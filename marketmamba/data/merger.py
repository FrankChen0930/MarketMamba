"""
MarketMamba V5.5 — 跨頻率時序融合模組
負責將日頻價量、籌碼、總經、VIP、低頻財報融合為單一主表
"""

import os
import glob
import shutil
import logging

import pandas as pd
from tqdm.auto import tqdm

from marketmamba.config import RAW_SUBDIRS, LOCAL_RAW_DIR, is_colab

logger = logging.getLogger('MarketMamba.merger')


def _load_parquet_dir(dir_path: str, date_col_from_filename: bool = True,
                      desc: str = "讀取中") -> pd.DataFrame:
    """通用：讀取資料夾內所有 parquet 並合併"""
    files = glob.glob(os.path.join(dir_path, '*.parquet'))
    if not files:
        logger.warning(f"⚠️ 目錄無資料: {dir_path}")
        return pd.DataFrame()

    dfs = []
    for f in tqdm(files, desc=desc):
        df = pd.read_parquet(f)
        if date_col_from_filename:
            df['Date'] = pd.to_datetime(os.path.basename(f)[:10])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def _cache_to_local(subdirs_to_cache: list[str]) -> None:
    """Colab 專用：將碎小 parquet 複製到本地端加速讀取"""
    if not is_colab():
        return

    os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
    for subdir in subdirs_to_cache:
        src = RAW_SUBDIRS.get(subdir, '')
        if os.path.exists(src):
            dst = os.path.join(LOCAL_RAW_DIR, subdir)
            if not os.path.exists(dst):
                logger.info(f"⚡ 快取到本地: {subdir}")
                shutil.copytree(src, dst)


def _cleanup_local_cache() -> None:
    """清理 Colab 本地暫存"""
    if is_colab() and os.path.exists(LOCAL_RAW_DIR):
        shutil.rmtree(LOCAL_RAW_DIR)
        logger.info("🧹 已清理本地暫存")


def _get_price_dir() -> str:
    """取得價量資料目錄 (Colab 用本地快取版，本機用原路徑)"""
    local = os.path.join(LOCAL_RAW_DIR, 'Daily_Price')
    return local if os.path.exists(local) else RAW_SUBDIRS['Daily_Price']


def _get_market_dir() -> str:
    """取得籌碼資料目錄"""
    local = os.path.join(LOCAL_RAW_DIR, 'Daily_Market')
    return local if os.path.exists(local) else RAW_SUBDIRS['Daily_Market']


def merge_all_data() -> pd.DataFrame:
    """
    跨頻率時序大融合：將所有原始資料融合為單一主表

    融合順序：
    1. 日頻價量 (Daily_Price)
    2. 日頻籌碼 (Daily_Market)
    3. 總經 (Daily_Macro)
    4. VIP 國家隊 (Daily_GovBank)
    5. VIP 除權息 (Daily_Dividend)
    6. 低頻月營收 (Monthly_Revenue) — merge_asof 防未來函數
    7. 低頻集保 (Weekly_Holdings) — merge_asof
    8. 低頻財報 (Quarterly_Financials) — merge_asof + IFRS 對齊

    回傳：融合後的 DataFrame
    """
    print("🌌 啟動跨頻率時序大融合...")

    # Colab 加速：先快取到本地
    _cache_to_local(['Daily_Price', 'Daily_Market'])

    # === 1. 日頻價量 ===
    df_price = _load_parquet_dir(_get_price_dir(), desc="📈 價量讀取")
    if df_price.empty:
        raise FileNotFoundError("❌ 找不到價量資料！請先執行 fetcher.run_full_data_sync()")

    # 只保留合法台股代號 (4位數字 或 ETF 如 0050, 00878)
    df_master = df_price[
        df_price['stock_id'].astype(str).str.match(r'^([1-9]\d{3}|00\d{2,4}[A-Za-z]?)$')
    ].copy()

    # === 2. 日頻籌碼 ===
    df_market = _load_parquet_dir(_get_market_dir(), desc="📊 籌碼讀取")
    if not df_market.empty:
        df_master = pd.merge(df_master, df_market, on=['Date', 'stock_id'], how='left')

    # === 3. 總經 ===
    macro_path = os.path.join(RAW_SUBDIRS['Daily_Macro'], 'macro_features.parquet')
    if os.path.exists(macro_path):
        df_macro = pd.read_parquet(macro_path)
        df_macro['Date'] = pd.to_datetime(df_macro['Date'])
        df_master = pd.merge(df_master, df_macro, on='Date', how='left')

    # === 4. VIP 國家隊 ===
    gov_dir = RAW_SUBDIRS['Daily_GovBank']
    gov_files = glob.glob(os.path.join(gov_dir, '*.parquet'))
    if gov_files:
        df_gov = _load_parquet_dir(gov_dir, desc="🏦 國家隊讀取")
        if not df_gov.empty and 'Gov_Bank_Buy' in df_gov.columns:
            df_master = pd.merge(
                df_master, df_gov[['stock_id', 'Date', 'Gov_Bank_Buy']],
                on=['Date', 'stock_id'], how='left'
            )

    # === 5. VIP 除權息 ===
    div_dir = RAW_SUBDIRS['Daily_Dividend']
    div_files = glob.glob(os.path.join(div_dir, '*.parquet'))
    if div_files:
        df_div = _load_parquet_dir(div_dir, desc="🛡️ 除權息讀取")
        if not df_div.empty:
            df_div['Is_Ex_Dividend'] = 1
            df_master = pd.merge(
                df_master, df_div[['stock_id', 'Date', 'Is_Ex_Dividend']],
                on=['Date', 'stock_id'], how='left'
            )

    # === 6~8. 低頻資料 (merge_asof 防未來函數) ===
    df_master = df_master.sort_values('Date')

    # 6. 月營收 — 隔月 10 號才公布
    rev_dir = RAW_SUBDIRS['Monthly_Revenue']
    rev_files = glob.glob(os.path.join(rev_dir, '*.parquet'))
    if rev_files:
        df_rev = pd.concat([
            pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:7] + '-01'))
            for f in rev_files
        ])
        df_rev['Pub_Date'] = df_rev['Date'] + pd.DateOffset(months=1, days=9)
        df_master = pd.merge_asof(
            df_master,
            df_rev.sort_values('Pub_Date').drop(columns=['Date']),
            left_on='Date', right_on='Pub_Date',
            by='stock_id', direction='backward'
        ).drop(columns=['Pub_Date'], errors='ignore')

    # 7. 集保庫存 — 延遲 3 天公布
    hold_dir = RAW_SUBDIRS['Weekly_Holdings']
    hold_files = glob.glob(os.path.join(hold_dir, '*.parquet'))
    if hold_files:
        df_hold = pd.concat([
            pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10]))
            for f in hold_files
        ])
        df_hold['Pub_Date'] = df_hold['Date'] + pd.Timedelta(days=3)
        df_master = pd.merge_asof(
            df_master,
            df_hold.sort_values('Pub_Date').drop(columns=['Date']),
            left_on='Date', right_on='Pub_Date',
            by='stock_id', direction='backward'
        ).drop(columns=['Pub_Date'], errors='ignore')

    # 8. 季報 — 使用 IFRS 法定截止日 (修復原本 +90d 的粗糙對齊)
    fin_dir = RAW_SUBDIRS['Quarterly_Financials']
    fin_files = glob.glob(os.path.join(fin_dir, '*.parquet'))
    if fin_files:
        df_fin = pd.concat([
            pd.read_parquet(f).assign(Date=pd.to_datetime(os.path.basename(f)[:10]))
            for f in fin_files
        ])
        df_fin['Pub_Date'] = df_fin['Date'].apply(_get_ifrs_pub_deadline)
        df_master = pd.merge_asof(
            df_master,
            df_fin.sort_values('Pub_Date').drop(columns=['Date']),
            left_on='Date', right_on='Pub_Date',
            by='stock_id', direction='backward'
        ).drop(columns=['Pub_Date'], errors='ignore')

    # 排序並清理
    df_master = df_master.sort_values(['stock_id', 'Date']).reset_index(drop=True)

    # 清理暫存
    _cleanup_local_cache()

    print(f"✅ 大融合完畢！總筆數: {len(df_master):,}")
    return df_master


def _get_ifrs_pub_deadline(report_date: pd.Timestamp) -> pd.Timestamp:
    """
    台灣上市櫃公司財報法定公布截止日 (IFRS)

    Q1 (Jan-Mar) → 最晚 5/15
    Q2 (Apr-Jun) → 最晚 8/14
    Q3 (Jul-Sep) → 最晚 11/14
    Q4 / 年報 (Oct-Dec) → 最晚隔年 3/31
    """
    m, y = report_date.month, report_date.year
    if m <= 3:
        return pd.Timestamp(y, 5, 15)
    elif m <= 6:
        return pd.Timestamp(y, 8, 14)
    elif m <= 9:
        return pd.Timestamp(y, 11, 14)
    else:
        return pd.Timestamp(y + 1, 3, 31)
