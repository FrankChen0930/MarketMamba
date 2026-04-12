"""
MarketMamba V5.5 — 資料擷取模組
負責從 FinMind API 與 yfinance 下載並快取所有原始資料
包含：大盤總經、個股價量、三大法人籌碼、低頻財報、VIP 國家隊/除權息
"""

import os
import time
import logging

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm.auto import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from marketmamba.config import (
    FINMIND_TOKEN, FINMIND_API_URL, FINMIND_MARKET_DATASETS,
    DATA_START_DATE, RAW_SUBDIRS, US_TICKERS,
    get_today_str,
)

logger = logging.getLogger('MarketMamba.fetcher')


# ==========================================
# HTTP Session (帶重試機制)
# ==========================================
def _create_session() -> requests.Session:
    """建立帶有自動重試的 HTTP Session"""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5))
    session.mount('https://', adapter)
    return session


# ==========================================
# FinMind API 通用擷取器
# ==========================================
def _fetch_finmind(session: requests.Session, dataset: str,
                   date_str: str, timeout: int = 15) -> pd.DataFrame | None:
    """
    通用 FinMind API 擷取函數
    成功回傳 DataFrame，失敗回傳 None 並記錄警告
    """
    try:
        res = session.get(FINMIND_API_URL, params={
            "dataset": dataset,
            "start_date": date_str,
            "end_date": date_str,
            "token": FINMIND_TOKEN,
        }, timeout=timeout)

        data = res.json()
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            df = pd.DataFrame(data["data"])
            df['stock_id'] = df['stock_id'].astype(str).str.strip()
            return df

    except requests.exceptions.Timeout:
        logger.warning(f"FinMind API 逾時: {dataset} @ {date_str}")
    except requests.exceptions.ConnectionError:
        logger.warning(f"FinMind API 連線失敗: {dataset} @ {date_str}")
    except Exception as e:
        logger.warning(f"FinMind API 異常: {dataset} @ {date_str} — {e}")

    return None


# ==========================================
# 1. 總經與大盤 (Macro)
# ==========================================
def fetch_macro(start_date: str = DATA_START_DATE,
                end_date: str = None) -> pd.DataFrame:
    """
    更新大盤 (TWII) 與國際盤 (SOX/QQQ/VIX/TNX/Gold/Oil) 資料
    包含美股時差對齊 (US_Visible_Date = US日期 + 1天)
    """
    end_date = end_date or get_today_str()
    logger.info(f"📊 擷取大盤總經資料 ({start_date} ~ {end_date})...")

    # 台灣加權指數
    twii = yf.Ticker("^TWII").history(start=start_date, end=end_date, auto_adjust=False)
    if twii.empty:
        raise ValueError("❌ 無法取得 TWII 資料，請檢查 yfinance 連線")
    twii.index = pd.to_datetime(twii.index).tz_localize(None)
    df_macro = pd.DataFrame({'Date': twii.index, 'TWII_Close': twii['Close'].values})

    # 國際盤
    df_us = pd.DataFrame(index=df_macro['Date'])
    for tick, name in US_TICKERS.items():
        try:
            tmp = yf.Ticker(tick).history(start=start_date, end=end_date, auto_adjust=False)
            if tmp.empty:
                logger.warning(f"  ⚠️ {tick} ({name}) 回傳空資料，跳過")
                continue
            tmp.index = pd.to_datetime(tmp.index).tz_localize(None)
            df_us = pd.merge(
                df_us,
                tmp[['Close']].rename(columns={'Close': name}),
                left_on='Date', right_index=True, how='outer'
            )
        except Exception as e:
            logger.warning(f"  ⚠️ {tick} ({name}) 擷取失敗: {e}")

    # 美股時差對齊：美股收盤資料要隔天才看得到
    df_us = df_us.sort_values('Date').dropna(how='all')
    df_us['US_Visible_Date'] = df_us['Date'] + pd.Timedelta(days=1)

    df_macro = pd.merge_asof(
        df_macro,
        df_us.drop(columns=['Date']).sort_values('US_Visible_Date'),
        left_on='Date', right_on='US_Visible_Date',
        direction='backward'
    )

    # 標記美股休市日 (Forward Fill 填補)
    df_macro['US_Market_Closed'] = np.where(
        (df_macro['Date'] - df_macro['US_Visible_Date']).dt.days > 0, 1, 0
    )
    df_macro = df_macro.drop(columns=['US_Visible_Date']).ffill().bfill()

    # 存檔
    save_path = os.path.join(RAW_SUBDIRS['Daily_Macro'], 'macro_features.parquet')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_macro.to_parquet(save_path, engine='pyarrow')
    logger.info(f"✅ 總經資料已更新 ({len(df_macro)} 筆)")

    return df_macro


# ==========================================
# 2. 個股價量 (OHLCV)
# ==========================================
def fetch_daily_prices(trading_days: list[str]) -> None:
    """
    逐日補齊個股價量資料 (有檔案就跳過)
    缺失策略：跳過該日 + 記錄 Warning
    """
    session = _create_session()
    save_dir = RAW_SUBDIRS['Daily_Price']
    os.makedirs(save_dir, exist_ok=True)
    skipped, fetched, failed = 0, 0, 0

    for date_str in tqdm(trading_days, desc="📈 掃描並補齊價量缺口"):
        save_path = os.path.join(save_dir, f"{date_str}_price.parquet")
        if os.path.exists(save_path):
            skipped += 1
            continue

        df_tmp = _fetch_finmind(session, "TaiwanStockPrice", date_str)

        if df_tmp is not None and not df_tmp.empty:
            df_clean = df_tmp[['stock_id', 'open', 'max', 'min', 'close', 'Trading_Volume']].rename(
                columns={'open': 'Open', 'max': 'High', 'min': 'Low',
                         'close': 'Close', 'Trading_Volume': 'Volume'}
            )
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            df_clean.to_parquet(save_path, engine='pyarrow')
            fetched += 1
        else:
            failed += 1

        time.sleep(0.2)

    logger.info(f"📈 價量完成 — 新增: {fetched}, 跳過: {skipped}, 失敗: {failed}")


# ==========================================
# 3. 全市場籌碼 (三大法人/信用/借券)
# ==========================================
def _process_inst_buysell(df_tmp: pd.DataFrame) -> pd.DataFrame:
    """處理三大法人買賣超 → Foreign_Buy / Trust_Buy / Dealer_Buy"""
    df_tmp['net_buy'] = (
        pd.to_numeric(df_tmp['buy'], errors='coerce').fillna(0) -
        pd.to_numeric(df_tmp['sell'], errors='coerce').fillna(0)
    )
    pivot = df_tmp.pivot_table(
        index='stock_id', columns='name', values='net_buy', aggfunc='sum'
    ).reset_index()

    rename_map = {
        '外資及陸資(不含外資自營商)': 'Foreign_Buy',
        'Foreign_Investor': 'Foreign_Buy',
        '投信': 'Trust_Buy',
        'Investment_Trust': 'Trust_Buy',
        '自營商(自行買賣)': 'Dealer_Buy',
        'Dealer_self': 'Dealer_Buy',
    }
    pivot = pivot.rename(columns=rename_map)
    keep_cols = ['stock_id'] + [c for c in ['Foreign_Buy', 'Trust_Buy', 'Dealer_Buy'] if c in pivot.columns]
    return pivot[keep_cols]


def _process_margin_short(df_tmp: pd.DataFrame) -> pd.DataFrame:
    """處理融資融券 → Margin_Balance / Short_Balance"""
    avail = [c for c in ['MarginPurchaseTodayBalance', 'ShortSaleTodayBalance'] if c in df_tmp.columns]
    if not avail:
        return pd.DataFrame(columns=['stock_id'])
    df_clean = df_tmp[['stock_id'] + avail].rename(columns={
        'MarginPurchaseTodayBalance': 'Margin_Balance',
        'ShortSaleTodayBalance': 'Short_Balance',
    })
    return df_clean


def _process_day_trading(df_tmp: pd.DataFrame) -> pd.DataFrame:
    """處理當沖比例 → Day_Trading_Ratio"""
    if 'BuyAfterSale' not in df_tmp.columns or 'Volume' not in df_tmp.columns:
        return pd.DataFrame(columns=['stock_id'])
    df_tmp['Day_Trading_Ratio'] = (
        pd.to_numeric(df_tmp['BuyAfterSale'], errors='coerce').fillna(0) /
        pd.to_numeric(df_tmp['Volume'], errors='coerce').fillna(1)
    )
    return df_tmp[['stock_id', 'Day_Trading_Ratio']]


def _process_sec_lending(df_tmp: pd.DataFrame) -> pd.DataFrame:
    """處理借券 → Securities_Lending"""
    if 'secLending' in df_tmp.columns:
        return df_tmp[['stock_id', 'secLending']].rename(columns={'secLending': 'Securities_Lending'})
    elif 'volume' in df_tmp.columns:
        return (df_tmp.groupby('stock_id')['volume'].sum()
                .reset_index().rename(columns={'volume': 'Securities_Lending'}))
    return pd.DataFrame(columns=['stock_id'])


# 籌碼資料處理器映射
_MARKET_PROCESSORS = {
    'Inst_BuySell': _process_inst_buysell,
    'Margin_Short': _process_margin_short,
    'Day_Trading':  _process_day_trading,
    'Sec_Lending':  _process_sec_lending,
}


def fetch_daily_market(trading_days: list[str], lookback: int = 30) -> None:
    """
    逐日補齊籌碼資料 (只掃最近 lookback 天)
    缺失策略：填 0 (無買賣紀錄 = 法人當天未操作)
    """
    session = _create_session()
    save_dir = RAW_SUBDIRS['Daily_Market']
    os.makedirs(save_dir, exist_ok=True)
    days_to_scan = trading_days[-lookback:] if len(trading_days) > lookback else trading_days

    for date_str in tqdm(days_to_scan, desc="📦 掃描並補齊籌碼缺口"):
        save_path = os.path.join(save_dir, f"{date_str}_market.parquet")
        if os.path.exists(save_path):
            continue

        daily_merged = None

        for key, dataset_name in FINMIND_MARKET_DATASETS.items():
            df_tmp = _fetch_finmind(session, dataset_name, date_str)

            if df_tmp is not None and not df_tmp.empty:
                processor = _MARKET_PROCESSORS.get(key)
                if processor:
                    df_clean = processor(df_tmp)
                    if len(df_clean.columns) > 1:
                        daily_merged = (
                            df_clean if daily_merged is None
                            else pd.merge(daily_merged, df_clean, on='stock_id', how='outer')
                        )
            time.sleep(0.2)

        if daily_merged is not None and not daily_merged.empty:
            daily_merged.to_parquet(save_path, engine='pyarrow')

    logger.info("📦 籌碼資料補齊完成")


# ==========================================
# 4. 低頻資料 (月營收)
# ==========================================
def fetch_monthly_revenue(end_date: str = None) -> None:
    """
    補齊近期月營收資料
    缺失策略：Forward Fill (營收公布前沿用上期)
    """
    session = _create_session()
    end_date = end_date or get_today_str()
    save_dir = RAW_SUBDIRS['Monthly_Revenue']
    os.makedirs(save_dir, exist_ok=True)

    months = pd.date_range(start=DATA_START_DATE, end=end_date, freq='MS').strftime('%Y-%m-%d').tolist()

    for month_str in tqdm(months, desc="💰 掃描月營收缺口"):
        save_path = os.path.join(save_dir, f"{month_str[:7]}_revenue.parquet")
        if os.path.exists(save_path):
            continue

        df_tmp = _fetch_finmind(session, "TaiwanStockMonthRevenue", month_str)

        if df_tmp is not None and not df_tmp.empty:
            df_clean = (
                df_tmp.groupby('stock_id')['revenue'].first()
                .reset_index().rename(columns={'revenue': 'Monthly_Revenue'})
            )
            df_clean['Monthly_Revenue'] = pd.to_numeric(df_clean['Monthly_Revenue'], errors='coerce').fillna(0)
            df_clean.to_parquet(save_path, engine='pyarrow')

        time.sleep(0.25)

    logger.info("💰 月營收補齊完成")


# ==========================================
# 5. VIP 專屬資料 (除權息/八大行庫)
# ==========================================
def fetch_vip_dividend(trading_days: list[str]) -> None:
    """補齊除權息結果表"""
    session = _create_session()
    save_dir = RAW_SUBDIRS['Daily_Dividend']
    os.makedirs(save_dir, exist_ok=True)

    for date_str in tqdm(trading_days, desc="🛡️ 掃描除權息防護網"):
        save_path = os.path.join(save_dir, f"{date_str}_dividend.parquet")
        if os.path.exists(save_path):
            continue

        df_tmp = _fetch_finmind(session, "TaiwanStockDividendResult", date_str)
        if df_tmp is not None and not df_tmp.empty:
            df_tmp.to_parquet(save_path, engine='pyarrow')

        time.sleep(0.2)

    logger.info("🛡️ 除權息資料補齊完成")


def fetch_vip_govbank(trading_days: list[str]) -> None:
    """補齊八大行庫買賣超"""
    session = _create_session()
    save_dir = RAW_SUBDIRS['Daily_GovBank']
    os.makedirs(save_dir, exist_ok=True)

    for date_str in tqdm(trading_days, desc="🏦 掃描國家隊護盤底線"):
        save_path = os.path.join(save_dir, f"{date_str}_govbank.parquet")
        if os.path.exists(save_path):
            continue

        df_tmp = _fetch_finmind(session, "TaiwanStockGovernmentBankBuySell", date_str)
        if df_tmp is not None and not df_tmp.empty:
            df_tmp['buy'] = pd.to_numeric(df_tmp['buy'], errors='coerce').fillna(0)
            df_tmp['sell'] = pd.to_numeric(df_tmp['sell'], errors='coerce').fillna(0)
            df_tmp['Gov_Bank_Buy'] = df_tmp['buy'] - df_tmp['sell']
            df_tmp[['stock_id', 'Gov_Bank_Buy']].to_parquet(save_path, engine='pyarrow')

        time.sleep(0.2)

    logger.info("🏦 八大行庫資料補齊完成")


# ==========================================
# 🚀 全量同步入口
# ==========================================
def run_full_data_sync(end_date: str = None) -> list[str]:
    """
    執行全量資料同步，回傳交易日清單
    
    呼叫順序：
    1. 總經與大盤 → 取得交易日清單
    2. 個股價量
    3. 全市場籌碼
    4. 月營收
    5. VIP 除權息 + 八大行庫
    """
    end_date = end_date or get_today_str()
    print(f"🚀 啟動 V5.5 全量資料同步 (掃描至 {end_date})...")

    # 1. 總經
    df_macro = fetch_macro(end_date=end_date)
    trading_days = df_macro['Date'].dt.strftime('%Y-%m-%d').tolist()

    # 2. 個股價量
    fetch_daily_prices(trading_days)

    # 3. 籌碼
    fetch_daily_market(trading_days)

    # 4. 月營收
    fetch_monthly_revenue(end_date=end_date)

    # 5. VIP
    fetch_vip_dividend(trading_days)
    fetch_vip_govbank(trading_days)

    print(f"🎉 全量資料同步完成！共 {len(trading_days)} 個交易日")
    return trading_days
