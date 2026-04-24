"""
MarketMamba V5.5 — 資料清洗模組
負責最終的資料品質把關：去重、補缺、NaN/Inf 清除、IPO 暖機期隔離
"""

import os
import logging

import numpy as np
import pandas as pd

from marketmamba.config import PROCESSED_DIR, MACRO_FILL_COLS

logger = logging.getLogger('MarketMamba.cleaner')


def clean_data(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    終極資料清洗

    步驟：
    1. 排序 + 去除重複欄位
    2. 替換 Inf 為 NaN
    3. 補班日 Macro 斷層修復 (Forward Fill)
    4. 剔除 IPO 暖機期不足的資料 (無 MA_60 / Alpha_1d)
    5. 剩餘缺失值填 0
    6. 清理輔助欄位
    7. 最終 NaN 檢查
    8. 存檔為 Parquet

    回傳：清洗完成的 DataFrame
    """
    print("🛁 啟動終極清洗 (隔離 IPO 暖機期與處理補班日)...")

    df = df.sort_values(['stock_id', 'Date']).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 1. 自動掃描並剔除重複的欄位 (修復 Is_Ex_Dividend 等重複問題)
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        logger.warning(f"⚠️ 發現重複欄位，已自動剔除: {dup_cols}")
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 2. 補班日 Macro 斷層修復
    for col in MACRO_FILL_COLS:
        if col in df.columns:
            df[col] = df.groupby('stock_id')[col].ffill().fillna(0)

    # 3. 剔除 IPO 暖機期不足的資料
    initial_len = len(df)
    required_cols = ['MA_60', 'Alpha_1d', 'Volatility_20d', 'Return_1d']
    available_required = [c for c in required_cols if c in df.columns]
    if available_required:
        df = df.dropna(subset=available_required)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"  🔪 剔除 IPO 暖機期不足資料: {dropped:,} 筆")

    # 4. 填補允許缺失的籌碼/基本面 (無資料 = 無事件)
    df = df.fillna(0)

    # 5. 清理不再需要的輔助欄位
    helper_cols = ['Date_1Y', 'Lookup', 'Rev_1Y_ago', 'TWII_MA_60']
    df = df.drop(columns=[c for c in helper_cols if c in df.columns], errors='ignore')

    # 6. 最終 NaN 檢查
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        logger.error(f"❌ 矩陣中仍有 {nan_count} 個 NaN 殘留！欄位: {nan_cols}")
        df = df.fillna(0)  # 安全網：強制歸零
    else:
        logger.info("✅ NaN 檢查通過，矩陣完全乾淨")

    # 7. 存檔
    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        final_path = os.path.join(PROCESSED_DIR, 'V5_Mamba_Matrix.parquet')
        df.to_parquet(final_path, engine='pyarrow')
        logger.info(f"💾 已存檔至: {final_path}")

    print(f"🎉 V5.5 終極資料庫建置完畢！")
    print(f"📊 最終實戰維度: {df.shape[0]:,} 列 x {df.shape[1]} 欄")

    return df
