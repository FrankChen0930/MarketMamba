"""
MarketMamba V5.5 — 全市場型態掃描模組
負責：多時間框架掃描全市場、結果輸出
"""

import os
import logging

import pandas as pd
import numpy as np

from marketmamba.config import PROCESSED_DIR, get_repo_output_dir
from marketmamba.pattern.detectors import (
    get_extrema,
    detect_standard_w,
    detect_spring_w,
    detect_m_top,
    detect_hns_bottom,
    detect_triangle,
    detect_bear_flag,
)

logger = logging.getLogger('MarketMamba.scanner')

# 多時間框架掃描尺度
SCALES = {
    '1_短線 (1~2週)': 3,
    '2_中線 (約1個月)': 8,
    '3_長線 (2~3個月)': 15,
    '4_半年大底': 30,
}

# 所有偵測函數
PATTERN_DETECTORS = [
    detect_standard_w,
    detect_spring_w,
    detect_m_top,
    detect_hns_bottom,
    detect_triangle,
    detect_bear_flag,
]


def scan_all_patterns(df: pd.DataFrame = None,
                      min_score: float = 60,
                      min_volume: float = 500000,
                      min_price: float = 10.0,
                      min_history: int = 100,
                      lookback_days: int = 180) -> pd.DataFrame:
    """
    全市場型態掃描

    過濾條件：
    - 只掃 4 位數字台股代號
    - 最近 5 天有交易紀錄
    - 近 20 日均量 > min_volume
    - 收盤價 > min_price
    - 至少 min_history 天歷史

    多時間框架：4 種尺度 (短/中/長/半年)

    Args:
        df: 特徵矩陣 (未傳入則從 parquet 讀取)
        min_score: 最低型態分數門檻
        min_volume: 最低均量門檻
        min_price: 最低價格門檻
        min_history: 最低歷史天數
        lookback_days: 回看天數

    Returns:
        掃描結果 DataFrame
    """
    if df is None:
        data_path = os.path.join(PROCESSED_DIR, 'V5_Mamba_Matrix.parquet')
        df = pd.read_parquet(data_path)
        df['Date'] = pd.to_datetime(df['Date'])

    latest_date = df['Date'].max()
    print(f"🕵️‍♂️ 啟動終極型態雷達 (六大結構 × 四時間框架)")
    print(f"📅 掃描基準日: {latest_date.strftime('%Y-%m-%d')}")

    grouped = df.sort_values('Date').groupby('stock_id')
    results = []

    for stock_id, group in grouped:
        stock_str = str(stock_id).strip()

        # 篩選條件
        if not (len(stock_str) == 4 and stock_str.isdigit()):
            continue
        if group['Date'].max() < latest_date - pd.Timedelta(days=5):
            continue

        df_recent = group.tail(lookback_days).copy()
        if len(df_recent) < min_history:
            continue
        if df_recent['Volume'].tail(20).mean() < min_volume:
            continue
        if df_recent['Close'].iloc[-1] < min_price:
            continue

        prices = df_recent['Close'].values

        # 多時間框架掃描
        for scale_name, order in SCALES.items():
            mins, maxs = get_extrema(prices, order)

            # 極值新鮮度檢查
            if len(mins) > 0 and (len(prices) - 1 - mins[-1]) > (order * 2.5):
                continue
            if len(maxs) > 0 and (len(prices) - 1 - maxs[-1]) > (order * 2.5):
                continue

            # 逐一嘗試六大型態
            matched = None
            for detector in PATTERN_DETECTORS:
                result = detector(prices, mins, maxs)
                if result and result[1] >= min_score:
                    matched = result
                    break  # 每支股票每個尺度只取最先匹配的型態

            if matched:
                p_name, score, neck, target = matched
                current_p = prices[-1]
                exp_return = ((target - current_p) / current_p) * 100

                results.append({
                    'Pattern': p_name,
                    'Stock_ID': stock_str,
                    'Scale': scale_name,
                    'Score': round(score, 1),
                    'Current_Price': round(current_p, 2),
                    'Neckline_Ref': round(neck, 2),
                    'Target_Price': round(target, 2),
                    'Exp_Return(%)': round(exp_return, 2),
                })
                break  # 每支股票只保留最短尺度的匹配

    # 輸出結果
    output_dir = get_repo_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'pattern_scan_results.csv')

    if results:
        df_patterns = pd.DataFrame(results)
        df_patterns = df_patterns.sort_values(
            ['Pattern', 'Scale', 'Score'], ascending=[False, True, False]
        ).reset_index(drop=True)
        df_patterns.to_csv(csv_path, index=False)

        print(f"\n🎉 掃描完成！共找出 {len(df_patterns)} 檔關鍵型態股！")
        print(f"📁 已儲存至: {csv_path}")
        return df_patterns
    else:
        # 空結果也要產生 CSV，前端才不會報錯
        empty_df = pd.DataFrame(columns=[
            'Pattern', 'Stock_ID', 'Scale', 'Score',
            'Current_Price', 'Neckline_Ref', 'Target_Price', 'Exp_Return(%)'
        ])
        empty_df.to_csv(csv_path, index=False)

        print("\n📉 今日市場無符合結構的股票。已產生空值報表。")
        return empty_df
