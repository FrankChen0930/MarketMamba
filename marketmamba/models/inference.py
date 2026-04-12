"""
MarketMamba V5.5 — 推論模組
負責：張量組裝、Z-Score 標準化、模型推論、夏普/凱利評分、CSV 輸出
"""

import os
import logging
import random

import numpy as np
import pandas as pd
import torch

from marketmamba.config import (
    FEATURE_COLS, FEATURE_COLS_V5, MODEL_CONFIG, PROCESSED_DIR, MODEL_DIR,
    get_repo_output_dir,
)

logger = logging.getLogger('MarketMamba.inference')


def set_seed(seed: int = 42) -> None:
    """設定全域隨機種子確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_tensors(df: pd.DataFrame,
                    feature_cols: list[str] = None,
                    min_days: int = 120) -> tuple:
    """
    從 DataFrame 中組裝推論用的 3D Tensor

    步驟：
    1. 對每支股票取最後 120 天
    2. 逐股 Z-Score 標準化
    3. 防毒防線：NaN/Inf 強制歸零

    回傳：
        test_x: (N, 120, D) float32 CUDA tensor
        final_tickers: list[str] 股票代號
        latest_volatility: list[float] 最新波動率
    """
    feature_cols = feature_cols or FEATURE_COLS

    # 確保所有需要的欄位都在
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df_sorted = df.sort_values(['stock_id', 'Date'])
    grouped = df_sorted.groupby('stock_id')

    all_stocks_data = []
    final_tickers = []
    latest_volatility = []

    for stock_id, group in grouped:
        if len(group) < min_days:
            continue

        # 取最後 min_days 天
        group_features = group.tail(min_days)[feature_cols].copy()

        # Z-Score 標準化 (每支股票獨立標準化)
        for col in feature_cols:
            mean = group_features[col].mean()
            std = group_features[col].std()
            group_features[col] = (group_features[col] - mean) / (std + 1e-8)

        all_stocks_data.append(group_features.values)
        final_tickers.append(str(stock_id))

        # 取最新波動率
        vol = group.iloc[-1].get('Volatility_20d', 0.02)
        latest_volatility.append(vol if (not pd.isna(vol) and vol > 0) else 0.02)

    # 組裝 Tensor
    test_x = torch.tensor(np.array(all_stocks_data), dtype=torch.float32)
    if torch.cuda.is_available():
        test_x = test_x.cuda()

    # 防毒防線：強制清洗 Tensor
    test_x = torch.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"📦 張量組裝完成: {test_x.shape} ({len(final_tickers)} 支股票)")
    return test_x, final_tickers, latest_volatility


def compute_kelly_sharpe(pred_alpha: np.ndarray,
                         final_tickers: list[str],
                         latest_volatility: list[float],
                         target_day: int = 14) -> pd.DataFrame:
    """
    計算夏普評分 + 凱利資金配置

    濾網：
    1. 正規化夏普值：Alpha / (Volatility + 1.5% 懲罰)
    2. 殭屍股狙擊：波動率 < 0.5% → -999 分
    3. 凱利公式：Exp_Return / Volatility²，半凱利 (×0.5)，單股上限 20%

    Args:
        pred_alpha: (N, 30) 預測 Alpha 矩陣
        final_tickers: 股票代號列表
        latest_volatility: 波動率列表
        target_day: 用於排名的目標天數 (0-indexed)

    Returns:
        排序後的 DataFrame，包含 Ticker, Exp_Return_15D, Volatility_Risk, Sharpe_Score, Kelly/Weight
    """
    raw_vol = np.array(latest_volatility)

    # 濾網 1：正規化夏普值
    sharpe = pred_alpha[:, target_day] / (raw_vol + 0.015)

    # 濾網 2：殭屍股狙擊
    zombie_mask = raw_vol < 0.005
    sharpe[zombie_mask] = -999.0

    # 顯示用波動率 (下限 0.50%)
    display_vol = np.clip(raw_vol, a_min=0.005, a_max=None)

    df_export = pd.DataFrame({
        'Ticker': final_tickers,
        'Exp_Return_15D': pred_alpha[:, target_day],
        'Volatility_Risk': display_vol,
        'Sharpe_Score': sharpe,
    })

    # 凱利公式 (半凱利，正報酬 & 非殭屍股才買)
    df_export['Kelly_Raw'] = np.where(
        (df_export['Exp_Return_15D'] > 0) & (df_export['Sharpe_Score'] > -100),
        df_export['Exp_Return_15D'] / (df_export['Volatility_Risk'] ** 2),
        0
    )
    df_export['Suggested_Weight'] = np.clip(
        df_export['Kelly_Raw'] * 0.5, a_min=0.0, a_max=0.20
    )

    df_export = df_export.sort_values('Sharpe_Score', ascending=False)
    return df_export


def run_inference(df: pd.DataFrame = None,
                  model_path: str = None,
                  feature_cols: list[str] = None,
                  use_v5_compat: bool = False) -> tuple:
    """
    端到端推論管線

    步驟：
    1. 讀取/接收特徵矩陣
    2. 組裝張量
    3. 載入模型
    4. One-Forward-Pass 推論
    5. 計算夏普/凱利
    6. 輸出 CSV

    Args:
        df: 特徵矩陣 DataFrame (如未傳入則從 Parquet 讀取)
        model_path: 模型權重路徑
        feature_cols: 特徵欄位列表 (None = 自動偵測 V5.0/V5.5)
        use_v5_compat: 強制使用 V5.0 相容模式 (46 維)

    Returns:
        (df_kelly, df_traj) 兩個 DataFrame
    """
    set_seed(42)
    print("🧠 喚醒 MarketMamba 大腦並組裝張量...")

    # 1. 讀取資料
    if df is None:
        data_path = os.path.join(PROCESSED_DIR, 'V5_Mamba_Matrix.parquet')
        df = pd.read_parquet(data_path)

    # 2. 決定特徵欄位與模型版本
    if use_v5_compat:
        feature_cols = FEATURE_COLS_V5
        input_dim = MODEL_CONFIG['input_dim_v5']
    else:
        feature_cols = feature_cols or FEATURE_COLS
        # 自動偵測：如果情緒特徵全為 0，退回 V5.0
        sentiment_present = any(
            col in df.columns and df[col].abs().sum() > 0
            for col in ['Sent_Stock_CN', 'Sent_Stock_EN']
        )
        if not sentiment_present:
            logger.info("ℹ️ 未偵測到情緒特徵，退回 V5.0 相容模式 (46 維)")
            feature_cols = FEATURE_COLS_V5
            input_dim = MODEL_CONFIG['input_dim_v5']
        else:
            input_dim = len(feature_cols)

    # 3. 組裝張量
    test_x, final_tickers, latest_vol = prepare_tensors(df, feature_cols)

    # 4. 載入模型
    if model_path is None:
        if input_dim == MODEL_CONFIG['input_dim_v5']:
            model_path = os.path.join(MODEL_DIR, 'V5_DynamicGAT_Production.pth')
        else:
            model_path = os.path.join(MODEL_DIR, 'V5_5_Production.pth')

    if input_dim == MODEL_CONFIG['input_dim_v5']:
        from marketmamba.models.architecture import MarketMambaV5
        model = MarketMambaV5(input_dim=input_dim)
    else:
        from marketmamba.models.architecture import MarketMambaV55
        model = MarketMambaV55(input_dim=input_dim)

    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # 5. 推論
    print("⚡ 執行 One-Forward-Pass 軌跡預測...")
    with torch.no_grad(), torch.autocast(
        device_type='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float16
    ):
        pred_alpha = model(test_x).cpu().numpy()

    # 防毒防線
    pred_alpha = np.nan_to_num(pred_alpha, nan=0.0, posinf=0.0, neginf=0.0)

    # 6. 夏普/凱利
    df_kelly = compute_kelly_sharpe(pred_alpha, final_tickers, latest_vol)

    # 7. 軌跡 DataFrame
    df_traj = pd.DataFrame(
        pred_alpha,
        columns=[f'Day_{i+1}' for i in range(MODEL_CONFIG['pred_days'])]
    )
    df_traj.insert(0, 'Ticker', final_tickers)

    # 8. 存檔
    output_dir = get_repo_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    kelly_path = os.path.join(output_dir, 'df_kelly.csv')
    traj_path = os.path.join(output_dir, 'df_traj.csv')
    df_kelly.to_csv(kelly_path, index=False)
    df_traj.to_csv(traj_path, index=False)

    # 清理 GPU
    del model, test_x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"✅ 預測結果生成完畢！共 {len(final_tickers)} 支股票")
    print(f"   📁 {kelly_path}")
    print(f"   📁 {traj_path}")

    return df_kelly, df_traj
