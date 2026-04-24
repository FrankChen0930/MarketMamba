"""
MarketMamba V5.5 — 特徵煉金模組
將融合後的原始資料轉換為模型所需的特徵矩陣
包含：大盤平穩化、報酬率/Alpha、波動率、技術指標、營收 YoY
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger('MarketMamba.feature_engineer')


def build_features(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    特徵煉金主函數：從融合主表中衍生所有技術與基本面特徵

    輸入：merge_all_data() 的產出
    輸出：加入衍生特徵後的 DataFrame (未清洗)
    """
    print("🧙‍♂️ 啟動特徵煉金：大盤平穩化、波動率、國家隊護盤與除權息防護...")

    df = df_master.copy()

    # 基礎清洗：去除無效紀錄
    df = df[(df['Close'] > 0) & (df['Volume'] > 0)].copy()

    # 基本面合理範圍限制
    if 'PER' in df.columns:
        df['PER'] = df['PER'].clip(0, 100)
    if 'PBR' in df.columns:
        df['PBR'] = df['PBR'].clip(0, 10)

    g = df.groupby('stock_id')
    new_features = {}

    # === 💎 VIP 專屬特徵 ===
    if 'Gov_Bank_Buy' in df.columns:
        df['Gov_Bank_Buy'] = df['Gov_Bank_Buy'].fillna(0)
        new_features['Gov_Bank_Sum_20d'] = g['Gov_Bank_Buy'].transform(
            lambda x: x.rolling(20).sum()
        )

    if 'Is_Ex_Dividend' in df.columns:
        new_features['Is_Ex_Dividend'] = df['Is_Ex_Dividend'].fillna(0)

    # === 🌟 1. 大盤與個股的絕對平穩化 ===
    df['TWII_MA_60'] = df['TWII_Close'].transform(lambda x: x.rolling(60).mean())
    new_features['TWII_Bias_60'] = (
        (df['TWII_Close'] - df['TWII_MA_60']) / (df['TWII_MA_60'] + 1e-8)
    )
    new_features['MA_60'] = g['Close'].transform(lambda x: x.rolling(60).mean())
    new_features['Bias_60'] = (
        (df['Close'] - new_features['MA_60']) / (new_features['MA_60'] + 1e-8)
    )

    # === 🌟 2. 報酬率與 Alpha ===
    new_features['Return_1d'] = g['Close'].pct_change(1)
    new_features['TWII_Return_1d'] = df['TWII_Close'].pct_change(1)
    new_features['Alpha_1d'] = new_features['Return_1d'] - new_features['TWII_Return_1d']

    # === 🌟 3. 風險認知：個股歷史波動率 ===
    new_features['Volatility_20d'] = (
        new_features['Return_1d']
        .groupby(df['stock_id'])
        .transform(lambda x: x.rolling(20).std())
    )

    # === 4. 傳統技術指標 ===
    # 移動平均線
    new_features['MA_20'] = g['Close'].transform(lambda x: x.rolling(20).mean())

    # 布林通道寬度
    std_20 = g['Close'].transform(lambda x: x.rolling(20).std())
    new_features['BB_Width'] = (4 * std_20) / (new_features['MA_20'] + 1e-8)

    # 量能均線與量比
    new_features['Vol_MA_20'] = g['Volume'].transform(lambda x: x.rolling(20).mean())
    new_features['Vol_Ratio'] = df['Volume'] / (new_features['Vol_MA_20'] + 1e-8)

    # 法人 20 日累積買賣超
    if 'Foreign_Buy' in df.columns:
        new_features['Foreign_Sum_20d'] = g['Foreign_Buy'].transform(
            lambda x: x.rolling(20).sum()
        )
    if 'Trust_Buy' in df.columns:
        new_features['Trust_Sum_20d'] = g['Trust_Buy'].transform(
            lambda x: x.rolling(20).sum()
        )

    # MACD 柱狀體
    ema_12 = g['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = g['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = ema_12 - ema_26
    macd_sig = macd.groupby(df['stock_id']).transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    new_features['MACD_Hist'] = macd - macd_sig

    # 併入大表
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    # === 🎯 營收 YoY (防未來函數) ===
    df = _compute_revenue_yoy(df)

    print("✅ V5.5 特徵煉金完成！")
    return df


def _compute_revenue_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算營收 YoY 成長率
    使用 merge_asof 搭配一年前的營收做比對，防止未來函數
    """
    if 'Monthly_Revenue' not in df.columns:
        logger.warning("⚠️ 無月營收資料，跳過 Rev_YoY 計算")
        df['Rev_YoY'] = 0.0
        return df

    # 建立一年前營收查找表
    rev_lookup = (
        df[['stock_id', 'Date', 'Monthly_Revenue']]
        .dropna()
        .rename(columns={'Date': 'Lookup', 'Monthly_Revenue': 'Rev_1Y_ago'})
        .sort_values('Lookup')
    )

    df['Date_1Y'] = df['Date'] - pd.DateOffset(years=1)
    df = df.sort_values('Date_1Y')

    df = pd.merge_asof(
        df, rev_lookup,
        left_on='Date_1Y', right_on='Lookup',
        by='stock_id', direction='backward',
        tolerance=pd.Timedelta(days=20)
    )

    df['Rev_YoY'] = (
        (df['Monthly_Revenue'] - df['Rev_1Y_ago']) / (df['Rev_1Y_ago'] + 1e-8)
    )

    # 清理輔助欄位
    df = df.drop(columns=['Date_1Y', 'Lookup', 'Rev_1Y_ago'], errors='ignore')

    return df


def build_feature_matrix(df_master: pd.DataFrame = None) -> pd.DataFrame:
    """
    完整特徵工程管線：融合 → 煉金 → 清洗

    如果未傳入 df_master，則自動從 merger 讀取
    """
    if df_master is None:
        from marketmamba.data.merger import merge_all_data
        df_master = merge_all_data()

    df = build_features(df_master)

    from marketmamba.data.cleaner import clean_data
    df = clean_data(df)

    return df
