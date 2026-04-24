"""
# ============================================================
# 🐍 MarketMamba V5.5 — 模型訓練管線 (Training Pipeline)
# 用途：訓練/重訓 Mamba+GAT 模型
# 環境：Google Colab GPU Runtime (A100 / T4)
# ============================================================
# 使用方式：
#   1. 在 Colab 中新增 Notebook
#   2. 依序將下方每個 「# %% Cell」 區塊複製貼入獨立的 Cell
#   3. 將 Runtime 類型改為 GPU (A100 推薦)
#   4. 依序執行每個 Cell
# ============================================================
"""

# %% Cell 1: 環境建置
# ==========================================
# 🚀 V5.5 訓練環境建置
# ==========================================
import os

print("🔧 建置訓練環境...")

# 1. 克隆程式碼
if not os.path.exists('/content/MarketMamba'):
    !git clone https://github.com/FrankChen0930/MarketMamba.git
else:
    %cd /content/MarketMamba
    !git pull origin main
    %cd /content

# 2. 安裝依賴
!pip install -q -r /content/MarketMamba/requirements.txt
!pip install -q FinMind

# 3. Mamba 算子 (A100/T4 通用)
os.makedirs("/content/mamba_core", exist_ok=True)
os.chdir("/content/mamba_core")
!wget -q "https://github.com/FrankChen0930/MarketMamba/releases/download/whl-for-mamba/causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl"
!wget -q "https://github.com/FrankChen0930/MarketMamba/releases/download/whl-for-mamba/mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl"
!pip install -q causal_conv1d-*.whl
!pip install -q mamba_ssm-*.whl
os.chdir("/content")

# 4. Python Path + Drive
import sys
sys.path.insert(0, '/content/MarketMamba')

from google.colab import drive
drive.mount('/content/drive')

print("✅ 訓練環境就緒！")

# 確認 GPU
import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
print(f"🖥️ GPU: {gpu_name}")
print(f"   VRAM: {vram:.1f} GB")

# 根據 GPU 自動建議 batch size
if 'A100' in gpu_name:
    SUGGESTED_BATCH = 512  # A100 80GB → 512
    print(f"   💎 A100 偵測到！建議 batch_size = {SUGGESTED_BATCH}")
elif 'V100' in gpu_name:
    SUGGESTED_BATCH = 128
elif 'T4' in gpu_name:
    SUGGESTED_BATCH = 64
else:
    SUGGESTED_BATCH = 32
    print(f"   ⚠️ 未知 GPU，保守設定 batch_size = {SUGGESTED_BATCH}")


# %% Cell 2: 準備訓練資料
# ==========================================
# 📊 資料同步 + 特徵工程 (帶快取加速)
# ==========================================
# 首次執行：全量同步 + 特徵工程 → 存檔為單一 parquet (~20 分鐘)
# 重啟後再執行：直接讀快取 (~10 秒)
# 需要重新從零跑：把 FORCE_REBUILD 改為 True
# ==========================================

import os
from marketmamba.config import PROCESSED_DIR

FORCE_REBUILD = False  # ← 改為 True 強制重建整個矩陣
MATRIX_CACHE = os.path.join(PROCESSED_DIR, 'V55_Mamba_Matrix.parquet')

if not FORCE_REBUILD and os.path.exists(MATRIX_CACHE):
    # ⚡ 快速載入 (10 秒內)
    import pandas as pd
    print(f"⚡ 偵測到快取矩陣，直接載入...")
    print(f"   路徑: {MATRIX_CACHE}")
    df = pd.read_parquet(MATRIX_CACHE)
    print(f"   ✅ 載入完成！")
else:
    # 🐌 首次完整建構 (~20 分鐘)
    from marketmamba.data.fetcher import run_full_data_sync
    from marketmamba.data.merger import merge_all_data
    from marketmamba.data.feature_engineer import build_features
    from marketmamba.data.cleaner import clean_data

    # 資料同步
    trading_days = run_full_data_sync()

    # 特徵工程
    df = merge_all_data()
    df = build_features(df)
    df = clean_data(df)

    # 💾 儲存快取 (下次重啟直接讀這個)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(MATRIX_CACHE, engine='pyarrow')
    print(f"\n💾 已儲存快取矩陣: {MATRIX_CACHE}")
    print(f"   下次重啟直接讀取，不用再跑 20 分鐘！")

print(f"\n📊 訓練矩陣: {df.shape[0]:,} 列 × {df.shape[1]} 欄")
print(f"📅 時間範圍: {df['Date'].min()} ~ {df['Date'].max()}")
print(f"📈 股票數量: {df['stock_id'].nunique()}")


# %% Cell 3: 消息面情緒特徵
# ==========================================
# 📰 情緒特徵注入 (自動偵測最佳資料來源)
# ==========================================
# 策略：
#   1. 先找 Drive 上的 processed 快取 (本機預跑 FinBERT 的結果)
#   2. 若只有 raw 快取 → 在 Colab 跑 FinBERT
#   3. 若都沒有 → 在 Colab 即時爬取 + FinBERT
#   4. 無論哪種方式，歷史資料的情緒欄位都是 0 (正常)
#
# ⚠️ 如果你要跳過情緒引擎 (純 V5.0 模式)，直接不執行這個 Cell 即可

import json, glob
import numpy as np
from marketmamba.config import (
    NEWS_CACHE_DIR, SENTIMENT_SCALAR_COLS,
    SENTIMENT_EMBED_EN_COLS, SENTIMENT_EMBED_CN_COLS,
)

# === 初始化所有情緒欄位為 0 ===
all_sentiment_cols = SENTIMENT_SCALAR_COLS + SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS
for col in all_sentiment_cols:
    if col not in df.columns:
        df[col] = 0.0

# === 偵測可用的情緒資料來源 ===
processed_dir = os.path.join(NEWS_CACHE_DIR, 'processed')
raw_dir = os.path.join(NEWS_CACHE_DIR, 'raw')

# 列出所有搜尋路徑 (幫助 debug)
print(f"🔍 搜尋情緒快取...")
print(f"   NEWS_CACHE_DIR: {NEWS_CACHE_DIR}")
print(f"   processed 目錄: {processed_dir} → 存在: {os.path.exists(processed_dir)}")
print(f"   raw 目錄: {raw_dir} → 存在: {os.path.exists(raw_dir)}")

# 方案 1: 有 processed 快取 (最快)
score_files = sorted(glob.glob(os.path.join(processed_dir, '*_sentiment_scores.json'))) if os.path.exists(processed_dir) else []

if score_files:
    print(f"✅ 找到 {len(score_files)} 天的情緒分數快取，直接載入...")
    for score_file in score_files:
        date_str = os.path.basename(score_file)[:10]
        with open(score_file) as f:
            scores = json.load(f)

        mask = df['Date'] == pd.to_datetime(date_str)
        df.loc[mask, 'Sent_Market_US'] = scores.get('market_us_score', 0)
        df.loc[mask, 'Sent_Geopolitical'] = scores.get('geopolitical_score', 0)
        df.loc[mask, 'Sent_Market_TW'] = scores.get('market_tw_score', 0)

        for stock_id, sdata in scores.get('stocks', {}).items():
            stock_mask = mask & (df['stock_id'].astype(str) == stock_id)
            df.loc[stock_mask, 'Sent_Stock_CN'] = sdata.get('score', 0)
            df.loc[stock_mask, 'News_Volume_Stock'] = np.log1p(sdata.get('count', 0))

    print(f"✅ 已載入 {len(score_files)} 天的情緒快取")

else:
    # 方案 2: 有 raw 快取 → 在 Colab 跑 FinBERT
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*_news_bundle.json'))) if os.path.exists(raw_dir) else []

    if raw_files:
        print(f"📰 找到 {len(raw_files)} 天的原始新聞，在 Colab 執行 FinBERT 分析...")
        # 用最新一天的新聞跑 integrator
        latest_raw = raw_files[-1]
        with open(latest_raw) as f:
            all_news = json.load(f)
        print(f"   來源: {os.path.basename(latest_raw)}")
        print(f"   新聞數: 市場EN={len(all_news.get('market_en',[]))} 地緣={len(all_news.get('geopolitical',[]))} 台股={len(all_news.get('market_tw',[]))} 個股={len(all_news.get('stocks',{}))}")

        # 執行 FinBERT
        from marketmamba.sentiment.integrator import compute_sentiment_features
        df = compute_sentiment_features(df, precomputed_news=all_news)

    else:
        # 方案 3: 什麼都沒有 → 即時爬取 + 分析
        print("⚠️ 未找到任何新聞快取")
        print("   🌐 啟動即時爬取 + FinBERT 分析 (可能需要 5~10 分鐘)...")
        from marketmamba.sentiment.integrator import compute_sentiment_features
        df = compute_sentiment_features(df)

# 確認情緒特徵狀態
non_zero_ratio = (df[SENTIMENT_SCALAR_COLS] != 0).any(axis=1).mean()
print(f"\n📊 情緒特徵狀態:")
print(f"   有消息面訊號的樣本比例: {non_zero_ratio:.2%}")
print(f"   （歷史資料為 0 是正常的，模型會自動學習區分）")


# %% Cell 4: 開始訓練
# ==========================================
# 🧠 訓練 Mamba + Dynamic GAT
# ==========================================
from marketmamba.models.trainer import train_model

# ===========================
# 🎛️ 訓練參數設定 (請自行調整)
# ===========================
TRAIN_PARAMS = {
    'epochs': 50,                   # 訓練輪數
    'batch_size': 256,              # ← 自己決定 (A100 80GB 建議 256~512)
    'lr': 1e-4,                     # 學習率
    'early_stop_patience': 7,       # 連續 N 輪無改善就停止
    'val_ratio': 0.15,              # 驗證集比例
}

print(f"🔧 訓練設定: batch_size={TRAIN_PARAMS['batch_size']} (GPU: {gpu_name})")

# 🚀 開始訓練！(tqdm 進度條、Loss 報告即時顯示)
model, history = train_model(df, **TRAIN_PARAMS)


# %% Cell 5: 訓練結果視覺化
# ==========================================
# 📈 Loss 曲線 + 過擬合診斷
# ==========================================
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss 曲線
ax1.plot(history['train_loss'], label='Train Loss', color='#ff4b4b', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', color='#00fa9a', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('訓練 / 驗證 Loss 曲線')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 學習率曲線
ax2.plot(history['lr'], color='#6c5ce7', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Cosine Annealing LR')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 診斷報告
best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
final_gap = history['train_loss'][-1] - history['val_loss'][-1]
print(f"\n🏆 最佳模型: Epoch {best_epoch}/{len(history['val_loss'])}")
print(f"   Best Val Loss: {min(history['val_loss']):.6f}")
print(f"   Train-Val Gap: {abs(final_gap):.6f} {'⚠️ 可能過擬合' if final_gap < -0.001 else '✅ 正常'}")


# %% Cell 6: 推論驗證
# ==========================================
# 🔍 用訓練好的模型跑一次推論，驗證結果
# ==========================================
from marketmamba.models.inference import run_inference

df_kelly, df_traj = run_inference(df)

print("\n🎯 Top 10 Alpha 潛力股 (新模型):")
print(df_kelly.head(10)[['Ticker', 'Exp_Return_15D', 'Sharpe_Score', 'Suggested_Weight']].to_string(index=False))


# %% Cell 7 (選配): 推送模型到 GitHub
# ==========================================
# 🚀 發布新模型的預測結果
# ==========================================
from marketmamba.pattern.scanner import scan_all_patterns
from marketmamba.robot.portfolio_manager import rebalance
from marketmamba.deploy.publisher import push_to_github

scan_all_patterns(df)
rebalance(df_kelly)
push_to_github()
