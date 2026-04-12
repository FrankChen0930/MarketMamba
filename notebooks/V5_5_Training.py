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
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3 if torch.cuda.is_available() else 0
print(f"🖥️ GPU: {gpu_name}")
print(f"   VRAM: {vram:.1f} GB")

# 根據 GPU 自動建議 batch size
if 'A100' in gpu_name:
    SUGGESTED_BATCH = 256  # A100 40GB → 256, 80GB → 512
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
# 📊 資料同步 + 特徵工程
# ==========================================
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

print(f"\n📊 訓練矩陣: {df.shape[0]:,} 列 × {df.shape[1]} 欄")
print(f"📅 時間範圍: {df['Date'].min()} ~ {df['Date'].max()}")
print(f"📈 股票數量: {df['stock_id'].nunique()}")


# %% Cell 3 (選配): 載入本機預爬的新聞情緒快取
# ==========================================
# 📰 載入本機預爬的新聞快取 (V5.5 情緒模式)
# ==========================================
# ⚠️ 前提：你已經在本機執行過 scripts/crawl_local.py
#    並將 News_Cache 資料夾上傳到 Google Drive:
#    MyDrive/MarketMamba_V5/News_Cache/
#
# ⚠️ 關於訓練時的新聞資料：
#    Google News RSS 只能抓最近 ~30 天的新聞，
#    所以 2019~2025 的歷史資料不會有情緒特徵。
#    ✅ 這是正常的！模型會學到：
#       - 情緒欄位 = 0 → 「沒有消息面訊號」→ 主要依賴量價籌碼
#       - 情緒欄位 ≠ 0 → 「有消息面訊號」→ 加權考慮
#    隨著每天累積新聞快取，訓練資料會越來越豐富。
#
# 方案 A: 不用情緒特徵 (V5.0 相容) → 跳過此 Cell
# 方案 B: 啟用情緒特徵 → 取消下面的註解

# import json, numpy as np
# from marketmamba.config import NEWS_CACHE_DIR, SENTIMENT_SCALAR_COLS, SENTIMENT_EMBED_EN_COLS, SENTIMENT_EMBED_CN_COLS
#
# # 讀取本機預爬的分數
# cache_files = sorted([
#     f for f in os.listdir(os.path.join(NEWS_CACHE_DIR, 'processed'))
#     if f.endswith('_sentiment_scores.json')
# ]) if os.path.exists(os.path.join(NEWS_CACHE_DIR, 'processed')) else []
#
# if cache_files:
#     # 初始化情緒欄位為 0
#     for col in SENTIMENT_SCALAR_COLS + SENTIMENT_EMBED_EN_COLS + SENTIMENT_EMBED_CN_COLS:
#         if col not in df.columns:
#             df[col] = 0.0
#
#     for cache_file in cache_files:
#         date_str = cache_file[:10]
#         with open(os.path.join(NEWS_CACHE_DIR, 'processed', cache_file)) as f:
#             scores = json.load(f)
#
#         mask = df['Date'] == pd.to_datetime(date_str)
#         df.loc[mask, 'Sent_Market_US'] = scores.get('market_us_score', 0)
#         df.loc[mask, 'Sent_Geopolitical'] = scores.get('geopolitical_score', 0)
#         df.loc[mask, 'Sent_Market_TW'] = scores.get('market_tw_score', 0)
#
#         for stock_id, sdata in scores.get('stocks', {}).items():
#             stock_mask = mask & (df['stock_id'].astype(str) == stock_id)
#             df.loc[stock_mask, 'Sent_Stock_CN'] = sdata.get('score', 0)
#             df.loc[stock_mask, 'News_Volume_Stock'] = np.log1p(sdata.get('count', 0))
#
#     print(f"✅ 已載入 {len(cache_files)} 天的情緒快取")
# else:
#     print("⚠️ 未找到新聞快取，情緒欄位全部為 0 (模型退回 V5.0 模式)")


# %% Cell 4: 開始訓練
# ==========================================
# 🧠 訓練 Mamba + Dynamic GAT
# ==========================================
from marketmamba.models.trainer import train_model

# ===========================
# 訓練參數設定 (GPU 自動適配)
# ===========================
TRAIN_PARAMS = {
    'epochs': 50,                   # 訓練輪數
    'batch_size': SUGGESTED_BATCH,  # 自動偵測 (A100=256, T4=64)
    'lr': 1e-4,                     # 學習率
    'early_stop_patience': 7,       # 連續 N 輪無改善就停止
    'val_ratio': 0.15,              # 驗證集比例
    # 'use_v5_compat': True,        # 取消註解 = 強制 V5.0 模式 (46 維)
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
