"""
# ============================================================
# 🐍 MarketMamba V5.5 — 日常推論管線 (Daily Pipeline)
# 用途：每日收盤後執行，產出預測 → 型態掃描 → 調倉 → 發布
# 環境：Google Colab GPU Runtime (A100 / T4)
# ============================================================
# 使用方式：
#   1. 在 Colab 中新增 Notebook
#   2. 依序將下方每個 「# %% Cell」 區塊複製貼入獨立的 Cell
#   3. 將 Runtime 切為 GPU (A100 推薦)
#   4. 依序執行每個 Cell
# ============================================================
"""

# %% Cell 1: 環境建置
# ==========================================
# 🚀 V5.5 環境建置 (只需首次或 Runtime 重啟後執行)
# ==========================================
import os

print("🚀 啟動 V5.5 終極端到端自動化管線...")

# 1. 克隆最新程式碼
if not os.path.exists('/content/MarketMamba'):
    !git clone https://github.com/FrankChen0930/MarketMamba.git
else:
    %cd /content/MarketMamba
    !git pull origin main
    %cd /content

# 2. 安裝依賴
!pip install -q -r /content/MarketMamba/requirements.txt
!pip install -q FinMind

# 3. 安裝 Mamba 底層算子 (A100/T4 通用)
os.makedirs("/content/mamba_core", exist_ok=True)
os.chdir("/content/mamba_core")
!wget -q "https://github.com/FrankChen0930/MarketMamba/releases/download/whl-for-mamba/causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl"
!wget -q "https://github.com/FrankChen0930/MarketMamba/releases/download/whl-for-mamba/mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl"
!pip install -q causal_conv1d-*.whl
!pip install -q mamba_ssm-*.whl
os.chdir("/content")

# 4. 設定 Python Path
import sys
sys.path.insert(0, '/content/MarketMamba')

# 5. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("✅ 環境建置完成！")


# %% Cell 2: 資料同步
# ==========================================
# 📊 全市場資料同步 (FinMind + yfinance)
# ==========================================
from marketmamba.data.fetcher import run_full_data_sync

trading_days = run_full_data_sync()
print(f"📅 交易日清單共 {len(trading_days)} 天，最後一天: {trading_days[-1]}")


# %% Cell 3: 跨頻融合 + 特徵工程
# ==========================================
# 🧬 跨頻率融合 + 特徵煉金
# ==========================================
from marketmamba.data.merger import merge_all_data
from marketmamba.data.feature_engineer import build_features
from marketmamba.data.cleaner import clean_data

df = merge_all_data()
df = build_features(df)
df = clean_data(df)

print(f"📊 特徵矩陣: {df.shape[0]:,} 列 × {df.shape[1]} 欄")


# %% Cell 4 (選配): 消息面情緒引擎
# ==========================================
# 📰 情緒特徵載入方式 (二選一)
#
# 方案 A: 使用本機預爬的快取 (推薦，速度快)
#   前提：已執行 scripts/crawl_local.py --run-finbert
#   並將 News_Cache 上傳到 Drive
#
# 方案 B: 直接在 Colab 爬 + 分析 (簡單但慢)
#   適合日常推論，新聞量少不會太久
#
# 方案 C: 跳過 → 自動退回 V5.0 模式 (46 維)
# ==========================================

# --- 方案 B: Colab 即時爬取 (日常推論推薦) ---
# from marketmamba.sentiment.integrator import compute_sentiment_features
# df = compute_sentiment_features(df)

# --- 方案 A: 使用本機預爬快取 ---
# (把 Cell 3 的 Training notebook 範例複製過來)


# %% Cell 5: AI 推論
# ==========================================
# 🧠 Mamba + GAT 推論 (夏普/凱利評分)
# ==========================================
from marketmamba.models.inference import run_inference

df_kelly, df_traj = run_inference(df)

# 顯示 Top 10
print("\n🎯 今日 Top 10 Alpha 潛力股:")
print(df_kelly.head(10)[['Ticker', 'Exp_Return_15D', 'Volatility_Risk', 'Sharpe_Score', 'Suggested_Weight']].to_string(index=False))


# %% Cell 6: 型態掃描
# ==========================================
# 📐 傳統型態學雷達 (六大結構 × 四時間框架)
# ==========================================
from marketmamba.pattern.scanner import scan_all_patterns

df_patterns = scan_all_patterns(df)
if not df_patterns.empty:
    print(df_patterns.to_string(index=False))


# %% Cell 7: 自動調倉
# ==========================================
# 🤖 百萬實盤機器人調倉
# ==========================================
from marketmamba.robot.portfolio_manager import rebalance

ledger = rebalance(df_kelly)


# %% Cell 8: 發布 + 收工
# ==========================================
# 🚀 推送至 GitHub → 觸發 Streamlit 更新
# ==========================================
from marketmamba.deploy.publisher import push_to_github, shutdown_colab

push_to_github()

# 完成後自動切斷 Colab (省 GPU 額度)
# 如果不想自動關機，把下面這行註解掉
shutdown_colab()
