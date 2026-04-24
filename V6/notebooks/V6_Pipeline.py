"""
MarketMamba V6 — Local Inference Notebook
===========================================
Use this on your local machine (RTX 3060 + WSL2) for daily inference.
This is the local equivalent of V5_5_Pipeline.py, but for V6.

Can also be run manually for ad-hoc inference or debugging.

Each # %% Cell is one Jupyter/VS Code cell.
"""

# %% Cell 1: Check Environment
# ==========================================
# 🔍 Verify WSL2 + CUDA is working correctly
# ==========================================
import sys, os
sys.path.insert(0, "/mnt/d/Desktop/work/MarketMamba/V6")
sys.path.insert(0, "/mnt/d/Desktop/work/MarketMamba")

import torch
print("=" * 50)
print("  MarketMamba V6 — Local Inference Check")
print("=" * 50)
print(f"Python: {sys.version[:20]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 50)

from marketmamba.config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    SEQ_LEN, INPUT_DIM, PRED_HORIZONS
)
print(f"Config loaded ✅")
print(f"  Model: seq_len={SEQ_LEN}, input_dim={INPUT_DIM}, horizons={PRED_HORIZONS}")


# %% Cell 2: Quick Data Update (Daily)
# ==========================================
# 📡 Fetch today's data (Hybrid: yfinance + TWSE + FinMind)
# ==========================================
from datetime import date, datetime
from marketmamba.data.fetcher import run_daily_update

TARGET_DATE = None   # None = today; or set "YYYY-MM-DD" for historical inference

today = TARGET_DATE or date.today().strftime("%Y-%m-%d")
print(f"Fetching data for: {today}")

today = run_daily_update(target_date=today)
print(f"✅ Data updated for {today}")


# %% Cell 3: Build Feature Matrix
# ==========================================
# 🔬 Load cached data + rebuild features for inference
# ==========================================
import pandas as pd
from marketmamba.data.merger import merge_all_data, validate_data_integrity
from marketmamba.data.feature_engineer import build_features, clean_and_scale

MATRIX_CACHE = PROCESSED_DIR / "V6_Feature_Matrix.parquet"

# Try to use cache (fast path for daily inference)
if MATRIX_CACHE.exists():
    print(f"⚡ Loading feature matrix from cache...")
    df = pd.read_parquet(MATRIX_CACHE)

    # Append today's data if not already present
    df["Date"] = pd.to_datetime(df["Date"])
    if pd.Timestamp(today) not in df["Date"].values:
        print(f"  Appending today's data ({today})...")
        data = merge_all_data()
        df_today = build_features(
            df_price    = data["prices"][pd.to_datetime(data["prices"]["Date"]).dt.strftime("%Y-%m-%d") == today],
            df_inst     = data["inst"],
            df_margin   = data["margin"],
            df_rev      = data["revenue"],
            df_fin      = data["financials"],
            df_macro    = data["macro"],
        )
        df_today = clean_and_scale(df_today)
        df = pd.concat([df, df_today], ignore_index=True)
        df.to_parquet(MATRIX_CACHE)
else:
    print("⚠️ No cache found — rebuilding from scratch (may take a few minutes)...")
    data = merge_all_data()
    validate_data_integrity(data)
    df = build_features(**{k: v for k, v in data.items() if v is not None})
    df = clean_and_scale(df)
    df.to_parquet(MATRIX_CACHE)

print(f"✅ Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"   Latest date: {df['Date'].max()}")


# %% Cell 4: Run Inference
# ==========================================
# 🧠 V6 Mamba+GAT Inference with MC-Dropout
# ==========================================
from marketmamba.models.inference import run_inference

device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on: {device_str}")

df_kelly, df_traj = run_inference(
    df          = df,
    device_str  = device_str,
    date_str    = today,
)

print(f"\n🎯 Top 10 Alpha Stocks [{today}]:")
print(
    df_kelly.head(10)[[
        "Ticker", "Exp_Alpha_20d", "Sharpe_Score", "Confidence", "Suggested_Weight"
    ]].to_string(index=False)
)


# %% Cell 5: LLM Market Report
# ==========================================
# 📝 Generate AI market commentary
# ==========================================
from marketmamba.llm.report_generator import generate_market_report, build_market_data

print("Generating LLM market report...")
try:
    market_data = build_market_data()
    report = generate_market_report(df_kelly, market_data, save=True)
    print(f"\n📊 Market Report [{report['date']}]")
    print("─" * 50)
    print(report["summary"])
    print("─" * 50)
except Exception as e:
    print(f"⚠️ LLM report failed: {e}")
    print("   Check ANTHROPIC_API_KEY or OPENAI_API_KEY environment variables")


# %% Cell 6: Portfolio Rebalancing
# ==========================================
# 🤖 Compute final positions with Kelly sizing
# ==========================================
from marketmamba.robot.portfolio_manager import rebalance

print("Computing portfolio positions...")
ledger = rebalance(df_kelly)

print(f"\n📊 Portfolio [{ledger.date}]: {ledger.total_stocks} positions")
for pos in ledger.positions[:10]:
    icon = {"高信心": "🟢", "中信心": "🟡", "低信心": "🔴"}.get(pos.confidence, "⚪")
    print(f"  {icon} {pos.ticker:6s} | {pos.weight:.1%} | ExpAlpha={pos.exp_alpha:+.2%}")


# %% Cell 7: Push to GitHub
# ==========================================
# 🚀 Publish results to GitHub → Streamlit auto-refresh
# ==========================================
from marketmamba.deploy.publisher import push_to_github

PUSH_NOW = False   # ← Set True when ready to publish

if PUSH_NOW:
    success = push_to_github(date_str=today)
    if success:
        print("✅ Results published successfully!")
    else:
        print("❌ Push failed — check git configuration")
else:
    print("⏭️ Push skipped (PUSH_NOW=False)")
    print(f"   Results saved locally at: {RESULTS_DIR}")
