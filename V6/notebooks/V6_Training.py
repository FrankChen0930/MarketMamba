"""
MarketMamba V6 — Colab Training Script
========================================
Run this on Google Colab with A100/T4 GPU.
Each # %% Cell block is one Colab cell.

Steps:
  Cell 0: Secrets (fill in your API tokens here)
  Cell 1: Environment setup (install deps, Mamba wheels, Drive mount)
  Cell 2: Config & paths check
  Cell 3: Full data sync (2012 → today, ~30 min first run)
  Cell 4: Build feature matrix + KG cache
  Cell 5: Walk-Forward baseline (optional)
  Cell 6: Train V6 model (single fold, for quick iteration)
  Cell 7: Full Walk-Forward with V6 model
  Cell 8: Save model + push to GitHub
"""

# %% Cell 0: Secrets <- FILL IN YOUR TOKENS HERE
# ==========================================
# Set API tokens as environment variables.
# This replaces the .env file for Colab.
# ==========================================
import os

# FinMind token (required for data sync)
os.environ["FINMIND_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiRnJhbmtDaGVuIiwiZW1haWwiOiJhMDk2NjQ2OTk2NEBnbWFpbC5jb20ifQ.rwJBGSwJyHDqXeVZKCMkKulKVk48Y2klu4pyUgiJyrE"

# LLM API key (optional - skip to disable LLM report)
os.environ["ANTHROPIC_API_KEY"] = ""   # paste your key if you have one
os.environ["OPENAI_API_KEY"]    = ""   # or OpenAI as fallback

assert os.environ.get("FINMIND_TOKEN"), "FinMind token is empty!"
print("Secrets loaded")


# %% Cell 1: Environment Setup
# ==========================================
# MarketMamba V6 Training Environment
# ==========================================
import os, sys

print("Setting up V6 training environment...")

# 1. Clone / update repo
if not os.path.exists("/content/MarketMamba"):
    os.system("git clone https://github.com/FrankChen0930/MarketMamba.git /content/MarketMamba")
    print("Repo cloned")
else:
    os.system("cd /content/MarketMamba && git pull origin main")
    print("Repo updated")

# 2. Create V6 directory structure
#    Needed until the first time you push V6/ to GitHub
for _d in [
    "/content/MarketMamba/V6/marketmamba/data",
    "/content/MarketMamba/V6/marketmamba/models",
    "/content/MarketMamba/V6/marketmamba/evaluation",
    "/content/MarketMamba/V6/marketmamba/knowledge",
    "/content/MarketMamba/V6/marketmamba/llm",
    "/content/MarketMamba/V6/marketmamba/robot",
    "/content/MarketMamba/V6/marketmamba/backtest",
    "/content/MarketMamba/V6/marketmamba/deploy",
    "/content/MarketMamba/V6/models",
    "/content/MarketMamba/V6/results",
    "/content/MarketMamba/V6/notebooks",
]:
    os.makedirs(_d, exist_ok=True)

# 3. Install Python dependencies
os.system("pip install -q yfinance requests pandas numpy scipy python-dotenv anthropic openai pyarrow")
os.system("pip install -q torch-geometric")

# 4. Mamba SSM kernel - Drive-Cache Strategy
# ============================================
# Layer 1: Google Drive pre-built wheel  (fast, ~10 sec - used on 2nd+ session)
# Layer 2: Binary pip install            (fast if PyPI has a matching wheel)
# Layer 3: Compile from source           (slow, ~40 min - only runs once per runtime version)
#           └─ saves wheel to Drive so future sessions skip compilation entirely
#
# The wheel is keyed by CUDA + PyTorch + Python version.
# Recompilation only needed if Colab upgrades its runtime stack.

import glob, shutil, torch

_cuda_ver  = (torch.version.cuda or "unknown").replace(".", "")    # e.g. "128"
_torch_ver = torch.__version__.split("+")[0].replace(".", "")       # e.g. "210"
_py_ver    = f"cp{sys.version_info.major}{sys.version_info.minor}"  # e.g. "cp312"
_wheel_key = f"cu{_cuda_ver}torch{_torch_ver}{_py_ver}"            # e.g. "cu128torch210cp312"

DRIVE_WHEEL_DIR = "/content/drive/MyDrive/MarketMamba/mamba_wheels"
WHEEL_BUILD_DIR = "/tmp/mamba_wheels"
os.makedirs(WHEEL_BUILD_DIR, exist_ok=True)

print(f"Mamba wheel key: {_wheel_key}")
print("Installing mamba_ssm...")

_installed = False

# --- Layer 1: Drive cache ---
if os.path.exists(DRIVE_WHEEL_DIR):
    _key_wheels = [
        f for f in glob.glob(f"{DRIVE_WHEEL_DIR}/*.whl")
        if _wheel_key in f
    ]
    if _key_wheels:
        print(f"  Found {len(_key_wheels)} cached wheel(s) in Drive - installing (fast path)...")
        for whl in _key_wheels:
            os.system(f"pip install -q {whl}")
        _installed = True
        print("  Installed from Drive cache")
    else:
        _all_wheels = glob.glob(f"{DRIVE_WHEEL_DIR}/*.whl")
        print(f"  No wheel matching key={_wheel_key} in Drive")
        if _all_wheels:
            print(f"  (Other runtime wheels present: {[os.path.basename(w) for w in _all_wheels]})")
else:
    print(f"  Drive wheel dir not found: {DRIVE_WHEEL_DIR}")

# --- Layer 2: Binary pip install ---
if not _installed:
    rc = os.system("pip install -q mamba-ssm causal-conv1d 2>/dev/null")
    if rc == 0:
        _installed = True
        print("  Binary pip install succeeded")

# --- Layer 3: Compile from source + save to Drive ---
if not _installed:
    print("  Compiling from source (~40 min). This only happens once per Colab runtime version.")
    print("  Wheel will be saved to Drive so next session takes ~10 sec.")

    rc3 = os.system(
        f"pip wheel mamba-ssm causal-conv1d "
        f"--no-build-isolation "
        f"--no-cache-dir "
        f"-w {WHEEL_BUILD_DIR}/ "
        f"2>&1 | tail -5"
    )

    if rc3 == 0:
        os.system(f"pip install -q {WHEEL_BUILD_DIR}/*.whl")

        # Save mamba_ssm + causal_conv1d wheels to Drive (skip dependency wheels)
        os.makedirs(DRIVE_WHEEL_DIR, exist_ok=True)
        for whl_path in glob.glob(f"{WHEEL_BUILD_DIR}/*.whl"):
            whl_name = os.path.basename(whl_path)
            if any(pkg in whl_name.lower() for pkg in ["mamba_ssm", "causal_conv1d"]):
                stem, ext = whl_name.rsplit(".", 1)
                tagged = f"{stem}___{_wheel_key}.{ext}"
                shutil.copy2(whl_path, f"{DRIVE_WHEEL_DIR}/{tagged}")
                print(f"  Saved to Drive: {tagged}")
        _installed = True
    else:
        print("  Source build failed - make sure GPU runtime is enabled in Colab")

# --- Verify ---
try:
    from mamba_ssm import Mamba
    print("mamba_ssm OK")
except ImportError as e:
    print(f"WARNING: mamba_ssm not available: {e}")
    print("  Cells 0-4 (data sync) can still run without it.")
    print("  Cell 6+ (training) requires mamba_ssm.")



# 5. Python path
# IMPORTANT: V6 must end up at sys.path[0] so its 'marketmamba' package
# takes priority over the OLD V5.5 'marketmamba' in /content/MarketMamba/.
# Insert in REVERSE order: the last insert wins position 0.
sys.path.insert(0, "/content/MarketMamba")    # pushed to [1] by next line
sys.path.insert(0, "/content/MarketMamba/V6") # ends up at [0] <- V6 wins

# Sanity check: confirm the right config is loaded
_config_path = "/content/MarketMamba/V6/marketmamba/config.py"
if not os.path.exists(_config_path):
    print(f"ERROR: V6 config not found at {_config_path}")
    print("The git clone may not have the V6 directory. Check the repo.")
else:
    print(f"V6 package path: OK")

# 6. Mount Google Drive (for data snapshot backup)
from google.colab import drive
drive.mount("/content/drive")

import torch
gpu  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
print(f"Environment ready | GPU: {gpu} ({vram:.0f} GB) | Torch: {torch.__version__}")


# %% Cell 2: Config & Paths
# ==========================================
# V6 Paths - adjust if needed
# ==========================================
from marketmamba.config import (
    PROCESSED_DIR, MODELS_DIR, DATA_START_DATE, SEQ_LEN, INPUT_DIM
)
print(f"Data range   : {DATA_START_DATE} -> today")
print(f"Model input  : seq_len={SEQ_LEN}, input_dim={INPUT_DIM}")
print(f"Processed dir: {PROCESSED_DIR}")
print(f"Models dir   : {MODELS_DIR}")

# Verify FinMind token reached config
from marketmamba.config import FINMIND_TOKEN
assert FINMIND_TOKEN, "FINMIND_TOKEN not loaded in config! Check Cell 0."
print(f"FinMind token: {FINMIND_TOKEN[:30]}... (OK)")

# Optional: restore data snapshot from Drive to speed up sync
DRIVE_DATA_PATH = "/content/drive/MyDrive/MarketMamba/data_snapshot_v6.tar.gz"
import os
if os.path.exists(DRIVE_DATA_PATH):
    print("Copying data snapshot from Drive...")
    os.makedirs(str(PROCESSED_DIR), exist_ok=True)
    os.system(f"tar xzf {DRIVE_DATA_PATH} -C /")
    print("Snapshot loaded - incremental sync will be much faster")
else:
    print("No Drive snapshot found - will do full sync (30-60 min first time)")


# %% Cell 3: Full Data Sync (2012 to today)
# ==========================================
# Hybrid Data Fetch
# ==========================================
# First run: ~30-60 min (2012 full history)
# Subsequent: ~2-5 min (incremental)

from marketmamba.data.fetcher import run_full_data_sync

FORCE_REBUILD = False   # Set True to re-download everything from scratch

trading_days = run_full_data_sync(force_rebuild=FORCE_REBUILD)
print(f"\nData sync complete")
print(f"   Trading days: {len(trading_days)}")
print(f"   Range: {trading_days[0]} -> {trading_days[-1]}")

# Back up to Drive
print("\nBacking up processed data to Drive...")
os.makedirs("/content/drive/MyDrive/MarketMamba", exist_ok=True)
os.system(f"tar czf /tmp/data_snapshot_v6.tar.gz -C / {str(PROCESSED_DIR).lstrip('/')}")
os.system(f"cp /tmp/data_snapshot_v6.tar.gz {DRIVE_DATA_PATH}")
print("Backup complete")


# %% Cell 4: Feature Matrix + KG Cache
# ==========================================
# Feature Engineering + Knowledge Graph
# ==========================================
import pandas as pd
from marketmamba.data.merger import merge_all_data, validate_data_integrity
from marketmamba.data.feature_engineer import build_features, clean_and_scale
from marketmamba.knowledge.graph_builder import build_knowledge_graph

MATRIX_CACHE = PROCESSED_DIR / "V6_Feature_Matrix.parquet"

if MATRIX_CACHE.exists() and not FORCE_REBUILD:
    print("Loading cached feature matrix...")
    df = pd.read_parquet(MATRIX_CACHE)
else:
    print("Building feature matrix from raw data...")
    data = merge_all_data()
    integrity = validate_data_integrity(data)
    print(f"   Stocks: {integrity.get('n_stocks')} | Dates: {integrity.get('n_dates')}")
    print(f"   Date range: {integrity.get('date_range')}")
    print(f"   Close NaN: {integrity.get('close_na_pct', 0):.1%}")

    df = build_features(
        df_price  = data["prices"],
        df_inst   = data["inst"],
        df_margin = data["margin"],
        df_rev    = data["revenue"],
        df_fin    = data["financials"],
        df_macro  = data["macro"],
    )
    df = clean_and_scale(df)
    df.to_parquet(MATRIX_CACHE)
    print(f"Feature matrix saved: {df.shape}")

print(f"\nFeature matrix: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Date range: {df['Date'].min()} -> {df['Date'].max()}")
print(f"Stocks: {df['stock_id'].nunique()}")

# Knowledge Graph
KG_REBUILD = False
from marketmamba.config import KG_CACHE_PATH

if not KG_CACHE_PATH.exists() or KG_REBUILD:
    print("\nBuilding Knowledge Graph...")
    data_for_kg = merge_all_data()
    df_universe = data_for_kg["prices"][["stock_id"]].drop_duplicates()
    try:
        from marketmamba.data.fetcher import _fetch_universe_from_finmind
        df_univ = _fetch_universe_from_finmind()
        df_universe = df_universe.merge(df_univ[["stock_id", "industry_category"]], on="stock_id", how="left")
    except Exception as e:
        print(f"  Could not fetch sector info: {e}")
        df_universe["industry_category"] = "Unknown"

    build_knowledge_graph(df_universe, data_for_kg["prices"], force_rebuild=KG_REBUILD)
    print(f"KG built -> {KG_CACHE_PATH}")
else:
    print(f"KG loaded from cache -> {KG_CACHE_PATH}")


# %% Cell 5: Walk-Forward Baseline (V5.5)
# ==========================================
# Establish baseline ICIR (optional)
# ==========================================
SKIP_BASELINE = True   # Set False if you have a V5.5 checkpoint

if not SKIP_BASELINE:
    print("V5.5 baseline: implement manually using V5.5 train_model")
else:
    print("Baseline skipped (SKIP_BASELINE=True)")


# %% Cell 6: Train V6 - Single Fold (Quick Test)
# ==========================================
# Quick single-fold training to verify pipeline works
# Use this before committing to the full Walk-Forward
# ==========================================
from marketmamba.models.trainer import train_model

all_dates = sorted(df["Date"].astype(str).unique().tolist())
cutoff_train_end = "2022-12-31"
cutoff_val_end   = "2023-12-31"

train_dates = [d for d in all_dates if d <= cutoff_train_end]
val_dates   = [d for d in all_dates if cutoff_train_end < d <= cutoff_val_end]

print(f"Quick Fold: train={len(train_dates)} days | val={len(val_dates)} days")
print(f"  Train: {train_dates[0]} -> {train_dates[-1]}")
print(f"  Val  : {val_dates[0]}  -> {val_dates[-1]}")

QUICK_EPOCHS = 20   # Use 20 for quick test; set 60 for full training

model, history = train_model(
    df              = df,
    train_dates     = train_dates,
    val_dates       = val_dates,
    epochs          = QUICK_EPOCHS,
    checkpoint_name = "v6_quick_test.pt",
)

# Show results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.train_loss, label="Train", color="#ff4b4b")
axes[0].plot(history.val_loss,   label="Val",   color="#00fa9a")
axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.val_ic, color="#6c5ce7")
axes[1].axhline(0.05, color="green", linestyle="--", label="IC=0.05 threshold")
axes[1].set_title("Val IC (20d Spearman)"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(history.lr, color="#fdcb6e")
axes[2].set_title("Learning Rate"); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nBest Epoch: {history.best_epoch} / {QUICK_EPOCHS}")
print(f"  Val Loss: {history.best_val_loss:.5f}")
print(f"  Best Val IC: {max(history.val_ic):.4f}")
model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model params: {model_n_params:,}")


# %% Cell 7: Full Walk-Forward Validation
# ==========================================
# Expanding-Window Walk-Forward (36 folds)
# WARNING: This takes 4-8 hours on A100
# Start only after Cell 6 confirms pipeline works
# ==========================================
RUN_FULL_WF = False   # Set True when ready for the full run

if RUN_FULL_WF:
    from marketmamba.evaluation.walk_forward import run_walk_forward

    def train_fold(df, train_dates, val_dates):
        m, _ = train_model(df, train_dates, val_dates, epochs=40, checkpoint_name="v6_wf_fold.pt")
        return m

    wf_summary = run_walk_forward(
        df           = df,
        train_fn     = train_fold,
        train_start  = "2012-01-01",
        save_results = True,
    )
    wf_summary.print_report()
else:
    print("Full Walk-Forward skipped (RUN_FULL_WF=False)")
    print("Set RUN_FULL_WF=True when ready for the full 4-8 hour run")


# %% Cell 8: Save Best Model + Push to GitHub
# ==========================================
# Package and distribute the trained model
# ==========================================
import shutil

PUSH_TO_GITHUB = False   # Set True when model is validated

quick_ckpt = MODELS_DIR / "v6_quick_test.pt"
best_ckpt  = MODELS_DIR / "v6_best.pt"
if quick_ckpt.exists():
    shutil.copy(quick_ckpt, best_ckpt)
    print(f"Model checkpoint: {best_ckpt}")

from google.colab import files
if best_ckpt.exists():
    files.download(str(best_ckpt))
    print("Download triggered - save to your MarketMamba/V6/models/ folder")
else:
    print("No checkpoint found - run Cell 6 first")

if PUSH_TO_GITHUB:
    os.chdir("/content/MarketMamba")
    os.system("git add V6/results/")
    os.system('git commit -m "V6 training results update"')
    os.system("git push origin main")
    print("Results pushed to GitHub")
