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

# FinMind Sponsor token (update after each renewal)
os.environ["FINMIND_TOKEN"] = "YOUR_SPONSOR_TOKEN_HERE"  # <-- paste new token

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
_torch_ver = torch.__version__.split("+")[0].replace(".", "")       # e.g. "2100"
_py_ver    = f"cp{sys.version_info.major}{sys.version_info.minor}"  # e.g. "cp312"
_wheel_key = f"cu{_cuda_ver}torch{_torch_ver}{_py_ver}"            # e.g. "cu128torch2100cp312"

# Wheels stored in a VERSION SUBDIRECTORY so filenames stay PEP-427 compliant.
# Drive/mamba_wheels/<key>/mamba_ssm-2.3.1-cp312-cp312-linux_x86_64.whl
DRIVE_WHEEL_BASE = "/content/drive/MyDrive/MarketMamba/mamba_wheels"
DRIVE_WHEEL_DIR  = f"{DRIVE_WHEEL_BASE}/{_wheel_key}"  # key = subdir, not filename suffix
WHEEL_BUILD_DIR  = "/tmp/mamba_wheels"
os.makedirs(WHEEL_BUILD_DIR, exist_ok=True)

print(f"Mamba wheel key : {_wheel_key}")
print(f"Drive wheel dir : {DRIVE_WHEEL_DIR}")
print("Installing mamba_ssm...")

_installed = False

# --- Layer 1: Drive cache (subdir keeps original PEP-427 filenames) ---
_cached = glob.glob(f"{DRIVE_WHEEL_DIR}/*.whl") if os.path.isdir(DRIVE_WHEEL_DIR) else []
if _cached:
    print(f"  Found {len(_cached)} cached wheel(s) - installing (fast path)...")
    for whl in _cached:
        rc1 = os.system(f"pip install -q {whl}")
        print(f"    {os.path.basename(whl)} (rc={rc1})")
    # Verify the install actually worked before claiming success
    try:
        import importlib, importlib.util
        if importlib.util.find_spec("mamba_ssm") is not None:
            _installed = True
            print("  Installed from Drive cache")
        else:
            print("  Drive cache install FAILED (module not found) - falling through...")
    except Exception:
        print("  Drive cache install FAILED - falling through...")
else:
    _all = glob.glob(f"{DRIVE_WHEEL_BASE}/**/*.whl", recursive=True)
    print(f"  No wheel for key={_wheel_key} in Drive")
    if _all:
        keys = list(set(os.path.basename(os.path.dirname(w)) for w in _all))
        print(f"  (Other cached keys: {keys})")

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
        rc_inst = os.system(f"pip install -q {WHEEL_BUILD_DIR}/*.whl")
        print(f"  Install rc={rc_inst}")

        # Save mamba_ssm + causal_conv1d to Drive SUBDIR (keep original filename)
        os.makedirs(DRIVE_WHEEL_DIR, exist_ok=True)
        for whl_path in glob.glob(f"{WHEEL_BUILD_DIR}/*.whl"):
            whl_name = os.path.basename(whl_path)
            if any(pkg in whl_name.lower() for pkg in ["mamba_ssm", "causal_conv1d"]):
                dest = f"{DRIVE_WHEEL_DIR}/{whl_name}"   # original name, no suffix mangling
                shutil.copy2(whl_path, dest)
                print(f"  Saved to Drive: {dest}")
        _installed = True
    else:
        print("  Source build failed - make sure GPU runtime is enabled in Colab")

# --- Verify ---
try:
    from mamba_ssm import Mamba
    print("mamba_ssm OK")
except ImportError as e:
    _err_str = str(e)
    if "is_opaque_value" in _err_str or "torch._library" in _err_str or "opaque_object" in _err_str:
        # PyTorch 2.10+ changed internal APIs - mamba_ssm 2.3.x not yet updated.
        # Fix: install the latest mamba_ssm directly from GitHub main branch.
        print(f"mamba_ssm import failed (PyTorch API change): {_err_str}")

        # Remove the incompatible wheel subdir from Drive
        if os.path.isdir(DRIVE_WHEEL_DIR):
            shutil.rmtree(DRIVE_WHEEL_DIR)
            print(f"  Removed incompatible cached wheel dir: {DRIVE_WHEEL_DIR}")

        print("  Installing latest mamba_ssm from GitHub (~5-10 min)...")
        rc_git = os.system(
            f"pip wheel git+https://github.com/state-spaces/mamba.git "
            f"git+https://github.com/Dao-AILab/causal-conv1d.git "
            f"--no-build-isolation --no-cache-dir "
            f"-w {WHEEL_BUILD_DIR}/ "
            f"2>&1 | tail -5"
        )
        if rc_git == 0:
            os.system(f"pip install -q {WHEEL_BUILD_DIR}/*.whl")
            # Save to Drive SUBDIR (original filename)
            os.makedirs(DRIVE_WHEEL_DIR, exist_ok=True)
            for whl_path in glob.glob(f"{WHEEL_BUILD_DIR}/*.whl"):
                whl_name = os.path.basename(whl_path)
                if any(pkg in whl_name.lower() for pkg in ["mamba_ssm", "causal_conv1d"]):
                    shutil.copy2(whl_path, f"{DRIVE_WHEEL_DIR}/{whl_name}")
                    print(f"  Saved GitHub wheel to Drive: {whl_name}")

        import importlib
        try:
            if "mamba_ssm" in sys.modules:
                importlib.reload(sys.modules["mamba_ssm"])
            from mamba_ssm import Mamba
            print("mamba_ssm OK (from GitHub latest)")
        except ImportError as e2:
            print(f"WARNING: still cannot import mamba_ssm: {e2}")
            print("  Cells 0-4 (data sync) can still run.")
    else:
        print(f"WARNING: mamba_ssm not available: {_err_str}")
        print("  Cells 0-4 (data sync) can still run without it.")
        print("  Cell 6+ (training) requires mamba_ssm.")



# 5. Python path
# IMPORTANT: V6 must end up at sys.path[0] so its 'marketmamba' package
# takes priority over the OLD V5.5 'marketmamba' in /content/MarketMamba/.
# Insert in REVERSE order: the last insert wins position 0.
sys.path.insert(0, "/content/MarketMamba")    # pushed to [1] by next line
sys.path.insert(0, "/content/MarketMamba/V6") # ends up at [0] <- V6 wins

# CRITICAL: purge any cached 'marketmamba' module Python already loaded.
# Without this, Python ignores the new sys.path and keeps using the old V5.5 package.
# This happens when torch, pip, or any earlier import accidentally triggered a
# marketmamba import and cached it in sys.modules.
_purged = [k for k in list(sys.modules.keys()) if k == "marketmamba" or k.startswith("marketmamba.")]
for _k in _purged:
    del sys.modules[_k]
if _purged:
    print(f"  Purged {len(_purged)} cached marketmamba module(s): {_purged}")

# Sanity check
_config_path = "/content/MarketMamba/V6/marketmamba/config.py"
if not os.path.exists(_config_path):
    print(f"ERROR: V6 config not found at {_config_path}")
else:
    # Quick import test to confirm the right package is now visible
    import importlib.util
    _spec = importlib.util.spec_from_file_location("marketmamba.config", _config_path)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    assert hasattr(_mod, "MODELS_DIR"), "V6 config missing MODELS_DIR"
    print(f"V6 package path: OK (MODELS_DIR confirmed)")

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
import sys

# Safety: re-purge sys.modules at the start of every cell that imports marketmamba.
# Colab cells share the same Python process but can be re-run independently,
# so this ensures we always pick up the V6 package.
for _k in list(sys.modules.keys()):
    if _k == "marketmamba" or _k.startswith("marketmamba."):
        del sys.modules[_k]

# Ensure V6 is still first in sys.path (idempotent)
for _p in ["/content/MarketMamba/V6", "/content/MarketMamba"]:
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, "/content/MarketMamba")
sys.path.insert(0, "/content/MarketMamba/V6")

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

# ── Drive snapshot restore ───────────────────────────────────────────────────
# Upload Data/processed_v6.zip to:
#   /MyDrive/MarketMamba_V6/processed_v6.zip
# This cell restores it so Cell 3+ can skip all FinMind fetching.

DRIVE_SNAPSHOT = "/content/drive/MyDrive/MarketMamba_V6/processed_v6.zip"
DRIVE_DATA_PATH = DRIVE_SNAPSHOT   # kept for Cell 3 backup reference

import os, time

# Force Drive to sync the directory before checking (Colab lazy-mounts)
_v6_dir = "/content/drive/MyDrive/MarketMamba_V6"
try:
    _files_in_dir = os.listdir(_v6_dir) if os.path.isdir(_v6_dir) else []
    print(f"Drive MarketMamba_V6/ contents: {_files_in_dir}")
except Exception as _e:
    print(f"Cannot list {_v6_dir}: {_e}")

if os.path.exists(DRIVE_SNAPSHOT):
    print("Restoring data snapshot from Drive...")
    os.makedirs(str(PROCESSED_DIR), exist_ok=True)
    rc = os.system(f"unzip -q -o {DRIVE_SNAPSHOT} -d {PROCESSED_DIR}")
    if rc == 0:
        n = len(list(PROCESSED_DIR.glob("*.parquet")))
        print(f"Snapshot restored: {n} parquet files in {PROCESSED_DIR}")
    else:
        print("unzip failed - trying python fallback...")
        import zipfile
        with zipfile.ZipFile(DRIVE_SNAPSHOT, 'r') as zf:
            zf.extractall(PROCESSED_DIR)
        print("Snapshot restored via Python zipfile")
else:
    print(f"Drive snapshot not found at: {DRIVE_SNAPSHOT}")
    print("Please upload Data/processed_v6.zip to that Drive path.")
    print("Cell 3 will fail if prices_raw.parquet is missing.")


# %% Cell 3: Verify Data Snapshot
# ==========================================
# Data was fetched locally and uploaded to Drive.
# This cell just verifies the parquet files are present
# and extracts the trading day list.
# ==========================================
import pandas as pd
from pathlib import Path

FORCE_REBUILD = False   # Set True to force feature re-engineering in Cell 4

prices_path = PROCESSED_DIR / "prices_raw.parquet"
if not prices_path.exists():
    raise FileNotFoundError(
        "prices_raw.parquet not found!\n"
        "Make sure you:\n"
        "  1. Uploaded Data/processed_v6.zip to /MyDrive/MarketMamba_V6/ on Drive\n"
        "  2. Re-ran Cell 2 (it unzips the snapshot)"
    )

df_cal = pd.read_parquet(prices_path, columns=["Date"]).drop_duplicates()
trading_days = sorted(df_cal["Date"].astype(str).unique().tolist())

print(f"Data snapshot verified")
print(f"  Trading days : {len(trading_days):,}")
print(f"  Range        : {trading_days[0]} -> {trading_days[-1]}")

# List all available parquet files
print("\nAvailable raw files:")
for f in sorted(PROCESSED_DIR.glob("*.parquet")):
    print(f"  {f.name:<40} {f.stat().st_size / 1_048_576:.1f} MB")


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
        df_price         = data["prices"],
        df_inst          = data["inst"],
        df_margin        = data["margin"],
        df_per           = data["per"],
        df_securities    = data["securities"],
        df_market_value  = data["market_value"],
        df_daytrade      = data["daytrade"],
        df_holdings      = data["holdings"],
        df_rev           = data["revenue"],
        df_fin           = data["financials"],
        df_balance_sheet = data["balance_sheet"],
        df_cashflow      = data["cashflow"],
        df_macro         = data["macro"],
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


# %% Cell 6: Train V6 - Final High-Quality Training
# ==========================================
# Architecture verified (IC=+0.0744 on Quick Fold).
# Now train with maximum data and quality settings:
#   - Full training window: 2005~2023
#   - True out-of-sample val: 2024~today
#   - N_SAMPLE_TRAIN = None → all stocks (test memory first)
#   - IC-based early stopping (saves best-IC model, not best-loss)
#   - Listnet uses mean (N-invariant scale)
# ==========================================
from marketmamba.models.trainer import train_model

all_dates = sorted(df["Date"].astype(str).unique().tolist())
cutoff_train_end = "2023-12-31"   # one full extra year vs Quick Fold
cutoff_val_end   = "2099-12-31"   # everything after → 2024~today

train_dates = [d for d in all_dates if d <= cutoff_train_end]
val_dates   = [d for d in all_dates if d >  cutoff_train_end]

print(f"Final Training: train={len(train_dates)} days | val={len(val_dates)} days")
print(f"  Train: {train_dates[0]} -> {train_dates[-1]}")
print(f"  Val  : {val_dates[0]}  -> {val_dates[-1]}")

FINAL_EPOCHS   = 100   # IC-based early stop (patience=15) decides actual stopping point
N_SAMPLE_TRAIN = None  # Use ALL stocks — if OOM, change to 2000 then 1500
EARLY_STOP_IC  = 15    # patience: stop if IC doesn't improve for 15 consecutive epochs

# Override config at runtime
from marketmamba import config as _cfg
_cfg.N_SAMPLE_TRAIN = N_SAMPLE_TRAIN
print(f"N_SAMPLE_TRAIN → {_cfg.N_SAMPLE_TRAIN} (None = all stocks)")


# ── Live visualization callback ────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")   # Colab-safe: prevents blank inline figure issues
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

def live_plot(history, epoch, epochs):
    """Updates training charts in-place after each epoch."""
    clear_output(wait=True)

    ep = list(range(1, len(history.train_loss) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#eee")
        ax.title.set_color("#eee")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Loss
    axes[0].plot(ep, history.train_loss, color="#ff6b6b", linewidth=2, label="Train")
    axes[0].plot(ep, history.val_loss,   color="#00fa9a", linewidth=2, label="Val")
    axes[0].set_title("Loss")
    axes[0].legend(facecolor="#222", labelcolor="#eee")
    axes[0].grid(alpha=0.2)

    # Val IC — use best_ic_epoch for accurate no_impr display
    axes[1].plot(ep, history.val_ic, color="#a29bfe", linewidth=2)
    axes[1].axhline(0.0,  color="#636e72", linestyle="--", linewidth=1)
    axes[1].axhline(0.05, color="#00cec9", linestyle="--", linewidth=1, label="IC=0.05")
    best_ic = max(history.val_ic)
    best_ic_ep = history.best_ic_epoch   # argmax(val_ic) — correct for ic_mode
    axes[1].axvline(best_ic_ep, color="#fdcb6e", linestyle=":", linewidth=1, label=f"best ep={best_ic_ep}")
    axes[1].set_title(f"Val IC   best={best_ic:+.4f} @ ep{best_ic_ep}")
    axes[1].legend(facecolor="#222", labelcolor="#eee")
    axes[1].grid(alpha=0.2)

    # Learning Rate
    axes[2].plot(ep, history.lr, color="#fdcb6e", linewidth=2)
    axes[2].set_title("Learning Rate")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.2)

    # no_impr: epochs since best IC (not best loss)
    no_impr = epoch - best_ic_ep
    suptitle = (
        f"Epoch {epoch}/{epochs}  |  "
        f"train={history.train_loss[-1]:.5f}  val={history.val_loss[-1]:.5f}  "
        f"IC={history.val_ic[-1]:+.4f}  |  "
        f"best_IC_epoch={best_ic_ep}  ({no_impr} ep no-improve)"
    )
    fig.suptitle(suptitle, color="#eee", fontsize=11)
    plt.tight_layout()
    display(fig)
    plt.close(fig)

# ── Run training ───────────────────────────────────────────────────────────────
# NOTE: If OOM on first batch, interrupt and set N_SAMPLE_TRAIN=2000 above, then retry
model, history = train_model(
    df              = df,
    train_dates     = train_dates,
    val_dates       = val_dates,
    epochs          = FINAL_EPOCHS,
    early_stop      = EARLY_STOP_IC,
    checkpoint_name = "v6_final.pt",     # separate file from quick test
    on_epoch_end    = live_plot,
    ic_mode         = True,              # IC-based early stop + checkpointing
)

# ── Final summary ─────────────────────────────────────────────────────────────
live_plot(history, len(history.train_loss), FINAL_EPOCHS)   # final static version

print(f"\nBest Epoch : {history.best_epoch} / {len(history.train_loss)}")
print(f"Val Loss   : {history.best_val_loss:.5f}")
print(f"Best Val IC: {max(history.val_ic):+.4f}")
model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params: {model_n_params:,}")


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
        train_start  = "2005-01-01",   # matches DATA_START_DATE
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

# Cell 6 saves the best-IC checkpoint as 'v6_final.pt'
# Cell 7 (walk-forward) saves fold checkpoints as 'v6_wf_fold.pt'
final_ckpt = MODELS_DIR / "v6_final.pt"
best_ckpt  = MODELS_DIR / "v6_best.pt"

if final_ckpt.exists():
    shutil.copy(final_ckpt, best_ckpt)
    print(f"✅ Checkpoint ready: {best_ckpt}")
else:
    print(f"❌ v6_final.pt not found at {MODELS_DIR}")
    print("   Make sure Cell 6 completed successfully (should print '✅ Checkpoint saved')")

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
