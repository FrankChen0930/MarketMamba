# %% Cell 4b: FULL RETRAIN — Production Model
# ==========================================
# Use ALL data for training. No early stopping.
# Fixed epoch count = best epoch from validated run.
#
# 💡 IMPORTANT: This is a SELF-CONTAINED script.
#    Run it in Colab with:  %run /content/MarketMamba/V6/notebooks/V6_FullRetrain_Cell.py
#    Or copy-paste into a new Colab cell (after Cell 1 + Cell 2 have run).
#
# 📝 HOW IT WORKS:
#    Phase ① (already done): Normal train/val split → found best epoch = 14
#    Phase ② (this cell):    Train on ALL data for exactly 14 epochs
#                            → produces the final production model
#
# ⚠️  Rules for Full Retrain:
#    - Hyperparameters are IDENTICAL to Phase ①
#    - Epoch count is FIXED at 14 (from Phase ① best IC epoch)
#    - Early stopping is DISABLED (no val set to judge)
#    - Val set is kept only for monitoring loss/IC curves (not for decisions)
# ==========================================

import os, sys, shutil
import pandas as pd
import torch

# ── Ensure Python path is set (same as Cell 1) ──
for _k in list(sys.modules.keys()):
    if _k == "marketmamba" or _k.startswith("marketmamba."):
        del sys.modules[_k]
for _p in ["/content/MarketMamba/V6", "/content/MarketMamba"]:
    while _p in sys.path: sys.path.remove(_p)
sys.path.insert(0, "/content/MarketMamba")
sys.path.insert(0, "/content/MarketMamba/V6")

from marketmamba.models.trainer import train_model
from marketmamba.config import PROCESSED_DIR, MODELS_DIR

# ── Load Feature Matrix (same logic as Cell 3) ──
DRIVE_V6_DIR = "/content/drive/MyDrive/MarketMamba_V6"
os.makedirs(DRIVE_V6_DIR, exist_ok=True)
DRIVE_FEATURE_CACHE = f"{DRIVE_V6_DIR}/V6_Feature_Matrix.parquet"
DRIVE_CKPT_DIR = f"{DRIVE_V6_DIR}/checkpoints"
os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

MATRIX_CACHE = PROCESSED_DIR / "V6_Feature_Matrix.parquet"

# Restore from Drive if not in local cache
if not MATRIX_CACHE.exists() and os.path.exists(DRIVE_FEATURE_CACHE):
    print("Restoring feature matrix from Drive...")
    os.makedirs(str(PROCESSED_DIR), exist_ok=True)
    shutil.copy(DRIVE_FEATURE_CACHE, str(MATRIX_CACHE))
    print(f"  Restored ({os.path.getsize(DRIVE_FEATURE_CACHE) / 1e9:.2f} GB)")

if MATRIX_CACHE.exists():
    print("Loading cached feature matrix...")
    df = pd.read_parquet(MATRIX_CACHE)
    n_features = df.shape[1] - 5
    print(f"✅ Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols ({n_features} features)")
    print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"   Stocks: {df['stock_id'].nunique():,}")
else:
    raise FileNotFoundError(
        f"Feature matrix not found at {MATRIX_CACHE}!\n"
        "Please run Cell 2 (Restore Data) and Cell 3 (Build Feature Matrix) first."
    )

# ── Build KG if needed (same logic as Cell 3b) ──
from marketmamba.knowledge.graph_builder import build_knowledge_graph
from marketmamba.config import KG_CACHE_PATH

DRIVE_KG_CACHE = f"{DRIVE_V6_DIR}/knowledge_graph_cache.npz"
if not KG_CACHE_PATH.exists() and os.path.exists(DRIVE_KG_CACHE):
    print("Restoring KG from Drive...")
    os.makedirs(str(KG_CACHE_PATH.parent), exist_ok=True)
    shutil.copy(DRIVE_KG_CACHE, str(KG_CACHE_PATH))
    print("  ✅ KG restored from Drive")

if not KG_CACHE_PATH.exists():
    print("Building Knowledge Graph...")
    df_prices = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet")
    df_universe = df_prices[["stock_id"]].drop_duplicates()
    stock_info_path = PROCESSED_DIR / "stock_info.parquet"
    if stock_info_path.exists():
        df_info = pd.read_parquet(stock_info_path)
        df_universe = df_universe.merge(
            df_info[["stock_id", "industry_category"]], on="stock_id", how="left"
        )
        df_universe["industry_category"] = df_universe["industry_category"].fillna("Unknown")
    else:
        df_universe["industry_category"] = "Unknown"
    build_knowledge_graph(df_universe, df_prices, force_rebuild=True)
    if KG_CACHE_PATH.exists():
        shutil.copy(str(KG_CACHE_PATH), DRIVE_KG_CACHE)
        print("  ✅ KG built & backed up to Drive")
    del df_prices, df_universe
    import gc; gc.collect()
else:
    print("✅ KG loaded from cache")

# ── Full Retrain Configuration ──────────────────────────────────────
FULL_RETRAIN       = True
BEST_EPOCH_PHASE1  = 14    # ← from Phase ① validated training (best IC @ep14)

all_dates = sorted(df["Date"].astype(str).unique().tolist())

if FULL_RETRAIN:
    # Phase ②: use ALL dates for training
    train_dates = all_dates
    # Val dates: use the last 20 trading days (monitor only, NOT for early stopping)
    val_dates   = all_dates[-20:]
    # Fixed epoch count from Phase ①
    FINAL_EPOCHS   = BEST_EPOCH_PHASE1
    # Disable early stopping (set patience way beyond epoch count)
    EARLY_STOP_IC  = 99999
    print("=" * 60)
    print("🔥 FULL RETRAIN MODE — Production Model")
    print("=" * 60)
    print(f"  ➤ Using ALL {len(train_dates)} training days")
    print(f"  ➤ Fixed epoch count: {FINAL_EPOCHS} (from Phase ① best)")
    print(f"  ➤ Early stopping: DISABLED")
    print(f"  ➤ Val dates (monitor only): last {len(val_dates)} days")
    print("=" * 60)
else:
    # Phase ①: normal train/val split (fallback)
    cutoff_train_end = "2023-12-31"
    train_dates = [d for d in all_dates if d <= cutoff_train_end]
    val_dates   = [d for d in all_dates if d > cutoff_train_end]
    FINAL_EPOCHS   = 100
    EARLY_STOP_IC  = 15

N_SAMPLE_TRAIN = 2000   # Sample 2000 stocks per batch to prevent OOM
                        # Full Retrain uses ALL dates (incl. 2024-2026) which have more
                        # listed stocks per cross-section → needs sampling to fit in GPU memory

from marketmamba import config as _cfg
_cfg.N_SAMPLE_TRAIN = N_SAMPLE_TRAIN

print(f"\nTraining Setup:")
print(f"  Mode:  {'FULL RETRAIN (Phase ②)' if FULL_RETRAIN else 'Validated (Phase ①)'}")
print(f"  Train: {len(train_dates)} days ({train_dates[0]} → {train_dates[-1]})")
print(f"  Val:   {len(val_dates)} days ({val_dates[0]} → {val_dates[-1]})")
print(f"  Epochs: {FINAL_EPOCHS} (early stop patience={EARLY_STOP_IC})")
print(f"  N_SAMPLE: {N_SAMPLE_TRAIN or 'ALL'}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── Live plot callback ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

def live_plot(history, epoch, epochs):
    clear_output(wait=True)
    ep = list(range(1, len(history.train_loss) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#eee")
        ax.title.set_color("#eee")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")

    # Loss
    axes[0].plot(ep, history.train_loss, color="#ff6b6b", lw=2, label="Train")
    axes[0].plot(ep, history.val_loss,   color="#00fa9a", lw=2, label="Val")
    axes[0].set_title("Loss"); axes[0].legend(facecolor="#222", labelcolor="#eee"); axes[0].grid(alpha=0.2)

    # IC
    axes[1].plot(ep, history.val_ic, color="#a29bfe", lw=2)
    axes[1].axhline(0, color="#636e72", ls="--", lw=1)
    axes[1].axhline(0.05, color="#00cec9", ls="--", lw=1, label="IC=0.05")
    best_ic = max(history.val_ic)
    best_ep = history.best_ic_epoch
    axes[1].axvline(best_ep, color="#fdcb6e", ls=":", lw=1, label=f"best={best_ep}")

    # Mode indicator
    mode_str = "FULL RETRAIN" if FULL_RETRAIN else "Validated"
    axes[1].set_title(f"Val IC  best={best_ic:+.4f} @ep{best_ep} [{mode_str}]")
    axes[1].legend(facecolor="#222", labelcolor="#eee"); axes[1].grid(alpha=0.2)

    # LR
    axes[2].plot(ep, history.lr, color="#fdcb6e", lw=2)
    axes[2].set_title("LR"); axes[2].set_yscale("log"); axes[2].grid(alpha=0.2)

    no_impr = epoch - best_ep
    title = f"Ep {epoch}/{epochs} | train={history.train_loss[-1]:.5f} val={history.val_loss[-1]:.5f} IC={history.val_ic[-1]:+.4f} | best@{best_ep} ({no_impr}ep no-impr)"
    if FULL_RETRAIN:
        title = f"🔥 FULL RETRAIN | {title}"
    fig.suptitle(title, color="#eee", fontsize=11)
    plt.tight_layout(); display(fig); plt.close(fig)

# ── Train ──
print(f"\n🚀 Starting {'FULL RETRAIN' if FULL_RETRAIN else 'training'}...")
print(f"   ⚡ Checkpoints auto-backup to Drive on every IC improvement")
model, history = train_model(
    df              = df,
    train_dates     = train_dates,
    val_dates       = val_dates,
    epochs          = FINAL_EPOCHS,
    early_stop      = EARLY_STOP_IC,
    checkpoint_name = "v6_final.pt",
    on_epoch_end    = live_plot,
    ic_mode         = True,
    checkpoint_backup_dir = DRIVE_CKPT_DIR,   # ← immediate Drive backup!
)

# ── Save summary ──
live_plot(history, len(history.train_loss), FINAL_EPOCHS)
print(f"\n✅ Training complete!")
print(f"   Mode:       {'FULL RETRAIN (Production)' if FULL_RETRAIN else 'Validated (Phase ①)'}")
print(f"   Best Epoch: {history.best_epoch} / {len(history.train_loss)}")
print(f"   Val Loss:   {history.best_val_loss:.5f}")
print(f"   Best IC:    {max(history.val_ic):+.4f}")
print(f"   Checkpoint: {DRIVE_CKPT_DIR}/v6_final.pt (already on Drive)")
if FULL_RETRAIN:
    print(f"\n   🔥 This is your PRODUCTION model — trained on ALL data for {FINAL_EPOCHS} epochs.")
    print(f"   🔥 Deploy this v6_final.pt for daily inference.")
