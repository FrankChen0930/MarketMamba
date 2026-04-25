"""
MarketMamba V6 — Local Feature Engineering
===========================================
Run AFTER fetch_v6_data.py completes.
Produces: Data/processed_v6/V6_Feature_Matrix.parquet

Usage:
    conda activate market_mamba
    python scripts/build_features_local.py

The output file is uploaded to Drive along with the raw parquets,
so Colab Cell 4 loads it from cache in seconds instead of rebuilding.

Estimated runtime: 30-90 min depending on CPU cores (9M rows × 46 features).
"""

import sys
import time
import logging
import warnings
import numpy as np
from pathlib import Path

# ── 0. Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
ROOT_DIR      = SCRIPT_DIR.parent
V6_DIR        = ROOT_DIR / "V6"
PROCESSED_DIR = ROOT_DIR / "Data" / "processed_v6"
OUTPUT_FILE   = PROCESSED_DIR / "V6_Feature_Matrix.parquet"

# Add V6 to sys.path so we can import marketmamba
sys.path.insert(0, str(V6_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT_DIR / "Data" / "build_features.log",
                            encoding="utf-8", mode="w"),
    ],
)
log = logging.getLogger("build_local")

# Suppress the divide-by-zero RuntimeWarning from np.log on 0-valued market caps
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── 1. Check if already built ──────────────────────────────────────────────────
if OUTPUT_FILE.exists():
    import pandas as pd
    df_existing = pd.read_parquet(OUTPUT_FILE)
    log.info(f"V6_Feature_Matrix.parquet already exists: {df_existing.shape}")
    log.info("  Delete it and re-run if you want to rebuild.")
    log.info("  Exiting.")
    sys.exit(0)

# ── 2. Import marketmamba modules ──────────────────────────────────────────────
try:
    from marketmamba.data.merger import merge_all_data, validate_data_integrity
    from marketmamba.data.feature_engineer import build_features, clean_and_scale
    log.info("marketmamba package loaded from V6/")
except ImportError as e:
    sys.exit(f"Cannot import marketmamba: {e}\n  Make sure conda env has pandas/numpy/pyarrow.")

# ── 3. Load all raw data ───────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Step 1/3: Loading raw parquet files...")
log.info("=" * 60)
t0 = time.time()

data = merge_all_data()
integrity = validate_data_integrity(data)
log.info(f"  Stocks : {integrity.get('n_stocks'):,}")
log.info(f"  Dates  : {integrity.get('n_dates'):,}")
log.info(f"  Range  : {integrity.get('date_range')}")
log.info(f"  Close NaN: {integrity.get('close_na_pct', 0):.2%}")
log.info(f"  Loaded in {time.time()-t0:.1f}s")

# ── 4. Build features ─────────────────────────────────────────────────────────
log.info("")
log.info("=" * 60)
log.info("Step 2/3: Building feature matrix (46 dims)...")
log.info("  This takes 30-90 min on a single core.")
log.info("=" * 60)
t1 = time.time()

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

log.info(f"  Raw feature matrix: {df.shape}")
log.info(f"  Built in {(time.time()-t1)/60:.1f} min")

# ── 5. Clean & scale ──────────────────────────────────────────────────────────
log.info("")
log.info("=" * 60)
log.info("Step 3/3: Cross-sectional winsorize + z-score...")
log.info("=" * 60)
t2 = time.time()

df = clean_and_scale(df)
log.info(f"  Final matrix: {df.shape}")
log.info(f"  Cleaned in {(time.time()-t2)/60:.1f} min")

# ── 6. Save ───────────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT_FILE)
size_mb = OUTPUT_FILE.stat().st_size / 1_048_576
log.info("")
log.info("=" * 60)
log.info(f"Saved: {OUTPUT_FILE}")
log.info(f"  Size : {size_mb:.1f} MB")
log.info(f"  Shape: {df.shape}")
log.info(f"  Dates: {df['Date'].min()} -> {df['Date'].max()}")
log.info(f"  Stocks: {df['stock_id'].nunique():,}")
log.info(f"  Total elapsed: {(time.time()-t0)/60:.1f} min")
log.info("")
log.info("Next step:")
log.info("  Re-zip Data/processed_v6/ and upload to Drive.")
log.info("  Colab Cell 4 will load from cache (~10 sec).")
log.info("=" * 60)
