"""
MarketMamba V6.1 — Local Feature Matrix Builder
==================================================
Builds V6_Feature_Matrix.parquet on local machine (no GPU needed).
This replaces Cell 4 of V6_Training.py for local execution.

Usage:
    python V6/scripts/build_feature_matrix.py
    python V6/scripts/build_feature_matrix.py --force   # rebuild even if cached

Output:
    Data/processed_v6/V6_Feature_Matrix.parquet (~2.5 GB)
"""

import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import time
import logging
from pathlib import Path

# ── Ensure V6 package is importable ──
ROOT = Path(__file__).resolve().parent.parent  # V6/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # MarketMamba/

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BuildMatrix")

import pandas as pd
from marketmamba.config import PROCESSED_DIR, INPUT_DIM
from marketmamba.data.merger import merge_all_data
from marketmamba.data.feature_engineer import build_features, clean_and_scale

MATRIX_CACHE = PROCESSED_DIR / "V6_Feature_Matrix.parquet"
FORCE = "--force" in sys.argv


def validate_data_integrity(data: dict) -> dict:
    """Quick data integrity check (same as V6_Training.py)."""
    df_p = data["prices"]
    return {
        "n_stocks": df_p["stock_id"].nunique(),
        "n_dates": df_p["Date"].nunique(),
        "date_range": f"{df_p['Date'].min()} -> {df_p['Date'].max()}",
        "close_na_pct": df_p["Close"].isna().mean() if "Close" in df_p.columns else 0,
    }


def main():
    log.info("=" * 65)
    log.info("  MarketMamba V6.1 — Local Feature Matrix Builder")
    log.info(f"  Target INPUT_DIM: {INPUT_DIM}")
    log.info(f"  Output: {MATRIX_CACHE}")
    log.info(f"  Force rebuild: {FORCE}")
    log.info("=" * 65)

    if MATRIX_CACHE.exists() and not FORCE:
        size = MATRIX_CACHE.stat().st_size / 1e9
        log.info(f"Feature matrix already exists ({size:.2f} GB)")
        log.info("Use --force to rebuild")

        # Quick validation
        df = pd.read_parquet(MATRIX_CACHE)
        log.info(f"  Shape: {df.shape}")
        log.info(f"  Dates: {df['Date'].nunique()}")
        log.info(f"  Stocks: {df['stock_id'].nunique()}")
        n_features = df.shape[1] - 2  # minus Date, stock_id
        log.info(f"  Features: {n_features} (target: {INPUT_DIM})")
        if n_features != INPUT_DIM:
            log.warning(f"  Feature count mismatch! {n_features} != {INPUT_DIM}")
            log.warning(f"  Run with --force to rebuild with V6.1 features")
        return

    # ── Step 1: Load all data ──
    t0 = time.time()
    log.info("")
    log.info("Step 1: Loading raw data sources...")
    data = merge_all_data()

    integrity = validate_data_integrity(data)
    log.info(f"  Stocks: {integrity['n_stocks']}")
    log.info(f"  Dates: {integrity['n_dates']}")
    log.info(f"  Range: {integrity['date_range']}")
    log.info(f"  Close NaN: {integrity['close_na_pct']:.1%}")

    # Show what V6.1 sources are available
    v61_sources = {
        "futures_inst": "期貨三大法人",
        "options_inst": "選擇權三大法人",
        "dividend": "股利政策",
        "foreign_shareholding": "外資持股比例",
        "fear_greed": "恐懼貪婪指數",
        "business_indicator": "景氣燈號",
        "fed_rate": "FED利率",
    }
    log.info("")
    log.info("  V6.1 data sources:")
    for key, desc in v61_sources.items():
        val = data.get(key)
        if val is not None:
            log.info(f"    [OK] {desc}: {len(val):,} rows")
        else:
            log.info(f"    [--] {desc}: not available (will use default)")

    # ── Step 2: Build features ──
    log.info("")
    log.info("Step 2: Building features (46D -> 56D)...")
    log.info("  This may take 5-15 minutes depending on data size...")

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
        # V6.1 new data sources
        df_futures_inst  = data.get("futures_inst"),
        df_options_inst  = data.get("options_inst"),
        df_dividend      = data.get("dividend"),
        df_foreign_shareholding = data.get("foreign_shareholding"),
        df_fear_greed    = data.get("fear_greed"),
        df_business_indicator = data.get("business_indicator"),
        df_fed_rate      = data.get("fed_rate"),
    )

    t1 = time.time()
    log.info(f"  Feature engineering done ({t1 - t0:.0f}s)")
    log.info(f"  Shape before scaling: {df.shape}")

    # ── Step 3: Clean & Scale ──
    log.info("")
    log.info("Step 3: Cleaning and scaling...")
    df = clean_and_scale(df)

    t2 = time.time()
    log.info(f"  Done ({t2 - t1:.0f}s)")

    # ── Step 4: Save ──
    log.info("")
    log.info("Step 4: Saving Feature Matrix...")
    df.to_parquet(MATRIX_CACHE)
    size = MATRIX_CACHE.stat().st_size / 1e9
    log.info(f"  Saved: {MATRIX_CACHE.name} ({size:.2f} GB)")

    # ── Summary ──
    total_time = time.time() - t0
    n_features = df.shape[1] - 2  # minus Date, stock_id
    log.info("")
    log.info("=" * 65)
    log.info("  BUILD COMPLETE")
    log.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    log.info(f"  Features: {n_features} (target: {INPUT_DIM})")
    log.info(f"  Dates: {df['Date'].nunique():,}")
    log.info(f"  Stocks: {df['stock_id'].nunique():,}")
    log.info(f"  Size: {size:.2f} GB")
    log.info(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)")

    if n_features != INPUT_DIM:
        log.warning(f"  MISMATCH: got {n_features} features, expected {INPUT_DIM}")
    else:
        log.info(f"  Feature count matches INPUT_DIM={INPUT_DIM}")

    log.info("")
    log.info("  Next: re-zip and upload to Drive")
    log.info(f"  powershell: Compress-Archive -Path '{PROCESSED_DIR}' -DestinationPath '{PROCESSED_DIR}.zip' -Force")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
