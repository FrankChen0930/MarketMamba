# -*- coding: utf-8 -*-
"""
MarketMamba V6.1 — Refetch foreign_shareholding_raw.parquet
============================================================
The cached foreign_shareholding_raw.parquet only has data from 2018 onwards.
This script forces a full re-fetch from FinMind (TaiwanStockShareholding).
FinMind free tier data for this dataset starts around 2016-01-01.

Usage:
    cd D:\\Desktop\\work\\ProjectForMe\\MarketMamba\\V6
    python scripts/refetch_shareholding.py

Expected runtime: ~3-10 minutes (API rate limited to ~60 req/min).
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from marketmamba.config import PROCESSED_DIR, DATA_START_DATE, FINMIND_TOKEN
from marketmamba.data.fetcher import _finmind_fetch_chunked

print("=" * 60)
print("  Refetching foreign_shareholding_raw.parquet")
print("=" * 60)

if not FINMIND_TOKEN:
    print("\n[ERROR] FINMIND_TOKEN is not set in .env file!")
    print("  Please add: FINMIND_TOKEN=your_token_here")
    sys.exit(1)

import requests
from datetime import date

# ── Step 1: 先探測 FinMind 最早能給幾年的資料 ──
print("\n[1/3] Probing FinMind dataset availability...")

FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"
test_years = ["2005-01-01", "2010-01-01", "2015-01-01", "2016-01-01",
              "2017-01-01", "2018-01-01"]

earliest_year = None
for start_probe in test_years:
    try:
        params = {
            "dataset":    "TaiwanStockShareholding",
            "start_date": start_probe,
            "end_date":   (pd.to_datetime(start_probe) + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            "token":      FINMIND_TOKEN,
        }
        r = requests.get(FINMIND_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        n = len(data.get("data", []))
        print(f"  {start_probe}: {n} rows returned")
        if n > 0 and earliest_year is None:
            earliest_year = start_probe
        time.sleep(0.5)
    except Exception as e:
        print(f"  {start_probe}: ERROR - {e}")

if earliest_year is None:
    print("\n[WARNING] No data found for any probe year. Trying 2018-01-01 as fallback...")
    earliest_year = "2018-01-01"

print(f"\n  Earliest available year: {earliest_year}")

# ── Step 2: Full fetch from earliest available year ──
end_date = date.today().strftime("%Y-%m-%d")
print(f"\n[2/3] Fetching {earliest_year} -> {end_date} (chunked yearly)...")

df = _finmind_fetch_chunked(
    "TaiwanStockShareholding",
    start_date=earliest_year,
    end_date=end_date,
    chunk_years=1,
)

if df is None or df.empty:
    print("\n[ERROR] No data returned from FinMind!")
    print("  Possible causes:")
    print("  - FINMIND_TOKEN has expired or is wrong")
    print("  - FinMind servers are down")
    print("  - Dataset name changed")
    sys.exit(1)

# ── Step 3: Clean and save ──
print(f"\n[3/3] Cleaning and saving...")
print(f"  Raw shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# Normalize date column
if "date" in df.columns and "Date" not in df.columns:
    df = df.rename(columns={"date": "Date"})
df["Date"] = pd.to_datetime(df["Date"])

# Show coverage summary
print(f"\n  Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
if "stock_id" in df.columns:
    print(f"  Stocks covered: {df['stock_id'].nunique():,}")

# Year-by-year coverage
print("\n  Coverage by year:")
df["year"] = df["Date"].dt.year
for yr, grp in df.groupby("year"):
    n_stocks = grp["stock_id"].nunique() if "stock_id" in grp.columns else len(grp)
    print(f"    {yr}: {len(grp):>8,} rows, {n_stocks:>5,} stocks")
df.drop(columns=["year"], inplace=True)

# Check the key percentage column
num_cols = df.select_dtypes(include="number").columns.tolist()
print(f"\n  Numeric columns: {num_cols}")
for c in num_cols[:4]:
    z = (df[c].fillna(0) == 0).mean() * 100
    n = df[c].isna().mean() * 100
    print(f"    {c}: NaN={n:.1f}%, Zero={z:.1f}%, range=[{df[c].min():.1f}, {df[c].max():.1f}]")

# Backup old file
cache_path = PROCESSED_DIR / "foreign_shareholding_raw.parquet"
backup_path = PROCESSED_DIR / "foreign_shareholding_raw.parquet.bak"
if cache_path.exists():
    import shutil
    shutil.copy2(cache_path, backup_path)
    print(f"\n  Backed up old file to: {backup_path.name}")

# Save new file
df.to_parquet(cache_path)
print(f"  Saved: {cache_path}")
print(f"  Final shape: {df.shape}")

print("\n" + "=" * 60)
print("  Done! Re-run quick_data_check.py to verify.")
print("=" * 60)
