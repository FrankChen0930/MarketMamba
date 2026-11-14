# -*- coding: utf-8 -*-
"""
MarketMamba V6.1 - Quick Data Quality Check
Runs in ~30 seconds. No feature matrix build needed.
Usage:  python scripts/quick_data_check.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from marketmamba.config import PROCESSED_DIR

SEP = "=" * 65

def check_file(filename, date_col_candidates=("Date","date","Week"),
               id_col="stock_id", max_num_cols=6):
    path = PROCESSED_DIR / filename
    if not path.exists():
        print(f"  [MISSING] {filename}")
        return None

    df = pd.read_parquet(path)
    print(f"\n{SEP}")
    print(f"  {filename}")
    print(SEP)
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Cols  : {df.columns.tolist()}")

    # Date range
    for dc in date_col_candidates:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            print(f"  Date  : {df[dc].min().date()} -> {df[dc].max().date()}")
            break

    # Stock coverage
    if id_col in df.columns:
        print(f"  Stocks: {df[id_col].nunique():,}")

    # Numeric columns quality
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\n  {'Column':<35} {'NaN%':>6}  {'Zero%':>6}  {'Min':>10}  {'Max':>10}")
        print(f"  {'-'*35} {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}")
        for c in num_cols[:max_num_cols]:
            s = df[c]
            nan_p  = s.isna().mean() * 100
            zero_p = (s.fillna(0) == 0).mean() * 100
            mn = s.min() if s.notna().any() else float('nan')
            mx = s.max() if s.notna().any() else float('nan')
            print(f"  {c:<35} {nan_p:>5.1f}%  {zero_p:>5.1f}%  {mn:>10.2f}  {mx:>10.2f}")

    return df

# ---- V6.1 new data sources ----
print("\n[V6.1 New Data Sources - Raw Parquet Check]")

check_file("holdings_raw.parquet")
check_file("securities_raw.parquet")
check_file("foreign_shareholding_raw.parquet")
check_file("dividend_raw.parquet")
check_file("cashflow_raw.parquet", date_col_candidates=("Date","date"))
check_file("fear_greed.parquet")
check_file("business_indicator.parquet")
check_file("fed_rate.parquet")
check_file("futures_institutional_raw.parquet")
check_file("options_institutional_raw.parquet")

# ---- Existing sources spot check ----
print(f"\n\n[Existing Data Sources - Spot Check]")
check_file("institutional_raw.parquet", max_num_cols=4)
check_file("macro_raw.parquet", id_col="index")

# ---- Year-by-year zero rate for key V6.1 features ----
print(f"\n\n{SEP}")
print("  Year-by-year Zero Rate: Key V6.1 Sources")
print(SEP)

# Securities Balance (borrowing)
sec_path = PROCESSED_DIR / "securities_raw.parquet"
if sec_path.exists():
    df_s = pd.read_parquet(sec_path)
    dc = "Date" if "Date" in df_s.columns else "date"
    df_s[dc] = pd.to_datetime(df_s[dc], errors="coerce")
    num_cols = df_s.select_dtypes(include="number").columns.tolist()
    if num_cols:
        col = num_cols[0]
        df_s["year"] = df_s[dc].dt.year
        print(f"\n  securities_raw -> [{col}] zero% by year:")
        for yr, grp in df_s.groupby("year"):
            z = (grp[col].fillna(0) == 0).mean() * 100
            n = grp[col].isna().mean() * 100
            bar = "#" * int(z / 5)
            print(f"    {yr}: NaN={n:5.1f}%  Zero={z:5.1f}%  {bar}")

# Holdings
hld_path = PROCESSED_DIR / "holdings_raw.parquet"
if hld_path.exists():
    df_h = pd.read_parquet(hld_path)
    dc = "Week" if "Week" in df_h.columns else "Date"
    df_h[dc] = pd.to_datetime(df_h[dc], errors="coerce")
    num_cols = df_h.select_dtypes(include="number").columns.tolist()
    if num_cols:
        col = num_cols[0]
        df_h["year"] = df_h[dc].dt.year
        print(f"\n  holdings_raw -> [{col}] zero% by year:")
        for yr, grp in df_h.groupby("year"):
            z = (grp[col].fillna(0) == 0).mean() * 100
            n = grp[col].isna().mean() * 100
            bar = "#" * int(z / 5)
            print(f"    {yr}: NaN={n:5.1f}%  Zero={z:5.1f}%  {bar}")

# Foreign shareholding
fsh_path = PROCESSED_DIR / "foreign_shareholding_raw.parquet"
if fsh_path.exists():
    df_f = pd.read_parquet(fsh_path)
    dc = "Date" if "Date" in df_f.columns else "date"
    df_f[dc] = pd.to_datetime(df_f[dc], errors="coerce")
    num_cols = df_f.select_dtypes(include="number").columns.tolist()
    if num_cols:
        col = num_cols[0]
        df_f["year"] = df_f[dc].dt.year
        print(f"\n  foreign_shareholding_raw -> [{col}] zero% by year:")
        for yr, grp in df_f.groupby("year"):
            z = (grp[col].fillna(0) == 0).mean() * 100
            n = grp[col].isna().mean() * 100
            bar = "#" * int(z / 5)
            print(f"    {yr}: NaN={n:5.1f}%  Zero={z:5.1f}%  {bar}")

print(f"\n{SEP}")
print("  Done. Look for high Zero% in early years (2005-2014).")
print(SEP)
