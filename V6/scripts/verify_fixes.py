# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
sys.path.insert(0, ".")
from marketmamba.config import PROCESSED_DIR

# --- Test 1: Holdings fix ---
print("=== TEST 1: Holdings (Retail_Hold_Ratio inversion) ===")
h = pd.read_parquet(PROCESSED_DIR / "holdings_raw.parquet")
print(f"Columns: {h.columns.tolist()}")
h["Retail_Hold_Ratio"] = pd.to_numeric(h["Retail_Hold_Ratio"], errors="coerce")
h["Holdings_Large_Pct"] = (1.0 - h["Retail_Hold_Ratio"] / 100.0).clip(0, 1)
z = (h["Holdings_Large_Pct"].fillna(0) == 0).mean() * 100
print(f"Holdings_Large_Pct  Zero%: {z:.1f}%  (was 100%, should now be <20%)")
print(h["Holdings_Large_Pct"].describe().to_string())

# --- Test 2: FED Rate from macro_raw ---
print()
print("=== TEST 2: FED_Rate from macro_raw ===")
m = pd.read_parquet(PROCESSED_DIR / "macro_raw.parquet")
print(f"FED_Rate in macro_raw: {'FED_Rate' in m.columns}")
if "FED_Rate" in m.columns:
    z = (m["FED_Rate"].fillna(0) == 0).mean() * 100
    print(f"FED_Rate Zero%: {z:.1f}%  (should be 0%)")
    print(f"FED_Rate range: [{m['FED_Rate'].min():.2f}, {m['FED_Rate'].max():.2f}]")
    m["Date"] = pd.to_datetime(m["Date"])
    yr_last = m.groupby(m["Date"].dt.year)[["Date","FED_Rate"]].last()
    print("Year-end FED_Rate values:")
    print(yr_last.to_string())

print()
print("=== TEST 3: CNN_FearGreed and TW_Biz_Signal in macro_raw ===")
for col, canon in [("CNN_FearGreed", "Fear_Greed"), ("TW_Biz_Signal", "Business_Signal")]:
    if col in m.columns:
        z = (m[col].fillna(0) == 0).mean() * 100
        mn, mx = m[col].min(), m[col].max()
        print(f"  {col} -> {canon}: Zero={z:.1f}%, range=[{mn:.1f}, {mx:.1f}]")
    else:
        print(f"  {col}: NOT FOUND in macro_raw")
