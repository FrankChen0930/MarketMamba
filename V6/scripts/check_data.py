import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import os

d = "Data/processed_v6"

files = [
    # V6.1 new (fetched today)
    ("futures_institutional_raw.parquet", "期貨三大法人", True),
    ("options_institutional_raw.parquet", "選擇權三大法人", True),
    ("dividend_raw.parquet",              "股利政策", True),
    ("foreign_shareholding_raw.parquet",  "外資持股比例", True),
    # Already downloaded, now activated by V6.1
    ("holdings_raw.parquet",              "大戶持股分級", False),
    ("securities_raw.parquet",            "借券餘額", False),
    ("cashflow_raw.parquet",              "現金流量表", False),
    ("fear_greed.parquet",                "恐懼貪婪指數", False),
    ("business_indicator.parquet",        "景氣燈號", False),
    ("fed_rate.parquet",                  "FED利率", False),
    # Core V6.0
    ("prices_raw.parquet",                "股價 OHLCV", False),
    ("institutional_raw.parquet",         "三大法人", False),
    ("margin_raw.parquet",                "融資融券", False),
    ("per_raw.parquet",                   "PER/PBR", False),
    ("market_value_raw.parquet",          "個股市值", False),
    ("daytrade_raw.parquet",              "當沖比率", False),
    ("revenue_raw.parquet",               "月營收", False),
    ("financials_raw.parquet",            "綜合損益表", False),
    ("balance_sheet_raw.parquet",         "資產負債表", False),
    ("macro_raw.parquet",                 "總經指標", False),
]

print("=" * 80)
print("  MarketMamba V6.1 - Complete Data Inventory")
print("=" * 80)

ok, missing = 0, 0
for fname, desc, is_new in files:
    p = os.path.join(d, fname)
    tag = "[NEW]" if is_new else "     "
    if not os.path.exists(p):
        print(f"  {tag} MISS  {desc:20s} ({fname})")
        missing += 1
        continue
    ok += 1
    size = os.path.getsize(p) / 1e6
    df = pd.read_parquet(p)
    dcol = None
    for c in ["Date", "date"]:
        if c in df.columns:
            dcol = c
            break
    if dcol:
        dmin = str(df[dcol].min())[:10]
        dmax = str(df[dcol].max())[:10]
        n = df[dcol].nunique()
        sid = ""
        if "stock_id" in df.columns:
            sid = f", {df['stock_id'].nunique()} stocks"
        print(f"  {tag}  OK   {desc:20s} {len(df):>10,} rows | {n:>5,} dates | {dmin}->{dmax} | {size:>7.1f}MB{sid}")
    else:
        print(f"  {tag}  OK   {desc:20s} {len(df):>10,} rows | {size:>7.1f}MB | cols={list(df.columns[:5])}")

print(f"\n  Total: {ok} OK, {missing} missing")
print("=" * 80)
