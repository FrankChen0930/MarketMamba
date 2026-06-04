import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba\Data\processed_v6")
CACHE_DIR     = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba\Data\cache_v6")

print("=" * 60)
print("1. prices_raw.parquet")
print("=" * 60)
df = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet")
df["Date"] = pd.to_datetime(df["Date"])

n_stocks = df["stock_id"].nunique()
n_dates  = df["Date"].nunique()
print(f"  Rows        : {len(df):,}")
print(f"  Unique stocks: {n_stocks}")
print(f"  Unique dates : {n_dates}")
print(f"  Date range   : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Columns      : {list(df.columns)}")

# 確認沒有非 4 位數字的 stock_id
bad = df[~df["stock_id"].str.match(r"^\d{4}$")]["stock_id"].unique()
if len(bad) == 0:
    print(f"  ✅ All stock_ids are 4-digit numeric (no '00400A'-style codes)")
else:
    print(f"  ❌ Found {len(bad)} bad stock_ids: {bad[:10]}")

# 每股平均資料天數
avg_days = len(df) / n_stocks
print(f"  Avg days/stock: {avg_days:.0f}")

# 資料最稀疏的 5 支股票（可能是問題股）
sparse = df.groupby("stock_id").size().nsmallest(5)
print(f"  Most sparse stocks: {sparse.to_dict()}")

print()
print("=" * 60)
print("2. ticker_universe.parquet（應該已刪）")
print("=" * 60)
cache_path = CACHE_DIR / "ticker_universe.parquet"
if cache_path.exists():
    print(f"  ❌ Still exists! Delete it before proceeding.")
else:
    print(f"  ✅ Deleted (will be rebuilt fresh next time)")

print()
print("=" * 60)
print("3. 其他 raw 檔案（僅確認存在和大小）")
print("=" * 60)
other_files = [
    "institutional_raw.parquet",
    "margin_raw.parquet",
    "revenue_raw.parquet",
    "financials_raw.parquet",
    "macro_raw.parquet",
    "foreign_shareholding_raw.parquet",
    "dividend_raw.parquet",
]
for fname in other_files:
    p = PROCESSED_DIR / fname
    if p.exists():
        size_mb = p.stat().st_size / 1_048_576
        print(f"  ✅ {fname:<45} {size_mb:.1f} MB")
    else:
        print(f"  ⚠️  {fname:<45} NOT FOUND")

print()
print("=" * 60)
print("4. V6_Feature_Matrix.parquet（如果存在要確認是否需要刪掉重建）")
print("=" * 60)
matrix_path = PROCESSED_DIR / "V6_Feature_Matrix.parquet"
if matrix_path.exists():
    df_m = pd.read_parquet(matrix_path, columns=["stock_id"])
    n_m = df_m["stock_id"].nunique()
    size_mb = matrix_path.stat().st_size / 1_048_576
    if n_m > 3000:
        print(f"  ❌ Feature matrix has {n_m} stocks ({size_mb:.0f} MB) — STALE, delete before re-zipping")
    else:
        print(f"  ✅ Feature matrix has {n_m} stocks ({size_mb:.0f} MB) — looks OK")
else:
    print(f"  ℹ️  Not found locally (will be rebuilt on Colab)")

print()
print("Done.")
