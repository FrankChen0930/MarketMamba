import pandas as pd
from pathlib import Path

prices_path = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba\Data\processed_v6\prices_raw.parquet")
cache_path  = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba\Data\cache_v6\ticker_universe.parquet")

# 過濾 prices_raw
df = pd.read_parquet(prices_path)
print(f"Before: {df['stock_id'].nunique()} unique stocks, {len(df):,} rows")

df_clean = df[df["stock_id"].str.match(r"^\d{4}$")].copy()
print(f"After:  {df_clean['stock_id'].nunique()} unique stocks, {len(df_clean):,} rows")

df_clean.to_parquet(prices_path)
print("Done. prices_raw.parquet updated.")

# 刪掉壞的 ticker_universe 快取
if cache_path.exists():
    cache_path.unlink()
    print("Deleted stale ticker_universe.parquet")
else:
    print("ticker_universe.parquet not found (already clean)")