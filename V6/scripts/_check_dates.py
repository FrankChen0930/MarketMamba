import sys; sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
df = pd.read_parquet("Data/processed_v6/prices_raw.parquet")
df["Date"] = pd.to_datetime(df["Date"])
last_dates = sorted(df["Date"].unique())[-7:]
for d in last_dates:
    n = len(df[df["Date"]==d])
    ds = d.strftime("%Y-%m-%d %a")
    print(f"{ds}: {n} stocks")
