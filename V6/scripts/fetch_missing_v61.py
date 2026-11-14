"""Fetch TaiwanStockShareholding (外資持股)"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import requests, time, os
import pandas as pd
from dotenv import load_dotenv
load_dotenv("V6/.env")
token = os.getenv("FINMIND_TOKEN")

OUT_DIR = "Data/processed_v6"
prices = pd.read_parquet(os.path.join(OUT_DIR, "prices_raw.parquet"), columns=["stock_id"])
stocks = sorted(set(s for s in prices["stock_id"].unique() if s.isdigit() and len(s) == 4))
print(f"Stocks: {len(stocks)}", flush=True)

all_rows = []
errs = 0
for i, sid in enumerate(stocks):
    if i % 200 == 0:
        print(f"  Shareholding [{i}/{len(stocks)}] {sid}... ({len(all_rows)} rows)", flush=True)
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={
            "dataset": "TaiwanStockShareholding",
            "data_id": sid,
            "start_date": "2018-01-01",
            "end_date": "2026-05-05",
            "token": token,
        }, timeout=15)
        rows = r.json().get("data", [])
        if rows:
            all_rows.extend(rows)
    except:
        errs += 1
    time.sleep(0.15)

print(f"\nShareholding: {len(all_rows)} total rows, {errs} errors", flush=True)
if all_rows:
    df = pd.DataFrame(all_rows)
    out = os.path.join(OUT_DIR, "foreign_shareholding_raw.parquet")
    df.to_parquet(out, index=False)
    print(f"Saved: {out} ({len(df):,} rows, {df['stock_id'].nunique()} stocks)", flush=True)
    print(f"Cols: {list(df.columns)}", flush=True)
else:
    print("No data!", flush=True)
