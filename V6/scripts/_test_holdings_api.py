"""
Confirm TaiwanStockHoldingSharesPer:
- Without data_id: full market data requires VIP plan AND specific date format
- With data_id: per-stock, but only from ~2018-06
Test wider date ranges for no-data_id mode.
"""
import os, sys, requests, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path("V6/.env"))
token = os.getenv("FINMIND_TOKEN", "")
BASE = "https://api.finmindtrade.com/api/v4/data"

# Test with wider date ranges (no data_id)
print("=== Test: no data_id (all stocks), wider date ranges ===")
for start, end in [
    ("2024-01-01", "2024-03-31"),
    ("2023-01-01", "2023-12-31"),
    ("2022-01-01", "2022-12-31"),
]:
    r = requests.get(BASE, params={
        "dataset": "TaiwanStockHoldingSharesPer",
        "start_date": start,
        "end_date": end,
        "token": token,
    }, timeout=60)
    d = r.json()
    rows = len(d.get("data", []))
    msg  = d.get("msg", "")
    print(f"{start}~{end}: status={d.get('status')} rows={rows} msg='{msg}'")
    if rows > 0:
        df = pd.DataFrame(d["data"])
        print(f"  Unique dates: {df['date'].nunique()}, range: {df['date'].min()} ~ {df['date'].max()}")

# Test per-stock (data_id) from 2018
print("\n=== Test: data_id=2330, with wider ranges ===")
for start, end in [
    ("2018-01-01", "2019-12-31"),
    ("2020-01-01", "2021-12-31"),
]:
    r = requests.get(BASE, params={
        "dataset": "TaiwanStockHoldingSharesPer",
        "data_id": "2330",
        "start_date": start,
        "end_date": end,
        "token": token,
    }, timeout=60)
    d = r.json()
    rows = len(d.get("data", []))
    if rows > 0:
        df = pd.DataFrame(d["data"])
        print(f"{start}~{end}: {rows} rows, dates: {df['date'].min()} ~ {df['date'].max()}")
    else:
        print(f"{start}~{end}: {rows} rows, msg={d.get('msg','')}")
