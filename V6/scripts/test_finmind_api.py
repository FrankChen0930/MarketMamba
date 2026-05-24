"""Test: Dividend fetch strategies."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import requests
import time

TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiRnJhbmtDaGVuIiwiZW1haWwiOiJhMDk2NjQ2OTk2NEBnbWFpbC5jb20ifQ.rwJBGSwJyHDqXeVZKCMkKulKVk48Y2klu4pyUgiJyrE"
BASE = "https://api.finmindtrade.com/api/v4/data"

# Strategy 1: Full market, small date range, NO data_id
print("Strategy 1: Full market, 1-year chunk, NO data_id")
for year in [2020, 2021, 2022, 2023, 2024]:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    t0 = time.time()
    r = requests.get(BASE, params={
        "dataset": "TaiwanStockDividend",
        "start_date": start,
        "end_date": end,
        "token": TOKEN,
    }, timeout=60)
    dt = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        rows = len(d.get("data", []))
        print(f"  {year}: {rows:>5} rows ({dt:.1f}s)")
    else:
        print(f"  {year}: HTTP {r.status_code} ({dt:.1f}s)")
    time.sleep(0.5)

# Strategy 2: Single stock, data_id
print("\nStrategy 2: Single stocks, data_id")
for sid in ["2330", "2317", "3008", "1301"]:
    t0 = time.time()
    r = requests.get(BASE, params={
        "dataset": "TaiwanStockDividend",
        "data_id": sid,
        "start_date": "2005-01-01",
        "end_date": "2026-05-05",
        "token": TOKEN,
    }, timeout=15)
    dt = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        rows = len(d.get("data", []))
        print(f"  {sid}: {rows:>3} rows ({dt:.1f}s)")
    else:
        print(f"  {sid}: HTTP {r.status_code} ({dt:.1f}s)")
    time.sleep(0.5)

# Strategy 3: Shareholding single stock small range
print("\nStrategy 3: Shareholding single stock, 1 month")
for sid in ["2330", "2317"]:
    t0 = time.time()
    r = requests.get(BASE, params={
        "dataset": "TaiwanStockShareholding",
        "data_id": sid,
        "start_date": "2024-12-01",
        "end_date": "2024-12-31",
        "token": TOKEN,
    }, timeout=15)
    dt = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        rows = len(d.get("data", []))
        print(f"  {sid}: {rows:>3} rows ({dt:.1f}s)")
    else:
        print(f"  {sid}: HTTP {r.status_code} ({dt:.1f}s)")
    time.sleep(0.3)
