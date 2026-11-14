import sys; sys.stdout.reconfigure(encoding="utf-8")
import requests, os
from dotenv import load_dotenv
load_dotenv("V6/.env")
token = os.getenv("FINMIND_TOKEN")

# Try different endpoints for dataset listing
for url in [
    "https://api.finmindtrade.com/api/v4/datasets",
    "https://api.finmindtrade.com/api/v4/dataset",
]:
    r = requests.get(url, params={"token": token}, timeout=10)
    d = r.json()
    ds = d.get("data", d.get("dataset", []))
    print(f"{url} -> {len(ds) if isinstance(ds, list) else 'not a list'}")
    if isinstance(ds, list) and ds:
        # filter relevant
        for x in sorted(ds):
            if any(k in x.lower() for k in ["share", "hold", "foreign", "interest", "rate", "fed", "dividend"]):
                print(f"  {x}")

# Now test specific datasets we know should work
print()
print("=== Direct tests for Shareholding/FedRate ===")
tests = [
    ("TaiwanStockShareholding", {"data_id": "2330"}),
    ("TaiwanForeignInvestorHolding", {"data_id": "2330"}),
    ("TaiwanStockHoldingSharesPer", {"data_id": "2330"}),
    ("InterestRate", {}),
    ("CentralBankInterestRate", {}),
    ("GovernmentBondsYield", {"data_id": "United States 10-Year"}),
]
for ds, extra in tests:
    params = {"dataset": ds, "start_date": "2024-01-01", "end_date": "2024-06-01", "token": token}
    params.update(extra)
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params=params, timeout=10)
        d = r.json()
        rows = d.get("data", [])
        msg = d.get("msg", "")[:60]
        print(f"  {ds}: {r.status_code} | {len(rows)} rows | {msg}")
    except Exception as e:
        print(f"  {ds}: {e}")
