"""
MarketMamba V6.1 — Local Data Fetcher (v4 Final)
===================================================
Handles FinMind API quirks:
  - Futures/Options: data_id=TX/TXO (market-level, fast)
  - Dividend: data_id per stock (sparse, manageable)
  - Shareholding: VERY SLOW (504 timeout), skip for now

Usage:
    python V6/scripts/fetch_v61_data.py
    python V6/scripts/fetch_v61_data.py --force
"""

import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import re
import time
import logging
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import requests

# ── Setup ──
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed_v6"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"
START_DATE = "2005-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")
FORCE = "--force" in sys.argv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("V61")


# ============================================================
# FinMind Helpers (matching fetcher.py patterns exactly)
# ============================================================

def _finmind_fetch(dataset, start_date, end_date, stock_id=None, data_id=None, timeout=30):
    """Generic FinMind API call with data_id support."""
    params = {
        "dataset": dataset,
        "start_date": start_date,
        "end_date": end_date,
        "token": FINMIND_TOKEN,
    }
    if stock_id:
        params["stock_id"] = stock_id
    if data_id:
        params["data_id"] = data_id
    try:
        resp = requests.get(FINMIND_BASE, params=params, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("status") != 200:
            return None
        df = pd.DataFrame(data["data"])
        return df if not df.empty else None
    except Exception:
        return None


def _finmind_fetch_chunked(dataset, start_date, end_date, stock_id=None, data_id=None, chunk_years=1):
    """Yearly chunks, same as fetcher.py."""
    _start = datetime.strptime(start_date, "%Y-%m-%d")
    _end = datetime.strptime(end_date, "%Y-%m-%d")
    frames = []
    current = _start
    while current <= _end:
        chunk_end = min(datetime(current.year + chunk_years - 1, 12, 31), _end)
        df = _finmind_fetch(
            dataset,
            start_date=current.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            stock_id=stock_id, data_id=data_id,
        )
        if df is not None and not df.empty:
            frames.append(df)
        current = datetime(current.year + chunk_years, 1, 1)
        time.sleep(0.6)
    return pd.concat(frames, ignore_index=True) if frames else None


def _get_stock_universe():
    """4-digit stock IDs only."""
    prices = PROCESSED_DIR / "prices_raw.parquet"
    df = pd.read_parquet(prices, columns=["stock_id"])
    all_ids = df["stock_id"].unique()
    return sorted([s for s in all_ids if re.match(r"^\d{4}$", str(s).strip())])


# ============================================================
# 1. Futures Institutional
# ============================================================
def fetch_futures():
    cache = PROCESSED_DIR / "futures_institutional_raw.parquet"
    if cache.exists() and not FORCE:
        log.info("[SKIP] futures_institutional_raw.parquet")
        return
    log.info("[1/3] Fetching TaiwanFuturesInstitutionalInvestors (data_id=TX)...")
    df = _finmind_fetch_chunked("TaiwanFuturesInstitutionalInvestors", START_DATE, END_DATE, data_id="TX")
    if df is None:
        log.error("  No data!")
        return
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df.to_parquet(cache)
    log.info(f"  OK: {len(df):,} rows")


# ============================================================
# 2. Options Institutional
# ============================================================
def fetch_options():
    cache = PROCESSED_DIR / "options_institutional_raw.parquet"
    if cache.exists() and not FORCE:
        log.info("[SKIP] options_institutional_raw.parquet")
        return
    log.info("[2/3] Fetching TaiwanOptionInstitutionalInvestors (data_id=TXO)...")
    df = _finmind_fetch_chunked("TaiwanOptionInstitutionalInvestors", START_DATE, END_DATE, data_id="TXO")
    if df is None:
        log.error("  No data!")
        return
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df.to_parquet(cache)
    log.info(f"  OK: {len(df):,} rows")


# ============================================================
# 3. Dividend (per stock, but sparse data)
# ============================================================
def fetch_dividends():
    cache = PROCESSED_DIR / "dividend_raw.parquet"
    if cache.exists() and not FORCE:
        log.info("[SKIP] dividend_raw.parquet")
        return

    log.info("[3/3] Fetching TaiwanStockDividend (per stock)...")
    stock_ids = _get_stock_universe()
    log.info(f"  {len(stock_ids)} stocks to fetch")

    frames = []
    errors = 0
    batch_size = 50

    for i, sid in enumerate(stock_ids):
        # Short timeout — dividend is sparse, should be fast
        df = _finmind_fetch("TaiwanStockDividend", START_DATE, END_DATE, data_id=sid, timeout=15)
        if df is not None and not df.empty:
            frames.append(df)
        else:
            errors += 1

        # Progress every batch
        if (i + 1) % batch_size == 0:
            log.info(f"  {i+1}/{len(stock_ids)} ... ({len(frames)} ok, {errors} empty)")
            time.sleep(1.0)  # longer pause every batch
        else:
            time.sleep(0.2)

    if not frames:
        log.error(f"  No data! ({errors} errors)")
        return
    df = pd.concat(frames, ignore_index=True)
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df.to_parquet(cache)
    log.info(f"  OK: {len(df):,} rows, {df['stock_id'].nunique()} stocks ({errors} no-data)")


# ============================================================
# Validation
# ============================================================
def validate():
    log.info("")
    log.info("=" * 70)
    log.info("  DATA VALIDATION REPORT")
    log.info("=" * 70)

    files = [
        ("futures_institutional_raw.parquet", "期貨三大法人"),
        ("options_institutional_raw.parquet", "選擇權三大法人"),
        ("dividend_raw.parquet",              "股利政策"),
        ("foreign_shareholding_raw.parquet",  "外資持股比例"),
        ("holdings_raw.parquet",              "大戶持股分級"),
        ("securities_raw.parquet",            "借券餘額"),
        ("cashflow_raw.parquet",              "現金流量表"),
        ("fear_greed.parquet",                "恐懼貪婪指數"),
        ("business_indicator.parquet",        "景氣燈號"),
        ("fed_rate.parquet",                  "FED利率"),
        ("prices_raw.parquet",                "股價 OHLCV"),
        ("institutional_raw.parquet",         "三大法人"),
        ("margin_raw.parquet",                "融資融券"),
        ("per_raw.parquet",                   "PER/PBR"),
        ("market_value_raw.parquet",          "個股市值"),
        ("daytrade_raw.parquet",              "當沖比率"),
        ("revenue_raw.parquet",               "月營收"),
        ("financials_raw.parquet",            "綜合損益表"),
        ("balance_sheet_raw.parquet",         "資產負債表"),
        ("macro_raw.parquet",                 "總經指標"),
    ]

    ok = 0
    missing = 0
    for fname, desc in files:
        path = PROCESSED_DIR / fname
        if not path.exists():
            log.info(f"  [MISS] {desc:20s} ({fname})")
            missing += 1
            continue
        ok += 1
        size = path.stat().st_size / 1e6
        df = pd.read_parquet(path)
        dcol = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
        if dcol:
            dmin, dmax = str(df[dcol].min())[:10], str(df[dcol].max())[:10]
            n = df[dcol].nunique()
            sid = f", {df['stock_id'].nunique()} sids" if "stock_id" in df.columns else ""
            log.info(f"  [ OK ] {desc:20s} {len(df):>10,} rows | {n:>5,} dates | {dmin}->{dmax} | {size:>7.1f}MB{sid}")
        else:
            log.info(f"  [ OK ] {desc:20s} {len(df):>10,} rows | {size:>7.1f}MB")

    log.info(f"\n  {ok} present, {missing} missing")
    if missing > 0:
        log.info("  NOTE: foreign_shareholding is expected missing (API too slow)")
        log.info("        Model will use Foreign_Holding_Pct=0 as default fallback")
    log.info("=" * 70)


def main():
    log.info("=" * 70)
    log.info("  MarketMamba V6.1 - Local Data Fetcher")
    log.info(f"  Range: {START_DATE} -> {END_DATE} | Force={FORCE}")
    log.info("=" * 70)

    fetch_futures()
    fetch_options()
    fetch_dividends()

    # NOTE: Skipping foreign_shareholding — API returns 504 timeout
    # The model handles this gracefully (Foreign_Holding_Pct defaults to 0)
    log.info("[SKIP] foreign_shareholding — FinMind API too slow (504 timeout)")
    log.info("  Model will use default fallback (Foreign_Holding_Pct=0)")

    validate()

    log.info("")
    log.info("  Next steps:")
    log.info("  1. Re-zip:  cd Data && powershell Compress-Archive -Path processed_v6 -DestinationPath processed_v6.zip -Force")
    log.info("  2. Upload processed_v6.zip to Google Drive /MyDrive/MarketMamba_V6/")
    log.info("  3. In Colab: FORCE_REBUILD=True to rebuild 56D Feature Matrix")


if __name__ == "__main__":
    main()
