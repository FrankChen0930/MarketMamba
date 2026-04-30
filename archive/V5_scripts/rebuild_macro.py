"""
MarketMamba V6 — Rebuild macro_raw.parquet
==========================================
Adds missing columns to macro_raw.parquet:
  - USD_TWD        (from FinMind TaiwanExchangeRate)
  - Oil            (from FinMind CrudeOilPrices)
  - FED_Rate       (from fed_rate.parquet — already downloaded)
  - CNN_FearGreed  (from fear_greed.parquet — already downloaded)
  - TW_Biz_Signal  (from business_indicator.parquet — already downloaded)

Usage:
    python scripts/rebuild_macro.py
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
ROOT_DIR      = SCRIPT_DIR.parent
V6_DIR        = ROOT_DIR / "V6"
DATA_DIR      = ROOT_DIR / "Data"
PROCESSED_DIR = DATA_DIR / "processed_v6"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("rebuild_macro")

# ── Token ────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(V6_DIR / ".env", override=False)
except ImportError:
    pass

TOKEN = os.getenv("FINMIND_TOKEN", "")
if not TOKEN:
    sys.exit("FINMIND_TOKEN not found. Check V6/.env")

API_URL = "https://api.finmindtrade.com/api/v4/data"
START_DATE = "2005-01-01"
END_DATE   = pd.Timestamp.now().strftime("%Y-%m-%d")

def _fm_get(dataset: str, start: str, end: str,
            stock_id: str | None = None, data_id: str | None = None) -> pd.DataFrame | None:
    params = {"dataset": dataset, "start_date": start, "end_date": end, "token": TOKEN}
    if data_id:
        params["data_id"] = data_id
    elif stock_id:
        params["stock_id"] = stock_id
    try:
        resp = requests.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == 200 and data.get("data"):
            return pd.DataFrame(data["data"])
        log.warning(f"  {dataset}: empty response")
        return None
    except Exception as e:
        log.error(f"  {dataset} failed: {e}")
        return None
    finally:
        time.sleep(0.7)

# ── 1. Load existing macro_raw ────────────────────────────────────────────────
macro_path = PROCESSED_DIR / "macro_raw.parquet"
if not macro_path.exists():
    sys.exit(f"macro_raw.parquet not found at {macro_path}")

log.info("Loading existing macro_raw.parquet ...")
df = pd.read_parquet(macro_path)
df["Date"] = pd.to_datetime(df["Date"])
log.info(f"  Shape: {df.shape} | Columns: {list(df.columns)}")

# ── 2. USD/TWD ────────────────────────────────────────────────────────────────
if "USD_TWD" not in df.columns:
    log.info("Fetching USD/TWD from FinMind ...")
    df_fx = _fm_get("TaiwanExchangeRate", START_DATE, END_DATE, data_id="USD")
    if df_fx is not None and not df_fx.empty:
        df_fx = df_fx.rename(columns={"date": "Date", "cash_sell": "USD_TWD"})[["Date", "USD_TWD"]]
        df_fx["Date"]    = pd.to_datetime(df_fx["Date"])
        df_fx["USD_TWD"] = pd.to_numeric(df_fx["USD_TWD"], errors="coerce")
        df = pd.merge_asof(df.sort_values("Date"), df_fx.sort_values("Date"),
                           on="Date", direction="backward")
        log.info(f"  USD_TWD added: {df['USD_TWD'].notna().sum()} non-null")
    else:
        log.warning("  USD_TWD fetch failed")
else:
    log.info("USD_TWD already present, skipping")

# ── 3. Oil (WTI) ──────────────────────────────────────────────────────────────
if "Oil" not in df.columns:
    log.info("Fetching Oil (WTI) from FinMind ...")
    df_oil = _fm_get("CrudeOilPrices", START_DATE, END_DATE, data_id="WTI")
    if df_oil is not None and not df_oil.empty:
        df_oil = df_oil.rename(columns={"date": "Date", "price": "Oil"})[["Date", "Oil"]]
        df_oil["Date"] = pd.to_datetime(df_oil["Date"])
        df_oil["Oil"]  = pd.to_numeric(df_oil["Oil"], errors="coerce")
        df = pd.merge_asof(df.sort_values("Date"), df_oil.sort_values("Date"),
                           on="Date", direction="backward")
        log.info(f"  Oil added: {df['Oil'].notna().sum()} non-null")
    else:
        log.warning("  Oil fetch failed")
else:
    log.info("Oil already present, skipping")

# ── 4. FED Rate (already downloaded) ─────────────────────────────────────────
if "FED_Rate" not in df.columns:
    fed_path = PROCESSED_DIR / "fed_rate.parquet"
    if fed_path.exists():
        log.info("Merging FED_Rate from fed_rate.parquet ...")
        df_fed = pd.read_parquet(fed_path)
        if "date" in df_fed.columns:
            df_fed = df_fed.rename(columns={"date": "Date"})
        df_fed["Date"] = pd.to_datetime(df_fed["Date"])
        # column name: 'interest_rate'
        rate_col = "interest_rate" if "interest_rate" in df_fed.columns else df_fed.columns[-1]
        df_fed = df_fed[["Date", rate_col]].rename(columns={rate_col: "FED_Rate"})
        df_fed["FED_Rate"] = pd.to_numeric(df_fed["FED_Rate"], errors="coerce")
        df = pd.merge_asof(df.sort_values("Date"), df_fed.sort_values("Date"),
                           on="Date", direction="backward")
        log.info(f"  FED_Rate added: {df['FED_Rate'].notna().sum()} non-null")
    else:
        log.warning("  fed_rate.parquet not found")
else:
    log.info("FED_Rate already present, skipping")

# ── 5. CNN Fear & Greed ───────────────────────────────────────────────────────
if "CNN_FearGreed" not in df.columns:
    cnn_path = PROCESSED_DIR / "fear_greed.parquet"
    if cnn_path.exists():
        log.info("Merging CNN_FearGreed from fear_greed.parquet ...")
        df_cnn = pd.read_parquet(cnn_path)
        if "date" in df_cnn.columns:
            df_cnn = df_cnn.rename(columns={"date": "Date"})
        df_cnn["Date"] = pd.to_datetime(df_cnn["Date"])
        df_cnn = df_cnn[["Date", "fear_greed"]].rename(columns={"fear_greed": "CNN_FearGreed"})
        df_cnn["CNN_FearGreed"] = pd.to_numeric(df_cnn["CNN_FearGreed"], errors="coerce")
        df = pd.merge_asof(df.sort_values("Date"), df_cnn.sort_values("Date"),
                           on="Date", direction="backward")
        log.info(f"  CNN_FearGreed added: {df['CNN_FearGreed'].notna().sum()} non-null")
    else:
        log.warning("  fear_greed.parquet not found")
else:
    log.info("CNN_FearGreed already present, skipping")

# ── 6. Taiwan Business Indicator ─────────────────────────────────────────────
if "TW_Biz_Signal" not in df.columns:
    biz_path = PROCESSED_DIR / "business_indicator.parquet"
    if biz_path.exists():
        log.info("Merging TW_Biz_Signal from business_indicator.parquet ...")
        df_biz = pd.read_parquet(biz_path)
        if "date" in df_biz.columns:
            df_biz = df_biz.rename(columns={"date": "Date"})
        df_biz["Date"] = pd.to_datetime(df_biz["Date"])
        # 'monitoring' column contains the numeric signal value
        sig_col = "monitoring" if "monitoring" in df_biz.columns else df_biz.columns[-1]
        df_biz = df_biz[["Date", sig_col]].rename(columns={sig_col: "TW_Biz_Signal"})
        df_biz["TW_Biz_Signal"] = pd.to_numeric(df_biz["TW_Biz_Signal"], errors="coerce")
        df = pd.merge_asof(df.sort_values("Date"), df_biz.sort_values("Date"),
                           on="Date", direction="backward")
        log.info(f"  TW_Biz_Signal added: {df['TW_Biz_Signal'].notna().sum()} non-null")
    else:
        log.warning("  business_indicator.parquet not found")
else:
    log.info("TW_Biz_Signal already present, skipping")

# ── 7. Forward-fill + save ────────────────────────────────────────────────────
log.info("Forward-filling and saving ...")
df = df.sort_values("Date").ffill().bfill().reset_index(drop=True)
df.to_parquet(macro_path)

log.info("")
log.info("=" * 55)
log.info("Done! macro_raw.parquet updated.")
log.info(f"  Shape   : {df.shape}")
log.info(f"  Columns : {list(df.columns)}")
log.info(f"  Dates   : {df['Date'].min().date()} -> {df['Date'].max().date()}")
log.info("")
for col in df.columns:
    if col == "Date":
        continue
    nan_pct = df[col].isna().mean() * 100
    status = "[OK]" if nan_pct < 5 else "[WARN]"
    log.info(f"  {status}  {col:<20}: NaN {nan_pct:.1f}%")
log.info("=" * 55)
