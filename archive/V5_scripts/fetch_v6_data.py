"""
MarketMamba V6 — Local Bulk Data Fetcher
==========================================
Prerequisites (run in terminal first):
    pip install requests pandas pyarrow yfinance tqdm python-dotenv

Usage:
    python scripts/fetch_v6_data.py

What it does:
  1. Fetches all Taiwan market data from FinMind (Sponsor plan, 6,000 req/hr)
     using the proven per-day approach (V4 style).
  2. Saves intermediate per-day parquet files in Data/raw_cache/ (resumable).
  3. Assembles final V6 parquet files in Data/processed_v6/.
  4. Fetches macro data via yfinance.

Estimated runtime: ~5-6 hours (safe at 0.65 s/request)
After completion: zip Data/processed_v6/ and upload to Google Drive.
"""

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# ── 0. Setup paths & logging ────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT_DIR     = SCRIPT_DIR.parent
V6_DIR       = ROOT_DIR / "V6"
DATA_DIR     = ROOT_DIR / "Data"
CACHE_DIR    = DATA_DIR / "raw_cache"
PROCESSED_DIR = DATA_DIR / "processed_v6"

for d in [CACHE_DIR / "daily", CACHE_DIR / "weekly",
          CACHE_DIR / "monthly", CACHE_DIR / "quarterly",
          PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "fetch_v6.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("fetch_v6")

# ── 1. Load token from .env ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(V6_DIR / ".env", override=False)
except ImportError:
    pass

TOKEN = os.getenv("FINMIND_TOKEN", "")
if not TOKEN:
    sys.exit("❌  FINMIND_TOKEN not found. Check V6/.env file.")

# Verify token and show quota
def _check_token() -> None:
    resp = requests.get(
        "https://api.web.finmindtrade.com/v2/user_info",
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=15,
    )
    info = resp.json()
    used  = info.get("user_count", "?")
    limit = info.get("api_request_limit", "?")
    log.info(f"✅  Token OK | Used: {used} / {limit} requests today")
    if limit not in ("?",) and int(limit) < 5000:
        log.warning("⚠️  Token might be free tier (limit < 5,000). Expected Sponsor (6,000/hr).")

_check_token()

# ── 2. HTTP Session with retry ───────────────────────────────────────────────
API_URL   = "https://api.finmindtrade.com/api/v4/data"
SLEEP_SEC = 0.65   # 6,000 req/hr ÷ 3,600s = 1.67/s; 0.65s gives ~1.5/s with headroom

session = requests.Session()
retry   = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry))

def _fm_get(dataset: str, start_date: str, end_date: str | None = None,
            stock_id: str | None = None) -> pd.DataFrame | None:
    """Single FinMind API call → DataFrame or None."""
    params: dict = {"dataset": dataset, "start_date": start_date, "token": TOKEN}
    if end_date:
        params["end_date"] = end_date
    if stock_id:
        params["stock_id"] = stock_id
    try:
        resp = session.get(API_URL, params=params, timeout=30)
        if resp.status_code == 402:
            log.error("🚨  API quota exceeded (HTTP 402). Stopping.")
            sys.exit(1)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == 200 and data.get("data"):
            return pd.DataFrame(data["data"])
        return None
    except Exception as exc:
        log.debug(f"  {dataset} {start_date}: {exc}")
        return None
    finally:
        time.sleep(SLEEP_SEC)

# ── 3. Get Taiwan trading days from FinMind ──────────────────────────────────
START_DATE = "2005-01-01"
END_DATE   = date.today().strftime("%Y-%m-%d")

log.info(f"📅  Fetching trading calendar {START_DATE} → {END_DATE} ...")
df_cal = _fm_get("TaiwanStockTradingDate", start_date=START_DATE, end_date=END_DATE)
if df_cal is None or df_cal.empty:
    # fallback: use pandas business days (imperfect but workable)
    log.warning("  Calendar fetch failed — using pandas bday as fallback")
    trading_days = pd.bdate_range(START_DATE, END_DATE).strftime("%Y-%m-%d").tolist()
else:
    trading_days = sorted(df_cal["date"].tolist())

log.info(f"  → {len(trading_days)} trading days to process")

# ── 4. Per-day daily data ────────────────────────────────────────────────────
#
#  For each trading day we call FinMind with start_date == end_date == date_str
#  and NO stock_id (Sponsor tier: returns all stocks at once).
#
#  Datasets per day:
#    A. TaiwanStockPriceAdj               (OHLCV, dividend-adjusted)  ← replaces unadjusted
#    B. TaiwanStockInstitutionalInvestorsBuySell  (三大法人)
#    C. TaiwanStockMarginPurchaseShortSale (融資融券)
#    D. TaiwanStockPER                     (PER / PBR / DY)
#    E. TaiwanStockDayTrading              (當沖, from 2014)
#    F. TaiwanStockSecuritiesLending       (借券, short-interest proxy)
#    G. TaiwanStockMarketValue             (市值, needed for Market_Cap_Log, from 2004)
#
DAILY_CACHE = CACHE_DIR / "daily"
MKT_VALUE_START = "2004-01-01"

def _save_daily(date_str: str, df: pd.DataFrame, tag: str) -> None:
    path = DAILY_CACHE / f"{date_str}_{tag}.parquet"
    df.to_parquet(path)

def _daily_done(date_str: str, tag: str) -> bool:
    return (DAILY_CACHE / f"{date_str}_{tag}.parquet").exists()

DAYTRADE_START = "2014-01-01"
log.info("📦  Starting per-day download (adj_price, chip, margin, PER, daytrade, securities, mktval)...")

for date_str in tqdm(trading_days, desc="Daily data"):
    tasks = [
        # NOTE: use TaiwanStockPriceAdj (還原股價) to avoid ex-dividend artifacts in ML training
        ("adj_price", "TaiwanStockPriceAdj"),
        ("chip",      "TaiwanStockInstitutionalInvestorsBuySell"),
        ("margin",    "TaiwanStockMarginPurchaseShortSale"),
        ("per",       "TaiwanStockPER"),
        ("securities","TaiwanStockSecuritiesLending"),
    ]
    if date_str >= MKT_VALUE_START:
        tasks.append(("mktval", "TaiwanStockMarketValue"))
    if date_str >= DAYTRADE_START:
        tasks.append(("daytrade", "TaiwanStockDayTrading"))

    for tag, dataset in tasks:
        if _daily_done(date_str, tag):
            time.sleep(0)   # no API call needed
            continue
        df = _fm_get(dataset, start_date=date_str, end_date=date_str)
        if df is not None and not df.empty:
            _save_daily(date_str, df, tag)
        # else: no data for this day (holiday / dataset gap) — skip silently

# ── 5. Per-week holdings ─────────────────────────────────────────────────────
HOLD_START   = "2010-01-29"   # TaiwanStockHoldingSharesPer earliest date
WEEKLY_CACHE = CACHE_DIR / "weekly"

log.info("📦  Starting per-week holdings download...")
week_fridays = (
    pd.date_range(start=HOLD_START, end=END_DATE, freq="W-FRI")
      .strftime("%Y-%m-%d").tolist()
)

for week_str in tqdm(week_fridays, desc="Weekly holdings"):
    path = WEEKLY_CACHE / f"{week_str}_holdings.parquet"
    if path.exists():
        continue
    # Try Friday; if empty, try Thursday (compensate for public holidays)
    for offset in range(4):
        search = (datetime.strptime(week_str, "%Y-%m-%d") - timedelta(days=offset)).strftime("%Y-%m-%d")
        df = _fm_get("TaiwanStockHoldingSharesPer", start_date=search, end_date=search)
        if df is not None and not df.empty:
            df.to_parquet(path)
            break

# ── 6. Per-month revenue ─────────────────────────────────────────────────────
MONTHLY_CACHE = CACHE_DIR / "monthly"

log.info("📦  Starting per-month revenue download...")
month_starts = (
    pd.date_range(start="2002-02-01", end=END_DATE, freq="MS")
      .strftime("%Y-%m-%d").tolist()
)

for month_str in tqdm(month_starts, desc="Monthly revenue"):
    path = MONTHLY_CACHE / f"{month_str[:7]}_revenue.parquet"
    if path.exists():
        continue
    df = _fm_get("TaiwanStockMonthRevenue", start_date=month_str, end_date=month_str)
    if df is not None and not df.empty:
        df.to_parquet(path)

# ── 7. Per-quarter financials ─────────────────────────────────────────────────
QUARTERLY_CACHE = CACHE_DIR / "quarterly"
(QUARTERLY_CACHE / "balance_sheet").mkdir(exist_ok=True)
(QUARTERLY_CACHE / "cashflow").mkdir(exist_ok=True)

log.info("📦  Starting per-quarter financial statements download...")
quarter_ends = (
    pd.date_range(start="2004-03-31", end=END_DATE, freq="QE")
      .strftime("%Y-%m-%d").tolist()
)

for q_str in tqdm(quarter_ends, desc="Quarterly financials"):
    # Income statement
    path = QUARTERLY_CACHE / f"{q_str}_financials.parquet"
    if not path.exists():
        df = _fm_get("TaiwanStockFinancialStatements", start_date=q_str, end_date=q_str)
        if df is not None and not df.empty:
            df.to_parquet(path)

    # Balance sheet (ROE, Book Value, Debt/Equity) — from 2011-12-01
    bs_path = QUARTERLY_CACHE / "balance_sheet" / f"{q_str}_balance.parquet"
    if not bs_path.exists() and q_str >= "2011-12-01":
        df = _fm_get("TaiwanStockBalanceSheet", start_date=q_str, end_date=q_str)
        if df is not None and not df.empty:
            df.to_parquet(bs_path)

    # Cash flow statement (FCF quality) — from 2008-06-01
    cf_path = QUARTERLY_CACHE / "cashflow" / f"{q_str}_cashflow.parquet"
    if not cf_path.exists() and q_str >= "2008-06-01":
        df = _fm_get("TaiwanStockCashFlowsStatement", start_date=q_str, end_date=q_str)
        if df is not None and not df.empty:
            df.to_parquet(cf_path)

# ── 7b. Bulk low-frequency macro supplements ─────────────────────────────────
# These are fetched in bulk (not per-day), so a single call covers years of data.

# Taiwan Business Indicator (景氣燈號: red/green lamp) — monthly, Backer/Sponsor
biz_path = PROCESSED_DIR / "business_indicator.parquet"
if not biz_path.exists():
    log.info("📦  Fetching TaiwanBusinessIndicator (景氣燈號)...")
    df_biz = _fm_get("TaiwanBusinessIndicator", start_date="2004-01-01", end_date=END_DATE)
    if df_biz is not None and not df_biz.empty:
        if "date" in df_biz.columns:
            df_biz = df_biz.rename(columns={"date": "Date"})
        df_biz["Date"] = pd.to_datetime(df_biz["Date"])
        df_biz.to_parquet(biz_path)
        log.info(f"    business_indicator: {df_biz.shape}")

# FED interest rate — free, bulk, critical macro signal
fed_path = PROCESSED_DIR / "fed_rate.parquet"
if not fed_path.exists():
    log.info("📦  Fetching FED interest rate...")
    df_fed = _fm_get("InterestRate", start_date="2004-01-01", end_date=END_DATE, stock_id="FED")
    if df_fed is not None and not df_fed.empty:
        if "date" in df_fed.columns:
            df_fed = df_fed.rename(columns={"date": "Date"})
        df_fed["Date"] = pd.to_datetime(df_fed["Date"])
        df_fed.to_parquet(fed_path)
        log.info(f"    fed_rate: {df_fed.shape}")

# CNN Fear & Greed Index — Backer/Sponsor, bulk, from 2011
cnn_path = PROCESSED_DIR / "fear_greed.parquet"
if not cnn_path.exists():
    log.info("📦  Fetching CNN Fear & Greed Index...")
    df_cnn = _fm_get("CnnFearGreedIndex", start_date="2011-01-01", end_date=END_DATE)
    if df_cnn is not None and not df_cnn.empty:
        if "date" in df_cnn.columns:
            df_cnn = df_cnn.rename(columns={"date": "Date"})
        df_cnn["Date"] = pd.to_datetime(df_cnn["Date"])
        df_cnn.to_parquet(cnn_path)
        log.info(f"    fear_greed: {df_cnn.shape}")

# ── 8. Macro data ─────────────────────────────────────────────────────────────
import yfinance as yf

log.info("🌍  Fetching macro data via yfinance + FinMind ...")
macro_cache = PROCESSED_DIR / "macro_raw.parquet"

def _fetch_macro() -> None:
    # 8a. Taiwan index via yfinance as anchor calendar
    twii = yf.download("^TWII", start=START_DATE, auto_adjust=True, progress=False)
    twii.index = pd.to_datetime(twii.index).tz_localize(None)
    df_macro = twii[["Close"]].rename(columns={"Close": "TWII_Close"}).reset_index()
    df_macro.rename(columns={"Date": "Date"}, inplace=True)
    df_macro["Date"] = pd.to_datetime(df_macro["Date"])

    # Handle MultiIndex from newer yfinance
    if isinstance(df_macro.columns, pd.MultiIndex):
        df_macro.columns = df_macro.columns.get_level_values(0)

    # 8b. US indices (shifted +1 trading day — visible next morning in Taiwan)
    us_tickers = {
        "^SOX": "US_SOX", "QQQ": "US_QQQ", "^VIX": "US_VIX",
        "^TNX": "US_TNX",
    }
    for ticker, col in us_tickers.items():
        try:
            tmp = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
            if isinstance(tmp.columns, pd.MultiIndex):
                tmp.columns = tmp.columns.get_level_values(0)
            tmp.index = pd.to_datetime(tmp.index).tz_localize(None)
            tmp = tmp[["Close"]].rename(columns={"Close": col}).reset_index()
            tmp.rename(columns={"Date": "US_Date"}, inplace=True)
            tmp["Date"] = tmp["US_Date"] + pd.Timedelta(days=1)
            df_macro = pd.merge_asof(
                df_macro.sort_values("Date"),
                tmp[["Date", col]].sort_values("Date"),
                on="Date", direction="backward",
            )
        except Exception as e:
            log.warning(f"  yfinance {ticker} failed: {e}")

    # 8c. USD/TWD from FinMind
    df_fx = _fm_get("TaiwanExchangeRate", start_date=START_DATE, end_date=END_DATE, stock_id="USD")
    if df_fx is not None and not df_fx.empty:
        df_fx = df_fx.rename(columns={"date": "Date", "cash_sell": "USD_TWD"})[["Date", "USD_TWD"]]
        df_fx["Date"] = pd.to_datetime(df_fx["Date"])
        df_fx["USD_TWD"] = pd.to_numeric(df_fx["USD_TWD"], errors="coerce")
        df_macro = pd.merge_asof(
            df_macro.sort_values("Date"),
            df_fx.sort_values("Date"),
            on="Date", direction="backward",
        )

    # 8d. Gold & Oil from FinMind (free)
    df_gold = _fm_get("GoldPrice", start_date=START_DATE, end_date=END_DATE)
    if df_gold is not None and not df_gold.empty:
        df_gold = df_gold.rename(columns={"date": "Date", "Price": "Gold"})[["Date", "Gold"]]
        df_gold["Date"] = pd.to_datetime(df_gold["Date"])
        df_gold["Gold"] = pd.to_numeric(df_gold["Gold"], errors="coerce")
        df_macro = pd.merge_asof(df_macro.sort_values("Date"), df_gold.sort_values("Date"),
                                 on="Date", direction="backward")

    df_oil = _fm_get("CrudeOilPrices", start_date=START_DATE, end_date=END_DATE, stock_id="WTI")
    if df_oil is not None and not df_oil.empty:
        df_oil = df_oil.rename(columns={"date": "Date", "price": "Oil"})[["Date", "Oil"]]
        df_oil["Date"] = pd.to_datetime(df_oil["Date"])
        df_oil["Oil"] = pd.to_numeric(df_oil["Oil"], errors="coerce")
        df_macro = pd.merge_asof(df_macro.sort_values("Date"), df_oil.sort_values("Date"),
                                 on="Date", direction="backward")

    # 8e. Merge FED rate into macro
    if fed_path.exists():
        df_fed = pd.read_parquet(fed_path)[["Date", "interest_rate"]].rename(
            columns={"interest_rate": "FED_Rate"})
        df_macro = pd.merge_asof(df_macro.sort_values("Date"),
                                 df_fed.sort_values("Date"),
                                 on="Date", direction="backward")

    # 8f. Merge CNN Fear & Greed into macro
    if cnn_path.exists():
        df_cnn = pd.read_parquet(cnn_path)[["Date", "fear_greed"]].rename(
            columns={"fear_greed": "CNN_FearGreed"})
        df_macro = pd.merge_asof(df_macro.sort_values("Date"),
                                 df_cnn.sort_values("Date"),
                                 on="Date", direction="backward")

    # 8g. Merge business indicator into macro
    if biz_path.exists():
        df_biz2 = pd.read_parquet(biz_path)[["Date", "monitoring"]].rename(
            columns={"monitoring": "TW_Business_Signal"})
        df_biz2["TW_Business_Signal"] = pd.to_numeric(
            df_biz2["TW_Business_Signal"], errors="coerce")
        df_macro = pd.merge_asof(df_macro.sort_values("Date"),
                                 df_biz2.sort_values("Date"),
                                 on="Date", direction="backward")

    df_macro = df_macro.ffill().bfill().sort_values("Date").reset_index(drop=True)
    df_macro.to_parquet(macro_cache)
    log.info(f"  Macro saved: {df_macro.shape}")

if not macro_cache.exists():
    _fetch_macro()
else:
    log.info("  Macro loaded from cache")

# ── 9. Assemble V6 flat parquet files ─────────────────────────────────────────
log.info("🔧  Assembling V6 flat parquet files from daily cache ...")

def _read_parquet_dir(cache_subdir: Path, suffix: str) -> pd.DataFrame:
    """Concat all parquet files matching *_<suffix>.parquet in a directory."""
    files = sorted(cache_subdir.glob(f"*_{suffix}.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in tqdm(files, desc=f"Loading {suffix}", leave=False):
        try:
            df = pd.read_parquet(f)
            # inject Date from filename if not present
            date_str = f.name[:10]
            if "date" not in df.columns and "Date" not in df.columns:
                df["Date"] = date_str
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# 9a. Prices (adjusted)  ───────────────────────────────────────────────────────
# Using TaiwanStockPriceAdj (還原股價) — critical for ML to avoid ex-dividend artifacts
log.info("  Assembling prices_raw.parquet (adjusted) ...")
df_price = _read_parquet_dir(DAILY_CACHE, "adj_price")
if not df_price.empty:
    if "date" in df_price.columns:
        df_price = df_price.rename(columns={"date": "Date"})
    df_price = df_price.rename(columns={
        "open": "Open", "max": "High", "min": "Low",
        "close": "Close", "Trading_Volume": "Volume",
    })
    keep = ["Date", "stock_id"] + [c for c in ["Open","High","Low","Close","Volume"] if c in df_price.columns]
    df_price = df_price[keep]
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df_price.columns:
            df_price[c] = pd.to_numeric(df_price[c], errors="coerce")
    # Filter: only 4-digit numeric IDs (regular stocks) + ETFs (00xxx)
    df_price = df_price[df_price["stock_id"].astype(str).str.match(r"^([1-9]\d{3}|00\d{2,4}[A-Za-z]?)$")]
    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_price.to_parquet(PROCESSED_DIR / "prices_raw.parquet")
    log.info(f"    prices_raw: {df_price.shape}")

# 9b. Institutional (三大法人)  ────────────────────────────────────────────────
log.info("  Assembling institutional_raw.parquet ...")
df_chip = _read_parquet_dir(DAILY_CACHE, "chip")
if not df_chip.empty:
    if "date" in df_chip.columns:
        df_chip = df_chip.rename(columns={"date": "Date"})
    df_chip["Date"] = pd.to_datetime(df_chip["Date"])
    # 'name' column contains: Foreign_Investor, Investment_Trust, Dealer_self etc.
    # pivot on name, compute net buy
    df_chip["buy"]  = pd.to_numeric(df_chip.get("buy",  0), errors="coerce").fillna(0)
    df_chip["sell"] = pd.to_numeric(df_chip.get("sell", 0), errors="coerce").fillna(0)
    df_chip["net"]  = df_chip["buy"] - df_chip["sell"]

    NAME_MAP = {
        "Foreign_Investor":    "Foreign",
        "Foreign_Investor_Dealer": "Foreign",
        "Investment_Trust":    "Trust",
        "Dealer_self":         "Dealer",
        "Dealer":              "Dealer",
    }
    df_chip["name_clean"] = df_chip["name"].map(NAME_MAP).fillna("Other")
    df_chip = df_chip[df_chip["name_clean"] != "Other"]

    pivot = df_chip.pivot_table(
        index=["Date","stock_id"], columns="name_clean",
        values=["buy","sell","net"], aggfunc="sum"
    ).reset_index()
    pivot.columns = ["Date","stock_id"] + [f"{v}_{k}" for v,k in pivot.columns[2:]]
    # Rename to V6 standard
    rename_chip = {
        "buy_Foreign":  "Foreign_Buy",   "sell_Foreign": "Foreign_Sell",  "net_Foreign": "Foreign_Net",
        "buy_Trust":    "Investment_Trust_Buy","sell_Trust":"Investment_Trust_Sell","net_Trust":"Investment_Trust_Net",
        "net_Dealer":   "Dealer_Net",
    }
    pivot = pivot.rename(columns=rename_chip)
    pivot.to_parquet(PROCESSED_DIR / "institutional_raw.parquet")
    log.info(f"    institutional_raw: {pivot.shape}")

# 9c. Margin / Short Sale ─────────────────────────────────────────────────────
log.info("  Assembling margin_raw.parquet ...")
df_margin = _read_parquet_dir(DAILY_CACHE, "margin")
if not df_margin.empty:
    if "date" in df_margin.columns:
        df_margin = df_margin.rename(columns={"date": "Date"})
    df_margin["Date"] = pd.to_datetime(df_margin["Date"])
    for c in ["MarginPurchaseTodayBalance", "ShortSaleTodayBalance",
              "MarginPurchaseBuy", "MarginPurchaseSell",
              "ShortSaleBuy", "ShortSaleSell"]:
        if c in df_margin.columns:
            df_margin[c] = pd.to_numeric(df_margin[c], errors="coerce").fillna(0)
    df_margin = df_margin.rename(columns={
        "MarginPurchaseTodayBalance": "Margin_Balance",
        "ShortSaleTodayBalance":      "Short_Balance",
        "MarginPurchaseBuy":          "Margin_Purchase",
        "MarginPurchaseSell":         "Margin_Repay",
        "ShortSaleBuy":               "Short_Cover",
        "ShortSaleSell":              "Short_Sale",
    })
    keep = ["Date","stock_id"] + [c for c in [
        "Margin_Balance","Short_Balance","Margin_Purchase","Margin_Repay",
        "Short_Cover","Short_Sale"
    ] if c in df_margin.columns]
    df_margin[keep].to_parquet(PROCESSED_DIR / "margin_raw.parquet")
    log.info(f"    margin_raw: {df_margin.shape}")

# 9d. PER / PBR / DY ─────────────────────────────────────────────────────────
log.info("  Assembling per_raw.parquet ...")
df_per = _read_parquet_dir(DAILY_CACHE, "per")
if not df_per.empty:
    if "date" in df_per.columns:
        df_per = df_per.rename(columns={"date": "Date"})
    df_per["Date"] = pd.to_datetime(df_per["Date"])
    for c in ["PER","PBR","dividend_yield"]:
        if c in df_per.columns:
            df_per[c] = pd.to_numeric(df_per[c], errors="coerce")
    df_per = df_per.rename(columns={"dividend_yield": "DY"})
    keep = ["Date","stock_id"] + [c for c in ["PER","PBR","DY"] if c in df_per.columns]
    df_per[keep].to_parquet(PROCESSED_DIR / "per_raw.parquet")
    log.info(f"    per_raw: {df_per.shape}")

# 9e. Securities Lending (借券 / short interest)  ────────────────────────────
log.info("  Assembling securities_raw.parquet ...")
df_sec = _read_parquet_dir(DAILY_CACHE, "securities")
if not df_sec.empty:
    if "date" in df_sec.columns:
        df_sec = df_sec.rename(columns={"date": "Date"})
    df_sec["Date"] = pd.to_datetime(df_sec["Date"])
    df_sec["volume"] = pd.to_numeric(df_sec.get("volume", 0), errors="coerce").fillna(0)
    # Aggregate: total securities lent per stock per day
    df_sec_agg = (df_sec.groupby(["Date","stock_id"])["volume"]
                  .sum().reset_index().rename(columns={"volume":"Securities_Lending"}))
    df_sec_agg.to_parquet(PROCESSED_DIR / "securities_raw.parquet")
    log.info(f"    securities_raw: {df_sec_agg.shape}")

# 9e2. Market Value (市值 → Market_Cap_Log)  ──────────────────────────────────
log.info("  Assembling market_value_raw.parquet ...")
df_mv = _read_parquet_dir(DAILY_CACHE, "mktval")
if not df_mv.empty:
    if "date" in df_mv.columns:
        df_mv = df_mv.rename(columns={"date": "Date"})
    df_mv["Date"] = pd.to_datetime(df_mv["Date"])
    df_mv["market_value"] = pd.to_numeric(df_mv.get("market_value", 0), errors="coerce")
    df_mv[["Date","stock_id","market_value"]].to_parquet(PROCESSED_DIR / "market_value_raw.parquet")
    log.info(f"    market_value_raw: {df_mv.shape}")

# 9e3. Balance Sheet (資產負債表)  ────────────────────────────────────────────
log.info("  Assembling balance_sheet_raw.parquet ...")
bs_files = sorted((QUARTERLY_CACHE / "balance_sheet").glob("*.parquet"))
if bs_files:
    df_bs = pd.concat([pd.read_parquet(f) for f in tqdm(bs_files, desc="Balance sheets", leave=False)],
                      ignore_index=True)
    if "date" in df_bs.columns:
        df_bs = df_bs.rename(columns={"date": "Date"})
    df_bs["Date"] = pd.to_datetime(df_bs["Date"])
    df_bs.to_parquet(PROCESSED_DIR / "balance_sheet_raw.parquet")
    log.info(f"    balance_sheet_raw: {df_bs.shape}")

# 9e4. Cash Flow Statement  ───────────────────────────────────────────────────
log.info("  Assembling cashflow_raw.parquet ...")
cf_files = sorted((QUARTERLY_CACHE / "cashflow").glob("*.parquet"))
if cf_files:
    df_cf = pd.concat([pd.read_parquet(f) for f in tqdm(cf_files, desc="Cash flows", leave=False)],
                      ignore_index=True)
    if "date" in df_cf.columns:
        df_cf = df_cf.rename(columns={"date": "Date"})
    df_cf["Date"] = pd.to_datetime(df_cf["Date"])
    df_cf.to_parquet(PROCESSED_DIR / "cashflow_raw.parquet")
    log.info(f"    cashflow_raw: {df_cf.shape}")

# 9f. Day Trading ─────────────────────────────────────────────────────────────
log.info("  Assembling daytrade_raw.parquet ...")
df_dt = _read_parquet_dir(DAILY_CACHE, "daytrade")
if not df_dt.empty:
    if "date" in df_dt.columns:
        df_dt = df_dt.rename(columns={"date": "Date"})
    df_dt["Date"] = pd.to_datetime(df_dt["Date"])
    df_dt["BuyAfterSale"] = pd.to_numeric(df_dt.get("BuyAfterSale", 0), errors="coerce").fillna(0)
    df_dt["Volume"]       = pd.to_numeric(df_dt.get("Volume", 1),       errors="coerce").replace(0, 1)
    df_dt["Day_Trade_Volume"] = df_dt["BuyAfterSale"] / df_dt["Volume"]
    df_dt[["Date","stock_id","Day_Trade_Volume"]].to_parquet(PROCESSED_DIR / "daytrade_raw.parquet")
    log.info(f"    daytrade_raw: {df_dt.shape}")

# 9f. Holdings (大戶持股)  ────────────────────────────────────────────────────
log.info("  Assembling holdings_raw.parquet ...")
hold_files = sorted((CACHE_DIR / "weekly").glob("*_holdings.parquet"))
if hold_files:
    frames = []
    for f in tqdm(hold_files, desc="Loading holdings", leave=False):
        df = pd.read_parquet(f)
        df["Week"] = f.name[:10]
        frames.append(df)
    df_hold = pd.concat(frames, ignore_index=True)
    if "date" in df_hold.columns:
        df_hold = df_hold.rename(columns={"date": "Date"})

    RETAIL_LEVELS = {"1","2","3","1-999","1,000-5,000","5,001-10,000"}
    WHALE_LEVELS  = {"15","1,000,001-more",">1,000,000"}
    df_hold["Level_Str"] = df_hold["HoldingSharesLevel"].astype(str).str.strip()
    df_hold["Percent"]   = pd.to_numeric(df_hold["percent"], errors="coerce").fillna(0)
    df_hold["is_retail"] = df_hold["Level_Str"].isin(RETAIL_LEVELS)
    df_hold["is_whale"]  = df_hold["Level_Str"].isin(WHALE_LEVELS)
    retail = (df_hold[df_hold["is_retail"]].groupby(["Week","stock_id"])["Percent"]
              .sum().reset_index().rename(columns={"Percent":"Retail_Hold_Ratio"}))
    whale  = (df_hold[df_hold["is_whale"]].groupby(["Week","stock_id"])["Percent"]
              .sum().reset_index().rename(columns={"Percent":"Whale_Hold_Ratio"}))
    df_agg = pd.merge(whale, retail, on=["Week","stock_id"], how="outer").fillna(0)
    df_agg["Week"] = pd.to_datetime(df_agg["Week"])
    df_agg.to_parquet(PROCESSED_DIR / "holdings_raw.parquet")
    log.info(f"    holdings_raw: {df_agg.shape}")

# 9g. Revenue  ────────────────────────────────────────────────────────────────
log.info("  Assembling revenue_raw.parquet ...")
rev_files = sorted((CACHE_DIR / "monthly").glob("*_revenue.parquet"))
if rev_files:
    df_rev = pd.concat([pd.read_parquet(f) for f in tqdm(rev_files, desc="Loading revenue", leave=False)],
                       ignore_index=True)
    if "date" in df_rev.columns:
        df_rev = df_rev.rename(columns={"date": "Date"})
    df_rev["Date"] = pd.to_datetime(df_rev["Date"])
    df_rev.to_parquet(PROCESSED_DIR / "revenue_raw.parquet")
    log.info(f"    revenue_raw: {df_rev.shape}")

# 9h. Financials  ─────────────────────────────────────────────────────────────
log.info("  Assembling financials_raw.parquet ...")
fin_files = sorted((CACHE_DIR / "quarterly").glob("*_financials.parquet"))
if fin_files:
    df_fin = pd.concat([pd.read_parquet(f) for f in tqdm(fin_files, desc="Loading financials", leave=False)],
                       ignore_index=True)
    if "date" in df_fin.columns:
        df_fin = df_fin.rename(columns={"date": "Date"})
    df_fin["Date"] = pd.to_datetime(df_fin["Date"])
    df_fin.to_parquet(PROCESSED_DIR / "financials_raw.parquet")
    log.info(f"    financials_raw: {df_fin.shape}")

# ── 10. Summary ───────────────────────────────────────────────────────────────
log.info("")
log.info("=" * 60)
log.info("✅  All done! Files in Data/processed_v6/:")
for f in sorted(PROCESSED_DIR.glob("*.parquet")):
    size_mb = f.stat().st_size / 1_048_576
    log.info(f"    {f.name:<40}  {size_mb:6.1f} MB")
log.info("")
log.info("Next steps:")
log.info("  1. Zip the Data/processed_v6/ folder")
log.info("  2. Upload to Google Drive → MarketMamba_V6/processed_v6/")
log.info("  3. Run Colab Cell 3 (will restore from Drive, skip re-fetching)")
log.info("=" * 60)
