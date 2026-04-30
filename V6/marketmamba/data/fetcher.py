"""
MarketMamba V6 — Data Fetcher
==============================
Hybrid data-source strategy:
  Layer 1 (fastest) : yfinance  → OHLCV price/volume (available 15 min after close)
  Layer 2 (fast)    : TWSE/TPEX direct → institutional investors (~16:30–17:00)
  Layer 3 (fallback): FinMind   → margin/short, 8 major banks, etc. (18:00–19:00)
                                  Forward-Fill used if not yet updated.

Target: inference can start at ~17:00 instead of 19:00+.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from marketmamba.config import (
    DATA_DIR,
    DATA_END_DATE,
    DATA_SOURCE_PRIORITY,
    DATA_START_DATE,
    FINMIND_TOKEN,
    MARGIN_FORWARD_FILL,
    PROCESSED_DIR,
    TPEX_INSTITUTIONAL_URL,
    TWSE_INSTITUTIONAL_URL,
)

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
RAW_DIR     = DATA_DIR / "raw_v6"
CACHE_DIR   = DATA_DIR / "cache_v6"
for _d in [RAW_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketMamba/6.0)"}
FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"

# ============================================================
# Ticker Universe Helpers
# ============================================================

def load_ticker_universe() -> tuple[list[str], list[str]]:
    """
    Load TSE (TWSE) and OTC (TPEX) stock ID lists.
    Returns (tse_ids, otc_ids) as plain 4-digit strings, e.g. ['2330', '2317', ...]
    Falls back to fetching from FinMind if local cache is missing.
    """
    cache_path = CACHE_DIR / "ticker_universe.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = _fetch_universe_from_finmind()
        df.to_parquet(cache_path)

    tse = df[df["market"] == "TSE"]["stock_id"].tolist()
    otc = df[df["market"] == "OTC"]["stock_id"].tolist()
    logger.info(f"Universe: {len(tse)} TSE + {len(otc)} OTC stocks")
    return tse, otc


def _fetch_universe_from_finmind() -> pd.DataFrame:
    """Fetch full stock list from FinMind (TaiwanStockInfo)."""
    logger.info("Fetching ticker universe from FinMind...")
    params = {
        "dataset": "TaiwanStockInfo",
        "token": FINMIND_TOKEN,
    }
    resp = requests.get(FINMIND_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != 200:
        raise RuntimeError(f"FinMind error: {data.get('msg')}")
    df = pd.DataFrame(data["data"])
    logger.info(f"  TaiwanStockInfo columns: {list(df.columns)}")

    # Filter: only ordinary 4-digit stock IDs
    df = df[df["stock_id"].str.match(r"^\d{4}$")].copy()

    # --- Determine exchange type ---
    # FinMind uses 'type' column (not 'market'), with values 'twse' / 'tpex'
    # Possible column names across different FinMind API versions:
    _market_col = None
    for _candidate in ["type", "market", "market_category", "exchange"]:
        if _candidate in df.columns:
            _market_col = _candidate
            break

    if _market_col:
        # Map FinMind exchange codes → our TSE / OTC convention
        _market_map = {
            "twse": "TSE", "TSE": "TSE", "sii": "TSE", "上市": "TSE",
            "tpex": "OTC", "OTC": "OTC", "otc": "OTC", "上櫃": "OTC",
        }
        df["market"] = df[_market_col].map(_market_map).fillna("TSE")
        logger.info(f"  Market column '{_market_col}' mapped → TSE/OTC")
    else:
        # No exchange column found — default everything to TSE
        # yfinance will fail on OTC suffix and we'll catch it via the missing list
        logger.warning("  No market/type column in TaiwanStockInfo — defaulting all to TSE")
        df["market"] = "TSE"

    tse_count = (df["market"] == "TSE").sum()
    otc_count  = (df["market"] == "OTC").sum()
    logger.info(f"  Universe: {tse_count} TSE + {otc_count} OTC stocks")

    return df[["stock_id", "stock_name", "industry_category", "market"]].reset_index(drop=True)


# ============================================================
# Layer 1: yfinance — Price/Volume
# ============================================================

def fetch_prices_yfinance(
    tse_ids: list[str],
    otc_ids: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Batch-download OHLCV for all stocks via yfinance.
    Returns long-format DataFrame with columns:
      [Date, stock_id, Open, High, Low, Close, Volume]
    Stocks missing from yfinance are flagged for FinMind fallback.
    """
    all_tickers = [f"{s}.TW" for s in tse_ids] + [f"{s}.TWO" for s in otc_ids]
    id_map = {f"{s}.TW": s for s in tse_ids}
    id_map.update({f"{s}.TWO": s for s in otc_ids})

    logger.info(f"yfinance: downloading {len(all_tickers)} tickers [{start} -> {end}]")

    # yfinance batch download — split into chunks to avoid rate limits
    BATCH_SIZE = 200  # yfinance starts rate-limiting above ~300 tickers at once
    all_records = []
    all_missing = []

    for batch_start in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[batch_start: batch_start + BATCH_SIZE]
        logger.info(f"  Batch {batch_start // BATCH_SIZE + 1}: {len(batch)} tickers...")

        for attempt in range(3):  # retry up to 3 times on rate limit
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                break
            except Exception as e:
                if "Too Many Requests" in str(e) or "429" in str(e):
                    wait = 30 * (attempt + 1)
                    logger.warning(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"  yfinance batch error: {e}")
                    raw = None
                    break
        else:
            # All 3 retries failed
            all_missing.extend(batch)
            continue

        if raw is None or raw.empty:
            all_missing.extend(batch)
            continue

        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    # Detect which level contains ticker names
                    # Old yfinance (group_by='ticker'): level 0 = Ticker, level 1 = Price
                    # New yfinance (>= 0.2.18):         level 0 = Price,  level 1 = Ticker
                    lvl0_vals = raw.columns.get_level_values(0).unique()
                    lvl1_vals = raw.columns.get_level_values(1).unique()
                    if ticker in lvl0_vals:
                        df_t = raw[ticker]           # old API
                    elif ticker in lvl1_vals:
                        df_t = raw.xs(ticker, axis=1, level=1)  # new API
                    else:
                        all_missing.append(ticker)
                        continue
                    df_t = df_t.dropna(subset=["Close"])
                else:
                    # Single-level columns (single ticker batch)
                    df_t = raw.dropna(subset=["Close"]) if len(batch) == 1 else pd.DataFrame()
            except (KeyError, ValueError):
                all_missing.append(ticker)
                continue
            if df_t.empty:
                all_missing.append(ticker)
                continue
            df_t = df_t.reset_index()
            df_t["stock_id"] = id_map[ticker]
            df_t["Date"] = pd.to_datetime(df_t["Date"]).dt.date
            all_records.append(df_t[["Date", "stock_id", "Open", "High", "Low", "Close", "Volume"]])

        if batch_start + BATCH_SIZE < len(all_tickers):
            time.sleep(3)  # polite pause between batches

    # Deduplicate: some delisted tickers appear in both TSE and OTC lists
    delisted_count  = sum(1 for t in all_missing if t in ["YFTzMissingError", "YFPricesMissingError"])
    rate_limit_count = len(all_missing) - delisted_count
    if all_missing:
        logger.warning(
            f"yfinance: {len(all_missing)} tickers unavailable "
            f"(many are delisted — this is expected for 2012+ historical data)"
        )
    if all_records:
        df_prices = pd.concat(all_records, ignore_index=True)
    else:
        df_prices = pd.DataFrame()
        logger.warning("yfinance returned no usable data")

    return df_prices, all_missing


# ============================================================
# Layer 2: TWSE / TPEX Direct — Institutional Investors
# ============================================================

def fetch_institutional_twse(date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetch TSE three-institutional investors (三大法人) directly from TWSE.
    Usually available by 16:30–17:00, ~30–60 min ahead of FinMind.

    Args:
        date_str: 'YYYY-MM-DD'
    Returns:
        DataFrame with columns [stock_id, Foreign_Net, Investment_Trust_Net, Dealer_Net]
        or None if data not yet published.
    """
    date_compact = date_str.replace("-", "")
    params = {"date": date_compact, "response": "json"}
    try:
        resp = requests.get(
            TWSE_INSTITUTIONAL_URL,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"TWSE institutional fetch failed: {e}")
        return None

    if data.get("stat") != "OK":
        logger.info(f"TWSE institutional: data not ready for {date_str} (stat={data.get('stat')})")
        return None

    rows = data.get("data", [])
    if not rows:
        return None

    records = []
    for row in rows:
        # TWSE T86 format: [股票代號, 股票名稱, 外資買, 外資賣, 外資淨, 投信買, 投信賣, 投信淨, 自營買, 自營賣, 自營淨, ...]
        try:
            stock_id = str(row[0]).strip()
            if not stock_id.isdigit() or len(stock_id) != 4:
                continue
            def _parse_int(s: str) -> int:
                return int(str(s).replace(",", "").replace("--", "0") or "0")

            records.append({
                "stock_id":            stock_id,
                "Foreign_Buy":         _parse_int(row[2]),
                "Foreign_Sell":        _parse_int(row[3]),
                "Foreign_Net":         _parse_int(row[4]),
                "Investment_Trust_Net": _parse_int(row[7]),
                "Dealer_Net":          _parse_int(row[10]),
            })
        except (IndexError, ValueError):
            continue

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info(f"TWSE direct: {len(df)} stocks institutional data for {date_str}")
    return df


def fetch_institutional_tpex(date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetch OTC (TPEX) institutional investor data directly from TPEX.
    Returns same schema as fetch_institutional_twse.
    """
    # TPEX uses Republic of China calendar year (民國年)
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    roc_year = dt.year - 1911
    date_roc = f"{roc_year}/{dt.month:02d}/{dt.day:02d}"

    params = {
        "l": "zh-tw",
        "se": "EW",
        "t": "D",
        "d": date_roc,
        "o": "json",
    }
    try:
        resp = requests.get(
            TPEX_INSTITUTIONAL_URL,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"TPEX institutional fetch failed: {e}")
        return None

    rows = data.get("aaData", [])
    if not rows:
        return None

    records = []
    for row in rows:
        try:
            stock_id = str(row[0]).strip()
            if not stock_id.isdigit() or len(stock_id) != 4:
                continue
            def _p(s: str) -> int:
                return int(str(s).replace(",", "") or "0")
            records.append({
                "stock_id":             stock_id,
                "Foreign_Buy":          _p(row[2]),
                "Foreign_Sell":         _p(row[3]),
                "Foreign_Net":          _p(row[4]),
                "Investment_Trust_Net": _p(row[10]),
                "Dealer_Net":           _p(row[13]),
            })
        except (IndexError, ValueError):
            continue

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info(f"TPEX direct: {len(df)} stocks institutional data for {date_str}")
    return df


# ============================================================
# Layer 3: FinMind — Margin, Short, Banks, etc.
# ============================================================

def _finmind_fetch(
    dataset:    str,
    start_date: str,
    end_date:   str,
    stock_id:   str | None = None,
) -> Optional[pd.DataFrame]:
    """Generic FinMind API call (single chunk). Returns None on failure."""
    if not FINMIND_TOKEN:
        logger.warning("FINMIND_TOKEN not set; skipping FinMind fetch")
        return None
    params = {
        "dataset":    dataset,
        "start_date": start_date,
        "end_date":   end_date,
        "token":      FINMIND_TOKEN,
    }
    if stock_id:
        params["stock_id"] = stock_id
    try:
        resp = requests.get(FINMIND_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != 200:
            logger.debug(f"FinMind {dataset}: {data.get('msg')} (stock={stock_id})")
            return None
        return pd.DataFrame(data["data"])
    except Exception as e:
        logger.debug(f"FinMind {dataset} chunk error: {e}")
        return None


def _finmind_fetch_chunked(
    dataset:    str,
    start_date: str,
    end_date:   str,
    stock_id:   str | None = None,
    chunk_years: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Fetch FinMind data in yearly chunks to respect free tier date limits.
    FinMind free tier rejects requests spanning > ~1825 days.
    Splits the [start_date, end_date] range into 'chunk_years'-year windows.
    """
    from datetime import datetime as _dt
    _start = _dt.strptime(start_date, "%Y-%m-%d")
    _end   = _dt.strptime(end_date,   "%Y-%m-%d")

    frames = []
    current = _start
    while current <= _end:
        chunk_end = min(
            _dt(current.year + chunk_years - 1, 12, 31),
            _end,
        )
        df = _finmind_fetch(
            dataset,
            start_date=current.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            stock_id=stock_id,
        )
        if df is not None and not df.empty:
            frames.append(df)
        current = _dt(current.year + chunk_years, 1, 1)
        time.sleep(0.6)  # ~60 req/min free tier

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def fetch_margin_finmind(date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetch margin purchase / short sale data from FinMind.
    If not yet updated and MARGIN_FORWARD_FILL=True, returns yesterday's cached data.
    """
    cache_path = CACHE_DIR / f"margin_{date_str}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = _finmind_fetch(
        "TaiwanStockMarginPurchaseShortSale",
        start_date=date_str,
        end_date=date_str,
    )

    if df is None or df.empty:
        if MARGIN_FORWARD_FILL:
            # Try to load yesterday's data
            yesterday = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday_cache = CACHE_DIR / f"margin_{yesterday}.parquet"
            if yesterday_cache.exists():
                logger.warning(f"Margin: FinMind not ready for {date_str}, using Forward Fill from {yesterday}")
                return pd.read_parquet(yesterday_cache)
        return None

    df.to_parquet(cache_path)
    return df


def fetch_prices_finmind(
    stock_ids: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Per-stock fallback price fetch for rate-limited tickers.
    Splits into YEARLY chunks to respect FinMind free tier 1825-day limit.
    Skips any stock where FinMind returns an error (likely delisted).
    """
    from datetime import datetime as _dt

    def _yearly_fetch(sid: str, y_start: str, y_end: str) -> pd.DataFrame | None:
        df = _finmind_fetch("TaiwanStockPrice", start_date=y_start, end_date=y_end, stock_id=sid)
        return df if (df is not None and not df.empty) else None

    start_year = int(start[:4])
    end_year   = int(end[:4])

    frames = []
    skipped = 0
    for i, sid in enumerate(stock_ids):
        stock_frames = []
        for yr in range(start_year, end_year + 1):
            y_start = f"{yr}-01-01" if yr > start_year else start
            y_end   = f"{yr}-12-31" if yr < end_year   else end
            chunk = _yearly_fetch(sid, y_start, y_end)
            if chunk is not None:
                stock_frames.append(chunk)
            time.sleep(0.3)  # ~60 req/min free tier

        if stock_frames:
            df_stock = pd.concat(stock_frames, ignore_index=True)
            # Normalise column names to V6 standard
            if "date" in df_stock.columns:
                df_stock = df_stock.rename(columns={"date": "Date"})
            if "stock_id" not in df_stock.columns:
                df_stock["stock_id"] = sid
            # Map FinMind OHLCV column names
            col_map = {"open": "Open", "max": "High", "min": "Low",
                       "close": "Close", "Trading_Volume": "Volume"}
            df_stock.rename(columns={k: v for k, v in col_map.items()
                                     if k in df_stock.columns}, inplace=True)
            keep = [c for c in ["Date", "stock_id", "Open", "High", "Low", "Close", "Volume"]
                    if c in df_stock.columns]
            frames.append(df_stock[keep])
        else:
            skipped += 1

        if i % 10 == 9:
            time.sleep(1.0)  # extra pause every 10 stocks

    logger.info(f"FinMind price fallback: {len(frames)} fetched, {skipped} skipped (likely delisted)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# Orchestrator — Full Historical Sync
# ============================================================

def run_full_data_sync(
    start: str = DATA_START_DATE,
    end:   str | None = DATA_END_DATE,
    force_rebuild: bool = False,
) -> list[str]:
    """
    Main entry point. Pulls all data needed for V6 training.

    Returns:
        List of trading day strings 'YYYY-MM-DD' that were synced.

    Strategy:
        1. Price/Volume: yfinance batch (fast) + FinMind fallback
        2. Institutional: TWSE/TPEX direct + FinMind fallback
        3. Margin/Short: FinMind (or Forward Fill if not ready)
        4. Monthly Revenue + Fundamentals: FinMind (monthly, cached)
        5. Macro (VIX, SPX, Gold, Oil): yfinance
    """
    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"=== V6 Full Data Sync: {start} → {end} ===")

    tse_ids, otc_ids = load_ticker_universe()

    # --- Step 1: Price/Volume ---
    price_cache = PROCESSED_DIR / "prices_raw.parquet"
    if force_rebuild or not price_cache.exists():
        df_prices, missing_tickers = fetch_prices_yfinance(tse_ids, otc_ids, start, end)
        # Fallback for missing
        if missing_tickers:
        # Fallback for missing — MUST replace .TWO before .TW to avoid '6516O' artefact
            missing_ids = [t.replace(".TWO", "").replace(".TW", "") for t in missing_tickers]
            df_fallback = fetch_prices_finmind(missing_ids, start, end)
            if not df_fallback.empty:
                df_prices = pd.concat([df_prices, df_fallback], ignore_index=True)
        df_prices.to_parquet(price_cache)
        logger.info(f"Prices saved: {len(df_prices):,} rows")
    else:
        df_prices = pd.read_parquet(price_cache)
        logger.info(f"Prices loaded from cache: {len(df_prices):,} rows")

    # --- Step 2: Institutional Investors (chunked — 14 years > FinMind free tier limit) ---
    inst_cache = PROCESSED_DIR / "institutional_raw.parquet"
    if force_rebuild or not inst_cache.exists():
        logger.info("Fetching institutional data via FinMind (chunked yearly)...")
        df_inst = _finmind_fetch_chunked(
            "TaiwanStockInstitutionalInvestors",
            start_date=start,
            end_date=end,
        )
        if df_inst is not None and not df_inst.empty:
            col_map = {
                "date":                         "Date",
                "Foreign_Investor_Buy":          "Foreign_Buy",
                "Foreign_Investor_Sell":         "Foreign_Sell",
                "Foreign_Investor_Buy__Sell":    "Foreign_Net",
                "Investment_Trust_Buy":          "Investment_Trust_Buy",
                "Investment_Trust_Sell":         "Investment_Trust_Sell",
                "Investment_Trust_Buy__Sell":    "Investment_Trust_Net",
                "Dealer_proprietary_Buy__Sell":  "Dealer_Net",
            }
            df_inst.rename(columns={k: v for k, v in col_map.items() if k in df_inst.columns}, inplace=True)
            df_inst.to_parquet(inst_cache)
            logger.info(f"Institutional saved: {len(df_inst):,} rows")
        else:
            logger.warning("FinMind institutional data unavailable")
    else:
        logger.info(f"Institutional loaded from cache: {(PROCESSED_DIR / 'institutional_raw.parquet').stat().st_size // 1024:,} KB")

    # --- Step 3: Margin / Short Sale (chunked — 14 years > FinMind free tier limit) ---
    margin_cache = PROCESSED_DIR / "margin_raw.parquet"
    if force_rebuild or not margin_cache.exists():
        logger.info("Fetching margin data via FinMind (chunked yearly)...")
        df_margin = _finmind_fetch_chunked(
            "TaiwanStockMarginPurchaseShortSale",
            start_date=start,
            end_date=end,
        )
        if df_margin is not None and not df_margin.empty:
            if "date" in df_margin.columns:
                df_margin = df_margin.rename(columns={"date": "Date"})
            df_margin.to_parquet(margin_cache)
            logger.info(f"Margin saved: {len(df_margin):,} rows")
        else:
            logger.warning("Margin data unavailable from FinMind")
            df_margin = pd.DataFrame()
    else:
        df_margin = pd.read_parquet(margin_cache)
        logger.info(f"Margin loaded from cache: {len(df_margin):,} rows")

    # --- Step 4: Monthly Revenue & Fundamentals ---
    _sync_monthly_data(force_rebuild)

    # --- Step 5: Macro ---
    _sync_macro_data(start, end, force_rebuild)

    trading_days = _get_trading_days(df_prices)
    logger.info(f"=== Sync complete: {len(trading_days)} trading days ===")
    return trading_days


# ============================================================
# Daily Update — Inference Mode (Fast Path)
# ============================================================

def run_daily_update(target_date: str | None = None) -> str:
    """
    Lightweight update for daily inference.
    Only fetches today's data; uses cache for history.

    Returns:
        The trading date string that was fetched ('YYYY-MM-DD').
    """
    today = target_date or date.today().strftime("%Y-%m-%d")
    logger.info(f"=== V6 Daily Update: {today} ===")

    tse_ids, otc_ids = load_ticker_universe()

    # Price — yfinance first, FinMind fallback if empty
    df_prices, missing = fetch_prices_yfinance(tse_ids, otc_ids, today, today)
    if df_prices.empty:
        logger.info("yfinance returned no data → falling back to FinMind for today's prices...")
        all_ids = tse_ids + otc_ids
        df_prices = fetch_prices_finmind(all_ids, today, today)
        if df_prices.empty:
            logger.warning("FinMind also returned no price data — prices not updated for today")
    if not df_prices.empty:
        _append_to_parquet(PROCESSED_DIR / "prices_raw.parquet", df_prices, today)

    # Institutional — TWSE/TPEX direct (~16:30)
    df_tse = fetch_institutional_twse(today)
    df_otc = fetch_institutional_tpex(today)
    inst_frames = [x for x in [df_tse, df_otc] if x is not None]
    if inst_frames:
        inst_today = pd.concat(inst_frames, ignore_index=True)
        if not inst_today.empty:
            inst_today["Date"] = today
            _append_to_parquet(PROCESSED_DIR / "institutional_raw.parquet", inst_today, today)
    else:
        logger.warning(
            f"No institutional data for {today} "
            f"(market may be closed or data not yet published — will use forward-fill)"
        )

    # Margin — FinMind or Forward Fill
    fetch_margin_finmind(today)  # caches to CACHE_DIR automatically

    logger.info(f"=== Daily update done: {today} ===")
    return today



# ============================================================
# Macro Data (VIX, SPX, Gold, Oil, USD/TWD)
# ============================================================

def _sync_macro_data(start: str, end: str, force: bool = False) -> None:
    macro_cache = PROCESSED_DIR / "macro_raw.parquet"
    if not force and macro_cache.exists():
        logger.info("Macro loaded from cache")
        return

    macro_tickers = {
        "^VIX":    "VIX",
        "^GSPC":   "SPX",
        "GC=F":    "Gold",
        "CL=F":    "Oil",
        "^TNX":    "TNX",
        "TWD=X":   "USD_TWD",
    }
    frames = []
    for ticker, col in macro_tickers.items():
        df_t = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df_t.empty:
            continue
        # --- FIX: flatten MultiIndex from new yfinance API ---
        # New yfinance (>= 0.2.18) returns MultiIndex (Price, Ticker) for all downloads
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
            df_t = df_t.loc[:, ~df_t.columns.duplicated()]  # drop duplicate col names
        # ---
        df_t = df_t[["Close"]].rename(columns={"Close": col}).reset_index()
        if "Date" not in df_t.columns and "index" in df_t.columns:
            df_t.rename(columns={"index": "Date"}, inplace=True)
        df_t["Date"] = pd.to_datetime(df_t["Date"])
        frames.append(df_t)

    if not frames:
        logger.warning("No macro data fetched")
        return

    df_macro = frames[0]
    for df_t in frames[1:]:
        df_macro = df_macro.merge(df_t, on="Date", how="outer")
    df_macro.sort_values("Date", inplace=True)
    df_macro.to_parquet(macro_cache)
    logger.info(f"Macro saved: {df_macro.shape}")


def _sync_monthly_data(force: bool = False) -> None:
    """Fetch monthly revenue and quarterly financials from FinMind in yearly chunks."""
    today = date.today().strftime("%Y-%m-%d")

    revenue_cache = PROCESSED_DIR / "revenue_raw.parquet"
    if not force and revenue_cache.exists():
        logger.info("Monthly revenue loaded from cache")
    else:
        logger.info("Fetching monthly revenue via FinMind (chunked yearly)...")
        df_rev = _finmind_fetch_chunked(
            "TaiwanStockMonthRevenue",
            start_date=DATA_START_DATE,
            end_date=today,
        )
        if df_rev is not None and not df_rev.empty:
            df_rev.to_parquet(revenue_cache)
            logger.info(f"Revenue saved: {df_rev.shape}")
        else:
            logger.warning("Revenue data unavailable from FinMind")

    fin_cache = PROCESSED_DIR / "financials_raw.parquet"
    if not force and fin_cache.exists():
        logger.info("Financial statements loaded from cache")
    else:
        logger.info("Fetching financial statements via FinMind (chunked yearly)...")
        df_fin = _finmind_fetch_chunked(
            "TaiwanStockFinancialStatements",
            start_date=DATA_START_DATE,
            end_date=today,
        )
        if df_fin is not None and not df_fin.empty:
            df_fin.to_parquet(fin_cache)
            logger.info(f"Financials saved: {df_fin.shape}")
        else:
            logger.warning("Financial statements unavailable from FinMind")


# ============================================================
# Utility Helpers
# ============================================================

def _get_trading_days(df_prices: pd.DataFrame) -> list[str]:
    """Extract sorted unique trading day strings from a price DataFrame."""
    days = pd.to_datetime(df_prices["Date"]).dt.strftime("%Y-%m-%d").unique().tolist()
    return sorted(days)


def _append_to_parquet(path: Path, df_new: pd.DataFrame, date_str: str) -> None:
    """
    Append new rows to an existing parquet, replacing rows for date_str if present.
    Always normalizes the Date column to YYYY-MM-DD string to prevent schema conflicts
    (e.g., existing parquet may have Date as int64/datetime, new data as string).
    """
    if df_new.empty:
        return

    # Normalize df_new Date to string
    df_new = df_new.copy()
    if "Date" in df_new.columns:
        df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.strftime("%Y-%m-%d")

    if path.exists():
        df_old = pd.read_parquet(path)
        # Normalize df_old Date to string (it might be int64/datetime from original sync)
        if "Date" in df_old.columns:
            df_old["Date"] = pd.to_datetime(df_old["Date"]).dt.strftime("%Y-%m-%d")
        # Remove any existing rows for this date before appending
        df_old = df_old[df_old["Date"] != date_str]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_parquet(path)

