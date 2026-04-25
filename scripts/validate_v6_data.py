"""
MarketMamba V6 — Data Validation
==================================
Run after fetch_v6_data.py to verify all downloaded data is complete and sane.

Usage:
    python scripts/validate_v6_data.py

Outputs a full report to console + Data/validate_v6.log
"""

import sys
import io
import logging
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
ROOT_DIR      = SCRIPT_DIR.parent
DATA_DIR      = ROOT_DIR / "Data"
PROCESSED_DIR = DATA_DIR / "processed_v6"
CACHE_DIR     = DATA_DIR / "raw_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "validate_v6.log", encoding="utf-8", mode="w"),
    ],
)
log = logging.getLogger("validate_v6")

PASS = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"

issues: list[str] = []

def _check(cond: bool, msg_ok: str, msg_fail: str, fatal=False) -> bool:
    if cond:
        log.info(f"  {PASS}  {msg_ok}")
    else:
        tag = FAIL if fatal else WARN
        log.info(f"  {tag}  {msg_fail}")
        issues.append(msg_fail)
    return cond

def _section(title: str) -> None:
    log.info("")
    log.info("─" * 60)
    log.info(f"  {title}")
    log.info("─" * 60)

# ── 1. File existence check ───────────────────────────────────────────────────
_section("1. File Existence")

EXPECTED_FILES = {
    "prices_raw.parquet":         ("Price OHLCV (adjusted)",     True),
    "institutional_raw.parquet":  ("三大法人 buy/sell",           True),
    "margin_raw.parquet":         ("融資融券 balance",             True),
    "per_raw.parquet":            ("PER / PBR / DY",              True),
    "securities_raw.parquet":     ("借券成交量",                   True),
    "market_value_raw.parquet":   ("個股市值",                    True),
    "daytrade_raw.parquet":       ("當沖比率",                    True),
    "holdings_raw.parquet":       ("大戶持股分級",                 True),
    "revenue_raw.parquet":        ("月營收",                      True),
    "financials_raw.parquet":     ("綜合損益表",                   True),
    "balance_sheet_raw.parquet":  ("資產負債表",                   True),
    "cashflow_raw.parquet":       ("現金流量表",                   True),
    "macro_raw.parquet":          ("宏觀指標",                    True),
    "business_indicator.parquet": ("景氣燈號",                    False),
    "fed_rate.parquet":           ("FED 利率",                    False),
    "fear_greed.parquet":         ("CNN 恐懼貪婪",                False),
}

existing = {}
for fname, (desc, required) in EXPECTED_FILES.items():
    path = PROCESSED_DIR / fname
    exists = path.exists()
    size_mb = path.stat().st_size / 1_048_576 if exists else 0
    _check(exists,
           f"{fname:<40}  {size_mb:6.1f} MB  ({desc})",
           f"{fname} MISSING — {desc}",
           fatal=required)
    if exists:
        existing[fname] = path

# ── 2. Per-day cache coverage ─────────────────────────────────────────────────
_section("2. Per-Day Cache Coverage")

DAILY_CACHE = CACHE_DIR / "daily"
EXPECTED_TAGS = ["adj_price", "chip", "margin", "per", "securities", "mktval"]

for tag in EXPECTED_TAGS:
    files = sorted(DAILY_CACHE.glob(f"*_{tag}.parquet"))
    _check(len(files) >= 4000,
           f"{tag:<12}: {len(files):,} day-files found",
           f"{tag:<12}: only {len(files):,} day-files found (expected ≥4,000 for 2005-now)")

# Check for trading calendar gap: compare adj_price vs chip coverage
adj_days  = {f.name[:10] for f in DAILY_CACHE.glob("*_adj_price.parquet")}
chip_days = {f.name[:10] for f in DAILY_CACHE.glob("*_chip.parquet")}
gap = adj_days.symmetric_difference(chip_days)
_check(len(gap) < 20,
       f"adj_price ↔ chip gap: only {len(gap)} mismatched days (normal)",
       f"adj_price ↔ chip gap: {len(gap)} days differ — possible fetch errors")
if gap:
    sample = sorted(gap)[:5]
    log.info(f"           Sample gaps: {sample}")

# ── 3. prices_raw.parquet ────────────────────────────────────────────────────
_section("3. prices_raw.parquet  (Adjusted OHLCV)")

if "prices_raw.parquet" in existing:
    df = pd.read_parquet(existing["prices_raw.parquet"])
    df["Date"] = pd.to_datetime(df["Date"])

    n_stocks = df["stock_id"].nunique()
    n_dates  = df["Date"].nunique()
    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()

    log.info(f"  Rows: {len(df):,} | Stocks: {n_stocks:,} | Dates: {n_dates:,}")
    log.info(f"  Date range: {date_min} → {date_max}")

    _check(date_min <= date(2005, 3, 1),
           f"Start date OK: {date_min}",
           f"Start date {date_min} is later than expected 2005-01-01")
    _check(date_max >= date.today().replace(month=date.today().month - 1) if date.today().month > 1
           else date.today().replace(year=date.today().year - 1, month=12),
           f"End date OK: {date_max}",
           f"End date {date_max} may be stale")
    _check(n_stocks >= 1500,
           f"Stock count OK: {n_stocks:,}",
           f"Low stock count: {n_stocks} (expected ≥1,500)")

    for col in ["Close", "Open", "High", "Low", "Volume"]:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            neg_pct = (df[col] < 0).mean() * 100 if col != "Volume" else (df[col] < 0).mean() * 100
            _check(nan_pct < 5.0,
                   f"{col} NaN: {nan_pct:.2f}%",
                   f"{col} NaN rate high: {nan_pct:.2f}%")
            _check(neg_pct < 0.01,
                   f"{col} negative: {neg_pct:.4f}%",
                   f"{col} has {neg_pct:.4f}% negative values — check adjusted prices")

    # Quick sanity: Close should be between 1 and 100,000 TWD
    if "Close" in df.columns:
        out_of_range = ((df["Close"] < 1) | (df["Close"] > 100_000)).mean() * 100
        _check(out_of_range < 1.0,
               f"Close price range OK (<1 or >100k TWD: {out_of_range:.2f}%)",
               f"Close price out-of-range: {out_of_range:.2f}%")

# ── 4. institutional_raw.parquet ─────────────────────────────────────────────
_section("4. institutional_raw.parquet  (三大法人)")

if "institutional_raw.parquet" in existing:
    df = pd.read_parquet(existing["institutional_raw.parquet"])
    df["Date"] = pd.to_datetime(df["Date"])
    log.info(f"  Rows: {len(df):,} | Stocks: {df['stock_id'].nunique():,} | Dates: {df['Date'].nunique():,}")
    log.info(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.info(f"  Columns: {list(df.columns)}")
    for col in ["Foreign_Net", "Investment_Trust_Net", "Dealer_Net"]:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            _check(nan_pct < 30, f"{col} NaN: {nan_pct:.1f}%",
                   f"{col} high NaN: {nan_pct:.1f}%")

# ── 5. margin_raw.parquet ────────────────────────────────────────────────────
_section("5. margin_raw.parquet  (融資融券)")

if "margin_raw.parquet" in existing:
    df = pd.read_parquet(existing["margin_raw.parquet"])
    df["Date"] = pd.to_datetime(df["Date"])
    log.info(f"  Rows: {len(df):,} | Stocks: {df['stock_id'].nunique():,} | Dates: {df['Date'].nunique():,}")
    log.info(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    for col in ["Margin_Balance", "Short_Balance"]:
        if col in df.columns:
            neg_pct = (df[col] < 0).mean() * 100
            _check(neg_pct < 0.01, f"{col} ≥0: OK", f"{col} has {neg_pct:.3f}% negative values")

# ── 6. market_value_raw.parquet ──────────────────────────────────────────────
_section("6. market_value_raw.parquet  (市值)")

if "market_value_raw.parquet" in existing:
    df = pd.read_parquet(existing["market_value_raw.parquet"])
    df["Date"] = pd.to_datetime(df["Date"])
    log.info(f"  Rows: {len(df):,} | Stocks: {df['stock_id'].nunique():,} | Dates: {df['Date'].nunique():,}")
    log.info(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    if "market_value" in df.columns:
        mv = df["market_value"].dropna()
        log.info(f"  Market value range: {mv.min():,.0f} ~ {mv.max():,.0f} TWD")
        _check(mv.median() > 1e8, f"Median market cap > 1億: {mv.median():,.0f}", "Median market cap suspiciously low")

# ── 7. Quarterly financials ──────────────────────────────────────────────────
_section("7. Quarterly Data  (財報 / 負債表 / 現金流)")

for fname, label in [
    ("financials_raw.parquet",    "綜合損益表"),
    ("balance_sheet_raw.parquet", "資產負債表"),
    ("cashflow_raw.parquet",      "現金流量表"),
]:
    if fname in existing:
        df = pd.read_parquet(existing[fname])
        df["Date"] = pd.to_datetime(df["Date"])
        n_quarters = df["Date"].nunique()
        n_stocks   = df["stock_id"].nunique()
        log.info(f"  {label}: {len(df):,} rows | {n_stocks:,} stocks | {n_quarters} quarters")
        log.info(f"    Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
        _check(n_quarters >= 40, f"{label} quarters OK: {n_quarters}",
               f"{label} low quarter count: {n_quarters} (expected ≥40)")

# ── 8. macro_raw.parquet ─────────────────────────────────────────────────────
_section("8. macro_raw.parquet")

if "macro_raw.parquet" in existing:
    df = pd.read_parquet(existing["macro_raw.parquet"])
    df["Date"] = pd.to_datetime(df["Date"])
    log.info(f"  Rows: {len(df):,} | Dates: {df['Date'].nunique():,}")
    log.info(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    log.info(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        if col == "Date":
            continue
        nan_pct = df[col].isna().mean() * 100
        _check(nan_pct < 20, f"{col}: NaN {nan_pct:.1f}%",
               f"Macro column {col} has {nan_pct:.1f}% NaN — check yfinance / FinMind fetch")

# ── 9. Coverage cross-check ──────────────────────────────────────────────────
_section("9. Cross-File Date Coverage Check")

if "prices_raw.parquet" in existing and "institutional_raw.parquet" in existing:
    df_p = pd.read_parquet(existing["prices_raw.parquet"])
    df_c = pd.read_parquet(existing["institutional_raw.parquet"])
    p_dates = set(pd.to_datetime(df_p["Date"]).dt.date)
    c_dates = set(pd.to_datetime(df_c["Date"]).dt.date)
    p_only  = len(p_dates - c_dates)
    c_only  = len(c_dates - p_dates)
    _check(p_only < 20, f"Price-only dates: {p_only} (normal — pre-2005 or holidays)",
           f"Price has {p_only} dates not in chip — possible coverage gap")
    _check(c_only < 20, f"Chip-only dates:  {c_only}",
           f"Chip has {c_only} dates not in price")

# ── 10. Revenue & Holdings ───────────────────────────────────────────────────
_section("10. Revenue & Holdings")

if "revenue_raw.parquet" in existing:
    df = pd.read_parquet(existing["revenue_raw.parquet"])
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    log.info(f"  Revenue: {len(df):,} rows | {df['stock_id'].nunique():,} stocks | "
             f"{df['Date'].dt.to_period('M').nunique()} months")

if "holdings_raw.parquet" in existing:
    df = pd.read_parquet(existing["holdings_raw.parquet"])
    log.info(f"  Holdings: {len(df):,} rows | {df['stock_id'].nunique():,} stocks")

# ── Summary ───────────────────────────────────────────────────────────────────
_section("SUMMARY")

if not issues:
    log.info(f"  {PASS}  All checks passed! Data is ready for V6 pipeline.")
else:
    log.info(f"  Found {len(issues)} issue(s):")
    for i, issue in enumerate(issues, 1):
        log.info(f"   {i}. {issue}")

log.info("")
log.info("Next step: zip Data/processed_v6/ and upload to Google Drive.")
log.info("           Then run V6_Training.py on Colab.")
