"""
MarketMamba V6 — Data Merger
==============================
Merges all raw data sources into a single training-ready DataFrame.
Cross-frequency alignment with look-ahead bias protection:

  Daily (price, institutional, macro) : aligned by exact Date
  Monthly (revenue)                    : available 10 days after month-end → as-of join
  Quarterly (financials/EPS/PER)       : available 45 days after quarter-end → as-of join

Output: a long-format DataFrame indexed by (Date, stock_id)
ready to pass into feature_engineer.build_features().
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from marketmamba.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


# ============================================================
# Main Merger
# ============================================================

def merge_all_data() -> dict[str, pd.DataFrame]:
    """
    Load all cached raw parquet files and return them as a dict.
    The feature engineer will merge them with proper look-ahead protection.

    Returns:
        {
          'prices'       : pd.DataFrame,          # adjusted OHLCV
          'inst'         : pd.DataFrame | None,   # 三大法人
          'margin'       : pd.DataFrame | None,   # 融資融券
          'per'          : pd.DataFrame | None,   # PER/PBR/DY
          'securities'   : pd.DataFrame | None,   # 借券 (short interest)
          'market_value' : pd.DataFrame | None,   # 個股市値
          'daytrade'     : pd.DataFrame | None,   # 當沖比率
          'holdings'     : pd.DataFrame | None,   # 大戶持股分級
          'revenue'      : pd.DataFrame | None,   # 月營收
          'financials'   : pd.DataFrame | None,   # 綜合損益表
          'balance_sheet': pd.DataFrame | None,   # 資產負債表
          'cashflow'     : pd.DataFrame | None,   # 現金流量表
          'macro'        : pd.DataFrame | None,   # 宏觀指標
        }
    """
    def _load(name: str) -> pd.DataFrame | None:
        path = PROCESSED_DIR / f"{name}_raw.parquet"
        if not path.exists():
            logger.warning(f"Raw data not found: {path}")
            return None
        df = pd.read_parquet(path)
        # Normalise date column to 'Date'
        if "date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"date": "Date"})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        logger.info(f"  {name}: {df.shape[0]:,} rows")
        return df

    def _load_plain(name: str) -> pd.DataFrame | None:
        """Load non-_raw suffixed files (fear_greed.parquet, etc.)"""
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            logger.warning(f"Data not found: {path}")
            return None
        df = pd.read_parquet(path)
        if "date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"date": "Date"})
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        logger.info(f"  {name}: {df.shape[0]:,} rows")
        return df

    logger.info("Loading raw data sources...")
    data = {
        "prices":        _load("prices"),
        "inst":          _load("institutional"),
        "margin":        _load("margin"),
        "per":           _load("per"),
        "securities":    _load("securities"),
        "market_value":  _load("market_value"),
        "daytrade":      _load("daytrade"),
        "holdings":      _load("holdings"),
        "revenue":       _load("revenue"),
        "financials":    _load("financials"),
        "balance_sheet": _load("balance_sheet"),
        "cashflow":      _load("cashflow"),
        "macro":         _load("macro"),
        # V6.1 new data sources
        "futures_inst":       _load("futures_institutional"),
        "options_inst":       _load("options_institutional"),
        "dividend":           _load("dividend"),
        "foreign_shareholding": _load("foreign_shareholding"),
        "fear_greed":         _load_plain("fear_greed"),
        "business_indicator": _load_plain("business_indicator"),
        "fed_rate":           _load_plain("fed_rate"),
    }

    if data["prices"] is None:
        raise FileNotFoundError(
            "prices_raw.parquet not found. "
            "Restore the Drive snapshot (Cell 2) before running this cell."
        )

    return data


# ============================================================
# Cross-Frequency Alignment Utilities
# ============================================================

def align_monthly_to_daily(
    df_daily:   pd.DataFrame,   # must have [Date, stock_id]
    df_monthly: pd.DataFrame,   # must have [date, stock_id, <value_cols>]
    value_cols: list[str],
    lag_days:   int = 11,       # publish lag after month-end (safe default: 11 days)
) -> pd.DataFrame:
    """
    As-of join: for each (Date, stock_id) in df_daily, find the latest monthly
    observation where (month_end + lag_days) <= Date.

    Args:
        df_daily   : daily base DataFrame
        df_monthly : monthly data (FinMind format)
        value_cols : which columns to bring into df_daily
        lag_days   : days after month-end before data is officially available

    Returns:
        df_daily with new columns from df_monthly (NaN if no data available yet)
    """
    df_m = df_monthly.copy()
    df_m["date"] = pd.to_datetime(df_m["date"])
    df_m["available_from"] = (
        df_m["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)
    )
    df_m = df_m.sort_values(["stock_id", "available_from"])

    df_daily = df_daily.copy()
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    for col in value_cols:
        df_daily[col] = np.nan

    result_frames = []
    for sid, sub_d in df_daily.groupby("stock_id", sort=False):
        sub_m = df_m[df_m["stock_id"] == sid].sort_values("available_from")
        if sub_m.empty:
            result_frames.append(sub_d)
            continue

        sub_d = sub_d.copy()
        avail_ts  = sub_m["available_from"].values
        val_arrays = {col: sub_m[col].values for col in value_cols if col in sub_m.columns}

        for col, vals in val_arrays.items():
            # Searchsorted: find latest published month for each date
            positions = np.searchsorted(avail_ts, sub_d["Date"].values, side="right") - 1
            valid_mask = positions >= 0
            sub_d.loc[valid_mask, col] = vals[positions[valid_mask]]

        result_frames.append(sub_d)

    return pd.concat(result_frames, ignore_index=True)


def align_quarterly_to_daily(
    df_daily:    pd.DataFrame,
    df_quarterly: pd.DataFrame,   # must have [date, stock_id, <value_cols>]
    value_cols:  list[str],
    lag_days:    int = 45,        # ~45 days after quarter-end for financial statements
) -> pd.DataFrame:
    """
    Same as align_monthly_to_daily, but for quarterly fundamental data.
    """
    return align_monthly_to_daily(
        df_daily, df_quarterly, value_cols, lag_days=lag_days
    )


def add_twii_to_macro(df_macro: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TWII (market-cap-weighted index proxy) from individual stock prices,
    or extract it from df_macro if already present.

    If neither source has TWII, adds a column of zeros (model handles missing macro).
    """
    if "TWII" in df_macro.columns or "TWII_Return" in df_macro.columns:
        return df_macro

    logger.info("TWII not in macro data — computing proxy from equal-weight avg close")
    df_m = df_macro.copy()
    df_prices["Date"] = pd.to_datetime(df_prices["Date"])

    daily_avg = (
        df_prices.groupby("Date")["Close"]
        .mean()
        .rename("TWII")
        .reset_index()
    )
    df_m = df_m.merge(daily_avg, on="Date", how="left")
    df_m["TWII"] = df_m["TWII"].ffill()
    return df_m


def validate_data_integrity(data: dict[str, pd.DataFrame | None]) -> dict[str, any]:
    """
    Run basic sanity checks on the raw data dict.
    Returns a dict with diagnostics.
    """
    report = {}
    prices = data.get("prices")
    if prices is not None:
        report["n_stocks"]      = prices["stock_id"].nunique()
        report["n_dates"]       = prices["Date"].nunique()
        report["date_range"]    = (str(prices["Date"].min()), str(prices["Date"].max()))
        report["close_na_pct"]  = float(prices["Close"].isna().mean())
        report["volume_zero_pct"] = float((prices["Volume"] == 0).mean())

    inst = data.get("inst")
    if inst is not None:
        report["inst_coverage"] = float(
            inst["stock_id"].nunique() / max(report.get("n_stocks", 1), 1)
        )

    margin = data.get("margin")
    if margin is not None:
        report["margin_coverage"] = float(
            margin["stock_id"].nunique() / max(report.get("n_stocks", 1), 1)
        )

    logger.info(f"Data integrity report: {report}")
    return report
