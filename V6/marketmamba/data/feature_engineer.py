"""
MarketMamba V6 — Feature Engineer
===================================
Builds the 46-dimensional pure-quant feature matrix from raw price, institutional,
margin, fundamental, and macro data.

Feature groups (must match config.FEATURE_GROUPS exactly):
  Group A — price_momentum      (12 dims): OHLCV, returns, MAs, RSI, ATR
  Group B — institutional_flow  (16 dims): 3 institutional, margin, KD, OBV, vol
  Group C — fundamentals        (10 dims): revenue, EPS, valuation, profitability
  Group D — macro_environment   ( 8 dims): index returns, VIX, rates, commodities
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from marketmamba.config import FEATURE_COLS, PROCESSED_DIR, SEQ_LEN

logger = logging.getLogger(__name__)


# ============================================================
# Main Entry Point
# ============================================================

def build_features(
    df_price:  pd.DataFrame,
    df_inst:   pd.DataFrame,
    df_margin: pd.DataFrame,
    df_rev:    pd.DataFrame,
    df_fin:    pd.DataFrame,
    df_macro:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all raw data sources and compute the 46-dim feature matrix.

    Args:
        df_price  : raw OHLCV [Date, stock_id, Open, High, Low, Close, Volume]
        df_inst   : institutional [Date, stock_id, Foreign_Buy/Sell/Net, ...]
        df_margin : margin/short  [Date, stock_id, MarginPurchase, ...]
        df_rev    : monthly revenue [Date, stock_id, Revenue, ...]
        df_fin    : financials [Date, stock_id, EPS, PER, PBR, ...]
        df_macro  : macro [Date, VIX, SPX, Gold, Oil, TNX, USD_TWD]

    Returns:
        df : MultiIndex [Date, stock_id] with all 46 feature columns + target columns
    """
    logger.info("Building V6 feature matrix (46D)...")

    df = df_price.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["stock_id", "Date"]).reset_index(drop=True)

    # -- Group A: Price / Momentum --
    df = _add_price_momentum_features(df)

    # -- Group B: Institutional / Margin / Technical --
    df = _merge_institutional(df, df_inst)
    df = _merge_margin(df, df_margin)
    df = _add_technical_b_features(df)

    # -- Group C: Fundamentals --
    df = _merge_fundamentals(df, df_rev, df_fin)

    # -- Group D: Macro --
    df = _merge_macro(df, df_macro)

    # -- Targets: 5d / 20d / 60d Alpha vs TWII --
    df = _add_alpha_targets(df, df_macro)

    # -- Sanity check --
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns (will be filled with 0): {missing_cols}")
        for c in missing_cols:
            df[c] = 0.0

    # Reorder to match FEATURE_COLS order
    meta_cols = ["Date", "stock_id", "Alpha_5d", "Alpha_20d", "Alpha_60d"]
    df = df[meta_cols + FEATURE_COLS].copy()

    logger.info(f"Feature matrix: {df.shape[0]:,} rows × {len(FEATURE_COLS)} features")
    return df


# ============================================================
# Group A — Price / Momentum (12 dims)
# ============================================================

def _add_price_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stock_id", sort=False)

    # Returns
    df["Return_1d"]  = g["Close"].pct_change(1)
    df["Return_5d"]  = g["Close"].pct_change(5)
    df["Return_20d"] = g["Close"].pct_change(20)

    # Moving Averages
    df["MA_20"] = g["Close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["MA_60"] = g["Close"].transform(lambda x: x.rolling(60, min_periods=30).mean())

    # RSI (14-day)
    df["RSI_14"] = g["Close"].transform(_compute_rsi)

    # ATR (14-day True Range)
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - g["Close"].shift(1)).abs(),
        (df["Low"]  - g["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = g["TR"].transform(lambda x: x.ewm(span=14, min_periods=7).mean())
    df.drop(columns=["TR"], inplace=True)

    # Volatility_20d (log-return std)
    log_ret = g["Close"].transform(lambda x: np.log(x / x.shift(1)))
    df["Volatility_20d"] = log_ret.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# Group B — Institutional / Technical (16 dims)
# ============================================================

def _merge_institutional(df: pd.DataFrame, df_inst: pd.DataFrame) -> pd.DataFrame:
    if df_inst is None or df_inst.empty:
        for col in ["Foreign_Buy", "Foreign_Sell", "Foreign_Net",
                    "Investment_Trust_Net", "Dealer_Net"]:
            df[col] = 0.0
        return df

    df_inst = df_inst.copy()
    df_inst["Date"] = pd.to_datetime(df_inst["Date"])
    merge_cols = ["Foreign_Buy", "Foreign_Sell", "Foreign_Net",
                  "Investment_Trust_Net", "Dealer_Net"]
    df = df.merge(df_inst[["Date", "stock_id"] + merge_cols],
                  on=["Date", "stock_id"], how="left")
    for col in merge_cols:
        df[col] = df[col].fillna(0.0)
    return df


def _merge_margin(df: pd.DataFrame, df_margin: pd.DataFrame) -> pd.DataFrame:
    """Merge margin purchase / short sale data, forward-fill gaps."""
    if df_margin is None or df_margin.empty:
        for col in ["Margin_Purchase", "Margin_Repay", "Short_Sale",
                    "Short_Cover", "Margin_Balance", "Short_Balance"]:
            df[col] = 0.0
        return df

    df_margin = df_margin.copy()
    df_margin["Date"] = pd.to_datetime(df_margin["Date"])
    margin_cols = ["Margin_Purchase", "Margin_Repay", "Short_Sale",
                   "Short_Cover", "MarginPurchaseBalance", "ShortSaleBalance"]

    # Map FinMind column names → V6 names
    col_map = {
        "MarginPurchase":        "Margin_Purchase",
        "MarginRepay":           "Margin_Repay",
        "ShortSale":             "Short_Sale",
        "ShortCover":            "Short_Cover",
        "MarginPurchaseBalance": "Margin_Balance",
        "ShortSaleBalance":      "Short_Balance",
    }
    df_margin.rename(columns=col_map, inplace=True)
    valid_cols = [c for c in col_map.values() if c in df_margin.columns]

    df = df.merge(df_margin[["Date", "stock_id"] + valid_cols],
                  on=["Date", "stock_id"], how="left")

    # Forward-fill within each stock (handles Forward Fill fallback days)
    for col in valid_cols:
        df[col] = df.groupby("stock_id")[col].transform(
            lambda x: x.ffill().fillna(0.0)
        )

    # Ensure all expected columns exist
    for col in ["Margin_Purchase", "Margin_Repay", "Short_Sale",
                "Short_Cover", "Margin_Balance", "Short_Balance"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _add_technical_b_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add KD stochastic, OBV, and Day_Trade_Volume placeholder."""
    g = df.groupby("stock_id", sort=False)

    # KD Stochastic (9,3,3)
    low_9  = g["Low"].transform(lambda x: x.rolling(9, min_periods=5).min())
    high_9 = g["High"].transform(lambda x: x.rolling(9, min_periods=5).max())
    rsv = ((df["Close"] - low_9) / (high_9 - low_9 + 1e-9)) * 100
    df["KD_K"] = g["stock_id"].transform(lambda _: None)  # placeholder
    # Compute K/D iteratively per stock
    k_vals, d_vals = [], []
    for sid, sub in df.groupby("stock_id", sort=False):
        rsv_s = rsv.loc[sub.index]
        k = rsv_s.ewm(com=2, adjust=False).mean()  # smoothing factor 1/3
        d = k.ewm(com=2, adjust=False).mean()
        k_vals.append(k)
        d_vals.append(d)
    df["KD_K"] = pd.concat(k_vals).reindex(df.index)
    df["KD_D"] = pd.concat(d_vals).reindex(df.index)

    # OBV — On Balance Volume
    direction = np.sign(df["Return_1d"].fillna(0))
    df["OBV"] = (direction * df["Volume"]).groupby(df["stock_id"]).cumsum()
    # Normalize OBV per stock (z-score rolling)
    df["OBV"] = g["OBV"].transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) /
                  (x.rolling(60, min_periods=20).std() + 1e-9)
    )

    # Day trade volume (placeholder — will be filled from FinMind if available)
    if "Day_Trade_Volume" not in df.columns:
        df["Day_Trade_Volume"] = 0.0

    return df


# ============================================================
# Group C — Fundamentals (10 dims)
# ============================================================

def _merge_fundamentals(
    df: pd.DataFrame,
    df_rev: pd.DataFrame,
    df_fin: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge monthly revenue and quarterly financial statements.
    Uses 'as-of' join to avoid look-ahead bias:
      - Revenue: published on 10th of following month → safe after that date
      - Financials: published ~45 days after quarter end → safe after that date
    """
    fund_defaults = {
        "PER": 15.0, "PBR": 1.5,
        "Revenue_MoM": 0.0, "Revenue_YoY": 0.0,
        "EPS": 0.0, "EPS_Surprise": 0.0,
        "Gross_Margin": 0.3, "ROE": 0.1,
        "Market_Cap_Log": 0.0, "Book_Value": 0.0,
    }
    for col, default in fund_defaults.items():
        if col not in df.columns:
            df[col] = default

    if df_rev is not None and not df_rev.empty:
        df = _merge_revenue(df, df_rev)

    if df_fin is not None and not df_fin.empty:
        df = _merge_financial_statements(df, df_fin)

    return df


def _merge_revenue(df: pd.DataFrame, df_rev: pd.DataFrame) -> pd.DataFrame:
    df_rev = df_rev.copy()
    # FinMind column: date (year-month), stock_id, revenue
    df_rev["date"] = pd.to_datetime(df_rev["date"])
    # Revenue is published on the 10th of following month; safe to use from 11th onward
    df_rev["available_from"] = df_rev["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=11)

    df_rev = df_rev.sort_values(["stock_id", "date"])
    df_rev["Revenue_MoM"] = df_rev.groupby("stock_id")["revenue"].pct_change(1).fillna(0)
    df_rev["Revenue_YoY"] = df_rev.groupby("stock_id")["revenue"].pct_change(12).fillna(0)

    # As-of merge: for each (stock_id, Date) in df, use latest revenue available
    merged_rows = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_rev = df_rev[df_rev["stock_id"] == sid].sort_values("available_from")
        sub_df = sub_df.copy()
        sub_df["Revenue_MoM"] = _asof_lookup(sub_df["Date"], sub_rev["available_from"], sub_rev["Revenue_MoM"])
        sub_df["Revenue_YoY"] = _asof_lookup(sub_df["Date"], sub_rev["available_from"], sub_rev["Revenue_YoY"])
        merged_rows.append(sub_df)

    if merged_rows:
        df = pd.concat(merged_rows, ignore_index=True)
    return df


def _merge_financial_statements(df: pd.DataFrame, df_fin: pd.DataFrame) -> pd.DataFrame:
    """Merge quarterly EPS, PER, PBR, gross margin, ROE with look-ahead protection."""
    df_fin = df_fin.copy()
    df_fin["date"] = pd.to_datetime(df_fin["date"])
    # Financials available ~45 days after quarter end
    df_fin["available_from"] = df_fin["date"] + pd.Timedelta(days=45)
    df_fin = df_fin.sort_values(["stock_id", "date"])

    fin_cols = [c for c in ["EPS", "PER", "PBR", "Gross_Margin", "ROE", "Book_Value"] if c in df_fin.columns]
    if not fin_cols:
        return df

    merged_rows = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_fin = df_fin[df_fin["stock_id"] == sid].sort_values("available_from")
        sub_df = sub_df.copy()
        for col in fin_cols:
            sub_df[col] = _asof_lookup(sub_df["Date"], sub_fin["available_from"], sub_fin[col])
        merged_rows.append(sub_df)

    if merged_rows:
        df = pd.concat(merged_rows, ignore_index=True)

    # EPS_Surprise = EPS_this_quarter / EPS_same_quarter_last_year - 1
    if "EPS" in df.columns:
        df["EPS_Surprise"] = (
            df.groupby("stock_id")["EPS"].pct_change(4).fillna(0)
        )

    # Market_Cap_Log
    if "Market_Cap_Log" not in df.columns or (df["Market_Cap_Log"] == 0).all():
        if "PBR" in df.columns and "Book_Value" in df.columns:
            market_cap = df["PBR"] * df["Book_Value"]
            df["Market_Cap_Log"] = np.log1p(market_cap.clip(lower=0))

    return df


# ============================================================
# Group D — Macro (8 dims)
# ============================================================

def _merge_macro(df: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    macro_defaults = {
        "TWII_Return": 0.0, "SPX_Return": 0.0,
        "VIX": 20.0, "TNX": 4.0,
        "Gold_Return": 0.0, "Oil_Return": 0.0,
        "USD_TWD": 30.0, "Market_Closed": 0.0,
    }
    for col, default in macro_defaults.items():
        if col not in df.columns:
            df[col] = default

    if df_macro is None or df_macro.empty:
        return df

    df_macro = df_macro.copy()
    df_macro["Date"] = pd.to_datetime(df_macro["Date"])

    # Compute returns for index-type columns
    for raw_col, ret_col in [("SPX", "SPX_Return"), ("Gold", "Gold_Return"), ("Oil", "Oil_Return")]:
        if raw_col in df_macro.columns:
            df_macro[ret_col] = df_macro[raw_col].pct_change(1).fillna(0)

    if "TWII" in df_macro.columns:
        df_macro["TWII_Return"] = df_macro["TWII"].pct_change(1).fillna(0)

    macro_cols = [c for c in macro_defaults if c in df_macro.columns]
    df = df.merge(df_macro[["Date"] + macro_cols], on="Date", how="left", suffixes=("", "_macro"))

    for col in macro_cols:
        merged_col = col + "_macro" if col + "_macro" in df.columns else col
        df[col] = df.get(merged_col, df.get(col, macro_defaults[col])).fillna(macro_defaults[col])
        if merged_col != col and merged_col in df.columns:
            df.drop(columns=[merged_col], inplace=True)

    return df


# ============================================================
# Alpha Targets (Multi-Horizon)
# ============================================================

def _add_alpha_targets(df: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward Alpha relative to TWII (benchmark) for 5d, 20d, 60d horizons.
    Alpha_Nd = stock cumulative return over N days - TWII cumulative return over N days

    Important: these are FUTURE returns, so they must only be used as training labels.
    They should never leak into the feature columns during inference.
    """
    g = df.groupby("stock_id", sort=False)

    def _fwd_cum_return(series: pd.Series, n: int) -> pd.Series:
        """Forward n-day cumulative return (look-ahead, safe for labels only)."""
        fwd = series.shift(-n) / series - 1
        return fwd

    df["Fwd_5d"]  = g["Close"].transform(lambda x: _fwd_cum_return(x, 5))
    df["Fwd_20d"] = g["Close"].transform(lambda x: _fwd_cum_return(x, 20))
    df["Fwd_60d"] = g["Close"].transform(lambda x: _fwd_cum_return(x, 60))

    # TWII benchmark returns
    if df_macro is not None and not df_macro.empty and "TWII" in df_macro.columns:
        df_macro = df_macro.copy()
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        df_macro["TWII_Fwd_5d"]  = _fwd_cum_return(df_macro["TWII"], 5)
        df_macro["TWII_Fwd_20d"] = _fwd_cum_return(df_macro["TWII"], 20)
        df_macro["TWII_Fwd_60d"] = _fwd_cum_return(df_macro["TWII"], 60)
        df = df.merge(
            df_macro[["Date", "TWII_Fwd_5d", "TWII_Fwd_20d", "TWII_Fwd_60d"]],
            on="Date", how="left",
        )
        df["Alpha_5d"]  = df["Fwd_5d"]  - df["TWII_Fwd_5d"].fillna(0)
        df["Alpha_20d"] = df["Fwd_20d"] - df["TWII_Fwd_20d"].fillna(0)
        df["Alpha_60d"] = df["Fwd_60d"] - df["TWII_Fwd_60d"].fillna(0)
        df.drop(columns=["TWII_Fwd_5d", "TWII_Fwd_20d", "TWII_Fwd_60d"], inplace=True)
    else:
        df["Alpha_5d"]  = df["Fwd_5d"]
        df["Alpha_20d"] = df["Fwd_20d"]
        df["Alpha_60d"] = df["Fwd_60d"]

    df.drop(columns=["Fwd_5d", "Fwd_20d", "Fwd_60d"], inplace=True)
    return df


# ============================================================
# Utility
# ============================================================

def _asof_lookup(dates: pd.Series, ref_dates: pd.Series, values: pd.Series) -> pd.Series:
    """
    For each date in `dates`, find the most recent value in `values`
    where ref_dates <= date (as-of join, no look-ahead bias).

    VECTORISED: uses numpy searchsorted for O(N log M) performance
    instead of a Python loop which would be O(N * M).
    """
    if ref_dates.empty or values.empty:
        return pd.Series(np.nan, index=dates.index)

    ref_arr = ref_dates.values   # sorted array of reference timestamps
    val_arr = values.values

    # For each date, find the rightmost ref_date <= date
    # searchsorted(..., 'right') gives the insertion point after all matching elements
    positions = np.searchsorted(ref_arr, dates.values, side="right") - 1

    # positions < 0 means no ref_date is available yet (before first publish date)
    result = np.where(positions >= 0, val_arr[positions], np.nan)
    return pd.Series(result.astype(np.float64), index=dates.index)


def clean_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning step:
      1. Drop rows with too many NaN features
      2. Winsorize at [1%, 99%] cross-sectionally (per Date)
      3. Z-score standardise cross-sectionally (per Date)

    This is intentionally cross-sectional, not time-series — we want relative
    ranks within a day, not absolute magnitudes.
    """
    # Drop rows where > 30% of feature columns are NaN
    threshold = int(0.7 * len(FEATURE_COLS))
    df = df.dropna(subset=FEATURE_COLS, thresh=threshold).copy()

    for col in FEATURE_COLS:
        # Cross-sectional winsorize
        df[col] = df.groupby("Date")[col].transform(
            lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
        )
        # Cross-sectional z-score
        df[col] = df.groupby("Date")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    # Fill any remaining NaNs with 0 (cross-sectional mean after z-score)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    return df
