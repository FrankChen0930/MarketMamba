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
    df_price:         pd.DataFrame,
    df_inst:          pd.DataFrame | None = None,
    df_margin:        pd.DataFrame | None = None,
    df_per:           pd.DataFrame | None = None,
    df_securities:    pd.DataFrame | None = None,
    df_market_value:  pd.DataFrame | None = None,
    df_daytrade:      pd.DataFrame | None = None,
    df_holdings:      pd.DataFrame | None = None,
    df_rev:           pd.DataFrame | None = None,
    df_fin:           pd.DataFrame | None = None,
    df_balance_sheet: pd.DataFrame | None = None,
    df_cashflow:      pd.DataFrame | None = None,
    df_macro:         pd.DataFrame | None = None,
    # V6.1 new data sources
    df_futures_inst:  pd.DataFrame | None = None,
    df_options_inst:  pd.DataFrame | None = None,
    df_dividend:      pd.DataFrame | None = None,
    df_foreign_shareholding: pd.DataFrame | None = None,
    df_fear_greed:    pd.DataFrame | None = None,
    df_business_indicator: pd.DataFrame | None = None,
    df_fed_rate:      pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge all raw data sources and compute the 56-dim feature matrix (V6.1).

    Returns:
        df : MultiIndex [Date, stock_id] with all 56 feature columns + target columns
    """
    logger.info("Building V6.1 feature matrix (56D)...")

    df = df_price.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["stock_id", "Date"]).reset_index(drop=True)

    # -- Group A: Price / Momentum --
    df = _add_price_momentum_features(df)

    # -- Group B: Institutional / Margin / Technical --
    df = _merge_institutional(df, df_inst)
    df = _merge_margin(df, df_margin)
    df = _add_technical_b_features(df)
    df = _merge_daytrade(df, df_daytrade)
    df = _merge_holdings(df, df_holdings)                        # V6.1
    df = _merge_securities(df, df_securities)                    # V6.1
    df = _merge_foreign_shareholding(df, df_foreign_shareholding)  # V6.1

    # -- Group C: Fundamentals --
    df = _merge_per_pbr(df, df_per)
    df = _merge_market_value_feature(df, df_market_value)
    df = _merge_fundamentals(df, df_rev, df_fin, df_balance_sheet)
    df = _merge_dividend_feature(df, df_dividend)                # V6.1
    df = _add_free_cash_flow(df, df_cashflow)                    # V6.1

    # -- Group D: Macro --
    df = _merge_macro(df, df_macro, df_fear_greed, df_business_indicator,
                      df_fed_rate, df_futures_inst, df_options_inst)  # V6.1 expanded

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
    """Merge margin purchase / short sale data.
    Columns in margin_raw.parquet are already renamed by fetch_v6_data.py."""
    EXPECTED = ["Margin_Purchase", "Margin_Repay", "Short_Sale",
                "Short_Cover", "Margin_Balance", "Short_Balance"]
    if df_margin is None or df_margin.empty:
        for col in EXPECTED:
            df[col] = 0.0
        return df

    df_m = df_margin.copy()
    df_m["Date"] = pd.to_datetime(df_m["Date"])

    # Support BOTH already-renamed cols AND original FinMind names
    legacy_map = {
        "MarginPurchaseBuy":          "Margin_Purchase",
        "MarginPurchaseSell":         "Margin_Repay",
        "ShortSaleSell":              "Short_Sale",
        "ShortSaleBuy":               "Short_Cover",
        "MarginPurchaseTodayBalance": "Margin_Balance",
        "ShortSaleTodayBalance":      "Short_Balance",
    }
    df_m.rename(columns=legacy_map, inplace=True)
    valid = [c for c in EXPECTED if c in df_m.columns]

    df = df.merge(df_m[["Date", "stock_id"] + valid],
                  on=["Date", "stock_id"], how="left")
    for col in valid:
        df[col] = df.groupby("stock_id")[col].transform(
            lambda x: x.ffill().fillna(0.0))
    for col in EXPECTED:
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

    if "Day_Trade_Volume" not in df.columns:
        df["Day_Trade_Volume"] = 0.0
    return df


def _merge_daytrade(df: pd.DataFrame, df_daytrade: pd.DataFrame | None) -> pd.DataFrame:
    """Merge Day_Trade_Volume ratio from daytrade_raw.parquet."""
    if df_daytrade is None or df_daytrade.empty:
        return df
    dt = df_daytrade[["Date", "stock_id", "Day_Trade_Volume"]].copy()
    dt["Date"] = pd.to_datetime(dt["Date"])
    dt["Day_Trade_Volume"] = pd.to_numeric(dt["Day_Trade_Volume"], errors="coerce").clip(0, 1)
    df = df.merge(dt, on=["Date", "stock_id"], how="left", suffixes=("", "_dt"))
    if "Day_Trade_Volume_dt" in df.columns:
        df["Day_Trade_Volume"] = df["Day_Trade_Volume_dt"].fillna(df["Day_Trade_Volume"])
        df.drop(columns=["Day_Trade_Volume_dt"], inplace=True)
    else:
        df["Day_Trade_Volume"] = df["Day_Trade_Volume"].fillna(0.0)
    return df


def _merge_per_pbr(df: pd.DataFrame, df_per: pd.DataFrame | None) -> pd.DataFrame:
    """Merge PER/PBR/DY from per_raw.parquet (daily, direct join)."""
    if df_per is None or df_per.empty:
        return df
    p = df_per.copy()
    p["Date"] = pd.to_datetime(p["Date"])
    for c in ["PER", "PBR", "DY", "dividend_yield"]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    if "dividend_yield" in p.columns and "DY" not in p.columns:
        p = p.rename(columns={"dividend_yield": "DY"})
    keep = ["Date", "stock_id"] + [c for c in ["PER", "PBR", "DY"] if c in p.columns]
    df = df.merge(p[keep], on=["Date", "stock_id"], how="left", suffixes=("", "_per"))
    for c in ["PER", "PBR"]:
        dup = c + "_per"
        if dup in df.columns:
            df[c] = df[dup].combine_first(df.get(c))
            df.drop(columns=[dup], inplace=True)
    # Forward-fill PER/PBR within each stock (not updated every day)
    for c in ["PER", "PBR"]:
        if c in df.columns:
            df[c] = df.groupby("stock_id")[c].transform(lambda x: x.ffill())
    return df


def _merge_market_value_feature(df: pd.DataFrame, df_mv: pd.DataFrame | None) -> pd.DataFrame:
    """Compute Market_Cap_Log from market_value_raw.parquet."""
    if df_mv is None or df_mv.empty:
        return df
    mv = df_mv[["Date", "stock_id", "market_value"]].copy()
    mv["Date"] = pd.to_datetime(mv["Date"])
    mv["market_value"] = pd.to_numeric(mv["market_value"], errors="coerce").clip(lower=0)
    mv["Market_Cap_Log"] = np.log1p(mv["market_value"])
    df = df.merge(mv[["Date", "stock_id", "Market_Cap_Log"]],
                  on=["Date", "stock_id"], how="left", suffixes=("", "_mv"))
    if "Market_Cap_Log_mv" in df.columns:
        df["Market_Cap_Log"] = df["Market_Cap_Log_mv"].combine_first(df.get("Market_Cap_Log"))
        df.drop(columns=["Market_Cap_Log_mv"], inplace=True)
    df["Market_Cap_Log"] = df.groupby("stock_id")["Market_Cap_Log"].transform(
        lambda x: x.ffill().fillna(0.0))
    return df


# ============================================================
# Group C — Fundamentals (10 dims)
# ============================================================

def _merge_fundamentals(
    df: pd.DataFrame,
    df_rev:          pd.DataFrame | None = None,
    df_fin:          pd.DataFrame | None = None,
    df_balance_sheet: pd.DataFrame | None = None,
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
        df = _merge_financial_statements(df, df_fin, df_balance_sheet)

    return df


def _merge_revenue(df: pd.DataFrame, df_rev: pd.DataFrame) -> pd.DataFrame:
    df_rev = df_rev.copy()
    # Support both 'date' (raw FinMind) and 'Date' (already renamed by merger)
    date_col = "Date" if "Date" in df_rev.columns else "date"
    df_rev["date"] = pd.to_datetime(df_rev[date_col])
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


def _merge_financial_statements(
    df: pd.DataFrame,
    df_fin: pd.DataFrame,
    df_balance_sheet: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge quarterly EPS, Gross_Margin, ROE with look-ahead protection.
    Handles FinMind long format: columns = [Date, stock_id, type, value, origin_name]"""
    df_fin = df_fin.copy()
    date_col = "Date" if "Date" in df_fin.columns else "date"
    df_fin["Date"] = pd.to_datetime(df_fin[date_col])
    df_fin["available_from"] = df_fin["Date"] + pd.Timedelta(days=45)

    # -- Detect wide vs long format --
    is_long = "type" in df_fin.columns and "value" in df_fin.columns

    if is_long:
        # Pivot long → wide, extracting key financial items
        TYPE_MAP = {
            # EPS variants
            "EPS": "EPS", "AfterTax_EPS": "EPS", "BasicEPS": "EPS",
            # Revenue / Gross
            "Operating_Revenue": "Revenue", "OperatingRevenue": "Revenue",
            "Gross_Profit": "GrossProfit", "GrossProfit": "GrossProfit",
            # ROE
            "ROE": "ROE",
            # Book value
            "Total_Equity": "Book_Value", "TotalEquity": "Book_Value",
            "StockholdersEquity": "Book_Value",
        }
        df_fin["mapped"] = df_fin["type"].map(TYPE_MAP)
        df_fin = df_fin[df_fin["mapped"].notna()].copy()
        if df_fin.empty:
            logger.warning("financial_statements: no recognisable type values found")
            return df
        df_fin["value"] = pd.to_numeric(df_fin["value"], errors="coerce")
        # Take last value per (stock_id, Date, mapped)
        df_wide = (df_fin.groupby(["stock_id", "Date", "available_from", "mapped"])["value"]
                   .last().unstack("mapped").reset_index())
        # Compute derived columns
        if "GrossProfit" in df_wide.columns and "Revenue" in df_wide.columns:
            df_wide["Gross_Margin"] = df_wide["GrossProfit"] / df_wide["Revenue"].replace(0, np.nan)
        df_fin = df_wide
        fin_cols = [c for c in ["EPS", "Gross_Margin", "ROE", "Book_Value"] if c in df_fin.columns]
    else:
        fin_cols = [c for c in ["EPS", "Gross_Margin", "ROE", "Book_Value"] if c in df_fin.columns]

    if not fin_cols:
        return df

    df_fin = df_fin.sort_values(["stock_id", "available_from"])
    for col in fin_cols:
        vals_by_stock = df_fin.groupby("stock_id").apply(
            lambda g: _asof_lookup(df.loc[df["stock_id"] == g.name, "Date"],
                                   g["available_from"].reset_index(drop=True),
                                   g[col].reset_index(drop=True))
        )
    # Vectorised as-of per stock
    result_rows = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_fin = df_fin[df_fin["stock_id"] == sid].sort_values("available_from")
        sub_df = sub_df.copy()
        for col in fin_cols:
            sub_df[col] = _asof_lookup(sub_df["Date"],
                                        sub_fin["available_from"].reset_index(drop=True),
                                        sub_fin[col].reset_index(drop=True))
        result_rows.append(sub_df)
    if result_rows:
        df = pd.concat(result_rows, ignore_index=True)

    if "EPS" in df.columns:
        df["EPS_Surprise"] = df.groupby("stock_id")["EPS"].pct_change(4).fillna(0)

    return df


# ============================================================
# V6.1 — Group B Additions (Holdings, Securities, Foreign Holding)
# ============================================================

def _merge_holdings(df: pd.DataFrame, df_holdings: pd.DataFrame | None) -> pd.DataFrame:
    """Merge 大戶持股分級 data → Holdings_Large_Pct + Holdings_Large_Change.

    holdings_raw.parquet contains weekly data (集保戶股權分散表).
    We compute the percentage of shares held by large holders (千張以上).
    """
    if df_holdings is None or df_holdings.empty:
        df["Holdings_Large_Pct"] = 0.0
        df["Holdings_Large_Change"] = 0.0
        return df

    h = df_holdings.copy()
    # holdings_raw uses 'Week' column instead of 'Date'
    if "Week" in h.columns and "Date" not in h.columns:
        h = h.rename(columns={"Week": "Date"})
    h["Date"] = pd.to_datetime(h["Date"])

    # holdings_raw has columns: [Week/Date, stock_id, Whale_Hold_Ratio, Retail_Hold_Ratio]
    # We need the row corresponding to the largest holding level (千張以上 / 1000+)
    pct_col = [c for c in h.columns if "percent" in c.lower() or "pct" in c.lower() or "ratio" in c.lower()]
    level_col = [c for c in h.columns if "level" in c.lower() or "holding" in c.lower()]

    if pct_col and level_col:
        # Filter for largest holder level (typically the last level or '1000+')
        h[pct_col[0]] = pd.to_numeric(h[pct_col[0]], errors="coerce")
        # Take the maximum level per (stock_id, Date) as the large holder percentage
        h_large = h.groupby(["Date", "stock_id"])[pct_col[0]].max().rename("Holdings_Large_Pct").reset_index()
    elif len(h.columns) >= 4:
        # Fallback: use the last numeric column as percentage
        num_cols = h.select_dtypes(include="number").columns.tolist()
        if num_cols:
            h_large = h.groupby(["Date", "stock_id"])[num_cols[-1]].max().rename("Holdings_Large_Pct").reset_index()
        else:
            df["Holdings_Large_Pct"] = 0.0
            df["Holdings_Large_Change"] = 0.0
            return df
    else:
        df["Holdings_Large_Pct"] = 0.0
        df["Holdings_Large_Change"] = 0.0
        return df

    # Left merge (weekly → daily), then forward fill per stock
    h_large = h_large.sort_values(["stock_id", "Date"])
    df = df.merge(h_large, on=["Date", "stock_id"], how="left", suffixes=("", "_hld"))
    if "Holdings_Large_Pct_hld" in df.columns:
        df["Holdings_Large_Pct"] = df["Holdings_Large_Pct_hld"].combine_first(df.get("Holdings_Large_Pct"))
        df.drop(columns=["Holdings_Large_Pct_hld"], inplace=True)
    df = df.sort_values(["stock_id", "Date"])
    df["Holdings_Large_Pct"] = df.groupby("stock_id")["Holdings_Large_Pct"].transform(
        lambda x: x.ffill().fillna(0.0))

    # Week-over-week change
    df["Holdings_Large_Change"] = df.groupby("stock_id")["Holdings_Large_Pct"].diff(5).fillna(0.0)

    return df


def _merge_securities(df: pd.DataFrame, df_sec: pd.DataFrame | None) -> pd.DataFrame:
    """Merge 借券餘額 → Securities_Balance (daily short lending balance)."""
    if df_sec is None or df_sec.empty:
        df["Securities_Balance"] = 0.0
        return df

    s = df_sec.copy()
    s["Date"] = pd.to_datetime(s["Date"])

    # Find the balance column
    bal_col = [c for c in s.columns if "balance" in c.lower() or "volume" in c.lower()]
    if not bal_col:
        num_cols = s.select_dtypes(include="number").columns.tolist()
        bal_col = num_cols[:1] if num_cols else []

    if not bal_col:
        df["Securities_Balance"] = 0.0
        return df

    s["Securities_Balance"] = pd.to_numeric(s[bal_col[0]], errors="coerce").fillna(0)
    df = df.merge(s[["Date", "stock_id", "Securities_Balance"]],
                  on=["Date", "stock_id"], how="left", suffixes=("", "_sec"))
    if "Securities_Balance_sec" in df.columns:
        df["Securities_Balance"] = df["Securities_Balance_sec"].combine_first(df.get("Securities_Balance"))
        df.drop(columns=["Securities_Balance_sec"], inplace=True)
    df["Securities_Balance"] = df.groupby("stock_id")["Securities_Balance"].transform(
        lambda x: x.ffill().fillna(0.0))
    return df


def _merge_foreign_shareholding(df: pd.DataFrame, df_fs: pd.DataFrame | None) -> pd.DataFrame:
    """Merge 外資持股比例 → Foreign_Holding_Pct (cumulative %)."""
    if df_fs is None or df_fs.empty:
        df["Foreign_Holding_Pct"] = 0.0
        return df

    fs = df_fs.copy()
    fs["Date"] = pd.to_datetime(fs["Date"])

    # Find percentage column
    pct_col = [c for c in fs.columns if "percent" in c.lower() or "pct" in c.lower() or "ratio" in c.lower() or "shareholding" in c.lower()]
    if not pct_col:
        num_cols = fs.select_dtypes(include="number").columns.tolist()
        pct_col = num_cols[:1] if num_cols else []

    if not pct_col:
        df["Foreign_Holding_Pct"] = 0.0
        return df

    fs["Foreign_Holding_Pct"] = pd.to_numeric(fs[pct_col[0]], errors="coerce")
    df = df.merge(fs[["Date", "stock_id", "Foreign_Holding_Pct"]],
                  on=["Date", "stock_id"], how="left", suffixes=("", "_fsp"))
    if "Foreign_Holding_Pct_fsp" in df.columns:
        df["Foreign_Holding_Pct"] = df["Foreign_Holding_Pct_fsp"].combine_first(df.get("Foreign_Holding_Pct"))
        df.drop(columns=["Foreign_Holding_Pct_fsp"], inplace=True)
    df["Foreign_Holding_Pct"] = df.groupby("stock_id")["Foreign_Holding_Pct"].transform(
        lambda x: x.ffill().fillna(0.0))
    return df


# ============================================================
# V6.1 — Group C Additions (Dividend, Free Cash Flow)
# ============================================================

def _merge_dividend_feature(df: pd.DataFrame, df_div: pd.DataFrame | None) -> pd.DataFrame:
    """Compute Dividend_Yield_Fwd from dividend announcements.

    Uses as-of join: dividends are available once announced (before ex-date).
    Dividend_Yield_Fwd = latest announced cash dividend / current Close price.
    """
    if df_div is None or df_div.empty:
        if "Dividend_Yield_Fwd" not in df.columns:
            df["Dividend_Yield_Fwd"] = 0.0
        return df

    div = df_div.copy()
    div["Date"] = pd.to_datetime(div["Date"])

    # Find cash dividend column
    cash_col = [c for c in div.columns if "cash" in c.lower() and ("earning" in c.lower() or "dividend" in c.lower())]
    if not cash_col:
        cash_col = [c for c in div.columns if "dividend" in c.lower()][:1]

    if not cash_col:
        df["Dividend_Yield_Fwd"] = 0.0
        return df

    div["_cash_div"] = pd.to_numeric(div[cash_col[0]], errors="coerce").fillna(0)
    # Keep latest dividend per stock per date
    div = div.sort_values(["stock_id", "Date"])
    div_latest = div.groupby(["stock_id", "Date"])["_cash_div"].sum().reset_index()

    # As-of merge: for each trading day, use the latest announced dividend
    merged = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_div = div_latest[div_latest["stock_id"] == sid].sort_values("Date")
        sub_df = sub_df.copy()
        if not sub_div.empty:
            sub_df["_latest_div"] = _asof_lookup(
                sub_df["Date"],
                sub_div["Date"].reset_index(drop=True),
                sub_div["_cash_div"].reset_index(drop=True),
            )
        else:
            sub_df["_latest_div"] = 0.0
        merged.append(sub_df)

    if merged:
        df = pd.concat(merged, ignore_index=True)

    # Compute forward yield = latest cash dividend / current price
    df["_latest_div"] = df.get("_latest_div", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["Dividend_Yield_Fwd"] = df["_latest_div"] / df["Close"].replace(0, np.nan)
    df["Dividend_Yield_Fwd"] = df["Dividend_Yield_Fwd"].clip(0, 0.2).fillna(0.0)  # cap at 20%
    df.drop(columns=["_latest_div"], inplace=True, errors="ignore")
    return df


def _add_free_cash_flow(df: pd.DataFrame, df_cashflow: pd.DataFrame | None) -> pd.DataFrame:
    """Compute Free_Cash_Flow from cashflow_raw.parquet.

    FCF = Operating Cash Flow - Capital Expenditure
    Uses as-of join (45-day lag) to prevent look-ahead bias.
    """
    if df_cashflow is None or df_cashflow.empty:
        if "Free_Cash_Flow" not in df.columns:
            df["Free_Cash_Flow"] = 0.0
        return df

    cf = df_cashflow.copy()
    date_col = "Date" if "Date" in cf.columns else "date"
    cf["Date"] = pd.to_datetime(cf[date_col])
    cf["available_from"] = cf["Date"] + pd.Timedelta(days=45)

    # FinMind long format: [Date, stock_id, type, value]
    if "type" in cf.columns and "value" in cf.columns:
        cf["value"] = pd.to_numeric(cf["value"], errors="coerce")
        # Look for operating CF and capex
        TYPE_MAP = {
            "CashFlowsFromOperatingActivities": "OperatingCF",
            "OperatingActivities": "OperatingCF",
            "CashFlowsFromInvestingActivities": "InvestingCF",
            "InvestingActivities": "InvestingCF",
        }
        cf["mapped"] = cf["type"].map(TYPE_MAP)
        cf = cf[cf["mapped"].notna()].copy()

        if cf.empty:
            df["Free_Cash_Flow"] = 0.0
            return df

        cf_wide = cf.pivot_table(
            index=["stock_id", "Date", "available_from"],
            columns="mapped", values="value", aggfunc="last"
        ).reset_index()

        if "OperatingCF" in cf_wide.columns:
            investing = cf_wide.get("InvestingCF", pd.Series(0.0, index=cf_wide.index))
            # FCF ≈ Operating CF + Investing CF (investing is negative for capex)
            cf_wide["Free_Cash_Flow"] = cf_wide["OperatingCF"].fillna(0) + investing.fillna(0)
        else:
            df["Free_Cash_Flow"] = 0.0
            return df
    else:
        df["Free_Cash_Flow"] = 0.0
        return df

    # As-of merge per stock
    cf_wide = cf_wide.sort_values(["stock_id", "available_from"])
    result_rows = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_cf = cf_wide[cf_wide["stock_id"] == sid].sort_values("available_from")
        sub_df = sub_df.copy()
        if not sub_cf.empty:
            sub_df["Free_Cash_Flow"] = _asof_lookup(
                sub_df["Date"],
                sub_cf["available_from"].reset_index(drop=True),
                sub_cf["Free_Cash_Flow"].reset_index(drop=True),
            )
        else:
            sub_df["Free_Cash_Flow"] = 0.0
        result_rows.append(sub_df)

    if result_rows:
        df = pd.concat(result_rows, ignore_index=True)
    df["Free_Cash_Flow"] = df["Free_Cash_Flow"].fillna(0.0)
    return df



def _merge_macro(
    df: pd.DataFrame,
    df_macro: pd.DataFrame | None,
    df_fear_greed: pd.DataFrame | None = None,
    df_business_indicator: pd.DataFrame | None = None,
    df_fed_rate: pd.DataFrame | None = None,
    df_futures_inst: pd.DataFrame | None = None,
    df_options_inst: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge macro data including V6.1 additions.
    Handles our macro_raw.parquet column names:
    TWII_Close, US_SOX, US_QQQ, US_VIX, US_TNX, Gold, Oil, USD_TWD, FED_Rate, ..."""
    DEFAULTS = {
        "TWII_Return": 0.0, "SPX_Return": 0.0,
        "VIX": 20.0, "TNX": 4.0,
        "Gold_Return": 0.0, "Oil_Return": 0.0,
        "USD_TWD": 30.0,
        # V6.1 macro features
        "Futures_OI_Foreign": 0.0,
        "Options_PC_Ratio": 1.0,
        "Fear_Greed": 50.0,
        "Business_Signal": 23.0,   # green light default
        "FED_Rate": 4.0,
    }
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    if df_macro is not None and not df_macro.empty:
        m = df_macro.copy()
        m["Date"] = pd.to_datetime(m["Date"])

        # Rename to canonical names used in features
        rename_map = {
            "TWII_Close": "TWII",
            "US_QQQ":     "SPX",    # QQQ as SPX proxy
            "US_VIX":     "VIX",
            "US_TNX":     "TNX",
        }
        m.rename(columns=rename_map, inplace=True)

        # Compute pct-change returns
        for raw, ret in [("TWII", "TWII_Return"), ("SPX", "SPX_Return"),
                         ("Gold", "Gold_Return"), ("Oil", "Oil_Return")]:
            if raw in m.columns:
                m[ret] = m[raw].pct_change(1).fillna(0)

        want = [c for c in ["TWII_Return", "SPX_Return", "VIX", "TNX",
                            "Gold_Return", "Oil_Return", "USD_TWD"] if c in m.columns]
        df = df.merge(m[["Date"] + want], on="Date", how="left", suffixes=("", "_m"))
        for col in want:
            dup = col + "_m"
            if dup in df.columns:
                df[col] = df[dup].combine_first(df[col])
                df.drop(columns=[dup], inplace=True)
            df[col] = df[col].fillna(DEFAULTS.get(col, 0.0))

    # -- V6.1: Futures OI Foreign --
    if df_futures_inst is not None and not df_futures_inst.empty:
        fi = df_futures_inst.copy()
        fi["Date"] = pd.to_datetime(fi["Date"])
        # Compute net OI for foreign investors
        fi_cols = fi.columns.tolist()
        if "institutional_investors" in fi_cols:
            fi_foreign = fi[fi["institutional_investors"].str.contains("外資|Foreign", case=False, na=False)].copy()
            if not fi_foreign.empty:
                oi_long_col = [c for c in fi_cols if "long_open_interest" in c.lower() or "long_oi" in c.lower()]
                oi_short_col = [c for c in fi_cols if "short_open_interest" in c.lower() or "short_oi" in c.lower()]
                if oi_long_col and oi_short_col:
                    fi_foreign["Futures_OI_Foreign"] = (
                        pd.to_numeric(fi_foreign[oi_long_col[0]], errors="coerce").fillna(0) -
                        pd.to_numeric(fi_foreign[oi_short_col[0]], errors="coerce").fillna(0)
                    )
                    fi_agg = fi_foreign.groupby("Date")["Futures_OI_Foreign"].sum().reset_index()
                    df = df.merge(fi_agg, on="Date", how="left", suffixes=("", "_fut"))
                    if "Futures_OI_Foreign_fut" in df.columns:
                        df["Futures_OI_Foreign"] = df["Futures_OI_Foreign_fut"].combine_first(df["Futures_OI_Foreign"])
                        df.drop(columns=["Futures_OI_Foreign_fut"], inplace=True)
                    df["Futures_OI_Foreign"] = df["Futures_OI_Foreign"].fillna(0.0)

    # -- V6.1: Options Put/Call Ratio --
    if df_options_inst is not None and not df_options_inst.empty:
        oi = df_options_inst.copy()
        oi["Date"] = pd.to_datetime(oi["Date"])
        oi_cols = oi.columns.tolist()
        # Try to compute Put/Call ratio from volume columns
        if "option_type" in oi_cols or "call_put" in oi_cols:
            type_col = "option_type" if "option_type" in oi_cols else "call_put"
            vol_col = [c for c in oi_cols if "volume" in c.lower() or "deal_volume" in c.lower()]
            if vol_col:
                oi["_vol"] = pd.to_numeric(oi[vol_col[0]], errors="coerce").fillna(0)
                put_vol = oi[oi[type_col].str.contains("put|P|賣", case=False, na=False)].groupby("Date")["_vol"].sum()
                call_vol = oi[oi[type_col].str.contains("call|C|買", case=False, na=False)].groupby("Date")["_vol"].sum()
                pc_ratio = (put_vol / call_vol.replace(0, np.nan)).rename("Options_PC_Ratio").reset_index()
                df = df.merge(pc_ratio, on="Date", how="left", suffixes=("", "_opt"))
                if "Options_PC_Ratio_opt" in df.columns:
                    df["Options_PC_Ratio"] = df["Options_PC_Ratio_opt"].combine_first(df["Options_PC_Ratio"])
                    df.drop(columns=["Options_PC_Ratio_opt"], inplace=True)
                df["Options_PC_Ratio"] = df["Options_PC_Ratio"].fillna(1.0)

    # -- V6.1: Fear & Greed Index --
    if df_fear_greed is not None and not df_fear_greed.empty:
        fg = df_fear_greed.copy()
        fg["Date"] = pd.to_datetime(fg["Date"])
        # Find the value column (varies by FinMind format)
        val_col = [c for c in fg.columns if c not in ["Date", "date"] and "value" in c.lower() or "index" in c.lower()]
        if not val_col:
            val_col = [c for c in fg.columns if c not in ["Date", "date"]][:1]
        if val_col:
            fg["Fear_Greed"] = pd.to_numeric(fg[val_col[0]], errors="coerce")
            df = df.merge(fg[["Date", "Fear_Greed"]], on="Date", how="left", suffixes=("", "_fg"))
            if "Fear_Greed_fg" in df.columns:
                df["Fear_Greed"] = df["Fear_Greed_fg"].combine_first(df["Fear_Greed"])
                df.drop(columns=["Fear_Greed_fg"], inplace=True)
            df["Fear_Greed"] = df["Fear_Greed"].ffill().fillna(50.0)

    # -- V6.1: Business Indicator (景氣燈號) --
    if df_business_indicator is not None and not df_business_indicator.empty:
        bi = df_business_indicator.copy()
        bi["Date"] = pd.to_datetime(bi["Date"])
        # The score column is typically 'score' or 'composite_index'
        score_col = [c for c in bi.columns if "score" in c.lower() or "composite" in c.lower() or "signal" in c.lower()]
        if not score_col:
            score_col = [c for c in bi.columns if c not in ["Date", "date"]][:1]
        if score_col:
            bi["Business_Signal"] = pd.to_numeric(bi[score_col[0]], errors="coerce")
            # Monthly data — as-of merge
            bi = bi.sort_values("Date")
            bi_dedup = bi.drop_duplicates(subset=["Date"], keep="last")
            df = pd.merge_asof(df.sort_values("Date"), bi_dedup[["Date", "Business_Signal"]],
                               on="Date", direction="backward", suffixes=("", "_bi"))
            if "Business_Signal_bi" in df.columns:
                df["Business_Signal"] = df["Business_Signal_bi"].combine_first(df["Business_Signal"])
                df.drop(columns=["Business_Signal_bi"], inplace=True)
            df["Business_Signal"] = df["Business_Signal"].ffill().fillna(23.0)

    # -- V6.1: FED Rate --
    if df_fed_rate is not None and not df_fed_rate.empty:
        fr = df_fed_rate.copy()
        fr["Date"] = pd.to_datetime(fr["Date"])
        rate_col = [c for c in fr.columns if "rate" in c.lower() or "value" in c.lower()]
        if not rate_col:
            rate_col = [c for c in fr.columns if c not in ["Date", "date"]][:1]
        if rate_col:
            fr["FED_Rate"] = pd.to_numeric(fr[rate_col[0]], errors="coerce")
            fr = fr.sort_values("Date")
            fr_dedup = fr.drop_duplicates(subset=["Date"], keep="last")
            df = pd.merge_asof(df.sort_values("Date"), fr_dedup[["Date", "FED_Rate"]],
                               on="Date", direction="backward", suffixes=("", "_fr"))
            if "FED_Rate_fr" in df.columns:
                df["FED_Rate"] = df["FED_Rate_fr"].combine_first(df["FED_Rate"])
                df.drop(columns=["FED_Rate_fr"], inplace=True)
            df["FED_Rate"] = df["FED_Rate"].ffill().fillna(4.0)

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
