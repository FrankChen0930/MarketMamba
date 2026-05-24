"""
MarketMamba V6 — Portfolio Manager
=====================================
Translates model output (df_kelly) into actionable position sizing.

Responsibilities:
  1. Hard filter: liquidity gate (min daily turnover)
  2. Soft filter: slippage-adjusted Net Alpha
  3. Position sizing: Kelly-inspired weighting clipped for safety
  4. Concentration limit: max 15% per single stock, min 5% to enter
  5. Sector concentration: max 40% in any single TWSE sector
  6. Output: updated robot_ledger.json with proposed trades
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from marketmamba.config import MIN_AVG_VOLUME_5D, RESULTS_DIR

logger = logging.getLogger(__name__)

LEDGER_PATH = RESULTS_DIR / "robot_ledger.json"

# Position sizing constraints
MAX_WEIGHT_PER_STOCK    = 0.15   # 單支最多 15%
MIN_WEIGHT_TO_ENTER     = 0.03   # 至少 3% 才值得買入（含交易成本後有意義）
MAX_SECTOR_WEIGHT       = 0.40   # 單一產業最多 40%
MAX_PORTFOLIO_STOCKS    = 20     # 最多持有 20 支


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Position:
    ticker:       str
    sector:       str
    weight:       float
    exp_alpha:    float
    sharpe_score: float
    confidence:   str
    entry_price:  float = 0.0
    entry_date:   str   = ""


@dataclass
class PortfolioLedger:
    date:          str
    positions:     list[Position] = field(default_factory=list)
    cash_reserve:  float = 0.05   # 5% 現金保留
    total_stocks:  int   = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ============================================================
# Filters
# ============================================================

def apply_liquidity_filter(
    df: pd.DataFrame,
    min_turnover: float = MIN_AVG_VOLUME_5D,
) -> pd.DataFrame:
    """
    Remove stocks below minimum daily turnover.
    Hard filter — these stocks simply cannot be traded at scale.
    """
    if "Turnover_5D" not in df.columns:
        logger.warning("Turnover_5D column missing — liquidity filter skipped")
        return df

    before = len(df)
    df = df[df["Turnover_5D"].fillna(0) >= min_turnover].copy()
    after = len(df)
    if before > after:
        logger.info(f"Liquidity filter: {before - after} stocks removed (turnover < {min_turnover:.0e})")
    return df


def apply_slippage_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduct estimated slippage from expected alpha.
    Slippage is inversely proportional to liquidity (market cap proxy).
    """
    if "Turnover_5D" not in df.columns:
        df["Net_Alpha_20d"] = df.get("Exp_Alpha_20d", 0)
        return df

    # Slippage model: 0.2% → 0.8% depending on liquidity tier
    high_liq  = df["Turnover_5D"] > 5e8   # > 5億: 0.2%
    mid_liq   = (df["Turnover_5D"] > 1e8) & ~high_liq  # 1-5億: 0.4%
    # low_liq: 0.8%

    df = df.copy()
    df["Slippage"] = 0.008   # default: small cap
    df.loc[mid_liq,  "Slippage"] = 0.004
    df.loc[high_liq, "Slippage"] = 0.002

    # Tax + commission: ~0.3% round trip (0.15% buy + 0.15% sell + 0.3% tax)
    df["Transaction_Cost"] = 0.003
    df["Net_Alpha_20d"] = (
        df.get("Exp_Alpha_20d", df.get("Net_Alpha_20d", 0))
        - df["Slippage"]
        - df["Transaction_Cost"]
    )
    return df


# ============================================================
# Position Sizing
# ============================================================

def compute_kelly_weights(
    df:              pd.DataFrame,
    sharpe_col:      str   = "Signal_Quality",
    max_weight:      float = MAX_WEIGHT_PER_STOCK,
    min_weight:      float = MIN_WEIGHT_TO_ENTER,
    max_stocks:      int   = MAX_PORTFOLIO_STOCKS,
    cash_reserve:    float = 0.05,
) -> pd.DataFrame:
    """
    Compute final portfolio weights using a simplified Kelly criterion.

    Method:
      1. Rank stocks by Sharpe score (net alpha / uncertainty)
      2. Assign raw weight proportional to Sharpe (positive only)
      3. Clip to [min_weight, max_weight]
      4. Apply sector concentration limit
      5. Renormalize to (1 - cash_reserve)

    Returns df with 'Final_Weight' column (0.0 for stocks not selected).
    """
    df = df.copy()
    df["Final_Weight"] = 0.0

    # Only consider stocks with positive net alpha
    eligible = df[df.get("Net_Alpha_20d", df.get("Exp_Alpha_20d", pd.Series())).fillna(-1) > 0].copy()

    if eligible.empty:
        logger.warning("No stocks with positive net alpha — portfolio is empty")
        return df

    # Rank by Sharpe score
    eligible = eligible.nlargest(max_stocks * 2, sharpe_col)

    # Raw weights proportional to Sharpe
    positive_sharpe = eligible[sharpe_col].clip(lower=0)
    total = positive_sharpe.sum()
    if total < 1e-10:
        return df
    eligible["Raw_Weight"] = positive_sharpe / total * (1 - cash_reserve)

    # Clip to per-stock limits
    eligible["Clipped_Weight"] = eligible["Raw_Weight"].clip(
        lower=min_weight, upper=max_weight
    )

    # Sector concentration limit
    if "Sector" in eligible.columns:
        eligible = _apply_sector_limit(eligible, MAX_SECTOR_WEIGHT, max_weight)

    # Final selection: top N by clipped weight, then renormalize
    selected = eligible.nlargest(max_stocks, "Clipped_Weight")
    total_selected = selected["Clipped_Weight"].sum()
    selected["Final_Weight"] = (
        selected["Clipped_Weight"] / total_selected * (1 - cash_reserve)
    ).round(4)

    # Only include stocks above minimum weight
    selected = selected[selected["Final_Weight"] >= min_weight]

    # Merge back into original df
    df.loc[selected.index, "Final_Weight"] = selected["Final_Weight"]
    return df


def _apply_sector_limit(
    df: pd.DataFrame,
    max_sector_weight: float,
    max_stock_weight:  float,
) -> pd.DataFrame:
    """
    Reduce weight of stocks if their sector's total exceeds max_sector_weight.
    """
    df = df.copy()
    sector_totals = df.groupby("Sector")["Clipped_Weight"].sum()

    for sector, total in sector_totals.items():
        if total > max_sector_weight:
            ratio = max_sector_weight / total
            mask = df["Sector"] == sector
            df.loc[mask, "Clipped_Weight"] *= ratio
            df["Clipped_Weight"] = df["Clipped_Weight"].clip(upper=max_stock_weight)

    return df


# ============================================================
# Main Rebalancing Function
# ============================================================

def rebalance(
    df_kelly:     pd.DataFrame,
    df_universe:  pd.DataFrame | None = None,   # optional: has 'industry_category'
    date_str:     str | None = None,
) -> PortfolioLedger:
    """
    Main portfolio rebalancing function.

    Args:
        df_kelly    : inference output from run_daily_inference
        df_universe : optional, for sector data
        date_str    : trading date string, defaults to today

    Returns:
        PortfolioLedger with final positions
    """
    date_str = date_str or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Portfolio rebalancing [{date_str}]")

    df = df_kelly.copy()

    # Attach sector info if available
    if df_universe is not None and "industry_category" in df_universe.columns:
        sector_map = df_universe.set_index("stock_id")["industry_category"].to_dict()
        df["Sector"] = df["Ticker"].map(sector_map).fillna("Unknown")

    # Step 1: Liquidity filter
    df = apply_liquidity_filter(df)
    if df.empty:
        logger.warning("All stocks filtered by liquidity — empty portfolio")
        return PortfolioLedger(date=date_str)

    # Step 2: Slippage penalty
    df = apply_slippage_penalty(df)

    # Step 3: Kelly weights
    sharpe_col = "Signal_Quality" if "Signal_Quality" in df.columns else df.columns[0]
    df = compute_kelly_weights(df, sharpe_col=sharpe_col)

    # Step 4: Extract positions
    selected = df[df["Final_Weight"] > 0].sort_values("Final_Weight", ascending=False)

    positions = []
    for _, row in selected.iterrows():
        positions.append(Position(
            ticker       = str(row.get("Ticker", "")),
            sector       = str(row.get("Sector", "Unknown")),
            weight       = float(row["Final_Weight"]),
            exp_alpha    = float(row.get("Exp_Alpha_20d", row.get("Net_Alpha_20d", 0))),
            sharpe_score = float(row.get("Signal_Quality", 0)),
            confidence   = str(row.get("Confidence", "中信心")),
            entry_price  = float(row.get("Close", 0)),
            entry_date   = date_str,
        ))

    ledger = PortfolioLedger(
        date=date_str,
        positions=positions,
        total_stocks=len(positions),
    )

    # Save ledger
    _save_ledger(ledger)
    _log_summary(ledger)
    return ledger


# ============================================================
# Utilities
# ============================================================

def _save_ledger(ledger: PortfolioLedger) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"Portfolio ledger saved → {LEDGER_PATH}")


def _log_summary(ledger: PortfolioLedger) -> None:
    logger.info(
        f"\n{'='*50}\n"
        f"  Portfolio [{ledger.date}]: {ledger.total_stocks} positions\n"
        f"{'='*50}"
    )
    for pos in ledger.positions:
        conf_icon = {"高信心": "🟢", "中信心": "🟡", "低信心": "🔴"}.get(pos.confidence, "⚪")
        logger.info(
            f"  {conf_icon} {pos.ticker:6s} | Weight={pos.weight:.1%} | "
            f"ExpAlpha={pos.exp_alpha:+.2%} | Sharpe={pos.sharpe_score:.2f}"
        )
    logger.info("=" * 50)


def load_ledger() -> dict:
    """Load the most recent portfolio ledger."""
    if not LEDGER_PATH.exists():
        return {}
    with open(LEDGER_PATH, encoding="utf-8") as f:
        return json.load(f)
