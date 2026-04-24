"""
MarketMamba V6 — Backtesting Engine
======================================
Simulates historical portfolio performance using model predictions.
Uses the stored daily df_kelly.csv files from the rolling 90-day archive.

Metrics produced:
  - Cumulative return vs TWII benchmark
  - Annualised Alpha / Sharpe
  - Maximum Drawdown
  - Win rate (days outperforming benchmark)
  - Turnover analysis (average monthly turnover %)

Comparison modes:
  A: Top-10 equal-weight
  B: Kelly-weighted (uses Suggested_Weight / Final_Weight)
  C: TWII index (benchmark)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from marketmamba.config import RESULTS_DIR

logger = logging.getLogger(__name__)

ARCHIVE_DIR = RESULTS_DIR  # daily results stored as RESULTS_DIR/YYYY-MM-DD/df_kelly.csv


# ============================================================
# Data Classes
# ============================================================

@dataclass
class BacktestResult:
    strategy:          str
    start_date:        str
    end_date:          str
    total_return:      float = 0.0
    annualized_return: float = 0.0
    annualized_sharpe: float = 0.0
    max_drawdown:      float = 0.0
    win_rate:          float = 0.0   # days beating benchmark
    avg_turnover:      float = 0.0
    n_trading_days:    int   = 0
    equity_curve:      list[float] = field(default_factory=list)
    dates:             list[str]   = field(default_factory=list)

    def print_report(self) -> None:
        logger.info(
            f"\n{'─'*55}\n"
            f"  Strategy: {self.strategy}\n"
            f"  Period  : {self.start_date} → {self.end_date} ({self.n_trading_days} days)\n"
            f"  Return  : {self.total_return:+.2%}  Annualized: {self.annualized_return:+.2%}\n"
            f"  Sharpe  : {self.annualized_sharpe:.3f}\n"
            f"  MaxDD   : {self.max_drawdown:.2%}\n"
            f"  Win Rate: {self.win_rate:.1%}\n"
            f"  Turnover: {self.avg_turnover:.1%}/month\n"
            f"{'─'*55}"
        )


# ============================================================
# Data Loader
# ============================================================

def load_historical_predictions(
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame:
    """
    Load all archived df_kelly.csv files and concatenate into one DataFrame.
    Archives are stored as RESULTS_DIR / YYYY-MM-DD / df_kelly.csv

    Returns:
        DataFrame with [Date, Ticker, Exp_Alpha_20d, Sharpe_Score, Final_Weight, ...]
    """
    frames = []
    for dated_dir in sorted(ARCHIVE_DIR.iterdir()):
        if not dated_dir.is_dir():
            continue
        date_str = dated_dir.name
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        kelly_path = dated_dir / "df_kelly.csv"
        if not kelly_path.exists():
            continue
        df = pd.read_csv(kelly_path)
        df["Date"] = date_str
        frames.append(df)

    if not frames:
        logger.warning(f"No archived predictions found in {ARCHIVE_DIR}")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(frames)} prediction files: {df_all.shape}")
    return df_all


def load_realised_returns(
    df_predictions: pd.DataFrame,
    prices_path:    Path | None = None,
    horizon_days:   int = 20,
) -> pd.DataFrame:
    """
    Load actual future returns for backtesting.
    For each (Date, Ticker) in df_predictions, find the close price
    `horizon_days` trading days later and compute actual return.

    Args:
        df_predictions : from load_historical_predictions()
        prices_path    : path to price data parquet
        horizon_days   : evaluation horizon (must match model's primary horizon)
    """
    from marketmamba.config import PROCESSED_DIR
    if prices_path is None:
        prices_path = PROCESSED_DIR / "prices_raw.parquet"

    if not prices_path.exists():
        logger.warning("Price data not found for backtesting — returning empty")
        return df_predictions

    df_prices = pd.read_parquet(prices_path)
    df_prices["Date"] = pd.to_datetime(df_prices["Date"])
    df_prices = df_prices.sort_values(["stock_id", "Date"])

    # Build forward-return lookup
    df_prices["Fwd_Return"] = df_prices.groupby("stock_id")["Close"].transform(
        lambda x: x.shift(-horizon_days) / x - 1
    )
    df_prices["Date_str"] = df_prices["Date"].dt.strftime("%Y-%m-%d")

    fwd_map = df_prices.set_index(["Date_str", "stock_id"])["Fwd_Return"].to_dict()

    df_predictions = df_predictions.copy()
    df_predictions["Realised_Return"] = df_predictions.apply(
        lambda row: fwd_map.get((row.get("Date", ""), row.get("Ticker", "")), np.nan),
        axis=1,
    )
    return df_predictions


# ============================================================
# Strategy Simulators
# ============================================================

def simulate_top_n_equal_weight(
    df:          pd.DataFrame,
    n:           int   = 10,
    date_col:    str   = "Date",
    rank_col:    str   = "Sharpe_Score",
    return_col:  str   = "Realised_Return",
    bench_col:   str | None = "TWII_Return",
) -> BacktestResult:
    """
    Simulate a top-N equal-weight strategy.
    Each day: select top-N by rank_col, equal-weight their actual returns.
    """
    daily_returns = []
    bench_returns = []
    dates_used    = []
    turnovers     = []
    prev_tickers: set[str] = set()

    for date, group in df.groupby(date_col):
        group = group.dropna(subset=[return_col])
        if len(group) < n:
            continue
        top_n = group.nlargest(n, rank_col)
        port_ret = float(top_n[return_col].mean())
        daily_returns.append(port_ret)

        curr_tickers = set(top_n["Ticker"].values)
        if prev_tickers:
            turnover = len(curr_tickers - prev_tickers) / max(len(curr_tickers), 1)
            turnovers.append(turnover)
        prev_tickers = curr_tickers

        if bench_col and bench_col in group.columns:
            bench_returns.append(float(group[bench_col].iloc[0]))
        else:
            bench_returns.append(0.0)

        dates_used.append(str(date))

    if not daily_returns:
        return BacktestResult(strategy=f"Top{n}_EW", start_date="", end_date="")

    return _compute_backtest_metrics(
        strategy=f"Top{n}_EqualWeight",
        daily_returns=np.array(daily_returns),
        bench_returns=np.array(bench_returns),
        dates=dates_used,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
    )


def simulate_kelly_weighted(
    df:          pd.DataFrame,
    date_col:    str  = "Date",
    weight_col:  str  = "Final_Weight",
    return_col:  str  = "Realised_Return",
    bench_col:   str | None = "TWII_Return",
) -> BacktestResult:
    """
    Simulate strategy using the model's Final_Weight (Kelly-adjusted).
    """
    daily_returns = []
    bench_returns = []
    dates_used    = []
    turnovers     = []
    prev_tickers: set[str] = set()

    for date, group in df.groupby(date_col):
        group = group[group[weight_col].fillna(0) > 0].dropna(subset=[return_col])
        if group.empty:
            continue
        # Weighted return
        weights  = group[weight_col].values
        returns  = group[return_col].values
        port_ret = float(np.dot(weights / weights.sum(), returns))
        daily_returns.append(port_ret)

        curr_tickers = set(group["Ticker"].values)
        if prev_tickers:
            turnovers.append(len(curr_tickers - prev_tickers) / max(len(curr_tickers), 1))
        prev_tickers = curr_tickers

        if bench_col and bench_col in group.columns:
            bench_returns.append(float(group[bench_col].iloc[0]))
        else:
            bench_returns.append(0.0)

        dates_used.append(str(date))

    if not daily_returns:
        return BacktestResult(strategy="Kelly_Weighted", start_date="", end_date="")

    return _compute_backtest_metrics(
        strategy="Kelly_Weighted",
        daily_returns=np.array(daily_returns),
        bench_returns=np.array(bench_returns),
        dates=dates_used,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
    )


# ============================================================
# Metric Computation
# ============================================================

def _compute_backtest_metrics(
    strategy:      str,
    daily_returns: np.ndarray,
    bench_returns: np.ndarray,
    dates:         list[str],
    avg_turnover:  float = 0.0,
    trading_days_per_year: int = 252,
) -> BacktestResult:
    """Compute all summary metrics from daily return series."""
    equity = np.cumprod(1 + daily_returns)

    total_return = float(equity[-1] - 1)
    n_days       = len(daily_returns)
    years        = n_days / trading_days_per_year
    ann_return   = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    # Sharpe (annualized, assuming 0% risk-free)
    daily_std = float(np.std(daily_returns))
    ann_sharpe = float(np.mean(daily_returns) / (daily_std + 1e-10) * np.sqrt(trading_days_per_year))

    # Max drawdown
    roll_max = np.maximum.accumulate(equity)
    drawdowns = (equity - roll_max) / (roll_max + 1e-10)
    max_dd = float(drawdowns.min())

    # Win rate vs benchmark
    excess = daily_returns - bench_returns
    win_rate = float((excess > 0).mean())

    return BacktestResult(
        strategy=strategy,
        start_date=dates[0],
        end_date=dates[-1],
        total_return=total_return,
        annualized_return=ann_return,
        annualized_sharpe=ann_sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        avg_turnover=avg_turnover,
        n_trading_days=n_days,
        equity_curve=equity.tolist(),
        dates=dates,
    )


# ============================================================
# Main Entry Point
# ============================================================

def run_backtest(
    start_date:   str | None = None,
    end_date:     str | None = None,
    horizon_days: int = 20,
    top_n:        int = 10,
) -> dict[str, BacktestResult]:
    """
    Load archived predictions, attach realised returns, and run all strategy simulations.

    Returns:
        {
          'top10_ew'     : BacktestResult,
          'kelly'        : BacktestResult,
        }
    """
    logger.info(f"=== V6 Backtest [{start_date} → {end_date}] ===")

    # Load predictions
    df = load_historical_predictions(start_date, end_date)
    if df.empty:
        logger.warning("No prediction data found — backtest aborted")
        return {}

    # Attach realised returns
    df = load_realised_returns(df, horizon_days=horizon_days)
    df = df.dropna(subset=["Realised_Return"])

    if df.empty:
        logger.warning("No realised returns available — check price data coverage")
        return {}

    logger.info(f"Backtesting on {df['Date'].nunique()} days, {df['Ticker'].nunique()} stocks")

    results = {}

    # Strategy A: Top-N equal weight
    results["top10_ew"] = simulate_top_n_equal_weight(df, n=top_n)
    results["top10_ew"].print_report()

    # Strategy B: Kelly weighted
    if "Final_Weight" in df.columns:
        results["kelly"] = simulate_kelly_weighted(df)
        results["kelly"].print_report()

    # Save results
    _save_backtest_results(results)
    return results


def _save_backtest_results(results: dict[str, BacktestResult]) -> None:
    import json
    save_path = RESULTS_DIR / "backtest_results.json"
    out = {}
    for key, res in results.items():
        d = {
            "strategy":          res.strategy,
            "start_date":        res.start_date,
            "end_date":          res.end_date,
            "total_return":      res.total_return,
            "annualized_return": res.annualized_return,
            "annualized_sharpe": res.annualized_sharpe,
            "max_drawdown":      res.max_drawdown,
            "win_rate":          res.win_rate,
            "avg_turnover":      res.avg_turnover,
            "n_trading_days":    res.n_trading_days,
        }
        out[key] = d
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Backtest results saved → {save_path}")
