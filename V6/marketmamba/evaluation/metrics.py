"""
MarketMamba V6 — IC / ICIR Metrics
=====================================
Standalone evaluation metrics module.
No dependency on model or trainer — can be used anywhere.

Primary metrics:
  IC    : Spearman rank correlation between predicted and realised alpha (cross-sectional)
  ICIR  : IC mean / IC std — signal stability metric
  Top-N : Equal-weight return of top-N ranked stocks
  MaxDD : Maximum drawdown of a cumulative return series
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


# ============================================================
# Core IC Functions
# ============================================================

def ic_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Cross-sectional Spearman Rank IC.
    Standard metric in quantitative finance for evaluating alpha factors.

    Args:
        pred   : (N,) predicted scores / ranks
        target : (N,) realised returns / alpha values
    Returns:
        Spearman correlation coefficient in [-1, 1]
    """
    if len(pred) < 5 or np.std(pred) < 1e-10:
        return 0.0
    corr, _ = spearmanr(pred, target, nan_policy="omit")
    return float(corr) if not np.isnan(corr) else 0.0


def ic_pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Cross-sectional Pearson IC (linear correlation).
    Less robust than Spearman for fat-tailed returns, but useful for comparison.
    """
    if len(pred) < 5 or np.std(pred) < 1e-10:
        return 0.0
    corr, _ = pearsonr(pred, target)
    return float(corr) if not np.isnan(corr) else 0.0


def icir(ic_series: list[float] | np.ndarray) -> float:
    """
    ICIR = mean(IC) / std(IC)
    Measures signal consistency. ICIR > 0.5 is the industry acceptance threshold.
    """
    arr = np.array(ic_series)
    std = np.std(arr)
    if std < 1e-10:
        return 0.0
    return float(np.mean(arr) / std)


def ic_decay(
    df:          pd.DataFrame,
    pred_col:    str,
    target_cols: list[str],  # e.g. ['Alpha_5d', 'Alpha_20d', 'Alpha_60d']
    date_col:    str = "Date",
) -> pd.DataFrame:
    """
    Compute IC at multiple forward horizons to assess signal decay.

    Returns DataFrame with columns [horizon, ic_mean, ic_std, icir, win_rate]
    """
    rows = []
    for col in target_cols:
        ics = []
        for date, group in df.groupby(date_col):
            valid = group[[pred_col, col]].dropna()
            if len(valid) < 5:
                continue
            ics.append(ic_spearman(valid[pred_col].values, valid[col].values))
        arr = np.array(ics)
        rows.append({
            "horizon":  col,
            "ic_mean":  float(np.mean(arr)),
            "ic_std":   float(np.std(arr)),
            "icir":     icir(arr),
            "win_rate": float((arr > 0).mean()),
            "n_days":   len(arr),
        })
    return pd.DataFrame(rows)


# ============================================================
# Cross-Sectional Evaluation
# ============================================================

@dataclass
class DailyEvalResult:
    date:        str
    ic_spearman: float = 0.0
    ic_pearson:  float = 0.0
    top5_return: float = 0.0
    top10_return: float = 0.0
    n_stocks:    int   = 0


def evaluate_cross_section(
    pred:    np.ndarray,
    target:  np.ndarray,
    date:    str = "",
    top_ns:  list[int] = [5, 10],
) -> DailyEvalResult:
    """
    Evaluate a single cross-section: IC + top-N returns.
    """
    n = len(pred)
    result = DailyEvalResult(date=date, n_stocks=n)
    result.ic_spearman = ic_spearman(pred, target)
    result.ic_pearson  = ic_pearson(pred, target)

    sorted_idx = np.argsort(pred)[::-1]
    for k in top_ns:
        if n >= k:
            top_k_return = float(np.mean(target[sorted_idx[:k]]))
            if k == 5:
                result.top5_return = top_k_return
            elif k == 10:
                result.top10_return = top_k_return
    return result


# ============================================================
# Aggregated Evaluation Over a Period
# ============================================================

@dataclass
class PeriodEvalSummary:
    ic_mean:      float = 0.0
    ic_std:       float = 0.0
    ic_ir:        float = 0.0
    win_rate:     float = 0.0
    top10_mean:   float = 0.0
    max_drawdown: float = 0.0
    n_days:       int   = 0
    daily_results: list[DailyEvalResult] = field(default_factory=list)

    def print_report(self, prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        status = "✅" if self.ic_ir > 0.5 and self.ic_mean > 0.05 else "❌"
        logger.info(
            f"{tag}IC={self.ic_mean:+.4f}±{self.ic_std:.4f} "
            f"ICIR={self.ic_ir:+.4f} Win={self.win_rate:.1%} "
            f"Top10={self.top10_mean:+.4f} MaxDD={self.max_drawdown:.4f} {status}"
        )


def evaluate_period(
    df:           pd.DataFrame,
    pred_col:     str    = "Pred_20d",
    target_col:   str    = "Alpha_20d",
    date_col:     str    = "Date",
    stock_col:    str    = "stock_id",
    benchmark_col: str | None = None,  # e.g. 'TWII_Return' for relative eval
) -> PeriodEvalSummary:
    """
    Evaluate model predictions over a full period (many trading days).

    For each day, computes cross-sectional IC.
    Then aggregates IC_mean, IC_std, ICIR, win_rate, top-10 return, max-drawdown.

    Args:
        df           : DataFrame with pred_col and target_col columns
        pred_col     : column name of model predictions
        target_col   : column name of realised alpha targets
        benchmark_col: if set, subtract benchmark return to get excess return
    """
    daily_results = []
    top10_equity  = [1.0]   # for max-drawdown computation

    for date, group in df.groupby(date_col):
        valid = group[[pred_col, target_col]].dropna()
        if len(valid) < 5:
            continue
        pred   = valid[pred_col].values
        target = valid[target_col].values

        res = evaluate_cross_section(pred, target, date=str(date))
        daily_results.append(res)

        # Equity curve for top-10
        top10_equity.append(top10_equity[-1] * (1 + res.top10_return))

    if not daily_results:
        return PeriodEvalSummary()

    ics = np.array([r.ic_spearman for r in daily_results])
    top10_arr = np.array(top10_equity)

    summary = PeriodEvalSummary(
        ic_mean    = float(np.mean(ics)),
        ic_std     = float(np.std(ics)),
        ic_ir      = icir(ics),
        win_rate   = float((ics > 0).mean()),
        top10_mean = float(np.mean([r.top10_return for r in daily_results])),
        max_drawdown = _max_drawdown(top10_arr),
        n_days     = len(daily_results),
        daily_results = daily_results,
    )
    return summary


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of an equity curve."""
    roll_max = np.maximum.accumulate(equity)
    drawdown = (equity - roll_max) / (roll_max + 1e-10)
    return float(drawdown.min())


# ============================================================
# Factor Decay Analysis
# ============================================================

def compute_autocorrelation(
    df:       pd.DataFrame,
    pred_col: str,
    date_col: str = "Date",
    stock_col: str = "stock_id",
    lags: list[int] = [1, 5, 10, 20],
) -> pd.DataFrame:
    """
    Compute signal auto-correlation at different lags.
    High auto-correlation means signal is persistent (good for low-turnover strategies).
    """
    rows = []
    for lag in lags:
        ics = []
        dates = sorted(df[date_col].unique())
        for i, date in enumerate(dates):
            if i < lag:
                continue
            past_date = dates[i - lag]
            cur  = df[df[date_col] == date][[stock_col, pred_col]].set_index(stock_col)
            past = df[df[date_col] == past_date][[stock_col, pred_col]].set_index(stock_col)
            merged = cur.join(past, how="inner", lsuffix="_cur", rsuffix="_past")
            if len(merged) < 5:
                continue
            ic = ic_spearman(merged[f"{pred_col}_cur"].values, merged[f"{pred_col}_past"].values)
            ics.append(ic)
        rows.append({
            "lag":     lag,
            "auto_ic": float(np.mean(ics)) if ics else 0.0,
        })
    return pd.DataFrame(rows)
