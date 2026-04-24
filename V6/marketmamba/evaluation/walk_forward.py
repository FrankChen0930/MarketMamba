"""
MarketMamba V6 — Walk-Forward Validation
==========================================
Expanding-Window Walk-Forward framework.

Design:
  - Fixed start: 2012-01-01
  - Test window: 6 months per fold
  - Roll step  : 3 months
  - Min train  : 3 years before first test fold
  - ~36 total folds (9 years × 4 folds/year)

Primary metric: ICIR (IC mean / IC std) across all folds.
Acceptance threshold: ICIR > 0.5, IC_mean > 0.05.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from marketmamba.config import (
    IC_THRESHOLD,
    ICIR_THRESHOLD,
    MODELS_DIR,
    PRED_HORIZONS,
    WF_MIN_TRAIN_YEARS,
    WF_STEP_MONTHS,
    WF_TEST_WINDOW_MONTHS,
)

logger = logging.getLogger(__name__)


# ============================================================
# Result Data Classes
# ============================================================

@dataclass
class FoldResult:
    fold_id:      int
    train_start:  str     # always '2012-01-01' for Expanding Window
    train_end:    str     # grows with each fold
    test_start:   str
    test_end:     str
    ic_daily:     list[float] = field(default_factory=list)  # IC per test day
    ic_mean:      float = 0.0
    ic_std:       float = 0.0
    ic_ir:        float = 0.0
    top10_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate:     float = 0.0   # fraction of days with positive IC

    def compute_metrics(self) -> None:
        if not self.ic_daily:
            return
        arr = np.array(self.ic_daily)
        self.ic_mean = float(np.mean(arr))
        self.ic_std  = float(np.std(arr))
        self.ic_ir   = self.ic_mean / (self.ic_std + 1e-9)
        self.win_rate = float((arr > 0).mean())

    def passed(self) -> bool:
        return self.ic_mean > IC_THRESHOLD and self.ic_ir > ICIR_THRESHOLD


@dataclass
class WalkForwardSummary:
    folds:       list[FoldResult] = field(default_factory=list)
    global_ic_mean: float = 0.0
    global_ic_std:  float = 0.0
    global_icir:    float = 0.0
    pass_rate:      float = 0.0   # fraction of folds that passed

    def compute_summary(self) -> None:
        all_ics = []
        for fold in self.folds:
            all_ics.extend(fold.ic_daily)
        if not all_ics:
            return
        arr = np.array(all_ics)
        self.global_ic_mean = float(np.mean(arr))
        self.global_ic_std  = float(np.std(arr))
        self.global_icir    = self.global_ic_mean / (self.global_ic_std + 1e-9)
        self.pass_rate      = float(np.mean([f.passed() for f in self.folds]))

    def print_report(self) -> None:
        logger.info("=" * 60)
        logger.info("Walk-Forward Validation Summary")
        logger.info("=" * 60)
        logger.info(f"  Total folds   : {len(self.folds)}")
        logger.info(f"  Passed folds  : {int(self.pass_rate * len(self.folds))} ({self.pass_rate:.1%})")
        logger.info(f"  Global IC     : {self.global_ic_mean:+.4f} ± {self.global_ic_std:.4f}")
        logger.info(f"  Global ICIR   : {self.global_icir:+.4f}")
        threshold_icon = "✅" if self.global_icir > ICIR_THRESHOLD else "❌"
        logger.info(f"  ICIR >{ICIR_THRESHOLD}: {threshold_icon}")
        logger.info("-" * 60)
        for fold in self.folds:
            status = "✅" if fold.passed() else "❌"
            logger.info(
                f"  Fold {fold.fold_id:02d} [{fold.test_start} → {fold.test_end}] "
                f"IC={fold.ic_mean:+.4f} ICIR={fold.ic_ir:+.4f} "
                f"Win={fold.win_rate:.1%} {status}"
            )
        logger.info("=" * 60)


# ============================================================
# Date Utilities
# ============================================================

def _add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    year  = dt.year + month // 12
    month = month % 12 + 1
    import calendar
    day   = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def generate_folds(
    all_dates: list[str],
    train_start: str  = "2012-01-01",
    test_window_months: int = WF_TEST_WINDOW_MONTHS,
    step_months:        int = WF_STEP_MONTHS,
    min_train_years:    int = WF_MIN_TRAIN_YEARS,
) -> list[dict]:
    """
    Generate Expanding-Window fold specifications.

    Returns list of dicts:
        {train_start, train_end, test_start, test_end, train_dates, test_dates}
    """
    dates = sorted(pd.to_datetime(all_dates))
    date_set = set(d.strftime("%Y-%m-%d") for d in dates)

    start_dt = pd.Timestamp(train_start)
    min_test_start = start_dt + pd.DateOffset(years=min_train_years)

    # First test window starts at min_test_start
    test_start_dt = min_test_start

    folds = []
    fold_id = 1

    while True:
        test_end_dt = _add_months(test_start_dt, test_window_months)

        if test_end_dt > dates[-1]:
            break

        train_end_dt = test_start_dt - pd.Timedelta(days=1)

        # Filter actual trading days
        train_dates = [
            d.strftime("%Y-%m-%d")
            for d in dates
            if start_dt <= d <= train_end_dt
        ]
        test_dates = [
            d.strftime("%Y-%m-%d")
            for d in dates
            if pd.Timestamp(test_start_dt) <= d < pd.Timestamp(test_end_dt)
        ]

        if len(train_dates) >= 100 and len(test_dates) >= 20:
            folds.append({
                "fold_id":     fold_id,
                "train_start": train_start,
                "train_end":   train_end_dt.strftime("%Y-%m-%d"),
                "test_start":  test_start_dt.strftime("%Y-%m-%d"),
                "test_end":    test_end_dt.strftime("%Y-%m-%d"),
                "train_dates": train_dates,
                "test_dates":  test_dates,
            })
            fold_id += 1

        test_start_dt = _add_months(test_start_dt, step_months)

    logger.info(f"Generated {len(folds)} Walk-Forward folds")
    return folds


# ============================================================
# IC Computation
# ============================================================

def compute_daily_ic(
    preds:    np.ndarray,   # (N,) predicted alpha scores
    targets:  np.ndarray,   # (N,) realised alpha
) -> float:
    """Spearman rank IC for one cross-section."""
    if len(preds) < 5:
        return 0.0
    corr, _ = spearmanr(preds, targets)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_top10_return(
    preds:   np.ndarray,
    targets: np.ndarray,
) -> float:
    """Equally-weighted return of top-10 predicted stocks."""
    if len(preds) < 10:
        return float(np.mean(targets))
    top_idx = np.argsort(preds)[-10:]
    return float(np.mean(targets[top_idx]))


# ============================================================
# Inference Helper (for Walk-Forward evaluation)
# ============================================================

def run_inference_for_eval(
    model:      torch.nn.Module,
    df:         pd.DataFrame,
    test_dates: list[str],
    device:     torch.device,
    horizon_idx: int = 1,   # 0=5d, 1=20d (default), 2=60d
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over test_dates and return (all_preds, all_targets) arrays.
    Used by Walk-Forward to compute IC per day.
    """
    from marketmamba.models.trainer import TemporalCrossSectionDataset, load_kg_edges
    from marketmamba.config import FEATURE_COLS, SEQ_LEN

    test_ds = TemporalCrossSectionDataset(df, test_dates, mode="test")
    edge_index, edge_attr = load_kg_edges(df["stock_id"].unique().tolist(), device)

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, Y in test_ds:
            X = X.to(device)
            preds = model(X, edge_index, edge_attr)
            all_preds.append(preds[:, horizon_idx].cpu().numpy())
            all_targets.append(Y[:, horizon_idx].numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ============================================================
# Main Walk-Forward Runner
# ============================================================

def run_walk_forward(
    df:              pd.DataFrame,
    train_fn:        Callable,   # signature: (df, train_dates, val_dates) → model
    train_start:     str = "2012-01-01",
    eval_horizon_idx: int = 1,   # 1 = 20d alpha (primary)
    device_str:      str = "cuda" if torch.cuda.is_available() else "cpu",
    save_results:    bool = True,
) -> WalkForwardSummary:
    """
    Full Expanding-Window Walk-Forward evaluation.

    Args:
        df           : complete feature DataFrame
        train_fn     : callable that trains the model and returns it
                       signature: train_fn(df, train_dates, val_dates) → nn.Module
        train_start  : expanding window fixed start date
        eval_horizon_idx : which pred horizon to evaluate (0=5d, 1=20d, 2=60d)
        device_str   : 'cuda' or 'cpu'
        save_results : save fold results to MODELS_DIR/walk_forward_results.parquet

    Returns:
        WalkForwardSummary with per-fold IC metrics
    """
    device = torch.device(device_str)
    all_dates = sorted(df["Date"].astype(str).unique().tolist())

    folds_spec = generate_folds(all_dates, train_start=train_start)

    if not folds_spec:
        raise ValueError(
            "No valid folds generated. Check that the DataFrame covers "
            f"at least {WF_MIN_TRAIN_YEARS + WF_TEST_WINDOW_MONTHS / 12:.0f} years."
        )

    summary = WalkForwardSummary()

    for spec in folds_spec:
        fold_id = spec["fold_id"]
        logger.info(
            f"\n{'='*55}\n"
            f"  Fold {fold_id:02d}: train=[{spec['train_start']} → {spec['train_end']}] "
            f"| test=[{spec['test_start']} → {spec['test_end']}]\n"
            f"{'='*55}"
        )

        # Split val from end of train (15% of train days)
        train_d = spec["train_dates"]
        n_val   = max(20, int(len(train_d) * 0.15))
        val_d   = train_d[-n_val:]
        train_d = train_d[:-n_val]

        # Train
        model = train_fn(df, train_d, val_d)
        model.to(device)

        # Evaluate on test set, day by day
        fold_result = FoldResult(
            fold_id=fold_id,
            train_start=spec["train_start"],
            train_end=spec["train_end"],
            test_start=spec["test_start"],
            test_end=spec["test_end"],
        )

        from marketmamba.models.trainer import TemporalCrossSectionDataset, load_kg_edges
        test_ds = TemporalCrossSectionDataset(df, spec["test_dates"], mode="test")
        edge_index, edge_attr = load_kg_edges(df["stock_id"].unique().tolist(), device)

        model.eval()
        test_dates_used = []
        with torch.no_grad():
            for i, (X, Y) in enumerate(test_ds):
                X = X.to(device)
                preds = model(X, edge_index, edge_attr)
                pred_np   = preds[:, eval_horizon_idx].cpu().numpy()
                target_np = Y[:, eval_horizon_idx].numpy()

                ic = compute_daily_ic(pred_np, target_np)
                fold_result.ic_daily.append(ic)
                test_dates_used.append(test_ds.valid_dates[i])

        fold_result.compute_metrics()

        # Top-10 return (aggregate over test period)
        # Compute once over all test stocks pooled
        all_preds, all_targets = run_inference_for_eval(model, df, spec["test_dates"], device, eval_horizon_idx)
        fold_result.top10_return = compute_top10_return(all_preds, all_targets)

        summary.folds.append(fold_result)
        logger.info(
            f"  Fold {fold_id:02d} result: IC={fold_result.ic_mean:+.4f} "
            f"ICIR={fold_result.ic_ir:+.4f} Win={fold_result.win_rate:.1%} "
            f"{'✅' if fold_result.passed() else '❌'}"
        )

        # Free GPU memory between folds
        del model
        torch.cuda.empty_cache()

    summary.compute_summary()
    summary.print_report()

    if save_results:
        _save_results(summary)

    return summary


def _save_results(summary: WalkForwardSummary) -> None:
    rows = []
    for fold in summary.folds:
        rows.append({
            "fold_id":      fold.fold_id,
            "train_start":  fold.train_start,
            "train_end":    fold.train_end,
            "test_start":   fold.test_start,
            "test_end":     fold.test_end,
            "ic_mean":      fold.ic_mean,
            "ic_std":       fold.ic_std,
            "ic_ir":        fold.ic_ir,
            "win_rate":     fold.win_rate,
            "top10_return": fold.top10_return,
            "passed":       fold.passed(),
        })
    df_results = pd.DataFrame(rows)
    save_path = MODELS_DIR / "walk_forward_results.parquet"
    df_results.to_parquet(save_path)
    logger.info(f"Walk-Forward results saved → {save_path}")
