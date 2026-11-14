"""
MarketMamba V6 — IC / ICIR Analyzer
=====================================
Validates the model's predictive power by computing the Information Coefficient (IC)
between predicted Alpha and actual realized returns.

IC (per day) = Spearman rank correlation between:
    - predicted Exp_Alpha_Nd  (from df_kelly.csv on prediction date)
    - actual N-day return     (realized from prices_raw.parquet)

ICIR = mean(IC) / std(IC)  — consistency of the factor

Output: V6/results/ic_analysis.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

HORIZONS = {
    "5d":  ("Exp_Alpha_5d",  5),
    "20d": ("Exp_Alpha_20d", 20),
}
MIN_STOCKS_FOR_IC = 50   # 至少需要 50 支股票才計算當天 IC


def _trading_days_later(date_str: str, n: int, available_dates: list[str]) -> str | None:
    """Return the date that is ~n trading days after date_str, from a sorted list."""
    try:
        idx = available_dates.index(date_str)
    except ValueError:
        return None
    target_idx = idx + n
    if target_idx >= len(available_dates):
        return None
    return available_dates[target_idx]


def compute_ic_series(
    results_dir: Path,
    prices_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute IC / ICIR for each horizon using archived df_kelly.csv files.

    Args:
        results_dir : V6/results/ — contains YYYY-MM-DD/ subdirectories
        prices_df   : prices_raw DataFrame with [Date, stock_id, Close, ...]

    Returns:
        Dict ready to save as ic_analysis.json
    """
    # ── Prepare prices pivot ──────────────────────────────────────────────────
    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.strftime("%Y-%m-%d")
    price_wide = prices_df.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    ).sort_index()

    available_price_dates = sorted(price_wide.index.tolist())

    # ── Discover archive dates ────────────────────────────────────────────────
    archive_dates = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and len(d.name) == 10
        and (d / "df_kelly.csv").exists()
    ])

    if not archive_dates:
        logger.warning("IC Analyzer: no archive dates found")
        return _empty_result()

    logger.info(
        f"IC Analyzer: {len(archive_dates)} dates "
        f"({archive_dates[0]} → {archive_dates[-1]})"
    )

    # ── Per-horizon IC series ─────────────────────────────────────────────────
    results_per_horizon: dict[str, list[dict]] = {h: [] for h in HORIZONS}

    for pred_date in archive_dates:
        try:
            df_pred = pd.read_csv(
                results_dir / pred_date / "df_kelly.csv",
                dtype={"Ticker": str},
            )
        except Exception as e:
            logger.warning(f"  [{pred_date}] cannot read: {e}")
            continue

        if "Ticker" not in df_pred.columns:
            continue

        # Support legacy column name
        if "Sharpe_Score" in df_pred.columns and "Signal_Quality" not in df_pred.columns:
            df_pred = df_pred.rename(columns={"Sharpe_Score": "Signal_Quality"})

        # Remove illiquid sentinel
        df_pred = df_pred[df_pred.get("Exp_Alpha_20d", pd.Series([0] * len(df_pred))).gt(-900)].copy()

        for horizon_label, (alpha_col, n_days) in HORIZONS.items():
            if alpha_col not in df_pred.columns:
                continue

            # Find the future date that is n trading days after pred_date
            future_date = _trading_days_later(pred_date, n_days, available_price_dates)
            if future_date is None:
                continue  # not enough future data yet

            # Get prices on pred_date and future_date
            if pred_date not in price_wide.index or future_date not in price_wide.index:
                continue

            p0 = price_wide.loc[pred_date]
            p1 = price_wide.loc[future_date]

            # Compute actual N-day return per stock
            common_tickers = df_pred["Ticker"].astype(str).values
            p0_aligned = p0.reindex(common_tickers)
            p1_aligned = p1.reindex(common_tickers)

            valid_mask = p0_aligned.notna() & p1_aligned.notna() & (p0_aligned > 0)
            n_valid = valid_mask.sum()

            if n_valid < MIN_STOCKS_FOR_IC:
                logger.debug(
                    f"  [{pred_date}] {horizon_label}: only {n_valid} valid stocks — skip"
                )
                continue

            actual_ret = (p1_aligned[valid_mask] / p0_aligned[valid_mask] - 1).values
            pred_alpha = df_pred.set_index("Ticker")[alpha_col].reindex(
                p0_aligned[valid_mask].index
            ).values

            # Drop any remaining NaN rows
            finite = np.isfinite(pred_alpha) & np.isfinite(actual_ret)
            if finite.sum() < MIN_STOCKS_FOR_IC:
                continue

            ic, _ = stats.spearmanr(pred_alpha[finite], actual_ret[finite])

            results_per_horizon[horizon_label].append({
                "pred_date":    pred_date,
                "future_date":  future_date,
                "ic":           round(float(ic), 4),
                "n_stocks":     int(finite.sum()),
            })
            logger.debug(
                f"  [{pred_date}] IC_{horizon_label} = {ic:+.4f} "
                f"(n={finite.sum()}, future={future_date})"
            )

    # ── Aggregate stats per horizon ───────────────────────────────────────────
    horizon_summaries: dict[str, dict] = {}
    for horizon_label, ic_list in results_per_horizon.items():
        if not ic_list:
            horizon_summaries[horizon_label] = {
                "n_days":     0,
                "mean_ic":    None,
                "icir":       None,
                "t_stat":     None,
                "ic_gt0_pct": None,
            }
            continue

        ic_values = np.array([r["ic"] for r in ic_list])
        n = len(ic_values)
        mean_ic  = float(np.mean(ic_values))
        std_ic   = float(np.std(ic_values))
        icir     = float(mean_ic / (std_ic + 1e-9))
        t_stat   = float(mean_ic / (std_ic / np.sqrt(n) + 1e-9)) if n >= 2 else 0.0
        ic_gt0   = float((ic_values > 0).mean() * 100)

        horizon_summaries[horizon_label] = {
            "n_days":     n,
            "mean_ic":    round(mean_ic, 4),
            "icir":       round(icir, 3),
            "t_stat":     round(t_stat, 2),
            "ic_gt0_pct": round(ic_gt0, 1),
        }

        logger.info(
            f"  IC_{horizon_label}: mean={mean_ic:+.4f}  "
            f"ICIR={icir:.3f}  t={t_stat:.2f}  "
            f"IC>0={ic_gt0:.0f}%  (n={n})"
        )

    # ── Rolling IC (5-day window, 5d horizon only) ────────────────────────────
    rolling_5d: list[dict] = []
    if results_per_horizon["5d"]:
        s = pd.Series(
            [r["ic"] for r in results_per_horizon["5d"]],
            index=[r["pred_date"] for r in results_per_horizon["5d"]],
        )
        roll = s.rolling(5, min_periods=3).mean()
        for date_str, val in roll.items():
            if not np.isnan(val):
                rolling_5d.append({"date": date_str, "rolling_ic": round(float(val), 4)})

    result: dict[str, Any] = {
        "generated_date":    pd.Timestamp.now().strftime("%Y-%m-%d"),
        "archive_count":     len(archive_dates),
        "period_start":      archive_dates[0] if archive_dates else None,
        "period_end":        archive_dates[-1] if archive_dates else None,
        "horizon_summary":   horizon_summaries,
        "ic_series_5d":      results_per_horizon["5d"],
        "ic_series_20d":     results_per_horizon["20d"],
        "rolling_ic_5d":     rolling_5d,
    }
    return result


def run_ic_analysis(
    results_dir: Path,
    prices_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Compute IC analysis and save to ic_analysis.json."""
    result = compute_ic_series(results_dir, prices_df)
    if not result:
        return {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s5 = result["horizon_summary"].get("5d", {})
    logger.info(
        f"✅ IC analysis saved → {output_path.name}  "
        f"IC_5d={s5.get('mean_ic', 'N/A')}  "
        f"ICIR={s5.get('icir', 'N/A')}  "
        f"t={s5.get('t_stat', 'N/A')}"
    )
    return result


def _empty_result() -> dict[str, Any]:
    return {
        "generated_date":  pd.Timestamp.now().strftime("%Y-%m-%d"),
        "archive_count":   0,
        "period_start":    None,
        "period_end":      None,
        "horizon_summary": {},
        "ic_series_5d":    [],
        "ic_series_20d":   [],
        "rolling_ic_5d":   [],
    }
