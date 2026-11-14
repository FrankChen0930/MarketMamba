"""
MarketMamba V6.2 — Dual Model IC / Top50 Analyzer
====================================================
驗證雙模型（短線 v6_short / 趨勢 v6_trend）的排名訊號在真實市場走勢下
是否有效，方法比照 V6.1 的 `ic_analyzer.py`，但改讀 dual 模型的每日歸檔：

    V6/results/archive/df_short_{date}.csv   (Score_5d/10d, Unc_5d/10d, SQ_5d)
    V6/results/archive/df_trend_{date}.csv   (Score_20d/60d, Unc_20d/60d, SQ_20d)

對每個 horizon（5d/10d/20d/60d）計算：
  - 每日 IC：Spearman(SQ 排名, 實際 N 日報酬)
  - Top50 by SQ 的實現超額報酬（去當日截面均值）—— 比 IC 更直覺回答
    「照這個排名選股，過去這段時間會不會賺錢」
  - 彙總統計：mean IC、ICIR、t 統計量、IC>0 比例、有效天數

輸出：V6/results/dual_ic_analysis.json

⚠️ 樣本量會隨時間自然增加，不需要手動介入：
  - 5d/10d 需要對應天數的價格資料才能計算，早期 archive 天數不足時該 horizon
    n_days=0（非錯誤，是正常的「還沒到時間」）
  - 20d/60d 需要更長時間才會有第一筆資料（約 archive 開始後 1~3 個月）
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

# horizon → (score_col, sq_col, n_trading_days, source)
HORIZONS = {
    "5d":  ("Score_5d",  "SQ_5d",  5,  "short"),
    "10d": ("Score_10d", "SQ_5d",  10, "short"),   # 10d 用 5d 的 SQ 排名（模型只出一組 SQ）
    "20d": ("Score_20d", "SQ_20d", 20, "trend"),
    "60d": ("Score_60d", "SQ_20d", 60, "trend"),
}
MIN_STOCKS_FOR_IC = 50
TOP_N = 50


def _trading_days_later(date_str: str, n: int, available_dates: list[str]) -> str | None:
    try:
        idx = available_dates.index(date_str)
    except ValueError:
        return None
    target_idx = idx + n
    if target_idx >= len(available_dates):
        return None
    return available_dates[target_idx]


def compute_dual_ic_series(
    archive_dir: Path,
    prices_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute IC / Top50 excess return for each dual-model horizon.

    Args:
        archive_dir : V6/results/archive/ — 內含 df_short_{date}.csv / df_trend_{date}.csv
        prices_df   : prices_raw DataFrame with [Date, stock_id, Close, ...]
    """
    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.strftime("%Y-%m-%d")
    prices_df["stock_id"] = prices_df["stock_id"].astype(str)
    prices_df = prices_df.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    price_wide = prices_df.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    ).sort_index()
    available_price_dates = sorted(price_wide.index.tolist())

    short_dates = sorted(
        p.stem.replace("df_short_", "") for p in archive_dir.glob("df_short_*.csv")
    )
    trend_dates = sorted(
        p.stem.replace("df_trend_", "") for p in archive_dir.glob("df_trend_*.csv")
    )
    if not short_dates and not trend_dates:
        logger.warning("Dual IC Analyzer: no archive files found")
        return _empty_result()

    logger.info(
        f"Dual IC Analyzer: short={len(short_dates)} days "
        f"({short_dates[0] if short_dates else '—'} → {short_dates[-1] if short_dates else '—'}), "
        f"trend={len(trend_dates)} days"
    )

    # Cache loaded archive frames per (source, date) to avoid re-reading short twice (5d+10d share it)
    _cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _load(source: str, date: str) -> pd.DataFrame | None:
        key = (source, date)
        if key in _cache:
            return _cache[key]
        path = archive_dir / f"df_{source}_{date}.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, dtype={"stock_id": str})
        except Exception as e:
            logger.warning(f"  [{source}/{date}] cannot read: {e}")
            return None
        _cache[key] = df
        return df

    results_per_horizon: dict[str, list[dict]] = {h: [] for h in HORIZONS}

    for horizon_label, (score_col, sq_col, n_days, source) in HORIZONS.items():
        dates = short_dates if source == "short" else trend_dates
        for pred_date in dates:
            df_pred = _load(source, pred_date)
            if df_pred is None or score_col not in df_pred.columns or sq_col not in df_pred.columns:
                continue

            future_date = _trading_days_later(pred_date, n_days, available_price_dates)
            if future_date is None:
                continue   # 還沒到時間，正常情況
            if pred_date not in price_wide.index or future_date not in price_wide.index:
                continue

            p0 = price_wide.loc[pred_date]
            p1 = price_wide.loc[future_date]
            tickers = df_pred["stock_id"].values
            p0_aligned = p0.reindex(tickers)
            p1_aligned = p1.reindex(tickers)
            valid = p0_aligned.notna() & p1_aligned.notna() & (p0_aligned > 0)
            if valid.sum() < MIN_STOCKS_FOR_IC:
                continue

            actual_ret = (p1_aligned[valid] / p0_aligned[valid] - 1)
            sub = df_pred.set_index("stock_id").loc[actual_ret.index]

            finite = np.isfinite(sub[sq_col].values) & np.isfinite(actual_ret.values)
            if finite.sum() < MIN_STOCKS_FOR_IC:
                continue

            sq_vals  = sub[sq_col].values[finite]
            ret_vals = actual_ret.values[finite]
            act = ret_vals - ret_vals.mean()   # 去市場均值的 alpha proxy

            ic, _ = stats.spearmanr(sq_vals, ret_vals)

            # Top50 by SQ 的實現超額報酬
            order = np.argsort(-sq_vals)[:TOP_N]
            top50_excess = float(act[order].mean() * 100) if len(order) > 0 else None

            results_per_horizon[horizon_label].append({
                "pred_date":       pred_date,
                "future_date":     future_date,
                "ic":              round(float(ic), 4),
                "n_stocks":        int(finite.sum()),
                "top50_excess_pct": round(top50_excess, 3) if top50_excess is not None else None,
            })

    horizon_summaries: dict[str, dict] = {}
    for horizon_label, ic_list in results_per_horizon.items():
        if not ic_list:
            horizon_summaries[horizon_label] = {
                "n_days": 0, "mean_ic": None, "icir": None, "t_stat": None,
                "ic_gt0_pct": None, "mean_top50_excess_pct": None,
            }
            continue

        ic_values = np.array([r["ic"] for r in ic_list])
        top_values = np.array([r["top50_excess_pct"] for r in ic_list if r["top50_excess_pct"] is not None])
        n = len(ic_values)
        mean_ic = float(np.mean(ic_values))
        std_ic  = float(np.std(ic_values))
        icir    = float(mean_ic / (std_ic + 1e-9))
        t_stat  = float(mean_ic / (std_ic / np.sqrt(n) + 1e-9)) if n >= 2 else 0.0
        ic_gt0  = float((ic_values > 0).mean() * 100)
        mean_top = float(np.mean(top_values)) if len(top_values) else None

        horizon_summaries[horizon_label] = {
            "n_days":     n,
            "mean_ic":    round(mean_ic, 4),
            "icir":       round(icir, 3),
            "t_stat":     round(t_stat, 2),
            "ic_gt0_pct": round(ic_gt0, 1),
            "mean_top50_excess_pct": round(mean_top, 3) if mean_top is not None else None,
        }
        logger.info(
            f"  [dual/{horizon_label}] n={n} mean_ic={mean_ic:+.4f} icir={icir:.3f} "
            f"top50_excess={mean_top if mean_top is not None else float('nan'):+.3f}%/期"
        )

    all_dates = sorted(set(short_dates) | set(trend_dates))
    result: dict[str, Any] = {
        "generated_date":     pd.Timestamp.now().strftime("%Y-%m-%d"),
        "archive_count_short": len(short_dates),
        "archive_count_trend": len(trend_dates),
        "period_start":       all_dates[0] if all_dates else None,
        "period_end":         all_dates[-1] if all_dates else None,
        "horizon_summary":    horizon_summaries,
        "ic_series_5d":       results_per_horizon["5d"],
        "ic_series_10d":      results_per_horizon["10d"],
        "ic_series_20d":      results_per_horizon["20d"],
        "ic_series_60d":      results_per_horizon["60d"],
    }
    return result


def run_dual_ic_analysis(
    archive_dir: Path,
    prices_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Compute dual-model IC analysis and save to dual_ic_analysis.json."""
    result = compute_dual_ic_series(archive_dir, prices_df)
    if not result:
        return {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s5  = result["horizon_summary"].get("5d", {})
    s20 = result["horizon_summary"].get("20d", {})
    logger.info(
        f"✅ dual_ic_analysis saved → {output_path.name} | "
        f"5d: n={s5.get('n_days', 0)} IC={s5.get('mean_ic', 'N/A')} | "
        f"20d: n={s20.get('n_days', 0)} IC={s20.get('mean_ic', 'N/A')}"
    )
    return result


def _empty_result() -> dict[str, Any]:
    empty_h = {"n_days": 0, "mean_ic": None, "icir": None, "t_stat": None,
               "ic_gt0_pct": None, "mean_top50_excess_pct": None}
    return {
        "generated_date":      pd.Timestamp.now().strftime("%Y-%m-%d"),
        "archive_count_short": 0,
        "archive_count_trend": 0,
        "period_start":        None,
        "period_end":          None,
        "horizon_summary":     {h: dict(empty_h) for h in HORIZONS},
        "ic_series_5d":        [],
        "ic_series_10d":       [],
        "ic_series_20d":       [],
        "ic_series_60d":       [],
    }
