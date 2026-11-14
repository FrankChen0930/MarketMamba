"""
MarketMamba — Investment Simulation Engine
==========================================
Computes a daily equity curve based on:
  - history_index.json  : each day's top-50 portfolio (Kelly weights)
  - prices_raw.parquet  : actual closing prices

Called from run_daily_inference.py after each inference run.
Output: V6/results/sim_backtest.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 1_000_000   # NT$100萬


def compute_sim_backtest(
    history_path: Path,
    output_path: Path,
    prices_df: pd.DataFrame | None = None,
    prices_path: Path | None = None,
) -> dict[str, Any]:
    """
    Read history_index.json and actual closing prices, then compute the
    daily equity curve for an investor who follows the model's top-50 portfolio.

    Args:
        history_path : path to V6/results/history_index.json
        output_path  : where to save sim_backtest.json
        prices_df    : pre-loaded prices DataFrame (Date, stock_id, Close, …)
                       Pass this from run_daily_inference to avoid a second disk read.
        prices_path  : used when prices_df is None
    """
    # ── Load history ──────────────────────────────────────────────────────────
    if not history_path.exists():
        logger.warning(f"history_index.json not found: {history_path}")
        return _save_empty(output_path)

    with open(history_path, encoding="utf-8") as f:
        raw = json.load(f)

    history: list[dict] = raw.get("history", [])
    if len(history) < 2:
        logger.warning("history_index has fewer than 2 entries — skipping backtest")
        return _save_empty(output_path)

    # Sort chronologically (oldest first)
    history = sorted(history, key=lambda x: x["date"])

    all_dates   = [e["date"] for e in history]
    all_tickers = {str(p["ticker"]) for e in history for p in e.get("portfolio", [])}

    # ── Load prices ───────────────────────────────────────────────────────────
    if prices_df is None:
        if prices_path is None or not prices_path.exists():
            logger.error("No prices source provided to sim_engine")
            return _save_empty(output_path)
        prices_df = pd.read_parquet(prices_path, columns=["Date", "stock_id", "Close"])

    # Normalise Date to string YYYY-MM-DD regardless of input dtype
    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.strftime("%Y-%m-%d")

    # Keep only the date range and tickers we need
    prices_df = prices_df[
        prices_df["Date"].between(all_dates[0], all_dates[-1])
        & prices_df["stock_id"].isin(all_tickers)
    ]

    # Pivot: rows = date, columns = stock_id, values = Close
    prices_wide = prices_df.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    )
    # Forward-fill within the date range to handle halted stocks
    prices_wide = prices_wide.sort_index().ffill()

    # ── Daily equity curve ────────────────────────────────────────────────────
    equity = float(INITIAL_CAPITAL)
    equity_curve: list[dict] = []
    missing_ticker_log: set[str] = set()

    for i, entry in enumerate(history):
        date = entry["date"]

        if i == 0:
            equity_curve.append({
                "date": date,
                "equity": int(round(equity)),
                "pnl": 0,
                "daily_return_pct": 0.0,
            })
            continue

        prev_date  = history[i - 1]["date"]
        prev_ports = history[i - 1].get("portfolio", [])

        # Portfolio held from prev_date close → date close
        daily_return = 0.0
        for pos in prev_ports:
            ticker = str(pos["ticker"])
            weight = float(pos.get("weight", 0.0))
            if weight <= 0:
                continue
            try:
                p0 = prices_wide.at[prev_date, ticker]
                p1 = prices_wide.at[date,      ticker]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    daily_return += weight * ((p1 / p0) - 1.0)
                else:
                    missing_ticker_log.add(ticker)
            except KeyError:
                missing_ticker_log.add(ticker)

        equity *= (1.0 + daily_return)
        equity_curve.append({
            "date": date,
            "equity": int(round(equity)),
            "pnl": int(round(equity - INITIAL_CAPITAL)),
            "daily_return_pct": round(daily_return * 100, 3),
        })

    if missing_ticker_log:
        logger.info(
            f"Sim: {len(missing_ticker_log)} tickers had missing/NaN prices "
            f"(treated as 0 return): {sorted(missing_ticker_log)[:10]}{'…' if len(missing_ticker_log)>10 else ''}"
        )

    # ── Summary stats ─────────────────────────────────────────────────────────
    final_equity = equity_curve[-1]["equity"]
    total_return_pct = round((final_equity / INITIAL_CAPITAL - 1) * 100, 2)

    # Max drawdown
    peak = float(INITIAL_CAPITAL)
    max_dd = 0.0
    for pt in equity_curve:
        eq = pt["equity"]
        peak = max(peak, eq)
        dd   = (eq - peak) / peak
        max_dd = min(max_dd, dd)
    max_drawdown_pct = round(max_dd * 100, 2)

    # Daily return array (skip day 0 which is always 0)
    daily_rets = [pt["daily_return_pct"] for pt in equity_curve[1:]]
    win_days  = sum(1 for r in daily_rets if r > 0)
    lose_days = sum(1 for r in daily_rets if r < 0)
    flat_days = len(daily_rets) - win_days - lose_days

    # Annualised Sharpe (risk-free = 0)
    if len(daily_rets) >= 2:
        r_arr = np.array(daily_rets) / 100.0
        sharpe = float(np.mean(r_arr) / (np.std(r_arr) + 1e-9) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Best / worst day
    if daily_rets:
        bi = int(np.argmax(daily_rets))
        wi = int(np.argmin(daily_rets))
        best_day  = {"date": equity_curve[bi + 1]["date"], "pct": daily_rets[bi]}
        worst_day = {"date": equity_curve[wi + 1]["date"], "pct": daily_rets[wi]}
    else:
        best_day = worst_day = {"date": "", "pct": 0.0}

    # Average deployed capital (sum of top-50 weights)
    deployed_pcts = [
        sum(float(p.get("weight", 0)) for p in e.get("portfolio", [])) * 100
        for e in history
    ]
    avg_deployed_pct = round(float(np.mean(deployed_pcts)) if deployed_pcts else 0, 1)

    result: dict[str, Any] = {
        "generated_date":   datetime.today().strftime("%Y-%m-%d"),
        "initial_capital":  INITIAL_CAPITAL,
        "period_start":     equity_curve[0]["date"],
        "period_end":       equity_curve[-1]["date"],
        "trading_days":     len(equity_curve) - 1,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe":           round(sharpe, 2),
        "win_days":         win_days,
        "lose_days":        lose_days,
        "flat_days":        flat_days,
        "avg_deployed_pct": avg_deployed_pct,
        "best_day":         best_day,
        "worst_day":        worst_day,
        "equity_curve":     equity_curve,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        f"✅ Sim backtest saved → {output_path.name}  "
        f"{len(equity_curve)} days  return={total_return_pct:+.2f}%  "
        f"sharpe={sharpe:.2f}  dd={max_drawdown_pct:.2f}%"
    )
    return result


def _save_empty(output_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_date":   datetime.today().strftime("%Y-%m-%d"),
        "initial_capital":  INITIAL_CAPITAL,
        "period_start":     None,
        "period_end":       None,
        "trading_days":     0,
        "total_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe":           0.0,
        "win_days":         0,
        "lose_days":        0,
        "flat_days":        0,
        "avg_deployed_pct": 0.0,
        "best_day":         {"date": "", "pct": 0.0},
        "worst_day":        {"date": "", "pct": 0.0},
        "equity_curve":     [],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
