"""
Trading Signal Scanner — MarketMamba V6.1
=========================================
Reads inference results + historical data to generate actionable BUY / SELL / WATCH signals.

Entry conditions (2/4 triggers BUY in normal market, 3/4 in cautious):
  1. Rank Stability  — Top 10 ≥2 days OR Top 50 ≥3 days
  2. High Confidence — Uncertainty < 0.02 (MC-Dropout)
  3. Relative Low    — RSI(14) < 40 OR Price < MA(20)
  4. Institutional Buy — Net foreign buy ≥2 consecutive days

Exit conditions (any triggers EXIT):
  - Rank drops out of Top 50 for 2 consecutive days
  - Foreign institutional selling ≥3 consecutive days

Market regime filter:
  - TWII > MA(60) → Normal (2/4 entry)
  - TWII < MA(60) → Cautious (3/4 entry)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # V6/
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR.parent / "Data" / "processed_v6"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _load_history_index() -> list:
    """Load history_index.json → list of daily entries."""
    path = RESULTS_DIR / "history_index.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("history", [])
    except Exception:
        return []


def _get_rank_history(history: list, ticker: str, n_days: int = 5) -> list[dict]:
    """Return the rank history of a ticker over the last n_days.
    Each entry: {"date": ..., "rank": int | None}
    """
    results = []
    for entry in history[:n_days]:
        rank = None
        for p in entry.get("portfolio", []):
            if p.get("ticker") == ticker:
                rank = p.get("rank")
                break
        results.append({"date": entry.get("date"), "rank": rank})
    return results


def _check_rank_stability(rank_history: list) -> dict:
    """Check if ticker has rank stability: Top10 ≥2d OR Top50 ≥3d."""
    # Count consecutive days in top 10 (starting from most recent)
    top10_streak = 0
    for rh in rank_history:
        if rh["rank"] is not None and rh["rank"] <= 10:
            top10_streak += 1
        else:
            break

    # Count consecutive days in top 50
    top50_streak = 0
    for rh in rank_history:
        if rh["rank"] is not None and rh["rank"] <= 50:
            top50_streak += 1
        else:
            break

    met = top10_streak >= 2 or top50_streak >= 3

    if top10_streak >= 2:
        detail = f"Top 10 連續 {top10_streak} 天"
    elif top50_streak >= 3:
        detail = f"Top 50 連續 {top50_streak} 天"
    elif top50_streak > 0:
        detail = f"Top 50 {top50_streak} 天 (需 ≥3)"
    else:
        detail = "未進入 Top 50"

    return {"met": met, "detail": detail, "top10_streak": top10_streak, "top50_streak": top50_streak}


def _check_confidence(uncertainty: float, threshold: float = 0.02) -> dict:
    """High confidence = uncertainty < threshold (dynamic: 30th-percentile of today's df_kelly)."""
    met = uncertainty < threshold
    return {
        "met": met,
        "detail": f"Uncertainty={uncertainty:.4f}" + (f" ✓ (<{threshold:.4f})" if met else f" (需 <{threshold:.4f})")
    }


def _check_relative_low(prices_df: pd.DataFrame, ticker: str) -> dict:
    """Check RSI(14) < 40 OR Price < MA(20)."""
    # Normalize ticker for price lookup
    sid = ticker.lstrip("0") if ticker.isdigit() else ticker

    stock_prices = prices_df[prices_df["stock_id"] == ticker]
    if stock_prices.empty:
        stock_prices = prices_df[prices_df["stock_id"] == sid]
    if stock_prices.empty:
        return {"met": False, "detail": "無價格資料", "rsi": None, "below_ma20": None}

    stock_prices = stock_prices.sort_values("Date").tail(60)
    closes = stock_prices["Close"]

    if len(closes) < 20:
        return {"met": False, "detail": "資料不足", "rsi": None, "below_ma20": None}

    rsi = _compute_rsi(closes).iloc[-1]
    ma20 = closes.rolling(20).mean().iloc[-1]
    current = closes.iloc[-1]
    below_ma20 = current < ma20

    met = (not np.isnan(rsi) and rsi < 40) or below_ma20

    parts = []
    if not np.isnan(rsi):
        parts.append(f"RSI={rsi:.0f}")
    if below_ma20:
        parts.append(f"價格<MA20 ({current:.1f}<{ma20:.1f})")
    else:
        parts.append(f"價格>MA20")

    return {"met": met, "detail": ", ".join(parts), "rsi": round(float(rsi), 1) if not np.isnan(rsi) else None, "below_ma20": below_ma20}


def _check_institutional(inst_df: pd.DataFrame, ticker: str) -> dict:
    """Check foreign institutional net buy ≥2 consecutive days."""
    if inst_df is None or inst_df.empty:
        return {"met": False, "detail": "無機構資料"}

    sid = ticker
    stock_inst = inst_df[inst_df["stock_id"] == sid]
    if stock_inst.empty:
        return {"met": False, "detail": "無此股機構資料"}

    stock_inst = stock_inst.sort_values("Date").tail(5)

    # Look for net buy column
    buy_col = None
    for c in stock_inst.columns:
        cl = c.lower()
        if "foreign" in cl and ("buy" in cl or "net" in cl):
            buy_col = c
            break
    if buy_col is None:
        for c in stock_inst.columns:
            if "buy" in c.lower():
                buy_col = c
                break

    if buy_col is None:
        return {"met": False, "detail": "無淨買超欄位"}

    # Count consecutive days of net buy from most recent
    streak = 0
    for val in reversed(stock_inst[buy_col].values):
        try:
            if float(val) > 0:
                streak += 1
            else:
                break
        except (ValueError, TypeError):
            break

    met = streak >= 2
    return {
        "met": met,
        "detail": f"外資連買 {streak} 天" if streak > 0 else "外資未連續買入"
    }


def _check_exit_rank(rank_history: list) -> dict:
    """Exit if rank drops out of Top 50 for 2 consecutive days."""
    out_of_50_streak = 0
    for rh in rank_history:
        if rh["rank"] is None or rh["rank"] > 50:
            out_of_50_streak += 1
        else:
            break

    met = out_of_50_streak >= 2
    return {
        "met": met,
        "detail": f"連續 {out_of_50_streak} 天在 Top 50 外" if out_of_50_streak > 0 else "仍在 Top 50 內"
    }


def _get_market_regime(prices_df: pd.DataFrame) -> dict:
    """Determine market regime from TWII vs 60-day MA."""
    # Try to find TWII data
    twii = prices_df[prices_df["stock_id"].isin(["TAIEX", "^TWII", "0000"])]
    if twii.empty:
        # Try macro data
        return {"regime": "NORMAL", "twii_vs_ma60": "N/A", "threshold": 2}

    twii = twii.sort_values("Date").tail(80)
    if len(twii) < 60:
        return {"regime": "NORMAL", "twii_vs_ma60": "N/A", "threshold": 2}

    closes = twii["Close"]
    ma60 = closes.rolling(60).mean().iloc[-1]
    current = closes.iloc[-1]
    pct = (current / ma60 - 1) * 100

    regime = "NORMAL" if current > ma60 else "CAUTIOUS"
    threshold = 2 if regime == "NORMAL" else 3

    return {
        "regime": regime,
        "twii_vs_ma60": f"{pct:+.1f}%",
        "twii_current": round(float(current), 0),
        "twii_ma60": round(float(ma60), 0),
        "threshold": threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Scanner
# ═══════════════════════════════════════════════════════════════════════════════

def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def run_scan(
    df_kelly_path: Optional[Path] = None,
    portfolio_positions: Optional[list] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run the signal scanner and output action_signals.json.

    Args:
        df_kelly_path: Path to df_kelly.csv (default: V6/results/df_kelly.csv)
        portfolio_positions: List of held stocks [{"ticker": "2330", ...}]
        output_path: Where to save output (default: V6/results/action_signals.json)

    Returns:
        dict with buy_signals, exit_signals, watch_list
    """
    if df_kelly_path is None:
        df_kelly_path = RESULTS_DIR / "df_kelly.csv"
    if output_path is None:
        output_path = RESULTS_DIR / "action_signals.json"

    logger.info("🎯 Signal Scanner starting...")

    # ── Load data ─────────────────────────────────────────────────────────
    df_kelly = pd.read_csv(df_kelly_path)
    history = _load_history_index()
    date_str = str(df_kelly["Date"].iloc[0]) if "Date" in df_kelly.columns else "unknown"

    # I1: Dynamic uncertainty threshold = 30th percentile of today's cross-section
    unc_q30 = float(df_kelly["Uncertainty"].quantile(0.30)) if "Uncertainty" in df_kelly.columns else 0.02
    logger.info(f"  Uncertainty Q30 threshold: {unc_q30:.4f}")

    # Price data (trimmed to recent)
    prices_path = DATA_DIR / "prices_raw.parquet"
    if prices_path.exists():
        prices_df = pd.read_parquet(prices_path)
        prices_df["Date"] = pd.to_datetime(prices_df["Date"])
        cutoff = prices_df["Date"].max() - pd.Timedelta(days=120)
        prices_df = prices_df[prices_df["Date"] >= cutoff]
    else:
        prices_df = pd.DataFrame()

    # Institutional data
    inst_path = DATA_DIR / "institutional_raw.parquet"
    if inst_path.exists():
        inst_df = pd.read_parquet(inst_path)
        inst_df["Date"] = pd.to_datetime(inst_df["Date"])
        cutoff_inst = inst_df["Date"].max() - pd.Timedelta(days=14)
        inst_df = inst_df[inst_df["Date"] >= cutoff_inst]
    else:
        inst_df = pd.DataFrame()

    # Market regime
    market = _get_market_regime(prices_df)
    entry_threshold = market["threshold"]
    logger.info(f"  Market regime: {market['regime']} (TWII vs MA60: {market['twii_vs_ma60']}) → entry ≥{entry_threshold}/4")

    # ── Scan Top 50 for entry signals ─────────────────────────────────────
    top50 = df_kelly.head(50)
    buy_signals = []
    watch_list = []

    # Weighted score weights (for display/reference only — BUY/WATCH uses conditions_met)
    _W_RANK  = 30
    _W_CONF  = 25
    _W_INST  = 25
    _W_LOW   = 20

    for _, row in top50.iterrows():
        ticker = str(row["Ticker"])
        uncertainty = float(row.get("Uncertainty", 1.0))
        alpha_20d = float(row.get("Exp_Alpha_20d", 0))
        sharpe = float(row.get("Signal_Quality", 0))
        confidence = str(row.get("Confidence", ""))
        weight = float(row.get("Suggested_Weight", 0))

        # Skip if alpha is negative
        if alpha_20d <= 0:
            continue

        rank_hist = _get_rank_history(history, ticker)
        c1 = _check_rank_stability(rank_hist)
        c2 = _check_confidence(uncertainty, unc_q30)
        c3 = _check_relative_low(prices_df, ticker)
        c4 = _check_institutional(inst_df, ticker)

        conditions_met = sum([c1["met"], c2["met"], c3["met"], c4["met"]])
        # Score kept for display purposes (shows relative strength within same conditions_met tier)
        score = _W_RANK * c1["met"] + _W_CONF * c2["met"] + _W_INST * c4["met"] + _W_LOW * c3["met"]

        signal_entry = {
            "ticker": ticker,
            "alpha_20d": round(alpha_20d, 4),
            "sharpe": round(sharpe, 3),
            "confidence": confidence,
            "uncertainty": round(uncertainty, 4),
            "suggested_weight": round(weight, 4),
            "score": score,
            "conditions_met": conditions_met,
            "conditions_total": 4,
            "rank_stability": c1,
            "high_confidence": c2,
            "relative_low": c3,
            "institutional_buy": c4,
        }

        if conditions_met >= entry_threshold:
            buy_signals.append(signal_entry)
            logger.info(f"  🔥 BUY: {ticker} (conditions={conditions_met}/4, score={score})")
        elif conditions_met >= 1:
            watch_list.append(signal_entry)

    # ── Check exit signals for current holdings ───────────────────────────
    exit_signals = []
    if portfolio_positions:
        for pos in portfolio_positions:
            ticker = pos.get("ticker") or pos.get("stock_id", "")
            if not ticker:
                continue

            rank_hist = _get_rank_history(history, ticker)
            exit_rank = _check_exit_rank(rank_hist)
            exit_inst = _check_institutional(inst_df, ticker)

            reasons = []
            if exit_rank["met"]:
                reasons.append("排名連續掉出 Top 50")
            # Reverse institutional check: selling ≥3 days
            # (reuse inst check but look for negative streaks)

            if reasons:
                exit_signals.append({
                    "ticker": ticker,
                    "reasons": reasons,
                    "rank_exit": exit_rank,
                    "current_rank": rank_hist[0]["rank"] if rank_hist else None,
                })
                logger.info(f"  🔴 EXIT: {ticker} — {', '.join(reasons)}")

    # ── Build output ──────────────────────────────────────────────────────
    # I3: Portfolio-level risk limits
    _MAX_POSITIONS = 15
    _MIN_POSITIONS = 5
    _MAX_WEIGHT    = 0.10   # 10% single-stock cap

    buy_signals = sorted(buy_signals, key=lambda x: x.get("score", 0), reverse=True)
    if len(buy_signals) > _MAX_POSITIONS:
        logger.info(f"  Portfolio cap: trimmed {len(buy_signals)} → {_MAX_POSITIONS} positions")
        buy_signals = buy_signals[:_MAX_POSITIONS]

    for sig in buy_signals:
        if sig.get("suggested_weight", 0) > _MAX_WEIGHT:
            sig["suggested_weight"] = _MAX_WEIGHT
    total_w = sum(s.get("suggested_weight", 0) for s in buy_signals)
    if total_w > 0:
        for sig in buy_signals:
            sig["suggested_weight"] = round(sig["suggested_weight"] / total_w, 4)

    portfolio_check = {
        "n_positions": len(buy_signals),
        "warnings": ([] if len(buy_signals) >= _MIN_POSITIONS else
                     [f"持倉不足 {_MIN_POSITIONS} 檔（{len(buy_signals)} 檔），建議觀望"]),
    }

    result = {
        "date": date_str,
        "scan_version": "1.2",
        "market_regime": market["regime"],
        "twii_vs_ma60": market.get("twii_vs_ma60", "N/A"),
        "entry_threshold": f"≥{entry_threshold}/4條件",
        "uncertainty_threshold": round(unc_q30, 6),
        "total_scanned": len(top50),
        "buy_signals": buy_signals,
        "exit_signals": exit_signals,
        "watch_list": watch_list,
        "portfolio_check": portfolio_check,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info(f"  Scanner output: {len(buy_signals)} BUY, {len(exit_signals)} EXIT, {len(watch_list)} WATCH")
    logger.info(f"  Saved → {output_path}")

    return result
