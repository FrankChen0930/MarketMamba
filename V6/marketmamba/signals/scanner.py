"""
Trading Signal Scanner — MarketMamba V6.2
=========================================
Reads inference results + historical data to generate actionable BUY / WATCH signals.

Entry scoring (v1.4 — unified with signal_conditions.compute_entry_score, same as
sim_engine_v3 / portfolio_checker; BUY = composite score ≥70 normal / ≥90 cautious):
  1. Rank Stability    (30) — Top 10 ≥2 days OR Top 50 ≥3 days
  2. High Confidence   (25) — Uncertainty < Q30 percentile (MC-Dropout)
  3. Institutional Buy (25) — Net foreign buy ≥2 consecutive days
  4. Relative Low      (20) — RSI(14) < 40 OR Price < MA(20)

Pattern bonus (added on top of base 100-pt score, max total 150):
  - pattern_score 60–74 → +20
  - pattern_score 75–89 → +30
  - pattern_score ≥90   → +40
  - dual_confirm (pattern ≥60 AND alpha_rank ≤200) → extra +10

Exit signals: NOT produced here. Full four-layer exit logic lives in
signal_conditions.check_exit_conditions() (used by sim_engine_v3 and
portfolio_checker → portfolio_exit_check.json). `exit_signals` in the output
JSON is kept as an empty list for backward compatibility.

Market regime filter:
  - TWII > MA(60) → Normal   (score ≥70)
  - TWII < MA(60) → Cautious (score ≥90)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from marketmamba.signals.signal_conditions import (
    DUAL_CONFIRM_RANK,
    ENTRY_THRESHOLD_CAUTIOUS,
    ENTRY_THRESHOLD_NORMAL,
    W_HIGH_CONFIDENCE,
    W_INSTITUTIONAL,
    W_RANK_STABILITY,
    W_RELATIVE_LOW,
    compute_entry_score,
)

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

    # 「外資連買」必須用淨買超（Foreign_Net = 買-賣）判斷。
    # 舊版名稱啟發式會先抓到 Foreign_Buy（總買進），大型股天天 >0 → 條件被灌水。
    if "Foreign_Net" in stock_inst.columns:
        net_values = stock_inst["Foreign_Net"].values
    elif "Foreign_Buy" in stock_inst.columns and "Foreign_Sell" in stock_inst.columns:
        net_values = (stock_inst["Foreign_Buy"].fillna(0) - stock_inst["Foreign_Sell"].fillna(0)).values
    else:
        return {"met": False, "detail": "無淨買超欄位"}

    # Count consecutive days of net buy from most recent
    streak = 0
    for val in reversed(net_values):
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


def _load_pattern_lookup() -> dict:
    """Load pattern_signals.json and build {stock_id: best_bullish_signal} lookup.

    Returns dict mapping ticker → pattern signal dict (with score, pattern_id,
    pattern_name, failure_stop, dual_confirm).  Empty dict if file missing.
    """
    path = RESULTS_DIR / "pattern_signals.json"
    if not path.exists():
        logger.warning("  pattern_signals.json not found — pattern bonus disabled")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        lookup: dict = {}
        for sig in data.get("signals", []):
            sid = str(sig.get("stock_id", ""))
            if not sid:
                continue
            # Keep the highest-score pattern per stock (list is already sorted desc)
            if sid not in lookup:
                lookup[sid] = sig
        logger.info(f"  Pattern lookup: {len(lookup)} stocks with bullish patterns loaded")
        return lookup
    except Exception as e:
        logger.warning(f"  Failed to load pattern_signals.json: {e}")
        return {}


def _apply_pattern_bonus(base_score: int, pattern_signal: dict | None, alpha_rank: int) -> tuple[int, dict | None]:
    """Composite score via the shared signal_conditions.compute_entry_score().

    Returns (composite_score, pattern_info_dict_or_None).
    pattern_info contains: pattern_score, pattern_id, pattern_name,
                            failure_stop, dual_confirm, pattern_bonus
                            (bonus shown includes the dual-confirm +10).
    """
    p_score = int(pattern_signal.get("score", 0)) if pattern_signal else None
    dual = p_score is not None and p_score >= 60 and alpha_rank <= DUAL_CONFIRM_RANK
    composite, breakdown = compute_entry_score(base_score, p_score, alpha_rank, dual)

    bonus_total = breakdown["pattern_bonus"] + breakdown["dual_confirm_bonus"]
    if pattern_signal is None or bonus_total == 0:
        return composite, None

    pattern_info = {
        "pattern_score":   p_score,
        "pattern_id":      str(pattern_signal.get("pattern_id", "")),
        "pattern_name":    str(pattern_signal.get("pattern_name", "")),
        "failure_stop":    pattern_signal.get("failure_stop"),
        "target_price":    pattern_signal.get("target_price"),
        "dual_confirm":    dual,
        "pattern_bonus":   bonus_total,
    }
    return composite, pattern_info


def _load_twii_from_macro(prices_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Fallback TWII source: macro_raw.parquet (TWII_Close).

    prices_raw only contains 4-digit stock ids (ticker universe filter), so the
    index never appears there and the regime gate silently stayed NORMAL.
    Freshness guard: macro must reach within 10 days of prices_raw's max date,
    otherwise a stale close vs MA60 would give a silently wrong regime.
    """
    macro_path = DATA_DIR / "macro_raw.parquet"
    if not macro_path.exists():
        return None
    try:
        macro = pd.read_parquet(macro_path, columns=["Date", "TWII_Close"]).dropna()
    except Exception as e:
        logger.warning(f"  macro_raw.parquet unreadable for regime check: {e}")
        return None
    if macro.empty:
        return None
    macro["Date"] = pd.to_datetime(macro["Date"])
    macro_max = macro["Date"].max()
    ref_max = prices_df["Date"].max() if not prices_df.empty else pd.Timestamp.today()
    lag_days = (ref_max - macro_max).days
    if lag_days > 10:
        logger.warning(
            f"  TWII regime check skipped: macro_raw stale "
            f"(last {macro_max.date()}, {lag_days} days behind prices) → regime=NORMAL"
        )
        return None
    return macro.rename(columns={"TWII_Close": "Close"})


def _get_market_regime(prices_df: pd.DataFrame) -> dict:
    """Determine market regime from TWII vs 60-day MA.

    threshold = BUY score threshold (shared with signal_conditions.entry_threshold):
    NORMAL → ≥70, CAUTIOUS → ≥90.
    """
    # Try to find TWII data in prices_raw, then fall back to macro_raw
    twii = prices_df[prices_df["stock_id"].isin(["TAIEX", "^TWII", "0000"])]
    if twii.empty:
        twii = _load_twii_from_macro(prices_df)
    if twii is None or twii.empty:
        return {"regime": "NORMAL", "twii_vs_ma60": "N/A", "threshold": ENTRY_THRESHOLD_NORMAL}

    twii = twii.sort_values("Date").tail(80)
    if len(twii) < 60:
        return {"regime": "NORMAL", "twii_vs_ma60": "N/A", "threshold": ENTRY_THRESHOLD_NORMAL}

    closes = twii["Close"]
    ma60 = closes.rolling(60).mean().iloc[-1]
    current = closes.iloc[-1]
    pct = (current / ma60 - 1) * 100

    regime = "NORMAL" if current > ma60 else "CAUTIOUS"
    threshold = ENTRY_THRESHOLD_NORMAL if regime == "NORMAL" else ENTRY_THRESHOLD_CAUTIOUS

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
    output_path: Optional[Path] = None,
) -> dict:
    """
    Run the signal scanner and output action_signals.json.

    Args:
        df_kelly_path: Path to df_kelly.csv (default: V6/results/df_kelly.csv)
        output_path: Where to save output (default: V6/results/action_signals.json)

    Returns:
        dict with buy_signals, watch_list (exit_signals kept as [] for compat;
        real exit logic = signal_conditions four-layer / portfolio_exit_check.json)
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

    # Pattern lookup (V6.2): {stock_id → best bullish pattern signal}
    pattern_lookup = _load_pattern_lookup()

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

    # Market regime → BUY score threshold (70 normal / 90 cautious)
    market = _get_market_regime(prices_df)
    score_threshold = market["threshold"]
    logger.info(f"  Market regime: {market['regime']} (TWII vs MA60: {market['twii_vs_ma60']}) → entry score ≥{score_threshold}")

    # ── Scan Top 50 for entry signals ─────────────────────────────────────
    top50 = df_kelly.head(50).reset_index(drop=True)
    buy_signals = []
    watch_list = []

    for idx, row in top50.iterrows():
        ticker = str(row["Ticker"])
        uncertainty = float(row.get("Uncertainty", 1.0))
        alpha_5d  = float(row.get("Exp_Alpha_5d",  0))
        alpha_20d = float(row.get("Exp_Alpha_20d", 0))
        alpha_60d = float(row.get("Exp_Alpha_60d", 0))
        sharpe = float(row.get("Signal_Quality", 0))
        confidence = str(row.get("Confidence", ""))
        weight = float(row.get("Suggested_Weight", 0))
        alpha_rank = int(idx) + 1   # rank within df_kelly (1-indexed)

        # Skip if alpha is negative
        if alpha_20d <= 0:
            continue

        rank_hist = _get_rank_history(history, ticker)
        c1 = _check_rank_stability(rank_hist)
        c2 = _check_confidence(uncertainty, unc_q30)
        c3 = _check_relative_low(prices_df, ticker)
        c4 = _check_institutional(inst_df, ticker)

        conditions_met = sum([c1["met"], c2["met"], c3["met"], c4["met"]])

        # Base score (max 100) — weights shared with signal_conditions
        base_score = (W_RANK_STABILITY  * c1["met"] + W_HIGH_CONFIDENCE * c2["met"]
                      + W_INSTITUTIONAL * c4["met"] + W_RELATIVE_LOW    * c3["met"])

        # V6.2: apply pattern bonus → composite score (max 150)
        p_signal = pattern_lookup.get(ticker)
        composite_score, pattern_info = _apply_pattern_bonus(base_score, p_signal, alpha_rank)

        signal_entry = {
            "ticker": ticker,
            "alpha_5d":  round(alpha_5d,  4),
            "alpha_20d": round(alpha_20d, 4),
            "alpha_60d": round(alpha_60d, 4),
            "sharpe": round(sharpe, 3),
            "confidence": confidence,
            "uncertainty": round(uncertainty, 4),
            "suggested_weight": round(weight, 4),
            "score": composite_score,
            "base_score": base_score,
            "conditions_met": conditions_met,
            "conditions_total": 4,
            "rank_stability": c1,
            "high_confidence": c2,
            "relative_low": c3,
            "institutional_buy": c4,
            # Pattern fields (None if no qualifying pattern)
            "pattern": pattern_info,
        }

        if composite_score >= score_threshold:
            buy_signals.append(signal_entry)
            pat_str = f", pattern={pattern_info['pattern_name']}({pattern_info['pattern_score']}) +{pattern_info['pattern_bonus']}" if pattern_info else ""
            logger.info(f"  🔥 BUY: {ticker} (score={composite_score}≥{score_threshold}, conditions={conditions_met}/4{pat_str})")
        elif conditions_met >= 1:
            watch_list.append(signal_entry)

    # Exit signals: intentionally always empty — real exit logic is the
    # four-layer check in signal_conditions (→ portfolio_exit_check.json).
    exit_signals = []

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
        "scan_version": "1.4",
        "market_regime": market["regime"],
        "twii_vs_ma60": market.get("twii_vs_ma60", "N/A"),
        "entry_threshold": f"≥{score_threshold}分",
        "entry_threshold_score": score_threshold,
        "uncertainty_threshold": round(unc_q30, 6),
        "total_scanned": len(top50),
        "buy_signals": buy_signals,
        "exit_signals": exit_signals,
        "watch_list": watch_list,
        "portfolio_check": portfolio_check,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
    pattern_count = sum(1 for s in buy_signals + watch_list if s.get("pattern"))
    logger.info(f"  Scanner output: {len(buy_signals)} BUY, {len(exit_signals)} EXIT, {len(watch_list)} WATCH, {pattern_count} with pattern bonus")
    logger.info(f"  Saved → {output_path}")

    return result
