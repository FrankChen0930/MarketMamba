"""
MarketMamba V6 — Traditional Chart Pattern Scanner
====================================================
Daily full-market scan (~2,500 stocks) for bullish chart patterns.
Patterns: W底, 彈簧型W底, 頭肩底, 三角收斂
Uses already-cached parquet files — zero new network calls.

Output: V6/results/pattern_signals.json
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from marketmamba.config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

_CACHE_DIR = DATA_DIR / "cache_v6"
PATTERN_SIGNALS_PATH = RESULTS_DIR / "pattern_signals.json"

# ── Timeframes ────────────────────────────────────────────────────────────────
TIMEFRAMES = [
    {"id": "short",  "label": "短線 約2週",   "days":  30, "order": 3},
    {"id": "medium", "label": "中線 約1個月",  "days":  60, "order": 8},
    {"id": "long",   "label": "長線 約3個月",  "days": 120, "order": 15},
    {"id": "season", "label": "季線 約半年",   "days": 180, "order": 30},
]

SCORE_THRESHOLD      = 60   # minimum score to emit a signal
DUAL_CONFIRM_RANK    = 200  # alpha rank ≤ this → dual_confirm = True


# ════════════════════════════════════════════════════════════════════════════════
# Local Extrema Detection  (pure numpy, no scipy)
# ════════════════════════════════════════════════════════════════════════════════

def _find_local_extrema_np(arr: np.ndarray, order: int = 5):
    """
    Find local minima and maxima with a rolling window of size (2*order+1).

    A point i is a local minimum  if arr[i] ≤ every value in arr[i-order:i+order+1].
    A point i is a local maximum  if arr[i] ≥ every value in arr[i-order:i+order+1].
    Consecutive plateaus of equal values yield only the first index.

    Returns
    -------
    (min_indices, max_indices) : tuple of integer numpy arrays
    """
    n = len(arr)
    if n < 2 * order + 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    min_list: list[int] = []
    max_list: list[int] = []
    prev_min = prev_max = False

    for i in range(order, n - order):
        window = arr[i - order: i + order + 1]
        v = arr[i]
        is_min = bool(v <= window.min())
        is_max = bool(v >= window.max())
        if is_min and not prev_min:
            min_list.append(i)
        if is_max and not prev_max:
            max_list.append(i)
        prev_min = is_min
        prev_max = is_max

    return np.array(min_list, dtype=int), np.array(max_list, dtype=int)


# ════════════════════════════════════════════════════════════════════════════════
# Technical Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _rsi_last(close: np.ndarray, period: int = 14) -> float:
    """Wilder's RSI at the last bar. Returns 50.0 if insufficient data."""
    if len(close) < period + 2:
        return 50.0
    delta = np.diff(close.astype(float))
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_g = float(gain[:period].mean())
    avg_l = float(loss[:period].mean())
    for g, l in zip(gain[period:], loss[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    rs = avg_g / (avg_l + 1e-10)
    return float(100 - 100 / (1 + rs))


def _vol_ratio(volume: Optional[np.ndarray], idx: int, window: int = 5) -> float:
    """Volume at *idx* relative to the mean of the prior *window* bars."""
    if volume is None or idx < window or idx >= len(volume):
        return 1.0
    past_avg = float(volume[max(0, idx - window): idx].mean())
    return float(volume[idx]) / (past_avg + 1e-10) if past_avg > 0 else 1.0


def _rr_str(target: float, current: float, stop: float) -> str:
    """Format risk-reward as '1:x.x'. Returns 'N/A' if geometry is invalid."""
    if current > stop and target > current:
        rr = (target - current) / (current - stop)
        return f"1:{rr:.1f}"
    return "N/A"


# ════════════════════════════════════════════════════════════════════════════════
# Pattern Detectors
# Each returns a dict {score, key_price, target_price, stop_loss, risk_reward}
# or None if the pattern is not present / score below threshold.
# ════════════════════════════════════════════════════════════════════════════════

def _detect_w_bottom(
    close: np.ndarray,
    volume: Optional[np.ndarray],
    min_idx: np.ndarray,
    max_idx: np.ndarray,
) -> Optional[dict]:
    """
    W底（雙底）
    ───────────
    • Two local lows within 5% of each other.
    • Intervening neckline ≥ 3% above the average of both lows.
    • Current price must be ≥ 95% of the neckline (near or above breakout).

    Scoring (max 100):
      Formation  40 — how similar the two lows are (0% spread = 40 pts)
      Volume     30 — volume surge at the second low (vs prior 5 bars)
      Position   20 — current price vs neckline
      RSI        10 — RSI ≥ 50 is bullish momentum confirmation
    """
    if len(min_idx) < 2:
        return None

    best: Optional[dict] = None

    for i in range(len(min_idx) - 1):
        l1_i, l2_i = int(min_idx[i]), int(min_idx[i + 1])
        l1, l2 = float(close[l1_i]), float(close[l2_i])

        spread = abs(l1 - l2) / max(l1, l2)
        if spread > 0.05:
            continue

        # Neckline = highest close between the two lows
        between = max_idx[(max_idx > l1_i) & (max_idx < l2_i)]
        if len(between) > 0:
            neck = float(close[between[np.argmax(close[between])]])
        else:
            neck = float(close[l1_i: l2_i + 1].max())

        avg_low = (l1 + l2) / 2.0
        if (neck - avg_low) / avg_low < 0.03:
            continue

        current = float(close[-1])
        if current < neck * 0.95:
            continue

        # ── Score ──
        form  = int(40 * max(0.0, 1.0 - spread / 0.05))
        vr    = _vol_ratio(volume, l2_i)
        vol_s = min(30, int(30 * min(vr / 1.5, 1.0))) if volume is not None else 18
        pos_p = (current - neck) / neck
        pos_s = (20 if pos_p >= 0.0
                 else 15 if pos_p >= -0.02
                 else 8  if pos_p >= -0.04
                 else 0)
        rsi   = _rsi_last(close)
        rsi_s = 10 if rsi >= 50 else (7 if rsi >= 45 else 3)

        score = form + vol_s + pos_s + rsi_s

        target = neck + (neck - avg_low)
        stop   = min(l1, l2) * 0.97
        entry  = {
            "score": score,
            "key_price":    round(neck, 2),
            "target_price": round(target, 2),
            "stop_loss":    round(stop, 2),
            "risk_reward":  _rr_str(target, current, stop),
        }
        if best is None or score > best["score"]:
            best = entry

    return best if (best and best["score"] >= SCORE_THRESHOLD) else None


def _detect_spring_w(
    close: np.ndarray,
    volume: Optional[np.ndarray],
    min_idx: np.ndarray,
    max_idx: np.ndarray,
) -> Optional[dict]:
    """
    彈簧型W底
    ──────────
    • Second low is 0.5%–2% below the first (bullish bear-trap / spring).
    • Gap between the two lows: 5–40 bars (not too fast, not too slow).
    • After the second low, price must recover above the first low.
    • Current price ≥ 94% of the neckline.

    Volume ideal: capitulation spike at the second low (high fear = good entry).
    """
    if len(min_idx) < 2:
        return None

    best: Optional[dict] = None

    for i in range(len(min_idx) - 1):
        l1_i, l2_i = int(min_idx[i]), int(min_idx[i + 1])
        l1, l2 = float(close[l1_i]), float(close[l2_i])

        dip = (l1 - l2) / l1          # positive when l2 < l1
        if not (0.005 <= dip <= 0.025):
            continue

        gap = l2_i - l1_i
        if gap < 5 or gap > 40:
            continue

        # Post-l2 recovery must exceed l1
        post = close[l2_i:]
        if len(post) < 3 or float(post.max()) <= l1:
            continue

        # Neckline between the two lows
        between = max_idx[(max_idx > l1_i) & (max_idx < l2_i)]
        neck = (float(close[between[np.argmax(close[between])]])
                if len(between) > 0
                else float(close[l1_i: l2_i + 1].max()))

        if (neck - l1) / l1 < 0.02:
            continue

        current = float(close[-1])
        if current < neck * 0.94:
            continue

        # ── Score ──
        ideal_dev = abs(dip - 0.012) / 0.012   # 1.2% spring = ideal
        form  = int(40 * max(0.0, 1.0 - min(ideal_dev, 1.0)))
        vr    = _vol_ratio(volume, l2_i)
        vol_s = min(30, int(30 * min(vr / 2.0, 1.0))) if volume is not None else 18
        pos_p = (current - neck) / neck
        pos_s = 20 if pos_p >= 0.0 else (15 if pos_p >= -0.02 else 5)
        rsi   = _rsi_last(close)
        rsi_s = 10 if rsi >= 50 else (7 if rsi >= 45 else 3)

        score = form + vol_s + pos_s + rsi_s

        target = neck + (neck - l2)
        stop   = l2 * 0.97
        entry  = {
            "score": score,
            "key_price":    round(neck, 2),
            "target_price": round(target, 2),
            "stop_loss":    round(stop, 2),
            "risk_reward":  _rr_str(target, current, stop),
        }
        if best is None or score > best["score"]:
            best = entry

    return best if (best and best["score"] >= SCORE_THRESHOLD) else None


def _detect_hs_bottom(
    close: np.ndarray,
    volume: Optional[np.ndarray],
    min_idx: np.ndarray,
    max_idx: np.ndarray,
) -> Optional[dict]:
    """
    頭肩底（倒頭肩）
    ────────────────
    • Three consecutive local lows: left shoulder, head (lowest), right shoulder.
    • Shoulders within 10% of each other.
    • Head ≥ 3% below average shoulder.
    • Neckline = average of the peaks between ls–h and h–rs.
    • Current price ≥ 94% of neckline.

    Volume ideal: right shoulder formed on lower volume (quiet accumulation).
    """
    if len(min_idx) < 3:
        return None

    best: Optional[dict] = None

    for i in range(len(min_idx) - 2):
        ls_i = int(min_idx[i])
        h_i  = int(min_idx[i + 1])
        rs_i = int(min_idx[i + 2])
        ls, hd, rs = float(close[ls_i]), float(close[h_i]), float(close[rs_i])

        if not (hd < ls and hd < rs):
            continue

        sh_diff = abs(ls - rs) / max(ls, rs)
        if sh_diff > 0.10:
            continue

        avg_sh = (ls + rs) / 2.0
        if (avg_sh - hd) / avg_sh < 0.03:
            continue

        lp_idx = max_idx[(max_idx > ls_i) & (max_idx < h_i)]
        rp_idx = max_idx[(max_idx > h_i)  & (max_idx < rs_i)]
        if len(lp_idx) == 0 or len(rp_idx) == 0:
            continue

        lp = float(close[lp_idx[np.argmax(close[lp_idx])]])
        rp = float(close[rp_idx[np.argmax(close[rp_idx])]])
        neckline = (lp + rp) / 2.0

        current = float(close[-1])
        if current < neckline * 0.94:
            continue

        # ── Score ──
        form  = int(40 * max(0.0, 1.0 - sh_diff / 0.10))
        vr    = _vol_ratio(volume, rs_i)
        # Right-shoulder: prefer lighter volume (quiet accumulation)
        vol_s = (min(30, int(30 * (1.5 - min(vr, 1.5))))
                 if volume is not None else 18)
        pos_p = (current - neckline) / neckline
        pos_s = (20 if pos_p >= 0.0
                 else 15 if pos_p >= -0.02
                 else 8  if pos_p >= -0.04
                 else 0)
        rsi   = _rsi_last(close)
        rsi_s = 10 if rsi >= 48 else (6 if rsi >= 42 else 2)

        score = form + vol_s + pos_s + rsi_s

        target = neckline + (neckline - hd)
        stop   = min(ls, rs) * 0.97
        entry  = {
            "score": score,
            "key_price":    round(neckline, 2),
            "target_price": round(target, 2),
            "stop_loss":    round(stop, 2),
            "risk_reward":  _rr_str(target, current, stop),
        }
        if best is None or score > best["score"]:
            best = entry

    return best if (best and best["score"] >= SCORE_THRESHOLD) else None


def _detect_triangle(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: Optional[np.ndarray],
    min_idx_l: np.ndarray,   # local minima of LOW  (support trendline)
    max_idx_h: np.ndarray,   # local maxima of HIGH (resistance trendline)
) -> Optional[dict]:
    """
    三角收斂（對稱三角）
    ────────────────────
    • Descending resistance trendline (lower highs).
    • Ascending  support   trendline (higher lows).
    • ≥ 2 touch points on each side (4 total).
    • Current price is inside the triangle.
    • Progress toward apex: 65%–90% is ideal (breakout imminent).
    • Triangle width < 25% of price.

    Volume ideal: contracting (drying up as the triangle tightens).
    Target = breakout level + triangle height at that point.
    """
    if len(max_idx_h) < 2 or len(min_idx_l) < 2:
        return None

    # Use up to last 4 swing highs and lows
    hi_idx = max_idx_h[-4:]
    lo_idx = min_idx_l[-4:]
    hi_p   = high[hi_idx].astype(float)
    lo_p   = low[lo_idx].astype(float)

    # Resistance: strictly decreasing highs
    if not all(hi_p[j] > hi_p[j + 1] for j in range(len(hi_p) - 1)):
        return None
    # Support: strictly increasing lows
    if not all(lo_p[j] < lo_p[j + 1] for j in range(len(lo_p) - 1)):
        return None

    r_t0, r_tn = int(hi_idx[0]), int(hi_idx[-1])
    s_t0, s_tn = int(lo_idx[0]), int(lo_idx[-1])
    if (r_tn - r_t0) < 1 or (s_tn - s_t0) < 1:
        return None

    r_slope = (hi_p[-1] - hi_p[0]) / (r_tn - r_t0)
    s_slope = (lo_p[-1] - lo_p[0]) / (s_tn - s_t0)

    if r_slope >= 0 or s_slope <= 0:
        return None

    r_int = float(hi_p[0]) - r_slope * r_t0
    s_int = float(lo_p[0]) - s_slope * s_t0

    if abs(r_slope - s_slope) < 1e-10:
        return None
    apex_t = (s_int - r_int) / (r_slope - s_slope)

    n = len(close) - 1          # current bar index
    r_now = r_slope * n + r_int
    s_now = s_slope * n + s_int

    if r_now <= s_now:
        return None

    current = float(close[-1])

    # Price must be inside (allow ±2% for noise)
    if not (s_now * 0.98 <= current <= r_now * 1.02):
        return None

    width_pct = (r_now - s_now) / max(s_now, 1.0)
    if width_pct > 0.25:
        return None

    # ── Score ──
    form = int(40 * max(0.0, 1.0 - width_pct / 0.20))

    if volume is not None and len(volume) >= 10:
        half    = len(volume) // 2
        early_v = float(volume[:half].mean())
        late_v  = float(volume[half:].mean())
        vr      = late_v / (early_v + 1e-10)
        vol_s   = (30 if vr < 0.70 else
                   22 if vr < 0.90 else
                   14 if vr < 1.10 else 6)
    else:
        vol_s = 18

    t0      = min(r_t0, s_t0)
    pct_apex = (n - t0) / max(apex_t - t0, 1.0)
    pos_s   = (20 if 0.65 <= pct_apex <= 0.90
               else 12 if 0.50 <= pct_apex <  0.65
               else  6 if 0.90 <  pct_apex <= 1.00
               else  2)

    rsi   = _rsi_last(close)
    rsi_s = 10 if rsi >= 45 else (6 if rsi >= 40 else 2)

    score = form + vol_s + pos_s + rsi_s

    tri_h  = r_now - s_now
    target = r_now + tri_h
    stop   = s_now * 0.97
    entry  = {
        "score": score,
        "key_price":    round(r_now, 2),      # resistance = breakout level
        "target_price": round(target, 2),
        "stop_loss":    round(stop, 2),
        "risk_reward":  _rr_str(target, current, stop),
    }
    return entry if score >= SCORE_THRESHOLD else None


# ════════════════════════════════════════════════════════════════════════════════
# Per-Stock Scanner
# ════════════════════════════════════════════════════════════════════════════════

def _scan_stock(
    stock_id: str,
    ohlcv: pd.DataFrame,
    alpha_rank: int,
    alpha_20d: float,
    confidence: str,
    name: str,
    sector: str,
) -> list[dict]:
    """
    Scan one stock across 4 timeframes × 4 patterns.
    Returns a (possibly empty) list of signal dicts.
    """
    close_all = ohlcv["Close"].values.astype(float)
    high_all  = ohlcv["High"].values.astype(float)
    low_all   = ohlcv["Low"].values.astype(float)
    vol_all   = (ohlcv["Volume"].values.astype(float)
                 if "Volume" in ohlcv.columns else None)

    # Rank bonus applied on top of raw pattern score
    alpha_bonus = 10 if alpha_rank <= 200 else (5 if alpha_rank <= 300 else 0)

    signals: list[dict] = []

    for tf in TIMEFRAMES:
        n_days = tf["days"]
        order  = tf["order"]

        if len(close_all) < n_days:
            continue

        close = close_all[-n_days:]
        high  = high_all[-n_days:]
        low   = low_all[-n_days:]
        vol   = vol_all[-n_days:] if vol_all is not None else None

        # Extrema on close (W / Spring / HS patterns)
        min_c, max_c = _find_local_extrema_np(close, order)
        # Extrema on raw high/low (triangle support & resistance lines)
        min_l, _     = _find_local_extrema_np(low,   order)
        _,     max_h = _find_local_extrema_np(high,  order)

        pattern_checks = [
            ("w_bottom",  "W底（雙底）",  _detect_w_bottom,  (close, vol, min_c, max_c)),
            ("spring_w",  "彈簧型W底",    _detect_spring_w,  (close, vol, min_c, max_c)),
            ("hs_bottom", "頭肩底",        _detect_hs_bottom, (close, vol, min_c, max_c)),
            ("triangle",  "三角收斂",      _detect_triangle,  (close, high, low, vol, min_l, max_h)),
        ]

        for pat_id, pat_name, fn, args in pattern_checks:
            try:
                result = fn(*args)
            except Exception:
                continue

            if result is None:
                continue

            final_score  = min(100, result["score"] + alpha_bonus)
            if final_score < SCORE_THRESHOLD:
                continue

            dual_confirm = (alpha_rank <= DUAL_CONFIRM_RANK)

            signals.append({
                "stock_id":        stock_id,
                "name":            name,
                "sector":          sector,
                "pattern_id":      pat_id,
                "pattern_name":    pat_name,
                "timeframe":       tf["id"],
                "timeframe_label": tf["label"],
                "score":           final_score,
                "key_price":       result["key_price"],
                "target_price":    result["target_price"],
                "stop_loss":       result["stop_loss"],
                "risk_reward":     result["risk_reward"],
                "alpha_rank":      alpha_rank,
                "alpha_20d":       round(float(alpha_20d), 4),
                "confidence":      confidence,
                "dual_confirm":    dual_confirm,
            })

    return signals


# ════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ════════════════════════════════════════════════════════════════════════════════

def run_pattern_scan(
    date_str: str,
    df_kelly_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Scan the full stock universe for bullish chart patterns.

    Parameters
    ----------
    date_str      : Target date string 'YYYY-MM-DD'
    df_kelly_path : Path to df_kelly.csv  (default: RESULTS_DIR/df_kelly.csv)
    output_path   : Where to write pattern_signals.json

    Returns
    -------
    dict   The saved payload (also written to disk as JSON).
    """
    t0 = time.time()
    if df_kelly_path is None:
        df_kelly_path = RESULTS_DIR / "df_kelly.csv"
    if output_path is None:
        output_path = PATTERN_SIGNALS_PATH

    logger.info(f"[pattern] Starting scan for {date_str}")

    # ── Load Alpha Rankings ────────────────────────────────────────────────────
    if not Path(df_kelly_path).exists():
        logger.error(f"[pattern] df_kelly.csv not found: {df_kelly_path}")
        return {}

    kelly = pd.read_csv(df_kelly_path)

    ticker_col = "Ticker" if "Ticker" in kelly.columns else "stock_id"
    alpha_col  = "Exp_Alpha_20d" if "Exp_Alpha_20d" in kelly.columns else "alpha_20d"

    if "rank" not in kelly.columns:
        kelly = kelly.sort_values(alpha_col, ascending=False,
                                  na_position="last").reset_index(drop=True)
        kelly["rank"] = kelly.index + 1

    kelly_map: dict[str, dict] = {
        str(row[ticker_col]): {
            "rank":       int(row.get("rank", 9999)),
            "alpha_20d":  float(row.get(alpha_col, 0.0)),
            "confidence": str(row.get("Confidence", row.get("confidence", "低信心"))),
            "name":       str(row.get("name", row[ticker_col])),
            "sector":     str(row.get("sector", "其他")),
        }
        for _, row in kelly.iterrows()
    }
    logger.info(f"[pattern] Loaded {len(kelly_map)} stocks from df_kelly")

    # ── Load Price History ─────────────────────────────────────────────────────
    prices_path = PROCESSED_DIR / "prices_raw.parquet"
    if not prices_path.exists():
        logger.error(f"[pattern] prices_raw.parquet not found: {prices_path}")
        return {}

    logger.info("[pattern] Loading prices parquet…")
    base_cols = ["Date", "stock_id", "Close", "High", "Low"]
    try:
        pr_all = pd.read_parquet(prices_path, columns=base_cols + ["Volume"])
    except Exception:
        pr_all = pd.read_parquet(prices_path, columns=base_cols)

    pr_all["Date"] = pd.to_datetime(pr_all["Date"])
    pr_all = pr_all[pr_all["Date"] <= pd.Timestamp(date_str)].copy()
    pr_all = pr_all[pr_all["stock_id"].isin(kelly_map)].copy()
    pr_all.sort_values(["stock_id", "Date"], inplace=True)
    logger.info(f"[pattern] Price rows: {len(pr_all):,}")

    # ── Per-Stock Scan ─────────────────────────────────────────────────────────
    all_signals: list[dict] = []
    scanned = 0

    for sid, grp in pr_all.groupby("stock_id", sort=False):
        sid = str(sid)
        grp = grp.sort_values("Date").tail(200)   # season uses 180 bars
        if len(grp) < 30:
            continue

        info = kelly_map[sid]
        sigs = _scan_stock(
            stock_id=sid,
            ohlcv=grp,
            alpha_rank=info["rank"],
            alpha_20d=info["alpha_20d"],
            confidence=info["confidence"],
            name=info["name"],
            sector=info["sector"],
        )
        all_signals.extend(sigs)
        scanned += 1

    logger.info(f"[pattern] Scanned {scanned} stocks, {len(all_signals)} raw hits")

    # ── Sort: dual-confirm first, then by score desc ───────────────────────────
    all_signals.sort(key=lambda x: (-int(x["dual_confirm"]), -x["score"]))

    # Keep at most 2 signals per stock (best across all timeframes & patterns)
    stock_count: dict[str, int] = {}
    final_signals: list[dict]   = []
    for sig in all_signals:
        cnt = stock_count.get(sig["stock_id"], 0)
        if cnt < 2:
            final_signals.append(sig)
            stock_count[sig["stock_id"]] = cnt + 1

    dual_count = sum(1 for s in final_signals if s["dual_confirm"])
    elapsed    = round(time.time() - t0, 1)

    payload = {
        "date":               date_str,
        "generated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_scanned":      scanned,
        "patterns_found":     len(final_signals),
        "dual_confirm_count": dual_count,
        "elapsed_seconds":    elapsed,
        "signals":            final_signals,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[pattern] Done: {len(final_signals)} patterns "
        f"({dual_count} dual-confirm) from {scanned} stocks "
        f"in {elapsed}s → {out}"
    )
    return payload
