"""
MarketMamba V6.2 — Traditional Chart Pattern Scanner (Rewrite)
===============================================================
Bullish patterns (進場用，5種):
  w_bottom   — W底：兩低點價差 <5%，突破頸線
  spring_w   — 彈簧型W底：第二低多跌 0.5~3%，快速站回
  hs_bottom  — 頭肩底：三低點，中間最低
  triangle   — 收斂三角底部：高低點收斂，有效突破上軌
  flag       — 上飄旗形：旗桿後小幅下斜整理（中繼）

Bearish patterns (退場用，2種):
  m_top          — M頭：雙頂，收盤確認跌破頸線
  false_breakout — 假突破向下：突破後迅速跌回確認

Scoring (bullish, 基礎 max 100 + beauty + alpha_bonus):
  型態強度：40  成交量：30  位置（波段跌幅）：20  RSI：10
  位置評分：跌幅 >70%→20, 50~70%→15, 30~50%→10, <30%→0
  Alpha 加成：rank≤200→+10, rank≤300→+5

Output: V6/results/pattern_signals.json
  signals         — 多方型態（含 failure_stop）
  bearish_signals — 空方型態（M頭 / 假突破向下）
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

PATTERN_SIGNALS_PATH = RESULTS_DIR / "pattern_signals.json"

TIMEFRAMES = [
    {"id": "short",  "label": "短線 約2週",  "days":  30, "order": 3},
    {"id": "medium", "label": "中線 約1個月", "days":  60, "order": 8},
    {"id": "long",   "label": "長線 約3個月", "days": 120, "order": 15},
    {"id": "season", "label": "季線 約半年",  "days": 180, "order": 30},
]

SCORE_THRESHOLD   = 60
DUAL_CONFIRM_RANK = 200


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _find_local_extrema_np(arr: np.ndarray, order: int = 5):
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


def _rsi_last(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 2:
        return 50.0
    delta = np.diff(close.astype(float))
    gain, loss = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
    avg_g, avg_l = float(gain[:period].mean()), float(loss[:period].mean())
    for g, l in zip(gain[period:], loss[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    return float(100 - 100 / (1 + avg_g / (avg_l + 1e-10)))


def _vol_ratio(volume: Optional[np.ndarray], idx: int, window: int = 5) -> float:
    if volume is None or idx < window or idx >= len(volume):
        return 1.0
    past_avg = float(volume[max(0, idx - window): idx].mean())
    return float(volume[idx]) / (past_avg + 1e-10) if past_avg > 0 else 1.0


def _rr_str(target: float, current: float, stop: float) -> str:
    if current > stop and target > current:
        return f"1:{(target - current) / (current - stop):.1f}"
    return "N/A"


def _position_score(close: np.ndarray, pattern_low: float) -> int:
    """波段跌幅評分：視窗內最高點 → 型態最低點。"""
    peak = float(close.max())
    if peak <= 0 or pattern_low <= 0:
        return 0
    dd = (peak - pattern_low) / peak
    if dd > 0.70:
        return 20
    if dd > 0.50:
        return 15
    if dd > 0.30:
        return 10
    return 0


def _count_trendline_touches(
    price_hi: np.ndarray, price_lo: np.ndarray,
    r_slope: float, r_int: float,
    s_slope: float, s_int: float,
    t_start: int, t_end: int,
    tol: float = 0.015,
) -> int:
    count = 0
    for i in range(t_start, min(t_end + 1, len(price_hi))):
        r_val = r_slope * i + r_int
        s_val = s_slope * i + s_int
        if r_val > 0 and abs(price_hi[i] - r_val) / r_val <= tol:
            count += 1
        if s_val > 0 and abs(price_lo[i] - s_val) / s_val <= tol:
            count += 1
    return count


# ════════════════════════════════════════════════════════════════════════════════
# Bullish Pattern Detectors
# Return dict {score, beauty_bonus, key_price, target_price,
#              stop_loss, failure_stop, risk_reward}  or  None
# ════════════════════════════════════════════════════════════════════════════════

def _detect_w_bottom(
    close: np.ndarray,
    volume: Optional[np.ndarray],
    min_idx: np.ndarray,
    max_idx: np.ndarray,
) -> Optional[dict]:
    """
    W底（雙底）
    進場：收盤突破頸線 > 3%
    失敗退場：跌破任一低點
    漂亮加分：兩低點價差 < 2% → +10
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
        between = max_idx[(max_idx > l1_i) & (max_idx < l2_i)]
        neck = (float(close[between[np.argmax(close[between])]])
                if len(between) > 0 else float(close[l1_i: l2_i + 1].max()))
        avg_low = (l1 + l2) / 2.0
        if (neck - avg_low) / avg_low < 0.03:
            continue
        current = float(close[-1])
        if current < neck * 0.95:
            continue
        form  = int(40 * max(0.0, 1.0 - spread / 0.05))
        vr    = _vol_ratio(volume, l2_i)
        vol_s = min(30, int(30 * min(vr / 1.5, 1.0))) if volume is not None else 18
        pos_s = _position_score(close, min(l1, l2))
        rsi_s = (10 if _rsi_last(close) >= 50 else 7 if _rsi_last(close) >= 45 else 3)
        beauty = 10 if spread < 0.02 else 0
        score  = form + vol_s + pos_s + rsi_s
        target = neck + (neck - avg_low)
        entry  = {
            "score": score, "beauty_bonus": beauty,
            "key_price": round(neck, 2), "target_price": round(target, 2),
            "stop_loss": round(min(l1, l2) * 0.97, 2),
            "failure_stop": round(min(l1, l2), 2),
            "risk_reward": _rr_str(target, current, min(l1, l2) * 0.97),
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
    彈簧型W底（破底翻）
    識別：第二低多跌 0.5%~3%，破底後 3 天內站回第一低之上
    失敗退場：第二低再被跌破
    漂亮加分：破底幅度 < 1.5% 且反彈 ≤ 2 天 → +15
    """
    if len(min_idx) < 2:
        return None
    best: Optional[dict] = None
    for i in range(len(min_idx) - 1):
        l1_i, l2_i = int(min_idx[i]), int(min_idx[i + 1])
        l1, l2 = float(close[l1_i]), float(close[l2_i])
        dip = (l1 - l2) / l1
        if not (0.005 <= dip <= 0.03):
            continue
        gap = l2_i - l1_i
        if gap < 5 or gap > 40:
            continue
        post = close[l2_i:]
        if len(post) < 3 or float(post.max()) <= l1:
            continue
        between = max_idx[(max_idx > l1_i) & (max_idx < l2_i)]
        neck = (float(close[between[np.argmax(close[between])]])
                if len(between) > 0 else float(close[l1_i: l2_i + 1].max()))
        if (neck - l1) / l1 < 0.02:
            continue
        current = float(close[-1])
        if current < neck * 0.94:
            continue
        recovery_days: Optional[int] = None
        for k in range(1, min(6, len(post))):
            if float(post[k]) > l1:
                recovery_days = k
                break
        ideal_dev = abs(dip - 0.012) / 0.012
        form  = int(40 * max(0.0, 1.0 - min(ideal_dev, 1.0)))
        vr    = _vol_ratio(volume, l2_i)
        vol_s = min(30, int(30 * min(vr / 2.0, 1.0))) if volume is not None else 18
        pos_s = _position_score(close, l2)
        rsi   = _rsi_last(close)
        rsi_s = 10 if rsi >= 50 else (7 if rsi >= 45 else 3)
        beauty = 15 if (dip < 0.015 and recovery_days is not None and recovery_days <= 2) else 0
        score  = form + vol_s + pos_s + rsi_s
        target = neck + (neck - l2)
        entry  = {
            "score": score, "beauty_bonus": beauty,
            "key_price": round(neck, 2), "target_price": round(target, 2),
            "stop_loss": round(l2 * 0.97, 2), "failure_stop": round(l2, 2),
            "risk_reward": _rr_str(target, current, l2 * 0.97),
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
    失敗退場：跌破右肩低點
    漂亮加分：兩肩高點差 < 5% → +10；型態寬度 < 視窗 60% → +5（中繼型態）
    """
    if len(min_idx) < 3:
        return None
    best: Optional[dict] = None
    for i in range(len(min_idx) - 2):
        ls_i, h_i, rs_i = int(min_idx[i]), int(min_idx[i+1]), int(min_idx[i+2])
        ls, hd, rs = float(close[ls_i]), float(close[h_i]), float(close[rs_i])
        if not (hd < ls and hd < rs):
            continue
        sh_diff = abs(ls - rs) / max(ls, rs)
        if sh_diff > 0.10:
            continue
        if ((ls + rs) / 2.0 - hd) / ((ls + rs) / 2.0) < 0.03:
            continue
        lp_idx = max_idx[(max_idx > ls_i) & (max_idx < h_i)]
        rp_idx = max_idx[(max_idx > h_i)  & (max_idx < rs_i)]
        if len(lp_idx) == 0 or len(rp_idx) == 0:
            continue
        lp = float(close[lp_idx[np.argmax(close[lp_idx])]])
        rp = float(close[rp_idx[np.argmax(close[rp_idx])]])
        neckline = (lp + rp) / 2.0
        current  = float(close[-1])
        if current < neckline * 0.94:
            continue
        form  = int(40 * max(0.0, 1.0 - sh_diff / 0.10))
        vr    = _vol_ratio(volume, rs_i)
        vol_s = (min(30, int(30 * (1.5 - min(vr, 1.5)))) if volume is not None else 18)
        pos_s = _position_score(close, hd)
        rsi   = _rsi_last(close)
        rsi_s = 10 if rsi >= 48 else (6 if rsi >= 42 else 2)
        beauty  = 0
        if sh_diff < 0.05:
            beauty += 10
        if (rs_i - ls_i) < len(close) * 0.60:
            beauty += 5
        score  = form + vol_s + pos_s + rsi_s
        target = neckline + (neckline - hd)
        entry  = {
            "score": score, "beauty_bonus": beauty,
            "key_price": round(neckline, 2), "target_price": round(target, 2),
            "stop_loss": round(rs * 0.97, 2), "failure_stop": round(rs, 2),
            "risk_reward": _rr_str(target, current, rs * 0.97),
        }
        if best is None or score > best["score"]:
            best = entry
    return best if (best and best["score"] >= SCORE_THRESHOLD) else None


def _detect_triangle(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: Optional[np.ndarray],
    min_idx_l: np.ndarray,
    max_idx_h: np.ndarray,
) -> Optional[dict]:
    """
    收斂三角底部
    有效突破：在三角形 1/2 至 3/4 處突破上軌
    失敗退場：跌破下軌
    漂亮加分：觸點 ≥ 6 → +10；突破位置在 1/2~2/3 → +5
    """
    if len(max_idx_h) < 2 or len(min_idx_l) < 2:
        return None
    hi_idx = max_idx_h[-4:]
    lo_idx = min_idx_l[-4:]
    hi_p   = high[hi_idx].astype(float)
    lo_p   = low[lo_idx].astype(float)
    if not all(hi_p[j] > hi_p[j+1] for j in range(len(hi_p)-1)):
        return None
    if not all(lo_p[j] < lo_p[j+1] for j in range(len(lo_p)-1)):
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
    n      = len(close) - 1
    r_now  = r_slope * n + r_int
    s_now  = s_slope * n + s_int
    if r_now <= s_now:
        return None
    current = float(close[-1])
    if not (s_now * 0.98 <= current <= r_now * 1.03):
        return None
    width_pct = (r_now - s_now) / max(s_now, 1.0)
    if width_pct > 0.25:
        return None
    t0       = min(r_t0, s_t0)
    pct_apex = (n - t0) / max(apex_t - t0, 1.0)
    if not (0.50 <= pct_apex <= 0.90):
        return None
    form = int(40 * max(0.0, 1.0 - width_pct / 0.20))
    if volume is not None and len(volume) >= 10:
        half    = len(volume) // 2
        vr      = float(volume[half:].mean()) / (float(volume[:half].mean()) + 1e-10)
        vol_s   = 30 if vr < 0.70 else (22 if vr < 0.90 else (14 if vr < 1.10 else 6))
    else:
        vol_s = 18
    pos_s = _position_score(close, float(low[lo_idx[-1]]))
    rsi   = _rsi_last(close)
    rsi_s = 10 if rsi >= 45 else (6 if rsi >= 40 else 2)
    beauty  = 0
    touches = _count_trendline_touches(high, low, r_slope, r_int, s_slope, s_int, t0, n)
    if touches >= 6:
        beauty += 10
    if 0.50 <= pct_apex <= 0.67:
        beauty += 5
    score  = form + vol_s + pos_s + rsi_s
    tri_h  = r_now - s_now
    target = r_now + tri_h
    entry  = {
        "score": score, "beauty_bonus": beauty,
        "key_price": round(r_now, 2), "target_price": round(target, 2),
        "stop_loss": round(s_now * 0.97, 2), "failure_stop": round(s_now, 2),
        "risk_reward": _rr_str(target, current, s_now * 0.97),
    }
    return entry if score >= SCORE_THRESHOLD else None


def _detect_flag(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: Optional[np.ndarray],
) -> Optional[dict]:
    """
    上飄旗形（中繼型態）
    旗桿：前有明顯上漲（≥15%，≤15 bars）
    旗形：小幅下斜整理，振幅 < 旗桿高度 60%
    失敗退場：跌破旗形最低點
    漂亮加分：旗形振幅 < 旗桿 1/3 → +10
    """
    n = len(close)
    if n < 30:
        return None
    best: Optional[dict] = None
    for flag_len in range(5, min(26, n - 10)):
        pole_end   = n - flag_len - 1
        pole_start = max(0, pole_end - 15)
        if pole_end - pole_start < 4:
            continue
        pole_top  = float(high[pole_end])
        pole_base = float(close[pole_start: pole_end + 1].min())
        pole_rise = (pole_top - pole_base) / max(pole_base, 1e-10)
        if pole_rise < 0.15:
            continue
        flag_h = high[-flag_len:]
        flag_l = low[-flag_len:]
        flag_high_max = float(flag_h.max())
        flag_low_min  = float(flag_l.min())
        if flag_high_max <= 0 or flag_high_max > pole_top * 1.03:
            continue
        flag_amplitude = (flag_high_max - flag_low_min) / flag_high_max
        if flag_amplitude > pole_rise * 0.60:
            continue
        mid = flag_len // 2
        if mid > 0 and float(flag_h[mid:].max()) > float(flag_h[:mid].max()) * 1.02:
            continue
        early_h   = float(flag_h[:max(1, flag_len // 3)].max())
        late_h    = float(flag_h[-max(1, flag_len // 3):].max())
        h_slope   = (late_h - early_h) / max(flag_len - flag_len // 3, 1)
        upper_now = late_h + h_slope
        current   = float(close[-1])
        if not (upper_now * 0.93 <= current <= upper_now * 1.04):
            continue
        form_s = int(40 * max(0.0, 1.0 - flag_amplitude / (pole_rise * 0.60)))
        if volume is not None:
            pv = volume[pole_start: n - flag_len]
            fv = volume[-flag_len:]
            pv_mean = float(pv.mean()) if len(pv) > 0 else 0.0
            ratio   = float(fv.mean()) / pv_mean if pv_mean > 0 else 1.0
            vol_s   = 30 if ratio < 0.40 else (22 if ratio < 0.60 else (14 if ratio < 0.80 else 6))
        else:
            vol_s = 18
        pos_s  = _position_score(close, pole_base)
        rsi    = _rsi_last(close)
        rsi_s  = 10 if rsi >= 50 else (7 if rsi >= 45 else 3)
        beauty = 10 if flag_amplitude < pole_rise / 3 else 0
        score  = form_s + vol_s + pos_s + rsi_s
        target = pole_top + (pole_top - pole_base)
        entry  = {
            "score": score, "beauty_bonus": beauty,
            "key_price": round(upper_now, 2), "target_price": round(target, 2),
            "stop_loss": round(flag_low_min * 0.97, 2), "failure_stop": round(flag_low_min, 2),
            "risk_reward": _rr_str(target, current, flag_low_min * 0.97),
        }
        if best is None or score > best["score"]:
            best = entry
    return best if (best and best["score"] >= SCORE_THRESHOLD) else None


# ════════════════════════════════════════════════════════════════════════════════
# Bearish Pattern Detectors
# Return dict {score, key_price, target_price, stop_loss, risk_reward}  or  None
# ════════════════════════════════════════════════════════════════════════════════

def _detect_m_top(
    close: np.ndarray,
    volume: Optional[np.ndarray],
    min_idx: np.ndarray,
    max_idx: np.ndarray,
) -> Optional[dict]:
    """
    M頭（雙頂） — 退場訊號，收盤確認跌破頸線才輸出
    目標跌幅：頸線 - (頂部 - 頸線)，供風險評估
    """
    if len(max_idx) < 2:
        return None
    best: Optional[dict] = None
    for i in range(len(max_idx) - 1):
        h1_i, h2_i = int(max_idx[i]), int(max_idx[i + 1])
        h1, h2 = float(close[h1_i]), float(close[h2_i])
        spread = abs(h1 - h2) / max(h1, h2)
        if spread > 0.05:
            continue
        between = min_idx[(min_idx > h1_i) & (min_idx < h2_i)]
        neck = (float(close[between[np.argmin(close[between])]])
                if len(between) > 0 else float(close[h1_i: h2_i + 1].min()))
        avg_top = (h1 + h2) / 2.0
        if (avg_top - neck) / avg_top < 0.03:
            continue
        current = float(close[-1])
        if current >= neck:   # 尚未確認跌破，不輸出
            continue
        form_s = int(40 * max(0.0, 1.0 - spread / 0.05))
        vr     = _vol_ratio(volume, h2_i)
        vol_s  = min(30, int(30 * max(0.0, 1.5 - min(vr, 1.5)))) if volume is not None else 18
        score  = form_s + vol_s
        entry  = {
            "score": score,
            "key_price": round(neck, 2),
            "target_price": round(neck - (avg_top - neck), 2),
            "stop_loss": round(avg_top, 2),
            "risk_reward": "N/A",
        }
        if best is None or score > best["score"]:
            best = entry
    return best


def _detect_false_breakout(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: Optional[np.ndarray],
    max_idx: np.ndarray,
) -> Optional[dict]:
    """
    假突破向下 — 近期突破壓力後迅速跌回，收盤確認
    退場觸發：假突破確立當天
    """
    n = len(close)
    if n < 15 or len(max_idx) < 1:
        return None
    valid_idx = max_idx[(max_idx < n - 3) & (max_idx >= max(0, n - 30))]
    if len(valid_idx) < 1:
        return None
    resistance_i = int(valid_idx[np.argmax(close[valid_idx])])
    resistance   = float(close[resistance_i])
    if resistance <= 0:
        return None
    recent = close[-5:]
    above_bars = [j for j in range(len(recent) - 1) if float(recent[j]) > resistance]
    if not above_bars:
        return None
    current = float(close[-1])
    if current >= resistance:
        return None
    fail_speed = (len(recent) - 1) - above_bars[0]
    if fail_speed <= 0 or fail_speed > 5:
        return None
    recent_high = float(high[-5:].max())
    score = max(40, 75 - fail_speed * 10)
    return {
        "score": score,
        "key_price": round(resistance, 2),
        "target_price": round(resistance - (recent_high - resistance) * 2, 2),
        "stop_loss": round(recent_high * 1.02, 2),
        "risk_reward": "N/A",
    }


# ════════════════════════════════════════════════════════════════════════════════
# Per-Stock Scanner
# ════════════════════════════════════════════════════════════════════════════════

def _scan_stock(
    stock_id:   str,
    ohlcv:      pd.DataFrame,
    alpha_rank: int,
    alpha_20d:  float,
    confidence: str,
    name:       str,
    sector:     str,
) -> tuple[list[dict], list[dict]]:
    """Returns (bullish_signals, bearish_signals)."""
    close_all = ohlcv["Close"].values.astype(float)
    high_all  = ohlcv["High"].values.astype(float)
    low_all   = ohlcv["Low"].values.astype(float)
    vol_all   = ohlcv["Volume"].values.astype(float) if "Volume" in ohlcv.columns else None

    alpha_bonus = 10 if alpha_rank <= 200 else (5 if alpha_rank <= 300 else 0)

    bullish: list[dict] = []
    bearish: list[dict] = []

    for tf in TIMEFRAMES:
        n_days = tf["days"]
        order  = tf["order"]
        if len(close_all) < n_days:
            continue

        close = close_all[-n_days:]
        high  = high_all[-n_days:]
        low   = low_all[-n_days:]
        vol   = vol_all[-n_days:] if vol_all is not None else None

        min_c, max_c = _find_local_extrema_np(close, order)
        min_l, _     = _find_local_extrema_np(low,   order)
        _,     max_h = _find_local_extrema_np(high,  order)

        # ── Bullish ──────────────────────────────────────────────────────────
        for pat_id, pat_name, fn, args in [
            ("w_bottom",  "W底（雙底）",  _detect_w_bottom,  (close, vol, min_c, max_c)),
            ("spring_w",  "彈簧型W底",    _detect_spring_w,  (close, vol, min_c, max_c)),
            ("hs_bottom", "頭肩底",        _detect_hs_bottom, (close, vol, min_c, max_c)),
            ("triangle",  "收斂三角底部",  _detect_triangle,  (close, high, low, vol, min_l, max_h)),
            ("flag",      "上飄旗形",      _detect_flag,      (close, high, low, vol)),
        ]:
            try:
                result = fn(*args)
            except Exception:
                continue
            if result is None:
                continue
            final_score = min(100, result["score"] + result.get("beauty_bonus", 0) + alpha_bonus)
            if final_score < SCORE_THRESHOLD:
                continue
            bullish.append({
                "stock_id":        stock_id,
                "name":            name,
                "sector":          sector,
                "pattern_id":      pat_id,
                "pattern_name":    pat_name,
                "timeframe":       tf["id"],
                "timeframe_label": tf["label"],
                "score":           final_score,
                "score_raw":       result["score"],
                "beauty_bonus":    result.get("beauty_bonus", 0),
                "alpha_bonus":     alpha_bonus,
                "key_price":       result["key_price"],
                "target_price":    result["target_price"],
                "stop_loss":       result["stop_loss"],
                "failure_stop":    result.get("failure_stop", result["stop_loss"]),
                "risk_reward":     result["risk_reward"],
                "alpha_rank":      alpha_rank,
                "alpha_20d":       round(float(alpha_20d), 4),
                "confidence":      confidence,
                "dual_confirm":    alpha_rank <= DUAL_CONFIRM_RANK,
            })

        # ── Bearish（只掃 medium 以上，60d+ 窗口較可靠）────────────────────
        if n_days < 60:
            continue
        for pat_id, pat_name, fn, args in [
            ("m_top",          "M頭",       _detect_m_top,          (close, vol, min_c, max_c)),
            ("false_breakout", "假突破向下", _detect_false_breakout, (close, high, low, vol, max_c)),
        ]:
            try:
                result = fn(*args)
            except Exception:
                continue
            if result is None:
                continue
            bearish.append({
                "stock_id":        stock_id,
                "name":            name,
                "sector":          sector,
                "pattern_id":      pat_id,
                "pattern_name":    pat_name,
                "timeframe":       tf["id"],
                "timeframe_label": tf["label"],
                "score":           result["score"],
                "key_price":       result["key_price"],
                "target_price":    result["target_price"],
                "stop_loss":       result["stop_loss"],
                "risk_reward":     result["risk_reward"],
                "alpha_rank":      alpha_rank,
                "alpha_20d":       round(float(alpha_20d), 4),
                "confidence":      confidence,
            })

    return bullish, bearish


# ════════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ════════════════════════════════════════════════════════════════════════════════

def run_pattern_scan(
    date_str:      str,
    df_kelly_path: Optional[Path] = None,
    output_path:   Optional[Path] = None,
) -> dict:
    """
    Scan full stock universe for bullish and bearish chart patterns.

    Output JSON:
      signals         — 多方型態（含 failure_stop，供 sim_engine_v3 進場評分）
      bearish_signals — 空方型態（M頭 / 假突破向下，供四層退場第一層）
    """
    t0 = time.time()
    if df_kelly_path is None:
        df_kelly_path = RESULTS_DIR / "df_kelly.csv"
    if output_path is None:
        output_path = PATTERN_SIGNALS_PATH

    logger.info(f"[pattern] Starting scan for {date_str}")

    if not Path(df_kelly_path).exists():
        logger.error(f"[pattern] df_kelly.csv not found: {df_kelly_path}")
        return {}

    kelly      = pd.read_csv(df_kelly_path)
    ticker_col = "Ticker" if "Ticker" in kelly.columns else "stock_id"
    alpha_col  = "Exp_Alpha_20d" if "Exp_Alpha_20d" in kelly.columns else "alpha_20d"

    if "rank" not in kelly.columns:
        kelly = kelly.sort_values(alpha_col, ascending=False, na_position="last").reset_index(drop=True)
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

    all_bullish: list[dict] = []
    all_bearish: list[dict] = []
    scanned = 0

    for sid, grp in pr_all.groupby("stock_id", sort=False):
        sid = str(sid)
        grp = grp.sort_values("Date").tail(200)
        if len(grp) < 30:
            continue
        info = kelly_map[sid]
        b, bear = _scan_stock(
            stock_id=sid, ohlcv=grp,
            alpha_rank=info["rank"], alpha_20d=info["alpha_20d"],
            confidence=info["confidence"], name=info["name"], sector=info["sector"],
        )
        all_bullish.extend(b)
        all_bearish.extend(bear)
        scanned += 1

    logger.info(f"[pattern] Scanned {scanned}: {len(all_bullish)} bullish / {len(all_bearish)} bearish")

    # Bullish: dual-confirm first, then score desc; cap 2 per stock
    all_bullish.sort(key=lambda x: (-int(x["dual_confirm"]), -x["score"]))
    stock_cnt: dict[str, int] = {}
    final_bullish: list[dict] = []
    for sig in all_bullish:
        if stock_cnt.get(sig["stock_id"], 0) < 2:
            final_bullish.append(sig)
            stock_cnt[sig["stock_id"]] = stock_cnt.get(sig["stock_id"], 0) + 1

    # Bearish: score desc; cap 1 per stock (strongest signal wins)
    all_bearish.sort(key=lambda x: -x["score"])
    seen: set[str] = set()
    final_bearish: list[dict] = []
    for sig in all_bearish:
        if sig["stock_id"] not in seen:
            final_bearish.append(sig)
            seen.add(sig["stock_id"])

    dual_count = sum(1 for s in final_bullish if s["dual_confirm"])
    elapsed    = round(time.time() - t0, 1)

    payload = {
        "date":               date_str,
        "generated_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_scanned":      scanned,
        "patterns_found":     len(final_bullish),
        "bearish_found":      len(final_bearish),
        "dual_confirm_count": dual_count,
        "elapsed_seconds":    elapsed,
        "signals":            final_bullish,
        "bearish_signals":    final_bearish,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[pattern] Done: {len(final_bullish)} bullish ({dual_count} dual-confirm), "
        f"{len(final_bearish)} bearish, {scanned} stocks, {elapsed}s → {out}"
    )
    return payload
