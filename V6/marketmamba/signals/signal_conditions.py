"""
signal_conditions.py — MarketMamba V6.2
共用進退場條件模組
=========================================
提供以下功能供 sim_engine_v3、portfolio_checker 等模組共用：

  compute_entry_score()   — 合計進場分數（Scanner 100 + 型態 40 + 雙確認 10 = 最高 150）
  entry_threshold()       — 根據市場狀態回傳門檻（正常 70 / 保守 90）
  update_trailing_stop()  — 更新四檔 Trailing Stop（只往上調）
  check_exit_conditions() — 執行四層退場邏輯，回傳 ExitTrigger 清單
  extract_main_conditions() — 從 scanner 訊號提取進場主要條件

資料結構：
  EntryRecord  — 每筆持倉狀態（存入 sim_state.json）
  ExitTrigger  — 退場觸發訊號
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# ── Entry score weights (scanner 4 conditions, sum = 100) ────────────────────
W_RANK_STABILITY  = 30
W_HIGH_CONFIDENCE = 25
W_INSTITUTIONAL   = 25
W_RELATIVE_LOW    = 20

# Pattern bonus tiers: (min_pattern_score, bonus_points)
PATTERN_BONUS_TIERS = [
    (90, 40),
    (75, 30),
    (60, 20),
]
DUAL_CONFIRM_BONUS = 10   # 型態 ≥ 60 且 alpha_rank ≤ 200，額外加分
DUAL_CONFIRM_RANK  = 200

# Entry thresholds
ENTRY_THRESHOLD_NORMAL   = 70
ENTRY_THRESHOLD_CAUTIOUS = 90

# Trailing stop: (peak_return_threshold, stop_level_relative_to_entry)
# 同 scanner_engine.py 的四檔設定
TRAILING_TIERS = [
    (0.15,  0.10),   # 峰值 ≥ +15% → 止損在進場價 +10%
    (0.10,  0.06),   # 峰值 ≥ +10% → 止損在進場價 +6%
    (0.05,  0.02),   # 峰值 ≥  +5% → 止損在進場價 +2%
    (-1.00, -0.05),  # 預設       → 止損在進場價 -5%
]

MAX_HOLD_DAYS          = 30   # 持有超過此天數直接退場（時間停損）
UNCERTAINTY_MULTIPLIER = 2.0  # 不確定度 > 進場時 × 此倍 → 退場
RS20D_NEGATIVE_DAYS    = 3    # RS_20d 連續負值天數 → 退場
ALPHA_DECLINING_DAYS   = 3    # Alpha_20d 連續下降天數 → 減半倉


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntryRecord:
    """
    每筆持倉狀態，每日結束後序列化存入 sim_state.json。

    main_conditions 記錄進場時觸發的主要條件，退場時優先檢查這些條件是否消失。
    可能的值：rank_stability / high_confidence / institutional_buy / relative_low
              / pattern:<pattern_id>
    """
    ticker:               str
    entry_date:           str
    entry_price:          float
    entry_score:          int
    main_conditions:      list[str]
    pattern_id:           Optional[str]   = None
    pattern_failure_stop: Optional[float] = None   # 型態失敗退場價
    entry_uncertainty:    float           = 0.0
    peak_return_pct:      float           = 0.0    # 歷史最高報酬率（用於 trailing stop）
    trailing_stop_price:  float           = 0.0    # 當前止損絕對價格
    hold_days:            int             = 0
    shares:               int             = 0
    cost_total:           float           = 0.0
    entry_alpha_rank:     int             = 9999

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EntryRecord":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class ExitTrigger:
    """
    退場觸發訊號。

    layer:
      1 = 停損類（立即全出）
      2 = 信號惡化類（立即全出）
      3 = 獲利保護類（減半倉）
      4 = 機會成本類（換倉）

    action:
      full_exit  — 全部出場
      half_exit  — 減半倉
      switch     — 賣出，換入新訊號
    """
    layer:     int
    ticker:    str
    condition: str
    action:    str
    detail:    str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_entry_score(
    scanner_score: int,
    pattern_score: Optional[int],
    alpha_rank:    int,
    dual_confirm:  bool = False,
) -> tuple[int, dict]:
    """
    合計進場分數：Scanner(100) + 型態加分(40) + 雙確認加分(10) = 最高 150。

    Parameters
    ----------
    scanner_score : scanner.py 計算的原始分數（0–100）
    pattern_score : pattern_scanner.py 輸出的型態分數（0–100），無型態訊號傳 None
    alpha_rank    : 當日 Alpha 排名（越小越好）
    dual_confirm  : 型態 ≥ 60 且 alpha_rank ≤ DUAL_CONFIRM_RANK 時為 True

    Returns
    -------
    (total_score, breakdown_dict)
    """
    pattern_bonus = 0
    if pattern_score is not None:
        for threshold, bonus in PATTERN_BONUS_TIERS:
            if pattern_score >= threshold:
                pattern_bonus = bonus
                break

    dc_bonus = 0
    if dual_confirm and pattern_score is not None and pattern_score >= 60:
        dc_bonus = DUAL_CONFIRM_BONUS

    total = scanner_score + pattern_bonus + dc_bonus

    return total, {
        "scanner_score":       scanner_score,
        "pattern_bonus":       pattern_bonus,
        "dual_confirm_bonus":  dc_bonus,
        "total":               total,
    }


def entry_threshold(regime: str) -> int:
    """根據市場狀態（NORMAL / CAUTIOUS）回傳進場門檻分數。"""
    return ENTRY_THRESHOLD_CAUTIOUS if regime == "CAUTIOUS" else ENTRY_THRESHOLD_NORMAL


def extract_main_conditions(scanner_signal: dict) -> list[str]:
    """
    從 action_signals.json 的一筆 buy_signal dict 提取已觸發的進場條件名稱清單。
    供建立 EntryRecord.main_conditions 使用。
    """
    conditions: list[str] = []
    if scanner_signal.get("rank_stability",    {}).get("met"):
        conditions.append("rank_stability")
    if scanner_signal.get("high_confidence",   {}).get("met"):
        conditions.append("high_confidence")
    if scanner_signal.get("institutional_buy", {}).get("met"):
        conditions.append("institutional_buy")
    if scanner_signal.get("relative_low",      {}).get("met"):
        conditions.append("relative_low")
    pat = scanner_signal.get("pattern_id")
    if pat:
        conditions.append(f"pattern:{pat}")
    return conditions


# ═══════════════════════════════════════════════════════════════════════════════
# Trailing Stop
# ═══════════════════════════════════════════════════════════════════════════════

def get_trailing_stop_pct(peak_return_pct: float) -> float:
    """
    四檔 trailing stop：根據歷史峰值報酬率，回傳相對進場價的止損比例。
    例：peak_return_pct=0.12 → 止損在進場價 +6%。
    """
    for threshold, stop in TRAILING_TIERS:
        if peak_return_pct >= threshold:
            return stop
    return -0.05


def update_trailing_stop(position: EntryRecord, current_price: float) -> EntryRecord:
    """
    更新 peak_return_pct 和 trailing_stop_price（只往上調，不往下）。
    直接修改傳入的 position 物件並回傳（方便鏈式呼叫）。
    """
    if position.entry_price <= 0:
        return position

    current_return = (current_price - position.entry_price) / position.entry_price
    if current_return > position.peak_return_pct:
        position.peak_return_pct = current_return

    new_stop = position.entry_price * (1.0 + get_trailing_stop_pct(position.peak_return_pct))
    if new_stop > position.trailing_stop_price:
        position.trailing_stop_price = new_stop

    return position


# ═══════════════════════════════════════════════════════════════════════════════
# Exit Conditions — Four Layers
# ═══════════════════════════════════════════════════════════════════════════════

def check_exit_conditions(
    position:    EntryRecord,
    market_data: dict,
) -> list[ExitTrigger]:
    """
    依四層優先順序執行退場條件檢查。
    一旦某層有觸發，立即回傳該層結果，不繼續檢查後面的層。

    market_data 必要鍵：
      current_price          float  — 今日收盤價
      alpha_rank             int    — 今日 Alpha 排名
      uncertainty            float  — 今日不確定度
      rank_out50_streak      int    — 連續掉出 Top 50 的天數
      inst_sell_streak       int    — 外資連續淨賣超天數
      rsi                    float  — 今日 RSI(14)
      rs_20d                 float  — 今日 RS_20d 值
      rs_20d_negative_days   int    — RS_20d 連續負值天數
      rs_20d_declining       bool   — RS_20d 是否持續下滑
      alpha_20d_declining_days int  — Alpha_20d 預測值連續下降天數
      signal_quality_pct     float  — 持倉股 SQ 排名百分位（0=最好, 1=最差）
      new_buy_available      bool   — 是否有新的 buy_signal
      max_positions_full     bool   — 是否已達持倉上限
      bearish_pattern        str|None — 空方型態：'m_top' / 'false_breakout' / None

    回傳值：
      list[ExitTrigger]，已排序（layer 由小到大）
      空清單 = 無退場條件觸發
    """
    triggers: list[ExitTrigger] = []
    ticker = position.ticker
    price  = float(market_data.get("current_price", 0.0))

    # ──────────────────────────────────────────────────────────────────────────
    # 第一層：停損類（立即全出）
    # ──────────────────────────────────────────────────────────────────────────

    # Trailing stop
    if price > 0 and position.trailing_stop_price > 0 and price <= position.trailing_stop_price:
        stop_pct = get_trailing_stop_pct(position.peak_return_pct)
        triggers.append(ExitTrigger(
            layer=1, ticker=ticker, action="full_exit",
            condition="Trailing Stop 觸及",
            detail=(
                f"現價 {price:.2f} ≤ 止損線 {position.trailing_stop_price:.2f} "
                f"(峰值 {position.peak_return_pct:+.1%} → 止損 {stop_pct:+.1%})"
            ),
        ))

    # 型態失敗退場價
    if (position.pattern_failure_stop is not None
            and price > 0
            and price < position.pattern_failure_stop):
        triggers.append(ExitTrigger(
            layer=1, ticker=ticker, action="full_exit",
            condition="型態關鍵價位跌破",
            detail=(
                f"現價 {price:.2f} < 型態失敗線 {position.pattern_failure_stop:.2f} "
                f"(型態：{position.pattern_id or '未知'})"
            ),
        ))

    # 外資連續 3 天淨賣出
    inst_sell = int(market_data.get("inst_sell_streak", 0))
    if inst_sell >= 3:
        triggers.append(ExitTrigger(
            layer=1, ticker=ticker, action="full_exit",
            condition="外資連續 3 天淨賣出",
            detail=f"已連續淨賣 {inst_sell} 天",
        ))

    # 空方型態確認
    bearish = market_data.get("bearish_pattern")
    if bearish == "m_top":
        triggers.append(ExitTrigger(
            layer=1, ticker=ticker, action="full_exit",
            condition="空方型態：M頭頸線跌破",
            detail="M頭確認，建議立即退場",
        ))
    elif bearish == "false_breakout":
        triggers.append(ExitTrigger(
            layer=1, ticker=ticker, action="full_exit",
            condition="空方型態：假突破向下確立",
            detail="假突破後收盤確認跌破頸線",
        ))

    # 持有超過 30 個交易日（時間停損，歸類第一層：無條件執行）
    if position.hold_days >= MAX_HOLD_DAYS:
        triggers.append(ExitTrigger(
            layer=1, ticker=ticker, action="full_exit",
            condition=f"持有超過 {MAX_HOLD_DAYS} 個交易日（時間停損）",
            detail=f"已持有 {position.hold_days} 天",
        ))

    if triggers:
        return sorted(triggers, key=lambda t: t.layer)

    # ──────────────────────────────────────────────────────────────────────────
    # 第二層：信號惡化類（立即全出）
    # ──────────────────────────────────────────────────────────────────────────

    rank_out50 = int(market_data.get("rank_out50_streak", 0))
    if rank_out50 >= 2:
        triggers.append(ExitTrigger(
            layer=2, ticker=ticker, action="full_exit",
            condition="Alpha 排名連續 2 天掉出 Top 50",
            detail=f"已連續 {rank_out50} 天排名 > 50",
        ))

    uncertainty = float(market_data.get("uncertainty", 0.0))
    if position.entry_uncertainty > 0 and uncertainty > position.entry_uncertainty * UNCERTAINTY_MULTIPLIER:
        triggers.append(ExitTrigger(
            layer=2, ticker=ticker, action="full_exit",
            condition="Uncertainty 超過進場當天的 2 倍",
            detail=f"當前 {uncertainty:.4f} > 進場時 {position.entry_uncertainty:.4f} × {UNCERTAINTY_MULTIPLIER}",
        ))

    rs_neg = int(market_data.get("rs_20d_negative_days", 0))
    if rs_neg >= RS20D_NEGATIVE_DAYS:
        triggers.append(ExitTrigger(
            layer=2, ticker=ticker, action="full_exit",
            condition="RS_20d 從正轉負並持續 3 天",
            detail=f"RS_20d 已連續負值 {rs_neg} 天",
        ))

    # 排名穩定性消失（原因消滅）：只有當初因排名穩定進場的才檢查
    if "rank_stability" in position.main_conditions and rank_out50 >= 1:
        triggers.append(ExitTrigger(
            layer=2, ticker=ticker, action="full_exit",
            condition="排名穩定性消失（進場主因消滅）",
            detail=f"原因排名穩定進場，但排名已掉出 Top 50 {rank_out50} 天",
        ))

    if triggers:
        return sorted(triggers, key=lambda t: t.layer)

    # ──────────────────────────────────────────────────────────────────────────
    # 第三層：獲利保護類（減半倉）
    # ──────────────────────────────────────────────────────────────────────────

    rsi          = float(market_data.get("rsi", 50.0))
    rs_declining = bool(market_data.get("rs_20d_declining", False))
    if rsi > 75 and rs_declining:
        triggers.append(ExitTrigger(
            layer=3, ticker=ticker, action="half_exit",
            condition="RSI > 75 且 RS_20d 開始下滑",
            detail=f"RSI={rsi:.0f}, RS_20d 動能轉弱",
        ))

    if price > 0 and position.entry_price > 0:
        current_return = (price - position.entry_price) / position.entry_price
        if current_return >= 0.20:
            triggers.append(ExitTrigger(
                layer=3, ticker=ticker, action="half_exit",
                condition="報酬率達 +20%（主動減半鎖利）",
                detail=f"當前報酬 {current_return:+.1%}",
            ))

    alpha_dec = int(market_data.get("alpha_20d_declining_days", 0))
    if alpha_dec >= ALPHA_DECLINING_DAYS:
        triggers.append(ExitTrigger(
            layer=3, ticker=ticker, action="half_exit",
            condition="Alpha_20d 預測值連續 3 天下降",
            detail=f"已連續下降 {alpha_dec} 天",
        ))

    if triggers:
        return sorted(triggers, key=lambda t: t.layer)

    # ──────────────────────────────────────────────────────────────────────────
    # 第四層：機會成本類（換倉）
    # ──────────────────────────────────────────────────────────────────────────

    sq_pct   = float(market_data.get("signal_quality_pct", 0.0))
    new_buy  = bool(market_data.get("new_buy_available", False))
    is_full  = bool(market_data.get("max_positions_full", False))

    if sq_pct >= 0.50:
        triggers.append(ExitTrigger(
            layer=4, ticker=ticker, action="switch",
            condition="Signal_Quality 排名落後市場後 50%",
            detail=f"SQ 排名位於後 {(1 - sq_pct):.0%}",
        ))

    if new_buy and is_full:
        triggers.append(ExitTrigger(
            layer=4, ticker=ticker, action="switch",
            condition="滿倉且有新進場訊號",
            detail="賣出本檔（最差 SQ 排名），換入新訊號",
        ))

    return sorted(triggers, key=lambda t: t.layer)
