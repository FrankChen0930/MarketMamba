"""
sim_engine_v3.py — MarketMamba V6.2
有狀態日更模擬機器人
====================================
核心設計：每日結束後將持倉狀態寫入 sim_state.json，
隔天讀取繼續執行，不重新開始。

每日流程：
  1. 讀取 sim_state.json（昨日持倉）
  2. 讀取今日 action_signals.json / pattern_signals.json / df_kelly.csv
  3. 更新市值 → 更新 Trailing Stop（只往上調）
  4. 逐倉執行四層退場條件（第一層有觸發立即停止）
  5. 計算可用資金，對 buy_signals 評分排序後依序進場
  6. 寫入 sim_state.json
  7. Append 交易紀錄到 sim_trades.jsonl

入口：
  run_daily_update()  — 日更（主要使用）
  run_backtest()      — 全量回放所有 archive 目錄
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from marketmamba.config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR
from marketmamba.signals.signal_conditions import (
    EntryRecord, ExitTrigger,
    check_exit_conditions, compute_entry_score, entry_threshold,
    extract_main_conditions, get_trailing_stop_pct, update_trailing_stop,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS   = 15
MAX_POS_PCT     = 0.10    # single-stock cap 10%
CASH_BUFFER_PCT = 0.05    # always keep ≥5% cash
BUY_FEE         = 0.0015  # 買進手續費
SELL_FEE        = 0.0045  # 賣出手續費 0.15% + 證交稅 0.30%
MIN_ALLOC_PCT   = 0.02    # 最小建倉比例

SIM_STATE_PATH  = RESULTS_DIR / "sim_state.json"
SIM_TRADES_PATH = RESULTS_DIR / "sim_trades.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# State Container
# ═══════════════════════════════════════════════════════════════════════════════

class SimState:
    def __init__(
        self,
        date:         str,
        cash:         float,
        positions:    dict[str, dict],
        equity_curve: list[dict],
    ):
        self.date         = date
        self.cash         = cash
        self.positions    = positions       # ticker → EntryRecord.to_dict()
        self.equity_curve = equity_curve

    def to_dict(self) -> dict:
        return {
            "date":         self.date,
            "cash":         self.cash,
            "positions":    self.positions,
            "equity_curve": self.equity_curve,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimState":
        return cls(
            date         = d.get("date", ""),
            cash         = float(d.get("cash", INITIAL_CAPITAL)),
            positions    = d.get("positions", {}),
            equity_curve = d.get("equity_curve", []),
        )

    @classmethod
    def fresh(cls) -> "SimState":
        return cls(date="", cash=float(INITIAL_CAPITAL), positions={}, equity_curve=[])


# ═══════════════════════════════════════════════════════════════════════════════
# Market Data Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json_safe(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def _rsi_14(prices_df: pd.DataFrame, ticker: str) -> float:
    stock = prices_df[prices_df["stock_id"] == ticker].sort_values("Date").tail(30)
    if len(stock) < 16:
        return 50.0
    closes = stock["Close"].values.astype(float)
    delta  = np.diff(closes)
    gain, loss = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
    ag = float(gain[:14].mean())
    al = float(loss[:14].mean())
    for g, l in zip(gain[14:], loss[14:]):
        ag = (ag * 13 + g) / 14
        al = (al * 13 + l) / 14
    return float(100 - 100 / (1 + ag / (al + 1e-10)))


def _inst_sell_streak(inst_df: pd.DataFrame, ticker: str) -> int:
    """外資連續淨賣超天數（負值代表淨賣）。"""
    if inst_df is None or inst_df.empty:
        return 0
    stock = inst_df[inst_df["stock_id"] == ticker].sort_values("Date").tail(10)
    if stock.empty:
        return 0
    buy_col = next(
        (c for c in stock.columns if "foreign" in c.lower() and ("buy" in c.lower() or "net" in c.lower())),
        next((c for c in stock.columns if "buy" in c.lower()), None),
    )
    if buy_col is None:
        return 0
    streak = 0
    for val in reversed(stock[buy_col].values):
        try:
            if float(val) < 0:
                streak += 1
            else:
                break
        except (ValueError, TypeError):
            break
    return streak


def _rank_out50_streak(history: list, ticker: str) -> int:
    """從 history_index 計算連續掉出 Top 50 天數。"""
    streak = 0
    for entry in history[:10]:
        rank = next(
            (p.get("rank") for p in entry.get("portfolio", []) if p.get("ticker") == ticker),
            None,
        )
        if rank is None or rank > 50:
            streak += 1
        else:
            break
    return streak


def _alpha_declining_days(recent_kellys: list[pd.DataFrame], ticker: str) -> int:
    """Alpha_20d 預測值連續下降天數（由最新到舊）。"""
    vals: list[float] = []
    for df in recent_kellys:
        row = df[df["Ticker"].astype(str) == ticker]
        if row.empty:
            break
        col = "Exp_Alpha_20d" if "Exp_Alpha_20d" in row.columns else "alpha_20d"
        vals.append(float(row.iloc[0].get(col, 0)))
    count = 0
    for i in range(len(vals) - 1):
        if vals[i] < vals[i + 1]:   # 最新 < 前一天 → 下降
            count += 1
        else:
            break
    return count


def _rs20d_info(recent_kellys: list[pd.DataFrame], ticker: str) -> tuple[int, bool]:
    """返回 (RS_20d 連續負值天數, 是否持續下滑)。"""
    if not recent_kellys or "RS_20d" not in recent_kellys[0].columns:
        return 0, False
    vals: list[float] = []
    for df in recent_kellys:
        row = df[df["Ticker"].astype(str) == ticker]
        if row.empty:
            break
        vals.append(float(row.iloc[0].get("RS_20d", 0)))
    if not vals:
        return 0, False
    neg_count = 0
    for v in vals:
        if v < 0:
            neg_count += 1
        else:
            break
    declining = len(vals) >= 2 and all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    return neg_count, declining


def _sq_percentile(ticker: str, df_kelly: pd.DataFrame) -> float:
    """SQ 排名百分位：0 = 最佳, 1 = 最差。未找到回傳 1.0。"""
    if df_kelly.empty or "Signal_Quality" not in df_kelly.columns:
        return 0.5
    # O3：優先用未截斷 raw 值排序（Top 區截斷並列 10.0 無法區分）
    _sq_col = "Signal_Quality_Raw" if "Signal_Quality_Raw" in df_kelly.columns else "Signal_Quality"
    sorted_df = (
        df_kelly.sort_values(_sq_col, ascending=False)
        .reset_index(drop=True)
    )
    total = len(sorted_df)
    idx = sorted_df.index[sorted_df["Ticker"].astype(str) == ticker].tolist()
    return (idx[0] + 1) / total if idx else 1.0


def _build_market_data(
    ticker:          str,
    position:        EntryRecord,
    current_price:   float,
    df_kelly:        pd.DataFrame,
    history:         list,
    inst_df:         pd.DataFrame,
    prices_df:       pd.DataFrame,
    bearish_map:     dict[str, str],
    recent_kellys:   list[pd.DataFrame],
    new_buy_tickers: set[str],
    n_positions:     int,
) -> dict:
    """組合 check_exit_conditions() 所需的 market_data dict。"""
    kelly_row = df_kelly[df_kelly["Ticker"].astype(str) == ticker]
    uncertainty = float(kelly_row.iloc[0].get("Uncertainty", 0.0)) if not kelly_row.empty else 0.0

    neg_days, rs_declining = _rs20d_info(recent_kellys, ticker)

    return {
        "current_price":            current_price,
        "alpha_rank":               int(kelly_row.iloc[0].get("rank", 9999)) if not kelly_row.empty else 9999,
        "uncertainty":              uncertainty,
        "rank_out50_streak":        _rank_out50_streak(history, ticker),
        "inst_sell_streak":         _inst_sell_streak(inst_df, ticker),
        "rsi":                      _rsi_14(prices_df, ticker),
        "rs_20d":                   float(kelly_row.iloc[0].get("RS_20d", 0)) if not kelly_row.empty else 0.0,
        "rs_20d_negative_days":     neg_days,
        "rs_20d_declining":         rs_declining,
        "alpha_20d_declining_days": _alpha_declining_days(recent_kellys, ticker),
        "signal_quality_pct":       _sq_percentile(ticker, df_kelly),
        "new_buy_available":        bool(new_buy_tickers - {ticker}),
        "max_positions_full":       n_positions >= MAX_POSITIONS,
        "bearish_pattern":          bearish_map.get(ticker),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SimEngine:
    def __init__(self, state: SimState):
        self.cash:         float                  = state.cash
        self.positions:    dict[str, EntryRecord] = {
            t: EntryRecord.from_dict(d) for t, d in state.positions.items()
        }
        self.equity_curve: list[dict]             = list(state.equity_curve)
        self._market_vals: dict[str, float]       = {}

    # ── Daily step ────────────────────────────────────────────────────────────

    def process_day(
        self,
        date:          str,
        prices:        dict[str, float],
        df_kelly:      pd.DataFrame,
        action_sigs:   dict,
        pattern_sigs:  dict,
        history:       list,
        inst_df:       pd.DataFrame,
        prices_df:     pd.DataFrame,
        recent_kellys: list[pd.DataFrame],
    ) -> list[dict]:
        """執行一個交易日。返回當日交易記錄清單。"""
        trades: list[dict] = []

        # 1. 市值更新
        self._update_market_values(prices)

        # 2. Trailing stop 往上更新 + hold_days 累加
        for ticker, pos in self.positions.items():
            p = prices.get(ticker)
            if p and p > 0:
                update_trailing_stop(pos, p)
            pos.hold_days += 1

        # 3. 輔助資料
        bearish_map: dict[str, str] = {
            s["stock_id"]: s["pattern_id"]
            for s in pattern_sigs.get("bearish_signals", [])
        }
        bullish_map: dict[str, dict] = {
            s["stock_id"]: s
            for s in pattern_sigs.get("signals", [])
        }
        regime = action_sigs.get("market_regime", "NORMAL")

        # 4. 計算合格進場候選（排除已持倉）
        candidates = self._compute_candidates(
            action_sigs, bullish_map, regime, prices, df_kelly,
        )
        new_buy_tickers = {c["ticker"] for c in candidates}

        # 5. 退場條件檢查
        layer4_candidates: list[tuple[str, ExitTrigger]] = []

        for ticker in list(self.positions.keys()):
            pos   = self.positions[ticker]
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue

            mdata = _build_market_data(
                ticker=ticker, position=pos,
                current_price=price, df_kelly=df_kelly,
                history=history, inst_df=inst_df, prices_df=prices_df,
                bearish_map=bearish_map, recent_kellys=recent_kellys,
                new_buy_tickers=new_buy_tickers,
                n_positions=len(self.positions),
            )

            triggers = check_exit_conditions(pos, mdata)
            if not triggers:
                continue

            top = triggers[0]
            if top.layer in (1, 2):
                t = self._execute_sell(ticker, price, date, top.condition, top.detail, "full")
                if t:
                    trades.append(t)
            elif top.layer == 3:
                t = self._execute_sell(ticker, price, date, top.condition, top.detail, "half")
                if t:
                    trades.append(t)
            elif top.layer == 4:
                layer4_candidates.append((ticker, top))

        # 6. Layer 4 換倉：找最差 SQ 持倉換出
        if layer4_candidates and candidates and len(self.positions) >= MAX_POSITIONS:
            layer4_candidates.sort(key=lambda x: _sq_percentile(x[0], df_kelly), reverse=True)
            worst_ticker, worst_trigger = layer4_candidates[0]
            price = prices.get(worst_ticker, 0.0)
            if price > 0:
                t = self._execute_sell(worst_ticker, price, date, worst_trigger.condition, worst_trigger.detail, "full")
                if t:
                    trades.append(t)

        # 7. 進場
        pv        = self._portfolio_value()
        deployed  = sum(self._market_vals.values())
        available = max(0.0, pv * (1 - CASH_BUFFER_PCT) - deployed)
        slots     = MAX_POSITIONS - len(self.positions)

        if slots > 0 and available >= pv * MIN_ALLOC_PCT:
            for cand in candidates:
                if slots <= 0 or available < pv * MIN_ALLOC_PCT:
                    break
                ticker = cand["ticker"]
                if ticker in self.positions:
                    continue
                price = prices.get(ticker)
                if not price or price <= 0:
                    continue
                alloc  = min(
                    float(cand.get("suggested_weight", MIN_ALLOC_PCT)) * pv,
                    pv * MAX_POS_PCT,
                    available,
                )
                if alloc < pv * MIN_ALLOC_PCT:
                    continue
                shares = int(alloc / price / 1000) * 1000
                if shares == 0:
                    shares = 100
                cost = price * shares * (1 + BUY_FEE)
                if cost > self.cash:
                    continue
                t = self._execute_buy(ticker, price, shares, date, cand)
                if t:
                    trades.append(t)
                    available -= price * shares
                    slots     -= 1

        # 8. 收盤後市值更新
        self._update_market_values(prices)

        # 9. 當日權益快照
        self.equity_curve.append(self._equity_snapshot(date))
        return trades

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_candidates(
        self,
        action_sigs: dict,
        bullish_map: dict[str, dict],
        regime:      str,
        prices:      dict[str, float],
        df_kelly:    pd.DataFrame,
    ) -> list[dict]:
        """
        從 buy_signals + pattern_scanner 計算合格進場候選清單。
        已排除持倉股與無今日報價者。
        """
        threshold = entry_threshold(regime)
        result: list[dict] = []

        for sig in action_sigs.get("buy_signals", []):
            ticker = str(sig.get("ticker", ""))
            if not ticker or ticker in self.positions:
                continue
            if not prices.get(ticker, 0):
                continue

            scanner_score = int(sig.get("score", 0))
            pat_sig = bullish_map.get(ticker)
            pat_score = int(pat_sig["score"]) if pat_sig else None
            dual_confirm = bool(pat_sig.get("dual_confirm", False)) if pat_sig else False

            # 優先從 df_kelly 取 rank
            kelly_row = df_kelly[df_kelly["Ticker"].astype(str) == ticker]
            alpha_rank = (int(kelly_row.iloc[0]["rank"]) if not kelly_row.empty and "rank" in kelly_row.columns
                          else 9999)

            total, breakdown = compute_entry_score(scanner_score, pat_score, alpha_rank, dual_confirm)
            if total < threshold:
                continue

            result.append({
                "ticker":           ticker,
                "combined_score":   total,
                "score_breakdown":  breakdown,
                "suggested_weight": float(sig.get("suggested_weight", MIN_ALLOC_PCT)),
                "scanner_signal":   sig,
                "pattern_signal":   pat_sig,
                "main_conditions":  extract_main_conditions({
                    **sig,
                    **({"pattern_id": pat_sig["pattern_id"]} if pat_sig else {}),
                }),
                "uncertainty":      float(sig.get("uncertainty", 0.0)),
                "alpha_rank":       alpha_rank,
            })

        result.sort(key=lambda x: -x["combined_score"])
        return result

    def _update_market_values(self, prices: dict[str, float]) -> None:
        for ticker, pos in self.positions.items():
            p = prices.get(ticker)
            if p and p > 0:
                self._market_vals[ticker] = p * pos.shares

    def _portfolio_value(self) -> float:
        return self.cash + sum(self._market_vals.values())

    def _equity_snapshot(self, date: str) -> dict:
        pv   = self._portfolio_value()
        prev = self.equity_curve[-1]["equity"] if self.equity_curve else INITIAL_CAPITAL
        return {
            "date":             date,
            "equity":           int(round(pv)),
            "cash":             int(round(self.cash)),
            "positions_value":  int(round(sum(self._market_vals.values()))),
            "n_positions":      len(self.positions),
            "daily_return_pct": round((pv / prev - 1) * 100, 3) if prev > 0 else 0.0,
        }

    def _execute_sell(
        self,
        ticker:  str,
        price:   float,
        date:    str,
        reason:  str,
        detail:  str,
        action:  str,  # "full" or "half"
    ) -> Optional[dict]:
        if ticker not in self.positions:
            return None
        pos = self.positions[ticker]

        # 計算出售股數
        if action == "half":
            shares_sell = max(100, (pos.shares // 2 // 100) * 100)
            if shares_sell >= pos.shares:
                action = "full"
                shares_sell = pos.shares
        else:
            shares_sell = pos.shares

        # 更新持倉或刪除
        if action == "full":
            del self.positions[ticker]
            self._market_vals.pop(ticker, None)
        else:
            pos.shares    -= shares_sell
            pos.cost_total = pos.entry_price * pos.shares * (1 + BUY_FEE)
            self._market_vals[ticker] = price * pos.shares

        proceeds  = price * shares_sell * (1 - SELL_FEE)
        cost_sold = pos.entry_price * shares_sell * (1 + BUY_FEE)
        pnl       = proceeds - cost_sold
        pnl_pct   = pnl / cost_sold * 100 if cost_sold > 0 else 0.0
        self.cash += proceeds

        logger.info(
            f"  SELL[{action}] {ticker} ×{shares_sell} @{price:.2f} "
            f"| P&L={pnl:+,.0f} ({pnl_pct:+.1f}%) | {reason}"
        )
        return {
            "date":       date,
            "action":     f"SELL_{action.upper()}",
            "ticker":     ticker,
            "price":      round(price, 2),
            "shares":     shares_sell,
            "value":      round(price * shares_sell, 0),
            "fee":        round(price * shares_sell * SELL_FEE, 0),
            "pnl":        round(pnl, 0),
            "return_pct": round(pnl_pct, 2),
            "reason":     reason,
            "detail":     detail,
            "hold_days":  pos.hold_days,
        }

    def _execute_buy(
        self,
        ticker: str,
        price:  float,
        shares: int,
        date:   str,
        cand:   dict,
    ) -> Optional[dict]:
        cost = price * shares * (1 + BUY_FEE)
        if cost > self.cash:
            return None
        self.cash -= cost

        pat = cand.get("pattern_signal")
        failure_stop: Optional[float] = (
            float(pat.get("failure_stop") or pat.get("stop_loss") or 0) or None
            if pat else None
        )

        pos = EntryRecord(
            ticker               = ticker,
            entry_date           = date,
            entry_price          = price,
            entry_score          = int(cand["combined_score"]),
            main_conditions      = cand["main_conditions"],
            pattern_id           = pat["pattern_id"] if pat else None,
            pattern_failure_stop = failure_stop,
            entry_uncertainty    = float(cand.get("uncertainty", 0.0)),
            peak_return_pct      = 0.0,
            trailing_stop_price  = price * (1 + get_trailing_stop_pct(0.0)),  # entry -5%
            hold_days            = 0,
            shares               = shares,
            cost_total           = cost,
            entry_alpha_rank     = int(cand.get("alpha_rank", 9999)),
        )
        self.positions[ticker] = pos
        self._market_vals[ticker] = price * shares

        logger.info(
            f"  BUY  {ticker} ×{shares} @{price:.2f} "
            f"| score={pos.entry_score} pattern={pos.pattern_id or '無'}"
        )
        return {
            "date":            date,
            "action":          "BUY",
            "ticker":          ticker,
            "price":           round(price, 2),
            "shares":          shares,
            "value":           round(price * shares, 0),
            "fee":             round(price * shares * BUY_FEE, 0),
            "pnl":             0,
            "return_pct":      0.0,
            "reason":          (
                f"進場 score={pos.entry_score} "
                f"主因={pos.main_conditions} "
                f"pattern={pos.pattern_id or '無'}"
            ),
            "entry_score":     pos.entry_score,
            "score_breakdown": cand.get("score_breakdown", {}),
        }

    def to_state(self, date: str) -> SimState:
        return SimState(
            date         = date,
            cash         = self.cash,
            positions    = {t: p.to_dict() for t, p in self.positions.items()},
            equity_curve = self.equity_curve,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_kelly(df: pd.DataFrame) -> pd.DataFrame:
    if "Sharpe_Score" in df.columns and "Signal_Quality" not in df.columns:
        df = df.rename(columns={"Sharpe_Score": "Signal_Quality"})
    if "rank" not in df.columns:
        sort_col = "Exp_Alpha_20d" if "Exp_Alpha_20d" in df.columns else "Signal_Quality"
        df = (df.sort_values(sort_col, ascending=False, na_position="last")
              .reset_index(drop=True))
        df["rank"] = df.index + 1
    return df


def _load_recent_kellys(results_dir: Path, before_date: str, n: int = 5) -> list[pd.DataFrame]:
    arch_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name < before_date
        and (d / "df_kelly.csv").exists()
    ], reverse=True)[:n]

    result: list[pd.DataFrame] = []
    for d in arch_dirs:
        try:
            result.append(_norm_kelly(pd.read_csv(d / "df_kelly.csv", dtype={"Ticker": str})))
        except Exception:
            pass
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Points
# ═══════════════════════════════════════════════════════════════════════════════

def run_daily_update(
    date:        str,
    results_dir: Path = RESULTS_DIR,
    data_dir:    Path = PROCESSED_DIR,
    state_path:  Path = SIM_STATE_PATH,
    trades_path: Path = SIM_TRADES_PATH,
) -> dict:
    """
    日更入口：讀取昨日 sim_state.json → 處理今日 → 寫回。
    """
    # 載入狀態
    state  = (SimState.from_dict(_load_json_safe(state_path))
              if state_path.exists() else SimState.fresh())
    engine = SimEngine(state)

    # 今日資料
    arch_dir = results_dir / date

    def _pick(fname: str) -> Path:
        """優先從 archive/date/ 取，否則從 results/ 取。"""
        p = arch_dir / fname
        return p if p.exists() else results_dir / fname

    kelly_path = _pick("df_kelly.csv")
    if not kelly_path.exists():
        logger.warning(f"[sim_v3] df_kelly.csv not found for {date}")
        return {}

    df_kelly     = _norm_kelly(pd.read_csv(kelly_path, dtype={"Ticker": str}))
    action_sigs  = _load_json_safe(_pick("action_signals.json"))
    pattern_sigs = _load_json_safe(results_dir / "pattern_signals.json")
    history      = _load_json_safe(results_dir / "history_index.json").get("history", [])

    # 價格資料
    prices_path = data_dir / "prices_raw.parquet"
    prices_df   = pd.DataFrame()
    prices: dict[str, float] = {}
    if prices_path.exists():
        pr = pd.read_parquet(prices_path, columns=["Date", "stock_id", "Close"])
        pr["Date"] = pd.to_datetime(pr["Date"])
        prices_df = pr[pr["Date"] <= pd.Timestamp(date)].copy()
        today_df  = prices_df[prices_df["Date"] == prices_df["Date"].max()]
        prices    = dict(zip(today_df["stock_id"].astype(str), today_df["Close"]))

    # 機構資料
    inst_df   = pd.DataFrame()
    inst_path = data_dir / "institutional_raw.parquet"
    if inst_path.exists():
        try:
            inst_raw = pd.read_parquet(inst_path)
            inst_raw["Date"] = pd.to_datetime(inst_raw["Date"])
            inst_df  = inst_raw[inst_raw["Date"] <= pd.Timestamp(date)].tail(10 * 3000)
        except Exception:
            pass

    recent_kellys = _load_recent_kellys(results_dir, date, n=5)

    trades = engine.process_day(
        date=date, prices=prices, df_kelly=df_kelly,
        action_sigs=action_sigs, pattern_sigs=pattern_sigs,
        history=history, inst_df=inst_df, prices_df=prices_df,
        recent_kellys=recent_kellys,
    )

    # 儲存狀態
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(engine.to_state(date).to_dict(), f, ensure_ascii=False, indent=2)

    # Append 交易紀錄
    if trades:
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trades_path, "a", encoding="utf-8") as f:
            for t in trades:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    pv      = engine._portfolio_value()
    summary = {
        "date":       date,
        "equity":     int(round(pv)),
        "cash":       int(round(engine.cash)),
        "positions":  len(engine.positions),
        "trades":     len(trades),
        "return_pct": round((pv / INITIAL_CAPITAL - 1) * 100, 2),
    }
    logger.info(
        f"[sim_v3] {date} | equity={pv:,.0f} | "
        f"pos={len(engine.positions)} | trades={len(trades)}"
    )
    return summary


def run_backtest(
    results_dir: Path = RESULTS_DIR,
    data_dir:    Path = PROCESSED_DIR,
    state_path:  Path = SIM_STATE_PATH,
    trades_path: Path = SIM_TRADES_PATH,
    reset:       bool = True,
) -> dict:
    """
    全量回放入口：依序處理所有 results/YYYY-MM-DD/ 目錄。
    reset=True 時清空既有狀態與交易紀錄重新開始。
    """
    arch_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and len(d.name) == 10 and (d / "df_kelly.csv").exists()
    ])
    if not arch_dirs:
        logger.warning("[sim_v3] No archive directories found")
        return {}

    if reset:
        state = SimState.fresh()
        if trades_path.exists():
            trades_path.unlink()
    else:
        state = (SimState.from_dict(_load_json_safe(state_path))
                 if state_path.exists() else SimState.fresh())

    engine = SimEngine(state)

    # 一次性載入大型資料
    prices_path = data_dir / "prices_raw.parquet"
    if not prices_path.exists():
        logger.error(f"[sim_v3] prices_raw.parquet not found: {prices_path}")
        return {}
    pr_all = pd.read_parquet(prices_path, columns=["Date", "stock_id", "Close"])
    pr_all["Date"] = pd.to_datetime(pr_all["Date"])

    inst_full = pd.DataFrame()
    inst_path = data_dir / "institutional_raw.parquet"
    if inst_path.exists():
        try:
            inst_full = pd.read_parquet(inst_path)
            inst_full["Date"] = pd.to_datetime(inst_full["Date"])
        except Exception:
            pass

    all_trades: list[dict] = []
    logger.info(
        f"[sim_v3] Backtest: {len(arch_dirs)} dates "
        f"({arch_dirs[0].name} → {arch_dirs[-1].name})"
    )

    for arch_dir in arch_dirs:
        date = arch_dir.name
        try:
            df_kelly = _norm_kelly(pd.read_csv(arch_dir / "df_kelly.csv", dtype={"Ticker": str}))
        except Exception:
            continue

        action_sigs  = _load_json_safe(arch_dir / "action_signals.json")
        pattern_sigs = _load_json_safe(results_dir / "pattern_signals.json")
        history      = _load_json_safe(results_dir / "history_index.json").get("history", [])

        cutoff   = pd.Timestamp(date)
        pr_day   = pr_all[pr_all["Date"] <= cutoff]
        today_df = pr_day[pr_day["Date"] == pr_day["Date"].max()]
        prices   = dict(zip(today_df["stock_id"].astype(str), today_df["Close"]))

        inst_day = (inst_full[inst_full["Date"] <= cutoff].copy()
                    if not inst_full.empty else pd.DataFrame())
        recent_kellys = _load_recent_kellys(results_dir, date, n=5)

        day_trades = engine.process_day(
            date=date, prices=prices, df_kelly=df_kelly,
            action_sigs=action_sigs, pattern_sigs=pattern_sigs,
            history=history, inst_df=inst_day, prices_df=pr_day,
            recent_kellys=recent_kellys,
        )
        all_trades.extend(day_trades)
        logger.info(f"[sim_v3] [{date}] pos={len(engine.positions)} trades={len(day_trades)}")

    # 儲存最終狀態
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(engine.to_state(arch_dirs[-1].name).to_dict(), f, ensure_ascii=False, indent=2)

    if all_trades:
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trades_path, "w", encoding="utf-8") as f:
            for t in all_trades:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # 績效彙總
    pv        = engine._portfolio_value()
    total_ret = (pv / INITIAL_CAPITAL - 1) * 100
    sells     = [t for t in all_trades if t["action"].startswith("SELL")]
    wins      = [t for t in sells if t.get("pnl", 0) > 0]
    win_rate  = len(wins) / len(sells) * 100 if sells else 0.0

    peak, max_dd = float(INITIAL_CAPITAL), 0.0
    for pt in engine.equity_curve:
        eq = pt["equity"]
        peak   = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak)

    summary = {
        "period":             f"{arch_dirs[0].name} → {arch_dirs[-1].name}",
        "trading_days":       len(engine.equity_curve),
        "total_return_pct":   round(total_ret, 2),
        "max_drawdown_pct":   round(max_dd * 100, 2),
        "win_rate_pct":       round(win_rate, 1),
        "total_trades":       len(sells),
        "final_equity":       int(round(pv)),
        "final_positions":    len(engine.positions),
        "final_cash":         int(round(engine.cash)),
    }
    logger.info(
        f"[sim_v3] Done | return={total_ret:+.2f}% dd={max_dd*100:.2f}% "
        f"win={win_rate:.1f}% trades={len(sells)}"
    )
    return summary
