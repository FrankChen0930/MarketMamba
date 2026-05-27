"""
MarketMamba V6 — Scanner-Driven Investment Robot
=================================================
Uses the 4-condition weighted scanner signals (action_signals.json archives)
for entry/exit decisions instead of raw Signal_Quality ranking.

Entry:  Stock in buy_signals (score ≥ 55 normal / ≥ 70 cautious)
Exit:   Stock in exit_signals OR trailing stop triggered
Size:   proportional to suggested_weight × available capital (≤ MAX_POS_PCT)

Trailing Stop (ratcheting on peak P&L):
  Peak P&L < +5%   → hard stop at entry −5%
  Peak P&L ≥ +5%   → lock-in at entry +2%
  Peak P&L ≥ +10%  → lock-in at entry +6%
  Peak P&L ≥ +15%  → lock-in at entry +10%

Input:
    V6/results/YYYY-MM-DD/action_signals.json  (daily scanner archives)
    Data/processed_v6/prices_raw.parquet       (for mark-to-market)

Output:
    V6/results/scanner_backtest.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000   # NT$100萬
MAX_POSITIONS   = 10          # 最多同時持有（Scanner 更嚴格篩選）
MAX_POS_PCT     = 0.15        # 單股最多 15%
MIN_POS_PCT     = 0.02        # 至少 2% 才建倉
CASH_RESERVE    = 0.05        # 維持 5% 現金緩衝
BUY_FEE         = 0.0015      # 買進手續費 0.15%
SELL_FEE        = 0.0045      # 賣出：手續費 0.15% + 證交稅 0.30%

# Trailing stop: (peak_pnl_threshold, stop_level_relative_to_entry)
TRAILING_TIERS = [
    (0.15,  0.10),   # Peak P&L ≥ +15% → stop at entry +10%
    (0.10,  0.06),   # Peak P&L ≥ +10% → stop at entry +6%
    (0.05,  0.02),   # Peak P&L ≥  +5% → stop at entry +2%
    (-1.0, -0.05),   # Default          → stop at entry −5%
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ScannerHeldPosition:
    ticker:       str
    entry_date:   str
    entry_price:  float
    shares:       int
    cost_total:   float
    entry_score:  int  = 0
    conditions_met: int = 0
    rank_stability_met:    bool = False
    high_confidence_met:   bool = False
    relative_low_met:      bool = False
    institutional_buy_met: bool = False
    peak_pnl_pct:  float = 0.0   # Highest P&L ratio ever reached (for trailing stop)


@dataclass
class ScannerTradeRecord:
    date:    str
    action:  str    # BUY / SELL
    ticker:  str
    price:   float
    shares:  int
    value:   float
    fee:     float
    pnl:     float
    reason:  str


# ── Trailing stop logic ───────────────────────────────────────────────────────

def _get_stop_pct(peak_pnl_pct: float) -> float:
    """Return the stop level (relative to entry) for a given peak P&L ratio."""
    for threshold, stop in TRAILING_TIERS:
        if peak_pnl_pct >= threshold:
            return stop
    return -0.05


def _check_trailing_stop(
    pos: ScannerHeldPosition, current_price: float
) -> tuple[bool, str]:
    """
    Check if trailing stop triggered. Updates pos.peak_pnl_pct in-place.
    Returns (triggered, reason_string).
    """
    if pos.entry_price <= 0:
        return False, ""

    pnl_pct = (current_price - pos.entry_price) / pos.entry_price

    # Ratchet up peak (never decreases)
    if pnl_pct > pos.peak_pnl_pct:
        pos.peak_pnl_pct = pnl_pct

    stop_pct   = _get_stop_pct(pos.peak_pnl_pct)
    stop_price = pos.entry_price * (1 + stop_pct)

    if current_price <= stop_price:
        return True, (
            f"Trailing Stop "
            f"(峰值{pos.peak_pnl_pct:+.1%}→止損{stop_pct:+.1%}, "
            f"跌破 {stop_price:.2f})"
        )
    return False, ""


# ── Scanner Robot ─────────────────────────────────────────────────────────────

class ScannerRobot:
    """Portfolio simulator driven by daily scanner 4-condition signals."""

    def __init__(self) -> None:
        self.cash: float = float(INITIAL_CAPITAL)
        self.holdings:              dict[str, ScannerHeldPosition] = {}
        self._pos_market_values:   dict[str, float] = {}
        self._pos_current_prices:  dict[str, float] = {}
        self.equity_curve:         list[dict]       = []
        self.transactions:         list[ScannerTradeRecord] = []
        self._hold_days:           dict[str, int]   = {}
        self._watchlist_history:   dict[str, int]   = {}

    @property
    def portfolio_value(self) -> float:
        return self.cash + sum(self._pos_market_values.values())

    # ── Main daily step ───────────────────────────────────────────────────────

    def process_day(
        self,
        date: str,
        scan_result: dict,
        prices_today: dict[str, float],
    ) -> None:
        # 1. Mark-to-market
        self._update_market_values(prices_today)

        # 2. Increment hold-day counters
        for t in list(self.holdings):
            self._hold_days[t] = self._hold_days.get(t, 0) + 1

        # 3. Check trailing stops (before processing scanner exits)
        self._check_trailing_stops(date, prices_today)

        # 4. Process scanner EXIT signals
        self._process_exits(date, scan_result, prices_today)

        # 5. Process scanner BUY signals
        self._process_entries(date, scan_result, prices_today)

        # 6. Final mark-to-market
        self._update_market_values(prices_today)

        # 7. Update watchlist tracker
        watch_tickers = {s['ticker'] for s in scan_result.get('watch_list', [])}
        new_wl: dict[str, int] = {}
        for t in watch_tickers:
            if t not in self.holdings:
                new_wl[t] = self._watchlist_history.get(t, 0) + 1
        self._watchlist_history = new_wl

        # 8. Snapshot equity curve
        pv_now   = self.portfolio_value
        prev_eq  = self.equity_curve[-1]["equity"] if self.equity_curve else INITIAL_CAPITAL
        daily_ret = (pv_now / prev_eq - 1.0) * 100 if prev_eq > 0 else 0.0
        self.equity_curve.append({
            "date":             date,
            "equity":           int(round(pv_now)),
            "cash":             int(round(self.cash)),
            "positions_value":  int(round(sum(self._pos_market_values.values()))),
            "daily_return_pct": round(daily_ret, 3),
            "n_positions":      len(self.holdings),
        })

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_market_values(self, prices: dict[str, float]) -> None:
        for ticker, pos in self.holdings.items():
            p = prices.get(ticker)
            if p and p > 0:
                self._pos_market_values[ticker]  = p * pos.shares
                self._pos_current_prices[ticker] = p

    def _check_trailing_stops(self, date: str, prices: dict[str, float]) -> None:
        for ticker in list(self.holdings):
            price = prices.get(ticker)
            if not price or price <= 0:
                continue
            triggered, reason = _check_trailing_stop(self.holdings[ticker], price)
            if triggered:
                self._execute_sell(ticker, price, date, reason)

    def _process_exits(
        self, date: str, scan_result: dict, prices: dict[str, float]
    ) -> None:
        exit_tickers = {s['ticker'] for s in scan_result.get('exit_signals', [])}
        for ticker in list(self.holdings):
            if ticker in exit_tickers:
                price = prices.get(ticker) or self.holdings[ticker].entry_price
                self._execute_sell(ticker, price, date, "Scanner EXIT 訊號")

    def _process_entries(
        self, date: str, scan_result: dict, prices: dict[str, float]
    ) -> None:
        buy_signals = sorted(
            scan_result.get('buy_signals', []),
            key=lambda x: -x.get('score', 0),
        )
        if not buy_signals:
            return

        pv        = self.portfolio_value
        max_deploy = pv * (1.0 - CASH_RESERVE)
        deployed  = sum(self._pos_market_values.values())
        available = max(0.0, max_deploy - deployed)
        slots     = MAX_POSITIONS - len(self.holdings)

        if slots <= 0 or available < pv * MIN_POS_PCT:
            return

        # Candidates: buy_signals not already held, up to available slots
        candidates = [s for s in buy_signals if s['ticker'] not in self.holdings][:slots]
        if not candidates:
            return

        # Normalize weights across candidates
        total_w = sum(max(s.get('suggested_weight', 0.01), 0.01) for s in candidates)

        for signal in candidates:
            ticker = signal['ticker']
            price  = prices.get(ticker)
            if not price or price <= 0:
                continue

            w         = max(signal.get('suggested_weight', 0.01), 0.01) / total_w
            alloc_pct = min(w * (available / pv), MAX_POS_PCT)
            if alloc_pct < MIN_POS_PCT:
                alloc_pct = MIN_POS_PCT

            alloc_val = pv * alloc_pct
            if self.cash * (1 - BUY_FEE) < alloc_val:
                alloc_val = self.cash * (1 - BUY_FEE) * 0.95

            shares = int(alloc_val / price / 1000) * 1000
            if shares == 0:
                shares = 1000   # 最小 1 張

            cost = price * shares * (1 + BUY_FEE)
            if cost > self.cash:
                continue

            self._execute_buy(ticker, price, shares, date, signal)

            # Recalculate available after this buy
            pv        = self.portfolio_value
            deployed  = sum(self._pos_market_values.values())
            available = max(0.0, pv * (1.0 - CASH_RESERVE) - deployed)
            if available < pv * MIN_POS_PCT:
                break

    def _execute_buy(
        self, ticker: str, price: float, shares: int, date: str, signal: dict
    ) -> None:
        cost = price * shares * (1 + BUY_FEE)
        self.cash -= cost

        rs = signal.get('rank_stability',    {})
        hc = signal.get('high_confidence',   {})
        rl = signal.get('relative_low',      {})
        ib = signal.get('institutional_buy', {})

        self._hold_days[ticker] = 0
        self.holdings[ticker]   = ScannerHeldPosition(
            ticker        = ticker,
            entry_date    = date,
            entry_price   = price,
            shares        = shares,
            cost_total    = cost,
            entry_score   = int(signal.get('score', 0)),
            conditions_met = int(signal.get('conditions_met', 0)),
            rank_stability_met    = bool(rs.get('met', False)),
            high_confidence_met   = bool(hc.get('met', False)),
            relative_low_met      = bool(rl.get('met', False)),
            institutional_buy_met = bool(ib.get('met', False)),
        )
        self._pos_market_values[ticker]  = price * shares
        self._pos_current_prices[ticker] = price

        self.transactions.append(ScannerTradeRecord(
            date   = date,
            action = "BUY",
            ticker = ticker,
            price  = round(price, 2),
            shares = shares,
            value  = round(price * shares, 0),
            fee    = round(price * shares * BUY_FEE, 0),
            pnl    = 0.0,
            reason = (f"Scanner BUY (score={signal.get('score', 0)}, "
                      f"{signal.get('conditions_met', 0)}/4條件)"),
        ))
        logger.info(
            f"  BUY  {ticker} × {shares} @ {price:.2f} | "
            f"score={signal.get('score', 0)} {signal.get('conditions_met', 0)}/4"
        )

    def _execute_sell(
        self, ticker: str, price: float, date: str, reason: str
    ) -> None:
        if ticker not in self.holdings:
            return
        pos      = self.holdings.pop(ticker)
        proceeds = price * pos.shares * (1 - SELL_FEE)
        pnl      = proceeds - pos.cost_total
        self.cash += proceeds
        self._pos_market_values.pop(ticker, None)
        self._pos_current_prices.pop(ticker, None)
        self._hold_days.pop(ticker, None)

        self.transactions.append(ScannerTradeRecord(
            date   = date,
            action = "SELL",
            ticker = ticker,
            price  = round(price, 2),
            shares = pos.shares,
            value  = round(price * pos.shares, 0),
            fee    = round(price * pos.shares * SELL_FEE, 0),
            pnl    = round(pnl, 0),
            reason = reason,
        ))
        pnl_pct = pnl / pos.cost_total * 100 if pos.cost_total > 0 else 0
        logger.info(
            f"  SELL {ticker} × {pos.shares} @ {price:.2f} | "
            f"P&L={pnl:+,.0f} ({pnl_pct:+.1f}%) | {reason}"
        )

    # ── Output builder ────────────────────────────────────────────────────────

    def build_output(
        self,
        date:            str,
        prices_latest:   dict[str, float],
        benchmark_curve: list[dict] | None,
        scan_meta:       dict | None = None,
    ) -> dict[str, Any]:
        pv = self.portfolio_value
        total_ret = (pv / INITIAL_CAPITAL - 1) * 100

        # Max drawdown
        peak, max_dd = float(INITIAL_CAPITAL), 0.0
        for pt in self.equity_curve:
            eq   = pt["equity"]
            peak = max(peak, eq)
            dd   = (eq - peak) / peak
            max_dd = min(max_dd, dd)

        # Annualised Sharpe
        daily_rets = [pt["daily_return_pct"] for pt in self.equity_curve[1:]]
        if len(daily_rets) >= 2:
            r = np.array(daily_rets) / 100
            sharpe = float(np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252))
        else:
            sharpe = 0.0

        sells    = [t for t in self.transactions if t.action == "SELL"]
        wins     = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0.0
        avg_hold = (
            float(np.mean([self._hold_days.get(t.ticker, 0) for t in sells]))
            if sells else 0.0
        )

        bench_ret = None
        if benchmark_curve and len(benchmark_curve) >= 2:
            bench_ret = round(benchmark_curve[-1]["level"] - 100.0, 2)

        # Holdings detail
        holdings_out = []
        for ticker, pos in sorted(
            self.holdings.items(),
            key=lambda x: -self._pos_market_values.get(x[0], 0),
        ):
            cur_price  = self._pos_current_prices.get(ticker, pos.entry_price)
            mv         = self._pos_market_values.get(ticker, pos.entry_price * pos.shares)
            pnl        = mv - pos.cost_total
            pnl_pct    = pnl / pos.cost_total * 100 if pos.cost_total > 0 else 0

            stop_pct   = _get_stop_pct(pos.peak_pnl_pct)
            stop_price = round(pos.entry_price * (1 + stop_pct), 2)

            holdings_out.append({
                "ticker":        ticker,
                "entry_date":    pos.entry_date,
                "entry_price":   round(pos.entry_price, 2),
                "current_price": round(cur_price, 2),
                "shares":        pos.shares,
                "market_value":  int(round(mv)),
                "weight_pct":    round(mv / pv * 100, 1) if pv > 0 else 0,
                "pnl":           int(round(pnl)),
                "pnl_pct":       round(pnl_pct, 2),
                "days_held":     self._hold_days.get(ticker, 0),
                "entry_score":   pos.entry_score,
                "conditions_met": pos.conditions_met,
                "rank_stability_met":    pos.rank_stability_met,
                "high_confidence_met":   pos.high_confidence_met,
                "relative_low_met":      pos.relative_low_met,
                "institutional_buy_met": pos.institutional_buy_met,
                "peak_pnl_pct":          round(pos.peak_pnl_pct * 100, 2),
                "trailing_stop_pct":     round(stop_pct * 100, 1),
                "trailing_stop_price":   stop_price,
            })

        # Watchlist (scanner's watch_list, tracked day count)
        watchlist_out = [
            {"ticker": t, "days_in_watch": d}
            for t, d in sorted(
                self._watchlist_history.items(), key=lambda x: -x[1]
            )[:10]
        ]

        return {
            "robot_type":      "scanner",
            "generated_date":  date,
            "initial_capital": INITIAL_CAPITAL,
            "period_start":    self.equity_curve[0]["date"] if self.equity_curve else date,
            "period_end":      date,
            "trading_days":    len(self.equity_curve),
            "scanner_meta":    scan_meta or {},
            "summary": {
                "total_return_pct":     round(total_ret, 2),
                "benchmark_return_pct": bench_ret,
                "excess_return_pct":    round(total_ret - bench_ret, 2) if bench_ret is not None else None,
                "max_drawdown_pct":     round(max_dd * 100, 2),
                "sharpe":              round(sharpe, 2),
                "win_rate_pct":        round(win_rate, 1),
                "total_trades":        len(sells),
                "avg_hold_days":       round(avg_hold, 1),
                "current_positions":   len(self.holdings),
                "cash":                int(round(self.cash)),
                "portfolio_value":     int(round(pv)),
                "deployed_pct":        round(
                    sum(self._pos_market_values.values()) / pv * 100, 1
                ) if pv > 0 else 0.0,
            },
            "current_holdings": holdings_out,
            "watchlist":        watchlist_out,
            "equity_curve":     self.equity_curve,
            "benchmark_curve":  benchmark_curve or [],
            "transactions":     [asdict(t) for t in self.transactions],
        }


# ── Benchmark (same pattern as sim_engine_v2) ────────────────────────────────

def _fetch_twii_benchmark(start: str, end: str) -> list[dict]:
    try:
        import yfinance as yf
        df = yf.download("^TWII", start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return []
        df = df[["Close"]].dropna().reset_index()
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        base = float(df.iloc[0]["close"])
        return [
            {"date": r["date"], "level": round(float(r["close"]) / base * 100, 3)}
            for _, r in df.iterrows()
        ]
    except Exception as e:
        logger.warning(f"TWII benchmark download failed: {e}")
        return []


# ── Main entry points ─────────────────────────────────────────────────────────

def run_scanner_backtest(
    results_dir: Path,
    prices_df:   pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Full rebuild from all YYYY-MM-DD/action_signals.json archives."""
    archive_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and len(d.name) == 10
        and (d / "action_signals.json").exists()
    ])

    if not archive_dirs:
        logger.warning("No action_signals.json archives found — scanner backtest skipped")
        return {}

    logger.info(
        f"Scanner backtest: {len(archive_dirs)} archive dates "
        f"({archive_dirs[0].name} → {archive_dirs[-1].name})"
    )

    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.strftime("%Y-%m-%d")
    price_wide = prices_df.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    ).sort_index().ffill()

    start_date = archive_dirs[0].name
    end_date   = archive_dirs[-1].name
    benchmark  = _fetch_twii_benchmark(start_date, end_date)

    robot = ScannerRobot()
    last_scan: dict = {}

    for arch_dir in archive_dirs:
        date = arch_dir.name
        try:
            with open(arch_dir / "action_signals.json", encoding="utf-8") as f:
                scan_result = json.load(f)
            last_scan = scan_result
        except Exception as e:
            logger.warning(f"  [{date}] Cannot read action_signals.json: {e}")
            continue

        prices_today = (
            price_wide.loc[date].dropna().to_dict()
            if date in price_wide.index else {}
        )
        logger.info(
            f"[{date}] Scanner: "
            f"{len(scan_result.get('buy_signals', []))} BUY "
            f"{len(scan_result.get('exit_signals', []))} EXIT "
            f"(holdings={len(robot.holdings)})"
        )
        robot.process_day(date, scan_result, prices_today)

    latest_prices = (
        price_wide.loc[end_date].dropna().to_dict()
        if end_date in price_wide.index else {}
    )
    scan_meta = _build_scan_meta(last_scan, end_date)
    result = robot.build_output(end_date, latest_prices, benchmark, scan_meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    logger.info(
        f"✅ Scanner backtest saved → {output_path.name}  "
        f"{result['trading_days']} days  "
        f"return={s['total_return_pct']:+.2f}%  "
        f"sharpe={s['sharpe']:.2f}  "
        f"positions={s['current_positions']}"
    )
    return result


def run_scanner_daily_update(
    date:        str,
    prices_df:   pd.DataFrame,
    output_path: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Incremental daily update. Falls back to full rebuild if output doesn't exist."""
    if not output_path.exists():
        return run_scanner_backtest(results_dir, prices_df, output_path)

    scan_path = results_dir / date / "action_signals.json"
    if not scan_path.exists():
        logger.warning(f"No action_signals.json archive for {date} — skipping scanner update")
        return {}

    with open(scan_path, encoding="utf-8") as f:
        scan_result = json.load(f)

    with open(output_path, encoding="utf-8") as f:
        existing = json.load(f)

    robot = _restore_scanner_robot(existing)

    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.strftime("%Y-%m-%d")
    price_wide = prices_df.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    ).sort_index().ffill()
    prices_today = (
        price_wide.loc[date].dropna().to_dict()
        if date in price_wide.index else {}
    )

    robot.process_day(date, scan_result, prices_today)

    # Extend benchmark by one day
    old_bench = existing.get("benchmark_curve", [])
    if old_bench:
        new_bench = _fetch_twii_benchmark(old_bench[-1]["date"], date)
        if new_bench and len(new_bench) > 1:
            base_level = old_bench[-1]["level"]
            base_raw   = new_bench[0]["level"]
            extended   = old_bench[:]
            for pt in new_bench[1:]:
                extended.append({
                    "date":  pt["date"],
                    "level": round(pt["level"] / base_raw * base_level, 3),
                })
        else:
            extended = old_bench
    else:
        extended = []

    scan_meta = _build_scan_meta(scan_result, date)
    result    = robot.build_output(date, prices_today, extended, scan_meta)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    logger.info(
        f"✅ Scanner daily update → {output_path.name}  "
        f"return={s['total_return_pct']:+.2f}%  "
        f"positions={s['current_positions']}  "
        f"deployed={s['deployed_pct']:.1f}%"
    )
    return result


def _build_scan_meta(scan_result: dict, fallback_date: str) -> dict:
    return {
        "market_regime":   scan_result.get("market_regime", "NORMAL"),
        "entry_threshold": scan_result.get("entry_threshold", "≥55分"),
        "scan_date":       scan_result.get("date", fallback_date),
        "n_buy_signals":   len(scan_result.get("buy_signals", [])),
        "n_exit_signals":  len(scan_result.get("exit_signals", [])),
    }


def _restore_scanner_robot(data: dict) -> ScannerRobot:
    """Reconstruct ScannerRobot from a saved JSON snapshot."""
    robot = ScannerRobot()
    robot.equity_curve = data.get("equity_curve", [])

    for h in data.get("current_holdings", []):
        ticker = str(h["ticker"])
        ep     = float(h["entry_price"])
        sh     = int(h["shares"])
        robot.holdings[ticker] = ScannerHeldPosition(
            ticker        = ticker,
            entry_date    = h["entry_date"],
            entry_price   = ep,
            shares        = sh,
            cost_total    = ep * sh * (1 + BUY_FEE),
            entry_score   = h.get("entry_score", 0),
            conditions_met = h.get("conditions_met", 0),
            rank_stability_met    = h.get("rank_stability_met",    False),
            high_confidence_met   = h.get("high_confidence_met",   False),
            relative_low_met      = h.get("relative_low_met",      False),
            institutional_buy_met = h.get("institutional_buy_met", False),
            peak_pnl_pct  = h.get("peak_pnl_pct", 0.0) / 100.0,  # stored as %, convert to ratio
        )
        robot._pos_market_values[ticker]  = h["market_value"]
        robot._pos_current_prices[ticker] = h["current_price"]
        robot._hold_days[ticker]          = h.get("days_held", 0)

    robot.cash = float(data.get("summary", {}).get("cash", INITIAL_CAPITAL))

    for w in data.get("watchlist", []):
        robot._watchlist_history[str(w["ticker"])] = w.get("days_in_watch", 1)

    for t in data.get("transactions", []):
        robot.transactions.append(ScannerTradeRecord(**t))

    return robot
