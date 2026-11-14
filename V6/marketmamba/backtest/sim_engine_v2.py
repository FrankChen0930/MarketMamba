"""
MarketMamba V6 — Signal-Driven Investment Robot (v2)
====================================================
Simulates a real portfolio with explicit BUY/EXIT decisions based on
Signal_Quality rankings from daily inference archives.

Entry rule  : enters Top-{ENTRY_RANK} by Signal_Quality, not already held
Exit rule   : drops below rank {EXIT_RANK}, or SQ < MIN_SQ, or held > MAX_HOLD_DAYS
Sizing      : Kelly-proportional weights from Suggested_Weight, capped at MAX_POS_PCT
Costs       : 0.15% buy fee + 0.45% sell fee (0.15% brokerage + 0.3% STT)
Benchmark   : ^TWII via yfinance, indexed to 100 at period start

Input:
    V6/results/YYYY-MM-DD/df_kelly.csv  (daily archives)
    Data/processed_v6/prices_raw.parquet

Output:
    V6/results/sim_backtest.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 1_000_000   # NT$100萬
MAX_POSITIONS    = 15          # 最多同時持有
ENTRY_RANK       = 20          # 進入 Top-20 才買
EXIT_RANK        = 35          # 跌出 Top-35 就賣
MIN_SQ           = 0.5         # Signal_Quality 低於此直接賣出
MAX_HOLD_DAYS    = 30          # 最多持有 30 個交易日（時間停損）
CASH_RESERVE     = 0.05        # 帳戶維持 5% 現金
MAX_POS_PCT      = 0.12        # 單股最多 12%
MIN_POS_PCT      = 0.02        # 至少 2% 才值得建倉
BUY_FEE          = 0.0015      # 買進手續費 0.15%
SELL_FEE         = 0.0045      # 賣出：手續費 0.15% + 證交稅 0.30%


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class HeldPosition:
    ticker:      str
    entry_date:  str
    entry_price: float
    shares:      int
    cost_total:  float   # 含手續費的總成本
    entry_sq:    float   # 入場時 Signal_Quality
    entry_rank:  int     # 入場時排名


@dataclass
class TradeRecord:
    date:    str
    action:  str    # BUY / SELL
    ticker:  str
    price:   float
    shares:  int
    value:   float
    fee:     float
    pnl:     float  # 0 for BUY; realized P&L for SELL
    reason:  str


# ── Robot ────────────────────────────────────────────────────────────────────

class SignalRobot:
    def __init__(self):
        self.cash: float = float(INITIAL_CAPITAL)
        self.holdings: dict[str, HeldPosition] = {}
        self._pos_market_values: dict[str, float] = {}
        self._pos_current_prices: dict[str, float] = {}
        self.equity_curve: list[dict] = []
        self.transactions: list[TradeRecord] = []
        self._hold_days: dict[str, int] = {}   # trading days held per ticker
        self._watchlist_history: dict[str, int] = {}  # days in watchlist

    @property
    def portfolio_value(self) -> float:
        return self.cash + sum(self._pos_market_values.values())

    # ── Core daily step ──────────────────────────────────────────────────────

    def process_day(
        self,
        date: str,
        df_day: pd.DataFrame,
        prices_today: dict[str, float],
    ) -> None:
        """Process one trading day: update positions → exit → enter → record."""
        # 1. Update market values for existing holdings
        self._update_market_values(prices_today)

        # 2. Increment hold-day counter for all current positions
        for t in list(self.holdings):
            self._hold_days[t] = self._hold_days.get(t, 0) + 1

        # 3. Build today's ranking map  {ticker: rank}
        df_ranked = (
            df_day
            .sort_values("Signal_Quality", ascending=False)
            .reset_index(drop=True)
        )
        df_ranked["rank_today"] = df_ranked.index + 1
        rank_map: dict[str, int] = dict(zip(
            df_ranked["Ticker"].astype(str),
            df_ranked["rank_today"]
        ))
        sq_map: dict[str, float] = dict(zip(
            df_ranked["Ticker"].astype(str),
            df_ranked["Signal_Quality"]
        ))

        # 4. Exits
        for ticker in list(self.holdings):
            rank = rank_map.get(ticker, 9999)
            sq   = sq_map.get(ticker, 0.0)
            days = self._hold_days.get(ticker, 0)

            should_exit = False
            reason = ""
            if rank > EXIT_RANK:
                should_exit = True
                reason = f"跌出Top-{EXIT_RANK} (rank={rank})"
            elif sq < MIN_SQ:
                should_exit = True
                reason = f"SQ過低 (SQ={sq:.2f})"
            elif days >= MAX_HOLD_DAYS:
                should_exit = True
                reason = f"超過{MAX_HOLD_DAYS}天時間停損"

            if should_exit:
                price = prices_today.get(ticker)
                if price and price > 0:
                    self._execute_sell(ticker, price, date, reason)
                else:
                    # 無價格資料：以成本清倉（標記虧損 = 費用）
                    pos = self.holdings[ticker]
                    fallback = pos.entry_price
                    self._execute_sell(ticker, fallback, date, reason + " [無價格,以成本清]")

        # 5. Entries
        # 可動用資金 = 組合總值 × (1 - 現金保留) - 已持倉市值
        pv = self.portfolio_value
        max_deploy = pv * (1.0 - CASH_RESERVE)
        deployed   = sum(self._pos_market_values.values())
        available  = max(0.0, max_deploy - deployed)

        n_positions = len(self.holdings)
        slots = MAX_POSITIONS - n_positions

        if slots > 0 and available > pv * MIN_POS_PCT:
            # 候選：Top-ENTRY_RANK 中未持有、有價格的股票
            candidates = df_ranked[
                (df_ranked["rank_today"] <= ENTRY_RANK) &
                (~df_ranked["Ticker"].astype(str).isin(self.holdings))
            ].head(slots).copy()

            # 計算每檔分配金額：按 Suggested_Weight 比例，上限 MAX_POS_PCT × pv
            if not candidates.empty and "Suggested_Weight" in candidates.columns:
                total_sw = candidates["Suggested_Weight"].clip(lower=0).sum()
                if total_sw > 0:
                    candidates["alloc_pct"] = (
                        candidates["Suggested_Weight"].clip(lower=0) / total_sw
                        * (available / pv)
                    ).clip(upper=MAX_POS_PCT)
                else:
                    candidates["alloc_pct"] = MAX_POS_PCT / len(candidates)
            else:
                if not candidates.empty:
                    candidates["alloc_pct"] = min(
                        available / pv / max(len(candidates), 1),
                        MAX_POS_PCT
                    )

            for _, row in candidates.iterrows():
                ticker = str(row["Ticker"])
                price  = prices_today.get(ticker)
                if not price or price <= 0:
                    continue

                alloc_value = pv * float(row.get("alloc_pct", MIN_POS_PCT))
                if alloc_value < pv * MIN_POS_PCT:
                    continue
                if self.cash * (1 - BUY_FEE) < alloc_value:
                    alloc_value = self.cash * (1 - BUY_FEE) * 0.95

                shares = int(alloc_value / price / 1000) * 1000   # 台股整張 1000 股
                if shares == 0:
                    shares = 100  # 零股最小單位（若交易所允許）

                cost = price * shares * (1 + BUY_FEE)
                if cost > self.cash:
                    continue

                self._execute_buy(
                    ticker=ticker,
                    price=price,
                    shares=shares,
                    date=date,
                    sq=float(row.get("Signal_Quality", 0)),
                    rank=int(row["rank_today"]),
                )

        # 6. Update market values again after trades
        self._update_market_values(prices_today)

        # 7. Update watchlist tracker (rank 21-35, not currently held)
        watching_now = set(
            str(r["Ticker"]) for _, r in df_ranked[
                (df_ranked["rank_today"] > ENTRY_RANK) &
                (df_ranked["rank_today"] <= EXIT_RANK) &
                (~df_ranked["Ticker"].astype(str).isin(self.holdings))
            ].iterrows()
        )
        new_wl = {}
        for t in watching_now:
            new_wl[t] = self._watchlist_history.get(t, 0) + 1
        self._watchlist_history = new_wl

        # 8. Snapshot equity curve
        pv_now = self.portfolio_value
        prev_eq = self.equity_curve[-1]["equity"] if self.equity_curve else INITIAL_CAPITAL
        daily_ret = (pv_now / prev_eq - 1.0) * 100 if prev_eq > 0 else 0.0

        self.equity_curve.append({
            "date":             date,
            "equity":           int(round(pv_now)),
            "cash":             int(round(self.cash)),
            "positions_value":  int(round(sum(self._pos_market_values.values()))),
            "daily_return_pct": round(daily_ret, 3),
            "n_positions":      len(self.holdings),
        })

    # ── Trade execution ──────────────────────────────────────────────────────

    def _execute_buy(
        self, ticker: str, price: float, shares: int, date: str, sq: float, rank: int
    ) -> None:
        cost = price * shares * (1 + BUY_FEE)
        self.cash -= cost
        self._hold_days[ticker] = 0
        self.holdings[ticker] = HeldPosition(
            ticker      = ticker,
            entry_date  = date,
            entry_price = price,
            shares      = shares,
            cost_total  = cost,
            entry_sq    = sq,
            entry_rank  = rank,
        )
        self._pos_market_values[ticker] = price * shares
        self._pos_current_prices[ticker] = price
        self.transactions.append(TradeRecord(
            date   = date,
            action = "BUY",
            ticker = ticker,
            price  = round(price, 2),
            shares = shares,
            value  = round(price * shares, 0),
            fee    = round(price * shares * BUY_FEE, 0),
            pnl    = 0.0,
            reason = f"Top-{ENTRY_RANK}進場 (rank={rank}, SQ={sq:.2f})",
        ))
        logger.info(f"  BUY  {ticker} × {shares} @ {price:.2f} | rank={rank} SQ={sq:.2f}")

    def _execute_sell(self, ticker: str, price: float, date: str, reason: str) -> None:
        if ticker not in self.holdings:
            return
        pos = self.holdings.pop(ticker)
        proceeds = price * pos.shares * (1 - SELL_FEE)
        pnl      = proceeds - pos.cost_total
        self.cash += proceeds
        self._pos_market_values.pop(ticker, None)
        self._pos_current_prices.pop(ticker, None)
        self._hold_days.pop(ticker, None)
        self.transactions.append(TradeRecord(
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
        pnl_pct = pnl / pos.cost_total * 100
        logger.info(
            f"  SELL {ticker} × {pos.shares} @ {price:.2f} "
            f"| P&L={pnl:+,.0f} ({pnl_pct:+.1f}%) | {reason}"
        )

    def _update_market_values(self, prices_today: dict[str, float]) -> None:
        for ticker, pos in self.holdings.items():
            p = prices_today.get(ticker)
            if p and p > 0:
                self._pos_market_values[ticker] = p * pos.shares
                self._pos_current_prices[ticker] = p

    # ── Serialisation ────────────────────────────────────────────────────────

    def build_output(
        self,
        date: str,
        df_today: pd.DataFrame,
        prices_latest: dict[str, float],
        benchmark_curve: list[dict] | None,
    ) -> dict[str, Any]:
        """Build the full output dict for saving to sim_backtest.json."""
        pv = self.portfolio_value
        total_ret = (pv / INITIAL_CAPITAL - 1) * 100

        # Max drawdown
        peak, max_dd = float(INITIAL_CAPITAL), 0.0
        for pt in self.equity_curve:
            eq   = pt["equity"]
            peak = max(peak, eq)
            dd   = (eq - peak) / peak
            max_dd = min(max_dd, dd)

        # Daily returns for Sharpe
        daily_rets = [pt["daily_return_pct"] for pt in self.equity_curve[1:]]
        if len(daily_rets) >= 2:
            r = np.array(daily_rets) / 100
            sharpe = float(np.mean(r) / (np.std(r) + 1e-9) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Realized trades stats
        sells = [t for t in self.transactions if t.action == "SELL"]
        wins  = [t for t in sells if t.pnl > 0]
        win_rate = len(wins) / len(sells) * 100 if sells else 0.0
        avg_hold = (
            np.mean([self._hold_days.get(t.ticker, 0) for t in sells])
            if sells else 0.0
        )

        # Benchmark return
        bench_ret = None
        if benchmark_curve and len(benchmark_curve) >= 2:
            bench_ret = round(benchmark_curve[-1]["level"] - 100.0, 2)

        # Current holdings detail
        df_ranked = (
            df_today.sort_values("Signal_Quality", ascending=False)
            .reset_index(drop=True)
        )
        df_ranked["rank_today"] = df_ranked.index + 1
        rank_map = dict(zip(df_ranked["Ticker"].astype(str), df_ranked["rank_today"]))
        sq_map   = dict(zip(df_ranked["Ticker"].astype(str), df_ranked["Signal_Quality"]))

        holdings_out = []
        for ticker, pos in sorted(
            self.holdings.items(),
            key=lambda x: -self._pos_market_values.get(x[0], 0)
        ):
            cur_price = self._pos_current_prices.get(ticker, pos.entry_price)
            mv        = self._pos_market_values.get(ticker, pos.entry_price * pos.shares)
            pnl       = mv - pos.cost_total
            pnl_pct   = pnl / pos.cost_total * 100 if pos.cost_total > 0 else 0
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
                "signal_quality": round(sq_map.get(ticker, pos.entry_sq), 2),
                "rank":          rank_map.get(ticker, 999),
            })

        # Watchlist
        watchlist_out = []
        for ticker, days_watching in sorted(
            self._watchlist_history.items(), key=lambda x: -x[1]
        )[:10]:
            row = df_today[df_today["Ticker"].astype(str) == ticker]
            if row.empty:
                continue
            r = row.iloc[0]
            watchlist_out.append({
                "ticker":         ticker,
                "rank":           int(rank_map.get(ticker, 999)),
                "signal_quality": round(float(r.get("Signal_Quality", 0)), 2),
                "close":          round(float(r.get("Close", 0)), 2),
                "days_in_watch":  days_watching,
            })

        return {
            "generated_date":   date,
            "initial_capital":  INITIAL_CAPITAL,
            "period_start":     self.equity_curve[0]["date"] if self.equity_curve else date,
            "period_end":       date,
            "trading_days":     len(self.equity_curve),
            "summary": {
                "total_return_pct":    round(total_ret, 2),
                "benchmark_return_pct": bench_ret,
                "excess_return_pct":   round(total_ret - bench_ret, 2) if bench_ret is not None else None,
                "max_drawdown_pct":    round(max_dd * 100, 2),
                "sharpe":              round(sharpe, 2),
                "win_rate_pct":        round(win_rate, 1),
                "total_trades":        len(sells),
                "avg_hold_days":       round(float(avg_hold), 1),
                "current_positions":   len(self.holdings),
                "cash":                int(round(self.cash)),
                "portfolio_value":     int(round(pv)),
                "deployed_pct":        round(
                    sum(self._pos_market_values.values()) / pv * 100, 1
                ) if pv > 0 else 0.0,
            },
            "current_holdings":  holdings_out,
            "watchlist":         watchlist_out,
            "equity_curve":      self.equity_curve,
            "benchmark_curve":   benchmark_curve or [],
            "transactions":      [asdict(t) for t in self.transactions],
        }


# ── Benchmark ────────────────────────────────────────────────────────────────

def _fetch_twii_benchmark(start: str, end: str) -> list[dict]:
    """Download TWII and return an array of {date, level} indexed to 100."""
    try:
        import yfinance as yf
        df = yf.download("^TWII", start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return []
        df = df[["Close"]].dropna().reset_index()
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        base = float(df.iloc[0]["close"])
        result = [
            {"date": r["date"], "level": round(float(r["close"]) / base * 100, 3)}
            for _, r in df.iterrows()
        ]
        logger.info(f"TWII benchmark: {len(result)} days loaded")
        return result
    except Exception as e:
        logger.warning(f"TWII benchmark download failed: {e}")
        return []


# ── Main entry point ─────────────────────────────────────────────────────────

def run_robot_backtest(
    results_dir: Path,
    prices_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """
    Re-run the full robot backtest from all available archives.

    Args:
        results_dir  : V6/results/ — contains YYYY-MM-DD/ subdirectories
        prices_df    : prices_raw DataFrame with columns [Date, stock_id, Close, ...]
        output_path  : where to write sim_backtest.json

    Returns:
        The output dict (same content as the saved JSON).
    """
    # ── Discover archive dates ────────────────────────────────────────────────
    archive_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and len(d.name) == 10
        and (d / "df_kelly.csv").exists()
    ])

    if not archive_dirs:
        logger.warning("No archive directories found — cannot run robot backtest")
        return {}

    logger.info(
        f"Robot backtest: {len(archive_dirs)} archive dates "
        f"({archive_dirs[0].name} → {archive_dirs[-1].name})"
    )

    # ── Prepare prices pivot ──────────────────────────────────────────────────
    prices_df = prices_df.copy()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.strftime("%Y-%m-%d")

    # Only keep Close; pivot to (date × stock_id) for fast lookups
    price_wide = prices_df.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    ).sort_index().ffill()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    start_date = archive_dirs[0].name
    end_date   = archive_dirs[-1].name
    benchmark  = _fetch_twii_benchmark(start_date, end_date)

    # ── Run robot through each archive date ───────────────────────────────────
    robot = SignalRobot()

    for arch_dir in archive_dirs:
        date = arch_dir.name
        try:
            df_day = pd.read_csv(arch_dir / "df_kelly.csv", dtype={"Ticker": str})
        except Exception as e:
            logger.warning(f"  [{date}] Cannot read df_kelly.csv: {e}")
            continue

        # Support legacy column name 'Sharpe_Score' (renamed to 'Signal_Quality')
        if "Sharpe_Score" in df_day.columns and "Signal_Quality" not in df_day.columns:
            df_day = df_day.rename(columns={"Sharpe_Score": "Signal_Quality"})

        if "Ticker" not in df_day.columns or "Signal_Quality" not in df_day.columns:
            logger.warning(f"  [{date}] Missing required columns — skipping")
            continue

        # Build price dict for this date
        if date in price_wide.index:
            series = price_wide.loc[date].dropna()
            prices_today = series.to_dict()
        else:
            logger.warning(f"  [{date}] No price data — mark-to-market skipped")
            prices_today = {}

        logger.info(f"[{date}] Processing day (holdings={len(robot.holdings)}) ...")
        robot.process_day(date, df_day, prices_today)

    # ── Final output ──────────────────────────────────────────────────────────
    # Load today's df_kelly for current state snapshot
    last_dir   = archive_dirs[-1]
    df_latest  = pd.read_csv(last_dir / "df_kelly.csv", dtype={"Ticker": str})
    latest_prices = (
        price_wide.loc[end_date].dropna().to_dict()
        if end_date in price_wide.index else {}
    )

    result = robot.build_output(
        date           = end_date,
        df_today       = df_latest,
        prices_latest  = latest_prices,
        benchmark_curve= benchmark,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    logger.info(
        f"✅ Robot backtest saved → {output_path.name}  "
        f"{result['trading_days']} days  "
        f"return={s['total_return_pct']:+.2f}%  "
        f"sharpe={s['sharpe']:.2f}  "
        f"dd={s['max_drawdown_pct']:.2f}%  "
        f"trades={s['total_trades']}"
    )
    return result


def run_daily_update(
    date: str,
    df_today: pd.DataFrame,
    prices_df: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """
    Incremental daily update: load existing state from output_path,
    process one more day, and save.

    Falls back to full backtest if output_path doesn't exist.
    """
    if not output_path.exists():
        from marketmamba.config import RESULTS_DIR, PROCESSED_DIR
        return run_robot_backtest(
            results_dir=RESULTS_DIR,
            prices_df=prices_df,
            output_path=output_path,
        )

    # Load existing output to reconstruct robot state
    with open(output_path, encoding="utf-8") as f:
        existing = json.load(f)

    robot = _restore_robot_from_json(existing, prices_df)

    prices_df_copy = prices_df.copy()
    prices_df_copy["Date"] = pd.to_datetime(prices_df_copy["Date"]).dt.strftime("%Y-%m-%d")
    price_wide = prices_df_copy.pivot_table(
        index="Date", columns="stock_id", values="Close", aggfunc="last"
    ).sort_index().ffill()
    prices_today = (
        price_wide.loc[date].dropna().to_dict()
        if date in price_wide.index else {}
    )

    robot.process_day(date, df_today, prices_today)

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

    result = robot.build_output(
        date           = date,
        df_today       = df_today,
        prices_latest  = prices_today,
        benchmark_curve= extended,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    logger.info(
        f"✅ Robot daily update → {output_path.name}  "
        f"return={s['total_return_pct']:+.2f}%  "
        f"positions={s['current_positions']}  "
        f"deployed={s['deployed_pct']:.1f}%"
    )
    return result


def _restore_robot_from_json(data: dict, prices_df: pd.DataFrame) -> "SignalRobot":
    """Reconstruct a SignalRobot from a previously saved JSON snapshot."""
    robot = SignalRobot()
    robot.equity_curve = data.get("equity_curve", [])

    # Rebuild holdings from current_holdings snapshot
    for h in data.get("current_holdings", []):
        ticker = str(h["ticker"])
        robot.holdings[ticker] = HeldPosition(
            ticker      = ticker,
            entry_date  = h["entry_date"],
            entry_price = h["entry_price"],
            shares      = h["shares"],
            cost_total  = h["entry_price"] * h["shares"] * (1 + BUY_FEE),
            entry_sq    = h.get("signal_quality", 0),
            entry_rank  = h.get("rank", 0),
        )
        robot._pos_market_values[ticker]   = h["market_value"]
        robot._pos_current_prices[ticker]  = h["current_price"]
        robot._hold_days[ticker]           = h.get("days_held", 0)

    # Restore cash from summary
    robot.cash = float(data.get("summary", {}).get("cash", INITIAL_CAPITAL))

    # Rebuild watchlist history
    for w in data.get("watchlist", []):
        robot._watchlist_history[str(w["ticker"])] = w.get("days_in_watch", 1)

    # Restore transactions
    for t in data.get("transactions", []):
        robot.transactions.append(TradeRecord(**t))

    return robot
