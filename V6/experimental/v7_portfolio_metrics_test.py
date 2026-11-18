from __future__ import annotations

import unittest
from datetime import datetime
from decimal import Context, Decimal, localcontext

from V6.experimental.v7_portfolio_contract import (
    ContractError,
    MarketQuote,
    PortfolioSpec,
)
from V6.experimental.v7_portfolio_engine import PortfolioEngine
from V6.experimental.v7_portfolio_metrics import (
    DEFAULT_EVIDENCE_CLASS,
    METRICS_VERSION,
    SessionRecord,
    SessionRecorder,
    canonical_sha256,
    compute_performance,
)

D = Decimal
DAYS = [f"2026-09-{day}T09:00:00+08:00" for day in (17, 18, 19, 20, 21)]


def rec(day: int, net: str, cost="0", buy="0", sell="0", counts=(0, 0, 0)):
    return SessionRecord(
        occurred_at=datetime.fromisoformat(DAYS[day]),
        net_value=D(net),
        total_cost=D(cost),
        buy_notional=D(buy),
        sell_notional=D(sell),
        fill_counts=(
            ("FILLED", counts[0]), ("PARTIAL", counts[1]), ("BLOCKED", counts[2])
        ),
    )


def series():
    return [
        rec(0, "100", buy="50", counts=(1, 0, 0)),
        rec(1, "120", buy="12", sell="8", counts=(2, 1, 0)),
        rec(2, "90"),
        rec(3, "108", sell="18", counts=(1, 0, 2)),
        rec(4, "108"),
    ]


def quote(ticker: str, price: str) -> MarketQuote:
    return MarketQuote(
        ticker=ticker, price=price, buy_fill_ratio="1", sell_fill_ratio="1",
        status="OPEN",
    )


def run_hold_engine():
    engine = PortfolioEngine(PortfolioSpec(
        holdings_count=1, buffer_multiple="1.5",
        rebalance_every_sessions=5, head="5d",
    ))
    recorder = SessionRecorder()
    engine.apply_signal("sig", "2026-09-16T17:00:00+08:00", "5d", {"A": "1"})
    for index, price in enumerate(("10", "11", "11")):
        result = engine.apply_market_session(
            f"s{index}", DAYS[index], {"A": quote("A", price)}
        )
        recorder.record(engine, result)
    return engine, recorder


class MetricsTest(unittest.TestCase):
    def test_exact_values_drawdown_with_recovery_and_turnover(self):
        out = compute_performance(series(), initial_net_value=100)
        self.assertEqual(out["metrics_version"], METRICS_VERSION)
        self.assertEqual(out["evidence_class"], DEFAULT_EVIDENCE_CLASS)
        self.assertEqual(out["total_return"], "0.08")
        self.assertEqual(out["max_drawdown"], {
            "fraction": "0.25",
            "peak_at": datetime.fromisoformat(DAYS[1]).isoformat(),
            "trough_at": datetime.fromisoformat(DAYS[2]).isoformat(),
        })
        self.assertEqual(
            [s["turnover"] for s in out["sessions"]],
            ["0.25", "0.1", "0", "0.1", "0"],
        )
        self.assertEqual(out["mean_turnover"], "0.09")
        self.assertEqual(out["fill_counts"],
                         {"filled": 4, "partial": 1, "blocked": 2})
        self.assertIsNone(out["benchmark"])

    def test_explicit_evidence_class(self):
        out = compute_performance(series(), evidence_class="CUSTOM")
        self.assertEqual(out["evidence_class"], "CUSTOM")

    def test_benchmark_metrics_exact(self):
        levels = dict(zip(DAYS, ("100", "110", "99", "99", "99")))
        out = compute_performance(series(), benchmark=levels)
        self.assertEqual(out["benchmark"], {
            "benchmark_return": "-0.01",
            "excess_return": "0.09",
            "tracking_difference": "0.0375",
        })

    def test_benchmark_misalignment_raises(self):
        short = dict(zip(DAYS[:4], ("100", "110", "99", "99")))
        with self.assertRaises(ContractError):
            compute_performance(series(), benchmark=short)
        shifted = dict(zip(DAYS, ("100", "110", "99", "99", "99")))
        shifted[DAYS[4]] = shifted.pop(DAYS[4]) and "99"
        shifted["2026-09-22T09:00:00+08:00"] = shifted.pop(DAYS[4])
        with self.assertRaises(ContractError):
            compute_performance(series(), benchmark=shifted)

    def test_fail_closed_inputs(self):
        with self.assertRaises(ContractError):
            compute_performance(series()[:1])
        with self.assertRaises(ContractError):
            compute_performance([])
        data = series()
        data[2], data[3] = data[3], data[2]
        with self.assertRaises(ContractError):
            compute_performance(data)
        data = series()
        data[1] = data[0]
        with self.assertRaises(ContractError):
            compute_performance(data)
        data = series()
        data[2] = rec(2, "0")
        with self.assertRaises(ContractError):
            compute_performance(data)
        data[2] = rec(2, "-5")
        with self.assertRaises(ContractError):
            compute_performance(data)

    def test_determinism_same_sha_and_tamper_changes_it(self):
        first = compute_performance(series(), initial_net_value=100)
        second = compute_performance(series(), initial_net_value=100)
        self.assertEqual(first, second)
        self.assertEqual(first["canonical_sha256"], second["canonical_sha256"])
        self.assertEqual(first["canonical_sha256"], canonical_sha256(first))
        other = compute_performance(series(), evidence_class="X")
        self.assertNotEqual(first["canonical_sha256"], other["canonical_sha256"])

    def test_engine_hold_has_zero_turnover_and_cost_drag_matches_engine(self):
        engine, recorder = run_hold_engine()
        self.assertEqual(len(recorder.records), 3)
        out = recorder.compute()
        turnovers = [s["turnover"] for s in out["sessions"]]
        self.assertNotEqual(turnovers[0], "0")
        self.assertEqual(turnovers[1:], ["0", "0"])
        self.assertGreater(engine.total_cost, D("0"))
        self.assertEqual(D(out["cumulative_cost"]), engine.total_cost)
        self.assertEqual(out["cost_drag"], out["cumulative_cost"])  # initial net value 1
        self.assertEqual(out["fill_counts"],
                         {"filled": 1, "partial": 0, "blocked": 0})
        first_buy = D(out["sessions"][0]["buy_notional"])
        self.assertEqual(out["sessions"][0]["sell_notional"], "0")
        with localcontext(Context(prec=50)):
            self.assertEqual(
                D(out["sessions"][0]["turnover"]), first_buy / D(2)
            )
        self.assertEqual(recorder.compute()["canonical_sha256"],
                         out["canonical_sha256"])


if __name__ == "__main__":
    unittest.main()
