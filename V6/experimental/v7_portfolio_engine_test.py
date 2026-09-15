from __future__ import annotations

import unittest
from datetime import datetime
from decimal import Decimal

from V6.experimental.v7_portfolio_contract import ContractError, PortfolioSpec
from V6.experimental.v7_portfolio_engine import (
    PortfolioEngine,
    select_target_tickers,
)


def aware(value: str) -> datetime:
    return datetime.fromisoformat(value)


def spec(
    *,
    holdings_count: int = 2,
    buffer_multiple: str = "1.5",
    rebalance_every_sessions: int = 5,
) -> PortfolioSpec:
    return PortfolioSpec(
        holdings_count=holdings_count,
        buffer_multiple=buffer_multiple,
        rebalance_every_sessions=rebalance_every_sessions,
        head="5d",
    )


class PortfolioPlannerTest(unittest.TestCase):
    def test_signal_tie_breaks_by_ticker_and_does_not_trade(self) -> None:
        engine = PortfolioEngine(spec())

        result = engine.apply_signal(
            "sig-1",
            aware("2026-09-16T17:00:00+08:00"),
            "5d",
            {"B": "1", "A": "1", "C": "0"},
        )

        self.assertTrue(result.due)
        self.assertEqual(result.target_tickers, ("A", "B"))
        self.assertEqual(engine.snapshot()["cash"], "1")
        self.assertEqual(engine.snapshot()["positions"], {})
        self.assertEqual(
            engine.snapshot()["pending"]["signal_id"],
            "sig-1",
        )

    def test_buffer_keeps_rank_three_and_fills_from_top_n(self) -> None:
        target = select_target_tickers(
            spec(),
            {"A": "4", "B": "3", "C": "2", "D": "1"},
            current_holdings=("C", "D"),
        )

        self.assertEqual(target, ("C", "A"))

    def test_pending_signal_is_not_superseded_before_frequency_is_due(self) -> None:
        engine = PortfolioEngine(spec(rebalance_every_sessions=5))
        first = engine.apply_signal(
            "sig-1",
            aware("2026-09-16T17:00:00+08:00"),
            "5d",
            {"A": "2", "B": "1"},
        )

        early = engine.apply_signal(
            "sig-2",
            aware("2026-09-17T17:00:00+08:00"),
            "5d",
            {"C": "2", "D": "1"},
        )

        self.assertTrue(first.due)
        self.assertFalse(early.due)
        self.assertEqual(early.target_tickers, ("A", "B"))
        self.assertEqual(engine.snapshot()["pending"]["signal_id"], "sig-1")

    def test_candidate_shortage_never_invents_a_ticker(self) -> None:
        target = select_target_tickers(
            spec(holdings_count=3),
            {"2330": "0.2"},
            current_holdings=(),
        )

        self.assertEqual(target, ("2330",))

    def test_head_mismatch_and_malformed_signal_are_rejected(self) -> None:
        engine = PortfolioEngine(spec())
        with self.assertRaises(ContractError):
            engine.apply_signal(
                "sig-x",
                aware("2026-09-16T17:00:00+08:00"),
                "10d",
                {"A": "1"},
            )
        with self.assertRaises(ContractError):
            engine.apply_signal(
                "sig-x",
                aware("2026-09-16T17:00:00+08:00"),
                "5d",
                {"A": "NaN"},
            )
        with self.assertRaises(ContractError):
            engine.apply_signal(
                " ",
                aware("2026-09-16T17:00:00+08:00"),
                "5d",
                {"A": "1"},
            )


if __name__ == "__main__":
    unittest.main()
