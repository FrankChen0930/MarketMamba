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


def quote(
    ticker: str,
    price: str,
    *,
    status: str = "OPEN",
    buy_ratio: str = "1",
    sell_ratio: str = "1",
):
    from V6.experimental.v7_portfolio_contract import MarketQuote

    return MarketQuote(
        ticker=ticker,
        price=price,
        status=status,
        buy_fill_ratio=buy_ratio,
        sell_fill_ratio=sell_ratio,
    )


class PortfolioExecutionTimeTest(unittest.TestCase):
    def test_same_timestamp_does_not_execute_and_unknown_stays_pending(self) -> None:
        engine = PortfolioEngine(spec(holdings_count=1))
        signal_time = aware("2026-09-16T17:00:00+08:00")
        engine.apply_signal("sig-1", signal_time, "5d", {"A": "1"})

        same = engine.apply_market_session(
            "m0", signal_time, {"A": quote("A", "10")}
        )
        self.assertEqual(same.fills, ())
        self.assertEqual(engine.cash, Decimal("1"))

        blocked = engine.apply_market_session(
            "m1",
            aware("2026-09-17T09:00:00+08:00"),
            {"A": quote(
                "A", "10", status="UNKNOWN",
                buy_ratio="0", sell_ratio="0",
            )},
        )
        self.assertEqual(blocked.fills[0].status, "BLOCKED")
        self.assertEqual(blocked.fills[0].reason, "BUY_NOT_ALLOWED_UNKNOWN")
        self.assertEqual(engine.cash, Decimal("1"))
        self.assertIsNotNone(engine.pending)


class PortfolioExecutionTest(unittest.TestCase):
    T0 = "2026-09-16T17:00:00+08:00"
    T1 = "2026-09-17T09:00:00+08:00"
    T2 = "2026-09-17T17:00:00+08:00"
    T3 = "2026-09-18T09:00:00+08:00"
    T4 = "2026-09-18T10:00:00+08:00"

    def _bought_engine(self) -> tuple[PortfolioEngine, object]:
        engine = PortfolioEngine(spec(
            holdings_count=1,
            rebalance_every_sessions=1,
        ))
        engine.apply_signal("sig-1", aware(self.T0), "5d", {"A": "1"})
        result = engine.apply_market_session(
            "market-1", aware(self.T1), {"A": quote("A", "10")}
        )
        return engine, result

    def test_full_buy_charges_cost_once_and_conserves_value(self) -> None:
        engine, result = self._bought_engine()
        fill = result.fills[0]
        expected_gross = Decimal(
            "0.99850224663005491762356465302046929605591612581128"
        )
        expected_fee = Decimal(
            "0.0014977533699450823764353469795307039440838741887169"
        )

        self.assertEqual(fill.side, "BUY")
        self.assertEqual(fill.status, "FILLED")
        self.assertEqual(fill.gross_notional, expected_gross)
        self.assertEqual(fill.fee, expected_fee)
        self.assertEqual(engine.total_cost, fill.fee)
        from decimal import localcontext

        with localcontext() as context:
            context.prec = 50
            conserved_value = engine.cash + engine.position_value()
        self.assertEqual(conserved_value, engine.net_value())
        self.assertGreaterEqual(engine.cash, Decimal("0"))
        self.assertIsNone(engine.pending)
        self.assertEqual(engine.last_rebalance_session, 1)

    def test_fill_ratio_one_half_leaves_pending(self) -> None:
        engine = PortfolioEngine(spec(holdings_count=1))
        engine.apply_signal("sig-1", aware(self.T0), "5d", {"A": "1"})

        result = engine.apply_market_session(
            "market-1",
            aware(self.T1),
            {"A": quote("A", "10", buy_ratio="0.5")},
        )
        fill = result.fills[0]

        self.assertEqual(
            fill.filled_quantity,
            Decimal(
                "0.049925112331502745881178232651023464802795806290564"
            ),
        )
        self.assertEqual(fill.status, "PARTIAL")
        self.assertEqual(fill.reason, "FILL_RATIO_LIMIT")
        self.assertIsNotNone(engine.pending)

    def test_missing_quote_is_blocked_without_inventing_price(self) -> None:
        engine = PortfolioEngine(spec(holdings_count=1))
        engine.apply_signal("sig-1", aware(self.T0), "5d", {"A": "1"})

        result = engine.apply_market_session("market-1", aware(self.T1), {})

        self.assertEqual(result.fills[0].status, "BLOCKED")
        self.assertEqual(result.fills[0].reason, "MISSING_QUOTE")
        self.assertNotIn("A", engine.positions)
        self.assertEqual(engine.cash, Decimal("1"))

    def test_sell_blocked_position_causes_cash_limited_buy(self) -> None:
        engine, _ = self._bought_engine()
        original_cost = engine.total_cost
        engine.apply_market_session(
            "market-2", aware(self.T2), {"A": quote("A", "10")}
        )
        change = engine.apply_signal(
            "sig-2", aware("2026-09-17T18:00:00+08:00"),
            "5d", {"B": "2", "A": "1"},
        )
        self.assertTrue(change.due)

        result = engine.apply_market_session(
            "market-3",
            aware(self.T3),
            {
                "A": quote(
                    "A", "10", status="SELL_BLOCKED",
                    buy_ratio="1", sell_ratio="0",
                ),
                "B": quote("B", "5"),
            },
        )

        sell = next(fill for fill in result.fills if fill.side == "SELL")
        buy = next(fill for fill in result.fills if fill.side == "BUY")
        self.assertEqual(sell.status, "BLOCKED")
        self.assertEqual(sell.reason, "SELL_NOT_ALLOWED_SELL_BLOCKED")
        self.assertEqual(buy.status, "PARTIAL")
        self.assertEqual(buy.reason, "INSUFFICIENT_CASH")
        self.assertEqual(buy.filled_quantity, Decimal("0"))
        self.assertIn("A", engine.positions)
        self.assertNotIn("B", engine.positions)
        self.assertEqual(engine.total_cost, original_cost)
        self.assertIsNotNone(engine.pending)

    def test_split_preserves_value_and_dividend_uses_pre_action_quantity(self) -> None:
        from V6.experimental.v7_portfolio_contract import CorporateAction

        engine, _ = self._bought_engine()
        before_split = engine.net_value()
        engine.apply_corporate_action(
            "ca-1",
            CorporateAction(
                ticker="A",
                occurred_at=aware(self.T2),
                quantity_multiplier="2",
                cash_per_old_share="0",
                post_action_price="5",
            ),
        )
        self.assertEqual(engine.net_value(), before_split)

        before_dividend = engine.net_value()
        old_cash = engine.cash
        old_quantity = engine.positions["A"].quantity
        engine.apply_corporate_action(
            "ca-2",
            CorporateAction(
                ticker="A",
                occurred_at=aware(self.T4),
                quantity_multiplier="1",
                cash_per_old_share="0.1",
                post_action_price="4.9",
            ),
        )

        from decimal import localcontext

        with localcontext() as context:
            context.prec = 50
            expected_cash = old_cash + old_quantity * Decimal("0.1")
        self.assertEqual(engine.cash, expected_cash)
        self.assertEqual(engine.net_value(), before_dividend)

    def test_side_permissions_are_explicit(self) -> None:
        for status, expected in (
            ("BUY_BLOCKED", "BLOCKED"),
            ("HALTED", "BLOCKED"),
            ("UNKNOWN", "BLOCKED"),
            ("SELL_BLOCKED", "FILLED"),
            ("OPEN", "FILLED"),
        ):
            with self.subTest(status=status):
                engine = PortfolioEngine(spec(holdings_count=1))
                engine.apply_signal(
                    "sig-1", aware(self.T0), "5d", {"A": "1"}
                )
                ratio = "1" if status in {"SELL_BLOCKED", "OPEN"} else "0"
                result = engine.apply_market_session(
                    "market-1",
                    aware(self.T1),
                    {"A": quote(
                        "A", "10", status=status,
                        buy_ratio=ratio, sell_ratio=ratio,
                    )},
                )
                self.assertEqual(result.fills[0].status, expected)


class PortfolioEventDispatchTest(unittest.TestCase):
    def test_signal_and_market_payloads_dispatch_through_public_boundary(self) -> None:
        engine = PortfolioEngine(spec(holdings_count=1))
        engine.apply_event(
            "SIGNAL",
            "sig-1",
            aware("2026-09-16T17:00:00+08:00"),
            {"head": "5d", "scores": {"A": "1"}},
        )

        result = engine.apply_event(
            "MARKET_SESSION",
            "market-1",
            aware("2026-09-17T09:00:00+08:00"),
            {"quotes": {"A": quote("A", "10").to_payload()}},
        )

        self.assertEqual(result.fills[0].status, "FILLED")
        self.assertIn("A", engine.positions)

    def test_corporate_action_dispatch_checks_record_time(self) -> None:
        from V6.experimental.v7_portfolio_contract import CorporateAction

        engine = PortfolioEngine(spec(holdings_count=1))
        engine.apply_signal(
            "sig-1", aware("2026-09-16T17:00:00+08:00"),
            "5d", {"A": "1"},
        )
        engine.apply_market_session(
            "market-1",
            aware("2026-09-17T09:00:00+08:00"),
            {"A": quote("A", "10")},
        )
        action = CorporateAction(
            ticker="A",
            occurred_at=aware("2026-09-18T09:00:00+08:00"),
            quantity_multiplier="2",
            cash_per_old_share="0",
            post_action_price="5",
        )

        result = engine.apply_event(
            "CORPORATE_ACTION",
            "ca-1",
            action.occurred_at,
            action.to_payload(),
        )
        self.assertEqual(result.status, "APPLIED")

        bad_payload = action.to_payload()
        bad_payload["occurred_at"] = "2026-09-19T09:00:00+08:00"
        with self.assertRaises(ContractError):
            engine.apply_event(
                "CORPORATE_ACTION",
                "ca-2",
                action.occurred_at,
                bad_payload,
            )

    def test_unknown_event_kind_is_rejected(self) -> None:
        engine = PortfolioEngine(spec())
        with self.assertRaises(ContractError):
            engine.apply_event(
                "UNKNOWN_EVENT",
                "event-1",
                aware("2026-09-16T17:00:00+08:00"),
                {},
            )


class PortfolioReviewRegressionTest(unittest.TestCase):
    def test_existing_target_without_quote_keeps_pending(self) -> None:
        engine = PortfolioEngine(spec(
            holdings_count=1,
            rebalance_every_sessions=1,
        ))
        engine.apply_signal(
            "sig-1", aware("2026-09-16T17:00:00+08:00"),
            "5d", {"A": "1"},
        )
        engine.apply_market_session(
            "market-1",
            aware("2026-09-17T09:00:00+08:00"),
            {"A": quote("A", "10")},
        )
        engine.apply_market_session(
            "market-2",
            aware("2026-09-17T17:00:00+08:00"),
            {"A": quote("A", "10")},
        )
        engine.apply_signal(
            "sig-2", aware("2026-09-17T18:00:00+08:00"),
            "5d", {"A": "1"},
        )

        result = engine.apply_market_session(
            "market-3",
            aware("2026-09-18T09:00:00+08:00"),
            {},
        )

        self.assertEqual(result.fills[0].reason, "MISSING_QUOTE")
        self.assertIsNotNone(engine.pending)

    def test_two_open_targets_complete_without_negative_cash(self) -> None:
        engine = PortfolioEngine(spec(holdings_count=2))
        engine.apply_signal(
            "sig-1", aware("2026-09-16T17:00:00+08:00"),
            "5d", {"A": "2", "B": "1"},
        )

        result = engine.apply_market_session(
            "market-1",
            aware("2026-09-17T09:00:00+08:00"),
            {"A": quote("A", "10"), "B": quote("B", "20")},
        )

        self.assertEqual([fill.status for fill in result.fills], [
            "FILLED", "FILLED",
        ])
        self.assertEqual(set(engine.positions), {"A", "B"})
        self.assertGreaterEqual(engine.cash, Decimal("0"))
        self.assertIsNone(engine.pending)

    def test_full_sell_charges_sell_cost_once(self) -> None:
        engine = PortfolioEngine(spec(
            holdings_count=1,
            rebalance_every_sessions=1,
        ))
        engine.apply_signal(
            "sig-1", aware("2026-09-16T17:00:00+08:00"),
            "5d", {"A": "1"},
        )
        first = engine.apply_market_session(
            "market-1",
            aware("2026-09-17T09:00:00+08:00"),
            {"A": quote("A", "10")},
        )
        first_cost = first.fills[0].fee
        engine.apply_market_session(
            "market-2",
            aware("2026-09-17T17:00:00+08:00"),
            {"A": quote("A", "10")},
        )
        engine.apply_signal(
            "sig-2", aware("2026-09-17T18:00:00+08:00"),
            "5d", {"B": "2", "A": "1"},
        )

        changed = engine.apply_market_session(
            "market-3",
            aware("2026-09-18T09:00:00+08:00"),
            {
                "A": quote(
                    "A", "10", status="BUY_BLOCKED",
                    buy_ratio="0", sell_ratio="1",
                ),
                "B": quote("B", "5"),
            },
        )

        sell = next(fill for fill in changed.fills if fill.side == "SELL")
        buy = next(fill for fill in changed.fills if fill.side == "BUY")
        self.assertEqual(sell.status, "FILLED")
        self.assertEqual(
            sell.fee,
            Decimal(
                "0.0044932601098352471293060409385921118322516225661508"
            ),
        )
        from decimal import localcontext
        with localcontext() as context:
            context.prec = 50
            expected_total_cost = first_cost + sell.fee + buy.fee
        self.assertEqual(engine.total_cost, expected_total_cost)

    def test_replay_math_does_not_depend_on_ambient_decimal_precision(self) -> None:
        from decimal import localcontext

        def run_once() -> dict:
            engine = PortfolioEngine(spec(holdings_count=2))
            engine.apply_signal(
                "sig-1", aware("2026-09-16T17:00:00+08:00"),
                "5d", {"A": "2", "B": "1"},
            )
            engine.apply_market_session(
                "market-1",
                aware("2026-09-17T09:00:00+08:00"),
                {"A": quote("A", "10"), "B": quote("B", "20")},
            )
            return engine.snapshot()

        default_precision = run_once()
        with localcontext() as context:
            context.prec = 10
            low_precision = run_once()

        self.assertEqual(low_precision, default_precision)


if __name__ == "__main__":
    unittest.main()
