from __future__ import annotations

import unittest
from datetime import datetime
from decimal import Decimal

from V6.experimental.v7_portfolio_contract import (
    ENGINE_VERSION,
    ContractError,
    CorporateAction,
    MarketQuote,
    PortfolioSpec,
    TradeStatus,
    parse_decimal,
    parse_timestamp,
)


class PortfolioContractTest(unittest.TestCase):
    def test_default_spec_round_trips_with_decimal_strings(self) -> None:
        spec = PortfolioSpec(
            holdings_count=2,
            buffer_multiple="1.5",
            rebalance_every_sessions=5,
            head="5d",
        )

        payload = spec.to_payload()

        self.assertEqual(payload, {
            "engine_version": ENGINE_VERSION,
            "holdings_count": 2,
            "buffer_multiple": "1.5",
            "rebalance_every_sessions": 5,
            "head": "5d",
            "buy_cost_rate": "0.0015",
            "sell_cost_rate": "0.0045",
        })
        self.assertEqual(PortfolioSpec.from_payload(payload), spec)

    def test_spec_rejects_unsafe_counts_rates_and_version(self) -> None:
        invalid = (
            {"holdings_count": 0},
            {"buffer_multiple": "0.99"},
            {"rebalance_every_sessions": 0},
            {"buy_cost_rate": "1"},
            {"sell_cost_rate": "-0.01"},
            {"head": " "},
            {"engine_version": "future-version"},
        )
        for change in invalid:
            values = {
                "holdings_count": 2,
                "buffer_multiple": "1.5",
                "rebalance_every_sessions": 5,
                "head": "5d",
            }
            values.update(change)
            with self.subTest(change=change), self.assertRaises(ContractError):
                PortfolioSpec(**values)

    def test_unknown_is_the_explicit_safe_default(self) -> None:
        quote = MarketQuote.from_payload({
            "ticker": "2330",
            "price": "100",
            "buy_fill_ratio": "0",
            "sell_fill_ratio": "0",
        })

        self.assertEqual(quote.status, TradeStatus.UNKNOWN)
        self.assertEqual(quote.to_payload(), {
            "ticker": "2330",
            "price": "100",
            "status": "UNKNOWN",
            "buy_fill_ratio": "0",
            "sell_fill_ratio": "0",
        })

    def test_quote_requires_explicit_ratios_and_rejects_invalid_values(self) -> None:
        with self.assertRaises(ContractError):
            MarketQuote.from_payload({"ticker": "2330", "price": "100"})
        invalid = (
            {"ticker": "2330", "price": "0", "buy_fill_ratio": "1", "sell_fill_ratio": "1"},
            {"ticker": "2330", "price": "100", "buy_fill_ratio": "1.1", "sell_fill_ratio": "1"},
            {"ticker": "2330", "price": "100", "buy_fill_ratio": "1", "sell_fill_ratio": "-0.1"},
            {"ticker": " ", "price": "100", "buy_fill_ratio": "1", "sell_fill_ratio": "1"},
            {"ticker": "2330", "price": "100", "status": "NOT_A_STATUS",
             "buy_fill_ratio": "1", "sell_fill_ratio": "1"},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                MarketQuote.from_payload(payload)

    def test_invalid_numbers_and_naive_timestamps_are_rejected(self) -> None:
        for value in ("NaN", "Infinity", "-Infinity", "", True):
            with self.subTest(value=value), self.assertRaises(ContractError):
                parse_decimal(value, field="price", minimum=Decimal("0"))
        with self.assertRaises(ContractError):
            parse_timestamp("2026-09-16T17:00:00")

        parsed = parse_timestamp("2026-09-16T17:00:00+08:00")
        self.assertEqual(parsed, datetime.fromisoformat("2026-09-16T17:00:00+08:00"))

    def test_corporate_action_round_trips_and_validates_explicit_terms(self) -> None:
        action = CorporateAction(
            ticker="2330",
            occurred_at="2026-09-17T09:00:00+08:00",
            quantity_multiplier="2",
            cash_per_old_share="0.1",
            post_action_price="49.9",
        )
        self.assertEqual(CorporateAction.from_payload(action.to_payload()), action)

        invalid = (
            {"quantity_multiplier": "0"},
            {"cash_per_old_share": "-0.1"},
            {"post_action_price": "0"},
            {"occurred_at": "2026-09-17T09:00:00"},
        )
        for change in invalid:
            values = {
                "ticker": "2330",
                "occurred_at": "2026-09-17T09:00:00+08:00",
                "quantity_multiplier": "1",
                "cash_per_old_share": "0",
                "post_action_price": "100",
            }
            values.update(change)
            with self.subTest(change=change), self.assertRaises(ContractError):
                CorporateAction(**values)


if __name__ == "__main__":
    unittest.main()
