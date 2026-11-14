from __future__ import annotations

import unittest

from V6.experimental.v7_ohlcv_provenance import (
    SourceClass,
    apply_source_eligibility,
    choose_source,
    classify_source,
    source_reliability_matrix,
)


class OhlcvProvenanceTest(unittest.TestCase):
    def test_exchange_regular_board_verified_is_admitted(self) -> None:
        source = classify_source(
            has_exchange_regular_board=True,
            has_exchange_derived=False,
            has_mixed_provider=True,
            was_empirically_backfilled=False,
            adjusted_prices=False,
        )
        self.assertEqual(source, SourceClass.EXCHANGE_REGULAR_BOARD_VERIFIED)
        result = apply_source_eligibility({"observation_present": True}, source)
        self.assertTrue(result["observation_present"])
        self.assertTrue(result["source_eligible"])

    def test_exchange_derived_with_clear_regular_board_semantics_is_admitted(self) -> None:
        source = classify_source(
            has_exchange_regular_board=False,
            has_exchange_derived=True,
            has_mixed_provider=True,
            was_empirically_backfilled=False,
            adjusted_prices=True,
        )
        self.assertEqual(source, SourceClass.EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR)
        self.assertTrue(apply_source_eligibility({"observation_present": True}, source)["source_eligible"])

    def test_ambiguous_provider_segment_cannot_execute(self) -> None:
        source = classify_source(
            has_exchange_regular_board=False,
            has_exchange_derived=False,
            has_mixed_provider=True,
            was_empirically_backfilled=False,
            adjusted_prices=False,
        )
        self.assertEqual(source, SourceClass.PROVIDER_MIXED_UNKNOWN_SEGMENT)
        result = apply_source_eligibility({"observation_present": True}, source)
        self.assertFalse(result["observation_present"])
        self.assertFalse(result["source_eligible"])

    def test_synthetic_backfill_is_rejected_even_with_finite_ohlcv(self) -> None:
        source = classify_source(
            has_exchange_regular_board=False,
            has_exchange_derived=False,
            has_mixed_provider=True,
            was_empirically_backfilled=True,
            adjusted_prices=True,
        )
        self.assertEqual(source, SourceClass.SYNTHETIC_BACKFILLED)
        result = apply_source_eligibility({"observation_present": True, "Open": 220.0}, source)
        self.assertFalse(result["observation_present"])

    def test_adjusted_price_without_segment_provenance_is_rejected(self) -> None:
        source = classify_source(
            has_exchange_regular_board=False,
            has_exchange_derived=False,
            has_mixed_provider=False,
            was_empirically_backfilled=False,
            adjusted_prices=True,
        )
        self.assertEqual(source, SourceClass.ADJUSTED_PRICE_ONLY)
        self.assertFalse(apply_source_eligibility({"observation_present": True}, source)["source_eligible"])

    def test_mixed_source_precedence_is_deterministic(self) -> None:
        candidates = [
            SourceClass.SYNTHETIC_BACKFILLED,
            SourceClass.PROVIDER_MIXED_UNKNOWN_SEGMENT,
            SourceClass.EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR,
        ]
        self.assertEqual(choose_source(candidates), SourceClass.EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR)
        self.assertEqual(choose_source(reversed(candidates)), SourceClass.EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR)

    def test_1264_regression_fails_closed_without_regular_board_anchor(self) -> None:
        source = classify_source(
            has_exchange_regular_board=False,
            has_exchange_derived=False,
            has_mixed_provider=True,
            was_empirically_backfilled=True,
            adjusted_prices=True,
        )
        row = apply_source_eligibility(
            {"observation_present": True, "Open": 220.76441634459184, "Volume": 4653.0},
            source,
        )
        self.assertEqual(row["source_class"], "SYNTHETIC/BACKFILLED")
        self.assertFalse(row["observation_present"])

    def test_source_reliability_matrix_is_complete_and_deterministic(self) -> None:
        first = source_reliability_matrix()
        second = source_reliability_matrix()
        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {
                "EXCHANGE_REGULAR_BOARD_VERIFIED",
                "EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR",
                "PROVIDER_MIXED_UNKNOWN_SEGMENT",
                "ADJUSTED_PRICE_ONLY",
                "SYNTHETIC/BACKFILLED",
                "UNKNOWN",
            },
        )
