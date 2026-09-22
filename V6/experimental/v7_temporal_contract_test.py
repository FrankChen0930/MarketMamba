from __future__ import annotations

import json
import unittest

from V6.experimental.v7_temporal_contract import (
    ContractError,
    active_intervals,
    canonical_manifest,
    first_session_after,
    normalize_publication_event,
    validate_effective_intervals,
)


CALENDAR = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-08"]


class CalendarContractTest(unittest.TestCase):
    def test_publication_uses_strictly_next_session(self) -> None:
        self.assertEqual(first_session_after(CALENDAR, "2026-01-02"), "2026-01-05")
        self.assertEqual(first_session_after(CALENDAR, "2026-01-03"), "2026-01-05")
        self.assertEqual(first_session_after(CALENDAR, "2026-01-06"), "2026-01-08")

    def test_unknown_timestamp_stays_unknown(self) -> None:
        event = normalize_publication_event(
            {"entity_id": "2330", "period": "2025-12", "published_at": None},
            calendar=CALENDAR,
            source="local-revenue",
            timestamp_semantics="official publication time",
        )
        self.assertEqual(event["verification_status"], "UNKNOWN")
        self.assertIsNone(event["available_session"])

    def test_verified_event_records_next_session_and_provenance(self) -> None:
        event = normalize_publication_event(
            {"entity_id": "2330", "period": "2025-12", "published_at": "2026-01-02T18:03:00+08:00"},
            calendar=CALENDAR,
            source="MOPS",
            timestamp_semantics="server publication timestamp",
        )
        self.assertEqual(event["verification_status"], "VERIFIED")
        self.assertEqual(event["publication_date"], "2026-01-02")
        self.assertEqual(event["available_session"], "2026-01-05")
        self.assertEqual(event["source"], "MOPS")

    def test_out_of_calendar_future_is_not_fabricated(self) -> None:
        event = normalize_publication_event(
            {"entity_id": "2330", "period": "2025-12", "published_at": "2026-01-09"},
            calendar=CALENDAR,
            source="MOPS",
            timestamp_semantics="publication date",
        )
        self.assertEqual(event["verification_status"], "VERIFIED")
        self.assertIsNone(event["available_session"])


class EffectiveIntervalTest(unittest.TestCase):
    def test_exact_boundaries_are_half_open(self) -> None:
        rows = [
            {"stock_id": "1101", "effective_from": "2020-01-01", "effective_to": "2021-01-01", "industry_code": "A"},
            {"stock_id": "1101", "effective_from": "2021-01-01", "effective_to": None, "industry_code": "B"},
        ]
        validate_effective_intervals(rows, key_fields=("stock_id",))
        self.assertEqual(active_intervals(rows, "2020-12-31")[0]["industry_code"], "A")
        self.assertEqual(active_intervals(rows, "2021-01-01")[0]["industry_code"], "B")

    def test_overlapping_intervals_fail_closed(self) -> None:
        rows = [
            {"stock_id": "1101", "effective_from": "2020-01-01", "effective_to": "2021-06-01"},
            {"stock_id": "1101", "effective_from": "2021-01-01", "effective_to": None},
        ]
        with self.assertRaisesRegex(ContractError, "overlap"):
            validate_effective_intervals(rows, key_fields=("stock_id",))

    def test_manifest_is_reproducible_and_json_safe(self) -> None:
        one = canonical_manifest("v7-test", {"b": 2, "a": 1}, {"x": "abc"})
        two = canonical_manifest("v7-test", {"a": 1, "b": 2}, {"x": "abc"})
        self.assertEqual(one, two)
        json.dumps(one, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
