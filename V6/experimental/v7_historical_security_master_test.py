from __future__ import annotations

import unittest

from V6.experimental.v7_historical_security_master import (
    LifecycleContractError,
    build_security_lifecycle,
)


class HistoricalSecurityMasterTest(unittest.TestCase):
    def test_listing_delisting_and_unknown_gap_fail_closed(self):
        result = build_security_lifecycle([
            {"event_type": "LIST", "stock_id": "1111", "canonical_security_id": "SEC-A", "market": "TPEX", "effective_at": "2020-01-02", "source": "official"},
            {"event_type": "DELIST", "stock_id": "1111", "canonical_security_id": "SEC-A", "market": "TPEX", "effective_at": "2021-01-05", "source": "official"},
            {"event_type": "DELIST", "stock_id": "2222", "canonical_security_id": "SEC-B", "market": "TWSE", "effective_at": "2019-06-03", "source": "official"},
        ])
        self.assertEqual(result["intervals"][0]["effective_to"], "2021-01-05")
        self.assertEqual(result["intervals"][0]["verification_state"], "VERIFIED")
        self.assertEqual(result["unresolved"][0]["reason"], "DELIST_WITHOUT_VERIFIED_LISTING_START")

    def test_suspension_preserves_existence_and_resume_boundary(self):
        result = build_security_lifecycle([
            {"event_type": "LIST", "stock_id": "2330", "canonical_security_id": "SEC-T", "market": "TWSE", "effective_at": "1994-09-05", "source": "official"},
            {"event_type": "SUSPEND", "stock_id": "2330", "canonical_security_id": "SEC-T", "market": "TWSE", "effective_at": "2026-01-05", "source": "official"},
            {"event_type": "RESUME", "stock_id": "2330", "canonical_security_id": "SEC-T", "market": "TWSE", "effective_at": "2026-01-08", "source": "official"},
        ])
        self.assertEqual([row["status"] for row in result["intervals"]], ["LISTED", "SUSPENDED", "LISTED"])
        self.assertEqual(result["intervals"][1]["effective_to"], "2026-01-08")

    def test_transfer_code_change_and_relisting_keep_identity(self):
        result = build_security_lifecycle([
            {"event_type": "LIST", "stock_id": "1234", "canonical_security_id": "SEC-X", "market": "TPEX", "effective_at": "2010-01-04", "source": "official"},
            {"event_type": "TRANSFER", "stock_id": "1234", "canonical_security_id": "SEC-X", "market": "TWSE", "effective_at": "2020-06-01", "source": "official"},
            {"event_type": "CODE_CHANGE", "stock_id": "5678", "previous_stock_id": "1234", "canonical_security_id": "SEC-X", "market": "TWSE", "effective_at": "2021-02-01", "source": "official"},
            {"event_type": "DELIST", "stock_id": "5678", "canonical_security_id": "SEC-X", "market": "TWSE", "effective_at": "2022-01-03", "source": "official"},
            {"event_type": "RELIST", "stock_id": "5678", "canonical_security_id": "SEC-X", "market": "TWSE", "effective_at": "2023-03-01", "source": "official"},
        ])
        rows = result["intervals"]
        self.assertEqual([(r["stock_id"], r["market"], r["effective_from"]) for r in rows], [
            ("1234", "TPEX", "2010-01-04"),
            ("1234", "TWSE", "2020-06-01"),
            ("5678", "TWSE", "2021-02-01"),
            ("5678", "TWSE", "2023-03-01"),
        ])
        self.assertTrue(all(r["canonical_security_id"] == "SEC-X" for r in rows))

    def test_overlapping_or_nondeterministic_events_are_rejected(self):
        events = [
            {"event_type": "LIST", "stock_id": "1111", "canonical_security_id": "S", "market": "TWSE", "effective_at": "2020-01-02", "source": "a"},
            {"event_type": "SUSPEND", "stock_id": "1111", "canonical_security_id": "S", "market": "TWSE", "effective_at": "2020-01-02", "source": "b"},
        ]
        with self.assertRaises(LifecycleContractError):
            build_security_lifecycle(events)


if __name__ == "__main__":
    unittest.main()
