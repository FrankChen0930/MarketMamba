import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from V6.experimental.v7_health_summary import build_health_summary, write_health_summary


class V7HealthSummaryTest(unittest.TestCase):
    def test_quarantine_is_degraded_but_publishable(self):
        document = {
            "graph": {"nodes": 12, "edges": 20, "output_sha256": "graph-hash"},
            "quality_report": {
                "blocking": False,
                "entries": [
                    {"severity": "WARN", "reason_code": "STALE", "dates": ["2026-09-15"], "stocks": ["2330"]},
                    {"severity": "QUARANTINE", "reason_code": "BAD_OHLC", "dates": ["2026-09-14", "2026-09-15"], "stocks": ["2317", "2330"]},
                ],
            },
        }
        result = build_health_summary(document, data_id="smoke-v1", generated_at="2026-09-16T00:00:00Z")
        self.assertEqual(result["schema"], "v7-health-summary-v1")
        self.assertEqual(result["state"], "degraded")
        self.assertTrue(result["publish_allowed"])
        self.assertEqual(result["counts"], {"entries": 2, "affected_dates": 2, "affected_stocks": 2})
        self.assertEqual(result["severity_counts"], {"BLOCK": 0, "QUARANTINE": 1, "WARN": 1, "INFO": 0})
        self.assertEqual(result["reason_counts"], {"BAD_OHLC": 1, "STALE": 1})
        self.assertEqual(result["affected_dates"], ["2026-09-14", "2026-09-15"])
        self.assertEqual(result["affected_stocks"], ["2317", "2330"])
        self.assertEqual(result["graph"]["nodes"], 12)

    def test_blocking_report_is_blocked(self):
        document = {"graph": {}, "quality_report": {"blocking": True, "entries": [
            {"severity": "BLOCK", "reason_code": "MISSING_COLUMN", "dates": [], "stocks": []}
        ]}}
        result = build_health_summary(document, data_id="blocked", generated_at="2026-09-16T00:00:00Z")
        self.assertEqual(result["state"], "blocked")
        self.assertFalse(result["publish_allowed"])

    def test_missing_blocking_and_unknown_severity_fail_closed(self):
        with self.assertRaises(ValueError):
            build_health_summary({"quality_report": {"entries": []}}, data_id="bad", generated_at="now")
        with self.assertRaises(ValueError):
            build_health_summary({"quality_report": {"blocking": False, "entries": [
                {"severity": "SURPRISE"}
            ]}}, data_id="bad", generated_at="now")

    def test_write_is_atomic_and_records_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data_health.json"
            output = root / "health-summary.json"
            raw = json.dumps({"graph": {}, "quality_report": {"blocking": False, "entries": []}})
            source.write_text(raw, encoding="utf-8")
            result = write_health_summary(source, output, data_id="empty", generated_at="2026-09-16T00:00:00Z")
            self.assertEqual(result, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(result["source"]["sha256"], hashlib.sha256(raw.encode()).hexdigest())
            previous = output.read_bytes()
            source.write_text("{broken", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                write_health_summary(source, output, data_id="bad", generated_at="later")
            self.assertEqual(output.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
