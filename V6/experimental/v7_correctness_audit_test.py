from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from V6.experimental.v7_correctness_audit import AuditValidationError, evaluate_audit, render_markdown, write_outputs


def clean_document() -> dict:
    checks = {}
    for domain in (
        "label_execution_alignment",
        "feature_point_in_time",
        "historical_universe",
        "preprocessing_leakage",
    ):
        checks[domain] = [
            {
                "id": f"{domain}.safe",
                "status": "PASS",
                "severity": "INFO",
                "finding": "Direct evidence supports this check.",
                "evidence": [{"path": "example.py", "lines": "1-2", "observation": "causal"}],
                "affected_scope": "none",
                "requires_rebuild": False,
                "minimum_fix": "none",
            }
        ]
    return {
        "schema_version": "v7-correctness-audit-v1",
        "baseline": "E5 three-seed ensemble",
        "timing": {
            "feature_information_cutoff": "session t close",
            "prediction_timestamp": "session t after close",
            "earliest_execution": "session t+1 open",
            "label_start": "session t+1 open",
            "label_end": "horizon-specific executable close",
        },
        "domains": checks,
    }


class AuditDecisionTest(unittest.TestCase):
    def test_complete_clean_document_passes(self) -> None:
        decision = evaluate_audit(clean_document())
        self.assertEqual(decision.status, "PASS")
        self.assertEqual(decision.blocking_findings, ())

    def test_major_failure_stops_and_preserves_remediation(self) -> None:
        document = copy.deepcopy(clean_document())
        check = document["domains"]["label_execution_alignment"][0]
        check.update(
            status="FAIL",
            severity="MAJOR",
            finding="The label starts before the first executable price.",
            affected_scope="all 5d and 10d labels",
            requires_rebuild=True,
            minimum_fix="Rebase labels on a frozen executable price.",
        )
        decision = evaluate_audit(document)
        self.assertEqual(decision.status, "STOP")
        self.assertEqual(len(decision.blocking_findings), 1)
        self.assertEqual(decision.blocking_findings[0]["id"], check["id"])
        self.assertTrue(decision.blocking_findings[0]["requires_rebuild"])

    def test_missing_required_domain_is_invalid(self) -> None:
        document = clean_document()
        del document["domains"]["historical_universe"]
        with self.assertRaisesRegex(AuditValidationError, "required audit domains"):
            evaluate_audit(document)


class AuditReportTest(unittest.TestCase):
    def test_markdown_exposes_timing_and_blocker_remediation(self) -> None:
        document = copy.deepcopy(clean_document())
        document["domains"]["label_execution_alignment"][0].update(
            status="FAIL", severity="MAJOR",
            affected_scope="all labels", requires_rebuild=True,
            minimum_fix="Use next-session executable prices.",
        )
        report = render_markdown(document, evaluate_audit(document))
        self.assertIn("Gate decision: STOP", report)
        self.assertIn("session t close -> session t after close -> session t+1 open", report)
        self.assertIn("Use next-session executable prices.", report)
        self.assertIn("example.py:1-2", report)


    def test_write_outputs_creates_matching_json_and_markdown(self) -> None:
        document = copy.deepcopy(clean_document())
        document["domains"]["historical_universe"][0].update(
            status="UNKNOWN", severity="CRITICAL"
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "audit.json"
            markdown_path = Path(directory) / "audit.md"
            decision = write_outputs(document, json_path, markdown_path)
            payload = json.loads(json_path.read_text())
            self.assertEqual(decision.status, "STOP")
            self.assertEqual(payload["decision"], "STOP")
            self.assertEqual(payload["blocking_findings"][0]["domain"], "historical_universe")
            self.assertIn("Gate decision: STOP", markdown_path.read_text())
            self.assertFalse(list(Path(directory).glob("*.tmp")))


class AuditCliTest(unittest.TestCase):
    def test_stop_is_completed_audit_exit_code_two(self) -> None:
        document = copy.deepcopy(clean_document())
        document["domains"]["preprocessing_leakage"][0].update(
            status="FAIL", severity="MAJOR"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evidence.json"
            source.write_text(json.dumps(document))
            command = [
                sys.executable, str(Path(__file__).with_name("v7_correctness_audit.py")),
                "--input", str(source), "--json-output", str(root / "out.json"),
                "--markdown-output", str(root / "out.md"),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "V7 correctness audit: STOP")
            self.assertTrue((root / "out.json").is_file())
            self.assertTrue((root / "out.md").is_file())


if __name__ == "__main__":
    unittest.main()
