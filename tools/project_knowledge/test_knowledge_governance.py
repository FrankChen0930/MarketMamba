#!/usr/bin/env python3
"""Focused tests for K2 governance behavior."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch
from subprocess import CompletedProcess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_knowledge_impact
import export_obsidian_links
import generate_indexes
import runtime_observations
import validate_governance
from authority_refs import parse_authority_ref


ROOT = Path(__file__).resolve().parents[2]


class AuthorityReferenceTests(unittest.TestCase):
    def test_durable_and_labeled_references(self) -> None:
        durable = parse_authority_ref("e3bce91:path/to/file.json")
        labeled = parse_authority_ref("feature/example@e3bce91:path/to/file.json#pointer")
        self.assertEqual(durable.commit, "e3bce91")
        self.assertIsNone(durable.branch)
        self.assertEqual(labeled.branch, "feature/example")
        self.assertEqual(labeled.fragment, "pointer")


class ImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "knowledge/knowledge-impact-map.json").read_text(encoding="utf-8"))

    def test_feature_manifest_requires_review(self) -> None:
        report = check_knowledge_impact.analyze(
            ["research/v7/x/feature-manifest.json"], self.config, None
        )
        self.assertEqual(report["status"], "WARNING")
        self.assertIn("knowledge/01_Domains/Features.md", {item["artifact"] for item in report["potential_knowledge_impact"]})

    def test_production_and_portfolio_mapping(self) -> None:
        report = check_knowledge_impact.analyze(
            ["V6/run_v62_daily.py", "V6/experimental/v7_portfolio_engine.py"], self.config, None
        )
        artifacts = {item["artifact"] for item in report["potential_knowledge_impact"]}
        self.assertIn("knowledge/01_Domains/Production.md", artifacts)
        self.assertIn("knowledge/01_Domains/Portfolio.md", artifacts)

    def test_acknowledgment_requires_reason(self) -> None:
        with self.assertRaises(ValueError):
            check_knowledge_impact.analyze(["V6/run_v62_daily.py"], self.config, " ")
        report = check_knowledge_impact.analyze(["V6/run_v62_daily.py"], self.config, "refactor only")
        self.assertEqual(report["status"], "ACKNOWLEDGED")
        self.assertFalse(report["automatic_updates_performed"])

    def test_governance_tool_requires_governance_review(self) -> None:
        report = check_knowledge_impact.analyze(
            ["tools/project_knowledge/knowledge_health.py"], self.config, None
        )
        self.assertEqual(report["status"], "WARNING")
        self.assertIn("knowledge/GOVERNANCE.md", {item["artifact"] for item in report["potential_knowledge_impact"]})

    def test_worktree_responsibility_requires_state_review(self) -> None:
        report = check_knowledge_impact.analyze(
            ["knowledge/00_Project_Map/Worktree_Register.md"], self.config, None
        )
        artifacts = {item["artifact"] for item in report["potential_knowledge_impact"]}
        self.assertIn("knowledge/00_Project_Map/Current_State.md", artifacts)
        self.assertIn("knowledge/00_Project_Map/Authority_Map.md", artifacts)


class RuntimeTests(unittest.TestCase):
    def test_fresh_and_stale(self) -> None:
        record = {
            "observation_id": "test", "topic": "Test", "observed_at": "2026-09-20T00:00:00+08:00",
            "ttl_days": 7, "actor": "tester", "method": "inspection", "evidence_ref": "evidence", "value": {"ok": True},
        }
        fresh = runtime_observations.classify(record, runtime_observations.parse_time("2026-09-21"))
        stale = runtime_observations.classify(record, runtime_observations.parse_time("2026-09-28"))
        self.assertEqual(fresh["status"], "FRESH")
        self.assertEqual(stale["status"], "STALE_OBSERVATION")

    def test_record_is_explicit_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(
                id="scheduler-test", topic="Scheduler", observed_at="2026-09-20T10:00:00+08:00",
                ttl_days=7, actor="human", method="manual inspection", evidence_ref="evidence.json",
                value_json='{"enabled": true}', confidence="HIGH", store=Path(temp),
            )
            payload = runtime_observations.record(args)
            saved = json.loads((Path(temp) / "scheduler-test.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, payload)


class GenerationTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_indexes.build_outputs()
        second = generate_indexes.build_outputs()
        self.assertEqual(first, second)

    def test_dirty_legacy_document_uses_committed_prose(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("unrelated user edit", encoding="utf-8")
            with patch.object(generate_indexes, "ROOT", root), patch.object(
                generate_indexes.subprocess, "run", side_effect=[
                    CompletedProcess([], 1),
                    CompletedProcess([], 0, stdout="committed prose"),
                ],
            ):
                self.assertEqual(generate_indexes.read_legacy_document("README.md"), "committed prose")


class CanonicalInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        knowledge = ROOT / "knowledge"
        cls.authority = json.loads((knowledge / "00_Project_Map/authority-map.json").read_text(encoding="utf-8"))
        cls.current = json.loads((knowledge / "00_Project_Map/current-state.json").read_text(encoding="utf-8"))
        cls.current_md = (knowledge / "00_Project_Map/Current_State.md").read_text(encoding="utf-8")
        cls.authority_md = (knowledge / "00_Project_Map/Authority_Map.md").read_text(encoding="utf-8")
        cls.features = (knowledge / "01_Domains/Features.md").read_text(encoding="utf-8")
        cls.operations = (knowledge / "01_Domains/Operations.md").read_text(encoding="utf-8")

    def validate(self, authority: dict | None = None, current: dict | None = None) -> None:
        validate_governance.validate_explicit_invariants(
            authority or self.authority, current or self.current, self.current_md,
            self.authority_md, self.features, self.operations,
        )

    def test_feature_authority_change_without_note_fails(self) -> None:
        authority = copy.deepcopy(self.authority)
        next(item for item in authority["topics"] if item["topic"] == "Feature contract")["primary_authority"] = "deadbee:replacement.json"
        with self.assertRaises(AssertionError):
            self.validate(authority=authority)

    def test_e1_status_mismatch_fails(self) -> None:
        current = copy.deepcopy(self.current)
        current["research"]["e1"]["status"] = "COMPLETE"
        with self.assertRaises(AssertionError):
            self.validate(current=current)

    def test_phase_a_deployed_mismatch_fails(self) -> None:
        current = copy.deepcopy(self.current)
        current["operations"]["deployed"] = True
        with self.assertRaises(AssertionError):
            self.validate(current=current)


class ObsidianTests(unittest.TestCase):
    def test_missing_vault_is_skipped(self) -> None:
        report = export_obsidian_links.discover(None)
        self.assertEqual(report["status"], "SKIPPED")
        self.assertFalse(report["vault_modified"])

    def test_export_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            note = Path(temp) / "note.md"
            note.write_text("MarketMamba\nCanonical: knowledge/01_Domains/Models.md\n", encoding="utf-8")
            before = note.read_bytes()
            report = export_obsidian_links.discover(Path(temp))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(note.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
