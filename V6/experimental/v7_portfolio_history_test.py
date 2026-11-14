"""Fixture-only history export acceptance."""
import fcntl
import multiprocessing
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from V6.experimental.v7_portfolio_journal import PortfolioJournal, _replay_records
from V6.experimental.v7_portfolio_journal_test import portfolio_spec, signal_payload, market_payload, T0, T1, T2
from V6.experimental.v7_portfolio_history import export_history, read_history, HistoryError, fixture_manifest

def _export_worker(journal, output, manifest, connection):
    original = fcntl.flock
    def observe(fd, operation):
        try: original(fd, operation | fcntl.LOCK_NB)
        except BlockingIOError:
            connection.send("blocked")
            original(fd, operation)
    fcntl.flock = observe
    try:
        result = export_history(journal, output, manifest, "a" * 40)
        connection.send(result["source"])
    finally: connection.close()

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.journal = PortfolioJournal.create(self.root / "fixture.jsonl", portfolio_spec(), "genesis", T0)
        self.journal.append("signal", "SIGNAL", T1, signal_payload())
        self.journal.append("session", "MARKET_SESSION", T2, market_payload())
        self.manifest = self.root / "fixture.json"
        self.manifest.write_text(json.dumps(fixture_manifest(self.journal.path, "unit-fixture")))
        self.output = self.root / "history.json"
    def export(self):
        return export_history(self.journal.path, self.output, self.manifest, "a" * 40)
    def test_every_prefix_and_determinism(self):
        result = self.export()
        raw = self.output.read_bytes()
        records = self.journal.records()
        self.assertEqual(result["source"]["journal_sha256"], hashlib.sha256(self.journal.path.read_bytes()).hexdigest())
        for i, event in enumerate(result["events"], 1):
            self.assertEqual(event["state"], _replay_records(records[:i]).state)
        self.assertEqual(result["final_state"], self.journal.replay().state)
        self.export()
        self.assertEqual(raw, self.output.read_bytes())
        self.assertEqual(result, read_history(self.output))
    def test_corrupt_missing_and_conflicting_inputs_preserve_output(self):
        self.export()
        saved = self.output.read_bytes()
        original = self.journal.path.read_bytes()
        for value in (b"", b"{", original[:-1], original.replace(b'"seq":1', b'"seq":2')):
            self.journal.path.write_bytes(value)
            with self.assertRaises(HistoryError): self.export()
            self.assertEqual(saved, self.output.read_bytes())
        self.journal.path.unlink()
        with self.assertRaises(HistoryError): self.export()
        self.assertEqual(saved, self.output.read_bytes())
    def test_alias_rejected_and_atomic_replace_failure(self):
        self.export()
        saved = self.output.read_bytes()
        with self.assertRaises(HistoryError):
            export_history(self.journal.path, self.journal.path, self.manifest, "a" * 40)
        alias = self.root / "alias"
        alias.symlink_to(self.journal.path)
        with self.assertRaises(HistoryError):
            export_history(self.journal.path, alias, self.manifest, "a" * 40)
        with patch("V6.experimental.v7_portfolio_history.os.replace", side_effect=OSError("failure")):
            with self.assertRaises(HistoryError): self.export()
        self.assertEqual(saved, self.output.read_bytes())
        self.assertEqual([], list(self.root.glob("*.tmp")))
    def test_legacy_universal_newline_bytes_identity(self):
        original = self.journal.path.read_bytes()
        for raw in (original.replace(b"\n", b"\r\n"), original.replace(b"\n", b"\r"), original.replace(b"\n", b"\r", 1)):
            with self.subTest(raw_sha256=hashlib.sha256(raw).hexdigest()):
                self.journal.path.write_bytes(raw)
                self.manifest.write_text(json.dumps(fixture_manifest(self.journal.path, "unit-fixture")))
                result = self.export()
                self.assertEqual(result["final_state"], self.journal.replay().state)
                self.assertEqual(result["source"]["journal_sha256"], hashlib.sha256(raw).hexdigest())
    def test_identity_flags_duplicate_keys_and_state_rejected(self):
        result = self.export()
        mutations = [lambda h: h.update(research_admission=True), lambda h: h["source"].update(source_commit="bad"), lambda h: h["source"].update(event_count=True), lambda h: h["final_state"].update(cash="NaN"), lambda h: h["events"][0].update(seq=2)]
        for mutate in mutations:
            h = json.loads(json.dumps(result)); mutate(h)
            self.output.write_text(json.dumps(h))
            with self.assertRaises(HistoryError): read_history(self.output)
        self.output.write_text('{"schema":1,"schema":2}')
        with self.assertRaises(HistoryError): read_history(self.output)
    def test_real_process_shared_lock_and_captured_prefix(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_export_worker, args=(self.journal.path, self.output, self.manifest, child))
        with self.journal.path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            process.start(); child.close()
            try:
                self.assertTrue(parent.poll(10)); self.assertEqual(parent.recv(), "blocked")
            finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            self.assertTrue(parent.poll(10)); source = parent.recv()
            process.join(10)
            self.assertEqual(process.exitcode, 0)
            self.assertEqual(source["journal_sha256"], hashlib.sha256(self.journal.path.read_bytes()).hexdigest())
        finally:
            if process.is_alive(): process.kill(); process.join()
            parent.close()
        captured = self.journal.read_prefix()
        self.journal.append("later", "MARKET_SESSION", "2026-09-18T09:00:00+08:00", market_payload())
        with patch.object(PortfolioJournal, "read_prefix", return_value=captured) as reader:
            result = self.export()
        reader.assert_called_once()
        self.assertEqual(result["source"]["event_count"], 3)
        self.assertEqual(result["final_state"], _replay_records(captured[1]).state)

    def test_manifest_is_required_and_hash_bound(self):
        self.manifest.write_text(json.dumps({"schema":"real"}))
        with self.assertRaises(HistoryError): self.export()
        self.manifest.unlink()
        with self.assertRaises(HistoryError): self.export()

if __name__ == "__main__": unittest.main()
