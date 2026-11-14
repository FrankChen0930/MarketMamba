"""Real-process WAL isolation evidence; synthetic ledgers, no timing sleeps."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from marketmamba.ops import projection
from marketmamba.ops.contracts import ArtifactIdentity, LogicalRunIdentity, RunState
from marketmamba.ops.lease import acquire_lease
from marketmamba.ops.schema import open_ledger
from marketmamba.ops.store import ArtifactRef, LedgerStore


NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
PAYLOAD = b"offline task4 isolation artifact"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
IDENTITY = ArtifactIdentity("isolation_fixture", DIGEST, "v1", "b" * 64)
TIMEOUT = 15


def receive(pipe):
    if not pipe.poll(TIMEOUT):
        raise TimeoutError("process synchronization deadline exceeded")
    return pipe.recv()


def writer(db, stage_id, lease, pipe):
    class GatedConnection(sqlite3.Connection):
        def commit(self):
            # This is the real store transaction after its SQL, before COMMIT.
            rows = self.execute(
                "SELECT artifact_id, content_sha256, byte_size, complete FROM artifacts"
            ).fetchall()
            pipe.send({"event": "uncommitted", "rows": rows, "pid": os.getpid()})
            action = receive(pipe)
            if action == "rollback":
                raise RuntimeError("controlled transaction failure")
            if action != "commit":
                raise ValueError(action)
            super().commit()

    connection = sqlite3.connect(db, isolation_level=None, factory=GatedConnection)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        try:
            LedgerStore(connection).commit_stage(
                stage_id, RunState.SUCCEEDED, lease=lease,
                artifacts=[ArtifactRef(IDENTITY, "fixture/artifact", len(PAYLOAD), True)],
                now=NOW + timedelta(seconds=1),
            )
        except RuntimeError as error:
            if str(error) != "controlled transaction failure":
                raise
            assert not connection.in_transaction
            pipe.send({"event": "rolled_back"})
        else:
            pipe.send({"event": "committed"})
    finally:
        connection.close()
        pipe.close()


def reader(db, boundary, pipe, omit_transaction=False):
    real_open = projection._open_readonly
    fired = False

    def open_with_barrier(path):
        connection = real_open(path)

        def trace(sql):
            nonlocal fired
            normalized = " ".join(sql.split())
            matches = {
                "before_snapshot": normalized == "BEGIN",
                "mid_watermark": normalized == "SELECT COUNT(*) FROM artifacts",
                "after_watermark": normalized.startswith("SELECT logical_run_id, requested_date"),
                "before_artifacts": normalized.startswith("SELECT a.artifact_type,"),
            }
            if not fired and matches.get(boundary, False):
                fired = True
                pipe.send({"event": "reader_barrier", "sql": normalized, "pid": os.getpid()})
                # sqlite trace callbacks swallow exceptions: report explicitly
                # and let the parent fail rather than silently accepting a run.
                try:
                    assert receive(pipe) == "continue"
                except BaseException as error:
                    pipe.send({"event": "barrier_error", "error": repr(error)})

        connection.set_trace_callback(trace)
        return connection

    try:
        # Instrument connection creation only; all SQL and transaction handling
        # remain the production implementation and real SQLite engine.
        with patch.object(projection, "_open_readonly", open_with_barrier):
            if omit_transaction:
                # Explicit negative control, never a production execution path.
                connection = open_with_barrier(Path(db))
                try:
                    state = projection._build_snapshot(connection)
                finally:
                    connection.close()
            else:
                state = projection.build_current_state(db)
        pipe.send({"event": "state", "state": state, "barrier_fired": fired})
    finally:
        pipe.close()


def negative_control_reader(db, boundary, pipe):
    reader(db, boundary, pipe, omit_transaction=True)


class ProjectionProcessIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "ledger.sqlite"
        connection = open_ledger(self.db)
        try:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            store = LedgerStore(connection)
            self.lease = acquire_lease(
                connection, "fixture", "owner", now=NOW, ttl=timedelta(minutes=5)
            )
            logical = store.open_logical_run(
                LogicalRunIdentity(requested_date=date(2026, 10, 20)), now=NOW
            )
            attempt = store.start_attempt(logical, lease=self.lease, host="fixture", now=NOW)
            self.stage = store.start_stage(attempt.attempt_id, "ingest_raw", "v1", now=NOW)
        finally:
            connection.close()
        self.before = projection.build_current_state(self.db)
        self.context = multiprocessing.get_context("spawn")
        self.children = []
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for process, pipe in self.children:
            if process.is_alive():
                process.kill()
            process.join(TIMEOUT)
            pipe.close()
            process.close()

    def start(self, target, *args):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=target, args=(*args, child))
        process.start()
        child.close()
        self.children.append((process, parent))
        return process, parent

    def start_writer(self):
        process, pipe = self.start(writer, str(self.db), self.stage.stage_run_id, self.lease)
        ready = receive(pipe)
        self.assertEqual(ready["event"], "uncommitted")
        self.assertNotEqual(ready["pid"], os.getpid())
        self.assertEqual(ready["rows"], [(IDENTITY.artifact_id, DIGEST, len(PAYLOAD), 1)])
        return process, pipe, ready

    def read_process(self):
        process, pipe = self.start(reader, str(self.db), None)
        result = receive(pipe)
        self.assertEqual(result["event"], "state")
        self.finish(process)
        return result["state"]

    def finish(self, process, expected=0):
        process.join(TIMEOUT)
        self.assertFalse(process.is_alive(), "child did not exit")
        self.assertEqual(process.exitcode, expected)

    def assert_generation(self, state):
        values = dict(state)
        digest = values.pop("source_generation")
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(digest, hashlib.sha256(encoded.encode()).hexdigest())

    def record(self, name, **values):
        record = {
            "case": name, "class": "REAL_PROCESS_SYNTHETIC_LEDGER",
            "sqlite_version": sqlite3.sqlite_version, "journal_mode": "wal",
            "artifact_identity": IDENTITY.artifact_id, "content_sha256": DIGEST,
            "before": self.before, **values,
        }
        print(json.dumps({
            "case": name, "before_generation": self.before["source_generation"],
            "during_generation": values["during"]["source_generation"],
            "after_generation": values["after"]["source_generation"],
            "writer_exitcode": values["writer_exitcode"],
        }, sort_keys=True), flush=True)
        destination = os.environ.get("TASK4_ISOLATION_EVIDENCE_DIR")
        if destination:
            # Explicit opt-in test evidence output; never overwrite a prior run.
            with (Path(destination) / f"{name}.json").open("x") as stream:
                json.dump(record, stream, sort_keys=True, indent=2)
                stream.write("\n")

    def uncommitted_case(self, action):
        process, pipe, ready = self.start_writer()
        during = self.read_process()
        self.assertEqual(during, self.before)
        if action == "kill":
            process.kill()
            self.finish(process, -9)
            outcome = "SIGKILL_BEFORE_COMMIT"
        else:
            pipe.send(action)
            outcome = receive(pipe)["event"]
            self.assertEqual(outcome, "committed" if action == "commit" else "rolled_back")
            self.finish(process)
        after = self.read_process()
        if action == "commit":
            self.assertNotEqual(after["source_generation"], self.before["source_generation"])
            self.assertEqual(after["artifact_summary"]["count"], 1)
            self.assertEqual(after["artifact_summary"]["total_bytes"], len(PAYLOAD))
            self.assertEqual(after["source_watermark"]["row_counts"]["artifacts"], 1)
            self.assertEqual(after["source_watermark"]["row_counts"]["stage_artifacts"], 1)
            self.assertIsNone(after["active_stage"])
        else:
            self.assertEqual(after, self.before)
        self.assert_generation(after)
        with sqlite3.connect(self.db) as connection:
            rows = connection.execute(
                "SELECT artifact_id, content_sha256, byte_size, complete FROM artifacts"
            ).fetchall()
            self.assertEqual(rows, ready["rows"] if action == "commit" else [])
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.record(action, staged=ready, during=during, after=after,
                    outcome=outcome, writer_exitcode=process.exitcode, artifact_rows=rows)

    def test_uncommitted_then_commit(self):
        self.uncommitted_case("commit")

    def test_uncommitted_then_rollback(self):
        self.uncommitted_case("rollback")

    def test_uncommitted_writer_process_failure(self):
        self.uncommitted_case("kill")

    def test_negative_control_detects_missing_read_transaction(self):
        process, pipe, ready = self.start_writer()
        reading, reader_pipe = self.start(
            negative_control_reader, str(self.db), "after_watermark"
        )
        barrier = receive(reader_pipe)
        self.assertEqual(barrier["event"], "reader_barrier")
        pipe.send("commit")
        self.assertEqual(receive(pipe)["event"], "committed")
        self.finish(process)
        after = self.read_process()
        reader_pipe.send("continue")
        result = receive(reader_pipe)
        self.assertEqual(result["event"], "state")
        self.finish(reading)
        mixed = result["state"]
        self.assertNotEqual(mixed, self.before)
        self.assertNotEqual(mixed, after)
        self.assertEqual(mixed["source_watermark"]["row_counts"]["artifacts"], 0)
        self.assertEqual(mixed["artifact_summary"]["count"], 1)
        self.record(
            "negative-control-no-BEGIN", staged=ready, barrier=barrier,
            during=mixed, after=after, writer_exitcode=process.exitcode,
            reader_exitcode=reading.exitcode, expected="MIXED_VERSION_DETECTED",
        )

    def test_commit_at_deterministic_reader_boundaries(self):
        # Separate ledgers for each legal ordering; no sleep or race lottery.
        for boundary in ("before_snapshot", "mid_watermark", "after_watermark", "before_artifacts"):
            for repetition in range(2):
                with self.subTest(boundary=boundary, repetition=repetition):
                    case = ProjectionProcessIsolationTest()
                    case.setUp()
                    try:
                        process, pipe, ready = case.start_writer()
                        reading, reader_pipe = case.start(reader, str(case.db), boundary)
                        barrier = receive(reader_pipe)
                        case.assertEqual(barrier["event"], "reader_barrier")
                        case.assertNotEqual(barrier["pid"], ready["pid"])
                        pipe.send("commit")
                        case.assertEqual(receive(pipe)["event"], "committed")
                        case.finish(process)
                        after = case.read_process()
                        reader_pipe.send("continue")
                        result = receive(reader_pipe)
                        case.assertEqual(result["event"], "state")
                        case.assertTrue(result["barrier_fired"])
                        case.finish(reading)
                        expected = after if boundary == "before_snapshot" else case.before
                        case.assertEqual(result["state"], expected)
                        case.assertNotEqual(after["source_generation"], case.before["source_generation"])
                        case.assertEqual(after["artifact_summary"]["count"], 1)
                        case.assert_generation(result["state"])
                        case.assert_generation(after)
                        case.record(
                            f"{boundary}-{repetition}", staged=ready, barrier=barrier,
                            during=result["state"], after=after, writer_exitcode=process.exitcode,
                            reader_exitcode=reading.exitcode,
                        )
                    finally:
                        case.doCleanups()


if __name__ == "__main__":
    unittest.main()
