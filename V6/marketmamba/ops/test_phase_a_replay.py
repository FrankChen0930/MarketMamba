"""Fixed offline Phase A replay, process recovery and timing evidence.

The callback is a deterministic legacy-output fixture, not a real provider.
SQLite/store/shadow/projection and process boundaries are real implementations.
No model/data pipeline is started.
"""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import subprocess
import tempfile
import time
import unittest

from marketmamba.ops.contracts import LogicalRunIdentity, RunState
from marketmamba.ops.lease import acquire_lease, release_lease, LeaseConflictError, StaleFenceError
from marketmamba.ops.projection import build_current_state, write_current_state
from marketmamba.ops.schema import open_ledger, MIGRATIONS
from marketmamba.ops.shadow import observe_fetch_only, artifact_ref_from_stat
from marketmamba.ops.store import LedgerStore


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
SOURCES = ("prices_raw", "margin_raw")
GOOD = ({name: True for name in SOURCES}, [])
CASES = (
    ("clean_success", ("good",)),
    ("missing_source", ("missing",)),
    ("freshness_failure", ("stale",)),
    ("nonzero_exit", ("exit1",)),
    ("transient_provider_failure_then_success", ("exception", "good")),
    ("retry_same_logical_run", ("good", "good")),
    ("duplicate_hygiene_warning", ("warning",)),
    ("source_count_mismatch_not_success_authority", ("count_mismatch",)),
    ("projection_failure_preserves_output", ("good",)),
    ("missing_artifact_not_success_authority", ("good",)),
)


def recv(pipe):
    if not pipe.poll(15):
        raise TimeoutError("fixed process handshake timed out")
    return pipe.recv()


def lifecycle_child(db, artifact, mode, pipe):
    connection = open_ledger(db)
    try:
        store = LedgerStore(connection)
        when = T0 + timedelta(seconds=61) if mode == "recover" else T0
        lease = acquire_lease(connection, "replay", mode, now=when, ttl=timedelta(seconds=60))
        logical = store.open_logical_run(LogicalRunIdentity(date(2026, 1, 1)), now=when)
        attempt = store.start_attempt(logical, lease=lease, now=when, host="offline-child")
        stage = store.start_stage(attempt.attempt_id, "ingest_raw", "fixed-input", now=when)
        if mode == "crash":
            pipe.send({"logical": logical, "attempt": attempt.attempt_id,
                       "stage": stage.stage_run_id, "fence": lease.fencing_token})
            recv(pipe)  # Parent kills this process after durable RUNNING.
            raise AssertionError("crash child must not resume")
        ref = artifact_ref_from_stat(artifact, data_root=Path(artifact).parent, complete=True)
        store.commit_stage(stage.stage_run_id, RunState.SUCCEEDED, lease=lease,
                           artifacts=[ref], now=when)
        store.finish_attempt(attempt.attempt_id, RunState.SUCCEEDED,
                             lease=lease, exit_code=0, now=when)
        store.finish_logical_run(logical, RunState.SUCCEEDED, lease=lease, now=when)
        pipe.send({"logical": logical, "attempt": attempt.attempt_id,
                   "fence": lease.fencing_token, "artifact": ref.identity.artifact_id})
    finally:
        connection.close()
        pipe.close()


def lease_child(db, owner, hold, pipe):
    connection = open_ledger(db)
    try:
        try:
            lease = acquire_lease(connection, "concurrent", owner, now=T0,
                                  ttl=timedelta(minutes=5))
        except LeaseConflictError:
            pipe.send({"result": "CONFLICT"})
            return
        pipe.send({"result": "ACQUIRED", "fence": lease.fencing_token})
        if hold:
            assert recv(pipe) == "release"
        release_lease(connection, lease, now=T0)
    finally:
        connection.close()
        pipe.close()


def failing_callback_child(db, data, pipe):
    def fail():
        raise RuntimeError("synthetic child fetch failure")
    try:
        observe_fetch_only(
            fail, daily_sources=SOURCES, v6_dir=Path(data).parent / "V6",
            ledger_path=db, data_root=data, requested_at=T0,
            projection_callback=write_current_state, environ={},
        )
    finally:
        pipe.close()


class PhaseAReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = []
        cls.timings = None
        cls.schema = None
        cls.projection_example = None

    @classmethod
    def tearDownClass(cls):
        destination = os.environ.get("PHASE_A_EVIDENCE_DIR")
        if not destination:
            return
        root = Path(destination)
        # No acceptance is emitted here. Partial/failed test runs cannot approve
        # anything; these are raw observations to be reviewed with exit status.
        outputs = {
            "shadow-replay-results.json": {
                "evidence_class": "OFFLINE_SYNTHETIC_CALLBACKS_REAL_SQLITE_AND_PROCESSES",
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                "cases": cls.records, "benchmark": cls.timings,
                "projection_example": cls.projection_example,
                "human_acceptance": "NOT_GRANTED",
            },
            "schema.json": cls.schema,
        }
        for name, value in outputs.items():
            with (root / name).open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "ledger.sqlite"
        self.data = self.root / "Data"
        processed = self.data / "processed_v6"
        processed.mkdir(parents=True)
        for name in SOURCES:
            path = processed / f"{name}.parquet"
            path.write_bytes(b"synthetic stat-only artifact: " + name.encode())
            os.utime(path, ns=(1700000000000000000, 1700000000000000000))
        self.ctx = multiprocessing.get_context("spawn")
        self.children = []
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for process, pipe in self.children:
            if process.is_alive():
                process.kill()
            process.join(15)
            pipe.close()
            process.close()

    def spawn(self, target, *args):
        parent, child = self.ctx.Pipe()
        process = self.ctx.Process(target=target, args=(*args, child))
        process.start()
        child.close()
        self.children.append((process, parent))
        return process, parent

    def finish(self, process, exitcode=0):
        process.join(15)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, exitcode)

    def observe(self, operation, callback=write_current_state):
        return observe_fetch_only(
            operation, daily_sources=SOURCES, v6_dir=self.root / "V6",
            ledger_path=self.db, data_root=self.data, requested_at=T0,
            projection_callback=callback, environ={},
        )

    def snapshot_schema(self):
        connection = open_ledger(self.db)
        try:
            type(self).schema = {
                "schema_version": MIGRATIONS[-1].version,
                "migrations": [{"version": m.version, "checksum": m.checksum} for m in MIGRATIONS],
                "sqlite_master": connection.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
                ).fetchall(),
            }
        finally:
            connection.close()

    def test_fixed_observed_vs_shadow_replays(self):
        for name, actions in CASES:
            with self.subTest(case=name):
                case = PhaseAReplayTest()
                case.setUp()
                try:
                    if name == "missing_artifact_not_success_authority":
                        # Fixture-only omission; never points at real data.
                        (case.data / "processed_v6/margin_raw.parquet").unlink()
                    results = []
                    for ordinal, action in enumerate(actions, 1):
                        if action == "missing":
                            value = ({"prices_raw": True, "margin_raw": False}, ["margin_raw"])
                        elif action == "stale":
                            value = (dict(GOOD[0]), ["freshness:previous_date"])
                        elif action == "exit1":
                            value = (dict(GOOD[0]), ["legacy exit gate failed"])
                        elif action == "count_mismatch":
                            value = ({"prices_raw": True, "margin_raw": False}, [])
                        else:
                            value = GOOD
                        calls = []
                        error = RuntimeError("synthetic transient provider failure")

                        def operation():
                            calls.append("called")
                            if action == "exception":
                                raise error
                            if action == "warning":
                                logging.getLogger("phase-a-fixture").warning("synthetic duplicate hygiene warning")
                            return value

                        callback = write_current_state
                        if name == "projection_failure_preserves_output":
                            # A real filesystem failure, not mocked success.
                            target = case.root / "target-is-directory"
                            target.mkdir(exist_ok=True)
                            callback = lambda db: write_current_state(db, target)
                        if action == "exception":
                            with case.assertRaises(RuntimeError) as raised:
                                case.observe(operation, callback)
                            case.assertIs(raised.exception, error)
                            expected_state, expected_exit = "FAILED", 1
                            returned = "SAME_EXCEPTION_OBJECT"
                        else:
                            observed = case.observe(operation, callback)
                            case.assertIs(observed, value)
                            expected_state = "FAILED" if value[1] else "SUCCEEDED"
                            expected_exit = 1 if value[1] else 0
                            returned = observed
                        case.assertEqual(len(calls), 1)
                        state = build_current_state(case.db)
                        case.assertEqual(state["run_status"], expected_state)
                        case.assertEqual(state["latest_exit_code"], expected_exit)
                        case.assertEqual(state["attempt_count"], ordinal)
                        case.assertEqual(state["latest_attempt"], ordinal)
                        with sqlite3.connect(case.db) as connection:
                            case.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                            artifacts = connection.execute(
                                "SELECT artifact_id,fingerprint_kind,content_sha256,fingerprint_sha256 "
                                "FROM artifacts ORDER BY artifact_id"
                            ).fetchall()
                        for row in artifacts:
                            case.assertEqual(row[1], "file_stat_v1")
                            case.assertIsNone(row[2])
                        expected_rows = []
                        if action != "exception":
                            for source in SOURCES:
                                path = case.data / "processed_v6" / f"{source}.parquet"
                                if path.exists():
                                    ref = artifact_ref_from_stat(
                                        path, data_root=case.data,
                                        complete=bool(value[0].get(source, False)),
                                    )
                                    expected_rows.append((
                                        ref.identity.artifact_id, "file_stat_v1", None,
                                        ref.identity.fingerprint_sha256,
                                    ))
                        case.assertEqual(artifacts, sorted(expected_rows))
                        results.append({"action": action, "legacy_return": returned,
                                        "legacy_exit": expected_exit, "shadow_projection": state,
                                        "artifact_rows": artifacts, "calls": len(calls)})
                    self.records.append({
                        "case": name, "match": True, "attempts": results,
                        "limit": "Hygiene log is not exposed by legacy result; no Phase C warning certification."
                        if name == "duplicate_hygiene_warning" else None,
                    })
                    if name == "clean_success":
                        type(self).projection_example = state
                        case.snapshot_schema()
                finally:
                    self.assertTrue(case.doCleanups())

    def test_crash_running_artifact_before_ledger_and_restart(self):
        artifact = self.root / "orphan-artifact"
        artifact.write_bytes(b"fixed pre-ledger artifact")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        process, pipe = self.spawn(lifecycle_child, str(self.db), str(artifact), "crash")
        checkpoint = recv(pipe)
        before = build_current_state(self.db)
        self.assertEqual(before["latest_attempt_status"], "RUNNING")
        self.assertEqual(before["artifact_summary"]["count"], 0)
        process.kill()
        self.finish(process, -9)
        self.assertEqual(build_current_state(self.db), before)
        recovery, recovered_pipe = self.spawn(lifecycle_child, str(self.db), str(artifact), "recover")
        recovered = recv(recovered_pipe)
        self.finish(recovery)
        self.assertEqual(checkpoint["logical"], recovered["logical"])
        self.assertNotEqual(checkpoint["attempt"], recovered["attempt"])
        self.assertEqual(recovered["fence"], checkpoint["fence"] + 1)
        state = build_current_state(self.db)
        self.assertEqual(state["run_status"], "SUCCEEDED")
        self.assertEqual(state["attempt_count"], 2)
        self.assertEqual(state["artifact_summary"]["count"], 1)
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), digest)
        with sqlite3.connect(self.db) as connection:
            attempts = connection.execute("SELECT attempt_number,state FROM attempts ORDER BY attempt_number").fetchall()
            self.assertEqual(attempts, [(1, "RUNNING"), (2, "SUCCEEDED")])
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.records.append({
            "case": "crash_running_artifact_before_ledger_restart",
            "checkpoint": checkpoint, "recovered": recovered, "old_attempts_preserved": attempts,
            "artifact_sha256_before_and_after": digest, "writer_exit": -9, "restart_exit": 0,
            "projection": state,
            "limit": "Explicit fixed test clock crosses lease expiry. Old RUNNING remains historical, not falsely completed. Ledger is the checkpoint, not a model checkpoint; no Phase D automatic resume claim.",
        })

    def test_concurrent_lease_ownership(self):
        connection = open_ledger(self.db)
        connection.close()
        owner, owner_pipe = self.spawn(lease_child, str(self.db), "owner", True)
        first = recv(owner_pipe)
        self.assertEqual(first["result"], "ACQUIRED")
        competitor, competing_pipe = self.spawn(lease_child, str(self.db), "competitor", False)
        denied = recv(competing_pipe)
        self.finish(competitor)
        self.assertEqual(denied["result"], "CONFLICT")
        owner_pipe.send("release")
        self.finish(owner)
        successor, successor_pipe = self.spawn(lease_child, str(self.db), "successor", False)
        second = recv(successor_pipe)
        self.finish(successor)
        self.assertEqual(second["fence"], first["fence"] + 1)
        self.records.append({"case": "concurrent_owners", "first": first, "competitor": denied,
                             "successor": second, "exitcodes": [0, 0, 0]})

    def test_stale_fence_and_active_shadow_conflict(self):
        connection = open_ledger(self.db)
        try:
            old = acquire_lease(connection, "stale", "old", now=T0, ttl=timedelta(seconds=1))
            new = acquire_lease(connection, "stale", "new", now=T0 + timedelta(seconds=2),
                                ttl=timedelta(minutes=1))
            logical = LedgerStore(connection).open_logical_run(LogicalRunIdentity(date(2026, 1, 1)), now=T0)
            with self.assertRaises(StaleFenceError):
                LedgerStore(connection).start_attempt(logical, lease=old, now=T0 + timedelta(seconds=2))
            self.assertEqual(new.fencing_token, old.fencing_token + 1)
            active = acquire_lease(connection, "v62-fetch-only", "held", ttl=timedelta(minutes=5))
            result = self.observe(lambda: GOOD)
            self.assertIs(result, GOOD)
            count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            self.assertEqual(count, 0)
            release_lease(connection, active)
            self.records.append({"case": "stale_fence_and_active_shadow_conflict",
                                 "stale_rejected": True, "fences": [old.fencing_token, new.fencing_token],
                                 "legacy_return_unchanged": True, "shadow_attempts": count,
                                 "limit": "Active competing lease produces NO shadow observation, not a successful shadow run."})
        finally:
            connection.close()

    def test_real_nonzero_child_exit_matches_failed_shadow(self):
        child, _ = self.spawn(failing_callback_child, str(self.db), str(self.data))
        self.finish(child, 1)
        state = build_current_state(self.db)
        self.assertEqual(state["run_status"], "FAILED")
        self.assertEqual(state["latest_exit_code"], child.exitcode)
        self.assertEqual(state["freshness"]["status"], "FAIL")
        self.assertEqual(
            json.loads((self.root / "current-state.json").read_text()), state
        )
        self.records.append({
            "case": "real_nonzero_callback_child", "child_exitcode": child.exitcode,
            "projection": state,
            "limit": "Real child process using synthetic failing callback, not live provider or production CLI.",
        })

    def test_reproducible_overhead_benchmark(self):
        # Fixed protocol, measured elapsed time is machine-dependent.
        baseline, observed = [], []
        for _ in range(10):
            start = time.perf_counter_ns()
            result = (lambda: GOOD)()
            baseline.append(time.perf_counter_ns() - start)
            self.assertIs(result, GOOD)
            start = time.perf_counter_ns()
            self.assertIs(self.observe(lambda: GOOD), GOOD)
            observed.append(time.perf_counter_ns() - start)
        self.assertEqual(build_current_state(self.db)["attempt_count"], 10)
        type(self).timings = {
            "protocol": "10 paired callbacks vs real shadow+projection on one fresh ledger; first includes migrations, subsequent attempts grow same logical run",
            "baseline_ns": baseline, "shadow_ns": observed,
            "median_added_ns": statistics.median(observed) - statistics.median(baseline),
            "max_shadow_ns": max(observed),
            "threshold": "NONE_PREDECLARED; measurement, not a production performance approval",
            "limits": "Synthetic stat-only artifacts; excludes provider/network cost and large-ledger extrapolation.",
        }


if __name__ == "__main__":
    unittest.main()
