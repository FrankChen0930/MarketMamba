"""Focused transactional store and SQLite lease/fencing tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from marketmamba.ops.contracts import (
    ArtifactIdentity,
    CheckIdentity,
    CheckStatus,
    LogicalRunIdentity,
    RunState,
)
from marketmamba.ops.lease import (
    LeaseConflictError,
    StaleFenceError,
    acquire_lease,
    heartbeat_lease,
    release_lease,
)
from marketmamba.ops.schema import BUSY_TIMEOUT_MS, open_ledger
from marketmamba.ops.store import (
    ArtifactRef,
    CheckResult,
    LedgerStore,
    TerminalStateError,
)


UTC = timezone.utc
T0 = datetime(2026, 9, 17, 9, 0, tzinfo=UTC)


class StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "ledger.sqlite3"
        self.connection = open_ledger(self.database_path)
        self.addCleanup(self.connection.close)
        self.store = LedgerStore(self.connection)
        self.logical = LogicalRunIdentity(requested_date=date(2026, 9, 17))

    def start_attempt_and_stage(self) -> tuple[str, str]:
        logical_run_id = self.store.open_logical_run(self.logical, now=T0)
        attempt = self.store.start_attempt(
            logical_run_id,
            host="host-a",
            process_id=101,
            code_identity="commit-a",
            dependency_identity="lock-a",
            now=T0,
        )
        stage = self.store.start_stage(
            attempt.attempt_id,
            "fetch_prices",
            "inputs-v1",
            now=T0,
        )
        return attempt.attempt_id, stage.stage_run_id

    def acquire(self, owner: str = "owner-a"):
        return acquire_lease(
            self.connection,
            "daily-fetch",
            owner,
            now=T0,
            ttl=timedelta(minutes=5),
        )


class TransactionalStoreTests(StoreTestCase):
    def test_retry_same_logical_run_adds_attempt_without_rewriting_history(self) -> None:
        logical_run_id = self.store.open_logical_run(self.logical, now=T0)
        lease = self.acquire()
        first = self.store.start_attempt(logical_run_id, host="host-a", now=T0)
        self.store.finish_attempt(
            first.attempt_id,
            RunState.FAILED,
            lease=lease,
            exit_code=1,
            incident_ref="incident-1",
            now=T0 + timedelta(minutes=1),
        )
        second = self.store.start_attempt(
            logical_run_id,
            host="host-b",
            now=T0 + timedelta(minutes=2),
        )

        rows = self.connection.execute(
            "SELECT attempt_id, attempt_number, state, host, exit_code "
            "FROM attempts ORDER BY attempt_number"
        ).fetchall()
        self.assertEqual(
            rows,
            [
                (first.attempt_id, 1, "FAILED", "host-a", 1),
                (second.attempt_id, 2, "RUNNING", "host-b", None),
            ],
        )
        self.assertNotEqual(first.attempt_id, second.attempt_id)

    def test_terminal_attempt_logical_run_and_stage_are_immutable(self) -> None:
        attempt_id, stage_run_id = self.start_attempt_and_stage()
        lease = self.acquire()
        self.store.commit_stage(
            stage_run_id,
            RunState.SUCCEEDED,
            lease=lease,
            now=T0 + timedelta(seconds=10),
        )
        self.store.finish_attempt(
            attempt_id,
            RunState.SUCCEEDED,
            lease=lease,
            exit_code=0,
            now=T0 + timedelta(seconds=20),
        )
        self.store.finish_logical_run(
            self.logical.logical_run_id,
            RunState.SUCCEEDED,
            lease=lease,
            terminal_reason="complete",
            now=T0 + timedelta(seconds=30),
        )

        with self.assertRaises(TerminalStateError):
            self.store.commit_stage(
                stage_run_id,
                RunState.FAILED,
                lease=lease,
                now=T0 + timedelta(seconds=40),
            )
        with self.assertRaises(TerminalStateError):
            self.store.finish_attempt(
                attempt_id,
                RunState.FAILED,
                lease=lease,
                exit_code=2,
                now=T0 + timedelta(seconds=40),
            )
        with self.assertRaises(TerminalStateError):
            self.store.finish_logical_run(
                self.logical.logical_run_id,
                RunState.FAILED,
                lease=lease,
                now=T0 + timedelta(seconds=40),
            )

    def test_terminal_stage_commits_artifacts_checks_and_status_atomically(self) -> None:
        _, stage_run_id = self.start_attempt_and_stage()
        lease = self.acquire()
        artifact_identity = ArtifactIdentity(
            artifact_type="raw_snapshot",
            content_sha256="b" * 64,
            schema_version="v1",
            schema_sha256="a" * 64,
        )
        check_identity = CheckIdentity(scope=stage_run_id, check_name="row_count")

        self.store.commit_stage(
            stage_run_id,
            RunState.SUCCEEDED,
            lease=lease,
            artifacts=[
                ArtifactRef(
                    identity=artifact_identity,
                    uri="file:///data/raw.parquet",
                    byte_size=42,
                    complete=True,
                )
            ],
            checks=[
                CheckResult(
                    identity=check_identity,
                    status=CheckStatus.PASS,
                    severity="INFO",
                    observed_value="42",
                    threshold_value=">=1",
                )
            ],
            now=T0 + timedelta(seconds=10),
        )

        self.assertEqual(
            self.connection.execute(
                "SELECT state, finished_at FROM stage_runs WHERE stage_run_id = ?",
                (stage_run_id,),
            ).fetchone(),
            ("SUCCEEDED", "2026-09-17T09:00:10.000000Z"),
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT artifact_id, stage_run_id, complete FROM artifacts"
            ).fetchall(),
            [(artifact_identity.artifact_id, stage_run_id, 1)],
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT check_id, stage_run_id, status FROM checks"
            ).fetchall(),
            [(check_identity.check_id, stage_run_id, "PASS")],
        )

    def test_failed_artifact_insert_rolls_back_entire_terminal_commit(self) -> None:
        _, stage_run_id = self.start_attempt_and_stage()
        lease = self.acquire()
        valid = ArtifactIdentity("raw_snapshot", "b" * 64, "v1", "a" * 64)
        invalid = ArtifactIdentity("raw_snapshot_2", "c" * 64, "v1", "a" * 64)

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.commit_stage(
                stage_run_id,
                RunState.SUCCEEDED,
                lease=lease,
                artifacts=[
                    ArtifactRef(valid, "file:///valid", 42, True),
                    ArtifactRef(invalid, "file:///invalid", -1, True),
                ],
                now=T0 + timedelta(seconds=10),
            )

        self.assertEqual(
            self.connection.execute(
                "SELECT state, finished_at FROM stage_runs WHERE stage_run_id = ?",
                (stage_run_id,),
            ).fetchone(),
            ("RUNNING", None),
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 0
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM checks").fetchone()[0], 0
        )

    def test_persisted_errors_evidence_and_uris_are_redacted(self) -> None:
        secret_blob = (
            "Authorization: Bearer sk-live-ABC123\n"
            "https://api.example.test/data?token=url-secret&ok=1 "
            "https://api.telegram.org/bot123456789:AAExampleTelegramSecret123/sendMessage "
            "password=hunter2"
        )
        logical_run_id = self.store.open_logical_run(self.logical, now=T0)
        attempt = self.store.start_attempt(
            logical_run_id,
            host=secret_blob,
            code_identity=secret_blob,
            dependency_identity=secret_blob,
            now=T0,
        )
        stage = self.store.start_stage(
            attempt.attempt_id,
            "fetch_prices",
            "inputs-v1",
            now=T0,
        )
        attempt_id = attempt.attempt_id
        stage_run_id = stage.stage_run_id
        lease = self.acquire()
        artifact = ArtifactIdentity("error_evidence", "b" * 64, "v1", "a" * 64)
        check = CheckIdentity(stage_run_id, "provider_error")

        self.store.commit_stage(
            stage_run_id,
            RunState.FAILED,
            lease=lease,
            artifacts=[ArtifactRef(artifact, secret_blob, 1, False)],
            checks=[
                CheckResult(
                    check,
                    CheckStatus.FAIL,
                    "ERROR",
                    observed_value=secret_blob,
                    threshold_value="password=expected-secret",
                )
            ],
            retryable=True,
            error_fingerprint=secret_blob,
            incident_ref=secret_blob,
            now=T0 + timedelta(seconds=10),
        )
        self.store.finish_attempt(
            attempt_id,
            RunState.FAILED,
            lease=lease,
            exit_code=1,
            incident_ref=secret_blob,
            now=T0 + timedelta(seconds=20),
        )

        persisted = "\n".join(
            str(value)
            for row in self.connection.execute(
                "SELECT error_fingerprint, incident_ref FROM stage_runs "
                "UNION ALL SELECT observed_value, threshold_value FROM checks "
                "UNION ALL SELECT uri, NULL FROM artifacts "
                "UNION ALL SELECT incident_ref, host FROM attempts "
                "UNION ALL SELECT code_identity, dependency_identity FROM attempts"
            )
            for value in row
            if value is not None
        )
        for secret in (
            "sk-live-ABC123",
            "url-secret",
            "123456789:AAExampleTelegramSecret123",
            "hunter2",
            "expected-secret",
        ):
            self.assertNotIn(secret, persisted)
        self.assertGreaterEqual(persisted.count("[REDACTED]"), 5)


class LeaseFencingTests(StoreTestCase):
    def test_two_real_connections_racing_have_exactly_one_lease_winner(self) -> None:
        self.connection.close()
        barrier = threading.Barrier(2)
        outcomes: list[tuple[str, str]] = []
        outcomes_lock = threading.Lock()

        def contender(owner_id: str) -> None:
            connection = open_ledger(self.database_path)
            try:
                barrier.wait()
                try:
                    lease = acquire_lease(
                        connection,
                        "daily-fetch",
                        owner_id,
                        now=T0,
                        ttl=timedelta(minutes=5),
                    )
                except LeaseConflictError:
                    outcome = (owner_id, "conflict")
                else:
                    outcome = (owner_id, f"won:{lease.fencing_token}")
                with outcomes_lock:
                    outcomes.append(outcome)
            finally:
                connection.close()

        threads = [
            threading.Thread(target=contender, args=("owner-a",)),
            threading.Thread(target=contender, args=("owner-b",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(
            sorted(result.split(":")[0] for _, result in outcomes),
            ["conflict", "won"],
        )
        self.assertEqual(
            [result for _, result in outcomes if result.startswith("won:")],
            ["won:1"],
        )

    def test_stale_takeover_and_release_keep_fencing_token_monotonic(self) -> None:
        first = self.acquire("owner-a")
        second = acquire_lease(
            self.connection,
            "daily-fetch",
            "owner-b",
            now=T0 + timedelta(minutes=6),
            ttl=timedelta(minutes=5),
        )
        release_lease(
            self.connection,
            second,
            now=T0 + timedelta(minutes=7),
        )
        third = acquire_lease(
            self.connection,
            "daily-fetch",
            "owner-c",
            now=T0 + timedelta(minutes=7, seconds=1),
            ttl=timedelta(minutes=5),
        )

        self.assertEqual(
            (first.fencing_token, second.fencing_token, third.fencing_token),
            (1, 2, 3),
        )

    def test_old_holder_cannot_heartbeat_finalize_attempt_or_commit_stage(self) -> None:
        attempt_id, stage_run_id = self.start_attempt_and_stage()
        old = self.acquire("owner-a")
        current = acquire_lease(
            self.connection,
            "daily-fetch",
            "owner-b",
            now=T0 + timedelta(minutes=6),
            ttl=timedelta(minutes=5),
        )

        with self.assertRaises(StaleFenceError):
            heartbeat_lease(
                self.connection,
                old,
                now=T0 + timedelta(minutes=6, seconds=1),
                ttl=timedelta(minutes=5),
            )
        with self.assertRaises(StaleFenceError):
            self.store.finish_attempt(
                attempt_id,
                RunState.FAILED,
                lease=old,
                exit_code=1,
                now=T0 + timedelta(minutes=6, seconds=1),
            )
        with self.assertRaises(StaleFenceError):
            self.store.commit_stage(
                stage_run_id,
                RunState.SUCCEEDED,
                lease=old,
                now=T0 + timedelta(minutes=6, seconds=1),
            )

        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()[0],
            "RUNNING",
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM stage_runs WHERE stage_run_id = ?", (stage_run_id,)
            ).fetchone()[0],
            "RUNNING",
        )
        heartbeat_lease(
            self.connection,
            current,
            now=T0 + timedelta(minutes=6, seconds=1),
            ttl=timedelta(minutes=5),
        )

    def test_connection_safety_pragmas_apply_to_each_real_connection(self) -> None:
        second = open_ledger(self.database_path)
        self.addCleanup(second.close)
        for connection in (self.connection, second):
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertGreaterEqual(
                connection.execute("PRAGMA busy_timeout").fetchone()[0],
                BUSY_TIMEOUT_MS,
            )


if __name__ == "__main__":
    unittest.main()
