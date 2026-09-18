"""Regression tests for Task 2 review findings."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from marketmamba.ops.contracts import ArtifactIdentity, LogicalRunIdentity, RunState
from marketmamba.ops.lease import (
    StaleFenceError,
    acquire_lease,
    heartbeat_lease,
    immediate_transaction,
)
from marketmamba.ops.schema import open_ledger
from marketmamba.ops.store import (
    ArtifactRef,
    LedgerStateError,
    LedgerStore,
    redact_sensitive_text,
)


UTC = timezone.utc
T0 = datetime(2026, 9, 17, 9, 0, tzinfo=UTC)


class ReviewStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "ledger.sqlite3"
        self.connection = open_ledger(self.database_path)
        self.addCleanup(self.connection.close)
        self.store = LedgerStore(self.connection)
        self.logical = LogicalRunIdentity(requested_date=date(2026, 9, 17))
        self.logical_run_id = self.store.open_logical_run(self.logical, now=T0)
        self.lease = acquire_lease(
            self.connection,
            "daily-fetch",
            "owner-a",
            now=T0,
            ttl=timedelta(minutes=30),
        )

    def start_attempt_stage(
        self,
        *,
        attempt_time: datetime,
        stage_name: str = "fetch_prices",
    ) -> tuple[str, str]:
        attempt = self.store.start_attempt(
            self.logical_run_id,
            lease=self.lease,
            host="host-a",
            now=attempt_time,
        )
        stage = self.store.start_stage(
            attempt.attempt_id,
            stage_name,
            "inputs-v1",
            now=attempt_time,
        )
        return attempt.attempt_id, stage.stage_run_id

    @staticmethod
    def artifact_ref() -> ArtifactRef:
        return ArtifactRef(
            ArtifactIdentity("raw_snapshot", "b" * 64, "v1", "a" * 64),
            "file:///data/raw.parquet",
            42,
            True,
        )


class RedactionRegressionTests(ReviewStoreTestCase):
    def test_json_mapping_and_url_userinfo_secrets_are_redacted_before_db_write(
        self,
    ) -> None:
        json_evidence = (
            '{"api_key":"synthetic-json-secret",'
            '"password":"synthetic-json-password"}'
        )
        mapping_evidence = "{'token': 'synthetic-mapping-secret'}"
        credential_url = (
            "https://synthetic-user:synthetic-userinfo-password@example.test/path"
        )

        attempt = self.store.start_attempt(
            self.logical_run_id,
            lease=self.lease,
            host=json_evidence,
            code_identity=mapping_evidence,
            dependency_identity=credential_url,
            now=T0,
        )

        persisted = "|".join(
            value
            for value in self.connection.execute(
                "SELECT host, code_identity, dependency_identity "
                "FROM attempts WHERE attempt_id = ?",
                (attempt.attempt_id,),
            ).fetchone()
            if value is not None
        )
        for secret in (
            "synthetic-json-secret",
            "synthetic-json-password",
            "synthetic-mapping-secret",
            "synthetic-userinfo-password",
        ):
            self.assertNotIn(secret, persisted)
        self.assertGreaterEqual(persisted.count("[REDACTED]"), 4)

    def test_extended_credential_forms_share_one_redactor(self) -> None:
        evidence = (
            "authorization=Basic dXNlcjpwYXNz "
            "access_token=access-value client_secret=client-value "
            "AWS_SECRET_ACCESS_KEY=aws-value "
            "ftp://ftp-user:ftp-password@example.test/data "
            "postgresql://pg-user:pg-password@db.test/market "
            "safe=https://example.com/status"
        )

        redacted = redact_sensitive_text(evidence)

        self.assertIsNotNone(redacted)
        assert redacted is not None
        for sensitive in (
            "dXNlcjpwYXNz",
            "access-value",
            "client-value",
            "aws-value",
            "ftp-user",
            "ftp-password",
            "pg-user",
            "pg-password",
        ):
            self.assertNotIn(sensitive, redacted)
        self.assertIn("authorization=[REDACTED]", redacted)
        self.assertIn("access_token=[REDACTED]", redacted)
        self.assertIn("client_secret=[REDACTED]", redacted)
        self.assertIn("AWS_SECRET_ACCESS_KEY=[REDACTED]", redacted)
        self.assertIn("ftp://[REDACTED]@example.test/data", redacted)
        self.assertIn("postgresql://[REDACTED]@db.test/market", redacted)
        self.assertIn("safe=https://example.com/status", redacted)

    def test_equals_quoted_credentials_are_fully_redacted_before_db_write(
        self,
    ) -> None:
        evidence = (
            r'''authorization="Basic auth-one auth-two" '''
            r'''access_token='access-one access-two' '''
            r'''client_secret="client-one \"client-two\" client-three" '''
            r'''AWS_SECRET_ACCESS_KEY='aws-one \'aws-two\' aws-three' '''
            "safe=https://example.com/status "
            "relative=file:processed_v6/raw.parquet"
        )

        attempt = self.store.start_attempt(
            self.logical_run_id,
            lease=self.lease,
            host=evidence,
            now=T0,
        )
        persisted = self.connection.execute(
            "SELECT host FROM attempts WHERE attempt_id = ?",
            (attempt.attempt_id,),
        ).fetchone()[0]

        for secret_fragment in (
            "auth-one",
            "auth-two",
            "access-one",
            "access-two",
            "client-one",
            "client-two",
            "client-three",
            "aws-one",
            "aws-two",
            "aws-three",
        ):
            self.assertNotIn(secret_fragment, persisted)
        self.assertEqual(persisted.count("[REDACTED]"), 4)
        self.assertIn("safe=https://example.com/status", persisted)
    def test_url_userinfo_is_fully_redacted_without_changing_host_or_path(
        self,
    ) -> None:
        user_password_url = (
            "https://synthetic-user:synthetic-password@"
            "example.test/path?view=summary"
        )
        token_only_url = "https://synthetic-token-only@example.test/api/v1"
        normal_url = "https://example.test/path?email=user@example.test"

        attempt = self.store.start_attempt(
            self.logical_run_id,
            lease=self.lease,
            host=user_password_url,
            code_identity=token_only_url,
            dependency_identity=normal_url,
            now=T0,
        )

        persisted = self.connection.execute(
            "SELECT host, code_identity, dependency_identity "
            "FROM attempts WHERE attempt_id = ?",
            (attempt.attempt_id,),
        ).fetchone()
        self.assertEqual(
            persisted,
            (
                "https://[REDACTED]@example.test/path?view=summary",
                "https://[REDACTED]@example.test/api/v1",
                "https://example.test/path?email=user@example.test",
            ),
        )
        for credential_fragment in (
            "synthetic-user",
            "synthetic-password",
            "synthetic-token-only",
        ):
            self.assertNotIn(credential_fragment, "|".join(persisted))

    def test_escaped_json_secret_value_is_fully_redacted_before_db_write(
        self,
    ) -> None:
        escaped_json = (
            '{"api_key":"synthetic-prefix\\"synthetic-suffix","safe":"visible"}'
        )

        attempt = self.store.start_attempt(
            self.logical_run_id,
            lease=self.lease,
            host=escaped_json,
            now=T0,
        )

        persisted = self.connection.execute(
            "SELECT host FROM attempts WHERE attempt_id = ?",
            (attempt.attempt_id,),
        ).fetchone()[0]
        self.assertEqual(
            persisted,
            '{"api_key":"[REDACTED]","safe":"visible"}',
        )
        self.assertNotIn("synthetic-prefix", persisted)
        self.assertNotIn("synthetic-suffix", persisted)


class TransactionRollbackRegressionTests(ReviewStoreTestCase):
    def test_denied_commit_rolls_back_stage_artifacts_and_transaction(self) -> None:
        _, stage_run_id = self.start_attempt_stage(attempt_time=T0)
        denied_commit = False

        def deny_commit(
            action_code: int,
            argument_one: str | None,
            argument_two: str | None,
            database_name: str | None,
            trigger_name: str | None,
        ) -> int:
            nonlocal denied_commit
            del argument_two, database_name, trigger_name
            if (
                action_code == sqlite3.SQLITE_TRANSACTION
                and argument_one == "COMMIT"
                and not denied_commit
            ):
                denied_commit = True
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        self.connection.set_authorizer(deny_commit)
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                self.store.commit_stage(
                    stage_run_id,
                    RunState.SUCCEEDED,
                    lease=self.lease,
                    artifacts=[self.artifact_ref()],
                    now=T0 + timedelta(seconds=1),
                )
        finally:
            self.connection.set_authorizer(None)

        self.assertTrue(denied_commit)
        self.assertFalse(self.connection.in_transaction)
        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM stage_runs WHERE stage_run_id = ?",
                (stage_run_id,),
            ).fetchone()[0],
            "RUNNING",
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            0,
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM stage_artifacts").fetchone()[0],
            0,
        )

    def test_keyboard_interrupt_rolls_back_body_and_transaction(self) -> None:
        _, stage_run_id = self.start_attempt_stage(attempt_time=T0)

        with self.assertRaises(KeyboardInterrupt):
            with immediate_transaction(self.connection):
                self.connection.execute(
                    "UPDATE stage_runs SET state = ? WHERE stage_run_id = ?",
                    (RunState.FAILED.value, stage_run_id),
                )
                raise KeyboardInterrupt("synthetic interrupt")

        self.assertFalse(self.connection.in_transaction)
        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM stage_runs WHERE stage_run_id = ?",
                (stage_run_id,),
            ).fetchone()[0],
            "RUNNING",
        )


class LeaseClockRegressionTests(ReviewStoreTestCase):
    def test_heartbeat_reads_clock_after_write_lock_and_rejects_expired_lease(
        self,
    ) -> None:
        current_time = {"value": T0 + timedelta(minutes=29)}
        clock_called = threading.Event()
        worker_ready = threading.Event()
        start_heartbeat = threading.Event()
        worker_started = threading.Event()
        outcome: list[BaseException | None] = []

        def controlled_clock() -> datetime:
            clock_called.set()
            return current_time["value"]

        def heartbeat_worker() -> None:
            connection = open_ledger(self.database_path)
            try:
                worker_ready.set()
                start_heartbeat.wait()
                worker_started.set()
                try:
                    heartbeat_lease(
                        connection,
                        self.lease,
                        ttl=timedelta(minutes=5),
                        clock=controlled_clock,
                    )
                except BaseException as error:
                    outcome.append(error)
                else:
                    outcome.append(None)
            finally:
                connection.close()

        worker = threading.Thread(target=heartbeat_worker)
        worker.start()
        self.assertTrue(worker_ready.wait(timeout=2))

        blocker = open_ledger(self.database_path)
        self.addCleanup(blocker.close)
        blocker.execute("BEGIN IMMEDIATE")
        start_heartbeat.set()
        self.assertTrue(worker_started.wait(timeout=2))
        called_before_unlock = clock_called.wait(timeout=0.3)
        current_time["value"] = T0 + timedelta(minutes=31)
        blocker.rollback()
        worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertFalse(called_before_unlock)
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], StaleFenceError)


class ArtifactLineageRegressionTests(ReviewStoreTestCase):
    def _commit_first_attempt(self) -> tuple[ArtifactRef, str]:
        attempt_id, stage_run_id = self.start_attempt_stage(attempt_time=T0)
        artifact = self.artifact_ref()
        self.store.commit_stage(
            stage_run_id,
            RunState.SUCCEEDED,
            lease=self.lease,
            artifacts=[artifact],
            now=T0 + timedelta(seconds=1),
        )
        self.store.finish_attempt(
            attempt_id,
            RunState.FAILED,
            lease=self.lease,
            exit_code=1,
            now=T0 + timedelta(seconds=2),
        )
        return artifact, stage_run_id

    def test_retry_can_reference_same_immutable_artifact_with_distinct_lineage(
        self,
    ) -> None:
        artifact, first_stage_run_id = self._commit_first_attempt()
        _, second_stage_run_id = self.start_attempt_stage(
            attempt_time=T0 + timedelta(seconds=3)
        )

        self.store.commit_stage(
            second_stage_run_id,
            RunState.SUCCEEDED,
            lease=self.lease,
            artifacts=[artifact],
            now=T0 + timedelta(seconds=4),
        )

        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0],
            1,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT stage_run_id, artifact_id FROM stage_artifacts "
                "ORDER BY stage_run_id"
            ).fetchall(),
            sorted(
                [
                    (first_stage_run_id, artifact.identity.artifact_id),
                    (second_stage_run_id, artifact.identity.artifact_id),
                ]
            ),
        )

    def test_existing_artifact_with_conflicting_content_fails_closed(self) -> None:
        artifact, _ = self._commit_first_attempt()
        self.connection.execute(
            "UPDATE artifacts SET content_sha256 = ? WHERE artifact_id = ?",
            ("c" * 64, artifact.identity.artifact_id),
        )
        _, retry_stage_run_id = self.start_attempt_stage(
            attempt_time=T0 + timedelta(seconds=3)
        )

        with self.assertRaises(LedgerStateError):
            self.store.commit_stage(
                retry_stage_run_id,
                RunState.SUCCEEDED,
                lease=self.lease,
                artifacts=[artifact],
                now=T0 + timedelta(seconds=4),
            )

        self.assertEqual(
            self.connection.execute(
                "SELECT state FROM stage_runs WHERE stage_run_id = ?",
                (retry_stage_run_id,),
            ).fetchone()[0],
            "RUNNING",
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM stage_artifacts").fetchone()[0],
            1,
        )


if __name__ == "__main__":
    unittest.main()
