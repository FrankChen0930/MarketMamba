"""Focused contract tests for the operational shadow ledger."""

from __future__ import annotations

from datetime import date, datetime
import sqlite3
import tempfile
from pathlib import Path
import unittest

from marketmamba.ops.contracts import (
    DEFAULT_POLICY_VERSION,
    AttemptIdentity,
    ArtifactIdentity,
    Authority,
    CheckIdentity,
    LogicalRunIdentity,
    RunMode,
    RunState,
    StageRunIdentity,
)
from marketmamba.ops.schema import (
    _apply_migrations,
    LATEST_SCHEMA_VERSION,
    SchemaVersionError,
    open_ledger,
)


class ContractIdentityTests(unittest.TestCase):
    def test_logical_run_identity_is_canonical_and_never_uses_volatile_data(self) -> None:
        first = LogicalRunIdentity(requested_date=date(2026, 9, 17))
        repeated = LogicalRunIdentity(requested_date=date(2026, 9, 17))
        changed_date = LogicalRunIdentity(requested_date=date(2026, 9, 18))

        self.assertEqual(DEFAULT_POLICY_VERSION, "ops-policy-v1")
        self.assertEqual(first.mode, RunMode.FETCH_ONLY)
        self.assertEqual(first.authority, Authority.LEGACY_V62)
        self.assertEqual(first.logical_run_id, repeated.logical_run_id)
        self.assertNotEqual(first.logical_run_id, changed_date.logical_run_id)
        self.assertEqual(
            first.canonical_key,
            "2026-09-17|FETCH_ONLY|LEGACY_V62|ops-policy-v1",
        )
        self.assertNotIn("T", first.logical_run_id)
        self.assertNotIn("Z", first.logical_run_id)

    def test_identity_dataclasses_validate_their_required_fields(self) -> None:
        logical = LogicalRunIdentity(requested_date=date(2026, 9, 17))

        with self.assertRaises(TypeError):
            LogicalRunIdentity(requested_date=datetime(2026, 9, 17, 0, 0))
        with self.assertRaises(TypeError):
            LogicalRunIdentity(requested_date=date(2026, 9, 17), mode="FETCH_ONLY")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AttemptIdentity(logical_run_id=logical.logical_run_id, attempt_number=0)
        with self.assertRaises(ValueError):
            StageRunIdentity(attempt_id="attempt", stage_name="bad stage", input_identity="input")
        with self.assertRaises(ValueError):
            ArtifactIdentity(
                artifact_type="snapshot",
                content_sha256="not-a-checksum",
                schema_version="v1",
                schema_sha256="a" * 64,
            )
        with self.assertRaises(ValueError):
            CheckIdentity(scope="", check_name="integrity")

    def test_child_identities_are_deterministic_and_check_policy_is_explicit(self) -> None:
        logical = LogicalRunIdentity(requested_date=date(2026, 9, 17))
        attempt = AttemptIdentity(logical_run_id=logical.logical_run_id, attempt_number=1)
        stage = StageRunIdentity(
            attempt_id=attempt.attempt_id,
            stage_name="fetch_prices",
            input_identity="inputs-v1",
        )
        artifact = ArtifactIdentity(
            artifact_type="raw_snapshot",
            content_sha256="b" * 64,
            schema_version="v1",
            schema_sha256="a" * 64,
        )
        check = CheckIdentity(scope=stage.stage_run_id, check_name="row_count")

        self.assertEqual(attempt.attempt_id, AttemptIdentity(logical.logical_run_id, 1).attempt_id)
        self.assertEqual(stage.stage_run_id, StageRunIdentity(attempt.attempt_id, "fetch_prices", "inputs-v1").stage_run_id)
        self.assertTrue(artifact.artifact_id.startswith("artifact_v1_"))
        self.assertEqual(check.policy_version, DEFAULT_POLICY_VERSION)
        self.assertTrue(RunState.SUCCEEDED.is_terminal)
        self.assertTrue(RunState.FAILED.is_terminal)
        self.assertFalse(RunState.RUNNING.is_terminal)


class LedgerSchemaTests(unittest.TestCase):
    def test_open_creates_v1_core_tables_with_wal_foreign_keys_and_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            connection = open_ledger(database_path)
            self.addCleanup(connection.close)

            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = ?",
                    ("table",),
                )
            }
            self.assertEqual(
                table_names,
                {
                    "schema_migrations",
                    "logical_runs",
                    "attempts",
                    "stage_runs",
                    "artifacts",
                    "stage_artifacts",
                    "checks",
                    "leases",
                },
            )
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertGreaterEqual(connection.execute("PRAGMA busy_timeout").fetchone()[0], 5_000)
            self.assertEqual(
                connection.execute("SELECT version FROM schema_migrations").fetchone()[0],
                LATEST_SCHEMA_VERSION,
            )

            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO attempts (attempt_id, logical_run_id, attempt_number, state, started_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("attempt_v1_orphan", "missing-run", 1, "RUNNING", "2026-09-17T00:00:00Z"),
                )

    def test_repeated_open_is_idempotent_and_records_a_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            first = open_ledger(database_path)
            first.close()
            second = open_ledger(database_path)
            self.addCleanup(second.close)

            migrations = second.execute(
                "SELECT version, checksum, applied_at FROM schema_migrations ORDER BY version"
            ).fetchall()
            self.assertEqual(len(migrations), 1)
            self.assertEqual(migrations[0][0], 1)
            self.assertEqual(len(migrations[0][1]), 64)
            self.assertTrue(migrations[0][2].endswith("Z"))

    def test_future_schema_version_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute(
                "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                (LATEST_SCHEMA_VERSION + 1, "f" * 64, "2026-09-17T00:00:00Z"),
            )
            connection.commit()
            connection.close()

            with self.assertRaises(SchemaVersionError):
                open_ledger(database_path)

    def test_mid_migration_ddl_failure_rolls_back_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            connection = sqlite3.connect(database_path)

            def deny_attempt_table(
                action_code: int,
                argument_one: str | None,
                argument_two: str | None,
                database_name: str | None,
                trigger_name: str | None,
            ) -> int:
                del argument_two, database_name, trigger_name
                if action_code == sqlite3.SQLITE_CREATE_TABLE and argument_one == "attempts":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            connection.set_authorizer(deny_attempt_table)
            with self.assertRaises(sqlite3.DatabaseError):
                _apply_migrations(connection)
            connection.set_authorizer(None)

            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = ?",
                    ("table",),
                )
            }
            self.assertEqual(table_names, set())
            connection.close()

            retry = open_ledger(database_path)
            self.addCleanup(retry.close)
            self.assertEqual(
                retry.execute("SELECT version FROM schema_migrations").fetchone()[0],
                LATEST_SCHEMA_VERSION,
            )
    def test_migration_checksum_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            connection = open_ledger(database_path)
            connection.execute(
                "UPDATE schema_migrations SET checksum = ? WHERE version = ?",
                ("f" * 64, LATEST_SCHEMA_VERSION),
            )
            connection.commit()
            connection.close()

            with self.assertRaises(SchemaVersionError):
                open_ledger(database_path)


if __name__ == "__main__":
    unittest.main()
