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
    MIGRATIONS,
    SchemaVersionError,
    open_ledger,
)

BASE_V1_CHECKSUM = "8214cd665cfb21be8d809daa5348d2e930c5ce3071e2b53d66d0eaf0f9fd4aee"


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
        self.assertTrue(RunState.SKIPPED.is_terminal)
        self.assertFalse(RunState.RUNNING.is_terminal)

    def test_metadata_fingerprint_never_claims_a_content_hash(self) -> None:
        artifact = ArtifactIdentity(
            artifact_type="raw_parquet_metadata_ref",
            content_sha256=None,
            schema_version="stat-fingerprint-v1",
            schema_sha256="a" * 64,
            fingerprint_kind="file_stat_v1",
            fingerprint_sha256="b" * 64,
        )

        self.assertIsNone(artifact.content_sha256)
        self.assertEqual(artifact.fingerprint_kind, "file_stat_v1")
        with self.assertRaises(ValueError):
            ArtifactIdentity(
                "bad", "b" * 64, "v1", "a" * 64,
                fingerprint_kind="file_stat_v1", fingerprint_sha256="b" * 64,
            )


class LedgerSchemaTests(unittest.TestCase):
    def test_open_creates_latest_tables_with_wal_foreign_keys_and_busy_timeout(self) -> None:
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
            self.assertEqual(LATEST_SCHEMA_VERSION, 3)
            self.assertEqual(
                [
                    row[0]
                    for row in connection.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                ],
                [1, 2, 3],
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
            self.assertEqual([migration[0] for migration in migrations], [1, 2, 3])
            self.assertEqual(migrations[0][1], BASE_V1_CHECKSUM)
            for _, checksum, applied_at in migrations:
                self.assertEqual(len(checksum), 64)
                self.assertTrue(applied_at.endswith("Z"))

    def test_genuine_base_v1_database_upgrades_to_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            base_v1 = MIGRATIONS[0]
            self.assertEqual(base_v1.version, 1)
            self.assertEqual(base_v1.checksum, BASE_V1_CHECKSUM)

            connection = sqlite3.connect(database_path)
            connection.execute(
                "CREATE TABLE schema_migrations ("
                "version INTEGER PRIMARY KEY, "
                "checksum TEXT NOT NULL CHECK (length(checksum) = 64), "
                "applied_at TEXT NOT NULL)"
            )
            for statement in base_v1.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                (1, BASE_V1_CHECKSUM, "2026-09-17T00:00:00Z"),
            )
            connection.commit()
            pre_upgrade_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = ?",
                    ("table",),
                )
            }
            self.assertNotIn("stage_artifacts", pre_upgrade_tables)
            connection.close()

            upgraded = open_ledger(database_path)
            self.addCleanup(upgraded.close)
            post_upgrade_tables = {
                row[0]
                for row in upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type = ?",
                    ("table",),
                )
            }
            self.assertIn("stage_artifacts", post_upgrade_tables)
            migrations = upgraded.execute(
                "SELECT version, checksum FROM schema_migrations ORDER BY version"
            ).fetchall()
            self.assertEqual([migration[0] for migration in migrations], [1, 2, 3])
            self.assertEqual(migrations[0][1], BASE_V1_CHECKSUM)
            self.assertEqual(migrations[1][1], MIGRATIONS[1].checksum)

    def test_populated_v2_database_upgrades_to_explicit_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute(
                "CREATE TABLE schema_migrations ("
                "version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, "
                "applied_at TEXT NOT NULL)"
            )
            for migration in MIGRATIONS[:2]:
                for statement in migration.statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                    (
                        migration.version,
                        migration.checksum,
                        "2026-09-17T00:00:00Z",
                    ),
                )
            connection.execute(
                "INSERT INTO logical_runs VALUES "
                "(?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)",
                (
                    "logical",
                    "2026-09-17",
                    "FETCH_ONLY",
                    "LEGACY_V62",
                    "ops-policy-v1",
                    "RUNNING",
                    "2026-09-17T00:00:00Z",
                    "2026-09-17T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO attempts "
                "(attempt_id, logical_run_id, attempt_number, state, started_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("attempt", "logical", 1, "RUNNING", "2026-09-17T00:00:00Z"),
            )
            connection.execute(
                "INSERT INTO stage_runs "
                "(stage_run_id, attempt_id, stage_name, input_identity, state, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "stage",
                    "attempt",
                    "ingest_raw",
                    "input",
                    "SUCCEEDED",
                    "2026-09-17T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "artifact",
                    "logical",
                    "stage",
                    "raw_snapshot",
                    "a" * 64,
                    "b" * 64,
                    "v1",
                    "c" * 64,
                    "file:raw.parquet",
                    42,
                    1,
                    "2026-09-17T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO stage_artifacts VALUES (?, ?, ?)",
                ("stage", "artifact", "2026-09-17T00:00:00Z"),
            )
            connection.execute(
                "INSERT INTO checks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "check", "logical", "stage", "stage", "freshness",
                    "PASS", "INFO", "0", "0", "ops-policy-v1",
                    "2026-09-17T00:00:00Z",
                ),
            )
            connection.commit()
            connection.close()

            upgraded = open_ledger(database_path)
            self.addCleanup(upgraded.close)
            self.assertEqual(
                upgraded.execute(
                    "SELECT content_sha256, fingerprint_kind, fingerprint_sha256 "
                    "FROM artifacts WHERE artifact_id = 'artifact'"
                ).fetchone(),
                ("b" * 64, "content_sha256", "b" * 64),
            )
            self.assertEqual(
                upgraded.execute(
                    "SELECT stage_run_id, artifact_id FROM stage_artifacts"
                ).fetchall(),
                [("stage", "artifact")],
            )
            self.assertEqual(
                upgraded.execute(
                    "SELECT check_id, logical_run_id, stage_run_id, status FROM checks"
                ).fetchall(),
                [("check", "logical", "stage", "PASS")],
            )
            self.assertEqual(
                upgraded.execute(
                    "SELECT stage_run_id, attempt_id, state FROM stage_runs "
                    "WHERE stage_run_id = 'stage'"
                ).fetchone(),
                ("stage", "attempt", "SUCCEEDED"),
            )
            upgraded.execute(
                "INSERT INTO stage_runs "
                "(stage_run_id, attempt_id, stage_name, input_identity, state, started_at) "
                "VALUES ('calendar', 'attempt', 'calendar', 'input', 'SKIPPED', "
                "'2026-09-17T00:00:00Z')"
            )
            self.assertEqual(upgraded.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_v3_failure_rolls_back_rebuild_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "ledger.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute(
                "CREATE TABLE schema_migrations ("
                "version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, "
                "applied_at TEXT NOT NULL)"
            )
            for migration in MIGRATIONS[:2]:
                for statement in migration.statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                    (
                        migration.version,
                        migration.checksum,
                        "2026-09-17T00:00:00Z",
                    ),
                )
            connection.commit()

            def deny_v3_rename(
                action_code: int,
                argument_one: str | None,
                argument_two: str | None,
                database_name: str | None,
                trigger_name: str | None,
            ) -> int:
                del argument_one, argument_two, database_name, trigger_name
                if action_code == sqlite3.SQLITE_ALTER_TABLE:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            connection.set_authorizer(deny_v3_rename)
            with self.assertRaises(sqlite3.DatabaseError):
                _apply_migrations(connection)
            connection.set_authorizer(None)

            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertNotIn("stage_runs_v3", tables)
            self.assertEqual(
                [row[0] for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )],
                [1, 2],
            )
            self.assertNotIn(
                "fingerprint_kind",
                [row[1] for row in connection.execute("PRAGMA table_info(artifacts)")],
            )

            _apply_migrations(connection)
            self.assertEqual(
                [row[0] for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )],
                [1, 2, 3],
            )
            connection.close()

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
                [
                    row[0]
                    for row in retry.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                ],
                [1, 2, 3],
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
