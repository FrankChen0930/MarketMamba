"""SQLite v1 migration and connection setup for the operational shadow ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3


BUSY_TIMEOUT_MS = 5_000


class SchemaVersionError(RuntimeError):
    """Raised when a ledger schema cannot be read without risking data loss."""


def _utc_iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Migration:
    version: int
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        canonical_sql = "\n".join(statement.strip() for statement in self.statements)
        return hashlib.sha256(canonical_sql.encode("utf-8")).hexdigest()


_SCHEMA_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL CHECK (length(checksum) = 64),
    applied_at TEXT NOT NULL
)
"""


MIGRATIONS = (
    Migration(
        version=1,
        statements=(
            """
            CREATE TABLE logical_runs (
                logical_run_id TEXT PRIMARY KEY,
                requested_date TEXT NOT NULL CHECK (length(requested_date) = 10),
                mode TEXT NOT NULL CHECK (mode = "FETCH_ONLY"),
                authority TEXT NOT NULL CHECK (authority = "LEGACY_V62"),
                policy_version TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED")),
                terminal_reason TEXT,
                incident_ref TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (requested_date, mode, authority, policy_version)
            )
            """,
            """
            CREATE TABLE attempts (
                attempt_id TEXT PRIMARY KEY,
                logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id) ON DELETE RESTRICT,
                attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
                state TEXT NOT NULL CHECK (state IN ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED")),
                host TEXT,
                process_id INTEGER,
                code_identity TEXT,
                dependency_identity TEXT,
                started_at TEXT NOT NULL,
                heartbeat_at TEXT,
                finished_at TEXT,
                exit_code INTEGER,
                incident_ref TEXT,
                UNIQUE (logical_run_id, attempt_number)
            )
            """,
            """
            CREATE TABLE stage_runs (
                stage_run_id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id) ON DELETE RESTRICT,
                stage_name TEXT NOT NULL,
                input_identity TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED")),
                started_at TEXT NOT NULL,
                finished_at TEXT,
                retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
                error_fingerprint TEXT,
                incident_ref TEXT,
                UNIQUE (attempt_id, stage_name, input_identity)
            )
            """,
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id) ON DELETE RESTRICT,
                stage_run_id TEXT REFERENCES stage_runs(stage_run_id) ON DELETE RESTRICT,
                artifact_type TEXT NOT NULL,
                identity_sha256 TEXT NOT NULL CHECK (length(identity_sha256) = 64),
                content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
                schema_version TEXT NOT NULL,
                schema_sha256 TEXT NOT NULL CHECK (length(schema_sha256) = 64),
                uri TEXT NOT NULL,
                byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
                complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
                created_at TEXT NOT NULL,
                UNIQUE (artifact_type, identity_sha256)
            )
            """,
            """
            CREATE TABLE checks (
                check_id TEXT PRIMARY KEY,
                logical_run_id TEXT NOT NULL REFERENCES logical_runs(logical_run_id) ON DELETE RESTRICT,
                stage_run_id TEXT REFERENCES stage_runs(stage_run_id) ON DELETE RESTRICT,
                scope TEXT NOT NULL,
                check_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ("PASS", "WARN", "FAIL", "SKIP")),
                severity TEXT NOT NULL,
                observed_value TEXT,
                threshold_value TEXT,
                policy_version TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                UNIQUE (scope, check_name, policy_version)
            )
            """,
            """
            CREATE TABLE leases (
                lease_name TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                fencing_token INTEGER NOT NULL CHECK (fencing_token > 0)
            )
            """,
        ),
    ),
)

LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version


def _migrations_by_version() -> dict[int, Migration]:
    return {migration.version: migration for migration in MIGRATIONS}


def _apply_migrations(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(_SCHEMA_MIGRATIONS)
        known_migrations = _migrations_by_version()
        applied = dict(connection.execute("SELECT version, checksum FROM schema_migrations"))

        future_versions = [version for version in applied if version > LATEST_SCHEMA_VERSION]
        if future_versions:
            raise SchemaVersionError(
                f"ledger schema version {max(future_versions)} is newer than supported version {LATEST_SCHEMA_VERSION}"
            )

        for version, checksum in applied.items():
            migration = known_migrations.get(version)
            if migration is None:
                raise SchemaVersionError(f"ledger contains unsupported schema version {version}")
            if checksum != migration.checksum:
                raise SchemaVersionError(f"ledger checksum mismatch for schema version {version}")

        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations (version, checksum, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.checksum, _utc_iso_timestamp()),
            )
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def open_ledger(path: str | Path) -> sqlite3.Connection:
    """Open and migrate one file-backed ledger with the required safety pragmas."""

    connection = sqlite3.connect(str(path), isolation_level=None)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    try:
        _apply_migrations(connection)
    except Exception:
        connection.close()
        raise
    return connection
