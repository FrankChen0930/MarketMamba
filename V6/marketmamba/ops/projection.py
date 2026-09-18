"""Read-only current-state projection derived from committed ledger rows."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Iterator, Mapping

from marketmamba.ops.diagnostics import sanitize_output_text


PROJECTION_SCHEMA_VERSION = "current-state-v1"
INCIDENT_STATUS = "NOT_IMPLEMENTED_PHASE_C"
DATE_SEMANTICS = "REQUESTED_ASIA_TAIPEI_DATE_UNVERIFIED_CALENDAR"
DEFAULT_LOCK_TIMEOUT_SECONDS = 0.1
DEFAULT_LOCK_POLL_SECONDS = 0.01


class ProjectionLockTimeout(TimeoutError):
    """Raised when a projection writer cannot acquire authority promptly."""




def _open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro",
        uri=True,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    return max(0.0, round((finished - started).total_seconds(), 6))


def _safe(value: str | None) -> str | None:
    return sanitize_output_text(value)


def _source_watermark(connection: sqlite3.Connection) -> dict[str, object]:
    timestamp = connection.execute(
        """
        SELECT MAX(value) FROM (
            SELECT applied_at AS value FROM schema_migrations
            UNION ALL SELECT created_at FROM logical_runs
            UNION ALL SELECT updated_at FROM logical_runs
            UNION ALL SELECT started_at FROM attempts
            UNION ALL SELECT heartbeat_at FROM attempts
            UNION ALL SELECT finished_at FROM attempts
            UNION ALL SELECT started_at FROM stage_runs
            UNION ALL SELECT finished_at FROM stage_runs
            UNION ALL SELECT created_at FROM artifacts
            UNION ALL SELECT checked_at FROM checks
            UNION ALL SELECT created_at FROM stage_artifacts
        )
        """
    ).fetchone()[0]
    row_counts = {}
    for table in (
        "logical_runs",
        "attempts",
        "stage_runs",
        "artifacts",
        "checks",
        "stage_artifacts",
    ):
        row_counts[table] = connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
    return {"committed_at": timestamp, "row_counts": row_counts}


def _empty_artifact_summary() -> dict[str, object]:
    return {
        "count": 0,
        "complete_count": 0,
        "total_bytes": 0,
        "fingerprint_kinds": [],
        "by_type": [],
    }


def _artifact_summary(
    connection: sqlite3.Connection,
    attempt_id: str | None,
) -> dict[str, object]:
    if attempt_id is None:
        return _empty_artifact_summary()
    rows = connection.execute(
        """
        SELECT a.artifact_type, a.fingerprint_kind,
               COUNT(DISTINCT a.artifact_id) AS artifact_count,
               SUM(a.complete) AS complete_count,
               SUM(a.byte_size) AS total_bytes
        FROM stage_artifacts AS sa
        JOIN stage_runs AS sr ON sr.stage_run_id = sa.stage_run_id
        JOIN artifacts AS a ON a.artifact_id = sa.artifact_id
        WHERE sr.attempt_id = ?
        GROUP BY a.artifact_type, a.fingerprint_kind
        ORDER BY a.artifact_type, a.fingerprint_kind
        """,
        (attempt_id,),
    ).fetchall()
    by_type = [
        {
            "artifact_type": row["artifact_type"],
            "fingerprint_kind": row["fingerprint_kind"],
            "count": row["artifact_count"],
            "complete_count": row["complete_count"] or 0,
            "total_bytes": row["total_bytes"] or 0,
        }
        for row in rows
    ]
    return {
        "count": sum(item["count"] for item in by_type),
        "complete_count": sum(item["complete_count"] for item in by_type),
        "total_bytes": sum(item["total_bytes"] for item in by_type),
        "fingerprint_kinds": sorted(
            {item["fingerprint_kind"] for item in by_type}
        ),
        "by_type": by_type,
    }


def _check_summary(
    connection: sqlite3.Connection,
    attempt_id: str | None,
    name: str,
) -> dict[str, object]:
    if attempt_id is None:
        return {
            "status": "UNKNOWN",
            "observed_value": None,
            "threshold_value": None,
            "checked_at": None,
        }
    row = connection.execute(
        """
        SELECT c.status, c.observed_value, c.threshold_value, c.checked_at
        FROM checks AS c
        JOIN stage_runs AS sr ON sr.stage_run_id = c.stage_run_id
        WHERE sr.attempt_id = ? AND c.check_name = ?
        ORDER BY c.checked_at DESC, c.check_id DESC
        LIMIT 1
        """,
        (attempt_id, name),
    ).fetchone()
    if row is None:
        return {
            "status": "UNKNOWN",
            "observed_value": None,
            "threshold_value": None,
            "checked_at": None,
        }
    return {
        "status": row["status"],
        "observed_value": _safe(row["observed_value"]),
        "threshold_value": _safe(row["threshold_value"]),
        "checked_at": row["checked_at"],
    }


def _build_snapshot(connection: sqlite3.Connection) -> dict[str, object]:
    watermark = _source_watermark(connection)
    latest_run = connection.execute(
        """
        SELECT logical_run_id, requested_date, mode, state, updated_at
        FROM logical_runs
        WHERE mode = 'FETCH_ONLY'
        ORDER BY requested_date DESC, updated_at DESC, logical_run_id DESC
        LIMIT 1
        """
    ).fetchone()
    logical_run_id = latest_run["logical_run_id"] if latest_run else None
    latest_attempt = None
    attempt_count = 0
    if logical_run_id is not None:
        latest_attempt = connection.execute(
            """
            SELECT attempt_id, attempt_number, state, started_at, finished_at,
                   exit_code
            FROM attempts
            WHERE logical_run_id = ?
            ORDER BY attempt_number DESC
            LIMIT 1
            """,
            (logical_run_id,),
        ).fetchone()
        attempt_count = connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE logical_run_id = ?",
            (logical_run_id,),
        ).fetchone()[0]
    attempt_id = latest_attempt["attempt_id"] if latest_attempt else None
    stage_rows = []
    if attempt_id is not None:
        stage_rows = connection.execute(
            """
            SELECT stage_name, state, started_at, finished_at, retryable
            FROM stage_runs
            WHERE attempt_id = ?
            ORDER BY started_at, stage_name
            """,
            (attempt_id,),
        ).fetchall()
    stages = [
        {
            "stage": row["stage_name"].upper(),
            "status": row["state"],
            "duration_seconds": _duration_seconds(
                row["started_at"], row["finished_at"]
            ),
            "retryable": bool(row["retryable"]),
        }
        for row in stage_rows
    ]
    active = next(
        (item["stage"] for item in stages if item["status"] == "RUNNING"),
        None,
    )
    last_success = connection.execute(
        """
        SELECT lr.requested_date
        FROM logical_runs AS lr
        WHERE lr.mode = 'FETCH_ONLY' AND lr.state = 'SUCCEEDED'
          AND EXISTS (
              SELECT 1 FROM attempts AS a
              JOIN stage_runs AS sr ON sr.attempt_id = a.attempt_id
              WHERE a.logical_run_id = lr.logical_run_id
                AND sr.stage_name = 'ingest_raw'
                AND sr.state = 'SUCCEEDED'
          )
        ORDER BY lr.requested_date DESC, lr.updated_at DESC
        LIMIT 1
        """
    ).fetchone()
    warning_count = 0
    if attempt_id is not None:
        warning_count = connection.execute(
            """
            SELECT COUNT(*) FROM checks AS c
            JOIN stage_runs AS sr ON sr.stage_run_id = c.stage_run_id
            WHERE sr.attempt_id = ?
              AND (c.status = 'WARN' OR UPPER(c.severity) = 'WARN')
            """,
            (attempt_id,),
        ).fetchone()[0]
    calendar = _check_summary(connection, attempt_id, "calendar_observation")
    calendar["status"] = next(
        (
            item["status"]
            for item in stages
            if item["stage"] == "CALENDAR"
        ),
        "UNKNOWN",
    )
    calendar["knowledge"] = "UNKNOWN_NOT_EXPOSED_BY_LEGACY"
    state: dict[str, object] = {
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "as_of": watermark["committed_at"],
        "source_watermark": watermark,
        "mode": "FETCH_ONLY",
        "scope": {
            "included": ["CALENDAR_OBSERVATION", "INGEST_RAW"],
            "excluded": ["INFERENCE", "PORTFOLIO", "PUBLICATION"],
        },
        "date_semantics": DATE_SEMANTICS,
        "latest_requested_date": (
            latest_run["requested_date"] if latest_run else None
        ),
        "latest_trading_date": (
            latest_run["requested_date"] if latest_run else None
        ),
        "latest_logical_run_id": logical_run_id,
        "attempt_count": attempt_count,
        "latest_attempt": (
            latest_attempt["attempt_number"] if latest_attempt else None
        ),
        "latest_attempt_status": (
            latest_attempt["state"] if latest_attempt else None
        ),
        "latest_exit_code": (
            latest_attempt["exit_code"] if latest_attempt else None
        ),
        "run_status": latest_run["state"] if latest_run else "NO_RUNS",
        "active_stage": active,
        "duration_seconds": (
            _duration_seconds(
                latest_attempt["started_at"], latest_attempt["finished_at"]
            )
            if latest_attempt
            else None
        ),
        "stage_statuses": stages,
        "calendar": calendar,
        "last_successful_fetch_date": last_success[0] if last_success else None,
        "freshness": _check_summary(connection, attempt_id, "freshness"),
        "artifact_summary": _artifact_summary(connection, attempt_id),
        "warning_count": warning_count,
        "incident_status": INCIDENT_STATUS,
        "open_incident_count": None,
    }
    generation_payload = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    state["source_generation"] = hashlib.sha256(generation_payload).hexdigest()
    return state


def build_current_state(database_path: str | Path) -> dict[str, object]:
    """Build a deterministic projection from one committed read snapshot."""

    connection = _open_readonly(Path(database_path))
    try:
        connection.execute("BEGIN")
        try:
            return _build_snapshot(connection)
        finally:
            connection.rollback()
    finally:
        connection.close()


@contextmanager
def _projection_lock(
    target: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> Iterator[None]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
    ):
        raise ValueError("lock timeout must be a finite non-negative number")
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0
    ):
        raise ValueError("lock poll interval must be a finite positive number")
    lock_path = target.with_name(f".{target.name}.lock")
    with lock_path.open("a+b") as handle:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProjectionLockTimeout(
                        "projection writer lock deadline exceeded"
                    ) from error
                time.sleep(min(poll_seconds, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(target: Path, state: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                state,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_current_state(
    database_path: str | Path,
    json_path: str | Path | None = None,
    *,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    lock_poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
) -> Path:
    """Publish current state atomically; concurrent writers read in lock order."""

    database = Path(database_path)
    target = (
        Path(json_path)
        if json_path is not None
        else database.parent / "current-state.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with _projection_lock(
        target,
        timeout_seconds=lock_timeout_seconds,
        poll_seconds=lock_poll_seconds,
    ):
        # Snapshot after taking the writer lock: a delayed older callback must
        # publish first; the following writer then re-reads committed state.
        state = build_current_state(database)
        _atomic_write_json(target, state)
    return target
