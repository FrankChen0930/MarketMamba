"""Best-effort observation of the unchanged V6.2 fetch-only operation.

The legacy fetch result or exception is always authoritative.  This module may
diagnose that it could not obtain ledger authority, but never retries, skips, or
changes the result of the supplied fetch operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sqlite3
from typing import Callable, Mapping, MutableMapping, Sequence, TypeVar
import uuid
from zoneinfo import ZoneInfo

from marketmamba.ops.contracts import (
    ArtifactIdentity,
    CheckIdentity,
    CheckStatus,
    LogicalRunIdentity,
    RunState,
)
from marketmamba.ops.lease import (
    LeaseConflictError,
    LeaseHandle,
    acquire_lease,
    release_lease,
)
from marketmamba.ops.schema import open_ledger
from marketmamba.ops.store import ArtifactRef, CheckResult, LedgerStore


_Result = TypeVar("_Result")
_LEASE_NAME = "v62-fetch-only"
_LEASE_TTL = timedelta(hours=8)
_TAIPEI = ZoneInfo("Asia/Taipei")
_STAT_SCHEMA = "relative_path,size_bytes,mtime_ns"
_STAT_SCHEMA_SHA256 = hashlib.sha256(_STAT_SCHEMA.encode("utf-8")).hexdigest()


@dataclass
class _ShadowSession:
    connection: sqlite3.Connection
    store: LedgerStore
    lease: LeaseHandle
    logical_run_id: str
    attempt_id: str
    ingest_stage_id: str


def default_ledger_path(
    v6_dir: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the ignored runtime DB path, with an operations-root override."""

    environment = os.environ if environ is None else environ
    configured = environment.get("MARKETMAMBA_OPS_ROOT")
    ops_root = (
        Path(configured).expanduser()
        if configured
        else Path(v6_dir).resolve().parent / "Data" / "ops"
    )
    return ops_root / "operations.sqlite"


def _default_data_root(
    v6_dir: Path,
    environment: Mapping[str, str],
) -> Path:
    configured = environment.get("MARKETMAMBA_DATA_ROOT")
    return (
        Path(configured).expanduser()
        if configured
        else v6_dir.resolve().parent / "Data"
    )


def artifact_ref_from_stat(
    path: str | Path,
    *,
    data_root: str | Path,
    complete: bool,
) -> ArtifactRef:
    """Describe a parquet with an explicit stat fingerprint, not a content hash."""

    artifact_path = Path(path)
    root = Path(data_root)
    stat = artifact_path.stat()
    try:
        relative = artifact_path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(artifact_path.name)
    relative_text = relative.as_posix()
    metadata = {
        "mtime_ns": stat.st_mtime_ns,
        "relative_path": relative_text,
        "size_bytes": stat.st_size,
    }
    metadata_fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = ArtifactIdentity(
        artifact_type="raw_parquet_metadata_ref",
        content_sha256=None,
        schema_version="stat-fingerprint-v1",
        schema_sha256=_STAT_SCHEMA_SHA256,
        fingerprint_kind="file_stat_v1",
        fingerprint_sha256=metadata_fingerprint,
    )
    return ArtifactRef(
        identity=identity,
        uri=f"file:{relative_text}",
        byte_size=stat.st_size,
        complete=complete,
    )


def _owner_id() -> str:
    host = socket.gethostname().replace(":", "_")[:48] or "unknown-host"
    return f"{host}:{os.getpid()}:{uuid.uuid4().hex[:16]}"


def _check(
    stage_run_id: str,
    name: str,
    status: CheckStatus,
    *,
    observed: str | None,
    threshold: str | None,
    severity: str,
) -> CheckResult:
    return CheckResult(
        identity=CheckIdentity(stage_run_id, name),
        status=status,
        severity=severity,
        observed_value=observed,
        threshold_value=threshold,
    )


def _start_session(
    *,
    ledger_path: Path,
    requested_at: datetime,
) -> _ShadowSession:
    if requested_at.tzinfo is None or requested_at.utcoffset() is None:
        raise ValueError("requested_at must be timezone-aware")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    connection = open_ledger(ledger_path)
    lease = None
    try:
        lease = acquire_lease(
            connection,
            _LEASE_NAME,
            _owner_id(),
            ttl=_LEASE_TTL,
        )
        store = LedgerStore(connection)
        logical = LogicalRunIdentity(requested_at.astimezone(_TAIPEI).date())
        logical_run_id = store.open_logical_run(logical)
        attempt = store.start_attempt(
            logical_run_id,
            lease=lease,
            host=socket.gethostname(),
            process_id=os.getpid(),
            code_identity="v62-fetch-only-shadow-phase-a",
            dependency_identity="stdlib-sqlite",
        )
        calendar = store.start_stage(
            attempt.attempt_id,
            "calendar",
            logical.requested_date.isoformat(),
        )
        store.commit_stage(
            calendar.stage_run_id,
            RunState.SKIPPED,
            lease=lease,
            checks=(
                _check(
                    calendar.stage_run_id,
                    "calendar_observation",
                    CheckStatus.SKIP,
                    observed="unknown:not_exposed_by_legacy_fetcher",
                    threshold=None,
                    severity="INFO",
                ),
            ),
        )
        ingest = store.start_stage(
            attempt.attempt_id,
            "ingest_raw",
            logical.requested_date.isoformat(),
        )
        return _ShadowSession(
            connection=connection,
            store=store,
            lease=lease,
            logical_run_id=logical_run_id,
            attempt_id=attempt.attempt_id,
            ingest_stage_id=ingest.stage_run_id,
        )
    except BaseException:
        if lease is not None:
            try:
                release_lease(connection, lease)
            except BaseException:
                pass
        try:
            connection.close()
        except BaseException:
            pass
        raise


def _artifacts(
    complete: Mapping[str, object],
    *,
    daily_sources: Sequence[str],
    data_root: Path,
) -> tuple[ArtifactRef, ...]:
    processed = data_root / "processed_v6"
    refs = []
    for source in daily_sources:
        path = processed / f"{source}.parquet"
        if not path.is_file():
            continue
        refs.append(
            artifact_ref_from_stat(
                path,
                data_root=data_root,
                complete=bool(complete.get(source, False)),
            )
        )
    return tuple(refs)


def _record_result(
    session: _ShadowSession,
    result: tuple[Mapping[str, object], Sequence[str]],
    *,
    daily_sources: Sequence[str],
    data_root: Path,
) -> None:
    complete, missing = result
    expected = len(daily_sources)
    observed = sum(bool(complete.get(name, False)) for name in daily_sources)
    succeeded = not missing
    source_count_passed = observed == expected
    source_count_status = (
        CheckStatus.PASS if source_count_passed else CheckStatus.FAIL
    )
    outcome_status = CheckStatus.PASS if succeeded else CheckStatus.FAIL
    session.store.commit_stage(
        session.ingest_stage_id,
        RunState.SUCCEEDED if succeeded else RunState.FAILED,
        lease=session.lease,
        artifacts=_artifacts(
            complete,
            daily_sources=daily_sources,
            data_root=data_root,
        ),
        checks=(
            _check(
                session.ingest_stage_id,
                "source_count",
                source_count_status,
                observed=str(observed),
                threshold=str(expected),
                severity="INFO" if source_count_passed else "ERROR",
            ),
            _check(
                session.ingest_stage_id,
                "freshness",
                outcome_status,
                observed=f"missing={len(missing)}",
                threshold="missing=0",
                severity="ERROR" if not succeeded else "INFO",
            ),
            _check(
                session.ingest_stage_id,
                "exit_gate",
                outcome_status,
                observed="1" if missing else "0",
                threshold="0",
                severity="ERROR" if not succeeded else "INFO",
            ),
        ),
    )
    terminal = RunState.SUCCEEDED if succeeded else RunState.FAILED
    session.store.finish_attempt(
        session.attempt_id,
        terminal,
        lease=session.lease,
        exit_code=0 if succeeded else 1,
    )
    session.store.finish_logical_run(
        session.logical_run_id,
        terminal,
        lease=session.lease,
        terminal_reason=None if succeeded else "legacy_fetch_missing_sources",
    )


def _record_exception(session: _ShadowSession, error: BaseException) -> None:
    error_type = f"{type(error).__module__}.{type(error).__qualname__}"
    fingerprint = hashlib.sha256(
        f"{error_type}:{error}".encode("utf-8", errors="replace")
    ).hexdigest()
    session.store.commit_stage(
        session.ingest_stage_id,
        RunState.FAILED,
        lease=session.lease,
        checks=(
            _check(
                session.ingest_stage_id,
                "source_count",
                CheckStatus.SKIP,
                observed=None,
                threshold=None,
                severity="WARN",
            ),
            _check(
                session.ingest_stage_id,
                "freshness",
                CheckStatus.FAIL,
                observed=f"fetch_exception:{error_type}",
                threshold="fetch_completed",
                severity="ERROR",
            ),
            _check(
                session.ingest_stage_id,
                "exit_gate",
                CheckStatus.FAIL,
                observed="exception",
                threshold="0",
                severity="ERROR",
            ),
        ),
        error_fingerprint=fingerprint,
        incident_ref=f"{error_type}: {error}",
    )
    session.store.finish_attempt(
        session.attempt_id,
        RunState.FAILED,
        lease=session.lease,
        exit_code=1,
        incident_ref=f"{error_type}: {error}",
    )
    session.store.finish_logical_run(
        session.logical_run_id,
        RunState.FAILED,
        lease=session.lease,
        terminal_reason="legacy_fetch_exception",
        incident_ref=f"{error_type}: {error}",
    )


def _warn(logger: logging.Logger | None, phase: str, error: BaseException) -> None:
    """Attempt diagnostics without becoming an operational dependency."""

    try:
        target = logger if logger is not None else logging.getLogger(__name__)
        target.warning(
            "[shadow-ledger] %s failed (%s); legacy fetch outcome is unchanged",
            phase,
            type(error).__name__,
        )
    except BaseException:
        pass


def _finish_session(
    session: _ShadowSession,
    logger: logging.Logger | None,
) -> None:
    try:
        release_lease(session.connection, session.lease)
    except BaseException as error:
        _warn(logger, "lease release", error)
    try:
        session.connection.close()
    except BaseException as error:
        _warn(logger, "connection close", error)


def observe_fetch_only(
    fetch_operation: Callable[[], _Result],
    *,
    daily_sources: Sequence[str],
    v6_dir: str | Path,
    ledger_path: str | Path | None = None,
    data_root: str | Path | None = None,
    requested_at: datetime | None = None,
    logger: logging.Logger | None = None,
    projection_callback: Callable[[Path], object] | None = None,
    environ: MutableMapping[str, str] | Mapping[str, str] | None = None,
) -> _Result:
    """Run legacy fetch exactly once while recording best-effort shadow evidence."""

    log = logger
    session = None
    db_path = None
    raw_root = None
    try:
        environment = os.environ if environ is None else environ
        v6_path = Path(v6_dir)
        db_path = (
            Path(ledger_path)
            if ledger_path is not None
            else default_ledger_path(v6_path, environ=environment)
        )
        raw_root = (
            Path(data_root)
            if data_root is not None
            else _default_data_root(v6_path, environment)
        )
        when = requested_at or datetime.now(timezone.utc)
        session = _start_session(ledger_path=db_path, requested_at=when)
    except BaseException as error:
        phase = "lease conflict" if isinstance(error, LeaseConflictError) else "initialization"
        _warn(log, phase, error)

    try:
        result = fetch_operation()
    except BaseException as fetch_error:
        if session is not None:
            try:
                _record_exception(session, fetch_error)
            except BaseException as observer_error:
                _warn(log, "exception recording", observer_error)
            finally:
                _finish_session(session, log)
        raise

    if session is not None:
        assert raw_root is not None
        assert db_path is not None
        ledger_committed = False
        try:
            _record_result(
                session,
                result,
                daily_sources=daily_sources,
                data_root=raw_root,
            )
            ledger_committed = True
        except BaseException as observer_error:
            _warn(log, "result recording", observer_error)
        finally:
            _finish_session(session, log)

        if ledger_committed and projection_callback is not None:
            try:
                projection_callback(db_path)
            except BaseException as projection_error:
                _warn(log, "projection callback", projection_error)
    return result
