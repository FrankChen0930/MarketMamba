"""Transactional writes for the operational shadow ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
import sqlite3
from typing import Iterable

from marketmamba.ops.contracts import (
    ArtifactIdentity,
    AttemptIdentity,
    CheckIdentity,
    CheckStatus,
    LogicalRunIdentity,
    RunState,
    StageRunIdentity,
)
from marketmamba.ops.lease import LeaseHandle, immediate_transaction, validate_lease


class LedgerStateError(RuntimeError):
    """Raised when a requested ledger transition is not valid."""


class TerminalStateError(LedgerStateError):
    """Raised when code attempts to rewrite an immutable terminal record."""


class ArtifactConflictError(LedgerStateError):
    """Raised when an artifact identity conflicts with immutable stored data."""


_HEADER_SECRET = re.compile(
    r"(?im)\b(authorization|proxy-authorization|x-api-key|api-key)\s*:\s*[^\r\n]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_TELEGRAM_SECRET = re.compile(r"(?<!\d)\d{6,12}:[A-Za-z0-9_-]{20,}\b")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:access_token|api[_-]?key|apikey|token|password|secret)=)[^&#\s]+"
)
_KEY_VALUE_SECRET = re.compile(
    r"(?i)\b(password|passwd|token|api[_-]?key|secret)\s*[=:]\s*[^\s&,;]+"
)
_QUOTED_KEY_SECRET = re.compile(
    r"""(?i)(["']?(?:authorization|password|passwd|token|api[_-]?key|secret)["']?\s*:\s*)"""
    r"""(?P<quote>["'])(.*?)(?P=quote)"""
)
_URL_USERINFO_SECRET = re.compile(r"(?i)(https?://[^/\s:@]+:)([^@\s/]+)(@)")



def redact_sensitive_text(value: str | None) -> str | None:
    """Redact common credentials before any error or evidence persistence."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("persisted text must be a string or None")
    redacted = _HEADER_SECRET.sub(lambda match: f"{match.group(1)}: [REDACTED]", value)
    redacted = _URL_USERINFO_SECRET.sub(
        lambda match: f"{match.group(1)}[REDACTED]{match.group(3)}",
        redacted,
    )
    redacted = _QUOTED_KEY_SECRET.sub(
        lambda match: (
            f"{match.group(1)}{match.group('quote')}"
            f"[REDACTED]{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _BEARER_SECRET.sub("Bearer [REDACTED]", redacted)
    redacted = _TELEGRAM_SECRET.sub("[REDACTED]", redacted)
    redacted = _QUERY_SECRET.sub(
        lambda match: f"{match.group(1)}[REDACTED]",
        redacted,
    )
    return _KEY_VALUE_SECRET.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        redacted,
    )


def _utc_timestamp(value: datetime | None) -> str:
    timestamp = datetime.now(timezone.utc) if value is None else value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("ledger timestamps must be timezone-aware")
    return timestamp.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _require_terminal(state: RunState) -> None:
    if not isinstance(state, RunState) or not state.is_terminal:
        raise ValueError("terminal state must be SUCCEEDED, FAILED, or CANCELLED")


def _raise_if_terminal(entity: str, state: str) -> None:
    if RunState(state).is_terminal:
        raise TerminalStateError(f"{entity} is already terminal with state {state}")


@dataclass(frozen=True)
class ArtifactRef:
    identity: ArtifactIdentity
    uri: str
    byte_size: int
    complete: bool


@dataclass(frozen=True)
class CheckResult:
    identity: CheckIdentity
    status: CheckStatus
    severity: str
    observed_value: str | None = None
    threshold_value: str | None = None


class LedgerStore:
    """Explicit-transaction state transitions over a Task 1 ledger connection."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def open_logical_run(
        self,
        identity: LogicalRunIdentity,
        *,
        now: datetime | None = None,
    ) -> str:
        if not isinstance(identity, LogicalRunIdentity):
            raise TypeError("identity must be a LogicalRunIdentity")
        timestamp = _utc_timestamp(now)
        with immediate_transaction(self.connection):
            row = self.connection.execute(
                "SELECT requested_date, mode, authority, policy_version "
                "FROM logical_runs WHERE logical_run_id = ?",
                (identity.logical_run_id,),
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO logical_runs "
                    "(logical_run_id, requested_date, mode, authority, policy_version, "
                    "state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identity.logical_run_id,
                        identity.requested_date.isoformat(),
                        identity.mode.value,
                        identity.authority.value,
                        identity.policy_version,
                        RunState.PENDING.value,
                        timestamp,
                        timestamp,
                    ),
                )
            elif row != (
                identity.requested_date.isoformat(),
                identity.mode.value,
                identity.authority.value,
                identity.policy_version,
            ):
                raise LedgerStateError("logical run identity conflicts with stored fields")
        return identity.logical_run_id

    def start_attempt(
        self,
        logical_run_id: str,
        *,
        host: str | None = None,
        process_id: int | None = None,
        code_identity: str | None = None,
        dependency_identity: str | None = None,
        now: datetime | None = None,
    ) -> AttemptIdentity:
        timestamp = _utc_timestamp(now)
        with immediate_transaction(self.connection):
            row = self.connection.execute(
                "SELECT state FROM logical_runs WHERE logical_run_id = ?",
                (logical_run_id,),
            ).fetchone()
            if row is None:
                raise LedgerStateError(f"logical run {logical_run_id!r} does not exist")
            _raise_if_terminal("logical run", row[0])
            attempt_number = self.connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM attempts "
                "WHERE logical_run_id = ?",
                (logical_run_id,),
            ).fetchone()[0]
            identity = AttemptIdentity(logical_run_id, attempt_number)
            self.connection.execute(
                "INSERT INTO attempts "
                "(attempt_id, logical_run_id, attempt_number, state, host, process_id, "
                "code_identity, dependency_identity, started_at, heartbeat_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identity.attempt_id,
                    logical_run_id,
                    attempt_number,
                    RunState.RUNNING.value,
                    redact_sensitive_text(host),
                    process_id,
                    redact_sensitive_text(code_identity),
                    redact_sensitive_text(dependency_identity),
                    timestamp,
                    timestamp,
                ),
            )
            self.connection.execute(
                "UPDATE logical_runs SET state = ?, updated_at = ? "
                "WHERE logical_run_id = ?",
                (RunState.RUNNING.value, timestamp, logical_run_id),
            )
        return identity

    def finish_attempt(
        self,
        attempt_id: str,
        state: RunState,
        *,
        lease: LeaseHandle,
        exit_code: int | None = None,
        incident_ref: str | None = None,
        now: datetime | None = None,
    ) -> None:
        _require_terminal(state)
        timestamp = _utc_timestamp(now)
        with immediate_transaction(self.connection):
            validate_lease(self.connection, lease, now=now)
            row = self.connection.execute(
                "SELECT state FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise LedgerStateError(f"attempt {attempt_id!r} does not exist")
            _raise_if_terminal("attempt", row[0])
            cursor = self.connection.execute(
                "UPDATE attempts SET state = ?, heartbeat_at = ?, finished_at = ?, "
                "exit_code = ?, incident_ref = ? "
                "WHERE attempt_id = ? AND state = ?",
                (
                    state.value,
                    timestamp,
                    timestamp,
                    exit_code,
                    redact_sensitive_text(incident_ref),
                    attempt_id,
                    RunState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise LedgerStateError(f"attempt {attempt_id!r} is not running")

    def finish_logical_run(
        self,
        logical_run_id: str,
        state: RunState,
        *,
        lease: LeaseHandle,
        terminal_reason: str | None = None,
        incident_ref: str | None = None,
        now: datetime | None = None,
    ) -> None:
        _require_terminal(state)
        timestamp = _utc_timestamp(now)
        with immediate_transaction(self.connection):
            validate_lease(self.connection, lease, now=now)
            row = self.connection.execute(
                "SELECT state FROM logical_runs WHERE logical_run_id = ?",
                (logical_run_id,),
            ).fetchone()
            if row is None:
                raise LedgerStateError(f"logical run {logical_run_id!r} does not exist")
            _raise_if_terminal("logical run", row[0])
            self.connection.execute(
                "UPDATE logical_runs SET state = ?, terminal_reason = ?, "
                "incident_ref = ?, updated_at = ? WHERE logical_run_id = ?",
                (
                    state.value,
                    redact_sensitive_text(terminal_reason),
                    redact_sensitive_text(incident_ref),
                    timestamp,
                    logical_run_id,
                ),
            )

    def start_stage(
        self,
        attempt_id: str,
        stage_name: str,
        input_identity: str,
        *,
        now: datetime | None = None,
    ) -> StageRunIdentity:
        identity = StageRunIdentity(attempt_id, stage_name, input_identity)
        timestamp = _utc_timestamp(now)
        with immediate_transaction(self.connection):
            row = self.connection.execute(
                "SELECT state FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None:
                raise LedgerStateError(f"attempt {attempt_id!r} does not exist")
            if row[0] != RunState.RUNNING.value:
                raise LedgerStateError(f"attempt {attempt_id!r} is not running")
            self.connection.execute(
                "INSERT INTO stage_runs "
                "(stage_run_id, attempt_id, stage_name, input_identity, state, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    identity.stage_run_id,
                    attempt_id,
                    stage_name,
                    input_identity,
                    RunState.RUNNING.value,
                    timestamp,
                ),
            )
        return identity

    def commit_stage(
        self,
        stage_run_id: str,
        state: RunState,
        *,
        lease: LeaseHandle,
        artifacts: Iterable[ArtifactRef] = (),
        checks: Iterable[CheckResult] = (),
        retryable: bool = False,
        error_fingerprint: str | None = None,
        incident_ref: str | None = None,
        now: datetime | None = None,
    ) -> None:
        _require_terminal(state)
        timestamp = _utc_timestamp(now)
        artifact_rows = tuple(artifacts)
        check_rows = tuple(checks)
        for result in check_rows:
            if result.identity.scope != stage_run_id:
                raise ValueError("check scope must equal the terminal stage_run_id")
            if not isinstance(result.status, CheckStatus):
                raise TypeError("check status must be a CheckStatus")

        with immediate_transaction(self.connection):
            validate_lease(self.connection, lease, now=now)
            row = self.connection.execute(
                "SELECT stage_runs.state, attempts.logical_run_id "
                "FROM stage_runs JOIN attempts USING (attempt_id) "
                "WHERE stage_run_id = ?",
                (stage_run_id,),
            ).fetchone()
            if row is None:
                raise LedgerStateError(f"stage {stage_run_id!r} does not exist")
            _raise_if_terminal("stage", row[0])
            if row[0] != RunState.RUNNING.value:
                raise LedgerStateError(f"stage {stage_run_id!r} is not running")
            logical_run_id = row[1]

            for artifact in artifact_rows:
                identity_sha256 = hashlib.sha256(
                    artifact.identity.canonical_key.encode("utf-8")
                ).hexdigest()
                immutable_values = (
                    artifact.identity.artifact_type,
                    identity_sha256,
                    artifact.identity.content_sha256,
                    artifact.identity.schema_version,
                    artifact.identity.schema_sha256,
                    redact_sensitive_text(artifact.uri),
                    artifact.byte_size,
                    int(artifact.complete),
                )
                existing = self.connection.execute(
                    "SELECT artifact_type, identity_sha256, content_sha256, "
                    "schema_version, schema_sha256, uri, byte_size, complete "
                    "FROM artifacts WHERE artifact_id = ?",
                    (artifact.identity.artifact_id,),
                ).fetchone()
                if existing is None:
                    self.connection.execute(
                        "INSERT INTO artifacts "
                        "(artifact_id, logical_run_id, stage_run_id, artifact_type, "
                        "identity_sha256, content_sha256, schema_version, schema_sha256, "
                        "uri, byte_size, complete, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            artifact.identity.artifact_id,
                            logical_run_id,
                            stage_run_id,
                            *immutable_values,
                            timestamp,
                        ),
                    )
                elif existing != immutable_values:
                    raise ArtifactConflictError(
                        "artifact identity conflicts with immutable stored data: "
                        f"{artifact.identity.artifact_id}"
                    )

                self.connection.execute(
                    "INSERT INTO stage_artifacts "
                    "(stage_run_id, artifact_id, created_at) VALUES (?, ?, ?)",
                    (
                        stage_run_id,
                        artifact.identity.artifact_id,
                        timestamp,
                    ),
                )

            for result in check_rows:
                self.connection.execute(
                    "INSERT INTO checks "
                    "(check_id, logical_run_id, stage_run_id, scope, check_name, "
                    "status, severity, observed_value, threshold_value, "
                    "policy_version, checked_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.identity.check_id,
                        logical_run_id,
                        stage_run_id,
                        result.identity.scope,
                        result.identity.check_name,
                        result.status.value,
                        result.severity,
                        redact_sensitive_text(result.observed_value),
                        redact_sensitive_text(result.threshold_value),
                        result.identity.policy_version,
                        timestamp,
                    ),
                )

            cursor = self.connection.execute(
                "UPDATE stage_runs SET state = ?, finished_at = ?, retryable = ?, "
                "error_fingerprint = ?, incident_ref = ? "
                "WHERE stage_run_id = ? AND state = ?",
                (
                    state.value,
                    timestamp,
                    int(retryable),
                    redact_sensitive_text(error_fingerprint),
                    redact_sensitive_text(incident_ref),
                    stage_run_id,
                    RunState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise LedgerStateError(f"stage {stage_run_id!r} is not running")
