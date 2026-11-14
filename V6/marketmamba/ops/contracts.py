"""Validated, deterministic identities for the operational shadow ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import re


DEFAULT_POLICY_VERSION = "ops-policy-v1"
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_POLICY_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_STAGE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RunMode(str, Enum):
    """The only Task 1 operational mode."""

    FETCH_ONLY = "FETCH_ONLY"


class Authority(str, Enum):
    """The authority identifying the unchanged legacy V6.2 behavior."""

    LEGACY_V62 = "LEGACY_V62"


class RunState(str, Enum):
    """States shared by logical runs, attempts, and stage runs."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunState.SUCCEEDED,
            RunState.FAILED,
            RunState.CANCELLED,
            RunState.SKIPPED,
        }


class CheckStatus(str, Enum):
    """A compact result vocabulary for recorded operational checks."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


def _stable_id(prefix: str, canonical_key: str) -> str:
    digest = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()
    return f"{prefix}_v1_{digest}"


def _require_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty stable token")


def _require_checksum(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hexadecimal digest")


@dataclass(frozen=True)
class LogicalRunIdentity:
    """A logical run keyed only by the requested Asia Taipei calendar date and policy."""

    requested_date: date
    mode: RunMode = RunMode.FETCH_ONLY
    authority: Authority = Authority.LEGACY_V62
    policy_version: str = DEFAULT_POLICY_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.requested_date, datetime) or not isinstance(self.requested_date, date):
            raise TypeError("requested_date must be an Asia Taipei calendar date")
        if not isinstance(self.mode, RunMode):
            raise TypeError("mode must be a RunMode")
        if not isinstance(self.authority, Authority):
            raise TypeError("authority must be an Authority")
        if not isinstance(self.policy_version, str) or not _POLICY_VERSION.fullmatch(self.policy_version):
            raise ValueError("policy_version must be a stable policy token")

    @property
    def canonical_key(self) -> str:
        return "|".join(
            (
                self.requested_date.isoformat(),
                self.mode.value,
                self.authority.value,
                self.policy_version,
            )
        )

    @property
    def logical_run_id(self) -> str:
        return _stable_id("logical_run", self.canonical_key)


@dataclass(frozen=True)
class AttemptIdentity:
    """A deterministic ordinal attempt belonging to one logical run."""

    logical_run_id: str
    attempt_number: int

    def __post_init__(self) -> None:
        _require_token(self.logical_run_id, "logical_run_id")
        if not isinstance(self.attempt_number, int) or isinstance(self.attempt_number, bool) or self.attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")

    @property
    def canonical_key(self) -> str:
        return f"{self.logical_run_id}|{self.attempt_number}"

    @property
    def attempt_id(self) -> str:
        return _stable_id("attempt", self.canonical_key)


@dataclass(frozen=True)
class StageRunIdentity:
    """A deterministic stage identity that includes its declared input identity."""

    attempt_id: str
    stage_name: str
    input_identity: str

    def __post_init__(self) -> None:
        _require_token(self.attempt_id, "attempt_id")
        if not isinstance(self.stage_name, str) or not _STAGE_NAME.fullmatch(self.stage_name):
            raise ValueError("stage_name must be lowercase snake_case")
        _require_token(self.input_identity, "input_identity")

    @property
    def canonical_key(self) -> str:
        return "|".join((self.attempt_id, self.stage_name, self.input_identity))

    @property
    def stage_run_id(self) -> str:
        return _stable_id("stage_run", self.canonical_key)


@dataclass(frozen=True)
class ArtifactIdentity:
    """An immutable artifact identity with explicit fingerprint semantics."""

    artifact_type: str
    content_sha256: str | None
    schema_version: str
    schema_sha256: str
    fingerprint_kind: str = "content_sha256"
    fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_token(self.artifact_type, "artifact_type")
        if self.content_sha256 is not None:
            _require_checksum(self.content_sha256, "content_sha256")
        _require_token(self.schema_version, "schema_version")
        _require_checksum(self.schema_sha256, "schema_sha256")
        _require_token(self.fingerprint_kind, "fingerprint_kind")
        fingerprint = self.fingerprint_sha256
        if fingerprint is None:
            if self.content_sha256 is None:
                raise ValueError(
                    "fingerprint_sha256 is required when content_sha256 is absent"
                )
            fingerprint = self.content_sha256
            object.__setattr__(self, "fingerprint_sha256", fingerprint)
        _require_checksum(fingerprint, "fingerprint_sha256")
        if self.fingerprint_kind == "content_sha256":
            if self.content_sha256 is None or fingerprint != self.content_sha256:
                raise ValueError(
                    "content_sha256 fingerprint must equal the true content hash"
                )
        elif self.content_sha256 is not None:
            raise ValueError(
                "non-content fingerprints must not populate content_sha256"
            )

    @property
    def canonical_key(self) -> str:
        if self.fingerprint_kind == "content_sha256":
            # Preserve v1/v2 artifact IDs so upgraded content artifacts remain
            # reusable instead of being duplicated under a new identity.
            return "|".join(
                (
                    self.artifact_type,
                    self.content_sha256 or "",
                    self.schema_version,
                    self.schema_sha256,
                )
            )
        return "|".join(
            (
                self.artifact_type,
                self.fingerprint_kind,
                self.fingerprint_sha256 or "",
                self.content_sha256 or "",
                self.schema_version,
                self.schema_sha256,
            )
        )

    @property
    def artifact_id(self) -> str:
        return _stable_id("artifact", self.canonical_key)

@dataclass(frozen=True)
class CheckIdentity:
    """A check identity bound to its scope, name, and explicit policy version."""

    scope: str
    check_name: str
    policy_version: str = DEFAULT_POLICY_VERSION

    def __post_init__(self) -> None:
        _require_token(self.scope, "scope")
        _require_token(self.check_name, "check_name")
        if not isinstance(self.policy_version, str) or not _POLICY_VERSION.fullmatch(self.policy_version):
            raise ValueError("policy_version must be a stable policy token")

    @property
    def canonical_key(self) -> str:
        return "|".join((self.scope, self.check_name, self.policy_version))

    @property
    def check_id(self) -> str:
        return _stable_id("check", self.canonical_key)
