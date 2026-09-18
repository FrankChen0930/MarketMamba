"""Single-host SQLite lease management with monotonically increasing fencing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import sqlite3
from typing import Iterator


class LeaseConflictError(RuntimeError):
    """Raised when another owner still holds an active lease."""


class StaleFenceError(RuntimeError):
    """Raised when a lease handle is expired, released, or superseded."""


@dataclass(frozen=True)
class LeaseHandle:
    lease_name: str
    owner_id: str
    fencing_token: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc(value: datetime | None) -> datetime:
    timestamp = _utc_now() if value is None else value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("lease timestamps must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def _timestamp(value: datetime | None) -> str:
    return _require_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_ttl(ttl: timedelta) -> None:
    if not isinstance(ttl, timedelta) or ttl.total_seconds() <= 0:
        raise ValueError("ttl must be a positive timedelta")


def _validate_name(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{field_name} must be a non-empty string up to 128 characters")


@contextmanager
def immediate_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run one explicit write transaction and always restore autocommit state."""

    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def acquire_lease(
    connection: sqlite3.Connection,
    lease_name: str,
    owner_id: str,
    *,
    now: datetime | None = None,
    ttl: timedelta,
) -> LeaseHandle:
    """Acquire a new or expired lease, incrementing its persistent fence."""

    _validate_name(lease_name, "lease_name")
    _validate_name(owner_id, "owner_id")
    _validate_ttl(ttl)
    current = _require_utc(now)
    heartbeat_at = _timestamp(current)
    expires_at = _timestamp(current + ttl)

    with immediate_transaction(connection):
        row = connection.execute(
            "SELECT owner_id, expires_at, fencing_token FROM leases WHERE lease_name = ?",
            (lease_name,),
        ).fetchone()
        if row is None:
            fencing_token = 1
            connection.execute(
                "INSERT INTO leases "
                "(lease_name, owner_id, heartbeat_at, expires_at, fencing_token) "
                "VALUES (?, ?, ?, ?, ?)",
                (lease_name, owner_id, heartbeat_at, expires_at, fencing_token),
            )
        else:
            active_owner, active_expiry, previous_token = row
            if active_expiry > heartbeat_at:
                raise LeaseConflictError(
                    f"lease {lease_name!r} is active for owner {active_owner!r}"
                )
            fencing_token = previous_token + 1
            connection.execute(
                "UPDATE leases SET owner_id = ?, heartbeat_at = ?, expires_at = ?, "
                "fencing_token = ? WHERE lease_name = ? AND fencing_token = ?",
                (
                    owner_id,
                    heartbeat_at,
                    expires_at,
                    fencing_token,
                    lease_name,
                    previous_token,
                ),
            )

    return LeaseHandle(lease_name, owner_id, fencing_token)


def validate_lease(
    connection: sqlite3.Connection,
    lease: LeaseHandle,
    *,
    now: datetime | None = None,
) -> None:
    """Validate a fence inside the caller's current transaction."""

    current = _timestamp(now)
    row = connection.execute(
        "SELECT owner_id, expires_at, fencing_token FROM leases WHERE lease_name = ?",
        (lease.lease_name,),
    ).fetchone()
    if (
        row is None
        or row[0] != lease.owner_id
        or row[2] != lease.fencing_token
        or row[1] <= current
    ):
        raise StaleFenceError(
            f"lease {lease.lease_name!r} fence {lease.fencing_token} is not current"
        )


def heartbeat_lease(
    connection: sqlite3.Connection,
    lease: LeaseHandle,
    *,
    now: datetime | None = None,
    ttl: timedelta,
) -> None:
    """Extend only the exact active holder represented by the lease handle."""

    _validate_ttl(ttl)
    current = _require_utc(now)
    heartbeat_at = _timestamp(current)
    expires_at = _timestamp(current + ttl)
    with immediate_transaction(connection):
        validate_lease(connection, lease, now=current)
        cursor = connection.execute(
            "UPDATE leases SET heartbeat_at = ?, expires_at = ? "
            "WHERE lease_name = ? AND owner_id = ? AND fencing_token = ?",
            (
                heartbeat_at,
                expires_at,
                lease.lease_name,
                lease.owner_id,
                lease.fencing_token,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleFenceError(
                f"lease {lease.lease_name!r} fence {lease.fencing_token} is not current"
            )


def release_lease(
    connection: sqlite3.Connection,
    lease: LeaseHandle,
    *,
    now: datetime | None = None,
) -> None:
    """Expire an active lease without deleting or resetting its fence."""

    current = _require_utc(now)
    released_at = _timestamp(current)
    with immediate_transaction(connection):
        validate_lease(connection, lease, now=current)
        cursor = connection.execute(
            "UPDATE leases SET heartbeat_at = ?, expires_at = ? "
            "WHERE lease_name = ? AND owner_id = ? AND fencing_token = ?",
            (
                released_at,
                released_at,
                lease.lease_name,
                lease.owner_id,
                lease.fencing_token,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleFenceError(
                f"lease {lease.lease_name!r} fence {lease.fencing_token} is not current"
            )
