"""Fail-closed temporal, interval, provenance, and manifest primitives for V7."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


class ContractError(ValueError):
    """Raised when correctness-sensitive inputs are ambiguous."""


def normalize_calendar(calendar: Sequence[str]) -> tuple[str, ...]:
    try:
        days = tuple(date.fromisoformat(str(value)).isoformat() for value in calendar)
    except (TypeError, ValueError) as error:
        raise ContractError("calendar contains an invalid ISO date") from error
    if not days or list(days) != sorted(set(days)):
        raise ContractError("calendar must be non-empty, unique, and increasing")
    return days


def first_session_after(calendar: Sequence[str], event_date: str) -> str | None:
    days = normalize_calendar(calendar)
    try:
        normalized = date.fromisoformat(str(event_date)).isoformat()
    except (TypeError, ValueError) as error:
        raise ContractError("event_date must be an ISO date") from error
    position = bisect_right(days, normalized)
    return days[position] if position < len(days) else None


def _publication_datetime(value: Any) -> datetime | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def normalize_publication_event(
    event: Mapping[str, Any],
    *,
    calendar: Sequence[str],
    source: str,
    timestamp_semantics: str,
    retrieval_method: str = "local-or-official-source",
) -> dict[str, Any]:
    if not source.strip() or not timestamp_semantics.strip() or not retrieval_method.strip():
        raise ContractError("source, timestamp_semantics, and retrieval_method are required")
    published = _publication_datetime(event.get("published_at"))
    raw = event.get("published_at")
    status = "VERIFIED" if published is not None else "UNKNOWN"
    reason = None if published is not None else (
        "MISSING_PUBLICATION_TIMESTAMP" if raw is None or not str(raw).strip()
        else "INVALID_PUBLICATION_TIMESTAMP"
    )
    publication_date = published.date().isoformat() if published is not None else None
    available = first_session_after(calendar, publication_date) if publication_date else None
    return {
        "entity_id": str(event.get("entity_id", "")),
        "period": str(event.get("period", "")),
        "published_at": published.isoformat() if published is not None else None,
        "publication_date": publication_date,
        "available_session": available,
        "verification_status": status,
        "unknown_reason": reason,
        "source": source.strip(),
        "timestamp_semantics": timestamp_semantics.strip(),
        "retrieval_method": retrieval_method.strip(),
    }


def _iso_day(value: Any, field: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must be an ISO date") from error


def validate_effective_intervals(
    rows: Iterable[Mapping[str, Any]], *, key_fields: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    if not key_fields:
        raise ContractError("key_fields cannot be empty")
    normalized: list[dict[str, Any]] = []
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if any(not part for part in key):
            raise ContractError("effective interval key is missing")
        row["effective_from"] = _iso_day(row.get("effective_from"), "effective_from")
        if row.get("effective_to") not in (None, ""):
            row["effective_to"] = _iso_day(row["effective_to"], "effective_to")
            if row["effective_to"] <= row["effective_from"]:
                raise ContractError("effective interval must be non-empty")
        else:
            row["effective_to"] = None
        grouped[key].append(row)
    for key, group in grouped.items():
        group.sort(key=lambda item: item["effective_from"])
        previous_end: str | None = None
        for index, row in enumerate(group):
            if index and (previous_end is None or row["effective_from"] < previous_end):
                raise ContractError(f"effective interval overlap for {key}")
            previous_end = row["effective_to"]
            normalized.append(row)
    return tuple(sorted(normalized, key=lambda row: tuple(str(row[field]) for field in key_fields) + (row["effective_from"],)))


def active_intervals(rows: Iterable[Mapping[str, Any]], session: str) -> list[dict[str, Any]]:
    day = _iso_day(session, "session")
    return [
        dict(row)
        for row in rows
        if str(row["effective_from"]) <= day
        and (row.get("effective_to") in (None, "") or day < str(row["effective_to"]))
    ]


def canonical_manifest(
    schema_version: str,
    parameters: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if not schema_version.strip() or not input_sha256:
        raise ContractError("schema_version and input hashes are required")
    payload = {
        "schema_version": schema_version.strip(),
        "parameters": dict(parameters),
        "input_sha256": dict(input_sha256),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "manifest_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
