"""Effective-dated Taiwan security lifecycle reconstruction.

The builder consumes evidence-backed lifecycle events. It never infers listing
or tradability from price observations. Intervals use half-open date semantics;
an event without enough prior evidence is retained as unresolved.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Mapping


class LifecycleContractError(ValueError):
    """Raised when lifecycle evidence is internally inconsistent."""


_EVENT_TYPES = {"LIST", "DELIST", "SUSPEND", "RESUME", "TRANSFER", "CODE_CHANGE", "RELIST"}
_REQUIRED = {"event_type", "stock_id", "canonical_security_id", "market", "effective_at", "source"}


def _normalize_event(raw: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(field for field in _REQUIRED if not str(raw.get(field, "")).strip())
    if missing:
        raise LifecycleContractError(f"lifecycle event missing fields: {missing}")
    event = dict(raw)
    event["event_type"] = str(event["event_type"]).upper().strip()
    if event["event_type"] not in _EVENT_TYPES:
        raise LifecycleContractError(f"unsupported lifecycle event: {event['event_type']}")
    try:
        event["effective_at"] = date.fromisoformat(str(event["effective_at"])).isoformat()
    except ValueError as error:
        raise LifecycleContractError("effective_at must be an ISO date") from error
    for field in ("stock_id", "canonical_security_id", "market", "source"):
        event[field] = str(event[field]).strip()
    event["security_type"] = str(event.get("security_type", "COMMON_STOCK")).strip()
    event["verification_state"] = str(event.get("verification_state", "VERIFIED")).strip()
    event["provenance"] = str(event.get("provenance", event["source"])).strip()
    if not event["provenance"]:
        raise LifecycleContractError("event provenance is required")
    return event


def build_security_lifecycle(events: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic half-open lifecycle intervals from verified events."""

    normalized = [_normalize_event(event) for event in events]
    normalized.sort(
        key=lambda event: (
            event["canonical_security_id"],
            event["effective_at"],
            event["event_type"],
            event["stock_id"],
            event["source"],
        )
    )
    seen_dates: set[tuple[str, str]] = set()
    for event in normalized:
        key = (event["canonical_security_id"], event["effective_at"])
        if key in seen_dates:
            raise LifecycleContractError(
                "multiple lifecycle events for one security on the same effective date "
                "require an explicit combined event"
            )
        seen_dates.add(key)

    intervals: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    open_by_security: dict[str, dict[str, Any]] = {}
    listing_start_by_security: dict[str, str] = {}

    def open_interval(event: Mapping[str, Any], status: str) -> None:
        security = event["canonical_security_id"]
        if security in open_by_security:
            raise LifecycleContractError(f"{event['event_type']} while lifecycle is already open")
        if status == "LISTED":
            listing_start_by_security[security] = event["effective_at"]
        open_by_security[security] = {
            "stock_id": event["stock_id"],
            "canonical_security_id": security,
            "market": event["market"],
            "security_type": event["security_type"],
            "status": status,
            "listing_date": listing_start_by_security.get(security, event["effective_at"]),
            "delisting_date": None,
            "effective_from": event["effective_at"],
            "effective_to": None,
            "source": event["source"],
            "provenance": event["provenance"],
            "verification_state": event["verification_state"],
            "tradability_certified": status == "SUSPENDED",
        }

    def close_interval(event: Mapping[str, Any], *, delisted: bool = False) -> dict[str, Any] | None:
        security = event["canonical_security_id"]
        current = open_by_security.pop(security, None)
        if current is None:
            unresolved.append({
                **dict(event),
                "reason": (
                    "DELIST_WITHOUT_VERIFIED_LISTING_START"
                    if event["event_type"] == "DELIST"
                    else f"{event['event_type']}_WITHOUT_OPEN_LIFECYCLE"
                ),
                "verification_state": "UNKNOWN",
            })
            return None
        if event["effective_at"] <= current["effective_from"]:
            raise LifecycleContractError("lifecycle interval must have positive duration")
        if current["stock_id"] != event.get("previous_stock_id", current["stock_id"]) and event["event_type"] == "CODE_CHANGE":
            raise LifecycleContractError("code-change previous_stock_id does not match open lifecycle")
        current["effective_to"] = event["effective_at"]
        if delisted:
            current["delisting_date"] = event["effective_at"]
        intervals.append(current)
        return current

    for event in normalized:
        kind = event["event_type"]
        security = event["canonical_security_id"]
        if event["verification_state"] != "VERIFIED":
            unresolved.append({**event, "reason": "EVENT_NOT_VERIFIED"})
            continue
        if kind in {"LIST", "RELIST"}:
            open_interval(event, "LISTED")
        elif kind == "DELIST":
            close_interval(event, delisted=True)
            listing_start_by_security.pop(security, None)
        elif kind == "SUSPEND":
            previous = close_interval(event)
            if previous is not None:
                open_interval({**event, "stock_id": previous["stock_id"], "market": previous["market"]}, "SUSPENDED")
        elif kind == "RESUME":
            previous = close_interval(event)
            if previous is not None:
                if previous["status"] != "SUSPENDED":
                    raise LifecycleContractError("RESUME requires an open SUSPENDED interval")
                open_interval({**event, "stock_id": previous["stock_id"], "market": previous["market"]}, "LISTED")
        elif kind == "TRANSFER":
            previous = close_interval(event)
            if previous is not None:
                open_interval(event, "LISTED")
        elif kind == "CODE_CHANGE":
            previous = close_interval(event)
            if previous is not None:
                open_interval(event, previous["status"])

    intervals.extend(open_by_security.values())
    intervals.sort(key=lambda row: (row["canonical_security_id"], row["effective_from"], row["stock_id"]))
    unresolved.sort(key=lambda row: (row["canonical_security_id"], row["effective_at"], row["event_type"]))
    return {"events": normalized, "intervals": intervals, "unresolved": unresolved}
