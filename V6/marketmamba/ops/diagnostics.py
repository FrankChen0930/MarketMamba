"""Human-readable, secret-safe diagnostics for operational current state."""

from __future__ import annotations

import re
from typing import Mapping

from marketmamba.ops.store import redact_sensitive_text


_QUOTED_ABSOLUTE_PATH = re.compile(
    r'''(?P<quote>["'])(?:[A-Za-z]:[\\/]|\\\\|//|/).*?(?P=quote)'''
)
_FILE_ABSOLUTE_URI = re.compile(
    r'''(?i)\bfile:(?:/{1,3}|[A-Za-z]:[\\/])[^\s,;'"<>()\[\]{}]+'''
)
_UNC_ABSOLUTE_PATH = re.compile(
    r'''(?<![:\\/\w])(?:\\\\|//)'''
    r'''[^\\/\s,;'"<>()\[\]{}]+'''
    r'''(?:[\\/][^\\/\s,;'"<>()\[\]{}]+)+'''
)
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r'''(?<!\w)[A-Za-z]:[\\/]'''
    r'''(?:[^\\/\s,;'"<>()\[\]{}]+[\\/])*'''
    r'''[^\\/\s,;'"<>()\[\]{}]+'''
)
_POSIX_ABSOLUTE_PATH = re.compile(
    r'''(?<![:\\/\w])/'''
    r'''(?:[^/\s,;'"<>()\[\]{}]+/)*'''
    r'''[^/\s,;'"<>()\[\]{}]+'''
)
_FIELD_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_.-])[A-Za-z_][A-Za-z0-9_.-]*\s*=\s*"
)


def _contains_absolute_path(value: str) -> bool:
    return any(
        pattern.search(value)
        for pattern in (
            _QUOTED_ABSOLUTE_PATH,
            _FILE_ABSOLUTE_URI,
            _UNC_ABSOLUTE_PATH,
            _WINDOWS_ABSOLUTE_PATH,
            _POSIX_ABSOLUTE_PATH,
        )
    )


def _opaque_preserving_edge_space(value: str) -> str:
    leading_length = len(value) - len(value.lstrip())
    trailing_length = len(value) - len(value.rstrip())
    leading = value[:leading_length]
    trailing = value[len(value) - trailing_length :] if trailing_length else ""
    return f"{leading}[PATH]{trailing}"


def _redact_bounded_path_values(value: str) -> str:
    """Opaque whole path-bearing evidence values, including paths with spaces."""

    assignments = list(_FIELD_ASSIGNMENT.finditer(value))
    if not assignments:
        return "[PATH]" if _contains_absolute_path(value) else value

    parts: list[str] = []
    cursor = 0
    for index, assignment in enumerate(assignments):
        parts.append(value[cursor : assignment.end()])
        value_end = (
            assignments[index + 1].start()
            if index + 1 < len(assignments)
            else len(value)
        )
        field_value = value[assignment.end() : value_end]
        parts.append(
            _opaque_preserving_edge_space(field_value)
            if _contains_absolute_path(field_value)
            else field_value
        )
        cursor = value_end
    return "".join(parts)


def sanitize_output_text(value: str | None) -> str | None:
    """Redact credentials and opaque absolute paths in user-facing output."""

    redacted = redact_sensitive_text(value)
    if redacted is None:
        return None
    redacted = _redact_bounded_path_values(redacted)
    redacted = _QUOTED_ABSOLUTE_PATH.sub(
        lambda match: f"{match.group('quote')}[PATH]{match.group('quote')}",
        redacted,
    )
    for pattern in (
        _FILE_ABSOLUTE_URI,
        _UNC_ABSOLUTE_PATH,
        _WINDOWS_ABSOLUTE_PATH,
        _POSIX_ABSOLUTE_PATH,
    ):
        redacted = pattern.sub("[PATH]", redacted)
    return redacted


def _display(value: object) -> str:
    if value is None:
        return "—"
    return sanitize_output_text(str(value)) or "—"


def render_current_state(state: Mapping[str, object]) -> str:
    """Render a compact view that cannot be mistaken for full production."""

    stages = state.get("stage_statuses") or []
    stage_text = ", ".join(
        f"{_display(item.get('stage'))}={_display(item.get('status'))}"
        for item in stages
        if isinstance(item, Mapping)
    ) or "—"
    freshness = state.get("freshness")
    freshness_status = (
        freshness.get("status") if isinstance(freshness, Mapping) else None
    )
    freshness_observed = (
        freshness.get("observed_value") if isinstance(freshness, Mapping) else None
    )
    freshness_threshold = (
        freshness.get("threshold_value") if isinstance(freshness, Mapping) else None
    )
    calendar = state.get("calendar")
    calendar_status = (
        calendar.get("status") if isinstance(calendar, Mapping) else None
    )
    artifacts = state.get("artifact_summary")
    artifact_count = artifacts.get("count") if isinstance(artifacts, Mapping) else 0
    artifact_fingerprints = (
        artifacts.get("fingerprint_kinds") if isinstance(artifacts, Mapping) else []
    )
    return "\n".join(
        (
            "MarketMamba operational state",
            "Mode: FETCH_ONLY (no inference, portfolio, or publication)",
            f"As of committed ledger: {_display(state.get('as_of'))}",
            f"Requested date: {_display(state.get('latest_requested_date'))} "
            "(Asia/Taipei request date; trading calendar unverified)",
            f"Logical run: {_display(state.get('latest_logical_run_id'))}",
            f"Status: {_display(state.get('run_status'))}",
            f"Attempts: {_display(state.get('attempt_count'))} "
            f"(latest: {_display(state.get('latest_attempt'))})",
            f"Duration: {_display(state.get('duration_seconds'))} seconds",
            f"Active stage: {_display(state.get('active_stage'))}",
            f"Stages: {stage_text}",
            f"Calendar: {_display(calendar_status)} "
            "(unknown; not exposed by legacy fetcher)",
            f"Last successful fetch date: "
            f"{_display(state.get('last_successful_fetch_date'))}",
            f"Freshness: {_display(freshness_status)} "
            f"(observed: {_display(freshness_observed)}, "
            f"threshold: {_display(freshness_threshold)})",
            f"Artifacts: {_display(artifact_count)} "
            f"(fingerprint kinds: {_display(artifact_fingerprints)})",
            f"Warnings: {_display(state.get('warning_count'))}",
            f"Incidents: {_display(state.get('incident_status'))}",
        )
    )
