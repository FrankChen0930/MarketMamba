"""Publish a small, versioned V7 data-health summary from data_health.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "v7-health-summary-v1"
SEVERITIES = ("BLOCK", "QUARANTINE", "WARN", "INFO")


def build_health_summary(
    document: Mapping[str, Any], *, data_id: str, generated_at: str
) -> dict[str, Any]:
    quality = document.get("quality_report")
    if not isinstance(quality, Mapping):
        raise ValueError("quality_report must be an object")
    if "blocking" not in quality or not isinstance(quality["blocking"], bool):
        raise ValueError("quality_report.blocking must be an explicit boolean")

    entries = quality.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("quality_report.entries must be a list")

    severity_counts = Counter({severity: 0 for severity in SEVERITIES})
    reason_counts: Counter[str] = Counter()
    dates: set[str] = set()
    stocks: set[str] = set()

    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("every quality entry must be an object")
        severity = entry.get("severity")
        if severity not in SEVERITIES:
            raise ValueError(f"unknown quality severity: {severity!r}")
        severity_counts[severity] += 1
        reason = entry.get("reason_code")
        reason_counts[str(reason) if reason not in (None, "") else "UNKNOWN_REASON"] += 1

        entry_dates = entry.get("dates", [])
        entry_stocks = entry.get("stocks", [])
        if not isinstance(entry_dates, list) or not isinstance(entry_stocks, list):
            raise ValueError("quality entry dates and stocks must be lists")
        dates.update(str(value) for value in entry_dates)
        stocks.update(str(value) for value in entry_stocks)

    blocking = quality["blocking"] or severity_counts["BLOCK"] > 0
    if blocking:
        state = "blocked"
    elif severity_counts["QUARANTINE"] or severity_counts["WARN"]:
        state = "degraded"
    else:
        state = "healthy"

    affected_dates = sorted(dates)
    affected_stocks = sorted(stocks)
    graph = document.get("graph", {})
    if not isinstance(graph, Mapping):
        raise ValueError("graph must be an object")

    return {
        "schema": SCHEMA,
        "state": state,
        "publish_allowed": not blocking,
        "data_id": data_id,
        "generated_at": generated_at,
        "counts": {
            "entries": len(entries),
            "affected_dates": len(affected_dates),
            "affected_stocks": len(affected_stocks),
        },
        "severity_counts": {severity: severity_counts[severity] for severity in SEVERITIES},
        "reason_counts": dict(sorted(reason_counts.items())),
        "affected_dates": affected_dates,
        "affected_stocks": affected_stocks,
        "graph": dict(graph),
    }


def write_health_summary(
    source: str | Path,
    output: str | Path,
    *,
    data_id: str,
    generated_at: str,
) -> dict[str, Any]:
    source_path = Path(source)
    output_path = Path(output)
    raw = source_path.read_bytes()
    document = json.loads(raw)
    summary = build_health_summary(document, data_id=data_id, generated_at=generated_at)
    summary["source"] = {
        "path": str(source_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--generated-at", required=True)
    arguments = parser.parse_args()
    summary = write_health_summary(
        arguments.source,
        arguments.output,
        data_id=arguments.data_id,
        generated_at=arguments.generated_at,
    )
    print(json.dumps({
        "state": summary["state"],
        "publish_allowed": summary["publish_allowed"],
        "data_id": summary["data_id"],
        "counts": summary["counts"],
        "severity_counts": summary["severity_counts"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
