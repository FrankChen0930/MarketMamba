#!/usr/bin/env python3
"""Record explicit external runtime observations and report their freshness."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = ROOT / "knowledge/runtime-observations"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def parse_time(value: str) -> datetime:
    if "T" not in value:
        value += "T23:59:59+08:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return parsed


def classify(record: dict, as_of: datetime) -> dict:
    observed = parse_time(record["observed_at"])
    expires = observed + timedelta(days=int(record["ttl_days"]))
    status = "FRESH" if as_of <= expires else "STALE_OBSERVATION"
    return {
        "observation_id": record["observation_id"],
        "topic": record["topic"],
        "observed_at": observed.isoformat(),
        "expires_at": expires.isoformat(),
        "status": status,
        "actor": record["actor"],
        "method": record["method"],
        "evidence_ref": record["evidence_ref"],
        "value": record["value"],
    }


def load_status(store: Path, as_of: datetime) -> dict:
    records = []
    for path in sorted(store.glob("*.json")) if store.exists() else []:
        records.append(classify(json.loads(path.read_text(encoding="utf-8")), as_of))
    status = "WARNING" if not records or any(item["status"] != "FRESH" for item in records) else "PASS"
    return {"schema_version": 1, "status": status, "as_of": as_of.isoformat(), "observations": records}


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def record(args: argparse.Namespace) -> dict:
    if not ID_RE.match(args.id):
        raise ValueError("id must use lowercase letters, digits, and hyphens")
    observed = parse_time(args.observed_at)
    value = json.loads(args.value_json)
    if not isinstance(value, dict) or not value:
        raise ValueError("value-json must be a non-empty JSON object")
    for field in ("topic", "actor", "method", "evidence_ref"):
        if not getattr(args, field).strip():
            raise ValueError(f"{field.replace('_', '-')} must be non-empty")
    payload = {
        "schema_version": 1,
        "observation_id": args.id,
        "topic": args.topic.strip(),
        "observed_at": observed.isoformat(),
        "ttl_days": args.ttl_days,
        "actor": args.actor.strip(),
        "method": args.method.strip(),
        "evidence_ref": args.evidence_ref.strip(),
        "value": value,
        "confidence": args.confidence,
    }
    atomic_write(args.store / f"{args.id}.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    sub = parser.add_subparsers(dest="command", required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--as-of", default=datetime.now().astimezone().isoformat())
    record_parser = sub.add_parser("record")
    record_parser.add_argument("--id", required=True)
    record_parser.add_argument("--topic", required=True)
    record_parser.add_argument("--observed-at", required=True)
    record_parser.add_argument("--ttl-days", type=int, default=7)
    record_parser.add_argument("--actor", required=True)
    record_parser.add_argument("--method", required=True)
    record_parser.add_argument("--evidence-ref", required=True)
    record_parser.add_argument("--value-json", required=True)
    record_parser.add_argument("--confidence", choices=("LOW", "MEDIUM", "HIGH"), default="MEDIUM")
    args = parser.parse_args()
    try:
        if args.command == "record":
            if args.ttl_days <= 0:
                raise ValueError("ttl-days must be positive")
            result = record(args)
        else:
            result = load_status(args.store, parse_time(args.as_of))
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
