#!/usr/bin/env python3
"""Check authority identities, repin candidates, and runtime freshness."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from authority_refs import parse_authority_ref, validate_authority_ref  # noqa: E402
from runtime_observations import load_status, parse_time  # noqa: E402


def render_markdown(report: dict) -> str:
    lines = [
        "# Knowledge Drift Report", "",
        "> `GENERATED / DO NOT HAND EDIT` — validation output, not canonical authority.", "",
        f"- As of: `{report['as_of']}`",
        f"- Status: `{report['status']}`",
        f"- Runtime freshness: `{report['runtime_freshness']['status']}`", "",
        "## Authority references", "",
        "| Reference | Status | Findings |", "|---|---|---|",
    ]
    for item in report["authority_refs"]:
        codes = ", ".join(finding["code"] for finding in item["findings"]) or "—"
        lines.append(f"| `{item['raw']}` | `{item['status']}` | {codes} |")
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(f"- `{item['severity']}` `{item['code']}` — {item['message']}" for item in report["findings"])
    else:
        lines.append("- No drift detected.")
    return "\n".join(lines) + "\n"


def build_report(as_of: str) -> dict:
    parsed_as_of = parse_time(as_of)
    authority = json.loads((ROOT / "knowledge/00_Project_Map/authority-map.json").read_text(encoding="utf-8"))
    results = []
    findings = []
    for topic in authority["topics"]:
        ref = parse_authority_ref(topic.get("primary_authority") or "")
        if not ref:
            continue
        result = validate_authority_ref(ROOT, ref)
        result["topic"] = topic["topic"]
        results.append(result)
        for finding in result["findings"]:
            messages = {
                "INVALID_COMMIT": "Authority commit does not resolve.",
                "MISSING_AUTHORITY_PATH": "Authority path does not exist at the pinned commit.",
                "BRANCH_LABEL_MISSING": "Optional branch label is gone; durable commit identity still resolves.",
                "BRANCH_LABEL_MOVED": "Optional branch label moved; review whether authority should be superseded.",
                "AUTHORITY_REPIN_CANDIDATE": "Identical authority exists on main; review a durable repin.",
            }
            findings.append({**finding, "topic": topic["topic"], "message": messages[finding["code"]]})
    runtime = load_status(ROOT / "knowledge/runtime-observations", parsed_as_of)
    for observation in runtime["observations"]:
        if observation["status"] == "STALE_OBSERVATION":
            findings.append({
                "code": "STALE_OBSERVATION", "severity": "WARNING", "topic": observation["topic"],
                "message": "External observation exceeded its TTL; re-observe rather than refreshing the timestamp.",
            })
    if any(item["status"] == "FAIL" for item in results):
        status = "FAIL"
    elif findings or runtime["status"] == "WARNING":
        status = "WARNING"
    else:
        status = "PASS"
    return {
        "schema_version": 2,
        "authority": "VALIDATION_OUTPUT_ONLY",
        "as_of": parsed_as_of.isoformat(),
        "status": status,
        "authority_refs": results,
        "runtime_freshness": runtime,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=datetime.now().astimezone().isoformat())
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(args.as_of)
    except ValueError as error:
        parser.error(str(error))
    if args.write:
        generated = ROOT / "knowledge/_generated"
        generated.mkdir(parents=True, exist_ok=True)
        (generated / "drift-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (generated / "drift-report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
