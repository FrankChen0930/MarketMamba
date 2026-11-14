#!/usr/bin/env python3
"""Unified, CI-ready MarketMamba knowledge health command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent


def run_json(label: str, command: list[str]) -> dict:
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        payload = {}
    status = payload.get("status", "FAIL" if process.returncode else "PASS")
    if process.returncode and status != "WARNING":
        status = "FAIL"
    return {
        "label": label,
        "status": status,
        "details": payload,
        "error": process.stderr.strip() or None,
    }


def build_health(as_of: str) -> dict:
    components = [
        run_json("Project Map", [sys.executable, str(TOOLS / "validate_project_map.py")]),
        run_json("Agent Guides", [sys.executable, str(TOOLS / "validate_agent_guides.py")]),
        run_json("Domains", [sys.executable, str(TOOLS / "validate_domains.py")]),
        run_json("Governance", [sys.executable, str(TOOLS / "validate_governance.py")]),
        run_json("Authority Drift", [sys.executable, str(TOOLS / "check_drift.py"), "--as-of", as_of]),
        run_json("Runtime Freshness", [sys.executable, str(TOOLS / "runtime_observations.py"), "status", "--as-of", as_of]),
        run_json("Generated Index", [sys.executable, str(TOOLS / "generate_indexes.py"), "--check"]),
    ]
    legacy = json.loads((ROOT / "knowledge/_generated/legacy-documentation-drift.json").read_text(encoding="utf-8"))
    components.append({"label": "Legacy Docs", "status": legacy["status"], "details": legacy, "error": None})
    statuses = [item["status"] for item in components]
    overall = "FAIL" if "FAIL" in statuses else ("WARNING" if "WARNING" in statuses else "PASS")
    return {
        "schema_version": 1,
        "authority": "VALIDATION_OUTPUT_ONLY",
        "as_of": as_of,
        "overall": overall,
        "components": components,
    }


def render(report: dict) -> str:
    width = max(len(item["label"]) for item in report["components"])
    lines = ["KNOWLEDGE HEALTH", ""]
    lines.extend(f"{item['label']:<{width}}  {item['status']}" for item in report["components"])
    lines.extend(["", f"Overall: {report['overall']}"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=datetime.now().astimezone().isoformat())
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--ci-warning-fails", action="store_true")
    args = parser.parse_args()
    report = build_health(args.as_of)
    if args.write:
        path = ROOT / "knowledge/_generated/knowledge-health.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True) if args.json else render(report))
    if report["overall"] == "FAIL":
        return 1
    if args.ci_warning_fails and report["overall"] == "WARNING":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
