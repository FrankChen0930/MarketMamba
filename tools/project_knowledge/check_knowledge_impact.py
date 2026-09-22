#!/usr/bin/env python3
"""Report potential knowledge impact for a Git diff without editing notes."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = ROOT / "knowledge/knowledge-impact-map.json"
RANK = {"NO_KNOWLEDGE_IMPACT": 0, "UPDATE_IF_SEMANTIC_CHANGE": 1, "REVIEW_REQUIRED": 2}


def git_changed(diff_range: str | None, staged: bool) -> list[str]:
    args = ["git", "-C", str(ROOT), "diff", "--name-only"]
    if staged:
        args.append("--cached")
    elif diff_range:
        args.append(diff_range)
    else:
        args.append("HEAD")
    output = subprocess.check_output(args, text=True)
    return sorted({line.strip() for line in output.splitlines() if line.strip()})


def matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:])
    )


def analyze(changed: list[str], config: dict, acknowledgment: str | None) -> dict:
    impacts: dict[str, dict] = {}
    matched_rules: dict[str, list[str]] = {path: [] for path in changed}
    for path in changed:
        for rule in config["rules"]:
            if any(matches(path, pattern) for pattern in rule["patterns"]):
                matched_rules[path].append(rule["id"])
                for impact in rule["impacts"]:
                    current = impacts.get(impact["artifact"])
                    if current is None or RANK[impact["severity"]] > RANK[current["severity"]]:
                        impacts[impact["artifact"]] = {
                            "artifact": impact["artifact"],
                            "severity": impact["severity"],
                            "rules": [rule["id"]],
                        }
                    elif rule["id"] not in current["rules"]:
                        current["rules"].append(rule["id"])
        if not matched_rules[path]:
            matched_rules[path].append("DEFAULT")
    ordered = sorted(impacts.values(), key=lambda item: (-RANK[item["severity"]], item["artifact"]))
    actionable = any(RANK[item["severity"]] > 0 for item in ordered)
    if acknowledgment is not None and not acknowledgment.strip():
        raise ValueError("NO_KNOWLEDGE_CHANGE_REQUIRED requires a non-empty reason")
    status = "WARNING" if actionable else "PASS"
    ack = None
    if acknowledgment is not None:
        ack = {"decision": "NO_KNOWLEDGE_CHANGE_REQUIRED", "reason": acknowledgment.strip()}
        status = "ACKNOWLEDGED" if actionable else "PASS"
    return {
        "schema_version": 1,
        "status": status,
        "changed": changed,
        "matched_rules": matched_rules,
        "potential_knowledge_impact": ordered,
        "acknowledgment": ack,
        "automatic_updates_performed": False,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diff_range", nargs="?")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--ack-no-change", metavar="REASON")
    parser.add_argument("--write-record", type=Path)
    args = parser.parse_args()
    if args.staged and args.diff_range:
        parser.error("choose a diff range or --staged, not both")
    changed = sorted(set(args.changed_file or git_changed(args.diff_range, args.staged)))
    config = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    try:
        report = analyze(changed, config, args.ack_no_change)
    except ValueError as error:
        parser.error(str(error))
    if args.write_record:
        atomic_json(args.write_record, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
