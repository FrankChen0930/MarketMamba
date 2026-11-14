#!/usr/bin/env python3
"""Generate deterministic MarketMamba knowledge navigation indexes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "knowledge"
GENERATED = KNOWLEDGE / "_generated"
CONFIG_PATH = KNOWLEDGE / "governance-config.json"
STATUS_RE = re.compile(r"^- Status: `([^`]+)`$", re.MULTILINE)


def json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def domain_status(name: str) -> str:
    text = (KNOWLEDGE / "01_Domains" / f"{name}.md").read_text(encoding="utf-8")
    match = STATUS_RE.search(text)
    if not match:
        raise ValueError(f"domain note missing status: {name}")
    return match.group(1)


def read_legacy_document(relative: str) -> str:
    """Avoid recording unrelated, uncommitted legacy prose as generated authority."""
    path = ROOT / relative
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "--", relative],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if dirty.returncode:
        committed = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if committed.returncode == 0:
            return committed.stdout
    return path.read_text(encoding="utf-8")


def build_outputs() -> dict[Path, bytes]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = config["required_domains"]
    optional = config["optional_domains"]
    present = [name for name in required + optional if (KNOWLEDGE / "01_Domains" / f"{name}.md").is_file()]
    missing = [name for name in required if name not in present]
    tier_a = {
        "schema_version": 2,
        "generated": True,
        "notice": "GENERATED / DO NOT HAND EDIT",
        "authority": "NAVIGATION_ONLY",
        "tier": "A",
        "domains": [
            {"name": name, "path": f"knowledge/01_Domains/{name}.md", "status": domain_status(name)}
            for name in config["tier_a_domains"]
        ],
    }
    coverage = {
        "schema_version": 2,
        "generated": True,
        "notice": "GENERATED / DO NOT HAND EDIT",
        "authority": "NAVIGATION_ONLY",
        "required_domain_count": len(required),
        "optional_domain_count": len(optional),
        "present": present,
        "missing": missing,
        "result": "PASS" if not missing else "FAIL",
    }
    important = {
        "schema_version": 2,
        "generated": True,
        "notice": "GENERATED / DO NOT HAND EDIT",
        "authority": "NAVIGATION_ONLY",
        "local_paths": config["important_local_paths"],
        "authority_refs": config["important_authority_refs"],
    }
    dep_lines = [
        "# Domain Dependencies", "",
        "> `GENERATED / DO NOT HAND EDIT` — navigation only; source: `knowledge/governance-config.json`.", "",
        "| Upstream domain | Downstream domain |", "|---|---|",
    ]
    dep_lines.extend(f"| [{source}](../01_Domains/{source}.md) | [{target}](../01_Domains/{target}.md) |" for source, target in config["dependencies"])
    dep_lines.extend([
        "", "## Protected boundaries", "",
        "- Universe membership is not execution evidence.",
        "- Historical simulated execution is not verified execution.",
        "- Research completion is not production promotion.",
        "- Operations-ledger success is not upstream business success.",
        "- Source-code presence is not deployment evidence.", "",
    ])

    history = json.loads((GENERATED / "authority-history.json").read_text(encoding="utf-8"))
    topics: dict[str, int] = {}
    for event in history["events"]:
        topics[event["topic"]] = topics.get(event["topic"], 0) + 1
    history_summary = {
        "schema_version": 1,
        "generated": True,
        "notice": "GENERATED / DO NOT HAND EDIT",
        "authority": "NAVIGATION_ONLY",
        "event_count": len(history["events"]),
        "events_by_topic": dict(sorted(topics.items())),
        "tracking_start": "K2",
    }

    compiled = [(item, re.compile(item["pattern"])) for item in config["legacy_claim_patterns"]]
    hits = []
    for relative in config["legacy_documents"]:
        path = ROOT / relative
        if not path.is_file():
            continue
        for line_number, line in enumerate(read_legacy_document(relative).splitlines(), 1):
            for item, pattern in compiled:
                if pattern.search(line):
                    hits.append({
                        "document": relative,
                        "line": line_number,
                        "claim_id": item["id"],
                        "severity": item["severity"],
                        "excerpt": line.strip()[:240],
                    })
    legacy = {
        "schema_version": 1,
        "generated": True,
        "notice": "GENERATED / DO NOT HAND EDIT",
        "authority": "VALIDATION_OUTPUT_ONLY",
        "status": "WARNING" if hits else "PASS",
        "documents_scanned": config["legacy_documents"],
        "hits": hits,
        "automatic_edits_performed": False,
    }
    return {
        GENERATED / "tier-a-domain-index.json": json_bytes(tier_a),
        GENERATED / "coverage.json": json_bytes(coverage),
        GENERATED / "important-paths.json": json_bytes(important),
        GENERATED / "domain-dependencies.md": ("\n".join(dep_lines)).encode("utf-8"),
        GENERATED / "authority-history-summary.json": json_bytes(history_summary),
        GENERATED / "legacy-documentation-drift.json": json_bytes(legacy),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    mismatches = []
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != content:
                mismatches.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    report = {
        "status": "FAIL" if mismatches else "PASS",
        "mode": "CHECK" if args.check else "WRITE",
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
        "mismatches": mismatches,
    }
    print(json.dumps(report, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
