#!/usr/bin/env python3
"""Validate K1 domain notes and generated navigation without modifying files."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOMAIN_DIR = ROOT / "knowledge/01_Domains"
GENERATED = ROOT / "knowledge/_generated"
REQUIRED = (
    "Data", "Features", "Models", "Training", "Labels", "Production",
    "Universe", "Execution", "Portfolio", "Operations", "UI",
)
OPTIONAL = ("Research",)
HEADINGS = (
    "Metadata", "Purpose", "Current Boundary", "Authority", "Inputs", "Outputs",
    "Current Implementation", "Evidence Classes", "Current vs Legacy / Research",
    "Invariants", "Known Limitations", "Do Not Use / Do Not Mix", "Important Paths",
    "Related Domains", "Update Triggers",
)
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+)\)")
BRANCH_REF_RE = re.compile(r"(?P<branch>[A-Za-z0-9_./-]+)@(?P<commit>[0-9a-f]{7,40}):(?P<path>[^\s`]+)")


def fail(message: str) -> None:
    raise AssertionError(message)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def validate_note(name: str) -> dict[str, str]:
    path = DOMAIN_DIR / f"{name}.md"
    if not path.is_file():
        fail(f"missing domain note: {path.relative_to(ROOT)}")
    text = path.read_text(encoding="utf-8")
    for heading in HEADINGS:
        if f"## {heading}" not in text:
            fail(f"{name} missing heading: {heading}")
    status_match = re.search(r"^- Status: `([^`]+)`$", text, re.MULTILINE)
    if not status_match or status_match.group(1) not in {
        "CANONICAL_DOMAIN_NOTE", "CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN"
    }:
        fail(f"{name} has invalid status")
    if not re.search(r"^- Last reviewed: \d{4}-\d{2}-\d{2}$", text, re.MULTILINE):
        fail(f"{name} missing ISO last-reviewed date")
    for target in LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not (path.parent / clean).resolve().exists():
            fail(f"broken link in {path.relative_to(ROOT)}: {target}")
    for match in BRANCH_REF_RE.finditer(text):
        resolved = git("rev-parse", match.group("commit"))
        subprocess.check_call(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"{resolved}:{match.group('path')}"],
            stdout=subprocess.DEVNULL,
        )
    return {"name": name, "status": status_match.group(1)}


def require_phrases(name: str, phrases: tuple[str, ...]) -> None:
    text = (DOMAIN_DIR / f"{name}.md").read_text(encoding="utf-8")
    for phrase in phrases:
        if phrase not in text:
            fail(f"{name} missing protected semantic phrase: {phrase}")


def main() -> int:
    notes = [validate_note(name) for name in REQUIRED + OPTIONAL]
    tier_a = json.loads((GENERATED / "tier-a-domain-index.json").read_text(encoding="utf-8"))
    coverage = json.loads((GENERATED / "coverage.json").read_text(encoding="utf-8"))
    paths = json.loads((GENERATED / "important-paths.json").read_text(encoding="utf-8"))
    if len(tier_a["domains"]) != 6:
        fail("Tier A index must contain six domains")
    if coverage["missing"] or coverage["result"] != "PASS":
        fail("domain coverage is incomplete")
    for local in paths["local_paths"]:
        if not (ROOT / local).exists():
            fail(f"important local path missing: {local}")
    require_phrases("Labels", ("Strict Phase 0 is `STOP`", "5d=`0`, 10d=`0`", "Historical-simulation readiness is separately `PASS`"))
    require_phrases("Models", ("Seed 43 is incomplete", "correlation about `0.9983`"))
    require_phrases("Production", ("fetch-only", "2026-09-27", "`UNKNOWN`"))
    require_phrases("Operations", ("Task 4", "correctness approval is not established", "Task 5 has not started"))
    print(json.dumps({
        "status": "PASS",
        "required_domains": len(REQUIRED),
        "optional_domains": len(OPTIONAL),
        "validated_notes": len(notes),
        "important_local_paths": len(paths["local_paths"]),
        "important_authority_refs": len(paths["authority_refs"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
