#!/usr/bin/env python3
"""Validate agent-guide navigation without implementing full drift checking."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
CLAUDE = ROOT / "CLAUDE.md"
CURRENT = "knowledge/00_Project_Map/Current_State.md"
AUTHORITY = "knowledge/00_Project_Map/Authority_Map.md"
ARCHIVE = ROOT / "docs/archive/legacy-claude-project-guide.md"
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def fail(message: str) -> None:
    raise AssertionError(message)


def validate_links(path: Path) -> None:
    for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not (path.parent / clean).resolve().exists():
            fail(f"broken link in {path.relative_to(ROOT)}: {target}")


def main() -> int:
    for path in (AGENTS, CLAUDE, ROOT / CURRENT, ROOT / AUTHORITY, ARCHIVE):
        if not path.is_file():
            fail(f"missing knowledge navigation file: {path.relative_to(ROOT)}")

    agents = AGENTS.read_text(encoding="utf-8")
    claude = CLAUDE.read_text(encoding="utf-8")
    archive = ARCHIVE.read_text(encoding="utf-8")

    current_pos = agents.find(CURRENT)
    authority_pos = agents.find(AUTHORITY)
    contract_pos = agents.find("machine contract")
    domain_pos = agents.find("domain note")
    if min(current_pos, authority_pos, contract_pos, domain_pos) < 0:
        fail("AGENTS bootstrap order is incomplete")
    if not current_pos < authority_pos < contract_pos < domain_pos:
        fail("AGENTS bootstrap order is incorrect")
    for phrase in (
        "Knowledge-impact Gate", "UNMERGED", "NOT_DEPLOYED",
        "observed_at", "fail closed", "git diff --check",
        "Reconnaissance before creation", "Handoff contract",
        "knowledge_update: not_required", "MARKETMAMBA_DATA_ROOT",
    ):
        if phrase not in agents:
            fail(f"AGENTS missing stable rule: {phrase}")

    if "AGENTS.md" not in claude or CURRENT not in claude or AUTHORITY not in claude:
        fail("CLAUDE thin wrapper does not link canonical policy and project map")
    if "reconnaissance-before-creation" not in claude or "same coherent change" not in claude:
        fail("CLAUDE does not delegate precreation and same-change memory rules")
    if len(claude.encode("utf-8")) > 5000:
        fail("CLAUDE thin wrapper exceeds 5 KB")
    for volatile in (
        "## Current Status", "strict Phase 0", "48 features",
        "MarketMamba_V62", "SPECIFIED_NOT_EXECUTED",
    ):
        if volatile in claude:
            fail(f"CLAUDE contains duplicated volatile state: {volatile}")
    if "HISTORICAL_ONLY / NON_AUTHORITATIVE" not in archive:
        fail("legacy Claude archive is not explicitly non-authoritative")

    validate_links(CLAUDE)
    print(json.dumps({
        "status": "PASS",
        "agents_bytes": len(agents.encode("utf-8")),
        "claude_bytes": len(claude.encode("utf-8")),
        "claude_threshold_bytes": 5000,
        "legacy_archive_preserved": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"agent-guide validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
