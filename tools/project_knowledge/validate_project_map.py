#!/usr/bin/env python3
"""Read-only validator for the K1 project-map slice."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from authority_refs import parse_authority_ref, validate_authority_ref  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "knowledge" / "00_Project_Map"
CURRENT_MD = MAP / "Current_State.md"
AUTHORITY_MD = MAP / "Authority_Map.md"
CURRENT_JSON = MAP / "current-state.json"
AUTHORITY_JSON = MAP / "authority-map.json"
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def fail(message: str) -> None:
    raise AssertionError(message)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def validate_markdown_links(path: Path) -> None:
    for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#")):
            continue
        clean = target.split("#", 1)[0]
        resolved = (path.parent / clean).resolve()
        if not resolved.exists():
            fail(f"broken link in {path.relative_to(ROOT)}: {target}")


def validate_authority_reference(reference: str | None, authority_class: str) -> None:
    if authority_class == "UNKNOWN":
        if reference is not None:
            fail("UNKNOWN authority must have null primary_authority")
        return
    if reference is None:
        fail(f"{authority_class} authority requires primary_authority")

    parsed = parse_authority_ref(reference)
    if parsed:
        result = validate_authority_ref(ROOT, parsed)
        if result["status"] == "FAIL":
            fail(f"invalid durable authority reference: {reference}: {result['findings']}")
        return

    if reference.startswith(("external:", "run-local")) or " + " in reference:
        return
    clean = reference.split("#", 1)[0]
    if "/" in clean and not (ROOT / clean).exists():
        fail(f"missing primary authority path: {clean}")


def main() -> int:
    for path in (CURRENT_MD, AUTHORITY_MD, CURRENT_JSON, AUTHORITY_JSON):
        if not path.is_file():
            fail(f"missing deliverable: {path.relative_to(ROOT)}")

    current = json.loads(CURRENT_JSON.read_text(encoding="utf-8"))
    authority = json.loads(AUTHORITY_JSON.read_text(encoding="utf-8"))
    current_md = CURRENT_MD.read_text(encoding="utf-8")
    authority_md = AUTHORITY_MD.read_text(encoding="utf-8")

    validate_markdown_links(CURRENT_MD)
    validate_markdown_links(AUTHORITY_MD)

    topics = authority["topics"]
    names = [item["topic"] for item in topics]
    if len(names) != len(set(names)):
        fail("duplicate authority topic")
    allowed = {
        "CANONICAL", "CURRENT_REFERENCE", "RESEARCH_EVIDENCE",
        "HISTORICAL_ONLY", "RUNTIME_OBSERVATION",
        "UNMERGED_IMPLEMENTATION", "UNKNOWN",
    }
    for item in topics:
        if item["authority_class"] not in allowed:
            fail(f"invalid authority class: {item['authority_class']}")
        validate_authority_reference(item["primary_authority"], item["authority_class"])

    expected_status = "CANONICAL_CURRENT_PROJECT_MAP"
    if current["status"] != expected_status or authority["status"] != expected_status:
        fail("JSON review status mismatch")
    if expected_status not in current_md or expected_status not in authority_md:
        fail("Markdown review status mismatch")
    if "Authority_Map.md" not in current_md:
        fail("Current_State does not reference Authority_Map")
    if current["correctness"]["strict_phase0"] != "STOP":
        fail("strict Phase 0 mismatch")
    if current["correctness"]["historical_simulation_readiness"] != "PASS":
        fail("historical simulation readiness mismatch")
    if current["research"]["corrected_e5"]["formal_seeds"] != [17, 29]:
        fail("formal corrected E5 seeds must be exactly [17, 29]")
    seed43 = current["research"]["seed43"]
    if "INCOMPLETE" not in seed43["status"] or "EXCLUDED" not in seed43["status"]:
        fail("seed43 status must remain incomplete and excluded")
    if seed43["stale_pointer_metadata"]["best_epoch"] != 8:
        fail("seed43 stale pointer metadata mismatch")
    orphan = seed43["surviving_orphan"]
    if orphan["epoch"] != 15 or orphan["step"] != 39915:
        fail("seed43 surviving orphan identity mismatch")
    if orphan["authority"] != "convergence evidence only":
        fail("seed43 orphan authority mismatch")
    if seed43["promotion_allowed"] or seed43["direct_resume_allowed"]:
        fail("seed43 must not be promotable or directly resumable")
    lowered = current_md.lower()
    if "loadable epoch-8" in lowered or "loadable epoch 8" in lowered:
        fail("Current_State incorrectly describes epoch 8 as loadable")
    e1 = current["research"]["e1"]
    if (e1["status"] != "COMPLETED_RESEARCH_EXPERIMENT"
            or e1["outcome"] != "PASS_REFRESH_HYPOTHESIS"
            or e1["research_evidence_only"] is not True
            or e1["formal_seeds"] != [17, 29]):
        fail("E1 completed research result mismatch")
    if current["operations"]["deployed"] is not False:
        fail("Phase A deployment mismatch")
    tasks = current["operations"]["tasks"]
    if "APPROVAL_NOT_ESTABLISHED" not in tasks["task4_atomic_projection_read_only_cli"]:
        fail("Phase A Task 4 must remain approval-pending")
    if tasks["task5_replay_formal_acceptance_report"] != "NOT_STARTED":
        fail("Phase A Task 5 must remain not started")

    print(json.dumps({
        "status": "PASS",
        "markdown_files": 2,
        "json_files": 2,
        "authority_topics": len(topics),
        "duplicate_topics": 0,
        "broken_links": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"project-map validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
