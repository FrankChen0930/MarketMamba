#!/usr/bin/env python3
"""Validate K2 governance contracts, authority identities, and explicit invariants."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from authority_refs import parse_authority_ref, validate_authority_ref  # noqa: E402

LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def fail(message: str) -> None:
    raise AssertionError(message)


def require(text: str, phrase: str, source: str) -> None:
    if phrase not in text:
        fail(f"{source} missing protected phrase: {phrase}")


def validate_markdown_links(paths: list[Path]) -> int:
    checked = 0
    for path in paths:
        for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            clean = target.split("#", 1)[0]
            if not (path.parent / clean).resolve().exists():
                fail(f"broken link in {path.relative_to(ROOT)}: {target}")
            checked += 1
    return checked


def validate_explicit_invariants(
    authority: dict, current: dict, current_md: str, authority_md: str,
    features_text: str, operations_md: str,
) -> None:
    topics = authority["topics"]
    names = [item["topic"] for item in topics]
    if len(names) != len(set(names)):
        fail("duplicate canonical topic owner")
    topic_map = {item["topic"]: item for item in topics}
    feature_ref = topic_map["Feature contract"]["primary_authority"]
    require(features_text, feature_ref, "Features.md")

    require(current_md, "Strict Phase 0", "Current_State.md")
    require(current_md, "`STOP`", "Current_State.md")
    require(current_md, "Historical-simulation readiness", "Current_State.md")
    require(current_md, "`PASS`", "Current_State.md")
    if current["correctness"]["strict_phase0"] != "STOP" or current["correctness"]["historical_simulation_readiness"] != "PASS":
        fail("strict STOP / historical simulation PASS invariant violated")
    if topic_map["Strict executable evidence"]["status"] != "STOP_LABELS_5D_0_10D_0":
        fail("strict executable authority status drifted")

    e1 = current["research"]["e1"]
    if (e1["status"] != "COMPLETED_RESEARCH_EXPERIMENT"
            or e1["outcome"] != "PASS_REFRESH_HYPOTHESIS"
            or e1["research_evidence_only"] is not True
            or e1["formal_seeds"] != [17, 29]):
        fail("Current State E1 status drifted")
    if topic_map["E1 experiment specification"]["status"] != "FROZEN_EXPERIMENT_SPECIFICATION":
        fail("Authority Map E1 status drifted")
    result = topic_map["E1 completed result"]
    if (result["status"] != "PASS_REFRESH_HYPOTHESIS"
            or result["authority_class"] != "RESEARCH_EVIDENCE"
            or result["primary_authority"] != e1["result_authority"]):
        fail("E1 completed result authority drifted")
    require(current_md, "COMPLETED_RESEARCH_EXPERIMENT", "Current_State.md")
    require(authority_md, "PASS_REFRESH_HYPOTHESIS", "Authority_Map.md")

    if current["operations"]["deployed"] is not False:
        fail("Phase A unexpectedly represented as deployed")
    require(current_md, "`NOT_DEPLOYED`", "Current_State.md")
    require(operations_md, "not deployed", "Operations.md")


def main() -> int:
    knowledge = ROOT / "knowledge"
    generated = knowledge / "_generated"
    required_json = [
        knowledge / "knowledge-impact-map.json",
        knowledge / "governance-config.json",
        generated / "authority-history.json",
    ]
    for path in required_json:
        json.loads(path.read_text(encoding="utf-8"))
    json_files = sorted(knowledge.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    markdown_links = validate_markdown_links(sorted(knowledge.rglob("*.md")) + [ROOT / "AGENTS.md", ROOT / "CLAUDE.md"])

    generation = subprocess.run(
        [sys.executable, str(TOOLS / "generate_indexes.py"), "--check"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if generation.returncode:
        fail(f"generated indexes drifted: {generation.stdout or generation.stderr}")

    authority = json.loads((knowledge / "00_Project_Map/authority-map.json").read_text(encoding="utf-8"))
    current = json.loads((knowledge / "00_Project_Map/current-state.json").read_text(encoding="utf-8"))
    topics = authority["topics"]

    reference_results = []
    for item in topics:
        primary = item.get("primary_authority")
        parsed = parse_authority_ref(primary) if primary else None
        if parsed:
            result = validate_authority_ref(ROOT, parsed)
            reference_results.append(result)
            if result["status"] == "FAIL":
                fail(f"invalid authority: {primary}: {result['findings']}")

    features_text = (knowledge / "01_Domains/Features.md").read_text(encoding="utf-8")
    current_md = (knowledge / "00_Project_Map/Current_State.md").read_text(encoding="utf-8")
    authority_md = (knowledge / "00_Project_Map/Authority_Map.md").read_text(encoding="utf-8")
    operations_md = (knowledge / "01_Domains/Operations.md").read_text(encoding="utf-8")
    validate_explicit_invariants(authority, current, current_md, authority_md, features_text, operations_md)

    warnings = [
        finding
        for result in reference_results
        for finding in result["findings"]
        if finding["severity"] == "WARNING"
    ]
    runtime_files = sorted((knowledge / "runtime-observations").glob("*.json"))
    if not runtime_files:
        fail("runtime observation store is empty")
    for path in runtime_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        for field in ("observation_id", "topic", "observed_at", "ttl_days", "actor", "method", "evidence_ref", "value"):
            if field not in record:
                fail(f"runtime observation missing {field}: {path.relative_to(ROOT)}")

    history = json.loads((generated / "authority-history.json").read_text(encoding="utf-8"))
    for index, event in enumerate(history["events"]):
        for field in ("topic", "old_authority", "new_authority", "changed_at", "change_commit", "reason", "evidence"):
            if field not in event:
                fail(f"authority history event {index} missing {field}")

    status = "WARNING" if warnings else "PASS"
    print(json.dumps({
        "status": status,
        "authority_refs_checked": len(reference_results),
        "warnings": warnings,
        "explicit_invariants": 5,
        "runtime_observations": len(runtime_files),
        "generated_indexes": "PASS",
        "json_files": len(json_files),
        "markdown_links_checked": markdown_links,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
