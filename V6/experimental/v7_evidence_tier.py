"""Fail-closed evidence tier for V7 publications (VERIFIED / RESEARCH_ONLY / BLOCKED)."""

from __future__ import annotations

from typing import Any, Mapping

REQUIRED_FOR_VERIFIED = (
    "health_healthy_and_publish_allowed",
    "evidence_class_not_synthetic_or_fixture",
    "out_of_sample_validation",
    "baseline_comparison",
    "pit_audit",
)
# Allowlist: anything not explicitly listed (incl. "demo", "mock", synthetic/fixture) never upgrades.
REAL_EVIDENCE_CLASSES = frozenset({"real_pit_oos", "forward_live_shadow"})


def classify_evidence_tier(
    *,
    health_state: Any = None,
    publish_allowed: Any = None,
    evidence_class: Any = None,
    out_of_sample_validated: Any = None,
    baseline_compared: Any = None,
    pit_audited: Any = None,
) -> dict[str, Any]:
    """Return the tier; only explicit True flags and a known real evidence class upgrade."""
    required = list(REQUIRED_FOR_VERIFIED)
    if health_state == "blocked" or (health_state == "healthy" and publish_allowed is False):
        return {"tier": "BLOCKED", "reasons": ["health_blocked"], "required_for_verified": required}

    reasons: list[str] = []
    if not (health_state == "healthy" and publish_allowed is True):
        reasons.append("health_not_healthy_or_publish_not_allowed")
    cls = evidence_class.strip().lower() if isinstance(evidence_class, str) else ""
    if not cls:
        reasons.append("evidence_class_missing")
    elif cls not in REAL_EVIDENCE_CLASSES:
        reasons.append("evidence_class_not_real_allowlisted")
    for flag, reason in (
        (out_of_sample_validated, "out_of_sample_validation_missing"),
        (baseline_compared, "baseline_comparison_missing"),
        (pit_audited, "pit_audit_missing"),
    ):
        if flag is not True:
            reasons.append(reason)
    if reasons:
        return {"tier": "RESEARCH_ONLY", "reasons": reasons, "required_for_verified": required}
    return {"tier": "VERIFIED", "reasons": [], "required_for_verified": required}


def classify_from_publication(document: Mapping[str, Any]) -> dict[str, Any]:
    """Classify from a health publication; optional `evidence` object carries the flags."""
    evidence = document.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    return classify_evidence_tier(
        health_state=document.get("state"),
        publish_allowed=document.get("publish_allowed"),
        evidence_class=evidence.get("evidence_class"),
        out_of_sample_validated=evidence.get("out_of_sample_validated"),
        baseline_compared=evidence.get("baseline_compared"),
        pit_audited=evidence.get("pit_audited"),
    )
