"""Fail-closed correctness gate for the V7 signal-to-portfolio program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "v7-correctness-audit-v1"
REQUIRED_DOMAINS = (
    "label_execution_alignment",
    "feature_point_in_time",
    "historical_universe",
    "preprocessing_leakage",
)
VALID_STATUSES = {"PASS", "FAIL", "UNKNOWN"}
VALID_SEVERITIES = {"INFO", "MINOR", "MAJOR", "CRITICAL"}
BLOCKING_SEVERITIES = {"MAJOR", "CRITICAL"}


class AuditValidationError(ValueError):
    """Raised when evidence is incomplete or violates the audit contract."""


@dataclass(frozen=True)
class AuditDecision:
    status: str
    blocking_findings: tuple[Mapping[str, Any], ...]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditValidationError(f"{field} must be non-empty text")
    return value.strip()


def _validate_check(raw: Any, domain: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AuditValidationError(f"{domain} checks must be objects")
    required = {
        "id", "status", "severity", "finding", "evidence", "affected_scope",
        "requires_rebuild", "minimum_fix",
    }
    if set(raw) != required:
        raise AuditValidationError(
            f"{domain} check fields must be exactly {sorted(required)}"
        )
    check = dict(raw)
    for field in ("id", "finding", "affected_scope", "minimum_fix"):
        check[field] = _text(check[field], f"check.{field}")
    if check["status"] not in VALID_STATUSES:
        raise AuditValidationError(f"invalid status for {check['id']}")
    if check["severity"] not in VALID_SEVERITIES:
        raise AuditValidationError(f"invalid severity for {check['id']}")
    if not isinstance(check["requires_rebuild"], bool):
        raise AuditValidationError(f"requires_rebuild must be boolean for {check['id']}")
    if not isinstance(check["evidence"], list) or not check["evidence"]:
        raise AuditValidationError(f"evidence must be non-empty for {check['id']}")
    for item in check["evidence"]:
        if not isinstance(item, Mapping) or set(item) != {
            "path", "lines", "observation"
        }:
            raise AuditValidationError(f"invalid evidence entry for {check['id']}")
        for field in ("path", "lines", "observation"):
            _text(item[field], f"evidence.{field}")
    return check


def evaluate_audit(document: Any) -> AuditDecision:
    if not isinstance(document, Mapping):
        raise AuditValidationError("audit document must be an object")
    required = {"schema_version", "baseline", "timing", "domains"}
    if set(document) != required:
        raise AuditValidationError(
            f"audit document fields must be exactly {sorted(required)}"
        )
    if document["schema_version"] != SCHEMA_VERSION:
        raise AuditValidationError("unsupported schema_version")
    _text(document["baseline"], "baseline")

    timing_fields = {
        "feature_information_cutoff",
        "prediction_timestamp",
        "earliest_execution",
        "label_start",
        "label_end",
    }
    timing = document["timing"]
    if not isinstance(timing, Mapping) or set(timing) != timing_fields:
        raise AuditValidationError("timing must contain the five required fields")
    for field in timing_fields:
        _text(timing[field], f"timing.{field}")

    domains = document["domains"]
    if not isinstance(domains, Mapping) or set(domains) != set(REQUIRED_DOMAINS):
        raise AuditValidationError("domains must exactly match required audit domains")

    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for domain in REQUIRED_DOMAINS:
        raw_checks = domains[domain]
        if not isinstance(raw_checks, list) or not raw_checks:
            raise AuditValidationError(f"{domain} must contain at least one check")
        for raw in raw_checks:
            check = _validate_check(raw, domain)
            if check["id"] in seen:
                raise AuditValidationError(f"duplicate check id: {check['id']}")
            seen.add(check["id"])
            checks.append({"domain": domain, **check})

    blockers = tuple(
        check for check in checks
        if check["severity"] in BLOCKING_SEVERITIES
        and check["status"] in {"FAIL", "UNKNOWN"}
    )
    return AuditDecision("STOP" if blockers else "PASS", blockers)


def render_markdown(document: Mapping[str, Any], decision: AuditDecision) -> str:
    timing = document["timing"]
    lines = [
        "# V7 Phase 0 Correctness Audit",
        "",
        f"Gate decision: {decision.status}",
        "",
        "## Timing chain",
        "",
        " -> ".join(
            (
                timing["feature_information_cutoff"],
                timing["prediction_timestamp"],
                timing["earliest_execution"],
                timing["label_start"],
                timing["label_end"],
            )
        ),
        "",
        "## Findings",
        "",
    ]
    for domain in REQUIRED_DOMAINS:
        lines.extend((f"### {domain}", ""))
        for check in document["domains"][domain]:
            lines.extend(
                (
                    f"- [{check['status']}/{check['severity']}] `{check['id']}`: {check['finding']}",
                    f"  - Affected scope: {check['affected_scope']}",
                    f"  - Requires rebuild: {'yes' if check['requires_rebuild'] else 'no'}",
                    f"  - Minimum fix: {check['minimum_fix']}",
                )
            )
            for evidence in check["evidence"]:
                lines.append(
                    f"  - Evidence: {evidence['path']}:{evidence['lines']} - "
                    f"{evidence['observation']}"
                )
        lines.append("")
    if decision.blocking_findings:
        lines.extend(("## Blocking findings", ""))
        for check in decision.blocking_findings:
            lines.append(
                f"- `{check['id']}` ({check['severity']}): {check['minimum_fix']}"
            )
    else:
        lines.extend(("## Baseline designation", "", "V7 Baseline v1"))
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: "Path", content: str) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)


def write_outputs(
    document: Mapping[str, Any],
    json_path: "Path",
    markdown_path: "Path",
) -> AuditDecision:
    import json

    decision = evaluate_audit(document)
    payload = {
        "schema_version": document["schema_version"],
        "baseline": document["baseline"],
        "decision": decision.status,
        "timing": document["timing"],
        "domains": document["domains"],
        "blocking_findings": list(decision.blocking_findings),
    }
    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, render_markdown(document, decision))
    return decision


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run the V7 Phase 0 correctness gate.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        decision = write_outputs(document, args.json_output, args.markdown_output)
    except (OSError, json.JSONDecodeError, AuditValidationError) as error:
        parser.exit(1, f"V7 correctness audit invalid: {error}\n")
    print(f"V7 correctness audit: {decision.status}")
    return 0 if decision.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
