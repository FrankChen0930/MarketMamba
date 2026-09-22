#!/usr/bin/env python3
"""Parse and validate durable MarketMamba authority references."""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


AUTHORITY_RE = re.compile(
    r"^(?:(?P<branch>[A-Za-z0-9_./-]+)@)?(?P<commit>[0-9a-f]{7,40}):"
    r"(?P<path>[^#]+?)(?:#(?P<fragment>.*))?$"
)


@dataclass(frozen=True)
class AuthorityRef:
    raw: str
    commit: str
    path: str
    branch: str | None = None
    fragment: str | None = None


def parse_authority_ref(value: str) -> AuthorityRef | None:
    match = AUTHORITY_RE.match(value.strip())
    if not match:
        return None
    return AuthorityRef(raw=value, **match.groupdict())


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
    )


def validate_authority_ref(root: Path, ref: AuthorityRef) -> dict:
    result = {**asdict(ref), "status": "PASS", "findings": []}
    resolved = _git(root, "rev-parse", ref.commit, check=False)
    if resolved.returncode != 0:
        result["status"] = "FAIL"
        result["findings"].append({"code": "INVALID_COMMIT", "severity": "ERROR"})
        return result
    commit = resolved.stdout.strip()
    result["resolved_commit"] = commit
    exists = _git(root, "cat-file", "-e", f"{commit}:{ref.path}", check=False)
    if exists.returncode != 0:
        result["status"] = "FAIL"
        result["findings"].append({"code": "MISSING_AUTHORITY_PATH", "severity": "ERROR"})
        return result

    if ref.branch:
        branch = _git(root, "rev-parse", ref.branch, check=False)
        if branch.returncode != 0:
            result["status"] = "WARNING"
            result["findings"].append({"code": "BRANCH_LABEL_MISSING", "severity": "WARNING"})
        elif branch.stdout.strip() != commit:
            result["status"] = "WARNING"
            result["branch_tip"] = branch.stdout.strip()
            result["findings"].append({"code": "BRANCH_LABEL_MOVED", "severity": "WARNING"})

    main_obj = _git(root, "rev-parse", f"main:{ref.path}", check=False)
    source_obj = _git(root, "rev-parse", f"{commit}:{ref.path}", check=False)
    if main_obj.returncode == 0 and source_obj.returncode == 0:
        if main_obj.stdout.strip() == source_obj.stdout.strip() and ref.branch:
            if result["status"] == "PASS":
                result["status"] = "WARNING"
            result["findings"].append({"code": "AUTHORITY_REPIN_CANDIDATE", "severity": "WARNING"})
    return result
