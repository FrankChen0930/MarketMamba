#!/usr/bin/env python3
"""
Canonical lightweight operational regression runner.

This runner intentionally uses an explicit allowlist. It does not discover
arbitrary tests because this repository also contains live API and data-heavy
diagnostic scripts under V6/scripts.

Exit codes:
  0 = COMPLETE_PASS: all allowlisted regressions ran and passed with no skips
  1 = REAL_FAILURE: at least one allowlisted regression failed
  2 = INCOMPLETE_EVIDENCE: no failures, but at least one core regression skipped
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


COMPLETE_PASS = 0
REAL_FAILURE = 1
INCOMPLETE_EVIDENCE = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "V6" / "scripts"

ALLOWLIST = (
    "test_v62_fetch_only.py",
    "test_v61_legacy_config_isolation.py",
    "test_raw_parquet_append_integrity.py",
)

SKIP_PATTERNS = (
    re.compile(r"\bskipped=([1-9][0-9]*)\b"),
    re.compile(r"\bskipped\s+'"),
    re.compile(r"^s+$", re.MULTILINE),
)


@dataclass(frozen=True)
class RegressionResult:
    script: str
    exit_code: int
    output: str

    @property
    def skipped(self) -> bool:
        return any(pattern.search(self.output) for pattern in SKIP_PATTERNS)

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.skipped


def _run_script(script: str) -> RegressionResult:
    path = SCRIPTS_DIR / script
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return RegressionResult(script=script, exit_code=completed.returncode, output=completed.stdout)


def _print_result(result: RegressionResult) -> None:
    if result.exit_code != 0:
        status = "FAIL"
    elif result.skipped:
        status = "INCOMPLETE"
    else:
        status = "PASS"
    print(f"[{status}] {result.script} exit={result.exit_code}")
    print(result.output.rstrip() or "(no output)")
    print()


def main() -> int:
    print("Market Mamba lightweight operational regressions")
    print("allowlist:")
    for script in ALLOWLIST:
        print(f"  - {script}")
    print()

    results = [_run_script(script) for script in ALLOWLIST]
    for result in results:
        _print_result(result)

    failed = [result.script for result in results if result.exit_code != 0]
    incomplete = [result.script for result in results if result.exit_code == 0 and result.skipped]

    if failed:
        print("overall: REAL_FAILURE")
        print("failed:")
        for script in failed:
            print(f"  - {script}")
        return REAL_FAILURE

    if incomplete:
        print("overall: INCOMPLETE_EVIDENCE")
        print("skipped core regressions:")
        for script in incomplete:
            print(f"  - {script}")
        return INCOMPLETE_EVIDENCE

    print("overall: COMPLETE_PASS")
    return COMPLETE_PASS


if __name__ == "__main__":
    raise SystemExit(main())
