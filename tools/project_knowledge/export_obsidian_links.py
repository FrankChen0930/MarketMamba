#!/usr/bin/env python3
"""Read-only discovery of Obsidian notes that may link to canonical knowledge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def discover(vault: Path | None) -> dict:
    if vault is None or not vault.is_dir():
        return {
            "schema_version": 1,
            "status": "SKIPPED",
            "reason": "No readable vault was explicitly supplied.",
            "candidates": [],
            "vault_modified": False,
        }
    candidates = []
    for path in sorted(vault.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "MarketMamba" not in text:
            continue
        canonical = sorted({
            line.split("knowledge/", 1)[1].split()[0].rstrip(")],.;")
            for line in text.splitlines() if "knowledge/" in line
        })
        candidates.append({
            "note": str(path.relative_to(vault)),
            "canonical_paths": [f"knowledge/{item}" for item in canonical],
        })
    return {
        "schema_version": 1,
        "status": "PASS",
        "reason": None,
        "candidates": candidates,
        "vault_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = discover(args.vault)
    if args.write:
        output = ROOT / "knowledge/_generated/obsidian-link-candidates.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
