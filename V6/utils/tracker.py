"""
MarketMamba V6 — Model Tracker
================================
Read-side utility for V6/logs/model_tracker.jsonl.

Each line is a JSON record written after every daily inference run:
  date, val_ic, top50, ic_analysis, scanner_summary, inference_duration_sec

Write-side lives in run_daily_inference._append_tracker_entry() to keep
this module free of torch / heavy dependencies.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# model_tracker.jsonl sits in V6/logs/; this file is at V6/utils/tracker.py
TRACKER_PATH = Path(__file__).parent.parent / "logs" / "model_tracker.jsonl"


def load_tracker_history(last_n_days: int = 30) -> list[dict]:
    """
    Read V6/logs/model_tracker.jsonl and return the most recent entries.

    Args:
        last_n_days: only include records whose 'date' is within this many
                     calendar days from today (default 30).

    Returns:
        list of dicts, sorted newest-first, at most last_n_days entries.
        Returns [] if the file does not exist or every line is malformed.
    """
    if not TRACKER_PATH.exists():
        return []

    cutoff = (datetime.today() - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
    records: list[dict] = []

    with open(TRACKER_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("date", "") >= cutoff:
                records.append(rec)

    records.sort(key=lambda r: r.get("date", ""), reverse=True)
    return records
