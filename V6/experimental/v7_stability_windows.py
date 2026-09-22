"""Leakage-resistant rolling window definitions for V7 stability training."""
from __future__ import annotations

from datetime import date

WINDOWS = (
    ("W2024", 2022, 2023, 2024),
    ("W2025", 2023, 2024, 2025),
    ("W2026", 2024, 2025, 2026),
)


def _year(value):
    return date.fromisoformat(str(value)[:10]).year


def build_window_splits(calendar, *, purge_sessions=30, window_ids=None):
    if purge_sessions < 1:
        raise ValueError("purge_sessions must be positive")
    dates = sorted(str(value)[:10] for value in calendar)
    if len(dates) != len(set(dates)):
        raise ValueError("trading calendar contains duplicates")
    selected = set(window_ids) if window_ids is not None else None
    known = {item[0] for item in WINDOWS}
    if selected is not None and not selected <= known:
        raise ValueError("unknown window id")
    windows = []
    for identifier, train_end, selection_year, evaluation_year in WINDOWS:
        if selected is not None and identifier not in selected:
            continue
        nominal_train = [value for value in dates if 2013 <= _year(value) <= train_end]
        if len(nominal_train) <= purge_sessions:
            raise ValueError(f"{identifier} lacks enough training sessions for purge")
        selection = [value for value in dates if _year(value) == selection_year]
        evaluation = [value for value in dates if _year(value) == evaluation_year]
        if not selection or not evaluation:
            raise ValueError(f"{identifier} lacks selection or evaluation dates")
        train = nominal_train[:-purge_sessions]
        if not max(train) < min(selection) or not max(selection) < min(evaluation):
            raise ValueError(f"{identifier} windows overlap or are unordered")
        windows.append({
            "id": identifier,
            "train_start_year": 2013,
            "train_end_year": train_end,
            "selection_year": selection_year,
            "evaluation_year": evaluation_year,
            "purge_sessions": purge_sessions,
            "train": train,
            "selection": selection,
            "evaluation": evaluation,
            "checkpoint_selection_split": "selection",
            "evaluation_used_for_selection": False,
        })
    return windows
