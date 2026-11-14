"""Executable-return labels for an after-close signal and next-session entry."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from V6.experimental.v7_temporal_contract import ContractError, canonical_manifest, normalize_calendar


SCHEMA_VERSION = "v7-executable-labels-v1"


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def build_executable_labels(
    price_rows: Iterable[Mapping[str, Any]],
    calendar: Sequence[str],
    *,
    horizons: Sequence[int] = (5, 10),
    return_manifest: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    days = normalize_calendar(calendar)
    requested = tuple(int(value) for value in horizons)
    if not requested or any(value < 1 for value in requested) or len(set(requested)) != len(requested):
        raise ContractError("horizons must be unique positive integers")
    by_stock: dict[str, dict[str, dict[str, Any]]] = {}
    for raw in price_rows:
        row = dict(raw)
        stock_id, day = str(row.get("stock_id", "")).strip(), str(row.get("Date", ""))
        if not stock_id or day not in days:
            raise ContractError("price row has an invalid stock/date key")
        stock = by_stock.setdefault(stock_id, {})
        if day in stock:
            raise ContractError(f"duplicate price row for {stock_id} {day}")
        stock[day] = row

    output: list[dict[str, Any]] = []
    for stock_id in sorted(by_stock):
        stock = by_stock[stock_id]
        for signal_index, signal_date in enumerate(days):
            result: dict[str, Any] = {"signal_date": signal_date, "stock_id": stock_id}
            signal = stock.get(signal_date)
            signal_valid = bool(signal and signal.get("observation_valid") is True)
            for horizon in requested:
                suffix = f"{horizon}d"
                entry_index = signal_index + 1
                exit_index = entry_index + horizon - 1
                entry_day = days[entry_index] if entry_index < len(days) else None
                exit_day = days[exit_index] if exit_index < len(days) else None
                result[f"entry_session_{suffix}"] = entry_day
                result[f"exit_session_{suffix}"] = exit_day
                result[f"entry_price_{suffix}"] = None
                result[f"exit_price_{suffix}"] = None
                label = math.nan
                if not signal_valid:
                    status = "SIGNAL_OBSERVATION_INVALID"
                elif exit_day is None:
                    status = "INSUFFICIENT_FUTURE_SESSIONS"
                else:
                    holding_days = days[entry_index : exit_index + 1]
                    holding = [stock.get(day) for day in holding_days]
                    entry, exit_row = holding[0], holding[-1]
                    if entry is None or entry.get("open_executable") is not True:
                        status = (
                            "ENTRY_EXECUTABILITY_UNKNOWN"
                            if entry is not None and "open_executable" not in entry
                            else "ENTRY_NOT_EXECUTABLE"
                        )
                    elif not _finite_positive(entry.get("Open")):
                        status = "ENTRY_PRICE_INVALID"
                    elif any(
                        row is None
                        or row.get("observation_valid") is not True
                        or row.get("suspended") is True
                        for row in holding
                    ):
                        status = "INVALID_HOLDING_INTERVAL"
                    elif not _finite_positive(exit_row.get("Close")):
                        status = "EXIT_PRICE_INVALID"
                    else:
                        entry_price, exit_price = float(entry["Open"]), float(exit_row["Close"])
                        result[f"entry_price_{suffix}"] = entry_price
                        result[f"exit_price_{suffix}"] = exit_price
                        label = exit_price / entry_price - 1.0
                        status = "VALID"
                result[f"Alpha_{suffix}"] = label
                result[f"label_status_{suffix}"] = status
            output.append(result)

    manifest = canonical_manifest(
        SCHEMA_VERSION,
        {
            "signal_timing": "session t after close",
            "entry_rule": "next calendar session open; never silently rolled",
            "holding_session_counting": "entry session is 1; exit is holding session h",
            "label_formula": "exit_close / entry_open - 1",
            "horizons": list(requested),
            "invalid_interval_policy": "unknown label",
            "suspension_policy": "any declared suspension in holding interval invalidates label",
        },
        {"calendar_sha256": __import__("hashlib").sha256("\n".join(days).encode()).hexdigest()},
    )
    manifest["label_formula"] = manifest["parameters"]["label_formula"]
    manifest["holding_session_counting"] = manifest["parameters"]["holding_session_counting"]
    manifest["invalidates"] = [
        "legacy Close[t+h]/Close[t] labels",
        "checkpoints trained on legacy labels",
        "predictions, IC, and portfolio claims derived from those checkpoints",
    ]
    return (output, manifest) if return_manifest else output
