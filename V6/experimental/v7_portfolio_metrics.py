"""Reproducible performance metrics over the Decimal-exact V7 portfolio engine.

Fixture/research tooling only. No annualised ratios, no profit claims.

Definitions (all Decimal, computed at 50 digits, ROUND_HALF_EVEN):
- series: net_value after each recorded market session.
- total_return = last / first - 1 over the session series.
- turnover_t = (buy_notional_t + sell_notional_t) / 2 / prior net value, where
  prior net value is initial_net_value for session 0 and the previous
  recorded net value afterwards. Notional counts FILLED and PARTIAL fills
  (BLOCKED fills carry zero notional).
- cost_drag = cumulative engine total_cost / initial_net_value.
- max_drawdown = max over t of (peak_to_date - v_t) / peak_to_date, reported
  as a non-negative fraction with peak and trough timestamps.
- benchmark: excess_return = total_return - benchmark_return;
  tracking_difference = mean per-session (strategy return - benchmark return).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from typing import Any, Mapping, Sequence

from V6.experimental.v7_portfolio_contract import (
    ContractError,
    decimal_text,
    parse_decimal,
    parse_timestamp,
)

METRICS_VERSION = "v7-portfolio-metrics-v1"
DEFAULT_EVIDENCE_CLASS = "SYNTHETIC_OR_FIXTURE_ONLY_NOT_INVESTMENT_EVIDENCE"
ZERO = Decimal("0")
ONE = Decimal("1")
METRICS_CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)
FILL_STATUSES = ("FILLED", "PARTIAL", "BLOCKED")


@dataclass(frozen=True)
class SessionRecord:
    occurred_at: datetime
    net_value: Decimal
    total_cost: Decimal
    buy_notional: Decimal
    sell_notional: Decimal
    fill_counts: tuple[tuple[str, int], ...]


class SessionRecorder:
    """Collects one SessionRecord per apply_market_session call."""

    def __init__(self, initial_net_value: Any = ONE):
        self.initial_net_value = parse_decimal(
            initial_net_value, field="initial_net_value", minimum=ZERO,
            minimum_inclusive=False,
        )
        self.records: list[SessionRecord] = []

    def record(self, engine: Any, session_result: Any) -> SessionRecord:
        counts = {status: 0 for status in FILL_STATUSES}
        buy = ZERO
        sell = ZERO
        with localcontext(METRICS_CONTEXT):
            for fill in session_result.fills:
                if fill.status not in counts:
                    raise ContractError(f"unknown fill status: {fill.status}")
                counts[fill.status] += 1
                if fill.status == "BLOCKED":
                    continue
                if fill.side == "BUY":
                    buy += fill.gross_notional
                elif fill.side == "SELL":
                    sell += fill.gross_notional
                else:
                    raise ContractError(f"unknown fill side: {fill.side}")
            item = SessionRecord(
                occurred_at=parse_timestamp(session_result.occurred_at),
                net_value=engine.net_value(),
                total_cost=engine.total_cost,
                buy_notional=buy,
                sell_notional=sell,
                fill_counts=tuple(counts.items()),
            )
        self.records.append(item)
        return item

    def compute(self, benchmark=None, evidence_class=None) -> dict[str, Any]:
        return compute_performance(
            self.records,
            benchmark=benchmark,
            evidence_class=evidence_class,
            initial_net_value=self.initial_net_value,
        )


def _validate(records: Sequence[SessionRecord]) -> None:
    if len(records) < 2:
        raise ContractError("performance metrics require at least 2 records")
    previous = None
    for item in records:
        if not isinstance(item, SessionRecord):
            raise ContractError("records must be SessionRecord instances")
        if item.net_value <= ZERO:
            raise ContractError("net_value must be positive")
        if previous is not None and item.occurred_at <= previous:
            raise ContractError("record timestamps must be strictly increasing")
        previous = item.occurred_at


def _benchmark_levels(
    benchmark: Mapping[Any, Any], records: Sequence[SessionRecord]
) -> list[Decimal]:
    if not isinstance(benchmark, Mapping):
        raise ContractError("benchmark must be a mapping")
    parsed: dict[datetime, Decimal] = {}
    for key, value in benchmark.items():
        stamp = parse_timestamp(key, field="benchmark timestamp")
        if stamp in parsed:
            raise ContractError("duplicate benchmark timestamp")
        parsed[stamp] = parse_decimal(
            value, field="benchmark level", minimum=ZERO, minimum_inclusive=False
        )
    expected = {item.occurred_at for item in records}
    if set(parsed) != expected:
        raise ContractError("benchmark timestamps must align exactly to sessions")
    return [parsed[item.occurred_at] for item in records]


def compute_performance(
    records: Sequence[SessionRecord],
    benchmark: Mapping[Any, Any] | None = None,
    *,
    evidence_class: str | None = None,
    initial_net_value: Any = ONE,
) -> dict[str, Any]:
    records = tuple(records)
    _validate(records)
    initial = parse_decimal(
        initial_net_value, field="initial_net_value", minimum=ZERO,
        minimum_inclusive=False,
    )
    if evidence_class is not None and (
        not isinstance(evidence_class, str) or not evidence_class.strip()
    ):
        raise ContractError("evidence_class must be a non-empty string")
    levels = _benchmark_levels(benchmark, records) if benchmark is not None else None

    with localcontext(METRICS_CONTEXT):
        values = [item.net_value for item in records]
        total_return = values[-1] / values[0] - ONE

        peak = values[0]
        peak_at = records[0].occurred_at
        best_peak_at = peak_at
        best_trough_at = peak_at
        max_dd = ZERO
        for item in records:
            if item.net_value > peak:
                peak = item.net_value
                peak_at = item.occurred_at
            drawdown = (peak - item.net_value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                best_peak_at = peak_at
                best_trough_at = item.occurred_at

        turnovers: list[Decimal] = []
        sessions = []
        prior = initial
        for item in records:
            turnover = (item.buy_notional + item.sell_notional) / Decimal(2) / prior
            turnovers.append(turnover)
            sessions.append({
                "occurred_at": item.occurred_at.isoformat(),
                "net_value": decimal_text(item.net_value),
                "buy_notional": decimal_text(item.buy_notional),
                "sell_notional": decimal_text(item.sell_notional),
                "turnover": decimal_text(turnover),
            })
            prior = item.net_value
        mean_turnover = sum(turnovers, start=ZERO) / Decimal(len(turnovers))

        cumulative_cost = records[-1].total_cost
        counts = {status: 0 for status in FILL_STATUSES}
        for item in records:
            for status, number in item.fill_counts:
                counts[status] += number

        payload: dict[str, Any] = {
            "metrics_version": METRICS_VERSION,
            "evidence_class": evidence_class or DEFAULT_EVIDENCE_CLASS,
            "session_count": len(records),
            "start": records[0].occurred_at.isoformat(),
            "end": records[-1].occurred_at.isoformat(),
            "initial_net_value": decimal_text(initial),
            "first_net_value": decimal_text(values[0]),
            "final_net_value": decimal_text(values[-1]),
            "total_return": decimal_text(total_return),
            "max_drawdown": {
                "fraction": decimal_text(max_dd),
                "peak_at": best_peak_at.isoformat(),
                "trough_at": best_trough_at.isoformat(),
            },
            "mean_turnover": decimal_text(mean_turnover),
            "cumulative_cost": decimal_text(cumulative_cost),
            "cost_drag": decimal_text(cumulative_cost / initial),
            "fill_counts": {
                "filled": counts["FILLED"],
                "partial": counts["PARTIAL"],
                "blocked": counts["BLOCKED"],
            },
            "sessions": sessions,
            "benchmark": None,
        }
        if levels is not None:
            bench_return = levels[-1] / levels[0] - ONE
            diffs = [
                (values[i] / values[i - 1] - ONE) - (levels[i] / levels[i - 1] - ONE)
                for i in range(1, len(values))
            ]
            payload["benchmark"] = {
                "benchmark_return": decimal_text(bench_return),
                "excess_return": decimal_text(total_return - bench_return),
                "tracking_difference": decimal_text(
                    sum(diffs, start=ZERO) / Decimal(len(diffs))
                ),
            }

    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "canonical_sha256"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
