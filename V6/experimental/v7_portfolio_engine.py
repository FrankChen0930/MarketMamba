"""Deterministic state machine for the V7 normalized research portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable, Mapping

from V6.experimental.v7_portfolio_contract import (
    ContractError,
    PortfolioSpec,
    decimal_text,
    parse_decimal,
    parse_timestamp,
)


@dataclass
class Position:
    quantity: Decimal
    last_price: Decimal

    def to_payload(self) -> dict[str, str]:
        return {
            "quantity": decimal_text(self.quantity),
            "last_price": decimal_text(self.last_price),
        }


@dataclass(frozen=True)
class PendingTarget:
    signal_id: str
    as_of: datetime
    target_tickers: tuple[str, ...]
    created_session: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "as_of": self.as_of.isoformat(),
            "target_tickers": list(self.target_tickers),
            "created_session": self.created_session,
        }


@dataclass(frozen=True)
class SignalResult:
    event_id: str
    due: bool
    target_tickers: tuple[str, ...]
    superseded_signal_id: str | None = None


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value.strip()


def _normalized_scores(scores: Mapping[str, Any]) -> dict[str, Decimal]:
    if not isinstance(scores, Mapping) or not scores:
        raise ContractError("scores must be a non-empty mapping")
    normalized: dict[str, Decimal] = {}
    for raw_ticker, raw_score in scores.items():
        ticker = _text(raw_ticker, field="ticker")
        if ticker in normalized:
            raise ContractError(f"duplicate ticker after normalization: {ticker}")
        normalized[ticker] = parse_decimal(raw_score, field=f"score[{ticker}]")
    return normalized


def _ranked_tickers(scores: Mapping[str, Any]) -> tuple[str, ...]:
    normalized = _normalized_scores(scores)
    return tuple(
        ticker
        for ticker, _ in sorted(
            normalized.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def select_target_tickers(
    spec: PortfolioSpec,
    scores: Mapping[str, Any],
    *,
    current_holdings: Iterable[str],
) -> tuple[str, ...]:
    """Apply the frozen banding rule with a stable lexical tie-break."""
    ranked = _ranked_tickers(scores)
    rank = {ticker: index for index, ticker in enumerate(ranked, start=1)}
    threshold = int(spec.buffer_multiple * spec.holdings_count)

    normalized_holdings: list[str] = []
    seen: set[str] = set()
    for raw_ticker in current_holdings:
        ticker = _text(raw_ticker, field="current_holding")
        if ticker in seen:
            raise ContractError(f"duplicate current holding: {ticker}")
        seen.add(ticker)
        normalized_holdings.append(ticker)

    keep = sorted(
        (
            ticker
            for ticker in normalized_holdings
            if rank.get(ticker, threshold + 1) <= threshold
        ),
        key=rank.__getitem__,
    )[:spec.holdings_count]

    target = list(keep)
    for ticker in ranked[:spec.holdings_count]:
        if ticker not in target:
            target.append(ticker)
        if len(target) == spec.holdings_count:
            break
    return tuple(target)


class PortfolioEngine:
    """In-memory reducer. Persistence and event idempotency live in the journal."""

    def __init__(self, spec: PortfolioSpec):
        if not isinstance(spec, PortfolioSpec):
            raise ContractError("spec must be a PortfolioSpec")
        self.spec = spec
        self.cash = Decimal("1")
        self.positions: dict[str, Position] = {}
        self.pending: PendingTarget | None = None
        self.session_count = 0
        self.last_rebalance_session: int | None = None
        self.last_session_id: str | None = None
        self.total_cost = Decimal("0")

    def _signal_due(self) -> bool:
        if self.pending is not None:
            return (
                self.session_count - self.pending.created_session
                >= self.spec.rebalance_every_sessions
            )
        if self.last_rebalance_session is None:
            return True
        return (
            self.session_count - self.last_rebalance_session
            >= self.spec.rebalance_every_sessions
        )

    def apply_signal(
        self,
        event_id: str,
        as_of: datetime | str,
        head: str,
        scores: Mapping[str, Any],
    ) -> SignalResult:
        normalized_event_id = _text(event_id, field="event_id")
        signal_time = parse_timestamp(as_of, field="as_of")
        normalized_head = _text(head, field="head")
        if normalized_head != self.spec.head:
            raise ContractError(
                f"signal head {normalized_head!r} does not match "
                f"spec head {self.spec.head!r}"
            )
        _normalized_scores(scores)

        if not self._signal_due():
            assert self.pending is not None
            return SignalResult(
                event_id=normalized_event_id,
                due=False,
                target_tickers=self.pending.target_tickers,
            )

        target = select_target_tickers(
            self.spec,
            scores,
            current_holdings=self.positions,
        )
        superseded = self.pending.signal_id if self.pending is not None else None
        self.pending = PendingTarget(
            signal_id=normalized_event_id,
            as_of=signal_time,
            target_tickers=target,
            created_session=self.session_count,
        )
        return SignalResult(
            event_id=normalized_event_id,
            due=True,
            target_tickers=target,
            superseded_signal_id=superseded,
        )

    def net_value(self) -> Decimal:
        return self.cash + sum(
            (
                position.quantity * position.last_price
                for position in self.positions.values()
            ),
            start=Decimal("0"),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "engine_version": self.spec.engine_version,
            "spec": self.spec.to_payload(),
            "cash": decimal_text(self.cash),
            "positions": {
                ticker: self.positions[ticker].to_payload()
                for ticker in sorted(self.positions)
            },
            "pending": self.pending.to_payload() if self.pending else None,
            "session_count": self.session_count,
            "last_rebalance_session": self.last_rebalance_session,
            "last_session_id": self.last_session_id,
            "total_cost": decimal_text(self.total_cost),
            "net_value": decimal_text(self.net_value()),
        }
