"""Deterministic state machine for the V7 normalized research portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from functools import wraps
from typing import Any, Iterable, Mapping

from V6.experimental.v7_portfolio_contract import (
    ContractError,
    CorporateAction,
    MarketQuote,
    PortfolioSpec,
    TradeStatus,
    decimal_text,
    parse_decimal,
    parse_timestamp,
)


ZERO = Decimal("0")
ONE = Decimal("1")
ENGINE_DECIMAL_CONTEXT = Context(prec=50, rounding=ROUND_HALF_EVEN)


def _fixed_decimal_context(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with localcontext(ENGINE_DECIMAL_CONTEXT):
            return function(*args, **kwargs)
    return wrapped


def _multiply_exact(left: Decimal, right: Decimal) -> Decimal:
    precision = max(
        ENGINE_DECIMAL_CONTEXT.prec,
        len(left.as_tuple().digits) + len(right.as_tuple().digits) + 2,
    )
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        return left * right


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


@dataclass(frozen=True)
class TradeFill:
    side: str
    ticker: str
    requested_quantity: Decimal
    filled_quantity: Decimal
    gross_notional: Decimal
    fee: Decimal
    status: str
    reason: str


@dataclass(frozen=True)
class SessionResult:
    event_id: str
    occurred_at: datetime
    fills: tuple[TradeFill, ...]


@dataclass(frozen=True)
class CorporateActionResult:
    event_id: str
    ticker: str
    status: str
    cash_added: Decimal


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


@_fixed_decimal_context
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


def _normalize_quotes(
    quotes: Mapping[str, MarketQuote | Mapping[str, Any]],
) -> dict[str, MarketQuote]:
    if not isinstance(quotes, Mapping):
        raise ContractError("quotes must be a mapping")
    normalized: dict[str, MarketQuote] = {}
    for raw_ticker, raw_quote in quotes.items():
        ticker = _text(raw_ticker, field="quote ticker")
        quote = (
            raw_quote
            if isinstance(raw_quote, MarketQuote)
            else MarketQuote.from_payload(raw_quote)
        )
        if quote.ticker != ticker:
            raise ContractError(
                f"quote key {ticker!r} does not match ticker {quote.ticker!r}"
            )
        if ticker in normalized:
            raise ContractError(f"duplicate quote ticker: {ticker}")
        normalized[ticker] = quote
    return normalized


def _buy_allowed(status: TradeStatus) -> bool:
    return status in {TradeStatus.OPEN, TradeStatus.SELL_BLOCKED}


def _sell_allowed(status: TradeStatus) -> bool:
    return status in {TradeStatus.OPEN, TradeStatus.BUY_BLOCKED}


class PortfolioEngine:
    """In-memory reducer. Persistence and event idempotency live in the journal."""

    def __init__(self, spec: PortfolioSpec):
        if not isinstance(spec, PortfolioSpec):
            raise ContractError("spec must be a PortfolioSpec")
        self.spec = spec
        self.cash = ONE
        self.positions: dict[str, Position] = {}
        self.pending: PendingTarget | None = None
        self.session_count = 0
        self.last_rebalance_session: int | None = None
        self.last_session_id: str | None = None
        self.total_cost = ZERO

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

    def _blocked_fill(
        self,
        *,
        side: str,
        ticker: str,
        requested: Decimal,
        reason: str,
    ) -> TradeFill:
        return TradeFill(
            side=side,
            ticker=ticker,
            requested_quantity=requested,
            filled_quantity=ZERO,
            gross_notional=ZERO,
            fee=ZERO,
            status="BLOCKED",
            reason=reason,
        )

    def _sell(
        self,
        ticker: str,
        requested: Decimal,
        quote: MarketQuote | None,
    ) -> TradeFill:
        if quote is None:
            return self._blocked_fill(
                side="SELL",
                ticker=ticker,
                requested=requested,
                reason="MISSING_QUOTE",
            )
        if not _sell_allowed(quote.status):
            return self._blocked_fill(
                side="SELL",
                ticker=ticker,
                requested=requested,
                reason=f"SELL_NOT_ALLOWED_{quote.status.value}",
            )

        filled = min(requested, requested * quote.sell_fill_ratio)
        gross = filled * quote.price
        fee = gross * self.spec.sell_cost_rate
        if filled > ZERO:
            position = self.positions[ticker]
            position.quantity -= filled
            position.last_price = quote.price
            self.cash += gross - fee
            self.total_cost += fee
            if position.quantity == ZERO:
                del self.positions[ticker]

        if filled == requested:
            status, reason = "FILLED", "COMPLETE"
        else:
            status, reason = "PARTIAL", "FILL_RATIO_LIMIT"
        return TradeFill(
            side="SELL",
            ticker=ticker,
            requested_quantity=requested,
            filled_quantity=filled,
            gross_notional=gross,
            fee=fee,
            status=status,
            reason=reason,
        )

    def _buy(
        self,
        ticker: str,
        requested: Decimal,
        quote: MarketQuote | None,
    ) -> TradeFill:
        if quote is None:
            return self._blocked_fill(
                side="BUY",
                ticker=ticker,
                requested=requested,
                reason="MISSING_QUOTE",
            )
        if not _buy_allowed(quote.status):
            return self._blocked_fill(
                side="BUY",
                ticker=ticker,
                requested=requested,
                reason=f"BUY_NOT_ALLOWED_{quote.status.value}",
            )

        ratio_limited = requested * quote.buy_fill_ratio
        cash_capacity = self.cash / (
            quote.price * (ONE + self.spec.buy_cost_rate)
        )
        filled = min(requested, ratio_limited, cash_capacity)
        gross = filled * quote.price
        fee = gross * self.spec.buy_cost_rate
        while gross + fee > self.cash and filled > ZERO:
            filled = filled.next_minus()
            gross = filled * quote.price
            fee = gross * self.spec.buy_cost_rate
        if filled > ZERO:
            self.cash -= gross + fee
            if self.cash < ZERO:
                raise ArithmeticError("cash became negative")
            position = self.positions.get(ticker)
            if position is None:
                self.positions[ticker] = Position(filled, quote.price)
            else:
                position.quantity += filled
                position.last_price = quote.price
            self.total_cost += fee

        if filled == requested:
            status, reason = "FILLED", "COMPLETE"
        elif ratio_limited < requested and ratio_limited <= cash_capacity:
            status, reason = "PARTIAL", "FILL_RATIO_LIMIT"
        else:
            status, reason = "PARTIAL", "INSUFFICIENT_CASH"
        return TradeFill(
            side="BUY",
            ticker=ticker,
            requested_quantity=requested,
            filled_quantity=filled,
            gross_notional=gross,
            fee=fee,
            status=status,
            reason=reason,
        )

    @_fixed_decimal_context
    def apply_market_session(
        self,
        event_id: str,
        occurred_at: datetime | str,
        quotes: Mapping[str, MarketQuote | Mapping[str, Any]],
    ) -> SessionResult:
        normalized_event_id = _text(event_id, field="event_id")
        session_time = parse_timestamp(occurred_at)
        normalized_quotes = _normalize_quotes(quotes)

        self.session_count += 1
        self.last_session_id = normalized_event_id
        for ticker, position in self.positions.items():
            if ticker in normalized_quotes:
                position.last_price = normalized_quotes[ticker].price

        if self.pending is None or session_time <= self.pending.as_of:
            return SessionResult(normalized_event_id, session_time, ())

        targets = self.pending.target_tickers
        target_count = len(targets)
        if target_count == 0:
            raise ContractError("pending target cannot be empty")
        target_set = set(targets)
        provisional_each = self.net_value() / (
            Decimal(target_count) * (ONE + self.spec.buy_cost_rate)
        )
        fills: list[TradeFill] = []

        for ticker in sorted(tuple(self.positions)):
            position = self.positions[ticker]
            quote = normalized_quotes.get(ticker)
            desired = ZERO
            if ticker in target_set:
                if quote is None:
                    continue
                desired = provisional_each / quote.price
            requested = max(ZERO, position.quantity - desired)
            if requested > ZERO:
                fills.append(self._sell(ticker, requested, quote))

        final_each = self.net_value() / (
            Decimal(target_count) * (ONE + self.spec.buy_cost_rate)
        )
        for ticker in targets:
            quote = normalized_quotes.get(ticker)
            current = self.positions.get(ticker)
            if quote is None:
                fills.append(self._blocked_fill(
                    side="BUY",
                    ticker=ticker,
                    requested=ZERO,
                    reason="MISSING_QUOTE",
                ))
                continue
            current_quantity = current.quantity if current else ZERO
            requested = max(ZERO, final_each / quote.price - current_quantity)
            if requested > ZERO:
                fills.append(self._buy(ticker, requested, quote))

        incomplete = any(fill.status != "FILLED" for fill in fills)
        incomplete = incomplete or any(
            ticker not in target_set for ticker in self.positions
        )
        incomplete = incomplete or any(
            ticker not in self.positions for ticker in targets
        )
        if not incomplete:
            self.pending = None
            self.last_rebalance_session = self.session_count

        return SessionResult(
            normalized_event_id,
            session_time,
            tuple(fills),
        )

    @_fixed_decimal_context
    def apply_corporate_action(
        self,
        event_id: str,
        action: CorporateAction | Mapping[str, Any],
    ) -> CorporateActionResult:
        normalized_event_id = _text(event_id, field="event_id")
        normalized_action = (
            action
            if isinstance(action, CorporateAction)
            else CorporateAction.from_payload(action)
        )
        position = self.positions.get(normalized_action.ticker)
        if position is None:
            return CorporateActionResult(
                event_id=normalized_event_id,
                ticker=normalized_action.ticker,
                status="NO_POSITION",
                cash_added=ZERO,
            )

        old_quantity = position.quantity
        cash_added = _multiply_exact(
            old_quantity, normalized_action.cash_per_old_share
        )
        self.cash += cash_added
        position.quantity = _multiply_exact(
            old_quantity, normalized_action.quantity_multiplier
        )
        position.last_price = normalized_action.post_action_price
        return CorporateActionResult(
            event_id=normalized_event_id,
            ticker=normalized_action.ticker,
            status="APPLIED",
            cash_added=cash_added,
        )

    def apply_event(
        self,
        kind: str,
        event_id: str,
        occurred_at: datetime | str,
        payload: Mapping[str, Any],
    ) -> SignalResult | SessionResult | CorporateActionResult:
        normalized_kind = _text(kind, field="kind")
        if not isinstance(payload, Mapping):
            raise ContractError("payload must be a mapping")
        event_time = parse_timestamp(occurred_at)

        if normalized_kind == "SIGNAL":
            if set(payload) != {"head", "scores"}:
                raise ContractError("SIGNAL payload requires head and scores")
            return self.apply_signal(
                event_id,
                event_time,
                payload["head"],
                payload["scores"],
            )
        if normalized_kind == "MARKET_SESSION":
            if set(payload) != {"quotes"}:
                raise ContractError("MARKET_SESSION payload requires quotes")
            return self.apply_market_session(
                event_id,
                event_time,
                payload["quotes"],
            )
        if normalized_kind == "CORPORATE_ACTION":
            action_payload = dict(payload)
            action_payload.setdefault("occurred_at", event_time.isoformat())
            action = CorporateAction.from_payload(action_payload)
            if action.occurred_at != event_time:
                raise ContractError(
                    "corporate action payload time does not match event time"
                )
            return self.apply_corporate_action(event_id, action)
        raise ContractError(f"unknown event kind: {normalized_kind}")

    @_fixed_decimal_context
    def position_value(self) -> Decimal:
        return sum(
            (
                position.quantity * position.last_price
                for position in self.positions.values()
            ),
            start=ZERO,
        )

    @_fixed_decimal_context
    def net_value(self) -> Decimal:
        return self.cash + self.position_value()

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
