"""Strict, versioned value objects for the V7 research portfolio engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping


ENGINE_VERSION = "v7-portfolio-v1"


class ContractError(ValueError):
    """Raised when an event or portfolio value violates the frozen contract."""


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    BUY_BLOCKED = "BUY_BLOCKED"
    SELL_BLOCKED = "SELL_BLOCKED"
    HALTED = "HALTED"
    UNKNOWN = "UNKNOWN"


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def parse_decimal(
    value: Any,
    *,
    field: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ContractError(f"{field} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ContractError(f"{field} must be a finite decimal") from None
    if not parsed.is_finite():
        raise ContractError(f"{field} must be a finite decimal")
    if minimum is not None:
        invalid = parsed < minimum if minimum_inclusive else parsed <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ContractError(f"{field} must be {operator} {minimum}")
    if maximum is not None:
        invalid = parsed > maximum if maximum_inclusive else parsed >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ContractError(f"{field} must be {operator} {maximum}")
    return parsed


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def parse_timestamp(value: Any, *, field: str = "occurred_at") -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise ContractError(f"{field} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{field} must include a timezone offset")
    return parsed


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    object_name: str,
) -> None:
    missing = required - set(payload)
    extra = set(payload) - allowed
    if missing:
        raise ContractError(f"{object_name} missing fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"{object_name} unknown fields: {sorted(extra)}")


@dataclass(frozen=True)
class PortfolioSpec:
    holdings_count: int
    buffer_multiple: Decimal
    rebalance_every_sessions: int
    head: str
    buy_cost_rate: Decimal = Decimal("0.0015")
    sell_cost_rate: Decimal = Decimal("0.0045")
    engine_version: str = ENGINE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "holdings_count",
            _positive_int(self.holdings_count, field="holdings_count"),
        )
        object.__setattr__(
            self, "buffer_multiple",
            parse_decimal(
                self.buffer_multiple,
                field="buffer_multiple",
                minimum=Decimal("1"),
            ),
        )
        object.__setattr__(
            self, "rebalance_every_sessions",
            _positive_int(
                self.rebalance_every_sessions,
                field="rebalance_every_sessions",
            ),
        )
        object.__setattr__(self, "head", _require_text(self.head, field="head"))
        object.__setattr__(
            self, "buy_cost_rate",
            parse_decimal(
                self.buy_cost_rate,
                field="buy_cost_rate",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
                maximum_inclusive=False,
            ),
        )
        object.__setattr__(
            self, "sell_cost_rate",
            parse_decimal(
                self.sell_cost_rate,
                field="sell_cost_rate",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
                maximum_inclusive=False,
            ),
        )
        if self.engine_version != ENGINE_VERSION:
            raise ContractError(
                f"unsupported engine_version: {self.engine_version!r}"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "holdings_count": self.holdings_count,
            "buffer_multiple": decimal_text(self.buffer_multiple),
            "rebalance_every_sessions": self.rebalance_every_sessions,
            "head": self.head,
            "buy_cost_rate": decimal_text(self.buy_cost_rate),
            "sell_cost_rate": decimal_text(self.sell_cost_rate),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PortfolioSpec":
        allowed = {
            "engine_version", "holdings_count", "buffer_multiple",
            "rebalance_every_sessions", "head", "buy_cost_rate",
            "sell_cost_rate",
        }
        required = {
            "holdings_count", "buffer_multiple",
            "rebalance_every_sessions", "head",
        }
        _reject_unknown_keys(
            payload, allowed=allowed, required=required, object_name="spec"
        )
        return cls(
            holdings_count=payload["holdings_count"],
            buffer_multiple=payload["buffer_multiple"],
            rebalance_every_sessions=payload["rebalance_every_sessions"],
            head=payload["head"],
            buy_cost_rate=payload.get("buy_cost_rate", Decimal("0.0015")),
            sell_cost_rate=payload.get("sell_cost_rate", Decimal("0.0045")),
            engine_version=payload.get("engine_version", ENGINE_VERSION),
        )


@dataclass(frozen=True)
class MarketQuote:
    ticker: str
    price: Decimal
    buy_fill_ratio: Decimal
    sell_fill_ratio: Decimal
    status: TradeStatus = TradeStatus.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", _require_text(self.ticker, field="ticker"))
        object.__setattr__(
            self, "price",
            parse_decimal(
                self.price,
                field="price",
                minimum=Decimal("0"),
                minimum_inclusive=False,
            ),
        )
        for field_name in ("buy_fill_ratio", "sell_fill_ratio"):
            object.__setattr__(
                self,
                field_name,
                parse_decimal(
                    getattr(self, field_name),
                    field=field_name,
                    minimum=Decimal("0"),
                    maximum=Decimal("1"),
                ),
            )
        try:
            status = (
                self.status
                if isinstance(self.status, TradeStatus)
                else TradeStatus(self.status)
            )
        except ValueError:
            raise ContractError(f"unknown trade status: {self.status!r}") from None
        object.__setattr__(self, "status", status)

    def to_payload(self) -> dict[str, str]:
        return {
            "ticker": self.ticker,
            "price": decimal_text(self.price),
            "status": self.status.value,
            "buy_fill_ratio": decimal_text(self.buy_fill_ratio),
            "sell_fill_ratio": decimal_text(self.sell_fill_ratio),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MarketQuote":
        allowed = {
            "ticker", "price", "status",
            "buy_fill_ratio", "sell_fill_ratio",
        }
        required = {"ticker", "price", "buy_fill_ratio", "sell_fill_ratio"}
        _reject_unknown_keys(
            payload, allowed=allowed, required=required, object_name="quote"
        )
        return cls(
            ticker=payload["ticker"],
            price=payload["price"],
            status=payload.get("status", TradeStatus.UNKNOWN),
            buy_fill_ratio=payload["buy_fill_ratio"],
            sell_fill_ratio=payload["sell_fill_ratio"],
        )


@dataclass(frozen=True)
class CorporateAction:
    ticker: str
    occurred_at: datetime
    quantity_multiplier: Decimal
    cash_per_old_share: Decimal
    post_action_price: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", _require_text(self.ticker, field="ticker"))
        object.__setattr__(
            self, "occurred_at",
            parse_timestamp(self.occurred_at, field="occurred_at"),
        )
        object.__setattr__(
            self, "quantity_multiplier",
            parse_decimal(
                self.quantity_multiplier,
                field="quantity_multiplier",
                minimum=Decimal("0"),
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self, "cash_per_old_share",
            parse_decimal(
                self.cash_per_old_share,
                field="cash_per_old_share",
                minimum=Decimal("0"),
            ),
        )
        object.__setattr__(
            self, "post_action_price",
            parse_decimal(
                self.post_action_price,
                field="post_action_price",
                minimum=Decimal("0"),
                minimum_inclusive=False,
            ),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "ticker": self.ticker,
            "occurred_at": self.occurred_at.isoformat(),
            "quantity_multiplier": decimal_text(self.quantity_multiplier),
            "cash_per_old_share": decimal_text(self.cash_per_old_share),
            "post_action_price": decimal_text(self.post_action_price),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CorporateAction":
        allowed = {
            "ticker", "occurred_at", "quantity_multiplier",
            "cash_per_old_share", "post_action_price",
        }
        _reject_unknown_keys(
            payload,
            allowed=allowed,
            required=allowed,
            object_name="corporate_action",
        )
        return cls(
            ticker=payload["ticker"],
            occurred_at=payload["occurred_at"],
            quantity_multiplier=payload["quantity_multiplier"],
            cash_per_old_share=payload["cash_per_old_share"],
            post_action_price=payload["post_action_price"],
        )
