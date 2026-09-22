from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping


class SourceClass(str, Enum):
    EXCHANGE_REGULAR_BOARD_VERIFIED = "EXCHANGE_REGULAR_BOARD_VERIFIED"
    EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR = "EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR"
    PROVIDER_MIXED_UNKNOWN_SEGMENT = "PROVIDER_MIXED_UNKNOWN_SEGMENT"
    ADJUSTED_PRICE_ONLY = "ADJUSTED_PRICE_ONLY"
    SYNTHETIC_BACKFILLED = "SYNTHETIC/BACKFILLED"
    UNKNOWN = "UNKNOWN"


_ELIGIBILITY = {
    SourceClass.EXCHANGE_REGULAR_BOARD_VERIFIED: (
        True,
        "Direct exchange regular-board observation with verified market-segment semantics.",
    ),
    SourceClass.EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR: (
        True,
        "Derived values anchored to an exchange regular-board observation with clear semantics.",
    ),
    SourceClass.PROVIDER_MIXED_UNKNOWN_SEGMENT: (
        False,
        "Provider observation does not prove that OHLC belongs to the regular-board segment.",
    ),
    SourceClass.ADJUSTED_PRICE_ONLY: (
        False,
        "Adjusted price values lack independently verified regular-board provenance.",
    ),
    SourceClass.SYNTHETIC_BACKFILLED: (
        False,
        "Synthetic or empirically backfilled OHLC cannot establish historical executability.",
    ),
    SourceClass.UNKNOWN: (
        False,
        "Source lineage is unknown and therefore fails closed.",
    ),
}

_PRECEDENCE = (
    SourceClass.EXCHANGE_REGULAR_BOARD_VERIFIED,
    SourceClass.EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR,
    SourceClass.SYNTHETIC_BACKFILLED,
    SourceClass.PROVIDER_MIXED_UNKNOWN_SEGMENT,
    SourceClass.ADJUSTED_PRICE_ONLY,
    SourceClass.UNKNOWN,
)


def classify_source(
    *,
    has_exchange_regular_board: bool,
    has_exchange_derived: bool,
    has_mixed_provider: bool,
    was_empirically_backfilled: bool,
    adjusted_prices: bool,
) -> SourceClass:
    """Classify one OHLCV observation by its strongest evidenced lineage."""
    if has_exchange_regular_board:
        return SourceClass.EXCHANGE_REGULAR_BOARD_VERIFIED
    if has_exchange_derived:
        return SourceClass.EXCHANGE_DERIVED_BUT_SEMANTICS_CLEAR
    if was_empirically_backfilled:
        return SourceClass.SYNTHETIC_BACKFILLED
    if has_mixed_provider:
        return SourceClass.PROVIDER_MIXED_UNKNOWN_SEGMENT
    if adjusted_prices:
        return SourceClass.ADJUSTED_PRICE_ONLY
    return SourceClass.UNKNOWN


def choose_source(candidates: Iterable[SourceClass | str]) -> SourceClass:
    """Select a source deterministically, independent of candidate order."""
    normalized = {SourceClass(candidate) for candidate in candidates}
    for source in _PRECEDENCE:
        if source in normalized:
            return source
    return SourceClass.UNKNOWN


def apply_source_eligibility(
    row: Mapping[str, Any], source: SourceClass | str
) -> dict[str, Any]:
    """Attach provenance and fail closed before frozen proxy classification."""
    source = SourceClass(source)
    eligible, reason = _ELIGIBILITY[source]
    result = dict(row)
    result["source_class"] = source.value
    result["source_eligible"] = eligible
    result["source_reason"] = reason
    if not eligible:
        result["observation_present"] = False
    return result


def source_reliability_matrix() -> dict[str, dict[str, Any]]:
    """Return the canonical, serializable source eligibility contract."""
    return {
        source.value: {"eligible": eligible, "reason": reason}
        for source, (eligible, reason) in _ELIGIBILITY.items()
    }
