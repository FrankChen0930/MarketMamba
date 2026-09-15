"""Append-only event journal for the V7 research portfolio engine."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from V6.experimental.v7_portfolio_contract import (
    ContractError,
    CorporateAction,
    MarketQuote,
    PortfolioSpec,
    decimal_text,
    parse_decimal,
    parse_timestamp,
)
from V6.experimental.v7_portfolio_engine import PortfolioEngine


ZERO_HASH = "0" * 64
EVENT_KINDS = {"SIGNAL", "MARKET_SESSION", "CORPORATE_ACTION"}


class JournalIntegrityError(ValueError):
    """Raised when the persisted journal cannot be trusted."""


class EventConflictError(ValueError):
    """Raised when an event ID is reused for different content."""


@dataclass(frozen=True)
class ReplayResult:
    state: dict[str, Any]
    event_count: int


@dataclass(frozen=True)
class AppendResult:
    appended: bool
    state: dict[str, Any]
    event_count: int


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_hash(record_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical(record_without_hash).encode("utf-8")
    ).hexdigest()


def _make_record(
    *,
    seq: int,
    event_id: str,
    kind: str,
    occurred_at: str,
    payload: Mapping[str, Any],
    prev_hash: str,
) -> dict[str, Any]:
    base = {
        "seq": seq,
        "event_id": event_id,
        "kind": kind,
        "occurred_at": occurred_at,
        "payload": dict(payload),
        "prev_hash": prev_hash,
    }
    return {**base, "record_hash": _record_hash(base)}


def _event_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JournalIntegrityError("event_id must be a non-empty string")
    return value.strip()


def _normalize_payload(
    kind: str,
    occurred_at: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise JournalIntegrityError("payload must be a mapping")
    try:
        if kind == "SIGNAL":
            if set(payload) != {"head", "scores"}:
                raise ContractError("SIGNAL payload requires head and scores")
            if not isinstance(payload["scores"], Mapping) or not payload["scores"]:
                raise ContractError("scores must be a non-empty mapping")
            scores = {
                str(ticker).strip(): decimal_text(
                    parse_decimal(score, field=f"score[{ticker}]")
                )
                for ticker, score in payload["scores"].items()
            }
            if any(not ticker for ticker in scores):
                raise ContractError("ticker must be non-empty")
            if len(scores) != len(payload["scores"]):
                raise ContractError("duplicate ticker after normalization")
            return {"head": payload["head"], "scores": scores}
        if kind == "MARKET_SESSION":
            if set(payload) != {"quotes"} or not isinstance(
                payload["quotes"], Mapping
            ):
                raise ContractError("MARKET_SESSION payload requires quotes")
            quotes: dict[str, Any] = {}
            for ticker, raw_quote in payload["quotes"].items():
                quote = (
                    raw_quote
                    if isinstance(raw_quote, MarketQuote)
                    else MarketQuote.from_payload(raw_quote)
                )
                normalized_ticker = str(ticker).strip()
                if not normalized_ticker or normalized_ticker != quote.ticker:
                    raise ContractError("quote key does not match ticker")
                quotes[normalized_ticker] = quote.to_payload()
            return {"quotes": quotes}
        if kind == "CORPORATE_ACTION":
            action_payload = dict(payload)
            action_payload.setdefault("occurred_at", occurred_at)
            action = CorporateAction.from_payload(action_payload)
            if action.occurred_at.isoformat() != occurred_at:
                raise ContractError(
                    "corporate action payload time does not match event time"
                )
            return action.to_payload()
    except ContractError as exc:
        raise JournalIntegrityError(str(exc)) from exc
    raise JournalIntegrityError(f"unknown event kind: {kind}")


def _decode_lines(text: str) -> list[dict[str, Any]]:
    if not text:
        raise JournalIntegrityError("journal is empty")
    if not text.endswith("\n"):
        raise JournalIntegrityError("journal has a truncated final line")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise JournalIntegrityError(f"blank journal line: {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JournalIntegrityError(
                f"invalid JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise JournalIntegrityError(
                f"record on line {line_number} is not an object"
            )
        records.append(record)
    return records


RECORD_KEYS = {
    "seq", "event_id", "kind", "occurred_at",
    "payload", "prev_hash", "record_hash",
}


def _validate_records(records: list[dict[str, Any]]) -> None:
    expected_prev = ZERO_HASH
    previous_time = None
    seen_ids: set[str] = set()

    for expected_seq, record in enumerate(records, start=1):
        if set(record) != RECORD_KEYS:
            raise JournalIntegrityError(
                f"record {expected_seq} has invalid fields"
            )
        if (
            isinstance(record["seq"], bool)
            or record["seq"] != expected_seq
        ):
            raise JournalIntegrityError(
                f"record {expected_seq} has non-contiguous seq"
            )

        event_id = _event_id(record["event_id"])
        if event_id in seen_ids:
            raise JournalIntegrityError(f"duplicate event_id: {event_id}")
        seen_ids.add(event_id)

        kind = record["kind"]
        if expected_seq == 1:
            if kind != "GENESIS":
                raise JournalIntegrityError("first record must be GENESIS")
        elif kind not in EVENT_KINDS:
            raise JournalIntegrityError(
                f"record {expected_seq} has unknown event kind: {kind}"
            )

        if record["prev_hash"] != expected_prev:
            raise JournalIntegrityError(
                f"record {expected_seq} has broken prev_hash"
            )
        base = {key: record[key] for key in RECORD_KEYS - {"record_hash"}}
        actual_hash = _record_hash(base)
        if record["record_hash"] != actual_hash:
            raise JournalIntegrityError(
                f"record {expected_seq} hash mismatch"
            )

        try:
            occurred_at = parse_timestamp(
                record["occurred_at"], field="occurred_at"
            )
            if previous_time is not None and occurred_at < previous_time:
                raise JournalIntegrityError(
                    f"record {expected_seq} time moved backwards"
                )
            if not isinstance(record["payload"], Mapping):
                raise JournalIntegrityError(
                    f"record {expected_seq} payload is not an object"
                )
            if expected_seq == 1:
                normalized_payload = PortfolioSpec.from_payload(
                    record["payload"]
                ).to_payload()
            else:
                normalized_payload = _normalize_payload(
                    kind,
                    occurred_at.isoformat(),
                    record["payload"],
                )
        except ContractError as exc:
            raise JournalIntegrityError(
                f"record {expected_seq} contract error: {exc}"
            ) from exc
        if normalized_payload != record["payload"]:
            raise JournalIntegrityError(
                f"record {expected_seq} payload is not canonical"
            )

        previous_time = occurred_at
        expected_prev = record["record_hash"]


class PortfolioJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def create(
        cls,
        path: str | Path,
        spec: PortfolioSpec,
        event_id: str,
        occurred_at: Any,
    ) -> "PortfolioJournal":
        journal = cls(path)
        journal.path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = parse_timestamp(
            occurred_at, field="occurred_at"
        ).isoformat()
        record = _make_record(
            seq=1,
            event_id=_event_id(event_id),
            kind="GENESIS",
            occurred_at=timestamp,
            payload=spec.to_payload(),
            prev_hash=ZERO_HASH,
        )
        try:
            with journal.path.open("x", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(_canonical(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise JournalIntegrityError(
                f"journal already exists: {journal.path}"
            ) from None
        return journal

    def records(self) -> tuple[dict[str, Any], ...]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise JournalIntegrityError(
                f"cannot read journal: {exc}"
            ) from exc
        records = _decode_lines(text)
        _validate_records(records)
        return tuple(records)

    def replay(self) -> ReplayResult:
        records = self.records()
        first = records[0]
        if first.get("kind") != "GENESIS":
            raise JournalIntegrityError("first record must be GENESIS")
        try:
            spec = PortfolioSpec.from_payload(first["payload"])
            engine = PortfolioEngine(spec)
            for record in records[1:]:
                engine.apply_event(
                    record["kind"],
                    record["event_id"],
                    record["occurred_at"],
                    record["payload"],
                )
        except (KeyError, ContractError) as exc:
            raise JournalIntegrityError(
                f"cannot replay journal: {exc}"
            ) from exc
        return ReplayResult(
            state=engine.snapshot(),
            event_count=len(records),
        )

    def append(
        self,
        event_id: str,
        kind: str,
        occurred_at: Any,
        payload: Mapping[str, Any],
    ) -> AppendResult:
        normalized_id = _event_id(event_id)
        normalized_kind = str(kind).strip()
        if normalized_kind not in EVENT_KINDS:
            raise JournalIntegrityError(
                f"unknown event kind: {normalized_kind}"
            )
        timestamp = parse_timestamp(
            occurred_at, field="occurred_at"
        ).isoformat()
        normalized_payload = _normalize_payload(
            normalized_kind, timestamp, payload
        )

        try:
            with self.path.open("r+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                text = handle.read()
                records = _decode_lines(text)
                _validate_records(records)
                for record in records:
                    if record["event_id"] != normalized_id:
                        continue
                    if (
                        record["kind"] == normalized_kind
                        and record["occurred_at"] == timestamp
                        and record["payload"] == normalized_payload
                    ):
                        replayed = self.replay()
                        return AppendResult(
                            appended=False,
                            state=replayed.state,
                            event_count=replayed.event_count,
                        )
                    raise EventConflictError(
                        f"event_id {normalized_id!r} has different content"
                    )

                if parse_timestamp(timestamp) < parse_timestamp(
                    records[-1]["occurred_at"]
                ):
                    raise JournalIntegrityError(
                        "event time cannot move backwards"
                    )

                record = _make_record(
                    seq=len(records) + 1,
                    event_id=normalized_id,
                    kind=normalized_kind,
                    occurred_at=timestamp,
                    payload=normalized_payload,
                    prev_hash=records[-1]["record_hash"],
                )
                handle.seek(0, os.SEEK_END)
                handle.write(_canonical(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileNotFoundError:
            raise JournalIntegrityError(
                f"journal does not exist: {self.path}"
            ) from None

        replayed = self.replay()
        return AppendResult(
            appended=True,
            state=replayed.state,
            event_count=replayed.event_count,
        )

def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="只讀驗證並重放 V7 組合 journal"
    )
    parser.add_argument("--journal", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        result = PortfolioJournal(args.journal).replay()
    except JournalIntegrityError as exc:
        print(f"日誌驗證失敗：{exc}", file=sys.stderr)
        return 2

    state = result.state
    print(f"事件數：{result.event_count}")
    print(f"最後 session：{state['last_session_id'] or '尚無'}")
    print(f"淨值：{state['net_value']}")
    print(f"現金：{state['cash']}")
    print(f"持股數：{len(state['positions'])}")
    print(f"累計成本：{state['total_cost']}")
    print(f"待成交：{'是' if state['pending'] is not None else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
