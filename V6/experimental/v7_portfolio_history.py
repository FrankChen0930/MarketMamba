"""Deterministic, fixture-only views of one locked portfolio journal prefix.

The fixture manifest is an explicit operator declaration, not research admission.
Consumers validate shape/identity/consistency; the exporter alone replays accounting.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from V6.experimental.v7_portfolio_contract import PortfolioSpec, ContractError, parse_timestamp, parse_decimal, decimal_text
from V6.experimental.v7_portfolio_engine import PortfolioEngine
from V6.experimental.v7_portfolio_journal import PortfolioJournal, JournalIntegrityError, _unique_json_object

SCHEMA = "v7-portfolio-history-v1"
EVIDENCE = "OFFLINE_SYNTHETIC_ENGINEERING"

class HistoryError(ValueError):
    """Missing, corrupt, conflicting or unapproved history input."""

def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=lambda _: (_ for _ in ()).throw(HistoryError("non-finite JSON")))

def _require(condition, message):
    if not condition:
        raise HistoryError(message)

def _keys(value, expected):
    _require(isinstance(value, dict) and set(value) == set(expected.split()), "invalid fields")

def _hex(value, size):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{%d}" % size, value) is not None

def _text(value):
    return isinstance(value, str) and bool(value) and value == value.strip()

def _integer(value):
    return type(value) is int and value >= 0

def fixture_manifest(journal, fixture_id):
    """Declare a caller-created synthetic fixture; never invoke on research data."""
    raw, _ = PortfolioJournal(journal).read_prefix()
    _require(_text(fixture_id), "invalid fixture_id")
    return {"schema": "v7-portfolio-fixture-v1", "evidence_class": EVIDENCE,
            "fixture_id": fixture_id, "journal_sha256": digest(raw)}

def _state(state, spec):
    _keys(state, "engine_version spec cash positions pending session_count last_rebalance_session last_session_id total_cost net_value")
    _require(state["spec"] == spec and state["engine_version"] == spec["engine_version"], "state spec mismatch")
    def decimal(value):
        _require(isinstance(value, str) and decimal_text(parse_decimal(value, field="state")) == value, "invalid decimal")
    for field in ("cash", "total_cost", "net_value"):
        decimal(state[field])
    _require(_integer(state["session_count"]), "invalid session count")
    _require(state["last_rebalance_session"] is None or (_integer(state["last_rebalance_session"]) and state["last_rebalance_session"] <= state["session_count"]), "invalid rebalance session")
    _require(state["last_session_id"] is None or _text(state["last_session_id"]), "invalid last session")
    _require(isinstance(state["positions"], dict), "invalid positions")
    for ticker, position in state["positions"].items():
        _require(_text(ticker), "invalid ticker")
        _keys(position, "quantity last_price")
        decimal(position["quantity"]); decimal(position["last_price"])
    pending = state["pending"]
    if pending is not None:
        _keys(pending, "signal_id as_of created_session target_tickers")
        _require(_text(pending["signal_id"]) and _integer(pending["created_session"]), "invalid pending")
        parse_timestamp(pending["as_of"])
        targets = pending["target_tickers"]
        _require(isinstance(targets, list) and all(_text(t) for t in targets) and len(set(targets)) == len(targets), "invalid targets")

def validate_history(history):
    try:
        _keys(history, "schema evidence_class strict_phase0 research_admission source spec events final_state")
        _require(history["schema"] == SCHEMA and history["evidence_class"] == EVIDENCE and history["strict_phase0"] == "STOP" and history["research_admission"] is False, "unapproved evidence")
        source = history["source"]
        _keys(source, "source_commit engine_version journal_sha256 record_hash event_count fixture_id fixture_manifest_sha256")
        _require(_hex(source["source_commit"], 40) and all(_hex(source[k], 64) for k in ("journal_sha256", "record_hash", "fixture_manifest_sha256")) and _text(source["fixture_id"]), "invalid source identity")
        spec = PortfolioSpec.from_payload(history["spec"]).to_payload()
        _require(spec == history["spec"] and source["engine_version"] == spec["engine_version"], "invalid spec identity")
        events = history["events"]
        _require(isinstance(events, list) and len(events) > 0 and _integer(source["event_count"]) and source["event_count"] == len(events), "invalid event count")
        ids = set(); previous = None
        for seq, event in enumerate(events, 1):
            _keys(event, "seq event_id kind occurred_at state")
            _require(type(event["seq"]) is int and event["seq"] == seq and _text(event["event_id"]) and event["event_id"] not in ids, "invalid event identity")
            ids.add(event["event_id"])
            _require(event["kind"] in ({"GENESIS"} if seq == 1 else {"SIGNAL", "MARKET_SESSION", "CORPORATE_ACTION"}), "invalid event kind")
            when = parse_timestamp(event["occurred_at"])
            _require(previous is None or when >= previous, "event time moved backwards")
            previous = when
            _state(event["state"], spec)
        _state(history["final_state"], spec)
        _require(history["final_state"] == events[-1]["state"], "final state mismatch")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise HistoryError(f"invalid history: {exc}") from exc
    return history

def read_history(path, expected_sha256=None):
    try:
        raw = Path(path).read_bytes()
        if expected_sha256 is not None:
            _require(_hex(expected_sha256, 64) and digest(raw) == expected_sha256, "history identity mismatch")
        return validate_history(_json(raw))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise HistoryError(f"cannot read history: {exc}") from exc

def export_history(journal, output, manifest, source_commit):
    temporary = None
    try:
        journal, output, manifest = map(Path, (journal, output, manifest))
        _require(_hex(source_commit, 40), "invalid source commit")
        for source in (journal, manifest):
            _require(output.resolve() != source.resolve() and not (output.exists() and source.exists() and os.path.samefile(output, source)), "output aliases input")
        # Only task-local artifacts or temporary outputs; never formal data/models.
        allowed = (Path(__file__).resolve().parents[2] / ".artifacts", Path(tempfile.gettempdir()))
        _require(any(output.resolve().is_relative_to(root.resolve()) for root in allowed), "output must be in checkout .artifacts or temporary directory")
        raw, records = PortfolioJournal(journal).read_prefix()
        manifest_raw = manifest.read_bytes()
        declaration = _json(manifest_raw)
        _keys(declaration, "schema evidence_class fixture_id journal_sha256")
        _require(declaration["schema"] == "v7-portfolio-fixture-v1" and declaration["evidence_class"] == EVIDENCE and _text(declaration["fixture_id"]) and declaration["journal_sha256"] == digest(raw), "fixture identity mismatch")
        spec = PortfolioSpec.from_payload(records[0]["payload"])
        engine = PortfolioEngine(spec)
        events = []
        for record in records:
            if record["seq"] > 1:
                engine.apply_event(record["kind"], record["event_id"], record["occurred_at"], record["payload"])
            events.append({**{key: record[key] for key in ("seq", "event_id", "kind", "occurred_at")}, "state": engine.snapshot()})
        history = validate_history({"schema": SCHEMA, "evidence_class": EVIDENCE, "strict_phase0": "STOP", "research_admission": False,
            "source": {"source_commit": source_commit, "engine_version": spec.engine_version, "journal_sha256": digest(raw), "record_hash": records[-1]["record_hash"], "event_count": len(records), "fixture_id": declaration["fixture_id"], "fixture_manifest_sha256": digest(manifest_raw)},
            "spec": spec.to_payload(), "events": events, "final_state": engine.snapshot()})
        payload = canonical(history)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output)
        return history
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise HistoryError(f"cannot export history: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export synthetic fixture portfolio history (no admission)")
    for name in ("journal", "output", "fixture-manifest", "source-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        history = export_history(args.journal, args.output, args.fixture_manifest, args.source_commit)
    except HistoryError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps({"event_count": history["source"]["event_count"], "history_sha256": digest(Path(args.output).read_bytes())}))

if __name__ == "__main__": main()
