"""Synthetic F2 summary boundary. No production source policy or scheduler.

Source policy below is a frozen SYNTHETIC fixture, never a real source admission.
Private diagnostics never enter source projection or the publication timetable.
"""
from datetime import date, datetime, time, timedelta
from contextvars import ContextVar
import hashlib
import re
import secrets
from zoneinfo import ZoneInfo

from tools.forward_wp4 import Stop, canonical

SOURCE = 'SYNTHETIC_SOURCE'
FIELDS = ('OHLCV', 'base_features', 'lifecycle')
SOURCE_CODES = ('SOURCE_MISSING', 'SOURCE_COVERAGE', 'SOURCE_UNIVERSE',
                'SOURCE_INVALID', 'SOURCE_OPTIONAL')
PRIVATE_CODES = ('RETRAIN_TRANSIENT', 'RETRAIN_IDENTITY', 'PRIVATE_MARK_EXECUTION',
                 'BLIND_INTEGRITY', 'PRIVATE_EXPECTED', 'UNKNOWN_ERROR')
ATTEMPT_ACTIONS = ('WAIT', 'RETRY', 'STOP', 'INVALID_FOR_CONFIRMATION', 'SWITCH_ALL_THREE')
HEX = re.compile(r'[0-9a-f]{64}')
CONTRACT = hashlib.sha256(canonical(dict(schema='forward-public-summary-v1',
    sources=[SOURCE], field_classes=FIELDS, errors=SOURCE_CODES,
    synthetic=True, cadence='next_calendar_day_12:00_Asia/Taipei'))).hexdigest()
SOURCE_KEYS = {'schema_version', 'session', 'public_contract_sha256', 'source_files',
               'universe_rows', 'source_error_counts', 'missing_source_sessions'}
CHAIN_KEYS = {'previous_head', 'commitment', 'head'}
_ATTEMPT_INVENTORY = ContextVar('forward_attempt_inventory', default=None)


def register_attempt(journal):
    """Bind generated attempt evidence before the caller can discard or rename it.

    Direct journal callers retain their own store contract. A synthetic wrapper
    additionally requires every observed request/event in its sealed inventory.
    """
    from tools import forward_wp4 as w
    active = _ATTEMPT_INVENTORY.get()
    if active is None: return
    private, inventory = active
    paths = [journal/'request.json'] + list((journal/'events').iterdir())
    for path in paths:
        w.private_path(path)
        try: name = str(path.relative_to(private))
        except ValueError as exc: raise Stop('attempt outside private inventory') from exc
        info = w.file_info(path)
        if name in inventory and inventory[name] != info:
            raise Stop('attempt inventory changed')
        inventory[name] = info


def iso_day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise Stop('summary date')
    return value


def count(value):
    return type(value) is int and value >= 0


def validate_source(record):
    try:
        if set(record) != SOURCE_KEYS: raise ValueError('closed schema')
        iso_day(record['session'])
        if record['schema_version'] != 'forward-public-summary-v1' or record['public_contract_sha256'] != CONTRACT:
            raise ValueError('public contract')
        for name in ('source_files', 'source_error_counts', 'missing_source_sessions'):
            if set(record[name]) != {SOURCE}: raise ValueError('source enum')
        slots = record['source_files'][SOURCE]
        if set(slots) != {'expected', 'received', 'valid'} or not all(count(v) for v in slots.values()):
            raise ValueError('slots')
        if not slots['valid'] <= slots['received'] <= slots['expected']: raise ValueError('slots order')
        if set(record['universe_rows']) != set(FIELDS): raise ValueError('field enum')
        for values in record['universe_rows'].values():
            if set(values) != {'expected', 'valid'}: raise ValueError('coverage keys')
            if values == {'expected': None, 'valid': None}: continue
            if not all(count(v) for v in values.values()) or values['valid'] > values['expected']:
                raise ValueError('coverage counts')
        denominators = [v['expected'] for v in record['universe_rows'].values()]
        if any(v != denominators[0] for v in denominators): raise ValueError('universe denominator')
        errors = record['source_error_counts'][SOURCE]
        if set(errors) != set(SOURCE_CODES) or not all(count(v) and v <= slots['expected'] for v in errors.values()):
            raise ValueError('source errors')
        if any(v['expected'] is None for v in record['universe_rows'].values()) and errors['SOURCE_UNIVERSE'] == 0:
            raise ValueError('unknown universe')
        missing = record['missing_source_sessions'][SOURCE]
        if not isinstance(missing, list) or missing != sorted(set(missing)):
            raise ValueError('missing sessions')
        if any(iso_day(d) > record['session'] for d in missing): raise ValueError('future missing day')
        if (record['session'] in missing) != (slots['expected'] > 0 and slots['received'] == 0):
            raise ValueError('missing source projection')
    except (ValueError, KeyError, TypeError) as exc:
        raise Stop('unsafe public source summary') from exc
    return record


def coverage_ratio(values):
    return None if values['expected'] in (None, 0) else values['valid'] / values['expected']


def commitment(private_manifest, nonce):
    if not isinstance(nonce, bytes) or len(nonce) != 32: raise Stop('private nonce')
    return hashlib.sha256(b'MM-FWD-PRIVATE-v1\0' + nonce + canonical(private_manifest)).hexdigest()


def seal(source_record, private_manifest, previous_head='0' * 64):
    """Caller keeps returned nonce PRIVATE; never included in the public record."""
    validate_source(source_record)
    if not isinstance(previous_head, str) or not HEX.fullmatch(previous_head): raise Stop('previous head')
    nonce = secrets.token_bytes(32)
    record = dict(source_record, previous_head=previous_head,
                  commitment=commitment(private_manifest, nonce))
    record['head'] = hashlib.sha256(b'MM-FWD-PUBLIC-v1\0' + canonical(record)).hexdigest()
    validate_public(record)
    return record, nonce


def validate_public(record):
    if set(record) != SOURCE_KEYS | CHAIN_KEYS: raise Stop('public unknown fields')
    validate_source({k: record[k] for k in SOURCE_KEYS})
    if any(not isinstance(record[k], str) or not HEX.fullmatch(record[k]) for k in CHAIN_KEYS):
        raise Stop('public chain format')
    expected = hashlib.sha256(b'MM-FWD-PUBLIC-v1\0' + canonical({k: v for k, v in record.items() if k != 'head'})).hexdigest()
    if record['head'] != expected: raise Stop('public chain hash')
    return record


def verify_chain(records, expected_sessions, external_head):
    """Explicit frozen source sessions; missing days cannot disappear by intersection."""
    if [r['session'] for r in records] != list(expected_sessions): raise Stop('chain missing sessions')
    previous = '0' * 64
    for record in records:
        validate_public(record)
        if record['previous_head'] != previous: raise Stop('chain link')
        previous = record['head']
    if previous != external_head: raise Stop('external head')
    return True


def release_at(session):
    day = date.fromisoformat(iso_day(session)) + timedelta(days=1)
    return datetime.combine(day, time(12), ZoneInfo('Asia/Taipei'))


def release_bytes(record, at):
    """Synthetic clock gate, no timer/job. Private success is deliberately absent."""
    validate_public(record)
    if not isinstance(at, datetime) or at.tzinfo is None or at != release_at(record['session']):
        raise Stop('fixed release instant')
    return canonical(record)


def disposition(codes, phase):
    """Private incident routing; never return its result via the public schema."""
    if phase not in ('PRESTART', 'FORMAL'): raise Stop('unknown phase')
    if not codes or any(c not in SOURCE_CODES + PRIVATE_CODES for c in codes):
        codes = ['UNKNOWN_ERROR']
    fatal = set(codes) - {'SOURCE_OPTIONAL', 'PRIVATE_EXPECTED', 'RETRAIN_TRANSIENT'}
    if fatal: return 'STOP' if phase == 'PRESTART' else 'INVALID_FOR_CONFIRMATION'
    if 'RETRAIN_TRANSIENT' in codes: return 'RETRY_SAME_SNAPSHOT_R_PLUS_2_LIMIT'
    return 'NO_EFFECT'


def synthetic_source(session='2026-11-26', missing=False, unknown_lifecycle=False):
    """S-held/S-unheld and M-held/M-unheld share the exact same source inputs."""
    errors = dict.fromkeys(SOURCE_CODES, 0)
    errors['SOURCE_COVERAGE'] = int(missing)
    errors['SOURCE_UNIVERSE'] = int(unknown_lifecycle)
    return validate_source(dict(schema_version='forward-public-summary-v1', session=session,
        public_contract_sha256=CONTRACT,
        source_files={SOURCE: dict(expected=1, received=1, valid=1)},
        universe_rows={f: dict(expected=None, valid=None) if unknown_lifecycle else
                       dict(expected=100, valid=(100 if f == 'lifecycle' else 99 - int(missing))) for f in FIELDS},
        source_error_counts={SOURCE: errors}, missing_source_sessions={SOURCE: []}))


def private_operation_status(private, files=None):
    """Derive status from committed receipts and actual hash-linked journals.

    The manifest's descriptive incident field is never a decision authority.
    Same-user commitments detect changes relative to the retained external head;
    they do not provide OS isolation or authenticated signatures.
    """
    import json
    from tools import forward_wp4 as w
    actual = {str(p.relative_to(private)): w.file_info(p) for p in private.rglob('*')
              if p.is_file() and p not in (private/'private-manifest.json',private/'nonce.bin')}
    if files is None:
        files = json.loads((private/'private-manifest.json').read_bytes())['files']
    if actual != files: raise Stop('private inventory mismatch')
    receipt = json.loads((private/'operation-receipt.json').read_bytes())
    if set(receipt) != {'event','exception','attempt_files'}: raise Stop('operation receipt schema')
    inventory = receipt['attempt_files']
    if any(name not in files or files[name] != info for name,info in inventory.items()):
        raise Stop('missing/renamed attempt inventory')
    # Inspect the COMPLETE committed file set, independently of directory names.
    # An unregistered attempt cannot become an ordinary opaque output by renaming.
    for name in files:
        try:
            value = json.loads((private/name).read_bytes())
        except (ValueError, UnicodeError):
            continue
        is_attempt = isinstance(value,dict) and (
            value.get('schema') == 'forward-wp5-synthetic-attempt-v1'
            or {'decision','previous_event_sha256','request_sha256'} <= set(value)
            or isinstance(value.get('request'),dict) and
            value['request'].get('schema') == 'forward-wp5-synthetic-request-v1')
        if is_attempt and name not in inventory:
            raise Stop('unregistered attempt record')
    status = 'UNKNOWN_ERROR' if receipt['exception'] else 'PRIVATE_EXPECTED'
    returned = receipt['event']
    if returned is not None:
        status = returned['decision']['action']
    if status not in ('PRIVATE_EXPECTED', 'UNKNOWN_ERROR') + ATTEMPT_ACTIONS:
        raise Stop('unknown operation status')
    matched = False
    terminal_statuses = []
    plans = [name for name in inventory if name.endswith('/request.json')]
    covered = set(plans)
    for name in sorted(plans):
        plan_path = private/name
        events = plan_path.parent/'events'
        plan = json.loads(plan_path.read_bytes())
        previous = '0'*64; actions = []
        for i,path in enumerate(sorted(events.iterdir())):
            relative = str(path.relative_to(private))
            if relative not in inventory: raise Stop('unregistered journal event')
            covered.add(relative)
            event = json.loads(path.read_bytes()); decision = event['decision']
            if (path.name != f'{i:03}.json' or event['previous_event_sha256'] != previous
                    or event['request_sha256'] != hashlib.sha256(canonical(plan)).hexdigest()
                    or decision['origin'] != plan['request']['origin']
                    or decision['snapshot_sha256'] != plan['request']['snapshot_sha256']
                    or decision['delay_sessions'] != i
                    or decision['action'] not in ATTEMPT_ACTIONS
                    or (actions and actions[-1] != 'RETRY')):
                raise Stop('invalid attempt journal')
            actions.append(decision['action']); previous = hashlib.sha256(canonical(event)).hexdigest()
            matched = matched or event == returned
        if not actions and status != 'WAIT': raise Stop('empty attempt journal')
        if actions: terminal_statuses.append(actions[-1])
    if covered != set(inventory): raise Stop('orphan attempt inventory')
    if returned is not None and status != 'WAIT' and not matched:
        raise Stop('operation return not bound to journal')
    # P2 §訓練視窗、資料與封存 / RETRAIN_TRANSIENT: only RETRY is
    # recoverable; a valid terminal switch resolves it, never STOP/INVALID.
    if receipt['exception']: return 'UNKNOWN_ERROR'
    for terminal in ('INVALID_FOR_CONFIRMATION','STOP','RETRY','WAIT'):
        if terminal in terminal_statuses: return terminal
    if terminal_statuses: return 'SWITCH_ALL_THREE'
    return status


def run_synthetic_session(out, *, private_operation=None, source=None, previous_head='0'*64,
                          previous_private_commitment='0'*64, at=None):
    """Synthetic Human-runner interface; writes BOTH sides only under forward.

    This is a boundary rehearsal, NOT OS isolation or a deployable blind runner.
    No holdings/performance inputs are accepted. Source snapshot is captured
    before the private operation; private errors cannot change its projection.
    """
    from contextlib import redirect_stdout, redirect_stderr
    import json
    import os
    import sys
    from tools import forward_wp4 as w
    out = w.private_path(out)
    if not HEX.fullmatch(previous_private_commitment): raise Stop('private previous commitment')
    snapshot = canonical(validate_source(source if source is not None else synthetic_source()))
    source = json.loads(snapshot)
    if at is None: at = release_at(source['session'])  # SYNTHETIC clock only
    # Reject wrong clock before any private computation. No waiting/scheduling.
    if at != release_at(source['session']): raise Stop('synthetic release clock')
    out.mkdir(parents=True, exist_ok=False)
    private = out / 'private'; private.mkdir()
    w.write_sync(private / 'source-before-fit.json', snapshot)
    outcome = 'PRIVATE_EXPECTED'
    returned = None; exception = None
    operation = private_operation or (lambda directory: w.synthetic_run(directory / 'store'))
    # Python AND native library output stays private. The wrapper is deliberately
    # single-process/single-caller, not a concurrent stdout redirection service.
    sys.stdout.flush(); sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    inventory = {}
    token = _ATTEMPT_INVENTORY.set((private, inventory))
    try:
        with (private / 'runner.log').open('x') as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    result = operation(private)
                    if isinstance(result, dict) and 'decision' in result:
                        returned = result
                        outcome = result['decision']['action']
                        if outcome not in ATTEMPT_ACTIONS: outcome = 'UNKNOWN_ERROR'
                except Exception as exc:
                    exception = type(exc).__name__
                    outcome = 'UNKNOWN_ERROR'
                log.flush(); os.fsync(log.fileno())
    finally:
        _ATTEMPT_INVENTORY.reset(token)
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1])
    if any(p.is_symlink() for p in private.rglob('*')): raise Stop('private output symlink')
    w.write_sync(private/'operation-receipt.json', canonical(dict(
        event=returned,exception=exception,attempt_files=inventory)))
    files = {str(p.relative_to(private)): w.file_info(p) for p in private.rglob('*') if p.is_file()}
    outcome = private_operation_status(private, files)
    manifest = dict(schema='forward-wp4-synthetic-private-v1', synthetic=True,
        previous_private_commitment=previous_private_commitment, incident=outcome,
        files=files)
    public, nonce = seal(source, manifest, previous_head)
    w.write_sync(private / 'private-manifest.json', canonical(manifest))
    w.write_sync(private / 'nonce.bin', nonce)
    # Summary goes through the same closed-schema AND fixed-clock validation.
    w.write_sync(out / 'public-summary.json', release_bytes(public, at))
    w._sync_dir(private); w._sync_dir(out)
    return public


def verify_synthetic_sessions(directories, expected_sessions, external_head):
    """Synthetic integrity check only; no access to a real blind directory."""
    import json
    from tools import forward_wp4 as w
    records = []; previous_private = '0'*64
    for directory in directories:
        root = w.private_path(directory)
        if any(p.is_symlink() for p in root.rglob('*')): raise Stop('synthetic chain symlink')
        if {p.name for p in root.iterdir()} != {'private','public-summary.json'}:
            raise Stop('public file allowlist')
        private = root/'private'
        public = json.loads((root/'public-summary.json').read_text())
        manifest = json.loads((private/'private-manifest.json').read_text())
        nonce = (private/'nonce.bin').read_bytes()
        if (set(manifest) != {'schema','synthetic','previous_private_commitment','incident','files'}
                or manifest['schema'] != 'forward-wp4-synthetic-private-v1'
                or manifest['synthetic'] is not True
                or manifest['previous_private_commitment'] != previous_private
                or manifest['incident'] not in PRIVATE_CODES + ATTEMPT_ACTIONS
                or commitment(manifest,nonce) != public['commitment']):
            raise Stop('private commitment chain')
        files = {str(p.relative_to(private)):w.file_info(p) for p in private.rglob('*')
                 if p.is_file() and p not in (private/'private-manifest.json',private/'nonce.bin')}
        if manifest['files'] != files: raise Stop('private files')
        records.append(public); previous_private = public['commitment']
    return verify_chain(records,expected_sessions,external_head)
