"""Offline, append-only local-cache observations. RESEARCH_ONLY / NOT_DEPLOYED.

SQLite FULL-sync transactions publish all four tables together. Initial values
reside in verified immutable parquet copies with compressed per-key hash indexes.
Only diffs have full JSON values; immutable runs seal the base/index and event chain.
No fetcher imports, network, source writes, publication-time inference or labels.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import shutil
import tempfile

import pyarrow as pa
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

KEYS = {'revenue_raw': ('stock_id', 'revenue_year', 'revenue_month'),
        **{n: ('stock_id', 'Date', 'type') for n in
           ('financials_raw', 'balance_sheet_raw', 'cashflow_raw')}}
KINDS = ('INITIAL_STOCK', 'FIRST_SEEN', 'REVISED', 'DISAPPEARED')
SOURCE = Path('/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6')
VERSION = 'pit-local-observation-v2'
DEFAULT_LEDGER = Path('/home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2')


class Stop(RuntimeError):
    """Untrusted or incomplete evidence: stop without publishing a run."""


def normalized(x):
    if x is None or x is pd.NA or x is pd.NaT:
        return None
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if not math.isfinite(x):
            raise Stop('nonfinite numeric value')
        return int(x) if x.is_integer() else x
    if isinstance(x, (str, int, bool)):
        return x
    raise Stop(f'unsupported scalar: {type(x).__name__}')


def canonical(value):
    return json.dumps({k: normalized(v) for k, v in sorted(value.items())},
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def value_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def observation_time(now=None):
    now = now or datetime.now(ZoneInfo('Asia/Taipei'))
    if now.tzinfo is None or now.utcoffset() is None:
        raise Stop('observation time must carry timezone')
    return now.astimezone(ZoneInfo('Asia/Taipei')).isoformat()


def digest_rows(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(tuple(row), ensure_ascii=False, separators=(',', ':')).encode())
        h.update(b'\n')
    return h.hexdigest()


SCHEMA = '''
CREATE TABLE events(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, table_name TEXT NOT NULL,
 key TEXT NOT NULL, kind TEXT NOT NULL, value_hash TEXT, value_json TEXT,
 previous_hash TEXT, previous_json TEXT);
CREATE TABLE runs(seq INTEGER PRIMARY KEY, run_id TEXT UNIQUE NOT NULL,
 summary_json TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TRIGGER events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append only'); END;
CREATE TRIGGER runs_no_update BEFORE UPDATE ON runs BEGIN SELECT RAISE(ABORT,'append only'); END;
CREATE TRIGGER runs_no_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT,'append only'); END;
PRAGMA user_version=2;
'''
EVENT_SQL = 'SELECT table_name,key,kind,value_hash,value_json,previous_hash,previous_json FROM events WHERE run_id=? ORDER BY id'


def validate(c, ledger):
    if c.execute('PRAGMA user_version').fetchone()[0] != 2:
        raise Stop('ledger schema mismatch')
    if c.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise Stop('ledger SQLite integrity failed')
    previous, last, initial = None, None, None
    event_count = 0
    for expected_seq, (seq, run_id, payload, seal) in enumerate(c.execute('SELECT * FROM runs ORDER BY seq'), 1):
        summary = json.loads(payload)
        if (seq != expected_seq or summary['version'] != VERSION or summary['run_id'] != run_id or
                summary['previous_run_digest'] != previous or value_hash(payload) != seal):
            raise Stop('ledger run chain damaged')
        observation_time(datetime.fromisoformat(summary['observed_at']))
        if last and summary['observed_at'] < last['observed_at']:
            raise Stop('observation time moved backwards')
        if digest_rows(c.execute(EVENT_SQL, (run_id,))) != summary['events_digest']:
            raise Stop('ledger events damaged')
        event_count += sum(sum(t['counts'][k] for k in KINDS if k != 'INITIAL_STOCK') for t in summary['tables'].values())
        initial = initial or summary
        previous, last = seal, summary
    if last is None or c.execute('SELECT count(*) FROM events').fetchone()[0] != event_count:
        raise Stop('ledger missing run or orphan events')
    # Base/index bytes are the immutable initial state; the sealed diffs are its only changes.
    for info in initial['tables'].values():
        for kind in ('base', 'index'):
            identity = info[kind]
            path = ledger / 'blobs' / (identity['sha256'] + '.parquet')
            if path.is_symlink() or path.stat().st_mode & 0o222 or file_hash(path) != identity['sha256']:
                raise Stop('ledger base/index damaged')
    return previous, last, initial


def publish_blob(source, ledger, expected=None):
    """Copy, fsync, verify, chmod then atomic publish; never overwrite a blob."""
    blobs = ledger / 'blobs'
    blobs.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=blobs, suffix='.partial', delete=False) as f:
        temp = Path(f.name)
        try:
            with Path(source).open('rb') as inp:
                shutil.copyfileobj(inp, f, 4 * 1024 * 1024)
            f.flush()
            os.fsync(f.fileno())
            sha = file_hash(temp)
            if expected is not None and sha != expected:
                raise Stop('source changed while copying')
            target = blobs / (sha + '.parquet')
            os.chmod(temp, 0o444)
            if target.exists():
                if target.is_symlink() or target.stat().st_mode & 0o222 or file_hash(target) != sha:
                    raise Stop('existing blob damaged')
            else:
                os.link(temp, target)  # atomic no-clobber publication, same filesystem
                fsync_dir(blobs)
            return dict(sha256=sha, bytes=target.stat().st_size)
        finally:
            temp.unlink(missing_ok=True)


def records(path, name):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32768):
        for record in batch.to_pylist():
            if any(record[k] is None for k in KEYS[name]):
                raise Stop(f'{name}: null business key')
            yield canonical({k: record[k] for k in KEYS[name]}), canonical(record)


def _overlay(c, name, seq):
    result = {}
    for key, kind, val, sha in c.execute(
            '''SELECT e.key,e.kind,e.value_json,e.value_hash FROM events e
               JOIN runs r ON r.run_id=e.run_id WHERE e.table_name=? AND r.seq<=? ORDER BY e.id''', (name, seq)):
        result[key] = (kind, val, sha)
    return result


def _replay(c, ledger, initial, name, seq, selected=None):
    overlay = _overlay(c, name, seq)
    base = ledger / 'blobs' / (initial['tables'][name]['base']['sha256'] + '.parquet')
    for key, val in records(base, name):
        if key not in overlay and (selected is None or key in selected):
            yield key, val
    for key, (kind, val, sha) in overlay.items():
        if kind != 'DISAPPEARED' and (selected is None or key in selected):
            yield key, val


def replay(ledger, run_id, table_name):
    """Yield canonical full rows at any committed run (including NO_CHANGE)."""
    ledger = Path(ledger).resolve()
    if table_name not in KEYS:
        raise Stop('unknown table')
    try:
        with sqlite3.connect(f'file:{ledger / "ledger.sqlite3"}?mode=ro', uri=True) as c:
            _, _, initial = validate(c, ledger)
            row = c.execute('SELECT seq FROM runs WHERE run_id=?', (run_id,)).fetchone()
            if row is None:
                raise Stop('unknown replay run')
            yield from (val for _, val in _replay(c, ledger, initial, table_name, row[0]))
    except (sqlite3.Error, OSError, ValueError, KeyError) as e:
        raise Stop(f'replay validation STOP: {e}') from e


def atomic_json(path, value):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fsync_dir(path.parent)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot(source, ledger, now=None):
    source, ledger = Path(source).resolve(), Path(ledger).resolve()
    if ledger == source or source in ledger.parents or ledger in source.parents:
        raise Stop('source and research ledger must be disjoint')
    started = time.monotonic()
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _snapshot(source, ledger, now, started)


def _snapshot(source, ledger, now, started):
    db, marker = ledger / 'ledger.sqlite3', ledger / 'IDENTITY.json'
    if marker.exists() and not db.exists():
        raise Stop('ledger database missing; refuse reinitialization')
    exists = db.exists()
    temp = ledger / 'initial.sqlite3'
    if not exists and temp.exists():
        raise Stop('unpublished initial database exists; inspect before recovery')
    c = None
    try:
        meta = {}
        for name in KEYS:
            path = source / f'{name}.parquet'
            pf = pq.ParquetFile(path)
            if not set(KEYS[name]) <= set(pf.schema_arrow.names):
                raise Stop(f'{name}: missing business keys')
            meta[name] = dict(path=str(path), sha256=file_hash(path), rows=pf.metadata.num_rows,
                              schema=str(pf.schema_arrow))
        c = sqlite3.connect(db if exists else temp)
        c.execute('PRAGMA synchronous=FULL')
        c.execute('PRAGMA temp_store=FILE')
        c.execute('PRAGMA cache_size=-65536')
        if exists:
            previous, last, initial = validate(c, ledger)
            if last['source_root'] != str(source):
                raise Stop('source root mismatch')
        else:
            c.executescript(SCHEMA)
            previous, last, initial = None, None, None
        observed_at = observation_time(now)
        if last and observed_at < last['observed_at']:
            raise Stop('observation time moved backwards')
        c.execute('BEGIN IMMEDIATE')
        summary = dict(version=VERSION, run_id=uuid.uuid4().hex, observed_at=observed_at,
                       source_root=str(source), evidence_class='LOCAL_CACHE_OBSERVATION_ONLY',
                       publication_time='UNKNOWN', refresh_coverage='UNKNOWN',
                       previous_run_digest=previous, parser_sha256=file_hash(__file__),
                       source_commit=subprocess.check_output(
                           ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip(),
                       python=sys.version, tables={})
        rid = summary['run_id']
        summary['status'] = 'INITIAL_STOCK' if last is None else 'NO_CHANGE'
        for name, keys in KEYS.items():
            counts = dict.fromkeys(KINDS, 0)
            info = dict(meta[name], counts=counts, refresh_status='UNKNOWN_NOT_INFERRED_FROM_CACHE')
            summary['tables'][name] = info
            if last and last['tables'][name]['sha256'] == meta[name]['sha256']:
                info['same_source_as_previous'] = True
                continue
            c.execute('CREATE TEMP TABLE incoming(key TEXT PRIMARY KEY, value_hash TEXT NOT NULL) WITHOUT ROWID')
            if last is None:
                info['base'] = publish_blob(meta[name]['path'], ledger, meta[name]['sha256'])
                scan_path = ledger / 'blobs' / (info['base']['sha256'] + '.parquet')
                index_schema = pa.schema([('key', pa.string()), ('value_hash', pa.binary(32))])
                with tempfile.NamedTemporaryFile(suffix='.parquet') as idx:
                    with pq.ParquetWriter(idx.name, index_schema, compression='zstd', use_dictionary=False) as writer:
                        batch_keys, batch_hashes = [], []
                        for key, val in records(scan_path, name):
                            sha = value_hash(val)
                            batch_keys.append(key)
                            batch_hashes.append(bytes.fromhex(sha))
                            if len(batch_keys) == 32768:
                                c.executemany('INSERT INTO incoming VALUES(?,?)', zip(batch_keys, (h.hex() for h in batch_hashes)))
                                writer.write_table(pa.Table.from_arrays([pa.array(batch_keys), pa.array(batch_hashes)], schema=index_schema))
                                batch_keys, batch_hashes = [], []
                        if batch_keys:
                            c.executemany('INSERT INTO incoming VALUES(?,?)', zip(batch_keys, (h.hex() for h in batch_hashes)))
                            writer.write_table(pa.Table.from_arrays([pa.array(batch_keys), pa.array(batch_hashes)], schema=index_schema))
                    info['index'] = publish_blob(idx.name, ledger)
                counts['INITIAL_STOCK'] = c.execute('SELECT count(*) FROM incoming').fetchone()[0]
            else:
                summary['status'] = 'CHANGED'
                # Temporary hash index only; no second persistent full-value store.
                c.execute('CREATE TEMP TABLE prior(key TEXT PRIMARY KEY, value_hash TEXT NOT NULL) WITHOUT ROWID')
                idx = ledger / 'blobs' / (initial['tables'][name]['index']['sha256'] + '.parquet')
                for batch in pq.ParquetFile(idx).iter_batches(batch_size=32768):
                    c.executemany('INSERT INTO prior VALUES(?,?)', ((r['key'], r['value_hash'].hex()) for r in batch.to_pylist()))
                last_seq = c.execute('SELECT max(seq) FROM runs').fetchone()[0]
                for key, (kind, val, sha) in _overlay(c, name, last_seq).items():
                    if kind == 'DISAPPEARED':
                        c.execute('DELETE FROM prior WHERE key=?', (key,))
                    else:
                        c.execute('INSERT OR REPLACE INTO prior VALUES(?,?)', (key, sha))
                changed = {}
                for key, val in records(meta[name]['path'], name):
                    sha = value_hash(val)
                    c.execute('INSERT INTO incoming VALUES(?,?)', (key, sha))
                    old = c.execute('SELECT value_hash FROM prior WHERE key=?', (key,)).fetchone()
                    if old is None or old[0] != sha:
                        changed[key] = ('FIRST_SEEN' if old is None else 'REVISED', val, sha, old[0] if old else None)
                for key, sha in c.execute('SELECT key,value_hash FROM prior WHERE key NOT IN (SELECT key FROM incoming)'):
                    changed[key] = ('DISAPPEARED', None, None, sha)
                need_previous = {key for key, row in changed.items() if row[3] is not None}
                old_values = dict(_replay(c, ledger, initial, name, last_seq, need_previous)) if need_previous else {}
                for key, (kind, val, sha, old_sha) in sorted(changed.items()):
                    old_val = old_values.get(key)
                    if old_sha is not None and (old_val is None or value_hash(old_val) != old_sha):
                        raise Stop('replayed previous value mismatch')
                    c.execute('INSERT INTO events(run_id,table_name,key,kind,value_hash,value_json,previous_hash,previous_json) VALUES(?,?,?,?,?,?,?,?)',
                              (rid, name, key, kind, sha, val, old_sha, old_val))
                    counts[kind] += 1
                c.execute('DROP TABLE prior')
            if c.execute('SELECT count(*) FROM incoming').fetchone()[0] != meta[name]['rows']:
                raise Stop('row count mismatch')
            c.execute('DROP TABLE incoming')
            print(f'{name}: {meta[name]["rows"]:,} rows {counts}', flush=True)
        for name, info in meta.items():
            if file_hash(info['path']) != info['sha256']:
                raise Stop(f'{name}: source changed before commit')
        summary['events_digest'] = digest_rows(c.execute(EVENT_SQL, (rid,)))
        summary['elapsed_seconds_before_commit'] = time.monotonic() - started
        payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        c.execute('INSERT INTO runs(run_id,summary_json,digest) VALUES(?,?,?)', (rid, payload, value_hash(payload)))
        c.commit()
        c.close()
        c = None
        if not exists:
            os.rename(temp, db)
            fsync_dir(ledger)
        atomic_json(marker, dict(version=VERSION, source_root=str(source)))
        atomic_json(ledger / f'{rid}.summary.json', summary)
        return summary
    except (sqlite3.Error, ValueError, KeyError, OSError) as e:
        raise Stop(f'ledger/source validation STOP: {e}') from e
    finally:
        if c is not None:
            c.rollback()
            c.close()
        if not exists and temp.exists():
            temp.unlink()


def second_session(due, sessions, end):
    later = sessions[sessions > due]
    return later[1] if len(later) >= 2 and due <= end else pd.NaT


def arrival_audit(df, sessions, coverage_start, coverage_end):
    """General 10th-day proxy only; unknown issuer exceptions never become legal findings."""
    sessions = pd.DatetimeIndex(pd.to_datetime(sessions)).sort_values().unique()
    lo, hi = pd.Timestamp(coverage_start), pd.Timestamp(coverage_end)
    parsed = pd.to_datetime(df.create_time, errors='coerce', format='mixed')
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize('Asia/Taipei')
    else:
        parsed = parsed.dt.tz_convert('Asia/Taipei')
    valid = df.loc[parsed.notna()].copy()
    valid['arrival'] = parsed[parsed.notna()].dt.tz_localize(None)
    due = pd.to_datetime(dict(year=valid.revenue_year, month=valid.revenue_month, day=1)) + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)
    valid['due'] = due
    valid['days'] = (valid.arrival - due).dt.total_seconds() / 86400
    valid['period'] = valid.revenue_year.astype(str) + '-' + valid.revenue_month.astype(int).astype(str).str.zfill(2)
    valid['arrival_month'] = valid.arrival.dt.strftime('%Y-%m')
    valid['proxy_end'] = [second_session(d, sessions, hi) if d >= lo else pd.NaT for d in due]
    valid['late'] = valid.arrival.dt.normalize() > valid.proxy_end

    def stats(part):
        covered = part.proxy_end.notna()
        n = int(covered.sum())
        late = int(part.loc[covered, 'late'].sum())
        return dict(rows=len(part), days_from_10th_midnight_quantiles={str(q): float(part.days.quantile(q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]} if len(part) else {},
                    before_10th_count=int((part.days < 0).sum()), calendar_covered_rows=n,
                    calendar_unknown_rows=len(part)-n, late_plus_two_count=late,
                    late_plus_two_fraction=late/n if n else None)
    return dict(total_rows=len(df), parseable_rows=len(valid), unparseable_rows=len(df)-len(valid),
                legal_applicability='UNKNOWN', verified_legal_late_fraction=None,
                semantic_warning='create_time is unverified provider/cache metadata, not certified publication or local receipt; no issuer-specific exception/historical holiday-extension evidence',
                timezone_assumption='naive create_time interpreted Asia/Taipei',
                calendar_coverage=[str(lo.date()), str(hi.date())],
                general_10th_proxy=stats(valid),
                by_revenue_month={str(k): stats(g) for k,g in valid.groupby('period')},
                by_arrival_month={str(k): stats(g) for k,g in valid.groupby('arrival_month')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--ledger', type=Path, default=DEFAULT_LEDGER)
    args = ap.parse_args()
    try:
        result = snapshot(args.source, args.ledger)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Stop as e:
        print(f'STOP: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
