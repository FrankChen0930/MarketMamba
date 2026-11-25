"""Offline acceptance for local observation ledger; no publication/PIT admission."""
import json
import sqlite3
from datetime import datetime

import pandas as pd
import pytest

from tools import pit_ledger as p


def sources(tmp_path, rows=None):
    root = tmp_path / 'source'
    root.mkdir(exist_ok=True)
    rows = rows or [('2330', 1), ('2317', 2)]
    for name, keys in p.KEYS.items():
        data = [dict(stock_id=s, revenue_year=2026, revenue_month=10, revenue=v)
                if name == 'revenue_raw' else dict(stock_id=s, Date=pd.Timestamp('2026-09-30'), type='Revenue', value=v)
                for s, v in rows]
        pd.DataFrame(data).to_parquet(root / f'{name}.parquet', index=False)
    return root


def events(ledger):
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        return c.execute('SELECT kind, value_json, previous_json FROM events ORDER BY id').fetchall()


def test_initial_then_new_revision_disappearance_replay_and_retry(tmp_path):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    a = p.snapshot(root, ledger)
    assert all(t['counts']['INITIAL_STOCK'] == 2 for t in a['tables'].values())
    before = events(ledger)
    b = p.snapshot(root, ledger)
    assert all(sum(t['counts'].values()) == 0 for t in b['tables'].values())
    assert events(ledger) == before
    sources(tmp_path, [('2330', 9), ('2454', 3)])
    c = p.snapshot(root, ledger)
    for t in c['tables'].values():
        assert t['counts'] == dict(INITIAL_STOCK=0, FIRST_SEEN=1, REVISED=1, DISAPPEARED=1)
    revised = [r for r in events(ledger) if r[0] == 'REVISED']
    assert len(revised) == 4
    assert json.loads(revised[0][2])['revenue'] == 1
    assert json.loads(revised[0][1])['revenue'] == 9
    assert before == []
    # Reappearance is a newly seen active version, even if a historical SHA recurs.
    sources(tmp_path)
    d = p.snapshot(root, ledger)
    assert d['tables']['revenue_raw']['counts']['FIRST_SEEN'] == 1
    assert datetime.fromisoformat(a['observed_at']).utcoffset().total_seconds() == 28800
    assert all('first_observed_at' not in row for row in p.replay(ledger, a['run_id'], 'revenue_raw'))


def test_corruption_and_missing_db_stop(tmp_path):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    p.snapshot(root, ledger)
    db = ledger / 'ledger.sqlite3'
    db.write_bytes(b'broken')
    with pytest.raises(p.Stop):
        p.snapshot(root, ledger)
    db.unlink()
    with pytest.raises(p.Stop):
        p.snapshot(root, ledger)


def test_semantic_corruption_stops(tmp_path):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    summary = p.snapshot(root, ledger)
    blob = ledger/'blobs'/(summary['tables']['revenue_raw']['index']['sha256']+'.parquet')
    blob.chmod(0o644)
    blob.write_bytes(b'tampered')
    with pytest.raises(p.Stop):
        p.snapshot(root, ledger)


def test_duplicate_keys_and_partial_input_atomic_stop(tmp_path):
    root = sources(tmp_path, [('2330', 1), ('2330', 2)])
    ledger = tmp_path / 'ledger'
    with pytest.raises(p.Stop):
        p.snapshot(root, ledger)
    assert not (ledger / 'ledger.sqlite3').exists()
    sources(tmp_path)
    (root / 'cashflow_raw.parquet').unlink()
    with pytest.raises(p.Stop):
        p.snapshot(root, ledger)


def test_normalization_and_timezone():
    assert p.canonical({'v': 1.0, 't': pd.Timestamp('2026-11-25')}) == p.canonical({'t': datetime(2026, 11, 25), 'v': 1})
    with pytest.raises(p.Stop):
        p.observation_time(datetime(2026, 11, 25))
    with pytest.raises(p.Stop):
        p.canonical({'v': float('inf')})


def test_arrival_boundary_unknown_and_missing_calendar():
    df = pd.DataFrame(dict(stock_id=['2330']*4, revenue_year=[2026]*4,
                           revenue_month=[10]*4, create_time=['2026-11-12 23:59:59', '2026-11-13 00:00:00', '', '2026-11-10']))
    result = p.arrival_audit(df, ['2026-11-11', '2026-11-12'], '2026-11-01', '2026-11-30')
    assert result['parseable_rows'] == 3
    assert result['general_10th_proxy']['late_plus_two_count'] == 1
    assert result['general_10th_proxy']['late_plus_two_fraction'] == pytest.approx(1/3)
    assert result['legal_applicability'] == 'UNKNOWN'
    missing = p.arrival_audit(df, [], '2026-10-01', '2026-10-31')
    assert missing['general_10th_proxy']['late_plus_two_fraction'] is None


def test_two_in_memory_mutations_are_killed(tmp_path, monkeypatch):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    p.snapshot(root, ledger)
    sources(tmp_path, [('2330', 9), ('2317', 2)])
    original = p.value_hash
    monkeypatch.setattr(p, 'value_hash', lambda value: 'constant' if '"revenue":' in value else original(value))
    # The independent previous-value verification rejects the corrupted hash function.
    with pytest.raises(p.Stop, match='previous value mismatch'):
        p.snapshot(root, ledger)
    monkeypatch.setattr(p, 'value_hash', original)
    monkeypatch.setattr(p, 'second_session', lambda due, sessions, end: due)
    d = pd.DataFrame(dict(stock_id=['2330'], revenue_year=[2026], revenue_month=[10], create_time=['2026-11-11']))
    mutated = p.arrival_audit(d, ['2026-11-11', '2026-11-12'], '2026-11-01', '2026-11-30')
    with pytest.raises(AssertionError):
        assert mutated['general_10th_proxy']['late_plus_two_count'] == 0


def test_sample_large_file_scaling(tmp_path):
    import time
    root = sources(tmp_path, [(str(i), i) for i in range(10000)])
    start = time.monotonic()
    result = p.snapshot(root, tmp_path / 'ledger')
    elapsed = time.monotonic() - start
    assert sum(t['rows'] for t in result['tables'].values()) == 40000
    assert sum(t['counts']['INITIAL_STOCK'] for t in result['tables'].values()) == 40000
    print(f'SYNTHETIC_PERF rows=40000 seconds={elapsed:.6f} extrapolated_13760451_seconds={elapsed/40000*13760451:.3f}')


def test_append_only_event_and_run_protection(tmp_path):
    root = sources(tmp_path)
    p.snapshot(root, tmp_path / 'ledger')
    sources(tmp_path, [('2330', 9), ('2454', 3)])
    p.snapshot(root, tmp_path / 'ledger')
    with sqlite3.connect(tmp_path / 'ledger' / 'ledger.sqlite3') as c:
        for sql in ['DELETE FROM events', "UPDATE events SET kind='FIRST_SEEN'", 'DELETE FROM runs']:
            with pytest.raises(sqlite3.DatabaseError, match='append only'):
                c.execute(sql)


def test_changed_source_during_read_rolls_back(tmp_path, monkeypatch):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    p.snapshot(root, ledger)
    before = events(ledger)
    original = p.file_hash
    seen = {}
    def unstable(path):
        result = original(path)
        path = str(path)
        seen[path] = seen.get(path, 0) + 1
        return 'changed' if path.endswith('revenue_raw.parquet') and seen[path] > 1 else result
    monkeypatch.setattr(p, 'file_hash', unstable)
    with pytest.raises(p.Stop, match='source changed'):
        p.snapshot(root, ledger)
    assert events(ledger) == before


def test_v2_initial_has_no_json_and_replay_is_equivalent(tmp_path):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    a = p.snapshot(root, ledger)
    assert events(ledger) == []
    expected = {n: sorted(p.canonical(r) for r in pd.read_parquet(root/f'{n}.parquet').to_dict('records')) for n in p.KEYS}
    b = p.snapshot(root, ledger)
    assert b['status'] == 'NO_CHANGE'
    sources(tmp_path, [('2330', 9), ('2454', 3)])
    c = p.snapshot(root, ledger)
    for rid in (a['run_id'], b['run_id']):
        for n in p.KEYS:
            assert sorted(p.replay(ledger, rid, n)) == expected[n]
    for n in p.KEYS:
        actual = sorted(p.canonical(r) for r in pd.read_parquet(root/f'{n}.parquet').to_dict('records'))
        assert sorted(p.replay(ledger, c['run_id'], n)) == actual
    for f in (ledger/'blobs').glob('*.parquet'):
        assert f.stat().st_mode & 0o222 == 0


def test_no_change_never_decodes_rows_and_growth_is_small(tmp_path, monkeypatch):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    p.snapshot(root, ledger)
    before = sum(f.stat().st_size for f in ledger.rglob('*') if f.is_file())
    def forbidden(*args, **kwargs):
        raise AssertionError('NO_CHANGE must not rescan records')
    monkeypatch.setattr(p, 'records', forbidden)
    result = p.snapshot(root, ledger)
    assert result['status'] == 'NO_CHANGE'
    after = sum(f.stat().st_size for f in ledger.rglob('*') if f.is_file())
    assert after-before < 100_000


def test_synthetic_size_limit_and_small_diff(tmp_path):
    root = sources(tmp_path, [(str(i), i) for i in range(25000)])
    ledger = tmp_path / 'ledger'
    a = p.snapshot(root, ledger)
    total = sum(f.stat().st_size for f in ledger.rglob('*') if f.is_file())
    indexes = sum(t['index']['bytes'] for t in a['tables'].values())
    # Extrapolation is a synthetic budget check, not a real-source measurement.
    projected = indexes / 100000 * 13760451 + 85_000_000 + 1_000_000
    assert projected < 1_000_000_000
    assert total < 10_000_000
    sources(tmp_path, [('0', -1)] + [(str(i), i) for i in range(1,25000)])
    b = p.snapshot(root, ledger)
    assert all(t['counts']['REVISED'] == 1 for t in b['tables'].values())
    after = sum(f.stat().st_size for f in ledger.rglob('*') if f.is_file())
    assert after-total < 1_000_000
    assert len(events(ledger)) == 4
    print(f'SYNTHETIC_SIZE rows=100000 initial_bytes={total} index_bytes={indexes} projected_13760451_bytes={projected:.0f} one_revision_per_table_growth_bytes={after-total}')


def test_event_corruption_and_missing_base_stop(tmp_path):
    root = sources(tmp_path)
    ledger = tmp_path/'ledger'
    a = p.snapshot(root, ledger)
    sources(tmp_path, [('2330', 9), ('2454', 3)])
    p.snapshot(root, ledger)
    with sqlite3.connect(ledger/'ledger.sqlite3') as c:
        c.execute('DROP TRIGGER events_no_update')
        c.execute("UPDATE events SET value_json='{}' WHERE kind='REVISED'")
    with pytest.raises(p.Stop, match='schema|events damaged'):
        list(p.replay(ledger, a['run_id'], 'revenue_raw'))


def test_replay_empty_and_reappearance_and_old_run(tmp_path):
    root = sources(tmp_path)
    ledger = tmp_path/'ledger'
    a = p.snapshot(root, ledger)
    before = sorted(p.replay(ledger, a['run_id'], 'revenue_raw'))
    for name in p.KEYS:
        f = root/f'{name}.parquet'
        pd.read_parquet(f).iloc[:0].to_parquet(f, index=False)
    b = p.snapshot(root, ledger)
    assert list(p.replay(ledger, b['run_id'], 'revenue_raw')) == []
    sources(tmp_path)
    c = p.snapshot(root, ledger)
    assert sorted(p.replay(ledger, c['run_id'], 'revenue_raw')) == before
    assert sorted(p.replay(ledger, a['run_id'], 'revenue_raw')) == before
    base = ledger/'blobs'/(a['tables']['revenue_raw']['base']['sha256']+'.parquet')
    base.unlink()
    with pytest.raises(p.Stop):
        p.snapshot(root, ledger)


@pytest.fixture(autouse=True)
def isolated_ledger_allowlist(tmp_path, monkeypatch):
    # Explicit test-only allowlist; never write the real observation ledger.
    monkeypatch.setattr(p, 'ALLOWED_LEDGER_ROOTS', (tmp_path / 'ledger',), raising=False)


@pytest.mark.parametrize('bad', [None, float('nan'), '', '   '])
def test_normalized_missing_business_key_stops(tmp_path, bad):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / 'bad.parquet'
    pq.write_table(pa.Table.from_pylist([dict(stock_id=bad, revenue_year=2026,
                                           revenue_month=10, revenue=1)]), path)
    with pytest.raises(p.Stop, match='business key'):
        list(p.records(path, 'revenue_raw'))


@pytest.mark.parametrize('trigger', ['events_no_update', 'events_no_delete',
                                     'runs_no_update', 'runs_no_delete'])
@pytest.mark.parametrize('replacement', [False, True])
def test_missing_or_modified_trigger_stops(tmp_path, trigger, replacement):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    p.snapshot(root, ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        c.execute(f'DROP TRIGGER {trigger}')
        if replacement:
            table, action = trigger.split('_no_')
            c.execute(f'CREATE TRIGGER {trigger} BEFORE {action} ON {table} BEGIN SELECT 1; END')
        with pytest.raises(p.Stop, match='schema'):
            p.validate(c, ledger)
    with pytest.raises(p.Stop, match='schema'):
        p.snapshot(root, ledger)


def test_altered_table_schema_stops(tmp_path):
    ledger = tmp_path / 'ledger'
    p.snapshot(sources(tmp_path), ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        c.execute('ALTER TABLE events ADD COLUMN unexpected TEXT')
        with pytest.raises(p.Stop, match='schema'):
            p.validate(c, ledger)


@pytest.mark.parametrize('destination', ['arbitrary/ledger', 'Data/other', 'source/ledger'])
def test_outside_namespace_rejected_before_write(tmp_path, destination):
    root = sources(tmp_path)
    ledger = tmp_path / destination
    with pytest.raises(p.Stop, match='namespace|disjoint'):
        p.snapshot(root, ledger)
    assert not ledger.exists()


@pytest.mark.parametrize('component', ['root', 'ancestor', 'blobs', '.lock',
                                       'ledger.sqlite3', 'initial.sqlite3', 'IDENTITY.json'])
def test_symlink_write_targets_rejected(tmp_path, monkeypatch, component):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    outside = tmp_path / 'outside'
    outside.mkdir()
    if component == 'root':
        ledger.symlink_to(outside, target_is_directory=True)
    elif component == 'ancestor':
        (tmp_path / 'alias').symlink_to(tmp_path, target_is_directory=True)
        ledger = tmp_path / 'alias' / 'ledger'
        monkeypatch.setattr(p, 'ALLOWED_LEDGER_ROOTS', (ledger,))
    else:
        ledger.mkdir()
        (ledger / component).symlink_to(outside / 'victim')
    with pytest.raises(p.Stop, match='symlink'):
        p.snapshot(root, ledger)
    assert list(outside.iterdir()) == []


def test_index_and_sqlite_temporaries_confined(tmp_path, monkeypatch):
    original_temp = p.tempfile.NamedTemporaryFile
    original_connect = p.sqlite3.connect
    seen = []
    def confined(*args, **kwargs):
        assert kwargs.get('dir') is not None
        assert (tmp_path / 'ledger') in p.Path(kwargs['dir']).parents or p.Path(kwargs['dir']) == tmp_path / 'ledger'
        seen.append(kwargs['dir'])
        return original_temp(*args, **kwargs)
    class Connection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == 'BEGIN IMMEDIATE':
                assert super().execute('PRAGMA temp_store').fetchone()[0] == 2
            return super().execute(sql, *args, **kwargs)
    monkeypatch.setattr(p.tempfile, 'NamedTemporaryFile', confined)
    monkeypatch.setattr(p.sqlite3, 'connect', lambda *a, **kw: original_connect(*a, factory=Connection, **kw))
    p.snapshot(sources(tmp_path), tmp_path / 'ledger')
    assert len(seen) >= 4


def _fault_writer(source, ledger, pipe, mode):
    """Real file DB and rollback journal; no :memory: durability substitution."""
    import os
    import resource
    import signal
    p.ALLOWED_LEDGER_ROOTS = (ledger,)
    original = sqlite3.connect
    class Connection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            result = super().execute(sql, *args, **kwargs)
            if mode == 'kill' and sql.startswith('INSERT INTO events('):
                # Spill dirty pages into the DB while the transaction is uncommitted.
                super().execute('PRAGMA cache_size=1')
                for i in range(32):
                    super().execute('INSERT INTO events(run_id,table_name,key,kind,value_json) VALUES(?,?,?,?,?)',
                                    ('uncommitted', 'revenue_raw', str(i), 'FIRST_SEEN', 'x' * 65536))
                pipe.send('WRITING_WITH_JOURNAL')
                signal.pause()
            return result
    try:
        if mode == 'kill':
            p.sqlite3.connect = lambda *a, **kw: original(*a, factory=Connection, **kw)
        else:
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            limit = (ledger / 'ledger.sqlite3').stat().st_size + 4096
            resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        p.snapshot(source, ledger)
        pipe.send('COMMITTED')
    except p.Stop as exc:
        pipe.send('STOP:' + str(exc))
    finally:
        pipe.close()


@pytest.mark.parametrize('mode', ['kill', 'full'])
def test_real_sqlite_kill_and_file_size_limit_recovery(tmp_path, mode):
    import multiprocessing as mp
    import signal
    ctx = mp.get_context('spawn')
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    first = p.snapshot(root, ledger)
    before = {name: sorted(p.replay(ledger, first['run_id'], name)) for name in p.KEYS}
    if mode == 'full':
        # Larger than the per-file RLIMIT for the real rollback journal/DB.
        for name in p.KEYS:
            path = root / f'{name}.parquet'
            df = pd.read_parquet(path)
            df['payload'] = ['x' * 200000, 'y' * 200000]
            df.to_parquet(path, index=False)
    else:
        sources(tmp_path, [('2330', 9), ('2454', 3)])
    expected = {name: sorted(p.canonical(row) for row in pd.read_parquet(root / f'{name}.parquet').to_dict('records'))
                for name in p.KEYS}
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_fault_writer, args=(root, ledger, child, mode))
    process.start()
    child.close()
    try:
        assert parent.poll(30), 'writer failed to reach fault boundary'
        message = parent.recv()
        if mode == 'kill':
            assert message == 'WRITING_WITH_JOURNAL'
            assert (ledger / 'ledger.sqlite3-journal').stat().st_size > 0
            process.kill()
        else:
            assert message.startswith('STOP:'), message
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == (-signal.SIGKILL if mode == 'kill' else 0)
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        _, last, _ = p.validate(c, ledger)
        count = c.execute('SELECT count(*) FROM runs').fetchone()[0]
    recovered = {name: sorted(p.replay(ledger, last['run_id'], name)) for name in p.KEYS}
    assert recovered in (before, expected)
    assert count == 1  # Both controlled failures occurred before commit.
    assert recovered == before
    retry = p.snapshot(root, ledger)
    assert retry['status'] == 'CHANGED'
    assert {name: sorted(p.replay(ledger, retry['run_id'], name)) for name in p.KEYS} == expected
    event_count = len(events(ledger))
    assert p.snapshot(root, ledger)['status'] == 'NO_CHANGE'
    assert len(events(ledger)) == event_count
    print(f'REAL_SQLITE fault={mode} message={message} recovered_runs={count} retry=CHANGED repeat=NO_CHANGE events={event_count}')


@pytest.mark.parametrize('guard', ['normalized_key', 'schema_definition'])
def test_new_guard_mutations_are_killed(tmp_path, monkeypatch, guard):
    import inspect
    def requires_stop(action):
        blocked = False
        try:
            action()
        except p.Stop:
            blocked = True
        assert blocked, 'mutant accepted invalid evidence'
    if guard == 'normalized_key':
        path = tmp_path / 'nan.parquet'
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist([dict(stock_id=float('nan'), revenue_year=2026,
                                               revenue_month=10, revenue=1)]), path)
        action = lambda: list(p.records(path, 'revenue_raw'))
        name = 'records'
        code = inspect.getsource(p.records).replace(
            "if any(v is None or (isinstance(v, str) and not v.strip()) for v in key.values()):", 'if False:')
    else:
        ledger = tmp_path / 'ledger'
        p.snapshot(sources(tmp_path), ledger)
        with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
            c.execute('DROP TRIGGER events_no_delete')
        action = lambda: list(p.replay(ledger, 'nonexistent', 'revenue_raw'))
        # Direct validate oracle must fail specifically on the missing protection.
        def action():
            with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
                p.validate(c, ledger)
        name = 'validate'
        code = inspect.getsource(p.validate).replace('if schema_identity(c) != EXPECTED_SCHEMA:', 'if False:')
    requires_stop(action)
    namespace = dict(vars(p))
    assert code != inspect.getsource(getattr(p, name))
    exec(compile(code, p.__file__, 'exec'), namespace)
    monkeypatch.setattr(p, name, namespace[name])
    with pytest.raises(AssertionError, match='mutant accepted'):
        requires_stop(action)
    print(f'NEW_MUTATION {guard} KILLED')


@pytest.mark.parametrize('sql', [
    'CREATE TRIGGER sqliteXdrop BEFORE INSERT ON events BEGIN SELECT RAISE(IGNORE); END',
    'CREATE TABLE sqliteXextra(value TEXT)',
    'CREATE INDEX sqliteXindex ON events(key)',
    'CREATE VIEW sqliteXview AS SELECT * FROM events',
])
def test_extra_schema_objects_stop_before_run(tmp_path, sql):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    p.snapshot(root, ledger)
    sources(tmp_path, [('2330', 9), ('2317', 2)])
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        c.execute(sql)
        with pytest.raises(p.Stop, match='schema'):
            p.validate(c, ledger)
    with pytest.raises(p.Stop, match='schema'):
        p.snapshot(root, ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        assert c.execute('SELECT count(*) FROM runs').fetchone()[0] == 1
        assert c.execute('SELECT count(*) FROM events').fetchone()[0] == 0


@pytest.mark.parametrize('bad', [None, float('nan'), '', ' ', '\u3000', '\u00a0'])
def test_legacy_v2_invalid_index_key_stops_before_no_change(tmp_path, monkeypatch, bad):
    import pyarrow.parquet as pq
    root = sources(tmp_path)
    path = root / 'revenue_raw.parquet'
    df = pd.read_parquet(path)
    df['stock_id'] = [bad, bad]
    # Distinct remaining business key components avoid unrelated duplicate STOP.
    df['revenue_month'] = [9, 10]
    df.to_parquet(path, index=False)
    ledger = tmp_path / 'ledger'
    original = p.records
    def old_records(path, name):
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                yield p.canonical({k: row[k] for k in p.KEYS[name]}), p.canonical(row)
    monkeypatch.setattr(p, 'records', old_records)
    p.snapshot(root, ledger)
    monkeypatch.setattr(p, 'records', original)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        c.execute('PRAGMA user_version=2')  # Emulate the old writer's schema revision.
        with pytest.raises(p.Stop, match='business key'):
            p.validate(c, ledger)
    with pytest.raises(p.Stop, match='business key'):
        p.snapshot(root, ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        assert c.execute('SELECT count(*) FROM runs').fetchone()[0] == 1
        assert c.execute('SELECT count(*) FROM events').fetchone()[0] == 0


def test_clean_v2_upgrade_audits_once_and_preserves_no_change(tmp_path, monkeypatch):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    first = p.snapshot(root, ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        c.execute('PRAGMA user_version=2')
    calls = []
    audit = p.audit_index_keys
    def counted(path, name):
        calls.append(name)
        return audit(path, name)
    monkeypatch.setattr(p, 'audit_index_keys', counted)
    second = p.snapshot(root, ledger)
    assert second['status'] == 'NO_CHANGE'
    assert sorted(calls) == sorted(p.KEYS)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        assert c.execute('PRAGMA user_version').fetchone()[0] == 3
    calls.clear()
    third = p.snapshot(root, ledger)
    assert third['status'] == 'NO_CHANGE'
    assert calls == []
    assert list(p.replay(ledger, first['run_id'], 'revenue_raw'))
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        assert c.execute('SELECT count(*) FROM runs').fetchone()[0] == 3
        assert c.execute('SELECT count(*) FROM events').fetchone()[0] == 0
        c.execute('PRAGMA user_version=1')
        with pytest.raises(p.Stop, match='schema'):
            p.validate(c, ledger)


def test_v3_missing_audit_marker_stops(tmp_path):
    ledger = tmp_path / 'ledger'
    p.snapshot(sources(tmp_path), ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        c.execute('DROP TRIGGER runs_no_update')
        rid, payload = c.execute('SELECT run_id,summary_json FROM runs').fetchone()
        summary = json.loads(payload)
        summary.pop('business_key_audit')
        payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        c.execute('UPDATE runs SET summary_json=?,digest=? WHERE run_id=?', (payload, p.value_hash(payload), rid))
        c.execute("CREATE TRIGGER runs_no_update BEFORE UPDATE ON runs BEGIN SELECT RAISE(ABORT,'append only'); END")
        with pytest.raises(p.Stop, match='audit marker'):
            p.validate(c, ledger)


def test_v2_upgrade_rolls_back_with_failed_run(tmp_path, monkeypatch):
    root = sources(tmp_path)
    ledger = tmp_path / 'ledger'
    p.snapshot(root, ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        c.execute('PRAGMA user_version=2')
    sources(tmp_path, [('2330', 9), ('2317', 2)])
    original = p.records
    def broken(*args, **kwargs):
        raise p.Stop('injected source read failure')
    monkeypatch.setattr(p, 'records', broken)
    with pytest.raises(p.Stop, match='injected'):
        p.snapshot(root, ledger)
    with sqlite3.connect(ledger / 'ledger.sqlite3') as c:
        assert c.execute('PRAGMA user_version').fetchone()[0] == 2
        assert c.execute('SELECT count(*) FROM runs').fetchone()[0] == 1
    monkeypatch.setattr(p, 'records', original)
    assert p.snapshot(root, ledger)['status'] == 'CHANGED'
