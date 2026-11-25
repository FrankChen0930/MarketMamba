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
    with pytest.raises(p.Stop, match='events damaged'):
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
