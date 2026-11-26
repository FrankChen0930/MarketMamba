"""WP3 synthetic feature-only snapshots; no external data, training or evaluation."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools import forward_wp3 as w


@pytest.fixture(autouse=True)
def namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(w, 'ROOT', tmp_path)


def fixture():
    names = json.loads(Path('research/v7/corrected-baseline-v1/feature-manifest.json').read_text())['feature_order']
    days = np.arange('2024-01-01', '2024-04-20', dtype='datetime64[D]')
    # Includes missing stock sessions and a late entrant; fixed roster preserves
    # the floating point reduction axis when future names first appear.
    pairs = [(d, s) for i, d in enumerate(days) for s in ['A', 'B', 'C', 'Z']
             if not (s == 'Z' and i < 80) and not (s == 'B' and i % 11 == 0)]
    dates = np.array([p[0] for p in pairs], dtype='datetime64[D]')
    stocks = np.array([p[1] for p in pairs], dtype='U32')
    x = np.random.default_rng(9).normal(size=(len(pairs), 48)).astype('float32')
    mask = np.ones(x.shape, dtype=bool)
    mask[::7, 2] = False
    x[~mask] = 0
    arrays = dict(dates=dates, stock_ids=stocks, X=x, masks=mask,
                  splits=np.ones(len(x), dtype='uint8'), eligibility=np.ones(len(x), dtype=bool))
    context = w.make_context(names, [str(d) for d in days], ['A', 'B', 'C', 'Z'], 'a' * 64)
    return arrays, context, days


def slice_at(a, selector):
    return {k: v[selector].copy() for k, v in a.items()}


def start(tmp_path):
    a, c, days = fixture()
    old = slice_at(a, a['dates'] <= days[69])
    p = tmp_path / 'reference'
    w.create_reference(p, old, c, b'original-score-bytes\x00\xff', 'b' * 64)
    return a, c, days, p


def test_full_reference_then_append_is_bitwise_equal_to_full_rebuild(tmp_path):
    a, c, days, parent = start(tmp_path)
    original = {p.name: p.read_bytes() for p in parent.iterdir()}
    delta = slice_at(a, a['dates'] > days[69])
    out = tmp_path / 'append'
    w.append_features(parent, out, delta, w.snapshot_id(parent), c,
                      w.batch_id(delta), 'c' * 64)
    got, manifest = w.load_snapshot(out)
    full = w.context_features(a, c)
    assert got['C185'].shape == (len(a['X']), 185)
    assert got['C185'].tobytes() == full.tobytes()
    n = sum(a['dates'] <= days[69])
    old, _ = w.load_snapshot(parent)
    for k in old:
        assert got[k][:n].dtype == old[k].dtype
        assert got[k][:n].tobytes() == old[k].tobytes()
    assert (out / 'scores.bin').read_bytes() == original['scores.bin']
    assert {p.name: p.read_bytes() for p in parent.iterdir()} == original
    assert manifest['parent'] == w.snapshot_id(parent)
    assert not list(out.glob('y*.npy'))


@pytest.mark.parametrize('kind', ['future_ohlcv', 'future_universe', 'future_labels'])
def test_future_mutation_does_not_change_historical_features_or_scores(tmp_path, kind):
    a, c, days, parent = start(tmp_path)
    before, _ = w.load_snapshot(parent)
    changed = slice_at(a, np.ones(len(a['X']), bool))
    future = a['dates'] > days[69]
    if kind == 'future_ohlcv':
        changed['X'][future] = 1e12
        changed['X'][~changed['masks']] = 0
    elif kind == 'future_universe':
        changed = slice_at(changed, ~(future & (a['stock_ids'] == 'C')))
    else:
        # No feature API accepts labels. Poisoned future labels cannot be read.
        class Poison:
            def __array__(self, *args, **kwargs):
                raise AssertionError('future labels read')
        with pytest.raises(w.Stop):
            w.create_reference(tmp_path / 'poison', dict(changed, y5=Poison()), c, b'x', 'b'*64)
    historical = changed['dates'] <= days[69]
    expected = w.context_features(changed, c)[historical]
    assert expected.tobytes() == before['C185'].tobytes()
    assert (parent / 'scores.bin').read_bytes() == b'original-score-bytes\x00\xff'


@pytest.mark.parametrize('bad', ['duplicate', 'overlap', 'wrong_parent', 'wrong_context', 'wrong_batch', 'dtype', 'labels', 'unsorted', 'inf', 'missing_session', 'empty'])
def test_append_rejects_bad_inputs_without_publishing(tmp_path, bad):
    a, c, days, parent = start(tmp_path)
    delta = slice_at(a, a['dates'] > days[69])
    pid = w.snapshot_id(parent)
    bid = w.batch_id(delta)
    if bad == 'duplicate':
        delta = {k: np.concatenate([v[:1], v]) for k, v in delta.items()}
    elif bad == 'overlap': delta = slice_at(a, a['dates'] >= days[69])
    elif bad == 'wrong_parent': pid = '0'*64
    elif bad == 'wrong_context': c = dict(c, contract='0'*64)
    elif bad == 'wrong_batch': bid = '0'*64
    elif bad == 'dtype': delta['X'] = delta['X'].astype('float64')
    elif bad == 'labels': delta['y5'] = np.zeros(len(delta['X']))
    elif bad == 'unsorted': delta = {k: v[::-1] for k, v in delta.items()}
    elif bad == 'inf': delta['X'][0, 0] = np.inf
    elif bad == 'missing_session': delta = slice_at(delta, delta['dates'] != days[72])
    elif bad == 'empty': delta = slice_at(delta, np.zeros(len(delta['X']), bool))
    out = tmp_path / 'bad'
    with pytest.raises(w.Stop):
        w.append_features(parent, out, delta, pid, c, bid, 'c'*64)
    assert not out.exists()


def test_repeat_and_interrupted_outputs_refused(tmp_path, monkeypatch):
    a, c, days, parent = start(tmp_path)
    delta = slice_at(a, a['dates'] > days[69])
    args = (parent, tmp_path/'out', delta, w.snapshot_id(parent), c, w.batch_id(delta), 'c'*64)
    real = w.save_array
    def interrupt(path, array):
        real(path, array)
        raise OSError('simulated interruption')
    monkeypatch.setattr(w, 'save_array', interrupt)
    with pytest.raises(OSError): w.append_features(*args)
    with pytest.raises(w.Stop): w.load_snapshot(tmp_path/'out')
    monkeypatch.setattr(w, 'save_array', real)
    with pytest.raises(FileExistsError): w.append_features(*args)
    args = (parent, tmp_path/'ok', *args[2:])
    w.append_features(*args)
    with pytest.raises(FileExistsError): w.append_features(*args)
    with pytest.raises(w.Stop):
        w.append_features(tmp_path/'ok', tmp_path/'again', delta, w.snapshot_id(tmp_path/'ok'), c, w.batch_id(delta), 'c'*64)


def test_sigkill_during_append_keeps_parent_and_rejects_partial(tmp_path):
    import select
    import signal
    import subprocess
    import sys
    a, c, days, parent = start(tmp_path)
    old_id = w.snapshot_id(parent)
    delta = slice_at(a, a['dates'] > days[69])
    out = tmp_path/'killed'
    incoming = tmp_path/'delta'; incoming.mkdir()
    for k, v in delta.items(): w.save_array(incoming/(k+'.npy'), v)
    script = '''
import signal, sys
from pathlib import Path
import numpy as np
from tools import forward_wp3 as w
parent, out, incoming = map(Path, sys.argv[1:])
w.ROOT = parent.parent
_, manifest = w.load_snapshot(parent)
delta = {k: np.load(incoming/(k+'.npy'), allow_pickle=False) for k in w.FIELDS}
real = w.save_array
def stop_after_first_write(path, values):
    real(path, values)
    print('partial_written', flush=True)
    signal.pause()
w.save_array = stop_after_first_write
w.append_features(parent, out, delta, w.snapshot_id(parent), manifest['context'], w.batch_id(delta), 'c'*64)
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(parent), str(out), str(incoming)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Generous bound: child interpreter start-up is slow when the full suite loads the machine.
        ready = select.select([process.stdout], [], [], 120)[0]
        if not ready or process.poll() is not None and not ready:
            process.kill(); _, err = process.communicate(timeout=10)
            raise AssertionError('writer did not reach first durable array; stderr tail: ' + err[-1500:])
        assert process.stdout.readline().strip() == 'partial_written'
    finally:
        process.kill()
        process.communicate(timeout=10)
    assert process.returncode == -signal.SIGKILL
    assert w.snapshot_id(parent) == old_id
    with pytest.raises(w.Stop): w.load_snapshot(out)
    with pytest.raises(FileExistsError):
        w.append_features(parent, out, delta, old_id, c, w.batch_id(delta), 'c'*64)


@pytest.mark.parametrize('file', ['X.npy', 'C185.npy', 'masks.npy', 'scores.bin', 'manifest.json'])
def test_corrupt_reference_refused(tmp_path, file):
    _, _, _, parent = start(tmp_path)
    p = parent/file
    data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    with pytest.raises(w.Stop): w.load_snapshot(parent)


def test_labels_are_separate_version_and_maturity_checked(tmp_path):
    a, _, days, parent = start(tmp_path)
    old, _ = w.load_snapshot(parent)
    n = len(old['X'])
    labels = dict(y5=np.zeros(n, dtype='float32'), y10=np.full(n, np.nan, dtype='float32'),
                  available_at=old['dates']+np.timedelta64(10, 'D'),
                  exit5=old['dates']+np.timedelta64(5, 'D'), exit10=old['dates']+np.timedelta64(10, 'D'))
    before = w.snapshot_id(parent)
    with pytest.raises(w.Stop): w.write_labels(parent, tmp_path/'early', labels, str(days[49]), 'd'*64)
    w.write_labels(parent, tmp_path/'labels-v1', labels, str(days[90]), 'd'*64)
    labels['y5'] += 1
    w.write_labels(parent, tmp_path/'labels-v2', labels, str(days[91]), 'e'*64)
    assert w.snapshot_id(parent) == before
    assert not list(parent.glob('y*.npy'))
    with pytest.raises(FileExistsError): w.write_labels(parent, tmp_path/'labels-v1', labels, str(days[91]), 'e'*64)


@pytest.mark.parametrize('bad', ['nan_timestamp', 'early_exit', 'future_publication', 'infinite', 'wrong_identity'])
def test_bad_labels_never_publish_or_change_features(tmp_path, bad):
    _, _, days, parent = start(tmp_path)
    a, _ = w.load_snapshot(parent)
    n = len(a['dates'])
    labels = dict(y5=np.zeros(n, dtype='float32'), y10=np.zeros(n, dtype='float32'),
                  available_at=a['dates']+np.timedelta64(10, 'D'),
                  exit5=a['dates']+np.timedelta64(5, 'D'), exit10=a['dates']+np.timedelta64(10, 'D'))
    source = 'd'*64
    if bad == 'nan_timestamp': labels['available_at'][0] = np.datetime64('NaT', 'D')
    elif bad == 'early_exit': labels['exit5'][0] = a['dates'][0]
    elif bad == 'future_publication': labels['available_at'][0] = days[-1]
    elif bad == 'infinite': labels['y10'][0] = np.inf
    elif bad == 'wrong_identity': source = 'UNKNOWN'
    before = w.snapshot_id(parent)
    with pytest.raises(w.Stop): w.write_labels(parent, tmp_path/'invalid-labels', labels, str(days[90]), source)
    assert not (tmp_path/'invalid-labels').exists()
    assert before == w.snapshot_id(parent)


def test_namespace_symlink_and_missing_mask_fail_closed(tmp_path):
    a, c, _ = fixture()
    with pytest.raises(w.Stop): w.create_reference(Path('/tmp/outside-wp3'), a, c, b'x', 'b'*64)
    link = tmp_path/'link'; link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(w.Stop): w.create_reference(link/'escape', a, c, b'x', 'b'*64)
    del a['masks']
    with pytest.raises(w.Stop): w.create_reference(tmp_path/'missing', a, c, b'x', 'b'*64)


def test_historical_c185_mutation_gate_refuses_publication(tmp_path, monkeypatch):
    a, c, days, parent = start(tmp_path)
    original = w.context_features
    def changed(a, context):
        result = original(a, context)
        result[0, 0] = np.nextafter(result[0, 0], np.float32(np.inf))
        return result
    monkeypatch.setattr(w, 'context_features', changed)
    delta = slice_at(a, a['dates'] > days[69])
    with pytest.raises(w.Stop, match='historical bytes changed'):
        w.append_features(parent, tmp_path/'mutant', delta, w.snapshot_id(parent), c, w.batch_id(delta), 'c'*64)
    assert not (tmp_path/'mutant').exists()


@pytest.mark.parametrize('field', ['X', 'masks', 'dates', 'stock_ids', 'splits', 'eligibility', 'C185'])
def test_prefix_gate_checks_every_field_and_dtype(tmp_path, field):
    _, _, _, parent = start(tmp_path)
    old, _ = w.load_snapshot(parent)
    changed = {k: a.copy() for k, a in old.items()}
    if field in ('X', 'C185'):
        changed[field] = changed[field].astype('float64')
    elif field == 'dates': changed[field][0] += np.timedelta64(1, 'D')
    elif field == 'stock_ids': changed[field][0] = 'Q'
    elif field == 'splits': changed[field][0] = 4
    else: changed[field].flat[0] = ~changed[field].flat[0]
    with pytest.raises(w.Stop): w.verify_prefix(old, changed)


def test_direct_c0plus_reference_without_labels(tmp_path):
    from types import SimpleNamespace
    a, c, _ = fixture()
    days, date_idx, counts = np.unique(a['dates'], return_inverse=True, return_counts=True)
    stocks, stock_idx = np.unique(a['stock_ids'], return_inverse=True)
    data = SimpleNamespace(X=a['X'], ends=np.cumsum(counts), n_stocks=len(stocks),
                           date_idx=date_idx, stock_idx=stock_idx)
    direct, names = w.c0plus.build_features(data, np.arange(len(a['X'])), c['feature_order'])
    assert len(names) == 185
    assert direct.tobytes() == w.context_features(a, c).tobytes()


def test_snapshot_records_producer_and_input_identity(tmp_path):
    a, c, days, parent = start(tmp_path)
    old, manifest = w.load_snapshot(parent)
    producer = manifest['producer']
    assert len(producer['source_commit']) == 64 or len(producer['source_commit']) == 40
    assert producer['implementation_sha256'] == w.file_hash(Path(w.__file__))
    assert producer['environment']['numpy'] == np.__version__
    assert manifest['input_batch_identity'] == w.batch_id({k: old[k] for k in w.FIELDS})
    assert manifest['context'] == c


@pytest.mark.parametrize('problem', ['complete', 'missing_mask', 'wrong_x'])
def test_readonly_full_inventory_does_not_guess_masks(tmp_path, problem):
    a, c, _ = fixture()
    source = tmp_path/'existing'; source.mkdir()
    for k in ('X', 'dates', 'stock_ids', 'masks'):
        w.save_array(source/(k+'.npy'), a[k])
    w.save_array(source/'end_indices.npy', np.cumsum(np.unique(a['dates'], return_counts=True)[1]))
    files = {p.name: w.file_hash(p) for p in source.iterdir()}
    (source/'manifest.json').write_text(json.dumps(dict(feature_order=c['feature_order'], files=files)))
    if problem == 'missing_mask': (source/'masks.npy').unlink()
    if problem == 'wrong_x':
        a['X'][0, 0] += 1
        np.save(source/'X.npy', a['X'])
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = w.audit_existing(source, tmp_path/'inventory')
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert result['labels_read'] is False
    if problem == 'complete': assert result['status'] == 'COMPLETE'
    elif problem == 'missing_mask':
        assert result['missing'] == ['masks.npy']
        assert 'masks.npy' not in result['records']
    else: assert result['conflicts'] == ['X.npy']
    with pytest.raises(FileExistsError): w.audit_existing(source, tmp_path/'inventory')


@pytest.mark.parametrize('bad', ['X_columns', 'X_dtype', 'mask_shape', 'stock_dtype', 'ends'])
def test_full_inventory_rejects_self_consistent_hashes_with_wrong_schema(tmp_path, bad):
    a, c, _ = fixture()
    a = {k: a[k] for k in ('X', 'dates', 'stock_ids', 'masks')}
    a['end_indices'] = np.cumsum(np.unique(a['dates'], return_counts=True)[1])
    if bad == 'X_columns': a['X'] = a['X'][:, :-1]
    elif bad == 'X_dtype': a['X'] = a['X'].astype('float64')
    elif bad == 'mask_shape': a['masks'] = a['masks'][:, :-1]
    elif bad == 'stock_dtype': a['stock_ids'] = a['stock_ids'].astype('U8')
    else: a['end_indices'][-1] -= 1
    source = tmp_path/'bad-schema'; source.mkdir()
    for k, value in a.items(): w.save_array(source/(k+'.npy'), value)
    files = {p.name: w.file_hash(p) for p in source.iterdir()}
    (source/'manifest.json').write_text(json.dumps(dict(feature_order=c['feature_order'], files=files)))
    result = w.audit_existing(source, tmp_path/'inventory')
    assert result['status'] == 'STOP_INCOMPLETE_REFERENCE'
    assert result['schema_conflicts']


def test_relocated_artifact_identity(tmp_path):
    from tools.forward_wp3 import resolve_artifact
    target=tmp_path/'validation.json';target.write_bytes(b'synthetic evidence')
    sha=__import__('hashlib').sha256(target.read_bytes()).hexdigest()
    record=dict(path='/missing/retired-worktree/validation.json',sha256=sha)
    relocation=tmp_path/'relocation.json'
    relocation.write_text(__import__('json').dumps(dict(artifacts={sha:'validation.json'})))
    assert resolve_artifact(record,relocation)==target.resolve()
    target.write_bytes(b'corrupt')
    with pytest.raises(w.Stop): resolve_artifact(record,relocation)
