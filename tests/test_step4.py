"""Step4 engineering contracts; synthetic fixtures, no fits or label statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import step4 as s
from tools import step4_prepare as prep
from tools import step4_audit as audit


def peers():
    return pd.DataFrame(dict(
        date=pd.to_datetime(['2015-01-05'] * 5), stock_id=list('ABCDE'),
        eligible=[True, True, True, False, True],
        industry=['X', 'X', None, 'X', 'X'],
        peer_return=[.1, np.nan, np.nan, np.nan, .2],
        return_peer_count=[3, 2, 0, 0, 4], market_return=[.02] * 5,
        revenue_spillover=[.3, np.nan, np.nan, np.nan, .5],
        revenue_peer_count=[3, 4, 0, 0, 5],
        target_revenue_available=[False, True, False, False, False]))


def test_frozen_contract_and_widths():
    assert len(s.BASELINE_FEATURES) == 190
    assert len(s.FEATURE_ORDER['B1a']) == 193
    assert len(s.FEATURE_ORDER['B1b']) == 192
    assert s.FEATURE_ORDER['A'] == s.BASELINE_FEATURES
    assert s.HOLM_M == 2 and s.ARMS == ('A', 'B1a', 'B1b')
    assert s.SEEDS == (17, 29, 43) and s.YEARS == tuple(range(2015, 2027))
    assert s.frozen_decision()['sha256'] == s.DECISION_SHA


def assert_unknown_counts(normalizer):
    x = normalizer(peers())
    assert np.isnan(x[2, [1, 4]]).all()
    assert np.isnan(x[3, [1, 4]]).all()
    assert x[1, 1] == 2 and x[1, 4] == 4
    assert np.isnan(x[1, 3])  # Own announcement removes value, retains known count.


def test_unknown_counts_are_nan_known_sparse_counts_retained():
    assert_unknown_counts(s.normalize_peer)


def test_mutation_unknown_as_zero_killed():
    def mutant(f):
        return f[list(s.EXTRA_FEATURES)].to_numpy(np.float32)
    with pytest.raises(AssertionError):
        assert_unknown_counts(mutant)


def test_duplicate_or_invalid_peer_values_stop():
    f = peers()
    with pytest.raises(ValueError, match='duplicate'):
        s.normalize_peer(pd.concat([f, f.iloc[:1]]))
    f.loc[0, 'return_peer_count'] = 2
    with pytest.raises(ValueError, match='minimum peers'):
        s.normalize_peer(f)


def test_keyed_left_join_keeps_missing_and_input_order():
    f = peers().sample(frac=1, random_state=3)
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(['2015-01-05'] * 3), ['E', 'A', 'Z']])
    x = prep.align_peer(f, keys)
    assert x.shape == (3, 5)
    assert x[0, 0] == pytest.approx(.2)
    assert x[1, 0] == pytest.approx(.1)
    assert np.isnan(x[2]).all()


def assert_key_oracle(aligner):
    f = peers().iloc[[4, 0]]
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(['2015-01-05'] * 2), ['A', 'E']])
    result = aligner(f, keys)
    np.testing.assert_allclose(result[:, 0], [.1, .2])


def test_mutation_positional_join_killed():
    assert_key_oracle(prep.align_peer)
    with pytest.raises(AssertionError):
        assert_key_oracle(lambda f, k: s.normalize_peer(f))


def test_future_rows_do_not_change_past_join():
    f = peers()
    future = f.copy(); future['date'] += pd.Timedelta(days=10)
    future['peer_return'] *= 999
    keys = pd.MultiIndex.from_frame(f[['date', 'stock_id']])
    np.testing.assert_array_equal(prep.align_peer(f, keys), prep.align_peer(pd.concat([future, f]), keys))


def test_sparse_support_no_invented_coverage_threshold():
    x = np.full((100, 2), np.nan, np.float32)
    x[:2] = [[.1, 3], [.2, 4]]
    stats = audit.feature_support(x, s.FEATURES_1B)
    assert stats[0]['finite'] == 2 and stats[0]['nan_fraction'] == .98
    x[1, 0] = .1
    with pytest.raises(ValueError, match='support'):
        audit.feature_support(x, s.FEATURES_1B)


def test_admission_requires_exact_human_record(tmp_path):
    bad = tmp_path / 'decision.md'; bad.write_text('Human approved')
    with pytest.raises(ValueError, match='identity'):
        s.frozen_decision(bad)


def test_component_matrix_no_label_reads(tmp_path, monkeypatch):
    base = np.arange(760, dtype=np.float32).reshape(4, 190)
    extra = np.arange(20, dtype=np.float32).reshape(4, 5)
    np.save(tmp_path / 'base.npy', base); np.save(tmp_path / 'extra.npy', extra)
    bundle = {'files': {'base': {'path': 'base.npy'}, 'extra': {'path': 'extra.npy'}}}
    job = {'fit_base': 'base', 'fit_extra': 'extra', 'arm': 'B1b'}
    x = s.matrix(tmp_path, bundle, job, 'fit')
    assert x.shape == (4, 192)
    np.testing.assert_array_equal(x[:, :190], base)
    np.testing.assert_array_equal(x[:, 190:], extra[:, 3:])


def test_disk_estimate_fail_closed():
    assert prep.check_disk_budget(1, 10) == 1
    with pytest.raises(ValueError, match='60 GB'):
        prep.check_disk_budget(60_000_000_001, 100_000_000_000)
    with pytest.raises(ValueError, match='free space'):
        prep.check_disk_budget(10, 1)


def test_output_namespace_no_overwrite_or_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    out = tmp_path / '.artifacts/step4/bundle'
    assert s.new_output(out) == out
    with pytest.raises(ValueError, match='exists'):
        s.new_output(out)
    link = tmp_path / '.artifacts/step4/link'; link.symlink_to(out, target_is_directory=True)
    with pytest.raises(ValueError):
        s.new_output(link)


@pytest.fixture
def engineering_bundle(tmp_path, monkeypatch):
    """Small component bundle exercises the real label-blind audit.

    Original Step3 schema is tested in test_step3; only that delegated schema
    and source pins are substituted. All Step4 bytes, axes and gates run.
    """
    import shutil
    from tools import step3
    real_root = s.ROOT
    repo = tmp_path / 'repo'
    (repo / 'tools').mkdir(parents=True)
    for prefix in ('step3*.py', 'step4*.py'):
        for path in (real_root / 'tools').glob(prefix):
            shutil.copyfile(path, repo / 'tools' / path.name)
    monkeypatch.setattr(s, 'ROOT', repo)
    tmp_path = repo / '.artifacts/step4/bundle'
    tmp_path.mkdir(parents=True)
    monkeypatch.setattr(s, 'YEARS', (2015,))
    monkeypatch.setattr(s, 'SEEDS', (17,))
    monkeypatch.setattr(step3, 'validate_bundle_schema', lambda b: None)
    files = {}

    def save(ref, value):
        path = tmp_path / (ref + '.npy')
        np.save(path, value, allow_pickle=False)
        files[ref] = dict(path=path.name, sha256=s.sha256(path), bytes=path.stat().st_size)
        return ref

    def textfile(ref, value):
        path = tmp_path / (ref + '.json')
        s.write_json(path, value)
        files[ref] = dict(path=path.name, sha256=s.sha256(path), bytes=path.stat().st_size)
        return files[ref]['sha256']

    sessions = pd.DatetimeIndex([pd.Timestamp(f'{year}-01-05') + pd.Timedelta(days=i)
                                for year in range(2013, 2027) for i in range(2)])
    dates = np.repeat(sessions.to_numpy(), 3)
    stock = np.tile(np.arange(3, dtype=np.int32), len(sessions))
    di = np.repeat(np.arange(len(sessions)), 3)
    ids = np.array(list('ABC'))[stock]
    save('matrix_dates', dates); save('matrix_ends', np.arange(3, len(dates) + 1, 3))
    save('stock_index', stock); save('membership', ids)
    # A full increasing session axis is enough to check exact purge arithmetic.
    save('calendar', pd.date_range('2013-01-01', '2026-12-31').to_numpy())
    textfile('feature_order', {'feature_order': list(s.BASE48)})
    block = np.column_stack([np.arange(len(dates)) * .01, stock + 3,
                             np.arange(len(dates)) * .001, np.arange(len(dates)) * .02, stock + 4]).astype(np.float32)
    save('peer_block', block)
    outputs = {}
    for year in range(2013, 2027):
        take = (pd.DatetimeIndex(dates).year == year)
        f = pd.DataFrame(block[take], columns=s.EXTRA_FEATURES)
        f['date'] = dates[take]; f['stock_id'] = ids[take]
        f['eligible'] = True; f['industry'] = 'X'; f['target_revenue_available'] = False
        name = f'peer-daily-{year}.parquet'
        path = tmp_path / name; f.to_parquet(path, index=False)
        info = dict(path=name, sha256=s.sha256(path), bytes=path.stat().st_size)
        files['peer_' + name] = info
        outputs[name] = {k: info[k] for k in ('sha256', 'bytes')}
    monkeypatch.setattr(s, 'PEER_MANIFEST_SHA', textfile('peer_manifest', {'outputs': outputs}))
    monkeypatch.setattr(s, 'PEER_REVIEW_SHA', textfile('peer_review', {'synthetic': True}))
    for ref, path in [('spec', s.SPEC), ('decision', s.DECISION)]:
        dst = tmp_path / path.name; shutil.copyfile(path, dst)
        files[ref] = dict(path=dst.name, sha256=s.sha256(dst), bytes=dst.stat().st_size)
    er = np.flatnonzero(pd.DatetimeIndex(dates).year == 2015)
    rows = np.arange(6)
    save('sample', rows); save('target', np.arange(6, dtype=np.float32))
    save('keys', np.column_stack([di[er], stock[er]]))
    save('label_keys', np.column_stack([di[er], stock[er]]))
    save('eval_labels', np.full(len(er), np.nan))
    save('splits', np.ones(len(dates), np.int8))
    save('fit_base', np.ones((len(rows), 190), np.float32))
    save('eval_base', np.ones((len(er), 190), np.float32))
    save('fit_extra', block[rows]); save('eval_extra', block[er])
    save('fit_keys', np.column_stack([di[rows], stock[rows]])); save('eval_rows', er)
    original = dict(year=2015, seed=17, arm='B')
    for role in ('sample', 'target', 'keys', 'splits', 'label_keys', 'eval_labels'):
        original[role + '_file'] = role
    original.update(fit_file='fit_base', eval_file='eval_base')
    from tools.step3_runner import array_digest
    for role in ('sample', 'target', 'keys', 'fit', 'eval'):
        original[role + '_sha256'] = array_digest(np.load(tmp_path / files[original[role + '_file']]['path']))
    source = dict(files=dict(files), jobs={'fold2015-B-seed17': original})
    monkeypatch.setattr(s, 'STEP3_BUNDLE_SHA', textfile('step3_manifest', source))
    implementation = {}
    for prefix in ('step3*.py', 'step4*.py'):
        for path in (s.ROOT / 'tools').glob(prefix):
            dst = tmp_path / ('code_' + path.name); shutil.copyfile(path, dst)
            info = dict(path=dst.name, sha256=s.sha256(dst), bytes=dst.stat().st_size)
            files['code_' + path.name] = info
            implementation[str(path.relative_to(s.ROOT))] = info
    jobs = {}
    for arm in s.ARMS:
        jobs[s.job_id(2015, 17, arm)] = dict(year=2015, seed=17, arm=arm,
            **{r + '_file': original[r + '_file'] for r in ('sample', 'target', 'keys', 'splits', 'label_keys', 'eval_labels')},
            fit_base='fit_base', eval_base='eval_base', fit_extra='fit_extra', eval_extra='eval_extra',
            fit_keys='fit_keys', eval_rows='eval_rows',
            source_array_identities={r: original[r + '_sha256'] for r in ('fit', 'eval', 'sample', 'target', 'keys')})
    bundle = dict(schema='step4-component-bundle-v1', state='PREPARED_NOT_ADMITTED',
                  contract=s.contract(), files=files, jobs=jobs, implementation=implementation,
                  source_identities={'synthetic': True})
    s.write_json(tmp_path / 'bundle.json', bundle)
    return tmp_path, bundle


def test_full_admission_without_loading_target_or_eval_labels(engineering_bundle, monkeypatch):
    directory, bundle = engineering_bundle
    real_load = np.load
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in ('target.npy', 'eval_labels.npy'), 'labels loaded by admission'
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    result = audit.admission(directory)
    assert result['status'] == 'ADMIT', result['stops']
    assert result['jobs'] == 3 and result['fit_count'] == 0
    assert len(result['future_invariance']) == 3


@pytest.mark.parametrize('mutation', ['missing_job', 'wrong_contract', 'corrupt_target', 'wrong_keys', 'missing_code'])
def test_full_admission_mutations_stop(engineering_bundle, mutation):
    directory, bundle = engineering_bundle
    if mutation == 'missing_job':
        bundle['jobs'].pop('fold2015-B1b-seed17')
    elif mutation == 'wrong_contract':
        bundle['contract']['bootstrap']['holm_m'] = 1
    elif mutation == 'corrupt_target':
        (directory / 'target.npy').write_bytes(b'corrupt opaque target')
    elif mutation == 'wrong_keys':
        # A self-consistent local file still cannot replace frozen source keys.
        np.save(directory / 'keys.npy', np.zeros((6, 2), np.int64))
        bundle['files']['keys'].update(sha256=s.sha256(directory / 'keys.npy'))
    elif mutation == 'missing_code':
        bundle['implementation'].pop('tools/step4_audit.py')
    (directory / 'bundle.json').write_text(json.dumps(bundle))
    result = audit.admission(directory)
    assert result['status'] == 'STOP' and result['stops']
