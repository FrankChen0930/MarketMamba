import json
import numpy as np
import pytest
from research_lab import colab_runner as cr
from research_lab.colab_runner import IdentityError

FEATURES = ['f%d' % i for i in range(48)]
FEATURES[6], FEATURES[7], FEATURES[30] = 'Return_5d', 'Return_20d', 'Volatility_20d'


def build(root, name='mat', days=130, names=30, seed=0, train_end=90, eval_start=95, label_shift=0.):
    d = root/name
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    n = days*names
    day = np.repeat(np.arange(days), names)
    X = rng.normal(size=(n, 48)).astype(np.float32)
    y5 = (.5*X[:, 10] + label_shift + rng.normal(size=n)).astype(np.float32)
    arrays = dict(X=X, y5=y5, y10=y5.copy(),
                  dates=(np.datetime64('2020-01-01') + day.astype('timedelta64[D]')),
                  splits=np.where(day <= train_end, 1, np.where(day >= eval_start, 2, 4)).astype(np.uint8),
                  end_indices=np.arange(1, days+1, dtype=np.int64)*names)
    for k, v in arrays.items():
        np.save(d/f'{k}.npy', v)
    files = {f'{k}.npy': cr.sha256_file(d/f'{k}.npy') for k in arrays}
    manifest = dict(rows=n, feature_order=FEATURES, schema_version='t-v1', logical_identity=name*8,
                    label_contract=dict(evidence_class='SYNTH'), files=files)
    (d/'manifest.json').write_text(json.dumps(manifest))
    reg = dict(rows=n, logical_identity=name*8, schema='t-v1', files=files,
               train_end=str(np.datetime64('2020-01-01')+np.timedelta64(train_end,'D')), eval_start=str(np.datetime64('2020-01-01')+np.timedelta64(eval_start,'D')))
    return d, reg


@pytest.fixture
def env(tmp_path):
    d, reg = build(tmp_path/'src')
    d2, reg2 = build(tmp_path/'src', name='other', seed=1)
    return dict(src=d, reg={'E5': reg, 'E1': reg2}, out=tmp_path/'out', tmp=tmp_path)


def go(env, **kw):
    logs = []
    kw.setdefault('smoke', True)
    kw.setdefault('n_boot', 50)
    r = cr.run_experiment(env['src'], env['out'], 'E5', registry=env['reg'], log=logs.append, **kw)
    return r, logs


def test_full_run_writes_all_artifacts(env):
    r, _ = go(env)
    root = next(env['out'].iterdir())
    assert root.name.startswith('E5_y5_smoke_')
    for f in ('identity.json', 'config.json', 'result.json'):
        assert (root/f).exists()
    assert {p.stem for p in (root/'daily_ic').glob('*.csv')} == set(cr.MODELS)
    assert r['models']['ridge']['ic_mean'] > .2 and abs(r['models']['reversal_ret5']['ic_mean']) < .1
    assert r['evidence_tier'] == 'RESEARCH_ONLY' and r['semantics']['evaluation'][0] >= '2020-04-05'
    assert {x['model'] for x in r['vs_reversal_ret5']} == {'momentum_ret20', 'lowvol_20d', 'ridge', 'lightgbm'}
    assert all('p_hac_holm' in x for x in r['vs_reversal_ret5']) and r['lightgbm_vs_ridge']
    assert set(r['code']) == set(cr.CODE_FILES)


def test_rerun_skips_and_partial_resume_only_reruns_missing(env):
    r1, _ = go(env)
    r2, logs = go(env)
    assert any('skip' in l for l in logs) and r2 == r1
    root = next(env['out'].iterdir())
    (root/'result.json').unlink()
    (root/'models'/'ridge.json').unlink()
    r3, logs = go(env)
    assert [l for l in logs if l.startswith('[done]')][0].startswith('[done] ridge')
    assert sum(l.startswith('[done]') for l in logs) == 1 and sum('[resume]' in l for l in logs) == 4
    assert r3['models']['lightgbm']['ic_mean'] == r1['models']['lightgbm']['ic_mean']


def test_different_config_gets_separate_dir_and_never_overwrites(env):
    go(env)
    go(env, seed=29)
    assert len(list(env['out'].iterdir())) == 2


def test_wrong_experiment_is_rejected(env):
    with pytest.raises(IdentityError, match='not E1'):
        cr.run_experiment(env['src'], env['out'], 'E1', registry=env['reg'], smoke=True, log=lambda *_: None)
    assert not list(env['out'].glob('*/result.json'))


def test_corrupted_array_detected_and_partial_removed(env):
    y = np.load(env['src']/'y5.npy')
    y[10] += 1
    np.save(env['src']/'y5.npy', y)
    with pytest.raises(IdentityError, match='y5.npy'):
        go(env)
    assert not list(env['out'].rglob('*.part'))


def test_semantic_contract_violation_detected(tmp_path):
    d, reg = build(tmp_path)
    sp = np.load(d/'splits.npy')
    sp[-1] = 1  # a train row after the evaluation start
    np.save(d/'splits.npy', sp)
    with pytest.raises(IdentityError, match='train'):
        cr.check_semantics(d, 'y5', 'E5', {'E5': reg})
    d2, reg2 = build(tmp_path, name='x2')
    reg2['train_end'] = '2020-01-10'  # claims earlier train end than the data has
    with pytest.raises(IdentityError, match='train rows after'):
        cr.check_semantics(d2, 'y5', 'E5', {'E5': reg2})


def test_memory_plan_flags_insufficient_memory(env):
    assert cr.memory_plan(env['src'], 'y5', 1000, available_bytes=10**9)['ok']
    assert not cr.memory_plan(env['src'], 'y5', 1000, available_bytes=1000)['ok']


def test_e5_and_e1_registry_distinct_where_expected():
    a, b = cr.REGISTRY['E5'], cr.REGISTRY['E1']
    assert a['logical_identity'] != b['logical_identity']
    same = {k for k in a['files'] if a['files'][k] == b['files'][k]}
    assert same == {'X.npy', 'dates.npy', 'end_indices.npy', 'stock_ids.npy'}
    assert all(len(h) == 64 for r in (a, b) for h in r['files'].values())


def test_find_matrix_by_identity_not_path(tmp_path):
    a, reg_a = build(tmp_path/'drive/E5root/output/E5root', name='mat')
    b, reg_b = build(tmp_path/'drive/E1root/matrix/v1', name='other', seed=1)
    reg = {'E5': reg_a, 'E1': reg_b}
    roots = [tmp_path/'drive/E5root', tmp_path/'drive/E1root', tmp_path/'missing']
    assert cr.find_matrix('E5', roots, reg) == a and cr.find_matrix('E1', roots, reg) == b
    with pytest.raises(FileNotFoundError, match='no E5 matrix'):
        cr.find_matrix('E5', [tmp_path/'drive/E1root'], reg)                      # E1's matrix is never accepted as E5
