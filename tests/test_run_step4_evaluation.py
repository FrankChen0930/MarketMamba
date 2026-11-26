"""Synthetic Step4 evaluation only; never accesses the running queue."""
import inspect
import json
import numpy as np
import pytest
from tools import run_step4_evaluation as e
from tools import step4 as s


def synthetic():
    days = np.array(['2015-01-05', '2015-01-06', '2021-01-05', '2021-01-06'], dtype='datetime64[D]')
    keys = np.array([(d, i) for d in range(4) for i in range(24)], dtype=np.int64)
    y = np.tile(np.arange(24, dtype=float), 4)
    rng = np.random.default_rng(42)
    predictions = {(a, seed): y + rng.normal(0, 12 if a == 'A' else 3, len(y))
                   for a in s.ARMS for seed in s.SEEDS}
    return dict(keys={k: keys.copy() for k in predictions}, predictions=predictions,
                labels=y, label_keys=keys.copy(), calendar=days)


def evaluate(x):
    return e.evaluate_family(**x)


def test_family_exactly_reuses_step3_and_shared_na():
    x = synthetic()
    result = evaluate(x)
    assert set(result['periods']) == {'2015-2020', '2021-2026'}
    for p in result['periods'].values():
        assert set(p) == {'1a', '1b'}
        assert all(v['valid_days'] == 2 for v in p.values())
    from scipy.stats import rankdata, spearmanr
    for arm in s.ARMS:
        ranks = np.mean([rankdata(x['predictions'][arm, seed][:24], method='average') / 24
                         for seed in s.SEEDS], axis=0)
        assert result['daily'][arm][0] == pytest.approx(spearmanr(ranks, x['labels'][:24]).statistic)
    x['predictions']['B1b', 17][:24] = 1
    x['predictions']['B1b', 29][:24] = 1
    x['predictions']['B1b', 43][:24] = 1
    result = evaluate(x)
    assert all(np.isnan(result['daily'][a][0]) for a in s.ARMS)
    assert result['reasons'][0] == 'CONSTANT_CROSS_SECTION'


def label_oracle(evaluator):
    x = synthetic()
    x['label_keys'] = x['label_keys'][::-1].copy()
    with pytest.raises(ValueError, match='STOP.*label'):
        evaluator(**x)


def test_label_alignment_and_source_mutation():
    label_oracle(e.evaluate_family)
    source = inspect.getsource(e.evaluate_family)
    changed = source.replace("label_keys=label_keys,", "label_keys=keys['A', 17],")
    assert changed != source
    ns = dict(e.__dict__)
    exec(compile(changed, '<label-alignment-mutant>', 'exec'), ns)
    with pytest.raises(pytest.fail.Exception):
        label_oracle(ns['evaluate_family'])


def holm_oracle(adjust):
    p = adjust({'1a': .026, '1b': .03})
    assert p['1a'] == pytest.approx(.052)
    assert p['1b'] == pytest.approx(.052)
    assert adjust({'1a': .001, '1b': 1}) == {'1a': .002, '1b': 1.0}


def test_holm_and_source_mutation():
    holm_oracle(e.holm)
    source = inspect.getsource(e.holm)
    changed = source.replace('(2 - i) * p[name]', 'p[name]')
    assert changed != source
    ns = dict(e.__dict__)
    exec(compile(changed, '<holm-mutant>', 'exec'), ns)
    with pytest.raises(AssertionError):
        holm_oracle(ns['holm'])


def test_bootstrap_centered_two_sided_and_empty_stop():
    x = np.column_stack([np.arange(43) / 100 + .01, -np.arange(43) / 100 - .01])
    actual = e.bootstrap(x)
    rng = np.random.default_rng(17)
    means = []
    for _ in range(1000):
        idx = ((rng.integers(0, len(x), 3)[:, None] + np.arange(20)) % len(x)).ravel()[:len(x)]
        means.append(x[idx].mean(axis=0))
    means = np.asarray(means)
    expected_p = (1 + (np.abs(means - x.mean(axis=0)) >= np.abs(x.mean(axis=0))).sum(axis=0)) / 1001
    assert np.allclose(actual['p'], expected_p)
    assert np.allclose(actual['ci95'], np.percentile(means, [2.5, 97.5], axis=0).T)
    with pytest.raises(ValueError, match='STOP'):
        e.bootstrap(np.full((3, 2), np.nan))
    sparse = np.full((100, 2), np.nan); sparse[0] = [.1, .2]
    with pytest.raises(ValueError, match='STOP empty bootstrap replicate'):
        e.bootstrap(sparse)


@pytest.mark.parametrize('mutation', ['seed', 'keys', 'nonfinite', 'label_shape', 'small'])
def test_support_fail_closed(mutation):
    x = synthetic()
    if mutation == 'seed':
        del x['predictions']['A', 43]
    elif mutation == 'keys':
        x['keys']['B1b', 43] = x['keys']['B1b', 43][::-1]
    elif mutation == 'nonfinite':
        x['predictions']['B1a', 29][0] = np.nan
    elif mutation == 'label_shape':
        x['labels'] = x['labels'][:-1]
    else:
        x['labels'][np.tile(np.arange(24) >= 19, 4)] = np.nan
    with pytest.raises(ValueError, match='STOP'):
        evaluate(x)


def test_crosscheck_equal_and_difference_stop():
    x = synthetic()
    third = dict(x, keys={('B', seed): x['keys']['A', seed] for seed in s.SEEDS},
                 predictions={('B', seed): x['predictions']['A', seed].copy() for seed in s.SEEDS})
    assert e.crosscheck(x, third)['status'] == 'CONSISTENT'
    third['predictions']['B', 17][0] += .1
    with pytest.raises(ValueError, match='STOP.*baseline'):
        e.crosscheck(x, third)


def test_missing_job_stops_before_loading_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(e.r, 'require_admission', lambda _: {'jobs': {name: {} for name in e.r.queue()}})
    monkeypatch.setattr(e.r, 'resume', lambda *args: (_ for _ in ()).throw(ValueError('STOP incomplete')))
    monkeypatch.setattr(np, 'load', lambda *args, **kw: pytest.fail('array loaded before complete gate'))
    with pytest.raises(ValueError, match='STOP incomplete'):
        e.load_inputs(tmp_path, tmp_path / 'fits')


def test_pending_templates_and_no_implicit_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    public = tmp_path / 'reports/step4_results_20261127.md'
    public.parent.mkdir()
    public.write_text(e.render_report(e.report_template()))
    public.with_suffix('.json').write_text(json.dumps(e.report_template()))
    monkeypatch.setattr(e, 'load_inputs', lambda *args: pytest.fail('read queue before placeholder gate'))
    with pytest.raises(ValueError, match='STOP.*consume-placeholder'):
        e.run(tmp_path, tmp_path / 'fits', tmp_path, tmp_path / 'decision',
              tmp_path / '.artifacts/step4/evaluation', public)


@pytest.fixture
def complete_jobs(tmp_path, monkeypatch):
    """108 tiny native manifests; opaque fake model bytes, zero fitting."""
    monkeypatch.setattr(s, 'ROOT', tmp_path)
    directory = tmp_path / 'bundle'; directory.mkdir()
    fits = tmp_path / '.artifacts/step4/fits'
    files = {}; jobs = {}
    def save(ref, value):
        path = directory / (ref + '.npy')
        np.save(path, value)
        files[ref] = dict(path=path.name, sha256=s.sha256(path), bytes=path.stat().st_size)
        return ref
    calendar = np.array([f'{year}-01-{day:02d}' for year in s.YEARS for day in (5, 6)], dtype='datetime64[D]')
    dates = np.repeat(calendar, 24); stock = np.tile(np.arange(24), len(calendar))
    save('calendar', calendar); save('matrix_dates', dates)
    save('stock_index', stock); save('matrix_ends', np.arange(24, len(dates)+1, 24))
    save('splits', np.ones(len(dates), dtype=np.int8))
    rng = np.random.default_rng(9)
    bundle = dict(jobs=jobs, files=files, source_commit='SYNTHETIC')
    for yi, year in enumerate(s.YEARS):
        rows = np.arange(yi*48, (yi+1)*48)
        keys = np.column_stack([np.repeat([yi*2, yi*2+1], 24), np.tile(np.arange(24), 2)])
        values = dict(sample=rows[:4], target=np.arange(4, dtype=np.float32), keys=keys,
                      fit=np.ones((4,190), dtype=np.float32), eval=np.ones((48,190), dtype=np.float32),
                      labels=np.tile(np.arange(24, dtype=float), 2), fit_keys=keys[:4], eval_rows=rows,
                      fit_extra=np.ones((4,5), dtype=np.float32), eval_extra=np.ones((48,5), dtype=np.float32))
        refs = {k: save(f'{k}_{year}', v) for k,v in values.items()}
        for seed in s.SEEDS:
            for arm in s.ARMS:
                name = s.job_id(year, seed, arm)
                job = dict(year=year, seed=seed, arm=arm, sample_file=refs['sample'], target_file=refs['target'],
                    keys_file=refs['keys'], fit_base=refs['fit'], eval_base=refs['eval'],
                    label_keys_file=refs['keys'], eval_labels_file=refs['labels'], splits_file='splits',
                    fit_keys=refs['fit_keys'], eval_rows=refs['eval_rows'], fit_extra=refs['fit_extra'],
                    eval_extra=refs['eval_extra'], source_array_identities={k: e.r.array_digest(values[k])
                    for k in ('sample','target','keys','fit','eval')})
                jobs[name] = job
                request = e._expected_request(directory, bundle, job,
                    dict(device='cpu', python='synthetic', executable='synthetic', numpy='synthetic', lightgbm='synthetic'))
                pred = values['labels'] + rng.normal(0, 8 if arm == 'A' else 2, 48)
                # Fixture creation has no durability claim; real resume hashes
                # and validates every byte. Avoid 756 redundant fsync cycles.
                folder = fits / name; folder.mkdir(parents=True)
                for filename, value in [('keys.npy', keys), ('sample.npy', values['sample']), ('prediction.npy', pred)]:
                    np.save(folder / filename, value)
                (folder / 'model.txt').write_bytes(b'opaque synthetic model')
                outputs = {p.name: dict(sha256=s.sha256(p), bytes=p.stat().st_size) for p in folder.iterdir()}
                (folder / 'manifest.json').write_text(json.dumps(dict(state='COMPLETE', request=request,
                    request_sha256=e.r.digest_json(request), files=outputs)))
    monkeypatch.setattr(e.r, 'require_admission', lambda _: bundle)
    return directory, fits, bundle


def test_complete_108_loader_and_publication(complete_jobs, monkeypatch):
    directory, fits, bundle = complete_jobs
    inputs = e.load_inputs(directory, fits)
    assert len(inputs['manifests']) == 108
    assert len(inputs['labels']) == 576
    assert all(v['validation'] == 'COMPLETE_VERIFIED' for v in inputs['manifests'])
    third = dict(inputs, keys={('B', seed): inputs['keys']['A', seed] for seed in s.SEEDS},
                 predictions={('B', seed): inputs['predictions']['A', seed] for seed in s.SEEDS},
                 admission_sha256='synthetic', acceptance={'synthetic': True})
    monkeypatch.setattr(e.prior, 'load_inputs', lambda *args: third)
    public = s.ROOT / 'reports/step4_results_20261127.md'; public.parent.mkdir()
    public.write_text(e.render_report(e.report_template()))
    public.with_suffix('.json').write_text(json.dumps(e.report_template()))
    output = s.ROOT / '.artifacts/step4/evaluation'
    (directory / 'bundle.json').write_text('synthetic')
    result = e.run(directory, fits, directory, directory/'decision', output, public, consume_placeholder=True)
    assert result['status'] == 'COMPLETE_HISTORICAL_ONLY'
    assert result['complete_verified'] == 108 and result['formal_success'] is False
    manifest = json.loads((output/'manifest.json').read_text())
    for name, info in manifest['outputs'].items():
        assert s.sha256(output/name) == info['sha256']
    for path, info in manifest['public_outputs'].items():
        assert s.sha256(path) == info['sha256']
    assert json.loads(public.with_suffix('.json').read_text()) == result
    with pytest.raises(ValueError, match='STOP pending'):
        e.run(directory, fits, directory, directory/'decision', output/'repeat', public, consume_placeholder=True)


@pytest.mark.parametrize('mutation', ['missing', 'corrupt', 'request', 'source_sha', 'label_keys'])
def test_108_artifact_mutations_stop(complete_jobs, mutation):
    directory, fits, bundle = complete_jobs
    name = e.r.queue()[-1]; path = fits / name
    if mutation == 'missing':
        (path/'prediction.npy').unlink()
    elif mutation == 'corrupt':
        (path/'model.txt').write_bytes(b'corrupt')
    elif mutation == 'request':
        m = json.loads((path/'manifest.json').read_text())
        m['request']['seed'] = 99
        m['request_sha256'] = e.r.digest_json(m['request'])
        (path/'manifest.json').write_text(json.dumps(m))
    elif mutation == 'source_sha':
        bundle['jobs'][name]['source_array_identities']['fit'] = '0'*64
    else:
        ref = 'wrong_labels'
        keys = e._load(directory, bundle, bundle['jobs'][name]['keys_file'])[::-1].copy()
        np.save(directory/(ref+'.npy'), keys)
        bundle['files'][ref] = dict(path=ref+'.npy')
        bundle['jobs'][name]['label_keys_file'] = ref
    with pytest.raises(ValueError, match='STOP'):
        e.load_inputs(directory, fits)
