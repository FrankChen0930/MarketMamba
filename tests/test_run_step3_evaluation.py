"""Synthetic-only evaluator driver acceptance and deliberate mutation checks."""
import json
import shutil

import numpy as np
import pytest

from tools import step3 as s
from tools import step3_runner as r
from tools import run_step3_evaluation as driver


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    # Only original-source provenance is replaced: all bundle/output hashes,
    # schema, Human record and request identities exercise real validators.
    monkeypatch.setattr(s, 'verify_source_provenance', lambda b: None)
    monkeypatch.setattr(s, 'private_dir', lambda p: _mkdir(p))
    monkeypatch.setattr(r, 'private_dir', lambda p: _mkdir(p))
    days = np.array([f'{y}-01-04' for y in s.YEARS], dtype='datetime64[D]')
    dates = np.repeat(days, 20)
    files = {}
    def save(name, a):
        p = tmp_path / (name + '.npy')
        np.save(p, a, allow_pickle=False)
        files[name] = dict(path=str(p), sha256=s.sha256(p), bytes=p.stat().st_size)
        return name
    save('calendar', np.sort(np.concatenate([days, days - np.timedelta64(1, 'D')])))
    save('matrix_dates', dates)
    save('matrix_ends', np.arange(20, 241, 20))
    save('stock_index', np.tile(np.arange(20), 12))
    save('labels', np.tile(np.arange(20.), 12))
    save('splits', np.ones(240, dtype=int))
    save('dummy', np.ones(1))
    names = s.BUNDLE_CATEGORIES | {'price_manifest', 'finmind_revenue', 'source_manifest'}
    names |= {'code_' + p.name for p in (s.ROOT / 'tools').glob('step3*.py') if p.name != 'step3.py'}
    for name in names:
        files.setdefault(name, files['dummy'].copy())
    jobs = {}
    arrays = {}
    for i, year in enumerate(s.YEARS):
        keys = np.column_stack([np.repeat(i, 20), np.arange(20)])
        for seed in s.SEEDS:
            for arm in 'AB':
                name = s.job_id(year, seed, arm)
                values = dict(keys=keys, label_keys=keys.copy(), eval_labels=np.arange(20.),
                              sample=np.arange(3), target=np.arange(3.), fit=np.ones((3, 1)), eval=np.ones((20, 1)))
                job = dict(year=year, seed=seed, arm=arm, splits_file='splits')
                for role, a in values.items():
                    job[role + '_file'] = save(name + '_' + role, a)
                    if role not in ('label_keys', 'eval_labels'):
                        job[role + '_sha256'] = r.array_digest(a)
                jobs[name] = job
                arrays[name] = values
    bundle = dict(contract=s.SPEC_SHA, source_commit='synthetic', source_gate='HUMAN_DECISION_REQUIRED',
                  files=files, jobs=jobs, source_identities=files.copy())
    dump(tmp_path / 'bundle.json', bundle)
    decision = tmp_path / 'human.json'
    dump(decision, dict(decision_date='2026-11-26', decision='ACCEPT_HISTORICAL_SOURCE_GATE', decision_text='Synthetic fixture'))
    bp = tmp_path / 'bundle.json'
    gate = dict(contract_sha256=s.SPEC_SHA, status='HUMAN_DECISION_REQUIRED', bundle=dict(path=str(bp), sha256=s.sha256(bp), bytes=bp.stat().st_size))
    _mkdir(tmp_path / 'admission')
    dump(tmp_path / 'admission/admission.json', gate)
    for name, job in jobs.items():
        request = driver.expected_request(bundle, gate, job, s.human_source_acceptance(decision))
        v = arrays[name]
        prediction = np.arange(20.) if job['arm'] == 'B' or job['year'] >= 2021 else -np.arange(20.)
        folder = _mkdir(tmp_path / 'fits' / name)
        for filename, value in {'keys.npy':v['keys'], 'sample.npy':v['sample'], 'prediction.npy':prediction}.items():
            np.save(folder / filename, value, allow_pickle=False)
        (folder / 'model.txt').write_bytes(b'synthetic model')
        info = {p.name:dict(sha256=s.sha256(p), bytes=p.stat().st_size) for p in folder.iterdir()}
        dump(folder / 'manifest.json', dict(state='COMPLETE', request=request, request_sha256=r.digest_json(request),
             files=info, telemetry=dict(rows_fit=3, rows_eval=20, fit_seconds=0.01, peak_rss_kib=1)))
    return tmp_path, decision


def _mkdir(p):
    from pathlib import Path
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


def test_complete_two_periods_and_private_public_schema(synthetic):
    root, decision = synthetic
    summary = driver.run(root, decision, root / 'private', root / 'public.md')
    assert summary['status'] == 'COMPLETE_HISTORICAL_ONLY'
    assert summary['fit_count'] == 72
    assert set(summary['periods']) == {'2015-2020', '2021-2026'}
    for name, p in summary['periods'].items():
        expected = 2 if name == '2015-2020' else 0
        assert p['valid_days'] == 6
        assert p['scheduled_days'] == 12
        assert p['mean_a'] == pytest.approx(-1 if expected else 1)
        assert p['mean_b'] == pytest.approx(1)
        assert p['mean_difference'] == pytest.approx(expected)
        assert p['paired_ci95'] == pytest.approx([expected, expected])
        assert p['formal_success'] is None
    assert len(summary['yearly_descriptive']) == 12
    assert len(summary['folds']) == 12
    assert summary['evidence_class'] == '已曝光、量級與工程驗證、非確認'
    public = json.loads((root / 'public.json').read_text())
    assert 'daily' not in public and 'inputs' not in public
    assert len(public['spec_sha256']) == len(public['human_decision_sha256']) == 64
    assert (root / 'private/daily_ic.csv').read_text().count('\n') == 25
    manifest = json.loads((root / 'private/manifest.json').read_text())
    assert len(manifest['job_manifests']) == 72
    assert 'prediction.npy' in manifest['job_manifests'][0]['files']
    assert 'NaN' in (root / 'public.md').read_text()
    with pytest.raises(ValueError, match='STOP'):
        driver.run(root, decision, root / 'private', root / 'public.md')


@pytest.mark.parametrize('damage', ['missing_job', 'incomplete', 'manifest', 'keys', 'prediction', 'label_keys', 'bundle'])
def test_damage_stops_before_scoring(synthetic, monkeypatch, damage):
    root, decision = synthetic
    folder = root / 'fits' / s.job_id(2015,17,'A')
    m = json.loads((folder / 'manifest.json').read_text())
    if damage == 'missing_job': shutil.rmtree(folder)
    elif damage in ('incomplete', 'manifest'):
        if damage == 'incomplete': m['state'] = 'PARTIAL'
        else:
            m['request']['input_identity'] = 'wrong'
            m['request_sha256'] = r.digest_json(m['request'])
        dump(folder / 'manifest.json', m)
    elif damage in ('keys', 'prediction'):
        role = 'keys.npy' if damage == 'keys' else 'prediction.npy'
        value = np.load(folder / role)[:-1]
        np.save(folder / role, value)
        m['files'][role] = dict(sha256=s.sha256(folder / role), bytes=(folder / role).stat().st_size)
        dump(folder / 'manifest.json', m)
    else:
        bundle = json.loads((root / 'bundle.json').read_text())
        if damage == 'bundle': del bundle['jobs'][s.job_id(2026,43,'B')]
        else:
            p = root / (s.job_id(2015,17,'A') + '_label_keys.npy')
            np.save(p, np.array([[0,999]]))
            name = bundle['jobs'][s.job_id(2015,17,'A')]['label_keys_file']
            bundle['files'][name].update(sha256=s.sha256(p), bytes=p.stat().st_size)
        dump(root / 'bundle.json', bundle)
        gate = json.loads((root / 'admission/admission.json').read_text())
        gate['bundle'].update(sha256=s.sha256(root / 'bundle.json'), bytes=(root / 'bundle.json').stat().st_size)
        dump(root / 'admission/admission.json', gate)
    monkeypatch.setattr(driver, 'evaluate', lambda *a, **k: pytest.fail('scoring before complete validation'))
    with pytest.raises(ValueError, match='STOP'):
        driver.run(root, decision, root / 'private', root / 'public.md')
    assert not (root / 'private').exists()
    assert not (root / 'public.json').exists()


def assert_incomplete_rejected(root, decision):
    with pytest.raises(ValueError, match='STOP'):
        driver.load_inputs(root, decision)


def test_mutation_skip_complete_manifest_validation_is_killed(synthetic, monkeypatch):
    root, decision = synthetic
    folder = root / 'fits' / s.job_id(2026,43,'B')
    m = json.loads((folder / 'manifest.json').read_text())
    m['state'] = 'PARTIAL'
    dump(folder / 'manifest.json', m)
    assert_incomplete_rejected(root, decision)  # baseline oracle
    monkeypatch.setattr(driver, 'resume', lambda folder, request: json.loads((folder / 'manifest.json').read_text()))
    with pytest.raises(pytest.fail.Exception, match='DID NOT RAISE'):
        assert_incomplete_rejected(root, decision)


def assert_threshold_boundary():
    assert driver.threshold_text(.005, [.001,.009]).startswith('達數值門檻')
    assert driver.threshold_text(.0049, [.001,.009]).startswith('正向但不足')
    assert driver.threshold_text(.005, [0,.009]).startswith('未證實')


def test_mutation_threshold_inclusive_to_strict_is_killed(monkeypatch):
    assert_threshold_boundary()
    real = driver.threshold_text
    monkeypatch.setattr(driver, 'threshold_text', lambda mean, ci: real(mean - 1e-10, ci))
    with pytest.raises(AssertionError):
        assert_threshold_boundary()


def test_pending_templates_only_consumed_after_validation(synthetic):
    root, decision = synthetic
    template = driver.report_template()
    report = root / 'results.md'
    report.write_text(driver.render_report(template), encoding='utf-8')
    dump(root / 'results.json', template)
    driver.run(root, decision, root / 'private', report, consume_placeholder=True)
    assert '{{RESULTS_FILLED_AT_EXECUTION}}' not in report.read_text()
    complete = report.read_bytes()
    with pytest.raises(ValueError, match='STOP'):
        driver.run(root, decision, root / 'other-private', report, consume_placeholder=True)
    assert report.read_bytes() == complete
    assert not (root / 'other-private').exists()


def test_pending_template_tamper_stops_before_input_read(tmp_path, monkeypatch):
    report = tmp_path / 'results.md'
    report.write_text('modified template')
    dump(tmp_path / 'results.json', driver.report_template())
    monkeypatch.setattr(driver, 'load_inputs', lambda *args: pytest.fail('must not read inputs'))
    with pytest.raises(ValueError, match='STOP'):
        driver.run(tmp_path, tmp_path / 'decision.json', tmp_path / 'private', report, consume_placeholder=True)
