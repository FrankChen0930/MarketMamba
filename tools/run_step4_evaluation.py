"""Explicit post-queue Step4 Q-A historical evaluation. No fitting or monitoring.

Import is inert. All 108 jobs must validate before score/label arrays are loaded.
RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
from scipy.stats import rankdata

from tools import step4 as s
from tools import step4_runner as r
from tools import run_step3_evaluation as prior
from tools.step3_evaluation import evaluate as evaluate_pair

_check = prior._check
_json = prior._json
EVIDENCE = prior.EVIDENCE
COMPARISONS = ('1a', '1b')
LIMITATIONS = prior.LIMITATIONS + [
    '第四步分類 known_at UNKNOWN，strict PIT STOP；6161 全期分類排除及研究在市／同業缺值限制不因 IC 解除。',
    '本案歷史 Q-A 僅 1a／1b，逐期 Holm m=2；季報無歷史擬合或主 p 值。',
    '第三步 B 只作工程交叉檢查；第四步 A 使用重新擬合分數，不以舊分數代替。',
]


def holm(p):
    """Fixed two-member family; stable preregistered tie order."""
    _check(set(p) == set(COMPARISONS), 'Holm m=2 family')
    _check(all(np.isfinite(v) and 0 <= v <= 1 for v in p.values()), 'Holm p range')
    order = sorted(COMPARISONS, key=lambda name: (p[name], COMPARISONS.index(name)))
    adjusted = {}; running = 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (2 - i) * p[name]))
        adjusted[name] = float(running)
    return adjusted


def bootstrap(difference):
    """Joint circular 20-session/1000/seed17 draws, retaining NA positions."""
    x = np.asarray(difference, float)
    _check(x.ndim == 2 and x.shape[1] == 2 and len(x) > 0 and not np.isinf(x).any(), 'bootstrap shape')
    _check(np.isfinite(x).any(axis=0).all(), 'no paired period support')
    mean = np.nanmean(x, axis=0)
    rng = np.random.default_rng(17); means = []
    for _ in range(1000):
        starts = rng.integers(0, len(x), size=(len(x) + 19) // 20)
        pick = ((starts[:, None] + np.arange(20)) % len(x)).ravel()[:len(x)]
        sample = x[pick]
        _check(np.isfinite(sample).any(axis=0).all(), 'empty bootstrap replicate; insufficient data')
        means.append(np.nanmean(sample, axis=0))
    means = np.asarray(means)
    return dict(ci95=np.percentile(means, [2.5, 97.5], axis=0).T.tolist(),
                p=((1 + (np.abs(means - mean) >= np.abs(mean)).sum(axis=0)) / 1001).tolist())


def evaluate_family(keys, predictions, labels, label_keys, calendar):
    """Delegate rank averaging, Spearman and alignment to the frozen Step3 code."""
    family = {(a, seed) for a in s.ARMS for seed in s.SEEDS}
    _check(set(keys) == set(predictions) == family, 'nine outputs required')
    pairs = {}
    for name in COMPARISONS:
        arm = 'B' + name
        pk = {(a, seed): keys['A' if a == 'A' else arm, seed] for a in 'AB' for seed in s.SEEDS}
        pp = {(a, seed): predictions['A' if a == 'A' else arm, seed] for a in 'AB' for seed in s.SEEDS}
        pairs[name] = evaluate_pair(pk, pp, labels, keys['A', 17][:, 0], calendar,
            label_keys=label_keys, frozen_labels=np.asarray(labels).copy(), independent_execution_authorized=True)
    daily = {'A': pairs['1a']['daily']['A'].copy(),
             'B1a': pairs['1a']['daily']['B'].copy(), 'B1b': pairs['1b']['daily']['B'].copy()}
    reasons = pairs['1a']['reasons'].copy()
    valid = np.logical_and.reduce([np.isfinite(v) for v in daily.values()])
    for pair in pairs.values():
        constant = pair['reasons'] == 'CONSTANT_CROSS_SECTION'
        reasons[constant] = 'CONSTANT_CROSS_SECTION'
    for values in daily.values():
        values[~valid] = np.nan
    years = np.asarray(calendar, dtype='datetime64[D]').astype('datetime64[Y]').astype(int) + 1970
    periods = {}
    for period, lo, hi in [('2015-2020', 2015, 2020), ('2021-2026', 2021, 2026)]:
        mask = (years >= lo) & (years <= hi)
        diffs = np.column_stack([daily['B' + name][mask] - daily['A'][mask] for name in COMPARISONS])
        boot = bootstrap(diffs)
        adjusted = holm(dict(zip(COMPARISONS, boot['p'])))
        periods[period] = {}
        for j, name in enumerate(COMPARISONS):
            mean = float(np.nanmean(diffs[:, j])); ci = boot['ci95'][j]
            passed = mean >= .005 and ci[0] > 0 and adjusted[name] <= .05
            periods[period][name] = dict(mean_a=float(np.nanmean(daily['A'][mask])),
                mean_b=float(np.nanmean(daily['B' + name][mask])), mean_difference=mean,
                paired_ci95=ci, p_raw=boot['p'][j], p_holm=adjusted[name], holm_reject=adjusted[name] <= .05,
                valid_days=int(np.isfinite(diffs[:, j]).sum()), scheduled_days=int(mask.sum()),
                historical_numeric_threshold=passed, formal_success=False,
                threshold_text=('達歷史數值門檻；量級／工程 evidence，非確認成功' if passed else
                                '正向但不足；非確認成功' if mean > 0 and ci[0] > 0 else '未證實；不主張等效或確認成功'))
    return dict(periods=periods, daily=daily, reasons=reasons)


def _load(directory, bundle, ref):
    return s.load(directory, bundle, ref)


def _expected_request(directory, bundle, job, environment):
    roles = ['sample_file', 'target_file', 'keys_file', 'fit_base', 'eval_base', 'fit_keys', 'eval_rows']
    if job['arm'] != 'A':
        roles += ['fit_extra', 'eval_extra']
    inputs = {role: dict(ref=job[role], sha256=bundle['files'][job[role]]['sha256'],
                        bytes=bundle['files'][job[role]]['bytes']) for role in roles}
    arrays = {role: _load(directory, bundle, job[ref]) for role, ref in
              [('sample', 'sample_file'), ('target', 'target_file'), ('keys', 'keys_file'),
               ('fit', 'fit_base'), ('eval', 'eval_base')]}
    _check(all(r.array_digest(a) == job['source_array_identities'][role] for role, a in arrays.items()),
           'source array identity')
    arrays['fit'] = s.matrix(directory, bundle, job, 'fit')
    arrays['eval'] = s.matrix(directory, bundle, job, 'eval')
    _check(environment.get('device') == 'cpu' and all(isinstance(environment.get(k), str) and environment[k]
           for k in ('python', 'executable', 'numpy', 'lightgbm')), 'producer environment')
    return dict(contract=s.SPEC_SHA, decision=s.DECISION_SHA, bundle_sha256=r.BUNDLE_SHA,
        admission_sha256=r.ADMISSION_SHA, job=s.job_id(job['year'], job['seed'], job['arm']),
        year=job['year'], seed=job['seed'], arm=job['arm'], params=s.PARAMS,
        feature_order=list(s.FEATURE_ORDER[job['arm']]), executor=r.executor_identity(), inputs=inputs,
        array_sha256={k: r.array_digest(v) for k, v in arrays.items()}, environment=environment,
        evidence_class='HISTORICAL_RESEARCH_CPU_ENGINEERING_ONLY')


def load_inputs(bundle_dir, fits_dir):
    """Two passes: complete output integrity first, then frozen array identity."""
    directory = Path(bundle_dir).absolute(); fits = Path(fits_dir).absolute()
    _check(directory.resolve() == directory and fits.resolve() == fits, 'input symlink')
    bundle = r.require_admission(directory)
    names = r.queue()
    _check(len(names) == 108 and len(set(names)) == 108 and set(bundle['jobs']) == set(names), '108 jobs required')
    manifests = []
    for name in names:
        path = fits / name / 'manifest.json'
        # resume verifies state, required file set, sizes/SHA and request digest.
        try:
            m = r.resume(fits / name, json.loads(path.read_text())['request'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('STOP incomplete/corrupt job ' + name) from exc
        manifests.append(dict(job=name, sha256=s.sha256(path), manifest=m, validation='COMPLETE_VERIFIED'))
    # No prediction/label arrays have been opened before all 108 pass above.
    dates = np.asarray(_load(directory, bundle, 'matrix_dates'), dtype='datetime64[D]')
    ends = _load(directory, bundle, 'matrix_ends'); stock = _load(directory, bundle, 'stock_index')
    calendar = np.asarray(_load(directory, bundle, 'calendar'), dtype='datetime64[D]')
    _check(dates.ndim == ends.ndim == stock.ndim == calendar.ndim == 1 and len(ends) > 0, 'axis dimensions')
    _check(ends.dtype.kind in 'iu' and np.all(np.diff(ends, prepend=0) > 0)
           and ends[-1] == len(dates) == len(stock), 'matrix row axes')
    matrix_days = dates[ends - 1]
    _check(not np.isnat(calendar).any() and np.all(calendar[1:] > calendar[:-1]), 'calendar order')
    _check(not np.isnat(matrix_days).any() and np.all(matrix_days[1:] > matrix_days[:-1]), 'matrix dates')
    calendar = calendar[calendar <= matrix_days[-1]]
    di = np.repeat(np.arange(len(ends)), np.diff(ends, prepend=0))
    _check(np.array_equal(dates, matrix_days[di]) and stock.dtype.kind in 'iu' and np.all(stock >= 0), 'matrix axes')
    nstocks = int(stock.max()) + 1; encoded = di * nstocks + stock
    sorter = np.argsort(encoded, kind='stable'); ordered = encoded[sorter]
    _check(np.all(ordered[1:] > ordered[:-1]), 'duplicate matrix keys')
    mapping = np.searchsorted(calendar, matrix_days)
    _check(np.all(mapping < len(calendar)) and np.array_equal(calendar[mapping], matrix_days), 'calendar coverage')
    family = {(a, seed): [] for a in s.ARMS for seed in s.SEEDS}
    predictions = {k: [] for k in family}; labels = []; label_keys = []; folds = []
    records = {v['job']: v for v in manifests}
    for year in s.YEARS:
        year_y = None; year_keys = None
        for seed in s.SEEDS:
            for arm in s.ARMS:
                name = s.job_id(year, seed, arm); job = bundle['jobs'][name]
                _check((job['year'], job['seed'], job['arm']) == (year, seed, arm), 'job identity')
                m = records[name]['manifest']
                request = _expected_request(directory, bundle, job, m['request']['environment'])
                _check(m['request'] == request, name + ' reconstructed request')
                k = np.load(fits / name / 'keys.npy', allow_pickle=False)
                _check(k.ndim == 2 and k.shape[1] == 2 and len(k) > 0 and k.dtype.kind in 'iu', 'key schema')
                _check(np.all(k >= 0) and np.all(k[:, 0] < len(ends)) and np.all(k[:, 1] < nstocks), 'key range')
                _check(np.array_equal(k, _load(directory, bundle, job['keys_file'])), 'frozen keys')
                sample = np.load(fits / name / 'sample.npy', allow_pickle=False)
                _check(np.array_equal(sample, _load(directory, bundle, job['sample_file'])), 'sample identity')
                lk = _load(directory, bundle, job['label_keys_file']); y = _load(directory, bundle, job['eval_labels_file'])
                _check(np.array_equal(lk, k) and y.shape == (len(k),) and not np.isinf(y).any(), 'label keys/shape')
                code = k[:, 0] * nstocks + k[:, 1]; positions = np.searchsorted(ordered, code)
                _check(np.all(positions < len(ordered)) and np.array_equal(ordered[positions], code), 'missing matrix key')
                rows = sorter[positions]; split = _load(directory, bundle, job['splits_file'])
                _check(split.shape == dates.shape and np.all(split[rows] != 0), 'warmup/splits')
                _check(np.all(rows[1:] > rows[:-1]) and np.all(dates[rows].astype('datetime64[Y]').astype(int) + 1970 == year), 'fold order/year')
                _check(np.array_equal(rows, _load(directory, bundle, job['eval_rows'])), 'eval row identity')
                prediction = np.load(fits / name / 'prediction.npy', allow_pickle=False)
                _check(prediction.shape == (len(k),) and np.isfinite(prediction).all(), 'prediction support')
                mapped = k.copy(); mapped[:, 0] = mapping[k[:, 0]]
                if year_y is None:
                    year_y = np.asarray(y).copy(); year_keys = mapped
                _check(np.array_equal(mapped, year_keys) and np.array_equal(y, year_y, equal_nan=True), 'paired labels/keys')
                family[arm, seed].append(mapped); predictions[arm, seed].append(prediction)
        labels.append(year_y); label_keys.append(year_keys)
        folds.append(dict(year=year, inference_rows=len(year_y), mature_rows=int(np.isfinite(year_y).sum()),
                          immature_rows=int(np.isnan(year_y).sum()), last_signal=str(calendar[year_keys[:, 0].max()])))
    return dict(keys={k: np.concatenate(v) for k, v in family.items()},
        predictions={k: np.concatenate(v) for k, v in predictions.items()}, labels=np.concatenate(labels),
        label_keys=np.concatenate(label_keys), calendar=calendar, manifests=manifests, bundle=bundle, folds=folds)


def crosscheck(inputs, third):
    """Exact support; floating tolerance is explicit, never changes a statistical gate."""
    _check(np.array_equal(inputs['calendar'], third['calendar'])
           and np.array_equal(inputs['label_keys'], third['label_keys'])
           and np.array_equal(inputs['labels'], third['labels'], equal_nan=True), 'baseline label/calendar support')
    maxima = []; rank_maxima = []
    for seed in s.SEEDS:
        _check(np.array_equal(inputs['keys']['A', seed], third['keys']['B', seed]), 'baseline keys')
        a = inputs['predictions']['A', seed]; b = third['predictions']['B', seed]
        _check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
               and np.allclose(a, b, rtol=1e-10, atol=1e-12), 'baseline score mismatch')
        maxima.append(float(np.max(np.abs(a - b))))
    def baseline_result(source, arm):
        keys = {(a, seed): source['keys'][arm, seed] for a in 'AB' for seed in s.SEEDS}
        predictions = {(a, seed): source['predictions'][arm, seed] for a in 'AB' for seed in s.SEEDS}
        return evaluate_pair(keys, predictions, source['labels'], source['label_keys'][:, 0], source['calendar'],
            label_keys=source['label_keys'], frozen_labels=source['labels'], independent_execution_authorized=True)
    a = baseline_result(inputs, 'A')['daily']['A']; b = baseline_result(third, 'B')['daily']['A']
    _check(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True), 'baseline IC mismatch')
    for day in np.unique(inputs['label_keys'][:, 0]):
        rows = (inputs['label_keys'][:, 0] == day) & np.isfinite(inputs['labels'])
        if rows.sum() < 20:
            continue
        scores = []
        for source, arm in [(inputs, 'A'), (third, 'B')]:
            scores.append(np.mean([rankdata(source['predictions'][arm, seed][rows], method='average') / rows.sum()
                                   for seed in s.SEEDS], axis=0))
        _check(np.allclose(*scores, rtol=1e-10, atol=1e-12), 'baseline percentile score mismatch')
        rank_maxima.append(float(np.max(np.abs(scores[0] - scores[1]))))
    return dict(status='CONSISTENT', rtol=1e-10, atol=1e-12, raw_score_max_abs_difference=max(maxima),
        percentile_score_max_abs_difference=max(rank_maxima), ic_max_abs_difference=float(np.nanmax(np.abs(a - b))))


def report_template():
    return dict(schema_version=1, status='PENDING_EXECUTION', evidence_class=EVIDENCE,
        research_status='RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED', spec_sha256=s.SPEC_SHA,
        decision_sha256=s.DECISION_SHA, bundle_sha256=r.BUNDLE_SHA, admission_sha256=r.ADMISSION_SHA,
        expected_fit_count=108, fit_count=None, complete_verified=None, periods=None, engineering_crosscheck=None,
        bootstrap=dict(block_sessions=20, replicates=1000, seed=17, holm_m=2, alpha=.05),
        limitations=LIMITATIONS, formal_success=False)


def render_report(summary):
    text = '# 第四步同業代理 h20 歷史評估\n\n'
    text += f"狀態：{summary['status']}；{summary['research_status']}。\n\n證據等級：{EVIDENCE}。\n\n"
    if summary['periods'] is None:
        text += '{{RESULTS_FILLED_AT_EXECUTION}}\n\n尚未評估；必須 108／108 COMPLETE_VERIFIED，不做部分評估。\n'
    else:
        text += '| 期間 | 對比 | A IC | B IC | d=B−A | 95% CI | p | Holm p | 有效／排定日 | 門檻對照 |\n|---|---|---:|---:|---:|---|---:|---:|---|---|\n'
        for period, comparisons in summary['periods'].items():
            for name, p in comparisons.items():
                text += f"| {period} | {name} | {p['mean_a']:.8f} | {p['mean_b']:.8f} | {p['mean_difference']:.8f} | {p['paired_ci95']} | {p['p_raw']:.8f} | {p['p_holm']:.8f} | {p['valid_days']}/{p['scheduled_days']} | {p['threshold_text']} |\n"
        text += '\n工程交叉檢查：' + json.dumps(summary['engineering_crosscheck'], ensure_ascii=False) + '\n'
    text += '\n限制：\n\n' + '\n'.join('- ' + v for v in summary['limitations']) + '\n'
    return text + '\n獨立審查待辦；完成不等於 Human acceptance。逐日分數／IC 僅保存私有 namespace。\n'


def run(bundle_dir, fits_dir, step3_bundle_dir, decision, private_out, public_report, *, consume_placeholder=False):
    private_out = Path(private_out).absolute(); public_report = Path(public_report).absolute()
    base = s.ROOT / '.artifacts/step4/evaluation'; public_json = public_report.with_suffix('.json')
    _check(private_out.resolve() == private_out and (private_out == base or private_out.is_relative_to(base)), 'private evaluation namespace')
    _check(not private_out.exists() and not private_out.is_symlink(), 'private output exists')
    _check(public_report == s.ROOT / 'reports/step4_results_20261127.md'
           and public_report.parent.resolve() == public_report.parent, 'public report namespace')
    _check(consume_placeholder, '--consume-placeholder required for reviewed PENDING templates')
    pending = report_template()
    def check_pending():
        _check(not public_report.is_symlink() and not public_json.is_symlink(), 'public symlink')
        _check(public_report.read_text(encoding='utf-8') == render_report(pending)
               and json.loads(public_json.read_text(encoding='utf-8')) == pending, 'pending template changed/completed')
    check_pending()
    inputs = load_inputs(bundle_dir, fits_dir)
    third = prior.load_inputs(step3_bundle_dir, decision)
    consistency = crosscheck(inputs, third)
    result = evaluate_family(**{k: inputs[k] for k in ('keys', 'predictions', 'labels', 'label_keys', 'calendar')})
    summary = report_template()
    summary.update(status='COMPLETE_HISTORICAL_ONLY', fit_count=108, complete_verified=108, periods=result['periods'],
        engineering_crosscheck=consistency, folds=inputs['folds'], driver_sha256=s.sha256(Path(__file__)),
        source_commit=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                               'rev-parse', 'HEAD'], text=True).strip(),
        bundle_source_commit=inputs['bundle']['source_commit'],
        implementation_sha256={p.name: s.sha256(p) for p in [Path(__file__), Path(prior.__file__),
            Path(sys.modules[evaluate_pair.__module__].__file__), Path(r.__file__), Path(s.__file__)]},
        evaluation_environment=dict(python=platform.python_version(), executable=sys.executable,
                                    numpy=np.__version__, scipy=scipy.__version__,
                                    data_root_env=os.getenv('MARKETMAMBA_DATA_ROOT'), data_realpath=str((s.ROOT/'Data').resolve())),
        third_step_bundle_sha256=s.sha256(Path(step3_bundle_dir)/'bundle.json'),
        third_step_admission_sha256=third['admission_sha256'], human_source_acceptance=third['acceptance'])
    private_out.mkdir(parents=True, exist_ok=False)
    _json(private_out / 'summary.json', summary)
    with (private_out / 'daily_ic.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['session', 'A', 'B1a', 'B1b', 'd_1a', 'd_1b', 'reason'])
        for i, day in enumerate(inputs['calendar']):
            if 2015 <= int(str(day)[:4]) <= 2026:
                vals = [result['daily'][a][i] for a in s.ARMS]
                vals += [vals[1]-vals[0], vals[2]-vals[0]]
                writer.writerow([str(day), *[float(v) if np.isfinite(v) else 'NA' for v in vals], result['reasons'][i]])
    _json(private_out / 'input_identity.json', dict(spec=s.SPEC_SHA, decision=s.DECISION_SHA,
        bundle=r.BUNDLE_SHA, admission=r.ADMISSION_SHA, jobs=inputs['manifests'],
        third_step_jobs=third['manifests'], files=inputs['bundle']['files'], executor=r.executor_identity(),
        support_sha256={k: r.array_digest(inputs[k]) for k in ('labels', 'label_keys', 'calendar')}))
    check_pending()
    staged_md = private_out / 'public_summary.md'; staged_json = private_out / 'public_summary.json'
    staged_md.write_text(render_report(summary), encoding='utf-8'); _json(staged_json, summary)
    _json(private_out / 'manifest.json', dict(schema_version=1, status='COMPLETE_HISTORICAL_ONLY',
        evidence_class=EVIDENCE, driver_sha256=summary['driver_sha256'],
        outputs={p.name: dict(sha256=s.sha256(p), bytes=p.stat().st_size) for p in private_out.iterdir()
                 if p not in (staged_md, staged_json)},
        public_outputs={str(dest): dict(sha256=s.sha256(src), bytes=src.stat().st_size)
                        for src, dest in [(staged_md, public_report), (staged_json, public_json)]}))
    os.replace(staged_json, public_json); os.replace(staged_md, public_report)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--fits-dir', type=Path, required=True)
    parser.add_argument('--step3-bundle-dir', type=Path, required=True)
    parser.add_argument('--human-accepted-source-gate', type=Path, required=True)
    parser.add_argument('--private-out', type=Path, default=s.ROOT/'.artifacts/step4/evaluation')
    parser.add_argument('--public-report', type=Path, default=s.ROOT/'reports/step4_results_20261127.md')
    parser.add_argument('--consume-placeholder', action='store_true')
    args = parser.parse_args(argv)
    try:
        summary = run(args.bundle_dir, args.fits_dir, args.step3_bundle_dir, args.human_accepted_source_gate,
                      args.private_out, args.public_report, consume_placeholder=args.consume_placeholder)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(2, f'STOP: {exc}\n')
    print(json.dumps(dict(status=summary['status'], complete_verified=summary['complete_verified'], evidence_class=EVIDENCE), ensure_ascii=False))


if __name__ == '__main__':
    main()
