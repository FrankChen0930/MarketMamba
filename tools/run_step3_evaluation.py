"""Explicit post-queue Step3 historical evaluation; never starts or monitors fits.

Run as `python -m tools.run_step3_evaluation --bundle-dir ...
--human-accepted-source-gate ...`. The caller must separately authorize execution
only after the queue has finished. This module has no import-time I/O.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from tools import step3 as s
from tools.step3_runner import array_digest, resume, validate_outputs
from tools.step3_evaluation import evaluate

EVIDENCE = '已曝光、量級與工程驗證、非確認'
LIMITATIONS = [
    '歷史兩段均已曝光；C0+ 設計受後期資料影響，只作量級與工程驗證，非獨立確認。',
    '約 23% 營收單位未核驗，相關比較記 NaN，由模型原生處理；不代表已核驗或補零。',
    '公司更正版本未知；TEJ 首次單月值文件及 Human 歷史來源決定不能證明逐期公司更正鏈完整。',
    '發布日為來源記載的日期，嚴格晚於發布日首個 session 收盤生效；缺盤中時間與實際收到時間，不能證明 strict PIT。',
    '原價未含股利，拆併股、減資、除權息及公司行動證據不完整，不能解讀為含息收益或可成交績效。',
    '選樣僅有完整價格的研究股票；研究成員、W60、缺價及未知停牌／下市原因存在選樣與存活者偏誤。',
    '日曆為觀測價格日期聯集，非合格交易所日曆；lifecycle／tradability 未核驗，strict 前瞻 gate 仍 STOP。',
    '五欄缺值與跨期合併／幣別可比性受已接受歷史來源範圍限制；2013 前比較排除，不使用匯出去年／累計欄。',
    '每日需至少 20 股且兩側非恆定，尾端未成熟標籤保留 NaN；缺日位置保留，未知風險不記零。',
    '循環區塊為 20 sessions、1000 次、seed17；更長相依可能使 95% 區間偏窄，非交易獲利區間。',
    '兩段獨立估計，不合併投票；逐年差僅描述。+0.005 及下界>0 僅作門檻對照，歷史不得宣稱正式成功。',
    'builder 合成測試不是獨立驗收或 Human acceptance；無前瞻、P2 開盲、交易、部署或排程授權。',
]


def load_array(bundle, ref):
    return np.load(bundle['files'][ref]['path'], mmap_mode='r', allow_pickle=False)


def expected_request(bundle, gate, job, acceptance):
    """Reconstruct fit_job's complete request, never trust manifest self-claims."""
    request = dict(contract=s.SPEC_SHA, source=bundle['files']['code_step3_runner.py']['sha256'],
                   primitives=bundle['files']['implementation']['sha256'],
                   year=job['year'], seed=job['seed'], arm=job['arm'], params=s.PARAMS,
                   input_identity=gate['bundle']['sha256'], human_source_acceptance=acceptance)
    for role in ('sample', 'target', 'keys', 'fit', 'eval'):
        request[role + '_sha256'] = job[role + '_sha256']
    return request


def _check(condition, message):
    if not condition:
        raise ValueError('STOP ' + message)


def load_inputs(bundle_dir, decision):
    """Validate every COMPLETE job before loading any predictions for scoring."""
    directory = Path(bundle_dir).resolve()
    gate_path = directory / 'admission/admission.json'
    gate = json.loads(gate_path.read_text())
    _check(Path(gate['bundle']['path']).resolve() == directory / 'bundle.json', 'admission bundle path')
    bundle = s.require_admission(gate, decision)
    acceptance = s.human_source_acceptance(decision)
    _check(len(bundle['jobs']) == 72, '72 jobs required')
    manifests = []
    # No partial evaluation: all output files (including model/sample) are hashed
    # by resume before the first call to evaluate. No worker process is touched.
    for year in s.YEARS:
        for seed in s.SEEDS:
            for arm in 'AB':
                name = s.job_id(year, seed, arm)
                job = bundle['jobs'][name]
                m = resume(directory / 'fits' / name, expected_request(bundle, gate, job, acceptance))
                manifests.append(dict(job=name, sha256=s.sha256(directory / 'fits' / name / 'manifest.json'), **m))

    dates = np.asarray(load_array(bundle, 'matrix_dates'), dtype='datetime64[D]')
    ends = np.asarray(load_array(bundle, 'matrix_ends'))
    stock = np.asarray(load_array(bundle, 'stock_index'))
    frozen = np.asarray(load_array(bundle, 'labels'))
    calendar = np.asarray(load_array(bundle, 'calendar'), dtype='datetime64[D]')
    _check(dates.ndim == ends.ndim == stock.ndim == frozen.ndim == calendar.ndim == 1, 'axis dimensions')
    _check(len(ends) > 0 and ends.dtype.kind in 'iu' and np.all(np.diff(ends, prepend=0) > 0)
           and ends[-1] == len(dates) == len(stock) == len(frozen), 'matrix row axes')
    matrix_days = dates[ends - 1]
    _check(not np.isnat(calendar).any() and np.all(calendar[1:] > calendar[:-1]), 'full calendar')
    _check(not np.isnat(matrix_days).any() and np.all(matrix_days[1:] > matrix_days[:-1]), 'matrix session dates')
    # Snapshot horizon is the last session shared with the frozen matrix.
    # Keep all missing positions inside that horizon, not post-snapshot days.
    calendar = calendar[calendar <= matrix_days[-1]]
    di = np.repeat(np.arange(len(ends)), np.diff(ends, prepend=0))
    _check(np.array_equal(dates, matrix_days[di]), 'matrix dates/end alignment')
    _check(stock.dtype.kind in 'iu' and np.all(stock >= 0), 'stock indices')
    nstocks = int(stock.max()) + 1
    encoded = di * nstocks + stock
    sorter = np.argsort(encoded, kind='stable')
    ordered = encoded[sorter]
    _check(np.all(ordered[1:] > ordered[:-1]), 'duplicate matrix keys')
    mapping = np.searchsorted(calendar, matrix_days)
    _check(np.all(mapping < len(calendar)) and np.array_equal(calendar[mapping], matrix_days), 'calendar coverage')
    keys_family = {(a, seed): [] for a in 'AB' for seed in s.SEEDS}
    predictions = {k: [] for k in keys_family}
    labels = []; label_keys = []; folds = []
    for year in s.YEARS:
        outputs = {}; year_y = None; year_keys = None
        for seed in s.SEEDS:
            for arm in 'AB':
                name = s.job_id(year, seed, arm); job = bundle['jobs'][name]
                folder = directory / 'fits' / name
                k = np.load(folder / 'keys.npy', allow_pickle=False)
                _check(k.ndim == 2 and k.shape[1] == 2 and len(k) > 0 and k.dtype.kind in 'iu', name + ' key schema')
                _check(np.all(k >= 0) and np.all(k[:,0] < len(ends)) and np.all(k[:,1] < nstocks), name + ' key range')
                _check(array_digest(k) == job['keys_sha256'] and np.array_equal(k, load_array(bundle, job['keys_file'])), name + ' frozen keys')
                lk = load_array(bundle, job['label_keys_file']); y = load_array(bundle, job['eval_labels_file'])
                _check(np.array_equal(lk, k) and y.shape == (len(k),) and not np.isinf(y).any(), name + ' label keys/shape')
                code = k[:,0] * nstocks + k[:,1]
                positions = np.searchsorted(ordered, code)
                _check(np.all(positions < len(ordered)) and np.array_equal(ordered[positions], code), name + ' missing matrix key')
                rows = sorter[positions]
                split = load_array(bundle, job['splits_file'])
                _check(split.shape == dates.shape and np.all(split[rows] != 0), name + ' warmup/splits')
                _check(np.all(rows[1:] > rows[:-1]) and np.all(dates[rows].astype('datetime64[Y]').astype(int) + 1970 == year), name + ' fold order/year')
                _check(np.array_equal(y, frozen[rows], equal_nan=True), name + ' frozen labels')
                sample = np.load(folder / 'sample.npy', allow_pickle=False)
                _check(array_digest(sample) == job['sample_sha256'] and np.array_equal(sample, load_array(bundle, job['sample_file'])), name + ' sample identity')
                prediction = np.load(folder / 'prediction.npy', allow_pickle=False)
                _check(prediction.shape == (len(k),) and np.isfinite(prediction).all(), name + ' prediction support')
                outputs[arm,seed] = k
                # Fit keys index matrix sessions. Evaluation keys index the FULL
                # frozen calendar, preserving calendar sessions without matrix rows.
                mapped = k.copy(); mapped[:,0] = mapping[k[:,0]]
                keys_family[arm,seed].append(mapped); predictions[arm,seed].append(prediction)
                if year_y is None:
                    year_y = np.asarray(y).copy(); year_keys = mapped
                else:
                    _check(np.array_equal(y, year_y, equal_nan=True), name + ' paired labels')
        validate_outputs(outputs)
        labels.append(year_y); label_keys.append(year_keys)
        folds.append(dict(year=year, inference_rows=len(year_y), common_keys=len(year_y),
                          mature_rows=int(np.isfinite(year_y).sum()), immature_or_missing_label_rows=int(np.isnan(year_y).sum()),
                          first_signal=str(calendar[year_keys[:,0].min()]), last_signal=str(calendar[year_keys[:,0].max()])))
    return dict(keys={k:np.concatenate(v) for k,v in keys_family.items()},
                predictions={k:np.concatenate(v) for k,v in predictions.items()},
                labels=np.concatenate(labels), label_keys=np.concatenate(label_keys), calendar=calendar,
                bundle=bundle, gate=gate, acceptance=acceptance, manifests=manifests, folds=folds,
                admission_sha256=s.sha256(gate_path))


def threshold_text(mean_difference, ci):
    if mean_difference >= .005 and ci[0] > 0:
        return '達數值門檻對照；已曝光歷史不判正式成功'
    if mean_difference > 0 and ci[0] > 0:
        return '正向但不足；已曝光歷史不判正式成功'
    return '未證實；不主張等效，已曝光歷史不判正式成功'


def report_template():
    return dict(schema_version=1, status='PENDING_EXECUTION', evidence_class=EVIDENCE,
                research_status='RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED',
                spec_sha256=s.SPEC_SHA, bundle_sha256=None, human_decision_sha256=None,
                periods=None, yearly_descriptive=None, folds=None, limitations=LIMITATIONS,
                result='{{RESULTS_FILLED_AT_EXECUTION}}')


def render_report(summary):
    text = '# 第三步歷史月營收 h20 評估\n\n'
    text += f"狀態：{summary['status']}；{summary['research_status']}。\n\n證據等級：{EVIDENCE}。\n\n"
    text += f"規格 SHA：`{summary['spec_sha256']}`\n\nBundle SHA：`{summary['bundle_sha256']}`\n\nHuman 決定紀錄 SHA：`{summary['human_decision_sha256']}`\n\n"
    text += '## 結果\n\n'
    if summary['periods'] is None:
        text += '{{RESULTS_FILLED_AT_EXECUTION}}\n\n尚未執行，無真資料評估數字。\n'
    else:
        text += '| 期間 | A 平均 IC | B 平均 IC | 每日配對差平均 | 95% CI | 有效／排定日 | 門檻對照 |\n|---|---:|---:|---:|---|---|---|\n'
        for name,p in summary['periods'].items():
            text += f"| {name} | {p['mean_a']:.8f} | {p['mean_b']:.8f} | {p['mean_difference']:.8f} | {p['paired_ci95']} | {p['valid_days']}/{p['scheduled_days']} | {p['threshold_text']} |\n"
        text += '\n逐年配對差僅描述；完整彙總、各折支援與資源數值見同名 JSON。\n\n'
        text += '| 年度 | 配對差平均 | 有效日 |\n|---|---:|---:|\n'
        for y in summary['yearly_descriptive']:
            text += f"| {y['year']} | {y['mean_difference']} | {y['valid_days']} |\n"
    text += '\n## 限制\n\n' + '\n'.join('- ' + v for v in summary['limitations']) + '\n'
    text += '\n獨立審查：待辦。完成不等於 Human acceptance。私有逐日 IC 與完整 SHA 不得公開逐列。\n'
    return text


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(bundle_dir, decision, private_out, public_report, *, consume_placeholder=False):
    private_out = Path(private_out); public_report = Path(public_report)
    public_json = public_report.with_suffix('.json')
    _check(not private_out.exists() and not private_out.is_symlink(), 'private output exists')
    pending = report_template()
    if consume_placeholder:
        _check(not public_report.is_symlink() and not public_json.is_symlink(), 'public symlink')
        _check(public_report.read_text(encoding='utf-8') == render_report(pending)
               and json.loads(public_json.read_text(encoding='utf-8')) == pending, 'pending template changed/completed')
    else:
        _check(not any(p.exists() or p.is_symlink() for p in (public_report, public_json)), 'public output exists')
    inputs = load_inputs(bundle_dir, decision)
    # No publication before every job, key, label and prediction has validated.
    result = evaluate(inputs['keys'], inputs['predictions'], inputs['labels'], inputs['label_keys'][:,0],
                      inputs['calendar'], label_keys=inputs['label_keys'], frozen_labels=inputs['labels'].copy(),
                      independent_execution_authorized=True)
    summary = report_template()
    summary.update(status='COMPLETE_HISTORICAL_ONLY', result='FILLED_AT_EXECUTION',
                   bundle_sha256=inputs['gate']['bundle']['sha256'], human_decision_sha256=inputs['acceptance']['sha256'],
                   periods=result['periods'], folds=inputs['folds'], fit_count=72,
                   source_gate='HUMAN_ACCEPTED_HISTORICAL_ONLY_STRICT_FORWARD_STOP',
                   bootstrap=dict(method='circular_paired_full_session_axis', block_sessions=20, replicates=1000, seed=17),
                   environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__,
                                    data_root=os.environ.get('MARKETMAMBA_DATA_ROOT'), data_realpath=str((s.ROOT/'Data').resolve())),
                   driver_sha256=s.sha256(__file__), source_commit=inputs['bundle']['source_commit'], yearly_descriptive=[])
    for p in summary['periods'].values():
        p.update(formal_success=None, threshold_text=threshold_text(p['mean_difference'], p['paired_ci95']))
    difference = result['daily']['B'] - result['daily']['A']
    years = inputs['calendar'].astype('datetime64[Y]').astype(int) + 1970
    for year in s.YEARS:
        d = difference[years == year]; valid = np.isfinite(d)
        summary['yearly_descriptive'].append(dict(year=year, mean_difference=float(d[valid].mean()) if valid.any() else None,
                                                 valid_days=int(valid.sum()), scheduled_days=len(d)))
    summary['fit_telemetry'] = [dict(job=m['job'], **{k:m['telemetry'].get(k) for k in
        ('fit_seconds','inference_seconds','total_seconds','peak_rss_kib','rows_fit','rows_eval','fit_matrix_bytes','eval_matrix_bytes')})
        for m in inputs['manifests']]
    summary['job_manifest_sha256'] = {m['job']:m['sha256'] for m in inputs['manifests']}
    summary['period_signal_bounds'] = {name:dict(first_scheduled=str(inputs['calendar'][mask][0]),
        last_scheduled=str(inputs['calendar'][mask][-1]),
        last_valid=str(inputs['calendar'][mask & np.isfinite(difference)][-1]))
        for name,mask in [('2015-2020',(years>=2015)&(years<=2020)),('2021-2026',(years>=2021)&(years<=2026))]}
    # Output namespace is checked even for programmatic callers; tests replace
    # only this filesystem policy to keep all synthetic artifacts in tmp_path.
    s.private_dir(private_out)
    daily_path = private_out / 'daily_ic.csv'
    with daily_path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['session','ic_a','ic_b','difference','reason'])
        for i,day in enumerate(inputs['calendar']):
            if 2015 <= years[i] <= 2026:
                numbers = [result['daily']['A'][i], result['daily']['B'][i], difference[i]]
                writer.writerow([str(day), *[float(v) if np.isfinite(v) else 'NA' for v in numbers], result['reasons'][i]])
    _json(private_out / 'input_sha256.json', dict(spec=s.SPEC_SHA, bundle=summary['bundle_sha256'],
          human_decision=inputs['acceptance'], admission=inputs['admission_sha256'], files=inputs['bundle']['files']))
    if consume_placeholder:
        # Recheck exact pending bytes immediately before replacing the two
        # reviewed templates. COMPLETE output is never eligible for overwrite.
        _check(public_report.read_text(encoding='utf-8') == render_report(pending)
               and json.loads(public_json.read_text(encoding='utf-8')) == pending, 'pending template changed during evaluation')
        staged_json = private_out / 'public_summary.json'
        staged_md = private_out / 'public_summary.md'
        _json(staged_json, summary)
        staged_md.write_text(render_report(summary), encoding='utf-8')
        os.replace(staged_json, public_json)
        os.replace(staged_md, public_report)
    else:
        _json(public_json, summary)
        with public_report.open('x', encoding='utf-8') as stream:
            stream.write(render_report(summary))
    _json(private_out / 'manifest.json', dict(schema_version=1, status='COMPLETE_HISTORICAL_ONLY',
          evidence_class=EVIDENCE, job_manifests=inputs['manifests'], driver_sha256=summary['driver_sha256'],
          outputs={p.name:dict(sha256=s.sha256(p), bytes=p.stat().st_size) for p in [*private_out.iterdir(), public_json, public_report]}))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--human-accepted-source-gate', type=Path, required=True)
    parser.add_argument('--private-out', type=Path, default=s.ROOT/'.artifacts/step3/evaluation')
    parser.add_argument('--public-report', type=Path, default=s.ROOT/'reports/step3_results_20261126.md')
    parser.add_argument('--consume-placeholder', action='store_true', help='replace only the exact reviewed PENDING templates after successful evaluation')
    args = parser.parse_args(argv)
    try:
        summary = run(args.bundle_dir, args.human_accepted_source_gate, args.private_out,
                      args.public_report, consume_placeholder=args.consume_placeholder)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(2, f'STOP: {exc}\n')
    print(json.dumps(dict(status=summary['status'], fit_count=summary['fit_count'],
                          evidence_class=summary['evidence_class']), ensure_ascii=False))


if __name__ == '__main__':
    main()
