"""Label-blind Step5 admission. Hash labels as opaque bytes; never np.load them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from tools import step3
from tools import step5 as s
from tools.step5_prepare import axes, key_rows
from tools.step5_features import STREAK_REASONS


def feature_stats(x, names):
    if x.ndim != 2 or x.shape[1] != len(names) or x.dtype != np.float32 or not len(x):
        raise ValueError('STOP feature shape/dtype/empty')
    finite = np.zeros(len(names), np.int64)
    lo = np.full(len(names), np.inf)
    hi = np.full(len(names), -np.inf)
    for start in range(0, len(x), 50_000):
        a = x[start:start + 50_000]
        if np.isinf(a).any():
            raise ValueError('STOP infinite features')
        f = np.isfinite(a)
        finite += f.sum(axis=0)
        lo = np.minimum(lo, np.min(a, axis=0, where=f, initial=np.inf))
        hi = np.maximum(hi, np.max(a, axis=0, where=f, initial=-np.inf))
    return [dict(feature=name, dtype='float32', rows=len(x), finite=int(n),
                 nan_fraction=float(1 - n / len(x)), variable=bool(n >= s.MIN_FINITE and low < high))
            for name, n, low, high in zip(names, finite, lo, hi)]


def feature_support(x, names):
    stats = feature_stats(x, names)
    if not all(v['variable'] for v in stats):
        bad = [v['feature'] for v in stats if not v['variable']]
        raise ValueError('STOP observable/variable support: ' + ','.join(bad))
    return stats


def future_check(frame, days):
    from tools.step5_features import replay
    cutoff=days[len(days)//2]
    a=replay(frame,days);a=a[a.session<=len(days)//2]
    f=frame.copy();f.loc[pd.to_datetime(f.available_date)>cutoff,'revenue']=987654321.
    b=replay(f.sample(frac=1,random_state=17),days);b=b[b.session<=len(days)//2]
    if not a.equals(b):raise ValueError('STOP future replay invariance')
    return dict(cutoff=str(cutoff),rows=len(a),differences=0)


def _audit(directory, bundle, result):
    if bundle['schema'] != 'step5-component-bundle-v1' or bundle['state'] != 'PREPARED_NOT_ADMITTED':
        raise ValueError('STOP bundle schema/state')
    if bundle['contract'] != s.contract():
        raise ValueError('STOP frozen contract mismatch')
    s.verify_file(s.SPEC, s.SPEC_SHA)
    s.frozen_decision()  # Also require current exact Human record, not a PASS string.
    files = bundle['files']
    expected_pins = {'spec': s.SPEC_SHA, 'decision': s.DECISION_SHA,
                     'step3_manifest': s.STEP3_BUNDLE_SHA}
    for ref, digest in expected_pins.items():
        if files[ref]['sha256'] != digest:
            raise ValueError('STOP source identity ' + ref)
    seen = set()
    for info in files.values():
        path = s.file_path(directory, info)
        if path not in seen:
            s.verify_file(path, info['sha256'], info['bytes'])
            seen.add(path)
    result['verified_files'] = len(seen)
    expected_code = {str(p.relative_to(s.ROOT)) for prefix in ('step3*.py', 'step5*.py')
                     for p in (s.ROOT / 'tools').glob(prefix) if p.name!='step5_runner.py'}
    expected_code.update({'tools/step4.py','tools/step4_prepare.py','tools/step4_audit.py'})
    if set(bundle['implementation']) != expected_code:
        raise ValueError('STOP incomplete implementation identity')
    for path, info in bundle['implementation'].items():
        s.verify_file(s.ROOT / path, info['sha256'], info['bytes'])
    b = json.loads(s.file_path(directory, files['step3_manifest']).read_text())
    step3.validate_bundle_schema(b)
    fm=json.loads(s.file_path(directory,files['feature_manifest']).read_text())
    if fm['contract']!=s.contract():raise ValueError('STOP feature contract')
    for name,info in fm['outputs'].items():
        if any(files['feature_'+name][k]!=info[k] for k in ('sha256','bytes')):raise ValueError('STOP feature output identity')
    if set(bundle['jobs']) != {s.job_id(y, seed, a) for y in s.YEARS for seed in s.SEEDS for a in s.ARMS}:
        raise ValueError('STOP 72-job completeness')
    if tuple(json.loads(s.file_path(directory, files['feature_order']).read_text())['feature_order']) != s.BASE48:
        raise ValueError('STOP column order')
    dates, di, stock, codes, ordered, sorter = axes(bundle, directory)
    calendar = s.load(directory, bundle, 'calendar').astype('datetime64[D]')
    if not len(calendar) or np.isnat(calendar).any() or (np.diff(calendar).astype(int) <= 0).any():
        raise ValueError('STOP calendar')
    block = s.load(directory, bundle, 'revenue_block')
    if block.shape != (len(dates), 4) or block.dtype != np.float32:
        raise ValueError('STOP full block schema')
    from tools.step5_features import replay, align_events
    source=pd.read_parquet(s.file_path(directory,files['feature_source_events.parquet']))
    # Re-derive admitted inputs from frozen Step3 source evidence, not a self-declared feature hash.
    from tools.step3_prepare import admitted_events
    source_manifest=b['files']['source_manifest']
    s.verify_file(s.file_path(directory,files['source_manifest']),source_manifest['sha256'],source_manifest['bytes'])
    mi=json.loads(s.file_path(directory,files['source_manifest']).read_text())['outputs']['monthly_revenue']
    raw_path=s.file_path(directory,files['monthly_source'])
    s.verify_file(raw_path,mi['sha256'],mi['bytes'])
    pi=b['files']['cross_source_pairs'];pair_path=s.file_path(directory,files['cross_source_pairs']);s.verify_file(pair_path,pi['sha256'],pi['bytes'])
    derived=admitted_events(pd.read_parquet(raw_path),pd.read_parquet(pair_path))
    if not source.equals(derived):raise ValueError('STOP admitted source derivation')
    days=pd.DatetimeIndex(calendar)
    events=replay(source,days)
    saved=pd.read_parquet(s.file_path(directory,files['feature_events.parquet']))
    if not events.equals(saved):raise ValueError('STOP event replay')
    raw=s.load(directory,bundle,'feature_block.npy')
    for start in range(0,len(dates),500000):
        end=min(start+500000,len(dates))
        keys=pd.MultiIndex.from_arrays([pd.to_datetime(dates[start:end]),codes[stock[start:end]].astype(str)])
        actual=align_events(events,days,keys,True)
        if not np.array_equal(raw[start:end],actual,equal_nan=True) or not np.array_equal(block[start:end],actual[:,:4],equal_nan=True):raise ValueError('STOP full feature replay')
    years=dates.astype('datetime64[Y]').astype(int)+1970
    checked = set()
    for year in s.YEARS:
        first_keys = None
        for seed in s.SEEDS:
            for arm in s.ARMS:
                name = s.job_id(year, seed, arm)
                job = bundle['jobs'][name]
                original = b['jobs'][step3.job_id(year, seed, 'B')]
                if (job['year'], job['seed'], job['arm']) != (year, seed, arm):
                    raise ValueError('STOP job identity')
                for role in ('sample', 'target', 'keys', 'splits', 'label_keys', 'eval_labels', 'fit', 'eval'):
                    ref = job[role + '_base'] if role in ('fit', 'eval') else job[role + '_file']
                    expected = b['files'][original[role + '_file']]
                    if any(files[ref][k] != expected[k] for k in ('sha256', 'bytes')):
                        raise ValueError('STOP Step3 common support/array bytes ' + role)
                if job['source_array_identities'] != {r: original[r + '_sha256'] for r in ('fit', 'eval', 'sample', 'target', 'keys')}:
                    raise ValueError('STOP source array identity')
                keys = s.load(directory, bundle, job['keys_file'])
                if first_keys is None:
                    first_keys = keys
                if not np.array_equal(first_keys, keys) or not np.array_equal(keys, s.load(directory, bundle, job['label_keys_file'])):
                    raise ValueError('STOP paired keys')
                er = key_rows(keys, ordered, sorter, len(codes))
                if not np.array_equal(er, s.load(directory, bundle, job['eval_rows'])) or not (years[er] == year).all():
                    raise ValueError('STOP eval rows/year')
                rows = s.load(directory, bundle, job['sample_file'])
                if rows.ndim != 1 or rows.dtype.kind not in 'iu' or not 0 < len(rows) <= s.SAMPLE_CAP or (np.diff(rows) <= 0).any() or rows.min() < 0 or rows.max() >= len(dates):
                    raise ValueError('STOP sample rows')
                fit_keys = s.load(directory, bundle, job['fit_keys'])
                if not np.array_equal(fit_keys, np.column_stack([di[rows], stock[rows]])):
                    raise ValueError('STOP fit keys')
                origin = np.flatnonzero(calendar.astype('datetime64[Y]') == np.datetime64(str(year), 'Y'))[0]
                t = np.searchsorted(calendar, dates[rows].astype('datetime64[D]'))
                if not np.array_equal(calendar[t], dates[rows].astype('datetime64[D]')) or not (t + 21 < origin).all() or (years[rows] < 2013).any():
                    raise ValueError('STOP purge/calendar')
                split = s.load(directory, bundle, job['splits_file'])
                if (split[rows] == 0).any() or (split[er] == 0).any():
                    raise ValueError('STOP warmup')
                for phase, positions in (('fit', rows), ('eval', er)):
                    base_ref, extra_ref = job[phase + '_base'], job[phase + '_extra']
                    if (base_ref, extra_ref) in checked:
                        continue
                    checked.add((base_ref, extra_ref))
                    base = s.load(directory, bundle, base_ref)
                    extra = s.load(directory, bundle, extra_ref)
                    if base.shape != (len(positions), 190) or extra.shape != (len(positions), 4):
                        raise ValueError('STOP partition shapes')
                    if not np.array_equal(extra, block[positions], equal_nan=True):
                        raise ValueError('STOP partition extra rows')
                    result['baseline_nan'][base_ref] = feature_stats(base, s.BASELINE_FEATURES)
                    stats = feature_stats(extra, s.EXTRA_FEATURES)
                    for col,stat in enumerate(stats):
                        values=np.asarray(extra[:,col],dtype=np.float64)
                        values=values[np.isfinite(values)]
                        stat['variance']=float(np.var(values)) if len(values) else None
                    result['support'][extra_ref] = stats
                    result.setdefault('diagnostics',{})[extra_ref]=dict(
                        streak_left_censored_fraction=float(np.count_nonzero(raw[positions,4]>0)/len(positions)),
                        streak_nan_reasons={name:int(np.count_nonzero(raw[positions,4]==code)) for name,code in STREAK_REASONS.items()},
                        streak_reason_unknown=int(np.count_nonzero(~np.isfinite(raw[positions,4]))),
                        seasonal_support={str(k):int(np.count_nonzero(raw[positions,5]==k)) for k in range(4)},
                        seasonal_support_unknown=int(np.count_nonzero(~np.isfinite(raw[positions,5]))))
                    if not all(v['variable'] for v in stats):
                        result['stops'].append(f'STOP {year}/{seed}/{phase}/B-A observable/variable support')
        print(json.dumps(dict(audited_year=year, unique_partitions=len(checked))), flush=True)
    result['future_invariance'] = future_check(source,days)
    result['jobs'] = len(bundle['jobs'])
    result['last_signal'] = str(dates[-1].astype('datetime64[D]'))
    result['matrix_rows'] = len(dates)


def admission(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(s.OUTPUT_ROOT):
        raise ValueError('STOP audit namespace')
    target = directory / 'admission.json'
    if target.exists():
        raise ValueError('STOP admission exists (no overwrite)')
    start = time.monotonic()
    result = dict(status='STOP', evidence_class='LABEL_BLIND_CPU_ENGINEERING_ONLY',
        research_status='RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED',
        contract_sha256=s.SPEC_SHA, decision_sha256=s.DECISION_SHA,
        source_identities={}, strict_pit='STOP', forward_admitted=False, trading_admitted=False,
        fit_count=0, label_statistics_computed=False, support={}, baseline_nan={}, stops=[])
    try:
        bundle_path = directory / 'bundle.json'
        result['bundle'] = dict(path=str(bundle_path), sha256=s.sha256(bundle_path), bytes=bundle_path.stat().st_size)
        bundle = json.loads(bundle_path.read_text())
        result['source_identities'] = bundle['source_identities']
        _audit(directory, bundle, result)
        if not result['stops']:
            result['status'] = 'ADMIT'
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result['stops'].append(str(exc))
    result['resources'] = dict(seconds=time.monotonic() - start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    s.write_json(target, result)
    print(json.dumps({k: result[k] for k in ('status', 'stops', 'fit_count', 'resources')}, ensure_ascii=False), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir', type=Path, default=s.OUTPUT_ROOT / 'bundle')
    args = p.parse_args(argv)
    result = admission(args.bundle_dir)
    if result['status'] != 'ADMIT':
        p.exit(2)


if __name__ == '__main__':
    main()
