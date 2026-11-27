"""Build Step5 R-A component bundle from immutable Step3 B and first-month revenue refinements.

No model, scores, IC, eval-label array loads, or raw future-value reads.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from tools import step3
from tools import step5 as s

SOURCE = Path('/home/frank/projects/MarketMamba')
AXES = ('matrix_dates', 'matrix_ends', 'stock_index', 'membership', 'calendar', 'feature_order')


def check_disk_budget(estimated, free):
    if estimated > s.LIMIT_BYTES:
        raise ValueError('STOP estimate exceeds 40 GB')
    if estimated * 1.1 > free:
        raise ValueError('STOP insufficient free space with 10% margin')
    return estimated


def source_plan(step3_dir, feature_dir):
    step3_dir,feature_dir=Path(step3_dir).resolve(),Path(feature_dir).resolve()
    s.verify_file(s.SPEC,s.SPEC_SHA);decision=s.frozen_decision()
    bi=s.verify_file(step3_dir/'bundle.json',s.STEP3_BUNDLE_SHA)
    b=json.loads((step3_dir/'bundle.json').read_text())
    step3.validate_bundle_schema(b);step3.verify_source_provenance(b)
    p=json.loads((feature_dir/'manifest.json').read_text())
    if p['schema']!='step5-features-v1' or p['contract']!=s.contract() or p['source_bundle_sha256']!=s.STEP3_BUNDLE_SHA:
        raise ValueError('STOP feature contract')
    for path,digest in p['implementation'].items():s.verify_file(s.ROOT/path,digest)
    refs=set(AXES)|{'source_manifest','cross_source_pairs'}
    for year in s.YEARS:
        for seed in s.SEEDS:
            job=b['jobs'][step3.job_id(year,seed,'B')]
            refs.update(job[role+'_file'] for role in ('fit','eval','sample','target','keys','splits','label_keys','eval_labels'))
    dates=np.load(b['files']['matrix_dates']['path'],mmap_mode='r',allow_pickle=False)
    extra_bytes=len(dates)*4*4+10_000_000
    for year in s.YEARS:
        k=np.load(b['files'][b['jobs'][step3.job_id(year,17,'B')]['keys_file']]['path'],mmap_mode='r')
        extra_bytes+=len(k)*(4*4+8)
        for seed in s.SEEDS:
            a=np.load(b['files'][b['jobs'][step3.job_id(year,seed,'B')]['sample_file']]['path'],mmap_mode='r')
            extra_bytes+=len(a)*(4*4+16)
    copied={b['files'][ref]['path']:b['files'][ref]['bytes'] for ref in refs}
    estimated=sum(copied.values())+extra_bytes+sum(v['bytes'] for v in p['outputs'].values())+p['source_input']['bytes']
    return b,p,sorted(refs),estimated,dict(step3=bi,features=dict(sha256=s.sha256(feature_dir/'manifest.json'),bytes=(feature_dir/'manifest.json').stat().st_size),human_decision=decision)


def axes(bundle, directory):
    dates = s.load(directory, bundle, 'matrix_dates')
    ends = s.load(directory, bundle, 'matrix_ends')
    stock = s.load(directory, bundle, 'stock_index')
    ids = s.load(directory, bundle, 'membership')
    if ends.ndim != 1 or ends.dtype.kind not in 'iu' or (np.diff(ends, prepend=0) <= 0).any():
        raise ValueError('STOP matrix session ends')
    if ends[-1] != len(dates) or len(stock) != len(dates) or len(ids) != len(dates):
        raise ValueError('STOP matrix row axes')
    di = np.repeat(np.arange(len(ends), dtype=np.int32), np.diff(ends, prepend=0))
    if not np.array_equal(dates, dates[ends - 1][di]):
        raise ValueError('STOP matrix date axes')
    code_idx, first = np.unique(stock, return_index=True)
    if not np.array_equal(code_idx, np.arange(len(first))):
        raise ValueError('STOP stock axis')
    codes = ids[first]
    for start in range(0, len(ids), 100_000):
        if not np.array_equal(codes[stock[start:start + 100_000]], ids[start:start + 100_000]):
            raise ValueError('STOP stock mapping')
    encoded = di.astype(np.int64) * len(codes) + stock
    sorter = np.argsort(encoded, kind='stable')
    ordered = encoded[sorter]
    if (np.diff(ordered) <= 0).any():
        raise ValueError('STOP duplicate matrix key')
    return dates, di, stock, codes, ordered, sorter


def key_rows(keys, ordered, sorter, nstocks):
    if keys.ndim != 2 or keys.shape[1] != 2 or keys.dtype.kind not in 'iu' or (keys < 0).any():
        raise ValueError('STOP key schema')
    encoded = keys[:, 0].astype(np.int64) * nstocks + keys[:, 1]
    positions = np.searchsorted(ordered, encoded)
    if (keys[:, 1] >= nstocks).any() or (positions >= len(ordered)).any() or not np.array_equal(ordered[positions], encoded):
        raise ValueError('STOP missing matrix keys')
    rows = sorter[positions]
    if not len(rows) or (np.diff(rows) <= 0).any():
        raise ValueError('STOP unordered/duplicate keys')
    return rows


def prepare(out, step3_dir=SOURCE/'.artifacts/step3/bundle-72',
            feature_dir=SOURCE/'.artifacts/step5/features'):
    start = time.monotonic()
    b, p, refs, estimated, identities = source_plan(step3_dir, feature_dir)
    free = shutil.disk_usage(SOURCE).free
    check_disk_budget(estimated, free)
    print(json.dumps(dict(estimated_bytes=estimated, limit_bytes=s.LIMIT_BYTES, free_bytes=free)), flush=True)
    out = s.new_output(out)
    s.write_json(out / 'preflight.json', dict(estimated_bytes=estimated, free_bytes=free,
                 limit_bytes=s.LIMIT_BYTES, source_identities=identities))
    files, copied = {}, {}

    def record(ref, path):
        path = Path(path)
        files[ref] = dict(path=str(path.relative_to(out)), sha256=s.sha256(path), bytes=path.stat().st_size)

    def copy(ref, path, info):
        path = Path(path).resolve()
        if str(path) in copied:
            files[ref] = files[copied[str(path)]]
            return
        # Source and destination both verified; copying is opaque for labels.
        s.verify_file(path, info['sha256'], info['bytes'])
        destination = out / (ref + path.suffix)
        shutil.copyfile(path, destination)
        s.verify_file(destination, info['sha256'], info['bytes'])
        files[ref] = dict(path=destination.name, sha256=info['sha256'], bytes=info['bytes'])
        copied[str(path)] = ref

    copy('step3_manifest', Path(step3_dir) / 'bundle.json', identities['step3'])
    copy('feature_manifest', Path(feature_dir)/'manifest.json',identities['features'])
    copy('decision', s.DECISION, identities['human_decision'])
    copy('spec', s.SPEC, dict(sha256=s.SPEC_SHA, bytes=s.SPEC.stat().st_size))
    for name,info in p['outputs'].items():
        if Path(name).name!=name:raise ValueError('STOP unsafe feature member')
        copy('feature_'+name,Path(feature_dir)/name,info)
    copy('monthly_source',p['source_input']['path'],p['source_input'])
    for ref in refs:
        copy(ref, b['files'][ref]['path'], b['files'][ref])
        if ref.startswith('fit_'):
            print(json.dumps(dict(copied=ref)), flush=True)
    interim = dict(files=files)
    dates, di, stock, codes, ordered, sorter = axes(interim, out)
    actual_order = json.loads(s.file_path(out, files['feature_order']).read_text())['feature_order']
    if tuple(actual_order) != s.BASE48:
        raise ValueError('STOP baseline feature order')
    raw=s.load(out,interim,'feature_block.npy')
    if raw.shape!=(len(dates),6) or raw.dtype!=np.float32:raise ValueError('STOP feature block schema')
    block=np.lib.format.open_memmap(out/'revenue_block.npy',mode='w+',dtype=np.float32,shape=(len(dates),4))
    block[:]=raw[:,:4];block.flush();record('revenue_block',out/'revenue_block.npy')
    alignment=dict(rows=len(dates),identity='exact Step3 matrix axes; feature replay audited independently')

    def save(ref, value):
        path = out / (ref + '.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        record(ref, path)
        return ref

    jobs = {}
    for year in s.YEARS:
        original = b['jobs'][step3.job_id(year, 17, 'B')]
        keys = s.load(out, interim, original['keys_file'])
        er = key_rows(keys, ordered, sorter, len(codes))
        eval_rows = save(f'eval_rows_{year}', er)
        eval_extra = save(f'eval_extra_{year}', block[er])
        for seed in s.SEEDS:
            original = b['jobs'][step3.job_id(year, seed, 'B')]
            rows = s.load(out, interim, original['sample_file'])
            fit_keys = save(f'fit_keys_{year}_{seed}', np.column_stack([di[rows], stock[rows]]))
            fit_extra = save(f'fit_extra_{year}_{seed}', block[rows])
            for arm in s.ARMS:
                jobs[s.job_id(year, seed, arm)] = dict(year=year, seed=seed, arm=arm,
                    source_step3_job=step3.job_id(year, seed, 'B'),
                    sample_file=original['sample_file'], target_file=original['target_file'],
                    keys_file=original['keys_file'], splits_file=original['splits_file'],
                    label_keys_file=original['label_keys_file'], eval_labels_file=original['eval_labels_file'],
                    fit_base=original['fit_file'], eval_base=original['eval_file'],
                    fit_extra=fit_extra, eval_extra=eval_extra, fit_keys=fit_keys, eval_rows=eval_rows,
                    source_array_identities={r: original[r + '_sha256'] for r in ('fit', 'eval', 'sample', 'target', 'keys')})
        print(json.dumps(dict(prepared_year=year, jobs=len(jobs))), flush=True)
    implementation = {}
    for path in [*sorted((s.ROOT / 'tools').glob('step5*.py')), *sorted((s.ROOT / 'tools').glob('step3*.py')), s.ROOT/'tools/step4.py', s.ROOT/'tools/step4_prepare.py', s.ROOT/'tools/step4_audit.py']:
        ref = 'code_' + path.name
        info = dict(sha256=s.sha256(path), bytes=path.stat().st_size)
        copy(ref, path, info)
        implementation[str(path.relative_to(s.ROOT))] = files[ref]
    unique_bytes = sum(v['bytes'] for v in {v['path']: v for v in files.values()}.values())
    if unique_bytes > s.LIMIT_BYTES:
        raise ValueError('STOP actual bundle exceeds 40 GB')
    bundle = dict(schema='step5-component-bundle-v1', state='PREPARED_NOT_ADMITTED',
        contract=s.contract(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_identities=identities, original_step3_sources=b['source_identities'],
        evidence_class='HISTORICAL_RESEARCH_CPU_ENGINEERING_ONLY', strict_pit='STOP',
        files=files, jobs=jobs, implementation=implementation, alignment=alignment,
        model_identity=None, fit_count=0, eval_label_policy='OPAQUE_COPY_AND_HASH_ONLY_NO_ARRAY_LOAD',
        resources=dict(seconds=time.monotonic() - start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       estimated_bytes=estimated, artifact_bytes=unique_bytes),
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                         executable=os.sys.executable, data_root_env=os.getenv('MARKETMAMBA_DATA_ROOT'),
                         data_realpath=str((SOURCE / 'Data').resolve()), gpu_used=False))
    s.write_json(out / 'bundle.json', bundle)
    print(json.dumps(dict(status=bundle['state'], jobs=len(jobs), resources=bundle['resources'],
                          bundle_sha256=s.sha256(out / 'bundle.json'))), flush=True)
    return bundle


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=s.OUTPUT_ROOT / 'bundle')
    p.add_argument('--step3-dir', type=Path, default=SOURCE / '.artifacts/step3/bundle-72')
    p.add_argument('--feature-dir',type=Path,default=s.OUTPUT_ROOT/'features')
    a = p.parse_args(argv)
    try:
        prepare(a.out, a.step3_dir, a.feature_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f'STOP: {exc}\n')


if __name__ == '__main__':
    main()
