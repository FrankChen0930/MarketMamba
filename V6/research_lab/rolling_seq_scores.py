"""P3 immutable rolling neural scores. E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Dry-run by default. External matrix files are only opened for reading; derived
rank-gauss caches live in .artifacts/p3/work. No environment installation occurs.
The registered official-Mamba2/no-conv contradiction fails closed before CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .rolling_scores import MATRIX_FILES, identity, provenance, rank_average, write_json
from .portfolio_eval import content_hash, read_pinned

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT/'.artifacts/p3'
WORK_ROOT = ARTIFACT_ROOT/'work'
SYSTEMS = ('mamba2','crossstock')
RECIPE = dict(d_model=64, n_layers=2, mixer_kw=dict(d_state=16, headdim=8, expand=2),
              lr=3e-4, epochs=8, steps_per_epoch=400, dates_per_batch=8,
              stocks_per_date=256, listnet_w=0., n_cross=1)
MODEL_CONTRACT_STOP = ('P3 requests official mamba_ssm.Mamba2 AND no-conv, but historical '
    'seq_runner uses the official default convolution and cross_stock forces d_conv=2. '
    'No registered no-conv factory exists for this recipe; resolve frozen contract before training.')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class ThermalStop(RuntimeError):
    pass


def gpu_temperature():
    raw = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=temperature.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10)
    return float(raw.strip())


class TemperatureGuard:
    """Latched stop at 85C/unknown; pause at 80C until <=72C, including inference."""
    def __init__(self, read=gpu_temperature, sleep=time.sleep, interval=10, deadline=None):
        self.read, self.sleep, self.interval = read, sleep, interval
        self.records, self.stopped, self.deadline = [], False, deadline

    def check(self):
        if self.stopped:
            raise ThermalStop('STOP latched; manual new attempt required')
        paused = False
        while True:
            try:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    raise ValueError('14 hour runtime budget exhausted')
                value = float(self.read())
                if not math.isfinite(value) or not 0 <= value <= 150:
                    raise ValueError('unknown GPU temperature')
            except Exception as exc:
                self.stopped = True
                self.records.append(dict(observed_at=utc_now(), status='STOP', reason=str(exc)))
                raise ThermalStop(f'STOP temperature/budget unavailable: {exc}') from exc
            self.records.append(dict(observed_at=utc_now(), temperature_c=value,
                                     status='STOP' if value>=85 else 'PAUSE' if paused or value>=80 else 'RUN'))
            if value >= 85:
                self.stopped = True
                raise ThermalStop(f'STOP GPU temperature {value}C >=85C')
            if value <= 72 or (not paused and value < 80):
                return
            paused = True
            self.sleep(self.interval)


def require_same_rows(actual, reference):
    keys=['fold','date','code']
    a,b=actual[keys],reference[keys]
    if a.empty or b.empty or a.duplicated().any() or b.duplicated().any() or len(a)!=len(b):
        raise ValueError('P1 evaluation row mismatch/duplicates')
    aa=pd.MultiIndex.from_frame(a); bb=pd.MultiIndex.from_frame(b)
    if not aa.isin(bb).all():
        raise ValueError('P1 evaluation row mismatch')
    if 'label_y5' in actual and 'label_y5' in reference:
        labels=actual[keys+['label_y5']].merge(reference[keys+['label_y5']],on=keys)
        if not np.allclose(labels.label_y5_x,labels.label_y5_y,rtol=0,atol=0,equal_nan=True):
            raise ValueError('P1 evaluation row label mismatch')


def checked_splits(data, val_dates=126):
    tr,va,ev=data.split_rows(val_dates)
    if not len(tr) or not len(va) or not len(ev):
        raise ValueError('empty training/validation/evaluation support')
    # Verify the outer rolling_folds purge independently of inner validation.
    all_train=np.flatnonzero(data.splits==1)
    first=int(data.date_idx[ev].min())
    if len(all_train)==0 or np.any(data.date_idx[all_train]>=first-20):
        raise ValueError('outer rolling purge violated; immature training labels')
    for rows in (tr,va):
        if (not np.all(data.splits[rows]==1) or
            not all(np.isfinite(data.y[k][rows]).all() for k in ('y5','y10')) or
            np.any(data.date_idx[rows]+11>=first)):
            raise ValueError('training split/label maturity violated')
    if data.date_idx[tr].max()+19>=data.date_idx[va].min():
        raise ValueError('inner validation purge violated')
    return tr,va,ev


def prepare_work(source, year):
    """Only create links for allowlisted inputs; never link any writable cache."""
    source=Path(source).resolve(strict=True)
    inputs={name:identity(source/name) for name in MATRIX_FILES}
    target=WORK_ROOT/f'fold_{year}-{content_hash(inputs)[:20]}'
    target.mkdir(parents=True,exist_ok=True)
    for name in MATRIX_FILES:
        link=target/name
        src=(source/name).resolve(strict=True)
        if link.is_symlink():
            if link.resolve(strict=True)!=src:
                raise ValueError('work input identity mismatch')
        elif link.exists():
            raise ValueError('work input must be read-only reference')
        else:
            link.symlink_to(src)
    return target


def resume_fit(path, contract):
    path=Path(path); meta=path.with_suffix('.json')
    if not path.exists() and not meta.exists():
        return None
    if not path.is_file() or not meta.is_file():
        raise ValueError('incomplete fit publication; preserve orphan, use fresh namespace')
    saved=json.loads(meta.read_text())
    if saved['contract']!=contract or saved['contract_hash']!=content_hash(contract):
        raise ValueError('fit identity mismatch')
    if identity(path)!=saved['output']:
        raise ValueError('fit payload hash mismatch')
    return read_pinned(path)[0],saved['job']


def save_fit(path, frame, contract, job):
    path=Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('refusing overwrite of completed/orphan fit')
    with path.open('xb') as stream:
        frame.to_parquet(stream,index=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    write_json(path.with_suffix('.json'),dict(contract=contract,contract_hash=content_hash(contract),
                                           output=identity(path),job=job))


def collect(directory, year, model, seed, cfg, reference, guard, device='cuda', mixer_factory=None):
    """Reusable train_one/predict adapter; CPU injected mixers are engineering only."""
    import torch
    from .seq_runner import SeqData, train_one, predict
    if mixer_factory is None:
        raise RuntimeError(MODEL_CONTRACT_STOP)
    started=time.perf_counter()
    data=SeqData(directory)
    tr,va,ev=checked_splits(data,cfg['val_dates'])
    dates=np.load(Path(directory)/'dates.npy',mmap_mode='r',allow_pickle=False)
    codes=np.load(Path(directory)/'stock_ids.npy',mmap_mode='r',allow_pickle=False)
    frame=pd.DataFrame(dict(fold=year,date=pd.to_datetime(dates[ev]),code=np.asarray(codes[ev]),label_y5=data.y['y5'][ev]))
    require_same_rows(frame,reference)
    if cfg['val_max_rows'] and len(va)>cfg['val_max_rows']:
        va=np.sort(np.random.default_rng(0).choice(va,cfg['val_max_rows'],replace=False))
    # Hook the input boundary: every train/validation/inference batch is guarded.
    windows=data.windows
    def guarded_windows(rows):
        guard.check()
        return windows(rows)
    data.windows=guarded_windows
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    fitted,state=train_one(model,seed,data,tr,va,cfg,device,mixer_factory,ckpt=None)
    if not np.isfinite(state['best']) or state['best'] <= -9.:
        raise ValueError('no valid best validation checkpoint; refusing LAST fallback')
    frame['score']=predict(fitted,data,ev,device)[:,0]
    frame['seed'],frame['system']=int(seed),model
    if not np.isfinite(frame.score).all():
        raise ValueError('nonfinite predictions')
    return frame,dict(fold=year,system=model,seed=seed,fit_rows=len(tr),val_rows=len(va),eval_rows=len(ev),
        history=state['history'],best_val_ic5=state['best'],wall_seconds=time.perf_counter()-started,
        gpu_model=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else 'CPU injected mixer',
        peak_memory_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
        temperatures=guard.records)


def pinned_reference(directory):
    manifest_path=Path(directory)/'manifest.json'
    mf=json.loads(manifest_path.read_text())
    frame,sid=read_pinned(Path(directory)/'rank_average.parquet')
    if sid!=mf['outputs']['rank_average.parquet']:
        raise ValueError('P1 score manifest hash mismatch')
    return frame[frame.system.eq('c0plus')],mf,dict(manifest=identity(manifest_path),scores=sid)


def run_jobs(folds_root, years, models, seeds, out, p1_dir, cfg, smoke=False,
             device='cuda', mixer_factory=None, guard_factory=TemperatureGuard):
    """Fit-level durable resume. Caller owns model-contract admission.

    Injected CPU mixers are deliberately excluded from confirmatory evidence.
    Only a complete fold/model/seed is reusable; partial fits restart in a fresh
    namespace. This does not claim mid-epoch checkpoint recovery.
    """
    import importlib.metadata
    import torch
    if mixer_factory is None:
        raise RuntimeError(MODEL_CONTRACT_STOP)
    started=time.monotonic()
    out=Path(out).resolve()
    if out==ARTIFACT_ROOT.resolve() or not out.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError('output must be below .artifacts/p3/')
    if out.is_relative_to(WORK_ROOT.resolve()): raise ValueError('output collides with work cache')
    reference,_,ref_identity=pinned_reference(p1_dir)
    prov=provenance()
    prov['p3_spec']=identity(ROOT/'tasks/spec-mamba-portfolio-p3-20261120.md')
    prov['environment']['device']=device
    prov['environment']['packages']['torch']=torch.__version__
    try: prov['environment']['packages']['mamba-ssm']=importlib.metadata.version('mamba-ssm')
    except importlib.metadata.PackageNotFoundError: prov['environment']['packages']['mamba-ssm']='MISSING'
    inputs={str(Path(folds_root)/f'fold_{year}'/n):identity(Path(folds_root)/f'fold_{year}'/n)
            for year in years for n in MATRIX_FILES}
    contract=dict(parameters=cfg,smoke=smoke,years=sorted(years),systems=list(models),seeds=list(seeds),
                  inputs=inputs,reference=ref_identity,provenance=prov,
                  mixer='official' if mixer_factory is None else 'INJECTED_CPU_ENGINEERING_ONLY')
    out.mkdir(parents=True,exist_ok=True)
    admission=out/'run_identity.json'
    if admission.exists():
        if json.loads(admission.read_text())!=contract: raise ValueError('run identity mismatch')
    else: write_json(admission,contract)
    frames,jobs=[],[]
    # Count prior completed fit wall time against the same namespace's budget.
    spent=0.
    for path in out.glob('fold_*_seed_*.json'):
        spent+=float(json.loads(path.read_text())['job']['wall_seconds'])
    deadline=started+max(0.,14*3600-spent)
    for year in sorted(years):
        for model in models:
            for seed in seeds:
                path=out/f'fold_{year}_{model}_seed_{seed}.parquet'
                binding=dict(run_hash=content_hash(contract),fold=year,model=model,seed=seed)
                cached=resume_fit(path,binding)
                if cached is not None:
                    frame,job=cached
                else:
                    if time.monotonic()>=deadline: raise ThermalStop('STOP 14 hour cumulative runtime budget')
                    directory=prepare_work(Path(folds_root)/f'fold_{year}',year)
                    # Bind derived cache to hashes of all matrix inputs, not mtime alone.
                    cache_meta=directory/'rankgauss_identity.json'
                    cache=directory/'X_rankgauss.npy'
                    cache_contract=dict(inputs={n:identity(directory/n) for n in MATRIX_FILES},
                                        implementation=prov['implementation'])
                    if cache.exists():
                        if not cache_meta.exists(): raise ValueError('unverified rankgauss cache orphan')
                        saved=json.loads(cache_meta.read_text())
                        if saved['contract']!=cache_contract or saved['output']!=identity(cache):
                            raise ValueError('rankgauss cache hash/identity mismatch')
                    guard=guard_factory(deadline=deadline)
                    try:
                        frame,job=collect(directory,year,model,seed,cfg,reference[reference.fold==year],guard,device,mixer_factory)
                    except Exception as exc:
                        # Immutable failure evidence; never resume it as successful scores.
                        import uuid
                        write_json(out/f'STOP-{year}-{model}-{seed}-{uuid.uuid4().hex}.json',dict(
                            contract=binding,error=type(exc).__name__,reason=str(exc),observed_at=utc_now(),temperatures=guard.records))
                        raise
                    if not cache_meta.exists(): write_json(cache_meta,dict(contract=cache_contract,output=identity(cache)))
                    for value in inputs.values():
                        if identity(value['path'])!=value: raise RuntimeError('input changed during fit')
                    if prov['implementation']!=provenance()['implementation']: raise RuntimeError('source changed during fit')
                    save_fit(path,frame,binding,job)
                require_same_rows(frame,reference[reference.fold==year])
                frames.append(frame); jobs.append(job)
                print(json.dumps(dict(fold=year,system=model,seed=seed,wall_seconds=job['wall_seconds'],action='RESUMED' if cached else 'WROTE')),flush=True)
    ranks=rank_average(pd.concat(frames,ignore_index=True))
    average=out/'rank_average.parquet'
    if not average.exists():
        with average.open('xb') as stream: ranks.to_parquet(stream,index=False)
    else:
        pd.testing.assert_frame_equal(read_pinned(average)[0],ranks)
    manifest=dict(version='p3-scores-v1',artifact_class='GENERATED',evidence_class='E2 RESEARCH_ONLY',
                  contract_status='UNRESOLVED' if mixer_factory is None else 'CPU_ENGINEERING_ONLY',
                  provenance=prov,inputs=inputs,reference=ref_identity,years=sorted(years),seeds=list(seeds),systems=list(models),
                  parameters=dict(**cfg,smoke=smoke,eligibility_window=60,rank_method='average percentile per date then mean across seeds'),
                  jobs=jobs,outputs={p.name:identity(p) for p in sorted(out.glob('*.parquet'))},
                  wall_seconds=time.monotonic()-started)
    if not (out/'manifest.json').exists(): write_json(out/'manifest.json',manifest)
    else:
        saved=json.loads((out/'manifest.json').read_text())
        if saved['outputs']!=manifest['outputs']: raise ValueError('completion manifest hash mismatch')
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds-root',type=Path,required=True)
    p.add_argument('--p1-scores-dir',type=Path,default=ROOT/'.artifacts/p1/full-scores-ace2eef')
    p.add_argument('--years',type=int,nargs='+',default=list(range(2021,2027)))
    p.add_argument('--models',choices=SYSTEMS,nargs='+',default=list(SYSTEMS))
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29])
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true',help='1 epoch x 20 steps; never confirmatory')
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    out=a.out_dir.resolve(); root=ARTIFACT_ROOT.resolve()
    if out==root or not out.is_relative_to(root) or out.is_relative_to(WORK_ROOT.resolve()):
        p.error('output must be below .artifacts/p3/ outside work/')
    if (any(len(v)!=len(set(v)) for v in (a.years,a.models,a.seeds)) or
        not set(a.years)<=set(range(2021,2027)) or not set(a.seeds)<={17,29}):
        p.error('unique registered folds/models/seeds required')
    from .seq_runner import DEFAULT_CFG
    cfg={**DEFAULT_CFG,**RECIPE}
    if a.smoke:
        cfg.update(epochs=1,steps_per_epoch=20)
    plan=dict(dry_run=not a.run,years=sorted(a.years),models=a.models,seeds=a.seeds,
              fits=len(a.years)*len(a.models)*len(a.seeds),parameters=cfg,output=str(out),
              training_contract_status='STOP_MODEL_CONTRACT_AMBIGUITY',reason=MODEL_CONTRACT_STOP)
    if not a.run:
        print(json.dumps(plan)); return
    # No workaround or fallback silently changes the frozen architecture.
    raise RuntimeError(MODEL_CONTRACT_STOP)


if __name__=='__main__':
    main()
