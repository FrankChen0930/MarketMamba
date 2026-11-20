"""P3 immutable rolling neural scores. E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Dry-run by default. External matrix files are only opened for reading; derived
rank-gauss caches live in .artifacts/p3/work. No environment installation occurs.
Amendment 1 uses official default temporal convolution and cross d_conv=2.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
import platform
import sys
import uuid
import inspect
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from .rolling_scores import MATRIX_FILES, identity, provenance, rank_average
from .portfolio_eval import content_hash, read_pinned

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT/'.artifacts/p3'
WORK_ROOT = ARTIFACT_ROOT/'work'
SYSTEMS = ('mamba2','crossstock')
RECIPE = dict(d_model=64, n_layers=2, mixer_kw=dict(d_state=16, headdim=8, expand=2),
              lr=3e-4, epochs=8, steps_per_epoch=400, dates_per_batch=8,
              stocks_per_date=256, listnet_w=0., n_cross=1, dropout=.1, window=60,
              normalize="rankgauss", train_probe_sessions=40)
# Explicit snapshots: DEFAULT_CFG drift is rejected even for overridden keys.
FROZEN_DEFAULT_CFG = dict(d_model=64, n_layers=2,
    mixer_kw=dict(d_state=16, headdim=8, expand=2), attn_window=16, lr=.0003,
    weight_decay=.01, epochs=10, steps_per_epoch=400, patience=3,
    dates_per_batch=16, stocks_per_date=256, min_names=64, baseline_rows=500000,
    n_boot=1000, val_dates=126, val_max_rows=80000, listnet_w=0., listnet_t=1.)
FROZEN_CFG = dict(d_model=64, n_layers=2,
    mixer_kw=dict(d_state=16, headdim=8, expand=2), attn_window=16, lr=.0003,
    weight_decay=.01, epochs=8, steps_per_epoch=400, patience=3,
    dates_per_batch=8, stocks_per_date=256, min_names=64, baseline_rows=500000,
    n_boot=1000, val_dates=126, val_max_rows=80000, listnet_w=0., listnet_t=1., n_cross=1,
    dropout=.1, window=60, normalize="rankgauss", train_probe_sessions=40)


def write_json(path, payload):
    """Exclusive artifact publication, fsync contents then containing directory."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def atomic_json(path, payload):
    """Durable replace: temporary file on the same filesystem, fsync, rename, fsync."""
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    write_json(temporary,payload)
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def namespace_evidence():
    return sorted({p for pattern in ('run_identity.json','fold_*.parquet','fold_*.json')
                   for p in ARTIFACT_ROOT.rglob(pattern)})


def budget_snapshot(events):
    return dict(sequence=len(events),head_sha256=events[-1]['sha256'] if events else '0'*64,
                spent_seconds=budget_state(events)[0])


def validate_snapshot(snapshot,events):
    """A sidecar may pin an older head; the global anchor must pin the exact head."""
    sequence=snapshot['sequence']; spent=snapshot['spent_seconds']
    if (type(sequence) is not int or not 0<=sequence<=len(events) or
        isinstance(spent,bool) or not isinstance(spent,(int,float)) or
        not math.isfinite(spent) or spent<0 or
        snapshot!=budget_snapshot(events[:sequence])):
        raise ValueError('budget anchor/snapshot mismatch; use --rebuild-budget --reason TEXT')


def effective_config(smoke=False):
    from .seq_runner import DEFAULT_CFG
    if DEFAULT_CFG != FROZEN_DEFAULT_CFG or {**DEFAULT_CFG, **RECIPE} != FROZEN_CFG:
        raise ValueError('P3 frozen effective configuration drift')
    cfg=deepcopy(FROZEN_CFG)
    if smoke: cfg.update(epochs=1,steps_per_epoch=20)
    return cfg


def runtime_environment(device):
    import importlib.metadata
    import torch
    packages={'torch':torch.__version__}
    for name,dist in [('mamba_ssm','mamba-ssm'),('causal_conv1d','causal-conv1d')]:
        try: packages[name]=importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError: packages[name]='MISSING'
    return dict(platform=platform.platform(),system=platform.system(),release=platform.release(),
                python=sys.version,executable=sys.executable,packages=packages,device=device,
                gpu_model=torch.cuda.get_device_name(0) if str(device).startswith('cuda') else None)


def validate_frozen(cfg):
    if cfg != effective_config():
        raise ValueError('P3 frozen configuration mismatch')


def mixer_parameters(cfg):
    from mamba_ssm import Mamba2
    signature=inspect.signature(Mamba2)
    defaults={k:v.default for k,v in signature.parameters.items()
              if v.default is not inspect.Parameter.empty and
              v.kind not in (inspect.Parameter.VAR_KEYWORD,inspect.Parameter.VAR_POSITIONAL)}
    # Canonical JSON makes tuple defaults (e.g. A_init_range) stable on disk.
    def canonical(value):
        if isinstance(value,float) and math.isinf(value): return 'Infinity' if value>0 else '-Infinity'
        if isinstance(value,(tuple,list)): return [canonical(v) for v in value]
        return value
    defaults={k:canonical(v) for k,v in defaults.items()}
    if defaults.get('d_conv') != 4:
        raise ValueError('official temporal d_conv default differs from frozen amendment 1')
    temporal=dict(defaults,d_model=cfg['d_model'],**cfg['mixer_kw'])
    return dict(factory='mamba_ssm.Mamba2',temporal=temporal,cross=dict(temporal,d_conv=2))


STANDING_AUTHORIZATION = ('Human 在 2026-11-19／20 的對話中授權主控自行接續本機（RTX 3060）實驗、不得上 Colab。'
                         '此授權可由 Human 日後稽核；不表示 Human 已逐項審核。')
EXPECTED_PACKAGES=dict(torch='2.10.0+cu128',mamba_ssm='2.3.2.post1',causal_conv1d='1.7.0')


def environment_files():
    """Hash installed code/binaries and metadata, plus interpreter and exact lock."""
    import importlib.metadata
    paths={Path(sys.executable).resolve(), Path.home()/'.venvs/marketmamba-gpu.requirements.lock.txt'}
    for dist in importlib.metadata.distributions():
        if not dist.files: raise RuntimeError('installed environment file inventory missing')
        paths.update(Path(dist.locate_file(f)).resolve() for f in dist.files
                     if not str(f).endswith(('.pyc','.pyo')))
    return {str(p):identity(p) for p in sorted(paths)} | {
        'required_lock':identity(Path.home()/'.venvs/marketmamba-gpu.requirements.lock.txt')}


def smoke_digest(payload):
    return content_hash({k:v for k,v in payload.items() if k not in ('reviewed_by','review_source','payload_sha256')})


def validate_smoke(payload,environment,cfg,review=True):
    if (environment['packages']!=EXPECTED_PACKAGES or not environment.get('gpu_model') or
        not str(environment['device']).startswith('cuda') or
        payload.get('version')!='p3-official-gpu-smoke-v2' or
        payload.get('parameters')!=effective_config() or cfg not in (effective_config(),effective_config(True)) or
        payload.get('mixer_parameters')!=mixer_parameters(cfg) or
        payload.get('implementation')!=provenance()['implementation'] or
        payload.get('p3_spec')!=identity(ROOT/'tasks/spec-mamba-portfolio-p3-20261120.md') or
        payload.get('environment')!=environment or payload.get('environment_files')!=environment_files() or
        payload.get('payload_sha256')!=smoke_digest(payload) or 'human_reviewed' in payload or
        payload.get('standing_authorization')!=STANDING_AUTHORIZATION):
        raise RuntimeError('STOP official GPU smoke evidence mismatch; contract_status UNRESOLVED')
    measurements=payload.get('measurements',{})
    if set(measurements)!={'temporal','cross'}: raise RuntimeError('STOP smoke measurements missing')
    for row in measurements.values():
        if row.get('forward_finite') is not True or row.get('backward_finite') is not True:
            raise RuntimeError('STOP smoke nonfinite')
        for key in ('parameters','gradient_tensors','peak_memory_bytes','elapsed_seconds','temperature_start_c','temperature_end_c','output_abs_max','gradient_abs_max'):
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:
                raise RuntimeError('STOP smoke invalid measurement: '+key)
        if max(row['temperature_start_c'],row['temperature_end_c'])>=85:
            raise RuntimeError('STOP smoke thermal limit')
    if review:
        r=payload.get('reviewed_by',{})
        if (set(r)!={'kind','job_id','verdict','evidence_sha256'} or
            r.get('kind') not in ('independent_worker','human') or not r.get('job_id') or
            r.get('job_id')==payload.get('producer_job_id') or r.get('verdict')!='GO' or
            r.get('evidence_sha256')!=payload.get('review_source',{}).get('sha256')):
            raise RuntimeError('STOP independent review required')
        source=payload['review_source']
        if identity(source['path'])!=source:
            raise RuntimeError('STOP reviewed raw evidence hash mismatch')
        raw=json.loads(Path(source['path']).read_text())
        if ('reviewed_by' in raw or 'review_source' in raw or
            raw.get('payload_sha256')!=payload['payload_sha256'] or smoke_digest(raw)!=payload['payload_sha256']):
            raise RuntimeError('STOP reviewed raw evidence content mismatch')


def official_admission(evidence,environment,cfg):
    if evidence is None:
        raise RuntimeError('STOP official GPU smoke evidence required; contract_status UNRESOLVED')
    payload=json.loads(Path(evidence).read_text())
    validate_smoke(payload,environment,cfg)
    return dict(status='RESOLVED',evidence=identity(evidence))


def gpu_smoke(output,job_id=None):
    """Executor-owned CUDA measurements; never records its own review."""
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('STOP CUDA required for GPU smoke')
    if not job_id: raise ValueError('producer job_id required')
    if Path(output).exists(): raise FileExistsError(output)
    env=runtime_environment('cuda')
    if env['packages']!=EXPECTED_PACKAGES: raise RuntimeError('STOP specified GPU environment required')
    cfg=effective_config(); mixers=mixer_parameters(cfg)
    files=environment_files(); implementation=provenance()['implementation']
    measurements={}
    from mamba_ssm import Mamba2
    with BudgetLedger(Path(output).parent) as ledger:
        with ledger.attempt(dict(kind='gpu-smoke',job_id=job_id)) as deadline:
            guard=TemperatureGuard(deadline=deadline)
            for name,length in (('temporal',60),('cross',256)):
                guard.check(); start_temp=guard.records[-1]['temperature_c']
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
                kwargs=deepcopy(mixers[name])
                if 'dt_limit' in kwargs:
                    kwargs['dt_limit']=tuple(float(v) for v in kwargs['dt_limit'])
                model=Mamba2(**kwargs).cuda()
                x=torch.randn(2,length,cfg['d_model'],device='cuda',requires_grad=True)
                y=model(x); finite=bool(torch.isfinite(y).all().item())
                loss=y.square().mean(); loss.backward(); torch.cuda.synchronize()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                backward=bool(grads) and all(bool(torch.isfinite(g).all().item()) for g in grads)
                backward=backward and x.grad is not None and bool(torch.isfinite(x.grad).all().item())
                guard.check()
                measurements[name]=dict(forward_finite=finite,backward_finite=backward,
                    parameters=sum(p.numel() for p in model.parameters()),gradient_tensors=len(grads),
                    peak_memory_bytes=torch.cuda.max_memory_allocated(),gpu_model=env['gpu_model'],
                    temperature_start_c=start_temp,temperature_end_c=guard.records[-1]['temperature_c'],
                    elapsed_seconds=time.perf_counter()-started,output_abs_max=float(y.detach().abs().max()),
                    gradient_abs_max=max(float(g.abs().max()) for g in grads))
                if not finite or not backward: raise RuntimeError('STOP GPU smoke nonfinite')
                del model,x,y,loss,grads
            if files!=environment_files() or implementation!=provenance()['implementation']:
                raise RuntimeError('STOP smoke source/environment changed')
            payload=dict(version='p3-official-gpu-smoke-v2',producer_job_id=job_id,observed_at=utc_now(),
                parameters=cfg,mixer_parameters=mixers,implementation=implementation,environment=env,
                p3_spec=identity(ROOT/'tasks/spec-mamba-portfolio-p3-20261120.md'),
                environment_files=files,measurements=measurements,standing_authorization=STANDING_AUTHORIZATION)
            payload['payload_sha256']=smoke_digest(payload)
            validate_smoke(payload,env,cfg,review=False)
            write_json(Path(output),payload)
    return identity(output)


def record_review(evidence,output,kind,job_id,verdict):
    payload=json.loads(Path(evidence).read_text())
    validate_smoke(payload,runtime_environment('cuda'),effective_config(),review=False)
    if kind not in ('independent_worker','human') or not job_id or job_id==payload['producer_job_id']:
        raise ValueError('independent reviewer job required')
    if 'reviewed_by' in payload or 'review_source' in payload:
        raise ValueError('record-review requires raw unreviewed executor evidence')
    payload['review_source']=identity(evidence)
    payload['reviewed_by']=dict(kind=kind,job_id=job_id,verdict=verdict,evidence_sha256=payload['review_source']['sha256'])
    payload['payload_sha256']=smoke_digest(payload)
    write_json(Path(output),payload)  # exclusive new artifact; preserve raw smoke
    return identity(output)


class BudgetLedger:
    """Global hash-chain budget. Lock the trusted artifact directory inode.

    Deleting/replacing budget.lock or the ledger cannot bypass the held lock.
    ARTIFACT_ROOT and its ancestors must remain trusted, stable directories.
    """
    def __init__(self,directory,limit_seconds=14*3600,allow_stopped=False):
        self.directory=Path(directory); self.limit=limit_seconds; self.active=None
        self.allow_stopped=allow_stopped

    def __enter__(self):
        import fcntl
        ARTIFACT_ROOT.mkdir(parents=True,exist_ok=True)
        self.lock=os.open(ARTIFACT_ROOT,os.O_RDONLY|os.O_DIRECTORY)
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.path=ARTIFACT_ROOT/'budget-global.jsonl'
            self.anchor_path=ARTIFACT_ROOT/'budget-anchor.json'
            evidence=namespace_evidence()
            if not self.path.exists() or not self.anchor_path.exists():
                if evidence or self.path.exists() or self.anchor_path.exists():
                    raise ValueError('missing budget ledger/anchor; use --rebuild-budget --reason TEXT')
                with self.path.open('x') as stream:
                    stream.flush(); os.fsync(stream.fileno())
                atomic_json(self.anchor_path,dict(budget_snapshot([]),observed_at=utc_now()))
            self.events=read_budget(self.path)
            self._validate_anchor()
            self.spent,self.stops,pending=budget_state(self.events)
            if pending:
                elapsed=max(time.time()-pending['wall_start'],time.monotonic()-pending['monotonic_start'])
                if elapsed<0 or not math.isfinite(elapsed): raise ValueError('budget recovery clock invalid')
                self.stop('interrupted attempt recovered; inspect before clearance')
                self._append(dict(event='END',attempt=pending['attempt'],seconds=elapsed,status='INTERRUPTED'))
                self.spent+=elapsed
            # Files are supplementary evidence; ledger is the persistent latch.
            if (self.stops or list(ARTIFACT_ROOT.rglob('STOP-*.json'))) and not self.allow_stopped:
                raise ThermalStop('STOP latched; use --clear-stop with reason; preserve budget')
            if self.spent>=self.limit and not self.allow_stopped:
                self.stop('14 hour cumulative runtime budget exhausted')
                raise ThermalStop('STOP 14 hour cumulative runtime budget exhausted')
            return self
        except BaseException:
            os.close(self.lock); self.lock=None; raise

    def snapshot(self):
        return budget_snapshot(self.events)

    def _validate_anchor(self):
        validate_ledger_state(self.events,self.anchor_path,self.snapshot())

    def _append(self,event):
        event=dict(event,observed_at=utc_now(),previous_sha256=self.events[-1]['sha256'] if self.events else '0'*64)
        event['sha256']=content_hash(event)
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        os.fsync(self.lock)
        self.events.append(event)
        atomic_json(self.anchor_path,dict(self.snapshot(),observed_at=utc_now()))

    def stop(self,reason):
        name='STOP-'+uuid.uuid4().hex+'.json'
        self._append(dict(event='STOP',name=name,reason=reason))
        self.stops.add(name)
        write_json(ARTIFACT_ROOT/name,dict(reason=reason,observed_at=utc_now()))
        return name

    def clear_stop(self,name,reason):
        if not reason.strip() or Path(name).name!=name: raise ValueError('STOP name and reason required')
        file=ARTIFACT_ROOT/name
        if name not in self.stops and not file.exists(): raise ValueError('unknown STOP')
        if name not in self.stops:
            self._append(dict(event='STOP',name=name,reason='external STOP file adopted before explicit clearance'))
        self._append(dict(event='CLEARED',name=name,reason=reason))
        self.stops.discard(name)
        file.unlink(missing_ok=True); os.fsync(self.lock)

    def start(self,binding):
        if self.active is not None: raise ValueError('budget attempt already active')
        if self.stops or list(ARTIFACT_ROOT.rglob('STOP-*.json')): raise ThermalStop('STOP latched')
        if self.spent>=self.limit: raise ThermalStop('STOP 14 hour cumulative runtime budget exhausted')
        self.active=dict(event='START',attempt=uuid.uuid4().hex,wall_start=time.time(),monotonic_start=time.monotonic(),binding=binding)
        self.tick=time.monotonic(); self._append(self.active)
        return self.tick+self.limit-self.spent

    @contextmanager
    def attempt(self,binding):
        deadline=self.start(binding); status='COMPLETED'
        try:
            yield deadline
            if time.monotonic()>=deadline: raise ThermalStop('STOP 14 hour cumulative runtime budget exhausted')
        except BaseException as exc:
            status=type(exc).__name__; self.stop(f'{status}: {exc}'); raise
        finally:
            seconds=time.monotonic()-self.tick
            self._append(dict(event='END',attempt=self.active['attempt'],seconds=seconds,status=status))
            self.spent+=seconds; self.active=None

    def __exit__(self,*exc):
        os.close(self.lock); self.lock=None


def read_budget(path):
    events=[]
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        row=json.loads(line); digest=row.get('sha256')
        if (digest!=content_hash({k:v for k,v in row.items() if k!='sha256'}) or
            row.get('previous_sha256')!=(events[-1]['sha256'] if events else '0'*64)):
            raise ValueError('budget hash chain mismatch')
        events.append(row)
    return events



def validate_ledger_state(events,anchor_path,snapshot=None):
    """Raise unless ledger, anchor and surviving sidecars are mutually consistent."""
    snapshot_now=snapshot if snapshot is not None else budget_snapshot(events)
    anchor=json.loads(Path(anchor_path).read_text())
    saved_snapshot={k:anchor[k] for k in ('sequence','head_sha256','spent_seconds')}
    validate_snapshot(saved_snapshot,events)
    if saved_snapshot!=snapshot_now or not anchor.get('observed_at'):
        raise ValueError('budget anchor head mismatch; use --rebuild-budget --reason TEXT')
    for path in ARTIFACT_ROOT.rglob('fold_*.json'):
        saved=json.loads(path.read_text())
        if not isinstance(saved.get('budget'),dict):
            raise ValueError('sidecar missing budget snapshot; manual inventory required')
        rebuilds=[e for e in events if e['event']=='REBUILT']
        covered=any(identity(path) in e['sidecars'] for e in rebuilds)
        if not covered: validate_snapshot(saved['budget'],events)


def budget_state(events):
    spent=0.; stops=set(); pending=None
    for index,row in enumerate(events):
        event=row['event']
        if event=='REBUILT' and index==0:
            value=row['spent_seconds']
            if (isinstance(value,bool) or not isinstance(value,(int,float)) or
                not math.isfinite(value) or value<0 or not row.get('reason','').strip() or
                value<row['sidecar_spent_lower_bound']):
                raise ValueError('invalid rebuilt budget debit')
            spent=value
            names=[row.get('stop_name'),*row.get('inherited_stops',[])]
            if (not isinstance(row.get('stop_name'),str) or not isinstance(row.get('inherited_stops',[]),list)
                    or any(not isinstance(n,str) or not n or Path(n).name!=n for n in names)):
                raise ValueError('invalid REBUILT STOP latch')
            stops.update(names)
        elif event=='START' and pending is None: pending=row
        elif event=='END' and pending and row['attempt']==pending['attempt']:
            seconds=row['seconds']
            if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
                raise ValueError('invalid budget debit')
            spent+=seconds; pending=None
        elif event=='STOP': stops.add(row['name'])
        elif event=='CLEARED' and row['name'] in stops and row.get('reason','').strip(): stops.remove(row['name'])
        else: raise ValueError('corrupt budget ledger event order')
    return spent,stops,pending


def rebuild_budget(reason):
    """Explicit recovery only. Preserve damaged bytes; never silently reset spent.

    Existing outputs are historical and cannot acquire new completed START/END
    evidence through recovery. A STOP requires separate inspection/clearance.
    """
    import fcntl
    if not reason or not reason.strip(): raise ValueError('rebuild reason required')
    ARTIFACT_ROOT.mkdir(parents=True,exist_ok=True)
    lock=os.open(ARTIFACT_ROOT,os.O_RDONLY|os.O_DIRECTORY)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ARTIFACT_ROOT/'budget-global.jsonl'; anchor=ARTIFACT_ROOT/'budget-anchor.json'
        sidecars=[]; lower=0.; known=0.; preserved=[]
        def debit(value):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('unknown/invalid spent; manual inventory required')
            return value
        for sidecar in sorted(ARTIFACT_ROOT.rglob('fold_*.json')):
            saved=json.loads(sidecar.read_text())
            if not isinstance(saved.get('budget'),dict) or 'spent_seconds' not in saved['budget']:
                raise ValueError('sidecar missing spent; manual inventory required')
            lower=max(lower,debit(saved['budget']['spent_seconds']))
            sidecars.append(identity(sidecar))
        # A payload without its sidecar has unknown cost; recovery cannot invent it.
        for fit in ARTIFACT_ROOT.rglob('fold_*.parquet'):
            if not fit.with_suffix('.json').exists():
                raise ValueError('fit without sidecar; unknown spent; manual inventory required')
        inherited=set(p.name for p in ARTIFACT_ROOT.rglob('STOP-*.json'))
        if path.exists() and anchor.exists():
            try:
                healthy_events=read_budget(path)
                validate_ledger_state(healthy_events,anchor)
                budget_state(healthy_events)
                healthy=True
            except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError):
                healthy=False
            if healthy:
                raise ValueError('ledger, anchor and sidecars are consistent; rebuild refused (use --clear-stop for a latched STOP)')
        if anchor.exists():
            known=max(known,debit(json.loads(anchor.read_text())['spent_seconds']))
        try:
            events=read_budget(path)
            spent,old_stops,pending=budget_state(events)
            inherited|=set(old_stops)
            if pending:
                elapsed=max(time.time()-pending['wall_start'],time.monotonic()-pending['monotonic_start'])
                spent+=debit(elapsed)
            known=max(known,spent)
        except (ValueError,KeyError,TypeError):
            # Damaged chain is preserved below; surviving anchor/sidecars own the
            # known lower bound, not unvalidated debits from corrupt events.
            pass
        recovery=uuid.uuid4().hex
        for source in (path,anchor):
            if source.exists():
                copy=source.with_name('budget-rebuild-'+recovery+'-'+source.name)
                with copy.open('xb') as stream:
                    stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
                preserved.append(identity(copy))
        stop_name='STOP-'+uuid.uuid4().hex+'.json'
        row=dict(event='REBUILT',reason=reason,sidecar_spent_lower_bound=lower,
                 stop_name=stop_name,inherited_stops=sorted(inherited),
                 spent_seconds=max(known,lower),sidecars=sidecars,preserved=preserved,
                 observed_at=utc_now(),previous_sha256='0'*64)
        row['sha256']=content_hash(row)
        temporary=path.with_name(path.name+'.tmp-'+recovery)
        with temporary.open('x') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path); os.fsync(lock)
        atomic_json(anchor,dict(budget_snapshot([row]),observed_at=utc_now()))
        # The REBUILT row itself latches stop_name; the file is supplementary evidence.
        write_json(ARTIFACT_ROOT/stop_name,dict(reason='budget REBUILT; inspect preserved history and unknown interrupted costs before clearance',observed_at=utc_now()))
        snapshot=budget_snapshot([row])
        print(json.dumps(dict(rebuilt=True,**snapshot,sidecar_spent_lower_bound=lower,stop_name=stop_name)))
        return snapshot
    finally: os.close(lock)


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


def write_fit_payload(path, frame):
    path=Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('refusing overwrite of completed/orphan fit')
    with path.open('xb') as stream:
        frame.to_parquet(stream,index=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())

def publish_fit_sidecar(path, contract, job, budget):
    path=Path(path)
    write_json(path.with_suffix('.json'),dict(contract=contract,contract_hash=content_hash(contract),
                                           output=identity(path),job=job,budget=budget))


def save_fit(path, frame, contract, job, budget):
    write_fit_payload(path,frame)
    publish_fit_sidecar(path,contract,job,budget)



def collect(directory, year, model, seed, cfg, reference, guard, device='cuda', mixer_factory=None, gpu_smoke_evidence=None, _ledger=None):
    """Internal adapter: caller must hold an active global-budget attempt."""
    if _ledger is None or _ledger.active is None or _ledger.lock is None or _ledger.stops:
        raise RuntimeError('collect is internal; active global budget attempt required')
    import torch
    from .seq_runner import SeqData, train_one, predict
    effective_config()  # Always check inherited-default drift, including injected engineering.
    if mixer_factory is None:
        if cfg not in (effective_config(),effective_config(True)):
            raise ValueError('P3 frozen configuration mismatch')
        if gpu_smoke_evidence is None: official_admission(None,{},cfg)
        official_admission(gpu_smoke_evidence,runtime_environment(device),cfg)
    started=time.perf_counter()
    data=SeqData(directory,window=cfg.get("window",60),normalize=cfg.get("normalize","rankgauss"))
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
             device='cuda', mixer_factory=None, guard_factory=TemperatureGuard, gpu_smoke_evidence=None):
    """Fit-level durable resume. Caller owns model-contract admission.

    Injected CPU mixers are deliberately excluded from confirmatory evidence.
    Only a complete fold/model/seed is reusable; partial fits restart in a fresh
    namespace. This does not claim mid-epoch checkpoint recovery.
    """
    effective=effective_config(smoke)
    if mixer_factory is None and cfg!=effective:
        raise ValueError('P3 frozen configuration mismatch')
    if list(Path(out).glob('STOP-*.json')):
        raise ThermalStop('STOP latched; inspect legacy STOP; preserve global budget; use explicit recorded clearance')
    if mixer_factory is None and gpu_smoke_evidence is None:
        official_admission(None,{},cfg)
    environment=runtime_environment(device)
    admission_status=official_admission(gpu_smoke_evidence,environment,cfg) if mixer_factory is None else dict(status='CPU_ENGINEERING_ONLY')
    actual_mixer=mixer_parameters(cfg) if mixer_factory is None else dict(factory='INJECTED_CPU_ENGINEERING_ONLY',actual_parameters='caller-defined; not official')
    started=time.monotonic()
    out=Path(out).resolve()
    if out==ARTIFACT_ROOT.resolve() or not out.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError('output must be below .artifacts/p3/')
    if out.is_relative_to(WORK_ROOT.resolve()): raise ValueError('output collides with work cache')
    reference,_,ref_identity=pinned_reference(p1_dir)
    prov=provenance()
    prov['p3_spec']=identity(ROOT/'tasks/spec-mamba-portfolio-p3-20261120.md')
    prov['environment'].update(environment)
    inputs={str(Path(folds_root)/f'fold_{year}'/n):identity(Path(folds_root)/f'fold_{year}'/n)
            for year in years for n in MATRIX_FILES}
    contract=dict(parameters=cfg,smoke=smoke,years=sorted(years),systems=list(models),seeds=list(seeds),
                  inputs=inputs,reference=ref_identity,provenance=prov,
                  mixer=actual_mixer,admission=admission_status,frozen_effective_config=deepcopy(FROZEN_CFG))
    out.mkdir(parents=True,exist_ok=True)
    frames,jobs=[],[]
    with BudgetLedger(out) as ledger:
        admission=out/'run_identity.json'
        if admission.exists():
            if json.loads(admission.read_text())!=contract: raise ValueError('run identity mismatch')
            if not any(e['event']=='START' and e['binding'].get('run_hash')==content_hash(contract) for e in ledger.events):
                raise ValueError('run identity missing from global budget chain')
        else: write_json(admission,contract)
        for year in sorted(years):
            for model in models:
                for seed in seeds:
                    path=out/f'fold_{year}_{model}_seed_{seed}.parquet'
                    binding=dict(run_hash=content_hash(contract),fold=year,model=model,seed=seed)
                    cached=resume_fit(path,binding)
                    if cached is not None:
                        starts={e['attempt']:e for e in ledger.events if e['event']=='START' and e['binding']==binding}
                        if not any(e['event']=='END' and e['status']=='COMPLETED' and e['attempt'] in starts for e in ledger.events):
                            raise ValueError('fit missing from global budget chain')
                        frame,job=cached
                    else:
                        with ledger.attempt(binding) as deadline:
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
                            frame,job=collect(directory,year,model,seed,cfg,reference[reference.fold==year],guard,device,mixer_factory,gpu_smoke_evidence=gpu_smoke_evidence,_ledger=ledger)
                            if not cache_meta.exists(): write_json(cache_meta,dict(contract=cache_contract,output=identity(cache)))
                            for value in inputs.values():
                                if identity(value['path'])!=value: raise RuntimeError('input changed during fit')
                            if prov['implementation']!=provenance()['implementation']: raise RuntimeError('source changed during fit')
                            write_fit_payload(path,frame)
                        publish_fit_sidecar(path,binding,job,budget=ledger.snapshot())
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
                      contract_status=admission_status['status'],admission=admission_status,
                      frozen_effective_config=deepcopy(FROZEN_CFG),mixer_parameters=actual_mixer,
                      provenance=prov,environment_observed_at=utc_now(),inputs=inputs,reference=ref_identity,years=sorted(years),seeds=list(seeds),systems=list(models),
                      parameters=dict(**cfg,smoke=smoke,eligibility_window=60,rank_method='average percentile per date then mean across seeds'),
                      jobs=jobs,budget=dict(**ledger.snapshot(),ledger=identity(ledger.path),limit_seconds=ledger.limit),outputs={p.name:identity(p) for p in sorted(out.glob('*.parquet'))},
                      wall_seconds=time.monotonic()-started)
        if not (out/'manifest.json').exists(): write_json(out/'manifest.json',manifest)
        else:
            saved=json.loads((out/'manifest.json').read_text())
            if saved['outputs']!=manifest['outputs']: raise ValueError('completion manifest hash mismatch')
            return saved
        return manifest


def revalidate_manifest(manifest,directory):
    """Fail closed for confirmation; engineering evaluation can still report incomplete."""
    try:
        cfg=effective_config()
        if manifest['frozen_effective_config']!=cfg: return False
        if any(manifest['parameters'].get(k)!=v for k,v in cfg.items()): return False
        evidence=manifest['admission']['evidence']
        if identity(evidence['path'])!=evidence: return False
        env=runtime_environment('cuda')
        if official_admission(evidence['path'],env,cfg)!=manifest['admission']: return False
        if manifest['mixer_parameters']!=mixer_parameters(cfg): return False
        if manifest['provenance']['implementation']!=provenance()['implementation']: return False
        if manifest['provenance']['p3_spec']!=identity(ROOT/'tasks/spec-mamba-portfolio-p3-20261120.md'): return False
        if any(manifest['provenance']['environment'].get(k)!=v for k,v in env.items()): return False
        directory=Path(directory)
        contract=json.loads((directory/'run_identity.json').read_text())
        if (contract['parameters']!=cfg or contract['smoke'] is not False or
            contract['admission']!=manifest['admission'] or contract['mixer']!=manifest['mixer_parameters'] or
            contract['provenance']!=manifest['provenance'] or contract['inputs']!=manifest['inputs'] or
            contract['reference']!=manifest['reference'] or contract['frozen_effective_config']!=cfg or
            contract['years']!=list(range(2021,2027)) or sorted(contract['systems'])!=sorted(SYSTEMS) or
            sorted(contract['seeds'])!=[17,29]): return False
        run_hash=content_hash(contract)
        # Read while holding the same fixed-directory lock as every writer.
        with BudgetLedger(directory) as ledger:
            budget=manifest['budget']
            validate_snapshot({k:budget[k] for k in ('sequence','head_sha256','spent_seconds')},ledger.events)
            if (budget['ledger']!=identity(ledger.path) or budget['limit_seconds']!=14*3600 or
                budget['spent_seconds']!=ledger.spent or not 0<=ledger.spent<=14*3600): return False
            completed=set()
            pending=None
            for row in ledger.events:
                if row['event']=='START': pending=row
                elif row['event']=='END':
                    if row['status']=='COMPLETED' and pending:
                        binding=pending['binding']
                        if binding.get('run_hash')==run_hash:
                            completed.add((binding['fold'],binding['model'],binding['seed']))
                    pending=None
            expected={(y,m,s) for y in range(2021,2027) for m in SYSTEMS for s in (17,29)}
            if not expected<=completed or len(manifest['jobs'])!=24: return False
            for year,model,seed in sorted(expected):
                path=directory/f'fold_{year}_{model}_seed_{seed}.parquet'
                binding=dict(run_hash=run_hash,fold=year,model=model,seed=seed)
                cached=resume_fit(path,binding)
                if cached is None or manifest['outputs'][path.name]!=identity(path): return False
                frame,job=cached
                if job not in manifest['jobs'] or (job['fold'],job['system'],job['seed'])!=(year,model,seed): return False
                if set(frame.fold)!={year} or set(frame.system)!={model} or set(frame.seed)!={seed}: return False
            if manifest['outputs']['rank_average.parquet']!=identity(directory/'rank_average.parquet'): return False
        return True
    except (KeyError,TypeError,ValueError,RuntimeError,OSError,ImportError):
        return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds-root',type=Path)
    p.add_argument('--p1-scores-dir',type=Path,default=ROOT/'.artifacts/p1/full-scores-ace2eef')
    p.add_argument('--years',type=int,nargs='+',default=list(range(2021,2027)))
    p.add_argument('--models',choices=SYSTEMS,nargs='+',default=list(SYSTEMS))
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29])
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--smoke',action='store_true',help='1 epoch x 20 steps; never confirmatory')
    p.add_argument('--gpu-smoke-evidence',type=Path)
    p.add_argument('--run',action='store_true')
    p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--record-review',type=Path)
    p.add_argument('--review-kind',choices=('independent_worker','human'))
    p.add_argument('--job-id')
    p.add_argument('--verdict',choices=('GO','NO_GO'))
    p.add_argument('--rebuild-budget',action='store_true',help='explicit ledger recovery from surviving sidecars; requires --reason')
    p.add_argument('--clear-stop')
    p.add_argument('--reason')
    a=p.parse_args(argv)
    if sum((a.gpu_smoke,bool(a.record_review),bool(a.clear_stop),a.rebuild_budget,a.run))>1:
        p.error('choose one execution command')
    if a.rebuild_budget:
        if not a.reason or not a.reason.strip(): p.error('--reason required')
        return rebuild_budget(a.reason)
    if a.clear_stop:
        if not a.reason: p.error('--reason required')
        with BudgetLedger(ARTIFACT_ROOT,allow_stopped=True) as ledger: ledger.clear_stop(a.clear_stop,a.reason)
        return
    if not a.out_dir: p.error('--out-dir required')
    if a.gpu_smoke: return gpu_smoke(a.out_dir,a.job_id)
    if a.record_review:
        if not all((a.review_kind,a.job_id,a.verdict)): p.error('review-kind, job-id, verdict required')
        return record_review(a.record_review,a.out_dir,a.review_kind,a.job_id,a.verdict)
    if not a.folds_root: p.error('--folds-root required')
    out=a.out_dir.resolve(); root=ARTIFACT_ROOT.resolve()
    if out==root or not out.is_relative_to(root) or out.is_relative_to(WORK_ROOT.resolve()):
        p.error('output must be below .artifacts/p3/ outside work/')
    if (any(len(v)!=len(set(v)) for v in (a.years,a.models,a.seeds)) or
        not set(a.years)<=set(range(2021,2027)) or not set(a.seeds)<={17,29}):
        p.error('unique registered folds/models/seeds required')
    cfg=effective_config(a.smoke)
    plan=dict(dry_run=not a.run,years=sorted(a.years),models=a.models,seeds=a.seeds,
              fits=len(a.years)*len(a.models)*len(a.seeds),parameters=cfg,output=str(out),
              training_contract_status='UNRESOLVED',reason='official GPU smoke admission required')
    if not a.run:
        print(json.dumps(plan)); return
    return run_jobs(a.folds_root,a.years,a.models,a.seeds,out,a.p1_scores_dir,cfg,
                    smoke=a.smoke,gpu_smoke_evidence=a.gpu_smoke_evidence)


if __name__=='__main__':
    main()
