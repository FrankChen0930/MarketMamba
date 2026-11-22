"""P4 Colab engineering runner, isolated from the frozen P3 ledger/admission.

Epoch-boundary resume with identity-bound RNG/optimizer/scheduler state.
P4 record-only accounting is independent of the frozen P3 ledger.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid
import sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from .p4_bundle import digest, verify, MATRIX_FILES, prepare_source
from .rolling_seq_scores import checked_splits, require_same_rows, FROZEN_CFG, write_json, atomic_json

ARMS=('mamba2','crossstock','gru')


def hashed(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def require_cuda():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU runs cannot launch P4 speed/formal training')


def finite_nonnegative(*values):
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('finite nonnegative budget values required')


class BudgetStop(RuntimeError): pass


def budget_charge(wall_seconds, guard_wait_seconds, units_per_hour):
    finite_nonnegative(wall_seconds,guard_wait_seconds,units_per_hour)
    if guard_wait_seconds>wall_seconds or units_per_hour<=0: raise ValueError('invalid wait/rate')
    return dict(wall_hours=wall_seconds/3600,active_hours=(wall_seconds-guard_wait_seconds)/3600,
                compute_units=(wall_seconds-guard_wait_seconds)/3600*units_per_hour,guard_wait_seconds=float(guard_wait_seconds))


def check_budget(max_hours,max_units,spent_hours,spent_units,planned_hours,planned_units,reserve):
    finite_nonnegative(max_hours,max_units,spent_hours,spent_units,planned_hours,planned_units,reserve)
    if reserve<.25 or max_hours<=0 or max_units<=0: raise ValueError('at least 25% reserve and positive caps required')
    if spent_hours+planned_hours*(1+reserve)>max_hours or spent_units+planned_units*(1+reserve)>max_units:
        raise BudgetStop('STOP: plan plus reserve exceeds Colab wall-hour / compute-unit cap')


def validate_budget_plan(max_hours,max_units,rate,planned,reserve):
    finite_nonnegative(max_hours,max_units,rate,planned,reserve)
    if min(max_hours,max_units,rate,planned)<=0 or reserve<.25:
        raise ValueError('positive caps/rate/estimate and >=25% reserve required')


def check_plan_budget(contract,spent_hours,spent_units,remaining_fits,planned_hours):
    if type(remaining_fits) is not int or remaining_fits<0: raise ValueError('invalid remaining fit count')
    total=remaining_fits*planned_hours
    check_budget(contract['max_hours'],contract['max_units'],spent_hours,spent_units,
                 total,total*contract['units_per_hour'],contract['reserve'])


def validate_contract(contract,years,speed):
    if contract['recipe']!=FROZEN_CFG or contract['arms']!=list(ARMS) or contract['seeds']!=[17,29]:
        raise ValueError('P4 recipe/arms/seed drift')
    required=contract['confirmatory_years']+contract['descriptive_years']
    allowed=required+contract['optional_years']
    if not years or len(set(years))!=len(years) or not set(years)<=set(allowed):
        raise ValueError('P4 fold mismatch')
    if not speed:
        if contract['status']!='FROZEN' or contract.get('contract_status')!='FROZEN': raise RuntimeError('P4 contract DRAFT: formal execution requires Human/Claude freeze')
        if not set(required)<=set(years): raise ValueError('formal P4 requires all primary/descriptive folds')


def publish_bytes(path,payload):
    """Serialize/fsync locally, exclusive durable copy/fsync, reread SHA; never overwrite."""
    import tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryFile() as local:
        local.write(payload);local.flush();os.fsync(local.fileno());local.seek(0)
        expected=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        with path.open('xb') as dst:
            import shutil
            shutil.copyfileobj(local,dst);dst.flush();os.fsync(dst.fileno())
    if digest(path)!=expected: raise ValueError('durable copy hash mismatch')
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return expected


def validate_fit_outputs(directory, contract, outputs):
    """Required schema plus bytes and semantic checks, before publish and resume."""
    from .p4_bundle import safe_name
    directory=Path(directory)
    for name,ident in outputs.items():
        safe_name(name)
        if any(p.is_symlink() for p in (directory/name, *(directory/name).parents)):
            raise ValueError('fit output links forbidden')
        if digest(directory/name)!=ident: raise ValueError('fit output hash mismatch')
    parents={str(Path(n).parent) for n in outputs}
    if len(parents)!=1: raise ValueError('fit requires one complete attempt')
    attempt_name=next(iter(parents))
    attempt=directory/attempt_name
    names={Path(n).name for n in outputs}
    if contract.get('speed_test'):
        if names!={'speed.json'}: raise ValueError('speed outputs must contain timing only')
        report=json.loads((attempt/'speed.json').read_text())
        allowed={'version','evidence_class','arm','epochs','steps','epoch_seconds','step_seconds',
                 'step_seconds_mean','gpu_model','telemetry','total_seconds','guard_wait_seconds',
                 'environment_mismatch','comparability'}
        if set(report)!=allowed or report['evidence_class']!='ENGINEERING_SPEED_ONLY':
            raise ValueError('speed schema mismatch')
        return
    required={'best.pt','scores.parquet','reference.parquet','metrics.json','checkpoint-pointer.json'}
    if not required<=names: raise ValueError('completed fit missing required artifacts')
    metrics=json.loads((attempt/'metrics.json').read_text())
    if not metrics.get('history') or not np.isfinite(metrics.get('best_val_ic5',np.nan)):
        raise ValueError('invalid metrics')
    frame=pd.read_parquet(attempt/'scores.parquet')
    reference=pd.read_parquet(attempt/'reference.parquet')
    require_same_rows(frame,reference)
    if len(frame)!=metrics['eval_rows'] or not np.isfinite(frame['score']).all():
        raise ValueError('invalid predictions')
    pointer=json.loads((attempt/'checkpoint-pointer.json').read_text())
    import torch
    checkpoints={}
    for role in ('latest','previous','best'):
        entry=pointer.get(role)
        if not entry:
            if role=='previous' and pointer['latest']['epoch']==1:continue
            raise ValueError('required checkpoint pointer missing')
        safe_name(entry['file'])
        if Path(entry['file']).name!=entry['file']: raise ValueError('checkpoint filename must be local')
        name=(Path(attempt_name)/entry['file']).as_posix()
        if name not in outputs or outputs[name]!=entry['identity']:
            raise ValueError('checkpoint pointer identity mismatch')
        payload=torch.load(directory/name,map_location='cpu',weights_only=False)
        if not {'contract','model','opt','sched','state','best_sd','rng_state','torch_rng'}<=payload.keys():
            raise ValueError('checkpoint schema incomplete')
        if payload['contract']!=contract or payload['state']['epoch']!=entry['epoch']:
            raise ValueError('checkpoint contract/epoch mismatch')
        checkpoints[role]=payload
    if not ((pointer['previous'] is None and pointer['latest']['epoch']==1 or pointer['previous'] is not None and pointer['previous']['epoch']<pointer['latest']['epoch']) and
            pointer['best']['epoch']<=pointer['latest']['epoch']):
        raise ValueError('checkpoint pointer order mismatch')
    best=torch.load(attempt/'best.pt',map_location='cpu',weights_only=False)
    selected=checkpoints['best']['best_sd']
    if (best['contract']!=contract or best['best_val_ic5']!=metrics['best_val_ic5'] or
        best['best_val_ic5']!=checkpoints['best']['state']['best'] or set(best['model'])!=set(selected) or
        any(not torch.equal(v,selected[k]) for k,v in best['model'].items())):
        raise ValueError('BEST weights/pointer mismatch')


def complete_fit(directory,contract,job,files):
    directory=Path(directory)
    outputs={n:digest(directory/n) for n in files}
    validate_fit_outputs(directory,contract,outputs)
    if (directory/'complete.json').exists():raise FileExistsError(directory/'complete.json')
    atomic_json(directory/'complete.json',dict(version='p4-fit-v1',contract=contract,contract_hash=hashed(contract),
        job=job,outputs=outputs,evidence_class='ENGINEERING_SPEED_ONLY' if contract.get('speed_test') else 'E2 RESEARCH_ONLY / NOT_ACCEPTED'))


def resume_fit(directory,contract):
    path=Path(directory)/'complete.json'
    if not path.exists(): return None
    mf=json.loads(path.read_text())
    if mf['contract']!=contract or mf['contract_hash']!=hashed(contract): raise ValueError('resume identity mismatch')
    validate_fit_outputs(directory,contract,mf['outputs'])
    return mf


class Ledger:
    """Append-only full-state generations + exact head pointer; missing/truncated state stops."""
    def __init__(self,root,contract):
        self.root=Path(root);self.contract=contract
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=None

    def __enter__(self):
        import fcntl
        self.lock=(self.root/'writer.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            events=sorted(self.root.glob('event-*.json'));previous=None
            self.state=dict(contract=self.contract,spent_hours=0.,spent_units=0.,active=None,completed=[],stopped=False)
            for i,p in enumerate(events,1):
                if p.name!=f'event-{i:06d}.json': raise ValueError('ledger gap')
                e=json.loads(p.read_text())
                if e['previous']!=previous or e['state']['contract']!=self.contract: raise ValueError('ledger identity/chain mismatch')
                previous=digest(p);self.state=e['state']
            self.count=len(events);self.previous=previous
            head=self.root/'head.json'
            if events:
                if not head.exists() or json.loads(head.read_text())!=dict(sequence=self.count,identity=previous): raise ValueError('ledger head mismatch')
            elif head.exists() or any(self.root.glob('run-*')):
                raise ValueError('missing ledger; do not reset spent')
            if self.state['active']:
                # Unknown crash interval conservatively charged through restart, no invented guard wait.
                elapsed=max(0,time.time()-self.state['active']['started'])/3600
                self.state['spent_hours']+=elapsed;self.state['spent_units']+=elapsed*self.contract['units_per_hour']
                self.state['last_attempt']=dict(self.state['active'],wall_seconds=elapsed*3600,ok=False,interval='UNKNOWN_THROUGH_RESTART')
                self.state['active']=None
                self.state['stopped']=self.state['stopped'] or self.contract.get('policy')!='record_only'
                self.append('INTERRUPTED' if not self.state['stopped'] else 'INTERRUPTED_STOP')
            if self.state['stopped']: raise BudgetStop('STOP latched: inspect ledger; no automatic reset/clear')
            return self
        except BaseException:
            self.lock.close();raise

    def __exit__(self,*args): self.lock.close()

    def append(self,event):
        self.count+=1
        p=self.root/f'event-{self.count:06d}.json'
        write_json(p,dict(event=event,previous=self.previous,state=self.state,observed_at=time.time()))
        self.previous=digest(p)
        atomic_json(self.root/'head.json',dict(sequence=self.count,identity=self.previous))

    def start(self,binding,planned_hours):
        if self.contract.get('policy')=='record_only':
            self.state['active']=dict(binding=binding,started=time.time());self.append('START');return
        check_budget(self.contract['max_hours'],self.contract['max_units'],self.state['spent_hours'],self.state['spent_units'],
                     planned_hours,planned_hours*self.contract['units_per_hour'],self.contract['reserve'])
        self.state['active']=dict(binding=binding,started=time.time());self.append('START')

    def check(self):
        if self.contract.get('policy')=='record_only': return
        elapsed=(time.time()-self.state['active']['started'])/3600
        check_budget(self.contract['max_hours'],self.contract['max_units'],self.state['spent_hours'],self.state['spent_units'],
                     elapsed,elapsed*self.contract['units_per_hour'],self.contract['reserve'])

    def observe(self,record):
        active=self.state['active']
        if active is None:raise ValueError('telemetry outside active attempt')
        first='telemetry' not in active
        stats=active.setdefault('telemetry',dict(gpu=record['gpu'],samples=0,
            temperature_c_min=record['temperature_c'],temperature_c_max=record['temperature_c'],
            clock_mhz_min=record['clock_mhz'],clock_mhz_max=record['clock_mhz']))
        if stats['gpu']!=record['gpu']:raise ValueError('GPU identity changed during attempt')
        stats['samples']+=1;stats['last']=record
        for key in ('temperature_c','clock_mhz'):
            stats[key+'_min']=min(stats[key+'_min'],record[key])
            stats[key+'_max']=max(stats[key+'_max'],record[key])
        if record['temperature_c']>=85:
            self.state['stopped']=True;self.append('THERMAL_STOP')
        elif first:self.append('TELEMETRY_START')


    def end(self,binding,ok):
        elapsed=time.time()-self.state['active']['started']
        charge=(dict(wall_hours=elapsed/3600,active_hours=elapsed/3600,compute_units=0.,guard_wait_seconds=0.)
                if self.contract.get('policy')=='record_only' else budget_charge(elapsed,0,self.contract['units_per_hour']))
        self.state['spent_hours']+=charge['active_hours'];self.state['spent_units']+=charge['compute_units']
        self.state['last_attempt']=dict(self.state['active'],wall_seconds=elapsed,ok=ok)
        self.state['active']=None;self.state['last_charge']=charge
        if ok:self.state['completed'].append(binding)
        elif self.contract.get('policy')!='record_only':self.state['stopped']=True
        self.append('END' if ok else 'STOP')


def telemetry():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=result.stdout.strip().splitlines()
    if len(rows)!=1: raise RuntimeError('one visible GPU required')
    name,temp,clock,power=[v.strip() for v in rows[0].split(',')]
    values=[float(v) for v in (temp,clock,power)]
    finite_nonnegative(*values)
    return dict(gpu=name,temperature_c=values[0],clock_mhz=values[1],power_w=values[2],observed_at=time.time())


def speed_report(arm,history,records,total,steps,environment=None):
    if len(history)!=2 or len(steps)!=200 or not records: raise ValueError('speed test requires exactly 2 epochs x 100 steps')
    finite_nonnegative(total,*steps,*[v['seconds'] for v in history])
    return dict(version='p4-speed-v1',evidence_class='ENGINEERING_SPEED_ONLY',arm=arm,epochs=2,steps=200,
                epoch_seconds=[v['seconds'] for v in history],step_seconds=steps,step_seconds_mean=float(np.mean(steps)),
                gpu_model=records[0]['gpu'],telemetry=records,total_seconds=total,guard_wait_seconds=0,
                environment_mismatch=(environment or {}).get('environment_mismatch',[]),
                comparability=(environment or {}).get('comparability','UNVERIFIED'))


class Observer:
    def __init__(self,root,contract,ledger,device='cuda',speed=False):
        self.root=Path(root);self.contract=contract;self.ledger=ledger;self.device=device
        self.speed=speed
        self.steps=[];self.records=[];self.pointer=dict(latest=None,previous=None,best=None)
        self.best=-9.;self.step_start=None

    def load_resume(self):
        """Pointer is the commit record. Orphans are preserved and never promoted."""
        import torch
        pointer=self.root/'checkpoint-pointer.json'
        if not pointer.exists():
            if any(self.root.glob('epoch-*.pt')):
                raise ValueError('missing checkpoint pointer; orphan recovery requires explicit review')
            return None
        saved=json.loads(pointer.read_text());payloads={}
        for role in ('latest','previous','best'):
            entry=saved.get(role)
            if entry is None:
                if role=='previous':continue
                raise ValueError('checkpoint pointer incomplete')
            from .p4_bundle import safe_name
            safe_name(entry['file'])
            if Path(entry['file']).name!=entry['file']:raise ValueError('checkpoint path')
            path=self.root/entry['file']
            if path.is_symlink() or not path.is_file() or digest(path)!=entry['identity']:
                raise ValueError('checkpoint hash mismatch')
            c=torch.load(path,map_location='cpu',weights_only=False)
            if c.get('contract')!=self.contract:raise ValueError('checkpoint identity mismatch')
            required={'model','opt','sched','state','best_sd','rng_state','numpy_rng','torch_rng','cuda_rng'}
            if not required<=c.keys() or c['state']['epoch']!=entry['epoch'] or c['best_sd'] is None:
                raise ValueError('checkpoint schema/epoch mismatch')
            payloads[role]=c
        if saved['best']['epoch']>saved['latest']['epoch'] or (saved['previous'] and saved['previous']['epoch']>=saved['latest']['epoch']):
            raise ValueError('checkpoint pointer order')
        self.pointer=saved;self.best=payloads['latest']['state']['best']
        return payloads['latest']


    def check(self):
        self.ledger.check()
        r=telemetry();self.records.append(r)
        if hasattr(self.ledger,'observe'):self.ledger.observe(r)
        if r['temperature_c']>=85: raise RuntimeError('STOP: Colab GPU temperature >=85C (no pause/retry)')

    def __call__(self,event,payload):
        import torch
        if event=='step_start':
            self.check()
            if str(self.device).startswith('cuda'):torch.cuda.synchronize()
            self.step_start=time.perf_counter()
        elif event=='step_end':
            if str(self.device).startswith('cuda'):torch.cuda.synchronize()
            self.steps.append(time.perf_counter()-self.step_start)
        elif event=='epoch':
            if self.speed: return  # Never serialize state/history/weights in timing mode.
            epoch=payload['state']['epoch'];name=f'epoch-{epoch:03d}-{uuid.uuid4().hex}.pt'
            stream=io.BytesIO();torch.save(dict(contract=self.contract,**payload,
                cuda_rng=torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else None),stream)
            ident=publish_bytes(self.root/name,stream.getvalue())
            entry=dict(file=name,identity=ident,epoch=epoch)
            self.pointer['previous']=self.pointer['latest'];self.pointer['latest']=entry
            if payload['state']['best']>self.best:
                self.best=payload['state']['best'];self.pointer['best']=entry
            atomic_json(self.root/'checkpoint-pointer.json',self.pointer)
            if hasattr(self.ledger,'append'):self.ledger.append('EPOCH_COMMITTED')


def rankgauss(directory):
    from .seq_runner import ensure_rankgauss, _x_tag
    d=Path(directory);cache=d/'X_rankgauss.npy';meta=d/'p4-rankgauss.json'
    if any((d/n).is_symlink() for n in ('X_rankgauss.npy','X_rankgauss.tag','p4-rankgauss.json')):
        raise ValueError('rankgauss derived links forbidden')
    inputs={n:digest(d/n) for n in MATRIX_FILES}
    impl=digest(Path(__file__).with_name('seq_runner.py'))
    contract=dict(inputs=inputs,implementation=impl)
    if cache.exists() or meta.exists():
        if not cache.exists() or not meta.exists(): raise ValueError('rankgauss orphan; use verified fresh extraction')
        saved=json.loads(meta.read_text())
        if saved['contract']!=contract or saved['output']!=digest(cache) or (d/'X_rankgauss.tag').read_text()!=_x_tag(d):
            raise ValueError('rankgauss identity/tag mismatch')
    else:
        ensure_rankgauss(d)
        if (d/'X_rankgauss.tag').read_text()!=_x_tag(d): raise ValueError('rankgauss tag mismatch')
        write_json(meta,dict(contract=contract,output=digest(cache)))
    return digest(cache)


def environment_snapshot(package_names):
    import importlib.metadata
    import platform
    import sysconfig
    import torch
    packages={}
    for name in sorted(package_names):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return dict(packages=packages,python=platform.python_version(),cuda=torch.version.cuda,
                cxx11abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                platform=sysconfig.get_platform().replace('-','_'),gpu=torch.cuda.get_device_name(0))


def environment(lock, runtime_contract=None, speed=False):
    """Colab contract governs formal runs; local P3 lock differences remain explicit."""
    require_cuda()  # Never waived for engineering speed tests.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from urllib.parse import unquote, urlsplit
    requirements=[]
    for line in Path(lock).read_text().splitlines():
        if line.strip() and not line.startswith('#'): requirements.append(Requirement(line))
    if runtime_contract is None:
        runtime_contract=Path(__file__).resolve().parents[2]/'environments/p4a-colab-runtime.json'
    contract=json.loads(Path(runtime_contract).read_text())
    expected_packages={canonicalize_name(k):v for k,v in contract['packages'].items()}
    actual=environment_snapshot({canonicalize_name(r.name) for r in requirements}|set(expected_packages))
    local=[];mismatch=[]
    def difference(target,item,expected,observed):
        target.append(dict(item=item,expected=expected,actual=observed))
    for req in requirements:
        name=canonicalize_name(req.name);value=actual['packages'].get(name)
        if req.url:
            expected=str(parse_wheel_filename(Path(unquote(urlsplit(req.url).path)).name)[1])
            matches=value==expected
        else:
            expected=str(req.specifier);matches=value is not None and value in req.specifier
        if not matches: difference(local,'lock:'+name,expected,value)
    if contract.get('version')!='p4-colab-runtime-v1':
        difference(mismatch,'contract:version','p4-colab-runtime-v1',contract.get('version'))
    if contract.get('status')!='OBSERVED' or not contract.get('observed_at'):
        difference(mismatch,'contract:status','OBSERVED with observed_at',contract.get('status'))
    for key in ('python','cuda','cxx11abi','platform'):
        expected=contract['runtime'].get(key)
        if expected in (None,'TBD') or type(expected)!=type(actual.get(key)) or expected!=actual.get(key):
            difference(mismatch,'contract:'+key,expected,actual.get(key))
    # Empty/incomplete contracts cannot authorize a formal run.
    for name in sorted(set(expected_packages)|{'torch','mamba-ssm','causal-conv1d','numpy','pandas','pyarrow','scipy','scikit-learn','einops','packaging'}):
        expected=expected_packages.get(name);value=actual['packages'].get(name)
        if expected in (None,'TBD') or expected!=value:
            difference(mismatch,'contract:'+name,expected,value)
    if speed: mismatch=local+mismatch
    if mismatch and not speed:
        raise ValueError('exact Colab environment mismatch: '+json.dumps(mismatch,sort_keys=True))
    actual.update(lock=digest(lock),runtime_contract=digest(runtime_contract),
                  environment_mismatch=mismatch,local_lock_mismatch=local,
                  comparability='結果不可與正式環境比較' if mismatch else 'Colab contract matched; P3 cross-environment differences must be disclosed')
    print(json.dumps(dict(environment_mismatch=mismatch,local_lock_mismatch=local,comparability=actual['comparability'])))
    return actual


def train_fit(directory,year,arm,seed,cfg,reference,observer):
    """Reuse P3 checked_splits/row admission and existing train_one; P3 collect owns a different ledger."""
    from .seq_runner import SeqData,train_one,predict
    data=SeqData(directory,window=cfg.get('window',60),normalize='rankgauss');tr,va,ev=checked_splits(data,cfg['val_dates'])
    frame=pd.DataFrame(dict(fold=year,date=pd.to_datetime(np.load(directory/'dates.npy',mmap_mode='r')[ev]),
        code=np.asarray(np.load(directory/'stock_ids.npy',mmap_mode='r')[ev]),label_y5=data.y['y5'][ev]))
    if cfg.get('window',60)==60:require_same_rows(frame,reference)
    else:
        from .p4_eval import common_rows
        aligned,support=common_rows({'model':frame,'c0plus':reference})
        if len(aligned['model'])!=len(frame):raise ValueError('long-window rows outside C0+ reference')
    if cfg['val_max_rows'] and len(va)>cfg['val_max_rows']:
        va=np.sort(np.random.default_rng(0).choice(va,cfg['val_max_rows'],replace=False))
    windows=data.windows
    def guarded(rows):
        observer.check();return windows(rows)
    data.windows=guarded
    restored=observer.load_resume() if not observer.speed else None
    model,state=train_one(arm,seed,data,tr,va,cfg,'cuda',observer=observer,resume_state=restored,
                          log=(lambda *_: None) if observer.speed else print)
    if observer.speed:
        return None,dict(history=[dict(seconds=h['seconds']) for h in state['history']])
    if not np.isfinite(state['best']) or state['best']<=-9: raise ValueError('BEST unavailable; LAST prohibited')
    frame['score']=predict(model,data,ev,'cuda')[:,0];frame['seed']=seed;frame['system']=arm
    frame['window']=cfg.get('window',60);frame['lr']=cfg['lr']
    if not np.isfinite(frame.score).all(): raise ValueError('nonfinite predictions')
    best=io.BytesIO();import torch
    torch.save(dict(contract=observer.contract,model=model.state_dict(),best_val_ic5=state['best']),best)
    publish_or_validate(observer.root/'best.pt',best.getvalue())
    return frame,dict(history=state['history'],best_val_ic5=state['best'],fit_rows=len(tr),val_rows=len(va),eval_rows=len(ev),
        parameters=sum(p.numel() for p in model.parameters()),telemetry=observer.records)



def fit_queue(contract, speed=False, years=None):
    years=years or contract['confirmatory_years']+contract['descriptive_years']
    if speed:
        return [dict(queue='SPEED',year=years[0],arm=a,seed=17,window=60,lr=3e-4) for a in ARMS]
    jobs=[]
    for queue,folds in (('Q1',contract['confirmatory_years']),('Q2',contract['descriptive_years'])):
        jobs += [dict(queue=queue,year=y,arm=a,seed=seed,window=60,lr=3e-4)
                 for y in folds for a in ARMS for seed in contract['seeds']]
    jobs += [dict(queue='Q3',year=y,arm=a,seed=seed,window=w,lr=3e-4)
             for w in (120,250) for y in (2021,2022,2023) for a in ('mamba2','gru') for seed in (17,29)]
    jobs += [dict(queue='Q4',year=2022,arm=a,seed=17,window=60,lr=lr)
             for lr in (1e-4,1e-3) for a in ('mamba2','gru')]
    return jobs


def run(bundle,drive_root,speed=False,max_hours=None,max_units=None,rate=None,planned_hours=0,reserve=None,data_dir=None):
    """P4-only record accounting; old cap arguments are accepted but never constrain runs."""
    require_cuda()
    bundle=Path(bundle);drive_root=Path(drive_root)
    if (not os.path.ismount('/content/drive') or
        not drive_root.resolve().is_relative_to(Path('/content/drive/MyDrive'))):
        raise ValueError('durable output requires mounted /content/drive/MyDrive')
    run_started=time.perf_counter()
    budget_contract=dict(policy='record_only',units_per_hour=0,thermal_stop_c=85)
    with Ledger(drive_root,budget_contract) as ledger:
        setup_binding=dict(stage='verify_bundle_and_runtime',bundle_manifest=digest(bundle/'bundle-manifest.json'))
        ledger.start(setup_binding,0)
        try:
            mf=verify(bundle,allow_derived=True);prepare_source(bundle)
            source_root=Path(__file__).resolve().parents[2]
            for name,ident in mf['files'].items():
                if name.startswith('source/') and digest(source_root/name.removeprefix('source/'))!=ident:
                    raise ValueError('executed source differs from bundle')
            env=environment(bundle/'requirements.lock.txt',
                runtime_contract=bundle/'source/environments/p4a-colab-runtime.json',speed=speed)
            contract=json.loads((bundle/'source/tasks/p4a-colab-contract-20261121.json').read_text())
            validate_contract(contract,mf['metadata']['years'],speed)
            if digest(bundle/'source/environments/p4a-colab-runtime.json')['sha256']!=contract['colab_runtime_contract']['sha256']:
                raise ValueError('frozen runtime contract identity mismatch')
            if digest(bundle/'requirements.lock.txt')['sha256']!=contract['environment_lock']['sha256']:
                raise ValueError('P4 exact lock hash mismatch')
            external=mf['metadata'].get('external_dataset')
            if external:
                if data_dir is None:raise ValueError('SSD data-dir required')
                prepared=json.loads((Path(data_dir)/'prepared.json').read_text())
                if prepared['matrix_identity']!=external:raise ValueError('matrix identity mismatch')
                dataset=json.loads((bundle/'DATASET_MANIFEST.json').read_text())
                for name,ident in dataset['files'].items():
                    path=Path(data_dir)/name
                    if path.is_symlink() or digest(path)!=ident:raise ValueError('SSD matrix hash mismatch')
                cache_id=digest(Path(data_dir)/'shared/X_rankgauss.npy')
                if cache_id!=prepared['rankgauss']:raise ValueError('rankgauss hash mismatch')
            ref_name='keys.parquet' if external else 'rank_average.parquet'
            ref=pd.read_parquet(bundle/'reference'/ref_name)
            if 'system' in ref:ref=ref[ref.system.eq('c0plus')]
            expected=json.loads((bundle/'reference/manifest.json').read_text())['outputs'][ref_name]
            if digest(bundle/'reference'/ref_name)!= {k:expected[k] for k in ('bytes','sha256')}:
                raise ValueError('P1 reference manifest mismatch')
            queue=fit_queue(contract,speed,mf['metadata']['years'])
            binding=dict(bundle=digest(bundle/'bundle-manifest.json'),matrix=external,
                         environment=env,contract=hashed(contract) if speed else contract,
                         queue=queue,speed_test=speed)
            run_hash=hashed(binding);ledger.end(setup_binding,True)
        except BaseException:
            if ledger.state['active']:ledger.end(setup_binding,False)
            raise
        out=drive_root/f'run-{run_hash}'
        if out.exists():
            if json.loads((out/'run_identity.json').read_text())!=binding:raise ValueError('run identity mismatch')
        else:out.mkdir();write_json(out/'run_identity.json',binding)
        reports=[];caches={};fit_paths={}
        for job in queue:
            year,arm,seed=job['year'],job['arm'],job['seed']
            directory=(Path(data_dir) if external else bundle)/f'folds/fold_{year}'
            if year not in caches:
                prep_binding=dict(run_hash=run_hash,stage='rankgauss_verify_or_build',year=year)
                ledger.start(prep_binding,0)
                try:
                    caches[year]=cache_id if external else rankgauss(directory)
                    ledger.end(prep_binding,True)
                except BaseException:
                    if ledger.state['active']:ledger.end(prep_binding,False)
                    raise
            cfg=deepcopy(contract['recipe']);cfg.update(window=job['window'],lr=job['lr'])
            if speed:cfg.update(epochs=2,steps_per_epoch=100,patience=3)
            fit_contract=dict(run_hash=run_hash,**job,rankgauss=caches[year],speed_test=speed)
            fit=out/f"{job['queue']}-{year}-{arm}-{seed}-w{job['window']}-lr{job['lr']}"
            fit_paths[(year,arm,seed,job['window'],job['lr'])]=fit
            cached=resume_fit(fit,fit_contract)
            if cached:reports.append(cached);continue
            ledger.start(fit_contract,0)
            # Stable directory permits retry from the last pointer. Orphan epoch files remain forensic.
            attempt=fit/'attempt-active';attempt.mkdir(parents=True,exist_ok=True)
            observer=Observer(attempt,fit_contract,ledger,speed=speed);started=time.perf_counter()
            try:
                frame,metrics=train_fit(directory,year,arm,seed,cfg,ref[ref.fold.eq(year)],observer)
                if speed:
                    report=speed_report(arm,metrics['history'],observer.records,time.perf_counter()-started,observer.steps,environment=env)
                    atomic_json(attempt/'speed.json',report);metrics=report
                else:
                    # A prior crash after prediction publication must validate identical bytes.
                    reference=ref[ref.fold.eq(year)]
                    if job['window']!=60:
                        from .p4_eval import common_rows
                        aligned,_=common_rows({'model':frame,'reference':reference});reference=aligned['reference']
                    for name,table in (('scores.parquet',frame),('reference.parquet',reference)):
                        buffer=io.BytesIO();table.to_parquet(buffer,index=False)
                        publish_or_validate(attempt/name,buffer.getvalue())
                    metrics.update(wall_seconds=time.perf_counter()-started,guard_wait_seconds=0)
                    atomic_json(attempt/'metrics.json',metrics)
                files=[p.relative_to(fit).as_posix() for p in attempt.iterdir() if p.is_file()]
                complete_fit(fit,fit_contract,metrics,files)
                ledger.end(fit_contract,True);reports.append(resume_fit(fit,fit_contract))
            except BaseException as error:
                if 'temperature >=85C' in str(error):ledger.state['stopped']=True
                if ledger.state['active']:ledger.end(fit_contract,False)
                raise
        if not speed:
            for year in (2021,2022,2023):
                diagnostic=out/f'permutation-{year}.json'
                identity=dict(run_hash=run_hash,year=year,R=10)
                if diagnostic.exists():
                    saved=json.loads(diagnostic.read_text())
                    if saved['identity']!=identity or saved['sha256']!=hashed(saved['result']):
                        raise ValueError('permutation identity/hash mismatch')
                    continue
                ledger.start(dict(stage='Q5',**identity),0)
                try:
                    directory=(Path(data_dir) if external else bundle)/f'folds/fold_{year}'
                    result=permutation_fit(directory,year,contract['recipe'],fit_paths,ledger)
                    write_json(diagnostic,dict(identity=identity,result=result,sha256=hashed(result)))
                    ledger.end(dict(stage='Q5',**identity),True)
                except BaseException as error:
                    if 'temperature >=85C' in str(error):ledger.state['stopped']=True
                    if ledger.state['active']:ledger.end(dict(stage='Q5',**identity),False)
                    raise
        payload=dict(version='p4-run-v2',binding=binding,jobs=reports,budget=ledger.state,
                     wall_seconds=time.perf_counter()-run_started,guard_wait_seconds=0,
                     evidence_class='ENGINEERING_SPEED_ONLY' if speed else 'EXPOSED_HISTORICAL_RESEARCH / NOT_ACCEPTED')
        if not speed:payload['research_conclusion']=None
        atomic_json(out/'manifest.json',payload)
        return payload


def publish_or_validate(path,payload):
    expected=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    if Path(path).exists():
        if digest(path)!=expected:raise ValueError('existing artifact hash mismatch')
        return expected
    temporary=Path(path).with_name(Path(path).name+'.partial-'+uuid.uuid4().hex)
    result=publish_bytes(temporary,payload)
    os.replace(temporary,path)
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return result


def permutation_fit(directory,year,cfg,fit_paths,ledger):
    import torch
    from .seq_runner import SeqData,predict
    from .seq_models import SeqRanker
    from .cross_stock import CrossStockRanker,predict_cross
    from .p4_eval import permutation_summary
    data=SeqData(directory,window=60);_,va,ev=checked_splits(data,cfg['val_dates'])
    cross=CrossStockRanker(n_features=data.X.shape[1],d_model=cfg['d_model'],n_layers=cfg['n_layers'],
                          n_cross=cfg['n_cross'],mixer_kw=cfg['mixer_kw'],dropout=cfg['dropout']).cuda()
    mamba=SeqRanker('mamba2',data.X.shape[1],cfg['d_model'],cfg['n_layers'],mixer_kw=cfg['mixer_kw']).cuda()
    for arm,model in [('crossstock',cross),('mamba2',mamba)]:
        fit=fit_paths[(year,arm,17,60,3e-4)]
        complete=json.loads((fit/'complete.json').read_text())
        resume_fit(fit,complete['contract'])
        model.load_state_dict(torch.load(fit/'attempt-active/best.pt',map_location='cuda',weights_only=False)['model'])
    original=data.windows
    def guarded(rows):
        ledger.check();r=telemetry();ledger.observe(r)
        if r['temperature_c']>=85:raise RuntimeError('STOP: Colab GPU temperature >=85C (no pause/retry)')
        return original(rows)
    data.windows=guarded
    result={}
    for split,rows in [('validation',va),('evaluation',ev)]:
        base=predict_cross(cross,data,rows,'cuda',n_perm=2,seed=0)[:,0]
        temporal=predict(mamba,data,rows,'cuda')[:,0]
        permutations=[predict_cross(cross,data,rows,'cuda',n_perm=1,seed=100+r)[:,0] for r in range(10)]
        result[split]=permutation_summary(data.date_idx[rows],np.asarray(np.load(directory/'stock_ids.npy',mmap_mode='r')[rows]),
                                          data.y['y5'][rows],base,temporal,permutations)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--drive-root',type=Path,required=True)
    p.add_argument('--speed-test',action='store_true');p.add_argument('--data-dir',type=Path)
    # Legacy flags are optional and ignored: P4 has no wall/CU caps.
    p.add_argument('--max-hours',type=float);p.add_argument('--max-units',type=float)
    p.add_argument('--units-per-hour',type=float);p.add_argument('--planned-hours-per-fit',type=float,default=0)
    p.add_argument('--reserve',type=float)
    a=p.parse_args(argv)
    result=run(a.bundle,a.drive_root,speed=a.speed_test,data_dir=a.data_dir)
    print(json.dumps(dict(version=result['version'],jobs=len(result['jobs']),evidence_class=result['evidence_class'],
                          environment_mismatch=result['binding']['environment'].get('environment_mismatch',[]),
                          comparability=result['binding']['environment'].get('comparability'))))

if __name__=='__main__':main()
