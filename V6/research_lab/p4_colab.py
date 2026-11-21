"""P4 Colab engineering runner, isolated from the frozen P3 ledger/admission.

Completed-fit resume only. Interrupted epochs remain forensic checkpoints;
restart uses a fresh attempt with the same seed/recipe, never LAST weights.
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

import numpy as np
import pandas as pd
from .p4_bundle import digest, verify, MATRIX_FILES
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
        if contract['status']!='FROZEN': raise RuntimeError('P4 contract DRAFT: formal execution requires Human/Claude freeze')
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


def complete_fit(directory,contract,job,files):
    directory=Path(directory)
    write_json(directory/'complete.json',dict(version='p4-fit-v1',contract=contract,contract_hash=hashed(contract),
        job=job,outputs={n:digest(directory/n) for n in files},evidence_class='E2 RESEARCH_ONLY / NOT_ACCEPTED'))


def resume_fit(directory,contract):
    path=Path(directory)/'complete.json'
    if not path.exists(): return None
    mf=json.loads(path.read_text())
    if mf['contract']!=contract or mf['contract_hash']!=hashed(contract): raise ValueError('resume identity mismatch')
    for name,ident in mf['outputs'].items():
        from .p4_bundle import safe_name
        safe_name(name)
        if digest(Path(directory)/name)!=ident: raise ValueError('fit output hash mismatch')
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
                self.state['active']=None;self.state['stopped']=True;self.append('INTERRUPTED_STOP')
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
        check_budget(self.contract['max_hours'],self.contract['max_units'],self.state['spent_hours'],self.state['spent_units'],
                     planned_hours,planned_hours*self.contract['units_per_hour'],self.contract['reserve'])
        self.state['active']=dict(binding=binding,started=time.time());self.append('START')

    def check(self):
        elapsed=(time.time()-self.state['active']['started'])/3600
        check_budget(self.contract['max_hours'],self.contract['max_units'],self.state['spent_hours'],self.state['spent_units'],
                     elapsed,elapsed*self.contract['units_per_hour'],self.contract['reserve'])

    def end(self,binding,ok):
        elapsed=time.time()-self.state['active']['started']
        charge=budget_charge(elapsed,0,self.contract['units_per_hour'])
        self.state['spent_hours']+=charge['active_hours'];self.state['spent_units']+=charge['compute_units']
        self.state['active']=None;self.state['last_charge']=charge
        if ok:self.state['completed'].append(binding)
        else:self.state['stopped']=True
        self.append('END' if ok else 'STOP')


def telemetry():
    result=subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=result.stdout.strip().splitlines()
    if len(rows)!=1: raise RuntimeError('one visible GPU required')
    name,temp,clock,power=[v.strip() for v in rows[0].split(',')]
    values=[float(v) for v in (temp,clock,power)]
    finite_nonnegative(*values)
    return dict(gpu=name,temperature_c=values[0],clock_mhz=values[1],power_w=values[2],observed_at=time.time())


def speed_report(arm,history,records,total,steps):
    if len(history)!=2 or len(steps)!=200 or not records: raise ValueError('speed test requires exactly 2 epochs x 100 steps')
    finite_nonnegative(total,*steps,*[v['seconds'] for v in history])
    return dict(version='p4-speed-v1',evidence_class='ENGINEERING_SPEED_ONLY',arm=arm,epochs=2,steps=200,
                epoch_seconds=[v['seconds'] for v in history],step_seconds=steps,step_seconds_mean=float(np.mean(steps)),
                gpu_model=records[0]['gpu'],telemetry=records,total_seconds=total,guard_wait_seconds=0)


class Observer:
    def __init__(self,root,contract,ledger,device='cuda'):
        self.root=Path(root);self.contract=contract;self.ledger=ledger;self.device=device
        self.steps=[];self.records=[];self.pointer=dict(latest=None,previous=None,best=None)
        self.best=-9.;self.step_start=None

    def check(self):
        self.ledger.check()
        r=telemetry();self.records.append(r)
        if r['temperature_c']>=85: raise RuntimeError('STOP: Colab GPU temperature >=85C (no pause/retry)')

    def __call__(self,event,payload):
        import torch
        if event=='step_start':
            self.check();torch.cuda.synchronize();self.step_start=time.perf_counter()
        elif event=='step_end':
            torch.cuda.synchronize();self.steps.append(time.perf_counter()-self.step_start)
        elif event=='epoch':
            epoch=payload['state']['epoch'];name=f'epoch-{epoch:03d}.pt'
            stream=io.BytesIO();torch.save(dict(contract=self.contract,**payload,
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),stream)
            ident=publish_bytes(self.root/name,stream.getvalue())
            entry=dict(file=name,identity=ident,epoch=epoch)
            self.pointer['previous']=self.pointer['latest'];self.pointer['latest']=entry
            if payload['state']['best']>self.best:
                self.best=payload['state']['best'];self.pointer['best']=entry
            atomic_json(self.root/'checkpoint-pointer.json',self.pointer)


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


def environment(lock):
    import importlib.metadata
    import platform
    import torch
    from packaging.requirements import Requirement
    packages={}
    for line in Path(lock).read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        req=Requirement(line);actual=importlib.metadata.version(req.name);packages[req.name]=actual
        expected={'mamba-ssm':'2.3.2.post1','causal-conv1d':'1.7.0'}.get(req.name)
        # Direct wheel versions may carry the CUDA/torch ABI local suffix.
        if (expected and actual.split('+')[0]!=expected) or (not req.url and actual not in req.specifier):
            raise ValueError(f'exact environment mismatch: {req.name}={actual}')
    if (torch.__version__!='2.10.0+cu128' or platform.python_version_tuple()[:2]!=('3','12') or
        not torch._C._GLIBCXX_USE_CXX11_ABI):
        raise ValueError('exact Python/torch runtime mismatch')
    return dict(packages=packages,python=platform.python_version(),cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),lock=digest(lock))


def train_fit(directory,year,arm,seed,cfg,reference,observer):
    """Reuse P3 checked_splits/row admission and existing train_one; P3 collect owns a different ledger."""
    from .seq_runner import SeqData,train_one,predict
    data=SeqData(directory,window=60,normalize='rankgauss');tr,va,ev=checked_splits(data,cfg['val_dates'])
    frame=pd.DataFrame(dict(fold=year,date=pd.to_datetime(np.load(directory/'dates.npy',mmap_mode='r')[ev]),
        code=np.asarray(np.load(directory/'stock_ids.npy',mmap_mode='r')[ev]),label_y5=data.y['y5'][ev]))
    require_same_rows(frame,reference)
    if cfg['val_max_rows'] and len(va)>cfg['val_max_rows']:
        va=np.sort(np.random.default_rng(0).choice(va,cfg['val_max_rows'],replace=False))
    windows=data.windows
    def guarded(rows):
        observer.check();return windows(rows)
    data.windows=guarded
    model,state=train_one(arm,seed,data,tr,va,cfg,'cuda',observer=observer)
    if not np.isfinite(state['best']) or state['best']<=-9: raise ValueError('BEST unavailable; LAST prohibited')
    frame['score']=predict(model,data,ev,'cuda')[:,0];frame['seed']=seed;frame['system']=arm
    if not np.isfinite(frame.score).all(): raise ValueError('nonfinite predictions')
    best=io.BytesIO();import torch
    torch.save(dict(contract=observer.contract,model=model.state_dict(),best_val_ic5=state['best']),best)
    publish_bytes(observer.root/'best.pt',best.getvalue())
    return frame,dict(history=state['history'],best_val_ic5=state['best'],fit_rows=len(tr),val_rows=len(va),eval_rows=len(ev),
        parameters=sum(p.numel() for p in model.parameters()),telemetry=observer.records)


def run(bundle,drive_root,speed,max_hours,max_units,rate,planned_hours,reserve=.25):
    require_cuda()  # Before any directory/output creation.
    validate_budget_plan(max_hours,max_units,rate,planned_hours,reserve)
    bundle=Path(bundle);drive_root=Path(drive_root)
    if (not os.path.ismount('/content/drive') or
        not drive_root.resolve().is_relative_to(Path('/content/drive/MyDrive'))):
        raise ValueError('durable output requires mounted /content/drive/MyDrive')
    run_started=time.perf_counter()
    budget_contract=dict(max_hours=max_hours,max_units=max_units,units_per_hour=rate,reserve=reserve,
                         billing='operator-confirmed conservative CU/hour estimate; not provider measured CU')
    with Ledger(drive_root,budget_contract) as ledger:
        setup_binding=dict(stage='verify_bundle_and_runtime',bundle_manifest=digest(bundle/'bundle-manifest.json'))
        ledger.start(setup_binding,planned_hours)
        try:
            mf=verify(bundle,allow_derived=True)
            # Running imported code must be exactly the source included in this bundle.
            source_root=Path(__file__).resolve().parents[2]
            for name,ident in mf['files'].items():
                if name.startswith('source/') and digest(source_root/name.removeprefix('source/'))!=ident:
                    raise ValueError('executed source differs from bundle')
            env=environment(bundle/'requirements.lock.txt')
            contract=json.loads((bundle/'source/tasks/p4a-colab-contract-20261121.json').read_text())
            validate_contract(contract,mf['metadata']['years'],speed)
            if digest(bundle/'requirements.lock.txt')['sha256']!=contract['environment_lock']['sha256']:
                raise ValueError('P4 exact lock hash mismatch')
            cfg=deepcopy(FROZEN_CFG)
            if speed:cfg.update(epochs=2,steps_per_epoch=100,patience=3)
            years=mf['metadata']['years'][:1] if speed else mf['metadata']['years']
            # Speed only seed17; full grid seed17/29. Every arm uses same selected fold.
            seeds=[17] if speed else [17,29]
            ref=pd.read_parquet(bundle/'reference/rank_average.parquet');ref=ref[ref.system.eq('c0plus')]
            if any(digest(bundle/'reference/rank_average.parquet')[k]!=json.loads((bundle/'reference/manifest.json').read_text())['outputs']['rank_average.parquet'][k] for k in ('bytes','sha256')):
                raise ValueError('P1 reference manifest mismatch')
            binding=dict(bundle=digest(bundle/'bundle-manifest.json'),environment=env,contract=contract,cfg=cfg,
                         years=years,seeds=seeds,arms=list(ARMS),speed_test=speed)
            run_hash=hashed(binding)
            ledger.check()
            ledger.end(setup_binding,True)
        except BaseException:
            if ledger.state['active']: ledger.end(setup_binding,False)
            raise
        out=drive_root/f'run-{run_hash}'
        if out.exists():
            if json.loads((out/'run_identity.json').read_text())!=binding:raise ValueError('run identity mismatch')
        else:
            out.mkdir();write_json(out/'run_identity.json',binding)
        reports=[]
        # Check the complete remaining grid plus reserve before starting another fit.
        completed_here=set()
        for b in ledger.state['completed']:
            if b.get('run_hash')==run_hash and 'arm' in b:
                completed_fit=out/f"{b['year']}-{b['arm']}-{b['seed']}"
                if resume_fit(completed_fit,b) is not None: completed_here.add(hashed(b))
        remaining=len(years)*len(ARMS)*len(seeds)-len(completed_here)
        check_plan_budget(budget_contract,ledger.state['spent_hours'],ledger.state['spent_units'],remaining+len(years),planned_hours)
        for year in years:
            directory=bundle/f'folds/fold_{year}'
            # verify() refuses derived outputs; all original bytes were verified before cache construction.
            prep_binding=dict(run_hash=run_hash,stage='rankgauss_verify_or_build',year=year)
            ledger.start(prep_binding,planned_hours)
            try:
                cache_id=rankgauss(directory)
                ledger.check()
                ledger.end(prep_binding,True)
            except BaseException:
                if ledger.state['active']: ledger.end(prep_binding,False)
                raise
            for arm in ARMS:
                for seed in seeds:
                    fit_contract=dict(run_hash=run_hash,year=year,arm=arm,seed=seed,rankgauss=cache_id)
                    fit=out/f'{year}-{arm}-{seed}'
                    cached=resume_fit(fit,fit_contract)
                    if cached:
                        if fit_contract not in ledger.state['completed']:raise ValueError('fit missing from ledger')
                        reports.append(cached);continue
                    ledger.start(fit_contract,planned_hours)
                    attempt=fit/('attempt-'+uuid.uuid4().hex);attempt.mkdir(parents=True)
                    observer=Observer(attempt,fit_contract,ledger);started=time.perf_counter()
                    try:
                        frame,job=train_fit(directory,year,arm,seed,cfg,ref[ref.fold.eq(year)],observer)
                        buffer=io.BytesIO();frame.to_parquet(buffer,index=False);publish_bytes(attempt/'scores.parquet',buffer.getvalue())
                        if speed:
                            report=speed_report(arm,job['history'],observer.records,time.perf_counter()-started,observer.steps)
                            write_json(attempt/'speed.json',report)
                        job.update(wall_seconds=time.perf_counter()-started,guard_wait_seconds=0)
                        write_json(attempt/'metrics.json',job)
                        # Record all logs/checkpoints/predictions before ending the charged attempt.
                        files=[p.relative_to(fit).as_posix() for p in attempt.iterdir() if p.is_file()]
                        ledger.end(fit_contract,True)
                        complete_fit(fit,fit_contract,job,files)
                        reports.append(resume_fit(fit,fit_contract))
                    except BaseException:
                        if ledger.state['active']:ledger.end(fit_contract,False)
                        raise
        summary=out/'manifest.json'
        payload=dict(version='p4-run-v1',binding=binding,jobs=reports,budget=ledger.state,
                     wall_seconds=time.perf_counter()-run_started,guard_wait_seconds=0,
                     billing_coverage='runner verification/preprocessing/fits; excludes upload/pip/operator idle',
                     evidence_class='ENGINEERING_SPEED_ONLY' if speed else 'E2 RESEARCH_ONLY / NOT_ACCEPTED',research_conclusion=None)
        if not summary.exists():write_json(summary,payload)
        return payload


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--drive-root',type=Path,required=True)
    p.add_argument('--speed-test',action='store_true');p.add_argument('--max-hours',type=float,required=True)
    p.add_argument('--max-units',type=float,required=True);p.add_argument('--units-per-hour',type=float,required=True)
    p.add_argument('--planned-hours-per-fit',type=float,required=True);p.add_argument('--reserve',type=float,default=.25)
    a=p.parse_args(argv)
    result=run(a.bundle,a.drive_root,a.speed_test,a.max_hours,a.max_units,a.units_per_hour,a.planned_hours_per_fit,a.reserve)
    print(json.dumps(dict(version=result['version'],jobs=len(result['jobs']),evidence_class=result['evidence_class'])))

if __name__=='__main__':main()
