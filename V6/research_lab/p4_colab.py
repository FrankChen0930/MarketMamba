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
import threading
import tempfile
from datetime import datetime, timezone
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from .p4_bundle import digest, verify, MATRIX_FILES, prepare_source
from .rolling_seq_scores import checked_splits, require_same_rows, FROZEN_CFG, write_json, atomic_json

ARMS=('mamba2','crossstock','gru')


def hashed(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def budget_root(root, speed=False):
    """CLI takes the formal ledger root; timing owns a sibling, across deliveries."""
    root=Path(root)
    return root.with_name(root.name+'_speedtest') if speed else root


class Progress:
    """Best-effort observation only. No model, RNG, checkpoint or ledger authority.

    A wall-clock heartbeat keeps Drive visible even during validation/slow steps.
    Each file is atomically replaced; the pair is not a transaction. JSON is the
    summary authority and both carry the same observed_at generation.
    """
    def __init__(self, root, speed=False, interval=60., clock=time.monotonic):
        self.root=Path(root);self.speed=speed;self.interval=interval;self.clock=clock
        self.lock=threading.RLock();self.stop=threading.Event();self.durations=[]
        self.fit_started=None;self.completed=0;self.last_fit_elapsed=0.;self.step_origin=0
        self.data=dict(status='準備中',queue=None,fit_index=0,total_fits=0,epoch=0,step=0,
                       loss=None,temperature_c=None,clock_mhz=None,last_checkpoint_at=None)

    def __enter__(self):
        self.write()
        self.thread=threading.Thread(target=self._heartbeat,name='p4-progress',daemon=True)
        self.thread.start();return self

    def _heartbeat(self):
        while not self.stop.wait(self.interval):self.write()

    def __exit__(self,kind,error,tb):
        self.stop.set();self.thread.join()
        self.update(status='已中斷／失敗' if kind else
                    ('已完成（略過）' if self.data['status']=='已完成（略過）' else '已完成'))
        self.write()

    def update(self, **values):
        # Whitelist prevents history/weights/performance leaking through payloads.
        allowed={'status','queue','fit_index','total_fits','arm','year','seed','window','lr',
                 'epoch','step','epochs','steps_per_epoch','loss','temperature_c','clock_mhz','last_checkpoint_at',
                 'run_hash','bundle_identity','source_commit'}
        if not self.speed:allowed.add('val_ic5')
        with self.lock:self.data.update({k:v for k,v in values.items() if k in allowed})

    def start_fit(self,job,index,total,cfg,skipped=False):
        with self.lock:
            self.fit_started=None if skipped else self.clock();self.last_fit_elapsed=0.;self.step_origin=0
            self.data.pop('val_ic5',None)
            self.update(**job,fit_index=index,total_fits=total,epoch=0,step=0,loss=None,
                        epochs=cfg['epochs'],steps_per_epoch=cfg['steps_per_epoch'],
                        last_checkpoint_at=None,status='已完成（略過）' if skipped else '訓練中')
        self.emit();self.write()

    def resume_from(self,epoch,step):
        """Rebase observation after checkpoint validation; historical steps earn no throughput."""
        with self.lock:
            self.step_origin=step
            self.update(epoch=epoch,step=step)
        self.emit();self.write()

    def finish_fit(self, skipped=False):
        with self.lock:
            if not skipped and self.fit_started is not None:
                self.last_fit_elapsed=max(0.,self.clock()-self.fit_started)
                if self.last_fit_elapsed>=1.:
                    self.durations.append(self.last_fit_elapsed)
            self.completed+=1;self.fit_started=None
            self.update(status='已完成（略過）' if skipped else
                        ('已完成' if self.completed==self.data['total_fits'] else 'fit 已驗證完成'))
        self.emit();self.write()

    def snapshot(self):
        with self.lock:
            result=dict(self.data)
            elapsed=max(0.,self.clock()-self.fit_started) if self.fit_started is not None else self.last_fit_elapsed
            step=result.get('step',0);total=result.get('epochs',0)*result.get('steps_per_epoch',0)
            measured_steps=step-self.step_origin
            fit_eta=None
            if self.fit_started is None and self.completed:
                fit_eta=0.
            elif measured_steps>=10 and elapsed>=1. and total>=step:
                fit_eta=elapsed*(total-step)/measured_steps
            remaining=max(0,result['total_fits']-self.completed)
            active=self.fit_started is not None
            queue_eta=None
            if result['total_fits'] and remaining==0:
                queue_eta=0.
            elif self.durations:
                mean=sum(self.durations)/len(self.durations)
                # Completed durations are frozen at finish, never recomputed using
                # the next fit's clock. Cached skips contribute no timing samples.
                current=(fit_eta if fit_eta is not None else max(0.,mean-elapsed)) if active else 0.
                queue_eta=current+mean*max(0,remaining-int(active))
            elif active and remaining==1:
                queue_eta=fit_eta
            result.update(observed_at=datetime.now(timezone.utc).isoformat(),
                          mode='SPEED_TEST' if self.speed else 'FORMAL',completed_fits=self.completed,
                          fit_elapsed_seconds=elapsed,fit_eta_seconds=fit_eta,queue_eta_seconds=queue_eta)
            if result.get('queue')=='Q5':result['fit_eta_seconds']=result['queue_eta_seconds']=None
            return result

    @staticmethod
    def render(data):
        def value(key):
            v=data.get(key)
            return '預估中' if v is None and 'eta' in key else ('未知／尚無' if v is None else str(v))
        labels=[('observed_at','更新時間'),('status','狀態'),('mode','模式'),('queue','佇列階段'),
                ('fit_index','目前 fit'),('total_fits','總 fits'),('completed_fits','已完成 fits'),
                ('arm','臂'),('year','折'),('seed','種子'),('window','窗'),('lr','學習率'),
                ('epoch','epoch'),('step','累計 step'),('loss','loss'),
                ('fit_elapsed_seconds','本 fit 已用秒數'),('fit_eta_seconds','本 fit 預估剩餘秒數（最大 epochs 外推）'),
                ('queue_eta_seconds','佇列預估剩餘秒數（完成 fit 均值；不含 Q5）'),
                ('temperature_c','GPU 溫度 °C'),('clock_mhz','GPU 時脈 MHz'),('last_checkpoint_at','最近檢查點時間')]
        if data.get('mode')=='FORMAL':labels.insert(15,('val_ic5','驗證 IC5'))
        return ' | '.join(label+': '+value(key) for key,label in labels)

    def emit(self):print(self.render(self.snapshot()),flush=True)

    def write(self):
        try:
            data=self.snapshot();self.root.mkdir(parents=True,exist_ok=True)
            for name,content in [('progress.json',json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)),
                                 ('PROGRESS.txt',self.render(data)+'\n')]:
                temporary=None
                try:
                    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=self.root,suffix='.tmp',delete=False) as f:
                        temporary=Path(f.name);f.write(content);f.flush();os.fsync(f.fileno())
                    os.replace(temporary,self.root/name)
                finally:
                    if temporary is not None:temporary.unlink(missing_ok=True)
        except Exception as error:
            print(f'進度寫入失敗：步驟=Drive 進度快照；期望=原子更新；實際={error}；下一步=檢查 Drive 掛載與空間，訓練繼續',flush=True)


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


def effective_recipe(contract):
    """Bind each formal fit to its frozen queue entry and effective recipe."""
    frozen=json.loads((Path(__file__).resolve().parents[2]/'tasks/p4a-colab-contract-20261121.json').read_text())
    keys=('queue','year','arm','seed','window','lr')
    job={k:contract.get(k) for k in keys}
    if job not in fit_queue(frozen):raise ValueError('fit description not in frozen queue')
    recipe=dict(frozen['recipe'],window=job['window'],lr=job['lr'])
    if contract.get('recipe')!=recipe:raise ValueError('fit frozen recipe mismatch')
    return recipe


def validate_training_state(state,recipe):
    history=state.get('history',[]);epoch=state.get('epoch')
    if not history or type(epoch)!=int or epoch!=len(history) or epoch>recipe['epochs']:
        raise ValueError('invalid training history length')
    best=-9.;bad=0;best_epoch=None
    for i,row in enumerate(history,1):
        score=row.get('val_ic5',np.nan)
        if row.get('epoch')!=i or not np.isfinite(score):raise ValueError('invalid training history epoch/IC')
        if bad>=recipe['patience']:raise ValueError('training history continues after early stop')
        if score>best:best=score;bad=0;best_epoch=i
        else:bad+=1
    if state.get('best')!=best or state.get('bad')!=bad or best_epoch is None:
        raise ValueError('training BEST/bad history mismatch')
    return best_epoch


def validate_fit_outputs(directory, contract, outputs, *, rehearsal=False):
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
    recipe=contract['recipe'] if rehearsal else effective_recipe(contract)
    if rehearsal and contract.get('evidence_class')!='REHEARSAL / 不是研究結果':raise ValueError('rehearsal identity required')
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
    latest=checkpoints['latest']['state']
    best_epoch=validate_training_state(latest,recipe)
    if latest['epoch']!=recipe['epochs'] and latest['bad']<recipe['patience']:
        raise ValueError('training termination not reached')
    if latest['history']!=metrics['history'] or latest['best']!=metrics['best_val_ic5']:
        raise ValueError('metrics history/BEST mismatch')
    if pointer['best']['epoch']!=best_epoch:
        raise ValueError('BEST epoch does not match history')
    for payload in checkpoints.values():
        state=payload['state'];validate_training_state(state,recipe)
        if state['history']!=latest['history'][:state['epoch']]:
            raise ValueError('checkpoint history prefix mismatch')
    for column,value in dict(fold=contract['year'],system=contract['arm'],seed=contract['seed'],
                             window=recipe['window'],lr=recipe['lr']).items():
        if column not in frame or not frame[column].eq(value).all():
            raise ValueError('prediction fit setting mismatch: '+column)
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
    def __init__(self,root,contract,ledger,device='cuda',speed=False,progress=None):
        self.root=Path(root);self.contract=contract;self.ledger=ledger;self.device=device
        self.speed=speed;self.progress=progress
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
        if self.progress:
            epoch=payloads['latest']['state']['epoch']
            self.progress.update(last_checkpoint_at=datetime.fromtimestamp(pointer.stat().st_mtime,timezone.utc).isoformat())
            self.progress.resume_from(epoch,epoch*self.progress.data['steps_per_epoch'])
        return payloads['latest']


    def check(self):
        self.ledger.check()
        r=telemetry();self.records.append(r)
        if self.progress:self.progress.update(temperature_c=r['temperature_c'],clock_mhz=r['clock_mhz'])
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
        elif event=='progress':
            if self.progress:self.progress.update(**payload);self.progress.emit()
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
            if self.progress:self.progress.update(last_checkpoint_at=datetime.now(timezone.utc).isoformat())


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


def long_window_data(directory,year,window,val_dates):
    """Actionable checks before SeqData allocation/split indexing; W60 bypasses."""
    from .seq_runner import SeqData
    sessions=len(np.load(directory/'end_indices.npy',mmap_mode='r'))
    def insufficient(kind,actual,needed):
        raise ValueError(f'fold={year} W={window} {kind}: 需要={needed} 實際={actual}；'
                         '下一步=確認 fold 與資料完整性，提供足夠歷史 sessions／eligible 訓練日後重跑；不得縮窗或放寬契約')
    if sessions<window:insufficient('sessions',sessions,window)
    data=SeqData(directory,window=window,normalize='rankgauss')
    eligible=len(np.unique(data.date_idx[(data.splits==1)&data.eligible]))
    if eligible<val_dates+20:insufficient('eligible training dates',eligible,val_dates+20)
    return data


REHEARSAL_EVIDENCE='REHEARSAL / 不是研究結果'
REHEARSAL_STAGES=(('Q3 W120 mamba2','Q3',120,'mamba2',1),
                  ('Q3 W250 mamba2','Q3',250,'mamba2',1),
                  ('Q5（兩個 prerequisite fits）','Q5',None,None,2))


def validate_rehearsal_schema(value,schema,path='rehearsal'):
    """One strict recursive validator for stage publications and cache metadata.

    Objects are closed, numeric types exclude bool, and conversion/recursion
    errors become invalid evidence rather than escaping the rehearsal gate.
    """
    try:
        kind,*args=schema
        valid=False
        if kind=='object':
            valid=type(value) is dict and set(value)==set(args[0])
            if valid:
                for key,child in args[0].items():validate_rehearsal_schema(value[key],child,f'{path}.{key}')
        elif kind=='list':
            child,minimum,maximum=args
            valid=type(value) is list and len(value)>=minimum and (maximum is None or len(value)<=maximum)
            if valid:
                for i,item in enumerate(value):validate_rehearsal_schema(item,child,f'{path}[{i}]')
        elif kind=='tuple':
            valid=type(value) is list and len(value)==len(args[0])
            if valid:
                for i,(item,child) in enumerate(zip(value,args[0])):validate_rehearsal_schema(item,child,f'{path}[{i}]')
        elif kind in ('int','number'):
            low,high=args
            valid=(type(value) is int if kind=='int' else type(value) in (int,float))
            valid=valid and math.isfinite(value) and (low is None or value>=low) and (high is None or value<=high)
        elif kind=='exact':valid=type(value) is type(args[0]) and hashed(value)==hashed(args[0])
        elif kind=='bool':valid=type(value) is bool
        elif kind=='date':
            # numpy day strings and ISO timestamps from permutation_summary.
            from datetime import datetime
            valid=type(value) is str and len(value)>=10
            if valid:datetime.fromisoformat(value.replace('Z','+00:00'))
        elif kind=='string':valid=type(value) is str and bool(value)
        elif kind=='sha256':valid=type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        if not valid:raise ValueError(f'{path}: invalid {kind}')
    except (OverflowError,TypeError,RecursionError) as error:
        raise ValueError(f'{path}: invalid schema value ({type(error).__name__})') from error


def rehearsal_schema(kind,expected=None):
    """Schemas follow actual publications; R=10 is frozen by permutation_summary."""
    obj=lambda **fields:('object',fields)
    seq=lambda child,lo=0,hi=None:('list',child,lo,hi)
    exact=lambda value:('exact',value)
    number=lambda lo=None,hi=None:('number',lo,hi)
    integer=lambda lo=0,hi=None:('int',lo,hi)
    identity=obj(bytes=integer(1),sha256=('sha256',))
    if kind=='q5':
        return obj(R=exact(10),label_support=obj(input_rows=integer(1),used_rows=integer(1),excluded_nan_rows=integer(),excluded_dates=seq(('date',))),
            daily=seq(obj(permutation=integer(0,9),date=('date',),score_spearman=number(-1,1),ic=number(-1,1),ic_change=number(-2,2),top50_overlap=number(0,1)),1),
            permutation_mean_ic=seq(number(-1,1),10,10),ic_std=number(0),mamba_ic_gap=number(0,2),order_sensitive=('bool',),
            rule=exact('IC std >= absolute crossstock minus mamba2 IC gap'),evidence_class=exact('DESCRIPTIVE_ONLY'))
    if kind=='stage_identity':
        binding,queue,window,arm=expected
        return obj(binding=exact(binding),queue=exact(queue),window=exact(window),arm=exact(arm),steps=exact(1),result=identity)
    if kind=='marker':
        stages=[obj(stage=exact(name),seconds=number(0),identity=rehearsal_schema('stage_identity',(expected,queue,window,arm)),
                    evidence_class=exact(REHEARSAL_EVIDENCE),fit_count=exact(count))
                for name,queue,window,arm,count in REHEARSAL_STAGES]
        return obj(identity=exact(expected),passed=exact(True),stages=('tuple',stages),evidence_class=exact(REHEARSAL_EVIDENCE),digest=('sha256',))
    if kind=='result':
        request=expected
        fields=dict(evidence_class=exact(REHEARSAL_EVIDENCE),artifact_class=exact('TEMPORARY'),output=('string',),
            fits=seq(('string',),1,2),inputs=obj(data_realpath=('string',),matrix_manifest=identity,reference=identity),
            implementation=obj(**{n:identity for n in ('p4_colab.py','p4_eval.py','seq_runner.py')}),
            environment=obj(**{n:('string',) for n in ('python','torch','numpy','device','mixer')}),
            cuda_max_memory_allocated=integer(1),request=exact(request),context=exact(request['binding']['identity']),
            **{key:exact(request[key]) for key in ('queue','window','arm','steps')})
        if request['queue']=='Q5':fields['diagnostic']=obj(validation=rehearsal_schema('q5'),evaluation=rehearsal_schema('q5'))
        return ('object',fields)
    if kind=='identity':return identity
    raise ValueError('unknown rehearsal schema: '+kind)


def validate_rehearsal_marker(saved,binding):
    """Validate with the stage schema engine; corrupted caches always rerun."""
    try:
        validate_rehearsal_schema(saved,rehearsal_schema('marker',binding))
        return saved['digest']==hashed({k:v for k,v in saved.items() if k!='digest'})
    except (ValueError,OverflowError,TypeError,RecursionError,KeyError):return False


def validate_rehearsal_result(path,request,common):
    """Gate-owned validation of result, requested inputs, and completed publications."""
    try:
        if Path(path).is_symlink():raise ValueError('result metadata symlink prohibited')
        result=json.loads(Path(path).read_text())
        validate_rehearsal_schema(result,rehearsal_schema('result',request))
        root=Path(result['output'])
        if not root.is_absolute() or root.is_symlink() or root.parent!=Path('/tmp') or not root.name.startswith('p4-REHEARSAL-'):raise ValueError('output must be retained /tmp rehearsal directory')
        if (root/'REHEARSAL.json').is_symlink():raise ValueError('publication metadata symlink prohibited')
        publication=json.loads((root/'REHEARSAL.json').read_text())
        validate_rehearsal_schema(publication,rehearsal_schema('result',request))
        if hashed(publication)!=hashed(result):raise ValueError('result publication mismatch')
        data=Path(common[common.index('--data-dir')+1]).resolve(strict=True)
        bundle=Path(common[common.index('--bundle')+1])
        reference=bundle/'reference/keys.parquet'
        if not reference.exists():reference=bundle/'reference/rank_average.parquet'
        frozen=json.loads((Path(__file__).resolve().parents[2]/'tasks/p4a-colab-contract-20261121.json').read_text())
        queue=request['queue']
        jobs=[j for j in fit_queue(frozen) if j['queue']==('Q1' if queue=='Q5' else queue)]
        jobs=([next(j for j in jobs if j['arm']==a) for a in ('crossstock','mamba2')] if queue=='Q5' else
              [j for j in jobs if j['window']==request['window'] and j['arm']==request['arm']][:1])
        names=[f"{j['arm']}-w{j['window']}" for j in jobs]
        if result['fits']!=names or len(names)!=(2 if queue=='Q5' else 1):raise ValueError('fit count/selection mismatch')
        source=data/f"folds/fold_{jobs[0]['year']}" if (data/'folds').is_dir() else data
        if result['inputs']!=dict(data_realpath=str(source),matrix_manifest=digest(source/'manifest.json'),reference=digest(reference)):raise ValueError('input identity mismatch')
        implementation={name:digest(Path(__file__).with_name(name)) for name in ('p4_colab.py','p4_eval.py','seq_runner.py')}
        if result['implementation']!=implementation:raise ValueError('implementation identity mismatch')
        environment=result['environment']
        if environment['device']!='cuda' or environment['mixer']!='official':raise ValueError('official CUDA rehearsal required')
        observed=result['context']['environment']
        if not isinstance(observed,dict) or environment['python']!=observed['python'] or any(environment[n]!=observed['packages'][n] for n in ('torch','numpy')):raise ValueError('environment identity mismatch')
        for name,job in zip(names,jobs):
            fit=root/name
            if fit.is_symlink() or (fit/'REHEARSAL-complete.json').is_symlink():raise ValueError('complete metadata symlink prohibited')
            complete=json.loads((fit/'REHEARSAL-complete.json').read_text())
            contract=dict(**job,recipe=dict(frozen['recipe'],window=job['window'],lr=job['lr'],epochs=1,steps_per_epoch=1),speed_test=False,evidence_class=REHEARSAL_EVIDENCE,rehearsal_binding=hashed(request))
            validate_rehearsal_schema(complete,('object',dict(contract=('exact',contract),outputs=('object',{n:rehearsal_schema('identity') for n in complete['outputs']}),evidence_class=('exact',REHEARSAL_EVIDENCE))))
            validate_fit_outputs(fit,contract,complete['outputs'],rehearsal=True)
        if queue=='Q5':
            diagnostic=result['diagnostic']
            for split in diagnostic.values():
                support=split['label_support']
                if support['input_rows']!=support['used_rows']+support['excluded_nan_rows']:raise ValueError('Q5 mature support missing')
                daily=split['daily']
                if {row['permutation'] for row in daily}!=set(range(split['R'])):raise ValueError('Q5 permutations incomplete')
                keys=[(row['permutation'],row['date']) for row in daily]
                dates={row['date'] for row in daily}
                if len(keys)!=len(set(keys)) or len(keys)!=split['R']*len(dates):raise ValueError('Q5 daily permutation/date support mismatch')
        return dict(binding=request['binding'],queue=queue,window=request['window'],arm=request['arm'],steps=1,result=digest(path))
    except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,OverflowError,RecursionError) as error:
        raise ValueError(f'REHEARSAL 排練結果／產物驗證失敗: {error}') from error


def launch_with_rehearsal(common,env,progress,identity,stream,*,enabled=True):
    """Launch formal only after all three official CLI rehearsals pass.

    The marker is observational engineering evidence, never a completed fit.
    Caller supplies identities after bundle verification and SSD preparation.
    """
    progress=Path(progress);marker=progress/'rehearsal_passed.json'
    binding=dict(identity=identity,command=common,policy='Q3-W120-mamba2/Q3-W250-mamba2/Q5-1step-v2')
    saved={}
    try:
        if marker.is_symlink():raise ValueError('marker metadata symlink prohibited')
        saved=json.loads(marker.read_text())
    except (OSError,ValueError,OverflowError,TypeError,RecursionError):pass
    if not isinstance(saved,dict):saved={}
    if enabled and validate_rehearsal_marker(saved,binding):
        print('排練通過（不是研究結果）：同一 run/source/data/environment，略過；各步耗時：',saved.get('stages'),flush=True)
    elif enabled:
        print('排練標記無效或缺少；期望=完整 schema／有限數值／非符號連結；實際=無可採用標記；下一步=移除舊標記並重新執行三階段排練',flush=True)
        # A stale success must not survive a failed recheck.
        marker.unlink(missing_ok=True)
        base=common[:common.index('--drive-root')]+common[common.index('--drive-root')+2:]
        stages=[]
        for stage,queue,window,arm,count in REHEARSAL_STAGES:
            flags=['--queue',queue]+([] if window is None else ['--window',str(window),'--arm',arm])
            command=base+['--rehearse','--steps','1']+flags
            scratch=Path(tempfile.mkdtemp(prefix='p4-REHEARSAL-gate-',dir='/tmp'))
            request=dict(binding=binding,queue=queue,window=window,arm=arm,steps=1,nonce=uuid.uuid4().hex)
            atomic_json(scratch/'request.json',request)
            stage_env=dict(env,P4_REHEARSAL_REQUEST=str(scratch/'request.json'),P4_REHEARSAL_RESULT=str(scratch/'result.json'))
            started=time.perf_counter()
            try:
                stream(command,stage_env)
                verified=validate_rehearsal_result(scratch/'result.json',request,common)
            except BaseException as error:
                print(f'排練失敗階段={stage}；暫存產物位置=/tmp/p4-REHEARSAL-*（本步確切路徑見上方）；'
                      f'期望=回傳碼 0 且完整排練 JSON／產物／identity 驗證；實際={error!r}；下一步=將本格完整輸出貼給 Claude，修正後重跑；正式 runner 未啟動',flush=True)
                raise
            stages.append(dict(stage=stage,seconds=time.perf_counter()-started,identity=verified,evidence_class=REHEARSAL_EVIDENCE,fit_count=count))
        progress.mkdir(parents=True,exist_ok=True)
        record=dict(identity=binding,passed=True,stages=stages,evidence_class=REHEARSAL_EVIDENCE)
        record['digest']=hashed(record)
        atomic_json(marker,record)
        print('排練通過（不是研究結果）；各步耗時：',stages,flush=True)
    else:print('REHEARSE_FIRST=False：本次未執行自動排練',flush=True)
    stream(common,env)


def train_fit(directory,year,arm,seed,cfg,reference,observer,*,mixer_factory=None):
    """Reuse P3 checked_splits/row admission and existing train_one; P3 collect owns a different ledger."""
    from .seq_runner import SeqData,train_one,predict
    data=(SeqData(directory,window=60,normalize='rankgauss') if cfg.get('window',60)==60
          else long_window_data(directory,year,cfg.get('window',60),cfg['val_dates']))
    tr,va,ev=checked_splits(data,cfg['val_dates'])
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
    extra={} if mixer_factory is None else dict(mixer_factory=mixer_factory)
    device=getattr(observer,'device','cuda')
    model,state=train_one(arm,seed,data,tr,va,cfg,device,**extra,observer=observer,resume_state=restored,
                          log=lambda *_: None)
    if observer.speed:
        return None,dict(history=[dict(seconds=h['seconds']) for h in state['history']])
    if not np.isfinite(state['best']) or state['best']<=-9: raise ValueError('BEST unavailable; LAST prohibited')
    frame['score']=predict(model,data,ev,device)[:,0];frame['seed']=seed;frame['system']=arm
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


def run(bundle,drive_root,speed=False,max_hours=None,max_units=None,rate=None,planned_hours=0,reserve=None,data_dir=None,allow_new_run=False):
    """P4-only record accounting; old cap arguments are accepted but never constrain runs."""
    require_cuda()
    bundle=Path(bundle);drive_root=budget_root(drive_root,speed)
    if (not os.path.ismount('/content/drive') or
        not drive_root.resolve().is_relative_to(Path('/content/drive/MyDrive'))):
        raise ValueError('durable output requires mounted /content/drive/MyDrive')
    run_started=time.perf_counter()
    budget_contract=dict(policy='record_only',units_per_hour=0,thermal_stop_c=85)
    with Ledger(drive_root,budget_contract) as ledger, Progress(drive_root.parent/'progress'/('speedtest' if speed else 'formal'),speed=speed) as progress:
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
        from .p4_source_compat import resolve
        binding,upgrade=resolve(bundle,mf,binding,drive_root,budget_contract,allow_new=allow_new_run)
        run_hash=hashed(binding)
        out=drive_root/f'run-{run_hash}'
        if out.exists():
            if json.loads((out/'run_identity.json').read_text())!=binding:raise ValueError('run identity mismatch')
        else:out.mkdir();write_json(out/'run_identity.json',binding)
        if upgrade:
            # Preserve run_identity and all historical fit/checkpoint bytes.
            audit=out/f'source-upgrade-{uuid.uuid4().hex}.json'
            write_json(audit,upgrade)
            ledger.state['source_upgrade']=dict(audit=str(audit),identity=digest(audit),
                **{k:upgrade[k] for k in ('from_sources_hash','to_sources_hash','declaration','observed_at','environment')})
            ledger.append('SOURCE_UPGRADE')
        progress.update(run_hash=run_hash,bundle_identity=binding['bundle'],source_commit=mf['metadata'].get('source_commit'),total_fits=len(queue))
        reports=[];caches={};fit_paths={}
        for fit_index,job in enumerate(queue,1):
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
            fit_contract=dict(run_hash=run_hash,**job,recipe=cfg,rankgauss=caches[year],speed_test=speed)
            fit=out/f"{job['queue']}-{year}-{arm}-{seed}-w{job['window']}-lr{job['lr']}"
            fit_paths[(year,arm,seed,job['window'],job['lr'])]=fit
            cached=resume_fit(fit,fit_contract)
            if cached:
                progress.start_fit(job,fit_index,len(queue),cfg,skipped=True)
                reports.append(cached);progress.finish_fit(skipped=True);continue
            ledger.start(fit_contract,0)
            # Stable directory permits retry from the last pointer. Orphan epoch files remain forensic.
            attempt=fit/'attempt-active';attempt.mkdir(parents=True,exist_ok=True)
            progress.start_fit(job,fit_index,len(queue),cfg)
            observer=Observer(attempt,fit_contract,ledger,speed=speed,progress=progress);started=time.perf_counter()
            try:
                frame,metrics=train_fit(directory,year,arm,seed,cfg,ref[ref.fold.eq(year)],observer)
                if speed:
                    report=speed_report(arm,metrics['history'],observer.records,time.perf_counter()-started,observer.steps,environment=env)
                    atomic_json(attempt/'speed.json',report);metrics=report
                else:
                    # A prior crash after prediction publication must validate identical bytes.
                    publish_predictions(attempt,frame,ref[ref.fold.eq(year)],job['window'])
                    metrics.update(wall_seconds=time.perf_counter()-started,guard_wait_seconds=0)
                    atomic_json(attempt/'metrics.json',metrics)
                files=[p.relative_to(fit).as_posix() for p in attempt.iterdir() if p.is_file()]
                complete_fit(fit,fit_contract,metrics,files)
                ledger.end(fit_contract,True);reports.append(resume_fit(fit,fit_contract));progress.finish_fit()
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
                progress.update(queue='Q5',status='排列診斷中',year=year,epoch=0,step=0,loss=None,arm='crossstock');progress.emit();progress.write()
                ledger.start(dict(stage='Q5',**identity),0)
                try:
                    directory=(Path(data_dir) if external else bundle)/f'folds/fold_{year}'
                    result=permutation_fit(directory,year,contract['recipe'],fit_paths,ledger,progress)
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


def permutation_fit(directory,year,cfg,fit_paths,ledger,progress=None,*,device='cuda',mixer_factory=None,rehearsal=False):
    import torch
    from .seq_runner import SeqData,predict
    from .seq_models import SeqRanker
    from .cross_stock import CrossStockRanker,predict_cross
    from .p4_eval import permutation_summary
    data=SeqData(directory,window=60);_,va,ev=checked_splits(data,cfg['val_dates'])
    extra={} if mixer_factory is None else dict(mixer_factory=mixer_factory)
    cross=CrossStockRanker(n_features=data.X.shape[1],d_model=cfg['d_model'],n_layers=cfg['n_layers'],
                          n_cross=cfg['n_cross'],mixer_kw=cfg['mixer_kw'],dropout=cfg['dropout'],**extra).to(device)
    mamba=SeqRanker('mamba2',data.X.shape[1],cfg['d_model'],cfg['n_layers'],mixer_kw=cfg['mixer_kw'],**extra).to(device)
    for arm,model in [('crossstock',cross),('mamba2',mamba)]:
        fit=fit_paths[(year,arm,17,60,3e-4)]
        complete=json.loads((fit/('REHEARSAL-complete.json' if rehearsal else 'complete.json')).read_text())
        if rehearsal:validate_fit_outputs(fit,complete['contract'],complete['outputs'],rehearsal=True)
        else:resume_fit(fit,complete['contract'])
        model.load_state_dict(torch.load(fit/'attempt-active/best.pt',map_location=device,weights_only=False)['model'])
    original=data.windows
    def guarded(rows):
        ledger.check();r=telemetry();ledger.observe(r)
        if progress:progress.update(temperature_c=r['temperature_c'],clock_mhz=r['clock_mhz'])
        if r['temperature_c']>=85:raise RuntimeError('STOP: Colab GPU temperature >=85C (no pause/retry)')
        return original(rows)
    data.windows=guarded
    result={}
    for split,rows in [('validation',va),('evaluation',ev)]:
        if progress:progress.update(status=f'Q5 {year} {split} 推論中');progress.emit()
        base=predict_cross(cross,data,rows,device,n_perm=2,seed=0)[:,0]
        temporal=predict(mamba,data,rows,device)[:,0]
        permutations=[predict_cross(cross,data,rows,device,n_perm=1,seed=100+r)[:,0] for r in range(10)]
        result[split]=permutation_summary(np.load(directory/'dates.npy',mmap_mode='r')[rows],np.asarray(np.load(directory/'stock_ids.npy',mmap_mode='r')[rows]),
                                          data.y['y5'][rows],base,temporal,permutations)
    return result


def publish_predictions(attempt,frame,reference,window):
    """Shared formal/rehearsal publication, with full model support required."""
    if window!=60:
        from .p4_eval import common_rows
        aligned,_=common_rows({'model':frame,'reference':reference})
        if len(aligned['model'])!=len(frame):raise ValueError('long-window rows outside C0+ reference')
        reference=aligned['reference']
    require_same_rows(frame,reference)
    for name,table in (('scores.parquet',frame),('reference.parquet',reference)):
        buffer=io.BytesIO();table.to_parquet(buffer,index=False)
        publish_or_validate(Path(attempt)/name,buffer.getvalue())


def rehearse(data_dir,reference,*,queue='Q3',window=None,arm=None,steps=1,device='cuda',mixer_factory=None,recipe=None):
    """One first-fit rehearsal (Q5: two prerequisite fits plus diagnostics).

    Inputs are read only. All derived caches, checkpoints and publications live
    in a retained system temporary directory, never a Drive run or budget.
    CPU/mixer/recipe injection is for offline tests, not exposed by the CLI.
    """
    import shutil
    from .seq_runner import _x_tag
    if queue not in ('Q1','Q3','Q4','Q5') or type(steps)!=int or steps<1:raise ValueError('invalid rehearsal queue/steps')
    if device=='cuda':
        require_cuda()
        import torch
        torch.cuda.reset_peak_memory_stats()
    frozen=json.loads((Path(__file__).resolve().parents[2]/'tasks/p4a-colab-contract-20261121.json').read_text())
    jobs=[j for j in fit_queue(frozen) if j['queue']==('Q1' if queue=='Q5' else queue)]
    if queue=='Q5':
        if window not in (None,60) or arm not in (None,'crossstock'):raise ValueError('Q5 requires W60 crossstock and mamba2')
        jobs=[next(j for j in jobs if j['arm']==a) for a in ('crossstock','mamba2')]
    else:
        jobs=[j for j in jobs if (window is None or j['window']==window) and (arm is None or j['arm']==arm)]
        if not jobs:raise ValueError('rehearsal selection outside frozen queue')
        jobs=jobs[:1]
    root=Path(tempfile.mkdtemp(prefix='p4-REHEARSAL-',dir='/tmp'))
    print(f'REHEARSAL / 不是研究結果: {root}',flush=True)
    ref=pd.read_parquet(reference)
    if 'system' in ref:ref=ref[ref.system.eq('c0plus')]
    source=Path(data_dir).resolve(strict=True)
    year=jobs[0]['year'];source=source/f'folds/fold_{year}' if (source/'folds').is_dir() else source
    directory=root/'matrix';directory.mkdir()
    # Reuse immutable matrix bytes, but never allow SeqData to write caches to input.
    for name in MATRIX_FILES:
        (directory/name).symlink_to((source/name).resolve(strict=True))
    if (source/'X_rankgauss.npy').is_file() and (source/'X_rankgauss.tag').is_file() and (source/'X_rankgauss.tag').read_text()==_x_tag(source):
        (directory/'X_rankgauss.npy').symlink_to((source/'X_rankgauss.npy').resolve())
        shutil.copyfile(source/'X_rankgauss.tag',directory/'X_rankgauss.tag')
    class LocalGuard:
        def check(self):pass
        def observe(self,record):pass
    guard=LocalGuard();fits=[];paths={}
    for job in jobs:
        cfg=dict(frozen['recipe'] if recipe is None else recipe,window=job['window'],lr=job['lr'],epochs=1,steps_per_epoch=steps)
        contract=dict(**job,recipe=cfg,speed_test=False,evidence_class='REHEARSAL / 不是研究結果')
        if 'P4_REHEARSAL_REQUEST' in os.environ:
            contract['rehearsal_binding']=hashed(json.loads(Path(os.environ['P4_REHEARSAL_REQUEST']).read_text()))
        fit=root/f"{job['arm']}-w{job['window']}";attempt=fit/'attempt-active';attempt.mkdir(parents=True)
        observer=Observer(attempt,contract,guard,device=device)
        frame,metrics=train_fit(directory,year,job['arm'],job['seed'],cfg,ref[ref.fold.eq(year)],observer,mixer_factory=mixer_factory)
        publish_predictions(attempt,frame,ref[ref.fold.eq(year)],job['window'])
        atomic_json(attempt/'metrics.json',metrics)
        outputs={p.relative_to(fit).as_posix():digest(p) for p in attempt.iterdir() if p.is_file()}
        validate_fit_outputs(fit,contract,outputs,rehearsal=True)
        complete=dict(contract=contract,outputs=outputs,evidence_class=contract['evidence_class'])
        atomic_json(fit/'REHEARSAL-complete.json',complete)
        saved=json.loads((fit/'REHEARSAL-complete.json').read_text())
        validate_fit_outputs(fit,saved['contract'],saved['outputs'],rehearsal=True)
        fits.append(fit.name);paths[(year,job['arm'],17,60,3e-4)]=fit
    import torch,platform
    result=dict(evidence_class='REHEARSAL / 不是研究結果',artifact_class='TEMPORARY',output=str(root),queue=queue,window=window,arm=arm,steps=steps,fits=fits,
                inputs=dict(data_realpath=str(source),matrix_manifest=digest(source/'manifest.json'),reference=digest(reference)),
                implementation={name:digest(Path(__file__).with_name(name)) for name in ('p4_colab.py','p4_eval.py','seq_runner.py')},
                environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,mixer='official' if mixer_factory is None else 'TEST_DOUBLE'))
    if queue=='Q5':result['diagnostic']=permutation_fit(directory,year,cfg,paths,guard,device=device,mixer_factory=mixer_factory,rehearsal=True)
    result['cuda_max_memory_allocated']=torch.cuda.max_memory_allocated() if device=='cuda' else None
    print('torch.cuda.max_memory_allocated:',result['cuda_max_memory_allocated'],'bytes',flush=True)
    atomic_json(root/'REHEARSAL.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path);p.add_argument('--drive-root',type=Path)
    p.add_argument('--rehearse',action='store_true');p.add_argument('--reference',type=Path)
    p.add_argument('--queue',choices=('Q1','Q3','Q4','Q5'),default='Q3');p.add_argument('--window',type=int)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--steps',type=int,default=1)
    p.add_argument('--allow-new-run',action='store_true');p.add_argument('--speed-test',action='store_true');p.add_argument('--data-dir',type=Path)
    # Legacy flags are optional and ignored: P4 has no wall/CU caps.
    p.add_argument('--max-hours',type=float);p.add_argument('--max-units',type=float)
    p.add_argument('--units-per-hour',type=float);p.add_argument('--planned-hours-per-fit',type=float,default=0)
    p.add_argument('--reserve',type=float)
    a=p.parse_args(argv)
    if a.rehearse:
        reference=a.reference
        if reference is None and a.bundle is not None:
            reference=a.bundle/'reference/keys.parquet'
            if not reference.exists():reference=a.bundle/'reference/rank_average.parquet'
        if a.data_dir is None or reference is None:p.error('--rehearse requires --data-dir and --reference (or --bundle)')
        result=rehearse(a.data_dir,reference,queue=a.queue,window=a.window,arm=a.arm,steps=a.steps)
        if 'P4_REHEARSAL_REQUEST' in os.environ:
            request=json.loads(Path(os.environ['P4_REHEARSAL_REQUEST']).read_text())
            runtime=json.loads((a.bundle/'source/environments/p4a-colab-runtime.json').read_text())
            context=dict(source=digest(a.bundle/'bundle-manifest.json'),
                         data=json.loads((a.data_dir/'prepared.json').read_text()),
                         data_realpath=str(a.data_dir.resolve()),environment=environment_snapshot(runtime['packages']))
            result.update(request=request,context=context)
            atomic_json(Path(result['output'])/'REHEARSAL.json',result)
            atomic_json(Path(os.environ['P4_REHEARSAL_RESULT']),result)
        print(json.dumps(result,ensure_ascii=False));return
    if a.bundle is None or a.drive_root is None:p.error('formal execution requires --bundle and --drive-root')
    result=run(a.bundle,a.drive_root,speed=a.speed_test,data_dir=a.data_dir,allow_new_run=a.allow_new_run)
    print(json.dumps(dict(version=result['version'],jobs=len(result['jobs']),evidence_class=result['evidence_class'],
                          environment_mismatch=result['binding']['environment'].get('environment_mismatch',[]),
                          comparability=result['binding']['environment'].get('comparability'))))

if __name__=='__main__':main()
