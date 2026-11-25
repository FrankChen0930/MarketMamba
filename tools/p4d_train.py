"""Stage1 fresh fits; only anchor baselines select BEST, all other fits fix updates.

No runtime is launched by import. Formal execution requires a verified rehearsal.
"""
from __future__ import annotations
from copy import deepcopy,copy
import io
import json
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import pandas as pd
from . import p4d
from research_lab.p4_bundle import digest
from research_lab.p4_colab import Observer, Ledger, Progress, publish_bytes, require_cuda, telemetry
from research_lab.rolling_seq_scores import atomic_json


class Checkpoints(Observer):
    """Reuse hash/identity/orphan validation; fixed endpoints never select on IC."""
    def __init__(self,*args,selection,**kwargs):
        super().__init__(*args,**kwargs);self.selection=selection

    def __call__(self,event,payload):
        if event!='epoch':return super().__call__(event,payload)
        import torch
        epoch=payload['state']['epoch'];name=f'epoch-{epoch:03d}-{uuid.uuid4().hex}.pt'
        # Serialize/fsync/hash locally before the durable copy and pointer commit.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='p4d-checkpoint-') as scratch:
            local=Path(scratch)/'state.pt'
            with local.open('wb') as f:
                torch.save(dict(contract=self.contract,**payload),f);f.flush();os.fsync(f.fileno())
            ident=digest(local)
            if publish_bytes(self.root/name,local.read_bytes())!=ident:raise ValueError('durable checkpoint copy mismatch')
        entry=dict(file=name,identity=ident,epoch=epoch)
        self.pointer['previous']=self.pointer['latest'];self.pointer['latest']=entry
        if self.selection=='fixed' or payload['state']['best']>self.best:
            self.best=payload['state']['best'];self.pointer['best']=entry
        atomic_json(self.root/'checkpoint-pointer.json',self.pointer)
        if self.progress:
            from datetime import datetime,timezone
            self.progress.update(last_checkpoint_at=datetime.now(timezone.utc).isoformat());self.progress.write()
        if hasattr(self.ledger,'append'):self.ledger.append('EPOCH_COMMITTED')


class CpuLedger:
    """Synthetic tests only; formal runtime always uses the durable Ledger."""
    def check(self):pass


def fit_gru(data,tr,va,cfg,seed,device,root,binding,*,fixed_epochs=None,ledger=None,progress=None,after_epoch=None):
    import torch
    from research_lab.seq_models import SeqRanker
    from research_lab.seq_runner import centred_rank,sample_batch,session_loss,warmup_cosine,predict,daily_ic_rows
    if fixed_epochs is not None and (type(fixed_epochs)!=int or not 1<=fixed_epochs<=cfg['epochs']):
        raise ValueError('fixed update endpoint outside frozen schedule')
    if fixed_epochs is None and not len(va):raise ValueError('BEST needs pre2021 validation')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    model=SeqRanker('gru',len(data.feature_names),cfg['d_model'],cfg['n_layers']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    # Always the original eight-epoch cosine cycle, even with an earlier endpoint.
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda i:warmup_cosine(i,cfg['epochs']*cfg['steps_per_epoch']))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    contract=dict(binding=binding,cfg=cfg,seed=seed,fixed_epochs=fixed_epochs,
        fit_sha=p4d.array_sha(tr),validation_sha=p4d.array_sha(va))
    observer=Checkpoints(root,contract,ledger or CpuLedger(),device=device,progress=progress,
                         selection='BEST' if fixed_epochs is None else 'fixed')
    # CPU synthetic checks have no CUDA/telemetry side effects.
    if device=='cpu':observer.check=lambda:None
    state=dict(epoch=0,best=-9.,bad=0,history=[],selected_epoch=None,
               selection='BEST_pre2021' if fixed_epochs is None else 'fixed_endpoint')
    best_sd=None;restored=observer.load_resume()
    if restored:
        model.load_state_dict(restored['model']);opt.load_state_dict(restored['opt']);sched.load_state_dict(restored['sched'])
        state=restored['state'];best_sd=restored['best_sd'];rng.bit_generator.state=restored['rng_state']
        np.random.set_state(restored['numpy_rng']);random.setstate(restored['python_rng'])
        torch.set_rng_state(restored['torch_rng'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(restored['cuda_rng'])
    by_date={int(day):tr[data.date_idx[tr]==day] for day in np.unique(data.date_idx[tr])}
    by_date={day:rows for day,rows in by_date.items() if len(rows)>=cfg['min_names']}
    if not by_date:raise ValueError('no train sessions')
    targets=[torch.from_numpy(centred_rank(data.y[k][tr],data.date_idx[tr])).to(device) for k in ('y5','y10')]
    endpoint=fixed_epochs or cfg['epochs']
    while state['epoch']<endpoint and (fixed_epochs is not None or state['bad']<cfg['patience']):
        losses=[];model.train()
        for step in range(cfg['steps_per_epoch']):
            observer('step_start',None)
            groups=sample_batch(rng,by_date,sorted(by_date),cfg['dates_per_batch'],cfg['stocks_per_date']);flat=groups.ravel()
            x=torch.from_numpy(data.windows(flat)).to(device)
            idx=torch.from_numpy(np.searchsorted(tr,flat)).to(device)
            scores=model(x);local=torch.arange(len(flat),device=device).view(groups.shape)
            loss=session_loss(scores,targets[0][idx],targets[1][idx],local,0.)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite gradient')
            opt.step();sched.step();losses.append(float(loss.detach()));observer('step_end',None)
            if (step+1)%50==0:observer('progress',dict(epoch=state['epoch']+1,step=state['epoch']*cfg['steps_per_epoch']+step+1,loss=float(np.mean(losses))))
        state['epoch']+=1;val=None
        if len(va):
            ic=daily_ic_rows(data,va,predict(model,data,va,device)[:,0],'y5')
            if not len(ic) or not np.isfinite(ic).all():raise ValueError('unknown validation IC')
            val=float(ic.mean())
        # Full eligible train support; no random probe/cap for capacity diagnostics.
        tic=daily_ic_rows(data,tr,predict(model,data,tr,device)[:,0],'y5')
        if not len(tic) or not np.isfinite(tic).all():raise ValueError('unknown full train IC')
        state['history'].append(dict(epoch=state['epoch'],loss=float(np.mean(losses)),val_ic5=val,train_ic5=float(tic.mean())))
        if fixed_epochs is not None or val>state['best']:
            if val is not None:state['best']=val
            state['bad']=0;state['selected_epoch']=state['epoch'];best_sd=deepcopy(model.state_dict())
        else:state['bad']+=1
        observer('epoch',dict(model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),state=state,
            best_sd=best_sd,rng_state=rng.bit_generator.state,numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None))
        observer('progress',dict(epoch=state['epoch'],step=state['epoch']*cfg['steps_per_epoch'],
            loss=state['history'][-1]['loss'],val_ic5=val))
        if after_epoch:after_epoch(deepcopy(state))
    if best_sd is None:raise ValueError('selected weights absent')
    model.load_state_dict(best_sd)
    state=deepcopy(state)
    state['training_step_seconds_this_attempt']=sum(observer.steps)
    state['telemetry_this_attempt']=observer.records
    return model,state


def fit_c0plus(data,tr,va,cfg,seed,*,fixed_iterations=None):
    from lightgbm import LGBMRegressor
    from research_lab.c0plus import build_features
    from research_lab.seq_runner import centred_rank,daily_ic_rows
    xf,names=build_features(data,tr,data.feature_names)
    if len(names)!=185:raise ValueError('C0+ feature identity not 185')
    xv=build_features(data,va,data.feature_names)[0] if len(va) else None
    target=centred_rank(data.y['y5'][tr],data.date_idx[tr])
    model=LGBMRegressor(n_estimators=fixed_iterations or 200,num_leaves=31,learning_rate=.05,
        min_child_samples=200,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=10.,
        random_state=seed,n_jobs=4,verbose=-1)
    history=[];best=-9.;best_iter=None;bad=0
    def callback(env):
        nonlocal best,best_iter,bad
        iteration=env.iteration+1
        if fixed_iterations is not None or iteration%25:return
        scores=env.model.predict(xv,num_iteration=iteration)
        ic=daily_ic_rows(data,va,scores,'y5')
        if not len(ic) or not np.isfinite(ic).all():raise ValueError('unknown C0+ validation IC')
        val=float(ic.mean());history.append(dict(iteration=iteration,val_ic5=val))
        if val>best:best=val;best_iter=iteration;bad=0
        else:bad+=25
        if bad>=75:
            from lightgbm.callback import EarlyStopException
            raise EarlyStopException(best_iter-1,[('validation','IC5',best,True)])
    model.fit(xf,target,callbacks=[callback])
    chosen=fixed_iterations or best_iter
    if chosen is None:raise ValueError('C0+ BEST missing')
    return model,dict(selected_iteration=chosen,history=history,feature_names=names,fit_rows=len(tr))


def validate_complete(root,binding):
    """No existence-only skips: verify bytes and row, selection and pointer identity."""
    root=Path(root);mf=json.loads((root/'complete.json').read_text())
    if mf.get('binding')!=binding:raise ValueError('completed fit identity mismatch')
    for name,ident in mf['outputs'].items():
        if Path(name).name!=name or (root/name).is_symlink() or digest(root/name)!=ident:raise ValueError('completed output hash mismatch')
    required={'scores.parquet','metrics.json','model.pt' if binding['job']['arm']=='gru' else 'model.txt'}
    if not required<=set(mf['outputs']):raise ValueError('missing completed output')
    scores=pd.read_parquet(root/'scores.parquet');metrics=json.loads((root/'metrics.json').read_text())
    if scores.empty or scores.duplicated(['fold','date','code']).any() or not np.isfinite(scores.score).all():raise ValueError('invalid scores')
    if p4d.hashed(scores[['fold','date','code']].astype(str).to_dict('list'))!=metrics['score_keys_sha256']:raise ValueError('score support identity')
    job=binding['job']
    support=binding['support']['support']
    expected_rows=support['validation' if job['phase']=='anchor' else 'evaluation']['rows']
    if len(scores)!=expected_rows or metrics['fit_rows']!=support['fit']['rows'] or metrics['val_rows']!=support['validation']['rows']:raise ValueError('completed fit support counts')
    for key,value in dict(fold=job['origin'],seed=job['seed'],system=job['arm']).items():
        if not scores[key].eq(value).all():raise ValueError('score job identity')
    if job['arm']=='gru':
        import torch
        selected=torch.load(root/'model.pt',map_location='cpu',weights_only=False)
        if selected['binding']!=binding:raise ValueError('model identity')
        pointer=json.loads((root/'checkpoint-pointer.json').read_text())
        for role in ('latest','previous','best'):
            entry=pointer[role]
            if entry is None:
                if role=='previous' and pointer['latest']['epoch']==1:continue
                raise ValueError('missing checkpoint role')
            if entry['file'] not in mf['outputs'] or mf['outputs'][entry['file']]!=entry['identity']:raise ValueError('pointer hash mismatch')
        payload=torch.load(root/pointer['best']['file'],map_location='cpu',weights_only=False)
        if any(not torch.equal(v,payload['best_sd'][k]) for k,v in selected['model'].items()):raise ValueError('selected checkpoint weights mismatch')
        latest=torch.load(root/pointer['latest']['file'],map_location='cpu',weights_only=False)
        if latest['contract']['binding']!=binding:raise ValueError('checkpoint binding identity')
        if latest['state']['history']!=metrics['history']:raise ValueError('checkpoint history mismatch')
        if metrics['selected_epoch']!=pointer['best']['epoch']:raise ValueError('selected epoch/pointer mismatch')
        if binding.get('fixed_epochs') is None:
            history=metrics['history'];best_epoch=max(range(len(history)),key=lambda i:history[i]['val_ic5'])+1
            if best_epoch!=metrics['selected_epoch']:raise ValueError('BEST selection mismatch')
        endpoint=binding.get('fixed_epochs')
        if endpoint is not None and (metrics['epoch']!=endpoint or pointer['best']['epoch']!=endpoint):raise ValueError('fixed endpoint mismatch')
    return mf


def execute_job(data,support,job,root,run_binding,selected,*,device='cuda',ledger=None,progress=None,rehearsal=False):
    from research_lab.seq_runner import predict
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    job_started=time.perf_counter()
    data=copy(data)
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.reset_peak_memory_stats()
        original_windows=data.windows
        def guarded_windows(rows):
            if ledger is not None:ledger.check()
            record=telemetry()
            if record['temperature_c']>=85:raise RuntimeError('STOP: GPU >=85C during inference/training')
            if progress:progress.update(temperature_c=record['temperature_c'],clock_mhz=record['clock_mhz'])
            return original_windows(rows)
        data.windows=guarded_windows
    baseline=(job['width'],job['layers'])==(64,2) and job['seed'] in (17,29) and job['phase']=='anchor'
    anchor=p4d.anchor_seed(job['seed']);choice=selected.get(f"{job['arm']}:{anchor}")
    if not baseline and choice is None:raise ValueError('anchor selection missing; no outer fitting')
    cfg=dict(p4d.CONTRACT['recipe'],d_model=job['width'],n_layers=job['layers'])
    if rehearsal:cfg.update(epochs=1,steps_per_epoch=1,patience=1)
    fixed=None if baseline else choice
    if rehearsal:fixed=1 if job['arm']=='gru' else 1
    binding=dict(run=run_binding,job=job,support=support['manifest'],cfg=cfg,fixed_epochs=fixed)
    if (root/'complete.json').exists():return validate_complete(root,binding)
    tr,va=support['fit'],support['validation']
    rows=va if job['phase']=='anchor' else support['evaluation']
    if not len(rows):raise ValueError('empty prediction support')
    if job['arm']=='gru':
        model,metrics=fit_gru(data,tr,va,cfg,job['seed'],device,root,binding,
            fixed_epochs=fixed,ledger=ledger,progress=progress)
        inference_started=time.perf_counter()
        score=predict(model,data,rows,device)[:,0]
        inference_seconds=time.perf_counter()-inference_started
        import torch
        stream=io.BytesIO();torch.save(dict(binding=binding,model=model.state_dict()),stream)
        publish_bytes(root/'model.pt',stream.getvalue())
        model_name='model.pt'
    else:
        model,metrics=fit_c0plus(data,tr,va,cfg,job['seed'],fixed_iterations=fixed)
        from research_lab.c0plus import build_features
        inference_started=time.perf_counter()
        score=model.predict(build_features(data,rows,data.feature_names)[0],num_iteration=metrics['selected_iteration'])
        inference_seconds=time.perf_counter()-inference_started
        model.booster_.save_model(str(root/'model.txt'),num_iteration=metrics['selected_iteration']);model_name='model.txt'
    frame=pd.DataFrame(dict(fold=job['origin'],date=data.dates[data.date_idx[rows]],code=data.codes[rows],
        label_y5=data.y['y5'][rows],label_y10=data.y['y10'][rows],score=score,system=job['arm'],seed=job['seed']))
    if not np.isfinite(frame.score).all():raise ValueError('nonfinite predictions')
    frame.to_parquet(root/'scores.parquet',index=False)
    metrics.update(fit_rows=len(tr),val_rows=len(va),eval_rows=len(rows),
        inference_seconds=inference_seconds,inference_rows_per_second=len(rows)/inference_seconds,
        fit_seconds_this_attempt=time.perf_counter()-job_started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else None,
        score_keys_sha256=p4d.hashed(frame[['fold','date','code']].astype(str).to_dict('list')),
        parameters=sum(p.numel() for p in model.parameters()) if job['arm']=='gru' else None,
        evidence_class='REHEARSAL / 不是研究結果' if rehearsal else p4d.CONTRACT['evidence_class'])
    atomic_json(root/'metrics.json',metrics)
    files=['scores.parquet','metrics.json',model_name]
    if job['arm']=='gru':files+=['checkpoint-pointer.json']+[p.name for p in root.glob('epoch-*.pt')]
    mf=dict(binding=binding,outputs={name:digest(root/name) for name in files})
    atomic_json(root/'complete.json',mf);validate_complete(root,binding)
    return mf


def prepare(directory,cache,*,anchor):
    data=p4d.load_data(directory)
    origin=int(data.date_idx[np.flatnonzero(data.splits==2)].min())
    year=int(data.dates[origin].year)
    if year not in range(2021,2027) or (anchor and year!=2021):raise ValueError('P4d origin calendar year')
    support=p4d.split(data,origin,anchor=anchor)
    data.Xn,preprocess=p4d.rankgauss(data.X,data.ends,support['manifest'],cache,feature_names=data.feature_names)
    return data,support,preprocess


def run(data_root,output,bundle,*,rehearsal_evidence):
    """Called only through guarded_launch; no formal training without exact gate."""
    from .p4d_bundle import pb
    from .p4d_bundle import run_binding
    p4d.verify_contract();require_cuda()
    bundle=Path(bundle);manifest=pb.verify(bundle)
    binding=run_binding(data_root,bundle)
    from .p4d_bundle import validate_gate
    publication_root=Path(output)/'progress/formal/rehearsal'
    # Live /tmp evidence and persisted publications use the same strict gate.
    if all(Path(f['path']).resolve().is_relative_to(publication_root.resolve()) for f in rehearsal_evidence.get('fits',{}).values()) and rehearsal_evidence.get('fits'):
        validate_gate(rehearsal_evidence,binding,publication_root)
    else:
        validate_gate(rehearsal_evidence,binding)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    # New experiment owns budget_p4d_stage1; never reuse or migrate P4a budget.
    run_hash=p4d.hashed(binding);run_root=root/'budget_p4d_stage1'/('run-'+run_hash)
    ledger_contract=dict(p4d.CONTRACT['budget'],experiment='p4d_stage1',run_hash=run_hash)
    selected={};current=None
    with Ledger(root/'budget_p4d_stage1',ledger_contract) as ledger, Progress(root/'progress/formal') as progress:
        # Acquire and initialize the ledger before creating run-*; otherwise the
        # existing missing-ledger protection correctly treats a new run as orphaned.
        if ledger.count==0:ledger.append('RUN_CREATED')
        run_root.mkdir(parents=True,exist_ok=True)
        ident=run_root/'run_identity.json'
        if ident.exists() and json.loads(ident.read_text())!=binding:raise ValueError('run identity mismatch')
        if not ident.exists():atomic_json(ident,binding)
        for index,job in enumerate(p4d.queue(),1):
            if job['phase']=='outer' and len(selected)!=4:raise ValueError('all anchor baselines must be frozen before outer')
            origin=job['origin'];key=(origin,job['phase'])
            if current!=key:
                data,support,preprocessing=prepare(Path(data_root)/'folds'/f"fold_{2021 if origin=='anchor' else origin}",
                    root/'rankgauss_cache',anchor=job['phase']=='anchor')
                from .p4d_bundle import require_frozen_split
                require_frozen_split(binding,origin,support)
                atomic_json(run_root/f'split-{origin}.json',dict(split=support['manifest'],preprocess=preprocessing['identity']))
                current=key
            path=run_root/p4d.hashed(job);cached=(path/'complete.json').exists()
            pcfg=dict(p4d.CONTRACT['recipe'])
            progress.start_fit(dict(queue=f"P4d {job['phase']} d{job['width']}x{job['layers']}",year=origin,
                arm=job['arm'],seed=job['seed'],window=60,lr=3e-4),index,140,pcfg,skipped=cached)
            ledger.start(job,0)
            try:
                started=time.perf_counter()
                mf=execute_job(data,support,job,path,binding,selected,ledger=ledger,progress=progress)
                ledger.end(job,True)
            except BaseException:
                ledger.end(job,False);raise
            metrics=json.loads((path/'metrics.json').read_text())
            if origin=='anchor' and job['seed'] in (17,29) and (job['width'],job['layers'])==(64,2):
                selected[f"{job['arm']}:{job['seed']}"]=metrics['selected_epoch'] if job['arm']=='gru' else metrics['selected_iteration']
                target=run_root/'selected_recipe.json'
                record=dict(selected=selected,policy='pre2021 only; original cosine period; fresh outer fits')
                if target.exists():
                    old=json.loads(target.read_text())
                    if any(k in old['selected'] and old['selected'][k]!=v for k,v in selected.items()):raise ValueError('selected recipe changed')
                if len(selected)==4:atomic_json(target,record)
            progress.finish_fit(skipped=cached)
        outputs={}
        # Revalidate the whole durable queue before publishing completion/download.
        for job in p4d.queue():
            path=run_root/p4d.hashed(job)
            complete=json.loads((path/'complete.json').read_text())
            if complete['binding']['job']!=job or complete['binding']['run']!=binding:raise ValueError('final fit lineage mismatch')
            validate_complete(path,complete['binding'])
            outputs[path.name]=digest(path/'complete.json')
        if len(outputs)!=140:raise ValueError('INCOMPLETE P4d queue')
        atomic_json(run_root/'completed.json',dict(binding=binding,outputs=outputs,selected=selected,
            evidence_class=p4d.CONTRACT['evidence_class']))
    return run_root


def dataset_identity(root):
    """Hash raw prepared inputs, never the legacy rank-gauss cache."""
    root=Path(root);names=('X','y5','y10','dates','stock_ids','end_indices')
    identity={}
    # Cache repeated physical inputs only within this verification call.
    known={}
    for year in range(2021,2027):
        fold=root/'folds'/f'fold_{year}'
        for name in (*names,'splits'):
            path=(fold/(name+'.npy')).resolve(strict=True)
            if path not in known:known[path]=digest(path)
            identity[f'{year}/{name}']=known[path]
        identity[f'{year}/manifest']=digest(fold/'manifest.json')
    return identity
