"""Local-only P4d stage1 descriptive evaluator; never packaged into Colab.

All 140 completed fits are required; no P4a/P1 score reuse or Holm family reduction.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'V6'))
from tools import p4d
from tools.p4d_train import validate_complete
from research_lab.p4_bundle import digest
from research_lab.rolling_seq_scores import atomic_json
from research_lab import portfolio_eval as pe
from research_lab.p4_eval import portfolio_rows
from research_lab.significance import paired_ic_test


def load_run(root):
    root=Path(root);mf=json.loads((root/'completed.json').read_text());binding=json.loads((root/'run_identity.json').read_text())
    if mf['binding']!=binding or binding['experiment']!='p4d_stage1' or binding['queue']!=p4d.QUEUE_SHA or binding['contract']!=p4d.hashed(p4d.CONTRACT):raise ValueError('STOP: run identity')
    expected={p4d.hashed(job) for job in p4d.queue()}
    if set(mf['outputs'])!=expected:raise ValueError('STOP: INCOMPLETE 140-fit queue')
    frames={};metrics={}
    for job in p4d.queue():
        name=p4d.hashed(job);path=root/name
        if digest(path/'complete.json')!=mf['outputs'][name]:raise ValueError('STOP: complete identity')
        fit=json.loads((path/'complete.json').read_text())
        if fit['binding']['run']!=binding or fit['binding']['job']!=job:raise ValueError('STOP: fit lineage')
        validate_complete(path,fit['binding'])
        frames[name]=pd.read_parquet(path/'scores.parquet');metrics[name]=json.loads((path/'metrics.json').read_text())
    return frames,metrics,mf


def curves(frames):
    """Baseline, nested seed curve, all A cells and both single seeds, on full support."""
    outputs={};supports={}
    for origin in ('anchor',2021,2022,2023,2024,2025,2026):
        jobs=[j for j in p4d.queue() if j['origin']==origin]
        c0={j['seed']:frames[p4d.hashed(j)] for j in jobs if j['arm']=='c0plus'}
        reference,_=p4d.ensemble(c0,(17,29))
        baseline={j['seed']:frames[p4d.hashed(j)] for j in jobs if j['arm']=='gru' and (j['width'],j['layers'])==(64,2)}
        # Entire K8 universe must be complete before K1/K2 are even calculated.
        p4d.ensemble(baseline,p4d.SEEDS)
        for k in (1,2,4,8):
            f,s=p4d.ensemble({seed:baseline[seed] for seed in p4d.SEEDS[:k]},p4d.SEEDS[:k])
            name=f'{origin}/B/K{k}';outputs[name]=f;supports[name]=p4d.descriptive_pair(f,reference)
        for seed,f in baseline.items():
            name=f'{origin}/B/single{seed}';outputs[name]=f;supports[name]=p4d.descriptive_pair(f,reference)
        for width,layers in p4d.CAPACITIES:
            candidates={j['seed']:frames[p4d.hashed(j)] for j in jobs if j['arm']=='gru' and (j['width'],j['layers'])==(width,layers) and j['seed'] in (17,29)}
            f,s=p4d.ensemble(candidates,(17,29))
            name=f'{origin}/A/d{width}x{layers}/ensemble';outputs[name]=f;supports[name]=p4d.descriptive_pair(f,reference)
            for seed,frame in candidates.items():
                name=f'{origin}/A/d{width}x{layers}/single{seed}';outputs[name]=frame;supports[name]=p4d.descriptive_pair(frame,reference)
        outputs[f'{origin}/c0plus']=reference
    return outputs,supports


def frozen_prices(directory,manifest):
    snapshot=json.loads(Path(manifest).read_text());directory=Path(directory)
    if snapshot.get('primary_config')!=asdict(pe.p2_primary_config()):raise ValueError('STOP: frozen price Config')
    if snapshot.get('volume_units',{}).get('unit') not in ('shares','lots'):raise ValueError('STOP: volume source evidence')
    if not snapshot.get('source_provenance'):raise ValueError('STOP: source provenance missing')
    for name in ('prices_raw.parquet','market_value_raw.parquet'):
        if digest(directory/name)!=snapshot['files'][name]:raise ValueError('STOP: price snapshot identity')
    price=pd.read_parquet(directory/'prices_raw.parquet').rename(columns=dict(stock_id='code',Date='date',Open='open',Close='close',Volume='volume',Adj_Close='adj_close'))
    price['date']=pd.to_datetime(price.date);price['code']=price.code.astype(str)
    units=pe.volume_evidence(price)
    if units!=snapshot['volume_units']:raise ValueError('STOP: volume evidence mismatch')
    price=pe.normalize_volume(price,units)
    if price.duplicated(['date','code']).any():raise ValueError('STOP: price duplicate')
    return price,snapshot


def portfolios(outputs,prices):
    combined={}
    for key,f in outputs.items():
        origin,name=key.split('/',1)
        if origin=='anchor':continue
        combined.setdefault(name,[]).append(f)
    rows=portfolio_rows({name:pd.concat(fs,ignore_index=True) for name,fs in combined.items()},prices,1000)
    if any(r.get('status')!='COMPUTED' for r in rows.values()):raise ValueError('STOP: portfolio error')
    comparisons={}
    for name in combined:
        if name=='c0plus':continue
        pairs={}
        for year in range(2021,2027):
            pair,info=pe.p3_pair(rows[year,name],rows[year,'c0plus'])
            if info['excluded_model_dates'] or info['excluded_c0plus_dates']:raise ValueError('STOP: unpaired return dates')
            pairs[year]=pair
        pools={}
        for pool,years in (('2021-2023',(2021,2022,2023)),('2024-2026',(2024,2025,2026)),('all',range(2021,2027))):
            values=np.array([r['net'] for y in years for r in pairs[y]['daily_returns']])
            test=paired_ic_test(pd.Series(values),pd.Series(np.zeros(len(values))),5,n_boot=1000,block=20,seed=20261125)
            pools[pool]={k:test[k] for k in ('n','mean_diff','p_boot','ci95_boot','boot_block','n_boot','seed')}
        comparisons[name]=dict(pools=pools,fold_pairs=pairs,holm=None,
            decision='DESCRIPTIVE_ONLY; registered Holm2 family deferred (Mamba2 not approved)')
    return dict(rows=[dict(fold=y,system=n,**r) for (y,n),r in rows.items()],comparisons=comparisons,
        config=asdict(pe.p2_primary_config()),limitations=['exposed history','raw price returns exclude dividends','PIT/universe/delisting not verified','circular blocks may cross fold boundaries'])


def evaluate(run,output,*,price_root=None,snapshot=None):
    p4d.verify_contract();frames,metrics,mf=load_run(run);outputs,report=curves(frames)
    result=dict(evidence_class=p4d.CONTRACT['evidence_class'],independent_confirmation=False,
        run_identity=mf['binding'],support_and_ic=report,fit_metrics=metrics,
        resource_diagnostics=resource_diagnostics(metrics,frames),registered_family_complete=False,holm_m=2,decision='DESCRIPTIVE_ONLY; no stage1 family declaration')
    if price_root is not None:
        if snapshot is None:raise ValueError('STOP: frozen price snapshot required')
        prices,identity=frozen_prices(price_root,snapshot)
        result['portfolio']=portfolios(outputs,prices);result['price_snapshot']=identity
    else:result['portfolio_status']='UNVERIFIED / no frozen price snapshot supplied'
    output=Path(output)
    if output.exists():raise FileExistsError('fresh output namespace required')
    output.mkdir(parents=True)
    atomic_json(output/'evaluation.json',result)
    atomic_json(output/'manifest.json',dict(run=digest(Path(run)/'completed.json'),
        source=digest(__file__),outputs={'evaluation.json':digest(output/'evaluation.json')}))
    return result




def resource_diagnostics(metrics,frames):
    """Report every predeclared cell, never select a capacity/seed winner."""
    report={}
    for origin in ('anchor',2021,2022,2023,2024,2025,2026):
        jobs=[j for j in p4d.queue() if j['origin']==origin and j['arm']=='gru']
        base=[j for j in jobs if (j['width'],j['layers'])==(64,2)]
        base.sort(key=lambda j:p4d.SEEDS.index(j['seed']))
        costs={j['seed']:metrics[p4d.hashed(j)].get('fit_seconds_this_attempt') for j in base}
        points=[]
        for k in (1,2,4,8):
            values=[costs[s] for s in p4d.SEEDS[:k]]
            points.append(dict(k=k,fit_seconds=sum(values) if all(v is not None for v in values) else None))
        capacities=[]
        for j in jobs:
            if j['seed'] not in (17,29):continue
            m=metrics[p4d.hashed(j)]
            capacities.append(dict(width=j['width'],layers=j['layers'],seed=j['seed'],
                parameters=m.get('parameters'),seconds=m.get('fit_seconds_this_attempt'),peak_allocated=m.get('peak_allocated_bytes')))
        correlations=[]
        for i,a in enumerate(base):
            for b in base[i+1:]:
                fa,fb=frames[p4d.hashed(a)],frames[p4d.hashed(b)]
                p4d.descriptive_pair(fa,fb)  # Same keys/labels before any correlation.
                both=fa.merge(fb,on=['fold','date','code'],suffixes=('_a','_b'))
                daily=both.groupby('date').apply(lambda g:g.score_a.rank().corr(g.score_b.rank()),include_groups=False)
                if daily.isna().any():raise ValueError('unknown seed prediction correlation')
                correlations.append(dict(seed_a=a['seed'],seed_b=b['seed'],mean=float(daily.mean()),median=float(daily.median())))
        budgets=[]
        for k in (1,2,4,8):
            limit=costs[17]*k if costs[17] is not None else None
            eligible=[v for v in capacities if v['seed']==17 and v['seconds'] is not None and limit is not None and v['seconds']<=limit]
            budgets.append(dict(multiplier=k,seconds=limit,all_admissible_large_single_models=eligible,status='NO_COMPARABLE_POINT' if not eligible else 'DESCRIPTIVE_ONLY'))
        report[str(origin)]=dict(seed_costs=points,capacities=capacities,same_budget_points=budgets,prediction_correlations=correlations,
            timing_caveat='fit_seconds is this attempt only; use ledger for interruption costs; no interpolation; inference throughput not yet measured')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--price-root',type=Path);p.add_argument('--snapshot',type=Path)
    a=p.parse_args();evaluate(a.run,a.output,price_root=a.price_root,snapshot=a.snapshot)
