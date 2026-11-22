"""P4a exposed historical comparisons using the unchanged P2 portfolio engine.

No independent confirmation is possible on these exposed folds, even when a
registered CI/Holm threshold is met. Prices must carry verified volume units.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import portfolio_eval as pe
from .p4_bundle import digest
from .significance import holm_adjust, paired_ic_test

KEYS=['fold','date','code']


def common_rows(systems):
    if not systems:raise ValueError('empty comparison')
    normalized={};indices={}
    for name,frame in systems.items():
        f=frame.copy();f['date']=pd.to_datetime(f.date);f['code']=f.code.astype(str)
        if f.empty or f.duplicated(KEYS).any():raise ValueError('empty/duplicate comparison rows')
        if not np.isfinite(f.label_y5).all():raise ValueError('nonfinite label')
        normalized[name]=f;indices[name]=pd.MultiIndex.from_frame(f[KEYS])
    common=next(iter(indices.values()))
    for idx in indices.values():common=common.intersection(idx,sort=False)
    if not len(common):raise ValueError('empty common support')
    aligned={name:f.set_index(KEYS).loc[common].reset_index().sort_values(KEYS).reset_index(drop=True)
             for name,f in normalized.items()}
    first=next(iter(aligned.values())).label_y5.to_numpy()
    if any(not np.array_equal(first,f.label_y5.to_numpy()) for f in aligned.values()):raise ValueError('common row label mismatch')
    report=dict(common_rows=len(common),systems={name:dict(input_rows=len(f),excluded_rows=len(f)-len(common),
                excluded_fraction=1-len(common)/len(f)) for name,f in normalized.items()})
    return aligned,report


def permutation_summary(dates,codes,labels,base,mamba,permutations):
    from scipy.stats import spearmanr
    arrays=[np.asarray(a) for a in (dates,codes,labels,base,mamba)]
    dates,codes,labels,base,mamba=arrays
    if len(permutations)!=10 or any(len(a)!=len(dates) for a in arrays):raise ValueError('permutation schema')
    if not all(np.isfinite(a).all() for a in (labels,base,mamba,*permutations)):raise ValueError('nonfinite permutation input')
    days=sorted(set(dates));daily=[];means=[]
    def corr(a,b):
        if len(a)<10 or np.std(a)==0 or np.std(b)==0:raise ValueError('unknown permutation correlation')
        value=float(spearmanr(a,b).statistic)
        if not np.isfinite(value):raise ValueError('unknown permutation correlation')
        return value
    baseline=[];temporal=[]
    for day in days:
        mask=dates==day;baseline.append(corr(base[mask],labels[mask]));temporal.append(corr(mamba[mask],labels[mask]))
    gap=abs(float(np.mean(baseline)-np.mean(temporal)))
    for r,scores in enumerate(permutations):
        scores=np.asarray(scores);ics=[]
        for i,day in enumerate(days):
            mask=dates==day;c=codes[mask];a=base[mask];b=scores[mask]
            top=lambda values:set(c[np.lexsort((c,-values))[:50]])
            ic=corr(b,labels[mask]);ics.append(ic)
            daily.append(dict(permutation=r,date=str(day),score_spearman=corr(a,b),ic=ic,
                              ic_change=ic-baseline[i],top50_overlap=len(top(a)&top(b))/min(50,len(c))))
        means.append(float(np.mean(ics)))
    sd=float(np.std(means,ddof=1))
    return dict(R=10,daily=daily,permutation_mean_ic=means,ic_std=sd,mamba_ic_gap=gap,
                order_sensitive=sd>=gap,rule='IC std >= absolute crossstock minus mamba2 IC gap',evidence_class='DESCRIPTIVE_ONLY')


def averaged_scores(scores):
    """Each cell requires exactly seeds 17/29 with identical keys and labels."""
    required={'fold','system','seed','window','lr','date','code','label_y5','score'}
    if not required<=set(scores) or not np.isfinite(scores.score).all():raise ValueError('score schema/nonfinite')
    cells={}
    for (arm,w,lr),g in scores.groupby(['system','window','lr']):
        if set(g.seed)!={17,29} and not (w==60 and lr in (1e-4,1e-3) and set(g.seed)=={17}):
            raise ValueError('incomplete seed support')
        keys=KEYS+['seed']
        if g.duplicated(keys).any():raise ValueError('duplicate seed rows')
        per_seed={str(seed):f for seed,f in g.groupby('seed')}
        aligned,report=common_rows(per_seed)
        if any(r['excluded_rows'] for r in report['systems'].values()):raise ValueError('incomplete seed row support')
        frames=[]
        for seed,f in aligned.items():
            f=f.copy();f['score']=f.groupby(['fold','date']).score.rank(method='average',pct=True);frames.append(f)
        result=pd.concat(frames).groupby(KEYS,as_index=False).agg(score=('score','mean'),label_y5=('label_y5','first'))
        name=arm if w==60 and lr==3e-4 else f'{arm}_w{w}' if lr==3e-4 else f'{arm}_lr{lr}'
        cells[name]=result
    return cells


def portfolio_rows(frames,prices,samples):
    config=replace(pe.p2_primary_config(),bootstrap_samples=samples)
    results={}
    for name,frame in frames.items():
        for year,pred in frame.groupby('fold'):
            pred=pred.assign(model=name,horizon=5)
            calendar=sorted(prices.date.unique());before=[d for d in calendar if d<pred.date.min()];after=[d for d in calendar if d>pred.date.max()]
            lo=before[-1] if before else pred.date.min();hi=after[0] if after else pred.date.max()
            price=prices[prices.date.between(lo,hi)].copy()
            idx=pd.MultiIndex.from_frame(price[['date','code']]);support=pd.MultiIndex.from_frame(pred[['date','code']])
            if not support.isin(idx).all():raise ValueError('missing prediction price keys')
            price['universe_eligible']=idx.isin(support)
            row=pe.checked_cell(price,pred,config)
            results[(int(year),name)]=row
    return results


def evaluate(scores,c0plus,prices,contract,samples=1000):
    frames=averaged_scores(scores);frames['c0plus']=c0plus[c0plus.system.eq('c0plus')].copy()
    tests={};intersections={};fold_tables=[]
    families=contract['families']
    for family,definition in families.items():
        names=sorted({n for pair in definition['pairs'] for n in pair}|{'c0plus'})
        present={name:frames[name] for name in names if name in frames}
        if set(present)!=set(names):
            tests[family]=dict(status='INCOMPLETE_ENGINEERING_ONLY',missing=sorted(set(names)-set(present)));continue
        # Family C uses one support across all lookback/60/C0+ systems.
        aligned,report=common_rows(present);intersections[family]=report
        rows=portfolio_rows(aligned,prices,samples);hypotheses={}
        for i,(a,b) in enumerate(definition['pairs'],1):
            paired=[];support=[];complete=True
            for year in (2021,2022,2023):
                aa,bb=rows.get((year,a)),rows.get((year,b))
                if aa is None or bb is None or aa['status']!='COMPUTED' or bb['status']!='COMPUTED':
                    complete=False;support.append(dict(fold=year,status='MISSING_OR_ERROR'));continue
                pair,info=pe.p3_pair(aa,bb);paired.append(pair);support.append(dict(fold=year,**info))
                if info['days']<20:complete=False
            values=[r['net'] for part in paired for r in part['daily_returns']]
            result=dict(p_boot=1.,ci95=None,mean_daily_difference=None)
            if len(values)>=20:
                stat=paired_ic_test(pd.Series(values),pd.Series(np.zeros(len(values))),5,n_boot=samples,block=20)
                result.update(p_boot=stat['p_boot'],ci95=stat['ci95_boot'],mean_daily_difference=stat['mean_diff'])
            result.update(model=a,reference=b,pairing=support,complete=complete and samples==1000,
                          replication=a in ('mamba2','crossstock') and family=='A')
            hypotheses[f'H-{family}{i}']=result
        for result,pval in zip(hypotheses.values(),holm_adjust([r['p_boot'] for r in hypotheses.values()])):
            result['p_holm']=pval
            result['registered_threshold_met']=bool(result['complete'] and result['ci95'] is not None and result['ci95'][0]>0 and pval<.05)
            result['decision']='歷史比較通過登記門檻（非獨立確認）' if result['registered_threshold_met'] else '未證實'
        tests[family]=dict(status='EXPOSED_HISTORICAL_ONLY',m=definition['m'],tests=hypotheses)
        fold_tables += [dict(family=family,fold=y,system=name,**row) for (y,name),row in rows.items()]
    # Descriptive folds and LR cells: explicit C0+ intersection for each cell.
    descriptions=[]
    for name,frame in frames.items():
        if name=='c0plus':continue
        aligned,report=common_rows({name:frame,'c0plus':frames['c0plus']})
        intersections['descriptive-'+name]=report
        rows=portfolio_rows(aligned,prices,samples)
        descriptions += [dict(fold=y,system=n,**row) for (y,n),row in rows.items()]
    return dict(evidence_class='EXPOSED_HISTORICAL_RESEARCH / NOT_ACCEPTED',independent_confirmation=False,
                hypotheses=tests,fold_tables=fold_tables,descriptive=descriptions,intersections=intersections,
                primary_config=asdict(pe.p2_primary_config()),limitations=contract['known_limitations'])


def load_run(run):
    from .p4_colab import resume_fit,hashed
    run=Path(run);binding=json.loads((run/'run_identity.json').read_text())
    if binding.get('speed_test'):raise ValueError('speed results cannot be evaluated')
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['binding']!=binding:raise ValueError('run manifest identity mismatch')
    frames=[];complete_files=set()
    for job in manifest['jobs']:
        paths=[p for p in run.glob('*/complete.json') if json.loads(p.read_text())==job]
        if len(paths)!=1:raise ValueError('fit manifest missing/ambiguous')
        fit=paths[0].parent;resume_fit(fit,job['contract']);complete_files.add(paths[0])
        frames.append(pd.read_parquet(fit/'attempt-active/scores.parquet'))
    if len(frames)!=64 or len(complete_files)!=64:raise ValueError('complete 64-fit run required')
    actual=[{k:j['contract'][k] for k in ('queue','year','arm','seed','window','lr')} for j in manifest['jobs']]
    if actual!=binding['queue'] or any(j['contract']['run_hash']!=hashed(binding) for j in manifest['jobs']):
        raise ValueError('fit queue identity mismatch')
    return pd.concat(frames,ignore_index=True),binding


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--c0plus-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    scores,binding=load_run(args.run_dir)
    c0,prices,_,_,inputs,diagnostics=pe.load_inputs(args.c0plus_dir,args.data_root)
    result=evaluate(scores,c0,prices,binding['contract'])
    result.update(inputs=inputs,price_diagnostics=diagnostics,run_identity=digest(args.run_dir/'run_identity.json'),
                  implementation=digest(Path(__file__)))
    from .rolling_seq_scores import write_json
    write_json(args.output,result)
    print(json.dumps(dict(output=str(args.output),identity=digest(args.output),independent_confirmation=False)))

if __name__=='__main__':main()
