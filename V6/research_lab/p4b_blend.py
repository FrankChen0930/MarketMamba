"""Frozen P4b adapter. Exposed historical research; no training or acquisition.

The 36-fit admission is separate from the unchanged P4a 64-fit gate.
All readers are read-only, including the append-only ledger reader.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
import subprocess
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
from .p4_eval import common_rows, averaged_scores, portfolio_rows, KEYS
from . import portfolio_eval as pe
from .p4_bundle import digest
from .p4_colab import resume_fit, hashed, fit_queue
from .significance import paired_ic_test, holm_adjust

SPEC_SHA='8d4f194730c2a147b42afe69750f722fccab1d0111e721d2a1f279df369f53c2'
P1_SHA='dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8'
RUN_HASH='6fbaf815b0716a086a794dfbc12a21f67688263d116469e272e57783448c3730'
UPGRADE_SHA='a77a884005b942279429acede859de06db707d96ee8b73ae342edee3fa6967ca'
RANKGAUSS=dict(bytes=1425394112,sha256='2484139bb9a3ef4580816e3cda110e0a825474479f078239524e3f6784689e8d')
ARMS=('mamba2','crossstock','gru')
BLENDS=('B1','B2','B3')
YEARS=tuple(range(2021,2027))
ROOT=Path(__file__).resolve().parents[2]


def read_json(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('STOP: metadata symlink')
    return json.loads(path.read_text())


def subset_jobs(queue):
    expected=[dict(queue='Q1' if y<2024 else 'Q2',year=y,arm=a,seed=s,window=60,lr=.0003)
              for y in YEARS for a in ARMS for s in (17,29)]
    actual=[j for j in queue if j.get('queue') in ('Q1','Q2')]
    if actual!=expected:raise ValueError('STOP: fixed 36-fit subset mismatch')
    return expected


def ledger_identity(root,binding):
    """Verify generations/head without opening a writer lock or recovering state."""
    previous=None;state=None;events=[];upgrades=[]
    for i,path in enumerate(sorted(root.glob('event-*.json')),1):
        e=read_json(path)
        if path.name!=f'event-{i:06d}.json' or e['previous']!=previous:
            raise ValueError('STOP: ledger chain/gap')
        if e['state']['contract']!=dict(policy='record_only',units_per_hour=0,thermal_stop_c=85):
            raise ValueError('STOP: ledger contract')
        previous=digest(path);state=e['state'];events.append(previous)
        if e['event']=='SOURCE_UPGRADE':upgrades.append(e['state']['source_upgrade'])
    if not events or read_json(root/'head.json')!=dict(sequence=len(events),identity=previous):
        raise ValueError('STOP: ledger head')
    audits=[]
    run=root/('run-'+hashed(binding))
    for saved in upgrades:
        path=run/Path(saved['audit']).name
        if digest(path)!=saved['identity']:raise ValueError('STOP: source upgrade hash')
        audit=read_json(path)
        if digest(path)['sha256']!=UPGRADE_SHA:raise ValueError('STOP: frozen source-upgrade identity')
        for k in ('from_sources_hash','to_sources_hash','declaration','observed_at','environment'):
            if saved[k]!=audit[k]:raise ValueError('STOP: upgrade ledger mismatch')
        if audit['event']!='SOURCE_UPGRADE' or audit['from_binding']!=binding:
            raise ValueError('STOP: source lineage')
        for side in ('from','to'):
            if hashed(audit[side+'_sources'])!=audit[side+'_sources_hash']:
                raise ValueError('STOP: source map hash')
        for k in ('contract','matrix','environment','queue','speed_test'):
            if audit['to_binding'][k]!=binding[k]:raise ValueError('STOP: upgrade changes research identity')
        audits.append(dict(path=str(path),identity=digest(path)))
    if {x['path'] for x in audits}!={str(p) for p in run.glob('source-upgrade-*.json')}:
        raise ValueError('STOP: orphan/unrecorded source upgrade')
    return dict(head=digest(root/'head.json'),events=len(events),upgrades=audits),state


def load_subset(run, *, retain_years=YEARS):
    run=Path(run);binding=read_json(run/'run_identity.json');manifest=read_json(run/'manifest.json')
    contract=read_json(ROOT/'tasks/p4a-colab-contract-20261121.json')
    if (binding['speed_test'] is not False or binding['contract']!=contract or
        hashed(binding)!=RUN_HASH or run.name!='run-'+hashed(binding) or manifest['binding']!=binding):
        raise ValueError('STOP: run/binding/frozen contract identity')
    if binding['queue']!=fit_queue(contract):raise ValueError('STOP: full queue identity')
    jobs=subset_jobs(binding['queue']);ledger,state=ledger_identity(run.parent,binding)
    by_job={};frames=[];identities=[]
    for entry in manifest['jobs']:
        fc=entry['contract'];key=tuple(fc.get(k) for k in ('queue','year','arm','seed','window','lr'))
        if key in by_job:raise ValueError('STOP: duplicate manifest fit')
        by_job[key]=entry
    for j in jobs:
        key=tuple(j.values());entry=by_job.get(key)
        fit=run/f"{j['queue']}-{j['year']}-{j['arm']}-{j['seed']}-w60-lr0.0003"
        if entry is None or read_json(fit/'complete.json')!=entry:raise ValueError('STOP: missing/changed fit')
        expected=dict(run_hash=hashed(binding),**j,recipe=contract['recipe'],rankgauss=RANKGAUSS,speed_test=False)
        if entry['contract']!=expected or expected not in state['completed']:
            raise ValueError('STOP: fit recipe/matrix/ledger identity')
        validated=resume_fit(fit,expected)
        if validated is None:raise ValueError('STOP: incomplete fit')
        # Locate the validated attempt, rather than assuming LAST or a directory name.
        names=[n for n in validated['outputs'] if Path(n).name=='scores.parquet']
        if len(names)!=1:raise ValueError('STOP: ambiguous score output')
        identities.append(dict(job=j,complete=digest(fit/'complete.json'),outputs=validated['outputs']))
        if j['year'] in retain_years:frames.append(pd.read_parquet(fit/names[0]))
    if len(identities)!=36:raise ValueError('STOP: 36 fits required')
    return pd.concat(frames,ignore_index=True),dict(run_hash=hashed(binding),binding=digest(run/'run_identity.json'),
              manifest=digest(run/'manifest.json'),fits=identities,ledger=ledger)


def load_c0(directory, *, retain_years=YEARS):
    directory=Path(directory);mf=directory/'manifest.json';manifest=read_json(mf)
    if digest(mf)['sha256']!=P1_SHA:raise ValueError('STOP: frozen P1 manifest')
    frames=[];identities={};dates={};date_source=None
    # The independent, pre-score evaluation date domain comes from frozen split bytes.
    for y in YEARS:
        entries={Path(k).name:v for k,v in manifest['inputs'].items() if f'/fold_{y}/' in k}
        if date_source is None:
            path=Path(entries['dates.npy']['path'])
            if digest(path)!={k:entries['dates.npy'][k] for k in ('bytes','sha256')}:raise ValueError('STOP: dates hash')
            date_source=np.load(path,allow_pickle=False)
            identities['dates']=dict(path=str(path),**digest(path))
        path=Path(entries['splits.npy']['path']);ident=entries['splits.npy']
        if digest(path)!={k:ident[k] for k in ('bytes','sha256')}:raise ValueError('STOP: split hash')
        split=np.load(path,allow_pickle=False)
        dates[y]=list(pd.to_datetime(np.unique(date_source[split==2])))
        identities[f'split_{y}']=dict(path=str(path),**digest(path))
        for s in (17,29,43):
            name=f'fold_{y}_seed_{s}.parquet';path=directory/name;ident=manifest['outputs'][name]
            if digest(path)!={k:ident[k] for k in ('bytes','sha256')}:raise ValueError('STOP: P1 seed hash')
            identities[name]=digest(path)
            if y in retain_years:
                f=pd.read_parquet(path);f=f[f.system.eq('c0plus')].copy()
                if f.empty or not f.seed.eq(s).all() or not f.fold.eq(y).all():raise ValueError('STOP: P1 seed identity')
                frames.append(f)
    return pd.concat(frames,ignore_index=True),{y:dates[y] for y in retain_years},dict(manifest=digest(mf),files=identities)


def rank(frame):
    return frame.groupby(['fold','date']).score.rank(method='average',pct=True)


def blend(neural,c0,expected_dates):
    """Validate original seed supports, intersect once, then rerank raw seeds."""
    expected={(int(y),pd.Timestamp(d)) for y,days in expected_dates.items() for d in days}
    if not expected or len(expected)!=sum(map(len,expected_dates.values())):raise ValueError('STOP: date domain')
    if set(neural.system)!=set(ARMS) or set(c0.system)!={'c0plus'}:raise ValueError('STOP: arm set')
    inputs={};seeds={}
    for name,f in [('c0plus',c0)]+list(neural.groupby('system')):
        f=f.copy();f['date']=pd.to_datetime(f.date);f['code']=f.code.astype(str)
        want={17,29,43} if name=='c0plus' else {17,29}
        if set(f.seed)!=want or not np.isfinite(f.score).all():raise ValueError('STOP: seed/nonfinite scores')
        per={str(s):g for s,g in f.groupby('seed')}
        aligned,info=common_rows(per)
        if any(r['excluded_rows'] for r in info['systems'].values()):raise ValueError('STOP: seed row support')
        for y in expected_dates:
            if not f.fold.eq(y).any():raise ValueError('STOP: entire fold missing')
        observed=set(zip(f.fold,f.date))
        if observed-expected:raise ValueError('STOP: unplanned date')
        seeds[name]=f;inputs[name]=next(iter(aligned.values()))
    aligned,report=common_rows(inputs);support=pd.MultiIndex.from_frame(aligned['c0plus'][KEYS])
    counts=aligned['c0plus'].groupby(['fold','date']).size().to_dict();daily=[]
    for name,f in inputs.items():
        original=f.groupby(['fold','date']).size().to_dict()
        for y,d in sorted(expected):
            num=int(counts.get((y,d),0));den=int(original.get((y,d),0))
            daily.append(dict(system=name,fold=y,date=d.isoformat(),input_rows=den,common_rows=num,
                              excluded_rows=den-num,retained=num/den if den else None,
                              passed=bool(den>=50 and num>=50 and num/den>=.95)))
    report.update(daily=daily,eligible=all(r['passed'] for r in daily),
                  status='ELIGIBLE' if all(r['passed'] for r in daily) else 'NOT_ELIGIBLE_FOR_FORWARD_NOMINATION')
    # Missing dates are integrity failures, whereas partial row retention is disclosed as ineligible.
    if any(r['input_rows']==0 or r['common_rows']==0 for r in daily):
        raise ValueError('STOP: entire day missing; '+json.dumps(report,sort_keys=True))
    restricted={name:f[pd.MultiIndex.from_frame(f[KEYS]).isin(support)] for name,f in seeds.items()}
    frames=averaged_scores(pd.concat([restricted[a] for a in ARMS],ignore_index=True))
    # C0 has three seeds: same shared seed-rank semantics, explicit 3-seed adapter.
    c=restricted['c0plus'].sort_values(['seed']+KEYS).copy();c['score']=c.groupby(['fold','date','seed']).score.rank(method='average',pct=True)
    c=c.groupby(KEYS,as_index=False).agg(score=('score','mean'))
    c=c.merge(aligned['c0plus'][KEYS+['label_y5']],on=KEYS,validate='one_to_one')
    frames['c0plus']=c
    for name,f in frames.items():
        f=f.sort_values(KEYS).reset_index(drop=True);f['score']=rank(f);frames[name]=f
    allframe=frames['c0plus'].copy();allframe['score']=sum(frames[a].score for a in ARMS)/3
    allframe['score']=rank(allframe);frames['R_all']=allframe
    for b,a in zip(BLENDS,('mamba2','R_all','gru')):
        f=frames['c0plus'].copy();f['score']=.5*f.score+.5*frames[a].score;frames[b]=f
    return frames,report


def signal_diagnostics(frames):
    daily=[];base=frames['c0plus'].set_index(KEYS)
    for name,f in frames.items():
        if name=='c0plus':continue
        for (y,d),g in f.groupby(['fold','date']):
            ref=base.loc[pd.MultiIndex.from_frame(g[KEYS])].reset_index()
            corr=g.score.reset_index(drop=True).corr(ref.score,method='spearman') if g.score.nunique()>1 and ref.score.nunique()>1 and len(g)>=2 else np.nan
            top=lambda x:set(x.sort_values(['score','code'],ascending=[False,True]).head(50).code)
            mask=np.isfinite(g.label_y5)
            ic=g.loc[mask,'score'].corr(g.loc[mask,'label_y5'],method='spearman') if mask.sum()>=2 and g.loc[mask,'score'].nunique()>1 and g.loc[mask,'label_y5'].nunique()>1 else np.nan
            daily.append(dict(fold=int(y),date=d.isoformat(),system=name,n=len(g),
                              spearman=float(corr) if np.isfinite(corr) else None,
                              top50_overlap=len(top(g)&top(ref))/50 if len(g)>=50 else None,
                              ic=float(ic) if np.isfinite(ic) else None,ic_used=int(mask.sum()),ic_excluded_nan=int((~mask).sum())))
    summary={}
    for name,g in pd.DataFrame(daily).groupby('system'):
        values=g.spearman.dropna()
        summary[name]=dict(valid_n=len(values),unknown_n=int(g.spearman.isna().sum()),
                           mean=float(values.mean()) if len(values) else None,median=float(values.median()) if len(values) else None,
                           quantiles={str(q):float(values.quantile(q)) for q in (.05,.25,.75,.95)} if len(values) else None)
    return dict(daily=daily,summary=summary,complete=all(x['spearman'] is not None and x['top50_overlap'] is not None for x in daily))


def inference(rows,support_ok,diagnostics_ok):
    if any((y,b) not in rows or rows[y,b]['status']!='COMPUTED' for y in YEARS for b in ('c0plus',)+BLENDS):
        raise ValueError('STOP: incomplete six-fold family')
    results={}
    for b in BLENDS:
        values={};pairing=[]
        for y in YEARS:
            pair,info=pe.p3_pair(rows[y,b],rows[y,'c0plus']);pairing.append(dict(fold=y,**info))
            v=[x['net'] for x in pair['daily_returns']]
            if len(v)<20 or not np.isfinite(v).all():raise ValueError('STOP: insufficient/nonfinite paired returns')
            values[y]=v
        def stat(years):
            v=np.concatenate([values[y] for y in years])
            return paired_ic_test(pd.Series(v),pd.Series(np.zeros(len(v))),5,n_boot=1000,block=20,seed=17)
        results[b]=dict(folds={str(y):stat([y]) for y in YEARS},registered=stat(YEARS[:3]),
                        recent=stat(YEARS[3:]),all=stat(YEARS),pairing=pairing)
    for b,p in zip(BLENDS,holm_adjust([results[b]['registered']['p_boot'] for b in BLENDS])):
        r=results[b];r['p_holm']=p
        r['registered_threshold_met']=bool(support_ok and r['registered']['ci95_boot'][0]>0 and p<.05)
        r['forward_candidate']=bool(r['registered_threshold_met'] and diagnostics_ok and
            sum(r['folds'][str(y)]['ci95_boot'][0]>0 for y in YEARS[:3])>=2 and
            all(r['folds'][str(y)]['mean_diff']>0 for y in YEARS[:3]) and r['recent']['mean_diff']>=0 and
            sum(r['folds'][str(y)]['mean_diff']>0 for y in YEARS[3:])>=2)
    return dict(m=3,tests=results,priority_candidates=[b for b in BLENDS if results[b]['forward_candidate']],
                interpretation='EXPOSED_DESCRIPTIVE_ONLY; fold/recent/all p-values unadjusted')


def price_slice(prices,pred):
    calendar=sorted(prices.date.unique())
    before=[d for d in calendar if d<pred.date.min()];after=[d for d in calendar if d>pred.date.max()]
    lo=before[-1] if before else pred.date.min();hi=after[0] if after else pred.date.max()
    price=prices[prices.date.between(lo,hi)].copy()
    idx=pd.MultiIndex.from_frame(price[['date','code']]);support=pd.MultiIndex.from_frame(pred[['date','code']])
    if not support.isin(idx).all():raise ValueError('STOP: missing prediction prices')
    price['universe_eligible']=idx.isin(support)
    return price


def holding_diagnostics(frames,prices):
    """Replay filled trade quantities, audit marks against engine NAV, never decide fills.

    A second unchanged engine pass supplies trades and daily cash omitted by its
    public summary. Only bootstrap draw count is reduced in this diagnostic pass.
    """
    from dataclasses import replace
    histories={};risk=[]
    for name,f in frames.items():
        for y,pred in f.groupby('fold'):
            price=price_slice(prices,pred)
            result=pe.evaluate_cell(price,pred.assign(model=name,horizon=5),replace(pe.p2_primary_config(),bootstrap_samples=1))
            q={};marks={};history={};trades={}
            for t in result['trades']:trades.setdefault(t['date'],[]).append(t)
            price=price.set_index(['date','code']);entries={r['entry']:r['signal'] for r in result['schedule']}
            def valid(r):
                return (np.isfinite(r.close) and r.close>0 and np.isfinite(r.volume) and r.volume>0 and
                        all(pd.notna(r.get(flag,False)) and not r.get(flag,False) for flag in ('outlier_flag','split_or_outlier_flag')))
            for day in result['daily']:
                date=day['date'];stamp=pd.Timestamp(date);terminal=date==result['end']
                if date in entries:
                    signal=pd.Timestamp(entries[date])
                    for code in q:
                        if (stamp,code) in price.index and (signal,code) in price.index:
                            r=price.loc[(stamp,code)];prior=price.loc[(signal,code)]
                            if np.isfinite(r.open) and r.open>0 and valid(prior):marks[code]=r.open
                for t in trades.get(date,[]):
                    q[t['code']]=q.get(t['code'],0)+t['notional']/t['price'];marks[t['code']]=t['price']
                if not terminal:
                    for code in q:
                        if (stamp,code) in price.index:
                            r=price.loc[(stamp,code)]
                            if valid(r):marks[code]=r.close
                weights={code:quantity*marks[code]/day['nav'] for code,quantity in q.items() if quantity>1e-12}
                cash=day['cash']/day['nav']
                if not np.isclose(sum(weights.values())+cash,1,atol=1e-9,rtol=0):raise ValueError('STOP: holding replay NAV mismatch')
                history[date]=dict(weights=weights,cash=cash,rebalance=date in entries)
            histories[int(y),name]=history
            risk.append(dict(fold=int(y),system=name,statistics=result['statistics'],net_total_return=result['statistics']['total_return'],
                             gross_total_return=result['statistics']['total_return']+result['gross_minus_net_return'],gross_minus_net_return=result['gross_minus_net_return'],
                             transaction_cost=result['daily'][-1]['cumulative_cost'],annual_turnover=result['annual_turnover'],holding_count=result['holding_count'],
                             max_drawdown=result['metrics']['max_drawdown'],minimum_fee_cost_fraction=result['minimum_fee_cost_fraction'],
                             daily_cash=[dict(date=d,cash=h['cash']) for d,h in history.items()]))
    daily=[]
    for (y,name),history in histories.items():
        if name=='c0plus':continue
        base=histories[y,'c0plus']
        if set(base)!=set(history):raise ValueError('STOP: holding calendar mismatch')
        for d,h in history.items():
            a=h['weights'];b=base[d]['weights'];intersection=set(a)&set(b);union=set(a)|set(b)
            daily.append(dict(fold=y,date=d,system=name,rebalance=h['rebalance'],
                              jaccard=len(intersection)/len(union) if union else None,
                              intersection_smaller=len(intersection)/min(len(a),len(b)) if a and b else None,
                              weight_overlap=sum(min(a[c],b[c]) for c in intersection),cash_model=h['cash'],cash_c0plus=base[d]['cash']))
    summaries=[]
    for (y,name),g in pd.DataFrame(daily).groupby(['fold','system']):
        for scope,part in [('all_holding_days',g),('rebalance_days',g[g.rebalance])]:
            summaries.append(dict(fold=int(y),system=name,scope=scope,days=len(part),
                **{key:dict(valid_n=int(part[key].notna().sum()),unknown_n=int(part[key].isna().sum()),
                     mean=float(part[key].mean()) if part[key].notna().any() else None)
                   for key in ('jaccard','intersection_smaller','weight_overlap')}))
    # Empty holdings have explicitly UNKNOWN set overlap; that is expected cash,
    # whereas replay/row errors throw and block evaluation.
    return dict(daily=daily,summary=summaries,risk=risk,complete=True)


def evaluate(neural,c0,prices,expected_dates):
    if set(expected_dates)!=set(YEARS):raise ValueError('STOP: all six folds required')
    frames,support=blend(neural,c0,expected_dates)
    signals=signal_diagnostics(frames)
    rows=portfolio_rows(frames,prices,1000)
    if any(row['status']!='COMPUTED' for row in rows.values()):raise ValueError('STOP: portfolio cell error')
    holdings=holding_diagnostics(frames,prices)
    tests=inference(rows,support['eligible'],signals['complete'] and holdings['complete'])
    return dict(evidence_class='EXPOSED_HISTORICAL_RESEARCH / NOT_ACCEPTED',independent_confirmation=False,
                support=support,signals=signals,holdings=holdings,inference=tests,
                fold_tables=[dict(fold=y,system=name,**row) for (y,name),row in rows.items()],
                primary_config=asdict(pe.p2_primary_config()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--c0plus-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--evaluate',action='store_true')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--snapshot-manifest',type=Path,help='Pre-frozen prices/market_value identities and complete Config')
    mode.add_argument('--rehearse-2021',action='store_true',help='Integrity checks plus 2021 blends only; no portfolio evaluation')
    a=p.parse_args(argv)
    out=a.output_dir.resolve()
    if not out.is_relative_to(ROOT/'.artifacts/p4b') or out.exists():raise ValueError('fresh .artifacts/p4b namespace required')
    out.mkdir(parents=True)
    try:
        years=(2021,) if a.rehearse_2021 else YEARS
        snapshot=None;prices=None;price_info=None
        if a.evaluate:
            if a.data_root is None or a.snapshot_manifest is None:raise ValueError('STOP: frozen price snapshot required')
            snapshot=read_json(a.snapshot_manifest)
            if snapshot['primary_config']!=asdict(pe.p2_primary_config()):raise ValueError('STOP: frozen Config mismatch')
            for name in ('prices_raw.parquet','market_value_raw.parquet'):
                if digest(a.data_root/name)!=snapshot['files'][name]:raise ValueError('STOP: price snapshot mismatch')
            _,prices,_,_,price_info,price_diagnostics=pe.load_inputs(a.c0plus_dir,a.data_root)
            if price_diagnostics['volume_units']!=snapshot['volume_units']:raise ValueError('STOP: volume identity mismatch')
        n,ni=load_subset(a.run_dir,retain_years=years);c,d,ci=load_c0(a.c0plus_dir,retain_years=years)
        frames,support=blend(n,c,d);outputs={}
        for name,f in frames.items():
            path=out/(name+'.parquet');f.to_parquet(path,index=False);outputs[path.name]=digest(path)
        result=dict(status='CPU_ENGINEERING_ONLY',evidence_class='E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED',
                    spec_frozen_sha=SPEC_SHA,neural=ni,c0plus=ci,support=support,outputs=outputs,
                    primary_config=asdict(pe.p2_primary_config()),
                    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),device='CPU',
                                     numpy=np.__version__,pandas=pd.__version__,MARKETMAMBA_DATA_ROOT=os.getenv('MARKETMAMBA_DATA_ROOT'),
                                     Data_realpath=str((ROOT/'Data').resolve())),
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                    source_zip=dict(sha256='b9060b740d0e94db7ae12e483ac20a095f0a7b3fce7e1ef1e13f1fa660d12116',evidence='Human-provided hash; archive not re-opened'),
                    implementation={str(Path(x).relative_to(ROOT)):digest(x) for x in
                        [__file__,ROOT/'V6/research_lab/p4_eval.py',ROOT/'V6/research_lab/p4_colab.py',ROOT/'V6/research_lab/portfolio_eval.py',ROOT/'V6/research_lab/significance.py']},
                    limitations=['2021 blend calculation only; no portfolio, inference or nomination',
                                 'C0+ three tuned seeds vs neural two fixed seeds; exposed descriptive history',
                                 'raw prices, PIT/universe/delisting execution evidence remain unverified'])
        if a.evaluate:
            evaluation=evaluate(n,c,prices,d)
            path=out/'evaluation.json';path.write_text(json.dumps(evaluation,indent=2,allow_nan=False))
            result['outputs'][path.name]=digest(path)
            result.update(status='EXPOSED_DESCRIPTIVE_EVALUATION',price_inputs=price_info,
                          snapshot=dict(identity=digest(a.snapshot_manifest),content=snapshot),limitations=[
                          'Exposed descriptive history; C0+/neural tuning and seed asymmetry',
                          'Raw returns, unverified PIT/universe/delisting/tradability',
                          '2026 partial; pooled circular blocks may cross folds; 1000 draws'])
        (out/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps(dict(output=str(out),manifest=digest(out/'manifest.json'),rows=len(frames['B1']),eligible=support['eligible'])))
    except Exception as e:
        (out/'STOP.json').write_text(json.dumps(dict(status='STOP',type=type(e).__name__,error=str(e)),indent=2))
        raise

if __name__=='__main__':main()
