"""Explicit CPU-only leakage audit of exposed Step3 history, local inputs only.

Use --phase facts, fits, evaluation, perturbation, portfolios. All row artifacts
are private under the main checkout .artifacts/leak-audit; formal inputs immutable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd

from tools.leak_audit import (SOURCE, SHARED, feature_names, price_columns, scalefree,
                              perturb_future, bitwise_prefix)
from tools import step3 as s
from tools.step3_runner import array_digest
from tools.tradability_diagnostic import composite, daily_ic, day_groups, price_matrices

BUNDLE = SOURCE / '.artifacts/step3/bundle-72'
RAW = SOURCE / 'Data/v7_through_20260911/processed_v6'
CACHE = SOURCE / 'Data/v7_prepared_20260911/.prepare-cache'
EVIDENCE = 'EXPOSED_HISTORICAL_AUDIT_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED'


def write(path, value):
    def clean(v):
        if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [clean(x) for x in v]
        if isinstance(v, (float,np.floating)) and not np.isfinite(v): return None
        return v
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False, default=str))
    os.replace(tmp, path)


def identity(path):
    path = Path(path)
    return dict(path=str(path.absolute()), realpath=str(path.resolve()), bytes=path.stat().st_size,
                sha256=s.sha256(path))


def context():
    from tools.run_step3_evaluation import load_inputs
    print('Validating frozen input hashes and all 72 original fit outputs', flush=True)
    if feature_names()[:48] != json.loads((SHARED/'manifest.json').read_text())['feature_order']:
        raise ValueError('STOP shared feature order differs from versioned policy')
    result = load_inputs(BUNDLE, SOURCE / '.artifacts/step3/human-source-gate-20261126.json')
    ids = np.load(SHARED/'stock_ids.npy', mmap_mode='r')
    result['codes'] = np.unique(ids)
    return result


def axes():
    ends = np.load(SHARED/'end_indices.npy')
    dates = np.load(SHARED/'dates.npy', mmap_mode='r')
    si = np.load(SHARED/'stock_idx.npy', mmap_mode='r')
    return dates, np.repeat(np.arange(len(ends)), np.diff(ends, prepend=0)), si, dates[ends-1]


def row_lookup(keys, di, si, nstocks):
    encoded = di.astype(np.int64)*nstocks + si
    wanted = keys[:, 0].astype(np.int64)*nstocks + keys[:, 1]
    pos = np.searchsorted(encoded, wanted)
    if (pos >= len(encoded)).any() or not np.array_equal(encoded[pos], wanted):
        raise ValueError('STOP missing shared row')
    return pos


def period_summary(days, daily):
    year = days.astype('datetime64[Y]').astype(int)+1970
    return {name: dict(mean=float(np.nanmean(daily[(year >= lo)&(year <= hi)])),
                       days=int(np.isfinite(daily[(year >= lo)&(year <= hi)]).sum()))
            for name,lo,hi in [('2015-2020',2015,2020),('2021-2026',2021,2026)]}


def source_examples(out):
    """Record raw/cache evidence on three concrete dates; never invent release times."""
    dates = pd.to_datetime(['2024-01-05','2024-04-30','2024-07-04'])
    stock = '2330'
    path = CACHE / (hashlib.sha256(stock.encode()).hexdigest()+'.parquet')
    prepared = pd.read_parquet(path)
    cols = ['Date','stock_id','Close','Holdings_Large_Pct','Holdings_Large_Change',
            'Securities_Balance','Foreign_Net','Investment_Trust_Net','Margin_Balance',
            'Business_Signal','Fear_Greed','SPX_Return','VIX','TNX']
    examples = dict(prepared=prepared.loc[prepared.Date.isin(dates), cols].to_dict('records'), raw={}, identities=[identity(path)])
    for name in ['holdings_raw','securities_raw','institutional_raw','margin_raw','macro_raw',
                 'business_indicator','fear_greed','fed_rate']:
        p = RAW/(name+'.parquet')
        if not p.exists():
            examples['raw'][name] = {'status':'UNKNOWN_MISSING'}; continue
        # Filter stock at parquet read rather than load the full market tables.
        import pyarrow.parquet as pq
        schema = pq.read_schema(p).names
        f = pd.read_parquet(p, filters=[('stock_id','=',stock)] if 'stock_id' in schema else None)
        dc = 'Week' if 'Week' in f else 'Date'
        f[dc] = pd.to_datetime(f[dc])
        samples = []
        for d in dates:
            g = f[f[dc] <= d].sort_values(dc).tail(2)
            samples.append(dict(signal_date=str(d.date()), rows=g.to_dict('records')))
        examples['raw'][name] = dict(columns=schema, samples=samples,
                                     publication_timestamp='UNKNOWN_NO_FIELD')
        examples['identities'].append(identity(p))
    examples['raw_cache'] = []
    for d in dates:
        day=str(d.date())
        paths=[SOURCE/'Data/raw_cache/daily'/f'{day}_{suffix}.parquet'
               for suffix in ('adj_price','chip','margin','securities')]
        weekly=sorted((SOURCE/'Data/raw_cache/weekly').glob('*_holdings.parquet'))
        prior=[p for p in weekly if p.name[:10]<=day]
        if prior: paths.append(prior[-1])
        for p in paths:
            if not p.exists(): continue
            f=pd.read_parquet(p)
            g=f.loc[f.stock_id.astype(str)==stock] if 'stock_id' in f else f.iloc[:0]
            examples['raw_cache'].append(dict(signal_date=day,file=p.name,columns=list(f.columns),
                rows=g.to_dict('records'),publication_timestamp='UNKNOWN_NO_RECEIPT_EVIDENCE'))
            examples['identities'].append(identity(p))
    write(out/'t1_examples.json', examples)


def facts(out):
    start = time.monotonic()
    c = context(); days = c['calendar']; codes = c['codes']
    dates, di, si, matrix_days = axes()
    mapping = np.searchsorted(days, matrix_days)
    avg = np.full((len(days),len(codes)), np.nan, np.float32)
    sources = []
    rawdir = SOURCE/'Data/raw_cache/daily'
    for i,d in enumerate(days):
        p = rawdir/f'{d}_adj_price.parquet'
        if not p.exists(): continue
        f = pd.read_parquet(p, columns=['date','stock_id','Trading_money','Trading_Volume'])
        if f.stock_id.duplicated().any() or set(f.date.astype(str)) != {str(d)}:
            raise ValueError('STOP daily source axes '+str(p))
        f = f.set_index('stock_id').reindex(codes)
        money=f.Trading_money.to_numpy(float); vol=f.Trading_Volume.to_numpy(float)
        ok=np.isfinite(money)&np.isfinite(vol)&(money>0)&(vol>0)
        avg[i,ok]=(money[ok]/vol[ok]).astype(np.float32)
        sources.append(identity(p))
        if len(sources)%1000==0: print('Turnover source files',len(sources),flush=True)
    for filename,value in [('actual_average.npy',avg),('actual_average_rows.npy',avg[mapping[di],si])]:
        path=out/filename
        if path.exists():
            if not np.array_equal(np.load(path,mmap_mode='r'),value,equal_nan=True):
                raise ValueError('STOP existing turnover matrix mismatch')
        else: np.save(path,value)
    write(out/'daily_sources.json', sources)
    print('T1 concrete source examples',flush=True); source_examples(out)
    prices = price_matrices(Path(c['bundle']['files']['prices']['path']), days, codes)
    F = np.divide(prices['Close'],avg,out=np.full(avg.shape,np.nan),where=np.isfinite(avg)&(avg>0))
    keys=c['label_keys']; fv=F[keys[:,0],keys[:,1]]
    corr={'h20':daily_ic(keys,fv,c['labels'],len(days))}
    X=np.load(SHARED/'X.npy',mmap_mode='r'); base=feature_names()[:48]
    matrix_keys=keys.copy(); matrix_keys[:,0]=np.searchsorted(matrix_days,days[keys[:,0]])
    rows=row_lookup(matrix_keys,di,si,len(codes))
    for name in ('Open','High','Low','Close','MA_20','MA_60','ATR_14'):
        corr[name]=daily_ic(keys,fv,X[rows,base.index(name)],len(days))
    np.savez_compressed(out/'factor_daily.npz',days=days,**corr)
    summary={name:period_summary(days,v) for name,v in corr.items()}
    write(out/'t3a.json',dict(definition='adjusted_Close / (Trading_money / Trading_Volume); approximate, intraday VWAP differs from unadjusted close',
        correlations=summary,source_files=len(sources),finite_average_cells=int(np.isfinite(avg).sum()),
        average_days=[str(days[np.isfinite(avg).any(axis=1)][i]) for i in [0,-1]],elapsed_seconds=time.monotonic()-start))
    print('T3a',summary['h20'],flush=True)
    scores={a:composite(keys,c['labels'],[c['predictions'][a,seed] for seed in s.SEEDS]) for a in 'AB'}
    np.savez_compressed(out/'original.npz',keys=keys,labels=c['labels'],days=days,codes=codes,
        score_A=scores['A'],score_B=scores['B'],
        **{f'prediction_{a}_{seed}':c['predictions'][a,seed] for a in 'AB' for seed in s.SEEDS})
    write(out/'input_manifest.json',dict(evidence_class=EVIDENCE,source_commit='b447ca53ac5d63088b07c48129f27a1c7ee88be3',
        bundle=identity(BUNDLE/'bundle.json'),shared=identity(SHARED/'manifest.json'),
        python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        MARKETMAMBA_DATA_ROOT=os.getenv('MARKETMAMBA_DATA_ROOT'),data_realpath=str((SOURCE/'Data').resolve()),
        raw_realpath=str(RAW.resolve()),feature_names=feature_names(),
        arm_identity='A190 is frozen Step3 B (190); original Step3 A has 185',
        inputs_validated='load_inputs validates all source files and 72 COMPLETE fits; private original.npz is derived',
        original_keys_sha256=array_digest(keys),label_sha256=array_digest(c['labels'])))


def fits(out):
    from lightgbm import LGBMRegressor
    import lightgbm
    if shutil.disk_usage('/mnt/d').free < 60*1024**3: raise ValueError('STOP free <60GiB')
    # Revalidate frozen source and all fits immediately before any new fit.
    c=context(); del c; gc.collect()
    bundle=json.loads((BUNDLE/'bundle.json').read_text())
    avg=np.load(out/'actual_average_rows.npy',mmap_mode='r')
    dates,di,si,matrix_days=axes(); nstocks=int(si.max())+1
    names=feature_names(); root=out/'fits'; root.mkdir(exist_ok=True)
    for year in s.YEARS:
        keys=np.load(BUNDLE/f'keys_{year}.npy')
        er=row_lookup(keys,di,si,nstocks)
        xe=scalefree(np.load(BUNDLE/f'eval_{year}_B.npy',mmap_mode='r'),names,avg[er])
        for seed in s.SEEDS:
            if shutil.disk_usage('/mnt/d').free < 60*1024**3: raise ValueError('STOP free <60GiB')
            name=f'fold{year}-A190-scalefree-seed{seed}'; dest=root/name
            job=bundle['jobs'][s.job_id(year,seed,'B')]
            sample=np.load(BUNDLE/f'sample_{year}_{seed}.npy')
            target=np.load(bundle['files'][job['target_file']]['path'])
            xf=scalefree(np.load(BUNDLE/f'fit_{year}_{seed}_B.npy',mmap_mode='r'),names,avg[sample])
            request=dict(year=year,seed=seed,reference_job=s.job_id(year,seed,'B'),params=s.PARAMS,
                keys_sha256=array_digest(keys),sample_sha256=array_digest(sample),target_sha256=array_digest(target),
                fit_sha256=array_digest(xf),eval_sha256=array_digest(xe),
                transform_code=identity(Path(__file__).with_name('leak_audit.py')),lightgbm=lightgbm.__version__)
            if dest.exists():
                old=json.loads((dest/'manifest.json').read_text())
                if old['request']!=request: raise ValueError('STOP resume identity')
                for info in old['outputs'].values(): s.verify_file(info['path'],info['sha256'],info['bytes'])
                continue
            dest.mkdir(); started=time.monotonic()
            model=LGBMRegressor(**s.PARAMS,random_state=seed).fit(xf,target)
            pred=np.asarray(model.predict(xe),np.float64)
            if pred.shape!=(len(keys),) or not np.isfinite(pred).all(): raise ValueError('STOP predictions')
            np.save(dest/'prediction.npy',pred); model.booster_.save_model(str(dest/'model.txt'))
            write(dest/'manifest.json',dict(state='COMPLETE',request=request,
                  outputs={p.name:identity(p) for p in [dest/'prediction.npy',dest/'model.txt']},
                  seconds=time.monotonic()-started,fit_rows=len(xf),eval_rows=len(xe),features=xe.shape[1],
                  missing_actual_average_fit=int(np.isnan(xf[:,-1]).sum()),missing_actual_average_eval=int(np.isnan(xe[:,-1]).sum())))
            print(name,round(time.monotonic()-started,1),'seconds',flush=True)
            del xf,model; gc.collect()
        del xe; gc.collect()
    print('36/36 fits complete',flush=True)


def evaluation(out):
    from tools.step3_evaluation import _circular_paired_interval
    z=np.load(out/'original.npz'); keys=z['keys']; days=z['days']; y=z['labels']
    predictions=[]
    for seed in s.SEEDS:
        predictions.append(np.concatenate([np.load(out/f'fits/fold{year}-A190-scalefree-seed{seed}/prediction.npy') for year in s.YEARS]))
    score=composite(keys,y,predictions)
    old=daily_ic(keys,z['score_B'],y,len(days)); new=daily_ic(keys,score,y,len(days))
    old_a=daily_ic(keys,z['score_A'],y,len(days))
    year=days.astype('datetime64[Y]').astype(int)+1970
    results={}
    for name,lo,hi in [('2015-2020',2015,2020),('2021-2026',2021,2026)]:
        pick=(year>=lo)&(year<=hi); delta=old[pick]-new[pick]
        if not np.array_equal(np.isfinite(old[pick]),np.isfinite(new[pick])): raise ValueError('STOP paired support')
        drop=float(np.nanmean(delta))
        results[name]=dict(original_A185=float(np.nanmean(old_a[pick])),
            A190=float(np.nanmean(old[pick])),A_scalefree=float(np.nanmean(new[pick])),
            decline=drop,paired_days=int(np.isfinite(delta).sum()),decline_ci=_circular_paired_interval(delta),
            decision='PRICE_LEVEL_LEAK_MATERIAL' if drop>=.02 else 'MINOR' if drop>=.005 else 'NOT_MATERIAL')
    np.savez_compressed(out/'scalefree_daily.npz',days=days,original=old,scalefree=new,score=score)
    write(out/'t3b.json',dict(periods=results,block=20,replicates=1000,seed=17,
        reference='Original Step3 B190, called audit A190; 36 matched refits, 191 columns',
        caveat='Association and ablation, not proof that all IC difference is caused by future information'))
    print('T3b',results,flush=True)


def universe(out):
    from tools.forward_wp1 import load_lifecycle, tej_date, field
    raw=SOURCE/'.artifacts/tej/raw'
    book,summary=load_lifecycle(raw/'月營收盈餘_基本資料補充_2013_2026_export20261126.zip',raw/'SHA256SUMS.20261126')
    dates,di,si,matrix_days=axes()
    codes=np.unique(np.load(SHARED/'stock_ids.npy',mmap_mode='r'))
    rows=[]; omitted=[]
    for year in s.YEARS:
        lo=f'{year}-01-01'; hi=min(f'{year}-12-31',str(matrix_days[-1]))
        mask=(dates>=np.datetime64(lo))&(dates<=np.datetime64(hi))
        pool=set(codes[np.unique(si[mask])]); active=set(); dead=set(); unknown=0
        for code,row in book.rows.items():
            if row.get('snapshot_conflict'): unknown+=1; continue
            starts=[tej_date(row,n) for n in ['TSE上市日','OTC上市日'] if field(row,n)]
            end=tej_date(row,'下市日期')
            if starts and min(starts)<=hi and (end is None or end>lo): active.add(code)
            if starts and end and lo<=end<=hi: dead.add(code)
        missing=active-pool
        omitted.extend(dict(year=year,stock_id=v,delisted_in_year=v in dead) for v in sorted(missing))
        rows.append(dict(year=year,through=hi,pool=len(pool),tej_active=len(active),active_missing=len(missing),
            delisted=len(dead),delisted_present=len(dead&pool),delisted_missing=len(dead-pool),
            pool_outside_tej_active=len(pool-active),tej_conflicts=unknown,suspended='UNKNOWN_NO_SUSPENSION_HISTORY'))
    write(out/'t4.json',dict(summary=summary,years=rows,
        decision='SURVIVOR_BIAS' if any(r['active_missing'] for r in rows) else 'OK',
        qualification='Omission counts establish historical selection bias; omissions alone do not identify delisting as sole cause. TEJ snapshot effective dates are research proxies, not publication PIT.',
        source=identity(raw/'月營收盈餘_基本資料補充_2013_2026_export20261126.zip')))
    pd.DataFrame(omitted).to_parquet(out/'t4_omitted.parquet',index=False)
    print('T4',rows,flush=True)


def prior_session_values(panel, keys):
    out=np.full(len(keys),np.nan)
    valid=keys[:,0]>0
    out[valid]=panel[keys[valid,0]-1,keys[valid,1]]
    return out


def sanity(out):
    from scipy.stats import rankdata
    from tools.step3_evaluation import _spearman
    z=np.load(out/'original.npz'); keys=z['keys']; y=z['labels']; days=z['days']; score=z['score_B']
    permutations=np.full((10,len(days)),np.nan)
    rng=np.random.default_rng(17)
    for rows in day_groups(keys):
        good=rows[np.isfinite(y[rows])&np.isfinite(score[rows])]
        if len(good)<20: continue
        d=int(keys[rows[0],0]); labels=y[good]
        a=rankdata(score[good]); a-=a.mean(); b=rankdata(labels); b-=b.mean()
        den=np.linalg.norm(a)*np.linalg.norm(b)
        if den<=0: continue
        for trial in range(10): permutations[trial,d]=float(a@rng.permutation(b)/den)
    dates,di,si,matrix_days=axes(); mk=keys.copy(); mk[:,0]=np.searchsorted(matrix_days,days[keys[:,0]])
    loc=row_lookup(mk,di,si,len(z['codes']))
    X=np.load(SHARED/'X.npy',mmap_mode='r'); ret=X[loc,feature_names().index('Return_20d')]
    # Return_20d includes t close, all information known by signal close.
    momentum=daily_ic(keys,ret,y,len(days))
    panel=np.full((len(days),len(z['codes'])),np.nan,np.float32)
    panel[np.searchsorted(days,matrix_days)[di],si]=X[:,feature_names().index('Return_20d')]
    strict_momentum=daily_ic(keys,prior_session_values(panel,keys),y,len(days))
    write(out/'t5.json',dict(permutations={str(i):period_summary(days,v) for i,v in enumerate(permutations)},
        momentum=period_summary(days,strict_momentum),reversal=period_summary(days,-strict_momentum),
        momentum_including_t_close=period_summary(days,momentum),
        reversal_including_t_close=period_summary(days,-momentum),
        seed=17,trials=10,method='permute mature labels separately within each signal day, fixed score; average same-day rank IC',
        decision='Reported without a post-hoc significance threshold; inspect ten means against zero'))
    np.savez_compressed(out/'sanity_daily.npz',days=days,permutations=permutations,
                        momentum=strict_momentum,momentum_including_t_close=momentum)
    print('T5 strict pre-t momentum',period_summary(days,strict_momentum),flush=True)


def perturbation(out):
    """Replay original feature sub-builder, 20 stocks x 2023-2024 plus warmup.

    Tests temporal causality of the existing raw snapshot. It cannot test
    historical vintage or recompute backward adjustment from corporate actions.
    """
    import logging
    import pyarrow.parquet as pq
    from marketmamba.data.feature_engineer import build_features
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'V6/experimental'))
    from v7_integrated_train import RAW_SOURCES, gap_safe_features
    from v7_integrated_point_in_time import align_point_in_time_features
    logging.disable(logging.WARNING)
    codes=['1101','1102','1216','1301','1303','1402','2002','2301','2303','2308',
           '2317','2327','2330','2357','2382','2412','2454','2603','2881','2882']
    # Previous history starts in 2022 solely as warmup; evaluated prefix 2023-24.
    frames={}; identities=[]
    for name,file in RAW_SOURCES.items():
        path=RAW/file
        if not path.exists(): frames[name]=None; continue
        cols=pq.read_schema(path).names
        dc='Date' if 'Date' in cols else 'Week' if 'Week' in cols else None
        filters=[('stock_id','in',codes)] if 'stock_id' in cols else []
        f=pd.read_parquet(path,filters=filters or None)
        if dc:
            f[dc]=pd.to_datetime(f[dc])
            f=f[f[dc].between('2022-01-01','2024-12-31')].copy()
        frames[name]=f
        identities.append(identity(path))
    prices=frames['prices'].copy()
    for c in ('Open','High','Low','Close','Volume'):
        if c not in prices: raise ValueError('STOP price schema')
    prices['observation_valid']=np.isfinite(prices[['Open','High','Low','Close','Volume']]).all(axis=1)&(prices.Close>0)&(prices.Volume>=0)
    calendar=pd.DatetimeIndex(np.load(BUNDLE/'calendar.npy'))
    calendar=calendar[(calendar>='2022-01-01')&(calendar<='2024-12-31')]
    def replay(inputs, price):
        f=gap_safe_features(price,inputs,calendar,build_features)
        # Revenue_MoM/YoY are excluded from the frozen 48; replay only the
        # surviving Market_Cap_Log override. Legacy revenue PeriodIndex ctor is
        # incompatible with this installed pandas and irrelevant to this audit.
        f,_=align_point_in_time_features(f,None,inputs.get('market_value'),calendar)
        f=f.sort_values(['Date','stock_id']).reset_index(drop=True)
        return f[['Date','stock_id']],f[feature_names()[:48]].to_numpy(np.float32)
    keys,base=replay(frames,prices)
    np.save(out/'t2_baseline.npy',base)
    records=[]
    for t in ['2023-06-30','2023-12-29','2024-06-28']:
        prefix=(keys.Date>=pd.Timestamp('2023-01-01'))&(keys.Date<=pd.Timestamp(t))
        for seed in range(5):
            changed={k:(None if f is None else perturb_future(f,t,seed,
                    date_col='Date' if 'Date' in f else 'Week' if 'Week' in f else None)) for k,f in frames.items()}
            price=perturb_future(prices,t,seed)
            # observation_valid is a fixed support mask, not a numeric feature input.
            price['observation_valid']=prices.observation_valid
            k,b=replay(changed,price)
            same_keys=keys.equals(k)
            equal=same_keys and bitwise_prefix(base,b,prefix.to_numpy())
            count=int(np.count_nonzero(base[prefix].view('u4')!=b[prefix].view('u4'))) if same_keys else None
            records.append(dict(cut=t,seed=seed,keys_equal=same_keys,prefix_rows=int(prefix.sum()),
                bitwise_equal=equal,different_cells=count))
            print('T2',t,seed,equal,count,flush=True)
    frozen=json.loads((CACHE/'identity.json').read_text())['identity']
    write(out/'t2.json',dict(status='PASS_SUBBUILDER' if all(r['bitwise_equal'] for r in records) else 'FAIL',
        stocks=codes,window='2023-01-01..2024-12-31; 2022 warmup',columns=feature_names()[:48],
        checks=records,inputs=identities,
        sources_match_prepared_identity={Path(v['path']).name:v['sha256']==frozen['sources'].get(Path(v['path']).name) for v in identities},
        helper_sha256=s.sha256(Path(__file__).resolve().parents[1]/'V6/marketmamba/data/feature_engineer.py'),
        prepared_helper_sha256=frozen['helpers']['feature_engineer.py'],
        limitation='Original gap_safe_features + build_features + PIT overrides; excludes final universe join and zero-fill. Current helper code may differ from original source. Not end-to-end shared X reconstruction; snapshot perturbation does not regenerate backward corporate-action adjustment.'))


def portfolios(out):
    from tools.tradability_diagnostic import (load_finmind, admission, simulate_portfolios, performance)
    z=np.load(out/'original.npz'); days=z['days']; codes=z['codes']; keys=z['keys']
    # Original inference ranks include immature rows, as in prior diagnostic.
    scores={a:composite(keys,np.zeros(len(keys)),[z[f'prediction_{a}_{seed}'] for seed in s.SEEDS]) for a in 'AB'}
    mats,ids=load_finmind(days,codes,SOURCE/'Data/raw_cache/daily')
    prices=price_matrices(SOURCE/'.artifacts/p4b/snapshot-prices-20261125/prices_raw.parquet',days,codes)
    mask=np.zeros(len(keys),bool)
    for rows in day_groups(keys):
        d=int(keys[rows[0],0]); stock=keys[rows,1]
        if d+1>=len(days): continue
        mask[rows],_=admission(mats['amount20'][d,stock],mats['Trading_money'][d,stock],mats['Trading_Volume'][d,stock],
            mats['open'][d+1,stock],mats['close'][d+1,stock]-mats['spread'][d+1,stock],
            mats['Trading_Volume'][d+1,stock],days[d+1],10_000_000)
    rows=[]; years=days.astype('datetime64[Y]').astype(int)+1970
    coverage={}
    for policy in ['last_close','loss100']:
        paths,buy,sell,fail=simulate_portfolios(keys,scores,mask,prices,days,days[-1],
                                               kinds=('top10','equal'),missing_policy=policy)
        coverage[policy]=fail
        np.savez_compressed(out/f't6_{policy}.npz',days=days,**{f'{a}_{k}':paths[a,k,.001] for a in 'AB' for k in ('top10','equal')})
        for period,lo,hi in [('2015-2020',2015,2020),('2021-2026',2021,2026)]:
            pick=(years>=lo)&(years<=hi)
            metrics=[]
            for offset in range(20):
                result=performance(paths['B','top10',.001][offset,pick],paths['B','equal',.001][offset,pick],
                    buy['B','top10'][offset,pick],sell['B','top10'][offset,pick])
                if result: metrics.append(result)
            m=np.asarray(metrics,float)
            rows.append(dict(policy=policy,period=period,offsets=len(m),
                mean_offset_annual_return=float(m[:,0].mean()),mean_offset_annual_excess=float(m[:,1].mean()),
                min_offset_excess=float(m[:,1].min()),max_offset_excess=float(m[:,1].max()),
                mean_sessions=float(m[:,-1].mean())))
    write(out/'t6.json',dict(rows=rows,coverage=coverage,slippage=.001,threshold=10_000_000,
        source_identity_file='daily_sources.json',
        method='Mean of 20 fixed-offset annualized returns on supported segments; benchmark equal-weight same admitted universe. Unknown source entry remains excluded; future missing prices become cash exits. No claim of complete executable history.'))
    print('T6',rows,flush=True)


def validation(out):
    """Verify only new tests and this audit's identities, never run the full suite."""
    import subprocess
    import xml.etree.ElementTree as ET
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(root/'V6')+':'+str(root)}
    cmd=[sys.executable,'-m','pytest','tests/test_leak_audit.py','-q','-p','no:cacheprovider',
         '--junitxml='+str(out/'focused_tests.xml')]
    result=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True)
    (out/'focused_tests.log').write_text(result.stdout+result.stderr)
    if result.returncode: raise ValueError('STOP new tests failed; inspect focused_tests.log')
    xml=ET.parse(out/'focused_tests.xml').getroot()
    tests=sum(int(x.attrib.get('tests',0)) for x in xml.iter('testsuite'))
    original=json.loads((out/'input_manifest.json').read_text())
    for ref in ('bundle','shared'):
        info=original[ref];s.verify_file(info['path'],info['sha256'],info['bytes'])
    bundle=json.loads((BUNDLE/'bundle.json').read_text())
    z=np.load(out/'original.npz')
    reference=json.loads((root/'reports/step3_results_20261126.json').read_text())
    reproduction=[]
    for arm in 'AB':
        summary=period_summary(z['days'],daily_ic(z['keys'],z['score_'+arm],z['labels'],len(z['days'])))
        for period,stats in summary.items():
            expected=reference['periods'][period]['mean_'+arm.lower()]
            error=abs(stats['mean']-expected)
            if error>1e-9: raise ValueError('STOP original IC reproduction')
            reproduction.append(dict(period=period,arm=arm,expected=expected,actual=stats['mean'],absolute_error=error))
    models=[]
    dates,di,si,matrix_days=axes()
    average=np.load(out/'actual_average_rows.npy',mmap_mode='r')
    names=feature_names();matrix_checks=0
    for year in s.YEARS:
        keys=np.load(BUNDLE/f'keys_{year}.npy')
        eval_rows=row_lookup(keys,di,si,len(z['codes']))
        transformed_eval=scalefree(np.load(BUNDLE/f'eval_{year}_B.npy',mmap_mode='r'),names,average[eval_rows])
        eval_digest=array_digest(transformed_eval);matrix_checks+=1
        del transformed_eval
        for seed in s.SEEDS:
            p=out/f'fits/fold{year}-A190-scalefree-seed{seed}'
            m=json.loads((p/'manifest.json').read_text()); req=m['request']; job=bundle['jobs'][s.job_id(year,seed,'B')]
            if m['state']!='COMPLETE' or req['params']!=s.PARAMS or m['features']!=191:
                raise ValueError('STOP fit state/parameter/schema')
            for role in ('sample','target','keys'):
                if req[role+'_sha256']!=job[role+'_sha256']: raise ValueError('STOP original '+role+' identity')
            sample=np.load(bundle['files'][job['sample_file']]['path'])
            transformed_fit=scalefree(np.load(bundle['files'][job['fit_file']]['path'],mmap_mode='r'),names,average[sample])
            if array_digest(transformed_fit)!=req['fit_sha256'] or eval_digest!=req['eval_sha256']:
                raise ValueError('STOP transformed matrix recomputation identity')
            matrix_checks+=1;del transformed_fit
            for info in m['outputs'].values(): s.verify_file(info['path'],info['sha256'],info['bytes'])
            pred=np.load(p/'prediction.npy')
            if pred.shape!=(m['eval_rows'],) or not np.isfinite(pred).all(): raise ValueError('STOP prediction values')
            models.append(dict(job=p.name,seconds=m['seconds'],rows=m['fit_rows'],
                missing_actual_average_fit=m['missing_actual_average_fit'],missing_actual_average_eval=m['missing_actual_average_eval']))
    for file in ('t1_examples.json','t2.json','t3a.json','t3b.json','t4.json','t5.json','t6.json'):
        json.loads((out/file).read_text())
    from lightgbm import Booster
    booster=Booster(model_file=str(BUNDLE/'fits/fold2021-B-seed17/model.txt'))
    prediction=booster.predict(np.load(BUNDLE/'eval_2021_B.npy',mmap_mode='r'),num_threads=4)
    old=np.load(BUNDLE/'fits/fold2021-B-seed17/prediction.npy')
    prediction_error=float(np.max(np.abs(prediction-old)))
    if prediction_error>1e-12: raise ValueError('STOP original model inference reproduction')
    syntax=[]
    for p in [root/'tools/leak_audit.py',root/'tools/leak_audit_run.py',root/'tools/tradability_diagnostic.py',root/'tests/test_leak_audit.py']:
        compile(p.read_text(),str(p),'exec');syntax.append(str(p.relative_to(root)))
    check=subprocess.run(['git','diff','--check'],cwd=root,capture_output=True,text=True)
    if check.returncode: raise ValueError('STOP diff check '+check.stdout+check.stderr)
    write(out/'validation.json',dict(evidence_class=EVIDENCE,tests=tests,test_returncode=result.returncode,
        complete_fits=len(models),models=models,syntax=syntax,diff_check_returncode=check.returncode,
        transformed_matrices_recomputed_and_digest_matched=matrix_checks,
        baseline_reproduction=reproduction,
        original_model_2021_B_seed17_prediction_max_abs_error=prediction_error,
        original_bundle_and_shared_manifest_reverified=True,
        all_model_prediction_hashes_reverified=True,original_sample_target_keys_digests_match=True))
    print('Audit validation:',tests,'new tests,',len(models),'complete fits; syntax, identities, diff check passed',flush=True)


def finalize(out):
    """Preserve delivered uncommitted source and artifact inventory, no Git writes."""
    import subprocess
    from datetime import datetime,timezone
    root=Path(__file__).resolve().parents[1]
    v=json.loads((out/'validation.json').read_text())
    if v['test_returncode'] or v['complete_fits']!=36 or v['transformed_matrices_recomputed_and_digest_matched']!=48:
        raise ValueError('STOP incomplete validation')
    paths=['tools/leak_audit.py','tools/leak_audit_run.py','tools/tradability_diagnostic.py',
           'tests/test_leak_audit.py','reports/leak_audit_20261128.md',
           'knowledge/00_Project_Map/Current_State.md','knowledge/00_Project_Map/Authority_Map.md',
           'knowledge/01_Domains/Features.md','knowledge/01_Domains/Universe.md','knowledge/01_Domains/Data.md']
    write(out/'source_at_closure.json',dict(
        purpose='Preserve delivered source because Human prohibits commit; closure snapshot is not a claim that all phases loaded the final file version. Matrix digests are independently recomputed.',
        files={name:dict(**identity(root/name),content=(root/name).read_text()) for name in paths}))
    for log in Path('/tmp').glob('leak-*.log'):
        shutil.copyfile(log,out/log.name)
    for filename in ('leak-knowledge-impact.json','leak-knowledge-health.json'):
        p=Path('/tmp')/filename
        if p.exists(): shutil.copyfile(p,out/filename)
    status=subprocess.check_output(['git','status','--short','--branch'],cwd=root,text=True)
    inventory={str(p.relative_to(out)):identity(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    # Fit manifests are also outputs; exclude only the root manifest itself.
    for p in sorted((out/'fits').glob('*/manifest.json')):
        inventory[str(p.relative_to(out))]=identity(p)
    write(out/'manifest.json',dict(classification='GENERATED',external_payload_classification='EXTERNAL_LARGE_DATA',evidence_class=EVIDENCE,
        observed_at=datetime.now(timezone.utc).isoformat(),
        source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        source_worktree=str(root),git_status=status,commit='PROHIBITED_BY_TASK',push='PROHIBITED_BY_TASK',
        source_snapshot='source_at_closure.json',input_identity='input_manifest.json',
        results={name:json.loads((out/name).read_text()) for name in ['t3b.json','t6.json']},
        validation='validation.json',outputs=inventory,
        rebuildable_large_files=['actual_average.npy','actual_average_rows.npy','original.npz','t2_baseline.npy'],
        cleanup='No files deleted; verify immutable inputs/references before any later cleanup. Retain models, predictions, daily metrics, manifests and logs.'))
    print('Final inventory:',len(inventory),'files;',sum(v['bytes'] for v in inventory.values()),'bytes',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['facts','fits','evaluation','universe','sanity','perturbation','portfolios','source_examples','validation','finalize'],required=True)
    a=p.parse_args(); out=a.out.absolute(); allowed=SOURCE/'.artifacts/leak-audit'
    if out.resolve()!=out or not out.is_relative_to(allowed) or out==allowed:
        raise ValueError('STOP private output path')
    out.mkdir(parents=True,exist_ok=True)
    globals()[a.phase](out)


if __name__=='__main__': main()
