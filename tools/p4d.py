"""P4d stage1 identity, strict session splits and isolated preprocessing.

Research engineering only. Does not launch training at import or reuse P4a fits.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_lab.p4_colab import hashed
from research_lab.p4_bundle import digest

ROOT=Path(__file__).resolve().parents[1]
SPEC_SHA='c36aed96787e2b7e69fc92cd736ef3692c5542f1aecf350c4e0047470790fe54'
FEATURE_SHA='7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696'
SEEDS=(17,29,43,59,71,89,101,113)
CAPACITIES=((32,2),(64,2),(128,2),(256,2),(64,1),(64,4))
MACRO_NAMES=('TWII_Return','SPX_Return','VIX','TNX','Gold_Return','Oil_Return','USD_TWD',
             'Futures_OI_Foreign','Options_PC_Ratio','Fear_Greed','Business_Signal','FED_Rate')
SCHEMA=dict(version='p4d-rankgauss-v1',feature_order_sha256=FEATURE_SHA,
            zero_columns=list(range(36,48)),names=list(MACRO_NAMES),method='not_fitted',
            ties='average',nonfinite='nan/inf to zero before daily rank (frozen legacy policy)',dtype='float32')


def array_sha(a):
    a=np.asarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
    # Bounded contiguous chunks also work on mmap inputs.
    for start in range(0,len(a),65536):h.update(np.ascontiguousarray(a[start:start+65536]).tobytes())
    return h.hexdigest()


def anchor_seed(seed):
    return SEEDS[SEEDS.index(seed)%2]


def queue():
    jobs=[]
    # Entire anchor stage precedes any exposed outer score.
    for origin in ('anchor',2021,2022,2023,2024,2025,2026):
        for arm in ('gru','c0plus'):
            for seed in SEEDS[:2]:
                jobs.append(dict(phase='anchor' if origin=='anchor' else 'outer',origin=origin,
                    arm=arm,seed=seed,width=64,layers=2))
        for seed in SEEDS[2:]:
            jobs.append(dict(phase='anchor' if origin=='anchor' else 'outer',origin=origin,
                arm='gru',seed=seed,width=64,layers=2))
        for width,layers in CAPACITIES:
            if (width,layers)==(64,2):continue
            for seed in SEEDS[:2]:
                jobs.append(dict(phase='anchor' if origin=='anchor' else 'outer',origin=origin,
                    arm='gru',seed=seed,width=width,layers=layers))
    return jobs


QUEUE_SHA='32506a22ecc1154096ca5c616ef0f33dc565d1f14a45a412cea5c8f620d8892f'
CONTRACT=dict(version='p4d-stage1-v1',experiment='p4d_stage1',status='STAGE1_FROZEN',
    approval='Human 2026-11-24 conversation; no digital signature',spec_sha256=SPEC_SHA,
    seeds=list(SEEDS),capacities=[list(c) for c in CAPACITIES],macro_schema=SCHEMA,
    budget=dict(policy='record_only',units_per_hour=0,thermal_stop_c=85),
    deferred=['Mamba2','capacity beyond six cells','D','N12'],
    recipe=dict(window=60,d_model=64,n_layers=2,lr=3e-4,weight_decay=.01,epochs=8,
        steps_per_epoch=400,patience=3,dates_per_batch=8,stocks_per_date=256,min_names=64,
        listnet_w=0.,mixer_kw=dict(expand=2,headdim=8,d_state=16)),
    validation_sessions=126,purge=20,label_end_offset=11,history_year_sessions=252,min_history_years=4,
    evidence_class='E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED')


def verify_contract():
    stored=json.loads((ROOT/'tasks/p4d-stage1-contract-20261125.json').read_text())
    if stored!=dict(CONTRACT,queue_sha256=QUEUE_SHA,fit_count=140):raise ValueError('P4d contract mismatch')
    if hashed(queue())!=QUEUE_SHA:raise ValueError('P4d queue SHA mismatch')
    original=(ROOT/'tasks/spec-p4d-equal-tuning-20261125.md').read_bytes().split(
        b'\n## '+ '第一段分段凍結紀錄'.encode())[0]
    if hashlib.sha256(original).hexdigest()!=SPEC_SHA:raise ValueError('frozen specification bytes changed')


def split(data,origin,*,anchor=True,validation_sessions=126,min_names=64):
    """Indices use the frozen full calendar, never compressed eligible dates.

    Check all declared outer training rows before filtering; NaN evaluation labels
    remain identity rows. +11 is the inherited conservative label-end bound.
    """
    t=np.asarray(data.date_idx); n=len(data.dates)
    if type(origin)!=int or not 0<origin<n:raise ValueError('invalid origin')
    outer=np.flatnonzero(data.splits==1)
    if not len(outer) or np.any(t[outer]>=origin-20):raise ValueError('outer purge violated')
    for k in ('y5','y10'):
        if np.any(t[outer]+11>=origin) or np.isinf(data.y[k][outer]).any():raise ValueError('outer label maturity violated')
    # Strict < T-20, including precisely 20 excluded original sessions.
    validation=np.arange(origin-20-validation_sessions,origin-20) if anchor else np.array([],int)
    boundary=int(validation[0]) if anchor else origin
    last=boundary-21
    first=0
    # Account for the earliest complete W60 history, without removing calendar gaps.
    candidates=np.flatnonzero(data.eligible & (t>=first) & (t<=last))
    if not len(candidates):raise ValueError('empty fit support')
    start=int(t[candidates].min());years=(last-start+1)//252
    if years<4:raise ValueError('insufficient four-year history')
    fit_dates=np.arange(last-years*252+1,last+1)
    supports={}
    for name,days in (('fit',fit_dates),('validation',validation)):
        if not len(days):supports[name]=np.array([],dtype=np.int64);continue
        raw=np.flatnonzero(np.isin(t,days))
        if not np.array_equal(np.unique(t[raw]),days):raise ValueError(name+' missing session')
        # Maturity arithmetic is checked on unfiltered rows. Finite labels are
        # eligibility, but a completely immature selected day is a STOP.
        next_first=int(validation[0]) if name=='fit' and anchor else origin
        if np.any(t[raw]+11>=next_first):raise ValueError(name+' maturity boundary')
        finite=np.isfinite(data.y['y5'][raw]) & np.isfinite(data.y['y10'][raw])
        rows=raw[finite & data.eligible[raw]]
        days_used,counts=np.unique(t[rows],return_counts=True)
        if not np.array_equal(days_used,days) or np.any(counts<min_names):
            raise ValueError(name+' maturity/session support below min_names')
        supports[name]=rows
    ev=np.flatnonzero((data.splits==2)&data.eligible)
    if not len(ev) or int(t[ev].min())!=origin:raise ValueError('evaluation first session mismatch')
    # Missing a full evaluation date cannot be hidden by filtering.
    expected=np.unique(t[data.splits==2]);actual=np.unique(t[ev])
    if not np.array_equal(expected,actual):raise ValueError('evaluation session lost')
    supports['evaluation']=ev
    def info(rows):
        days=np.unique(t[rows]);return dict(rows=len(rows),sessions=len(days),
            first=str(data.dates[days[0]].date()) if len(days) else None,
            last=str(data.dates[days[-1]].date()) if len(days) else None,
            keys_sha256=hashed(dict(sessions=array_sha(t[rows]),codes=array_sha(data.codes[rows]))))
    supports['manifest']=dict(origin=origin,calendar_sha256=array_sha(np.asarray(data.dates,dtype='datetime64[D]')),
        support={k:info(v) for k,v in supports.items()},
        label_end_max={k:int(t[v].max()+11) if len(v) else None for k,v in supports.items()},
        halo_sessions=[int(fit_dates[0]-59),int(fit_dates[0]-1)],years=years,
        label_end_policy='conservative session +11; not verified execution/PIT evidence')
    return supports


def rankgauss(x,ends,support,cache=None,*,feature_names=None,rebuild=False):
    """Explicit macro schema; daily ranks never depend on later sessions."""
    from scipy.special import ndtri
    from scipy.stats import rankdata
    if x.ndim!=2 or x.shape[1]!=48:raise ValueError('48-feature shape required')
    if feature_names is not None:
        order=hashlib.sha256(json.dumps(feature_names,separators=(',',':')).encode()).hexdigest()
        if order!=FEATURE_SHA or tuple(feature_names[36:])!=MACRO_NAMES:raise ValueError('feature schema mismatch')
    ends=np.asarray(ends)
    if not len(ends) or ends[-1]!=len(x) or np.any(np.diff(ends,prepend=0)<=0):raise ValueError('session ends')
    binding=dict(schema=SCHEMA,implementation_sha256=digest(__file__)['sha256'],
        x_sha256=array_sha(x),feature_names=feature_names,calendar_ends_sha256=array_sha(ends),support=support)
    identity=hashed(binding);root=Path(cache)/identity if cache is not None else None
    if root and (root/'complete.json').exists() and not rebuild:
        mf=json.loads((root/'complete.json').read_text())
        if mf['binding']!=binding or mf['output']!=digest(root/'X.npy'):raise ValueError('rankgauss cache corrupt')
        return np.load(root/'X.npy',mmap_mode='r'),mf
    if root:
        root.mkdir(parents=True,exist_ok=True)
        target=root/'partial.npy'
        out=np.lib.format.open_memmap(target,mode='w+',dtype='float32',shape=x.shape)
    else:out=np.zeros(x.shape,dtype='float32')
    sessions=[];start=0
    for end in ends:
        block=np.asarray(x[start:end]);z=np.zeros(block.shape,dtype='float32')
        cleaned=np.nan_to_num(np.asarray(block,dtype=np.float64),nan=0.,posinf=0.,neginf=0.)
        for col in range(36):
            z[:,col]=ndtri((rankdata(cleaned[:,col],method='average')-.5)/len(block))
        out[start:end]=z
        sessions.append(dict(input=array_sha(block),output=array_sha(z)));start=int(end)
    mf=dict(identity=identity,binding=binding,sessions=sessions)
    if root:
        out.flush();del out
        target.replace(root/'X.npy');mf['output']=digest(root/'X.npy')
        from research_lab.rolling_seq_scores import atomic_json
        atomic_json(root/'complete.json',mf)
        return np.load(root/'X.npy',mmap_mode='r'),mf
    mf['output_array_sha256']=array_sha(out)
    return out,mf


def load_data(directory):
    from research_lab.seq_runner import SeqData
    # Never call legacy ensure_rankgauss/macro_columns or write to prepared data.
    directory=Path(directory)
    codes=np.load(directory/'stock_ids.npy',mmap_mode='r')
    cache=directory/'stock_idx.npy'
    if cache.exists():
        idx=np.load(cache,mmap_mode='r')
        if idx.shape!=codes.shape or idx.dtype.kind not in 'iu':raise ValueError('stock index cache shape/type')
        unique,first=np.unique(idx,return_index=True)
        if not np.array_equal(unique,np.arange(len(unique))):raise ValueError('stock index cache density')
        mapping=np.asarray(codes[first])
        if len(np.unique(mapping))!=len(mapping):raise ValueError('stock index cache code duplication')
        for start in range(0,len(idx),65536):
            if not np.array_equal(mapping[idx[start:start+65536]],codes[start:start+65536]):raise ValueError('stock index cache code identity')
    else:
        # Derive in memory; never let the legacy stock_index() write to a source.
        _,idx=np.unique(codes,return_inverse=True)
    d=SeqData(directory,stock_idx=idx,window=60,normalize=None)
    d.codes=codes
    return d


def ensemble(frames,expected_seeds):
    """Require identical support before daily ranks; tail NaNs are preserved."""
    from research_lab.p4_eval import common_rows
    if set(frames)!=set(expected_seeds):raise ValueError('INCOMPLETE seed set')
    aligned,report=common_rows({str(k):v for k,v in frames.items()})
    if any(v['excluded_rows'] for v in report['systems'].values()):raise ValueError('INCOMPLETE seed/date/code support')
    first=next(iter(aligned.values())).copy();ranks=[]
    if 'label_y10' in first:
        if any('label_y10' not in f or np.isinf(f.label_y10).any() or not np.array_equal(first.label_y10,f.label_y10,equal_nan=True) for f in aligned.values()):raise ValueError('label_y10 identity mismatch')
    for f in aligned.values():
        if not np.isfinite(f.score).all():raise ValueError('nonfinite seed score')
        ranks.append(f.groupby(['fold','date']).score.rank(method='average',pct=True).to_numpy())
    first['score']=np.mean(ranks,axis=0)
    return first,report


def descriptive_pair(neural,c0):
    from research_lab.p4_eval import common_rows
    aligned,support=common_rows({'gru':neural,'c0plus':c0})
    if any(s['excluded_rows'] for s in support['systems'].values()):raise ValueError('INCOMPLETE paired support')
    if 'label_y10' in neural or 'label_y10' in c0:
        if not all('label_y10' in f for f in aligned.values()) or not np.array_equal(aligned['gru'].label_y10,aligned['c0plus'].label_y10,equal_nan=True) or np.isinf(aligned['gru'].label_y10).any():raise ValueError('label_y10 identity mismatch')
    rows=[]
    for name,f in aligned.items():
        for (fold,date),g in f.groupby(['fold','date']):
            finite=np.isfinite(g.label_y5)
            if 'label_y10' in g:finite &= np.isfinite(g.label_y10)
            m=g[finite]
            ic=m.score.rank().corr(m.label_y5.rank()) if len(m)>=10 else np.nan
            if len(m) and not np.isfinite(ic):raise ValueError('unknown paired IC')
            rows.append(dict(system=name,fold=str(fold),date=str(date.date()),ic5=float(ic) if len(m) else None,
                input_rows=len(g),mature_rows=len(m),excluded_nan_rows=int((~finite).sum())))
    return dict(evidence_class=CONTRACT['evidence_class'],independent_confirmation=False,
        registered_family='Holm m=2 deferred; Mamba2 not approved; no family decision',support=support,daily=rows)
