"""Deferred Step3 historical evaluation. NOT EXECUTED in engineering task.

No CLI and no import-time evaluation. Only an independently authorized later
execution may call evaluate and persist the result privately. Not trading evidence.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from tools.step3 import SEEDS
from tools.step3_runner import validate_outputs


def _spearman(a,b):
    a=rankdata(a,method='average');b=rankdata(b,method='average')
    a=a-a.mean();b=b-b.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom>0 else np.nan


def _circular_paired_interval(difference):
    x=np.asarray(difference,float);n=len(x)
    if not n or not np.isfinite(x).any():raise ValueError('STOP no paired evaluation days')
    rng=np.random.default_rng(17);means=[];offset=np.arange(20)
    for _ in range(1000):
        starts=rng.integers(0,n,size=(n+19)//20)
        pick=((starts[:,None]+offset)%n).ravel()[:n]
        sample=x[pick]
        if not np.isfinite(sample).any():raise ValueError('STOP empty bootstrap replicate')
        means.append(float(np.nanmean(sample)))
    return [float(v) for v in np.percentile(means,[2.5,97.5])]


def evaluate(keys_by_job,predictions,labels,row_session,session_dates,*,label_keys=None,frozen_labels=None,independent_execution_authorized=False):
    """Full ordered keys already concatenate folds in original row order.

    Both arms and all seeds must cover the entire W60 inference support, including
    immature tails. Common maturity is applied only after this identity check.
    No inference row may be removed due to revenue NaN or future price absence.
    """
    if not independent_execution_authorized:raise ValueError('STOP deferred evaluation requires separate authorization')
    validate_outputs(keys_by_job)
    if set(predictions)!=set(keys_by_job):raise ValueError('STOP predictions family')
    keys=keys_by_job['A',17];n=len(keys)
    y=np.asarray(labels,float);t=np.asarray(row_session);days=np.asarray(session_dates,dtype='datetime64[D]')
    if y.shape!=(n,) or t.shape!=(n,) or t.dtype.kind not in 'iu' or (t<0).any() or (t>=len(days)).any():
        raise ValueError('STOP evaluation alignment')
    if label_keys is None or not np.array_equal(np.asarray(label_keys),keys) or not np.array_equal(t,keys[:,0]):
        raise ValueError('STOP label keys/session alignment')
    if frozen_labels is None or not np.array_equal(y,np.asarray(frozen_labels,float),equal_nan=True):
        raise ValueError('STOP labels differ from frozen identity')
    if np.isinf(y).any():raise ValueError('STOP infinite labels')
    if len(np.unique(days))!=len(days) or np.any(days[1:]<=days[:-1]):raise ValueError('STOP full session axis')
    for v in predictions.values():
        if np.asarray(v).shape!=(n,) or not np.isfinite(v).all():raise ValueError('STOP prediction finite support')
    daily={a:np.full(len(days),np.nan) for a in 'AB'};reason=np.full(len(days),'NO_COMMON_MATURE_SUPPORT',dtype=object)
    for d in range(len(days)):
        rows=np.flatnonzero((t==d)&np.isfinite(y))
        if len(rows)<20:reason[d]='FEWER_THAN_20';continue
        vals={}
        for arm in 'AB':
            seed_ranks=[rankdata(np.asarray(predictions[arm,seed])[rows],method='average')/len(rows) for seed in SEEDS]
            vals[arm]=_spearman(np.mean(seed_ranks,axis=0),y[rows])
        if not all(np.isfinite(v) for v in vals.values()):reason[d]='CONSTANT_CROSS_SECTION';continue
        for arm in 'AB':daily[arm][d]=vals[arm]
        reason[d]='VALID'
    periods={}
    year=days.astype('datetime64[Y]').astype(int)+1970
    for name,lo,hi in [('2015-2020',2015,2020),('2021-2026',2021,2026)]:
        mask=(year>=lo)&(year<=hi);a=daily['A'][mask];b=daily['B'][mask];diff=b-a
        if not np.isfinite(diff).any():raise ValueError('STOP no paired period support')
        periods[name]=dict(mean_a=float(np.nanmean(a)),mean_b=float(np.nanmean(b)),mean_difference=float(np.nanmean(diff)),
            paired_ci95=_circular_paired_interval(diff),valid_days=int(np.isfinite(diff).sum()),scheduled_days=int(mask.sum()),
            interpretation='EXPOSED_HISTORICAL_ENGINEERING_ONLY_NO_CONFIRMATORY_SUCCESS')
    return dict(periods=periods,daily=daily,reasons=reason)
