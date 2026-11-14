"""Read-only W25 diagnostics. Reuses W08 ranks and E1 circular-block sampler.

No model fit, metric redefinition, sample-selection policy or research admission.
"""
import numpy as np
from V6.experimental.v7_e1_evaluate import block_bootstrap_lower
from V6.experimental.v7_research_evaluation import daily_metric

def describe(values):
    v=np.asarray(values,dtype=float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('FINITE_NONEMPTY_SERIES_REQUIRED')
    out={'n':len(v),'mean':float(v.mean()),'median':float(np.median(v)),
         'sample_std':float(v.std(ddof=1)) if len(v)>1 else None,
         'positive_fraction':float((v>0).mean()),'negative_fraction':float((v<0).mean()),'zero_fraction':float((v==0).mean()),
         'quantiles':dict(zip(['q10','q25','q75','q90'],map(float,np.quantile(v,[.1,.25,.75,.9])))),
         'lag_correlation':{}}
    for lag in (1,5,10):
        out['lag_correlation'][str(lag)]=float(np.corrcoef(v[:-lag],v[lag:])[0,1]) if len(v)>lag+1 and np.std(v[:-lag])>0 and np.std(v[lag:])>0 else None
    return out

def circular_bounds(values,calendar_positions,block,repeats,seed):
    v=np.asarray(values,dtype=float);p=np.asarray(calendar_positions)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all() or p.shape!=v.shape or p.dtype.kind not in 'iu' or not np.all(np.diff(p)==1):raise ValueError('FINITE_CONTIGUOUS_MARKET_SERIES_REQUIRED_NO_GAP_COMPRESSION')
    if type(block)is not int or not 1<=block<=len(v) or type(repeats)is not int or repeats<=0:raise ValueError('INVALID_BOOTSTRAP_CONFIGURATION')
    return {'lower':block_bootstrap_lower(v,block,repeats,seed),'upper':-block_bootstrap_lower(-v,block,repeats,seed),
            'block':block,'replicates':repeats,'seed':seed,'interval':'EXPLORATORY_CIRCULAR_BLOCK_PERCENTILES_2.5_97.5_NOT_FORMAL_CI'}

def leave_one(l0,ridge,security,day,horizon,policy):
    a=[r for r in l0 if r['stock_id']!=security];b=[r for r in ridge if r['stock_id']!=security]
    ka={(r['stock_id'],r['label_id'],r['label_metadata_sha256'],r['label']) for r in a};kb={(r['stock_id'],r['label_id'],r['label_metadata_sha256'],r['label']) for r in b}
    if ka!=kb or len(ka)!=len(a) or len(kb)!=len(b):raise ValueError('PAIRED_SUPPORT_OR_DUPLICATE_MISMATCH')
    ma=daily_metric(a,day,horizon,policy);mb=daily_metric(b,day,horizon,policy)
    assert ma['support_sha256']==mb['support_sha256']
    delta=mb['rank_ic']-ma['rank_ic'] if ma['rank_ic'] is not None and mb['rank_ic'] is not None else None
    return {'L0':ma,'Ridge':mb,'delta':delta}
