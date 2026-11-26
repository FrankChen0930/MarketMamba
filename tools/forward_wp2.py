"""WP2 offline C0+ origin adapter; RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

No IC, returns, selection or production publication. A supplied calendar is an
arithmetic dependency, never PIT/forward admission. Historical observed-union
calendars are allowed only for explicitly labelled P1 reproduction.
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from tools.forward_wp1 import Calendar, Stop, load_w34
from V6.research_lab import c0plus, matrix_baselines as mb
from V6.research_lab.seq_runner import SeqData
from V6.research_lab.rolling_scores import identity, provenance, rank_average, write_json

ROOT = Path(__file__).resolve().parents[1]/'.artifacts/forward'
SEEDS = (17, 29, 43)
PARAMS = dict(n_estimators=200, num_leaves=31, learning_rate=.05,
              min_child_samples=200, subsample=.8, subsample_freq=1,
              colsample_bytree=.8, reg_lambda=10., n_jobs=4, verbose=-1)
FILES = ('manifest.json','X.npy','dates.npy','end_indices.npy','splits.npy',
         'stock_ids.npy','stock_idx.npy','y5.npy','y10.npy')


def output_path(path):
    out=Path(path).resolve()
    if out==ROOT.resolve() or out.exists() or Path(path).is_symlink():
        raise FileExistsError(out)
    if not out.is_relative_to(ROOT.resolve()):
        raise Stop('output must be below checkout .artifacts/forward/')
    return out


def origin_splits(dates, base_splits, calendar, origin, eval_end):
    """P1-compatible inclusive cutoff origin-21; origin itself is scored.

    WP1 origin_minus21 returns the inclusive last allowed training session,
    consistent with P2's explicit requirement t < origin-20.
    Complete historical calendar alignment is mandatory; never infer holidays.
    """
    origin=calendar.require_session(origin)
    eval_end=calendar.require_session(eval_end)
    boundary=np.datetime64(calendar.origin_minus21(origin))
    dates,base=np.asarray(dates).astype('datetime64[D]'),np.asarray(base_splits)
    if (dates.ndim!=1 or base.shape!=dates.shape or len(dates)==0 or
            np.isnat(dates).any() or not np.isin(base,[0,1,2,3,4]).all() or
            np.any(dates[1:]<dates[:-1]) or eval_end<origin):
        raise Stop('invalid rows/splits/evaluation interval')
    sessions=np.unique(dates)
    # The calendar may extend past the immutable matrix, but may not omit any
    # historical session. Missing individual stocks remain gaps for SeqData.
    expected=np.array([d for d in calendar.sessions if d<=str(sessions[-1])],dtype='datetime64[D]')
    if not np.array_equal(sessions,expected) or np.datetime64(eval_end)>sessions[-1]:
        raise Stop('matrix/calendar session mismatch or unavailable endpoint')
    train=(dates<=boundary)&(base!=0)
    ev=(dates>=np.datetime64(origin))&(dates<=np.datetime64(eval_end))
    if not train.any() or not ev.any():raise Stop('empty fit/evaluation support')
    out=np.full(len(dates),3,dtype=np.uint8)
    out[(dates<np.datetime64(origin))&(base!=0)]=4
    out[(dates<np.datetime64(origin))&(base==0)]=0
    out[train]=1;out[ev]=2
    return out


def prepare_view(source, out, calendar, origin, eval_end):
    """Fresh split-only view. All source arrays remain read-only symlinks."""
    source,out=Path(source).resolve(strict=True),output_path(out)
    if out.exists() or out.is_symlink():raise FileExistsError(out)
    paths={name:(source/name).resolve(strict=True) for name in FILES}
    dates=np.load(paths['dates.npy'],mmap_mode='r',allow_pickle=False)
    base=np.load(paths['splits.npy'],mmap_mode='r',allow_pickle=False)
    splits=origin_splits(dates,base,calendar,origin,eval_end)
    out.mkdir(parents=True,exist_ok=False)
    for name,path in paths.items():
        if name!='splits.npy':(out/name).symlink_to(path)
    with (out/'splits.npy').open('xb') as f:np.save(f,splits,allow_pickle=False)
    write_json(out/'origin.json',dict(origin=origin,eval_end=eval_end,
        train_last_allowed=calendar.origin_minus21(origin),cutoff_inclusive=True,
        calendar=calendar.identity,forward_admitted=False))
    return out


def fit_predict(directory, seed, max_train_rows=500_000):
    """Same C0+ feature/target/sample/fit defaults as ace2eef, no eval labels.

    Unlike c0plus.run, omits the extra day-t model and all evaluation diagnostics.
    y10 is not a fit or prediction mask. W60 applies only to score eligibility.
    """
    from lightgbm import LGBMRegressor
    if seed not in SEEDS or type(max_train_rows) is not int or max_train_rows<1:
        raise Stop('unregistered seed or invalid sample cap')
    if not (Path(directory)/'stock_idx.npy').is_file():raise Stop('read-only stock index required')
    data=SeqData(directory,normalize=None)
    ctx=mb.Context(directory,'y5',max_train_rows,seed,'rank',row_mask=data.eligible)
    if len(ctx.fit_idx)==0 or len(ctx.eval_idx)==0:raise Stop('empty support')
    xf,names=c0plus.build_features(data,ctx.fit_idx,data.feature_names)
    model=LGBMRegressor(**PARAMS,random_state=seed).fit(xf,ctx.y_fit)
    xe=c0plus.build_features(data,ctx.eval_idx,data.feature_names)[0]
    scores=model.predict(xe)
    if not np.isfinite(scores).all():raise Stop('nonfinite scores')
    dates=np.load(Path(directory)/'dates.npy',mmap_mode='r',allow_pickle=False)
    return dict(train_idx=np.flatnonzero((ctx.splits==1)&np.isfinite(ctx.y)),
        fit_idx=ctx.fit_idx,eval_idx=ctx.eval_idx,target=ctx.y_fit,
        scores=scores,dates=np.asarray(dates[ctx.eval_idx]),feature_order=names,
        fit_probe_scores=model.predict(xf[:32]),model_text=model.booster_.model_to_string())


def save_replay(directory,result,out,seed):
    """Store original ordered row indices AND date/code keys, not hash-only claims."""
    out=output_path(out);out.mkdir(parents=True,exist_ok=False)
    ids=np.load(Path(directory)/'stock_ids.npy',mmap_mode='r',allow_pickle=False)
    dates=np.load(Path(directory)/'dates.npy',mmap_mode='r',allow_pickle=False)
    records={}
    for name,key in [('train','train_idx'),('sample','fit_idx')]:
        idx=result[key];p=out/f'{name}.npy'
        with p.open('xb') as f:np.save(f,idx,allow_pickle=False)
        records[name]=identity(p)
        p=out/f'{name}-keys.parquet'
        pd.DataFrame(dict(row=idx,date=dates[idx],code=ids[idx])).to_parquet(p,index=False)
        records[name+'_keys']=identity(p)
    records['seed']=seed
    return records


def replay_keys(directory,records,seed,max_train_rows=500_000):
    if records['seed']!=seed:raise Stop('replay seed mismatch')
    for key in ('train','sample','train_keys','sample_keys'):
        if identity(records[key]['path'])!=records[key]:raise Stop('replay hash mismatch')
    y=np.load(Path(directory)/'y5.npy',mmap_mode='r',allow_pickle=False)
    splits=np.load(Path(directory)/'splits.npy',mmap_mode='r',allow_pickle=False)
    train=np.flatnonzero((splits==1)&np.isfinite(y))
    sample=mb._sample_train(train,max_train_rows,seed)
    ids=np.load(Path(directory)/'stock_ids.npy',mmap_mode='r',allow_pickle=False)
    dates=np.load(Path(directory)/'dates.npy',mmap_mode='r',allow_pickle=False)
    for key,idx in [('train',train),('sample',sample)]:
        if not np.array_equal(idx,np.load(records[key]['path'],allow_pickle=False)):
            raise Stop('replayed indices differ')
        frame=pd.read_parquet(records[key+'_keys']['path'])
        if (not np.array_equal(frame.row,idx) or not np.array_equal(frame.code,ids[idx]) or
                not np.array_equal(frame.date.to_numpy().astype('datetime64[D]'),dates[idx].astype('datetime64[D]'))):
            raise Stop('replayed business keys differ')
    return True


def same_keys(actual, reference, keys):
    """Compare ordered business-key values across lossless timestamp units.

    pandas.equals also compares dtype, while Parquet can change seconds to
    milliseconds. Score and rank comparisons remain raw-byte comparisons.
    """
    return len(actual)==len(reference) and all(
        np.array_equal(actual[key].to_numpy(),reference[key].to_numpy())
        for key in keys)


def run(source,out,calendar,origin,eval_end,reference=None,year=None):
    out=output_path(out);source=Path(source)
    inputs={name:identity(source/name) for name in FILES}
    prov=provenance();started=time.perf_counter()
    adapter_inputs={str(p):identity(p) for p in [Path(__file__),Path('tools/forward_wp1.py')]}
    out.mkdir(parents=True,exist_ok=False)
    view=prepare_view(source,out/'matrix-view',calendar,origin,eval_end)
    comparisons=[];frames=[];replays={};reference_inputs={}
    if reference:
        reference=Path(reference);refmanifest=json.loads((reference/'manifest.json').read_text())
        reference_inputs['manifest']=identity(reference/'manifest.json')
        if refmanifest['provenance']['source_commit']!='ace2eef0c580d3484a4c6299de16b1c3d5420898':
            raise Stop('wrong P1 reference commit')
        for name in FILES:
            matches=[v['sha256'] for k,v in refmanifest['inputs'].items() if f'/fold_{year}/' in k and k.endswith('/'+name)]
            if matches!=[inputs[name]['sha256']]:raise Stop('P1 matrix identity mismatch: '+name)
        oldsplit=np.load(source/'splits.npy',mmap_mode='r')
        if not np.array_equal(oldsplit,np.load(view/'splits.npy')):raise Stop('P1 split reproduction failed')
    for seed in SEEDS:
        tick=time.perf_counter();r=fit_predict(view,seed)
        replay=save_replay(view,r,out/f'seed-{seed}',seed)
        replay_keys(view,replay,seed);replays[str(seed)]=replay
        ids=np.load(view/'stock_ids.npy',mmap_mode='r',allow_pickle=False)
        frame=pd.DataFrame(dict(code=ids[r['eval_idx']],date=pd.to_datetime(r['dates']),
            score=r['scores'],fold=year or int(origin[:4]),seed=seed,system='c0plus'))
        frame.to_parquet(out/f'scores-{seed}.parquet',index=False)
        (out/f'seed-{seed}'/'model.txt').write_text(r['model_text'])
        check=dict(seed=seed,fit_rows=len(r['fit_idx']),train_rows=len(r['train_idx']),
            eval_rows=len(frame),features=len(r['feature_order']),wall_seconds=time.perf_counter()-tick)
        if reference:
            name=f'fold_{year}_seed_{seed}.parquet';p=reference/name
            reference_inputs[name]=identity(p)
            if reference_inputs[name]['sha256']!=refmanifest['outputs'][name]['sha256']:raise Stop('reference SHA mismatch')
            ref=pd.read_parquet(p);ref=ref[ref.system=='c0plus'].reset_index(drop=True)
            keys=['fold','seed','system','date','code']
            check['keys_equal']=same_keys(frame,ref,keys)
            check['scores_bitwise_equal']=frame.score.to_numpy().tobytes()==ref.score.to_numpy().tobytes()
            check['mismatched_scores']=int(np.count_nonzero(frame.score.to_numpy().view('uint64')!=ref.score.to_numpy().view('uint64'))) if len(frame)==len(ref) else None
            check['seed_ranks_bitwise_equal']=frame.groupby('date').score.rank(pct=True).to_numpy().tobytes()==ref.groupby('date').score.rank(pct=True).to_numpy().tobytes()
        frames.append(frame);comparisons.append(check);print(json.dumps(check),flush=True)
    ranks=rank_average(pd.concat(frames,ignore_index=True))
    ranks.to_parquet(out/'rank_average.parquet',index=False)
    ranks_equal=None
    if reference:
        p=reference/'rank_average.parquet';reference_inputs[p.name]=identity(p)
        if reference_inputs[p.name]['sha256']!=refmanifest['outputs'][p.name]['sha256']:raise Stop('rank reference SHA mismatch')
        old=pd.read_parquet(p);old=old[(old.fold==year)&(old.system=='c0plus')].reset_index(drop=True)
        ranks_equal=(same_keys(ranks,old,['fold','system','date','code']) and ranks.score.to_numpy().tobytes()==old.score.to_numpy().tobytes())
    if inputs!={name:identity(source/name) for name in FILES}:raise Stop('source mutated during fit')
    if prov['implementation']!=provenance()['implementation'] or adapter_inputs!={p:identity(p) for p in adapter_inputs}:
        raise Stop('source code mutated during fit')
    outputs={str(p.relative_to(out)):identity(p) for p in sorted(out.rglob('*')) if p.is_file() and not p.is_symlink()}
    passed=all(c.get('keys_equal') and c.get('scores_bitwise_equal') and c.get('seed_ranks_bitwise_equal') for c in comparisons) and ranks_equal is True
    manifest=dict(schema='forward-wp2-v1',artifact_class='GENERATED',evidence_class='CPU_ENGINEERING_ONLY / EXPOSED_HISTORICAL_REPRODUCTION',
        status='PASS_BITWISE' if reference and passed else ('STOP_BITWISE_MISMATCH' if reference else 'ENGINEERING_ONLY_NOT_ACCEPTED'),
        forward_admitted=False,provenance=prov,inputs=inputs,reference_inputs=reference_inputs,
        implementation=adapter_inputs,
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
        contracts={str(p):identity(p) for p in [Path('tasks/spec-forward-p2-primary-20261124.md'),Path('tasks/spec-portfolio-eval-20261119.md')]},
        calendar=calendar.identity,origin=origin,eval_end=eval_end,train_last_allowed=calendar.origin_minus21(origin),
        cutoff_inclusive=True,parameters=dict(**PARAMS,seeds=list(SEEDS),max_train_rows=500000,fit_target='full daily centred rank then sample',eligibility_window=60),
        replay=replays,jobs=comparisons,rank_average_bitwise_equal=ranks_equal,outputs=outputs,wall_seconds=time.perf_counter()-started)
    write_json(out/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],rank_average_bitwise_equal=ranks_equal,wall_seconds=manifest['wall_seconds'])),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--origin',required=True);p.add_argument('--eval-end',required=True)
    p.add_argument('--calendar',type=Path);p.add_argument('--historical-p1-reference',type=Path)
    p.add_argument('--year',type=int)
    a=p.parse_args()
    if a.historical_p1_reference:
        if not a.year or a.calendar:p.error('historical reproduction requires year and observed reference calendar only')
        dates=np.load(a.source/'dates.npy',mmap_mode='r',allow_pickle=False)
        days=np.unique(dates.astype('datetime64[D]'))
        cal=Calendar([str(d) for d in days],str(days[0]),str(days[-1]),
            dict(kind='P1_OBSERVED_UNION_REPRODUCTION_ONLY',forward_admitted=False,dates=identity(a.source/'dates.npy')))
    else:
        if not a.calendar:p.error('explicit WP1 calendar required; no inferred sessions')
        cal=load_w34(a.calendar)
    m=run(a.source,a.out,cal,a.origin,a.eval_end,a.historical_p1_reference,a.year)
    if m['status']=='STOP_BITWISE_MISMATCH':raise SystemExit(2)


if __name__=='__main__':main()
