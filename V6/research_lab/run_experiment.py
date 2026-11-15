"""Bounded CPU walk-forward ranking experiment. Invoke from repo with PYTHONPATH=V6."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from .features import BASE_FEATURES
from .splits import walk_forward_splits
from .models import GRURanker, SelectiveRanker, seed_all
from .metrics import daily_metrics, summarize

FEATURES = [f'{c}_cs_z' for c in BASE_FEATURES]
SEEDS = [17,29,43]
DEFAULT_DATA = '/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/research/dataset.parquet'
DEFAULT_OUT = '/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/research/predictions'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_panel(frame, features, window=60):
    """Dense observed-union calendar; missing sessions never compress history."""
    days = pd.DatetimeIndex(frame.date.unique()).sort_values()
    codes = sorted(frame.code.unique())
    index = pd.MultiIndex.from_product([days, codes], names=['date','code'])
    dense = frame.set_index(['date','code']).reindex(index)
    x = dense[features].to_numpy(np.float32).reshape(len(days),len(codes),-1)
    valid = np.isfinite(x).all(-1)
    support = pd.DataFrame(valid).rolling(window,min_periods=window).sum().eq(window).to_numpy()
    # Missing values are never admitted by support, zero only makes storage finite.
    return np.nan_to_num(x, nan=0., posinf=0., neginf=0.), support


def windows(x, ids):
    day, code = ids
    offsets = np.arange(-59,1)
    return torch.from_numpy(x[day[:,None]+offsets, code[:,None]])


def neural_fit_predict(cls, x, train_ids, y, test_ids, seed):
    seed_all(seed)
    model = cls(x.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    day, code = train_ids
    # Inner chronological validation; labels stop >=20 sessions before validation.
    cutoff = day.max()-63
    fit = np.flatnonzero(day < cutoff-20)
    val = np.flatnonzero(day >= cutoff)
    if not len(fit) or not len(val):
        raise ValueError('insufficient inner purged train/validation support')
    rng = np.random.default_rng(seed)
    fit = rng.choice(fit, min(4096,len(fit)), replace=False)
    val = rng.choice(val, min(1024,len(val)), replace=False)
    best, best_loss, losses = None, float('inf'), []
    for epoch in range(2):
        model.train()
        for batch in np.array_split(rng.permutation(fit), max(1,int(np.ceil(len(fit)/256)))):
            pred = model(windows(x,(day[batch],code[batch])))
            loss = torch.nn.functional.mse_loss(pred,torch.from_numpy(y[batch]))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad():
            loss = float(torch.nn.functional.mse_loss(model(windows(x,(day[val],code[val]))),torch.from_numpy(y[val])))
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best = {k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0,len(test_ids[0]),1024):
            ids = tuple(v[start:start+1024] for v in test_ids)
            result.append(model(windows(x,ids)).numpy())
    return np.concatenate(result), dict(validation_mse=losses, best_epoch=int(np.argmin(losses))+1, fit_samples=len(fit), validation_samples=len(val))


def validate_output_paths(output, report_dir):
    output, report_dir = Path(output), Path(report_dir)
    if output.exists() and list(output.glob("*.parquet")):
        raise FileExistsError("prediction directory already populated")
    if (report_dir/"results.json").exists():
        raise FileExistsError("results.json already exists; preserve the completed run")


def run(args):
    started = time.monotonic()
    validate_output_paths(args.out, "V6/research_lab/experiments")
    source_hashes = {str(p):sha(p) for p in Path("V6/research_lab").rglob("*.py")}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = Path(args.out)
    if output.exists() and list(output.glob('*.parquet')):
        raise FileExistsError('prediction directory already populated: refusing to overwrite')
    manifest_path = Path(args.data).with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if sha(args.data) != manifest['output_file_sha256']:
        raise ValueError('dataset bytes do not match manifest')
    frame = pd.read_parquet(args.data).sort_values(['date','code']).reset_index(drop=True)
    frame.code = frame.code.astype(str)
    x, support = sequence_panel(frame,FEATURES)
    days = pd.DatetimeIndex(frame.date.unique()).sort_values()
    codes = sorted(frame.code.unique())
    di = days.get_indexer(frame.date)
    ci = pd.Index(codes).get_indexer(frame.code)
    complete = np.isfinite(frame[FEATURES]).all(axis=1).to_numpy() & support[di,ci]
    folds = walk_forward_splits(frame,min_train_sessions=504,test_sessions=252,horizon=20,purge=20)
    predictions, logs = {}, []
    def save(name,h,idx,values,fold):
        f = frame.loc[idx,['code','date']].copy()
        f['fold'],f['model'],f['horizon'],f['score'] = fold,name,h,np.asarray(values,dtype=float)
        predictions.setdefault((name,h),[]).append(f)
    for fold in folds:
        tick = time.monotonic()
        tr = fold['train'].row_index.to_numpy()
        te = fold['test'].row_index.to_numpy()
        tr,te = tr[complete[tr]], te[complete[te]]
        # Identical point-in-time feature support across all model families.
        tr = tr[np.isfinite(frame.loc[tr,['rank_label_5d','rank_label_20d']]).all(axis=1)]
        log = dict(fold=fold['fold'],train_rows=len(tr),test_rows=len(te),
                   train_start=str(frame.loc[tr,'date'].min()),train_end=str(frame.loc[tr,'date'].max()),
                   test_start=str(frame.loc[te,'date'].min()),test_end=str(frame.loc[te,'date'].max()), neural=[])
        for h in (20,5):
            for name,col,sign in [('momentum20','return_20d',1),('reversal5','return_5d',-1),('lowvol','volatility_20d',-1)]:
                save(name,h,te,sign*frame.loc[te,col],fold['fold'])
            for name,model in [('ridge',Ridge(alpha=100.)),('lightgbm',LGBMRegressor(n_estimators=60,num_leaves=15,max_depth=5,learning_rate=.05,min_child_samples=100,reg_lambda=10,n_jobs=2,random_state=17,verbosity=-1,deterministic=True,force_col_wise=True))]:
                model.fit(frame.loc[tr,FEATURES],frame.loc[tr,f'rank_label_{h}d']-.5)
                save(name,h,te,model.predict(frame.loc[te,FEATURES]),fold['fold'])
        y = frame.loc[tr,['rank_label_20d','rank_label_5d']].to_numpy(np.float32)-.5
        for name,cls in [('gru',GRURanker),('selective',SelectiveRanker)]:
            scores = []
            for seed in SEEDS:
                scores_i, details = neural_fit_predict(cls,x,(di[tr],ci[tr]),y,(di[te],ci[te]),seed)
                log['neural'].append(dict(model=name,seed=seed,**details))
                scores.append(scores_i)
                for j,h in enumerate((20,5)):
                    save(f'{name}_s{seed}',h,te,scores_i[:,j],fold['fold'])
            # Per-date percentile averaging prevents differing seed score scales dominating.
            for j,h in enumerate((20,5)):
                ranks = [pd.Series(s[:,j]).groupby(frame.loc[te,'date'].to_numpy()).rank(pct=True).to_numpy() for s in scores]
                save(name,h,te,np.mean(ranks,axis=0),fold['fold'])
        log['seconds'] = time.monotonic()-tick
        logs.append(log)
        print(json.dumps(log),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    results = dict(evidence_class='RESEARCH_ONLY',status='UNMERGED / NOT_DEPLOYED',
                   git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   dataset=str(Path(args.data).resolve()),dataset_sha256=sha(args.data),manifest_sha256=sha(manifest_path),
                   environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device='cpu',torch_threads=2,marketmamba_data_root=os.getenv('MARKETMAMBA_DATA_ROOT'),data_realpath=str(Path('Data').resolve())),
                   config=dict(features=FEATURES,window=60,seeds=SEEDS,min_train_sessions=504,test_sessions=252,purge=20,epochs=2,max_train_samples=4096,width=12,state=4,lr=.003,ridge_alpha=100,lightgbm_trees=60,lightgbm_leaves=15),folds=logs,models={})
    for (name,h), parts in predictions.items():
        p = pd.concat(parts,ignore_index=True)
        path = output/f'{name}_h{h}.parquet'
        if p.duplicated(['code','date']).any() or not np.isfinite(p.score).all():
            raise ValueError('invalid prediction contract')
        with path.open("xb") as stream:
            p.to_parquet(stream,index=False)
        evaluated = p.merge(frame[['code','date',f'fwd_return_{h}d']],on=['code','date'],validate='one_to_one').rename(columns={f'fwd_return_{h}d':'target'})
        daily = daily_metrics(evaluated, days[(days >= p.date.min()) & (days <= p.date.max())])
        results['models'][f'{name}_h{h}'] = dict(summary=summarize(daily,h),years={str(year):summarize(g,h) for year,g in daily.groupby(daily.date.dt.year)},prediction_path=str(path),prediction_sha256=sha(path),prediction_rows=len(p))
    results['compute_seconds'] = time.monotonic()-started
    results['source_hashes'] = source_hashes
    report_dir = Path('V6/research_lab/experiments')
    report_dir.mkdir(exist_ok=True)
    with (report_dir/'results.json').open('x') as stream:
        stream.write(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',results['compute_seconds'],flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default=DEFAULT_DATA)
    parser.add_argument('--out',default=DEFAULT_OUT)
    run(parser.parse_args())
