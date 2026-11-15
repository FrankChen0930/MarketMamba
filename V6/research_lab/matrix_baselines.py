"""Fair tabular baselines on a V7 matrix directory (X/y5/y10/dates/splits/end_indices/manifest).

Same X, same label, same split for every model. Fit uses split==1 (train) rows with finite
labels only; evaluation uses split==2 rows. Per-date Spearman IC is NOT comparable with
sequence-model IC from another label/split/universe. RESEARCH_ONLY.
Usage: python -m research_lab.matrix_baselines --matrix DIR --label y5 --out result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .significance import compare_models

TRAIN, EVAL = 1, 2
SPLIT_NAMES = {0: 'warmup', 1: 'train', 2: 'research_evaluation', 3: 'out_of_contract', 4: 'purge'}


def load_matrix(directory, labels=('y5', 'y10')):
    d = Path(directory)
    manifest = json.loads((d/'manifest.json').read_text())
    arr = {k: np.load(d/f'{k}.npy', mmap_mode='r') for k in ('X', *labels, 'dates', 'splits', 'end_indices')}
    n = manifest['rows']
    for k, v in arr.items():
        if k != 'end_indices' and len(v) != n:
            raise ValueError(f'{k} length {len(v)} != manifest rows {n}')
    ends = np.asarray(arr['end_indices'])
    if ends[-1] != n or (np.diff(ends) <= 0).any():
        raise ValueError('end_indices inconsistent with rows')
    return manifest, arr


def daily_rank_ic(score, y, ends, mask, min_names=10, segments=None):
    """Spearman IC per date segment on rows where mask & finite; returns Series index=segment id."""
    out = {}
    starts = np.concatenate(([0], ends[:-1]))
    for i in (range(len(ends)) if segments is None else segments):
        sl = slice(int(starts[i]), int(ends[i]))
        m = mask[sl] & np.isfinite(y[sl]) & np.isfinite(score[sl])
        if m.sum() >= min_names:
            a = pd.Series(score[sl][m]).rank().to_numpy()
            b = pd.Series(y[sl][m]).rank().to_numpy()
            c = np.corrcoef(a, b)[0, 1]
            out[i] = c if np.isfinite(c) else np.nan
    return pd.Series(out, dtype=float)


def _sample_train(idx, max_rows, seed):
    if len(idx) <= max_rows:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, max_rows, replace=False))


RULES = ('reversal_ret5', 'momentum_ret20', 'lowvol_20d')
DEFAULT_PARAMS = dict(ridge_alpha=100., lgbm_estimators=200, lgbm_leaves=31, lgbm_lr=.05,
                      lgbm_min_child=200, lgbm_reg_lambda=10., lgbm_threads=2)


class Context:
    """Prepared arrays for one (matrix, label). X stays memory-mapped; only fit/eval rows are copied."""

    def __init__(self, directory, label='y5', max_train_rows=500_000, seed=17, fit_target='rank', row_mask=None, eval_split=EVAL, fit_mask=None):
        if fit_target not in ('raw', 'demean', 'rank'):
            raise ValueError('fit_target must be raw, demean or rank')
        self.directory, self.label, self.seed = Path(directory), label, seed
        self.manifest, a = load_matrix(directory, labels=(label,))
        self.y = np.asarray(a[label], dtype=np.float32)
        self.splits = np.asarray(a['splits'])
        self.ends = np.asarray(a['end_indices'])
        self.horizon = int(label[1:])
        self.names = self.manifest['feature_order']
        self.col = {n: i for i, n in enumerate(self.names)}
        fit_ok = (self.splits == TRAIN) if fit_mask is None else np.asarray(fit_mask, dtype=bool)   # fit rows may be restricted (validation diagnostics)
        train_idx = np.flatnonzero(fit_ok & np.isfinite(self.y))
        self.train_rows_available = int(len(train_idx))
        self.fit_target = fit_target
        self.fit_idx = _sample_train(train_idx, max_train_rows, seed)
        self.y_fit = self._fit_labels(train_idx)
        self.eval_mask = self.splits == eval_split
        if row_mask is not None:      # evaluate on a common row subset (e.g. rows a sequence model can score)
            self.eval_mask = self.eval_mask & np.asarray(row_mask, dtype=bool)
        self.eval_idx = np.flatnonzero(self.eval_mask)
        self.X_fit = np.asarray(a['X'][self.fit_idx], dtype=np.float32)
        self.X_eval = np.asarray(a['X'][self.eval_idx], dtype=np.float32)
        seg_dates = pd.to_datetime(np.asarray(a['dates'])[self.ends-1])
        self.seg_dates = seg_dates
        # Only segments that contain evaluation rows are scored.
        starts = np.concatenate(([0], self.ends[:-1]))
        self.eval_segments = [i for i in range(len(self.ends)) if self.eval_mask[starts[i]:self.ends[i]].any()]

    def _fit_labels(self, train_idx):
        """Per-session transform of the fit label, computed on the FULL train cross-section then sampled.
        The matrix label is a plain forward return (not market-relative), so 'rank'/'demean' remove
        the common daily component that a pooled regression would otherwise chase."""
        y = self.y[self.fit_idx]
        if self.fit_target == 'raw':
            return y
        seg = np.searchsorted(self.ends, train_idx, side='right')
        s = pd.Series(self.y[train_idx])
        g = s.groupby(seg)
        t = (g.rank(pct=True)-(1+1/g.transform('count'))/2) if self.fit_target == 'rank' else (s-g.transform('mean'))
        pos = np.searchsorted(train_idx, self.fit_idx)
        return t.to_numpy(dtype=np.float32)[pos]

    def memory_gb(self):
        return (self.X_fit.nbytes + self.X_eval.nbytes + self.y.nbytes + self.splits.nbytes) / 1e9


def compute_scores(ctx, model, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    col, X = ctx.col, ctx.X_eval
    if model == 'reversal_ret5':
        return -X[:, col['Return_5d']]
    if model == 'momentum_ret20':
        return X[:, col['Return_20d']]
    if model == 'lowvol_20d':
        return -X[:, col['Volatility_20d']]
    if model == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge(alpha=p['ridge_alpha']).fit(ctx.X_fit, ctx.y_fit).predict(X)
    if model == 'lightgbm':
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(n_estimators=p['lgbm_estimators'], num_leaves=p['lgbm_leaves'], learning_rate=p['lgbm_lr'],
                          min_child_samples=p['lgbm_min_child'], subsample=.8, subsample_freq=1,
                          colsample_bytree=.8, reg_lambda=p['lgbm_reg_lambda'], random_state=ctx.seed,
                          n_jobs=p['lgbm_threads'], verbose=-1)
        return m.fit(ctx.X_fit, ctx.y_fit).predict(X)
    raise ValueError(f'unknown model {model}')


def model_daily_ic(ctx, scores):
    buf = np.full(len(ctx.y), np.nan, dtype=np.float32)
    buf[ctx.eval_idx] = scores
    ic = daily_rank_ic(buf, ctx.y, ctx.ends, ctx.eval_mask, segments=ctx.eval_segments)
    ic.index = ctx.seg_dates[ic.index]
    return ic


def summarize_daily(ic):
    return dict(ic_mean=float(ic.mean()), ic_std=float(ic.std(ddof=1)), days=int(ic.notna().sum()),
                icir=float(ic.mean()/ic.std(ddof=1)) if ic.std(ddof=1) > 0 else None,
                first=str(ic.dropna().index.min().date()), last=str(ic.dropna().index.max().date()))


def run(directory, label='y5', max_train_rows=500_000, seed=17, models=('ridge', 'lightgbm'), n_boot=1000, params=None, fit_target='rank'):
    ctx = Context(directory, label, max_train_rows, seed, fit_target)
    daily = {m: model_daily_ic(ctx, compute_scores(ctx, m, params)) for m in (*RULES, *models)}
    full = {k: summarize_daily(v) for k, v in daily.items()}
    common = pd.concat(daily, axis=1).dropna().index
    daily = {k: v.loc[common] for k, v in daily.items()}
    sig = compare_models(daily, 'reversal_ret5', ctx.horizon, n_boot=n_boot, seed=seed) if len(common) >= 20 else []
    mf = ctx.manifest
    return dict(matrix=str(directory), manifest_schema=mf.get('schema_version'),
                manifest_logical_identity=mf.get('logical_identity'),
                x_sha256_manifest=mf.get('files', {}).get('X.npy'),
                label=label, horizon=ctx.horizon, label_evidence=mf.get('label_contract', {}).get('evidence_class'),
                split_contract=mf.get('split_contract'),
                fit_rows=int(len(ctx.fit_idx)), train_rows_available=ctx.train_rows_available,
                eval_rows=int(len(ctx.eval_idx)), eval_sessions=int(len(common)), models=full,
                vs_reversal_ret5=sig, seed=seed,
                caveats=['Same X/label/split for all models; IC not comparable to other matrices/labels',
                         'research-used evaluation window (already seen by prior architecture selection)',
                         'tabular, no sequence context, subsampled fit; not Mamba', 'RESEARCH_ONLY'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', required=True, type=Path)
    p.add_argument('--label', choices=['y5', 'y10'], default='y5')
    p.add_argument('--max-train-rows', type=int, default=500_000)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    r = run(args.matrix, args.label, args.max_train_rows)
    args.out.write_text(json.dumps(r, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in r['models'].items()}, indent=1))


if __name__ == '__main__':
    main()
