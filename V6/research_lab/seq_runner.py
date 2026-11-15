"""B8: train/evaluate official Mamba2 / Samba rankers on a V7 matrix with B4's X, label and split. RESEARCH_ONLY.

Same contract as matrix_baselines: fit on split==1 (inner chronological validation = last VAL_DATES train sessions,
purged from fitting), evaluate once on split==2. A sample needs a complete WINDOW-session history of the same stock
(no gaps); baselines are re-scored on exactly the same evaluation rows so IC and paired tests are like-for-like.
Loss: per-session centred-rank MSE + LISTNET_W * ListNet-top1, heads 5d:10d = 1:.5 (V7 style; ListNet temperature is
this module's choice, not a V7 constant). Mixer is the official mamba_ssm.Mamba2 unless a test injects one.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import matrix_baselines as mb
from .seq_models import SeqRanker
from .significance import compare_models

WINDOW, VAL_DATES, LISTNET_W, LISTNET_T = 60, 126, 0.5, 10.


def centred_rank(values, seg):
    """Per-session centred percentile rank (mean exactly 0), computed on the full cross-section given."""
    s = pd.Series(values)
    g = s.groupby(seg)
    return (g.rank(pct=True)-(1+1/g.transform('count'))/2).to_numpy(np.float32)


def macro_columns(X, ends, sessions=60, seed=0):
    """Columns that are identical for every stock within a session (macro series): detected from sampled sessions."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ends), min(sessions, len(ends)), replace=False)
    starts = np.concatenate(([0], ends[:-1]))
    const = np.ones(X.shape[1], bool)
    for d in pick:
        blk = np.asarray(X[starts[d]:ends[d]], dtype=np.float64)
        if len(blk) > 1:
            const &= blk.std(0) < 1e-9
    return np.flatnonzero(const)


def check_input_scale(x, max_sat=.01):
    """Hard guard against the 'raw-scale features clipped to constants' failure: network inputs must already be O(1).
    Fails if >1% of values are at/over the clip bound or any non-zero-variance column has std outside [0.3, 3]."""
    x = np.asarray(x, dtype=np.float64)
    if (np.abs(x) >= 10).mean() > max_sat:
        raise ValueError('network input is not standardised: >1% of values reach the clip bound (use normalize="rankgauss")')
    sd = x.std(0)
    bad = np.flatnonzero((sd > 1e-9) & ((sd < .3) | (sd > 3.)))
    if len(bad) > len(sd)//4:
        raise ValueError(f'network input scale suspicious: {len(bad)} columns with std outside [0.3, 3]')


def _x_tag(d):
    st = (d/'X.npy').stat()
    return f'{st.st_size}-{st.st_mtime_ns}'


def ensure_rankgauss(directory, log=print):
    """Per-session cross-sectional rank-gauss copy of X (cached as X_rankgauss.npy). Uses only same-session values, so it
    is point-in-time. Macro (same-for-every-stock) columns are set to 0; non-finite values -> 0. Needed because X is NOT
    standardised for most columns (Volume std ~8e6, flows up to 1e8, prices raw) although the manifest says it is."""
    from scipy.stats import norm, rankdata
    d = Path(directory)
    out = d/'X_rankgauss.npy'
    tag = d/'X_rankgauss.tag'
    if out.exists() and tag.exists() and tag.read_text() == _x_tag(d):
        return out
    X = np.load(d/'X.npy', mmap_mode='r')
    ends = np.asarray(np.load(d/'end_indices.npy'))
    macro = macro_columns(X, ends)
    log(f'[rankgauss] macro columns zeroed: {macro.tolist()}')
    tmp = d/'X_rankgauss.tmp.npy'
    Y = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=X.shape)
    start = 0
    for e in ends:
        blk = np.nan_to_num(np.asarray(X[start:e], dtype=np.float64), nan=0., posinf=0., neginf=0.)
        n = len(blk)
        r = rankdata(blk, axis=0, method='average') if n > 1 else np.full_like(blk, (n+1)/2)
        z = norm.ppf((r-.5)/n)
        z[:, macro] = 0.
        Y[start:e] = z
        start = e
    Y.flush(); del Y
    tmp.replace(out); tag.write_text(_x_tag(d))
    return out


def stock_index(directory):
    """Integer id per matrix row from stock_ids.npy (U32 strings, ~1 GB): computed once, cached next to the matrix copy."""
    f = Path(directory)/'stock_idx.npy'
    if f.exists():
        return np.load(f)
    _, inv = np.unique(np.load(Path(directory)/'stock_ids.npy', mmap_mode='r'), return_inverse=True)
    inv = inv.astype(np.int32)
    tmp = Path(str(f)+'.tmp.npy')
    np.save(tmp, inv)
    tmp.replace(f)
    return inv


class SeqData:
    def __init__(self, directory, stock_idx=None, labels=('y5', 'y10'), window=WINDOW, normalize='rankgauss'):
        self.window = window
        self.manifest, a = mb.load_matrix(directory, labels=labels)
        self.X = a['X']                                   # raw (feature_report, LightGBM)
        self.Xn = np.load(ensure_rankgauss(directory), mmap_mode='r') if normalize == 'rankgauss' else self.X   # network input
        self.y = {k: np.asarray(a[k], dtype=np.float32) for k in labels}
        self.splits, self.ends = np.asarray(a['splits']), np.asarray(a['end_indices'])
        n = len(self.splits)
        self.date_idx = np.repeat(np.arange(len(self.ends), dtype=np.int32), np.diff(self.ends, prepend=0))
        if stock_idx is None:
            stock_idx = stock_index(directory)
        self.stock_idx = np.asarray(stock_idx, dtype=np.int32)
        self.n_stocks = int(self.stock_idx.max())+1
        self.row_of = np.full((len(self.ends), self.n_stocks), -1, dtype=np.int32)
        self.row_of[self.date_idx, self.stock_idx] = np.arange(n, dtype=np.int32)
        present = (self.row_of >= 0).astype(np.int16).cumsum(axis=0)
        lag = np.vstack([np.zeros((window, self.n_stocks), np.int16), present[:-window]])
        full = (present-lag) == window                                # complete `window` sessions ending at t
        full[:window-1] = False
        self.eligible = full[self.date_idx, self.stock_idx]           # per matrix row
        self.dates = pd.to_datetime(np.asarray(a['dates'])[self.ends-1])
        self.feature_names = self.manifest['feature_order']
        self._scale_ok = False

    def windows(self, rows):
        t, s = self.date_idx[rows], self.stock_idx[rows]
        idx = self.row_of[t[:, None]+np.arange(-self.window+1, 1), s[:, None]]
        assert (idx >= 0).all(), 'incomplete window requested'
        x = np.asarray(self.Xn[idx.reshape(-1)], dtype=np.float32).reshape(len(rows), self.window, -1)
        if not self._scale_ok:
            check_input_scale(x.reshape(-1, x.shape[-1])); self._scale_ok = True
        return np.clip(np.nan_to_num(x, nan=0., posinf=0., neginf=0.), -10., 10.)   # clip is a no-op for rank-gauss inputs

    def split_rows(self, val_dates=VAL_DATES):
        train_dates = np.unique(self.date_idx[(self.splits == mb.TRAIN) & self.eligible])
        val_first = train_dates[-val_dates]
        fit_last = train_dates[-val_dates-20]           # 20-session embargo >= max label horizon
        e = self.eligible & np.isfinite(self.y['y5']) & np.isfinite(self.y['y10'])
        tr = np.flatnonzero((self.splits == mb.TRAIN) & e & (self.date_idx <= fit_last))
        va = np.flatnonzero((self.splits == mb.TRAIN) & e & (self.date_idx >= val_first))
        ev = np.flatnonzero((self.splits == mb.EVAL) & self.eligible)
        return tr, va, ev


def listnet(score, target, temperature=LISTNET_T):
    return -(nn_softmax(target*temperature)*log_softmax(score)).sum(-1).mean()


def nn_softmax(x):
    import torch
    return torch.softmax(x, -1)


def log_softmax(x):
    import torch
    return torch.log_softmax(x, -1)


def session_loss(out, t5, t10, groups, listnet_w=0., listnet_t=LISTNET_T):
    """groups: (D, S) index matrix of one session per row. Per-head loss = MSE on centred ranks (+ listnet_w * ListNet-top1).
    ListNet is OFF by default. Real E5 MLP check (val IC, 3 epochs): temperature 10 -> -0.01..-0.03; T=3 0.080; T=1 0.093; MSE only 0.094-0.101.
    """
    import torch
    total = 0.
    for w, key, tgt in ((1., 'score5', t5), (.5, 'score10', t10)):
        sc, tg = out[key][groups], tgt[groups]
        total = total + w*(torch.mean((sc-tg)**2)+(listnet_w*listnet(sc, tg, listnet_t) if listnet_w else 0.))
    return total/1.5


def sample_batch(rng, rows_by_date, dates, d, s):
    chosen = rng.choice(len(dates), size=min(d, len(dates)), replace=False)
    picks = [rng.choice(rows_by_date[dates[i]], size=s, replace=len(rows_by_date[dates[i]]) < s) for i in chosen]
    return np.stack(picks)


def predict(model, data, rows, device, chunk=2048):
    import torch
    model.eval()
    out = np.zeros((len(rows), 2), np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            x = torch.from_numpy(data.windows(rows[i:i+chunk])).to(device)
            o = model(x)
            out[i:i+chunk] = torch.stack([o['score5'], o['score10']], 1).float().cpu().numpy()
    return out


def daily_ic_rows(data, rows, scores, label):
    """Spearman IC per session over `rows` (scores aligned to rows)."""
    df = pd.DataFrame(dict(d=data.date_idx[rows], s=scores, y=data.y[label][rows]))
    df = df[np.isfinite(df.y) & np.isfinite(df.s)]
    r = df.groupby('d').apply(lambda g: g.s.rank().corr(g.y.rank()) if len(g) >= 10 else np.nan, include_groups=False)
    r.index = data.dates[r.index.to_numpy()]
    return r.dropna()


def diagnose_validation(directory, cfg=None, models=('reversal_ret5', 'momentum_ret20', 'lowvol_20d', 'ridge', 'lightgbm'), stock_idx=None):
    """No training: how do the tabular baselines score on the *validation* rows used for early stopping (fit on the
    purged training rows only)? Tells a hostile validation period apart from a model that fails to learn."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    data = SeqData(directory, stock_idx)
    tr, va, _ = data.split_rows(cfg['val_dates'])
    if cfg['val_max_rows'] and len(va) > cfg['val_max_rows']:
        va = np.sort(np.random.default_rng(0).choice(va, cfg['val_max_rows'], replace=False))
    fit, val = np.zeros(len(data.splits), bool), np.zeros(len(data.splits), bool)
    fit[tr], val[va] = True, True
    ctx = mb.Context(directory, 'y5', cfg['baseline_rows'], 17, 'rank', row_mask=val, eval_split=mb.TRAIN, fit_mask=fit)
    out = {m: mb.summarize_daily(mb.model_daily_ic(ctx, mb.compute_scores(ctx, m))) for m in models}
    out['_validation'] = dict(first=str(data.dates[data.date_idx[va].min()].date()), last=str(data.dates[data.date_idx[va].max()].date()),
                              rows=int(len(va)), fit_rows=int(len(ctx.fit_idx)), fit_last=str(data.dates[data.date_idx[tr].max()].date()))
    return out


def feature_report(directory, sessions=200, seed=0, stock_idx=None):
    """Per-feature scale diagnostics on sampled eligible sessions: cross-sectional std within a session (0 => the column is
    the same for every stock that day, e.g. macro series), spread of session means, zero fraction, tail size."""
    data = SeqData(directory, stock_idx, window=1, normalize=None)
    rng = np.random.default_rng(seed)
    dts = np.unique(data.date_idx[(data.splits == mb.TRAIN)])
    pick = np.sort(rng.choice(dts, min(sessions, len(dts)), replace=False))
    starts = np.concatenate(([0], data.ends[:-1]))
    rows = [np.arange(starts[d], data.ends[d]) for d in pick]
    X = [np.asarray(data.X[r], dtype=np.float64) for r in rows]
    within = np.mean([x.std(0) for x in X], axis=0)
    means = np.stack([x.mean(0) for x in X])
    allx = np.vstack(X)
    out = {}
    for i, n in enumerate(data.feature_names):
        out[n] = dict(within_session_std=float(within[i]), session_mean_std=float(means[:, i].std()), zero_fraction=float((allx[:, i] == 0).mean()),
                      p99_abs=float(np.quantile(np.abs(allx[:, i]), .99)), max_abs=float(np.abs(allx[:, i]).max()),
                      day_constant=bool(within[i] < 1e-9))
    return out


def control_mlp(directory, cfg=None, seed=17, device='cuda', log=print, stock_idx=None):
    """Same data, loss, optimiser and schedule as the sequence models but the network only sees the LAST session
    (window=1, no sequence mixing). If this cannot reach roughly Ridge-level validation IC, the problem is the
    input/loss/optimisation set-up, not sequence modelling."""
    cfg = {**DEFAULT_CFG, 'epochs': 3, 'steps_per_epoch': 300, **(cfg or {})}
    data = SeqData(directory, stock_idx, window=1)
    tr, va, _ = data.split_rows(cfg['val_dates'])
    if cfg['val_max_rows'] and len(va) > cfg['val_max_rows']:
        va = np.sort(np.random.default_rng(0).choice(va, cfg['val_max_rows'], replace=False))
    m, st = train_one('mlp', seed, data, tr, va, cfg, device, None, {}, None, log)
    return st['history']


def train_one(kind, seed, data, tr, va, cfg, device='cuda', mixer_factory=None, mixer_kw=None, ckpt=None, log=print):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SeqRanker(kind, n_features=len(data.feature_names), d_model=cfg['d_model'], n_layers=cfg['n_layers'],
                      mixer_kw=mixer_kw if mixer_kw is not None else cfg['mixer_kw'], window=cfg.get('attn_window', 16),
                      mixer_factory=mixer_factory).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps = cfg['epochs']*cfg['steps_per_epoch']
    warm = max(1, int(.15*steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i+1)/warm if i < warm else .5*(1+math.cos(math.pi*(i-warm)/max(1, steps-warm))))
    t5, t10 = (centred_rank(data.y[k][tr], data.date_idx[tr]) for k in ('y5', 'y10'))
    dtr = data.date_idx[tr]
    cuts = np.flatnonzero(np.diff(dtr))+1                      # tr is sorted by row == sorted by date
    by_date = {int(data.date_idx[g[0]]): g for g in np.split(tr, cuts) if len(g) >= cfg['min_names']}
    tt5, tt10 = torch.from_numpy(t5).to(device), torch.from_numpy(t10).to(device)
    dates = sorted(by_date)
    state = dict(epoch=0, best=-9., bad=0, history=[])
    best_sd = None
    prng = np.random.default_rng(1)       # in-sample probe = whole sampled sessions (>=10 names/session), not scattered rows
    pdays = prng.choice(sorted(by_date), min(len(by_date), cfg.get('train_probe_sessions', 40)), replace=False)
    tr_probe = np.sort(np.concatenate([prng.choice(by_date[d], min(len(by_date[d]), 500), replace=False) for d in pdays]))
    if ckpt and Path(ckpt).exists():
        c = torch.load(ckpt, map_location=device)
        model.load_state_dict(c['model']); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched']); state = c['state']
        best_sd = c.get('best_sd')
        if best_sd is None and state['best'] > -9:
            raise RuntimeError(f'checkpoint {ckpt} has no best weights (written by an older version); delete it and retrain')
        rng = np.random.default_rng(seed+state['epoch']); log(f'[resume] {kind} s{seed} from epoch {state["epoch"]}')
    while state['epoch'] < cfg['epochs'] and state['bad'] < cfg['patience']:
        model.train(); t0 = time.time(); losses = []
        for _ in range(cfg['steps_per_epoch']):
            g = sample_batch(rng, by_date, dates, cfg['dates_per_batch'], cfg['stocks_per_date'])
            flat = g.reshape(-1)
            x = torch.from_numpy(data.windows(flat)).to(device)
            idx = torch.from_numpy(np.searchsorted(tr, g.reshape(-1))).to(device)
            sc = model(x)
            local = torch.arange(flat.size, device=device).view(g.shape)
            loss = session_loss(sc, tt5[idx], tt10[idx], local, cfg.get('listnet_w', 0.), cfg.get('listnet_t', LISTNET_T))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); sched.step()
            losses.append(float(loss.detach()))
        ic = daily_ic_rows(data, va, predict(model, data, va, device)[:, 0], 'y5')
        state['epoch'] += 1
        score = float(ic.mean()) if len(ic) else -9.
        tic = daily_ic_rows(data, tr_probe, predict(model, data, tr_probe, device)[:, 0], 'y5')
        state['history'].append(dict(epoch=state['epoch'], loss=float(np.mean(losses)), val_ic5=score,
                                     train_probe_ic5=float(tic.mean()) if len(tic) else None, seconds=time.time()-t0))
        if score > state['best']:
            state['best'], state['bad'] = score, 0
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            state['bad'] += 1
        log(f'[{kind} s{seed}] epoch {state["epoch"]} loss {np.mean(losses):.4f} val_ic5 {score:+.4f} train_probe_ic5 {state["history"][-1]["train_probe_ic5"] or 0:+.4f} ({time.time()-t0:.0f}s)')
        if ckpt:
            tmp = str(ckpt)+'.tmp'
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=state, best_sd=best_sd), tmp)
            Path(tmp).replace(ckpt)
    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, state


def run_experiment(directory, out_root, expect, kinds=('mamba2', 'samba'), seeds=(17,), cfg=None, device='cuda',
                   mixer_factory=None, stock_idx=None, tag='', log=print):
    """Train each (kind, seed) once on the inner split, score the evaluation rows, compare with baselines on identical rows."""
    import hashlib
    import json
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    # The directory depends only on the training config, so models/seeds added later reuse finished work;
    # each (kinds, seeds) request gets its own result file.
    root = Path(out_root)/f'{expect}_seq_{hashlib.sha256(json.dumps([cfg, tag], sort_keys=True).encode()).hexdigest()[:10]}'
    result_file = root/f'result_{"-".join(kinds)}_s{"-".join(map(str, seeds))}.json'
    if result_file.exists():
        log(f'[skip] complete: {result_file}')
        return json.loads(result_file.read_text())
    (root/'ckpt').mkdir(parents=True, exist_ok=True); (root/'scores').mkdir(exist_ok=True); (root/'daily_ic').mkdir(exist_ok=True)
    data = SeqData(directory, stock_idx)
    tr, va, ev = data.split_rows(cfg['val_dates'])
    if cfg['val_max_rows'] and len(va) > cfg['val_max_rows']:     # early-stopping signal only; fixed subset so epochs are comparable
        va = np.sort(np.random.default_rng(0).choice(va, cfg['val_max_rows'], replace=False))
    rowmask = np.zeros(len(data.splits), bool); rowmask[ev] = True
    ctx = mb.Context(directory, 'y5', cfg['baseline_rows'], 17, 'rank', row_mask=rowmask)
    daily = {m: mb.model_daily_ic(ctx, mb.compute_scores(ctx, m)) for m in ('reversal_ret5', 'ridge', 'lightgbm')}
    models = {}
    for kind in kinds:
        for seed in seeds:
            key = f'{kind}_s{seed}'
            sc_file = root/'scores'/f'{key}.npy'
            if sc_file.exists():
                sc = np.load(sc_file); log(f'[resume] {key} scores exist')
            else:
                m, st = train_one(kind, seed, data, tr, va, cfg, device, mixer_factory, None, root/'ckpt'/f'{key}.pt', log)
                sc = predict(m, data, ev, device); np.save(sc_file, sc)
                models[key] = dict(history=st['history'], best_val_ic5=st['best'])
            ic = daily_ic_rows(data, ev, sc[:, 0], 'y5')
            ic.rename('ic').rename_axis('date').reset_index().to_csv(root/'daily_ic'/f'{key}.csv', index=False)
            daily[key] = ic
            models.setdefault(key, {}).update(mb.summarize_daily(ic))
    for k in ('reversal_ret5', 'ridge', 'lightgbm'):
        models[k] = mb.summarize_daily(daily[k])
    common = pd.concat(daily, axis=1).dropna().index
    aligned = {k: v.loc[common] for k, v in daily.items()}
    sig = compare_models(aligned, 'lightgbm', 5, n_boot=cfg['n_boot'], seed=17) if len(common) >= 20 else []
    res = dict(experiment=expect, config=cfg, kinds=list(kinds), seeds=list(seeds), train_rows=int(len(tr)), val_rows=int(len(va)),
               eval_rows=int(len(ev)), common_sessions=int(len(common)), models=models, vs_lightgbm=sig,
               caveats=['rows limited to complete 60-session windows; baselines re-scored on the same evaluation rows',
                        'evaluation window is research-used; labels are HISTORICAL_SIMULATION_PROXY; universe survivorship-biased',
                        'IC only; not portfolio performance', 'RESEARCH_ONLY'])
    result_file.write_text(json.dumps(res, indent=2, allow_nan=False, default=float))
    return res


DEFAULT_CFG = dict(d_model=64, n_layers=2, mixer_kw=dict(d_state=16, headdim=8, expand=2), attn_window=16, lr=3e-4,
                   weight_decay=.01, epochs=10, steps_per_epoch=400, patience=3, dates_per_batch=16, stocks_per_date=256,
                   min_names=64, baseline_rows=500_000, n_boot=1000, val_dates=VAL_DATES, val_max_rows=80_000, listnet_w=0., listnet_t=LISTNET_T)
