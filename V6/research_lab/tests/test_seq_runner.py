import json
import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip('torch')
from torch import nn
from research_lab import seq_runner as sr

FEATURES = ['f%d' % i for i in range(12)]
FEATURES[2], FEATURES[3], FEATURES[4] = 'Return_5d', 'Return_20d', 'Volatility_20d'


class CausalConvMixer(nn.Module):
    def __init__(self, d, k=4):
        super().__init__()
        self.k, self.conv = k, nn.Conv1d(d, d, k, groups=d)

    def forward(self, x):
        return self.conv(nn.functional.pad(x.transpose(1, 2), (self.k-1, 0))).transpose(1, 2)


def build(tmp, days=200, names=40, train_end=120, eval_start=130, gap=None, seed=0, signal=2.5):
    rng = np.random.default_rng(seed)
    rows_day, rows_stock = [], []
    for d in range(days):
        for s in range(names):
            if gap and s == gap[0] and gap[1] <= d < gap[2]:
                continue
            rows_day.append(d); rows_stock.append(s)
    day, stock = np.array(rows_day), np.array(rows_stock)
    n = len(day)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    y5 = (signal*X[:, 7] + rng.normal(size=n)).astype(np.float32)          # signal in the last session's feature
    y10 = (signal*X[:, 7] + rng.normal(size=n)).astype(np.float32)
    splits = np.where(day <= train_end, 1, np.where(day >= eval_start, 2, 4)).astype(np.uint8)
    ends = np.cumsum(np.bincount(day, minlength=days)).astype(np.int64)
    arrays = dict(X=X, y5=y5, y10=y10, splits=splits, end_indices=ends,
                  dates=np.datetime64('2020-01-01') + day.astype('timedelta64[D]'),
                  stock_ids=np.array([f'{1000+s}' for s in stock], dtype='U32'))
    for k, v in arrays.items():
        np.save(tmp/f'{k}.npy', v)
    (tmp/'manifest.json').write_text(json.dumps(dict(rows=n, feature_order=FEATURES, schema_version='t', files={})))
    return tmp, day, stock


CFG = dict(sr.DEFAULT_CFG, d_model=16, n_layers=1, mixer_kw={}, attn_window=8, lr=3e-3, epochs=3, steps_per_epoch=25, patience=3,
           dates_per_batch=6, stocks_per_date=32, min_names=20, baseline_rows=10_000, n_boot=50, val_dates=15)


def test_windows_gaps_and_alignment(tmp_path):
    d, day, stock = build(tmp_path, gap=(5, 100, 110))
    data = sr.SeqData(d)
    assert not data.eligible[(day < 59)].any()                                    # fewer than 60 sessions of history
    gap_covered = (stock == 5) & (day >= 110) & (day < 110+59)
    assert not data.eligible[gap_covered].any() and data.eligible[(stock == 5) & (day >= 110+59)].all()
    assert data.eligible[(stock == 6) & (day >= 59)].all()
    r = np.flatnonzero((stock == 3) & (day == 100))[:1]
    w = data.windows(r)[0]
    expect = np.clip(data.X[np.flatnonzero((stock == 3) & (day >= 41) & (day <= 100))], -10, 10)
    assert w.shape == (60, 12) and np.allclose(w, expect)
    with pytest.raises(AssertionError):
        data.windows(np.flatnonzero((stock == 5) & (day == 112)))


def test_split_rows_disjoint_embargoed_eligible(tmp_path):
    d, day, stock = build(tmp_path)
    data = sr.SeqData(d)
    tr, va, ev = data.split_rows(15)
    assert not (set(tr) & set(va)) and not (set(va) & set(ev)) and not (set(tr) & set(ev))
    assert data.splits[ev].min() == 2 and data.splits[tr].max() == 1 and data.splits[va].max() == 1
    assert day[va].min() - day[tr].max() >= 20                                       # embargo >= label horizon
    assert data.eligible[tr].all() and data.eligible[va].all() and data.eligible[ev].all()


def test_centred_rank_mean_zero_per_session():
    seg = np.repeat(np.arange(4), 10)
    r = sr.centred_rank(np.random.default_rng(0).normal(size=40), seg)
    assert np.allclose(pd.Series(r).groupby(seg).mean(), 0, atol=1e-6)


def test_training_learns_planted_signal_resumes_and_skips(tmp_path):
    d, day, stock = build(tmp_path)
    data = sr.SeqData(d)
    tr, va, ev = data.split_rows(15)
    logs = []
    m, st = sr.train_one('mamba2', 1, data, tr, va, CFG, 'cpu', CausalConvMixer, {}, tmp_path/'c.pt', logs.append)
    assert st['best'] > .2 and len(st['history']) == 3 and (tmp_path/'c.pt').exists()
    ic = sr.daily_ic_rows(data, ev, sr.predict(m, data, ev, 'cpu')[:, 0], 'y5')
    assert ic.mean() > .2
    m2, st2 = sr.train_one('mamba2', 1, data, tr, va, CFG, 'cpu', CausalConvMixer, {}, tmp_path/'c.pt', logs.append)
    assert any('[resume]' in x for x in logs) and st2['epoch'] == 3 and len(st2['history']) == 3     # nothing retrained


def test_run_experiment_same_rows_for_baselines_and_resume(tmp_path):
    d, day, stock = build(tmp_path/'m') if (tmp_path/'m').mkdir() is None else None
    logs = []
    r = sr.run_experiment(d, tmp_path/'out', 'EX', kinds=('mamba2', 'samba'), seeds=(1,), cfg=CFG, device='cpu',
                          mixer_factory=CausalConvMixer, log=logs.append)
    assert set(r['models']) >= {'mamba2_s1', 'samba_s1', 'lightgbm', 'ridge', 'reversal_ret5'}
    assert r['models']['lightgbm']['days'] == r['models']['mamba2_s1']['days'] == r['common_sessions']
    assert {x['model'] for x in r['vs_lightgbm']} == {'mamba2_s1', 'samba_s1', 'ridge', 'reversal_ret5'}
    assert r['models']['mamba2_s1']['ic_mean'] > .15
    r2 = sr.run_experiment(d, tmp_path/'out', 'EX', kinds=('mamba2', 'samba'), seeds=(1,), cfg=CFG, device='cpu',
                           mixer_factory=CausalConvMixer, log=logs.append)
    assert r2 == r and any('[skip]' in x for x in logs)


def test_adding_seeds_later_reuses_finished_training(tmp_path):
    (tmp_path/'m').mkdir()
    d, _, _ = build(tmp_path/'m')
    logs = []
    cfg = dict(CFG, val_max_rows=300)
    r1 = sr.run_experiment(d, tmp_path/'out', 'EX', kinds=('mamba2',), seeds=(1,), cfg=cfg, device='cpu', mixer_factory=CausalConvMixer, log=logs.append)
    n_epochs = sum('epoch' in x and 'val_ic5' in x for x in logs)
    r2 = sr.run_experiment(d, tmp_path/'out', 'EX', kinds=('mamba2',), seeds=(1, 2), cfg=cfg, device='cpu', mixer_factory=CausalConvMixer, log=logs.append)
    assert len(list((tmp_path/'out').iterdir())) == 1                          # same training config -> same directory
    assert sum('epoch' in x and 'val_ic5' in x for x in logs) == n_epochs + 3  # only seed 2 trained (3 epochs)
    assert any('scores exist' in x for x in logs)
    assert r2['models']['mamba2_s1']['ic_mean'] == r1['models']['mamba2_s1']['ic_mean'] and 'mamba2_s2' in r2['models']


def test_diagnose_validation_fits_only_on_purged_training_rows(tmp_path):
    d, day, stock = build(tmp_path)
    r = sr.diagnose_validation(d, dict(CFG, val_max_rows=0), models=('reversal_ret5', 'ridge'))
    v = r['_validation']
    assert r['ridge']['ic_mean'] > .2                                   # planted signal, fit on training rows only
    data = sr.SeqData(d)
    tr, va, _ = data.split_rows(15)
    assert v['rows'] == len(va) and v['first'] > v['fit_last']          # validation strictly after the last fit session
    assert abs(r['reversal_ret5']['ic_mean']) < .15


def test_training_history_records_in_sample_probe_ic(tmp_path):
    d, day, stock = build(tmp_path)
    data = sr.SeqData(d)
    tr, va, ev = data.split_rows(15)
    m, st = sr.train_one('mamba2', 1, data, tr, va, dict(CFG, train_probe_rows=500), 'cpu', CausalConvMixer, {}, None, lambda *_: None)
    assert all(h['train_probe_ic5'] is not None for h in st['history']) and st['history'][-1]['train_probe_ic5'] > .2


def test_mlp_control_uses_only_last_session_and_learns_last_day_signal(tmp_path):
    d, day, stock = build(tmp_path)
    h = sr.control_mlp(d, dict(CFG, epochs=3, steps_per_epoch=30, d_model=16, val_max_rows=0), device='cpu', log=lambda *_: None)
    assert len(h) == 3 and h[-1]['val_ic5'] > .2
    from research_lab.seq_models import SeqRanker
    m = SeqRanker('mlp', 12, 16, 1).eval()
    x = torch.randn(4, 1, 12)
    assert m(x)['score5'].shape == (4,)
    x2 = torch.randn(4, 5, 12); x2[:, -1] = x[:, 0]
    assert torch.allclose(m(x2)['score5'], m(x)['score5'], atol=1e-6)       # earlier sessions are ignored


def test_feature_report_flags_day_constant_columns(tmp_path):
    d, day, stock = build(tmp_path)
    X = np.load(d/'X.npy'); X[:, 9] = day.astype(np.float32)/50.            # a macro-like column: same for every stock on a day
    np.save(d/'X.npy', X)
    r = sr.feature_report(d, sessions=40)
    assert r['f9']['day_constant'] and not r['f0']['day_constant'] and r['f9']['session_mean_std'] > .5
