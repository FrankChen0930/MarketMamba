import numpy as np
import pytest
import torch
from torch import nn
from research_lab import cross_stock as cs, seq_runner as sr
from test_seq_runner import build, CFG


class Scan(nn.Module):
    """Order-dependent stand-in for Mamba (causal running mean); same (B,L,d)->(B,L,d) contract, CPU only."""
    def __init__(self, d):
        super().__init__()
        self.p = nn.Linear(d, d)

    def forward(self, x):
        return self.p(torch.cumsum(x, 1)/torch.arange(1, x.shape[1]+1, device=x.device).view(1, -1, 1))


def fac(d, **k):
    return Scan(d)


def test_bidirectional_block_matches_s_mamba_formula():
    torch.manual_seed(0)
    b = cs.BiMambaBlock(8, fac, {}, d_ff=16, dropout=0.).eval()
    x = torch.randn(2, 6, 8)
    ref = b.norm1(x + b.fwd(x) + b.rev(x.flip(1)).flip(1))
    ref = b.norm2(ref + b.ffn(ref))
    assert torch.allclose(b(x), ref, atol=1e-6)


def test_cross_stock_mixing_depends_on_other_stocks_and_noncross_does_not():
    torch.manual_seed(0)
    m = cs.CrossStockRanker(5, 8, 1, 1, mixer_factory=fac, dropout=0.).eval()
    x = torch.randn(6, 4, 5)                                   # one session of 6 stocks
    a = m(x, (1, 6))['score5']
    x2 = x.clone(); x2[5] += 3.                                # change the LAST stock
    b = m(x2, (1, 6))['score5']
    assert not torch.allclose(a[:5], b[:5])                    # backward scan: earlier stocks see the later one
    m0 = cs.CrossStockRanker(5, 8, 1, 0, mixer_factory=fac, dropout=0.).eval()    # n_cross=0 = temporal-only ablation (C1)
    assert torch.allclose(m0(x, (1, 6))['score5'][:5], m0(x2, (1, 6))['score5'][:5])


def test_chunking_covers_every_stock_once_per_permutation():
    rng = np.random.default_rng(0)
    parts = cs.chunk_session(1000, 256, rng)
    cat = np.concatenate(parts)
    assert sorted(cat) == list(range(1000)) and len(parts) == 4 and max(map(len, parts))-min(map(len, parts)) <= 1


def test_predict_cross_aligns_scores_to_rows_and_is_deterministic(tmp_path):
    d, day, stock = build(tmp_path)
    data = sr.SeqData(d, window=1, normalize=None)
    tr, va, ev = data.split_rows(15)
    m = cs.CrossStockRanker(data.X.shape[1], 8, 1, 1, mixer_factory=fac, dropout=0.)
    s1 = sr.predict(m, data, ev, 'cpu'); s2 = sr.predict(m, data, ev, 'cpu')
    assert s1.shape == (len(ev), 2) and np.isfinite(s1).all() and np.array_equal(s1, s2)
    sub = ev[::2]
    assert np.isfinite(sr.predict(m, data, sub, 'cpu')).all()


def test_cross_model_trains_through_the_shared_runner(tmp_path):
    d, day, stock = build(tmp_path)
    data = sr.SeqData(d, window=1, normalize=None)
    tr, va, ev = data.split_rows(15)
    cfg = dict(CFG, epochs=2, steps_per_epoch=15, d_model=8, n_layers=1, n_cross=1, dates_per_batch=4, stocks_per_date=32, dropout=0.)
    m, st = sr.train_one('crossstock', 1, data, tr, va, cfg, 'cpu', fac, {}, None, lambda *_: None)
    assert len(st['history']) == 2 and np.isfinite(st['history'][-1]['val_ic5'])


def reference_predict(model, data, rows, device, target, n_perm, seed):
    """Per-job accumulation with no batching: the ground truth for predict_cross."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(rows), 2)); by = {}
    for i, r in enumerate(rows):
        by.setdefault(int(data.date_idx[r]), []).append(i)
    model.eval()
    with torch.no_grad():
        for _ in range(n_perm):
            for pos in by.values():
                pos = np.asarray(pos)
                for piece in cs.chunk_session(len(pos), target, rng):
                    idx = pos[piece]
                    o = model(torch.from_numpy(data.windows(rows[idx])), (1, len(idx)))
                    out[idx] += torch.stack([o['score5'], o['score10']], 1).double().numpy()
    return out/n_perm


@pytest.mark.parametrize('n_perm', [1, 2, 3])
def test_predict_cross_equals_unbatched_reference_even_when_chunks_share_a_batch(tmp_path, n_perm):
    d, day, stock = build(tmp_path, names=60)            # 60 stocks/session, target 20 -> 3 equal chunks per permutation
    data = sr.SeqData(d, window=1, normalize=None)
    tr, va, ev = data.split_rows(15)
    ev = ev[:120]                                         # 2 sessions -> all chunks of all permutations land in ONE batch (repeated indices)
    torch.manual_seed(0)
    m = cs.CrossStockRanker(data.X.shape[1], 8, 1, 1, mixer_factory=fac, dropout=0.)
    got = cs.predict_cross(m, data, ev, 'cpu', target=20, n_perm=n_perm, seed=5)
    ref = reference_predict(m, data, ev, 'cpu', 20, n_perm, 5)
    assert np.abs(got-ref).max() < 1e-5


def test_chunks_never_exceed_target():
    rng = np.random.default_rng(0)
    for n in (1, 130, 255, 256, 257, 300, 383, 384, 1800):
        assert max(map(len, cs.chunk_session(n, 256, rng))) <= 256
