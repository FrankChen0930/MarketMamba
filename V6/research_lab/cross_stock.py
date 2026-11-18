"""C2: temporal Mamba2 per stock -> bidirectional Mamba2 ACROSS the stocks of one session (S-Mamba style).

Conformance to S-Mamba (Wang et al., arXiv 2403.11144; official code github.com/wzhwzhwzh0921/S-D-Mamba, model/S_Mamba.py and
layers/Mamba_EncDec.py, read 2026-11-16):
  SAME  : bidirectional layer = Mamba(x) + Mamba_r(x.flip(1)).flip(1) with two separate Mamba modules; x = x + that (no dropout on it);
          LayerNorm; position-wise FFN d_model->d_ff->d_model (ReLU, dropout after each); residual; LayerNorm; final LayerNorm.
  DIFFERS (deliberate, each to be judged by ablation, not assumed harmless):
    1. tokens: S-Mamba tokens are variates (each embedded from its whole lookback by one Linear); here tokens are stocks whose
       representation comes from a temporal Mamba2 over a 60-session window (the original MarketMamba design).
    2. mixer: S-Mamba uses Mamba v1 (d_conv=2, expand=1); here the official Mamba2 (user's design). d_conv defaults to 2.
    3. no instance (RevIN-style) normalisation and no Linear(d_model->pred_len): inputs are per-session rank-gauss, outputs are
       two rank-score heads (5d, 10d).
    4. token ORDER: stocks have no natural order, a Mamba scan is order-dependent. Training uses random order per batch; scoring
       averages `n_perm` random orders/chunkings. (V7 original used a fixed sorted-by-stock-code order and averaged the two
       directions instead of adding them; neither is claimed equivalent.)
    5. sequence length: training tokens = exactly `stocks_per_date` (256) distinct stocks (sessions with fewer are skipped, never
       sampled with replacement); scoring cuts a session into ceil(n/256) near-equal pieces (<=256, e.g. 300 -> 2x150), so scoring
       lengths are at most the training length but can be shorter. Sensitivity to this is an open ablation (chunk 128/256/384).
    6. no Dropout after the embedding (S-Mamba's DataEmbedding_inverted has one); cross blocks inherit mixer_kw (expand=2,
       headdim=8, d_state=16; only d_conv is set to 2) whereas S-Mamba uses expand=1.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .seq_models import Residual, RMSNorm, official_mamba2_factory


class BiMambaBlock(nn.Module):
    """S-Mamba EncoderLayer (layers/Mamba_EncDec.py) with the dead code (unused d_model=11 Mamba/attention) omitted."""

    def __init__(self, d_model, make, mixer_kw, d_ff=None, dropout=.1):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.fwd, self.rev = make(d_model, **mixer_kw), make(d_model, **mixer_kw)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):                                   # x: (D, S, d)
        x = self.norm1(x + self.fwd(x) + self.rev(x.flip(1)).flip(1))
        return self.norm2(x + self.ffn(x))


class CrossStockRanker(nn.Module):
    cross_stock = True

    def __init__(self, n_features=48, d_model=64, n_layers=2, n_cross=1, mixer_kw=None, cross_kw=None, d_ff=None, dropout=.1,
                 mixer_factory=None):
        super().__init__()
        make = mixer_factory or official_mamba2_factory
        mkw = dict(mixer_kw or {})
        ckw = dict(cross_kw if cross_kw is not None else {**mkw, 'd_conv': 2})
        self.embed = nn.Linear(n_features, d_model)
        mk = (lambda d, **k: make(d, **k)) if mixer_factory is None else (lambda d, **k: make(d))
        self.temporal = nn.Sequential(*[Residual(d_model, mk(d_model, **mkw)) for _ in range(n_layers)])
        self.t_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([BiMambaBlock(d_model, make if mixer_factory is None else (lambda d, **k: make(d)), ckw, d_ff, dropout)
                                     for _ in range(n_cross)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head5, self.head10 = nn.Linear(d_model, 1), nn.Linear(d_model, 1)

    def forward(self, x, shape):
        """x: (D*S, W, F) flat rows grouped session by session; shape=(D, S). Returns flat scores (D*S,) per head."""
        D, S = shape
        h = self.t_norm(self.temporal(self.embed(x)))[:, -1].reshape(D, S, -1)
        for b in self.blocks:
            h = b(h)
        h = self.final_norm(h).reshape(D*S, -1)
        return dict(score5=self.head5(h).squeeze(-1), score10=self.head10(h).squeeze(-1))


def chunk_session(n, target, rng):
    """Random order of n stocks cut into near-equal pieces of about `target` tokens (>=1 piece)."""
    perm = rng.permutation(n)
    return np.array_split(perm, max(1, -(-n//target)))      # ceil: every piece <= target tokens (training always uses exactly `target`)


def predict_cross(model, data, rows, device, target=256, n_perm=2, seed=0, max_batch_rows=2048):
    """Score `rows` (sorted by session). Each session is cut into ~`target`-token chunks in random order, n_perm times; scores averaged."""
    model.eval()
    rng = np.random.default_rng(seed)
    out = np.zeros((len(rows), 2), np.float64)
    by_date = {}
    for i, r in enumerate(rows):
        by_date.setdefault(int(data.date_idx[r]), []).append(i)
    jobs = []
    for _ in range(n_perm):
        for pos in by_date.values():
            pos = np.asarray(pos)
            for piece in chunk_session(len(pos), target, rng):
                jobs.append(pos[piece])
    jobs.sort(key=len)                                          # batch equal-length chunks together
    with torch.no_grad():
        i = 0
        while i < len(jobs):
            L = len(jobs[i])
            j = i
            while j < len(jobs) and len(jobs[j]) == L and (j-i+1)*L <= max_batch_rows:
                j += 1
            idx = np.concatenate(jobs[i:j])
            x = torch.from_numpy(data.windows(rows[idx])).to(device)
            o = model(x, (j-i, L))
            np.add.at(out, idx, torch.stack([o['score5'], o['score10']], 1).double().cpu().numpy())   # idx may repeat across permutations
            i = j
    return (out/n_perm).astype(np.float32)
