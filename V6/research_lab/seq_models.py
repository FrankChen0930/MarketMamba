"""Official-Mamba2 and Samba ranker definitions for the V7 (N stocks x 60 sessions x F) contract. RESEARCH_ONLY.

* Mamba2 mixer is the *official* mamba_ssm.Mamba2 (CUDA/Triton only), imported lazily. Nothing here is a
  simplified selective SSM. For CPU tests a mixer_factory can be injected; such runs prove shape/causality
  contracts only, never model quality.
* Samba = repeated [Mamba -> MLP -> sliding-window attention -> MLP] (Ren et al. 2024) built on the same mixer.
* Attention is plain torch (scaled_dot_product_attention + causal sliding-window mask), runnable on CPU.
* No training loop here: B7 is definition + compatibility only.
"""
from __future__ import annotations

import argparse
import json
import platform

import torch
from torch import nn
import torch.nn.functional as F


def mamba2_dims(d_model, expand=2, headdim=64, d_state=128, strict_kernel=True):
    """Constraints read from mamba_ssm 2.3.2.post1 modules/mamba2.py (d_ssm % headdim == 0) plus the kernel's
    preference that nheads is a multiple of 8 (documented in the mamba_ssm README)."""
    d_inner = expand*d_model
    if d_inner % headdim:
        raise ValueError(f'expand*d_model={d_inner} must be divisible by headdim={headdim}')
    nheads = d_inner//headdim
    if strict_kernel and nheads % 8:
        raise ValueError(f'nheads={nheads} should be a multiple of 8 for the fused kernel (headdim={headdim})')
    return dict(d_inner=d_inner, nheads=nheads, d_state=d_state)


def official_mamba2_factory(d_model, **kw):
    try:
        from mamba_ssm import Mamba2
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError('official mamba_ssm.Mamba2 not importable (needs CUDA build); '
                          'run on Colab GPU or inject mixer_factory for CPU contract tests') from e
    return Mamba2(d_model=d_model, **kw)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)*self.w


class Residual(nn.Module):
    """x + f(norm(x))"""
    def __init__(self, d, fn):
        super().__init__()
        self.norm, self.fn = RMSNorm(d), fn

    def forward(self, x):
        return x+self.fn(self.norm(x))


class MLP(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        self.up, self.gate, self.down = nn.Linear(d, mult*d, bias=False), nn.Linear(d, mult*d, bias=False), nn.Linear(mult*d, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x))*self.up(x))


def sliding_window_mask(t, window, device=None):
    """True where attention is allowed: causal and within `window` most recent positions (incl. self)."""
    i = torch.arange(t, device=device)
    d = i[:, None]-i[None, :]
    return (d >= 0) & (d < window)


class SlidingWindowAttention(nn.Module):
    def __init__(self, d, heads=4, window=16):
        super().__init__()
        if d % heads:
            raise ValueError('d_model must be divisible by heads')
        self.h, self.window = heads, window
        self.qkv, self.out = nn.Linear(d, 3*d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d//self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=sliding_window_mask(t, self.window, x.device))
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class SambaLayer(nn.Module):
    def __init__(self, d, mixer, heads, window):
        super().__init__()
        self.blocks = nn.Sequential(Residual(d, mixer), Residual(d, MLP(d)),
                                    Residual(d, SlidingWindowAttention(d, heads, window)), Residual(d, MLP(d)))

    def forward(self, x):
        return self.blocks(x)


class SeqRanker(nn.Module):
    """(B, T, F) -> dict(score5, score10) from the last session's representation."""

    def __init__(self, kind, n_features=48, d_model=64, n_layers=1, mixer_kw=None, attn_heads=4, window=16,
                 mixer_factory=None):
        super().__init__()
        if kind not in ('mamba2', 'samba'):
            raise ValueError('kind must be mamba2 or samba')
        self.kind = kind
        make = mixer_factory or official_mamba2_factory
        mkw = dict(mixer_kw or {})
        self.embed = nn.Linear(n_features, d_model)
        layers = []
        for i in range(n_layers):
            mixer = make(d_model, **mkw) if mixer_factory is None else make(d_model)
            layers.append(Residual(d_model, mixer) if kind == 'mamba2' else SambaLayer(d_model, mixer, attn_heads, window))
        self.layers = nn.Sequential(*layers)
        self.norm = RMSNorm(d_model)
        self.head5, self.head10 = nn.Linear(d_model, 1), nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.norm(self.layers(self.embed(x)))[:, -1]
        return dict(score5=self.head5(h).squeeze(-1), score10=self.head10(h).squeeze(-1))


def probe():
    """Environment/compat probe; safe on CPU (reports what is missing instead of failing)."""
    info = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.cuda.is_available(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    for mod in ('mamba_ssm', 'causal_conv1d', 'triton'):
        try:
            info[mod] = getattr(__import__(mod), '__version__', 'present')
        except Exception as e:  # noqa: BLE001
            info[mod] = f'missing: {type(e).__name__}'
    info['mamba2_dims_width64_headdim8'] = mamba2_dims(64, 2, 8)
    if info['cuda'] and not str(info['mamba_ssm']).startswith('missing'):
        out = {}
        for kind in ('mamba2', 'samba'):
            m = SeqRanker(kind, 48, 64, 1, mixer_kw=dict(d_state=16, headdim=8, expand=2)).cuda()
            x = torch.randn(32, 60, 48, device='cuda', requires_grad=False)
            y = m(x)
            (y['score5'].mean()+y['score10'].mean()).backward()
            out[kind] = dict(params=sum(p.numel() for p in m.parameters()), score5_shape=list(y['score5'].shape),
                             finite=bool(torch.isfinite(y['score5']).all()),
                             grad_finite=all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None),
                             peak_mem_mb=round(torch.cuda.max_memory_allocated()/1e6, 1))
        info['smoke'] = out
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    ap.parse_args()
    print(json.dumps(probe(), indent=2, default=str))
