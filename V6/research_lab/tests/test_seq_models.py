import pytest
import torch
from torch import nn
from research_lab.seq_models import (SeqRanker, SlidingWindowAttention, mamba2_dims, sliding_window_mask, probe)


class CausalConvMixer(nn.Module):
    """CPU stand-in for the official mixer: causal depthwise conv. Contract tests only, not model quality."""
    def __init__(self, d, k=4):
        super().__init__()
        self.k, self.conv = k, nn.Conv1d(d, d, k, groups=d)

    def forward(self, x):
        y = self.conv(nn.functional.pad(x.transpose(1, 2), (self.k-1, 0)))
        return y.transpose(1, 2)


def fake(d):
    return CausalConvMixer(d)


def test_mamba2_dim_constraints_from_official_source():
    assert mamba2_dims(64, 2, 8)['nheads'] == 16
    with pytest.raises(ValueError, match='divisible'):
        mamba2_dims(64, 2, 48)
    with pytest.raises(ValueError, match='multiple of 8'):
        mamba2_dims(64, 2, 64)            # nheads=2: valid for the module, not for the fused kernel
    assert mamba2_dims(64, 2, 64, strict_kernel=False)['nheads'] == 2


def test_sliding_window_mask_hand_case():
    m = sliding_window_mask(5, 2)
    assert m.int().tolist() == [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]]


def test_attention_window_and_causality():
    torch.manual_seed(0)
    a = SlidingWindowAttention(16, heads=2, window=3).eval()
    x = torch.randn(1, 10, 16)
    base = a(x)
    far = x.clone(); far[:, 0] += 5          # outside window of position 9 -> no effect
    fut = x.clone(); fut[:, 9] += 5          # future of position 5 -> no effect
    near = x.clone(); near[:, 8] += 5        # inside window of position 9 -> effect
    assert torch.allclose(a(far)[:, 9], base[:, 9], atol=1e-6)
    assert torch.allclose(a(fut)[:, :9], base[:, :9], atol=1e-6)
    assert not torch.allclose(a(near)[:, 9], base[:, 9], atol=1e-4)


@pytest.mark.parametrize('kind', ['mamba2', 'samba'])
def test_ranker_shapes_causality_and_gradients(kind):
    torch.manual_seed(1)
    m = SeqRanker(kind, n_features=12, d_model=16, n_layers=2, attn_heads=2, window=8, mixer_factory=fake).eval()
    x = torch.randn(5, 60, 12)
    y = m(x)
    assert y['score5'].shape == (5,) and y['score10'].shape == (5,)
    # the last-session score must depend on recent sessions but the model never sees the future: appending is not possible
    # (T is the window), so check dependence on the last input and independence of per-stock batch members
    x2 = x.clone(); x2[0, -1] += 3
    y2 = m(x2)
    assert not torch.allclose(y['score5'][0], y2['score5'][0], atol=1e-5)
    assert torch.allclose(y['score5'][1:], y2['score5'][1:], atol=1e-6)    # stocks are independent
    m.train()
    out = m(x)
    (out['score5'].sum()+out['score10'].sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_samba_has_attention_and_mamba2_does_not():
    mk = lambda k: SeqRanker(k, 12, 16, 1, attn_heads=2, mixer_factory=fake)
    assert any(isinstance(x, SlidingWindowAttention) for x in mk('samba').modules())
    assert not any(isinstance(x, SlidingWindowAttention) for x in mk('mamba2').modules())
    with pytest.raises(ValueError):
        SeqRanker('selective_ssm', mixer_factory=fake)


def test_official_mixer_missing_gives_clear_error_on_cpu_env():
    try:
        import mamba_ssm  # noqa: F401
        pytest.skip('mamba_ssm present')
    except ImportError:
        with pytest.raises(ImportError, match='official mamba_ssm.Mamba2'):
            SeqRanker('mamba2', 12, 16, 1)


def test_probe_runs_on_cpu_and_reports_missing():
    info = probe()
    assert info['torch'] and 'smoke' not in info or info['cuda']
    assert info['mamba2_dims_width64_headdim8']['nheads'] == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_official_mamba2_gpu_smoke():
    pytest.importorskip('mamba_ssm')
    for kind in ('mamba2', 'samba'):
        m = SeqRanker(kind, 48, 64, 1, mixer_kw=dict(d_state=16, headdim=8, expand=2)).cuda()
        y = m(torch.randn(8, 60, 48, device='cuda'))
        assert torch.isfinite(y['score5']).all()
