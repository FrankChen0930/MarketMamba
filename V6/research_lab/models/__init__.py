"""Small CPU research rankers; no relation to frozen production weights."""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def selective_scan(u, delta, a, b, c):
    """Diagonal selective SSM: ZOH transition, first-order input discretization.

    u/delta: B,T,D; a: D,N (negative); b/c: B,T,N.
    Returns B,T,D; recurrence is causal and differentiable, with zero initial state.
    """
    state = u.new_zeros(u.shape[0], u.shape[2], a.shape[1])
    outputs = []
    for t in range(u.shape[1]):
        step = delta[:, t, :, None]
        state = (step * a).exp() * state + step * b[:, t, None, :] * u[:, t, :, None]
        outputs.append((state * c[:, t, None, :]).sum(-1))
    return torch.stack(outputs, dim=1)


class GRURanker(nn.Module):
    def __init__(self, features, width=12):
        super().__init__()
        self.gru = nn.GRU(features, width, batch_first=True)
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        return self.head(self.gru(x)[0][:, -1])


class SelectiveRanker(nn.Module):
    """S6-like input-selective delta/B/C, gated residual, pure PyTorch scan.

    Deliberately small; not an implementation/equivalence claim for Mamba-1/2.
    """
    def __init__(self, features, width=12, state=4):
        super().__init__()
        self.input = nn.Linear(features, width * 2)
        self.select = nn.Linear(width, width + 2 * state)
        self.log_a = nn.Parameter(torch.arange(1, state+1).float().log().repeat(width, 1))
        self.skip = nn.Parameter(torch.ones(width))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)
        self.width, self.state = width, state

    def forward(self, x):
        u, gate = self.input(x).chunk(2, -1)
        u = F.silu(u)
        dt, b, c = torch.split(self.select(u), [self.width, self.state, self.state], -1)
        y = selective_scan(u, F.softplus(dt), -self.log_a.exp(), b, c)
        y = (y + self.skip*u) * F.silu(gate)
        return self.head(self.norm(y[:, -1]))
