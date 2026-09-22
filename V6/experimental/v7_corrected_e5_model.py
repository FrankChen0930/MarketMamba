"""Graph-free corrected E5: temporal Mamba plus bidirectional cross-stock Mamba."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from V6.experimental.v7_corrected_e5_replication import (
    CorrectedE5Config,
    canonical_stock_order,
)


class FactorGroupedEmbedding(nn.Module):
    def __init__(self, config: CorrectedE5Config):
        super().__init__()
        widths = [
            config.d_model * size // len(config.feature_order)
            for size in config.group_dims
        ]
        largest = max(range(len(widths)), key=lambda i: config.group_dims[i])
        widths[largest] += config.d_model - sum(widths)
        self.group_dims = config.group_dims
        self.projections = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(config.group_dims, widths)
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value: Tensor) -> Tensor:
        groups = torch.split(value, self.group_dims, dim=-1)
        embedded = torch.cat(
            [projection(group) for projection, group in zip(self.projections, groups)],
            dim=-1,
        )
        return self.dropout(self.norm(embedded))


class GatedRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value: Tensor, gate: Tensor) -> Tensor:
        value = value * F.silu(gate)
        value = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return value * self.weight


class Mamba2NoConv(nn.Module):
    """Incumbent no-convolution Mamba-2 block with injected official SSD."""
    def __init__(
        self,
        config: CorrectedE5Config,
        ssd_fn: Callable[..., Tensor],
        *,
        dt_min: float = .001,
        dt_max: float = .1,
        dt_floor: float = 1e-4,
        a_min: float = 1.,
        a_max: float = 16.,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.config = config
        self.ssd_fn = ssd_fn
        self.dt_floor = dt_floor
        self.chunk_size = chunk_size
        self.inner = config.expand * config.d_model
        self.n_heads = self.inner // config.head_dim
        projection_size = (
            2 * self.inner + 2 * config.n_groups * config.d_state + self.n_heads
        )
        self.in_proj = nn.Linear(config.d_model, projection_size, bias=False)
        self.out_proj = nn.Linear(self.inner, config.d_model, bias=False)
        self.norm = GatedRMSNorm(self.inner)
        self.A_log = nn.Parameter(torch.empty(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.n_heads))
        with torch.no_grad():
            a = torch.empty(self.n_heads).uniform_(a_min, a_max)
            self.A_log.copy_(torch.log(a))
            log_dt = torch.empty(self.n_heads).uniform_(
                math.log(dt_min), math.log(dt_max)
            )
            dt = torch.exp(log_dt).clamp_min(dt_floor)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

    def forward(self, value: Tensor) -> Tensor:
        z, x, b, c, dt = torch.split(
            self.in_proj(value),
            [
                self.inner,
                self.inner,
                self.config.n_groups * self.config.d_state,
                self.config.n_groups * self.config.d_state,
                self.n_heads,
            ],
            dim=-1,
        )
        x, b, c = F.silu(x), F.silu(b), F.silu(c)
        batch, length, _ = x.shape
        x = x.view(batch, length, self.n_heads, self.config.head_dim)
        b = b.view(batch, length, self.config.n_groups, self.config.d_state)
        c = c.view(batch, length, self.config.n_groups, self.config.d_state)
        output = self.ssd_fn(
            x, dt, -torch.exp(self.A_log.float()), b, c,
            D=self.D.float(), dt_bias=self.dt_bias.float(),
            dt_softplus=True, chunk_size=self.chunk_size,
        ).reshape(batch, length, self.inner)
        return self.out_proj(self.norm(output, z))


class ResidualMixer(nn.Module):
    def __init__(
        self,
        config: CorrectedE5Config,
        mixer_factory: Callable[[CorrectedE5Config], nn.Module],
    ):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = mixer_factory(config)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.mixer(self.norm(value))




def official_mixer_factory() -> Callable[[CorrectedE5Config], nn.Module]:
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError as exc:
        raise RuntimeError(
            "pinned mamba-ssm is required for formal training; no silent fallback"
        ) from exc
    return lambda config: Mamba2NoConv(config, mamba_chunk_scan_combined)


class CorrectedE5Model(nn.Module):
    """Input (N, 60, 48), canonical stock IDs; output (N, 2)."""
    def __init__(
        self,
        config: CorrectedE5Config,
        *,
        mixer_factory: Callable[[CorrectedE5Config], nn.Module] | None = None,
    ):
        super().__init__()
        config.validate()
        self.config = config
        factory = mixer_factory or official_mixer_factory()
        self.embedding = FactorGroupedEmbedding(config)
        self.missing_observation = nn.Parameter(torch.zeros(config.d_model))
        self.temporal = nn.ModuleList([
            ResidualMixer(config, factory) for _ in range(config.temporal_layers)
        ])
        self.cross_forward = nn.Sequential(*[
            ResidualMixer(config, factory) for _ in range(config.forward_layers)
        ])
        self.cross_reverse = nn.Sequential(*[
            ResidualMixer(config, factory) for _ in range(config.reverse_layers)
        ])
        self.output_norm = nn.LayerNorm(config.d_model)
        self.head_5d = nn.Linear(config.d_model, 1)
        self.head_10d = nn.Linear(config.d_model, 1)

    def forward(
        self,
        x: Tensor,
        stock_ids: Sequence[object],
        observation_mask: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            self.config.sequence_length, len(self.config.feature_order)
        ):
            raise ValueError("x must have shape (N, 60, 48)")
        if len(stock_ids) != x.shape[0]:
            raise ValueError("stock_ids length must match N")
        if observation_mask is None:
            observation_mask = torch.isfinite(x).all(dim=-1)
        if tuple(observation_mask.shape) != tuple(x.shape[:2]):
            raise ValueError("observation_mask must have shape (N, 60)")
        clean = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        encoded = self.embedding(clean)
        encoded = torch.where(
            observation_mask.to(x.device).bool().unsqueeze(-1),
            encoded,
            self.missing_observation,
        )
        for layer in self.temporal:
            encoded = layer(encoded)
        temporal = encoded[:, -1]

        order_values = canonical_stock_order(stock_ids)
        order = torch.tensor(order_values, dtype=torch.long, device=x.device)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=x.device)
        sequence = temporal[order].unsqueeze(0)
        forward = self.cross_forward(sequence)
        reverse = torch.flip(
            self.cross_reverse(torch.flip(sequence, dims=(1,))), dims=(1,)
        )
        cross = (forward + reverse).squeeze(0) * .5
        cross = cross[inverse]
        fused = self.output_norm(cross)
        return torch.cat((self.head_5d(fused), self.head_10d(fused)), dim=-1)
