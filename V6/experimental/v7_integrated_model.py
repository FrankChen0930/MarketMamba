"""Isolated V7 candidate: no-conv Mamba-2 temporal + GATv2 + Bi-Mamba-2.

The SSD operation is a required dependency injection.  This module does not
import mamba_ssm, probe CUDA, or silently substitute a synthetic operation.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from v7_integrated_config import V7IntegratedConfig  # noqa: E402

SSDCallable = Callable[..., Tensor]


def canonical_stock_order(stock_ids: Sequence[object]) -> tuple[Tensor, Tensor]:
    """Return stable lexical order and its inverse; duplicate IDs are rejected."""
    keys = [str(value) for value in stock_ids]
    if len(set(keys)) != len(keys):
        raise ValueError("stock IDs must be unique")
    ordered = sorted(range(len(keys)), key=lambda index: (keys[index], index))
    order = torch.tensor(ordered, dtype=torch.long)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order))
    return order, inverse


def bidirectional_scan(sequence: Tensor, scan: Callable[[Tensor], Tensor]) -> Tensor:
    """Scan both directions and restore the reverse result before averaging."""
    forward = scan(sequence)
    reverse = torch.flip(scan(torch.flip(sequence, dims=(1,))), dims=(1,))
    return 0.5 * (forward + reverse)


class FactorGroupedEmbedding(nn.Module):
    """V6 proportional factor-group projection, isolated from eager Mamba imports."""
    def __init__(self, config: V7IntegratedConfig):
        super().__init__()
        raw = [config.d_model * size // config.input_dim for size in config.group_dims]
        raw[max(range(len(raw)), key=lambda i: config.group_dims[i])] += config.d_model - sum(raw)
        self.projections = nn.ModuleList(nn.Linear(src, dst) for src, dst in zip(config.group_dims, raw))
        self.group_dims = config.group_dims
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        pieces = torch.split(x, self.group_dims, dim=-1)
        return self.dropout(self.norm(torch.cat([layer(part) for layer, part in zip(self.projections, pieces)], -1)))


class GatedRMSNorm(nn.Module):
    """Mamba-2-style RMS normalization gated by SiLU(z)."""
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = x * F.silu(z)
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class Mamba2NoConv(nn.Module):
    """Projection/SSD/gating portion of official Mamba-2 with convolution removed."""
    def __init__(self, config: V7IntegratedConfig, ssd_fn: SSDCallable | None):
        super().__init__()
        self.config = config
        self.ssd_fn = ssd_fn
        self.inner = config.expand * config.d_model
        self.n_heads = self.inner // config.head_dim
        projection_size = 2 * self.inner + 2 * config.n_groups * config.d_state + self.n_heads
        self.in_proj = nn.Linear(config.d_model, projection_size, bias=False)
        self.out_proj = nn.Linear(self.inner, config.d_model, bias=False)
        self.norm = GatedRMSNorm(self.inner)
        self.A_log = nn.Parameter(torch.empty(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.n_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            A = torch.empty(self.n_heads).uniform_(self.config.a_min, self.config.a_max)
            self.A_log.copy_(torch.log(A))
            log_dt = torch.empty(self.n_heads).uniform_(math.log(self.config.dt_min), math.log(self.config.dt_max))
            dt = torch.exp(log_dt).clamp_min(self.config.dt_floor)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

    def forward(self, u: Tensor) -> Tensor:
        if self.ssd_fn is None:
            raise RuntimeError("Mamba2NoConv requires an explicit SSD callable; no fallback is provided")
        z, x, B, C, dt = torch.split(
            self.in_proj(u),
            [self.inner, self.inner, self.config.n_groups * self.config.d_state,
             self.config.n_groups * self.config.d_state, self.n_heads],
            dim=-1,
        )
        # In upstream Mamba-2 the causal convolution is followed by SiLU over
        # x/B/C.  This candidate removes the convolution itself, not that
        # activation or any of the projection/SSD/gating semantics.
        x, B, C = F.silu(x), F.silu(B), F.silu(C)
        batch, length, _ = x.shape
        x_heads = x.view(batch, length, self.n_heads, self.config.head_dim)
        B = B.view(batch, length, self.config.n_groups, self.config.d_state)
        C = C.view(batch, length, self.config.n_groups, self.config.d_state)
        y = self.ssd_fn(
            x_heads, dt, -torch.exp(self.A_log.float()), B, C,
            D=self.D.float(), dt_bias=self.dt_bias.float(), dt_softplus=True,
            chunk_size=self.config.chunk_size,
        ).reshape(batch, length, self.inner)
        return self.out_proj(self.norm(y, z))


class ResidualMamba2(nn.Module):
    def __init__(self, config: V7IntegratedConfig, ssd_fn: SSDCallable | None):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mixer = Mamba2NoConv(config, ssd_fn)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))


class ExistingGATv2Layer(nn.Module):
    """Same GATv2 parameters and residual convention as V6 GraphAttentionLayer."""
    def __init__(self, config: V7IntegratedConfig, n_heads: int = 4):
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError as exc:
            raise RuntimeError("torch_geometric GATv2 is required; inject a graph layer for tests") from exc
        if config.d_model % n_heads:
            raise ValueError("d_model must be divisible by GAT heads")
        self.norm = nn.LayerNorm(config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.gat = GATv2Conv(config.d_model, config.d_model // n_heads, heads=n_heads,
                             edge_dim=1, dropout=config.dropout, add_self_loops=False)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.shape[1] == 0:
            return h
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        return h + self.drop(self.gat(self.norm(h), edge_index, edge_attr))


class BoundedFusion(nn.Module):
    def __init__(self, d_model: int, limit: float):
        super().__init__()
        self.logits = nn.Linear(3 * d_model, 3)
        self.lower = 1.0 - limit
        self.span = 1.0 - 3.0 * self.lower

    def forward(self, temporal: Tensor, graph: Tensor, scan: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.lower + self.span * torch.softmax(self.logits(torch.cat((temporal, graph, scan), -1)), -1)
        stacked = torch.stack((temporal, graph, scan), dim=1)
        return (weights.unsqueeze(-1) * stacked).sum(1), weights


class MarketMambaV7Integrated(nn.Module):
    """External contract: (x, edge_index, edge_attr, padding_mask) -> (N, 2)."""
    def __init__(self, config: V7IntegratedConfig, *, ssd_fn: SSDCallable | None = None,
                 graph_layer: nn.Module | None = None):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = FactorGroupedEmbedding(config)
        self.missing_observation = nn.Parameter(torch.zeros(config.d_model))
        self.temporal = nn.ModuleList(ResidualMamba2(config, ssd_fn) for _ in range(config.temporal_layers))
        self.graph_layer = graph_layer if graph_layer is not None else ExistingGATv2Layer(config)
        self.cross_forward = Mamba2NoConv(config, ssd_fn)
        self.cross_reverse = Mamba2NoConv(config, ssd_fn)
        self.fusion = BoundedFusion(config.d_model, config.fusion_limit)
        self.fused_norm = nn.LayerNorm(config.d_model)
        self.head_5d = nn.Linear(config.d_model, 1)
        self.head_10d = nn.Linear(config.d_model, 1)

    @staticmethod
    def _eligible_graph(edge_index: Tensor, edge_attr: Tensor, eligible: Tensor) -> tuple[Tensor, Tensor]:
        mapping = torch.full((eligible.numel(),), -1, dtype=torch.long, device=edge_index.device)
        mapping[eligible] = torch.arange(int(eligible.sum()), device=edge_index.device)
        if edge_index.numel() == 0:
            return edge_index, edge_attr
        valid_bounds = (edge_index >= 0).all(0) & (edge_index < eligible.numel()).all(0)
        safe = edge_index[:, valid_bounds]
        attrs = edge_attr[valid_bounds]
        keep = eligible[safe[0]] & eligible[safe[1]]
        return mapping[safe[:, keep]], attrs[keep]

    def _temporal_last(self, x: Tensor, padding_mask: Tensor, observation_mask: Tensor) -> Tensor:
        """Remove leading pre-history only; every internal calendar hole remains."""
        first = padding_mask.long().argmax(1)
        representations = x.new_empty((x.shape[0], self.config.d_model))
        for start in sorted(set(first.tolist())):
            indices = torch.nonzero(first == start, as_tuple=False).flatten()
            valid = observation_mask[indices, start:]
            sequence = torch.where(valid.unsqueeze(-1), x[indices, start:], 0.)
            sequence = self.embedding(sequence)
            sequence = torch.where(valid.unsqueeze(-1), sequence, self.missing_observation)
            for layer in self.temporal:
                sequence = layer(sequence)
            representations[indices] = sequence[:, -1]
        return representations

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                padding_mask: Tensor | None = None, observation_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError("x must have shape (N, T, input_dim)")
        if padding_mask is None:
            padding_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif padding_mask.shape != x.shape[:2]:
            raise ValueError("padding_mask must have shape (N, T)")
        else:
            padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a long tensor with shape (2, E)")
        if edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_attr length must match edge_index")
        if observation_mask is None:
            observation_mask = padding_mask
        if observation_mask.shape != padding_mask.shape:
            raise ValueError("observation_mask must have shape (N, T)")
        observation_mask = observation_mask.to(x.device).bool() & padding_mask
        eligible = observation_mask[:, -1]
        output = x.new_zeros((x.shape[0], 2))
        if not bool(eligible.any()):
            return output
        temporal = self._temporal_last(x[eligible], padding_mask[eligible], observation_mask[eligible])
        remapped_edges, remapped_attrs = self._eligible_graph(edge_index, edge_attr, eligible)
        graph = self.graph_layer(temporal, remapped_edges, remapped_attrs)
        seq = temporal.unsqueeze(0)
        forward = self.cross_forward(seq)
        reverse = torch.flip(self.cross_reverse(torch.flip(seq, dims=(1,))), dims=(1,))
        scan = (forward + reverse).squeeze(0) * 0.5
        fused, _ = self.fusion(temporal, graph, scan)
        fused = self.fused_norm(fused)
        output[eligible] = torch.cat((self.head_5d(fused), self.head_10d(fused)), -1)
        return output
