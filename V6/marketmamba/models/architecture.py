"""
MarketMamba V6 — Model Architecture
=====================================
Core components:
  FactorGroupedEmbedding  : factor-aware input projection (46D → d_model)
  MultiScaleMambaEncoder  : parallel short/mid/long Mamba branches + adaptive fusion
  MarketMambaV6           : full model = Embedding → MultiScaleMamba → GATv2 → MultiHead

All components are designed so that:
  - input tensor shape: (N_stocks, seq_len, input_dim)
  - output tensor shape: (N_stocks, n_horizons)  where n_horizons = 3 (5d, 20d, 60d)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "mamba_ssm is required. Install it via the pre-built wheel:\n"
        "  pip install mamba_ssm-*.whl causal_conv1d-*.whl\n"
        "These must be compiled for your CUDA version."
    )

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    raise ImportError(
        "torch_geometric is required:\n"
        "  pip install torch_geometric"
    )

from marketmamba.config import (
    D_MODEL,
    D_STATE,
    DROPOUT,
    GROUP_DIMS,
    INPUT_DIM,
    MAX_NEIGHBORS_GAT,
    MULTI_SCALE_LAYERS,
    MULTI_SCALE_SEQLENS,
    N_HEADS_GAT,
    PRED_HORIZONS,
)


# ============================================================
# Helper: MambaBlock stack
# ============================================================

class MambaStack(nn.Module):
    """
    Stack of n Mamba SSM layers with residual connections and LayerNorm.
    Input/output shape: (batch, seq_len, d_model)
    """
    def __init__(self, n_layers: int, d_model: int = D_MODEL, d_state: int = D_STATE):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, T, d_model)
        for mamba, norm in zip(self.layers, self.norms):
            x = x + mamba(norm(x))   # pre-norm residual
        return x


# ============================================================
# Component 1: FactorGroupedEmbedding
# ============================================================

class FactorGroupedEmbedding(nn.Module):
    """
    Projects each factor group into its own sub-space, then concatenates.

    Groups (from config.FEATURE_GROUPS):
      A: price_momentum      (12 dims) → d_model // 4
      B: institutional_flow  (16 dims) → d_model // 4
      C: fundamentals        (10 dims) → d_model // 4
      D: macro_environment   ( 8 dims) → d_model // 4
    Concat → LayerNorm → d_model

    Benefits:
      - Different factor types learn in their own sub-space
      - No cross-contamination between price signals and macro signals
      - Initialisation is cleaner than a single large linear
    """
    def __init__(
        self,
        group_dims: dict[str, int] = GROUP_DIMS,
        d_model: int = D_MODEL,
    ):
        super().__init__()
        sub_dim = d_model // 4   # 64 per group (for d_model=256)

        self.proj_A = nn.Linear(group_dims["price_momentum"],     sub_dim)
        self.proj_B = nn.Linear(group_dims["institutional_flow"], sub_dim)
        self.proj_C = nn.Linear(group_dims["fundamentals"],       sub_dim)
        self.proj_D = nn.Linear(group_dims["macro_environment"],  sub_dim)
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(DROPOUT)

        # Store slice positions for clarity
        dim_A = group_dims["price_momentum"]
        dim_B = group_dims["institutional_flow"]
        dim_C = group_dims["fundamentals"]
        self._slices = {
            "A": (0, dim_A),
            "B": (dim_A, dim_A + dim_B),
            "C": (dim_A + dim_B, dim_A + dim_B + dim_C),
            "D": (dim_A + dim_B + dim_C, INPUT_DIM),
        }

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, T, 46)
        sA, eA = self._slices["A"]
        sB, eB = self._slices["B"]
        sC, eC = self._slices["C"]
        sD, eD = self._slices["D"]

        hA = self.proj_A(x[..., sA:eA])
        hB = self.proj_B(x[..., sB:eB])
        hC = self.proj_C(x[..., sC:eC])
        hD = self.proj_D(x[..., sD:eD])

        out = self.norm(torch.cat([hA, hB, hC, hD], dim=-1))
        return self.drop(out)   # (N, T, d_model)


# ============================================================
# Component 2: MultiScaleMambaEncoder
# ============================================================

class MultiScaleMambaEncoder(nn.Module):
    """
    Three parallel Mamba branches capturing different temporal horizons:
      Short (20d,  2 layers): momentum, chipset signals
      Mid   (60d,  3 layers): quarterly trends, institutional mid-term
      Long  (252d, 3 layers): fundamental cycles, annual patterns

    Each branch takes the last T_k timesteps of the sequence.
    Outputs are fused via a learnable attention-weighted sum.

    Input:  (N, 252, d_model)
    Output: (N, d_model)
    """
    def __init__(
        self,
        d_model:    int = D_MODEL,
        d_state:    int = D_STATE,
        n_layers:   list[int] = MULTI_SCALE_LAYERS,    # [2, 3, 3]
        seq_lens:   list[int] = MULTI_SCALE_SEQLENS,   # [20, 60, 252]
    ):
        super().__init__()
        self.seq_short = seq_lens[0]
        self.seq_mid   = seq_lens[1]
        self.seq_long  = seq_lens[2]

        self.mamba_short = MambaStack(n_layers[0], d_model, d_state)
        self.mamba_mid   = MambaStack(n_layers[1], d_model, d_state)
        self.mamba_long  = MambaStack(n_layers[2], d_model, d_state)

        # Scale fusion: learn which branch to trust more per stock
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model * 3, 3),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 252, d_model) — full year window required
        h_short = self.mamba_short(x[:, -self.seq_short:, :])[:, -1, :]   # (N, d_model)
        h_mid   = self.mamba_mid(x[:, -self.seq_mid:,   :])[:, -1, :]
        h_long  = self.mamba_long(x)[:, -1, :]

        # Adaptive scale fusion
        cat_h  = torch.cat([h_short, h_mid, h_long], dim=-1)   # (N, d_model*3)
        scales = self.scale_gate(cat_h)                          # (N, 3)

        fused = (
            scales[:, 0:1] * h_short
            + scales[:, 1:2] * h_mid
            + scales[:, 2:3] * h_long
        )
        return self.norm(fused)   # (N, d_model)


# ============================================================
# Component 3: GATv2 Graph Layer
# ============================================================

class GraphAttentionLayer(nn.Module):
    """
    One GATv2 cross-sectional layer.
    Nodes: stocks in today's cross-section.
    Edges: from knowledge graph cache (supply chain + rolling correlation).

    Input:  h (N, d_model), edge_index (2, E), edge_attr (E,)
    Output: h (N, d_model)
    """
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS_GAT):
        super().__init__()
        assert d_model % n_heads == 0
        head_dim = d_model // n_heads
        self.gat = GATv2Conv(
            in_channels=d_model,
            out_channels=head_dim,
            heads=n_heads,
            edge_dim=1,
            dropout=DROPOUT,
            add_self_loops=True,
            concat=True,   # output: N × (heads × head_dim) = N × d_model
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(DROPOUT)

    def forward(
        self,
        h:          Tensor,
        edge_index: Tensor,
        edge_attr:  Tensor,
    ) -> Tensor:
        # Residual + pre-norm
        h_in = h
        h_attn = self.gat(self.norm(h), edge_index, edge_attr.unsqueeze(-1))
        return h_in + self.drop(h_attn)


# ============================================================
# Component 4: MultiHorizonHead
# ============================================================

class MultiHorizonHead(nn.Module):
    """
    Three independent linear prediction heads:
      head_5d  → pred_5d  (short-term, auxiliary)
      head_20d → pred_20d (primary output)
      head_60d → pred_60d (long-term, auxiliary)

    Returns a stacked tensor: (N, 3) in order [5d, 20d, 60d]
    """
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.head_5d  = nn.Linear(d_model, 1)
        self.head_20d = nn.Linear(d_model, 1)
        self.head_60d = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, d_model)
        p5  = self.head_5d(x)   # (N, 1)
        p20 = self.head_20d(x)
        p60 = self.head_60d(x)
        return torch.cat([p5, p20, p60], dim=-1)   # (N, 3)


# ============================================================
# Full Model: MarketMambaV6
# ============================================================

class MarketMambaV6(nn.Module):
    """
    MarketMamba V6 — Pure-Quant Multi-Scale SSM + Graph Attention

    Architecture:
      (N, 252, 46)
        → FactorGroupedEmbedding       → (N, 252, d_model)
        → MultiScaleMambaEncoder       → (N, d_model)   ← temporal encoding
        → GraphAttentionLayer          → (N, d_model)   ← cross-stock relations
        → Gating Fusion (temporal + graph)
        → MultiHorizonHead             → (N, 3)

    Args:
        d_model     : model dimension (default 256)
        d_state     : Mamba state dimension (default 32)
        n_heads_gat : GAT attention heads (default 4)
        dropout     : dropout probability (default 0.1)
    """
    def __init__(
        self,
        d_model:     int = D_MODEL,
        d_state:     int = D_STATE,
        n_heads_gat: int = N_HEADS_GAT,
        dropout:     float = DROPOUT,
    ):
        super().__init__()

        self.embedding = FactorGroupedEmbedding(d_model=d_model)
        self.encoder   = MultiScaleMambaEncoder(d_model=d_model, d_state=d_state)
        self.graph_layer = GraphAttentionLayer(d_model=d_model, n_heads=n_heads_gat)

        # Gating fusion: combine temporal encoding with graph-updated encoding
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.norm_fuse = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

        self.head = MultiHorizonHead(d_model=d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x:          Tensor,
        edge_index: Tensor,
        edge_attr:  Tensor,
    ) -> Tensor:
        """
        Args:
            x          : (N, seq_len=252, input_dim=46)
            edge_index : (2, E) — pre-computed from KG cache, for this cross-section
            edge_attr  : (E,)   — edge weights

        Returns:
            preds : (N, 3) — [pred_5d, pred_20d, pred_60d]
        """
        # Step 1: Factor-grouped embedding
        h = self.embedding(x)          # (N, 252, d_model)

        # Step 2: Multi-scale temporal encoding
        h_temporal = self.encoder(h)   # (N, d_model)

        # Step 3: Cross-stock graph attention
        h_graph = self.graph_layer(h_temporal, edge_index, edge_attr)   # (N, d_model)

        # Step 4: Gating fusion
        gate_input  = torch.cat([h_temporal, h_graph], dim=-1)  # (N, d_model*2)
        gate_weight = self.gate(gate_input)                      # (N, d_model)
        h_fused = self.norm_fuse(
            gate_weight * h_temporal + (1 - gate_weight) * h_graph
        )
        h_fused = self.dropout(h_fused)

        # Step 5: Multi-horizon prediction
        preds = self.head(h_fused)     # (N, 3)
        return preds

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
