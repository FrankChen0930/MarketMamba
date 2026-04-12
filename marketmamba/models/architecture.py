"""
MarketMamba V5.5 — 模型架構定義
包含：
  - PositionalEncoding (正弦位置編碼)
  - MarketMambaV55 (Mamba SSM + KG-Enhanced GAT + FinBERT 情緒融合)
  - MarketMambaV5  (V5.0 向下相容版本，46 維輸入)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch_geometric.nn import GATv2Conv

from marketmamba.config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """正弦位置編碼 (Sinusoidal Positional Encoding)"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class MarketMambaV55(nn.Module):
    """
    MarketMamba V5.5: Mamba SSM + Knowledge Graph Enhanced GAT

    特色：
    1. 84 維輸入 (46 量價籌碼 + 38 情緒特徵)
    2. 4 層 Mamba SSM + Residual + LayerNorm
    3. KG-Enhanced GAT：支援知識圖譜與 cosine similarity 混合邊建構
    4. Gating Fusion：自動平衡個股訊號 vs 群體訊號
    5. 30 天 Alpha 軌跡預測

    向下相容：
    - use_knowledge_graph=False 時退回純 cosine similarity KNN
    - 可透過 input_dim 參數相容 V5.0 (46 維)
    """

    def __init__(
        self,
        input_dim: int = MODEL_CONFIG['input_dim'],
        seq_len: int = MODEL_CONFIG['seq_len'],
        d_model: int = MODEL_CONFIG['d_model'],
        pred_days: int = MODEL_CONFIG['pred_days'],
        num_mamba_layers: int = MODEL_CONFIG['num_mamba_layers'],
        d_state: int = MODEL_CONFIG['d_state'],
        d_conv: int = MODEL_CONFIG['d_conv'],
        expand: int = MODEL_CONFIG['expand'],
        dropout_rate: float = MODEL_CONFIG['dropout_rate'],
        k_neighbors: int = MODEL_CONFIG['k_neighbors'],
        use_knowledge_graph: bool = True,
    ):
        super().__init__()
        self.pred_days = pred_days
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.use_kg = use_knowledge_graph

        # 輸入嵌入層
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len)

        # Mamba SSM 層堆疊
        self.mamba_layers = nn.ModuleList()
        self.mamba_norms = nn.ModuleList()
        for _ in range(num_mamba_layers):
            self.mamba_layers.append(
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            )
            self.mamba_norms.append(nn.LayerNorm(d_model))

        self.dropout = nn.Dropout(dropout_rate)

        # GATv2 圖注意力 (4 head, concat)
        self.gat = GATv2Conv(
            d_model, d_model // 4, heads=4, concat=True, dropout=dropout_rate
        )

        # Gating 融合
        self.gating_linear = nn.Linear(d_model * 2, d_model)

        # 軌跡預測頭
        self.trajectory_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, pred_days),
        )

    def forward(
        self,
        x: torch.Tensor,
        knowledge_edges: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: (N, seq_len, input_dim) 輸入序列
            knowledge_edges: (2, E) 預計算的知識圖譜增強邊
                             若為 None 則退回純 cosine similarity KNN

        Returns:
            (N, pred_days) Alpha 軌跡預測
        """
        # === 嵌入 + 位置編碼 ===
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # === Mamba SSM 時序處理 ===
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            x = x + self.dropout(mamba(norm(x)))

        # 取最後時間步作為時序摘要
        x_temporal = x[:, -1, :]

        # === 防毒防線：保護 GAT 注意力矩陣不崩潰 ===
        x_temporal = torch.nan_to_num(x_temporal, nan=0.0)

        # === 圖建構 + GAT ===
        if self.use_kg and knowledge_edges is not None:
            # 使用預計算的知識圖譜增強邊
            edge_index = knowledge_edges.to(x.device)
        else:
            # 退回：純 cosine similarity KNN 動態圖
            edge_index = self._build_knn_edges(x_temporal)

        x_graph = self.gat(x_temporal, edge_index)

        # === Gating Fusion ===
        combined = torch.cat([x_temporal, x_graph], dim=1)
        gate = torch.sigmoid(self.gating_linear(combined))
        x_fused = gate * x_temporal + (1 - gate) * x_graph

        # === 軌跡預測 ===
        return self.trajectory_head(x_fused)

    def _build_knn_edges(self, x_temporal: torch.Tensor) -> torch.Tensor:
        """建構 cosine similarity KNN 動態圖邊"""
        N = x_temporal.size(0)
        k = min(self.k_neighbors + 1, N)

        x_norm = F.normalize(x_temporal, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.t())

        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        source_nodes = torch.arange(N, device=x_temporal.device).repeat_interleave(k)
        target_nodes = topk_indices.reshape(-1)

        return torch.stack([source_nodes, target_nodes], dim=0)


# ==========================================
# V5.0 向下相容版本 (46 維，無知識圖譜)
# ==========================================
class MarketMambaV5(MarketMambaV55):
    """V5.0 相容版本：46 維輸入，純 cosine similarity KNN"""

    def __init__(self, input_dim: int = MODEL_CONFIG['input_dim_v5'], **kwargs):
        super().__init__(input_dim=input_dim, use_knowledge_graph=False, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, knowledge_edges=None)
