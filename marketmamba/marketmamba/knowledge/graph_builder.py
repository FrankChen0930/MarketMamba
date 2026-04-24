"""
MarketMamba V5.5 — 知識圖譜增強邊建構器
混合 cosine similarity + 產業分類關係 建構 GAT 的 edge_index

策略：
  edge_score = (1 - kg_weight) × cosine_similarity + kg_weight × knowledge_score

  knowledge_score:
    - 同產業 = 0.5
    - 未分類 = 0.0
"""

import logging

import torch
import torch.nn.functional as F

from marketmamba.knowledge.sector_mapping import get_sector

logger = logging.getLogger('MarketMamba.graph_builder')


def _build_knowledge_similarity(stock_ids: list[str]) -> torch.Tensor:
    """
    建構知識圖譜相似度矩陣

    同產業 = 0.5, 不同產業 = 0.0

    Args:
        stock_ids: 股票代號列表

    Returns:
        (N, N) similarity matrix
    """
    N = len(stock_ids)
    sectors = [get_sector(sid) for sid in stock_ids]

    kg_sim = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i == j:
                kg_sim[i, j] = 1.0
            elif sectors[i] == sectors[j] and sectors[i] != "未分類":
                kg_sim[i, j] = 0.5

    return kg_sim


def build_knowledge_enhanced_edges(
    stock_ids: list[str],
    x_temporal: torch.Tensor,
    k_neighbors: int = 10,
    kg_weight: float = 0.3,
) -> torch.Tensor:
    """
    混合圖邊建構

    結合 Mamba 輸出特徵的 cosine similarity 和知識圖譜先驗關係

    Args:
        stock_ids: 股票代號列表 (長度 N)
        x_temporal: (N, d_model) Mamba 最後時間步輸出
        k_neighbors: 每個節點的鄰居數量
        kg_weight: 知識圖譜權重 (0~1)

    Returns:
        edge_index: (2, E) 邊索引 tensor
    """
    N = x_temporal.size(0)
    k = min(k_neighbors + 1, N)
    device = x_temporal.device

    # 1. 數值相似度 (cosine)
    x_norm = F.normalize(x_temporal.detach(), p=2, dim=1)
    cosine_sim = torch.mm(x_norm, x_norm.t())  # (N, N)

    # 2. 知識圖譜相似度
    kg_sim = _build_knowledge_similarity(stock_ids).to(device)  # (N, N)

    # 3. 混合分數
    combined_sim = (1 - kg_weight) * cosine_sim + kg_weight * kg_sim

    # 4. Top-K 鄰居選取
    _, topk_indices = torch.topk(combined_sim, k=k, dim=1)
    source_nodes = torch.arange(N, device=device).repeat_interleave(k)
    target_nodes = topk_indices.reshape(-1)

    edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    # 記錄統計
    kg_edges = (kg_sim[edge_index[0], edge_index[1]] > 0).sum().item()
    total_edges = edge_index.shape[1]
    logger.info(
        f"🕸️ 圖建構完成: {total_edges} 條邊, "
        f"其中 {kg_edges} 條受知識圖譜影響 ({kg_edges/total_edges*100:.1f}%)"
    )

    return edge_index
