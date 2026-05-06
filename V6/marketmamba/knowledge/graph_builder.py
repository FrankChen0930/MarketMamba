"""
MarketMamba V6 — Knowledge Graph Builder
==========================================
Builds the hybrid edge set for GATv2 cross-stock attention.

Four edge types, combined into a single (edge_index, edge_attr) cache:
  1. TWSE Sector     : static, TWSE industry classification (weight 0.5)
  2. Conglomerate    : manual table of major groups (weight 0.8)
  3. TPEX Supply Chain: crawled from TPEX industry chain platform (weight 0.6)
  4. Rolling Correlation: 60-day Pearson correlation > threshold (weight dynamic)

Output: PROCESSED_DIR / knowledge_graph_cache.npz
  - stock_ids   : (N,) array of stock ID strings
  - edge_index  : (2, E) int32 array
  - edge_attr   : (E,) float32 array of edge weights

The cache is loaded once per training run and stays on CPU until moved to GPU.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from marketmamba.config import (
    KG_CACHE_PATH,
    KG_CORR_THRESHOLD,
    KG_CORR_WINDOW,
    KG_EDGE_WEIGHTS,
    MAX_NEIGHBORS_GAT,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketMamba/6.0)"}


# ============================================================
# Edge Type 1: TWSE Sector (Static)
# ============================================================

def build_sector_edges(
    df_universe: pd.DataFrame,   # must have [stock_id, industry_category]
    stock_ids:   list[str],
    weight:      float = KG_EDGE_WEIGHTS["twse_sector"],
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Connect stocks in the same TWSE industry category.
    Returns (edges, weights) as lists of (src_idx, dst_idx) tuples.
    """
    id_to_idx = {sid: i for i, sid in enumerate(stock_ids)}
    sector_map: dict[str, list[int]] = {}

    for _, row in df_universe.iterrows():
        sid = str(row["stock_id"])
        sec = str(row.get("industry_category", "Unknown"))
        if sid not in id_to_idx:
            continue
        sector_map.setdefault(sec, []).append(id_to_idx[sid])

    edges, weights = [], []
    for sector, indices in sector_map.items():
        if len(indices) < 2:
            continue
        # Cap: each stock connects to at most MAX_SECTOR_NEIGHBORS in same sector
        # (prevents O(N²) blowup for large sectors like "其他")
        MAX_SECTOR_NEIGHBORS = 15
        if len(indices) <= MAX_SECTOR_NEIGHBORS + 1:
            # Small sector: full connectivity
            for i in indices:
                for j in indices:
                    if i != j:
                        edges.append((i, j))
                        weights.append(weight)
        else:
            # Large sector: each node connects to random subset
            import random
            for i in indices:
                neighbors = random.sample([j for j in indices if j != i],
                                          min(MAX_SECTOR_NEIGHBORS, len(indices) - 1))
                for j in neighbors:
                    edges.append((i, j))
                    weights.append(weight)

    logger.info(f"Sector edges: {len(edges)} (from {len(sector_map)} sectors)")
    return edges, weights


# ============================================================
# Edge Type 2: Conglomerate (Manual Table)
# ============================================================

# Major Taiwan conglomerate groups
# Format: group_name → list of stock_ids
CONGLOMERATE_TABLE: dict[str, list[str]] = {
    "鴻海集團":    ["2317", "2354", "2353", "6005"],
    "台積電生態圈": ["2330", "2379", "2308", "3034", "2357"],
    "聯發科生態圈": ["2454", "3443", "3711", "6770"],
    "統一集團":    ["1216", "9904", "2912", "5903"],
    "遠東集團":    ["2401", "2遠東新世紀", "2845", "4904"],
    "富邦集團":    ["2881", "2882", "2883", "5880"],
    "國泰集團":    ["2882", "2884", "9910"],
    "台塑集團":    ["1301", "1303", "1326", "6505"],
    "友達光電":    ["2409", "3481", "3532"],
    "廣達集團":    ["2382", "3376"],
}


def build_conglomerate_edges(
    stock_ids: list[str],
    weight:    float = KG_EDGE_WEIGHTS["conglomerate"],
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Connect stocks within the same conglomerate.
    These edges have the highest default weight (0.8) as group relationships
    are the strongest structural signal.
    """
    id_to_idx = {sid: i for i, sid in enumerate(stock_ids)}
    edges, weights = [], []

    for group, members in CONGLOMERATE_TABLE.items():
        indices = [id_to_idx[sid] for sid in members if sid in id_to_idx]
        if len(indices) < 2:
            continue
        for i in indices:
            for j in indices:
                if i != j:
                    edges.append((i, j))
                    weights.append(weight)

    logger.info(f"Conglomerate edges: {len(edges)}")
    return edges, weights


# ============================================================
# Edge Type 3: TPEX Supply Chain Crawler
# ============================================================

def crawl_tpex_supply_chain(
    tse_ids:    list[str],
    cache_path: Path | None = None,
    weight:     float = KG_EDGE_WEIGHTS["supply_chain"],
) -> tuple[list[tuple], list[str], list[str]]:
    """
    Crawl TPEX Industry Chain platform to get supply-chain relationships.
    URL: https://ic.tpex.org.tw/

    Returns:
        (edges_raw, upstream_ids, downstream_ids)
        edges_raw: list of (upstream_id, downstream_id) string pairs

    Note: TPEX platform requires stock code lookup, not bulk download.
    We crawl a representative set and cache the results quarterly.
    """
    if cache_path is None:
        cache_path = PROCESSED_DIR / "tpex_supply_chain.parquet"

    if cache_path.exists():
        logger.info(f"Loading TPEX supply chain from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        pairs = list(zip(df["upstream_id"], df["downstream_id"]))
        return pairs

    logger.info("Crawling TPEX supply chain data (this may take a few minutes)...")
    pairs = []
    base_url = "https://ic.tpex.org.tw/introduce.php"

    # We query a subset of large-cap stocks that are known to be in supply chains
    priority_ids = [sid for sid in tse_ids if sid in [
        "2330", "2317", "2454", "2382", "2357", "2379", "2308",
        "3034", "2308", "2303", "2337", "2344", "3711", "2409",
    ]]

    for sid in priority_ids:
        try:
            resp = requests.get(
                base_url,
                params={"stock_code": sid},
                headers=HEADERS,
                timeout=10,
            )
            if resp.status_code == 200:
                extracted = _parse_tpex_html(resp.text, sid)
                pairs.extend(extracted)
            time.sleep(1.0)  # polite crawl
        except Exception as e:
            logger.debug(f"TPEX crawl skipped for {sid}: {e}")

    if pairs:
        df = pd.DataFrame(pairs, columns=["upstream_id", "downstream_id"])
        df.to_parquet(cache_path)
        logger.info(f"TPEX supply chain: {len(pairs)} pairs cached → {cache_path}")
    else:
        logger.warning("TPEX crawl returned no results. Supply chain edges will be skipped.")

    return pairs


def _parse_tpex_html(html: str, source_id: str) -> list[tuple[str, str]]:
    """
    Parse TPEX supply chain HTML to extract related stock IDs.
    The platform shows upstream/downstream supplier tables with stock codes.
    """
    import re
    # Look for 4-digit stock codes in the HTML
    codes = re.findall(r'\b(\d{4})\b', html)
    codes = [c for c in codes if c != source_id and c.startswith(("1", "2", "3", "4", "5", "6"))]
    codes = list(set(codes))[:20]  # cap at 20 relations per stock
    return [(source_id, c) for c in codes]


def build_supply_chain_edges(
    supply_pairs: list[tuple[str, str]],
    stock_ids:    list[str],
    weight:       float = KG_EDGE_WEIGHTS["supply_chain"],
) -> tuple[list[tuple[int, int]], list[float]]:
    """Convert supply chain string pairs to integer edge index."""
    id_to_idx = {sid: i for i, sid in enumerate(stock_ids)}
    edges, weights = [], []
    for up_id, down_id in supply_pairs:
        if up_id in id_to_idx and down_id in id_to_idx:
            i, j = id_to_idx[up_id], id_to_idx[down_id]
            edges.append((i, j))
            edges.append((j, i))   # bidirectional
            weights.extend([weight, weight])

    logger.info(f"Supply chain edges: {len(edges)}")
    return edges, weights


# ============================================================
# Edge Type 4: Rolling Correlation (Dynamic)
# ============================================================

def build_rolling_correlation_edges(
    df_prices:  pd.DataFrame,   # [Date, stock_id, Close]
    stock_ids:  list[str],
    window:     int   = KG_CORR_WINDOW,
    threshold:  float = KG_CORR_THRESHOLD,
    max_neighbors: int = MAX_NEIGHBORS_GAT,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Build dynamic correlation edges based on the most recent `window` trading days.

    For each stock pair with correlation > threshold, add a directed edge.
    Edge weight = the correlation value itself.

    Complexity: O(N^2) in stock count — acceptable for N~2000 with vectorized operations.
    """
    id_to_idx = {sid: i for i, sid in enumerate(stock_ids)}

    # Get latest `window` days of close prices
    df_prices = df_prices.copy()
    df_prices["Date"] = pd.to_datetime(df_prices["Date"])
    latest_dates = sorted(df_prices["Date"].unique())[-window:]
    df_window = df_prices[df_prices["Date"].isin(latest_dates)]

    # Pivot to (T, N) matrix
    pivot = df_window.pivot_table(index="Date", columns="stock_id", values="Close")
    pivot = pivot[[c for c in stock_ids if c in pivot.columns]]  # keep ordered

    # Log-returns for correlation (more stable than price levels)
    log_ret = np.log(pivot / pivot.shift(1)).dropna()

    # Vectorised correlation matrix
    if log_ret.shape[0] < 10 or log_ret.shape[1] < 2:
        logger.warning("Insufficient data for rolling correlation edges")
        return [], []

    corr_matrix = log_ret.corr(method="pearson").values  # (N', N')
    available_ids = list(log_ret.columns)
    avail_idx = {sid: id_to_idx.get(sid) for sid in available_ids if sid in id_to_idx}

    edges, weights = [], []
    local_avail = list(avail_idx.keys())

    for local_i, sid_i in enumerate(local_avail):
        global_i = avail_idx[sid_i]
        # Get correlation row, sort descending, pick top-k above threshold
        corr_row = corr_matrix[local_i]
        candidates = [
            (corr_row[local_j], avail_idx[sid_j])
            for local_j, sid_j in enumerate(local_avail)
            if local_j != local_i
            and not np.isnan(corr_row[local_j])
            and corr_row[local_j] > threshold
        ]
        candidates.sort(reverse=True)
        for corr_val, global_j in candidates[:max_neighbors]:
            edges.append((global_i, global_j))
            weights.append(float(corr_val))

    logger.info(
        f"Correlation edges: {len(edges)} "
        f"(threshold={threshold}, window={window}d, max_neighbors={max_neighbors})"
    )
    return edges, weights


# ============================================================
# Deduplication & Normalisation
# ============================================================

def _merge_and_deduplicate(
    all_edges:   list[tuple[int, int]],
    all_weights: list[float],
    n_stocks:    int,
    max_neighbors: int = MAX_NEIGHBORS_GAT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge edges from all sources:
      - If the same (i,j) pair appears multiple times, take the MAXIMUM weight
      - For each node, keep at most max_neighbors edges
    Returns (edge_index, edge_attr) as numpy arrays.
    """
    from collections import defaultdict
    pair_weight: dict[tuple[int, int], float] = defaultdict(float)
    for (i, j), w in zip(all_edges, all_weights):
        pair_weight[(i, j)] = max(pair_weight[(i, j)], w)

    # Per-node neighbour cap
    out_edges: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (i, j), w in pair_weight.items():
        out_edges[i].append((j, w))

    final_edges, final_weights = [], []
    for i in range(n_stocks):
        nbrs = sorted(out_edges[i], key=lambda x: -x[1])[:max_neighbors]
        for j, w in nbrs:
            final_edges.append((i, j))
            final_weights.append(w)

    if not final_edges:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.float32)

    edge_index = np.array(final_edges, dtype=np.int32).T   # (2, E)
    edge_attr  = np.array(final_weights, dtype=np.float32) # (E,)
    return edge_index, edge_attr


# ============================================================
# Main Builder
# ============================================================

def build_knowledge_graph(
    df_universe: pd.DataFrame,   # [stock_id, industry_category, market]
    df_prices:   pd.DataFrame,   # [Date, stock_id, Close] — for rolling correlation
    force_rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build and cache the full hybrid knowledge graph.

    Args:
        df_universe  : stock universe with sector labels
        df_prices    : historical price data for rolling correlation
        force_rebuild: ignore existing cache

    Returns:
        (edge_index, edge_attr, stock_ids)
        edge_index : (2, E) int32
        edge_attr  : (E,)   float32
        stock_ids  : ordered list of stock ID strings (index = node index)
    """
    if not force_rebuild and KG_CACHE_PATH.exists():
        logger.info(f"Loading KG from cache: {KG_CACHE_PATH}")
        data = np.load(KG_CACHE_PATH, allow_pickle=True)
        return data["edge_index"], data["edge_attr"], list(data["stock_ids"])

    logger.info("Building V6 Knowledge Graph from scratch...")

    # Ordered stock universe
    stock_ids = sorted(df_universe["stock_id"].astype(str).unique().tolist())
    n_stocks  = len(stock_ids)
    logger.info(f"Stock universe: {n_stocks} stocks")

    all_edges, all_weights = [], []

    # -- Edge type 1: Sector --
    e, w = build_sector_edges(df_universe, stock_ids)
    all_edges.extend(e); all_weights.extend(w)

    # -- Edge type 2: Conglomerate --
    e, w = build_conglomerate_edges(stock_ids)
    all_edges.extend(e); all_weights.extend(w)

    # -- Edge type 3: TPEX Supply Chain --
    try:
        supply_pairs = crawl_tpex_supply_chain(stock_ids)
        e, w = build_supply_chain_edges(supply_pairs, stock_ids)
        all_edges.extend(e); all_weights.extend(w)
    except Exception as ex:
        logger.warning(f"Supply chain crawl failed, skipping: {ex}")

    # -- Edge type 4: Rolling Correlation --
    try:
        e, w = build_rolling_correlation_edges(df_prices, stock_ids)
        all_edges.extend(e); all_weights.extend(w)
    except Exception as ex:
        logger.warning(f"Rolling correlation failed, skipping: {ex}")

    # -- Merge & Deduplicate --
    edge_index, edge_attr = _merge_and_deduplicate(all_edges, all_weights, n_stocks)

    # -- Save Cache --
    KG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        KG_CACHE_PATH,
        stock_ids=np.array(stock_ids),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    logger.info(
        f"KG saved: {n_stocks} nodes, {edge_index.shape[1]} edges → {KG_CACHE_PATH}"
    )
    return edge_index, edge_attr, stock_ids


def update_correlation_edges(
    df_prices:     pd.DataFrame,
    force_rebuild: bool = False,
) -> None:
    """
    Lightweight weekly update: only refresh rolling correlation edges.
    Sector and supply chain edges are expensive to rebuild; update them quarterly.
    """
    if not KG_CACHE_PATH.exists():
        logger.warning("No KG cache found. Run build_knowledge_graph() first.")
        return

    data = np.load(KG_CACHE_PATH, allow_pickle=True)
    stock_ids  = list(data["stock_ids"])
    edge_index = data["edge_index"]
    edge_attr  = data["edge_attr"]

    # Remove existing correlation edges (identified by weight < sector_weight)
    # Strategy: rebuild only correlation layer and merge back
    logger.info("Updating rolling correlation edges in KG...")
    corr_edges, corr_weights = build_rolling_correlation_edges(df_prices, stock_ids)

    # Re-merge with existing non-correlation edges (keep only non-correlation part)
    # Simple heuristic: keep edges with weight >= 0.5 (sector/conglomerate/supply)
    keep_mask = edge_attr >= 0.5
    kept_idx    = edge_index[:, keep_mask]
    kept_attr   = edge_attr[keep_mask]

    all_edges   = list(zip(kept_idx[0].tolist(), kept_idx[1].tolist())) + corr_edges
    all_weights = kept_attr.tolist() + corr_weights

    edge_index_new, edge_attr_new = _merge_and_deduplicate(
        all_edges, all_weights, len(stock_ids)
    )
    np.savez_compressed(
        KG_CACHE_PATH,
        stock_ids=np.array(stock_ids),
        edge_index=edge_index_new,
        edge_attr=edge_attr_new,
    )
    logger.info(f"KG correlation edges updated: {edge_index_new.shape[1]} total edges")
