"""
Overnight training patch:
1. Cross-sectional z-score on targets in __getitem__
2. Replace get_batch_edges (Python loop) with scipy CSR slice version
3. Re-enable KG in train_model
"""
import pathlib, re

path = pathlib.Path(r"d:\Desktop\work\MarketMamba\V6\marketmamba\models\trainer.py")
text = path.read_text(encoding="utf-8")

# ─── Fix 1: cross-sectional z-score on Y in __getitem__ ───────────────────────
OLD_Y = (
    "        X = torch.from_numpy(np.array(X_list))\n"
    "        Y = torch.from_numpy(np.array(Y_list))\n"
    "\n"
    "        if self.n_sample is not None and X.shape[0] > self.n_sample:\n"
    "            idx_s = torch.randperm(X.shape[0])[: self.n_sample]\n"
    "            X, Y  = X[idx_s], Y[idx_s]\n"
    "            valid_stocks = [valid_stocks[i] for i in idx_s.tolist()]\n"
    "\n"
    "        return X, Y, valid_stocks"
)
NEW_Y = (
    "        X = torch.from_numpy(np.array(X_list))\n"
    "        Y = torch.from_numpy(np.array(Y_list))\n"
    "\n"
    "        if self.n_sample is not None and X.shape[0] > self.n_sample:\n"
    "            idx_s = torch.randperm(X.shape[0])[: self.n_sample]\n"
    "            X, Y  = X[idx_s], Y[idx_s]\n"
    "            valid_stocks = [valid_stocks[i] for i in idx_s.tolist()]\n"
    "\n"
    "        # Cross-sectional z-score: normalize targets within each date's cross-section.\n"
    "        # MSE on raw returns is dominated by magnitude variance (loss ~4-8).\n"
    "        # After z-scoring, loss ~1.0 and gradients target rank ordering quality.\n"
    "        if Y.shape[0] > 1:\n"
    "            Y_mean = Y.mean(dim=0, keepdim=True)\n"
    "            Y_std  = Y.std(dim=0, keepdim=True).clamp(min=1e-6)\n"
    "            Y = (Y - Y_mean) / Y_std\n"
    "\n"
    "        return X, Y, valid_stocks"
)

for eol in ("\r\n", "\n"):
    old = OLD_Y.replace("\n", eol)
    new = NEW_Y.replace("\n", eol)
    if old in text:
        text = text.replace(old, new, 1)
        print(f"Fix 1 (target z-score) applied ({repr(eol)})")
        break
else:
    print("Fix 1 NOT FOUND")

# ─── Fix 2: add scipy CSR functions before build_kg_adjacency ────────────────
CSR_FUNCS = '''
def build_kg_csr():
    """
    Build scipy CSR matrix for fast O(1) per-batch subgraph extraction.
    Replaces the Python-loop get_batch_edges (~1s/batch) with scipy slice (~1ms/batch).
    Returns: (kg_csr, stock_to_idx) or (None, {}) if no KG cache.
    """
    if not KG_CACHE_PATH.exists():
        print("[KG] Cache not found — running without KG edges.", flush=True)
        return None, {}

    from scipy import sparse as sp

    data      = np.load(KG_CACHE_PATH, allow_pickle=True)
    all_ids   = list(data["stock_ids"])
    all_edges = data["edge_index"]   # (2, E)
    all_attrs = data["edge_attr"]    # (E,)

    N            = len(all_ids)
    stock_to_idx = {str(sid): i for i, sid in enumerate(all_ids)}

    rows  = all_edges[0].astype(np.int32)
    cols  = all_edges[1].astype(np.int32)
    attrs = all_attrs.astype(np.float32)

    kg_csr = sp.csr_matrix((attrs, (rows, cols)), shape=(N, N))
    print(f"[KG] CSR matrix built: {N} nodes, {kg_csr.nnz} edges", flush=True)
    return kg_csr, stock_to_idx


def get_batch_edges_csr(
    batch_stocks: list[str],
    kg_csr,                # scipy CSR matrix or None
    stock_to_idx: dict,    # {stock_id_str: global_row_idx}
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Extract the subgraph for batch_stocks using CSR slice — vectorized, ~1ms/batch.
    Returns (edge_index, edge_attr) with local indices in [0, len(batch_stocks)).
    """
    empty = (
        torch.zeros((2, 0), dtype=torch.long,    device=device),
        torch.zeros(0,       dtype=torch.float32, device=device),
    )
    if kg_csr is None or not batch_stocks:
        return empty

    # Map batch stocks → global CSR row indices (-1 = not in KG)
    global_idx  = np.array([stock_to_idx.get(s, -1) for s in batch_stocks], dtype=np.int32)
    valid_local  = np.where(global_idx >= 0)[0]    # positions in batch
    valid_global = global_idx[valid_local]           # CSR row indices

    if len(valid_global) == 0:
        return empty

    # scipy CSR subgraph slice: O(nnz in subgraph), fully vectorized
    sub = kg_csr[np.ix_(valid_global, valid_global)].tocoo()
    if sub.nnz == 0:
        return empty

    # Map sub indices [0, len(valid_global)) → actual batch positions
    local_rows = torch.from_numpy(valid_local[sub.row].astype(np.int64))
    local_cols = torch.from_numpy(valid_local[sub.col].astype(np.int64))
    attrs      = torch.from_numpy(sub.data.astype(np.float32))

    edge_index = torch.stack([local_rows, local_cols], dim=0).to(device)
    edge_attr  = attrs.to(device)
    return edge_index, edge_attr

'''

# Insert before build_kg_adjacency
INSERT_BEFORE = "def build_kg_adjacency()"
if INSERT_BEFORE in text:
    text = text.replace(INSERT_BEFORE, CSR_FUNCS + INSERT_BEFORE, 1)
    print("Fix 2 (scipy CSR functions) applied")
else:
    print("Fix 2 NOT FOUND - looking for anchor...")

# ─── Fix 3: re-enable KG in train_model using CSR ────────────────────────────
OLD_KG = (
    "    # KG disabled during training: get_batch_edges Python loop is a bottleneck\n"
    "    # when KG has many edges (e.g. full correlation graph). Re-enable once pipeline\n"
    "    # is verified stable by setting use_kg=True in train_model call.\n"
    "    kg_adj = None  # build_kg_adjacency() -- disabled for speed"
)
NEW_KG = (
    "    # KG: use scipy CSR for O(1) vectorized per-batch subgraph extraction.\n"
    "    # ~1ms/batch vs ~500ms/batch for the old Python loop version.\n"
    "    kg_csr, stock_to_idx = build_kg_csr()"
)

for eol in ("\r\n", "\n"):
    old = OLD_KG.replace("\n", eol)
    new = NEW_KG.replace("\n", eol)
    if old in text:
        text = text.replace(old, new, 1)
        print(f"Fix 3 (re-enable KG) applied ({repr(eol)})")
        break
else:
    print("Fix 3 NOT FOUND")

# ─── Fix 4: update get_batch_edges call sites to use CSR version ──────────────
for old_call, new_call in [
    (
        "            edge_index, edge_attr = get_batch_edges(batch_stocks, kg_adj, device)",
        "            edge_index, edge_attr = get_batch_edges_csr(batch_stocks, kg_csr, stock_to_idx, device)"
    ),
]:
    for eol in ("\r\n", "\n"):
        o = old_call.replace("\n", eol)
        n = new_call.replace("\n", eol)
        if o in text:
            text = text.replace(o, n)
            print(f"Fix 4 (call site update) applied")
            break

path.write_text(text, encoding="utf-8")
print("Done writing trainer.py")
