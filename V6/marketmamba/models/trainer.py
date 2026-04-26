"""
MarketMamba V6 — Trainer (Fixed)
==================================
Critical fixes vs initial version:
  BUG FIX 1: TemporalCrossSectionDataset is now LAZY — tensors built on-demand
              in __getitem__, NOT pre-computed in __init__.
              Pre-computing 3000 dates × 2000 stocks would require ~280 GB RAM.
  BUG FIX 2: DataLoader uses batch_size=1 + _identity_collate to correctly
              handle full-cross-section batches without adding an extra batch dim.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from marketmamba.config import (
    AMP_ENABLED,
    EARLY_STOP,
    EPOCHS,
    FEATURE_COLS,
    GRAD_CLIP_NORM,
    INPUT_DIM,
    KG_CACHE_PATH,
    LOSS_WEIGHTS,
    LR,
    MODELS_DIR,
    N_SAMPLE_TRAIN,
    PRED_HORIZONS,
    SEQ_LEN,
    WARMUP_PCT,
)
from marketmamba.models.architecture import MarketMambaV6

logger = logging.getLogger(__name__)

TARGET_COLS = [f"Alpha_{h}d" for h in PRED_HORIZONS]


# ============================================================
# Dataset — LAZY LOADING (critical fix)
# ============================================================

class TemporalCrossSectionDataset(Dataset):
    """
    Correct cross-sectional dataset for Mamba + GAT training.

    DESIGN: Tensors are built ON-DEMAND in __getitem__, NOT pre-computed.
    Pre-computing everything would require ~280 GB RAM for 3000 days × 2000 stocks.

    Each __getitem__ returns ONE FULL TRADING DAY's cross-section:
      X: (N_stocks, SEQ_LEN, INPUT_DIM)
      Y: (N_stocks, 3)   — [Alpha_5d, Alpha_20d, Alpha_60d]

    The df is stored as a reference (not copied per call), with a pre-built
    index for O(1) stock/date lookup.
    """

    def __init__(
        self,
        df:       pd.DataFrame,
        dates:    list[str],
        seq_len:  int = SEQ_LEN,
        mode:     str = "train",
        n_sample: Optional[int] = N_SAMPLE_TRAIN,
    ):
        self.seq_len  = seq_len
        self.mode     = mode
        self.n_sample = n_sample if mode == "train" else None

        # Prepare indexed DataFrame (store reference, don't copy)
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["stock_id", "Date"]).reset_index(drop=True)

        self._df = df

        # Global date index — normalize to pd.Timestamp so dict keys are resolution-independent.
        # pandas 2.x may return datetime64[us] from unique() but datetime64[ns] from .values,
        # causing hash mismatches. pd.Timestamp is always compatible with both.
        all_dates         = sorted(pd.Timestamp(d) for d in df["Date"].unique())
        self._all_dates   = all_dates
        self._date_to_idx = {d: i for i, d in enumerate(all_dates)}   # keys: pd.Timestamp

        date_stocks = df.groupby("Date")["stock_id"].apply(list).to_dict()
        # Normalize groupby keys (pandas 2.x returns Timestamp, older versions differ)
        date_stocks = {pd.Timestamp(k): v for k, v in date_stocks.items()}
        self._date_stocks = {dt: [str(s) for s in sl] for dt, sl in date_stocks.items()}

        # Valid dates
        self.valid_dates = []
        for ds in dates:
            dt     = pd.Timestamp(ds)
            dt_idx = self._date_to_idx.get(dt)
            if dt_idx is None or dt_idx < seq_len:
                continue
            if dt in self._date_stocks and len(self._date_stocks[dt]) > 0:
                self.valid_dates.append(ds)

        # Pre-index per-stock numpy arrays (eliminates pandas in __getitem__)
        # Memory: ~1.9 GB for 1754 stocks x 5500 dates x 49 features x float32
        n_stocks_total = df["stock_id"].nunique()
        print(f"[Dataset init] pre-indexing {n_stocks_total} stocks...", flush=True)
        self._stock_index: dict[str, dict] = {}
        for sid, grp in df.groupby("stock_id"):
            grp  = grp.sort_values("Date")
            # Convert each date to pd.Timestamp for consistent dict lookup
            didx = np.array(
                [self._date_to_idx[pd.Timestamp(d)] for d in grp["Date"].values],
                dtype=np.int32,
            )
            self._stock_index[str(sid)] = {
                "date_idx": didx,
                "feats":    grp[FEATURE_COLS].values.astype(np.float32),
                "targets":  grp[TARGET_COLS].values.astype(np.float32),
            }

        print(f"[Dataset init] {len(self.valid_dates)} valid days | {len(self._stock_index)} stocks pre-indexed", flush=True)

    def __len__(self) -> int:
        return len(self.valid_dates)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, list]:
        date_str  = self.valid_dates[idx]
        dt        = pd.Timestamp(date_str)
        dt_idx    = self._date_to_idx[dt]
        win_start = max(0, dt_idx - self.seq_len + 1)

        stocks_today = self._date_stocks.get(dt, [])

        X_list, Y_list, valid_stocks = [], [], []
        for sid in stocks_today:
            stock = self._stock_index.get(sid)
            if stock is None:
                continue
            didx = stock["date_idx"]

            # O(n_dates_per_stock) numpy mask -- no pandas
            mask = (didx >= win_start) & (didx <= dt_idx)
            n    = int(mask.sum())
            if n < int(self.seq_len * 0.8):
                continue

            feats = stock["feats"][mask]
            if n < self.seq_len:
                pad   = np.zeros((self.seq_len - n, INPUT_DIM), dtype=np.float32)
                feats = np.vstack([pad, feats])
            else:
                feats = feats[-self.seq_len:]

            target_mask = didx == dt_idx
            if not target_mask.any():
                continue
            targets = stock["targets"][target_mask][-1]

            X_list.append(feats)
            Y_list.append(targets)
            valid_stocks.append(sid)

        if not X_list:
            return (
                torch.zeros(1, self.seq_len, INPUT_DIM, dtype=torch.float32),
                torch.zeros(1, len(PRED_HORIZONS), dtype=torch.float32),
                [],
            )

        X = torch.from_numpy(np.array(X_list))
        Y = torch.from_numpy(np.array(Y_list))

        if self.n_sample is not None and X.shape[0] > self.n_sample:
            idx_s = torch.randperm(X.shape[0])[: self.n_sample]
            X, Y  = X[idx_s], Y[idx_s]
            valid_stocks = [valid_stocks[i] for i in idx_s.tolist()]

        return X, Y, valid_stocks


# ============================================================
# DataLoader Collate Function (critical fix)
# ============================================================

def _identity_collate(batch):
    """
    Custom collate for full-cross-section batches.
    batch = [(X, Y, stock_ids)] with batch_size=1.
    """
    assert len(batch) == 1, "batch_size must be 1 for cross-section dataset"
    return batch[0]  # returns (X, Y, stock_ids)


def make_dataloader(dataset: TemporalCrossSectionDataset, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for TemporalCrossSectionDataset.
    Always uses batch_size=1 + identity collate.
    """
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,          # 0 is safest for large per-sample tensors
        collate_fn=_identity_collate,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Loss Functions
# ============================================================

def listnet_loss(pred: Tensor, target: Tensor, eps: float = 1e-10) -> Tensor:
    """
    ListNet ranking loss (top-1 probability).
    Optimises the KL divergence between softmax distributions.
    Fully differentiable; no Heaviside or step functions.
    """
    pred_prob   = F.softmax(pred,   dim=0)
    target_prob = F.softmax(target, dim=0)
    return -(target_prob * torch.log(pred_prob + eps)).sum()


def multi_horizon_loss(
    preds:   Tensor,    # (N, 3) — columns: [5d, 20d, 60d]
    targets: Tensor,    # (N, 3)
    weights: dict = LOSS_WEIGHTS,
) -> tuple[Tensor, dict]:
    """
    Combined multi-horizon MSE + ListNet ranking loss.
    Returns (total_loss, breakdown_dict).
    """
    p5d,  p20d,  p60d  = preds[:,  0], preds[:,  1], preds[:,  2]
    t5d,  t20d,  t60d  = targets[:, 0], targets[:, 1], targets[:, 2]

    # Skip samples where target is NaN
    valid_5d  = ~torch.isnan(t5d)
    valid_20d = ~torch.isnan(t20d)
    valid_60d = ~torch.isnan(t60d)

    mse_5d  = F.mse_loss(p5d[valid_5d],   t5d[valid_5d])   if valid_5d.any()  else torch.tensor(0.0)
    mse_20d = F.mse_loss(p20d[valid_20d], t20d[valid_20d]) if valid_20d.any() else torch.tensor(0.0)
    mse_60d = F.mse_loss(p60d[valid_60d], t60d[valid_60d]) if valid_60d.any() else torch.tensor(0.0)
    ln_20d  = listnet_loss(p20d[valid_20d], t20d[valid_20d]) if valid_20d.sum() > 1 else torch.tensor(0.0)

    total = (
        weights["mse_20d"]     * mse_20d
        + weights["mse_5d"]    * mse_5d
        + weights["mse_60d"]   * mse_60d
        + weights["listnet_20d"] * ln_20d
    )

    breakdown = {
        "loss_total":  total.item(),
        "mse_20d":     mse_20d.item(),
        "mse_5d":      mse_5d.item(),
        "mse_60d":     mse_60d.item(),
        "listnet_20d": ln_20d.item(),
    }
    return total, breakdown


# ============================================================
# Knowledge Graph Edge Loader
# ============================================================

def load_kg_edges(
    stock_ids: list[str],
    device:    torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Load pre-computed KG edges for the given stock universe.
    Returns (edge_index, edge_attr) — empty tensors if cache not found.
    """
    if not KG_CACHE_PATH.exists():
        logger.warning(
            f"KG cache not found at {KG_CACHE_PATH}. "
            "No cross-stock edges will be used."
        )
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
        )

    data       = np.load(KG_CACHE_PATH, allow_pickle=True)
    all_ids    = list(data["stock_ids"])
    all_edges  = data["edge_index"]   # (2, E_total)
    all_attrs  = data["edge_attr"]    # (E_total,)

    id_to_local = {sid: i for i, sid in enumerate(stock_ids)}

    rows, cols, attrs = [], [], []
    for i in range(all_edges.shape[1]):
        src = all_ids[all_edges[0, i]]
        dst = all_ids[all_edges[1, i]]
        if src in id_to_local and dst in id_to_local:
            rows.append(id_to_local[src])
            cols.append(id_to_local[dst])
            attrs.append(all_attrs[i])

    if not rows:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
        )

    edge_index = torch.tensor([rows, cols], dtype=torch.long,    device=device)
    edge_attr  = torch.tensor(attrs,        dtype=torch.float32, device=device)
    logger.info(f"KG: {edge_index.shape[1]} edges loaded for {len(stock_ids)} stocks")
    return edge_index, edge_attr


def build_kg_adjacency() -> dict | None:
    """
    Pre-build KG adjacency dict for fast per-batch edge lookup.
    Returns: {src_stock_id: [(dst_stock_id, edge_attr), ...]} or None if no KG.
    """
    if not KG_CACHE_PATH.exists():
        return None
    data      = np.load(KG_CACHE_PATH, allow_pickle=True)
    all_ids   = list(data["stock_ids"])
    all_edges = data["edge_index"]
    all_attrs = data["edge_attr"]
    adj: dict[str, list] = {}
    for i in range(all_edges.shape[1]):
        src = all_ids[all_edges[0, i]]
        dst = all_ids[all_edges[1, i]]
        adj.setdefault(src, []).append((dst, float(all_attrs[i])))
    logger.info(f"KG adjacency built: {len(all_ids)} nodes, {all_edges.shape[1]} edges")
    return adj


def get_batch_edges(
    batch_stocks: list[str],
    kg_adj:       dict | None,
    device:       torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Build a LOCAL edge_index for the current batch from the pre-built KG adj dict.
    All indices are in [0, len(batch_stocks)) — no out-of-bounds possible.
    """
    empty = (
        torch.zeros((2, 0), dtype=torch.long,    device=device),
        torch.zeros(0,       dtype=torch.float32, device=device),
    )
    if kg_adj is None or not batch_stocks:
        return empty

    local = {sid: i for i, sid in enumerate(batch_stocks)}
    rows, cols, attrs = [], [], []
    for src in batch_stocks:
        for dst, attr in kg_adj.get(src, []):
            if dst in local:
                rows.append(local[src])
                cols.append(local[dst])
                attrs.append(attr)
    if not rows:
        return empty

    edge_index = torch.tensor([rows, cols], dtype=torch.long,    device=device)
    edge_attr  = torch.tensor(attrs,        dtype=torch.float32, device=device)
    return edge_index, edge_attr


# ============================================================
# IC Metric
# ============================================================

def compute_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation (cross-sectional IC)."""
    from scipy.stats import spearmanr
    mask = ~(np.isnan(pred) | np.isnan(target))
    if mask.sum() < 5:
        return 0.0
    corr, _ = spearmanr(pred[mask], target[mask])
    return float(corr) if not np.isnan(corr) else 0.0


# ============================================================
# Training History
# ============================================================

@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_ic:     list[float] = field(default_factory=list)
    lr:         list[float] = field(default_factory=list)
    breakdown:  list[dict]  = field(default_factory=list)

    @property
    def best_epoch(self) -> int:
        return int(np.argmin(self.val_loss)) + 1 if self.val_loss else 0

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss) if self.val_loss else float("inf")


# ============================================================
# Main Train Function
# ============================================================

def train_model(
    df:              pd.DataFrame,
    train_dates:     list[str],
    val_dates:       list[str],
    epochs:          int   = EPOCHS,
    lr:              float = LR,
    early_stop:      int   = EARLY_STOP,
    checkpoint_name: str   = "v6_best.pt",
    device_str:      str   = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[MarketMambaV6, TrainingHistory]:
    """
    Train MarketMamba V6 on a (train_dates, val_dates) split.
    Walk-Forward compatible: accepts explicit date lists.

    Returns: (best_model, history)
    """
    device = torch.device(device_str)
    logger.info(
        f"Training on {device} | train={len(train_dates)} days | val={len(val_dates)} days"
    )

    # Datasets & DataLoaders (lazy loading)
    train_ds = TemporalCrossSectionDataset(df, train_dates, mode="train")
    val_ds   = TemporalCrossSectionDataset(df, val_dates,   mode="val")
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds,   shuffle=False)

    # Model
    model = MarketMambaV6().to(device)
    logger.info(f"Model parameters: {model.n_parameters:,}")

    # Optimiser + Scheduler
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps  = epochs * max(len(train_loader), 1)
    scheduler    = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=WARMUP_PCT,
        anneal_strategy="cos",
    )
    scaler = GradScaler('cuda', enabled=AMP_ENABLED and device_str == "cuda")

    # KG disabled during training: get_batch_edges Python loop is a bottleneck
    # when KG has many edges (e.g. full correlation graph). Re-enable once pipeline
    # is verified stable by setting use_kg=True in train_model call.
    kg_adj = None  # build_kg_adjacency() -- disabled for speed

    history  = TrainingHistory()
    best_val = float("inf")
    no_impr  = 0
    ckpt_path = MODELS_DIR / checkpoint_name

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────
        model.train()
        train_losses = []
        epoch_bd: dict[str, list] = {k: [] for k in
                                      ["mse_20d", "mse_5d", "mse_60d", "listnet_20d"]}

        for batch_idx, (X, Y, batch_stocks) in enumerate(train_loader):
            if X.shape[0] <= 1:   # skip empty / degenerate cross-sections
                continue

            # Timing diagnostic for first batch of first epoch
            if epoch == 1 and batch_idx == 0:
                print(f"  [diag] First batch: X={tuple(X.shape)} stocks={len(batch_stocks)} | {time.time()-t0:.1f}s since epoch start", flush=True)

            X, Y = X.to(device), Y.to(device)
            edge_index, edge_attr = get_batch_edges(batch_stocks, kg_adj, device)

            if epoch == 1 and batch_idx == 0:
                print(f"  [diag] KG edges: {edge_index.shape[1]}", flush=True)

            optimizer.zero_grad()
            with autocast('cuda', enabled=AMP_ENABLED and device_str == "cuda"):
                preds       = model(X, edge_index, edge_attr)
                loss, brkdn = multi_horizon_loss(preds, Y)

            if epoch == 1 and batch_idx == 0:
                print(f"  [diag] Forward OK. loss={loss.item():.4f} | batch took {time.time()-t0:.1f}s total", flush=True)

            # Check for NaN loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                if epoch == 1 and batch_idx < 5:
                    print(f'  [WARN] NaN/Inf loss at batch {batch_idx}: {loss.item()}', flush=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(brkdn["loss_total"])
            for k in epoch_bd:
                epoch_bd[k].append(brkdn.get(k, 0.0))

            # Progress every 200 batches
            if (batch_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                total_b = len(train_loader)
                eta     = elapsed / (batch_idx + 1) * (total_b - batch_idx - 1)
                print(f"  Ep {epoch:03d} [{batch_idx+1}/{total_b}] loss={float(np.mean(train_losses)):.5f} | {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

        if not train_losses:
            logger.warning(f"Epoch {epoch}: no valid training batches")
            continue

        avg_train = float(np.mean(train_losses))

        # ── Validate ────────────────────────────────────────
        model.eval()
        val_losses, val_ics = [], []
        with torch.no_grad():
            for X, Y, batch_stocks in val_loader:
                if X.shape[0] <= 1:
                    continue
                X, Y = X.to(device), Y.to(device)
                edge_index, edge_attr = get_batch_edges(batch_stocks, kg_adj, device)
                with autocast('cuda', enabled=AMP_ENABLED and device_str == "cuda"):
                    preds      = model(X, edge_index, edge_attr)
                    loss, _    = multi_horizon_loss(preds, Y)
                val_losses.append(loss.item())
                ic = compute_ic(
                    preds[:, 1].cpu().numpy(),
                    Y[:, 1].cpu().numpy(),
                )
                val_ics.append(ic)

        avg_val  = float(np.mean(val_losses)) if val_losses else float("inf")
        avg_ic   = float(np.mean(val_ics))    if val_ics   else 0.0
        cur_lr   = scheduler.get_last_lr()[0]

        history.train_loss.append(avg_train)
        history.val_loss.append(avg_val)
        history.val_ic.append(avg_ic)
        history.lr.append(cur_lr)
        history.breakdown.append({k: float(np.mean(v)) for k, v in epoch_bd.items()})

        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={avg_train:.5f} val={avg_val:.5f} IC={avg_ic:+.4f} "
            f"lr={cur_lr:.2e} | {time.time()-t0:.1f}s"
        )

        if avg_val < best_val:
            best_val = avg_val
            no_impr  = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "val_loss": avg_val, "val_ic": avg_ic, "history": history},
                ckpt_path,
            )
            logger.info(f"  ✅ Checkpoint saved → {ckpt_path.name}")
        else:
            no_impr += 1
            if no_impr >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

    # Reload best weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    logger.info(
        f"Training done. Best epoch={history.best_epoch}, "
        f"val_loss={history.best_val_loss:.5f}"
    )
    return model, history
