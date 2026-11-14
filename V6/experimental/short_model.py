"""
Phase 2 步驟 2 — 單尺度短線模型（隔離副本）
================================================
B 已證明多尺度對 5d 冗餘，故短線模型砍成「單一時間分支」：
  Embedding → SingleScaleEncoder(一個 Mamba 分支, window=60) → GAT → gating 融合 → 2 頭(5d/10d)

目標工程：訓練時對每個 cross-section 的 target 做 per-date pct-rank 置中（±0.5），
          MSE 與 listnet 都吃這個 ranked target（跟 IC 指標對齊、解 MSE/listnet 打架）。

⚠️ 隔離副本：正式 marketmamba/ 一個字不動（用 runtime 指定 TARGET_COLS=[Alpha_5d,Alpha_10d]，
   跟 listnet_5d 那次同款）；V6.1 線上推論零影響。主輸出 (N,2)=[5d,10d]。
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from marketmamba.config import (
    AMP_ENABLED, D_MODEL, D_STATE, DROPOUT, GRAD_CLIP_NORM, LR, MODELS_DIR,
    N_HEADS_GAT, WARMUP_PCT,
)
from marketmamba.models.architecture import (
    FactorGroupedEmbedding, MambaStack, GraphAttentionLayer,
)
from marketmamba.models.trainer import (
    TemporalCrossSectionDataset, make_dataloader, build_kg_csr,
    get_batch_edges_csr, compute_ic, listnet_loss, TrainingHistory,
)

# 5d 為主、10d 次要
SHORT_WEIGHTS = {"mse_5d": 1.0, "mse_10d": 0.5, "listnet_5d": 0.5, "listnet_10d": 0.25}


# ============================================================
# 1. 單尺度 encoder：一個 Mamba 分支，取最後 window 步
# ============================================================
class SingleScaleEncoder(nn.Module):
    """取輸入最後 `window` 步 → MambaStack → 最後 hidden → norm。
    window≤200 時最後 window 步全為真實資料（dataset 要求 ≥202 天），不需 padding mask。"""

    def __init__(self, d_model: int = D_MODEL, d_state: int = D_STATE,
                 window: int = 60, n_layers: int = 3):
        super().__init__()
        self.window = window
        self.mamba = MambaStack(n_layers, d_model, d_state)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:        # x: (N, 252, d_model)
        h = self.mamba(x[:, -self.window:, :])[:, -1, :]   # (N, d_model)
        return self.norm(h)


# ============================================================
# 2. 短線模型：單尺度 + GAT + gating 融合 + 2 頭(5d/10d)
# ============================================================
class ShortModelV6(nn.Module):
    """
    Args:
        use_gat:
          True（預設，線上 `run_dual_inference.py` 走這條）：GAT + gating 融合。
          False（決策2 消融用）：完全跳過 GAT 與 gate，直接用 h_temporal。

          ⚠️ `use_gat=False` 的 state_dict **少了 graph_layer / gate / norm_fuse**，
             與 `v6_short.pt` 不相容。消融一律用獨立的 checkpoint 檔名。

    【為什麼要有這個開關】2026-07-29 實測發現線上知識圖譜是壞的
    （42,864 個節點只有 2,510 是真股票、供應鏈邊來自 regex 抓 HTML 裡的 4 位數字、
    2330 的鄰居是電器電纜與綠能環保）。在確認 GATv2 到底有沒有貢獻之前，
    不應該投入更複雜的圖設計——所以要先能把它整個關掉做對照。
    三組：無 GAT ／ 舊圖 GAT ／ v2 圖 GAT，其餘變因全凍結。
    """

    def __init__(self, d_model: int = D_MODEL, d_state: int = D_STATE,
                 n_heads_gat: int = N_HEADS_GAT, dropout: float = DROPOUT,
                 window: int = 60, n_layers: int = 3, use_gat: bool = True):
        super().__init__()
        self.use_gat     = use_gat
        self.embedding   = FactorGroupedEmbedding(d_model=d_model)
        self.encoder     = SingleScaleEncoder(d_model=d_model, d_state=d_state,
                                              window=window, n_layers=n_layers)
        if use_gat:
            self.graph_layer = GraphAttentionLayer(d_model=d_model, n_heads=n_heads_gat)
            self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
            self.norm_fuse = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.head_5d   = nn.Linear(d_model, 1)
        self.head_10d  = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, edge_index, edge_attr, padding_mask=None) -> Tensor:
        # padding_mask 收下但不用（單尺度取最後 window 步全為真實資料）
        h = self.embedding(x)                              # (N, 252, d_model)
        h_temporal = self.encoder(h)                       # (N, d_model)
        if self.use_gat:
            h_graph = self.graph_layer(h_temporal, edge_index, edge_attr)
            gate_input  = torch.cat([h_temporal, h_graph], dim=-1)
            gate_weight = self.gate(gate_input)
            h_fused = self.norm_fuse(gate_weight * h_temporal + (1 - gate_weight) * h_graph)
        else:
            # 消融組：edge_index / edge_attr 仍照收（呼叫端簽章不變），但完全不用
            h_fused = h_temporal
        h_fused = self.dropout(h_fused)
        return torch.cat([self.head_5d(h_fused), self.head_10d(h_fused)], dim=-1)  # (N, 2)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 3. 目標工程：per-cross-section pct-rank 置中（±0.5）
# ============================================================
def rank_transform(Y: Tensor) -> Tensor:
    """對每一欄（horizon）在這個 cross-section 內做 pct-rank 置中。NaN 維持 NaN。
    （Y 進來時已是 dataset 的 per-date z-score；rank 對 z-score 不變 = rank 原始 Alpha。）"""
    out = torch.full_like(Y, float("nan"))
    for j in range(Y.shape[1]):
        col = Y[:, j]
        valid = ~torch.isnan(col)
        if valid.sum() < 2:
            continue
        v = col[valid]
        order = torch.argsort(v)
        ranks = torch.empty_like(v)
        ranks[order] = torch.arange(len(v), dtype=v.dtype, device=v.device)
        out[valid, j] = ranks / (len(v) - 1) - 0.5     # [-0.5, 0.5]
    return out


# ============================================================
# 4. 損失：MSE + listnet，對 ranked target
# ============================================================
def short_loss(preds, targets_raw, weights=SHORT_WEIGHTS):
    t = rank_transform(targets_raw)                 # (N,2) ranked
    p5, p10 = preds[:, 0], preds[:, 1]
    t5, t10 = t[:, 0], t[:, 1]
    v5, v10 = ~torch.isnan(t5), ~torch.isnan(t10)
    z = torch.zeros((), device=preds.device)
    mse5  = F.mse_loss(p5[v5],  t5[v5])  if v5.any()  else z
    mse10 = F.mse_loss(p10[v10], t10[v10]) if v10.any() else z
    ln5   = listnet_loss(p5[v5],  t5[v5])  if v5.sum()  > 1 else z
    ln10  = listnet_loss(p10[v10], t10[v10]) if v10.sum() > 1 else z
    total = (weights["mse_5d"] * mse5 + weights["mse_10d"] * mse10
             + weights["listnet_5d"] * ln5 + weights["listnet_10d"] * ln10)
    bd = {"loss_total": float(total.item()), "mse_5d": float(mse5.item()),
          "mse_10d": float(mse10.item()), "ln_5d": float(ln5.item()), "ln_10d": float(ln10.item())}
    return total, bd


# ============================================================
# 5. 訓練：patch target 欄為 [Alpha_5d, Alpha_10d]，單尺度模型，雙 IC
# ============================================================
def train_short_model(
    df,
    train_dates,
    val_dates,
    epochs:          int = 100,
    lr:              float = LR,
    early_stop:      int = 15,
    weights:         dict = SHORT_WEIGHTS,
    window:          int = 60,
    n_layers:        int = 3,
    checkpoint_name: str = "v6_short.pt",
    device_str:      str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_backup_dir: str | None = None,
    status_path:     str | None = None,
    use_gat:         bool = True,
    kg_cache_path:   str | None = None,
):
    # ── 把資料集的 target 欄暫時指到 5d/10d（不動正式檔，跟 listnet_5d 同款 runtime 覆蓋）──
    #
    # ⚠️ 這是 **module 級全域變數**，跑完必須還原（2026-07-31 補）。
    #    原本沒還原，於是在同一個 Colab session 先跑短線模型、再跑 Cell 4 的
    #    多 horizon 訓練時，dataset 只給 2 欄標籤而 `multi_horizon_loss` 要取
    #    `targets[:, 2]` → `IndexError: index 2 is out of bounds`。
    #    錯誤訊息指向 trainer.py，跟真正的原因（這裡改了全域沒還原）差很遠，
    #    很難從堆疊追回來，所以用 try/finally 保證還原。
    import marketmamba.config as cfg
    import marketmamba.models.trainer as T
    _orig_globals = (cfg.PRED_HORIZONS, T.PRED_HORIZONS, T.TARGET_COLS)
    cfg.PRED_HORIZONS = [5, 10]
    T.PRED_HORIZONS   = [5, 10]
    T.TARGET_COLS     = ["Alpha_5d", "Alpha_10d"]
    try:
        return _train_short_model_inner(
            df, train_dates, val_dates, epochs=epochs, early_stop=early_stop, lr=lr,
            weights=weights, window=window, n_layers=n_layers,
            checkpoint_name=checkpoint_name, checkpoint_backup_dir=checkpoint_backup_dir,
            status_path=status_path, device_str=device_str,
            use_gat=use_gat, kg_cache_path=kg_cache_path)
    finally:
        cfg.PRED_HORIZONS, T.PRED_HORIZONS, T.TARGET_COLS = _orig_globals
        print(f"[short] 已還原 TARGET_COLS={T.TARGET_COLS} / PRED_HORIZONS={T.PRED_HORIZONS}",
              flush=True)


def _train_short_model_inner(
    df,
    train_dates,
    val_dates,
    *,
    epochs, early_stop, lr, weights, window, n_layers,
    checkpoint_name, checkpoint_backup_dir, status_path, device_str,
    use_gat, kg_cache_path,
):
    """實際訓練迴圈。外層 `train_short_model` 負責 patch/還原全域變數。"""
    import marketmamba.config as cfg
    import marketmamba.models.trainer as T          # 下方 KG monkeypatch 會用到

    device = torch.device(device_str)
    print(f"[short] device={device} | train={len(train_dates)} val={len(val_dates)} | "
          f"window={window} layers={n_layers} | targets=[Alpha_5d, Alpha_10d] | weights={weights}", flush=True)

    _n_sample = cfg.N_SAMPLE_TRAIN
    train_ds = TemporalCrossSectionDataset(df, train_dates, mode="train", n_sample=_n_sample)
    val_ds   = TemporalCrossSectionDataset(df, val_dates,   mode="val",   n_sample=None)
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds,   shuffle=False)

    model = ShortModelV6(window=window, n_layers=n_layers, use_gat=use_gat).to(device)
    print(f"[short] model params: {model.n_parameters:,} | use_gat={use_gat}", flush=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
                           pct_start=WARMUP_PCT, anneal_strategy="cos")
    scaler = GradScaler('cuda', enabled=AMP_ENABLED and device_str == "cuda")

    # 圖的來源：`build_kg_csr()` 在受保護的 trainer.py 裡讀 module-level KG_CACHE_PATH，
    # 沿用本檔既有的 monkeypatch 慣例（上面第 190 行已經在這樣改 PRED_HORIZONS）。
    if kg_cache_path is not None:
        from pathlib import Path as _P
        _orig_kg = T.KG_CACHE_PATH
        T.KG_CACHE_PATH = _P(kg_cache_path)
        print(f"[short] KG 來源改為 {kg_cache_path}", flush=True)
        try:
            kg_csr, stock_to_idx = build_kg_csr()
        finally:
            T.KG_CACHE_PATH = _orig_kg          # 還原，避免污染同一個 session 的後續實驗
    else:
        kg_csr, stock_to_idx = build_kg_csr()

    if not use_gat:
        # 消融組不需要圖；印出來是為了讓「這一輪確實沒用到 KG」看得見，
        # 而不是靠讀程式碼推論（規則 7）
        print("[short] use_gat=False → 本輪完全不使用知識圖譜邊", flush=True)

    history = TrainingHistory()
    history.val_ic_10d = []   # 額外記 10d IC
    best_ic, no_impr = float("-inf"), 0
    ckpt_path = MODELS_DIR / checkpoint_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"── Epoch {epoch:03d}/{epochs} 開始 ──", flush=True)

        # ── Train ──
        model.train()
        train_losses = []
        total_b = len(train_loader)
        for batch_idx, (X, Y, batch_stocks, padding_mask) in enumerate(train_loader):
            if X.shape[0] <= 1:
                continue
            if epoch == 1 and batch_idx == 0:
                print(f"  [diag] 第一個 batch: stocks={len(batch_stocks)} Y={tuple(Y.shape)} | {time.time()-t0:.0f}s", flush=True)
            X, Y = X.to(device), Y.to(device)
            edge_index, edge_attr = get_batch_edges_csr(batch_stocks, kg_csr, stock_to_idx, device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=AMP_ENABLED and device_str == "cuda"):
                preds = model(X, edge_index, edge_attr)
                loss, brkdn = short_loss(preds, Y, weights)
            if not loss.requires_grad or torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(brkdn["loss_total"])
            if (batch_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (batch_idx + 1) * (total_b - batch_idx - 1)
                print(f"  Ep {epoch:03d} [{batch_idx+1}/{total_b}] loss={float(np.mean(train_losses)):.5f} "
                      f"| {elapsed:.0f}s | ETA {eta:.0f}s", flush=True)

        if not train_losses:
            continue
        avg_train = float(np.mean(train_losses))

        # ── Validate ── (IC 用 dataset 的 Y，compute_ic 是 Spearman、與歷次可比)
        model.eval()
        val_losses, ic5, ic10 = [], [], []
        with torch.no_grad():
            for X, Y, batch_stocks, padding_mask in val_loader:
                if X.shape[0] <= 1:
                    continue
                X, Y = X.to(device), Y.to(device)
                edge_index, edge_attr = get_batch_edges_csr(batch_stocks, kg_csr, stock_to_idx, device)
                with autocast('cuda', enabled=AMP_ENABLED and device_str == "cuda"):
                    preds = model(X, edge_index, edge_attr)
                    loss, _ = short_loss(preds, Y, weights)
                val_losses.append(loss.item())
                ic5.append(compute_ic(preds[:, 0].cpu().numpy(),  Y[:, 0].cpu().numpy()))
                ic10.append(compute_ic(preds[:, 1].cpu().numpy(), Y[:, 1].cpu().numpy()))

        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
        avg_ic5  = float(np.mean(ic5))  if ic5  else 0.0
        avg_ic10 = float(np.mean(ic10)) if ic10 else 0.0
        cur_lr   = scheduler.get_last_lr()[0]

        history.train_loss.append(avg_train)
        history.val_loss.append(avg_val)
        history.val_ic.append(avg_ic5)          # 主指標 = 5d IC
        history.val_ic_10d.append(avg_ic10)
        history.lr.append(cur_lr)

        print(f"Epoch {epoch:03d}/{epochs} | train={avg_train:.5f} val={avg_val:.5f} "
              f"IC(5d)={avg_ic5:+.4f} IC(10d)={avg_ic10:+.4f} lr={cur_lr:.2e} | {time.time()-t0:.0f}s", flush=True)

        if status_path is not None:
            best_ep = int(np.argmax(history.val_ic)) + 1 if history.val_ic else 0
            with open(status_path, "w") as f:
                json.dump({
                    "experiment": "short_model_single_scale",
                    "epoch": epoch, "epochs_max": epochs,
                    "best_val_ic_5d": max(history.val_ic) if history.val_ic else 0.0,
                    "best_ic_epoch": best_ep,
                    "config": {"window": window, "n_layers": n_layers, "weights": weights},
                    "history": {
                        "train_loss": history.train_loss, "val_loss": history.val_loss,
                        "val_ic_5d": history.val_ic, "val_ic_10d": history.val_ic_10d, "lr": history.lr,
                    },
                }, f, indent=1)

        # checkpoint / early-stop 追 5d IC
        if avg_ic5 > best_ic:
            best_ic, no_impr = avg_ic5, 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_ic_5d": avg_ic5, "val_ic_10d": avg_ic10, "val_loss": avg_val,
                        "config": {"window": window, "n_layers": n_layers}}, ckpt_path)
            if checkpoint_backup_dir:
                os.makedirs(checkpoint_backup_dir, exist_ok=True)
                import shutil
                shutil.copy(str(ckpt_path), os.path.join(checkpoint_backup_dir, checkpoint_name))
                print(f"  ✅ ckpt saved (5d IC={avg_ic5:+.4f}) → {ckpt_path.name} + Drive", flush=True)
            else:
                print(f"  ✅ ckpt saved (5d IC={avg_ic5:+.4f}) → {ckpt_path.name}", flush=True)
        else:
            no_impr += 1
            if no_impr >= early_stop:
                print(f"  🛑 Early stop at epoch {epoch} (5d IC no-improve {early_stop} ep)", flush=True)
                break

    print(f"[short] done. best 5d IC={max(history.val_ic) if history.val_ic else 0.0:+.4f}", flush=True)
    return model, history
