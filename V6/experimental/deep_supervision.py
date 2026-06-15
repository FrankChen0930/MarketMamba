"""
方案 B — Deep Supervision（診斷實驗，隔離副本）
================================================
目的：給三個時間尺度分支（Short/Mid/Long）各掛一個輔助頭、各自直接預測 5d，
      強迫每個分支不靠 scale_gate 也要學會預測，藉此判定：
        - Short 是真沒料、還是被 gate 餓死？
        - Mid 與 Long 是各有獨立價值、還是互相冗餘？

設計重點：
  - 輔助頭接在「GAT 與 gating 融合之前」的分支原始表徵上（h_short/h_mid/h_long）。
  - 輔助頭預測 5d（與主線一致）。
  - 損失 = 主損失(5d 主導、listnet 關) + w_aux × (MSE_short + MSE_mid + MSE_long)。
  - 跑在純 MSE 的 5d baseline 上（與 Phase 0 一個變因對比）。

⚠️ 這是 V6/experimental/ 隔離副本，完全不影響 production marketmamba/。
   production 的 MarketMambaV6 / train_model / 線上推論一個字都不會被動到。
   主輸出 forward 仍回傳 (N, 3)，與正式模型相容。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# ── 全部從正式 package 匯入，不複製、不修改 ──
from marketmamba.config import (
    AMP_ENABLED, D_MODEL, D_STATE, GRAD_CLIP_NORM, LR, MODELS_DIR, WARMUP_PCT,
)
from marketmamba.models.architecture import MultiScaleMambaEncoder, MarketMambaV6
from marketmamba.models.trainer import (
    TemporalCrossSectionDataset, make_dataloader, build_kg_csr,
    get_batch_edges_csr, compute_ic, listnet_loss, TrainingHistory,
)

# baseline(5d 主導、listnet 關) + 輔助損失權重
DS_WEIGHTS = {
    "mse_5d":      1.0,
    "mse_20d":     0.3,
    "mse_60d":     0.3,
    "listnet_20d": 0.0,
    "aux":         0.3,   # 每個分支輔助損失的權重
}


# ============================================================
# 1. Encoder：複製 forward，多存三個分支表徵（保留梯度）
# ============================================================
class MultiScaleMambaEncoderDS(MultiScaleMambaEncoder):
    """與正式 encoder 完全相同，唯一差別：把 h_short/h_mid/h_long 存進 self._branch_reps。"""

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        h_short = self.mamba_short(x[:, -self.seq_short:, :])[:, -1, :]
        h_mid   = self.mamba_mid(x[:, -self.seq_mid:,   :])[:, -1, :]

        h_long_seq = self.mamba_long(x)
        if padding_mask is not None:
            h_long_seq = h_long_seq * padding_mask.unsqueeze(-1).float()
        h_long = h_long_seq[:, -1, :]

        # ← 唯一新增：保留梯度，供輔助頭使用
        self._branch_reps = (h_short, h_mid, h_long)

        cat_h  = torch.cat([h_short, h_mid, h_long], dim=-1)
        scales = self.scale_gate(cat_h)
        self._last_scales = scales.detach()

        fused = (
            scales[:, 0:1] * h_short
            + scales[:, 1:2] * h_mid
            + scales[:, 2:3] * h_long
        )
        return self.norm(fused)


# ============================================================
# 2. Model：換上 DS encoder、加三個輔助頭
# ============================================================
class MarketMambaV6DS(MarketMambaV6):
    """正式模型 + deep supervision。主輸出維持 (N,3)；輔助預測存 self._aux_preds。"""

    def __init__(self, d_model: int = D_MODEL, d_state: int = D_STATE, **kw):
        super().__init__(d_model=d_model, d_state=d_state, **kw)
        # 換成會暴露分支表徵的 encoder
        self.encoder = MultiScaleMambaEncoderDS(d_model=d_model, d_state=d_state)
        # 三個輔助頭：各分支表徵 → 預測 5d
        self.aux_short = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.aux_mid   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.aux_long  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self._aux_preds: Tensor | None = None
        self._init_weights()   # 重新初始化（含換掉的 encoder 與輔助頭）

    def forward(self, x, edge_index, edge_attr, padding_mask=None) -> Tensor:
        preds = super().forward(x, edge_index, edge_attr, padding_mask=padding_mask)
        h_short, h_mid, h_long = self.encoder._branch_reps
        self._aux_preds = torch.cat(
            [self.aux_short(h_short), self.aux_mid(h_mid), self.aux_long(h_long)], dim=-1
        )  # (N, 3) = [short→5d, mid→5d, long→5d]
        return preds   # 主輸出不變


# ============================================================
# 3. 損失：主損失 + 三分支輔助損失
# ============================================================
def multi_horizon_loss_ds(preds, targets, aux_preds, weights=DS_WEIGHTS):
    p5, p20, p60 = preds[:, 0], preds[:, 1], preds[:, 2]
    t5, t20, t60 = targets[:, 0], targets[:, 1], targets[:, 2]
    v5, v20, v60 = ~torch.isnan(t5), ~torch.isnan(t20), ~torch.isnan(t60)
    z = torch.zeros((), device=preds.device)

    mse5  = F.mse_loss(p5[v5],   t5[v5])   if v5.any()  else z
    mse20 = F.mse_loss(p20[v20], t20[v20]) if v20.any() else z
    mse60 = F.mse_loss(p60[v60], t60[v60]) if v60.any() else z
    ln20  = (listnet_loss(p20[v20], t20[v20])
             if (weights.get("listnet_20d", 0.0) > 0 and v20.sum() > 1) else z)

    main = (weights["mse_5d"] * mse5 + weights["mse_20d"] * mse20
            + weights["mse_60d"] * mse60 + weights.get("listnet_20d", 0.0) * ln20)

    # 輔助：每個分支獨力預測 5d
    a_s, a_m, a_l = aux_preds[:, 0], aux_preds[:, 1], aux_preds[:, 2]
    aux_s = F.mse_loss(a_s[v5], t5[v5]) if v5.any() else z
    aux_m = F.mse_loss(a_m[v5], t5[v5]) if v5.any() else z
    aux_l = F.mse_loss(a_l[v5], t5[v5]) if v5.any() else z
    w_aux = weights.get("aux", 0.3)

    total = main + w_aux * (aux_s + aux_m + aux_l)
    bd = {
        "loss_total": float(total.item()),
        "mse_5d": float(mse5.item()), "mse_20d": float(mse20.item()), "mse_60d": float(mse60.item()),
        "aux_short": float(aux_s.item()), "aux_mid": float(aux_m.item()), "aux_long": float(aux_l.item()),
    }
    return total, bd


# ============================================================
# 4. 訓練：複製 train_model，改用 DS 模型 + DS 損失 + 記錄三分支 aux IC
# ============================================================
def train_model_ds(
    df,
    train_dates,
    val_dates,
    epochs:          int = 100,
    lr:              float = LR,
    early_stop:      int = 15,
    weights:         dict = DS_WEIGHTS,
    checkpoint_name: str = "v6_exp_ds.pt",
    device_str:      str = "cuda" if torch.cuda.is_available() else "cpu",
    on_epoch_end=None,                  # 重用 Cell 4 的 live_plot/_on_epoch（標準 training_status.json）
    checkpoint_backup_dir: str | None = None,
    ds_status_path:  str | None = None,  # 額外輸出含三分支 aux IC 的 JSON
):
    device = torch.device(device_str)
    print(f"[DS] device={device} | train={len(train_dates)} val={len(val_dates)} | weights={weights}", flush=True)

    import marketmamba.config as _live_cfg
    _n_sample = _live_cfg.N_SAMPLE_TRAIN
    train_ds = TemporalCrossSectionDataset(df, train_dates, mode="train", n_sample=_n_sample)
    val_ds   = TemporalCrossSectionDataset(df, val_dates,   mode="val",   n_sample=None)
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds,   shuffle=False)

    model = MarketMambaV6DS().to(device)
    print(f"[DS] model params: {model.n_parameters:,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
                           pct_start=WARMUP_PCT, anneal_strategy="cos")
    scaler = GradScaler('cuda', enabled=AMP_ENABLED and device_str == "cuda")
    kg_csr, stock_to_idx = build_kg_csr()

    history = TrainingHistory()
    # 額外記錄三分支 aux IC（TrainingHistory 沒有的欄位，動態掛上）
    history.aux_ic_short, history.aux_ic_mid, history.aux_ic_long = [], [], []
    best_ic, no_impr = float("-inf"), 0
    ckpt_path = MODELS_DIR / checkpoint_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # ── Train ──
        model.train()
        train_losses = []
        for X, Y, batch_stocks, padding_mask in train_loader:
            if X.shape[0] <= 1:
                continue
            X, Y = X.to(device), Y.to(device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            edge_index, edge_attr = get_batch_edges_csr(batch_stocks, kg_csr, stock_to_idx, device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=AMP_ENABLED and device_str == "cuda"):
                preds = model(X, edge_index, edge_attr, padding_mask=padding_mask)
                loss, brkdn = multi_horizon_loss_ds(preds, Y, model._aux_preds, weights)
            if not loss.requires_grad:
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(brkdn["loss_total"])

        if not train_losses:
            continue
        avg_train = float(np.mean(train_losses))

        # ── Validate ──
        model.eval()
        val_losses = []
        ic_main, ic_s, ic_m, ic_l = [], [], [], []
        with torch.no_grad():
            for X, Y, batch_stocks, padding_mask in val_loader:
                if X.shape[0] <= 1:
                    continue
                X, Y = X.to(device), Y.to(device)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)
                edge_index, edge_attr = get_batch_edges_csr(batch_stocks, kg_csr, stock_to_idx, device)
                with autocast('cuda', enabled=AMP_ENABLED and device_str == "cuda"):
                    preds = model(X, edge_index, edge_attr, padding_mask=padding_mask)
                    loss, _ = multi_horizon_loss_ds(preds, Y, model._aux_preds, weights)
                val_losses.append(loss.item())
                y5 = Y[:, 0].cpu().numpy()
                ap = model._aux_preds
                ic_main.append(compute_ic(preds[:, 0].cpu().numpy(), y5))
                ic_s.append(compute_ic(ap[:, 0].cpu().numpy(), y5))
                ic_m.append(compute_ic(ap[:, 1].cpu().numpy(), y5))
                ic_l.append(compute_ic(ap[:, 2].cpu().numpy(), y5))

        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
        avg_ic  = float(np.mean(ic_main)) if ic_main else 0.0
        aux_ic  = (float(np.mean(ic_s)) if ic_s else 0.0,
                   float(np.mean(ic_m)) if ic_m else 0.0,
                   float(np.mean(ic_l)) if ic_l else 0.0)
        cur_lr  = scheduler.get_last_lr()[0]

        history.train_loss.append(avg_train)
        history.val_loss.append(avg_val)
        history.val_ic.append(avg_ic)
        history.lr.append(cur_lr)
        history.aux_ic_short.append(aux_ic[0])
        history.aux_ic_mid.append(aux_ic[1])
        history.aux_ic_long.append(aux_ic[2])
        if model.encoder._last_scales is not None:
            w = model.encoder._last_scales.mean(dim=0).cpu()
            history.scale_gates.append([float(w[0]), float(w[1]), float(w[2])])
        else:
            history.scale_gates.append([0.333, 0.333, 0.334])

        print(f"Epoch {epoch:03d}/{epochs} | train={avg_train:.5f} val={avg_val:.5f} "
              f"IC(main 5d)={avg_ic:+.4f} lr={cur_lr:.2e} | {time.time()-t0:.0f}s", flush=True)
        print(f"  [aux IC 5d] Short: {aux_ic[0]:+.4f}  Mid: {aux_ic[1]:+.4f}  Long: {aux_ic[2]:+.4f}", flush=True)
        sg = history.scale_gates[-1]
        print(f"  [scale_gate] Short: {sg[0]:.3f}  Mid: {sg[1]:.3f}  Long: {sg[2]:.3f}", flush=True)

        if on_epoch_end is not None:
            on_epoch_end(history, epoch, epochs)

        # 額外 JSON（含三分支 aux IC），自包含、不依賴 production dump
        if ds_status_path is not None:
            best_ep = int(np.argmax(history.val_ic)) + 1 if history.val_ic else 0
            with open(ds_status_path, "w") as f:
                json.dump({
                    "experiment": "deep_supervision",
                    "epoch": epoch, "epochs_max": epochs,
                    "best_val_ic": max(history.val_ic) if history.val_ic else 0.0,
                    "best_ic_epoch": best_ep,
                    "weights": weights,
                    "history": {
                        "train_loss": history.train_loss, "val_loss": history.val_loss,
                        "val_ic": history.val_ic, "lr": history.lr,
                        "scale_gates": history.scale_gates,
                        "aux_ic_short": history.aux_ic_short,
                        "aux_ic_mid": history.aux_ic_mid,
                        "aux_ic_long": history.aux_ic_long,
                    },
                }, f, indent=1)

        # IC-based checkpoint（追主 5d IC）
        if avg_ic > best_ic:
            best_ic, no_impr = avg_ic, 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_ic": avg_ic, "val_loss": avg_val, "history": history}, ckpt_path)
            if checkpoint_backup_dir:
                os.makedirs(checkpoint_backup_dir, exist_ok=True)
                import shutil
                shutil.copy(str(ckpt_path), os.path.join(checkpoint_backup_dir, checkpoint_name))
                print(f"  ✅ ckpt saved (IC={avg_ic:+.4f}) → {ckpt_path.name} + Drive", flush=True)
            else:
                print(f"  ✅ ckpt saved (IC={avg_ic:+.4f}) → {ckpt_path.name}", flush=True)
        else:
            no_impr += 1
            if no_impr >= early_stop:
                print(f"  🛑 Early stop at epoch {epoch} (no-improve {early_stop} ep)", flush=True)
                break

    print(f"[DS] done. best main 5d IC={max(history.val_ic) if history.val_ic else 0.0:+.4f}", flush=True)
    return model, history
