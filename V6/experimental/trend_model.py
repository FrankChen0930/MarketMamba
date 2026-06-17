"""
Phase 2 步驟 3 — 多尺度趨勢模型（隔離副本）
================================================
趨勢模型沿用正式多尺度架構 MarketMambaV6（3 分支 + scale gate + GAT + 3 頭 5/20/60），
主攻 20d / 60d。與短線模型一致使用 rank 目標——rank 把短線 5d IC 從 0.049 翻到 0.095，
趨勢同樣套用。

⚠️ 隔離副本：直接用正式 MarketMambaV6（多尺度沒被 B 推翻，20d 趨勢吃長窗有用），
   marketmamba/ 一個字不動；target 用 dataset 預設 [Alpha_5d, Alpha_20d, Alpha_60d]（免 patch）。
   主輸出 (N,3)。
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

from marketmamba.config import AMP_ENABLED, GRAD_CLIP_NORM, LR, MODELS_DIR, WARMUP_PCT
from marketmamba.models.architecture import MarketMambaV6
from marketmamba.models.trainer import (
    TemporalCrossSectionDataset, make_dataloader, build_kg_csr,
    get_batch_edges_csr, compute_ic, listnet_loss, TrainingHistory,
)

# 20d 為主、60d 次要（沿用原 V6 趨勢配置 + 加 listnet_60d；mse_5d 保留輕度多任務）
TREND_WEIGHTS = {"mse_5d": 0.3, "mse_20d": 1.0, "mse_60d": 0.5, "listnet_20d": 0.5, "listnet_60d": 0.25}


def rank_transform(Y: Tensor) -> Tensor:
    """每欄(horizon)在 cross-section 內做 pct-rank 置中(±0.5)，NaN 維持 NaN。"""
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
        out[valid, j] = ranks / (len(v) - 1) - 0.5
    return out


def trend_loss(preds, targets_raw, weights=TREND_WEIGHTS):
    t = rank_transform(targets_raw)                 # (N,3) [5,20,60] ranked
    p5, p20, p60 = preds[:, 0], preds[:, 1], preds[:, 2]
    t5, t20, t60 = t[:, 0], t[:, 1], t[:, 2]
    v5, v20, v60 = ~torch.isnan(t5), ~torch.isnan(t20), ~torch.isnan(t60)
    z = torch.zeros((), device=preds.device)
    mse5  = F.mse_loss(p5[v5],   t5[v5])   if v5.any()  else z
    mse20 = F.mse_loss(p20[v20], t20[v20]) if v20.any() else z
    mse60 = F.mse_loss(p60[v60], t60[v60]) if v60.any() else z
    ln20  = listnet_loss(p20[v20], t20[v20]) if v20.sum() > 1 else z
    ln60  = listnet_loss(p60[v60], t60[v60]) if v60.sum() > 1 else z
    total = (weights["mse_5d"] * mse5 + weights["mse_20d"] * mse20 + weights["mse_60d"] * mse60
             + weights["listnet_20d"] * ln20 + weights["listnet_60d"] * ln60)
    bd = {"loss_total": float(total.item()), "mse_20d": float(mse20.item()),
          "mse_60d": float(mse60.item()), "ln_20d": float(ln20.item()), "ln_60d": float(ln60.item())}
    return total, bd


def train_trend_model(
    df, train_dates, val_dates,
    epochs: int = 100, lr: float = LR, early_stop: int = 15,
    weights: dict = TREND_WEIGHTS, checkpoint_name: str = "v6_trend.pt",
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_backup_dir: str | None = None, status_path: str | None = None,
):
    device = torch.device(device_str)
    print(f"[trend] device={device} | train={len(train_dates)} val={len(val_dates)} | "
          f"targets=[Alpha_5d,20d,60d] | weights={weights}", flush=True)

    import marketmamba.config as cfg
    _n_sample = cfg.N_SAMPLE_TRAIN
    train_ds = TemporalCrossSectionDataset(df, train_dates, mode="train", n_sample=_n_sample)
    val_ds   = TemporalCrossSectionDataset(df, val_dates,   mode="val",   n_sample=None)
    train_loader = make_dataloader(train_ds, shuffle=True)
    val_loader   = make_dataloader(val_ds,   shuffle=False)

    model = MarketMambaV6().to(device)
    print(f"[trend] model params: {model.n_parameters:,}", flush=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps,
                           pct_start=WARMUP_PCT, anneal_strategy="cos")
    scaler = GradScaler('cuda', enabled=AMP_ENABLED and device_str == "cuda")
    kg_csr, stock_to_idx = build_kg_csr()

    history = TrainingHistory()
    history.val_ic_60d, history.val_ic_5d = [], []
    best_ic, no_impr = float("-inf"), 0
    ckpt_path = MODELS_DIR / checkpoint_name
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"── Epoch {epoch:03d}/{epochs} 開始 ──", flush=True)

        model.train()
        train_losses = []
        total_b = len(train_loader)
        for batch_idx, (X, Y, batch_stocks, padding_mask) in enumerate(train_loader):
            if X.shape[0] <= 1:
                continue
            if epoch == 1 and batch_idx == 0:
                print(f"  [diag] 第一個 batch: stocks={len(batch_stocks)} Y={tuple(Y.shape)} | {time.time()-t0:.0f}s", flush=True)
            X, Y = X.to(device), Y.to(device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            edge_index, edge_attr = get_batch_edges_csr(batch_stocks, kg_csr, stock_to_idx, device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=AMP_ENABLED and device_str == "cuda"):
                preds = model(X, edge_index, edge_attr, padding_mask=padding_mask)
                loss, brkdn = trend_loss(preds, Y, weights)
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

        model.eval()
        val_losses, ic20, ic60, ic5 = [], [], [], []
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
                    loss, _ = trend_loss(preds, Y, weights)
                val_losses.append(loss.item())
                ic20.append(compute_ic(preds[:, 1].cpu().numpy(), Y[:, 1].cpu().numpy()))
                ic60.append(compute_ic(preds[:, 2].cpu().numpy(), Y[:, 2].cpu().numpy()))
                ic5.append(compute_ic(preds[:, 0].cpu().numpy(),  Y[:, 0].cpu().numpy()))

        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
        a20 = float(np.mean(ic20)) if ic20 else 0.0
        a60 = float(np.mean(ic60)) if ic60 else 0.0
        a5  = float(np.mean(ic5))  if ic5  else 0.0
        cur_lr = scheduler.get_last_lr()[0]

        history.train_loss.append(avg_train)
        history.val_loss.append(avg_val)
        history.val_ic.append(a20)          # 主指標 = 20d IC
        history.val_ic_60d.append(a60)
        history.val_ic_5d.append(a5)
        history.lr.append(cur_lr)
        if model.encoder._last_scales is not None:
            w = model.encoder._last_scales.mean(dim=0).cpu()
            history.scale_gates.append([float(w[0]), float(w[1]), float(w[2])])

        print(f"Epoch {epoch:03d}/{epochs} | train={avg_train:.5f} val={avg_val:.5f} "
              f"IC(20d)={a20:+.4f} IC(60d)={a60:+.4f} IC(5d)={a5:+.4f} lr={cur_lr:.2e} | {time.time()-t0:.0f}s", flush=True)
        if history.scale_gates:
            sg = history.scale_gates[-1]
            print(f"  [scale_gate] Short:{sg[0]:.3f} Mid:{sg[1]:.3f} Long:{sg[2]:.3f}", flush=True)

        if status_path is not None:
            best_ep = int(np.argmax(history.val_ic)) + 1 if history.val_ic else 0
            with open(status_path, "w") as f:
                json.dump({
                    "experiment": "trend_model_multi_scale", "epoch": epoch, "epochs_max": epochs,
                    "best_val_ic_20d": max(history.val_ic) if history.val_ic else 0.0, "best_ic_epoch": best_ep,
                    "weights": weights,
                    "history": {"train_loss": history.train_loss, "val_loss": history.val_loss,
                                "val_ic_20d": history.val_ic, "val_ic_60d": history.val_ic_60d,
                                "val_ic_5d": history.val_ic_5d, "lr": history.lr,
                                "scale_gates": history.scale_gates},
                }, f, indent=1)

        if a20 > best_ic:
            best_ic, no_impr = a20, 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_ic_20d": a20, "val_ic_60d": a60, "val_loss": avg_val}, ckpt_path)
            if checkpoint_backup_dir:
                os.makedirs(checkpoint_backup_dir, exist_ok=True)
                import shutil
                shutil.copy(str(ckpt_path), os.path.join(checkpoint_backup_dir, checkpoint_name))
                print(f"  ✅ ckpt saved (20d IC={a20:+.4f}) → {ckpt_path.name} + Drive", flush=True)
            else:
                print(f"  ✅ ckpt saved (20d IC={a20:+.4f}) → {ckpt_path.name}", flush=True)
        else:
            no_impr += 1
            if no_impr >= early_stop:
                print(f"  🛑 Early stop at epoch {epoch} (20d IC no-improve {early_stop} ep)", flush=True)
                break

    print(f"[trend] done. best 20d IC={max(history.val_ic) if history.val_ic else 0.0:+.4f}", flush=True)
    return model, history
