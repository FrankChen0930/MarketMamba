"""
Phase 3 — 實驗 C：趨勢模型單尺度簡化（診斷）
==========================================================
目的
----
趨勢模型 `v6_trend.pt` 是多尺度 MarketMambaV6（Short20/Mid60/Long252 三分支 + scale
gate），但訓練時 **scale_gate 在 ep3 就 100% 塌向 Long**——Short/Mid 形同虛設、
多尺度對 20d 趨勢沒加值。本實驗把趨勢砍成 **單尺度＝只留 Long 分支**，驗證：
  「單尺度 Long 的峰值 20d IC 能否持平多尺度的 0.0961」
若持平＝多尺度確實冗餘、可砍 → 未來趨勢訓練更省、模型更乾淨（少 2 個 Mamba 分支
＋ scale gate）。這是「確認既有發現」的便宜診斷，IC 升幅期望 ~0，價值在簡化。

單一變因
--------
只把 encoder 從「多尺度 3 分支 + gate」換成「單尺度 Long（完整 252 步）」，其餘
全部與 v6_trend.pt 基準一致：
  - 同 GAT + gating 融合 + 3 頭(5/20/60)、同 TREND_WEIGHTS（20d 主 + listnet）
  - dropout=0.1, LR=7e-5, wd=1e-4, 切分 train≤2023 / val 2024-
  - **Long 分支照樣套 padding_mask**（個股歷史 202~252 天、全 252 步含 zero-pad，
    與多尺度 Long 分支同款：padding 位置乘 0 截斷梯度）

隔離保證
--------
- 不修改 `trend_model.py` / `architecture.py`（前者被本實驗 import、後者是受保護的
  production 架構）。單尺度模型在本檔內定義；用 monkeypatch 把 `trend_model.MarketMambaV6`
  暫時換成單尺度版、跑完還原，整套 `train_trend_model`（loss/排程/IC/checkpoint/status）重用。
- checkpoint 用獨立檔名 `v6_trend_C_single.pt`，**絕不覆蓋** production `v6_trend.pt`。

============================================================
怎麼用（Colab）
============================================================
前置：跑完 Cell 0→1→2→3（df 就緒、sys.path 含 /content/MarketMamba/V6）。新開 cell：

    from experimental.phase3_c_trend_single_scale import run_trend_single
    result = run_trend_single(
        df,
        epochs=16, early_stop=8,              # 多尺度峰在 ep13；夠看到峰 + 確認即可
        dropout=0.1,                          # 與趨勢基準一致；要驗 dropout 才改
        drive_dir="/content/drive/MyDrive/MarketMamba_V6",
    )

跑完印「單尺度 vs 多尺度基準」對照 + 判定，完整結果寫到
  {drive_dir}/phase3_C_trend_single_result.json

判讀標準
--------
  (1) 單尺度峰值 20d IC 是否持平 0.0961（容差內＝多尺度冗餘、可砍）
  (2) 60d IC 是否持平 0.1030
  (3) 訓練開頭印的 model params 會明顯少於多尺度（少 Short/Mid 兩分支 + scale gate）
若持平：趨勢可改單尺度當未來基準（Phase 3 後續實驗都在更便宜的單尺度上做）。
若明顯掉：代表 Long 之外仍有殘餘貢獻，保留多尺度。
"""

from __future__ import annotations

import functools
import json
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from marketmamba.config import D_MODEL, D_STATE, DROPOUT, N_HEADS_GAT
from marketmamba.models.architecture import (
    FactorGroupedEmbedding, MambaStack, GraphAttentionLayer, MultiHorizonHead,
)

# 多尺度趨勢基準（v6_trend.pt，2026-06-15）——拿來對照
BASELINE = {"val_ic_20d": 0.0961, "val_ic_60d": 0.1030, "peak_epoch": 13}


# ============================================================
# 單尺度 encoder：只留 Long 分支（完整 252 步 + padding_mask）
# ============================================================
class LongOnlyEncoder(nn.Module):
    """多尺度趨勢 gate ep3 就 100% 塌 Long → 直接只保留 Long 分支。
    取完整 252 步 MambaStack，套 padding_mask 把 zero-pad 位置乘 0 截斷梯度
    （與 MultiScaleMambaEncoder 的 Long 分支同款），取最後一步 → LayerNorm。"""

    def __init__(self, d_model: int = D_MODEL, d_state: int = D_STATE, n_layers: int = 3):
        super().__init__()
        self.mamba_long = MambaStack(n_layers, d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        # train_trend_model 會讀 model.encoder._last_scales；單尺度無 gate，設 None 讓它跳過 scale_gate 紀錄
        self._last_scales: Optional[Tensor] = None

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:  # x:(N,252,d)
        h_long_seq = self.mamba_long(x)                                   # (N,252,d)
        if padding_mask is not None:
            h_long_seq = h_long_seq * padding_mask.unsqueeze(-1).float()  # 截斷 padding 梯度
        h_long = h_long_seq[:, -1, :]                                     # (N,d)
        return self.norm(h_long)


# ============================================================
# 單尺度趨勢模型：LongOnlyEncoder + GAT + gating 融合 + 3 頭(5/20/60)
# （除 encoder 外，與 MarketMambaV6 結構逐行對齊）
# ============================================================
class TrendSingleModel(nn.Module):
    def __init__(self, d_model: int = D_MODEL, d_state: int = D_STATE,
                 n_heads_gat: int = N_HEADS_GAT, dropout: float = DROPOUT, n_layers: int = 3):
        super().__init__()
        self.embedding   = FactorGroupedEmbedding(d_model=d_model)
        self.encoder     = LongOnlyEncoder(d_model=d_model, d_state=d_state, n_layers=n_layers)
        self.graph_layer = GraphAttentionLayer(d_model=d_model, n_heads=n_heads_gat)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm_fuse = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.head      = MultiHorizonHead(d_model=d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x, edge_index, edge_attr, padding_mask=None) -> Tensor:
        h = self.embedding(x)                          # (N,252,d)
        h_temporal = self.encoder(h, padding_mask)     # (N,d) — 單尺度 Long
        h_graph = self.graph_layer(h_temporal, edge_index, edge_attr)
        gate_input  = torch.cat([h_temporal, h_graph], dim=-1)
        gate_weight = self.gate(gate_input)
        h_fused = self.norm_fuse(gate_weight * h_temporal + (1 - gate_weight) * h_graph)
        h_fused = self.dropout(h_fused)
        return self.head(h_fused)                       # (N,3) = [5d,20d,60d]

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------
# 資料切分（與趨勢基準 run 同款）
# ------------------------------------------------------------
def build_dates(df, cutoff_train_end: str = "2023-12-31"):
    all_dates = sorted(df["Date"].astype(str).unique().tolist())
    train_dates = [d for d in all_dates if d <= cutoff_train_end]
    val_dates   = [d for d in all_dates if d > cutoff_train_end]
    print(f"[phase3-C] 切分 cutoff={cutoff_train_end} | "
          f"train={len(train_dates)} 天（…{train_dates[-1] if train_dates else '—'}）| "
          f"val={len(val_dates)} 天（{val_dates[0] if val_dates else '—'}…）", flush=True)
    return train_dates, val_dates


# ------------------------------------------------------------
# 主入口：跑單尺度趨勢、對照多尺度基準
# ------------------------------------------------------------
def run_trend_single(
    df,
    train_dates: Optional[Sequence[str]] = None,
    val_dates:   Optional[Sequence[str]] = None,
    epochs:      int = 16,
    early_stop:  int = 8,
    dropout:     float = 0.1,
    n_layers:    int = 3,
    cutoff_train_end: str = "2023-12-31",
    drive_dir:   Optional[str] = "/content/drive/MyDrive/MarketMamba_V6",
    ic_tol:      float = 0.003,
) -> dict:
    """
    跑一次單尺度（Long-only）趨勢模型，對照多尺度基準 20d IC 0.0961。
    monkeypatch trend_model.MarketMambaV6 -> TrendSingleModel，重用 train_trend_model；跑完還原。
    回傳 dict（含峰值 20d/60d IC、峰值 epoch、與基準差、判定），並把結果寫進 Drive JSON。
    """
    if train_dates is None or val_dates is None:
        train_dates, val_dates = build_dates(df, cutoff_train_end)

    print(f"[phase3-C] 單尺度趨勢（Long-only, n_layers={n_layers}, dropout={dropout}）"
          f"| 對照多尺度基準 20d IC={BASELINE['val_ic_20d']} @ep{BASELINE['peak_epoch']}", flush=True)

    ckpt_name   = "v6_trend_C_single.pt"
    status_path = None
    backup_dir  = None
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        status_path = f"{drive_dir}/status_trend_C_single.json"
        backup_dir  = f"{drive_dir}/checkpoints"

    import experimental.trend_model as tm   # 重用其 train_trend_model；只暫時替換模型類別

    original_cls = tm.MarketMambaV6
    tm.MarketMambaV6 = functools.partial(TrendSingleModel, dropout=dropout, n_layers=n_layers)
    try:
        model, history = tm.train_trend_model(
            df, train_dates, val_dates,
            epochs=epochs,
            early_stop=early_stop,
            checkpoint_name=ckpt_name,
            checkpoint_backup_dir=backup_dir,
            status_path=status_path,
        )
    finally:
        tm.MarketMambaV6 = original_cls

    ic20 = list(history.val_ic)                          # 主指標 = 20d IC
    ic60 = list(getattr(history, "val_ic_60d", []))
    ic5  = list(getattr(history, "val_ic_5d", []))
    if ic20:
        peak_idx = int(np.argmax(ic20))
        peak_20, peak_ep = float(ic20[peak_idx]), peak_idx + 1
        peak_60 = float(ic60[peak_idx]) if peak_idx < len(ic60) else 0.0
    else:
        peak_20, peak_60, peak_ep = 0.0, 0.0, 0

    delta20 = peak_20 - BASELINE["val_ic_20d"]
    n_params = int(model.n_parameters)
    # 判定：持平（容差內或更好）＝多尺度冗餘、可砍
    if peak_20 >= BASELINE["val_ic_20d"] - ic_tol:
        verdict = "持平/更好 → 多尺度冗餘、趨勢可改單尺度（未來實驗在更便宜的單尺度上做）"
    else:
        verdict = "明顯掉 → Long 之外仍有殘餘貢獻，保留多尺度"

    result = {
        "experiment": "phase3_C_trend_single_scale",
        "checkpoint": ckpt_name,
        "config": {"encoder": "long_only", "n_layers": n_layers, "dropout": dropout,
                   "epochs": epochs, "early_stop": early_stop},
        "n_parameters": n_params,
        "peak_ic_20d": peak_20, "peak_ic_60d": peak_60, "peak_epoch": peak_ep,
        "baseline_multi_scale": BASELINE,
        "delta_ic_20d_vs_baseline": delta20,
        "verdict": verdict,
        "val_ic_20d_curve": [round(x, 4) for x in ic20],
        "val_ic_60d_curve": [round(x, 4) for x in ic60],
        "val_ic_5d_curve":  [round(x, 4) for x in ic5],
        "val_loss_curve":   [round(x, 5) for x in list(history.val_loss)],
    }

    if drive_dir:
        out = f"{drive_dir}/phase3_C_trend_single_result.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        print(f"[phase3-C] 已寫入 {out}", flush=True)

    _print_summary(result)
    return result


def _print_summary(r: dict) -> None:
    """對照摘要（規則 7：數值明確顯示）。"""
    b = r["baseline_multi_scale"]
    print("\n" + "=" * 68, flush=True)
    print("[phase3-C] 單尺度(Long-only) vs 多尺度基準 對照", flush=True)
    print("-" * 68, flush=True)
    print(f"{'指標':>16} | {'單尺度':>12} | {'多尺度基準':>12}", flush=True)
    print(f"{'峰值 20d IC':>16} | {r['peak_ic_20d']:>+12.4f} | {b['val_ic_20d']:>+12.4f}", flush=True)
    print(f"{'峰值 60d IC':>16} | {r['peak_ic_60d']:>+12.4f} | {b['val_ic_60d']:>+12.4f}", flush=True)
    print(f"{'峰值 epoch':>16} | {r['peak_epoch']:>12} | {b['peak_epoch']:>12}", flush=True)
    print(f"{'參數量':>16} | {r['n_parameters']:>12,} | {'(多分支較多)':>12}", flush=True)
    print("-" * 68, flush=True)
    print(f"  20d IC 對基準差：{r['delta_ic_20d_vs_baseline']:+.4f}", flush=True)
    print(f"  判定：{r['verdict']}", flush=True)
    print("=" * 68, flush=True)
