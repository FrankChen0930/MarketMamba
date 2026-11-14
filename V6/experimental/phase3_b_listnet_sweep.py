"""
Phase 3 — 實驗 B：listnet 權重 sweep（短線模型）
==========================================================
目的
----
短線模型 `v6_short.pt` 的損失 = rank-MSE + listnet（兩者都吃 per-cross-section
pct-rank 後的 target）。rank 目標是把 IC 從 0.049 近翻倍到 0.0951 的主功臣；
本實驗掃 **listnet 損失的權重**，看「排名損失強度」這個旋鈕能否再把峰值
5d IC 推過 0.0951，或讓 Phase 1 觀察到的「IC 峰高但脆」變穩。

⚠️ 與 Phase 1 的差別（避免混淆）
-------------------------------
Phase 1 那次「listnet=Mid↔Long 旋鈕、加 listnet 會把 gate 塌向 Long」是
**多尺度模型**的現象。短線模型是**單尺度**（無 Short/Mid/Long gate），故這裡
listnet 權重純粹是「listwise 排名損失 vs 逐點 MSE」的比重，不涉及 gate。

單一變因（嚴格遵守「一次只改一個變因」）
-----------------------------------------
只動 listnet 權重，其他全部凍結，與 v6_short.pt 基準 run 相同：
  - mse_5d=1.0, mse_10d=0.5（固定）
  - window=60, n_layers=3, dropout=0.1, LR=7e-5, wd=1e-4, 切分 train≤2023 / val 2024-
基準 listnet_5d=0.5（峰 0.0951@ep8）。掃法：保持 listnet_10d = listnet_5d / 2
（與基準 0.5:0.25 = 2:1 比例一致），listnet_5d ∈ {0.0, 0.25, 0.5, 1.0}：
  - 0.0  = 純 rank-MSE，無 listnet（隔離 listnet 的邊際貢獻＝控制組）
  - 0.25 = 弱
  - 0.5  = 基準
  - 1.0  = 強

隔離保證
--------
- 不修改 `short_model.py`（線上 dual 推論 import 它）。weights 透過 train_short_model
  的現成參數傳入；dropout 用 monkeypatch 暫時固定、跑完還原（與實驗 A 同款）。
- checkpoint / status 用獨立檔名（v6_short_B_lnXX.pt / status_short_B_lnXX.json），
  **絕不覆蓋** production `v6_short.pt`。

============================================================
怎麼用（Colab）
============================================================
前置：跑完 Cell 0→1→2→3（df 就緒、sys.path 含 /content/MarketMamba/V6）。新開 cell：

    from experimental.phase3_b_listnet_sweep import run_listnet_sweep
    results = run_listnet_sweep(
        df,
        listnet_5d_values=(0.0, 0.25, 0.5, 1.0),
        epochs=18, early_stop=10,
        dropout=0.1,                          # 純 B 實驗保持基準 0.1；
                                              # 若想帶 A 的最佳 dropout 做組合確認再改
        drive_dir="/content/drive/MyDrive/MarketMamba_V6",
    )

跑完印對照表，完整結果寫到 {drive_dir}/phase3_B_listnet_sweep_result.json。

判讀標準
--------
  (1) 峰值 5d IC 是否 >0.0951
  (2) 0.0（無 listnet）的峰值——量化 listnet 到底貢獻多少 IC
  (3) 峰值 epoch 是否延後、峰後下滑是否變緩（listnet 是否讓 IC 更穩/更脆）

備註
----
- 四組各跑一次完整訓練；每跑完一組落盤 Drive JSON（Colab 斷線可續）。
- 先試窄一點：run_listnet_sweep(df, listnet_5d_values=(0.0, 0.5))。
"""

from __future__ import annotations

import functools
import json
import os
from typing import Iterable, Optional, Sequence

import numpy as np


# ------------------------------------------------------------
# 資料切分（與短線基準 run 同款；與實驗 A 一致）
# ------------------------------------------------------------
def build_dates(df, cutoff_train_end: str = "2023-12-31"):
    all_dates = sorted(df["Date"].astype(str).unique().tolist())
    train_dates = [d for d in all_dates if d <= cutoff_train_end]
    val_dates   = [d for d in all_dates if d > cutoff_train_end]
    print(f"[phase3-B] 切分 cutoff={cutoff_train_end} | "
          f"train={len(train_dates)} 天（…{train_dates[-1] if train_dates else '—'}）| "
          f"val={len(val_dates)} 天（{val_dates[0] if val_dates else '—'}…）", flush=True)
    return train_dates, val_dates


def _ln_tag(ln: float) -> str:
    """0.5 -> 'ln0p5'，0.25 -> 'ln0p25'，0.0 -> 'ln0p0'，1.0 -> 'ln1p0'（檔名安全、不碰撞）。"""
    return "ln" + str(ln).replace(".", "p")


def _make_weights(listnet_5d: float) -> dict:
    """固定 mse_5d=1.0 / mse_10d=0.5；listnet_10d = listnet_5d / 2（保 2:1 比例）。"""
    return {
        "mse_5d": 1.0, "mse_10d": 0.5,
        "listnet_5d": float(listnet_5d), "listnet_10d": float(listnet_5d) / 2.0,
    }


# ------------------------------------------------------------
# 單組訓練：固定一組 listnet 權重，跑一次完整短線訓練
# ------------------------------------------------------------
def _train_one_listnet(
    df,
    train_dates,
    val_dates,
    listnet_5d: float,
    dropout: float,
    epochs: int,
    early_stop: int,
    drive_dir: Optional[str],
) -> dict:
    import experimental.short_model as sm   # 線上 dual 推論用的同一支；只暫時包裝、跑完還原

    tag = _ln_tag(listnet_5d)
    weights = _make_weights(listnet_5d)
    ckpt_name   = f"v6_short_B_{tag}.pt"
    status_path = None
    backup_dir  = None
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        status_path = f"{drive_dir}/status_short_B_{tag}.json"
        backup_dir  = f"{drive_dir}/checkpoints"

    print("\n" + "=" * 64, flush=True)
    print(f"[phase3-B] ▶ 訓練 listnet_5d={listnet_5d} (listnet_10d={weights['listnet_10d']}) "
          f"| dropout={dropout} | ckpt={ckpt_name} | epochs={epochs} early_stop={early_stop}", flush=True)
    print(f"           weights={weights}", flush=True)
    print("=" * 64, flush=True)

    # dropout 仍用 monkeypatch 固定（預設 0.1=基準，與 A 同款）；weights 走現成參數
    original_cls = sm.ShortModelV6
    sm.ShortModelV6 = functools.partial(original_cls, dropout=dropout)
    try:
        model, history = sm.train_short_model(
            df, train_dates, val_dates,
            epochs=epochs,
            early_stop=early_stop,
            weights=weights,                       # ← 本實驗的變因
            checkpoint_name=ckpt_name,
            checkpoint_backup_dir=backup_dir,
            status_path=status_path,
        )
    finally:
        sm.ShortModelV6 = original_cls

    ic5  = list(history.val_ic)
    ic10 = list(getattr(history, "val_ic_10d", []))
    if ic5:
        peak_idx = int(np.argmax(ic5))
        peak_ic5, peak_ep = float(ic5[peak_idx]), peak_idx + 1
        tail = ic5[peak_idx:]
        decay = float((tail[0] - tail[-1]) / max(len(tail) - 1, 1)) if len(tail) > 1 else 0.0
    else:
        peak_ic5, peak_ep, decay = 0.0, 0, 0.0

    print(f"[phase3-B] ✔ listnet_5d={listnet_5d} 完成 | 峰值 5d IC={peak_ic5:+.4f} @ep{peak_ep} "
          f"| 峰後每 epoch 平均下滑 {decay:+.4f}", flush=True)

    return {
        "listnet_5d": listnet_5d,
        "listnet_10d": weights["listnet_10d"],
        "dropout": dropout,
        "checkpoint": ckpt_name,
        "peak_ic_5d": peak_ic5,
        "peak_epoch": peak_ep,
        "post_peak_decay_per_epoch": decay,
        "epochs_ran": len(ic5),
        "val_ic_5d_curve": [round(x, 4) for x in ic5],
        "val_ic_10d_curve": [round(x, 4) for x in ic10],
        "val_loss_curve": [round(x, 5) for x in list(history.val_loss)],
    }


# ------------------------------------------------------------
# 主入口：listnet 權重 sweep
# ------------------------------------------------------------
def run_listnet_sweep(
    df,
    train_dates: Optional[Sequence[str]] = None,
    val_dates:   Optional[Sequence[str]] = None,
    listnet_5d_values: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    epochs:      int = 18,
    early_stop:  int = 10,
    dropout:     float = 0.1,
    cutoff_train_end: str = "2023-12-31",
    drive_dir:   Optional[str] = "/content/drive/MyDrive/MarketMamba_V6",
) -> list[dict]:
    """
    短線模型 listnet 權重 sweep。train/val 不傳則由 df 依 cutoff 自動切。
    dropout 預設 0.1（基準）；純 B 實驗請保持，若要驗 A×B 組合再改。
    每跑完一組就把累計結果寫進 Drive JSON（斷線可續）。回傳 list[dict] 並印對照表。
    """
    if train_dates is None or val_dates is None:
        train_dates, val_dates = build_dates(df, cutoff_train_end)

    values = list(listnet_5d_values)
    print(f"[phase3-B] listnet_5d sweep = {values} | dropout={dropout} | "
          f"基準 listnet_5d=0.5（峰 0.0951@ep8）", flush=True)

    result_json = f"{drive_dir}/phase3_B_listnet_sweep_result.json" if drive_dir else None
    results: list[dict] = []
    for ln in values:
        r = _train_one_listnet(df, train_dates, val_dates, ln, dropout, epochs, early_stop, drive_dir)
        results.append(r)
        if result_json:
            with open(result_json, "w") as f:
                json.dump({"experiment": "phase3_B_listnet_sweep",
                           "fixed": {"mse_5d": 1.0, "mse_10d": 0.5, "dropout": dropout},
                           "baseline": {"listnet_5d": 0.5, "peak_ic_5d": 0.0951, "peak_epoch": 8},
                           "results": results}, f, indent=1, ensure_ascii=False)
            print(f"[phase3-B] 已寫入 {result_json}", flush=True)

    _print_table(results)
    return results


def _print_table(results: list[dict]) -> None:
    """對照表（規則 7：數值明確顯示）。"""
    print("\n" + "=" * 70, flush=True)
    print("[phase3-B] listnet 權重 sweep 對照表（基準 listnet_5d=0.5 → 峰 0.0951@ep8）", flush=True)
    print("-" * 70, flush=True)
    print(f"{'listnet_5d':>10} | {'峰值5d IC':>10} | {'峰值ep':>6} | {'峰後/ep下滑':>11} | {'跑了ep':>6}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        flag = "  ← 勝基準" if r["peak_ic_5d"] > 0.0951 else ""
        print(f"{r['listnet_5d']:>10} | {r['peak_ic_5d']:>+10.4f} | {r['peak_epoch']:>6} | "
              f"{r['post_peak_decay_per_epoch']:>+11.4f} | {r['epochs_ran']:>6}{flag}", flush=True)
    print("=" * 70, flush=True)
    print("判讀：峰值 IC>0.0951／listnet=0 看貢獻多少／峰值延後或崩壞變緩＝listnet 強度有效旋鈕。",
          flush=True)
