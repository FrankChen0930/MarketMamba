"""
Phase 3 — 實驗 D：短線模型窗口 sweep（window 90/120）
==========================================================
目的
----
短線模型 `v6_short.pt` 是單尺度、取輸入最後 `window=60` 步（≈3 個月）。本實驗掃
window ∈ {60(基準), 90, 120}，看「給單尺度分支更長的回看」能否把峰值 5d IC 推過
0.0951——短線訊號可能在 1~2 個月有更多可用脈絡，但太長也可能稀釋近期動能。

⚠️ 前提脈絡（B/④ 已削弱本實驗）
-------------------------------
deep supervision（Phase 1 ②）顯示「多尺度 20/60/252 對 5d 完全冗餘」，故窗口長短
對 5d 的邊際效益可能有限；D 仍值得一跑＝它是「≤252 純切片、免重建矩陣」的便宜驗證，
即使持平也是「窗口長短對 5d 不敏感」的乾淨證據（穩賺資訊）。

單一變因
--------
只動 `window`，其餘全部凍結，與 v6_short.pt 基準 run 相同：
  - n_layers=3, dropout=0.1, weights=SHORT_WEIGHTS, LR=7e-5, wd=1e-4
  - 切分 train≤2023 / val 2024-
基準 window=60（峰 0.0951@ep8）→ 掃 90 → 120。
（window≤200 時取最後 window 步全為真實資料、不需 padding mask；120<200 安全。）

隔離保證
--------
- 不修改 `short_model.py`（線上 dual 推論 import 它）。window 走 train_short_model
  的現成參數；dropout 用 monkeypatch 留可選（預設 0.1=基準，與 A/B 同款）。
- checkpoint / status 用獨立檔名（v6_short_D_winXX.pt / status_short_D_winXX.json），
  **絕不覆蓋** production `v6_short.pt`。

============================================================
怎麼用（Colab）
============================================================
前置：跑完 Cell 0→1→2→3（df 就緒、sys.path 含 /content/MarketMamba/V6）。新開 cell：

    from experimental.phase3_d_window_sweep import run_window_sweep
    results = run_window_sweep(
        df,
        windows=(60, 90, 120),
        epochs=18, early_stop=10,
        dropout=0.1,                          # 純 D 實驗保持基準；要驗組合再改
        drive_dir="/content/drive/MyDrive/MarketMamba_V6",
    )

跑完印對照表，完整結果寫到 {drive_dir}/phase3_D_window_sweep_result.json。

判讀標準
--------
  (1) 峰值 5d IC 是否 >0.0951（更長窗口是否有用）
  (2) 峰值 epoch / 峰後下滑（更長窗口是否更穩或更易過擬合）
  (3) 若全部持平＝5d 對窗口長短不敏感（呼應「多尺度對 5d 冗餘」）

備註
----
- 三組各跑一次完整訓練；每跑完一組落盤 Drive JSON（Colab 斷線可續）。
- 想省時可先試 (60, 120) 兩端。
"""

from __future__ import annotations

import functools
import json
import os
from typing import Iterable, Optional, Sequence

import numpy as np


# ------------------------------------------------------------
# 資料切分（與短線基準 run 同款；與 A/B 一致）
# ------------------------------------------------------------
def build_dates(df, cutoff_train_end: str = "2023-12-31"):
    all_dates = sorted(df["Date"].astype(str).unique().tolist())
    train_dates = [d for d in all_dates if d <= cutoff_train_end]
    val_dates   = [d for d in all_dates if d > cutoff_train_end]
    print(f"[phase3-D] 切分 cutoff={cutoff_train_end} | "
          f"train={len(train_dates)} 天（…{train_dates[-1] if train_dates else '—'}）| "
          f"val={len(val_dates)} 天（{val_dates[0] if val_dates else '—'}…）", flush=True)
    return train_dates, val_dates


def _win_tag(win: int) -> str:
    return f"win{int(win)}"


# ------------------------------------------------------------
# 單組訓練：固定一個 window，跑一次完整短線訓練
# ------------------------------------------------------------
def _train_one_window(
    df,
    train_dates,
    val_dates,
    window: int,
    dropout: float,
    epochs: int,
    early_stop: int,
    drive_dir: Optional[str],
) -> dict:
    import experimental.short_model as sm   # 線上 dual 推論用的同一支；只暫時包裝 dropout、跑完還原

    tag = _win_tag(window)
    ckpt_name   = f"v6_short_D_{tag}.pt"
    status_path = None
    backup_dir  = None
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        status_path = f"{drive_dir}/status_short_D_{tag}.json"
        backup_dir  = f"{drive_dir}/checkpoints"

    print("\n" + "=" * 64, flush=True)
    print(f"[phase3-D] ▶ 訓練 window={window} | dropout={dropout} | ckpt={ckpt_name} | "
          f"epochs={epochs} early_stop={early_stop}", flush=True)
    print("=" * 64, flush=True)

    # window 走 train_short_model 現成參數；dropout 用 monkeypatch 固定（預設 0.1=基準）
    original_cls = sm.ShortModelV6
    sm.ShortModelV6 = functools.partial(original_cls, dropout=dropout)
    try:
        model, history = sm.train_short_model(
            df, train_dates, val_dates,
            epochs=epochs,
            early_stop=early_stop,
            window=window,                         # ← 本實驗的變因
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

    print(f"[phase3-D] ✔ window={window} 完成 | 峰值 5d IC={peak_ic5:+.4f} @ep{peak_ep} "
          f"| 峰後每 epoch 平均下滑 {decay:+.4f}", flush=True)

    return {
        "window": window,
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
# 主入口：window sweep
# ------------------------------------------------------------
def run_window_sweep(
    df,
    train_dates: Optional[Sequence[str]] = None,
    val_dates:   Optional[Sequence[str]] = None,
    windows:     Iterable[int] = (60, 90, 120),
    epochs:      int = 18,
    early_stop:  int = 10,
    dropout:     float = 0.1,
    cutoff_train_end: str = "2023-12-31",
    drive_dir:   Optional[str] = "/content/drive/MyDrive/MarketMamba_V6",
) -> list[dict]:
    """
    短線模型 window sweep。train/val 不傳則由 df 依 cutoff 自動切。
    dropout 預設 0.1（基準）。每跑完一組就把累計結果寫進 Drive JSON（斷線可續）。
    回傳 list[dict] 並印對照表。
    """
    if train_dates is None or val_dates is None:
        train_dates, val_dates = build_dates(df, cutoff_train_end)

    wins = [int(w) for w in windows]
    print(f"[phase3-D] window sweep = {wins} | dropout={dropout} | "
          f"基準 window=60（峰 0.0951@ep8）", flush=True)

    result_json = f"{drive_dir}/phase3_D_window_sweep_result.json" if drive_dir else None
    results: list[dict] = []
    for w in wins:
        r = _train_one_window(df, train_dates, val_dates, w, dropout, epochs, early_stop, drive_dir)
        results.append(r)
        if result_json:
            with open(result_json, "w") as f:
                json.dump({"experiment": "phase3_D_window_sweep",
                           "fixed": {"n_layers": 3, "dropout": dropout},
                           "baseline": {"window": 60, "peak_ic_5d": 0.0951, "peak_epoch": 8},
                           "results": results}, f, indent=1, ensure_ascii=False)
            print(f"[phase3-D] 已寫入 {result_json}", flush=True)

    _print_table(results)
    return results


def _print_table(results: list[dict]) -> None:
    """對照表（規則 7：數值明確顯示）。"""
    print("\n" + "=" * 66, flush=True)
    print("[phase3-D] window sweep 對照表（基準 window=60 → 峰 0.0951@ep8）", flush=True)
    print("-" * 66, flush=True)
    print(f"{'window':>7} | {'峰值5d IC':>10} | {'峰值ep':>6} | {'峰後/ep下滑':>11} | {'跑了ep':>6}", flush=True)
    print("-" * 66, flush=True)
    for r in results:
        flag = "  ← 勝基準" if r["peak_ic_5d"] > 0.0951 else ""
        print(f"{r['window']:>7} | {r['peak_ic_5d']:>+10.4f} | {r['peak_epoch']:>6} | "
              f"{r['post_peak_decay_per_epoch']:>+11.4f} | {r['epochs_ran']:>6}{flag}", flush=True)
    print("=" * 66, flush=True)
    print("判讀：峰值 IC>0.0951＝更長窗口有用；全部持平＝5d 對窗口長短不敏感（呼應多尺度冗餘）。",
          flush=True)
