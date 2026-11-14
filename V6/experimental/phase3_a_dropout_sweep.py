"""
Phase 3 — 實驗 A：正則救峰值（dropout sweep，短線模型）
==========================================================
目的
----
短線模型 `v6_short.pt` 峰值 5d IC = 0.0951 @ep8，之後過擬合崩壞
（ep10→0.080、ep12→0.067）。checkpoint 存的是 best IC，所以部署模型本身
已是峰值；本實驗的目標不是「救 checkpoint」，而是用更強的 dropout **把峰值
本身推更高 / 更晚**——若過擬合被壓住，崩壞前的泛化更好，峰值 IC 可能 >0.0951。

單一變因（嚴格遵守「一次只改一個變因」）
-----------------------------------------
只動 `dropout`，其他全部凍結，與 v6_short.pt 基準 run 完全相同：
  - window=60, n_layers=3, weights=SHORT_WEIGHTS
  - LR=7e-5, weight_decay=1e-4, WARMUP_PCT=0.15, GRAD_CLIP_NORM=1.0
  - 資料切分：train ≤2023-12-31 / val 2024-01-01~（與短線基準同款）
基準 dropout=0.1（已知峰 0.0951@ep8）→ 掃 0.2 → 0.3。

隔離保證
--------
- 不修改 `short_model.py`（它被 run_dual_inference.py import 做線上推論）。
  本檔用 monkeypatch 暫時把 `ShortModelV6` 包成固定 dropout 的版本，跑完還原。
- checkpoint / status 一律用獨立檔名（v6_short_A_doXX.pt / status_short_A_doXX.json），
  **絕不覆蓋 production v6_short.pt**。
- weight_decay 不在本實驗變動（保 1e-4）；若 dropout 不夠，留作 A 之後的下一個單變因。

============================================================
怎麼用（Colab）
============================================================
前置：跑完 Cell 0→1→2→3（df / 特徵矩陣就緒、sys.path 已含 /content/MarketMamba/V6）。
然後新開一個 cell：

    from experimental.phase3_a_dropout_sweep import run_dropout_sweep
    results = run_dropout_sweep(
        df,                                   # Cell 3 建好的 feature matrix
        dropouts=(0.1, 0.2, 0.3),             # 要掃的 dropout 值
        epochs=18, early_stop=10,             # 觀察到峰值 + 一段崩壞即可
        drive_dir="/content/drive/MyDrive/MarketMamba_V6",   # 結果 JSON 存這（可 None）
    )

跑完畫面尾端會印一張對照表，並把完整結果寫到
  {drive_dir}/phase3_A_dropout_sweep_result.json
（含每個 dropout 的峰值 IC、峰值 epoch、完整 IC 曲線）。把那張表或 JSON 貼回來即可。

判讀標準
--------
看三件事任一改善＝dropout 是有效正則旋鈕、帶進趨勢模型：
  (1) 峰值 5d IC 是否 >0.0951
  (2) 峰值 epoch 是否延後（崩壞更晚來）
  (3) 崩壞是否變緩（峰後 IC 下滑斜率、val_loss 上升斜率）

備註
----
- 三個 dropout 各跑一次完整訓練；單尺度短線一個 epoch 不長，但三組合計仍是數小時，
  Colab 可能斷線——故每跑完一組就把當組結果寫進 Drive JSON（斷線後重跑只需補沒跑的值）。
- 若只想先跑一組：run_dropout_sweep(df, dropouts=(0.2,))。
"""

from __future__ import annotations

import functools
import json
import os
from typing import Iterable, Optional, Sequence

import numpy as np


# ------------------------------------------------------------
# 資料切分：與短線基準 run 同款（train ≤ cutoff / val 之後）
# ------------------------------------------------------------
def build_dates(df, cutoff_train_end: str = "2023-12-31"):
    """從 df 切出 train/val 日期（val = 2024-01-01 起，與 v6_short.pt 基準一致）。"""
    all_dates = sorted(df["Date"].astype(str).unique().tolist())
    train_dates = [d for d in all_dates if d <= cutoff_train_end]
    val_dates   = [d for d in all_dates if d > cutoff_train_end]
    print(f"[phase3-A] 切分 cutoff={cutoff_train_end} | "
          f"train={len(train_dates)} 天（…{train_dates[-1] if train_dates else '—'}）| "
          f"val={len(val_dates)} 天（{val_dates[0] if val_dates else '—'}…）", flush=True)
    return train_dates, val_dates


def _do_tag(do: float) -> str:
    """0.2 -> 'do0p2'，0.1 -> 'do0p1'，0.3 -> 'do0p3'（檔名安全、不碰撞）。"""
    return "do" + str(do).replace(".", "p")


# ------------------------------------------------------------
# 單組訓練：固定一個 dropout，跑一次完整短線訓練
# ------------------------------------------------------------
def _train_one_dropout(
    df,
    train_dates,
    val_dates,
    dropout: float,
    epochs: int,
    early_stop: int,
    drive_dir: Optional[str],
) -> dict:
    """monkeypatch ShortModelV6 -> 固定 dropout -> 呼叫現成 train_short_model -> 取峰值。"""
    import experimental.short_model as sm   # 線上 dual 推論用的同一支；只暫時包裝、跑完還原

    tag = _do_tag(dropout)
    ckpt_name   = f"v6_short_A_{tag}.pt"
    status_path = None
    backup_dir  = None
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        status_path = f"{drive_dir}/status_short_A_{tag}.json"
        backup_dir  = f"{drive_dir}/checkpoints"   # 與其他 Drive 備份同目錄，但檔名獨立

    print("\n" + "=" * 64, flush=True)
    print(f"[phase3-A] ▶ 訓練 dropout={dropout} | ckpt={ckpt_name} | "
          f"epochs={epochs} early_stop={early_stop}", flush=True)
    print("=" * 64, flush=True)

    # ── monkeypatch：把模組全域的 ShortModelV6 包成「dropout 固定」的版本 ──
    # train_short_model 內部用 ShortModelV6(window=.., n_layers=..) 建模、沒傳 dropout，
    # 故在此把預設 dropout 固定下來；weight_decay 仍由 train_short_model 內部維持 1e-4。
    original_cls = sm.ShortModelV6
    sm.ShortModelV6 = functools.partial(original_cls, dropout=dropout)
    try:
        model, history = sm.train_short_model(
            df, train_dates, val_dates,
            epochs=epochs,
            early_stop=early_stop,
            checkpoint_name=ckpt_name,
            checkpoint_backup_dir=backup_dir,
            status_path=status_path,
        )
    finally:
        sm.ShortModelV6 = original_cls   # 還原，避免 partial 疊加污染下一組

    ic5  = list(history.val_ic)                       # 主指標 = 5d IC（逐 epoch）
    ic10 = list(getattr(history, "val_ic_10d", []))   # 次要 = 10d IC
    if ic5:
        peak_idx = int(np.argmax(ic5))
        peak_ic5, peak_ep = float(ic5[peak_idx]), peak_idx + 1
        # 峰後 IC 下滑斜率（每 epoch 平均掉多少）——量化「崩壞快慢」
        tail = ic5[peak_idx:]
        decay = float((tail[0] - tail[-1]) / max(len(tail) - 1, 1)) if len(tail) > 1 else 0.0
    else:
        peak_ic5, peak_ep, decay = 0.0, 0, 0.0

    print(f"[phase3-A] ✔ dropout={dropout} 完成 | 峰值 5d IC={peak_ic5:+.4f} @ep{peak_ep} "
          f"| 峰後每 epoch 平均下滑 {decay:+.4f}", flush=True)

    return {
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
# 主入口：dropout sweep
# ------------------------------------------------------------
def run_dropout_sweep(
    df,
    train_dates: Optional[Sequence[str]] = None,
    val_dates:   Optional[Sequence[str]] = None,
    dropouts:    Iterable[float] = (0.1, 0.2, 0.3),
    epochs:      int = 18,
    early_stop:  int = 10,
    cutoff_train_end: str = "2023-12-31",
    drive_dir:   Optional[str] = "/content/drive/MyDrive/MarketMamba_V6",
) -> list[dict]:
    """
    短線模型 dropout sweep。train/val 不傳則由 df 依 cutoff 自動切（與基準一致）。
    每跑完一組就把累計結果寫進 Drive JSON（斷線可續）。回傳 list[dict]，並印對照表。
    """
    if train_dates is None or val_dates is None:
        train_dates, val_dates = build_dates(df, cutoff_train_end)

    dropouts = list(dropouts)
    print(f"[phase3-A] dropout sweep = {dropouts} | 基準 dropout=0.1（峰 0.0951@ep8）", flush=True)

    result_json = f"{drive_dir}/phase3_A_dropout_sweep_result.json" if drive_dir else None
    results: list[dict] = []
    for do in dropouts:
        r = _train_one_dropout(df, train_dates, val_dates, do, epochs, early_stop, drive_dir)
        results.append(r)
        # 每組跑完即落盤（Colab 斷線保險）
        if result_json:
            with open(result_json, "w") as f:
                json.dump({"experiment": "phase3_A_dropout_sweep",
                           "baseline": {"dropout": 0.1, "peak_ic_5d": 0.0951, "peak_epoch": 8},
                           "results": results}, f, indent=1, ensure_ascii=False)
            print(f"[phase3-A] 已寫入 {result_json}", flush=True)

    _print_table(results)
    return results


def _print_table(results: list[dict]) -> None:
    """對照表（規則 7：數值明確顯示）。"""
    print("\n" + "=" * 64, flush=True)
    print("[phase3-A] dropout sweep 對照表（基準 dropout=0.1 → 峰 0.0951@ep8）", flush=True)
    print("-" * 64, flush=True)
    print(f"{'dropout':>8} | {'峰值5d IC':>10} | {'峰值ep':>6} | {'峰後/ep下滑':>11} | {'跑了ep':>6}", flush=True)
    print("-" * 64, flush=True)
    for r in results:
        flag = "  ← 勝基準" if r["peak_ic_5d"] > 0.0951 else ""
        print(f"{r['dropout']:>8} | {r['peak_ic_5d']:>+10.4f} | {r['peak_epoch']:>6} | "
              f"{r['post_peak_decay_per_epoch']:>+11.4f} | {r['epochs_ran']:>6}{flag}", flush=True)
    print("=" * 64, flush=True)
    print("判讀：峰值 IC>0.0951 / 峰值 ep 延後 / 峰後下滑變緩 任一改善＝dropout 有效，帶進趨勢模型。",
          flush=True)
