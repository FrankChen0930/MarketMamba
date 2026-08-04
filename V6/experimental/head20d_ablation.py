"""
head20d_ablation.py — 標籤 horizon 實驗的 Mamba 版（Colab，尚未執行）
=====================================================================
問題（使用者 2026-08-03 提出）
-----------------------------
現行最佳設定是「**預測 5 天後的排序、每 20 天再平衡**」。直覺上再平衡頻率
應該不高於預測天數才對——持有 20 天卻只預測 5 天，聽起來怪。

本機已用 Ridge / GBDT 在 5d / 10d / 20d 三個標籤上做完（見
`docs/label-horizon-vs-holding-period-2026-08-03.md`）。本檔是 Mamba 版。

⚠️ 為什麼需要重訓，不能像 10d 頭那樣直接換讀輸出
------------------------------------------------
`ShortModelV6` 只有兩顆頭 `[head_5d, head_10d]`（`short_model.py:97-98`），
**沒有 20d 頭**。10d 的分數可以直接換讀 `forward` 輸出的第 1 欄（不必重訓，
`score_mamba_local.py --head 10d` 已經在做），但 20d 必須真的訓練出來。

設計：兩組，唯一變因是「第二顆頭學的是哪個 horizon」
----------------------------------------------------
    h10   第二顆頭學 Alpha_10d（控制組）
    h20   第二顆頭學 Alpha_20d

**實作用「換欄」而不是改程式**：`train_short_model` 在函式內部寫死
`T.TARGET_COLS = ["Alpha_5d", "Alpha_10d"]`（第 198–201 行，而且有 try/finally
還原——那是 2026-07-31 修過的坑），從外面 monkeypatch 會被它覆蓋。
所以改成把 `df["Alpha_10d"]` 這一欄的**值**暫時換成 `Alpha_20d`：

  - 架構、參數量、RNG 消耗、資料順序**完全相同** → 乾淨配對
  - 不動 `short_model.py`（它被 `run_dual_inference.py` import 做線上推論）
  - 與 F5 的「換欄」技巧同一招（那次省掉 4 次 27 分鐘的矩陣重建）

⚠️ 為什麼控制組不能沿用 2×2 最佳格
----------------------------------
那一格是 `purge_horizon=10` 跑的（`kg_ablation.build_dates` 寫死 horizon=10）。
20d 標籤需要 `purge_horizon=20`，否則訓練尾端與測試集有 20 天標籤重疊，
**會系統性偏袒 20d 那一組**——正是本實驗要檢定的對象。
兩組都改用 horizon=20 → 控制組必須跟著重訓，不能匯入。
（本機 Ridge/GBDT 版同樣三組共用 `purge=20`，理由一致。）

其餘設定逐項對齊 2×2 最佳格
---------------------------
seed 20260730｜dropout 0.2｜epochs 10｜early_stop 5｜use_gat=True｜
`knowledge_graph_v2.npz`｜**Group D 歸零**｜train 起點 2013｜val_end 2026-06-02

成本：約 3.6h × 2 = 7.2 小時 GPU。

用法（Colab，Cell 0→1→2→3 跑完、df 就緒後）
-------------------------------------------
    from experimental.head20d_ablation import run_head20d_ablation
    results = run_head20d_ablation(
        df, drive_dir="/content/drive/MyDrive/MarketMamba_V6")

跑完把兩個 checkpoint 下載到 `D:\\Downloads\\`，再在本機 WSL 產分數過 portfolio_lab：
    python V6/experimental/score_mamba_local.py --arms <arm> --head 10d
（第二顆頭一律用 `--head 10d` 讀，不論它學的是 10d 還是 20d。）
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# 切分與評估沿用同一份程式（F5 方法紀律 ③：不另寫一份，否則會多一個實作變因）
from experimental.kg_ablation import (           # noqa: E402
    DROPOUT, SEED, TRAIN_START_DEFAULT, VAL_END_DEFAULT, _eval_best_checkpoint,
)
from experimental.groupd_ablation import describe_macro, macro_cols, zeroed_macro  # noqa: E402

ARMS: dict[str, str] = {         # arm -> 第二顆頭實際學的標籤欄
    "h10": "Alpha_10d",
    "h20": "Alpha_20d",
}
KG_FILE = "knowledge_graph_v2.npz"
USE_GAT = True
ZERO_MACRO = True                # 2×2 最佳格是 no_macro
EPOCHS = 10
EARLY_STOP = 5
PURGE_HORIZON = 20               # ← 與 2×2 最佳格（10）不同，理由見檔頭
EMBARGO_DAYS = 20


@contextlib.contextmanager
def swapped_second_label(df, src_col: str):
    """
    暫時把 `Alpha_10d` 的值換成 `src_col` 的值，離開時還原。

    不還原的話，同一個 Colab session 裡後續任何用到 `Alpha_10d` 的東西
    都會靜默吃到 20d 的值——與 `train_short_model` 沒還原 `TARGET_COLS`
    是同一類坑（2026-07-31 已修過一次）。
    """
    if src_col == "Alpha_10d":
        print("[h20-abl] 第二顆頭用原本的 Alpha_10d（控制組，不換欄）", flush=True)
        yield
        return
    saved = df["Alpha_10d"].to_numpy(copy=True)
    try:
        df["Alpha_10d"] = df[src_col].to_numpy(copy=True)
        a, b = df["Alpha_10d"].to_numpy(), df[src_col].to_numpy()
        same = np.array_equal(a[np.isfinite(a)], b[np.isfinite(b)])
        print(f"[h20-abl] 已把 Alpha_10d 換成 {src_col} 的值｜逐位元相同 = {same}｜"
              f"std {np.nanstd(a):.5f}（換之前 {np.nanstd(saved):.5f}，"
              f"horizon 越長應越大）", flush=True)
        assert same, "換欄失敗"
        yield
    finally:
        df["Alpha_10d"] = saved
        print(f"[h20-abl] Alpha_10d 已還原｜std {np.nanstd(saved):.5f}", flush=True)


def build_dates_h20(df, cutoff: str = "2023-12-31"):
    """與 kg_ablation.build_dates 相同，但 horizon 改 20（那支寫死 10）。"""
    from experimental.splitters import train_val_split_dates
    all_dates = sorted(df["Date"].astype(str).unique().tolist())
    n0 = len(all_dates)
    all_dates = [d for d in all_dates if d >= TRAIN_START_DEFAULT]
    print(f"[h20-abl] 訓練起點 {TRAIN_START_DEFAULT}：可用日期 {n0} → {len(all_dates)} 天",
          flush=True)
    tr, va = train_val_split_dates(all_dates, cutoff, horizon=PURGE_HORIZON,
                                   embargo_days=EMBARGO_DAYS, label="h20-abl")
    if VAL_END_DEFAULT:
        va = [d for d in va if d <= VAL_END_DEFAULT]
    print(f"[h20-abl] purge_horizon={PURGE_HORIZON}（2×2 最佳格是 10，20d 標籤必須加長）"
          f"｜embargo={EMBARGO_DAYS}｜train {len(tr)} 天 / val {len(va)} 天", flush=True)
    return tr, va


def run_head20d_ablation(df, arms: Sequence[str] = ("h10", "h20"),
                         drive_dir: Optional[str] = None) -> dict:
    from experimental.short_model import train_short_model
    from marketmamba.config import PROCESSED_DIR

    train_dates, val_dates = build_dates_h20(df)
    kg_path = str(Path(PROCESSED_DIR) / KG_FILE) if USE_GAT else None
    cols = macro_cols()
    if ZERO_MACRO:
        describe_macro(df, cols)      # 規則 7：先讓 Group D 的實測狀態看得見

    out = {"experiment": "head20d_ablation",
           "design": "唯一變因＝第二顆頭學的 horizon；用『換欄』保持架構/RNG 完全相同",
           "seed": SEED, "dropout": DROPOUT, "epochs": EPOCHS, "early_stop": EARLY_STOP,
           "use_gat": USE_GAT, "kg_file": KG_FILE, "zero_macro": ZERO_MACRO,
           "purge_horizon": PURGE_HORIZON, "embargo_days": EMBARGO_DAYS,
           "cutoff_train_end": "2023-12-31", "val_end": VAL_END_DEFAULT,
           "n_train_days": len(train_dates), "n_val_days": len(val_dates),
           "note": ("控制組不可沿用 2×2 最佳格：那一格 purge_horizon=10，"
                    "與本輪的 20 不同，並列會混進第二個變因"),
           "arms": {}}

    for arm in arms:
        src = ARMS[arm]
        ckpt = f"v6_short_H_{arm}.pt"
        print(f"\n{'='*70}\n[h20-abl] ▶ arm={arm}｜第二顆頭學 {src}｜ckpt={ckpt}\n{'='*70}",
              flush=True)
        status = f"{drive_dir}/status_short_H_{arm}.json" if drive_dir else None

        # ⚠️ 2026-08-04 修：原本**完全沒有設 seed**（SEED 只被寫進輸出 JSON、沒套用）。
        #    `groupd_ablation._train_one_arm` 有做這件事，但本檔是直接呼叫
        #    `train_short_model`、繞過了它 → arm 1 跑完會消耗大量 RNG，
        #    arm 2 的參數初始化與 DataLoader 打亂順序都不同 ＝ **髒配對**，
        #    「唯一變因是第二顆頭的 horizon」這個宣稱會不成立。
        #    與 GAT 消融 A vs B 踩過的是同一個坑（見 f6-training-log-and-readout.md）。
        #    必須在建模與建 dataloader **之前**設（兩者都消耗 RNG）。
        import random as _random

        import torch as _torch
        _torch.manual_seed(SEED)
        _torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        _random.seed(SEED)
        print(f"[h20-abl]   seed={SEED} 已重設"
              f"（兩組架構相同＝逐參數同初始化、同 DataLoader 順序）", flush=True)

        mctx = zeroed_macro(df, cols) if ZERO_MACRO else contextlib.nullcontext()
        with swapped_second_label(df, src), mctx:
            hist = train_short_model(
                df, train_dates, val_dates,
                epochs=EPOCHS, early_stop=EARLY_STOP, checkpoint_name=ckpt,
                checkpoint_backup_dir=drive_dir, status_path=status,
                use_gat=USE_GAT, kg_cache_path=kg_path,
            )
            curve = list(getattr(hist, "val_ic", []) or [])
            peak = max(curve) if curve else float("nan")
            print(f"[h20-abl] ✔ {arm} 完成｜峰值 val IC（第 0 顆頭 5d）={peak:+.4f}"
                  f" @ep{curve.index(peak)+1 if curve else -1}｜共跑 {len(curve)} epoch",
                  flush=True)
            extra = _eval_best_checkpoint(df, val_dates, arm, USE_GAT, kg_path, ckpt)
        out["arms"][arm] = {"arm": arm, "second_head_label": src, "checkpoint": ckpt,
                            "val_ic_5d_curve": curve, "peak_ic_5d": peak, **(extra or {})}

    if drive_dir:
        p = Path(drive_dir) / "head20d_ablation_result.json"
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str),
                     encoding="utf-8")
        print(f"[h20-abl] 結果 → {p}", flush=True)

    print(f"\n{'='*70}\n判讀提醒\n{'='*70}")
    print("⚠️ 上面印的 val IC 是**第 0 顆頭（5d）**的，兩組本來就該接近——"
          "那正好是配對是否乾淨的檢查點，不是本實驗的結論。", flush=True)
    print("真正要比的是「第二顆頭的分數過 portfolio_lab 之後的組合層表現」：", flush=True)
    print("  1. 下載 v6_short_H_h10.pt / v6_short_H_h20.pt 到 D:\\Downloads\\", flush=True)
    print("  2. 在 ARMS 加這兩組，跑 score_mamba_local.py --head 10d", flush=True)
    print("  3. portfolio_lab --sweep，看 N=50/k=1.5/20 日那一格", flush=True)
    return out
