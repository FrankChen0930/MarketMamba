"""
MarketMamba V6 — Training Status Recorder
==========================================
把 TrainingHistory 序列化為 training_status.json，供 PersonalOS「模型狀態」頁面使用。

資料流（2026-06 定案）：
  Colab 訓練中 → 每 epoch 寫到 Google Drive (MyDrive/MarketMamba_V6/training_status.json)
  訓練完成後 → 手動複製到 V6/results/training_status.json → git push
  Render 後端 → 從 GitHub raw 讀取 → 前端 ModelStatus 頁面

不依賴 torch / colab，本機與 Colab 皆可 import。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def dump_training_status(
    history,                        # TrainingHistory（duck-typed，避免 import torch）
    epoch: int,                     # 目前 epoch（1-based）
    epochs_max: int,                # 本輪訓練的目標 epoch 上限
    status: str,                    # "training" / "completed" / "early_stopped"
    meta: dict,                     # 靜態快照：model_version / started_at / early_stop_patience / config
    out_paths: list,                # 要寫出的檔案路徑（str 或 Path，可多個）
    epoch_seconds: list | None = None,
    quiet: bool = False,
) -> dict:
    """每個 epoch 結束時呼叫。寫出完整 history 曲線 + 最佳指標 + config 快照。"""
    best_ic = float(max(history.val_ic)) if history.val_ic else None
    best_ic_ep = int(history.best_ic_epoch) if history.val_ic else 0

    payload = {
        "model_version": meta.get("model_version", "V6"),
        "status": status,
        "started_at": meta.get("started_at"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "epochs_max": int(epochs_max),
        "best_val_ic": best_ic,
        "best_ic_epoch": best_ic_ep,
        "best_val_loss": float(min(history.val_loss)) if history.val_loss else None,
        "early_stop_patience": meta.get("early_stop_patience"),
        "history": {
            "train_loss":  [round(float(v), 5) for v in history.train_loss],
            "val_loss":    [round(float(v), 5) for v in history.val_loss],
            "val_ic":      [round(float(v), 5) for v in history.val_ic],
            "lr":          [float(f"{float(v):.3e}") for v in history.lr],
            "scale_gates": [
                [round(float(x), 4) for x in g]
                for g in getattr(history, "scale_gates", [])
            ],
            "epoch_seconds": [round(float(s), 1) for s in (epoch_seconds or [])],
        },
        "config": meta.get("config", {}),
    }

    written = []
    for p in out_paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        written.append(str(p))

    if not quiet:
        last_ic = history.val_ic[-1] if history.val_ic else float("nan")
        print(
            f"[training_status] {status} | ep {epoch}/{epochs_max} | "
            f"val_ic={last_ic:+.4f} | best={best_ic if best_ic is not None else float('nan'):+.4f}@ep{best_ic_ep} "
            f"| → {', '.join(written)}",
            flush=True,
        )
    return payload
