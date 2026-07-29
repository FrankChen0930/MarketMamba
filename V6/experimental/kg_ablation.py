"""
kg_ablation.py — 決策2 的實質答案：GATv2 到底有沒有貢獻？
============================================================
目的
----
2026-07-29 實測發現線上知識圖譜是**壞的**：

  - 42,864 個節點，只有 2,510 個是真股票（其餘 ETF/權證）
  - 642,451 條邊中，滾動相關性邊 **0 條**（動態層從未生效）
  - 供應鏈邊來自 `_parse_tpex_html`，那支函式是 **regex 抓 HTML 裡所有 4 位數字**
  - 2330 的鄰居是電器電纜、化學工業、綠能環保；**產業邊一條都沒進去**
    （15 個名額被 4 條集團邊 + 11 條爬蟲垃圾佔滿）
  - 產業邊用未設 seed 的 `random.sample`，每次重建都不一樣

在這個狀態下，「GATv2 有沒有用」這個問題從來沒有被真正回答過——
線上 IC ~0.08 是**帶著一張雜訊圖**達到的。所以在投入更複雜的圖設計
（動態相關性、產業鏈）之前，先做最便宜也最決定性的三組對照。

三組（單一變因：圖）
--------------------
  A  no_gat    完全不使用 GAT（`use_gat=False`，跳過 graph_layer 與 gate）
  B  old_kg    現行圖 `knowledge_graph_cache.npz`
  C  v2_kg     重建圖 `knowledge_graph_v2.npz`（見 kg_builder_v2.py）

其餘全部凍結，與 v6_short 基準 run 相同：
  window=60, n_layers=3, dropout=0.2（Phase 3-A 定案）, weights=SHORT_WEIGHTS,
  LR=7e-5, weight_decay=1e-4, WARMUP_PCT=0.15, 切分 train ≤2023-12-31 / val 之後

判讀
----
  B > A 明顯      → 圖有貢獻，即使是雜訊圖（可能只是多了一層平滑/正則）
  C > B 明顯      → 圖的**品質**有貢獻 → 值得再投資選項 C（動態相關性、產業鏈）
  A ≈ B ≈ C       → GATv2 對 5d 沒有加值 → 砍掉，省參數與算力，
                    也不必再花時間做產業鏈邊
  C ≈ A < B       → 反直覺，要先懷疑實驗本身（例如 v2 圖的 stock_id 對不上）

⚠️ **要跟同 harness 的重跑值比，不要跟歷史值比**（Phase 3-A 的教訓：
   歷史 0.0951 沒重現，同 harness 重跑 dropout=0.1 只有 0.0870，
   直接拿歷史值當基準會把「有效」誤判成「無效」）。本檔三組都是同一輪跑出來的。

隔離保證
--------
- checkpoint / status 一律獨立檔名（`v6_short_KG_{arm}.pt`），**不覆蓋 v6_short.pt**
- 圖的切換用 monkeypatch `trainer.KG_CACHE_PATH`，不動受保護的 `trainer.py` 檔案
- `use_gat=False` 的 state_dict 少了 graph_layer/gate/norm_fuse，與線上 checkpoint
  不相容——這正是要用獨立檔名的原因

============================================================
怎麼用（Colab）
============================================================
前置：Cell 0→1→2→3 跑完（df 就緒、sys.path 已含 /content/MarketMamba/V6），
且 `knowledge_graph_v2.npz` 已上傳到 PROCESSED_DIR（本機用 kg_builder_v2.py 產生後上傳）。

    from experimental.kg_ablation import run_kg_ablation
    results = run_kg_ablation(
        df,
        epochs=18, early_stop=10,
        drive_dir="/content/drive/MyDrive/MarketMamba_V6",
    )

跑完尾端會印對照表，完整結果寫到 {drive_dir}/kg_ablation_result.json。
只想先跑一組：run_kg_ablation(df, arms=("no_gat",))。
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# 三組的定義：arm -> (use_gat, KG 檔名或 None)
ARMS: dict[str, tuple[bool, Optional[str]]] = {
    "no_gat": (False, None),
    "old_kg": (True, "knowledge_graph_cache.npz"),
    "v2_kg":  (True, "knowledge_graph_v2.npz"),
}

DROPOUT = 0.2      # Phase 3-A 定案值；本實驗不動它（單一變因＝圖）


def build_dates(df, cutoff_train_end: str = "2023-12-31", purge: bool = True):
    """
    切分。**預設啟用 purge + embargo**（協定 v2.0），這是與 phase3_a 的差別。

    短線模型的 label 是 5d/10d → horizon 取 **10**（最長者）。
    embargo 20 天，與協定 §5 一致。

    ⚠️ 若要與 phase3_a 的舊結果直接並列，須設 `purge=False`——
       但那組數字含邊界洩漏，並列時必須標註。
    """
    all_dates = sorted(df["Date"].astype(str).unique().tolist())
    if not purge:
        print("[kg-abl] ⚠ purge=False：使用零 purge 的舊切分，"
              "train 尾端 10 天的 label 落在 val 區間內", flush=True)
        return ([d for d in all_dates if d <= cutoff_train_end],
                [d for d in all_dates if d > cutoff_train_end])

    from experimental.splitters import train_val_split_dates
    return train_val_split_dates(all_dates, cutoff_train_end,
                                 horizon=10, embargo_days=20, label="kg-abl")


def _kg_summary(path: Path) -> str:
    """訓練前先把圖的規模印出來——不能只靠檔名相信自己載對了圖（規則 7）。"""
    if not path.exists():
        return f"{path.name}：**不存在**"
    import re
    d = np.load(path, allow_pickle=True)
    ids = [str(s) for s in d["stock_ids"]]
    real = sum(1 for s in ids if re.match(r"^\d{4}$", s))
    ea = d["edge_attr"]
    uniq, cnt = np.unique(np.round(ea, 2), return_counts=True)
    w = "｜".join(f"{a}:{b:,}" for a, b in zip(uniq, cnt))
    return (f"{path.name}：{len(ids):,} 節點（真股票 {real:,}）"
            f"｜{d['edge_index'].shape[1]:,} 邊｜權重 {w}")


def _train_one_arm(df, train_dates, val_dates, arm: str,
                   epochs: int, early_stop: int,
                   drive_dir: Optional[str]) -> dict:
    import experimental.short_model as sm
    from marketmamba.config import PROCESSED_DIR

    use_gat, kg_file = ARMS[arm]
    kg_path = str(Path(PROCESSED_DIR) / kg_file) if kg_file else None

    ckpt_name = f"v6_short_KG_{arm}.pt"
    status_path = backup_dir = None
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        status_path = f"{drive_dir}/status_short_KG_{arm}.json"
        backup_dir  = f"{drive_dir}/checkpoints"

    print("\n" + "=" * 68, flush=True)
    print(f"[kg-abl] ▶ arm={arm} | use_gat={use_gat} | ckpt={ckpt_name}", flush=True)
    if kg_path:
        print(f"[kg-abl]   圖：{_kg_summary(Path(kg_path))}", flush=True)
    else:
        print("[kg-abl]   圖：不使用", flush=True)
    print("=" * 68, flush=True)

    # dropout 固定 0.2（同 phase3_a 的 monkeypatch 慣例：
    # train_short_model 內部建模時沒傳 dropout）
    original_cls = sm.ShortModelV6
    sm.ShortModelV6 = functools.partial(original_cls, dropout=DROPOUT)
    try:
        model, history = sm.train_short_model(
            df, train_dates, val_dates,
            epochs=epochs, early_stop=early_stop,
            checkpoint_name=ckpt_name,
            checkpoint_backup_dir=backup_dir,
            status_path=status_path,
            use_gat=use_gat,
            kg_cache_path=kg_path,
        )
    finally:
        sm.ShortModelV6 = original_cls

    ic5  = list(history.val_ic)
    ic10 = list(getattr(history, "val_ic_10d", []))
    if ic5:
        pk = int(np.argmax(ic5))
        peak, peak_ep = float(ic5[pk]), pk + 1
        tail = ic5[pk:]
        decay = float((tail[0] - tail[-1]) / max(len(tail) - 1, 1)) if len(tail) > 1 else 0.0
    else:
        peak, peak_ep, decay = 0.0, 0, 0.0

    print(f"[kg-abl] ✔ {arm} 完成 | 峰值 5d IC={peak:+.4f} @ep{peak_ep}"
          f" | 峰後每 epoch 平均下滑 {decay:+.4f}"
          f" | 參數 {model.n_parameters:,}", flush=True)

    return {
        "arm": arm, "use_gat": use_gat, "kg_file": kg_file,
        "checkpoint": ckpt_name,
        "n_parameters": int(model.n_parameters),
        "peak_ic_5d": peak, "peak_epoch": peak_ep,
        "post_peak_decay_per_epoch": decay,
        "epochs_ran": len(ic5),
        "val_ic_5d_curve": [round(x, 4) for x in ic5],
        "val_ic_10d_curve": [round(x, 4) for x in ic10],
        "val_loss_curve": [round(x, 5) for x in list(history.val_loss)],
    }


def run_kg_ablation(df, arms: Sequence[str] = ("no_gat", "old_kg", "v2_kg"),
                    epochs: int = 18, early_stop: int = 10,
                    cutoff_train_end: str = "2023-12-31",
                    purge: bool = True,
                    drive_dir: Optional[str] = None) -> dict:
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"未知的 arm {a!r}，可用：{list(ARMS)}")

    train_dates, val_dates = build_dates(df, cutoff_train_end, purge=purge)
    out_path = f"{drive_dir}/kg_ablation_result.json" if drive_dir else None

    # 斷線續跑：已完成的 arm 直接沿用（Colab 三組合計數小時）
    results: dict[str, dict] = {}
    if out_path and os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f).get("arms", {})
            if results:
                print(f"[kg-abl] 沿用既有結果：{list(results)}", flush=True)
        except Exception:                                     # noqa: BLE001
            results = {}

    for arm in arms:
        if arm in results:
            print(f"[kg-abl] {arm} 已有結果，跳過（要重跑請刪 {out_path}）", flush=True)
            continue
        results[arm] = _train_one_arm(df, train_dates, val_dates, arm,
                                      epochs, early_stop, drive_dir)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"experiment": "kg_ablation", "dropout": DROPOUT,
                           "cutoff_train_end": cutoff_train_end,
                           # 切分設定必須寫進結果檔：purge 與否會改變 IC 的絕對水位，
                           # 沒記錄的話半年後看到這份 JSON 會不知道能不能跟別的數字並列
                           "purge": purge,
                           "purge_horizon": 10 if purge else 0,
                           "embargo_days": 20 if purge else 0,
                           "n_train_days": len(train_dates),
                           "n_val_days": len(val_dates),
                           "arms": results}, f, ensure_ascii=False, indent=2)
            print(f"[kg-abl] 已寫入 {out_path}", flush=True)

    print("\n" + "=" * 68)
    print("GAT 消融對照表（同 harness、同一輪；單一變因＝圖）")
    print("=" * 68)
    print(f"{'arm':10s} {'峰值 5d IC':>12s} {'峰值 ep':>8s} {'峰後下滑':>10s} {'參數':>12s}")
    for a in arms:
        r = results.get(a)
        if not r:
            continue
        print(f"{a:10s} {r['peak_ic_5d']:>+12.4f} {r['peak_epoch']:>8d} "
              f"{r['post_peak_decay_per_epoch']:>+10.4f} {r['n_parameters']:>12,}")

    base = results.get("no_gat", {}).get("peak_ic_5d")
    if base is not None:
        print("\n相對於 no_gat 的增量：")
        for a in arms:
            if a == "no_gat" or a not in results:
                continue
            d = results[a]["peak_ic_5d"] - base
            print(f"  {a:10s} {d:+.4f}"
                  + ("   （幅度小於 Phase 3-A 的 dropout 效應 +0.009，"
                     "不足以視為圖有貢獻）" if abs(d) < 0.009 else ""))
    print("=" * 68)
    return results
