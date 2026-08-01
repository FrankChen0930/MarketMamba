"""
groupd_ablation.py — F6 決策3：Group D（總經 12 維）到底有沒有貢獻？
====================================================================
背景
----
2026-07-29 實測確認：**V6.1 下 Group D 全部 12 維都是 0**。原始值其實有訊息
（`Fear_Greed` 5~78 共 67 個相異值、`Business_Signal` 23~40），是被
`macro_norm="cross"` 消滅的——macro 同日對所有股票同值 → 橫斷面 z-score → std=0。
也就是說**線上 IC ~0.08 是在整組 Group D 都死掉的狀態下達到的**，
到今天為止沒有任何證據顯示這 12 維有用。

V6.2 改成 `macro_norm="ts"`（expanding 時序 z-score）之後它們才第一次真的活著。
既有決策紀錄寫得很清楚：**先證實 Group D 有貢獻，再補資料源**
（`fear_greed` 停在 2026-04-24、`business_indicator` 停在 2026-02-01、
`fed_rate` 根本只有 1 個相異日期），否則是為未經證實的特徵接資料源。
這支腳本就是那個「先證實」。

兩組（單一變因：Group D 有沒有資訊）
------------------------------------
  with_macro   現況：Group D 12 維照常
  no_macro     **把那 12 欄在輸入端歸零**（維度／架構／參數量全部不變）

為什麼是 mask 而不是砍成 47 維
------------------------------
原始計畫寫的是「砍掉 Group D → INPUT_DIM 47」。但那會重演 GAT 消融
A(`no_gat`) vs B(`old_kg`) 的三個干擾項：`GROUP_DIMS` 一改，
`FactorGroupedEmbedding` 的 sub_dim 就重新分配（proj_A/B/C 都變大）、
**參數量改變**、建模時消耗的 RNG 也不同 →
(i) head 初始化不同 (ii) DataLoader 的打亂順序不同（實測 GAT 消融時
A 的第一個 batch 是 1,558 支、B 是 1,668 支，同一份 dataset 同一個 seed）。

改成把那 12 欄的**值**歸零：兩組逐參數同初始化、同資料順序、同參數量，
是真正乾淨的配對比較。這正是 2026-08-01 決策紀錄寫的
「要嘛所有組架構等價（用 mask 而非拿掉模組）」。

代價（誠實揭露）：`proj_D` 仍然存在，會貢獻一個 **bias 常數向量**。
但常數不含任何 macro 資訊、且對每一支股票都相同，模型可自行吸收，
資訊層面與「拿掉」等價。腳本會驗證兩組參數量完全相同。

跑之前先定好的判讀（不看到數字才選規則）
----------------------------------------
1. **`macro_norm="ts"` 下 Group D 是每日橫斷面常數**（同一天所有股票同值）。
   對線性模型，它對日內 rank IC 的貢獻**恰為 0**；Mamba 只能透過非線性層
   與個股特徵的**交互作用**用到它。
   → **「Δ ≈ 0」是預期中的合理結果之一，不代表程式壞掉。**
2. 門檻沿用 F6：峰值 Δ ≥ **+0.009** **且**配對 NW t ≥ **2** 才算有貢獻。
3. 結論一律寫成「**11 維有效特徵的貢獻**」——`FED_Rate` 是死維
   （`fed_rate.parquet` 只有 8 列 / 1 個相異日期 2004-01-01）。
   本檔啟動時會**實測列印**那 12 欄各自的相異值數與 std，把死維直接顯示出來，
   不靠讀文件相信（規則 7）。
4. 只有 Δ 顯著為正，才值得回頭補 `fear_greed`/`business_indicator`/`fed_rate`
   的資料源，以及把 macro 接進 `run_daily_update`。

省一半 GPU 時間：控制組直接沿用 `no_gat`
----------------------------------------
本實驗用 `use_gat=False`（理由：GAT 三組還沒出結果，不預設圖有用；
而且 no_gat 最便宜）。這使得 `with_macro` 與**已完成的 `no_gat`**
設定逐項相同（同 seed、同 dropout、同 epochs、同切分、同 df、同架構）
→ 預設 `control="import"` 直接從 `kg_ablation_result.json` 匯入當控制組，
**只需訓練 `no_macro` 一組（約 3.6 小時而不是 7.2）**。

匯入前會逐項核對 seed / dropout / purge / val_end / train_start /
train 與 val 天數 / use_gat，任一不符就拒絕匯入並要求改用 `control="rerun"`。
`control="rerun"` 會自己重跑控制組——那時兩者峰值 IC 應該**完全相同**，
本身就是一個很強的環境一致性檢查。

隔離保證
--------
- checkpoint 一律獨立檔名（`v6_short_GD_{arm}.pt`），**不覆蓋 `v6_short.pt`**
- 不改 `short_model.py`、不改受保護的 `marketmamba/models/`
- `df` 的 macro 欄在 `finally` 還原（訓練＋評估都在同一個 context 內完成）

====================================================================
怎麼用（Colab）
====================================================================
前置：Cell 0→1→2→3 跑完（df 就緒），且 GAT 消融的 `no_gat` 已完成。

    from experimental.groupd_ablation import run_groupd_ablation
    results = run_groupd_ablation(
        df,
        drive_dir="/content/drive/MyDrive/MarketMamba_V6",
    )

只想重跑控制組驗證環境一致：
    run_groupd_ablation(df, control="rerun", drive_dir=...)
"""
from __future__ import annotations

import contextlib
import functools
import json
import os
from typing import Optional, Sequence

import numpy as np

# 切分與評估**直接沿用 kg_ablation 的同一份程式**：
# 控制組要能跨檔重用，前提是切分與 IC 計算逐行相同。自己另寫一份就等於
# 在「有沒有 macro」之外多塞一個實作差異的變因（F5 方法紀律第 ③ 條）。
from experimental.kg_ablation import (           # noqa: E402
    DROPOUT, SEED, TRAIN_START_DEFAULT, VAL_END_DEFAULT,
    _eval_best_checkpoint, build_dates,
)

# arm -> 是否把 Group D 歸零
ARMS: dict[str, bool] = {
    "with_macro": False,
    "no_macro":   True,
}

# 控制組沿用 `no_gat` 的前提：那一輪是 epochs=10 / early_stop=5 跑的。
# `epochs` 同時決定 OneCycleLR 的排程長度（Cell 4 就是栽在這裡：epochs=100
# 讓暖身長達 15 個 epoch、模型從頭到尾沒進入退火階段），所以 epochs 不同
# 的兩輪**不可比**，匯入控制組時必須擋下來。
CONTROL_ARM = "no_gat"
CONTROL_EPOCHS = 10
CONTROL_EARLY_STOP = 5

MACRO_GROUP = "macro_environment"


# ============================================================
# 1. 診斷：Group D 這 12 維現在到底長什麼樣
# ============================================================
def macro_cols() -> list[str]:
    """從 config 取 Group D 欄名（runtime 讀取，因 Colab 會把 config patch 成 59 維）。"""
    import marketmamba.config as cfg
    return list(cfg.FEATURE_GROUPS[MACRO_GROUP])


def describe_macro(df, cols: Optional[list[str]] = None) -> dict:
    """
    列印 Group D 每一維的實測狀態（規則 7：不可只做邏輯、要看得到數字）。

    要回答三件事：
      1. **哪幾維是死的**（相異值數 ≤ 1 或 std = 0）→ 決定結論要寫「幾維有效」
      2. **是不是每日橫斷面常數**（同一天所有股票同值）→ 這決定了「Δ≈0 是預期
         結果之一」的前提成不成立。如果實測不是常數，代表 `macro_norm` 沒吃到
         `"ts"`，那整個實驗的解讀會不一樣，必須當場發現
      3. 整體尺度（std）是否已經標準化
    """
    cols = cols or macro_cols()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"df 缺少 Group D 欄位：{missing}（df 是不是不含 macro？）")

    print("\n" + "=" * 78, flush=True)
    print(f"[gd-abl] Group D 實測狀態（{len(cols)} 維，{len(df):,} 列）", flush=True)
    print("=" * 78, flush=True)
    print(f"{'欄位':22s} {'相異值':>10s} {'std':>10s} {'min':>9s} {'max':>9s}  狀態", flush=True)

    dead: list[str] = []
    stats: dict[str, dict] = {}
    for c in cols:
        s = df[c]
        nuq = int(s.nunique(dropna=True))
        sd = float(s.std())
        is_dead = nuq <= 1 or not np.isfinite(sd) or sd == 0.0
        if is_dead:
            dead.append(c)
        stats[c] = {"n_unique": nuq, "std": round(sd, 6),
                    "min": round(float(s.min()), 4), "max": round(float(s.max()), 4),
                    "dead": is_dead}
        print(f"{c:22s} {nuq:>10,} {sd:>10.6f} {float(s.min()):>9.3f} {float(s.max()):>9.3f}"
              f"  {'❌ 死維（整維無訊息）' if is_dead else '✓'}", flush=True)

    # 每日橫斷面常數檢查：抽 5 個交易日，看同一天內的相異值數
    sample_days = sorted(df["Date"].astype(str).unique())
    sample_days = sample_days[::max(len(sample_days) // 5, 1)][:5]
    per_day_uniques = []
    for d in sample_days:
        sub = df[df["Date"].astype(str) == d]
        per_day_uniques.append(max(int(sub[c].nunique()) for c in cols))
    all_const = all(u <= 1 for u in per_day_uniques)
    print(f"\n[gd-abl] 每日橫斷面常數檢查（抽 {len(sample_days)} 天，取 12 維中最大相異值數）："
          f"{per_day_uniques} → {'✓ 確為每日常數（macro_norm=ts 生效）' if all_const else ''}"
          f"{'' if all_const else '⚠️ 不是常數！macro_norm 可能不是 ts，判讀前提要重新確認'}",
          flush=True)

    live = len(cols) - len(dead)
    print(f"[gd-abl] 有效維度：**{live} 活 / {len(dead)} 死**"
          f"{'（死維：' + ', '.join(dead) + '）' if dead else ''}", flush=True)
    print(f"[gd-abl] → 本實驗的結論必須寫成「{live} 維有效特徵的貢獻」，不是 {len(cols)} 維",
          flush=True)
    print("=" * 78, flush=True)

    return {"cols": cols, "n_live": live, "dead": dead,
            "per_day_max_unique": per_day_uniques,
            "is_daily_constant": bool(all_const), "stats": stats}


# ============================================================
# 2. mask：把 Group D 歸零（可還原）
# ============================================================
@contextlib.contextmanager
def zeroed_macro(df, cols: Optional[list[str]] = None):
    """
    暫時把 `cols` 全部歸零，離開時還原。

    為什麼要還原：同一個 Colab session 裡兩組共用同一份 `df`，不還原的話
    控制組（若選 rerun）或後續任何實驗都會靜默吃到被歸零的 macro——
    那正是「不會報錯的錯誤」，跟 `train_short_model` 沒還原 `TARGET_COLS`
    是同一類坑（2026-07-31 已用 try/finally 修過一次）。

    記憶體：備份 12 欄 float32，約 8.0M 列 × 12 × 4B ≈ 385 MB（會印出實測值）。
    """
    cols = cols or macro_cols()
    saved = {c: df[c].to_numpy(copy=True) for c in cols}
    mb = sum(a.nbytes for a in saved.values()) / 2 ** 20
    try:
        for c in cols:
            df[c] = np.zeros(len(df), dtype=np.float32)
        # 驗證真的歸零了（規則 7：不能只做邏輯、要看得到數字）
        absmax = float(np.nanmax(np.abs(df[cols].to_numpy(dtype=np.float32))))
        print(f"[gd-abl] Group D 已歸零：{len(cols)} 欄 | 歸零後 absmax = {absmax:.6f}"
              f"（應為 0.000000）| 備份佔用 {mb:.0f} MB", flush=True)
        assert absmax == 0.0, f"歸零失敗，absmax={absmax}"
        yield
    finally:
        for c in cols:
            df[c] = saved[c]
        chk = float(np.nanmax(np.abs(df[cols].to_numpy(dtype=np.float32))))
        print(f"[gd-abl] Group D 已還原：absmax = {chk:.4f}（應 > 0）", flush=True)
        del saved


# ============================================================
# 3. 單組訓練
# ============================================================
def _train_one_arm(df, train_dates, val_dates, arm: str,
                   epochs: int, early_stop: int, use_gat: bool,
                   kg_file: Optional[str], drive_dir: Optional[str],
                   tag: str = "") -> dict:
    import experimental.short_model as sm
    from marketmamba.config import PROCESSED_DIR
    from pathlib import Path

    mask_macro = ARMS[arm]
    kg_path = str(Path(PROCESSED_DIR) / kg_file) if (use_gat and kg_file) else None

    # tag：同一個 arm 在不同設定下重跑時避免 checkpoint 撞名
    # （例如 no_macro 分別在 use_gat=False 與 use_gat=True 下各跑一次）
    ckpt_name = f"v6_short_GD_{arm}{tag}.pt"
    status_path = backup_dir = None
    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        status_path = f"{drive_dir}/status_short_GD_{arm}{tag}.json"
        backup_dir = f"{drive_dir}/checkpoints"

    print("\n" + "=" * 68, flush=True)
    print(f"[gd-abl] ▶ arm={arm} | Group D {'歸零' if mask_macro else '照常'} | "
          f"use_gat={use_gat} | ckpt={ckpt_name}", flush=True)
    print("=" * 68, flush=True)

    # seed 必須在建模與建 dataloader **之前**設（兩者都消耗 RNG）。
    # 兩組架構完全相同 → 逐參數同初始化、同資料打亂順序，是乾淨的配對比較。
    import random as _random

    import torch as _torch
    _torch.manual_seed(SEED)
    _torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    _random.seed(SEED)
    print(f"[gd-abl]   seed={SEED}（兩組架構相同＝逐參數同初始化、同 DataLoader 順序）",
          flush=True)

    cols = macro_cols()
    ctx = zeroed_macro(df, cols) if mask_macro else contextlib.nullcontext()

    with ctx:
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

        ic5 = list(history.val_ic)
        ic10 = list(getattr(history, "val_ic_10d", []))
        if ic5:
            pk = int(np.argmax(ic5))
            peak, peak_ep = float(ic5[pk]), pk + 1
            tail = ic5[pk:]
            decay = float((tail[0] - tail[-1]) / max(len(tail) - 1, 1)) if len(tail) > 1 else 0.0
        else:
            peak, peak_ep, decay = 0.0, 0, 0.0

        print(f"[gd-abl] ✔ {arm} 完成 | 峰值 5d IC={peak:+.4f} @ep{peak_ep}"
              f" | 峰後每 epoch 平均下滑 {decay:+.4f} | 參數 {model.n_parameters:,}", flush=True)

        # 逐日 IC + Top50 組合層。**必須在 context 內**，否則評估時 macro 已還原
        # ＝訓練與評估的特徵定義不一致（而且完全不會報錯）。
        extra = _eval_best_checkpoint(df, val_dates, arm, use_gat, kg_path, ckpt_name)

    return {
        "arm": arm, "macro_zeroed": mask_macro, "use_gat": use_gat,
        "seed": SEED, "checkpoint": ckpt_name,
        "epochs_param": epochs, "early_stop_param": early_stop,
        **extra,
        "n_parameters": int(model.n_parameters),
        "peak_ic_5d": peak, "peak_epoch": peak_ep,
        "post_peak_decay_per_epoch": decay,
        "epochs_ran": len(ic5),
        "val_ic_5d_curve": [round(x, 4) for x in ic5],
        "val_ic_10d_curve": [round(x, 4) for x in ic10],
        "val_loss_curve": [round(x, 5) for x in list(history.val_loss)],
    }


# ============================================================
# 4. 控制組匯入（沿用 GAT 消融的 no_gat）
# ============================================================
def import_control(kg_result_path: str, *, epochs: int, early_stop: int, purge: bool,
                   val_end: Optional[str], train_start: Optional[str],
                   n_train_days: int, n_val_days: int, use_gat: bool) -> Optional[dict]:
    """
    把 `kg_ablation_result.json` 的 `no_gat` 當作 `with_macro` 控制組。

    **逐項核對後才匯入**。不核對的話，一個「其實不同 harness」的控制組
    會給出一個看起來很合理的 Δ，而且永遠不會有人發現——這比沒有控制組更糟。
    """
    if not os.path.exists(kg_result_path):
        print(f"[gd-abl] ⚠ 找不到 {kg_result_path} → 無法匯入控制組", flush=True)
        return None
    with open(kg_result_path, encoding="utf-8") as f:
        d = json.load(f)
    arm = (d.get("arms") or {}).get(CONTROL_ARM)
    if not arm:
        print(f"[gd-abl] ⚠ {kg_result_path} 裡沒有 `{CONTROL_ARM}` → 無法匯入控制組", flush=True)
        return None

    checks = [
        ("seed",         d.get("seed"),                       SEED),
        ("dropout",      d.get("dropout"),                    DROPOUT),
        ("purge",        d.get("purge"),                      purge),
        ("val_end",      d.get("val_end"),                    val_end),
        ("train_start",  (d.get("spec") or {}).get("train_start"), train_start),
        ("n_train_days", d.get("n_train_days"),               n_train_days),
        ("n_val_days",   d.get("n_val_days"),                 n_val_days),
        ("use_gat",      arm.get("use_gat"),                  use_gat),
        # epochs 同時是 OneCycleLR 的排程長度 → 不同就不可比（Cell 4 的教訓）
        ("epochs",       CONTROL_EPOCHS,                      epochs),
        ("early_stop",   CONTROL_EARLY_STOP,                  early_stop),
    ]
    bad = [(k, a, b) for k, a, b in checks if a != b]
    print(f"\n[gd-abl] 控制組匯入核對（來源 {os.path.basename(kg_result_path)} / "
          f"arm={CONTROL_ARM}）：", flush=True)
    for k, a, b in checks:
        print(f"           {k:14s} 控制組={a!s:14s} 本輪={b!s:14s} "
              f"{'✓' if a == b else '❌'}", flush=True)
    if bad:
        print(f"[gd-abl] ❌ {len(bad)} 項不符 → **拒絕匯入**。"
              f"請改用 control='rerun' 自己重跑控制組。", flush=True)
        return None

    out = dict(arm)
    out.update({
        "arm": "with_macro", "macro_zeroed": False,
        "checkpoint": arm.get("checkpoint"),
        "epochs_param": CONTROL_EPOCHS, "early_stop_param": CONTROL_EARLY_STOP,
        "imported_from": {"file": kg_result_path, "arm": CONTROL_ARM},
        "note": ("直接沿用 GAT 消融的 no_gat：設定逐項相同（seed/dropout/epochs/切分/"
                 "架構/df 皆同），等價於重跑一次 with_macro，省約 3.6 小時 GPU"),
    })
    print(f"[gd-abl] ✅ 控制組匯入成功：峰值 5d IC {out['peak_ic_5d']:+.4f} @ep"
          f"{out['peak_epoch']} | 重評 IC {out.get('eval_mean_ic_5d')} | "
          f"參數 {out['n_parameters']:,}", flush=True)
    return out


# ============================================================
# 5. 主流程
# ============================================================
def run_groupd_ablation(df, arms: Sequence[str] = ("with_macro", "no_macro"),
                        epochs: int = CONTROL_EPOCHS,
                        early_stop: int = CONTROL_EARLY_STOP,
                        cutoff_train_end: str = "2023-12-31",
                        purge: bool = True,
                        val_end: Optional[str] = VAL_END_DEFAULT,
                        train_start: Optional[str] = TRAIN_START_DEFAULT,
                        use_gat: bool = False,
                        kg_file: Optional[str] = None,
                        control: str = "import",
                        kg_result_path: Optional[str] = None,
                        tag: str = "",
                        drive_dir: Optional[str] = None) -> dict:
    """
    Args:
        use_gat: 預設 **False**——GAT 三組還沒出結果，不預設圖有用；而且
                 `use_gat=False` 讓控制組可以直接沿用已完成的 `no_gat`。
                 若 F6 判定圖有貢獻，可改 True + 指定 `kg_file`，
                 但那時控制組必須用 `control="rerun"` 重跑。
        control: "import" = 沿用 `kg_ablation_result.json` 的 `no_gat`（預設，省 3.6 小時）
                 "rerun"  = 自己重跑控制組（峰值 IC 應與 no_gat 完全相同＝環境一致性檢查）
    """
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"未知的 arm {a!r}，可用：{list(ARMS)}")
    if control not in ("import", "rerun"):
        raise ValueError("control 只能是 'import' 或 'rerun'")
    if use_gat and not kg_file:
        raise ValueError("use_gat=True 時必須指定 kg_file（例如 'knowledge_graph_v2.npz'）")
    if use_gat and control == "import" and "with_macro" in arms:
        raise ValueError("use_gat=True 無法沿用 no_gat 當控制組 → 請用 control='rerun'")

    # ── 先把 Group D 的實測狀態印出來（決定結論要寫「幾維有效」）──
    diag = describe_macro(df)

    train_dates, val_dates = build_dates(df, cutoff_train_end, purge=purge,
                                         val_end=val_end, train_start=train_start)
    out_path = f"{drive_dir}/groupd_ablation_result{tag}.json" if drive_dir else None
    kg_result_path = kg_result_path or (f"{drive_dir}/kg_ablation_result.json"
                                        if drive_dir else None)

    # 斷線續跑
    results: dict[str, dict] = {}
    if out_path and os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                results = json.load(f).get("arms", {})
            if results:
                print(f"[gd-abl] 沿用既有結果：{list(results)}", flush=True)
        except Exception:                                     # noqa: BLE001
            results = {}

    def _dump() -> None:
        if not out_path:
            return
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "groupd_ablation",
                "design": "mask（Group D 歸零），非砍維度 → 兩組架構/參數量/RNG 完全相同",
                "dropout": DROPOUT, "seed": SEED,
                "epochs": epochs, "early_stop": early_stop,
                "use_gat": use_gat, "kg_file": kg_file, "tag": tag,
                "val_end": val_end,
                "spec": {"input_dim": 59, "availability_flags": False,
                         "fundamentals_v2": True, "macro_norm": "ts",
                         "neutralize": "none", "train_start": train_start,
                         "dead_features": diag["dead"]},
                "group_d_diagnosis": diag,
                "cutoff_train_end": cutoff_train_end,
                "purge": purge,
                "purge_horizon": 10 if purge else 0,
                "embargo_days": 20 if purge else 0,
                "n_train_days": len(train_dates),
                "n_val_days": len(val_dates),
                "arms": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[gd-abl] 已寫入 {out_path}", flush=True)

    for arm in arms:
        if arm in results:
            print(f"[gd-abl] {arm} 已有結果，跳過（要重跑請刪 {out_path}）", flush=True)
            continue
        if arm == "with_macro" and control == "import":
            imported = import_control(
                kg_result_path or "", epochs=epochs, early_stop=early_stop, purge=purge,
                val_end=val_end, train_start=train_start,
                n_train_days=len(train_dates), n_val_days=len(val_dates), use_gat=use_gat)
            if imported is not None:
                results[arm] = imported
                _dump()
                continue
            print("[gd-abl] → 改為自行訓練控制組", flush=True)
        results[arm] = _train_one_arm(df, train_dates, val_dates, arm,
                                      epochs, early_stop, use_gat, kg_file, drive_dir, tag)
        _dump()

    _print_table(results, arms, diag)
    return results


def _print_table(results: dict, arms: Sequence[str], diag: dict) -> None:
    print("\n" + "=" * 76)
    print("Group D 消融對照表（同 harness；單一變因＝Group D 有沒有資訊）")
    print("=" * 76)
    print(f"{'arm':12s} {'峰值 5d IC':>12s} {'峰值 ep':>8s} {'重評 IC':>9s} "
          f"{'ICIR':>7s} {'年化':>8s} {'參數':>12s}")
    for a in arms:
        r = results.get(a)
        if not r:
            continue
        pf = r.get("test_portfolio") or {}
        ann = f"{pf['ann_return']:+.1%}" if "ann_return" in pf else "—"
        ev = r.get("eval_mean_ic_5d")
        print(f"{a:12s} {r['peak_ic_5d']:>+12.4f} {r['peak_epoch']:>8d} "
              f"{(f'{ev:+.4f}' if ev is not None else '—'):>9s} "
              f"{str(r.get('eval_icir_5d', '—')):>7s} {ann:>8s} {r['n_parameters']:>12,}")

    base, test = results.get("with_macro"), results.get("no_macro")
    if base and test:
        # mask 設計的驗證：兩組參數量必須完全相同，不同就代表 mask 沒生效
        if base["n_parameters"] != test["n_parameters"]:
            print(f"\n❌ 兩組參數量不同（{base['n_parameters']:,} vs "
                  f"{test['n_parameters']:,}）→ mask 設計失效，結果不可用")
        else:
            print(f"\n✓ 兩組參數量相同（{base['n_parameters']:,}）＝ mask 設計成立、"
                  f"架構不是變因")

        d = base["peak_ic_5d"] - test["peak_ic_5d"]
        line = f"\nΔ = with_macro − no_macro（正 = Group D 有貢獻）：峰值 Δ={d:+.4f}"
        pa, pb = base.get("ic_by_day"), test.get("ic_by_day")
        t = float("nan")
        dd = None
        if pa and pb:
            common = sorted(set(pa) & set(pb))
            if len(common) >= 30:
                dd = np.array([pa[k] - pb[k] for k in common])
                try:
                    from experimental.baseline_common import newey_west_t
                    t = newey_west_t(dd, lag=5)
                except Exception:                            # noqa: BLE001
                    t = float("nan")
                line += (f"\n{'':4s}配對 Δ={dd.mean():+.4f} NW t={t:+.2f} "
                         f"Δ>0 {(dd > 0).mean():.1%} n={len(common)}")
        print(line)

        big = abs(d) >= 0.009
        sig = dd is not None and not np.isnan(t) and abs(t) >= 2.0
        if big and sig and d > 0:
            verdict = (f"✅ 有貢獻 → 值得回頭補 fear_greed / business_indicator / "
                       f"fed_rate 的資料源，並把 macro 接進 run_daily_update")
        elif big and sig and d < 0:
            verdict = "❌ 顯著變差 → Group D 是雜訊，考慮移除（INPUT_DIM 59 → 47）"
        elif big and not sig:
            verdict = "待覆核（幅度夠但不顯著）→ 看 Δ>0 比例；必要時多跑 seed 量 σ"
        else:
            verdict = ("無效應 → **不補資料源**。與「macro 是每日橫斷面常數、"
                       "只能靠交互作用發揮」的事前預期一致，不是程式壞掉")
        print(f"\n判定（門檻：|Δ|≥0.009 且 |NW t|≥2，跑之前已定）：{verdict}")
        print(f"\n⚠️ 結論必須寫成「**{diag['n_live']} 維有效特徵的貢獻**」"
              f"{'（死維：' + ', '.join(diag['dead']) + '）' if diag['dead'] else ''}")
        print("⚠️ 兩組差距落在 0.01 以內時，先拿一組多跑 2 個 seed 量 run-to-run σ，"
              "再宣告「沒有貢獻」——否則是拿雜訊當結論")
    print("=" * 76)
