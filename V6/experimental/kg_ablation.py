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
且對應的 `knowledge_graph_*.npz` 已放進 PROCESSED_DIR。

⚠️ **KG 檔不在 `processed_v6.zip` 裡也沒關係**——Cell 2 的 zip 只管 parquet，
   npz 只有幾十 KB～2 MB，單獨丟到 Drive 再複製過去即可，**不需要重壓縮上傳**：

    import shutil, os
    from marketmamba.config import PROCESSED_DIR
    for f in ("knowledge_graph_v2.npz", "knowledge_graph_v3.npz"):
        src = f"/content/drive/MyDrive/MarketMamba_V6/{f}"
        if os.path.exists(src):
            shutil.copy(src, str(PROCESSED_DIR / f))
            print(f"✅ {f} → {PROCESSED_DIR}")

    from experimental.kg_ablation import run_kg_ablation
    results = run_kg_ablation(
        df,
        epochs=10, early_stop=5,           # 預設值；改了就與既有三組不可比（§3 的教訓）
        drive_dir="/content/drive/MyDrive/MarketMamba_V6",
    )

跑完尾端會印對照表，完整結果寫到 {drive_dir}/kg_ablation_result.json。
**已有結果的 arm 會自動跳過**，所以要補跑 v3 只需：

    run_kg_ablation(df, arms=("v3_kg",), drive_dir="...")

訓練前會印出圖摘要（節點/邊/權重分布），**用它確認載對了圖**：
`v2_kg` 應是 32,083 邊、`v3_kg` 應是 **36,587** 邊（權重 0.6 有 4,504 條）。
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
    # 2026-08-01 新增：v2 + PIT 安全的殘差相關性邊（`kg_builder_v3.py`）。
    # **additive 模式**：v2 的 32,083 條邊逐條保留、只多加 4,504 條相關性邊
    # → 與 v2_kg 的差異**只有「多了相關性邊」一個變因**，
    #   是與 C−B（配對 NW t=5.17）同級的乾淨比較。
    # 判讀基準：v2_kg 的 +0.0991（峰值 Δ ≥ +0.009 且配對 NW t ≥ 2 才算有效）。
    "v3_kg":  (True, "knowledge_graph_v3.npz"),
}

DROPOUT = 0.2      # Phase 3-A 定案值；本實驗不動它（單一變因＝圖）

# ── seed（2026-07-30 補）────────────────────────────────────────────────
# 實測發現 `trainer.py` / `short_model.py` / `config.py` **全都沒有設 seed**，
# 三組原本除了圖還差「隨機初始化 + dataloader 打亂順序」。而 Phase 3-A 的教訓是
# 峰值 IC 是單 epoch 尖峰、+0.009 就算有效——run-to-run 變異可能與要測的效應同量級。
#
# 固定 seed 後：
#   B vs C（old_kg vs v2_kg）架構完全相同 → **初始化逐參數相同，是乾淨的配對比較**
#   A vs B  `use_gat=False` 少建 graph_layer/gate/norm_fuse，RNG 流在那之後分岔
#           → embedding 與 encoder 相同（它們在 __init__ 裡先建），但 head 不同。
#           已知限制，不假裝解決。
# ⚠️ 固定 seed 不等於量到了變異。若三組結果落在 0.01 以內，必須拿其中一組
#    多跑 2 個 seed 量出 σ，才能宣告「A≈B≈C」——否則是拿雜訊當結論。
SEED = 20260730

# F5 的 test 窗（協定 v2.0 §5）。val 上限截在這裡，理由有兩個：
#   ① 與 F5 的 Ridge/GBDT 用**同一批 582 個交易日**，數字才能並列
#   ② `macro_raw` 停在 2026-04-24，之後 Group D 是凍結值；截到 06-02 只殘留
#      約 6 週凍結，截更早會犧牲樣本，取平衡
VAL_END_DEFAULT = "2026-06-02"


# F6 定案的訓練起點（協定 v2.0 §9.4 / F5 的 R1）。
# 理由不是 IC——R1 實測 2012→2013 的 Δ 只有 +0.0001（t=0.46，無效應）——
# 而是資料品質：`institutional_raw` 對 prices 宇宙的命中率 2011 年 17%、
# 2012 年 56%、2013 年 74%、之後 80–96%。2013 之前 Group B 有 4~8 成的列
# 是 fillna(0) 補出來的，而「淨買超為 0」是合法值，模型在那段學到的是
# 「這支有沒有被資料涵蓋」而不是籌碼訊號。
TRAIN_START_DEFAULT = "2013-01-01"


def build_dates(df, cutoff_train_end: str = "2023-12-31", purge: bool = True,
                val_end: Optional[str] = VAL_END_DEFAULT,
                train_start: Optional[str] = TRAIN_START_DEFAULT):
    """
    切分。**預設啟用 purge + embargo**（協定 v2.0），這是與 phase3_a 的差別。

    ⚠️ `train_start`（2026-07-31 補）：`train_val_split_dates` 只在 cutoff **切一刀**，
       不看起點——不傳這個參數的話訓練集會是**矩陣的全部歷史 2005–2023**，
       而不是 F6 定案的 2013 起。那是靜默的規格違反（不會報錯、只是多吃 8 年
       籌碼欄多半是 fillna(0) 的資料），而且訓練集多 74%、每 epoch 從約 19 分
       變約 33 分、三組從 8.7 小時變約 15 小時。

    短線模型的 label 是 5d/10d → horizon 取 **10**（最長者）。
    embargo 20 天，與協定 §5 一致。

    ⚠️ 若要與 phase3_a 的舊結果直接並列，須設 `purge=False`——
       但那組數字含邊界洩漏，並列時必須標註。
    """
    all_dates = sorted(df["Date"].astype(str).unique().tolist())
    if train_start:
        n0 = len(all_dates)
        all_dates = [d for d in all_dates if d >= train_start]
        print(f"[kg-abl] 訓練起點 {train_start}（F6 定案）：可用日期 {n0} → {len(all_dates)} 天",
              flush=True)
    if purge:
        from experimental.splitters import train_val_split_dates
        train_dates, val_dates = train_val_split_dates(
            all_dates, cutoff_train_end, horizon=10, embargo_days=20, label="kg-abl")
    else:
        print("[kg-abl] ⚠ purge=False：使用零 purge 的舊切分，"
              "train 尾端 10 天的 label 落在 val 區間內", flush=True)
        train_dates = [d for d in all_dates if d <= cutoff_train_end]
        val_dates   = [d for d in all_dates if d > cutoff_train_end]

    if val_end:
        n0 = len(val_dates)
        val_dates = [d for d in val_dates if d <= val_end]
        print(f"[kg-abl] val 上限截至 {val_end}（與 F5 test 窗同批交易日）："
              f"{n0} → {len(val_dates)} 天", flush=True)
    print(f"[kg-abl] train {len(train_dates)} 天（…{train_dates[-1]}）"
          f"｜val {len(val_dates)} 天（{val_dates[0]}…{val_dates[-1]}）", flush=True)
    return train_dates, val_dates


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


def _eval_best_checkpoint(df, val_dates, arm: str, use_gat: bool,
                          kg_path: Optional[str], ckpt_name: str) -> dict:
    """
    載入該組**最佳 checkpoint**、重跑一次 val，輸出兩樣訓練迴圈沒給的東西：

      1. **逐日 val IC**（582 天的序列）—— 有這個才能做**配對** Newey-West 檢定。
         只有每 epoch 一個平均 IC 的話，三組之間只能比大小、比不出顯著性，
         而 F5 已經證明 Δ=0.002 級別的差異用肉眼判不了。
         同時也讓 F6 能與 F5 的 Ridge/GBDT 逐日配對。
      2. **Top50 組合層回測** —— 沿用 F5 同一套 `portfolio_backtest`
         （等權、5 日再平衡、買 0.15%/賣 0.45%），年化才能與 baseline 並列。
         Step 3 的教訓：IC 排名 ≠ 落袋排名，只看 IC 會選錯模型。

    ⚠️ `portfolio_backtest` 來自 `baseline_common`，而 **import 它時絕對不能設
       `MM_PROTOCOL=v2`**——那會觸發 `patch_config_67d()` 把 config 改成 66 維，
       與本輪 59 維的模型衝突。不設環境變數走 v1 分支即可（v1 分支只是把
       INPUT_DIM/FEATURE_COLS 設成 59 + RS，與 main 的 config 完全一致＝idempotent）。
    """
    import torch
    import experimental.short_model as sm
    from marketmamba.config import AMP_ENABLED, MODELS_DIR
    from marketmamba.models.trainer import (
        TemporalCrossSectionDataset, build_kg_csr, compute_ic,
        get_batch_edges_csr, make_dataloader,
    )

    ckpt_path = MODELS_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"[kg-abl] ⚠ 找不到 {ckpt_path}，跳過逐日 IC 與組合層", flush=True)
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sm.ShortModelV6(use_gat=use_gat, dropout=DROPOUT).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)          # strict=True：載錯組會當場失敗而不是靜默降級
    model.eval()

    val_ds = TemporalCrossSectionDataset(df, val_dates, mode="val", n_sample=None)
    val_loader = make_dataloader(val_ds, shuffle=False)

    if kg_path is not None:
        import marketmamba.models.trainer as T
        from pathlib import Path as _P
        _orig = T.KG_CACHE_PATH
        T.KG_CACHE_PATH = _P(kg_path)
        try:
            kg_csr, stock_to_idx = build_kg_csr()
        finally:
            T.KG_CACHE_PATH = _orig
    else:
        kg_csr, stock_to_idx = build_kg_csr()

    ic_by_day: dict[str, float] = {}
    all_dates, all_stocks, all_scores = [], [], []
    with torch.no_grad():
        for i, (X, Y, batch_stocks, _mask) in enumerate(val_loader):
            if X.shape[0] <= 1:
                continue
            date_str = val_ds.valid_dates[i]          # shuffle=False → 逐項對應
            X, Y = X.to(device), Y.to(device)
            edge_index, edge_attr = get_batch_edges_csr(batch_stocks, kg_csr, stock_to_idx, device)
            with torch.amp.autocast('cuda', enabled=AMP_ENABLED and device.type == "cuda"):
                preds = model(X, edge_index, edge_attr)
            s = preds[:, 0].float().cpu().numpy()
            y = Y[:, 0].float().cpu().numpy()
            # Spearman 對「每日 z-score」免疫 → 與 F5 對原始 Alpha 算的 rank IC 等價
            ic_by_day[str(date_str)[:10]] = float(compute_ic(s, y))
            all_dates.append(np.array([np.datetime64(str(date_str)[:10])] * len(s)))
            all_stocks.append(np.array([str(x) for x in batch_stocks]))
            all_scores.append(s)

    ics = np.array([v for v in ic_by_day.values() if not np.isnan(v)])
    out = {
        "eval_n_days": len(ics),
        "eval_mean_ic_5d": round(float(ics.mean()), 4) if len(ics) else None,
        "eval_icir_5d": round(float(ics.mean() / ics.std()), 3) if len(ics) and ics.std() > 0 else None,
        "eval_pct_pos": round(float((ics > 0).mean()), 3) if len(ics) else None,
        "ic_by_day": {k: round(v, 6) for k, v in ic_by_day.items() if not np.isnan(v)},
    }
    print(f"[kg-abl] {arm} 最佳 checkpoint 重評：{len(ics)} 天 | mean IC "
          f"{out['eval_mean_ic_5d']:+.4f} | ICIR {out['eval_icir_5d']} | "
          f"正比例 {out['eval_pct_pos']:.1%}", flush=True)

    try:
        from experimental.baseline_common import newey_west_t, portfolio_backtest
        out["eval_t_newey_west"] = round(float(newey_west_t(ics, lag=5)), 2)
        out["test_portfolio"] = portfolio_backtest(
            np.concatenate(all_dates), np.concatenate(all_stocks), np.concatenate(all_scores))
        pf = out["test_portfolio"]
        print(f"[kg-abl] {arm} 組合層：年化 {pf['ann_return']:+.1%} | Sharpe "
              f"{pf['ann_sharpe']} | MDD {pf['max_drawdown']:+.1%} | "
              f"換手 {pf['avg_turnover_per_rebalance']:.0%}", flush=True)
    except Exception as e:                                    # noqa: BLE001
        # 組合層失敗不能拖垮整輪訓練結果（那才是主產出）
        print(f"[kg-abl] ⚠ 組合層/NW t 計算失敗（不影響 IC 結果）：{type(e).__name__}: {e}",
              flush=True)
    return out


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

    # seed：三組共用同一個，讓「隨機初始化 + dataloader 打亂順序」不再是第二個變因。
    # 必須在建模與建 dataloader **之前**設（兩者都會消耗 RNG）。
    import torch as _torch
    _torch.manual_seed(SEED)
    _torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    import random as _random
    _random.seed(SEED)
    print(f"[kg-abl]   seed={SEED}（三組相同；B vs C 架構相同＝逐參數同初始化，"
          f"A 因少建 GAT 層而在 head 之後分岔）", flush=True)

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

    # 最佳 checkpoint 重評：逐日 IC（供配對檢定）+ Top50 組合層（供與 F5 並列）
    extra = _eval_best_checkpoint(df, val_dates, arm, use_gat, kg_path, ckpt_name)

    return {
        "arm": arm, "use_gat": use_gat, "kg_file": kg_file,
        "seed": SEED,
        "checkpoint": ckpt_name,
        **extra,
        "n_parameters": int(model.n_parameters),
        "peak_ic_5d": peak, "peak_epoch": peak_ep,
        "post_peak_decay_per_epoch": decay,
        "epochs_ran": len(ic5),
        "val_ic_5d_curve": [round(x, 4) for x in ic5],
        "val_ic_10d_curve": [round(x, 4) for x in ic10],
        "val_loss_curve": [round(x, 5) for x in list(history.val_loss)],
    }


def run_kg_ablation(df, arms: Sequence[str] = ("no_gat", "old_kg", "v2_kg"),
                    epochs: int = 10, early_stop: int = 5,
                    cutoff_train_end: str = "2023-12-31",
                    purge: bool = True,
                    val_end: Optional[str] = VAL_END_DEFAULT,
                    train_start: Optional[str] = TRAIN_START_DEFAULT,
                    drive_dir: Optional[str] = None) -> dict:
    """
    epochs/early_stop 預設 **10/5**（原為 18/10）。依據：Phase 3-A 兩組實測
    val 5d IC 峰值在 **ep3 與 ep4**，到 ep8 已從峰值掉 20–30% 且單調下滑
    （`status_short_A_do0p1/0p2.json` 的 `val_ic_5d` 曲線）。18/10 會在峰值之後
    多跑約 10 個 epoch，以每 epoch 約 19 分計＝每組浪費超過 3 小時。
    """
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"未知的 arm {a!r}，可用：{list(ARMS)}")

    train_dates, val_dates = build_dates(df, cutoff_train_end, purge=purge, val_end=val_end,
                                         train_start=train_start)
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
                           "seed": SEED,
                           "val_end": val_end,
                           # F6 規格指紋：日後看到這份 JSON 才知道它是哪一套特徵下的數字
                           "spec": {"input_dim": 59, "availability_flags": False,
                                    "fundamentals_v2": True, "macro_norm": "ts",
                                    "neutralize": "none", "train_start": train_start,
                                    # 已知：FED_Rate 整維恆為 0（fed_rate.parquet
                                    # 只有 <=1 個相異日期）→ 實質 58 維有訊息
                                    "dead_features": ["FED_Rate"]},
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
    print(f"{'arm':10s} {'峰值 5d IC':>12s} {'峰值 ep':>8s} {'重評 IC':>9s} "
          f"{'ICIR':>7s} {'年化':>8s} {'參數':>12s}")
    for a in arms:
        r = results.get(a)
        if not r:
            continue
        pf = r.get("test_portfolio") or {}
        ann = f"{pf['ann_return']:+.1%}" if "ann_return" in pf else "—"
        ev = r.get("eval_mean_ic_5d")
        print(f"{a:10s} {r['peak_ic_5d']:>+12.4f} {r['peak_epoch']:>8d} "
              f"{(f'{ev:+.4f}' if ev is not None else '—'):>9s} "
              f"{str(r.get('eval_icir_5d', '—')):>7s} {ann:>8s} {r['n_parameters']:>12,}")

    base = results.get("no_gat")
    if base is not None:
        print("\n相對於 no_gat 的增量（Δ = arm − no_gat）：")
        for a in arms:
            if a == "no_gat" or a not in results:
                continue
            d = results[a]["peak_ic_5d"] - base["peak_ic_5d"]
            line = f"  {a:10s} 峰值 Δ={d:+.4f}"
            # 配對日 IC 差 + NW t：只比兩個平均值分不出 0.002 級的差異
            pa, pb = results[a].get("ic_by_day"), base.get("ic_by_day")
            if pa and pb:
                common = sorted(set(pa) & set(pb))
                if len(common) >= 30:
                    dd = np.array([pa[k] - pb[k] for k in common])
                    try:
                        from experimental.baseline_common import newey_west_t
                        t = newey_west_t(dd, lag=5)
                    except Exception:                        # noqa: BLE001
                        t = float("nan")
                    line += (f" ｜配對 Δ={dd.mean():+.4f} NW t={t:+.2f} "
                             f"Δ>0 {(dd > 0).mean():.1%} n={len(common)}")
            if abs(d) < 0.009:
                line += "   （峰值幅度 <+0.009＝Phase 3-A 的 dropout 效應，不足以視為圖有貢獻）"
            print(line)
    print("\n⚠️ 判讀紀律：① 只與本輪同 harness 的組比，**不要**跟 Phase 3-A 的 0.0959 比"
          "（訓練集少 42% 且加了 purge）② 三組落在 0.01 以內時，先拿一組多跑 2 個 seed "
          "量 run-to-run σ，再宣告 A≈B≈C ③ 峰值 IC 與組合層年化可能不同調（Step 3 教訓）")
    print("=" * 68)
    return results
