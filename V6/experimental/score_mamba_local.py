"""
score_mamba_local.py — 在本機 WSL 產生 Mamba 各組的組合層分數
==============================================================
用途：`portfolio_lab.py` 需要每個模型在 test 窗（582 天）的逐股分數。
Ridge/GBDT 本機可算，Mamba 需要 GPU 前向——**但不需要重訓**，
所以把 Colab 的 checkpoint 載下來、在 WSL 的 RTX 3060 上前向一次即可。

⚠️ 一個必須先講清楚的差異
--------------------------
F6 訓練/驗證用的特徵矩陣是在 Colab 建的（**自 2005 起**、59 維、無 Avail 旗標）。
本機沒有那一份，只有 `baseline_cache_v2/baseline_base_66d.parquet`（**自 2011 起**、
66 維 = 59 + 7 個 Avail 旗標）。兩者的 59 個特徵欄**建構方式完全相同**
（`fundamentals_v2=True`、`macro_norm="ts"`、同一套宇宙過濾），唯一的差別是：

    `macro_norm="ts"` 是 **expanding** z-score → 起點不同會讓 Group D 的 12 維數值不同。

這正是 `docs/f6-training-log-and-readout.md` §4.4 已經記錄的二階差異。
Group D 是每日橫斷面常數，但 Mamba 是非線性模型，**不能假設它免疫**。

→ 所以本檔**內建一個決定性的驗證**：把算出來的逐日 IC 拿去對 Colab 記錄的
`ic_by_day`（KG 四組對 `kg_ablation_result.json`，`v2_kg_nomacro` 對
`groupd_ablation_result_gatv2.json`）。**對得上就代表本機矩陣可用；對不上就不可用，
必須改回 Colab 產生分數**（`portfolio_lab.py --colab`）。

`v2_kg_nomacro`（2026-08-03 新增）
----------------------------------
F6 2×2 的最佳格：`no_macro + v2 圖`，訊號層 IC **+0.1145 / ICIR 1.340**（全專案新高）。
與其他 arm 的唯一差別是**輸入端把 Group D 那 12 欄歸零**——這一步不可省：
checkpoint 是在 Group D 全 0 的輸入上訓練的，若拿沒歸零的 macro 去前向，
推論輸入與訓練輸入不一致，**而且不會報錯**（維度相同）。這正是 `macro_norm`、
`fundamentals_v2` 已經踩過兩次的同一種坑。

歸零直接 import `groupd_ablation.zeroed_macro`，不另寫一份——另寫一份等於在
「有沒有 macro」之外多塞一個實作差異的變因（F5 方法紀律第 ③ 條）。
歸零的 context **必須包住整個推論迴圈**：`TemporalCrossSectionDataset` 是
lazy loading（tensor 在 `__getitem__` 才建），提早還原就會吃到未歸零的 macro。
另有一道保險——每個 arm 的**第一個 batch** 會印出 X 張量裡 Group D 那 12 個
位置的 absmax（歸零組應為 0.000000、其餘組應 > 0），確認歸零真的傳到模型輸入。

用法（WSL）
-----------
    wsl -d Ubuntu -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && \\
      conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && \\
      python V6/experimental/score_mamba_local.py --arms v2_kg_nomacro --max-days 40"

先用 `--max-days` 小樣本驗證 + 測速，過了再跑全部 582 天。
⚠️ `--max-days` 模式一律寫到 `{arm}__partialNd.parquet`，**不會覆蓋**正式的
`{arm}.parquet`（正式掃描用的分數檔必須是全量 582 天）。
輸出：`V6/experimental/result/scores/{arm}.parquet`（欄位 Date / stock_id / score）
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent.parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

# ── 59 維 config patch：必須在 import 任何 marketmamba.models.* 之前 ──────
# `architecture.py` 在 import 當下就把 GROUP_DIMS/INPUT_DIM 綁進
# `FactorGroupedEmbedding.__init__` 的預設參數（def 執行時求值）。
import marketmamba.config as cfg                                   # noqa: E402

_RS = ["RS_5d", "RS_20d", "RS_60d"]
if not all(r in cfg.FEATURE_GROUPS["price_momentum"] for r in _RS):
    cfg.FEATURE_GROUPS["price_momentum"] = cfg.FEATURE_GROUPS["price_momentum"] + _RS
cfg.INPUT_DIM = 59
cfg.FEATURE_COLS = (cfg.FEATURE_GROUPS["price_momentum"] + cfg.FEATURE_GROUPS["institutional_flow"]
                    + cfg.FEATURE_GROUPS["fundamentals"] + cfg.FEATURE_GROUPS["macro_environment"])
cfg.GROUP_DIMS = {k: len(v) for k, v in cfg.FEATURE_GROUPS.items()}
assert len(cfg.FEATURE_COLS) == 59, f"expected 59, got {len(cfg.FEATURE_COLS)}"
# RS 必須在 group A 的**末端**（位置 12–14）——放錯 proj_A 會吃到錯欄位且不會報錯
assert cfg.FEATURE_COLS[12:15] == _RS, f"RS 位置錯誤：{cfg.FEATURE_COLS[9:16]}"

from marketmamba.config import PROCESSED_DIR                        # noqa: E402

FEATURE_COLS = list(cfg.FEATURE_COLS)
MACRO_COLS = list(cfg.FEATURE_GROUPS["macro_environment"])
# Group D 在 59 維張量裡的位置（用欄名反查，不寫死 47:59）
MACRO_IDX = [FEATURE_COLS.index(c) for c in MACRO_COLS]
BASE_MATRIX = PROCESSED_DIR / "baseline_cache_v2" / "baseline_base_66d.parquet"
CKPT_DIRS = [Path("/mnt/d/Downloads"), Path("D:/Downloads"), PROCESSED_DIR.parent / "models"]
JSON_DIRS = [Path("/mnt/d/Downloads"), Path("D:/Downloads")]
RESULT_DIR = Path(__file__).resolve().parent / "result"
SCORE_DIR = RESULT_DIR / "scores"
KG_RESULT_JSON = "kg_ablation_result.json"      # 也是 val 日期的權威來源


@dataclass(frozen=True)
class Arm:
    """一組要評分的設定。`ref` = (Colab 結果 JSON 檔名, 該 JSON 裡的 arm key)。"""
    use_gat: bool
    kg_file: Optional[str]
    ckpt: str
    zero_macro: bool
    ref: tuple[str, str]


ARMS: dict[str, Arm] = {
    # ── F6 GAT 消融三組 + v3（2026-08-01，行為與改版前逐項相同） ──
    "no_gat":        Arm(False, None,                       "v6_short_KG_no_gat.pt",
                         False, (KG_RESULT_JSON, "no_gat")),
    "old_kg":        Arm(True,  "knowledge_graph_cache.npz", "v6_short_KG_old_kg.pt",
                         False, (KG_RESULT_JSON, "old_kg")),
    "v2_kg":         Arm(True,  "knowledge_graph_v2.npz",    "v6_short_KG_v2_kg.pt",
                         False, (KG_RESULT_JSON, "v2_kg")),
    "v3_kg":         Arm(True,  "knowledge_graph_v3.npz",    "v6_short_KG_v3_kg.pt",
                         False, (KG_RESULT_JSON, "v3_kg")),
    # ── F6 2×2 最佳格（2026-08-03 新增）：no_macro + v2 圖，IC +0.1145 ──
    "v2_kg_nomacro": Arm(True,  "knowledge_graph_v2.npz",    "v6_short_GD_no_macro_gatv2.pt",
                         True,  ("groupd_ablation_result_gatv2.json", "no_macro")),
}
DROPOUT = 0.2                  # 與 kg_ablation / groupd_ablation 相同（eval 模式下不生效，但架構要一致）
HISTORY_START = "2022-01-01"   # val 起點前需 ≥252 個交易日；2022-01 給約 490 天緩衝


def _find_ckpt(arm: str) -> Path:
    name = ARMS[arm].ckpt
    for d in CKPT_DIRS:
        p = d / name
        if p.exists():
            return p
    raise SystemExit(f"❌ 找不到 {name}，找過：{[str(d) for d in CKPT_DIRS]}\n"
                     f"   （Colab checkpoint 在 Drive 的 MarketMamba_V6/，下載到 D:\\Downloads\\）")


def _find_json(name: str) -> Optional[Path]:
    for d in JSON_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def _val_dates() -> list[str]:
    """直接沿用 Colab 記錄的 582 個 val 日期，避免本機重算切分時差一天。"""
    p = _find_json(KG_RESULT_JSON)
    if p is None:
        raise SystemExit(f"❌ 找不到 {KG_RESULT_JSON}（需要它取 val 日期）")
    d = json.loads(p.read_text(encoding="utf-8"))
    arm = next(iter(d["arms"].values()))
    return sorted(arm["ic_by_day"].keys())


def _ref_ic(arm: str) -> dict[str, float]:
    """取該 arm 在 Colab 記錄的逐日 IC（對不到就回空字典，驗證會標成 ⚠）。"""
    fname, key = ARMS[arm].ref
    p = _find_json(fname)
    if p is None:
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return ((d.get("arms") or {}).get(key) or {}).get("ic_by_day") or {}


def check_ref_windows(arms: list[str], val_dates: list[str]) -> None:
    """
    跑之前先確認：每個 arm 的對照 JSON 與 val 窗是**同一批日期**。
    不同批的話，後面那個「本機 vs Colab」的驗證就是在比兩段不同的期間——
    數字看起來會像有差異，但那是窗不同造成的，屬於會誤導判讀的靜默錯誤。
    """
    ok = True
    for arm in arms:
        fname, key = ARMS[arm].ref
        ref = _ref_ic(arm)
        if not ref:
            print(f"[窗檢查] ⚠ {arm:14s} 找不到對照 {fname}:{key} → 本輪無法驗證", flush=True)
            continue
        same = sorted(ref.keys()) == val_dates
        print(f"[窗檢查] {'✓' if same else '❌'} {arm:14s} ← {fname}:{key}"
              f"｜{len(ref)} 天（{min(ref)} → {max(ref)}）"
              f"｜與 val 窗{'完全相同' if same else '不同！'}", flush=True)
        ok &= same
    if not ok:
        raise SystemExit("❌ 對照窗與 val 窗不一致，停止（比出來的差異會混進「窗不同」這個變因）")


def load_df(val_dates: list[str]) -> pd.DataFrame:
    """從 v2 base matrix 取 59 維 + 標籤，組成 Dataset 需要的 df。"""
    t0 = time.time()
    cols = ["Date", "stock_id"] + FEATURE_COLS + ["Alpha_5d", "Alpha_20d"]
    filt = [("Date", ">=", pd.Timestamp(HISTORY_START)),
            ("Date", "<=", pd.Timestamp(val_dates[-1]))]
    df = pd.read_parquet(BASE_MATRIX, columns=cols, filters=filt)
    df["Date"] = pd.to_datetime(df["Date"])
    df["stock_id"] = df["stock_id"].astype(str)
    # ShortModelV6 是 2 頭 [5d, 10d]；矩陣沒有 Alpha_10d，用 Alpha_20d 佔位。
    # **評分只取 preds[:,0]，Y 完全不參與**，佔位不影響分數；
    # 只有下面的 IC 驗證會用到 Alpha_5d，那一欄是真值。
    # 真正的 Alpha_10d 來自 `label_10d.py` 建的統一標籤快照（窗內已證明與 base
    # matrix 逐位元同源）。舊版拿 Alpha_20d 當佔位是因為當時沒有 10d——現在有了，
    # 用真值才能算「同一份分數對三個評估 horizon 的 IC」。
    # ⚠️ 路徑不可從 `baseline_common.CACHE_DIR` 取——那個值依 `MM_PROTOCOL` 在
    #    import 期決定（沒設就是 v1 的 `baseline_cache`），而本檔的 WSL 指令不帶
    #    該環境變數 → 會靜默找不到檔案、退回 `Alpha_10d = Alpha_20d` 的佔位，
    #    表現成「vs Alpha_10d 與 vs Alpha_20d 的 IC 完全相同」（2026-08-03 實際踩到）。
    #    改為與本檔 BASE_MATRIX 相同的寫死路徑，兩者永遠指向同一個目錄。
    _lp = BASE_MATRIX.parent / "baseline_label_10d.parquet"
    if _lp.exists():
        _lab = pd.read_parquet(_lp, columns=["Date", "stock_id", "Alpha_10d"])
        _lab["Date"] = pd.to_datetime(_lab["Date"]); _lab["stock_id"] = _lab["stock_id"].astype(str)
        _n0 = len(df)
        df = df.merge(_lab, on=["Date", "stock_id"], how="left")
        assert len(df) == _n0, f"併 Alpha_10d 後列數變了：{_n0} → {len(df)}"
        print(f"[score] 併入真實 Alpha_10d：非空 {df['Alpha_10d'].notna().sum():,} / {len(df):,}",
              flush=True)
    else:
        df["Alpha_10d"] = df["Alpha_20d"]
        print("[score] ⚠ 找不到 baseline_label_10d.parquet → Alpha_10d 用 Alpha_20d 佔位"
              "（只影響診斷用的 IC，不影響分數）", flush=True)
    n_hist = df[df["Date"] < pd.Timestamp(val_dates[0])]["Date"].nunique()
    print(f"[score] 矩陣 {len(df):,} 列 × {df['stock_id'].nunique()} 支｜"
          f"{df['Date'].min().date()} → {df['Date'].max().date()}｜"
          f"val 起點前有 {n_hist} 個交易日（需 ≥252）｜{time.time()-t0:.0f}s", flush=True)
    assert n_hist >= 252, f"歷史不足（{n_hist} < 252），請把 HISTORY_START 往前調"
    return df


def score_arm(df, val_dates: list[str], arm: str, partial: bool = False,
              head: str = "5d") -> dict:
    import torch
    import experimental.short_model as sm
    import marketmamba.models.trainer as T
    from marketmamba.config import AMP_ENABLED
    from marketmamba.models.trainer import (
        TemporalCrossSectionDataset, build_kg_csr, compute_ic, get_batch_edges_csr,
        make_dataloader,
    )
    # Group D 歸零沿用 groupd_ablation 的同一份實作（F5 方法紀律 ③：不另寫一份）
    from experimental.groupd_ablation import describe_macro, zeroed_macro

    spec = ARMS[arm]
    # ShortModelV6 是雙頭 `[head_5d, head_10d]`（short_model.py:97-98，**沒有 20d 頭**）。
    # 換頭只是換讀 forward 輸出的哪一欄，不需要重訓、不動任何權重。
    head_idx = {"5d": 0, "10d": 1}[head]
    ck = _find_ckpt(arm)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*66}\n[score] arm={arm} use_gat={spec.use_gat} "
          f"kg={spec.kg_file or '—'} macro={'歸零' if spec.zero_macro else '照常'}\n"
          f"[score] ckpt={ck.name} dev={dev}\n{'='*66}", flush=True)

    # dataset 的標籤欄：ShortModelV6 是 2 頭
    _orig = (T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS)
    T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = ["Alpha_5d", "Alpha_10d"], [5, 10], [5, 10]
    try:
        model = sm.ShortModelV6(use_gat=spec.use_gat, dropout=DROPOUT).to(dev)
        state = torch.load(ck, map_location=dev)
        model.load_state_dict(state.get("state_dict", state))   # strict=True：載錯當場失敗
        model.eval()
        print(f"[score] checkpoint ep{state.get('epoch')} val_ic_5d={state.get('val_ic_5d')}"
              f"｜參數 {model.n_parameters:,}", flush=True)

        if spec.kg_file:
            _o = T.KG_CACHE_PATH
            T.KG_CACHE_PATH = Path(PROCESSED_DIR) / spec.kg_file
            try:
                kg, s2i = build_kg_csr()
            finally:
                T.KG_CACHE_PATH = _o
        else:
            kg, s2i = build_kg_csr()

        if spec.zero_macro:
            describe_macro(df, MACRO_COLS)      # 規則 7：歸零前先讓 Group D 的實測狀態看得見

        # ⚠️ context 必須包住整個推論迴圈：Dataset 是 lazy loading，
        #    tensor 在 __getitem__ 才建，提早還原就會吃到未歸零的 macro（且不報錯）
        ctx = zeroed_macro(df, MACRO_COLS) if spec.zero_macro else contextlib.nullcontext()
        with ctx:
            ds = TemporalCrossSectionDataset(df, val_dates, mode="val", n_sample=None)
            loader = make_dataloader(ds, shuffle=False)

            rows, ic_by_day = [], {}
            probe = None
            t0 = time.time()
            with torch.no_grad():
                for i, (X, Y, stks, _m) in enumerate(loader):
                    if X.shape[0] <= 1:
                        continue
                    if probe is None:       # 保險：確認歸零真的傳到模型輸入張量
                        probe = float(X[:, :, MACRO_IDX].abs().max())
                        exp = "應為 0.000000" if spec.zero_macro else "應 > 0"
                        bad = (probe > 0) if spec.zero_macro else (probe == 0)
                        print(f"[score] 第一個 batch：{X.shape[0]} 支｜"
                              f"Group D 12 個位置 absmax = {probe:.6f}（{exp}）"
                              f"{' ❌ 與預期不符！' if bad else ' ✓'}", flush=True)
                        if bad:
                            raise SystemExit("❌ Group D 輸入狀態與 arm 設定不符，停止"
                                             "（推論輸入與訓練輸入不一致，分數不可用）")
                    d = str(ds.valid_dates[i])[:10]
                    ei, ea = get_batch_edges_csr(stks, kg, s2i, dev)
                    with torch.amp.autocast('cuda', enabled=AMP_ENABLED and dev.type == "cuda"):
                        p = model(X.to(dev), ei, ea)
                    s = p[:, head_idx].float().cpu().numpy()
                    ic_by_day[d] = float(compute_ic(s, Y[:, 0].float().cpu().numpy()))
                    rows.append(pd.DataFrame({"Date": d, "stock_id": [str(x) for x in stks],
                                              "score": s.astype(np.float32)}))
                    if (i + 1) % 100 == 0:
                        el = time.time() - t0
                        print(f"  {i+1}/{len(ds.valid_dates)}｜{el:.0f}s｜"
                              f"ETA {el/(i+1)*(len(ds.valid_dates)-i-1):.0f}s", flush=True)
    finally:
        T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = _orig

    out = pd.concat(rows, ignore_index=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    # 小樣本模式絕不覆蓋正式分數檔——`portfolio_lab --sweep` 是 glob 整個 scores/，
    # 混進一個只有 N 天的 {arm}.parquet 會讓那個模型的整份掃描結果變成錯的
    sfx = "" if head == "5d" else f"__head{head}"
    dst = SCORE_DIR / (f"{arm}{sfx}__partial{len(ic_by_day)}d.parquet" if partial
                       else f"{arm}{sfx}.parquet")
    out.to_parquet(dst, index=False)

    ics = np.array([v for v in ic_by_day.values() if np.isfinite(v)])
    mine = float(ics.mean())
    print(f"[score] {arm}：{len(out):,} 列 / {len(ic_by_day)} 天 → {dst.name}"
          f"｜本機 mean IC {mine:+.4f}（{(time.time()-t0)/60:.1f} 分）", flush=True)

    # ── horizon 衰減矩陣：同一份分數對三個評估 horizon 的逐日 Spearman IC ──
    # 回答「標籤 horizon 要不要對齊持有期」時，這是最直接的證據：
    # 若對齊有實質內容，10d 頭對 Alpha_20d 的 IC 應明顯高於 5d 頭對 Alpha_20d。
    lab = df[["Date", "stock_id", "Alpha_5d", "Alpha_10d", "Alpha_20d"]].copy()
    lab["Date"] = lab["Date"].astype(str).str.slice(0, 10)
    mg = out.merge(lab, on=["Date", "stock_id"], how="left")
    assert len(mg) == len(out), f"併標籤後列數變了：{len(out)} → {len(mg)}"
    hz_ic = {}
    for h in ("5d", "10d", "20d"):
        per_day = []
        for d, g in mg.groupby("Date", sort=True):
            yv = g[f"Alpha_{h}"].to_numpy(np.float64)
            sv = g["score"].to_numpy(np.float64)
            m = np.isfinite(yv) & np.isfinite(sv)
            if m.sum() >= 30:
                per_day.append(float(compute_ic(sv[m], yv[m])))
        arr = np.array([x for x in per_day if np.isfinite(x)])
        hz_ic[h] = {"mean_ic": round(float(arr.mean()), 4),
                    "icir": round(float(arr.mean() / arr.std()), 3) if arr.std() else None,
                    "pct_pos": round(float((arr > 0).mean()), 3), "n_days": len(arr)}
    print(f"[horizon] {arm}｜head={head} 的分數對三個評估 horizon 的 IC：" +
          "｜".join(f"vs Alpha_{h} {hz_ic[h]['mean_ic']:+.4f}"
                    f"(ICIR {hz_ic[h]['icir']}, {hz_ic[h]['n_days']}天)"
                    for h in ("5d", "10d", "20d")), flush=True)

    # ── 決定性驗證：對 Colab 記錄的逐日 IC（只有 5d 頭有對照基準）──
    ref = _ref_ic(arm) if head == "5d" else {}
    v = {"arm": arm, "head": head, "n_days": len(ic_by_day),
         "local_mean_ic": round(mine, 4), "horizon_ic": hz_ic,
         "output": dst.name, "partial": partial}
    if head != "5d":
        print(f"[驗證] head={head} 無 Colab 逐日 IC 可對照（`val_ic_10d_curve` 是逐 epoch "
              f"純量、且對的是 Colab 自己的 Alpha_10d）→ 本輪不做決定性驗證。"
              f"該 checkpoint 的 5d 頭已在同一支程式下驗證通過。", flush=True)
    common = sorted(set(ic_by_day) & set(ref))
    if common:
        a = np.array([ic_by_day[k] for k in common])
        b = np.array([ref[k] for k in common])
        v.update({
            "colab_mean_ic": round(float(b.mean()), 4),
            "diff": round(float(a.mean() - b.mean()), 4),
            "corr_daily": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "max_abs_daily_diff": round(float(np.abs(a - b).max()), 4),
            "n_common": len(common),
        })
        ok = abs(v["diff"]) < 0.005 and v["corr_daily"] > 0.95
        v["verdict"] = "✅ 一致，本機矩陣可用" if ok else "❌ 不一致 → 改用 Colab 產生分數"
        print(f"[驗證] {arm}｜本機 {mine:+.4f} vs Colab {v['colab_mean_ic']:+.4f}"
              f"（差 {v['diff']:+.4f}）｜逐日相關 {v['corr_daily']:.4f}"
              f"｜最大單日差 {v['max_abs_daily_diff']:.4f}｜n={len(common)}\n"
              f"[驗證] {v['verdict']}", flush=True)
    else:
        v["verdict"] = "⚠ 無 Colab 逐日 IC 可對照"
        print(f"[驗證] {v['verdict']}", flush=True)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--head", choices=("5d", "10d"), default="5d",
                    help="讀 ShortModelV6 的哪一顆頭（沒有 20d 頭，那需要 Colab 重訓）")
    ap.add_argument("--max-days", type=int, default=0,
                    help="只跑前 N 個 val 日（小樣本驗證＋測速用；0 = 全部 582 天）"
                         "。輸出寫到 {arm}__partialNd.parquet，不覆蓋正式檔")
    a = ap.parse_args()

    vd = _val_dates()
    print(f"[score] Colab 記錄的 val 窗：{len(vd)} 天（{vd[0]} → {vd[-1]}）", flush=True)
    check_ref_windows(a.arms, vd)
    if a.max_days:
        vd = vd[:a.max_days]
        print(f"[score] ⚠ 小樣本模式：只跑前 {len(vd)} 天（輸出檔名加 __partial 後綴）", flush=True)

    df = load_df(vd)
    ver = [score_arm(df, vd, arm, partial=bool(a.max_days), head=a.head) for arm in a.arms]

    print(f"\n{'='*66}\n驗證彙總\n{'='*66}")
    for v in ver:
        print(f"  {v['arm']:14s} head={v.get('head','5d'):4s} 本機 {v['local_mean_ic']:+.4f}"
              f"｜Colab {v.get('colab_mean_ic', float('nan')):+.4f}"
              f"｜差 {v.get('diff', float('nan')):+.4f}"
              f"｜逐日相關 {v.get('corr_daily', float('nan'))}｜{v['verdict']}")
    if a.max_days:
        print("\n⚠ 小樣本模式的 parquet 帶 __partial 後綴，正式掃描前要用全量重跑。")
    out_json = RESULT_DIR / ("score_mamba_validation_partial.json" if a.max_days
                             else "score_mamba_validation.json")
    # merge key 要含 head，否則 10d 那輪會蓋掉同 arm 的 5d 紀錄

    # 依 arm 合併，不整份覆寫——舊版是直接 write，只跑一個 arm 就會把其他 arm
    # 先前的驗證紀錄靜默清掉（實際發生過：檔案裡只剩最後跑的那一個）
    prev: dict[str, dict] = {}
    if out_json.exists():
        try:
            prev = {(x["arm"], x.get("head", "5d")): x
                    for x in json.loads(out_json.read_text(encoding="utf-8"))}
        except Exception as e:                      # 壞檔不該讓整輪白跑
            print(f"[score] ⚠ 既有 {out_json.name} 讀不動（{e}），本輪改為重建", flush=True)
    new_keys = {(v["arm"], v.get("head", "5d")) for v in ver}
    kept = [f"{k[0]}/{k[1]}" for k in prev if k not in new_keys]
    prev.update({(v["arm"], v.get("head", "5d")): v for v in ver})
    out_json.write_text(json.dumps(list(prev.values()), indent=1, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[score] 驗證明細 → {out_json.name}"
          f"（本輪更新 {[v['arm'] for v in ver]}；保留既有 {kept or '無'}）", flush=True)
