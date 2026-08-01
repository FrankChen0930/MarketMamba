"""
wf_scores.py — Walk-forward 的**逐股 OOS 預測**（給 portfolio_lab 用）
=====================================================================
為什麼要這支
------------
`portfolio_lab` 目前所有結論都來自 **2024-01 ~ 2026-06 這單一 2.3 年多頭窗口**
＋ 1,500 格網格。這是唯一可能讓整套結論作廢的風險：
「20 日再平衡最好」「緩衝有效」「N 依賴因模型而異」可能只是這段行情的特性。

既有的 `_walk_forward_ridge` 只存每個 fold 的 **IC**，沒有存逐股預測 → 組合層用不了。
本檔補上：expanding walk-forward，每一季用「該季之前的全部資料」訓練，
在該季做**全解析度**預測，接起來就是一條 **2015–2026 連續、不重疊的 OOS 預測序列**。

有了它，就能把同一套組合建構網格套在十幾個不同 regime 上，
檢驗那些結論是**結構性**的還是**這段行情特有**的。

設計
----
- **expanding**：train = 該季開始日 − (purge + embargo) 之前的**全部**歷史
- **purge 5 + embargo 20 個交易日**：label 是 `rank_5d`（horizon=5），
  比協定凍結的 60 小——那個 60 是「多 horizon 模型取 max」用的，本檔只有 5d
- **α 固定** = 主切分選出的值（同既有 `_walk_forward_ridge` 的作法）。
  每 fold 各自選 α 會讓超參數吸收掉 regime 差異，就不再是乾淨的 OOS
- **逐年落檔可續跑**：`scores_wf_ridge_{year}.parquet`，已存在就跳過
  （本機長跑會被外部終止，見 f6 訓練紀錄 §6.1）

用法
----
    MM_PROTOCOL=v2 python V6/experimental/wf_scores.py --years 2015 2016 2017
    MM_PROTOCOL=v2 python V6/experimental/wf_scores.py --merge     # 併成 scores/wf_ridge.parquet
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

if os.environ.get("MM_PROTOCOL") != "v2":
    raise SystemExit("❌ 請設 MM_PROTOCOL=v2")

from experimental.baseline_common import (            # noqa: E402
    PROTOCOL, all_feature_names, load_xy,
)
from experimental.baseline_ridge_lasso import (       # noqa: E402
    TRAIN_STRIDE, gram_stats, ridge_solve, stats_add,
)

WF_DIR = _THIS / "result" / "wf"
SCORE_DIR = _THIS / "result" / "scores"

ALPHA = 10.0            # 主切分選出的 best α（見 baseline_ridge_lasso_result.json）
# GBDT 的輪數**固定**（不逐 fold early-stopping）：F5 在主切分上選出的 best_iteration
# 是 100/101/116，取 100。逐 fold 各自選輪數會讓超參數吸收掉 regime 差異，
# 就不再是乾淨的 OOS——與 Ridge 固定 α 同一個理由。
GBDT_ROUNDS = 100
PURGE_DAYS = 5          # label horizon（rank_5d）
EMBARGO_DAYS = 20       # 同協定 §5
MIN_TRAIN_YEARS = 3
WF_START_YEAR = 2015    # 2013 起 + 至少 3 年訓練 → 2016 才夠；2015 留給 2012 起的資料
DATA_START = "2012-01-01"


def _keep_mask() -> np.ndarray:
    """旗標關（同 R1／portfolio_lab 產生 ridge.parquet 時的設定）→ 300 維。"""
    return np.array([not n.startswith("Avail_") for n in all_feature_names()])


def build_quarterly_gram() -> tuple[list[pd.Timestamp], list[dict]]:
    """一次載入 stride-2 的全歷史，累積成「每季」的 Gram 統計量後就把大矩陣釋放。"""
    t0 = time.time()
    keep = _keep_mask()
    print(f"[wf] 載入 {DATA_START} → {PROTOCOL['TEST_END']}（stride={TRAIN_STRIDE}）...", flush=True)
    tr = load_xy(DATA_START, PROTOCOL["TEST_END"], day_stride=TRAIN_STRIDE)
    tr["X"] = np.ascontiguousarray(tr["X"][:, keep])
    gc.collect()
    print(f"[wf] {tr['X'].shape[0]:,} 列 × {tr['X'].shape[1]} 維 "
          f"({tr['X'].nbytes/2**30:.2f} GB)", flush=True)

    y = tr["rank_5d"]
    ok = ~np.isnan(y)
    d = pd.DatetimeIndex(tr["dates"])
    q = d.to_period("Q")
    qs, stats = [], []
    for p in sorted(q.unique()):
        m = np.asarray(q == p) & ok
        if m.sum() == 0:
            continue
        qs.append(p.to_timestamp())
        stats.append(gram_stats(tr["X"], y, m))
    print(f"[wf] 季 Gram 完成：{len(qs)} 季（{qs[0].date()} → {qs[-1].date()}）"
          f"｜{time.time()-t0:.0f}s", flush=True)
    del tr
    gc.collect()
    return qs, stats


def load_train_rows() -> dict:
    """GBDT 版：整份 train span 留在記憶體（無法像 Ridge 用 Gram 累積）。"""
    keep = _keep_mask()
    print(f"[wf] 載入 {DATA_START} → {PROTOCOL['TEST_END']}（stride={TRAIN_STRIDE}）...", flush=True)
    tr = load_xy(DATA_START, PROTOCOL["TEST_END"], day_stride=TRAIN_STRIDE)
    tr["X"] = np.ascontiguousarray(tr["X"][:, keep])
    gc.collect()
    print(f"[wf] {tr['X'].shape[0]:,} 列 × {tr['X'].shape[1]} 維 "
          f"({tr['X'].nbytes/2**30:.2f} GB)", flush=True)
    return tr


def run_year_gbdt(year: int, tr: dict) -> Path | None:
    """與 Ridge 版**完全相同的 fold 切分**，只換模型 → 兩者可直接並列。"""
    import lightgbm as lgb
    from experimental.f5_r_series import GBDT_PARAMS
    out = WF_DIR / f"scores_wf_gbdt_{year}.parquet"
    if out.exists():
        print(f"[wf] gbdt {year} 已存在，跳過", flush=True)
        return out
    t0 = time.time()
    keep = _keep_mask()
    te = load_xy(f"{year}-01-01", min(f"{year}-12-31", PROTOCOL["TEST_END"]), day_stride=1)
    if te["X"].shape[0] == 0:
        return None
    te["X"] = np.ascontiguousarray(te["X"][:, keep])
    gc.collect()
    dts = pd.DatetimeIndex(te["dates"])
    tr_dates = pd.DatetimeIndex(tr["dates"])
    y = tr["rank_5d"]
    ok = ~np.isnan(y)

    rows = []
    for qi in range(4):
        q_start = pd.Timestamp(year=year, month=1 + qi * 3, day=1)
        q_end = q_start + pd.offsets.QuarterEnd(0)
        sel = (dts >= q_start) & (dts <= q_end)
        if sel.sum() == 0:
            continue
        cut = q_start - pd.Timedelta(days=int((PURGE_DAYS + EMBARGO_DAYS) * 1.45))
        m = ok & (tr_dates < cut)
        n_years = (cut - pd.Timestamp(DATA_START)).days / 365.25
        if m.sum() == 0 or n_years < MIN_TRAIN_YEARS:
            print(f"[wf] gbdt {year}Q{qi+1} 訓練資料不足（{n_years:.1f} 年），跳過", flush=True)
            continue
        tq = time.time()
        ds = lgb.Dataset(tr["X"][m], label=y[m].astype(np.float64), free_raw_data=True)
        b = lgb.train(GBDT_PARAMS, ds, num_boost_round=GBDT_ROUNDS)
        sc = b.predict(te["X"][sel])
        rows.append(pd.DataFrame({"Date": te["dates"][sel], "stock_id": te["stock_ids"][sel],
                                  "score": sc.astype(np.float32)}))
        print(f"[wf] gbdt {year}Q{qi+1}｜train {int(m.sum()):>9,} 列 / {n_years:.1f} 年"
              f"｜test {int(sel.sum()):>7,} 列（{time.time()-tq:.0f}s）", flush=True)
        del ds, b
        gc.collect()

    del te
    gc.collect()
    if not rows:
        return None
    WF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"✅ [wf] gbdt {year}：{len(df):,} 列 → {out.name}（{(time.time()-t0)/60:.1f} 分）",
          flush=True)
    return out


def run_year(year: int, qs: list[pd.Timestamp], stats: list[dict]) -> Path | None:
    """把 `year` 的四季各自用『該季之前的全部歷史』訓練，在該季做全解析度預測。"""
    out = WF_DIR / f"scores_wf_ridge_{year}.parquet"
    if out.exists():
        print(f"[wf] {year} 已存在，跳過", flush=True)
        return out
    t0 = time.time()
    keep = _keep_mask()

    y0, y1 = f"{year}-01-01", f"{year}-12-31"
    te = load_xy(y0, min(y1, PROTOCOL["TEST_END"]), day_stride=1)
    if te["X"].shape[0] == 0:
        print(f"[wf] {year} 無資料，跳過", flush=True)
        return None
    te["X"] = np.ascontiguousarray(te["X"][:, keep])
    gc.collect()
    dts = pd.DatetimeIndex(te["dates"])

    rows = []
    for qi in range(4):
        q_start = pd.Timestamp(year=year, month=1 + qi * 3, day=1)
        q_end = q_start + pd.offsets.QuarterEnd(0)
        sel = (dts >= q_start) & (dts <= q_end)
        if sel.sum() == 0:
            continue
        # train：該季開始日往前推 purge+embargo 個「交易日」→ 用日曆日近似（×1.45 保守）
        cut = q_start - pd.Timedelta(days=int((PURGE_DAYS + EMBARGO_DAYS) * 1.45))
        use = [s for qq, s in zip(qs, stats) if qq < cut]
        n_years = (cut - pd.Timestamp(DATA_START)).days / 365.25
        if len(use) == 0 or n_years < MIN_TRAIN_YEARS:
            print(f"[wf] {year}Q{qi+1} 訓練資料不足（{n_years:.1f} 年），跳過", flush=True)
            continue
        st = use[0]
        for s in use[1:]:
            st = stats_add(st, s)
        w, c, _ = ridge_solve(st, ALPHA)
        sc = te["X"][sel] @ w + c
        rows.append(pd.DataFrame({"Date": te["dates"][sel],
                                  "stock_id": te["stock_ids"][sel],
                                  "score": sc.astype(np.float32)}))
        print(f"[wf] {year}Q{qi+1}｜train {st['n']:>9,} 列 / {n_years:.1f} 年"
              f"（截至 {cut.date()}）｜test {int(sel.sum()):>7,} 列", flush=True)

    del te
    gc.collect()
    if not rows:
        return None
    WF_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"✅ [wf] {year}：{len(df):,} 列 → {out.name}（{(time.time()-t0)/60:.1f} 分）", flush=True)
    return out


def merge(model: str = "ridge") -> Path:
    files = sorted(WF_DIR.glob(f"scores_wf_{model}_*.parquet"))
    if not files:
        raise SystemExit(f"❌ {WF_DIR} 裡沒有年度檔")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date", "stock_id"], keep="last").sort_values(["Date", "stock_id"])
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    dst = SCORE_DIR / f"wf_{model}.parquet"
    df.to_parquet(dst, index=False)
    n_days = df["Date"].nunique()
    print(f"✅ [wf] 合併 {len(files)} 個年度檔 → {dst.name}｜{len(df):,} 列 / {n_days} 個交易日"
          f"｜{df['Date'].min().date()} → {df['Date'].max().date()}", flush=True)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int,
                    default=list(range(WF_START_YEAR, 2027)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--model", choices=("ridge", "gbdt"), default="ridge")
    a = ap.parse_args()
    if a.merge:
        merge(a.model)
    elif a.model == "gbdt":
        tr = load_train_rows()
        for yr in a.years:
            run_year_gbdt(yr, tr)
    else:
        qs, stats = build_quarterly_gram()
        for yr in a.years:
            run_year(yr, qs, stats)
