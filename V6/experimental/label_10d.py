"""
label_10d.py — 建 `Alpha_10d` / `rank_10d`（標籤 horizon 實驗用）
=================================================================
`baseline_cache_v2/baseline_base_66d.parquet` 只留了 `Alpha_5d` 與 `Alpha_20d`
（`baseline_common.py:372` 的 `keep` 把 `Alpha_10d` 丟掉了），但 `build_features`
其實有算。要做「標籤 horizon vs 持有期」的實驗就需要 10d 這一格。

設計原則
--------
1. **不碰凍結的 base matrix**。輸出獨立 side file，逐列對齊 `BASE_PATH` 的行序，
   與 `baseline_common._derived_parts()` 的既有慣例相同（`load_xy` 會 assert 行序）。
2. **不重寫標籤邏輯**，直接呼叫 `feature_engineer._add_alpha_targets`
   （F5 方法紀律 ③：自己另寫一份 = 在「horizon」之外多塞一個實作差異的變因）。
3. **決定性驗證**：用同一支程式重建 `Alpha_5d` 與 `Alpha_20d`，要求與 base matrix
   內的值**逐位元相同**。對得上，才代表 `Alpha_10d` 是同一條路徑產出的；
   對不上就中止，不硬跑。

⚠️ 已查證的事實（2026-08-03，實測非讀碼）
-----------------------------------------
`Alpha_Nd` **不是**相對大盤的超額報酬，而是**原始前瞻報酬** `Close.shift(-N)/Close - 1`。
根因：`_add_alpha_targets` 用 `"TWII" in df_macro.columns` 決定要不要減基準，
而 `macro_raw` 的欄位叫 **`TWII_Close`**；把它改名成 `TWII` 的 `_merge_macro`
是在 `m = df_macro.copy()` 上做的 → **呼叫端的 `df_macro` 沒有 `TWII` 欄** → 走 else 分支。
實測：`Alpha_5d` vs `Fwd_5d` 的 `max|Δ| = 0.000e+00`（對照組差 0.183）。

**影響是零，不需要修**：標籤最後都轉成每日橫斷面 rank，而大盤前瞻報酬是**當日常數**，
減一個當日常數不改變當日排序；IC 是 Spearman，同樣對此免疫。
→ 本檔**刻意複製實際行為**（呼叫同一個函式、傳同一份 raw macro）。
   把它「修好」反而會讓 10d 與既有 5d/20d 不同義，破壞整個比較。

用法
----
    python V6/experimental/label_10d.py            # 建（已存在則跳過）
    python V6/experimental/label_10d.py --force    # 重建
輸出：`Data/processed_v6/baseline_cache_v2/baseline_label_10d.parquet`
      （欄位 Date / stock_id / Alpha_10d / rank_10d，行序與 base matrix 完全一致）
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

# `_DIM` 在 baseline_common 的 import 期就被綁定（66 vs 59），而 BASE_PATH 依它命名。
# 沒設 MM_PROTOCOL=v2 會指到不存在的 `baseline_base_59d.parquet`——那是「參數沒設對
# 就靜默讀到別份資料」的同一類坑，寧可在這裡當場失敗。
if os.environ.get("MM_PROTOCOL") != "v2":
    raise SystemExit("❌ 請設 MM_PROTOCOL=v2 再跑（否則 BASE_PATH 會指向 59d 那份）")

_V6 = Path(__file__).resolve().parent.parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

from experimental.baseline_common import (                      # noqa: E402
    BASE_PATH, CACHE_DIR, PROTOCOL, ROW_GROUP, _filter_universe, _load_raw,
)
from marketmamba.data.feature_engineer import _add_alpha_targets  # noqa: E402

OUT_PATH = CACHE_DIR / "baseline_label_10d.parquet"
HORIZONS_CHECK = (5, 20)        # 用來驗證重建路徑正確性的兩個 horizon


def _rank_transform(df: pd.DataFrame, col: str, out_col: str) -> pd.Series:
    """
    per-date pct-rank 置中 [-0.5, +0.5]。
    **逐行照抄 `baseline_common.build_base_matrix` 的 label 段**（該檔 396–405 行），
    包含 `method="average"`、`n > 1` 的守衛、以及 float32 轉型——
    這裡若與 rank_5d/rank_20d 有任何語意差異，三個 horizon 就不可比。
    """
    mask = (df["eligible"] & df[col].notna()).to_numpy()
    sub = df.loc[mask, ["Date", col]]
    r = sub.groupby("Date")[col].rank(method="average") - 1.0
    n = sub.groupby("Date")[col].transform("count")
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[mask] = np.where(n > 1, r / (n - 1.0) - 0.5, np.nan)
    return out.astype(np.float32)


def rebuild_labels() -> pd.DataFrame:
    """照 `build_base_matrix` 的前置步驟重建 Alpha_5d/10d/20d（全 horizon）。"""
    t0 = time.time()
    prices = _load_raw("prices_raw")
    n0 = len(prices)
    prices = _filter_universe(prices)
    prices = prices.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    # build_features 的第一件事就是這個排序（feature_engineer.py:126），
    # 而 _add_alpha_targets 的 shift(-n) 是**位置性**的 → 排序錯了值就全錯
    prices = prices.sort_values(["stock_id", "Date"]).reset_index(drop=True)
    print(f"[label10d] prices_raw {n0:,} → 過濾後 {len(prices):,} 列 | "
          f"{prices['stock_id'].nunique()} 支 | "
          f"{prices['Date'].min().date()} → {prices['Date'].max().date()}"
          f"（{time.time()-t0:.0f}s）", flush=True)

    macro = _load_raw("macro_raw")
    has_twii = macro is not None and "TWII" in macro.columns
    print(f"[label10d] macro 欄位有 'TWII' = {has_twii} → "
          f"{'減去大盤前瞻報酬' if has_twii else '**不減基準**，Alpha = 原始前瞻報酬'}"
          f"（與 base matrix 建立時同一個分支；見檔頭說明）", flush=True)

    lab = _add_alpha_targets(prices[["Date", "stock_id", "Close"]].copy(), macro)
    lab = lab[lab["Date"] >= pd.Timestamp(PROTOCOL["MATRIX_START"])]
    keep = ["Date", "stock_id", "Alpha_5d", "Alpha_10d", "Alpha_20d"]
    lab = lab[keep].reset_index(drop=True)
    assert not lab.duplicated(subset=["Date", "stock_id"]).any(), "重建的標籤有重複鍵"
    print(f"[label10d] 重建標籤 {len(lab):,} 列（{time.time()-t0:.0f}s）", flush=True)
    del prices
    gc.collect()
    return lab


def verify(base: pd.DataFrame, merged: pd.DataFrame) -> bool:
    """
    決定性驗證：重建的 Alpha_5d/20d 必須與 base matrix 內的值逐位元相同。

    **閘門只加在協定窗 [TRAIN_START, TEST_END] 之內，窗外據實列出但不擋。**
    理由（2026-08-03 實測後定的，不是看到數字才放寬）：`prices_raw` 是活的檔案，
    base matrix 建好之後仍會 ① 尾端 append 新交易日 ② 偶爾有歷史格被除息還原改寫。
    實測就抓到一格：**2949 在 2026-07-29 的 Close 由 59.0 → 56.969006**
    （比值 0.9656 ＝ 還原因子，隱含股利約 2.03 元），連帶讓
    `Alpha_20d @ 2026-06-29` 與 `Alpha_5d @ 2026-07-22` 兩列改變。
    這兩列都在 TEST_END（2026-06-02）之後 → 對訓練與測試皆無影響。

    窗外不擋、窗內零容忍，才能同時做到「不被無關的資料漂移卡住」與
    「真正會影響結論的漂移一定擋下來」。窗外的差異一律印出來供人工判讀。

    NaN 型態也分開看：只有重建有值 = 尾端新資料算得出標籤了（預期）；
    只有快取有值 = 資料倒退（不該發生，窗內視為失敗）。
    """
    lo = pd.Timestamp(PROTOCOL["TRAIN_START"])
    hi = pd.Timestamp(PROTOCOL["TEST_END"])
    inw = ((merged["Date"] >= lo) & (merged["Date"] <= hi)).to_numpy()
    ok_all = True
    print("=" * 78, flush=True)
    print(f"[驗證] 重建路徑正確性｜協定窗 {lo.date()} → {hi.date()}："
          f"{inw.sum():,} 列（窗外 {(~inw).sum():,} 列，僅報告不擋）", flush=True)
    for h in HORIZONS_CHECK:
        a = merged[f"Alpha_{h}d__cached"].to_numpy(np.float32)
        b = merged[f"Alpha_{h}d__rebuilt"].to_numpy(np.float32)
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb

        for tag, sel, gate in (("窗內", both & inw, True), ("窗外", both & ~inw, False)):
            if not sel.any():
                continue
            d = np.abs(a[sel] - b[sel])
            mx, n_bad = float(d.max()), int((d > 0).sum())
            print(f"  Alpha_{h}d {tag}：{sel.sum():,} 列｜**max|Δ| = {mx:.3e}**｜"
                  f"不同 {n_bad:,} 列（{n_bad/sel.sum():.6%}）"
                  f"{'  ← 閘門' if gate else '  ← 僅報告'}", flush=True)
            if n_bad:
                bad = merged.loc[sel & (np.abs(a - b) > 0), ["Date", "stock_id"]]
                for _, r in bad.head(5).iterrows():
                    print(f"            差異列：{r['Date'].date()} {r['stock_id']}", flush=True)
            if gate and mx != 0.0:
                print(f"  ❌ Alpha_{h}d 窗內出現差異 → base matrix 已不可信，中止", flush=True)
                ok_all = False

        only_cached_in = int((fa & ~fb & inw).sum())
        only_rebuilt = int((~fa & fb).sum())
        print(f"  Alpha_{h}d NaN 型態：窗內「只有快取有值」{only_cached_in:,}（應為 0）｜"
              f"全期「只有重建有值」{only_rebuilt:,}（尾端新資料，預期 > 0）", flush=True)
        if only_cached_in:
            print(f"  ❌ Alpha_{h}d 窗內資料倒退", flush=True)
            ok_all = False
        if only_rebuilt:
            dts = merged.loc[(~fa) & fb, "Date"]
            print(f"            → 範圍 {dts.min().date()} → {dts.max().date()}"
                  f"（全在 TEST_END 之後：{bool((dts > hi).all())}）", flush=True)
    print(f"[驗證] {'✅ 通過：協定窗內與 base matrix 逐位元同源' if ok_all else '❌ 未通過'}",
          flush=True)
    print("=" * 78, flush=True)
    return ok_all


def build(force: bool = False, check_only: bool = False) -> Path:
    """check_only=True 時只跑漂移檢查、不寫任何檔（回答「prices_raw 自 base matrix
    建立後有沒有被改過」——這是每次要拿凍結快取做結論之前該問的一句話）。"""
    if OUT_PATH.exists() and not force and not check_only:
        print(f"[label10d] 已存在，跳過：{OUT_PATH.name}（--force 可重建）", flush=True)
        return OUT_PATH

    t0 = time.time()
    base = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id", "eligible",
                                               "Alpha_5d", "Alpha_20d",
                                               "rank_5d", "rank_20d"])
    base["Date"] = pd.to_datetime(base["Date"])
    base["stock_id"] = base["stock_id"].astype(str)
    # 先改名，否則下面重算 rank 時會把快取的值蓋掉、無從驗證
    base = base.rename(columns={"rank_5d": "rank_5d__cached",
                                "rank_20d": "rank_20d__cached"})
    print(f"[label10d] base matrix {len(base):,} 列｜{base['stock_id'].nunique()} 支｜"
          f"{base['Date'].min().date()} → {base['Date'].max().date()}", flush=True)

    lab = rebuild_labels()
    lab["Date"] = pd.to_datetime(lab["Date"])
    lab["stock_id"] = lab["stock_id"].astype(str)

    n_before = len(base)
    merged = base.merge(lab, on=["Date", "stock_id"], how="left",
                        suffixes=("__cached", "__rebuilt"))
    # how="left" 保序，但明講出來並驗證，避免日後 pandas 行為變動時靜默出錯
    assert len(merged) == n_before, f"merge 後列數變了：{n_before} → {len(merged)}"
    assert merged["Date"].equals(base["Date"]) and merged["stock_id"].equals(base["stock_id"]), \
        "merge 後行序與 base 不一致"
    n_miss = int(merged["Alpha_10d"].isna().sum() - base["Alpha_5d"].isna().sum())
    print(f"[label10d] 併回 base：{len(merged):,} 列，行序一致 ✓", flush=True)

    ok = verify(base, merged)
    if check_only:
        print(f"[label10d] --check-only：{'✅ 未偵測到協定窗內的漂移' if ok else '❌ 偵測到窗內漂移'}"
              f"｜未寫入任何檔案", flush=True)
        raise SystemExit(0 if ok else 1)
    if not ok:
        raise SystemExit("❌ 驗證未通過，不產生 Alpha_10d（寧可沒有，也不要一個錯的標籤）")

    # ── 三個 horizon 一律用「重建」這一份，理由見下 ──
    # 快取與重建在協定窗內已證明逐位元相同（上面的閘門），但兩者的**標籤覆蓋率**
    # 在窗尾不同：`prices_raw` 後來多了交易日，於是原本算不出的標籤現在算得出來。
    # 若 5d/20d 取快取、10d 取重建，三個 horizon 就站在兩個不同的資料快照上，
    # 窗尾會有幾千列的覆蓋差異——那是「不會報錯的不公平」。
    # 全部取自同一次重建，三者才真正只差 horizon 一個變因。
    lo, hi = pd.Timestamp(PROTOCOL["TRAIN_START"]), pd.Timestamp(PROTOCOL["TEST_END"])
    inw = (merged["Date"] >= lo) & (merged["Date"] <= hi)
    for h in (5, 20):
        gained = int((merged[f"Alpha_{h}d__cached"].isna()
                      & merged[f"Alpha_{h}d__rebuilt"].notna() & inw).sum())
        print(f"[label10d] 窗內覆蓋差異 Alpha_{h}d：重建比快取多 {gained:,} 列"
              f"（{gained/int(inw.sum()):.4%}）→ 三個 horizon 統一取重建", flush=True)
    merged = merged.rename(columns={"Alpha_5d__rebuilt": "Alpha_5d",
                                    "Alpha_20d__rebuilt": "Alpha_20d"})

    # ⚠️ rank 必須在 **float32** 上算。`build_base_matrix` 的順序是
    # `_downcast_f32(df)` → 排序 → eligible → rank（baseline_common.py 的 label 段），
    # 所以真正的 rank_5d/rank_20d 是在 float32 值上做的。在 float64 上算會讓
    # 平手（tie）的判定不同 → 實測 rank 差到 1.4e-3，而 alpha 本身逐位元相同。
    # 這種差異小到不會被任何斷言擋下，卻會讓三個 horizon 與既有結果不同源。
    for h in (5, 10, 20):
        merged[f"Alpha_{h}d"] = merged[f"Alpha_{h}d"].astype(np.float32)
    for h in (5, 10, 20):
        merged[f"rank_{h}d"] = _rank_transform(merged, f"Alpha_{h}d", f"rank_{h}d")

    # ── rank 的二次驗證：重算的 rank_5d/rank_20d 必須與快取逐位元相同 ──
    # 這比 Alpha 的比對更嚴格——rank 同時取決於值、eligible 遮罩、每日橫斷面
    # 成員、以及 tie 的處理，四者任一不同都會現形。
    # rank 是**當日橫斷面**的量：只要當天有一列多出來，分母就變、整天的 rank 全體微移。
    # 所以閘門只加在「當日覆蓋數相同」的日子；覆蓋數不同的日子據實列出。
    # （實測：20d 有 3 列覆蓋差，落在 3 個日子上，rank 差 5.1e-4——是已知差異的
    #   必然結果，不是 tie／eligible 出問題。）
    rank_ok = True
    for h in (5, 20):
        cov = merged.groupby("Date").agg(
            n_cached=(f"Alpha_{h}d__cached", "count"), n_rebuilt=(f"Alpha_{h}d", "count"))
        same_days = set(cov.index[cov["n_cached"] == cov["n_rebuilt"]])
        on_same = merged["Date"].isin(same_days).to_numpy()
        c = merged[f"rank_{h}d__cached"].to_numpy(np.float32)
        r = merged[f"rank_{h}d"].to_numpy(np.float32)
        fin = np.isfinite(c) & np.isfinite(r)
        sel = fin & inw.to_numpy() & on_same
        mx = float(np.abs(c[sel] - r[sel]).max()) if sel.any() else float("nan")
        nan_mm = int(((np.isfinite(c) != np.isfinite(r)) & inw.to_numpy() & on_same).sum())
        diff_days = sorted(set(merged.loc[inw & ~pd.Series(on_same, index=merged.index),
                                          "Date"].unique()))
        print(f"[驗證] rank_{h}d 窗內／覆蓋數相同的日子：{sel.sum():,} 列｜"
              f"**max|Δ| = {mx:.3e}**｜NaN 型態不同 {nan_mm:,} 列（皆應為 0）  ← 閘門",
              flush=True)
        if diff_days:
            print(f"            覆蓋數不同的日子 {len(diff_days)} 天"
                  f"（{diff_days[0].date()} → {diff_days[-1].date()}）：僅報告，不擋",
                  flush=True)
        if mx != 0.0 or nan_mm:
            rank_ok = False
    if not rank_ok:
        raise SystemExit("❌ rank 重算與快取不一致 → tie 處理／eligible／橫斷面成員有差異，中止")
    print("[驗證] ✅ rank 重算與 base matrix 逐位元相同（覆蓋數相同的日子）", flush=True)

    cols = ["Date", "stock_id"] + [f"Alpha_{h}d" for h in (5, 10, 20)] \
                                + [f"rank_{h}d" for h in (5, 10, 20)]
    out = merged[cols].copy()
    for h in (5, 10, 20):
        out[f"Alpha_{h}d"] = out[f"Alpha_{h}d"].astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False, row_group_size=ROW_GROUP)

    # ── 健檢（規則 7：數值明確輸出）──
    win = merged[merged["eligible"] & inw]
    print(f"[健檢] 輸出 {len(out):,} 列 × {len(cols)} 欄 → {OUT_PATH.name}", flush=True)
    print(f"[健檢] 協定窗內 eligible {len(win):,} 列｜標籤非空："
          + "｜".join(f"{h}d {win[f'Alpha_{h}d'].notna().sum():,}" for h in (5, 10, 20)),
          flush=True)
    for h in (5, 10, 20):
        r = out.loc[out[f"rank_{h}d"].notna(), f"rank_{h}d"]
        a = out.loc[out[f"Alpha_{h}d"].notna(), f"Alpha_{h}d"]
        print(f"[健檢] rank_{h}d：min={r.min():+.4f} mean={r.mean():+.6f} max={r.max():+.4f}"
              f"（應 −0.5 / ~0 / +0.5）｜Alpha_{h}d：median={a.median():+.5f} "
              f"std={a.std():.5f} p1={a.quantile(.01):+.4f} p99={a.quantile(.99):+.4f}",
              flush=True)
    # 單調性交叉檢查：horizon 越長，前瞻報酬的離散度應越大
    s = {h: float(out[f"Alpha_{h}d"].std()) for h in (5, 10, 20)}
    mono = s[5] < s[10] < s[20]
    print(f"[健檢] std 單調性 5d {s[5]:.5f} < 10d {s[10]:.5f} < 20d {s[20]:.5f} → "
          f"{'✓ 合理' if mono else '⚠️ 不單調，需人工確認'}", flush=True)
    # 相鄰 horizon 的相關性應該高但不為 1（否則代表兩欄其實是同一個東西）
    for x, y in ((5, 10), (10, 20)):
        sub = out[[f"Alpha_{x}d", f"Alpha_{y}d"]].dropna()
        print(f"[健檢] corr(Alpha_{x}d, Alpha_{y}d) = "
              f"{sub.iloc[:,0].corr(sub.iloc[:,1]):.4f}（應明顯 <1）", flush=True)
    print(f"[label10d] 完成（{(time.time()-t0)/60:.1f} 分）", flush=True)
    return OUT_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="只檢查 prices_raw 相對 base matrix 有無漂移，不寫檔")
    _a = ap.parse_args()
    build(force=_a.force, check_only=_a.check_only)
