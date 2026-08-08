"""
backfill_prices.py — 補回 `prices_raw.parquet` 遺失的 (stock_id, Date)
=====================================================================
2026-08-08 查出 production 的 `prices_raw.parquet` 有真實資料遺失：

  · 582 天評估窗內缺 **194,909 列 = 15.15%**，涉及 859 支（489 支在可交易宇宙內）
  · 190 支在窗內幾乎整段消失；缺漏散布在 **98% 的交易日**
  · 用 TWSE 官方資料判定過：4169 於 2026-06-01 確實有交易
    （收盤 158.50 / 量 162,506），與備份吻合、正式檔就是少那一列

**根因**（已於 `fetcher._append_to_parquet` 修掉）：寫入端原本是「整天替換」——
先刪掉該日全部舊列、再寫入這次抓到的。只要某天抓到的股票比已存的少
（端點掛掉、限流、universe 縮水），差額就被靜默刪除。

本檔負責把資料補回來。**來源是備份的原始價，不是重新抓**——
備份 `prices_raw_backup_before_adj_20260729.parquet` 是 B-3 還原之前的快照，
已用交易所資料抽驗過；重抓 572 個交易日 × 兩個交易所要數小時，沒有必要。

價格口徑：**從資料學回來，不假設公式**
--------------------------------------
正式檔是還原價、備份是原始價。第一版假設

    adjusted(t) = raw(t) × Π{ adj_factor(e) : e 為除權息/減資日, e > t }

**實測不成立**：在 791 萬個重疊鍵上只有 **11.4%** 落在 1e-4 內。
例：5364 於 2011-11-18，正式 23.893900、備份 23.90（幾乎相同），
但因子表說 2013 年那筆減資 `adj_factor=40.03` 應該讓兩者差 30 倍。
→ **正式檔的還原口徑無法用 `ex_rights_raw` 重現。**

但正式檔**確實有還原**（除權息當日報酬 median +0.0021、
跌 >3% 的比例 9.5%，而未還原的備份是 13.5%）。所以問題不是「沒還原」，
是「用了我無法反推的口徑」。

→ 改成**經驗比值法**：逐股在重疊區算 `ratio = 正式 / 備份`。
   該比值是**分段常數**（只在除權息事件變動；實測看到的數百個相異值
   其實是備份只有 2 位小數造成的浮點雜訊）。
   補一列時，取它前後最近的重疊日比值，**兩側一致才採用**——
   一致就代表中間沒有發生事件。**兩側不一致或取不到就不補**，並回報跳過幾列。

   這個方法的好處是**完全不需要知道正式檔用什麼口徑**，它只是把
   正式檔自己的轉換照樣套到缺的列上。

★ 驗證（`--check`）：留出法
--------------------------
把重疊列隨機留出 5%，假裝它們是缺的，用上述規則預測其正式價，
再與真值比。這才是對「補值規則」本身的檢定——
拿公式去對重疊列只能檢定公式，檢定不了規則。

用法
----
    python V6/experimental/backfill_prices.py --check      # 只驗證，不寫入
    python V6/experimental/backfill_prices.py --apply      # 寫入（會先備份）
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent.parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

from marketmamba.config import PROCESSED_DIR                       # noqa: E402
# ex_rights_multiplier 不再使用（因子表無法重現正式檔口徑，見檔頭）

D = Path(PROCESSED_DIR)
PROD = D / "prices_raw.parquet"
BACKUP = D / "prices_raw_backup_before_adj_20260729.parquet"
EXR = D / "ex_rights_raw.parquet"
PRICE_COLS = ["Open", "High", "Low", "Close"]

# 判準（跑之前定死）
RATIO_TOL = 1e-3          # 前後兩側比值視為「相同」的容差（分段常數 + 2 位小數雜訊）
CHECK_TOL = 5e-3          # 留出法：預測價與真值的相對誤差容差（0.5%）
CHECK_MIN_PASS = 0.99     # 至少 99% 的留出列要落在容差內
HOLDOUT_FRAC = 0.05
SEED = 20260808


def _load(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"].astype(str))
    df["stock_id"] = df["stock_id"].astype(str)
    return df


def learn_ratio(target: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """逐列學出 `正式/備份` 的比值；學不出來的回 NaN（那些列就不補）。

    `target`：要補的列（Date, stock_id）。`ref`：重疊列的比值表（Date, stock_id, ratio）。

    規則：取同一支股票在該日**前後最近**的重疊日比值，**兩側一致才採用**
    （比值是分段常數，兩側一致 ⇒ 中間沒有除權息事件）。
    只有一側存在時也接受——那代表該列在該股資料的頭或尾。
    """
    # ⚠️ `merge_asof` 要求兩邊都**依 `on` 欄全域排序**（`by=` 只是分組，不代表排序）。
    #    依 ["stock_id","Date"] 排會 ValueError: left keys must be sorted。
    ref = ref.sort_values("Date", kind="mergesort")
    t = target[["Date", "stock_id"]].copy()
    t["_i"] = np.arange(len(t))
    t = t.sort_values("Date", kind="mergesort")

    bwd = pd.merge_asof(t, ref.rename(columns={"ratio": "r_bwd"}),
                        on="Date", by="stock_id", direction="backward")
    fwd = pd.merge_asof(t, ref.rename(columns={"ratio": "r_fwd"}),
                        on="Date", by="stock_id", direction="forward")
    r_b = bwd["r_bwd"].to_numpy(float)
    r_f = fwd["r_fwd"].to_numpy(float)

    out = np.full(len(t), np.nan)
    both = np.isfinite(r_b) & np.isfinite(r_f)
    agree = both & (np.abs(r_b - r_f) <= RATIO_TOL * np.maximum(np.abs(r_b), 1e-12))
    out[agree] = r_b[agree]
    only_b = np.isfinite(r_b) & ~np.isfinite(r_f)
    only_f = np.isfinite(r_f) & ~np.isfinite(r_b)
    out[only_b] = r_b[only_b]
    out[only_f] = r_f[only_f]

    s = pd.Series(out, index=t["_i"].to_numpy())
    return s.sort_index().to_numpy()


def overlap_ratio(prod: pd.DataFrame, bak: pd.DataFrame) -> pd.DataFrame:
    m = prod[["Date", "stock_id", "Close"]].merge(
        bak[["Date", "stock_id", "Close"]], on=["Date", "stock_id"], suffixes=("_p", "_b"))
    a = pd.to_numeric(m["Close_p"], errors="coerce")
    b = pd.to_numeric(m["Close_b"], errors="coerce")
    m["ratio"] = a / b
    m = m[np.isfinite(m["ratio"]) & (m["ratio"] > 0)]
    return m[["Date", "stock_id", "ratio"]]


def run(apply: bool) -> int:
    prod, bak = _load(PROD), _load(BACKUP)
    kp = set(map(tuple, prod[["Date", "stock_id"]].values))
    bak["_key"] = list(map(tuple, bak[["Date", "stock_id"]].values))
    miss = bak[~bak["_key"].isin(kp)].drop(columns="_key").copy()
    over = bak[bak["_key"].isin(kp)].drop(columns="_key").copy()

    print("=" * 84)
    print(f"正式 {len(prod):,} 列｜備份 {len(bak):,} 列｜"
          f"重疊 {len(over):,}｜**只在備份 {len(miss):,}**")
    print("=" * 84)
    if miss.empty:
        print("✅ 沒有缺漏，不需要補")
        return 0

    ratio = overlap_ratio(prod, bak)
    print(f"\n[比值] 重疊區學到 {len(ratio):,} 個比值｜"
          f"涉及 {ratio['stock_id'].nunique()} 支｜"
          f"median {ratio['ratio'].median():.6f}")

    # ── 驗證：留出法（檢定「補值規則」本身，不是檢定某個公式）──
    rng = np.random.default_rng(SEED)
    hold_mask = rng.random(len(ratio)) < HOLDOUT_FRAC
    hold = ratio[hold_mask]
    keep = ratio[~hold_mask]
    print(f"\n[驗證] 留出 {len(hold):,} 列（{HOLDOUT_FRAC:.0%}），"
          f"用剩下 {len(keep):,} 列的比值去預測它們 …", flush=True)
    # 真值與預測必須逐列對齊 → 用同一個排序後的鍵表去取兩邊的 Close
    hk = hold[["Date", "stock_id"]].sort_values(["stock_id", "Date"]).reset_index(drop=True)
    hb = hk.merge(bak[["Date", "stock_id", "Close"]], on=["Date", "stock_id"], how="left")
    hp = hk.merge(prod[["Date", "stock_id", "Close"]], on=["Date", "stock_id"], how="left")
    pr = learn_ratio(hk, keep)
    truth = pd.to_numeric(hp["Close"], errors="coerce").to_numpy()
    predi = pd.to_numeric(hb["Close"], errors="coerce").to_numpy() * pr
    ok = np.isfinite(truth) & np.isfinite(predi) & (truth != 0)
    cover = float(np.isfinite(pr).mean())
    rel = np.abs(predi[ok] - truth[ok]) / np.abs(truth[ok])
    pass_rate = float((rel <= CHECK_TOL).mean())
    print(f"[驗證] 可預測比例（規則給得出比值）= {cover:.2%}")
    print(f"[驗證] 相對誤差：median {np.median(rel):.2e}｜p99 {np.percentile(rel,99):.2e}"
          f"｜max {rel.max():.2e}")
    print(f"[驗證] 落在 {CHECK_TOL:.0e} 內 = **{pass_rate:.4%}**（判準 ≥{CHECK_MIN_PASS:.1%}）")
    good = pass_rate >= CHECK_MIN_PASS
    print(f"[驗證] {'✅ 補值規則可信' if good else '❌ 未過，不可寫入'}")
    if not good:
        return 1

    # ── 轉換缺漏列 ──
    miss = miss.sort_values(["stock_id", "Date"])
    r = learn_ratio(miss, ratio)
    usable = np.isfinite(r)
    print(f"\n[補回] 缺漏 {len(miss):,} 列｜學得到比值 {int(usable.sum()):,} "
          f"({usable.mean():.1%})｜**學不到、不補 {int((~usable).sum()):,} 列**")
    out = miss[usable].copy()
    rr = r[usable]
    for c in PRICE_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").to_numpy() * rr
    print(f"[補回] 涉及 {out['stock_id'].nunique()} 支、{out['Date'].nunique()} 個交易日"
          f"（{out['Date'].min().date()} → {out['Date'].max().date()}）")
    if (~usable).any():
        sk = miss[~usable]
        print(f"[補回] 跳過的集中在 {sk['stock_id'].nunique()} 支"
              f"（那些股票在正式檔完全沒有重疊列，無從學起）")

    if not apply:
        print("\n（--check 模式，未寫入。要寫入請加 --apply）")
        return 0

    # ── 寫入：型別必須與正式檔完全相同（CLAUDE.md「只改值、不改型別」）──
    import pyarrow.parquet as pq
    sch_before = pq.ParquetFile(PROD).schema_arrow
    bkp = PROD.with_name(f"prices_raw_backup_before_backfill_{pd.Timestamp.now():%Y%m%d}.parquet")
    shutil.copy2(PROD, bkp)
    print(f"\n[寫入] 已備份 → {bkp.name}")

    out = out[[c for c in prod.columns if c in out.columns]]
    merged = pd.concat([prod, out], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date", "stock_id"], keep="first")  # 正式檔優先
    merged = merged.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)
    # Date 轉回 large_string（production 的型別）
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    merged.to_parquet(PROD, index=False)

    sch_after = pq.ParquetFile(PROD).schema_arrow
    same_type = {f.name: str(f.type) for f in sch_before if f.name != "__index_level_0__"} == \
                {f.name: str(f.type) for f in sch_after if f.name != "__index_level_0__"}
    after = _load(PROD)
    ka = set(map(tuple, after[["Date", "stock_id"]].values))
    print(f"[寫入] {len(prod):,} → {len(after):,} 列（+{len(after)-len(prod):,}）")
    print(f"[寫入] 舊鍵全部保留：{kp <= ka}｜備份鍵全部涵蓋："
          f"{set(map(tuple, bak[['Date','stock_id']].values)) <= ka}")
    print(f"[寫入] schema 型別不變：{same_type}")
    print("\n⚠️ 下一步：base matrix 與衍生快取都建在舊資料上，**必須重建**"
          "（`MM_PROTOCOL=v2 python V6/experimental/baseline_common.py --build --force`），"
          "否則模型仍吃缺 15% 的舊面板。")
    return 0 if (kp <= ka and same_type) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只驗證轉換口徑，不寫入")
    g.add_argument("--apply", action="store_true", help="驗證通過後寫入（會先備份）")
    a = ap.parse_args()
    raise SystemExit(run(apply=a.apply))
