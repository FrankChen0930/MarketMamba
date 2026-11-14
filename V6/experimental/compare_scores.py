"""
compare_scores.py — 兩份分數檔的逐日比對（B 類重建 vs 既有參考）
================================================================
用途：2026-08-03 那次「重出 p30」沒有留下模型也沒有留下可重現的設定
（`baseline_ridge_lasso.py` 當時沒有 `--purge`、也不寫 parquet）。
2026-08-08 重建之後，必須回答一件事：

    **重建出來的模型，跟八模型定稿表裡的那一格是不是同一個東西？**

判準（跑之前定死，與 `run_v62_inference.verify()` 同一套）
--------------------------------------------------------
  · 逐日 Spearman ρ 的中位數 ≥ 0.95
  · 每日 Top50 重疊 ≥ 40/50
兩項都過 → 當作同一個模型，沿用既有基準；
任一項未過 → **誠實記為新模型**，另立基準，不可混用。

用法
----
    MM_PROTOCOL=v2 python V6/experimental/compare_scores.py \
        --new ridge__p30_rebuild20260808 --ref ridge__lab5d_p30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_DIR = Path(__file__).resolve().parent / "result" / "scores"
MIN_RHO, MIN_OVERLAP = 0.95, 40


def _load(name: str) -> pd.DataFrame:
    p = SCORE_DIR / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(f"❌ 找不到 {p}")
    d = pd.read_parquet(p)
    d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    d["stock_id"] = d["stock_id"].astype(str)
    return d


def compare(new: str, ref: str) -> int:
    a, b = _load(new), _load(ref)
    mg = a.merge(b, on=["Date", "stock_id"], how="inner", suffixes=("_new", "_ref"))
    print("=" * 78)
    print(f"新：{new}  {len(a):,} 列 / {a['Date'].nunique()} 天")
    print(f"參：{ref}  {len(b):,} 列 / {b['Date'].nunique()} 天")
    print(f"交集 {len(mg):,} 列 / {mg['Date'].nunique()} 天"
          f"（只在新 {len(a)-len(mg):,}／只在參 {len(b)-len(mg):,}）")
    print("=" * 78)
    if mg.empty:
        print("❌ 沒有交集")
        return 1

    rho = mg.groupby("Date").apply(
        lambda g: g["score_new"].corr(g["score_ref"], method="spearman"),
        include_groups=False).dropna()

    def _ov(g):
        tn = set(g.nlargest(50, "score_new")["stock_id"])
        tr = set(g.nlargest(50, "score_ref")["stock_id"])
        return len(tn & tr)
    ov = mg.groupby("Date").apply(_ov, include_groups=False)

    r = rho.to_numpy()
    o = ov.to_numpy()
    h1, h2 = r[:len(r) // 2], r[len(r) // 2:]
    print(f"逐日 Spearman ρ：median {np.median(r):.4f}｜mean {r.mean():.4f}"
          f"｜min {r.min():.4f}｜p05 {np.percentile(r, 5):.4f}")
    print(f"                前半 {h1.mean():.4f} / 後半 {h2.mean():.4f}"
          f"（差很多＝影響集中在某一段，不是均勻的模型差異）")
    print(f"Top50 重疊     ：median {np.median(o):.0f}/50｜mean {o.mean():.1f}"
          f"｜min {o.min():.0f}｜<{MIN_OVERLAP} 的天數 {(o < MIN_OVERLAP).sum()}/{len(o)}")

    worst = rho.nsmallest(5)
    print(f"\n最低的 5 天：")
    for d, v in worst.items():
        print(f"  {d}  ρ={v:.4f}  重疊={int(ov[d])}/50")

    ok = bool(np.median(r) >= MIN_RHO and np.median(o) >= MIN_OVERLAP)
    print(f"\n{'='*78}")
    print(f"判準：ρ median ≥{MIN_RHO} 且 Top50 重疊 median ≥{MIN_OVERLAP}")
    print("✅ 通過 → 視為同一個模型，可沿用既有基準" if ok else
          "❌ 未過 → 誠實記為**新模型**，另立基準，不可與八模型表混用")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    raise SystemExit(compare(**vars(ap.parse_args())))
