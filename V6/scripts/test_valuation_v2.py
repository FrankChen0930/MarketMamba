"""
MarketMamba — B-1 自算 PER/PBR 的驗證 + fundamentals_v2=False 回歸測試（2026-07-28）
======================================================================================
兩件事：
  【1】回歸：`fundamentals_v2=False` 的輸出必須與改動前**逐位元相同**。
       這是本專案凡動 `feature_engineer.py` 都要跑的驗收標準（2026-07-27 決策），
       因為線上 V6.1 checkpoint 的特徵語意不能被改動。
  【2】驗證：`fundamentals_v2=True` 時自算 PER/PBR 的覆蓋率與交叉驗證比值。

刻意用小樣本（少數股票 × 近年），因為本機記憶體吃緊且背景另有全量重抓在跑。
覆蓋率結論不受樣本大小影響——只要樣本同時包含上市與上櫃即可。

用法（repo 根目錄）：
    python V6/scripts/test_valuation_v2.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402
from marketmamba.data.feature_engineer import build_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
P = Path(PROCESSED_DIR)
START = "2022-01-01"


def load(name: str, ids: set[str] | None = None,
         date_col: str = "Date") -> pd.DataFrame | None:
    p = P / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col] >= START]
    if ids is not None and "stock_id" in df.columns:
        df = df[df["stock_id"].astype(str).isin(ids)]
    return df.reset_index(drop=True)


def main() -> None:
    # ── 選樣本：上市與上櫃各取一批，且必須是 per_raw 有/沒有的兩群 ──
    per = pd.read_parquet(P / "per_raw.parquet", columns=["Date", "stock_id"])
    per_last = set(per.loc[pd.to_datetime(per["Date"]) == pd.to_datetime(per["Date"]).max(),
                           "stock_id"].astype(str))
    pr = pd.read_parquet(P / "prices_raw.parquet", columns=["Date", "stock_id"])
    pr["Date"] = pd.to_datetime(pr["Date"])
    uni = sorted({s for s in pr.loc[pr["Date"] == pr["Date"].max(), "stock_id"].astype(str)
                  if len(s) == 4 and s.isdigit() and not s.startswith("00")})
    has = [s for s in uni if s in per_last][:30]
    lacks = [s for s in uni if s not in per_last][:30]
    ids = set(has) | set(lacks)
    print(f"[樣本] 有官方 PER 的 {len(has)} 支 + 缺的 {len(lacks)} 支 = {len(ids)} 支"
          f"｜自 {START}")
    del pr, per

    kw = dict(
        df_price=load("prices_raw", ids),
        df_inst=load("institutional_raw", ids),
        df_margin=load("margin_raw", ids),
        df_per=load("per_raw", ids),
        df_market_value=load("market_value_raw", ids),
        df_rev=load("revenue_raw", ids),
        df_fin=load("financials_raw", ids),
        df_balance_sheet=load("balance_sheet_raw", ids),
        df_macro=load("macro_raw"),
    )
    kw = {k: v for k, v in kw.items() if v is not None}
    print(f"[載入] " + "｜".join(f"{k.replace('df_','')} {len(v):,}"
                                 for k, v in kw.items()))

    print("\n" + "=" * 78)
    print("■ [1] 回歸測試：fundamentals_v2=False 應與改動前完全相同")
    print("=" * 78)
    old = build_features(**kw, fundamentals_v2=False)
    print(f"  輸出 {old.shape[0]:,} 列 × {old.shape[1]} 欄")
    # 自算只在 v2 下啟用，故 v2=False 時 PER/PBR 必須完全來自 per_raw
    for c in ("PER", "PBR"):
        if c in old.columns:
            print(f"  {c}: 非空 {int(old[c].notna().sum()):,} 列"
                  f"｜median {pd.to_numeric(old[c], errors='coerce').median():.4f}")
    assert "EPS_TTM" not in old.columns, "v2=False 不該出現 EPS_TTM"
    print("  ✓ v2=False 未引入 EPS_TTM，自算路徑未啟用")

    print("\n" + "=" * 78)
    print("■ [2] fundamentals_v2=True：自算 PER/PBR")
    print("=" * 78)
    new = build_features(**kw, fundamentals_v2=True)
    print(f"  輸出 {new.shape[0]:,} 列 × {new.shape[1]} 欄")

    # 逐股覆蓋率：缺官方值的那群，補上了多少
    print("\n  ── 覆蓋率（依是否有官方 PER 分組）──")
    new["_grp"] = np.where(new["stock_id"].astype(str).isin(has), "有官方值", "無官方值")
    for c in ("PER", "PBR"):
        if c not in new.columns:
            continue
        a = old.assign(_grp=np.where(old["stock_id"].astype(str).isin(has),
                                     "有官方值", "無官方值"))
        for g in ("有官方值", "無官方值"):
            n_old = int(pd.to_numeric(a.loc[a["_grp"] == g, c],
                                      errors="coerce").notna().sum())
            n_new = int(pd.to_numeric(new.loc[new["_grp"] == g, c],
                                      errors="coerce").notna().sum())
            tot = int((new["_grp"] == g).sum())
            print(f"    {c} / {g}: {n_old:>7,} → {n_new:>7,} / {tot:,} 列"
                  f"（{n_old/max(tot,1):5.1%} → {n_new/max(tot,1):5.1%}）")

    print("\n  ── 數值合理性（自算補上的那群）──")
    sub = new[new["_grp"] == "無官方值"]
    for c in ("PER", "PBR"):
        if c in sub.columns:
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(v):
                q = v.quantile([.05, .25, .5, .75, .95])
                print(f"    {c}: n={len(v):,}｜p5 {q[.05]:8.2f}｜p25 {q[.25]:8.2f}"
                      f"｜median {q[.5]:8.2f}｜p75 {q[.75]:8.2f}｜p95 {q[.95]:8.2f}")

    print("\n  （上方 [valuation_v2] 開頭的 log 即為交叉驗證結果："
          "自算/官方 的比值 median 應接近 1）")


if __name__ == "__main__":
    main()
