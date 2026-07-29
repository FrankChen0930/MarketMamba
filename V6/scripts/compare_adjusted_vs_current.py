"""
MarketMamba — 切換前比對：`prices_adj_raw` vs 現行 `prices_raw`（2026-07-28）
==============================================================================
產出「換過去會有什麼改變」的完整數字，供拍板是否切換。
本檔**唯讀**，不改動任何資料。

用法：python V6/scripts/compare_adjusted_vs_current.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402

P = Path(PROCESSED_DIR)
COLS = ["Date", "stock_id", "Close", "Volume"]


def head(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> None:
    new = pd.read_parquet(P / "prices_adj_raw.parquet")
    old = pd.read_parquet(P / "prices_raw.parquet", columns=COLS)
    for d in (new, old):
        d["Date"] = pd.to_datetime(d["Date"])
        d["stock_id"] = d["stock_id"].astype(str)

    head("1  規模與涵蓋")
    print(f"  新 {len(new):>10,} 列｜{new['stock_id'].nunique():,} 支"
          f"｜{new['Date'].min().date()} → {new['Date'].max().date()}"
          f"｜交易日 {new['Date'].nunique():,}")
    print(f"  舊 {len(old):>10,} 列｜{old['stock_id'].nunique():,} 支"
          f"｜{old['Date'].min().date()} → {old['Date'].max().date()}"
          f"｜交易日 {old['Date'].nunique():,}")
    if "src" in new.columns:
        print(f"  新檔來源分布：{dict(new['src'].value_counts())}")

    head("2  最近 10 個交易日的檔數（確認每日推論可用）")
    for tag, d in [("舊", old), ("新", new)]:
        g = d[d["Date"] >= d["Date"].max() - pd.Timedelta(days=20)]
        c = g.groupby(g["Date"].dt.strftime("%Y-%m-%d"))["stock_id"].nunique().tail(10)
        print(f"  {tag}: " + " ".join(f"{k[5:]}={v}" for k, v in c.items()))

    head("3  極端報酬（資料品質的關鍵指標）")
    for tag, d in [("舊", old), ("新", new)]:
        s = d[["Date", "stock_id", "Close"]].sort_values(["stock_id", "Date"])
        s["ret"] = s.groupby("stock_id")["Close"].pct_change()
        r = s["ret"].dropna()
        for th in (0.4, 1.0):
            sub = s[s["ret"].abs() > th]
            n26 = int((sub["Date"].dt.year == 2026).sum())
            print(f"  {tag}｜|單日報酬| > {th:.0%}：{len(sub):>7,} 筆"
                  f"（2026 年 {n26:,} 筆）")
        print(f"  {tag}｜報酬 std {r.std():.4f}｜p0.1 {r.quantile(.001):+.2%}"
              f"｜p99.9 {r.quantile(.999):+.2%}")

    head("4  已知案例：2412 中華電 2026-07-09 除息（官方 adj_factor=0.962724）")
    for tag, d in [("舊", old), ("新", new)]:
        s = d[(d["stock_id"] == "2412")
              & d["Date"].between("2026-07-07", "2026-07-10")][["Date", "Close"]]
        s = s.sort_values("Date")
        if len(s) >= 2:
            v = s["Close"].to_numpy()
            print(f"  {tag}: " + "｜".join(
                f"{r.Date.date()} {r.Close:.2f}" for r in s.itertuples()))
            print(f"      07-08→07-09 報酬 "
                  f"{(v[-1] / v[-2] - 1) if len(v) >= 2 else float('nan'):+.2%}"
                  f"（官方參考價隱含 -3.73%；還原後應接近當日真實漲跌）")

    head("5  共同列的價格差異（新舊都有的 key）")
    m = old.merge(new[["Date", "stock_id", "Close"]], on=["Date", "stock_id"],
                  how="inner", suffixes=("_old", "_new"))
    m["ratio"] = pd.to_numeric(m["Close_new"], errors="coerce") / \
        pd.to_numeric(m["Close_old"], errors="coerce").replace(0, np.nan)
    r = m["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  共同 {len(m):,} 列｜新/舊 價格比 median {r.median():.4f}"
          f"｜p5 {r.quantile(.05):.4f}｜p95 {r.quantile(.95):.4f}")
    print(f"  比值 == 1（完全相同）的比例：{(r.sub(1).abs() < 1e-6).mean():.1%}")
    yr = m.assign(y=m["Date"].dt.year).groupby("y")["ratio"].median()
    print("  逐年 新/舊 中位數比（越接近 1 代表兩者還原基準越一致）：")
    print("    " + " ".join(f"{y}={v:.3f}" for y, v in yr.items()))

    head("6  結論用的檢查清單")
    n_bad = int((pd.to_numeric(new["Close"], errors="coerce") <= 0).sum())
    n_dup = int(new.duplicated(subset=["Date", "stock_id"]).sum())
    n_nan = int(new[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
    print(f"  Close <= 0        : {n_bad:,} 列 {'✓' if n_bad == 0 else '❌'}")
    print(f"  (Date, stock_id) 重複: {n_dup:,} 列 {'✓' if n_dup == 0 else '❌'}")
    print(f"  OHLC 有 NaN       : {n_nan:,} 列 {'✓' if n_nan == 0 else '❌'}")
    last = new["Date"].max()
    n_last = int(new[new["Date"] == last]["stock_id"].nunique())
    print(f"  最新交易日 {last.date()} 檔數：{n_last:,} "
          f"{'✓' if n_last > 1800 else '❌ 偏少'}")


if __name__ == "__main__":
    main()
