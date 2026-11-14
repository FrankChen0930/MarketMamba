"""
MarketMamba — 極端報酬逐筆歸因（2026-07-29）
=============================================
台股有 ±10% 單日漲跌幅限制，因此「相鄰兩個交易日之間 |報酬| > 40%」
**在制度上不可能是真實的單日波動**——每一筆都必須有結構性解釋。
本腳本把它們逐筆歸類，剩下無法解釋的才是真正需要人工追查的資料問題。

【為什麼現在才做】此分析必須在還原之後：還原會改變「哪些報酬算極端」。
2026-07-29 完成官方還原後，|報酬|>40% 已從 850 筆降到 471 筆
（2026 年從 93 筆降到 4 筆），而且現在有 **667 筆減資事件清單**可直接對消。

【歸因順序】由最確定到最寬鬆，每筆只計入第一個命中的類別：
  1. 除權息/減資事件      → 因子表有紀錄但價格未被還原（理論上應為 0，是還原的自我檢查）
  2. 交易中斷（gap > 5 天）→ 停牌/處置/長假後復牌，「單日」其實跨了數週
  3. 上市首日             → IPO 首日無漲跌幅限制
  4. 極低價（前收 < 1 元） → 最小跳動單位 0.01 元即為 1% 以上，百分比失真
  5. 無法解釋             → 需人工追查

用法：python V6/scripts/classify_extreme_returns.py [--threshold 0.4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402

P = Path(PROCESSED_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.4)
    a = ap.parse_args()
    TH = a.threshold

    pr = pd.read_parquet(P / "prices_raw.parquet")
    pr["Date"] = pd.to_datetime(pr["Date"])
    pr["stock_id"] = pr["stock_id"].astype(str)
    pr = pr.sort_values(["stock_id", "Date"])
    g = pr.groupby("stock_id")
    pr["prev_close"] = g["Close"].shift(1)
    pr["prev_date"] = g["Date"].shift(1)
    pr["ret"] = pr["Close"] / pr["prev_close"] - 1
    pr["gap_days"] = (pr["Date"] - pr["prev_date"]).dt.days
    first_date = g["Date"].transform("min")
    pr["is_first"] = pr["Date"] == first_date

    ex = pd.read_parquet(P / "ex_rights_raw.parquet")
    ex["Date"] = pd.to_datetime(ex["Date"])
    ex["stock_id"] = ex["stock_id"].astype(str)
    # 事件日 ±3 個交易日內都算命中（官方恢復買賣日與實際首個交易日可能差幾天）
    ev = {(r.stock_id, r.Date) for r in ex.itertuples()}
    ev_ids = set(ex["stock_id"])

    big = pr[pr["ret"].abs() > TH].dropna(subset=["ret"]).copy()
    print("=" * 74)
    print(f"■ |單日報酬| > {TH:.0%} 逐筆歸因")
    print("=" * 74)
    print(f"  全體 {len(pr):,} 列｜命中 {len(big):,} 筆"
          f"（{len(big) / len(pr):.4%}）｜{big['stock_id'].nunique():,} 支")

    def near_event(sid: str, d: pd.Timestamp) -> bool:
        if sid not in ev_ids:
            return False
        return any((sid, d + pd.Timedelta(days=k)) in ev
                   for k in range(-5, 6))

    cats = []
    for r in big.itertuples():
        if near_event(r.stock_id, r.Date):
            cats.append("1_除權息或減資")
        elif pd.notna(r.gap_days) and r.gap_days > 5:
            cats.append("2_交易中斷後復牌")
        elif r.is_first:
            cats.append("3_上市首日")
        elif pd.notna(r.prev_close) and r.prev_close < 1.0:
            cats.append("4_極低價股")
        else:
            cats.append("5_無法解釋")
    big["cat"] = cats

    print()
    print("  歸因結果：")
    vc = big["cat"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"    {k:<18} {v:>5,} 筆 ({v / len(big):6.1%})")

    print()
    print("  依年份（無法解釋者）：")
    un = big[big["cat"] == "5_無法解釋"]
    if len(un):
        yr = un.groupby(un["Date"].dt.year).size()
        print("    " + " ".join(f"{y}={c}" for y, c in yr.items()))
        print()
        print(f"  無法解釋的 {len(un):,} 筆，依 |報酬| 前 12 名：")
        cols = ["Date", "stock_id", "prev_close", "Close", "ret",
                "Volume", "gap_days"]
        show = un.reindex(un["ret"].abs().sort_values(ascending=False).index)
        print(show[cols].head(12).to_string(index=False,
                                            float_format=lambda x: f"{x:,.4f}"))
        print()
        v = pd.to_numeric(un["Volume"], errors="coerce")
        pc = pd.to_numeric(un["prev_close"], errors="coerce")
        print(f"  無法解釋者的特徵：成交量 median {v.median():,.0f}"
              f"｜<10,000 股 {(v < 10000).mean():.1%}"
              f"｜前收 median {pc.median():.2f} 元"
              f"｜前收 <5 元 {(pc < 5).mean():.1%}")
    else:
        print("    無 ✓")

    # 第 1 類理論上應為 0——還原若正確，除權息日的報酬不該超過 40%
    n1 = int((big["cat"] == "1_除權息或減資").sum())
    print()
    print("=" * 74)
    print(f"■ 還原自我檢查：第 1 類（除權息/減資日仍極端）= {n1} 筆 "
          f"{'✓' if n1 == 0 else '⚠️ 表示這些事件的因子未生效，需追查'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
