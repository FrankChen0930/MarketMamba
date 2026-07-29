"""
MarketMamba — 修正 margin_raw 的融券流量欄位互換（2026-07-27）
================================================================
問題：FinMind `TaiwanStockMarginPurchaseShortSale` 對**上櫃股**把
`ShortSaleSell`（券賣）與 `ShortSaleBuy`（券買回補）標反了，上市股正確。
`margin_raw.parquet` 自 2005 年起由 FinMind 建立，因此 `Short_Sale` /
`Short_Cover` 兩欄在上櫃股全歷史都是互換的（`Short_Balance` 餘額正確）。

證據（2026-07-27 交叉驗證 TWSE MI_MARGN / TPEX margin-balance 直連）：
  - 恆等式 `融券今日餘額 = 前日 + 賣出 - 買進 - 現券償還`
    交易所欄位 100.00% 成立、FinMind 欄位僅 88.01%
  - `corr(ΔShort_Balance, Short_Sale - Short_Cover)`（正確應為正）
    TSE  +0.71 ~ +0.95（符號一致 95–97%）
    OTC  −0.85 ~ −0.93（符號一致 **1%**）→ 交換後 +0.927

判定方式（刻意不用市場別分類）：
  現存 TWSE/TPEX 成員名單無法涵蓋已下市股（17% 的列），且有「上櫃轉上市」
  的股票會在歷史中換邊。因此改用**每支股票每年**自己的餘額變動一致性判定：
      net = Short_Sale - Short_Cover
      若 corr(ΔShort_Balance, net) < -CORR_THRESH → 該 (股票, 年) 需交換
  年度粒度可處理轉市場的情況（誤判風險僅限轉換當年）。樣本不足的年度
  依序回退到「該股全歷史判定」→「不動」。

用法（repo 根目錄）：
    python V6/scripts/fix_margin_short_swap.py --dry-run   # 只診斷、不寫檔
    python V6/scripts/fix_margin_short_swap.py --apply     # 備份後實際修正
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402

MARGIN_PATH = PROCESSED_DIR / "margin_raw.parquet"
CORR_THRESH = 0.30      # |corr| 低於此值視為判定不明
MIN_OBS = 20            # 每個 (股票, 年) 至少要這麼多有效觀測才判定


def _diagnose(df: pd.DataFrame) -> pd.DataFrame:
    """回傳每個 (stock_id, year) 的判定：swap / keep / unknown。"""
    df = df.sort_values(["stock_id", "Date"])
    d_bal = df.groupby("stock_id", sort=False)["Short_Balance"].diff()
    net = df["Short_Sale"] - df["Short_Cover"]
    work = pd.DataFrame({
        "stock_id": df["stock_id"].to_numpy(),
        "year": df["Date"].dt.year.to_numpy(),
        "d_bal": d_bal.to_numpy(),
        "net": net.to_numpy(),
    }).dropna(subset=["d_bal"])
    # 只留「有動作」的列，否則全 0 列會把相關稀釋掉
    work = work[(work["d_bal"] != 0) | (work["net"] != 0)]

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        c = g["d_bal"].corr(g["net"]) if n >= MIN_OBS else np.nan
        return pd.Series({"n": n, "corr": c})

    by_year = work.groupby(["stock_id", "year"]).apply(_agg, include_groups=False).reset_index()
    by_stock = work.groupby("stock_id").apply(_agg, include_groups=False).reset_index()
    by_stock = by_stock.rename(columns={"corr": "corr_stock", "n": "n_stock"})

    m = by_year.merge(by_stock[["stock_id", "corr_stock", "n_stock"]], on="stock_id", how="left")

    def _verdict(row) -> str:
        for c in (row["corr"], row["corr_stock"]):      # 年度優先，退回全歷史
            if pd.notna(c):
                if c < -CORR_THRESH:
                    return "swap"
                if c > CORR_THRESH:
                    return "keep"
        return "unknown"

    m["verdict"] = m.apply(_verdict, axis=1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="實際寫回 margin_raw.parquet（會先備份）")
    ap.add_argument("--dry-run", action="store_true", help="只診斷不寫檔")
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.print_help()
        return

    df = pd.read_parquet(MARGIN_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"[讀入] {MARGIN_PATH.name}：{len(df):,} 列 | "
          f"{df['stock_id'].nunique()} 支 | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}", flush=True)

    verdict = _diagnose(df)
    vc = verdict["verdict"].value_counts()
    print(f"\n[診斷] (股票, 年) 組合共 {len(verdict):,}")
    for k in ("swap", "keep", "unknown"):
        n = int(vc.get(k, 0))
        print(f"    {k:<8} {n:>7,} 組（{n/max(len(verdict),1):.1%}）")

    # 把判定貼回逐列
    df["year"] = df["Date"].dt.year
    df = df.merge(verdict[["stock_id", "year", "verdict"]], on=["stock_id", "year"], how="left")
    df["verdict"] = df["verdict"].fillna("unknown")
    n_swap_rows = int((df["verdict"] == "swap").sum())
    print(f"\n[診斷] 逐列：需交換 {n_swap_rows:,} 列（{n_swap_rows/len(df):.1%}）| "
          f"保持原樣 {int((df['verdict']=='keep').sum()):,} 列 | "
          f"判定不明 {int((df['verdict']=='unknown').sum()):,} 列（不動）")

    # 修正前後的整體一致性（規則 7：數值明確輸出）
    def _corr_overall(frame: pd.DataFrame, sale: str, cover: str) -> tuple[float, float, int]:
        f = frame.sort_values(["stock_id", "Date"])
        d = f.groupby("stock_id", sort=False)["Short_Balance"].diff()
        n = f[sale] - f[cover]
        ok = d.notna() & ((d != 0) | (n != 0))
        return (float(d[ok].corr(n[ok])),
                float((np.sign(d[ok]) == np.sign(n[ok])).mean()),
                int(ok.sum()))

    mask = df["verdict"] == "swap"
    before = _corr_overall(df, "Short_Sale", "Short_Cover")
    print(f"\n[修正前] 全體 corr(ΔShort_Balance, Sale-Cover) = {before[0]:+.4f} | "
          f"符號一致率 {before[1]:.2%} | 樣本 {before[2]:,}")

    if n_swap_rows:
        s = df.loc[mask, "Short_Sale"].to_numpy().copy()
        c = df.loc[mask, "Short_Cover"].to_numpy().copy()
        df.loc[mask, "Short_Sale"] = c
        df.loc[mask, "Short_Cover"] = s

    after = _corr_overall(df, "Short_Sale", "Short_Cover")
    print(f"[修正後] 全體 corr(ΔShort_Balance, Sale-Cover) = {after[0]:+.4f} | "
          f"符號一致率 {after[1]:.2%} | 樣本 {after[2]:,}")

    # 分期間複驗
    print(f"\n[分期複驗] {'期間':<14}{'corr':>9}{'符號一致':>10}{'樣本':>12}")
    for lo, hi, lbl in [("2010-01-01", "2012-12-31", "2010–2012"),
                        ("2016-01-01", "2018-12-31", "2016–2018"),
                        ("2022-01-01", "2024-12-31", "2022–2024"),
                        ("2026-01-01", "2026-12-31", "2026")]:
        seg = df[(df["Date"] >= lo) & (df["Date"] <= hi)]
        if len(seg) < 1000:
            continue
        c, a, n = _corr_overall(seg, "Short_Sale", "Short_Cover")
        flag = "✅" if c > 0.5 else "⚠️"
        print(f"           {lbl:<14}{c:>+9.3f}{a:>9.1%}{n:>12,}  {flag}")

    # 餘額欄未被動到（保險）
    print(f"\n[保險] Short_Balance / Margin_* 四欄未被修改：僅交換 Short_Sale ↔ Short_Cover")

    if args.dry_run:
        print("\n--dry-run：未寫檔。確認上方數字後改用 --apply 實際修正。", flush=True)
        return

    out_cols = ["Date", "stock_id", "Margin_Balance", "Short_Balance",
                "Margin_Purchase", "Margin_Repay", "Short_Cover", "Short_Sale"]
    backup = PROCESSED_DIR / f"margin_raw_backup_{date.today().strftime('%Y%m%d')}.parquet"
    if not backup.exists():
        pd.read_parquet(MARGIN_PATH).to_parquet(backup, index=False)
        print(f"\n[備份] → {backup.name}", flush=True)
    df[out_cols].to_parquet(MARGIN_PATH, index=False)
    print(f"[寫回] {MARGIN_PATH.name}：{len(df):,} 列，欄位順序與原檔一致", flush=True)
    print("✅ 完成", flush=True)


if __name__ == "__main__":
    main()
