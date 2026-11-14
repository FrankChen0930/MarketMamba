"""
MarketMamba — 停更資料回補健檢（2026-07-25）
================================================
在把 _staging_202607_backfill/ 五個檔案合併回 production 之前，逐項檢查：
  A. 結構完整性：重複列、缺值比例
  B. 數值合理性：範圍是否符合常識
  C. 交易日曆完整性：跟 prices_raw 真實交易日比對，找出缺漏的交易日
  D. 股票覆蓋率：涵蓋幾成的當日有效股票
  E. 與 production 舊資料的銜接：日期是否重疊/有斷層

用法：python V6/scripts/backfill_stale_202607.py 執行完後
      python V6/scripts/healthcheck_backfill_202607.py
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

STAGING_DIR = PROCESSED_DIR / "_staging_202607_backfill"

FILES = {
    "per_raw":          ("per_raw_backfill.parquet",          ["PER", "PBR", "dividend_yield"]),
    "securities_raw":   ("securities_raw_backfill.parquet",   ["Securities_Balance"]),
    "market_value_raw": ("market_value_raw_backfill.parquet", ["market_value"]),
    "margin_raw":       ("margin_raw_backfill.parquet",       ["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]),
    "daytrade_raw":     ("daytrade_raw_backfill.parquet",     None),  # 欄位待印出後決定
}

VALUE_RULES = {
    # (欄位, 允許最小值, 允許最大值, 說明)
    "PER":                       (0, 3000, "本益比：負值代表虧損公司（合理），但不該出現超過三位數的離譜值"),
    "PBR":                       (0, 200, "股價淨值比：應為正值"),
    "dividend_yield":            (0, 50, "殖利率(%)：超過 50% 幾乎必為資料錯誤"),
    "Securities_Balance":        (0, None, "可借券賣出股數：不應為負"),
    "market_value":              (0, None, "市值：不應為負或零"),
    "MarginPurchaseTodayBalance": (0, None, "融資今日餘額：不應為負"),
    "ShortSaleTodayBalance":     (0, None, "融券今日餘額：不應為負"),
}


def _load(name: str) -> pd.DataFrame:
    fname, _ = FILES[name]
    df = pd.read_parquet(STAGING_DIR / fname)
    dcol = "Date" if "Date" in df.columns else "date"
    df[dcol] = pd.to_datetime(df[dcol])
    if dcol != "Date":
        df = df.rename(columns={dcol: "Date"})
    return df


def check_structure(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name}：結構完整性 ---")
    n = len(df)
    dup = df.duplicated(subset=["Date", "stock_id"]).sum()
    print(f"  總列數 {n:,} | 重複 (Date, stock_id) 列數 {dup:,}"
          f"（{'✅ 無重複' if dup == 0 else '⚠️ 有重複，需要 drop_duplicates'}）")

    _, cols = FILES[name]
    if cols is None:
        cols = [c for c in df.columns if c not in ("Date", "stock_id")]
    for c in cols:
        if c not in df.columns:
            print(f"  欄位 {c}：⚠️ 不存在於此檔案")
            continue
        n_na = df[c].isna().sum()
        print(f"  欄位 {c}：缺值 {n_na:,} / {n:,}（{n_na/n:.2%}）")


def check_values(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name}：數值合理性 ---")
    _, cols = FILES[name]
    if cols is None:
        return
    for c in cols:
        if c not in df.columns or c not in VALUE_RULES:
            continue
        lo, hi, desc = VALUE_RULES[c]
        s = df[c].dropna()
        n_bad_lo = int((s < lo).sum()) if lo is not None else 0
        n_bad_hi = int((s > hi).sum()) if hi is not None else 0
        print(f"  {c}（{desc}）：min={s.min():.2f} max={s.max():.2f} "
              f"median={s.median():.2f} | 低於下限 {n_bad_lo:,} 筆、超過上限 {n_bad_hi:,} 筆"
              f"（合計異常 {n_bad_lo+n_bad_hi:,} / {len(s):,} = {(n_bad_lo+n_bad_hi)/len(s):.3%}）")


def check_calendar(name: str, df: pd.DataFrame, real_trading_days: set) -> None:
    print(f"\n--- {name}：交易日曆完整性 ---")
    got_days = set(df["Date"].dt.strftime("%Y-%m-%d").unique())
    missing = sorted(real_trading_days - got_days)
    extra = sorted(got_days - real_trading_days)
    print(f"  應涵蓋交易日 {len(real_trading_days)} 天 | 實際涵蓋 {len(got_days)} 天")
    if missing:
        print(f"  ⚠️ 缺少 {len(missing)} 個交易日：{missing}")
    else:
        print(f"  ✅ 無缺漏交易日")
    if extra:
        print(f"  ⚠️ 出現 {len(extra)} 個不在 prices_raw 交易日曆裡的日期：{extra[:10]}"
              f"{' ...' if len(extra) > 10 else ''}")


def check_coverage(name: str, df: pd.DataFrame, prices: pd.DataFrame) -> None:
    print(f"\n--- {name}：股票覆蓋率（抽樣最新一天） ---")
    latest = df["Date"].max()
    today_ids = set(df.loc[df["Date"] == latest, "stock_id"])
    universe_today = set(prices.loc[prices["Date"] == latest, "stock_id"])
    if not universe_today:
        # 找最近的 prices_raw 交易日
        near = prices.loc[prices["Date"] <= latest, "Date"].max()
        universe_today = set(prices.loc[prices["Date"] == near, "stock_id"])
    cov = len(today_ids & universe_today) / max(len(universe_today), 1)
    print(f"  {latest.date()}：{len(today_ids)} 檔有資料，當日 prices_raw 股票池 {len(universe_today)} 檔，"
          f"覆蓋率 {cov:.1%}")


def check_boundary(name: str) -> None:
    print(f"\n--- {name}：與 production 舊資料的銜接 ---")
    fname, _ = FILES[name]
    prod_path = PROCESSED_DIR / f"{name}.parquet"
    if not prod_path.exists():
        print("  ⚠️ production 檔案不存在，無法比對")
        return
    prod = pd.read_parquet(prod_path)
    dcol = "Date" if "Date" in prod.columns else "date"
    prod[dcol] = pd.to_datetime(prod[dcol])
    prod_last = prod[dcol].max()

    staged = pd.read_parquet(STAGING_DIR / fname)
    scol = "Date" if "Date" in staged.columns else "date"
    staged[scol] = pd.to_datetime(staged[scol])
    staged_first = staged[scol].min()
    staged_last = staged[scol].max()

    overlap = staged[(staged[scol] <= prod_last)]
    print(f"  production 最後日期：{prod_last.date()} | 暫存檔範圍：{staged_first.date()} ~ {staged_last.date()}")
    if len(overlap) > 0:
        print(f"  ⚠️ 有 {len(overlap):,} 列跟 production 既有資料日期重疊（合併時需要用暫存檔覆蓋，不是直接 concat）")
    else:
        gap_days = (staged_first - prod_last).days
        print(f"  {'✅ 緊接無重疊' if gap_days <= 3 else f'⚠️ 中間有 {gap_days} 天間隔（需對照是否為真實缺口）'}")


def main() -> None:
    prices = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet", columns=["Date", "stock_id"])
    prices["Date"] = pd.to_datetime(prices["Date"])

    real_trading_days = set(
        prices.loc[
            (prices["Date"] >= "2026-04-25") & (prices["Date"] <= prices["Date"].max()),
            "Date",
        ].dt.strftime("%Y-%m-%d").unique()
    )

    for name in FILES:
        print("=" * 78)
        print(f"[{name}]")
        df = _load(name)
        check_structure(name, df)
        check_values(name, df)
        check_calendar(name, df, real_trading_days)
        check_coverage(name, df, prices)
        check_boundary(name)

    print("\n" + "=" * 78)
    print("健檢完成。")


if __name__ == "__main__":
    main()
