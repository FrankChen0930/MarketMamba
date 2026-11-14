"""
MarketMamba — 驗證「用 dividend_raw 自算除權息參考價」的公式可信度（2026-07-28）
================================================================================
【為什麼需要這一步】
  `ex_rights_raw.parquet`（TWSE TWT49U）是官方還原因子，但**只涵蓋上市**：
  現有宇宙 1,942 支中，上櫃 823 支只有 2 支有紀錄。若只用它重建全歷史還原價，
  會造成「上市已還原、上櫃未還原」的市場別系統性偏差——比現況（yfinance 對
  兩邊一致還原）更糟。

  `dividend_raw`（FinMind）涵蓋上櫃 863/1,098 支且欄位齊全，可依交易所公式自算。
  但 FinMind 對上櫃股已有前科（融券賣出/買進標反，2,638,958 列），**不能直接信**。

【驗證設計：純公式比對，不碰價格資料】
  TWT49U 同時給「除權息前收盤價」與「除權息參考價」。因此可以：
      用官方 close_before + dividend_raw 的股利欄位 → 自算 ref_price
      與官方 ref_price 直接比對
  完全不需要 prices_raw，所以**不受「歷史已還原、近期未還原」的混源狀態影響**。
  這是這個驗證能成立的關鍵——若改用 prices_raw 的前收盤價，公式裡的現金股利
  是絕對金額（元）而價格是被縮放過的，兩者尺度不一致，驗證本身就會失真。

【交易所除權息參考價公式】
      參考價 = (前收盤價 − 現金股利 + 現金增資認購價 × 現金增資配股率)
               ÷ (1 + 無償配股率 + 現金增資配股率)
  其中無償配股率 = 股票股利(元/股) ÷ 10（面額 10 元）。

【判定標準】
  自算/官方 的比值 median 應 ≈ 1.0000，且落在 ±0.1% 內的比例要夠高。
  過了才把同一套公式用到上櫃；沒過就代表 dividend_raw 不可信，上櫃維持現況。

用法（repo 根目錄）：
    python V6/scripts/validate_dividend_formula.py
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

EX_PATH = PROCESSED_DIR / "ex_rights_raw.parquet"
DIV_PATH = PROCESSED_DIR / "dividend_raw.parquet"

# dividend_raw 欄位 → 公式成分
CASH_COLS = ["CashEarningsDistribution", "CashStatutorySurplus"]
STOCK_COLS = ["StockEarningsDistribution", "StockStatutorySurplus"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def build_events(div: pd.DataFrame, ci_rate_div: float) -> pd.DataFrame:
    """
    把 dividend_raw 攤成 (stock_id, ex_date) 的事件表。

    一筆配息紀錄可能有兩個除權息日（現金與股票分開），也可能同日（權息）。
    因此拆成兩條候選、再按 (stock_id, ex_date) 加總——同日者自然合併成「權息」。

    ci_rate_div: 現金增資配股率的單位除數。FinMind 未註明單位，
                 由呼叫端掃描 {1, 100, 1000} 找出讓誤差最小者（見 main）。
    """
    div = div.copy()
    cash = sum(_num(div[c]) for c in CASH_COLS if c in div.columns)
    stock = sum(_num(div[c]) for c in STOCK_COLS if c in div.columns)
    ci_rate = _num(div.get("CashIncreaseSubscriptionRate", 0)) / ci_rate_div
    ci_price = _num(div.get("CashIncreaseSubscriptionpRrice", 0))

    parts = []
    for date_col, take_cash, take_stock in [
        ("CashExDividendTradingDate", True, False),
        ("StockExDividendTradingDate", False, True),
    ]:
        if date_col not in div.columns:
            continue
        d = pd.to_datetime(div[date_col], errors="coerce")
        m = d.notna()
        if not m.any():
            continue
        parts.append(pd.DataFrame({
            "stock_id": div.loc[m, "stock_id"].astype(str).values,
            "Date": d[m].values,
            "cash": (cash[m] if take_cash else 0.0 * cash[m]).values,
            "stock_ratio": ((stock[m] / 10.0) if take_stock else 0.0 * stock[m]).values,
            # 現金增資歸在股票除權日（同屬「權」）
            "ci_rate": (ci_rate[m] if take_stock else 0.0 * ci_rate[m]).values,
            "ci_price": (ci_price[m] if take_stock else 0.0 * ci_price[m]).values,
        }))
    if not parts:
        return pd.DataFrame()
    ev = pd.concat(parts, ignore_index=True)
    agg = ev.groupby(["stock_id", "Date"], as_index=False).agg(
        cash=("cash", "sum"), stock_ratio=("stock_ratio", "sum"),
        ci_rate=("ci_rate", "max"), ci_price=("ci_price", "max"))
    return agg


def main() -> None:
    ex = pd.read_parquet(EX_PATH)
    ex["Date"] = pd.to_datetime(ex["Date"])
    ex["stock_id"] = ex["stock_id"].astype(str)
    div = pd.read_parquet(DIV_PATH)

    print("=" * 78)
    print("■ 輸入")
    print("=" * 78)
    print(f"  官方 ex_rights : {len(ex):,} 筆｜{ex['stock_id'].nunique():,} 支"
          f"｜{ex['Date'].min().date()} → {ex['Date'].max().date()}")
    print(f"  dividend_raw   : {len(div):,} 筆｜{div['stock_id'].nunique():,} 支")

    # ── 現金增資配股率的單位未註明，掃描三種可能 ──────────────────
    print()
    print("=" * 78)
    print("■ 現金增資配股率單位掃描（FinMind 未註明，用資料自己決定）")
    print("=" * 78)
    best, best_div = None, None
    for cd in (1.0, 100.0, 1000.0):
        agg = build_events(div, cd)
        if agg.empty:
            continue
        m = ex.merge(agg, on=["stock_id", "Date"], how="inner")
        if m.empty:
            continue
        calc = ((m["close_before"] - m["cash"] + m["ci_price"] * m["ci_rate"])
                / (1.0 + m["stock_ratio"] + m["ci_rate"]))
        r = calc / m["ref_price"]
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        within = float((r.sub(1).abs() <= 0.001).mean())
        print(f"  除以 {cd:>6.0f}: 配對 {len(m):>6,} 筆｜比值 median {r.median():.6f}"
              f"｜±0.1% 內 {within:6.2%}")
        if best is None or within > best:
            best, best_div = within, cd
    print(f"  → 採用除以 {best_div:.0f}")

    # ── 正式驗證 ────────────────────────────────────────────────
    agg = build_events(div, best_div)
    m = ex.merge(agg, on=["stock_id", "Date"], how="inner")
    m["calc_ref"] = ((m["close_before"] - m["cash"] + m["ci_price"] * m["ci_rate"])
                     / (1.0 + m["stock_ratio"] + m["ci_rate"]))
    m["ratio"] = m["calc_ref"] / m["ref_price"]
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])

    print()
    print("=" * 78)
    print("■ 驗證結果：自算參考價 vs 官方參考價")
    print("=" * 78)
    n_ex_matchable = int(ex["Date"].between(
        pd.to_datetime(div[["CashExDividendTradingDate",
                            "StockExDividendTradingDate"]]
                       .apply(pd.to_datetime, errors="coerce").min().min()),
        pd.to_datetime(div[["CashExDividendTradingDate",
                            "StockExDividendTradingDate"]]
                       .apply(pd.to_datetime, errors="coerce").max().max()),
    ).sum())
    print(f"  官方事件 {len(ex):,} 筆｜落在 dividend_raw 時間範圍內 {n_ex_matchable:,} 筆"
          f"｜實際配對上 {len(m):,} 筆（配對率 {len(m)/max(n_ex_matchable,1):.1%}）")
    r = m["ratio"]
    print(f"  比值 median {r.median():.6f}｜mean {r.mean():.6f}｜std {r.std():.6f}")
    for tol in (0.0005, 0.001, 0.005, 0.01):
        print(f"    落在 ±{tol:.2%} 內: {(r.sub(1).abs() <= tol).mean():7.2%}"
              f"  ({int((r.sub(1).abs() <= tol).sum()):,} 筆)")

    # 依事件類型拆解（權/息/權息 的公式成分不同，分開看才知道哪一塊有問題）
    print()
    print("  依官方 kind 拆解：")
    for k, g in m.groupby("kind"):
        gr = g["ratio"]
        print(f"    {str(k):<6} n={len(g):>6,}｜median {gr.median():.6f}"
              f"｜±0.1% 內 {(gr.sub(1).abs() <= 0.001).mean():6.2%}")

    # 誤差最大的樣本——用來判斷離群是「公式錯」還是「個案特殊」
    print()
    print("  誤差最大的 8 筆：")
    worst = m.reindex(r.sub(1).abs().sort_values(ascending=False).index).head(8)
    cols = ["Date", "stock_id", "kind", "close_before", "ref_price", "calc_ref",
            "cash", "stock_ratio", "ci_rate", "ci_price", "ratio"]
    print(worst[cols].to_string(index=False))

    # ── 判定 ────────────────────────────────────────────────────
    ok = (abs(r.median() - 1.0) < 0.0005) and ((r.sub(1).abs() <= 0.001).mean() > 0.90)
    print()
    print("=" * 78)
    print(f"■ 判定：{'✅ 通過' if ok else '❌ 未通過'}"
          f"（門檻：median 偏離 <0.05% 且 ±0.1% 內 >90%）")
    print("=" * 78)
    if ok:
        print("  → dividend_raw 的公式自算可信，可用於上櫃 823 支的還原因子。")
    else:
        print("  → dividend_raw 自算不可信，上櫃維持現況並標記為已知限制。")


if __name__ == "__main__":
    main()
