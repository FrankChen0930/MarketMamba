"""
diag_window_panel.py — 尾端窗 base matrix 能不能重現全歷史快取？（診斷，不進 production）
================================================================================
B 類（Ridge / GBDT / GRU）要每日推論，就得每天生出 v2 協定的 base matrix。
但 `build_base_matrix()` 是**全歷史重建**（5 個 chunk、5.8 GB 衍生檔），
不可能每天跑。唯一可行的是「只建最近 N 天的尾端窗」——就像 Mamba 那條線
`run_v62_inference.build_feature_df()` 做的一樣。

**但 Mamba 那條線繞過了一個問題，B 類繞不過**：
`clean_and_scale(macro_norm="ts")` 的 Group D 用 **expanding** 統計量
（至少 252 天暖機）。尾端窗只有 N 天歷史，expanding 的分母就跟全歷史不同
→ **12 個 macro 欄會系統性偏掉**。Mamba 的上線 arm 把 Group D 整個歸零，
所以那條線量不到這件事；Ridge/GBDT/GRU **沒有歸零**，全 66 維都吃。

這支就是去量：尾端窗 vs 快取，**逐欄**差多少、是不是只有 macro 有問題。

⚠️ 2026-08-08 第一版的設計缺陷（保留紀錄，不偷改）
--------------------------------------------------
第一版拿「尾端窗」直接對「快取」比，得到 5 個非 macro 欄偏掉
（`Free_Cash_Flow` ρ=0.356、`Dividend_Yield_Fwd` 0.909、`Securities_Balance` 0.933…）。
**但那不是窗長造成的**——快取建於 2026-07-30，而 FCF 兩層修正與 MOPS 補齊是 08-04/05。
也就是說第一版**同時改了兩個變因**（窗長 + 資料版本），結果無法歸因，
正好違反本專案「一次只改一個變因」的規則。

修正：改成 **兩個不同長度的尾端窗互比**（`--lookback` vs `--lookback2`）。
同一份資料、同一天、只有窗長不同 → 這才是乾淨的配對，量到的就是純窗長效應。

另一個缺陷：macro 欄是**每日橫斷面常數**，單日的 Spearman 無定義（全是 NaN），
第一版因此一欄都沒量到。macro 一律改用 **max|Δ| 與逐欄值**判讀，不用 ρ。

判準（跑之前定死）
------------------
兩個窗長之間，同一天同一批股票：
  · 非 macro 欄  ρ ≥ 0.999 且 max|Δ| ≤ 0.01  → 窗長不影響 → 尾端窗可行
  · macro 欄     報 max|Δ|（12 維、每日常數）；≤0.05 視為可忽略

用法
----
    # 純窗長效應（推薦）
    MM_PROTOCOL=v2 python V6/experimental/diag_window_panel.py --date 2026-03-02 \
        --lookback 1200 --lookback2 2400
    # 對快取比（會混進資料版本差異，只能當參考）
    MM_PROTOCOL=v2 python V6/experimental/diag_window_panel.py --date 2026-03-02 --vs-cache
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experimental import baseline_common as B
from marketmamba.data.feature_engineer import build_features, clean_and_scale


def build_window_base(target_date: str, lookback_days: int) -> pd.DataFrame:
    """只建 `target_date` 往前 `lookback_days` 曆日的 base matrix（不分 chunk，窗小）。

    逐步對齊 `baseline_common.build_base_matrix()`：
      同樣的 universe 過濾、同樣的 `build_features` 參數、
      同樣的 `clean_and_scale(macro_norm='ts', neutralize=...)`、同樣的 eligible 規則。
    唯一的差別就是**餵進去的日期範圍**——這正是要量的變因。
    """
    end = pd.Timestamp(target_date)
    cutoff = end - pd.Timedelta(days=lookback_days)
    t0 = time.time()

    prices = B._load_raw("prices_raw")
    prices = B._filter_universe(prices)
    prices = prices.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    prices = prices[(prices["Date"] >= cutoff) & (prices["Date"] <= end)].copy()
    stocks = sorted(prices["stock_id"].unique())
    print(f"[win] prices {len(prices):,} 列 | {len(stocks)} 支 | "
          f"{prices['Date'].min().date()} → {prices['Date'].max().date()}", flush=True)

    def _trim(d):
        if d is None or "Date" not in getattr(d, "columns", []):
            return d
        d = d.copy()
        d["Date"] = pd.to_datetime(d["Date"])
        return d[(d["Date"] >= cutoff) & (d["Date"] <= end)]

    stock_kw = {k: _trim(B._load_raw(v, stock_ids=stocks)) for k, v in B._STOCK_RAWS.items()}
    # 市場層級 raw **不 trim**：macro 的 expanding 統計量正是本實驗的重點，
    # 要讓 clean_and_scale 看到多少歷史，由 df 的日期範圍決定，不是由這裡。
    market_kw = {k: B._load_raw(v) for k, v in B._MARKET_RAWS.items()}

    df = build_features(
        df_price=prices, **stock_kw, **market_kw,
        fundamentals_v2=B.PROTOCOL.get("FUNDAMENTALS_V2", False),
        availability_flags=B.PROTOCOL.get("AVAILABILITY_FLAGS", False),
    )
    keep = ["Date", "stock_id"] + list(B.FEATURE_COLS) + ["Alpha_5d", "Alpha_20d"]
    df = df[keep]
    del prices, stock_kw, market_kw
    gc.collect()

    _neu = B.PROTOCOL.get("NEUTRALIZE", "none")
    n0 = len(df)
    df = clean_and_scale(df, macro_norm="ts", neutralize=_neu)
    df = df.sort_values(["Date", "stock_id"], kind="mergesort").reset_index(drop=True)
    df["eligible"] = df.groupby("stock_id", sort=False).cumcount() >= (
        B.PROTOCOL["MIN_HISTORY_DAYS"] - 1)
    print(f"[win] clean_and_scale {n0:,} → {len(df):,} 列 | "
          f"eligible {int(df['eligible'].sum()):,} | {(time.time()-t0)/60:.1f} 分", flush=True)
    return df


def _macro_cols() -> set[str]:
    import marketmamba.config as _c
    return set(_c.FEATURE_GROUPS["macro_environment"])


def compare(a_df: pd.DataFrame, b_df: pd.DataFrame, date: str,
            name_a: str, name_b: str) -> int:
    macro = _macro_cols()
    a_df = a_df[a_df["Date"] == pd.Timestamp(date)].copy()
    b_df = b_df[b_df["Date"] == pd.Timestamp(date)].copy()
    a_df["stock_id"] = a_df["stock_id"].astype(str)
    b_df["stock_id"] = b_df["stock_id"].astype(str)
    mg = a_df.merge(b_df, on="stock_id", suffixes=("_a", "_b"))
    print(f"\n[比對] {name_a} {len(a_df)} 支 / {name_b} {len(b_df)} 支 / 交集 {len(mg)} 支")
    for nm, d in ((name_a, a_df), (name_b, b_df)):
        if "eligible" in d:
            print(f"[比對] {nm} eligible = {int(d['eligible'].sum())}")

    rows = []
    for c in B.FEATURE_COLS:
        x, y = mg[f"{c}_a"], mg[f"{c}_b"]
        ok = x.notna() & y.notna()
        if ok.sum() < 30:
            rows.append((c, np.nan, np.nan, c in macro))
            continue
        mad = float((x[ok] - y[ok]).abs().max())
        # macro 是每日橫斷面常數 → Spearman 無定義，只看 max|Δ|
        rho = np.nan if c in macro or x[ok].nunique() < 2 else \
            float(x[ok].corr(y[ok], method="spearman"))
        rows.append((c, rho, mad, c in macro))

    res = pd.DataFrame(rows, columns=["col", "rho", "max_abs_diff", "is_macro"])
    nm_ = res[~res["is_macro"]]
    nm_rho = nm_.dropna(subset=["rho"])
    mac = res[res["is_macro"]]

    print(f"\n{'─'*78}")
    print(f"非 macro 欄（{len(nm_)} 欄，其中 {len(nm_rho)} 欄可算 ρ）：")
    print(f"  ρ min={nm_rho['rho'].min():.6f}｜median={nm_rho['rho'].median():.6f}"
          f"｜<0.999 的有 {int((nm_rho['rho'] < 0.999).sum())} 欄")
    print(f"  max|Δ| 最大={nm_['max_abs_diff'].max():.6f}"
          f"｜>0.01 的有 {int((nm_['max_abs_diff'] > 0.01).sum())} 欄")
    worst = nm_.nlargest(8, "max_abs_diff")
    print(f"\n{'欄位':26s}{'ρ':>12s}{'max|Δ|':>12s}")
    for _, r in worst.iterrows():
        rr = f"{r['rho']:12.6f}" if np.isfinite(r["rho"]) else f"{'—':>12s}"
        print(f"{r['col']:26s}{rr}{r['max_abs_diff']:12.6f}")

    print(f"\nmacro 欄（{len(mac)} 欄，每日常數 → 只看 max|Δ|）：")
    print(f"{'欄位':26s}{'max|Δ|':>12s}")
    for _, r in mac.iterrows():
        print(f"{r['col']:26s}{r['max_abs_diff']:12.6f}")
    mac_worst = float(mac["max_abs_diff"].max())

    ok_nm = bool((nm_rho["rho"] >= 0.999).all()
                 and (nm_["max_abs_diff"].dropna() <= 0.01).all())
    ok_mac = mac_worst <= 0.05
    print(f"\n{'='*78}")
    print(f"非 macro：{'✅ 通過（ρ≥0.999 且 max|Δ|≤0.01）' if ok_nm else '❌ 未過'}")
    print(f"macro   ：{'✅' if ok_mac else '❌'} max|Δ| = {mac_worst:.6f}（判準 ≤0.05）")
    print("=" * 78)
    return 0 if (ok_nm and ok_mac) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-03-02")
    ap.add_argument("--lookback", type=int, default=1200, help="尾端窗 A 曆日數")
    ap.add_argument("--lookback2", type=int, default=2400, help="尾端窗 B 曆日數")
    ap.add_argument("--vs-cache", action="store_true",
                    help="改成對快取比（⚠️ 會混進資料版本差異，見檔頭）")
    a = ap.parse_args()

    if a.vs_cache:
        print("=" * 78)
        print(f"⚠️ 尾端窗 vs 快取｜{a.date}｜lookback {a.lookback}")
        print("⚠️ 快取建於 2026-07-30，FCF/MOPS 修正在 08-04/05 → "
              "偏差含資料版本差異，不可歸因為窗長")
        print("=" * 78)
        ref = pd.read_parquet(B.BASE_PATH, filters=[("Date", "==", pd.Timestamp(a.date))])
        if ref.empty:
            print(f"❌ 快取沒有 {a.date}")
            return 2
        return compare(ref, build_window_base(a.date, a.lookback), a.date, "快取", "尾端窗")

    print("=" * 78)
    print(f"純窗長效應｜{a.date}｜lookback {a.lookback} vs {a.lookback2} 曆日")
    print("（同一份資料、同一天，唯一變因＝窗長）")
    print("=" * 78)
    wa = build_window_base(a.date, a.lookback)
    wb = build_window_base(a.date, a.lookback2)
    return compare(wa, wb, a.date, f"窗{a.lookback}", f"窗{a.lookback2}")


if __name__ == "__main__":
    raise SystemExit(main())
