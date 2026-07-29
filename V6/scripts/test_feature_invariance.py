"""
test_feature_invariance.py — 子集不變性測試（檢查表 A 的例行化）
==================================================================
凡新增／修改特徵都要跑。與 `test_availability_flags.py` 驗的**不是同一件事**：

  test_availability_flags [1]  「與 git HEAD 逐位元比對」
                                → 驗「這次改動有沒有影響舊行為」
  本檔                          「宇宙縮小後個股特徵值有沒有變」
                                → 驗「時序特徵有沒有偷吃到橫斷面資訊」

兩者不能互相取代。第二種抓的是這類錯誤：某個特徵表面上是逐股算的，
實際上卻用到了「當天全市場」的資訊（例如在 z-score 之後才做 rolling），
於是換一批股票進來、同一支股票的歷史特徵就變了。
這種錯誤**不會報錯**，只會讓 IC 莫名偏低——`baseline_common` 的 rolling 特徵
（2026-07-27 修正的 G4）就是踩了這個。

---------------------------------------------------------------------------
三種欄位、三種預期，測試必須分開處理（這是本檔設計的重點）

  ① 應該完全不變：純逐股時序 + as-of join 的欄位
     OHLCV、Return_*、MA_*、RSI、ATR、KD、OBV、法人、融資融券、營收、財報…
     判準：**逐位元相同**。任何差異都是 bug。

  ② 設計上就會變：橫斷面操作
     - `_derive_valuation_fallback`（fundamentals_v2=True）的**當日橫斷面校準**
       用當天同時有官方值與自算值的股票算 median 係數 → 換宇宙必然改變
     - `clean_and_scale` 的 winsorize / z-score / 中性化
     判準：值可以變，但**同一天、同一批股票之間的排序必須完全一致**
     （z-score 是單調轉換，Spearman 必須 = 1.0）。
     排序若也變了，代表橫斷面操作本身有 bug。

  ③ 市場層級常數：Group D macro
     同日全市場同值，與宇宙無關 → 逐位元相同。

用法（repo 根目錄）：
    python -u -W ignore V6/scripts/test_feature_invariance.py
    python -u -W ignore V6/scripts/test_feature_invariance.py --full 40 --sub 6
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from marketmamba.config import PROCESSED_DIR                      # noqa: E402

P = Path(PROCESSED_DIR)
START = "2018-01-01"      # 涵蓋所有資料源都有資料的期間，讓比對有內容

# 設計上允許因宇宙改變而改值的欄位（見上方 ②）
_CROSS_SECTIONAL_COLS = {"PER", "PBR"}      # fundamentals_v2 的當日橫斷面校準


def load(name: str, ids: set[str] | None = None) -> pd.DataFrame | None:
    p = P / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    for dc in ("Date", "date", "Week"):
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            df = df[df[dc] >= START]
            break
    if ids is not None and "stock_id" in df.columns:
        df = df[df["stock_id"].astype(str).isin(ids)]
    return df.reset_index(drop=True)


def make_kwargs(ids: set[str]) -> dict:
    kw = dict(
        df_price=load("prices_raw", ids),
        df_inst=load("institutional_raw", ids),
        df_margin=load("margin_raw", ids),
        df_daytrade=load("daytrade_raw", ids),
        df_holdings=load("holdings_raw", ids),
        df_securities=load("securities_raw", ids),
        df_foreign_shareholding=load("foreign_shareholding_raw", ids),
        df_per=load("per_raw", ids),
        df_market_value=load("market_value_raw", ids),
        df_rev=load("revenue_raw", ids),
        df_fin=load("financials_raw", ids),
        df_balance_sheet=load("balance_sheet_raw", ids),
        df_cashflow=load("cashflow_raw", ids),
        df_dividend=load("dividend_raw", ids),
        df_macro=load("macro_raw"),
    )
    return {k: v for k, v in kw.items() if v is not None and not v.empty}


def compare_bitwise(a: pd.DataFrame, b: pd.DataFrame, cols: list[str],
                    label: str) -> tuple[list[str], float]:
    """回傳 (有差異的欄位, 最大絕對差)。兩個 df 需已對齊到相同的 (Date, stock_id)。"""
    bad, worst = [], 0.0
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        x = a[c].to_numpy(dtype="float64")
        y = b[c].to_numpy(dtype="float64")
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            bad.append(c)
            continue
        m = ~np.isnan(x)
        d = float(np.max(np.abs(x[m] - y[m]))) if m.any() else 0.0
        if d > 0:
            bad.append(c)
        worst = max(worst, d)
    return bad, worst


def pick_universe(n_full: int, n_sub: int) -> tuple[set[str], set[str], int]:
    """
    挑樣本時**必須刻意納入「缺官方 PER」的股票**（沿用 test_valuation_v2.py 的作法）。

    原因：`_derive_valuation_fallback` 的最後一行是
        df[name] = official.where(obs).combine_first(calc).combine_first(official)
    只要那一列有官方觀測，官方值就直接勝出、自算與當日橫斷面校準完全不參與。
    若樣本全是老牌大型股（官方 PER 覆蓋率極高），這條路徑從頭到尾不會被踩到，
    測試會綠燈但什麼都沒驗到——這與樣本大小無關，是**選樣**的問題。

    回傳 (完整宇宙, 子集, 完整宇宙中缺官方 PER 的支數)。
    """
    pr = pd.read_parquet(P / "prices_raw.parquet", columns=["Date", "stock_id"])
    pr["Date"] = pd.to_datetime(pr["Date"])
    last = pr["Date"].max()
    uni = sorted({s for s in pr.loc[pr["Date"] == last, "stock_id"].astype(str)
                  if len(s) == 4 and s.isdigit() and not s.startswith("00")})
    del pr

    per = pd.read_parquet(P / "per_raw.parquet", columns=["Date", "stock_id"])
    per["Date"] = pd.to_datetime(per["Date"])
    has_per = set(per.loc[per["Date"] == per["Date"].max(), "stock_id"].astype(str))
    del per

    has = [s for s in uni if s in has_per]
    lacks = [s for s in uni if s not in has_per]
    half = n_full // 2
    full_ids = set(has[:half] + lacks[:n_full - half])

    # 子集也要**兩種都有**，否則子集這一側同樣踩不到自算路徑
    sub_has = [s for s in sorted(full_ids) if s in has_per][: max(n_sub // 2, 1)]
    sub_lack = [s for s in sorted(full_ids) if s not in has_per][: n_sub - len(sub_has)]
    return full_ids, set(sub_has + sub_lack), len([s for s in full_ids if s not in has_per])


def align(full: pd.DataFrame, sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把 full 裁到 sub 的 (Date, stock_id)，兩邊排序一致後回傳。"""
    key = ["Date", "stock_id"]
    f = full.merge(sub[key], on=key, how="inner")
    f = f.sort_values(key, kind="mergesort").reset_index(drop=True)
    s = sub.sort_values(key, kind="mergesort").reset_index(drop=True)
    assert f[key].equals(s[key]), "對齊失敗：(Date, stock_id) 不一致"
    return f, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=int, default=40, help="完整宇宙的股票數")
    ap.add_argument("--sub", type=int, default=6, help="子集的股票數")
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v1",
                    help="v2：patch 成 67 維，把 8 個 Avail_* 旗標與中性化納入比對")
    args = ap.parse_args()

    # config patch 必須早於 `feature_engineer` 的 import——該模組在 import 時就
    # 綁定 FEATURE_COLS（與 architecture.py 綁 GROUP_DIMS 是同一個陷阱）。
    v2 = args.protocol == "v2"
    if v2:
        from marketmamba.data.feature_spec import patch_config_67d
        print(f"[protocol v2] config patched → INPUT_DIM={patch_config_67d()}")

    from marketmamba.data.feature_engineer import build_features, clean_and_scale
    from marketmamba.config import FEATURE_COLS

    # v2 下旗標與 fundamentals_v2 是綁在一起的（協定 v2.0 的 PROTOCOL 設定），
    # 所以 [1] 的「應完全不變」基準也要帶旗標，否則測到的還是 59 維那條路徑。
    BF = dict(availability_flags=True) if v2 else {}

    full_ids, sub_ids, n_lack = pick_universe(args.full, args.sub)

    print("=" * 76)
    print(f"子集不變性測試｜完整宇宙 {len(full_ids)} 支 → 子集 {len(sub_ids)} 支｜自 {START}")
    print(f"  其中缺官方 PER 的 {n_lack} 支（用來踩到 _derive_valuation_fallback 的自算路徑）")
    print(f"  子集：{sorted(sub_ids)}")
    print("=" * 76)

    # ================================================================
    print("\n■ [1] build_features(fundamentals_v2=False)：應完全不變")
    print("-" * 76)
    f0 = build_features(**make_kwargs(full_ids), fundamentals_v2=False, **BF)
    s0 = build_features(**make_kwargs(sub_ids), fundamentals_v2=False, **BF)
    fa, sa = align(f0, s0)
    cols = [c for c in FEATURE_COLS if c in fa.columns] + \
           [c for c in ("Alpha_5d", "Alpha_20d", "Alpha_60d") if c in fa.columns]
    bad, worst = compare_bitwise(fa, sa, cols, "v2=False")
    print(f"  比對 {len(cols)} 欄 × {len(fa):,} 列｜最大絕對差 {worst:.3e}｜有差異的欄位 {len(bad)}")
    if bad:
        print(f"  ✗ {bad}")
    assert not bad, (
        "有欄位違反子集不變性——代表某個「逐股時序」特徵實際上吃到了橫斷面資訊。"
        "這種錯誤不會報錯，只會讓 IC 莫名偏低。")
    print("  ✓ 全部逐位元相同")

    # ================================================================
    print("\n■ [2] build_features(fundamentals_v2=True)：只有橫斷面校準的欄位可以變")
    print("-" * 76)
    f1 = build_features(**make_kwargs(full_ids), fundamentals_v2=True, **BF)
    s1 = build_features(**make_kwargs(sub_ids), fundamentals_v2=True, **BF)
    fb, sb = align(f1, s1)
    bad1, worst1 = compare_bitwise(fb, sb, cols, "v2=True")
    unexpected = [c for c in bad1 if c not in _CROSS_SECTIONAL_COLS]
    print(f"  有差異的欄位 {len(bad1)}：{bad1}")
    print(f"  預期會變的（當日橫斷面校準）：{sorted(_CROSS_SECTIONAL_COLS)}")
    if unexpected:
        print(f"  ✗ 非預期的差異：{unexpected}")
    assert not unexpected, (
        f"這些欄位不該因宇宙改變而變：{unexpected}")

    # 「斷言通過」與「真的驗到東西」是兩回事——一個什麼都沒測到的測試不該是綠燈。
    # 若完全沒有欄位改變，代表自算/校準這條路徑根本沒被執行
    # （最常見原因：樣本全是官方 PER 覆蓋率高的大型股，官方值一律勝出），
    # 此時「PER/PBR 只在預期範圍內改變」這個結論是空的，必須讓測試失敗。
    assert bad1, (
        "本關空轉：完全沒有欄位改變，代表 _derive_valuation_fallback 的自算/校準"
        "路徑沒有被執行過（樣本可能全是官方 PER 覆蓋率高的股票）。"
        "測試若在這裡給綠燈，等於宣稱驗過了一條從未執行的程式碼。"
        "請確認 pick_universe() 有納入缺官方 PER 的股票。")
    print("  ✓ 只有 PER/PBR 因當日橫斷面校準而改變（設計如此，B-1 的決策），"
          "且該路徑確實被執行到")

    # ================================================================
    print("\n■ [3] clean_and_scale：值會變（橫斷面），但子集內排序必須完全一致")
    print("-" * 76)
    # ⚠️ 這一關**必須用 neutralize="none"**。
    #
    # 中性化是對 [產業 dummies, log 市值] 取 OLS 殘差：
    #     resid = x − X(XᵀX)⁻¹Xᵀx
    # 每一支扣掉的擬合值不同，**它不是單調轉換**——兩支股票的相對順序
    # 本來就可能因宇宙改變而互換。拿「排序必須不變」去驗中性化是**驗錯性質**。
    # （實測：v2 開 industry_mktcap 後 2021-06-16 的 Spearman 是 −0.809524，
    #   那是中性化的正常行為，不是 bug。）
    #
    # 中性化的正確驗收在 test_availability_flags.py [4]：
    # 產業內殘差均值 ≈ 0、殘差與 log 市值相關 ≈ 0。
    fc = clean_and_scale(f0.copy(), macro_norm="cross", neutralize="none")
    sc = clean_and_scale(s0.copy(), macro_norm="cross", neutralize="none")
    fd, sd = align(fc, sc)

    # 值本身應該不同——若完全相同，反而代表 clean_and_scale 沒有真的做橫斷面
    _, w3 = compare_bitwise(fd, sd, ["Return_5d"], "scaled")
    print(f"  Return_5d 值的最大絕對差 {w3:.3e}（**應該 > 0**，證明確實有做橫斷面標準化）")
    assert w3 > 0, "clean_and_scale 後值完全相同，橫斷面標準化可能沒生效"

    # 排序：z-score 是單調轉換，所以子集內的相對排名不該因宇宙而變——
    # **但要先扣掉三個「合法會改變排名」的機制**，否則驗到的是它們而不是 z-score。
    #
    # 這三個機制是實際跑出來、逐一追出來的（斷言被打臉三次）：
    #
    #  (a) winsorize 同分
    #      `clean_and_scale` 是 winsorize → z-score 兩步。夾到同一個 1%/99%
    #      邊界的股票會變成同分；40 支宇宙與 8 支宇宙的分位數不同，
    #      「在 full 側同分、在 subset 側沒有」的股票對排名自然不同。
    #      那是 winsorize 的設計目的（極端處刻意丟棄資訊）。
    #
    #  (b) NaN 填補（**最初兩次都沒想到的那個**）
    #      尾端的 `fillna(0.0)` 把缺值填成 0 = 「該宇宙的橫斷面平均」。
    #      實例：2018-08-13 的 1240 標準化前是 NaN，填成 0 之後——
    #        full 宇宙其他 7 支 z-score 全為正(0.005~1.74) → 0 排第 1（最低）
    #        subset 其他值落在 -0.88~1.81                 → 0 排第 6
    #      同一個填補值在不同宇宙落在不同位置，兩邊都是正確陳述。
    #
    #  (c) 橫斷面統計量本身
    #      這才是要驗的對象：只要 (a)(b) 扣乾淨，剩下的排名就必須完全一致。
    from scipy.stats import spearmanr

    # (b) 標準化前就是 NaN 的列 → 排除
    pre_nan = fa["Return_5d"].isna().to_numpy()

    def _untied(a: np.ndarray) -> np.ndarray:
        """回傳「這個值在該日是唯一的」的遮罩；有重複＝被 winsorize 夾成同分。"""
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        return cnt[inv] == 1

    fd = fd.reset_index(drop=True)
    sd = sd.reset_index(drop=True)
    worst_rho, worst_day, n_day, n_skip, n_tied, n_nan = 1.0, None, 0, 0, 0, 0
    for d, idx in fd.groupby("Date").groups.items():
        pos = np.asarray(idx, dtype=int)
        x = fd.loc[pos, "Return_5d"].to_numpy(dtype="float64")
        y = sd.loc[pos, "Return_5d"].to_numpy(dtype="float64")
        m = ~(np.isnan(x) | np.isnan(y)) & ~pre_nan[pos]     # (b) 排除填補列
        n_nan += int(pre_nan[pos].sum())
        if m.sum() < 3:
            n_skip += 1
            continue
        keep = m.copy()
        keep[m] = _untied(x[m]) & _untied(y[m])              # (a) 排除同分
        n_tied += int(m.sum() - keep.sum())
        if keep.sum() < 3:
            n_skip += 1
            continue
        rho = spearmanr(x[keep], y[keep]).statistic
        n_day += 1
        if rho < worst_rho:
            worst_rho, worst_day = float(rho), d
    print(f"  逐日 Spearman（扣掉 NaN 填補與 winsorize 同分後）：{n_day} 天可比"
          f"｜最差 {worst_rho:.6f}"
          f"（{str(worst_day)[:10] if worst_day is not None else '—'}）")
    print(f"  排除的觀測：NaN 填補 {n_nan:,}｜winsorize 同分 {n_tied:,}"
          f"｜可比不足 3 筆的日子 {n_skip}")
    assert n_day > 0, "沒有任何一天可比——遮罩邏輯可能過嚴"
    assert worst_rho > 0.999999, (
        "扣掉 NaN 填補與 winsorize 同分之後，橫斷面標準化仍改變了子集內的相對排序。"
        "z-score 是單調轉換，這代表分組或分位數邏輯真的有問題。")
    print("  ✓ 扣掉兩個合法機制後排序完全一致（z-score 單調性成立）")

    # ================================================================
    print("\n■ [4] Group D macro：市場層級常數，與宇宙無關")
    print("-" * 76)
    from marketmamba.config import FEATURE_GROUPS
    macro = [c for c in FEATURE_GROUPS["macro_environment"] if c in fa.columns]
    bad4, worst4 = compare_bitwise(fa, sa, macro, "macro")
    print(f"  比對 {len(macro)} 欄｜最大絕對差 {worst4:.3e}｜有差異 {len(bad4)}")
    assert not bad4, f"macro 欄位不該隨宇宙改變：{bad4}"
    print("  ✓ 逐位元相同")

    print("\n" + "=" * 76)
    print("全部通過")
    print("=" * 76)


if __name__ == "__main__":
    main()
