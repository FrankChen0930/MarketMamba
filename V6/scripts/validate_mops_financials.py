"""
MarketMamba — MOPS 財報直連的上線前驗證（2026-08-04）
======================================================================
本腳本**只讀不寫**：不碰任何 parquet，純粹回答「MOPS 抓回來的值可以直接
取代／補進 FinMind 的資料嗎」。三項驗證全過才把 fetcher 接進每日流程。

【為什麼一定要跑這個】
本專案累積的 13 條雷區，絕大多數是「照欄名直譯 → 值錯了但不報錯」：
  · MOPS 股利把現金股利拆三欄 → 只讀一欄會讓台泥變成 0 元股利
  · TAIFEX 期貨用「多方/空方」、選擇權用「買方/賣方」→ 選擇權 8 欄靜默落空
財報這批同樣有四個對映陷阱（單位千元、累計 vs 單季、全形括號、兩個毛利欄），
其中任何一個沒處理好，產出的都是「看起來正常但值是錯的」特徵——
那比現在的「值是舊的」更糟，因為舊值看得見、錯值看不見。

【三項驗證】
  V1 差分規則   MOPS 2025 累計相減 vs FinMind 2025 單季（逐季、逐 type）
  V2 量級交叉   重疊期 (股,季,type) 的 MOPS/FinMind 比值，median 應 ≈ 1.000
  V3 手工核對   2330 / 2317 / 2454 的毛利率與 ROE 是否對得上實際
  V4 接縫連續   Q1 2026（MOPS）對 Q4 2025（FinMind）不得有量級跳階

用法（Windows，本機 pandas 2.2.2）：
    python V6/scripts/validate_mops_financials.py            # 全部
    python V6/scripts/validate_mops_financials.py --only v1  # 單項
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parents[1]
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR                      # noqa: E402
from marketmamba.data.fetcher import (                            # noqa: E402
    fetch_financial_statement_mops_direct,
    fetch_revenue_mops_direct,
)

KINDS = ("financials", "balance_sheet", "cashflow")
# 下游 `feature_engineer` 真正消費的 type——這幾個錯了會直接汙染特徵
CRITICAL = {
    "financials": ["Revenue", "GrossProfit", "EPS", "IncomeAfterTaxes"],
    "balance_sheet": ["EquityAttributableToOwnersOfParent", "Equity"],
    "cashflow": ["CashFlowsFromOperatingActivities",
                 "CashProvidedByInvestingActivities"],
}


def _load(kind: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"{kind}_raw.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    df["stock_id"] = df["stock_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _ratio_report(mops: pd.DataFrame, fin: pd.DataFrame, label: str,
                  types: list[str] | None = None) -> bool:
    """逐 type 比對 MOPS 與 FinMind 的比值，回傳是否通過。

    用**比值的 median** 而非平均：財報有大量接近 0 的科目，平均會被
    少數 0/0 的極端比值主導。判準只看關鍵 type（其餘僅列出供參考）。
    """
    key = ["stock_id", "type"]
    m = mops.merge(fin, on=key, how="inner", suffixes=("_mops", "_fin"))
    if m.empty:
        print(f"    ⚠️ {label}: 無重疊列，無法比對")
        return False
    ok_all = True
    rows = []
    for t, g in m.groupby("type"):
        a, b = g["value_mops"].to_numpy(), g["value_fin"].to_numpy()
        good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6)
        if good.sum() < 5:
            continue
        r = a[good] / b[good]
        med = float(np.median(r))
        if t == "EPS":
            # ⚠️ EPS 不能用相對誤差判：它只報到小數第 2 位，累計相減後
            #    「0.04 − 0.01 = 0.03」對上 FinMind 的 0.04 就是 25% 相對誤差，
            #    但絕對差只有 0.01＝一個報告單位，屬捨入而非錯誤。
            #    改用絕對容差 ±0.02（兩個報告單位，涵蓋兩次相減的捨入）。
            within = float(np.mean(np.abs(a[good] - b[good]) <= 0.02))
            note = "（絕對差 ≤0.02）"
        else:
            within = float(np.mean(np.abs(r - 1.0) < 0.01))
            note = ""
        is_crit = types is not None and t in types
        # 判準：關鍵 type 的 median 必須落在 1.000 ±0.5%，且 ≥90% 的列在容差內
        passed = (abs(med - 1.0) < 0.005) and (within >= 0.90)
        if is_crit and not passed:
            ok_all = False
        rows.append((t, int(good.sum()), med, within, is_crit, passed, note))
    rows.sort(key=lambda x: (not x[4], x[0]))
    print(f"    {'type':<58} {'n':>6} {'median比值':>11} {'容差內':>8}")
    for t, n, med, within, is_crit, passed, note in rows:
        mark = "★" if is_crit else " "
        flag = "" if (passed or not is_crit) else "   ❌"
        print(f"   {mark}{t:<58} {n:>6,} {med:>11.6f} {within:>7.1%}{flag}{note}")
    return ok_all


def v1_differencing() -> bool:
    """V1：累計 → 單季的差分規則是否重現 FinMind 的值。

    這是四個陷阱裡唯一**無法從單一季別看出來**的：Q1 不需相減，所以就算
    規則寫錯，只補 2026Q1 也完全正常；等 Q2（8/14 截止）進來才會整批錯。
    故拿 2025 年（FinMind 有完整資料）當對照，逐季驗。
    """
    print("\n" + "=" * 78)
    print("V1  差分規則驗證：MOPS 累計相減 vs FinMind 單季（2025 年）")
    print("=" * 78)
    ok = True
    for kind in ("financials", "cashflow"):
        fin = _load(kind)
        for season in (2, 3):        # Q2/Q3 需相減；Q1 不需、Q4 留給 V2
            date = pd.Timestamp({2: "2025-06-30", 3: "2025-09-30"}[season])
            print(f"\n  ── {kind} 2025Q{season}（基準日 {date.date()}）")
            mops = fetch_financial_statement_mops_direct(kind, 2025, season)
            if mops is None or mops.empty:
                print("    ⚠️ MOPS 無資料，跳過")
                ok = False
                continue
            sub = fin[fin["Date"] == date]
            if sub.empty:
                print("    ⚠️ FinMind 無該季資料，跳過")
                continue
            if not _ratio_report(mops[["stock_id", "type", "value"]],
                                 sub[["stock_id", "type", "value"]],
                                 f"{kind} 2025Q{season}", CRITICAL[kind]):
                ok = False
    print(f"\n  V1 結果：{'✅ 通過' if ok else '❌ 未通過'}")
    return ok


def v2_magnitude() -> bool:
    """V2：重疊期的量級交叉驗證（單位換算 + 對映是否正確）。

    用 2025-12-31（Q4）——FinMind 三張表都有 2,225~2,475 支，樣本最大。
    ⚠️ Q4 是流量表相減幅度最大的一季，同時也檢驗了差分在年底邊界的正確性。
    """
    print("\n" + "=" * 78)
    print("V2  量級交叉驗證：MOPS vs FinMind（2025Q4，重疊期最大樣本）")
    print("=" * 78)
    ok = True
    for kind in KINDS:
        fin = _load(kind)
        date = pd.Timestamp("2025-12-31")
        print(f"\n  ── {kind} 2025Q4")
        mops = fetch_financial_statement_mops_direct(kind, 2025, 4)
        if mops is None or mops.empty:
            print("    ⚠️ MOPS 無資料")
            ok = False
            continue
        sub = fin[fin["Date"] == date]
        print(f"    覆蓋：MOPS {mops['stock_id'].nunique():,} 支"
              f"｜FinMind {sub['stock_id'].nunique():,} 支")
        if not _ratio_report(mops[["stock_id", "type", "value"]],
                             sub[["stock_id", "type", "value"]],
                             f"{kind} 2025Q4", CRITICAL[kind]):
            ok = False

    # 月營收一併驗（FinMind 2026-04 尚未腰斬，樣本完整）
    print("\n  ── 月營收 2026-03（FinMind Date=2026-04-01）")
    rev_m = fetch_revenue_mops_direct(2026, 3)
    rev_f = pd.read_parquet(PROCESSED_DIR / "revenue_raw.parquet")
    rev_f["Date"] = pd.to_datetime(rev_f["Date"])
    rf = rev_f[rev_f["Date"] == pd.Timestamp("2026-04-01")]
    if rev_m is None or rf.empty:
        print("    ⚠️ 無法比對")
        ok = False
    else:
        m = rev_m.merge(rf[["stock_id", "revenue"]], on="stock_id",
                        how="inner", suffixes=("_mops", "_fin"))
        good = m["revenue_fin"].abs() > 0
        r = m.loc[good, "revenue_mops"] / m.loc[good, "revenue_fin"]
        med, within = float(r.median()), float((r.sub(1).abs() < 0.01).mean())
        print(f"    覆蓋：MOPS {len(rev_m):,} 支｜FinMind {len(rf):,} 支"
              f"｜重疊 {len(m):,} 支")
        print(f"    median 比值 = {med:.6f}｜±1% 內 = {within:.1%}"
              f"｜逐位元相同 = {(m['revenue_mops'] == m['revenue_fin']).mean():.1%}")
        if not (abs(med - 1.0) < 0.005 and within >= 0.90):
            print("    ❌ 未通過")
            ok = False
    print(f"\n  V2 結果：{'✅ 通過' if ok else '❌ 未通過'}")
    return ok


def v3_manual() -> bool:
    """V3：抽三支核對毛利率／ROE／EPS 是否對得上實際。

    這是唯一能抓到「所有比值都 1.000、但科目對映整組挑錯」的檢查——
    比值驗證只證明「MOPS 的 X 欄 == FinMind 的 X type」，
    不證明「X 就是我們以為的那個科目」。

    ⚠️ 判準刻意**不硬編**我記憶中的水準。第一版寫死「台積電毛利率 55~62%」，
    實測 2026Q1 是 66.2% 而判定失敗——但用 FinMind 自己的資料回頭看，
    毛利率是 53.1→57.8→59.0→59.5→62.3 一路往上，66.2% 是延續而非異常，
    **過時的是我的參考值、不是資料**。
    若當時把區間放寬到剛好通過，那就是拿結果去改測試、驗證等於白做。
    改成「必須落在 FinMind 自身近況的合理延伸」：門檻寬到不會被單一結果牽著走，
    但仍然擋得住 1000 倍單位錯誤與挑錯欄位（那會差好幾個量級）。
    完整軌跡一併印出，供人眼判讀。
    """
    print("\n" + "=" * 78)
    print("V3  指標核對：2330 台積電 / 2317 鴻海 / 2454 聯發科（2026Q1）")
    print("=" * 78)
    inc = fetch_financial_statement_mops_direct("financials", 2026, 1)
    bs = fetch_financial_statement_mops_direct("balance_sheet", 2026, 1)
    if inc is None or bs is None:
        print("  ⚠️ 抓取失敗")
        return False

    fin_i, fin_b = _load("financials"), _load("balance_sheet")

    def _mops(df, sid, t):
        s = df[(df["stock_id"] == sid) & (df["type"] == t)]["value"]
        return float(s.iloc[0]) if len(s) else float("nan")

    def _hist(sid: str) -> pd.DataFrame:
        """FinMind 自身近 8 季的營收／毛利率／EPS／權益（獨立於 MOPS）。"""
        a = fin_i[(fin_i["stock_id"] == sid) & (fin_i["Date"] >= "2024-01-01")]
        p = a.pivot_table(index="Date", columns="type", values="value", aggfunc="last")
        b = fin_b[(fin_b["stock_id"] == sid) & (fin_b["Date"] >= "2024-01-01")]
        q = b.pivot_table(index="Date", columns="type", values="value", aggfunc="last")
        out = pd.DataFrame(index=p.index)
        out["營收(億)"] = (p.get("Revenue") / 1e8).round(0)
        out["毛利率%"] = (p.get("GrossProfit") / p.get("Revenue") * 100).round(1)
        out["EPS"] = p.get("EPS")
        eq = q.get("EquityAttributableToOwnersOfParent")
        if eq is not None:
            out["年化ROE%"] = (p.get("IncomeAfterTaxes") * 4 / eq * 100).round(1)
        return out

    NAMES = {"2330": "台積電", "2317": "鴻海", "2454": "聯發科"}
    ok = True
    for sid, nm in NAMES.items():
        h = _hist(sid)
        rev = _mops(inc, sid, "Revenue")
        gp = _mops(inc, sid, "GrossProfit")
        ni = _mops(inc, sid, "IncomeAfterTaxes")
        eps = _mops(inc, sid, "EPS")
        eq = _mops(bs, sid, "EquityAttributableToOwnersOfParent")
        gm = gp / rev * 100 if rev else float("nan")
        roe = ni * 4 / eq * 100 if eq else float("nan")

        print(f"\n  ── {sid} {nm}")
        print("     FinMind 歷史（獨立於 MOPS）:")
        for d, r in h.iterrows():
            print(f"       {d.date()}  營收 {r['營收(億)']:>8,.0f} 億"
                  f"｜毛利率 {r['毛利率%']:>5.1f}%｜EPS {r['EPS']:>6.2f}"
                  f"｜年化ROE {r.get('年化ROE%', float('nan')):>6.1f}%")
        print(f"     MOPS 2026Q1  營收 {rev / 1e8:>8,.0f} 億"
              f"｜毛利率 {gm:>5.1f}%｜EPS {eps:>6.2f}｜年化ROE {roe:>6.1f}%")

        last = h.iloc[-1]
        checks = [
            # 毛利率：與上一季差距 ≤15 個百分點（挑錯欄位會差數十 pp）
            ("毛利率", abs(gm - last["毛利率%"]) <= 15,
             f"Δ={gm - last['毛利率%']:+.1f}pp（上限 ±15pp）"),
            # 營收：季變動落在 0.5×~2.0×（單位錯 1000 倍會直接爆掉）
            ("營收", 0.5 <= (rev / 1e8) / last["營收(億)"] <= 2.0,
             f"季比={(rev / 1e8) / last['營收(億)']:.2f}×（0.5~2.0×）"),
            # EPS：唯一不套 ×1000 的欄，季變動同上
            ("EPS", 0.3 <= abs(eps / last["EPS"]) <= 3.0 if last["EPS"] else True,
             f"季比={eps / last['EPS']:.2f}×（0.3~3.0×）"),
            # ROE：與上一季差距 ≤20pp
            ("年化ROE", abs(roe - last.get("年化ROE%", roe)) <= 20,
             f"Δ={roe - last.get('年化ROE%', roe):+.1f}pp（上限 ±20pp）"),
        ]
        for label, passed, detail in checks:
            if not passed:
                ok = False
            print(f"       {label:<8} {'✓' if passed else '❌'}  {detail}")

    print(f"\n  V3 結果：{'✅ 通過' if ok else '❌ 未通過'}")
    return ok


def _median_ratio(cur: pd.DataFrame, prev: pd.DataFrame, t: str) -> tuple[float, int]:
    a = cur[cur["type"] == t].set_index("stock_id")["value"]
    b = prev[prev["type"] == t].set_index("stock_id")["value"]
    common = a.index.intersection(b.index)
    if len(common) < 20:
        return float("nan"), len(common)
    b2 = b[common]
    r = (a[common] / b2).replace([np.inf, -np.inf], np.nan)
    r = r[b2.abs() > 1e-6].dropna()
    return (float(r.median()) if len(r) else float("nan")), len(r)


def v4_seam() -> bool:
    """V4：接縫連續性——2026Q1（MOPS）接在 2025Q4（FinMind）之後不得跳階。

    ⚠️ 判準**自我校準**，不硬編任何數字。第一版寫死「median 季比須落在
    0.3~3.0」，結果現金流量表報 0.164 判定失敗——但那是對的：
    cashflow 是「年初至今累計」，Q4 是全年、Q1 只有一季，比值本來就 ≈1/4。
    純 FinMind 內部的同一個轉換實測 Q1/Q4 = 0.130(2023) / 0.169(2024) / 0.171(2025)，
    我們算出的 0.164 正落在其中。

    故改成：**參考值直接取 FinMind 前一年的同一個季度轉換**（2025Q1/2024Q4），
    再要求本次觀測值與它相差不超過 3 倍。這樣三張表各自的慣例
    （單季／累計／時點）自動被編碼進參考值，測試不需要知道差異存在。
    """
    print("\n" + "=" * 78)
    print("V4  接縫連續性：2026Q1（MOPS）vs 2025Q4（FinMind）")
    print("     參考值 = FinMind 自身 2025Q1 vs 2024Q4 的同一個轉換（自我校準）")
    print("=" * 78)
    ok = True
    for kind in KINDS:
        fin = _load(kind)
        prev = fin[fin["Date"] == pd.Timestamp("2025-12-31")]
        ref_cur = fin[fin["Date"] == pd.Timestamp("2025-03-31")]
        ref_prev = fin[fin["Date"] == pd.Timestamp("2024-12-31")]
        mops = fetch_financial_statement_mops_direct(kind, 2026, 1)
        if mops is None or prev.empty:
            print(f"  ⚠️ {kind}: 無法比對")
            ok = False
            continue
        print(f"\n  ── {kind}")
        for t in CRITICAL[kind]:
            obs, n = _median_ratio(mops, prev, t)
            ref, _ = _median_ratio(ref_cur, ref_prev, t)
            if not np.isfinite(obs) or not np.isfinite(ref) or abs(ref) < 1e-9:
                print(f"    {t:<45} 樣本不足（n={n}），略過")
                continue
            rel = obs / ref
            passed = 1 / 3 <= abs(rel) <= 3.0
            if not passed:
                ok = False
            print(f"    {t:<45} n={n:>5,} 觀測 {obs:>7.3f}"
                  f"｜參考 {ref:>7.3f}｜倍率 {rel:>6.2f}×"
                  f"{'  ✓' if passed else '  ❌ 疑似跳階'}")
    print(f"\n  V4 結果：{'✅ 通過' if ok else '❌ 未通過'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["v1", "v2", "v3", "v4"], default=None)
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    runners = {"v1": v1_differencing, "v2": v2_magnitude,
               "v3": v3_manual, "v4": v4_seam}
    todo = [args.only] if args.only else ["v1", "v2", "v3", "v4"]
    results = {}
    for k in todo:
        try:
            results[k] = runners[k]()
        except Exception as e:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"\n  {k} 執行時例外：{e}")
            results[k] = False

    print("\n" + "=" * 78)
    print("總結")
    print("=" * 78)
    for k in todo:
        print(f"  {k.upper()}  {'✅ 通過' if results.get(k) else '❌ 未通過'}")
    allok = all(results.get(k) for k in todo)
    print(f"\n  {'✅ 全部通過 → 可以接進 run_daily_update' if allok else '❌ 有項目未通過 → 先不要接上去'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
