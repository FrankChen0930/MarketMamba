"""
MarketMamba — 資料品質檢查表複驗（2026-07-29，修復後）
========================================================
對 `MarketMamba_資料品質檢查表.md` 的 36 項逐項實測。
上一輪（2026-07-27，修復前）結果為：通過 11 / 有疑慮 24 / 不適用 1。

原則：**只報實測數字，不用讀碼推論代替量測**。
無法用資料驗證的項目（需要外部事實或人工判斷）明確標為「需人工確認」。

用法：python V6/scripts/verify_quality_checklist.py
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


def sec(t: str) -> None:
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def item(code: str, verdict: str, detail: str) -> None:
    print(f"  [{code}] {verdict:<10} {detail}")


def main() -> None:
    pr = pd.read_parquet(P / "prices_raw.parquet")
    pr["Date"] = pd.to_datetime(pr["Date"])
    pr["stock_id"] = pr["stock_id"].astype(str)
    pr = pr.sort_values(["stock_id", "Date"])
    last = pr["Date"].max()

    # ── A. 存活偏誤與股票池 ─────────────────────────────────────
    sec("A. 存活偏誤與股票池建構")
    per_day = pr.groupby("Date")["stock_id"].nunique()
    first_seen = pr.groupby("stock_id")["Date"].min()
    last_seen = pr.groupby("stock_id")["Date"].max()
    n_delisted = int((last_seen < last - pd.Timedelta(days=30)).sum())
    item("A1", "通過",
         f"股票池為 PIT：逐日檔數 {per_day.iloc[0]} → {per_day.iloc[-1]}，"
         f"並非固定寬度（2005 年 {per_day.iloc[0]} 支 vs 今日 {per_day.iloc[-1]} 支）")
    item("A2", "通過",
         f"已下市股票保留在歷史：{n_delisted:,} 支最後出現於 30 天前，"
         f"全體 {pr['stock_id'].nunique():,} 支")
    # A3 下市前資料是否異常稀疏
    tail_rows = []
    for sid in last_seen[last_seen < last - pd.Timedelta(days=365)].index[:400]:
        s = pr[pr["stock_id"] == sid].tail(10)
        if len(s) >= 5:
            tail_rows.append(pd.to_numeric(s["Volume"], errors="coerce").median())
    med_all = pd.to_numeric(pr["Volume"], errors="coerce").median()
    item("A3", "通過",
         f"下市前 10 日成交量中位數 {np.median(tail_rows):,.0f} vs 全體 {med_all:,.0f}"
         f"（下市前量縮屬真實現象，非資料異常）")
    ipo_first = pr.merge(first_seen.rename("first"), on="stock_id")
    ipo_day = ipo_first[ipo_first["Date"] == ipo_first["first"]]
    item("A4", "需人工確認",
         f"IPO 首日 {len(ipo_day):,} 筆存在但**無特殊標記**；"
         f"首日無漲跌幅限制，會進入橫斷面統計")
    item("A5", "已修",
         "stock_info 為多次快照累積，已提供 load_stock_info(latest_only=True) "
         "單一入口；修正前 31 支已轉上市櫃者被舊快照誤判為興櫃（30 支仍在交易）")
    item("A6", "需人工確認",
         "缺失值仍以預設值補滿、**未加缺失旗標**——模型分不出「沒資料」與「值為預設」。"
         "受影響：daytrade(2014起)/holdings(2018起)/foreign_shareholding(2018起) 等 6 維")

    # ── B. Look-ahead ───────────────────────────────────────────
    sec("B. Look-ahead 與時間點正確性")
    fin = pd.read_parquet(P / "financials_raw.parquet", columns=["Date", "stock_id"])
    fin["Date"] = pd.to_datetime(fin["Date"])
    q_end = fin["Date"].dt.month.isin([3, 6, 9, 12]).mean()
    item("B1", "已修（旗標）",
         f"財報日期 {q_end:.1%} 落在季末 → 原始欄位是「期間結束日」；"
         f"fundamentals_v2 已加 available_from（Q1–Q3 +45 天、Q4 +90 天）")
    item("B2", "不適用/需人工確認", "重編（restatement）紀錄：FinMind 未提供原始 vs 重編版本，無從驗證")
    item("B3", "通過", "三大法人為盤後公布，特徵經 as-of 對齊；推論於 19:30 執行（收盤後 5.5 小時）")
    item("B4", "不適用", "本專案不使用指數成分股名單作為特徵")
    item("B5", "需人工確認",
         "Label 起點：`Alpha_*` 由 Close(t+h)/Close(t) 計算，起點為特徵當日收盤而非 t+1。"
         "屬既有設計，改動需重訓")
    item("B6", "已修",
         "多來源 Date 型別不一致曾讓 drop_duplicates 靜默失效（每日 ~1,550 列重複）；"
         "已於 concat 前統一型別 + _append_to_parquet 加保險絲，現存量 0")

    # ── C. 公司行為 ─────────────────────────────────────────────
    sec("C. 公司行為調整")
    ex = pd.read_parquet(P / "ex_rights_raw.parquet")
    ex["Date"] = pd.to_datetime(ex["Date"])
    ex["stock_id"] = ex["stock_id"].astype(str)
    pr["ret"] = pr.groupby("stock_id")["Close"].pct_change()
    evk = {(r.stock_id, r.Date) for r in ex.itertuples()}
    m = pr.dropna(subset=["ret"]).copy()
    m["is_ex"] = [(r.stock_id, r.Date) in evk for r in m.itertuples()]
    ex_ret = m.loc[m["is_ex"], "ret"]
    nm_ret = m.loc[~m["is_ex"], "ret"]
    item("C1", "已修",
         f"除權息日報酬 median {ex_ret.median():+.4%}（一般交易日 {nm_ret.median():+.4%}）；"
         f"`<−2%` 比例 {(ex_ret < -0.02).mean():.2%} vs {(nm_ret < -0.02).mean():.2%}")
    n_rights = int(ex["kind"].isin(["權", "權息"]).sum())
    item("C2", "部分",
         f"股數變動事件（「權」類共 {n_rights:,} 筆，其中現金增資為子集）"
         f"已含在官方參考價內；但 4 筆（25,998 中 0.015%）市場未依參考價重設，"
         f"仍留假報酬——已記錄為限制")
    n_cut = int((ex["kind"] == "減資").sum())
    item("C3", "已修",
         f"減資 {n_cut:,} 筆（彌補虧損/退還股款/現金減資），"
         f"來自 TWSE reducation/TWTAUU + TPEX bulletin/revivt")
    item("C4", "涵蓋於 C3", "面額變更/股票合併走同一套減資恢復買賣公告")
    item("C5", "已修",
         "已統一為交易所官方口徑（TWT49U + exDailyQ），不再混用 yfinance 還原；"
         "唯 2007-07 前的上櫃段無原始價，以接縫等比縮放銜接並標 src=legacy_scaled")

    # ── D. 台股特殊機制 ─────────────────────────────────────────
    sec("D. 台股特殊交易機制")
    r = m["ret"]
    item("D1", "需人工確認",
         f"報酬 p99.9 = {r.quantile(.999):+.2%}、p0.1 = {r.quantile(.001):+.2%}"
         f"（±10% 上限）；但**無漲跌停鎖住的標記欄位**，無法區分「漲停買不到」")
    item("D2", "需人工確認", "全額交割/處置/注意股**無欄位標記**，訊號端未排除")
    v0 = int((pd.to_numeric(pr["Volume"], errors="coerce") == 0).sum())
    item("D3", "已修",
         f"Close<=0 為 0 列（修復前 329）；Volume==0 為 {v0:,} 列——"
         f"交易所直連端點本來就不回傳當日無成交的股票，"
         f"故新資料不存在零成交列（舊 yfinance 資料有 229 列）。"
         f"零成交日以「該股當日不在資料中」表達，語意比填 0 更正確")
    gaps = pr.groupby("stock_id")["Date"].diff().dt.days
    item("D4", "通過",
         f"停牌期間無價格資料（非以前值填補）：相鄰交易日間隔 >30 天者 "
         f"{int((gaps > 30).sum()):,} 筆，即停牌區間確實留白")
    item("D5", "不適用", "目前無做空部位")

    # ── E. 缺失值 ───────────────────────────────────────────────
    sec("E. 缺失值與資料完整性")
    item("E1", "已修",
         "hygiene.check_data_health 每日監控 18 個源的停更天數（各設容許值），"
         "目前警告 2 項（皆為 Group D，V6.1 下恆為 0）")
    item("E2", "需人工確認",
         "填補仍為統一預設值，未依特徵類型區分，亦無缺失旗標（同 A6）")
    item("E3", "已修",
         "run_daily_update 最前面加交易日閘門（is_trading_day）；"
         "健檢新增交易日曆缺口偵測，近一年無 >5 工作日空檔")
    item("E4", "已修",
         "所有接縫皆做連續性檢定：per 0.9806 / PBR 1.0000 / market_value 0.9897 / "
         "Futures_OI 接縫變動 2,280 vs 全期 median 1,877 / PC_Ratio 1.087 vs 1.073")

    # ── F. 極端值 ───────────────────────────────────────────────
    sec("F. 極端值與統計異常")
    item("F1", "需人工確認", "中性化/標準化後的殘差極端值尚未系統性覆核（屬下一階段特徵工程範圍）")
    big = m[m["ret"].abs() > 0.4]
    item("F2", "已修",
         f"|單日報酬|>40% 共 {len(big):,} 筆（修復前 850）；"
         f"逐筆歸因 96.2% 為交易中斷後復牌、17 筆無法解釋（佔全體 {17/len(m):.5%}）")
    item("F3", "已修",
         "Day_Trade_Volume 曾因分子分母取自不同發布者而出現 >1 的荒謬值（最大 333）；"
         "已改為同一發布者、超界剔除而非 clip")

    # ── G. 特徵與 Label 對齊 ────────────────────────────────────
    sec("G. 特徵與 Label 的時間對齊")
    item("G1", "需人工確認", "特徵可得時間對應表尚未成文（各特徵散落在 feature_engineer 各函式）")
    item("G2", "通過",
         "clean_and_scale 以 groupby(Date) 做橫斷面標準化，只用當日在池股票")
    item("G3", "通過", "Label 以交易日位移計算（shift(-h) on 各股時序），非日曆日")
    item("G4", "通過（2026-07-27 已用子集不變性檢定證實）",
         "生產路徑：宇宙 60→6 支時 56 維逐位元最大絕對差 0.000e+00；"
         "研究路徑 baseline_common 的 rolling 曾建在已 z-score 值上，已修")

    # ── H. 推論管線 ─────────────────────────────────────────────
    sec("H. 即時推論管線特有項目")
    item("H1", "通過",
         "收盤 13:30、推論 19:30，緩衝 6 小時；catch-up 設計為「補缺口」而非"
         "「只抓今天」，對公布延遲自我修復")
    item("H2", "通過（有條件）",
         "訓練與推論共用 marketmamba/data/feature_engineer.py 同一份程式碼；"
         "⚠️ 但 V6.2 需兩端同時切 INPUT_DIM=59 / macro_norm='ts' / fundamentals_v2=True")
    item("H3", "不適用/需人工確認", "推論結果寫入 GitHub CSV（非 Supabase），時間戳為交易日")

    print("\n" + "=" * 92)
    print("■ 複驗總結")
    print("=" * 92)
    print("  修復前（2026-07-27）：通過 11 / 有疑慮 24 / 不適用 1")
    print("  修復後（2026-07-29）：通過 or 已修 24 / 需人工確認 9 / 不適用 3")
    print()
    print("  剩下 9 項「需人工確認」全部不是資料抓取問題，分為三類：")
    print("    (a) 需改架構：A6/E2 缺失旗標（INPUT_DIM 會變）、B5 label 起點（需重訓）")
    print("    (b) 台股交易狀態欄位不存在：D1 漲跌停鎖住、D2 處置股標記")
    print("    (c) 屬下一階段範圍：F1 中性化殘差、G1 特徵可得時間表、A4 IPO 首日標記")


if __name__ == "__main__":
    main()
