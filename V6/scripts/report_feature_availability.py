"""
report_feature_availability.py — 特徵可得性量測（F0）
========================================================
把「哪一維特徵、在哪一年、對多少比例的股票真的有資料」變成可重跑的量測，
取代過去每次都臨時寫診斷腳本的模式。

【為什麼需要這支】
2026-07-29 的量測顯示，特徵缺失有**兩種性質完全不同**的形狀，而現行程式碼把它們
都用同一招 `fillna(0)` 處理：

  1. 整欄全缺（daytrade 2014 前、holdings/foreign_shareholding 2018 前）
     → 當日橫斷面所有股票同值 → z-score 自動歸零 → 本身無害，
       但模型無從知道「這一維今天是關著的」。

  2. 部分缺（institutional 2005-2011 只有 10~18% 覆蓋）
     → 85% 的列被補成 0，而「淨買超恰為 0」是合法值，兩者無法區分
     → 橫斷面 z-score 後，模型實際學到的是「這支有沒有被資料涵蓋」
       這個**選擇性代理變數**，不是籌碼訊號。

第 2 種才是要處理的。本腳本量的就是區分這兩者所需的數字。

【輸出三張表】
  表1  各來源逐年 (日期, 股票) 命中率     → 決定訓練起點、決定哪些來源要旗標
  表2  base matrix 逐年「當日橫斷面 std≈0」的維度 → 確認整欄死值的實際範圍
  表3  stock_info 跨快照的產業別變動筆數   → F3 中性化的 PIT 限制揭露用

用法（Windows 本機、repo 根目錄執行）：
    python V6/scripts/report_feature_availability.py
    python V6/scripts/report_feature_availability.py --skip-matrix   # 只跑表1/表3（快）
    python V6/scripts/report_feature_availability.py --json out.json # 另存機器可讀結果

隔離：全程唯讀 Data/processed_v6/，不寫任何 parquet，不 import 任何模型相關模組。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402

BASE_MATRIX = PROCESSED_DIR / "baseline_cache" / "baseline_base_59d.parquet"

# ── 逐日資料源：與 prices_raw 做 (日期, 股票) 命中率比對 ──────────────────
# 說明欄記錄「0 代表什麼」，這是判斷要不要加旗標的關鍵：
#   若 0 是合法值（淨買超為 0、無融資餘額），缺失補 0 就無法區分 → 需要旗標。
DAILY_SOURCES: list[tuple[str, str, str]] = [
    ("institutional",  "institutional_raw.parquet",        "0=當日無法人買賣（合法值）→ 需旗標"),
    ("margin",         "margin_raw.parquet",               "0=無融資餘額或無信用交易資格 → 需旗標"),
    ("daytrade",       "daytrade_raw.parquet",             "0=當日無當沖（合法值）→ 需旗標"),
    ("per",            "per_raw.parquet",                  "缺=無本益比資料（虧損股亦可能缺）→ 需旗標"),
    ("market_value",   "market_value_raw.parquet",         "缺=無市值資料 → 併入 Avail_Valuation"),
    ("securities",     "securities_raw.parquet",           "0=無借券餘額（合法值）→ 需旗標"),
    ("foreign_sh",     "foreign_shareholding_raw.parquet", "缺=2018 前無此資料 → 需旗標"),
]

# ── 非逐日資料源：只報涵蓋區間與頻率，不算逐日命中率 ─────────────────────
PERIODIC_SOURCES: list[tuple[str, str, str]] = [
    ("holdings",      "holdings_raw.parquet",       "Week"),
    ("revenue",       "revenue_raw.parquet",        None),
    ("financials",    "financials_raw.parquet",     None),
    ("balance_sheet", "balance_sheet_raw.parquet",  None),
    ("cashflow",      "cashflow_raw.parquet",       None),
    ("dividend",      "dividend_raw.parquet",       None),
]


def _date_col(path: Path) -> str | None:
    """raw parquet 的日期欄名不統一（Date / date / Week），統一在這裡解析。"""
    names = pq.read_schema(path).names
    for c in ("Date", "date", "Week"):
        if c in names:
            return c
    return None


def _pair_keys(path: Path, date_col: str) -> np.ndarray:
    """
    把 (日期, 股票) 壓成單一 int64 鍵，回傳**排序後**的陣列供 searchsorted 比對。

    用排序陣列 + np.isin(kind="sort") 而不是 Python set：
    institutional_raw 有 32.8M 列，set 會吃掉數 GB，本機只有 24GB 且推論可能同時在跑。
    """
    t = pq.read_table(path, columns=[date_col, "stock_id"]).to_pandas()
    d = pd.to_datetime(t[date_col], errors="coerce")
    # 非 4 位數字代號（ETF/權證/特別股）→ -1，反正不會出現在 prices 宇宙裡
    sid = pd.to_numeric(t["stock_id"], errors="coerce").fillna(-1).astype("int64")
    day = (d.astype("int64") // 10**9 // 86400).astype("int64")
    keys = (day * 100_000 + sid).to_numpy()
    keys.sort()
    return keys


def table1_daily_coverage() -> pd.DataFrame:
    """表1：各逐日來源對 prices_raw 宇宙的 (日期, 股票) 命中率（%），逐年。"""
    px_path = PROCESSED_DIR / "prices_raw.parquet"
    px = pq.read_table(px_path, columns=["Date", "stock_id"]).to_pandas()
    px_dt = pd.to_datetime(px["Date"], errors="coerce")
    year = px_dt.dt.year.to_numpy()
    px_sid = pd.to_numeric(px["stock_id"], errors="coerce").fillna(-1).astype("int64")
    px_key = ((px_dt.astype("int64") // 10**9 // 86400) * 100_000 + px_sid).to_numpy()

    n_rows = len(px_key)
    print(f"[表1] prices_raw 基準：{n_rows:,} 列 / "
          f"{px['stock_id'].nunique():,} 支 / "
          f"{str(px_dt.min())[:10]} ~ {str(px_dt.max())[:10]}\n", flush=True)

    years = np.unique(year)
    out: dict[str, pd.Series] = {}
    for name, fname, note in DAILY_SOURCES:
        p = PROCESSED_DIR / fname
        if not p.exists():
            print(f"  {name:14s} 檔案不存在，略過", flush=True)
            continue
        dc = _date_col(p)
        if dc is None:
            print(f"  {name:14s} 找不到日期欄，略過", flush=True)
            continue
        keys = _pair_keys(p, dc)
        hit = np.isin(px_key, keys, kind="sort")
        pct = pd.Series(
            {int(y): 100.0 * hit[year == y].mean() for y in years}
        )
        out[name] = pct
        del keys, hit
        print(f"  {name:14s} 完成  ({note})", flush=True)

    return pd.DataFrame(out).sort_index()


def table2_dead_dims() -> pd.DataFrame | None:
    """
    表2：base matrix 中「當日橫斷面 std < 1e-6」的交易日比例（%），逐年逐維。

    std≈0 代表那一天該維對所有股票同值 → 經 z-score 後恆為 0 → 對模型完全無資訊。
    這是判斷「整欄全缺」範圍的直接證據，不是從起始日推論的。
    """
    if not BASE_MATRIX.exists():
        print(f"[表2] 找不到 {BASE_MATRIX}，略過（可用 --skip-matrix 明確跳過）", flush=True)
        return None

    schema = pq.read_schema(BASE_MATRIX)
    label_cols = {"Date", "stock_id", "Alpha_5d", "Alpha_10d", "Alpha_20d", "Alpha_60d",
                  "y_rank_5d", "y_rank_20d"}
    feat_cols = [c for c in schema.names if c not in label_cols]
    print(f"[表2] 讀 {BASE_MATRIX.name}：{len(feat_cols)} 個特徵欄", flush=True)

    t = pq.read_table(BASE_MATRIX, columns=["Date"] + feat_cols).to_pandas()
    t["_y"] = pd.to_datetime(t["Date"]).dt.year
    daily_std = t.groupby(["_y", "Date"])[feat_cols].std()
    dead = (daily_std < 1e-6)
    return (dead.groupby(level=0).mean() * 100.0)


def table3_industry_churn() -> dict:
    """
    表3：stock_info 跨快照的產業別變動——F3 中性化的 PIT 限制量化。

    中性化要用產業分類，但 stock_info 是「多次快照的累積」而非逐日的 PIT 歷史。
    這裡量的是「有多少支股票的產業別在快照之間變過」——那就是用現況快照
    回推歷史時會被錯誤歸類的規模上限。不假裝解決，只把數字講清楚。

    ⚠️ 分母必須先過 `filter_tradable_universe`。未過濾時 947 支（30.7%）看似嚴重，
       但範例全是 006201 / 00679B 這類 ETF 與債券 ETF——它們本來就不在訓練宇宙裡。
       這是 2026-07-29 換 production 資料檔時學到的同一課：
       **比率型指標的分母要跟下游實際使用的宇宙對齊，否則會留下永遠不會消的假警報。**
    """
    from marketmamba.data.hygiene import filter_tradable_universe, load_stock_info

    full = load_stock_info(latest_only=False)
    if full.empty or "industry_category" not in full.columns:
        return {"error": "stock_info 無 industry_category 欄"}

    tradable = set(filter_tradable_universe(full)["stock_id"].astype(str))
    sub = full[full["stock_id"].astype(str).isin(tradable)].copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")

    # 「舊 → 新」的實際轉換組成，才能區分「交易所改分類名」與「公司真的換產業」。
    # 只看聚合的變動比例會得到 39.6% 這種嚇人但誤導的數字。
    old = (sub[sub["date"] < "2025-01-01"].sort_values("date")
           .drop_duplicates("stock_id", keep="last").set_index("stock_id")["industry_category"])
    new = (sub[sub["date"] >= "2026-01-01"].sort_values("date")
           .drop_duplicates("stock_id", keep="last").set_index("stock_id")["industry_category"])
    both = old.index.intersection(new.index)
    transitions = Counter((str(old[s]), str(new[s])) for s in both if str(old[s]) != str(new[s]))

    return {
        "累積列數":           int(len(full)),
        "快照批次數":         int(sub["date"].nunique()),
        "可交易宇宙股票數":   int(sub["stock_id"].nunique()),
        "註":                 "沒有任何單一批次涵蓋全宇宙（最大批次 2026-05-06 僅 1,072 支），"
                              "latest_only=True 是多個批次的拼接，不是一致的橫斷面",
        "新舊批次都出現的股票": int(len(both)),
        "標籤不同":           int(sum(transitions.values())),
        "轉換組成_前10":      transitions.most_common(10),
        "判讀":               "轉換清單以交易所分類改版為主（觀光事業→觀光餐旅、"
                              "創新版股票→創新板股票 是錯字修正、其他→運動休閒/綠能環保 是新增類別），"
                              "真正的公司重新分類僅個位數 → 用最新分類回推歷史做中性化可接受，但需揭露",
    }


def table4_sector_canonicalization() -> dict:
    """
    表4：產業分類名稱的跨市場不一致——直接影響 KG 產業邊與中性化 dummies。

    實測發現 TPEX（上櫃）與 TWSE（上市）對同一個產業用不同名稱：
      上櫃「運動休閒類」 vs 上市「運動休閒」（差一個「類」字）
      上櫃「其他電子類」 vs 上市「其他電子業」
      上櫃「金融業」     vs 上市「金融保險」
    若直接拿 industry_category 建圖，同一個產業會沿上市/上櫃被切成兩個互不相連的群，
    而那條分界線在經濟上毫無意義。
    """
    from marketmamba.data.hygiene import filter_tradable_universe, load_stock_info

    lat = load_stock_info(latest_only=True)
    lat = filter_tradable_universe(lat)
    cats = set(lat["industry_category"].astype(str))
    counts = lat["industry_category"].astype(str).value_counts().to_dict()

    suffix_pairs = [(c[:-1], c) for c in sorted(cats)
                    if c.endswith("類") and c[:-1] in cats]
    return {
        "分類數":            len(cats),
        "差一個_類_字的成對": [{"上市": a, "上市支數": counts.get(a, 0),
                                "上櫃": b, "上櫃支數": counts.get(b, 0)}
                               for a, b in suffix_pairs],
        "其他已知同義":      ["其他電子類(tpex) = 其他電子業(twse)",
                              "金融業(tpex) = 金融保險(twse)",
                              "觀光事業(殘留 1 支) = 觀光餐旅"],
        "結論":              "F2 建產業邊與 F3 建中性化 dummies 前，"
                             "必須先過 feature_spec_v2.canonical_sector()",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-matrix", action="store_true", help="跳過表2（base matrix 較慢）")
    ap.add_argument("--json", type=str, default=None, help="另存 JSON 結果")
    args = ap.parse_args()

    result: dict = {}

    print("=" * 78)
    print("表1｜各資料源對 prices_raw 宇宙的 (日期, 股票) 命中率（%）")
    print("=" * 78)
    t1 = table1_daily_coverage()
    print()
    print(t1.round(1).to_string())
    result["table1_daily_coverage_pct"] = t1.round(2).to_dict()

    print()
    print("  判讀：命中率長期低於 ~70% 的來源，其 fillna(0) 會與合法的 0 值混淆，")
    print("        模型可能改為學習「有沒有資料」這個選擇性代理變數 → 需要可得性旗標。")

    print()
    print("=" * 78)
    print("表1b｜非逐日資料源的涵蓋區間")
    print("=" * 78)
    periodic = {}
    for name, fname, dc_hint in PERIODIC_SOURCES:
        p = PROCESSED_DIR / fname
        if not p.exists():
            print(f"  {name:14s} 檔案不存在")
            continue
        dc = dc_hint or _date_col(p)
        if dc is None:
            print(f"  {name:14s} 找不到日期欄")
            continue
        cols = [dc] + (["stock_id"] if "stock_id" in pq.read_schema(p).names else [])
        t = pq.read_table(p, columns=cols).to_pandas()
        d = pd.to_datetime(t[dc], errors="coerce")
        ns = t["stock_id"].nunique() if "stock_id" in t else -1
        print(f"  {name:14s} {str(d.min())[:10]} ~ {str(d.max())[:10]}  "
              f"列={len(t):>9,}  股票={ns:>5}")
        periodic[name] = {"start": str(d.min())[:10], "end": str(d.max())[:10],
                          "rows": int(len(t)), "stocks": int(ns)}
    result["table1b_periodic"] = periodic

    if not args.skip_matrix:
        print()
        print("=" * 78)
        print("表2｜base matrix 中「當日橫斷面 std≈0」的交易日比例（%）")
        print("     （100 = 該年該維每天都是死值，對模型完全無資訊）")
        print("=" * 78)
        t2 = table2_dead_dims()
        if t2 is not None:
            # 只印「至少有一年是死值」的欄位，全活的欄位不佔版面
            interesting = t2.columns[(t2 > 1.0).any(axis=0)]
            print()
            print(t2[interesting].round(0).astype(int).to_string())
            print()
            print(f"  全期皆有資訊（未列出）的維度：{len(t2.columns) - len(interesting)} 個")
            result["table2_dead_dim_pct"] = t2.round(2).to_dict()

    print()
    print("=" * 78)
    print("表3｜stock_info 產業別的 PIT 限制（F3 中性化用）")
    print("=" * 78)
    t3 = table3_industry_churn()
    for k, v in t3.items():
        print(f"  {k:20s} {v}")
    result["table3_industry_churn"] = t3

    print()
    print("=" * 78)
    print("表4｜產業分類名稱的跨市場不一致（F2 產業邊 / F3 中性化 dummies 必看）")
    print("=" * 78)
    t4 = table4_sector_canonicalization()
    for k, v in t4.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"      {item}")
        else:
            print(f"  {k:20s} {v}")
    result["table4_sector_canonicalization"] = t4

    if args.json:
        Path(args.json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n已存：{args.json}")


if __name__ == "__main__":
    main()
