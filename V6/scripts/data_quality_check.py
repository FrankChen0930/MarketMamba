"""
MarketMamba V6.1 — Data Quality Diagnostic
============================================
針對所有 V6.1 新增特徵（及部分舊特徵）做資料品質檢查：
  1. NaN 缺失率（整體 + 按年份）
  2. 零值比率（整體 + 按年份）
  3. 股票覆蓋率（有多少 stock_id 有資料）
  4. 時間覆蓋（最早/最晚有效資料日期）
  5. 特別針對：被刪掉的那個「幾乎都是 0」的特徵找線索

用法：
  cd V6
  python scripts/data_quality_check.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketmamba.config import PROCESSED_DIR, FEATURE_COLS, FEATURE_GROUPS

# ──────────────────────────────────────────────
# 0. 輔助函數
# ──────────────────────────────────────────────

def pct(n, total):
    return f"{n/total*100:.1f}%" if total > 0 else "N/A"

def print_header(title: str):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def yearly_stats(series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """計算每年的 NaN 率和零值率"""
    years = dates.dt.year
    rows = []
    for yr in sorted(years.unique()):
        mask = years == yr
        s = series[mask]
        n = len(s)
        nan_pct = s.isna().sum() / n * 100
        zero_pct = (s.fillna(0) == 0).sum() / n * 100
        near_zero_pct = (s.fillna(0).abs() < 1e-6).sum() / n * 100
        rows.append({
            "Year": yr,
            "N": n,
            "NaN%": f"{nan_pct:.1f}%",
            "Zero%": f"{zero_pct:.1f}%",
            "~Zero%": f"{near_zero_pct:.1f}%",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 1. 載入各個 parquet 原始資料
# ──────────────────────────────────────────────

print_header("載入原始 Parquet 檔案")

def load(name: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / name
    if not path.exists():
        print(f"  ❌ 找不到: {name}")
        return None
    df = pd.read_parquet(path)
    print(f"  ✅ {name:<45} {df.shape[0]:>10,} rows × {df.shape[1]} cols")
    return df

df_price         = load("prices_raw.parquet")
df_inst          = load("institutional_raw.parquet")
df_margin        = load("margin_raw.parquet")
df_per           = load("per_raw.parquet")
df_securities    = load("securities_raw.parquet")
df_market_value  = load("market_value_raw.parquet")
df_daytrade      = load("daytrade_raw.parquet")
df_holdings      = load("holdings_raw.parquet")
df_rev           = load("revenue_raw.parquet")
df_fin           = load("financials_raw.parquet")
df_balance_sheet = load("balance_sheet_raw.parquet")
df_cashflow      = load("cashflow_raw.parquet")
df_macro         = load("macro_raw.parquet")
df_futures       = load("futures_institutional_raw.parquet")
df_options       = load("options_institutional_raw.parquet")
df_dividend      = load("dividend_raw.parquet")
df_foreign_sh    = load("foreign_shareholding_raw.parquet")
df_fear_greed    = load("fear_greed.parquet")
df_business      = load("business_indicator.parquet")
df_fed_rate      = load("fed_rate.parquet")


# ──────────────────────────────────────────────
# 2. 逐一檢查 V6.1 新增的原始資料源
# ──────────────────────────────────────────────

print_header("V6.1 新增資料源：基本資訊")

# ── 2-1. Holdings（大戶持股）──
if df_holdings is not None:
    print("\n📌 holdings_raw.parquet（大戶持股分級）")
    print(f"   Columns: {df_holdings.columns.tolist()}")
    date_col = "Week" if "Week" in df_holdings.columns else "Date"
    if date_col in df_holdings.columns:
        df_holdings[date_col] = pd.to_datetime(df_holdings[date_col])
        print(f"   日期範圍: {df_holdings[date_col].min()} → {df_holdings[date_col].max()}")
    n_stocks = df_holdings["stock_id"].nunique() if "stock_id" in df_holdings.columns else "N/A"
    print(f"   涵蓋股票數: {n_stocks}")
    # 看數值欄位
    num_cols = df_holdings.select_dtypes(include="number").columns.tolist()
    print(f"   數值欄位: {num_cols}")
    for c in num_cols[:5]:
        zero_pct = (df_holdings[c].fillna(0) == 0).mean() * 100
        nan_pct  = df_holdings[c].isna().mean() * 100
        print(f"     {c:<35} NaN={nan_pct:.1f}%  Zero={zero_pct:.1f}%")

# ── 2-2. Securities（借券餘額）──
if df_securities is not None:
    print("\n📌 securities_raw.parquet（借券餘額）")
    print(f"   Columns: {df_securities.columns.tolist()}")
    if "Date" in df_securities.columns:
        df_securities["Date"] = pd.to_datetime(df_securities["Date"])
        print(f"   日期範圍: {df_securities['Date'].min()} → {df_securities['Date'].max()}")
    n_stocks = df_securities["stock_id"].nunique() if "stock_id" in df_securities.columns else "N/A"
    print(f"   涵蓋股票數: {n_stocks}")
    num_cols = df_securities.select_dtypes(include="number").columns.tolist()
    print(f"   數值欄位: {num_cols}")
    for c in num_cols[:5]:
        zero_pct = (df_securities[c].fillna(0) == 0).mean() * 100
        nan_pct  = df_securities[c].isna().mean() * 100
        print(f"     {c:<35} NaN={nan_pct:.1f}%  Zero={zero_pct:.1f}%")

# ── 2-3. Foreign Shareholding（外資持股比例）──
if df_foreign_sh is not None:
    print("\n📌 foreign_shareholding_raw.parquet（外資持股比例）")
    print(f"   Columns: {df_foreign_sh.columns.tolist()}")
    date_col = "Date" if "Date" in df_foreign_sh.columns else "date"
    if date_col in df_foreign_sh.columns:
        df_foreign_sh[date_col] = pd.to_datetime(df_foreign_sh[date_col])
        print(f"   日期範圍: {df_foreign_sh[date_col].min()} → {df_foreign_sh[date_col].max()}")
    n_stocks = df_foreign_sh["stock_id"].nunique() if "stock_id" in df_foreign_sh.columns else "N/A"
    print(f"   涵蓋股票數: {n_stocks}")
    num_cols = df_foreign_sh.select_dtypes(include="number").columns.tolist()
    print(f"   數值欄位: {num_cols}")
    for c in num_cols[:5]:
        zero_pct = (df_foreign_sh[c].fillna(0) == 0).mean() * 100
        nan_pct  = df_foreign_sh[c].isna().mean() * 100
        print(f"     {c:<35} NaN={nan_pct:.1f}%  Zero={zero_pct:.1f}%")

# ── 2-4. Dividend（股利）──
if df_dividend is not None:
    print("\n📌 dividend_raw.parquet（股利資料）")
    print(f"   Columns: {df_dividend.columns.tolist()}")
    date_col = "Date" if "Date" in df_dividend.columns else "date"
    if date_col in df_dividend.columns:
        df_dividend[date_col] = pd.to_datetime(df_dividend[date_col])
        print(f"   日期範圍: {df_dividend[date_col].min()} → {df_dividend[date_col].max()}")
    n_stocks = df_dividend["stock_id"].nunique() if "stock_id" in df_dividend.columns else "N/A"
    print(f"   涵蓋股票數: {n_stocks}")
    # 找 cash dividend column
    cash_col = [c for c in df_dividend.columns if "cash" in c.lower()]
    print(f"   現金股利欄位: {cash_col}")
    for c in cash_col[:3]:
        df_dividend[c] = pd.to_numeric(df_dividend[c], errors="coerce")
        zero_pct = (df_dividend[c].fillna(0) == 0).mean() * 100
        nan_pct  = df_dividend[c].isna().mean() * 100
        print(f"     {c:<35} NaN={nan_pct:.1f}%  Zero={zero_pct:.1f}%")

# ── 2-5. Cashflow（自由現金流）──
if df_cashflow is not None:
    print("\n📌 cashflow_raw.parquet（現金流量表）")
    print(f"   Columns: {df_cashflow.columns.tolist()}")
    date_col = "Date" if "Date" in df_cashflow.columns else "date"
    if date_col in df_cashflow.columns:
        df_cashflow[date_col] = pd.to_datetime(df_cashflow[date_col])
        print(f"   日期範圍: {df_cashflow[date_col].min()} → {df_cashflow[date_col].max()}")
    n_stocks = df_cashflow["stock_id"].nunique() if "stock_id" in df_cashflow.columns else "N/A"
    print(f"   涵蓋股票數: {n_stocks}")
    if "type" in df_cashflow.columns:
        types = df_cashflow["type"].value_counts().head(10)
        print(f"   type 欄位 top-10:\n{types.to_string()}")

# ── 2-6. Fear & Greed ──
if df_fear_greed is not None:
    print("\n📌 fear_greed.parquet（CNN 恐懼貪婪指數）")
    print(f"   Columns: {df_fear_greed.columns.tolist()}")
    date_col = "Date" if "Date" in df_fear_greed.columns else "date"
    if date_col in df_fear_greed.columns:
        df_fear_greed[date_col] = pd.to_datetime(df_fear_greed[date_col])
        print(f"   日期範圍: {df_fear_greed[date_col].min()} → {df_fear_greed[date_col].max()}")
    print(f"   總行數: {len(df_fear_greed):,}")
    num_cols = df_fear_greed.select_dtypes(include="number").columns.tolist()
    for c in num_cols[:3]:
        nan_pct  = df_fear_greed[c].isna().mean() * 100
        zero_pct = (df_fear_greed[c].fillna(0) == 0).mean() * 100
        print(f"     {c:<35} NaN={nan_pct:.1f}%  Zero={zero_pct:.1f}%  range=[{df_fear_greed[c].min():.1f}, {df_fear_greed[c].max():.1f}]")

# ── 2-7. Business Indicator（景氣燈號）──
if df_business is not None:
    print("\n📌 business_indicator.parquet（台灣景氣燈號）")
    print(f"   Columns: {df_business.columns.tolist()}")
    date_col = "Date" if "Date" in df_business.columns else "date"
    if date_col in df_business.columns:
        df_business[date_col] = pd.to_datetime(df_business[date_col])
        print(f"   日期範圍: {df_business[date_col].min()} → {df_business[date_col].max()}")
    print(f"   總行數: {len(df_business):,}")
    num_cols = df_business.select_dtypes(include="number").columns.tolist()
    for c in num_cols[:3]:
        nan_pct  = df_business[c].isna().mean() * 100
        zero_pct = (df_business[c].fillna(0) == 0).mean() * 100
        print(f"     {c:<35} NaN={nan_pct:.1f}%  Zero={zero_pct:.1f}%  range=[{df_business[c].min():.1f}, {df_business[c].max():.1f}]")

# ── 2-8. FED Rate ──
if df_fed_rate is not None:
    print("\n📌 fed_rate.parquet（FED 利率）")
    print(f"   Columns: {df_fed_rate.columns.tolist()}")
    date_col = "Date" if "Date" in df_fed_rate.columns else "date"
    if date_col in df_fed_rate.columns:
        df_fed_rate[date_col] = pd.to_datetime(df_fed_rate[date_col])
        print(f"   日期範圍: {df_fed_rate[date_col].min()} → {df_fed_rate[date_col].max()}")
    print(f"   總行數: {len(df_fed_rate):,}")
    num_cols = df_fed_rate.select_dtypes(include="number").columns.tolist()
    for c in num_cols[:3]:
        nan_pct  = df_fed_rate[c].isna().mean() * 100
        zero_pct = (df_fed_rate[c].fillna(0) == 0).mean() * 100
        print(f"     {c:<35} NaN={nan_pct:.1f}%  Zero={zero_pct:.1f}%  range=[{df_fed_rate[c].min():.3f}, {df_fed_rate[c].max():.3f}]")

# ── 2-9. Futures Institutional ──
if df_futures is not None:
    print("\n📌 futures_institutional_raw.parquet（外資期貨）")
    print(f"   Columns: {df_futures.columns.tolist()}")
    date_col = "Date" if "Date" in df_futures.columns else "date"
    if date_col in df_futures.columns:
        df_futures[date_col] = pd.to_datetime(df_futures[date_col])
        print(f"   日期範圍: {df_futures[date_col].min()} → {df_futures[date_col].max()}")
    print(f"   總行數: {len(df_futures):,}")
    if "institutional_investors" in df_futures.columns:
        print(f"   institutional_investors 分類:\n{df_futures['institutional_investors'].value_counts().to_string()}")

# ── 2-10. Options Institutional ──
if df_options is not None:
    print("\n📌 options_institutional_raw.parquet（選擇權）")
    print(f"   Columns: {df_options.columns.tolist()}")
    date_col = "Date" if "Date" in df_options.columns else "date"
    if date_col in df_options.columns:
        df_options[date_col] = pd.to_datetime(df_options[date_col])
        print(f"   日期範圍: {df_options[date_col].min()} → {df_options[date_col].max()}")
    print(f"   總行數: {len(df_options):,}")


# ──────────────────────────────────────────────
# 3. 建立完整 feature matrix 並做欄位級別診斷
# ──────────────────────────────────────────────

print_header("建立 Feature Matrix 並做全欄位診斷")
print("（這一步需要幾分鐘，正在建立完整 56D feature matrix...）\n")

try:
    from marketmamba.data.feature_engineer import build_features, clean_and_scale

    df_feat = build_features(
        df_price         = df_price,
        df_inst          = df_inst,
        df_margin        = df_margin,
        df_per           = df_per,
        df_securities    = df_securities,
        df_market_value  = df_market_value,
        df_daytrade      = df_daytrade,
        df_holdings      = df_holdings,
        df_rev           = df_rev,
        df_fin           = df_fin,
        df_balance_sheet = df_balance_sheet,
        df_cashflow      = df_cashflow,
        df_macro         = df_macro,
        df_futures_inst  = df_futures,
        df_options_inst  = df_options,
        df_dividend      = df_dividend,
        df_foreign_shareholding = df_foreign_sh,
        df_fear_greed    = df_fear_greed,
        df_business_indicator = df_business,
        df_fed_rate      = df_fed_rate,
    )

    df_feat["Date"] = pd.to_datetime(df_feat["Date"])
    n_total = len(df_feat)
    n_stocks = df_feat["stock_id"].nunique()
    date_min = df_feat["Date"].min()
    date_max = df_feat["Date"].max()

    print(f"✅ Feature matrix 建立成功")
    print(f"   總行數: {n_total:,}")
    print(f"   股票數: {n_stocks:,}")
    print(f"   日期範圍: {date_min.date()} → {date_max.date()}")
    print()

    # ── 3-1. 全欄位摘要表 ──
    print_header("各特徵欄位品質摘要（排序：缺失率 + 零值率 由高到低）")

    rows = []
    for col in FEATURE_COLS:
        if col not in df_feat.columns:
            rows.append({"Feature": col, "Group": "?", "NaN%": "❌MISSING", "Zero%": "N/A",
                         "~Zero%": "N/A", "Min": "N/A", "Max": "N/A", "Std": "N/A"})
            continue

        s = df_feat[col]
        nan_pct   = s.isna().mean() * 100
        zero_pct  = (s.fillna(0) == 0).mean() * 100
        nzero_pct = (s.fillna(0).abs() < 1e-6).mean() * 100
        grp = next((k for k, v in FEATURE_GROUPS.items() if col in v), "?")
        rows.append({
            "Feature": col,
            "Group":   grp[:10],
            "NaN%":    f"{nan_pct:.1f}%",
            "Zero%":   f"{zero_pct:.1f}%",
            "~Zero%":  f"{nzero_pct:.1f}%",
            "Min":     f"{s.min():.4f}" if s.notna().any() else "N/A",
            "Max":     f"{s.max():.4f}" if s.notna().any() else "N/A",
            "Std":     f"{s.std():.4f}" if s.notna().any() else "N/A",
        })

    summary_df = pd.DataFrame(rows)
    # 排序：先看 NaN 最高，再看 Zero 最高
    summary_df["_sort_nan"]  = summary_df["NaN%"].str.replace("%","").replace("❌MISSING","100").astype(float)
    summary_df["_sort_zero"] = summary_df["Zero%"].str.replace("%","").replace("N/A","0").astype(float)
    summary_df = summary_df.sort_values(["_sort_nan", "_sort_zero"], ascending=False).drop(columns=["_sort_nan","_sort_zero"])
    print(summary_df.to_string(index=False))

    # ── 3-2. V6.1 新增欄位的年度分佈 ──
    NEW_FEATURES_V61 = [
        "Holdings_Large_Pct", "Holdings_Large_Change",
        "Securities_Balance", "Foreign_Holding_Pct",
        "Dividend_Yield_Fwd", "Free_Cash_Flow",
        "Futures_OI_Foreign", "Options_PC_Ratio",
        "Fear_Greed", "Business_Signal", "FED_Rate",
    ]

    print_header("V6.1 新增特徵：按年份的零值率分佈（關鍵診斷）")
    for col in NEW_FEATURES_V61:
        if col not in df_feat.columns:
            print(f"\n  ❌ {col} 不存在")
            continue
        print(f"\n  ── {col} ──")
        ys = yearly_stats(df_feat[col], df_feat["Date"])
        print(ys.to_string(index=False))

    # ── 3-3. 找出最可能是「幾乎全 0」的那個被刪除的特徵 ──
    print_header("🔍 尋找「幾乎全是 0」的疑似被刪掉的特徵")
    print("（對照 V6 中曾經存在但 V6.1 被移除的：Market_Closed）\n")

    # Market_Closed 應該是原本 V6 的 Group D 的一個 feature
    # 按照文件：取代 Market_Closed → Business_Signal
    # 來重新確認 Business_Signal 的資料狀況
    if "Business_Signal" in df_feat.columns:
        bs = df_feat["Business_Signal"]
        zero_pct = (bs == 0).mean() * 100
        default_pct = (bs == 23.0).mean() * 100   # 23 是我們設的 default
        print(f"Business_Signal:")
        print(f"  - 零值率:              {zero_pct:.1f}%")
        print(f"  - 等於 default(23.0):  {default_pct:.1f}%  ← 高代表資料根本沒讀到")
        print(f"  - 非 default 的比率:   {100-default_pct:.1f}%")
        ys = yearly_stats(bs, df_feat["Date"])
        print(ys.to_string(index=False))

    # ── 3-4. 找出哪些特徵在 2005~2014 早期資料有嚴重缺失 ──
    print_header("早期資料（2005-2014）覆蓋率問題")
    early_mask = df_feat["Date"].dt.year <= 2014
    df_early = df_feat[early_mask]
    if len(df_early) == 0:
        print("  ⚠️ 沒有 2014 以前的資料")
    else:
        print(f"  2005-2014 行數: {len(df_early):,}")
        print()
        problem_cols = []
        for col in FEATURE_COLS:
            if col not in df_early.columns:
                continue
            zero_or_nan = (df_early[col].isna() | (df_early[col].fillna(0).abs() < 1e-6)).mean() * 100
            if zero_or_nan > 50:
                problem_cols.append((col, zero_or_nan))
        problem_cols.sort(key=lambda x: -x[1])
        if problem_cols:
            print("  以下特徵在 2005-2014 的零/缺失率 > 50%：")
            for col, pct_val in problem_cols:
                grp = next((k for k, v in FEATURE_GROUPS.items() if col in v), "?")
                print(f"    [{grp[:10]:>10}] {col:<30} {pct_val:.1f}%")
        else:
            print("  ✅ 沒有嚴重問題（無欄位在早期資料中缺失 > 50%）")

except Exception as e:
    import traceback
    print(f"\n❌ Feature matrix 建立失敗：{e}")
    traceback.print_exc()
    print("\n改為只做原始 parquet 的基礎診斷（以上已完成）")


print_header("診斷完成")
print("請重點關注：")
print("  1. NaN% 或 Zero% 很高的欄位 → 資料根本沒被讀到")
print("  2. 早期（2005-2014）缺失嚴重的欄位 → 模型在學假資料（default 值）")
print("  3. Business_Signal 的 default 佔比 → 是否取代 Market_Closed 成功")
