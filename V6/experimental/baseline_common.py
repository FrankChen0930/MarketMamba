"""
baseline_common.py — 方向二 Baseline 對照：共用資料層 + 評估層
================================================================
對應協定：docs/baseline-experiment-protocol-draft-2026-07-11.md（v1.0 凍結）
隔離原則：只讀 Data/processed_v6/ raw parquet，輸出到 Data/processed_v6/baseline_cache/
          （不進 git）；不動 production marketmamba/、不動 V6/models/。

提供三層功能（Ridge/Lasso、GBDT、LSTM/GRU 三階 baseline 共用）：
  1. build_base_matrix()    : 2010 起完整歷史 59 維 feature matrix（分 chunk 建構防 OOM）
                              + 協定 §2 universe 過濾 + clean_and_scale(macro_norm="ts")
                              + rank(Alpha_5d/20d) label（同 v6_short rank_transform 語意）
  2. build_derived()        : 協定 §4 的 lag/rolling/動能特徵（241 維，合計 300 維）
  3. load_xy() / 評估工具   : 日 Spearman IC、Newey-West t、Top50 組合回測（含成本）

記憶體設計（本機 24GB RAM，Colab 曾用高 RAM 建 8.7M 列矩陣，本機必須分塊）：
  - raw parquet 逐 chunk 用 pyarrow filters 篩股票讀取，不整檔載入
  - 全程 float32；衍生特徵分 4 個 part 檔案計算與儲存
  - 訓練端建議 day_stride=2（隔日抽樣）：5d 窗口重疊本就高度冗餘，樣本減半資訊損失小

用法（Windows 本機、repo 根目錄執行）：
  python V6/experimental/baseline_common.py --build         # 建 base matrix + derived（一次性，~1hr）
  python V6/experimental/baseline_common.py --rebuild-roll  # 只重建 rolling part（G4 修正）

⚠️ 2026-07-27 協定變更（v1.0 → v2.0）
  rolling 特徵（*_rmean / *_rstd 共 60 維）原本建在 clean_and_scale 之後的
  橫斷面 z-score 上，屬於「先橫斷面、再時序」的順序顛倒（資料品質檢查表 G4）。
  已改為建在 chunk 檔的原始值上、再逐日 winsorize + z-score。
  → 既有的 Ridge / GBDT / GRU 三階結果是舊特徵下的數字，重跑後才可與新結果並列。
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── sys.path：讓本檔從任何 cwd 都能 import marketmamba / experimental ──
_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

# ── 59 維 config 自切（與 run_dual_inference.py 同款；必須在 import feature 模組之前）──
import marketmamba.config as cfg

_RS = ["RS_5d", "RS_20d", "RS_60d"]
if not all(r in cfg.FEATURE_GROUPS["price_momentum"] for r in _RS):
    cfg.FEATURE_GROUPS["price_momentum"] = cfg.FEATURE_GROUPS["price_momentum"] + _RS
cfg.INPUT_DIM = 59
cfg.FEATURE_COLS = (cfg.FEATURE_GROUPS["price_momentum"] + cfg.FEATURE_GROUPS["institutional_flow"]
                    + cfg.FEATURE_GROUPS["fundamentals"] + cfg.FEATURE_GROUPS["macro_environment"])
cfg.GROUP_DIMS = {k: len(v) for k, v in cfg.FEATURE_GROUPS.items()}
assert len(cfg.FEATURE_COLS) == 59, f"expected 59 features, got {len(cfg.FEATURE_COLS)}"

from marketmamba.config import PROCESSED_DIR                      # noqa: E402
from marketmamba.data.feature_engineer import build_features, clean_and_scale  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("baseline_common")

FEATURE_COLS: list[str] = list(cfg.FEATURE_COLS)

# ============================================================
# 協定常數（v1.0 凍結，見 docs/baseline-experiment-protocol-draft-2026-07-11.md）
# ============================================================
PROTOCOL = {
    "RAW_START":    "2009-01-01",   # raw 起點（給滾動窗/asof join 的 lookback 緩衝）
    "MATRIX_START": "2010-01-01",   # matrix 起點（train 前留 ~2 年：202 天門檻 + macro ts 暖機）
    "TRAIN_START":  "2012-01-01",
    "TRAIN_END":    "2023-12-31",
    "TEST_START":   "2024-01-01",
    "TEST_END":     "2026-06-02",   # 與 Phase 3 harness 同窗
    "VAL_RATIO":    0.15,           # train 尾端 15% 交易日當 val（選超參數用）
    "MIN_HISTORY_DAYS": 202,        # SEQ_LEN 252 × 0.8，與 TemporalCrossSectionDataset 一致
    "TOP_N":        50,
    "REBALANCE_DAYS": 5,
    "COST_BUY":     0.0015,
    "COST_SELL":    0.0045,
}

CACHE_DIR  = PROCESSED_DIR / "baseline_cache"
CHUNK_DIR  = CACHE_DIR / "chunks"
BASE_PATH  = CACHE_DIR / "baseline_base_59d.parquet"
ROW_GROUP  = 200_000                # 小 row group → 讀取時 date filter 可有效裁剪

# ============================================================
# 協定 §4 附錄：扁平模型衍生特徵規格（凍結；GBDT 共用同一份）
# ============================================================
LAGS = [1, 5, 20]                                   # 59 × 3 = 177 維
ROLL_CORE = [                                       # 價量 + 籌碼核心 12 欄
    "Close", "Volume", "Return_1d",
    "Foreign_Net", "Investment_Trust_Net", "Dealer_Net",
    "Margin_Balance", "Short_Balance", "OBV",
    "RSI_14", "Volatility_20d", "Foreign_Holding_Pct",
]
ROLL_MEAN_WINDOWS = [5, 20, 60]                     # 12 × 3 = 36 維
ROLL_STD_WINDOWS  = [20, 60]                        # 12 × 2 = 24 維
MOM_WINDOWS       = [5, 10, 20, 60]                 # 4 維（原始還原收盤價的累積報酬，再橫斷面標準化）


def lag_names(n: int) -> list[str]:
    return [f"{c}_lag{n}" for c in FEATURE_COLS]


def roll_names() -> list[str]:
    names = [f"{c}_rmean{w}" for c in ROLL_CORE for w in ROLL_MEAN_WINDOWS]
    names += [f"{c}_rstd{w}" for c in ROLL_CORE for w in ROLL_STD_WINDOWS]
    names += [f"Mom_{w}d" for w in MOM_WINDOWS]
    return names


def all_feature_names() -> list[str]:
    """完整 300 維欄位（順序固定 = X 矩陣欄位順序）：59 base + 177 lag + 60 rolling + 4 momentum"""
    return FEATURE_COLS + lag_names(1) + lag_names(5) + lag_names(20) + roll_names()


# 衍生特徵分 4 個 part 檔（各檔行序 = base matrix 的 (Date, stock_id) 排序，逐列對齊）
def _derived_parts() -> list[tuple[Path, list[str]]]:
    return [
        (CACHE_DIR / "baseline_derived_lag1.parquet",  lag_names(1)),
        (CACHE_DIR / "baseline_derived_lag5.parquet",  lag_names(5)),
        (CACHE_DIR / "baseline_derived_lag20.parquet", lag_names(20)),
        (CACHE_DIR / "baseline_derived_roll.parquet",  roll_names()),
    ]


assert len(all_feature_names()) == 300, f"expected 300 dims, got {len(all_feature_names())}"


# ============================================================
# Raw 載入（帶 pyarrow filters，避免整檔進記憶體）
# ============================================================
_STOCK_RAWS = {   # build_features kwarg -> parquet 檔名（含 stock_id 欄，逐 chunk 篩讀）
    "df_inst":                 "institutional_raw",
    "df_margin":               "margin_raw",
    "df_per":                  "per_raw",
    "df_securities":           "securities_raw",
    "df_market_value":         "market_value_raw",
    "df_daytrade":             "daytrade_raw",
    "df_holdings":             "holdings_raw",
    "df_rev":                  "revenue_raw",
    "df_fin":                  "financials_raw",
    "df_balance_sheet":        "balance_sheet_raw",
    "df_cashflow":             "cashflow_raw",
    "df_dividend":             "dividend_raw",
    "df_foreign_shareholding": "foreign_shareholding_raw",
}
_MARKET_RAWS = {  # 市場層級（小檔，整檔載入一次後重用）
    "df_macro":              "macro_raw",
    "df_futures_inst":       "futures_institutional_raw",
    "df_options_inst":       "options_institutional_raw",
    "df_fear_greed":         "fear_greed",
    "df_business_indicator": "business_indicator",
    "df_fed_rate":           "fed_rate",
}


def _load_raw(name: str, stock_ids: list[str] | None = None) -> pd.DataFrame | None:
    """讀 raw parquet；stock_ids 給定時用 pyarrow filter 只解出該批股票（防 OOM 關鍵）。
    Date 欄名/型別正規化比照 merger._load。"""
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        logger.warning(f"  raw 不存在：{path.name}（該資料源以 None 傳入，特徵以預設值補）")
        return None
    filters = [("stock_id", "in", list(stock_ids))] if stock_ids is not None else None
    try:
        df = pd.read_parquet(path, filters=filters)
    except Exception:
        df = pd.read_parquet(path)                     # 檔案無 stock_id 欄等情況 → 整檔載入
        if stock_ids is not None and "stock_id" in df.columns:
            df = df[df["stock_id"].isin(stock_ids)]
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"] >= pd.Timestamp(PROTOCOL["RAW_START"])]
    return df


def _downcast_f32(df: pd.DataFrame, exclude: tuple[str, ...] = ("Date", "stock_id")) -> pd.DataFrame:
    for c in df.columns:
        if c not in exclude and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    return df


# ============================================================
# 1) Base matrix：59 維 + label（分 chunk 建構）
# ============================================================
def build_base_matrix(n_chunks: int = 5, force: bool = False) -> None:
    if BASE_PATH.exists() and not force:
        logger.info(f"base matrix 已存在，跳過：{BASE_PATH}")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── prices：全檔載入（8.7M 列尚可）→ 協定 §2 過濾 ──
    prices = _load_raw("prices_raw")
    n0 = len(prices)
    prices = prices[prices["stock_id"].astype(str).str.match(r"^\d{4}$")]
    prices = prices.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    stocks = sorted(prices["stock_id"].unique())
    print(f"[build] prices_raw {n0:,} → 過濾後 {len(prices):,} 列 | {len(stocks)} 支 | "
          f"{prices['Date'].min().date()} → {prices['Date'].max().date()}", flush=True)

    # ── 市場層級 raw：載一次重用 ──
    market_kwargs = {k: _load_raw(v) for k, v in _MARKET_RAWS.items()}

    # ── 逐 chunk 建特徵 ──
    chunks = np.array_split(np.array(stocks), n_chunks)
    for i, chunk in enumerate(chunks):
        out = CHUNK_DIR / f"base_chunk_{i}.parquet"
        if out.exists() and not force:
            print(f"[build] chunk {i+1}/{n_chunks} 已存在，跳過", flush=True)
            continue
        tc = time.time()
        chunk = list(chunk)
        p = prices[prices["stock_id"].isin(chunk)].copy()
        kwargs = {k: _load_raw(v, stock_ids=chunk) for k, v in _STOCK_RAWS.items()}
        mk = {k: (v.copy() if v is not None else None) for k, v in market_kwargs.items()}
        df = build_features(df_price=p, **kwargs, **mk)
        df = df[df["Date"] >= pd.Timestamp(PROTOCOL["MATRIX_START"])]
        keep = ["Date", "stock_id"] + FEATURE_COLS + ["Alpha_5d", "Alpha_20d"]
        df = _downcast_f32(df[keep])
        df.to_parquet(out, index=False)
        print(f"[build] chunk {i+1}/{n_chunks}：{len(chunk)} 支 → {len(df):,} 列 "
              f"({time.time()-tc:.0f}s)", flush=True)
        del df, p, kwargs
        gc.collect()

    # ── 合併 → clean_and_scale（橫斷面統計需要完整 cross-section，必須在合併後做）──
    print("[build] 合併 chunk + clean_and_scale(macro_norm='ts') ...", flush=True)
    df = pd.concat([pd.read_parquet(CHUNK_DIR / f"base_chunk_{i}.parquet") for i in range(n_chunks)],
                   ignore_index=True)
    n_before = len(df)
    df = clean_and_scale(df, macro_norm="ts")
    print(f"[build] clean_and_scale：{n_before:,} → {len(df):,} 列（NaN 剔除 {n_before-len(df):,}）",
          flush=True)
    df = _downcast_f32(df)
    df = df.sort_values(["Date", "stock_id"], kind="mergesort").reset_index(drop=True)

    # ── 協定 §2：≥202 天歷史才納入 cross-section（cumcount 以 clean 後資料計，同 Dataset 語意）──
    df["eligible"] = df.groupby("stock_id", sort=False).cumcount() >= (PROTOCOL["MIN_HISTORY_DAYS"] - 1)

    # ── label：per-date pct-rank 置中 [-0.5, +0.5]（同 short_model.rank_transform 語意）──
    for h in (5, 20):
        col, out_col = f"Alpha_{h}d", f"rank_{h}d"
        mask = df["eligible"] & df[col].notna()
        sub = df.loc[mask, ["Date", col]]
        r = sub.groupby("Date")[col].rank(method="average") - 1.0
        n = sub.groupby("Date")[col].transform("count")
        df[out_col] = np.nan
        df.loc[mask, out_col] = np.where(n > 1, r / (n - 1.0) - 0.5, np.nan)
        df[out_col] = df[out_col].astype(np.float32)

    df.to_parquet(BASE_PATH, index=False, row_group_size=ROW_GROUP)

    # ── 健檢（規則 7：數值明確輸出）──
    elig = df[df["eligible"]]
    per_day = elig.groupby("Date").size()
    tr = elig[(elig["Date"] >= PROTOCOL["TRAIN_START"]) & (elig["Date"] <= PROTOCOL["TEST_END"])]
    print("=" * 70, flush=True)
    print(f"[健檢] base matrix：{len(df):,} 列 × {df.shape[1]} 欄 | "
          f"{df['stock_id'].nunique()} 支 | {df['Date'].nunique()} 個交易日", flush=True)
    print(f"[健檢] eligible 列數：{len(elig):,}（{len(elig)/len(df):.1%}）| "
          f"每日 eligible 檔數 min/median/max = {per_day.min()}/{int(per_day.median())}/{per_day.max()}",
          flush=True)
    print(f"[健檢] 協定窗內（2012–2026-06）eligible：{len(tr):,} 列 | "
          f"rank_5d 非空 {tr['rank_5d'].notna().sum():,} | rank_20d 非空 {tr['rank_20d'].notna().sum():,}",
          flush=True)
    r5 = tr["rank_5d"].dropna()
    print(f"[健檢] rank_5d 分布：min={r5.min():+.3f} mean={r5.mean():+.4f} max={r5.max():+.3f}"
          f"（應 ≈ -0.5 / 0 / +0.5）", flush=True)
    feat_nan = int(df[FEATURE_COLS].isna().sum().sum())
    print(f"[健檢] 特徵 NaN 總數：{feat_nan}（應為 0）| 耗時 {(time.time()-t0)/60:.1f} 分", flush=True)


# ============================================================
# 2) 衍生特徵（lag / rolling / momentum，241 維，分 part 檔）
# ============================================================
def build_derived(force: bool = False) -> None:
    parts = _derived_parts()
    if all(p.exists() for p, _ in parts) and not force:
        logger.info("derived parts 已存在，跳過")
        return
    t0 = time.time()
    base = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id"] + FEATURE_COLS)
    keys = base[["Date", "stock_id"]]
    # (stock_id, Date) 排序視圖（穩定排序保留原 index，算完 sort_index 還原 canonical 行序）
    srt_idx = base[["stock_id", "Date"]].sort_values(["stock_id", "Date"], kind="mergesort").index
    f = base.loc[srt_idx, FEATURE_COLS]
    gid = base.loc[srt_idx, "stock_id"].to_numpy()
    del base
    gc.collect()

    # ── lag parts ──
    for n in LAGS:
        out = CACHE_DIR / f"baseline_derived_lag{n}.parquet"
        if out.exists() and not force:
            print(f"[derived] lag{n} 已存在，跳過", flush=True)
            continue
        tc = time.time()
        lag = f.groupby(gid, sort=False).shift(n)
        lag.columns = lag_names(n)
        lag = lag.sort_index().astype(np.float32)
        lag = pd.concat([keys, lag.reset_index(drop=True)], axis=1)
        lag.to_parquet(out, index=False, row_group_size=ROW_GROUP)
        nan_pct = float(lag[lag_names(n)[0]].isna().mean())
        print(f"[derived] lag{n}：{lag.shape[0]:,} × {len(lag_names(n))} "
              f"(首欄 NaN {nan_pct:.2%}，應≈各股前 {n} 列) ({time.time()-tc:.0f}s)", flush=True)
        del lag
        gc.collect()

    del f, gid
    gc.collect()

    # ── rolling + momentum part ──
    build_derived_roll(keys, force=force)

    print(f"[derived] 完成，總耗時 {(time.time()-t0)/60:.1f} 分", flush=True)


def build_derived_roll(keys: pd.DataFrame | None = None, force: bool = False) -> None:
    """
    rolling（*_rmean / *_rstd）+ momentum 特徵。

    ⚠️ 2026-07-27 修正——特徵計算順序（檢查表 G4）
    ------------------------------------------------
    舊版是從 BASE_PATH 讀特徵再做 rolling，但 BASE_PATH 已經過 clean_and_scale
    （實測最後一日 Close 欄 mean=-0.0000、std=1.0000 = 已橫斷面 z-score），
    等於「先橫斷面標準化、再算時序特徵」——正是檢查表 G4 點名的順序顛倒。
    後果不是 look-ahead，而是語意錯亂：`Close_rstd20` 量到的是「橫斷面排名的波動」
    而非價格波動，且各日 z-score 的尺度不同還被平均在一起。

    現在改成從 CHUNK_DIR 的 chunk 檔取值——那是 build_base_matrix 在
    clean_and_scale **之前** 寫出的原始特徵——先在各股自己的時序上做 rolling，
    再逐日 winsorize + z-score，與同函式中 Mom_* 的既有正確作法一致。

    lag 特徵（*_lag1/5/20）刻意不改：純位移不是時序聚合，不會混到不同日的尺度，
    「該股 N 日前的橫斷面排名」本身就是合理且可解釋的特徵。
    """
    out = CACHE_DIR / "baseline_derived_roll.parquet"
    if out.exists() and not force:
        print("[derived] roll 已存在，跳過（要套用 G4 修正請加 --force）", flush=True)
        return
    tc = time.time()

    if keys is None:
        keys = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id"])

    # ── 來源：clean_and_scale 之前的 chunk 檔（原始值）──
    chunk_files = sorted(CHUNK_DIR.glob("base_chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(
            f"找不到 {CHUNK_DIR}/base_chunk_*.parquet。rolling 特徵必須建在 "
            f"clean_and_scale 之前的原始值上，請先跑 --build 重新產生 chunk。"
        )
    src = pd.concat(
        [pd.read_parquet(p, columns=["Date", "stock_id"] + ROLL_CORE) for p in chunk_files],
        ignore_index=True,
    )
    src["Date"] = pd.to_datetime(src["Date"])
    src = src.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)
    print(f"[derived] roll 來源（chunk 原始值）：{len(src):,} 列 × {len(ROLL_CORE)} 欄",
          flush=True)

    gid = src["stock_id"].to_numpy()
    grp = src[ROLL_CORE].groupby(gid, sort=False)

    # 逐窗口計算並立刻降成 float32，算完就釋放，避免同時存在多份 float64 中間物
    cols: dict[str, np.ndarray] = {}
    for w in ROLL_MEAN_WINDOWS:
        rm = grp.rolling(w, min_periods=w).mean()
        rm.index = rm.index.droplevel(0)
        rm = rm.sort_index()
        for c in ROLL_CORE:
            cols[f"{c}_rmean{w}"] = rm[c].to_numpy(np.float32)
        del rm
        gc.collect()
    for w in ROLL_STD_WINDOWS:
        rs = grp.rolling(w, min_periods=w).std()
        rs.index = rs.index.droplevel(0)
        rs = rs.sort_index()
        for c in ROLL_CORE:
            cols[f"{c}_rstd{w}"] = rs[c].to_numpy(np.float32)
        del rs
        gc.collect()
    del grp
    gc.collect()

    roll_src = pd.DataFrame(cols, index=src.index)
    roll_src.insert(0, "stock_id", src["stock_id"].to_numpy())
    roll_src.insert(0, "Date", src["Date"].to_numpy())
    del cols, src, gid
    gc.collect()

    # ── 對齊回 base matrix 的 canonical 行序（chunk 是 base 的超集：含 clean 掉的列）──
    _n0 = len(roll_src)
    roll_src = roll_src.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    if len(roll_src) < _n0:
        print(f"[derived] ⚠️ chunk 內有 {_n0 - len(roll_src):,} 列 (Date, stock_id) 重複，"
              f"已取最後一筆（否則 merge 會列膨脹）", flush=True)
    roll = keys.merge(roll_src, on=["Date", "stock_id"], how="left")
    assert len(roll) == len(keys), f"對齊後列數 {len(roll)} != base {len(keys)}"
    del roll_src
    gc.collect()

    # ── 逐日 winsorize + z-score（與 clean_and_scale / Mom_* 同慣例）──
    rolled_names = [f"{c}_rmean{w}" for c in ROLL_CORE for w in ROLL_MEAN_WINDOWS] \
                 + [f"{c}_rstd{w}" for c in ROLL_CORE for w in ROLL_STD_WINDOWS]
    for c in rolled_names:
        s = roll.groupby("Date")[c].transform(
            lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
        roll[c] = ((s - s.groupby(roll["Date"]).transform("mean"))
                   / (s.groupby(roll["Date"]).transform("std") + 1e-9)).astype(np.float32)
        del s
    gc.collect()

    # ── momentum：原始（還原）收盤價的累積報酬 → 每日橫斷面 winsorize + z-score ──
    pr = _load_raw("prices_raw")
    pr = pr[pr["stock_id"].astype(str).str.match(r"^\d{4}$")]
    pr = pr.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    pr = pr.sort_values(["stock_id", "Date"], kind="mergesort")
    g = pr.groupby("stock_id", sort=False)["Close"]
    for w in MOM_WINDOWS:
        pr[f"Mom_{w}d"] = g.shift(0) / g.shift(w) - 1.0
    mom_cols = [f"Mom_{w}d" for w in MOM_WINDOWS]
    merged = keys.merge(pr[["Date", "stock_id"] + mom_cols], on=["Date", "stock_id"], how="left")
    for c in mom_cols:
        merged[c] = merged.groupby("Date")[c].transform(
            lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
        roll[c] = merged.groupby("Date")[c].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)
    del pr, merged
    gc.collect()

    roll = roll[["Date", "stock_id"] + roll_names()]        # 固定欄序
    roll.to_parquet(out, index=False, row_group_size=ROW_GROUP)

    # 健檢（規則 7：數值明確輸出）
    smp = roll["Close_rmean20"].dropna()
    print(f"[derived] roll+mom：{roll.shape[0]:,} × {len(roll_names())} "
          f"({time.time()-tc:.0f}s)", flush=True)
    print(f"[derived] G4 修正健檢 — Close_rmean20（rolling 建於原始價、再橫斷面標準化）："
          f"mean={smp.mean():+.4f} std={smp.std():.4f} 非空 {len(smp):,} 列"
          f"（應 ≈ 0 / 1，且不再等同『z-score 的移動平均』）", flush=True)
    del roll
    gc.collect()


# ============================================================
# 3) 載入 X / y（訓練與評估用）
# ============================================================
def load_xy(date_from: str, date_to: str, day_stride: int = 1,
            with_derived: bool = True) -> dict:
    """回傳 dict：X (np.float32, n×300 或 n×59)、rank_5d/rank_20d/alpha_5d/alpha_20d、dates、stock_ids。
    只回傳 eligible 列；label NaN 列保留（各模型自行 mask）。
    day_stride=k：每 k 個交易日取一天（訓練抽樣；5d 重疊窗冗餘高，k=2 資訊損失小）。"""
    filt = [("Date", ">=", pd.Timestamp(date_from)), ("Date", "<=", pd.Timestamp(date_to))]
    base = pd.read_parquet(BASE_PATH, filters=filt)
    mask = base["eligible"].to_numpy()
    if day_stride > 1:
        days = np.sort(base["Date"].unique())
        keep_days = set(days[::day_stride])
        mask &= base["Date"].isin(keep_days).to_numpy()

    out = {
        "dates":     base.loc[mask, "Date"].to_numpy(),
        "stock_ids": base.loc[mask, "stock_id"].to_numpy(),
        "rank_5d":   base.loc[mask, "rank_5d"].to_numpy(np.float32),
        "rank_20d":  base.loc[mask, "rank_20d"].to_numpy(np.float32),
        "alpha_5d":  base.loc[mask, "Alpha_5d"].to_numpy(np.float32),
        "alpha_20d": base.loc[mask, "Alpha_20d"].to_numpy(np.float32),
    }
    blocks = [base.loc[mask, FEATURE_COLS].to_numpy(np.float32)]
    ref_keys = base.loc[mask, ["Date", "stock_id"]].reset_index(drop=True)
    del base
    gc.collect()

    if with_derived:
        for path, names in _derived_parts():
            part = pd.read_parquet(path, filters=filt)
            assert len(part) == len(mask), f"{path.name} 列數 {len(part)} != base {len(mask)}"
            pk = part.loc[mask, ["Date", "stock_id"]].reset_index(drop=True)
            if not (pk["Date"].equals(ref_keys["Date"]) and pk["stock_id"].equals(ref_keys["stock_id"])):
                raise AssertionError(f"{path.name} 行序與 base 不一致")
            blocks.append(part.loc[mask, names].to_numpy(np.float32))
            del part
            gc.collect()

    X = np.hstack(blocks)
    del blocks
    gc.collect()
    np.nan_to_num(X, copy=False)                      # 衍生特徵前段 NaN → 0（= 橫斷面均值，同 clean 慣例）
    out["X"] = X
    out["feature_names"] = all_feature_names() if with_derived else list(FEATURE_COLS)
    return out


# ============================================================
# 4) 評估工具（三階 baseline 共用）
# ============================================================
def daily_spearman_ic(dates: np.ndarray, scores: np.ndarray, realized: np.ndarray) -> pd.Series:
    """每日 Spearman IC（預測分數 vs 實際 Alpha）。realized NaN 的列自動剔除。"""
    df = pd.DataFrame({"Date": dates, "s": scores, "r": realized}).dropna(subset=["r"])
    def _ic(g):
        if len(g) < 30 or g["s"].nunique() < 2:
            return np.nan
        return g["s"].corr(g["r"], method="spearman")
    return df.groupby("Date").apply(_ic, include_groups=False).dropna()


def newey_west_t(x: np.ndarray, lag: int) -> float:
    """mean(x) 的 Newey-West t 值（Bartlett kernel，lag = horizon，處理重疊窗自相關）。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < lag + 2:
        return float("nan")
    e = x - x.mean()
    lrv = float(e @ e) / n
    for l in range(1, lag + 1):
        gamma = float(e[l:] @ e[:-l]) / n
        lrv += 2.0 * (1.0 - l / (lag + 1.0)) * gamma
    return float(x.mean() / np.sqrt(lrv / n))


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    m, s = float(ic.mean()), float(ic.std())
    return {
        "n_days":  int(len(ic)),
        "mean_ic": round(m, 4),
        "ic_std":  round(s, 4),
        "icir":    round(m / s, 3) if s > 0 else None,
        "pct_pos": round(float((ic > 0).mean()), 3),
        "t_naive": round(m / (s / np.sqrt(len(ic))), 2) if s > 0 else None,
        "t_newey_west": round(newey_west_t(ic.to_numpy(), lag=horizon), 2),
    }


def _load_close_pivot(date_from: str, date_to: str) -> pd.DataFrame:
    pr = _load_raw("prices_raw")
    pr = pr[(pr["Date"] >= pd.Timestamp(date_from)) & (pr["Date"] <= pd.Timestamp(date_to))]
    pr = pr[pr["stock_id"].astype(str).str.match(r"^\d{4}$")]
    pr = pr.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    px = pr.pivot(index="Date", columns="stock_id", values="Close").sort_index()
    return px.where(px > 0)                            # Close ≤ 0（停牌等髒資料）一律視為缺值


def _load_twii(date_from: str, date_to: str) -> pd.Series | None:
    m = _load_raw("macro_raw")
    if m is None:
        return None
    col = next((c for c in ("TWII", "TWII_Close") if c in m.columns), None)
    if col is None:
        return None
    s = m.drop_duplicates(subset=["Date"], keep="last").set_index("Date")[col].sort_index()
    return s[(s.index >= pd.Timestamp(date_from)) & (s.index <= pd.Timestamp(date_to))].dropna()


def portfolio_backtest(dates: np.ndarray, stock_ids: np.ndarray, scores: np.ndarray,
                       top_n: int = PROTOCOL["TOP_N"],
                       rebalance_days: int = PROTOCOL["REBALANCE_DAYS"]) -> dict:
    """協定 §7 組合層：Top-N 等權、每 rebalance_days 個交易日再平衡、收盤價成交、
    成本買 0.15% / 賣 0.45%。報酬用 prices_raw 真實收盤價（不可用 z-score 後的 Close）。"""
    sig = pd.DataFrame({"Date": dates, "stock_id": stock_ids, "score": scores}).dropna()
    trade_days = np.sort(sig["Date"].unique())
    px = _load_close_pivot(str(pd.Timestamp(trade_days[0]).date()),
                           str((pd.Timestamp(trade_days[-1]) + pd.Timedelta(days=20)).date()))
    px = px[px.index.isin(trade_days) | (px.index > trade_days[-1])]

    reb_dates = trade_days[::rebalance_days]
    daily_ret, held_prev, turnovers, n_missing = [], set(), [], 0
    for k, d in enumerate(reb_dates):
        top = sig[sig["Date"] == d].nlargest(top_n, "score")["stock_id"].tolist()
        avail = [s for s in top if s in px.columns and not np.isnan(px.loc[d, s])]
        n_missing += len(top) - len(avail)
        if not avail:
            continue
        # 持有窗：d 收盤買進 → 下一個再平衡日收盤（或最後一天）
        end = reb_dates[k + 1] if k + 1 < len(reb_dates) else px.index[px.index >= d][-1]
        win = px.loc[(px.index >= d) & (px.index <= end), avail]
        win = win.ffill()
        rel = win / win.iloc[0]                        # 等權買進後的價格相對值
        vpath = rel.mean(axis=1)
        rets = vpath.pct_change().dropna()

        cur = set(avail)
        sell_frac = len(held_prev - cur) / max(len(held_prev), 1) if held_prev else 0.0
        buy_frac = len(cur - held_prev) / len(cur)
        cost = buy_frac * PROTOCOL["COST_BUY"] + sell_frac * PROTOCOL["COST_SELL"]
        turnovers.append(buy_frac)
        if len(rets) > 0:
            rets.iloc[0] -= cost
        held_prev = cur
        daily_ret.append(rets)

    r = pd.concat(daily_ret).sort_index()
    r = r.groupby(r.index).sum()                       # 邊界日只會屬於一個窗，防重複保險
    n_bad = int((~np.isfinite(r)).sum())
    r = r[np.isfinite(r)]
    cum = (1 + r).cumprod()
    n = len(r)
    ann_ret = float(cum.iloc[-1] ** (252 / n) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else None
    mdd = float((cum / cum.cummax() - 1).min())

    out = {
        "n_days": n, "n_rebalances": len(reb_dates),
        "n_bad_return_days": n_bad,
        "ann_return": round(ann_ret, 4),
        "ann_sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown": round(mdd, 4),
        "avg_turnover_per_rebalance": round(float(np.mean(turnovers)), 3),
        "total_return": round(float(cum.iloc[-1] - 1), 4),
        "n_price_missing": int(n_missing),
    }
    twii = _load_twii(str(pd.Timestamp(trade_days[0]).date()), str(pd.Timestamp(trade_days[-1]).date()))
    if twii is not None and len(twii) > 20:
        tr = twii.pct_change().dropna()
        common = r.index.intersection(tr.index)        # macro_raw 停在 2026-04-24 → 超額只算到該日
        if len(common) > 20:
            pr_c, tw_c = (1 + r[common]).prod(), (1 + tr[common]).prod()
            out["excess_vs_twii"] = round(float(pr_c - tw_c), 4)
            out["excess_window"] = f"{common[0].date()} ~ {common[-1].date()}（macro TWII 覆蓋範圍）"
    return out


# ============================================================
# 5) 序列張量介面（階段二 2d，見 planing/資料基礎升級計畫_baseline_common扶正.md）
# ============================================================
# 懶加載設計，比照 production TemporalCrossSectionDataset：不材料化整個 (N, 252, 59)
# （若 N~200 萬筆 eligible 列全展開約需 118GB，本機 24GB RAM 裝不下），改成 __getitem__
# 時才動態切片。索引清單直接沿用 BASE_PATH 的 eligible==True 列，跟階段三-1 的 flat
# baseline（Ridge/GBDT/GRU）用同一批樣本，模型階梯之間才可比。

SEQ_LEN = 252  # 與 production marketmamba.config.SEQ_LEN 一致


class BaselineSequenceDataset:
    """
    __getitem__(i) 回傳 dict：
      X            : (SEQ_LEN, 59) float32 — 結束於該筆索引日期（含當日）的最近 SEQ_LEN
                     個交易日特徵；不足 SEQ_LEN 天者前面補 0
      padding_mask : (SEQ_LEN,) bool — True=真實資料、False=補 0 的 padding 位置
      rank_5d / rank_20d / stock_id / date
    """

    def __init__(self, date_from: str, date_to: str, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        cols = ["Date", "stock_id", "eligible", "rank_5d", "rank_20d"] + FEATURE_COLS
        base = pd.read_parquet(BASE_PATH, columns=cols)
        base = base.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)

        # 每支股票在自己時序中的位置（= feats 陣列的 row index），向量化計算
        base["_pos"] = base.groupby("stock_id", sort=False).cumcount()

        # per-stock 特徵陣列，供 __getitem__ O(1) 切片（groupby 順序與上面 sort 一致）
        self._feat_by_stock: dict[str, np.ndarray] = {
            sid: g[FEATURE_COLS].to_numpy(np.float32)
            for sid, g in base.groupby("stock_id", sort=False)
        }

        mask = (
            base["eligible"]
            & (base["Date"] >= pd.Timestamp(date_from))
            & (base["Date"] <= pd.Timestamp(date_to))
        )
        self.index = base.loc[
            mask, ["Date", "stock_id", "rank_5d", "rank_20d", "_pos"]
        ].reset_index(drop=True)

        n_dims = len(FEATURE_COLS)
        mem_gb = sum(a.nbytes for a in self._feat_by_stock.values()) / 1e9
        print(
            f"[BaselineSequenceDataset] {len(self.index):,} 筆索引 | "
            f"{self.index['stock_id'].nunique()} 支股票 | seq_len={seq_len} | "
            f"{self.index['Date'].min().date()} ~ {self.index['Date'].max().date()} | "
            f"per-stock 特徵陣列常駐記憶體 {mem_gb:.2f} GB", flush=True,
        )
        del base
        gc.collect()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        row = self.index.iloc[i]
        sid, pos = row["stock_id"], int(row["_pos"])
        feats = self._feat_by_stock[sid]
        lo = max(0, pos - self.seq_len + 1)
        window = feats[lo: pos + 1]                       # (actual_len, 59)
        actual_len = window.shape[0]

        X = np.zeros((self.seq_len, len(FEATURE_COLS)), dtype=np.float32)
        padding_mask = np.zeros(self.seq_len, dtype=bool)
        X[self.seq_len - actual_len:] = window
        padding_mask[self.seq_len - actual_len:] = True

        return {
            "X": X,
            "padding_mask": padding_mask,
            "rank_5d": np.float32(row["rank_5d"]),
            "rank_20d": np.float32(row["rank_20d"]),
            "stock_id": sid,
            "date": row["Date"],
        }


def _check_sequence_dataset(date_from: str, date_to: str, n_sample: int = 200) -> None:
    """驗證用（規則 7：數值明確輸出）：抽樣檢查 shape、padding 比例，並跟 flat base
    matrix 的對應列數值逐欄比對（序列最後一天必須完全等於 flat 版本該列的值）。"""
    ds = BaselineSequenceDataset(date_from, date_to)
    if len(ds) == 0:
        print(f"[check_sequence] {date_from}~{date_to} 區間內沒有 eligible 樣本"
              "（可能早於 202 天門檻累積完成的時間點），無法抽樣", flush=True)
        return
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(ds), size=min(n_sample, len(ds)), replace=False)

    flat = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id"] + FEATURE_COLS)
    flat = flat.set_index(["stock_id", "Date"])

    pad_ratios, max_abs_diff, shape_ok = [], 0.0, True
    for idx in sample_idx:
        item = ds[int(idx)]
        if item["X"].shape != (ds.seq_len, len(FEATURE_COLS)):
            shape_ok = False
        pad_ratios.append(item["padding_mask"].mean())
        flat_row = flat.loc[(item["stock_id"], pd.Timestamp(item["date"]))][FEATURE_COLS].to_numpy(np.float32)
        diff = float(np.max(np.abs(item["X"][-1] - flat_row)))
        max_abs_diff = max(max_abs_diff, diff)

    print(
        f"[check_sequence] 抽樣 {len(sample_idx)} 筆 | shape 正確={shape_ok} | "
        f"padding_mask 真實資料佔比 min/median/max = "
        f"{min(pad_ratios):.3f}/{np.median(pad_ratios):.3f}/{max(pad_ratios):.3f} | "
        f"序列末日 vs flat base matrix 逐欄最大絕對差 = {max_abs_diff:.2e}（應為 0）",
        flush=True,
    )


# ============================================================
# 6) KG 邊介面（階段二 2e）
# ============================================================
# 直接重用 production 現成的 build_kg_csr()/get_batch_edges_csr()
# （marketmamba/models/trainer.py——只 import 不修改；此路徑不在規則 5 保護範圍內，
# 規則 5 保護的是 V6/models/ 下的 checkpoint，不是 marketmamba/models/ 原始碼）。
# 短線/趨勢實驗模型已在用同一套，這裡只是讓 baseline_common 的使用者不必重複串接。

_KG_CSR_CACHE: dict = {}


def _get_kg_csr():
    if "csr" not in _KG_CSR_CACHE:
        from marketmamba.models.trainer import build_kg_csr
        kg_csr, stock_to_idx = build_kg_csr()
        _KG_CSR_CACHE["csr"] = kg_csr
        _KG_CSR_CACHE["idx"] = stock_to_idx
    return _KG_CSR_CACHE["csr"], _KG_CSR_CACHE["idx"]


def load_kg_edges_for_stocks(stock_ids: list[str], device: str = "cpu"):
    """回傳 (edge_index, edge_attr)：stock_ids 對應的 KG 子圖，local index 為
    stock_ids 的位置（0-based），跟 production get_batch_edges_csr 語意一致。"""
    import torch
    from marketmamba.models.trainer import get_batch_edges_csr

    kg_csr, stock_to_idx = _get_kg_csr()
    edge_index, edge_attr = get_batch_edges_csr(stock_ids, kg_csr, stock_to_idx, torch.device(device))
    n_covered = sum(1 for s in stock_ids if s in stock_to_idx)
    print(
        f"[KG] {len(stock_ids)} 支股票 → KG 覆蓋 {n_covered} 支 "
        f"({n_covered / max(len(stock_ids), 1):.1%}) | 子圖邊數 {edge_index.shape[1]:,}",
        flush=True,
    )
    return edge_index, edge_attr


def _check_kg_interface(date_from: str, date_to: str, n_sample: int = 1) -> None:
    """驗證用：從 base matrix 抽一天的 eligible 股票清單，實際跑一次子圖擷取。"""
    base = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id", "eligible"])
    days = sorted(base.loc[base["eligible"], "Date"].unique())[-n_sample:]
    for d in days:
        stocks = base.loc[(base["Date"] == d) & base["eligible"], "stock_id"].tolist()
        load_kg_edges_for_stocks(stocks, device="cpu")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="建 base matrix + derived features 快取")
    ap.add_argument("--force", action="store_true", help="忽略既有快取重建")
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--rebuild-roll", action="store_true",
                    help="只重建 rolling/momentum part（套用 2026-07-27 的 G4 順序修正）。"
                         "base matrix 與 lag part 不動，chunk 檔必須還在。")
    ap.add_argument("--check-sequence", action="store_true", help="驗證 2d：序列張量介面健檢")
    ap.add_argument("--check-kg", action="store_true", help="驗證 2e：KG 邊介面健檢")
    args = ap.parse_args()
    if args.build:
        build_base_matrix(n_chunks=args.chunks, force=args.force)
        build_derived(force=args.force)
        print("✅ baseline 快取建構完成：", CACHE_DIR, flush=True)
    elif args.rebuild_roll:
        build_derived_roll(force=True)
        print("✅ roll part 已依 G4 修正重建：", CACHE_DIR, flush=True)
    elif args.check_sequence:
        _check_sequence_dataset(PROTOCOL["TEST_START"], PROTOCOL["TEST_END"])
    elif args.check_kg:
        _check_kg_interface(PROTOCOL["TEST_START"], PROTOCOL["TEST_END"])
    else:
        ap.print_help()
