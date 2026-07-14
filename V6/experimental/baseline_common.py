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
  python V6/experimental/baseline_common.py --build     # 建 base matrix + derived（一次性，~1hr）
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

    # ── rolling + momentum part ──
    out = CACHE_DIR / "baseline_derived_roll.parquet"
    if not out.exists() or force:
        tc = time.time()
        grp = f[ROLL_CORE].groupby(gid, sort=False)
        cols = {}
        for w in ROLL_MEAN_WINDOWS:
            rm = grp.rolling(w, min_periods=w).mean()
            rm.index = rm.index.droplevel(0)
            for c in ROLL_CORE:
                cols[f"{c}_rmean{w}"] = rm[c]
            del rm
        for w in ROLL_STD_WINDOWS:
            rs = grp.rolling(w, min_periods=w).std()
            rs.index = rs.index.droplevel(0)
            for c in ROLL_CORE:
                cols[f"{c}_rstd{w}"] = rs[c]
            del rs
        roll = pd.DataFrame(cols).sort_index().astype(np.float32)
        del cols
        gc.collect()

        # momentum：原始（還原）收盤價的過去累積報酬 → 每日橫斷面 winsorize + z-score（同 clean_and_scale 慣例）
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
            merged[c] = merged.groupby("Date")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
            roll[c] = merged[c].astype(np.float32).to_numpy()
        del pr, merged
        gc.collect()

        roll = roll[roll_names()]                      # 固定欄序
        roll = pd.concat([keys, roll.reset_index(drop=True)], axis=1)
        roll.to_parquet(out, index=False, row_group_size=ROW_GROUP)
        print(f"[derived] roll+mom：{roll.shape[0]:,} × {len(roll_names())} ({time.time()-tc:.0f}s)",
              flush=True)
        del roll
        gc.collect()

    print(f"[derived] 完成，總耗時 {(time.time()-t0)/60:.1f} 分", flush=True)


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
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="建 base matrix + derived features 快取")
    ap.add_argument("--force", action="store_true", help="忽略既有快取重建")
    ap.add_argument("--chunks", type=int, default=5)
    args = ap.parse_args()
    if args.build:
        build_base_matrix(n_chunks=args.chunks, force=args.force)
        build_derived(force=args.force)
        print("✅ baseline 快取建構完成：", CACHE_DIR, flush=True)
    else:
        ap.print_help()
