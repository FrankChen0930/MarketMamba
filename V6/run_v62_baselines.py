"""
run_v62_baselines.py — B 類經典模型每日推論（Ridge / GBDT / GRU）
================================================================
與 `run_v62_inference.py`（Mamba）**完全分開的獨立行程**，理由是 config patch
是 module 級全域：Mamba 線切 59 維、baseline 線切 66 維（`MM_PROTOCOL=v2`），
同一個 process 混跑會**靜默**吃到錯的設定（CLAUDE.md 記過 47 維那次的坑）。

為什麼不能直接用 `baseline_common.load_xy()`
--------------------------------------------
那支讀的是全歷史快取（`baseline_base_66d.parquet` + 4 個共 5.8 GB 的衍生檔），
建於 2026-07-30、**不會自己往前長**。每日推論只能建「尾端窗」。

尾端窗的三個已量測落差（2026-08-08，`experimental/diag_window_panel.py`）
------------------------------------------------------------------------
拿兩個不同長度的窗互比（同資料、同一天，唯一變因＝窗長）：

  ① **51/54 個非 macro 欄完全相同**（ρ=1.000000、max|Δ|=0）→ 尾端窗本身沒問題
  ② `Dividend_Yield_Fwd` / `Securities_Balance` / `Avail_Securities` 會偏
     → 那三欄依賴被 trim 掉的歷史。**解法：非價格 raw 一律不 trim**（見 `_RAW_TRIM`）
  ③ **12 個 macro 欄全部偏很大**（`Oil_Return` max|Δ|=2.27、`TNX` 1.07、`VIX` 0.87）
     → `clean_and_scale(macro_norm="ts")` 的 expanding 統計量是算在**傳進去的
     日期範圍**上。解法：全歷史算一次那 12 欄、按 Date 貼回（`_splice_macro`），
     且**必須用 `feature_engineer.macro_ts_zscore`**，不可在這裡複製那五行。

  ⚠️ Mamba 那條線量不到 ③——它的上線 arm 把 Group D 整個歸零。
     Ridge/GBDT/GRU 吃全部 66 維，躲不掉。

三個模型的輸入契約**不一樣**
----------------------------
  Ridge / GBDT : 307 維扁平（66 base + 59×3 lag + 60 rolling + 4 momentum）
  GRU          : (B, 60, 59) 序列 —— 只要 base 的 59 欄（66 扣掉 7 個 Avail_ 旗標），
                 **完全不需要衍生特徵**

可重現性現況（2026-08-08 實測，見 `experimental/compare_scores.py`）
------------------------------------------------------------------
  Ridge : ✅ 重建 vs 參考 ρ=0.9970、Top50 重疊 47/50 → 沿用既有基準
  GRU   : ✅ 不需重訓，`gru_5d__p30_s20260713.pt` 就是產出參考分數的那一顆
  GBDT  : ⚠️ **要分兩層講，不可簡化成「不可重現」**
            訊號層 ❌：ρ=0.9203、Top50 重疊 25/50。不是設定丟失——我自己兩次跑
                      （只差 purge 30 vs 60 天）彼此也才 ρ=0.9434、重疊 28/50。
                      樹的切點是離散決策，訓練窗動 1% 就翻，151 輪 boosting 再放大。
            組合層 ✅：11.0% vs 參考 11.2%、Sharpe 0.653 vs 0.639、換手 79% vs 81%
                      （decile spread Sharpe 1.714 vs 1.806）。
            → **換進來的那半個 Top50 與被換掉的一樣好。** 所以「每天的持股名單
              與參考模型對不上」是事實，但「這是不同的策略」不成立。
              基準用重建模型自己量的 11.0%，不借用參考值。

⚠️ 執行環境：**WSL**（GRU 需要 torch，而本機 Windows 的 torch DLL 已損壞）。
   pandas 3.0 的唯讀 `to_numpy()` 問題已於 2026-08-08 修掉（portfolio_lab:286、
   f5_r_series:248），兩邊數字實測一致（582 天 replay：Windows 38.02% = WSL 38.02%）。

用法
----
    MM_PROTOCOL=v2 python V6/run_v62_baselines.py                 # 全部三個
    MM_PROTOCOL=v2 python V6/run_v62_baselines.py --arms ridge
    MM_PROTOCOL=v2 python V6/run_v62_baselines.py --verify-date 2026-03-02
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

if os.environ.get("MM_PROTOCOL") != "v2":
    raise SystemExit("❌ 請設 MM_PROTOCOL=v2 再跑（baseline 協定是 66 維，"
                     "與 Mamba 線的 59 維不同，混用會靜默算錯）")

from experimental import baseline_common as B                      # noqa: E402
from marketmamba.data.feature_engineer import (                    # noqa: E402
    build_features, clean_and_scale, macro_ts_zscore,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v62_baselines")

RESULT_DIR = _V6 / "experimental" / "result"
SCORE_DIR = RESULT_DIR / "scores"
RESULTS_DIR = _V6 / "results"

# 窗長：需涵蓋 eligible 門檻 202 天 + GRU 60 天視窗 + rolling 60 + lag 20 + 緩衝。
# 用曆日算 → 900 曆日 ≈ 610 個交易日，餘裕充足。
LOOKBACK_DAYS = 900

# 只 trim 價格；其餘 raw **一律不 trim**（落差 ② 的解法）。
# 代價是多吃記憶體，但 trim 掉的歷史正是 Dividend_Yield_Fwd / Securities_Balance
# 需要的東西——「省記憶體」不值得用「靜默算錯三欄」去換。
_RAW_TRIM = False

VERIFY_MIN_RHO, VERIFY_MIN_OVERLAP = 0.95, 40


@dataclass(frozen=True)
class BaseArm:
    kind:      str            # ridge / gbdt / gru
    artifact:  str            # 權重檔（相對 RESULT_DIR）
    ref_score: str | None     # 對照分數檔（SCORE_DIR）
    out_name:  str
    trust:     str            # reproduced / new_model —— 前端與紀錄要能區分
    note:      str = ""


ARMS: dict[str, BaseArm] = {
    "ridge": BaseArm("ridge", "ridge_5d__p30_rebuild20260808.npz",
                     "ridge__lab5d_p30.parquet", "df_v62_ridge", "reproduced",
                     note="重建 vs 參考 ρ=0.9970／重疊 47/50"),
    "gbdt":  BaseArm("gbdt", "gbdt_5d__p30fix_20260808.txt",
                     "gbdt__p30fix_20260808.parquet", "df_v62_gbdt", "new_model",
                     note="⚠️ 無法重現八模型表那格（ρ=0.9203）→ 另立基準"),
    "gru":   BaseArm("gru", "gru_5d__p30_s20260713.pt",
                     "gru__p30_s20260713.parquet", "df_v62_gru", "reproduced",
                     note="checkpoint 即產出參考分數的那一顆，未重訓"),
}


# ============================================================
# 1. 尾端窗面板
# ============================================================
def _splice_macro(df: pd.DataFrame) -> pd.DataFrame:
    """把 12 個 Group D 欄換成**全歷史** expanding z-score 的版本。

    落差 ③ 的解法。macro 是每日橫斷面常數，所以只需要「每日一列」的序列——
    不需要任何個股列，成本極低（約 5,700 列）。

    ⚠️ 用 `feature_engineer.macro_ts_zscore`，**不在這裡複製那五行**。
       2026-08-08 為此把它從 `clean_and_scale` 抽出來（逐位元回歸通過）。
    """
    import marketmamba.config as _c
    macro_cols = [c for c in _c.FEATURE_GROUPS["macro_environment"] if c in df.columns]
    if not macro_cols:
        logger.warning("[macro] df 沒有 Group D 欄位 → 跳過貼回")
        return df

    # 全歷史的原始 macro（clean_and_scale 之前的值）：從 chunk 檔取，
    # 那是 build_base_matrix 在標準化之前寫出的。沒有 chunk 就退回不貼、但要講明白。
    chunks = sorted(B.CHUNK_DIR.glob("base_chunk_*.parquet"))
    if not chunks:
        logger.warning(f"[macro] 找不到 {B.CHUNK_DIR}/base_chunk_*.parquet → "
                       f"**沿用尾端窗自算的 macro**（12 欄會偏，"
                       f"實測 Oil_Return max|Δ| 可達 2.27）")
        return df

    # 每個 chunk 都含全部日期、且 macro 每日同值 → 讀第一個 chunk 就夠
    src = pd.read_parquet(chunks[0], columns=["Date"] + macro_cols)
    src["Date"] = pd.to_datetime(src["Date"])
    per_day = src.groupby("Date")[macro_cols].first().sort_index()
    del src
    gc.collect()

    before = {c: float(df[c].iloc[0]) for c in macro_cols[:3]}
    for c in macro_cols:
        z = macro_ts_zscore(per_day[c])
        df[c] = df["Date"].map(z).astype(np.float32)
    after = {c: float(df[c].iloc[0]) for c in macro_cols[:3]}

    cov = float(df[macro_cols[0]].notna().mean())
    logger.info(f"[macro] 全歷史貼回 {len(macro_cols)} 欄"
                f"（來源 {len(per_day):,} 個交易日：{per_day.index.min().date()} → "
                f"{per_day.index.max().date()}）｜覆蓋率 {cov:.1%}")
    # 規則 7：貼回前後的實際數值要看得見，否則「有做但看不出來」等同沒做
    for c in list(before):
        logger.info(f"[macro]   {c:<22s} 窗內自算 {before[c]:+.4f} → 全歷史 {after[c]:+.4f}")
    if cov < 0.99:
        logger.warning(f"[macro] ⚠️ 覆蓋率只有 {cov:.1%}——推論日可能超出 chunk 的日期範圍，"
                       f"那幾天的 macro 會是 NaN（下游 nan→0）")
    return df


def build_panel(target_date: str | None = None,
                lookback_days: int = LOOKBACK_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回 (cleaned, raw)：cleaned 是 clean_and_scale 之後（base + lag 用），
    raw 是之前的原始值（rolling 用，G4 順序修正的要求）。

    逐步對齊 `baseline_common.build_base_matrix()`，唯一差別是日期範圍。
    """
    t0 = time.time()
    prices = B._load_raw("prices_raw")
    prices = B._filter_universe(prices)
    prices = prices.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    end = pd.Timestamp(target_date) if target_date else prices["Date"].max()
    cutoff = end - pd.Timedelta(days=lookback_days)
    prices = prices[(prices["Date"] >= cutoff) & (prices["Date"] <= end)].copy()
    stocks = sorted(prices["stock_id"].unique())
    logger.info(f"[panel] 價格窗 {len(prices):,} 列｜{len(stocks)} 支｜"
                f"{prices['Date'].min().date()} → {prices['Date'].max().date()}")

    def _trim(d):
        if d is None or not _RAW_TRIM or "Date" not in getattr(d, "columns", []):
            return d
        d = d.copy()
        d["Date"] = pd.to_datetime(d["Date"])
        return d[(d["Date"] >= cutoff) & (d["Date"] <= end)]

    stock_kw = {k: _trim(B._load_raw(v, stock_ids=stocks)) for k, v in B._STOCK_RAWS.items()}
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

    raw = df[["Date", "stock_id"] + list(B.ROLL_CORE)].copy()   # rolling 的來源（未標準化）

    cleaned = clean_and_scale(df, macro_norm="ts",
                              neutralize=B.PROTOCOL.get("NEUTRALIZE", "none"))
    cleaned = cleaned.sort_values(["Date", "stock_id"], kind="mergesort").reset_index(drop=True)
    cleaned["eligible"] = cleaned.groupby("stock_id", sort=False).cumcount() >= (
        B.PROTOCOL["MIN_HISTORY_DAYS"] - 1)
    cleaned = _splice_macro(cleaned)

    logger.info(f"[panel] cleaned {len(cleaned):,} 列｜eligible "
                f"{int(cleaned['eligible'].sum()):,}｜{(time.time()-t0)/60:.1f} 分")
    return cleaned, raw


# ============================================================
# 2. 衍生特徵（307 維扁平；Ridge / GBDT 用）
# ============================================================
def build_flat_X(cleaned: pd.DataFrame, raw: pd.DataFrame,
                 date: str) -> tuple[np.ndarray, pd.DataFrame]:
    """回 (X, keys)：X 是 `date` 當日的 307 維矩陣，欄序 = `all_feature_names()`。

    三段各自的來源**逐行照抄 `baseline_common.build_derived*`**：
      lag      ← cleaned（刻意建在標準化後的值上，見 build_derived_roll docstring）
      rolling  ← raw（clean 之前的原始值）→ 再逐日 winsorize + z-score（G4）
      momentum ← 還原收盤價的累積報酬 → 逐日 winsorize + z-score
    """
    names = B.all_feature_names()
    d = pd.Timestamp(date)

    # ── lag：整窗算完再取當日 ──
    src = cleaned.sort_values(["stock_id", "Date"], kind="mergesort")
    lag_src = [c for c in B.FEATURE_COLS if c not in B._NO_LAG_COLS]
    gid = src["stock_id"].to_numpy()
    f = src[lag_src]
    lag_blocks = {}
    for n in B.LAGS:
        lg = f.groupby(gid, sort=False).shift(n)
        lg.columns = B.lag_names(n)
        lag_blocks[n] = lg.set_index([src["Date"].to_numpy(), src["stock_id"].to_numpy()])
    del f, gid
    gc.collect()

    # ── rolling：在 raw 上做，再逐日標準化 ──
    r = raw.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)
    rg = r[list(B.ROLL_CORE)].groupby(r["stock_id"].to_numpy(), sort=False)
    cols: dict[str, np.ndarray] = {}
    for w in B.ROLL_MEAN_WINDOWS:
        rm = rg.rolling(w, min_periods=w).mean()
        rm.index = rm.index.droplevel(0)
        rm = rm.sort_index()
        for c in B.ROLL_CORE:
            cols[f"{c}_rmean{w}"] = rm[c].to_numpy(np.float32)
        del rm
    for w in B.ROLL_STD_WINDOWS:
        rs = rg.rolling(w, min_periods=w).std()
        rs.index = rs.index.droplevel(0)
        rs = rs.sort_index()
        for c in B.ROLL_CORE:
            cols[f"{c}_rstd{w}"] = rs[c].to_numpy(np.float32)
        del rs
    del rg
    gc.collect()
    roll = pd.DataFrame(cols, index=r.index)
    roll.insert(0, "stock_id", r["stock_id"].to_numpy())
    roll.insert(0, "Date", r["Date"].to_numpy())
    del cols, r
    gc.collect()

    # ── 當日 keys（只保留 eligible，與 load_xy 一致）──
    day = cleaned[(cleaned["Date"] == d) & cleaned["eligible"]].copy()
    if day.empty:
        raise SystemExit(f"❌ {date} 沒有 eligible 列（該日不存在，或歷史不足 202 天）")
    keys = day[["Date", "stock_id"]].reset_index(drop=True)

    rolled_names = [f"{c}_rmean{w}" for c in B.ROLL_CORE for w in B.ROLL_MEAN_WINDOWS] \
                 + [f"{c}_rstd{w}" for c in B.ROLL_CORE for w in B.ROLL_STD_WINDOWS]
    roll = roll.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    # ⚠️ winsorize + z-score 必須在**當日完整橫斷面**上做，不能只對 eligible 子集，
    #    否則分母（分位數與 std）與訓練時不同 → 每一維都會偏且不報錯。
    rday = roll[roll["Date"] == d].copy()
    for c in rolled_names:
        s = rday[c].clip(lower=rday[c].quantile(0.01), upper=rday[c].quantile(0.99))
        rday[c] = ((s - s.mean()) / (s.std() + 1e-9)).astype(np.float32)
    del roll
    gc.collect()

    # ── momentum ──
    pr = B._load_raw("prices_raw")
    pr = B._filter_universe(pr)
    pr = pr.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    pr = pr.sort_values(["stock_id", "Date"], kind="mergesort")
    g = pr.groupby("stock_id", sort=False)["Close"]
    mom_cols = [f"Mom_{w}d" for w in B.MOM_WINDOWS]
    for w in B.MOM_WINDOWS:
        pr[f"Mom_{w}d"] = g.shift(0) / g.shift(w) - 1.0
    pday = pr[pr["Date"] == d][["Date", "stock_id"] + mom_cols].copy()
    for c in mom_cols:
        s = pday[c].clip(lower=pday[c].quantile(0.01), upper=pday[c].quantile(0.99))
        pday[c] = ((s - s.mean()) / (s.std() + 1e-9)).astype(np.float32)
    del pr
    gc.collect()

    # ── 組裝：欄序必須完全等於 all_feature_names() ──
    out = keys.copy()
    base_day = day[["Date", "stock_id"] + list(B.FEATURE_COLS)]
    out = out.merge(base_day, on=["Date", "stock_id"], how="left")
    for n in B.LAGS:
        lb = lag_blocks[n]
        lb.index.names = ["Date", "stock_id"]
        lb = lb.reset_index()
        lb = lb[lb["Date"] == d]
        out = out.merge(lb, on=["Date", "stock_id"], how="left")
    out = out.merge(rday[["Date", "stock_id"] + rolled_names], on=["Date", "stock_id"], how="left")
    out = out.merge(pday, on=["Date", "stock_id"], how="left")

    missing = [c for c in names if c not in out.columns]
    if missing:
        raise SystemExit(f"❌ 缺少 {len(missing)} 個特徵欄，前 5 個：{missing[:5]}")
    X = out[names].to_numpy(np.float32).copy()      # .copy()：pandas 3.0 唯讀
    np.nan_to_num(X, copy=False)
    logger.info(f"[flat] {date}：{X.shape[0]} 支 × {X.shape[1]} 維"
                f"（應為 {len(names)}）｜NaN 已補 0")
    return X, keys


# ============================================================
# 3. 三個推論路徑
# ============================================================
def predict_ridge(X: np.ndarray, arm: BaseArm) -> np.ndarray:
    z = np.load(RESULT_DIR / arm.artifact, allow_pickle=True)
    w, c = z["w"], float(z["intercept"])
    if len(w) != X.shape[1]:
        raise SystemExit(f"❌ 權重 {len(w)} 維 != X {X.shape[1]} 維")
    logger.info(f"[ridge] α={float(z['alpha']):.0e}｜purge={int(z['purge_days'])}"
                f"｜train_end={str(z['train_end'])}")
    return (X.astype(np.float64) @ w + c).astype(np.float32)


def predict_gbdt(X: np.ndarray, arm: BaseArm) -> np.ndarray:
    import lightgbm as lgb
    bst = lgb.Booster(model_file=str(RESULT_DIR / arm.artifact))
    if bst.num_feature() != X.shape[1]:
        raise SystemExit(f"❌ 模型 {bst.num_feature()} 維 != X {X.shape[1]} 維")
    logger.info(f"[gbdt] {bst.num_trees()} 棵樹｜{bst.num_feature()} 維")
    return bst.predict(X).astype(np.float32)


def predict_gru(cleaned: pd.DataFrame, date: str, arm: BaseArm) -> tuple[np.ndarray, pd.DataFrame]:
    """GRU 吃 (B, 60, 59) 序列，**不用衍生特徵**。逐行對齊 `baseline_rnn.Panel`。"""
    import torch
    import torch.nn as nn

    WINDOW = 60
    feats = [c for c in B.FEATURE_COLS if not c.startswith("Avail_")]
    src = cleaned.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)
    F = np.ascontiguousarray(src[feats].to_numpy(np.float32))
    dates = src["Date"].to_numpy()
    sids = src["stock_id"].to_numpy()
    cum = src.groupby("stock_id", sort=False).cumcount().to_numpy()
    window_ok = cum >= (WINDOW - 1)

    m = (dates == np.datetime64(pd.Timestamp(date))) & src["eligible"].to_numpy() & window_ok
    rows = np.flatnonzero(m)
    if len(rows) == 0:
        raise SystemExit(f"❌ {date} 沒有可用列（需 eligible 且股內 ≥{WINDOW} 列）")
    offs = rows[:, None] + np.arange(-(WINDOW - 1), 1)[None, :]
    Xs = F[offs]                                     # (B, 60, 59)
    logger.info(f"[gru] {Xs.shape[0]} 支 × {Xs.shape[1]} 步 × {Xs.shape[2]} 維")

    class RNNReg(nn.Module):
        def __init__(self, hidden=64, layers=2):
            super().__init__()
            self.rnn = nn.GRU(len(feats), hidden, num_layers=layers,
                              batch_first=True, dropout=0.2)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1]).squeeze(-1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(RESULT_DIR / arm.artifact, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "state" in sd:
        sd = sd["state"]
    model = RNNReg().to(dev)
    model.load_state_dict(sd)          # strict=True：架構不符當場失敗
    model.eval()
    with torch.no_grad():
        s = model(torch.from_numpy(Xs).to(dev)).float().cpu().numpy()
    keys = pd.DataFrame({"Date": pd.Timestamp(date), "stock_id": sids[rows]})
    return s.astype(np.float32), keys


# ============================================================
# 4. 驗證 / 主流程
# ============================================================
def verify(out: pd.DataFrame, date: str, arm: BaseArm) -> bool:
    """對既有參考分數比同一天的 ρ 與 Top50 重疊。判準在檔頭，跑之前定死。"""
    if not arm.ref_score:
        logger.warning("[驗證] 此 arm 無對照分數檔")
        return False
    p = SCORE_DIR / arm.ref_score
    if not p.exists():
        logger.warning(f"[驗證] 找不到對照 {p}")
        return False
    ref = pd.read_parquet(p)
    ref["Date"] = pd.to_datetime(ref["Date"]).dt.strftime("%Y-%m-%d")
    ref = ref[ref["Date"] == date]
    if ref.empty:
        logger.warning(f"[驗證] 對照檔沒有 {date}")
        return False
    ref["stock_id"] = ref["stock_id"].astype(str)
    mg = out.merge(ref[["stock_id", "score"]], on="stock_id",
                   how="inner", suffixes=("_new", "_ref"))
    rho = float(mg["score_new"].corr(mg["score_ref"], method="spearman"))
    ov = len(set(out.nlargest(50, "score")["stock_id"]) &
             set(ref.nlargest(50, "score")["stock_id"]))
    ok = rho >= VERIFY_MIN_RHO and ov >= VERIFY_MIN_OVERLAP
    print(f"\n{'='*70}\n[驗證] {arm.kind}｜{date}｜新 {len(out)} 支 vs 參考 {len(ref)} 支"
          f"｜交集 {len(mg)}\n"
          f"[驗證] Spearman ρ = {rho:.4f}（判準 ≥{VERIFY_MIN_RHO}）"
          f"｜Top50 重疊 = {ov}/50（判準 ≥{VERIFY_MIN_OVERLAP}）\n"
          f"[驗證] {'✅ 通過' if ok else '❌ 未過'}｜trust={arm.trust}\n{'='*70}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None, help="預設全部")
    ap.add_argument("--date", default=None)
    ap.add_argument("--verify-date", default=None, help="驗證模式：對既有分數比對")
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    arms = a.arms or list(ARMS)
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        return logger.error(f"未知 arm：{unknown}（可用 {list(ARMS)}）") or 2

    target = a.verify_date or a.date
    cleaned, raw = build_panel(target, a.lookback)
    date = target or cleaned["Date"].max().strftime("%Y-%m-%d")

    need_flat = any(ARMS[x].kind in ("ridge", "gbdt") for x in arms)
    X, keys = (build_flat_X(cleaned, raw, date) if need_flat else (None, None))

    ok_all, failed = True, []
    for name in arms:
        arm = ARMS[name]
        try:
            if arm.kind == "gru":
                s, k = predict_gru(cleaned, date, arm)
            else:
                s = predict_ridge(X, arm) if arm.kind == "ridge" else predict_gbdt(X, arm)
                k = keys
            out = pd.DataFrame({"stock_id": k["stock_id"].astype(str), "Date": date,
                                "score": s})
            out = out.sort_values("score", ascending=False).reset_index(drop=True)
            out["rank"] = np.arange(1, len(out) + 1)
            logger.info(f"[{name}] {len(out)} 支｜min {out['score'].min():+.4f}"
                        f"｜median {out['score'].median():+.4f}｜max {out['score'].max():+.4f}")
            if a.verify_date:
                ok_all &= verify(out, date, arm)
            elif not a.no_save:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                out.to_csv(RESULTS_DIR / f"{arm.out_name}.csv", index=False)
                arch = RESULTS_DIR / "archive"
                arch.mkdir(parents=True, exist_ok=True)
                out.to_csv(arch / f"{arm.out_name}_{date}.csv", index=False)
                logger.info(f"[{name}] ✅ → {arm.out_name}.csv")
        except Exception as e:                                  # noqa: BLE001
            import traceback
            failed.append(name)
            logger.error(f"[{name}] 失敗：{e}\n{traceback.format_exc()[:1500]}")

    if failed:
        logger.error(f"❌ 失敗的 arm：{failed}")
        return 1
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
