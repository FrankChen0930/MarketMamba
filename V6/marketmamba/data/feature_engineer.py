"""
MarketMamba V6 — Feature Engineer
===================================
Builds the 46-dimensional pure-quant feature matrix from raw price, institutional,
margin, fundamental, and macro data.

Feature groups (must match config.FEATURE_GROUPS exactly):
  Group A — price_momentum      (12 dims): OHLCV, returns, MAs, RSI, ATR
  Group B — institutional_flow  (16 dims): 3 institutional, margin, KD, OBV, vol
  Group C — fundamentals        (10 dims): revenue, EPS, valuation, profitability
  Group D — macro_environment   ( 8 dims): index returns, VIX, rates, commodities
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from marketmamba.config import FEATURE_COLS, FEATURE_GROUPS, PROCESSED_DIR, SEQ_LEN
from marketmamba.data.feature_spec import AVAIL_COLS, NEUTRALIZE_EXCLUDE

logger = logging.getLogger(__name__)


# ============================================================
# 可得性旗標 helper（V6.3 / 決策1）
# ============================================================

def _mark_avail(df: pd.DataFrame, flag: str, observed: "pd.Series | bool") -> pd.DataFrame:
    """
    寫入一個可得性旗標欄。

    語意：1 = 該來源對這筆 (日期, 股票) 有**真實觀測**（可能是 ffill 帶下來的，
    但源頭確實存在過）；0 = 純粹捏造的填補值。詳見 `feature_spec.py` 的說明。

    呼叫時機一律是「ffill 之後、fillna(0) 之前」——那個瞬間的 notna 才是正確語意。
    """
    if isinstance(observed, bool):
        df[flag] = np.float32(1.0 if observed else 0.0)
    else:
        df[flag] = observed.fillna(False).to_numpy(dtype="float32")
    return df


# ============================================================
# Main Entry Point
# ============================================================

def build_features(
    df_price:         pd.DataFrame,
    df_inst:          pd.DataFrame | None = None,
    df_margin:        pd.DataFrame | None = None,
    df_per:           pd.DataFrame | None = None,
    df_securities:    pd.DataFrame | None = None,
    df_market_value:  pd.DataFrame | None = None,
    df_daytrade:      pd.DataFrame | None = None,
    df_holdings:      pd.DataFrame | None = None,
    df_rev:           pd.DataFrame | None = None,
    df_fin:           pd.DataFrame | None = None,
    df_balance_sheet: pd.DataFrame | None = None,
    df_cashflow:      pd.DataFrame | None = None,
    df_macro:         pd.DataFrame | None = None,
    # V6.1 new data sources
    df_futures_inst:  pd.DataFrame | None = None,
    df_options_inst:  pd.DataFrame | None = None,
    df_dividend:      pd.DataFrame | None = None,
    df_foreign_shareholding: pd.DataFrame | None = None,
    df_fear_greed:    pd.DataFrame | None = None,
    df_business_indicator: pd.DataFrame | None = None,
    df_fed_rate:      pd.DataFrame | None = None,
    fundamentals_v2:  bool = False,
    availability_flags: bool = False,
) -> pd.DataFrame:
    """
    Merge all raw data sources and compute the 59-dim feature matrix (V6.2)
    或 67 維（V6.3，`availability_flags=True` 且 config 已 patch 成 67 維時）。

    Args:
        availability_flags:
          False（預設，V6.1/V6.2 行為）：不產生可得性旗標，輸出維度不變。
          True（V6.3 起）：額外產生 8 個 `Avail_*` 旗標欄，把「捏造的 0」與
            「真實的 0」分開（決策1）。必須搭配
            `feature_spec.patch_config_67d()`，且該 patch 要在 import
            `marketmamba.models.*` 之前執行。
          ⚠️ 若 config 已是 67 維但這裡傳 False，旗標欄會走 build_features 尾端的
             「缺欄補 0」路徑而變成整欄 0——那等於告訴模型「所有資料一律不可得」。
             故下方有顯式檢查會擋下這個組合。

        fundamentals_v2:
          False（預設，V6.1 行為）：維持現行基本面語意，與線上 v6_best.pt / v6_short.pt /
            v6_trend.pt 的訓練資料一致。
          True（V6.2 重訓起）：修正兩個 2026-07-27 稽核找到的基本面 bug——
            ① EPS_Surprise 改在季頻上算 pct_change(4)（= 去年同季），
               舊行為套在日頻列上、實際只比 4 個交易日；
            ② Q4/年報的 available_from 從 +45 天改 +90 天（法定申報期限為次年 3/31），
               舊行為每年 2/14–3/31 之間存在 45 天的年報 look-ahead 窗。
          ⚠️ 兩者都會改變 2005 年起所有歷史特徵值。推論端必須等對應的 V6.2 checkpoint
             上線後才可切換，提前切換等於注入訓練/推論不一致（同 D1 macro_norm 的坑）。

    Returns:
        df : MultiIndex [Date, stock_id] with all 59 feature columns + target columns
    """
    _cfg_has_flags = any(c in FEATURE_COLS for c in AVAIL_COLS)
    if _cfg_has_flags and not availability_flags:
        raise ValueError(
            "config 已 patch 成含 Avail_* 的 67 維，但 build_features(availability_flags=False)。"
            "這個組合會讓旗標欄走尾端『缺欄補 0』路徑、整欄變成 0，"
            "等於對模型宣告「所有資料一律不可得」，而且不會有任何錯誤訊息。"
            "請傳 availability_flags=True。"
        )
    if availability_flags and not _cfg_has_flags:
        logger.warning(
            "availability_flags=True 但 config 尚未 patch 成 67 維——"
            "旗標會被算出來，但 build_features 尾端重排欄位時會被丟掉。"
            "請先呼叫 feature_spec.patch_config_67d()。"
        )
    logger.info(
        f"Building feature matrix ({len(FEATURE_COLS)}D, "
        f"fundamentals_v2={fundamentals_v2}, availability_flags={availability_flags})..."
    )

    df = df_price.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["stock_id", "Date"]).reset_index(drop=True)

    # -- Group A: Price / Momentum --
    df = _add_price_momentum_features(df)

    # -- Group B: Institutional / Margin / Technical --
    df = _merge_institutional(df, df_inst, availability_flags=availability_flags)
    df = _merge_margin(df, df_margin, availability_flags=availability_flags)
    df = _add_technical_b_features(df)
    df = _merge_daytrade(df, df_daytrade, availability_flags=availability_flags)
    df = _merge_holdings(df, df_holdings, availability_flags=availability_flags)
    df = _merge_securities(df, df_securities, availability_flags=availability_flags)
    df = _merge_foreign_shareholding(df, df_foreign_shareholding,
                                     availability_flags=availability_flags)

    # -- Group C: Fundamentals --
    df = _merge_per_pbr(df, df_per, fundamentals_v2=fundamentals_v2)
    df = _merge_market_value_feature(df, df_market_value)
    df = _merge_fundamentals(df, df_rev, df_fin, df_balance_sheet,
                             fundamentals_v2=fundamentals_v2)
    # 必須在 _merge_fundamentals 之後：推算 PER/PBR 需要它 as-of join 進來的
    # EPS_TTM 與 Book_Value（PIT 保護就是靠那個 join，不可提前）。
    df = _derive_valuation_fallback(df, fundamentals_v2=fundamentals_v2)
    df = _merge_dividend_feature(df, df_dividend)                # V6.1
    df = _add_free_cash_flow(df, df_cashflow,
                             fundamentals_v2=fundamentals_v2)    # V6.1

    # Group C 的兩個旗標必須等所有 Group C 來源都合併完才算：
    # PER/PBR 還會被 _derive_valuation_fallback 用自算值補上（B-1），
    # 在個別 _merge_* 裡算會低估實際可得性。
    if availability_flags:
        df = _add_group_c_avail_flags(df)

    # -- Group D: Macro --
    df = _merge_macro(df, df_macro, df_fear_greed, df_business_indicator,
                      df_fed_rate, df_futures_inst, df_options_inst)  # V6.1 expanded

    # -- Group A addition: RS relative-strength vs TWII (V6.2) --
    # Must run after _merge_macro so TWII_Return_5d/20d/60d are available.
    df = _add_rs_features(df)

    # -- Targets: 5d / 20d / 60d Alpha vs TWII --
    df = _add_alpha_targets(df, df_macro)

    # -- Sanity check --
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns (will be filled with 0): {missing_cols}")
        for c in missing_cols:
            df[c] = 0.0

    # Reorder to match FEATURE_COLS order
    meta_cols = ["Date", "stock_id", "Alpha_5d", "Alpha_10d", "Alpha_20d", "Alpha_60d"]
    df = df[meta_cols + FEATURE_COLS].copy()

    logger.info(f"Feature matrix: {df.shape[0]:,} rows × {len(FEATURE_COLS)} features")
    return df


# ============================================================
# Group A — Price / Momentum (12 dims)
# ============================================================

def _add_price_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stock_id", sort=False)

    # Returns
    df["Return_1d"]  = g["Close"].pct_change(1)
    df["Return_5d"]  = g["Close"].pct_change(5)
    df["Return_20d"] = g["Close"].pct_change(20)

    # Moving Averages
    df["MA_20"] = g["Close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["MA_60"] = g["Close"].transform(lambda x: x.rolling(60, min_periods=30).mean())

    # RSI (14-day)
    df["RSI_14"] = g["Close"].transform(_compute_rsi)

    # ATR (14-day True Range)
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - g["Close"].shift(1)).abs(),
        (df["Low"]  - g["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR_14"] = g["TR"].transform(lambda x: x.ewm(span=14, min_periods=7).mean())
    df.drop(columns=["TR"], inplace=True)

    # Volatility_20d (log-return std)
    log_ret = g["Close"].transform(lambda x: np.log(x / x.shift(1)))
    df["Volatility_20d"] = log_ret.groupby(df["stock_id"]).transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )

    return df


# ============================================================
# Group A Addition — RS Relative Strength vs TWII (V6.2)
# ============================================================

def _add_rs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RS_5d, RS_20d, RS_60d = stock return - TWII return over the same window.

    Prerequisites (both must already be in df before this function is called):
      - Return_5d, Return_20d: computed by _add_price_momentum_features()
      - TWII_Return_5d, TWII_Return_20d, TWII_Return_60d: merged by _merge_macro()

    Return_60d is not in FEATURE_COLS so it is computed here as a temporary column.
    """
    g = df.groupby("stock_id", sort=False)

    # Return_60d: temporary intermediate, not added to FEATURE_COLS
    tmp_return_60d = g["Close"].pct_change(60)

    twii_5d  = df.get("TWII_Return_5d",  pd.Series(0.0, index=df.index))
    twii_20d = df.get("TWII_Return_20d", pd.Series(0.0, index=df.index))
    twii_60d = df.get("TWII_Return_60d", pd.Series(0.0, index=df.index))

    df["RS_5d"]  = df["Return_5d"]  - twii_5d
    df["RS_20d"] = df["Return_20d"] - twii_20d
    df["RS_60d"] = tmp_return_60d   - twii_60d

    # Fill forward within stock then fill remaining NaN with 0
    for col in ["RS_5d", "RS_20d", "RS_60d"]:
        df[col] = df.groupby("stock_id")[col].transform(lambda x: x.ffill().fillna(0.0))

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# Group B — Institutional / Technical (16 dims)
# ============================================================

def _merge_institutional(df: pd.DataFrame, df_inst: pd.DataFrame,
                         availability_flags: bool = False) -> pd.DataFrame:
    if df_inst is None or df_inst.empty:
        for col in ["Foreign_Buy", "Foreign_Sell", "Foreign_Net",
                    "Investment_Trust_Net", "Dealer_Net"]:
            df[col] = 0.0
        if availability_flags:
            df = _mark_avail(df, "Avail_Institutional", False)
        return df

    df_inst = df_inst.copy()
    df_inst["Date"] = pd.to_datetime(df_inst["Date"])
    merge_cols = ["Foreign_Buy", "Foreign_Sell", "Foreign_Net",
                  "Investment_Trust_Net", "Dealer_Net"]
    df = df.merge(df_inst[["Date", "stock_id"] + merge_cols],
                  on=["Date", "stock_id"], how="left")
    if availability_flags:
        # Foreign_Net 是這組的代表欄；join 沒命中時五欄同時為 NaN
        df = _mark_avail(df, "Avail_Institutional", df["Foreign_Net"].notna())
    for col in merge_cols:
        df[col] = df[col].fillna(0.0)
    return df


def _merge_margin(df: pd.DataFrame, df_margin: pd.DataFrame,
                  availability_flags: bool = False) -> pd.DataFrame:
    """Merge margin purchase / short sale data.
    Columns in margin_raw.parquet are already renamed by fetch_v6_data.py."""
    EXPECTED = ["Margin_Purchase", "Margin_Repay", "Short_Sale",
                "Short_Cover", "Margin_Balance", "Short_Balance"]
    if df_margin is None or df_margin.empty:
        for col in EXPECTED:
            df[col] = 0.0
        if availability_flags:
            df = _mark_avail(df, "Avail_Margin", False)
        return df

    df_m = df_margin.copy()
    df_m["Date"] = pd.to_datetime(df_m["Date"])

    # Support BOTH already-renamed cols AND original FinMind names
    legacy_map = {
        "MarginPurchaseBuy":          "Margin_Purchase",
        "MarginPurchaseSell":         "Margin_Repay",
        "ShortSaleSell":              "Short_Sale",
        "ShortSaleBuy":               "Short_Cover",
        "MarginPurchaseTodayBalance": "Margin_Balance",
        "ShortSaleTodayBalance":      "Short_Balance",
    }
    df_m.rename(columns=legacy_map, inplace=True)
    valid = [c for c in EXPECTED if c in df_m.columns]

    df = df.merge(df_m[["Date", "stock_id"] + valid],
                  on=["Date", "stock_id"], how="left")
    if availability_flags:
        # 取 ffill 之後、fillna(0) 之前的 notna：代表「這支曾有過真實融資融券資料」，
        # 而不是「今天剛好有新公布」（後者只會反映公布頻率）
        _obs = (df.groupby("stock_id")[valid[0]].transform(lambda x: x.ffill()).notna()
                if valid else pd.Series(False, index=df.index))
        df = _mark_avail(df, "Avail_Margin", _obs)
    for col in valid:
        df[col] = df.groupby("stock_id")[col].transform(
            lambda x: x.ffill().fillna(0.0))
    for col in EXPECTED:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _add_technical_b_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add KD stochastic, OBV, and Day_Trade_Volume placeholder."""
    g = df.groupby("stock_id", sort=False)

    # KD Stochastic (9,3,3)
    low_9  = g["Low"].transform(lambda x: x.rolling(9, min_periods=5).min())
    high_9 = g["High"].transform(lambda x: x.rolling(9, min_periods=5).max())
    rsv = ((df["Close"] - low_9) / (high_9 - low_9 + 1e-9)) * 100
    df["KD_K"] = g["stock_id"].transform(lambda _: None)  # placeholder
    # Compute K/D iteratively per stock
    k_vals, d_vals = [], []
    for sid, sub in df.groupby("stock_id", sort=False):
        rsv_s = rsv.loc[sub.index]
        k = rsv_s.ewm(com=2, adjust=False).mean()  # smoothing factor 1/3
        d = k.ewm(com=2, adjust=False).mean()
        k_vals.append(k)
        d_vals.append(d)
    df["KD_K"] = pd.concat(k_vals).reindex(df.index)
    df["KD_D"] = pd.concat(d_vals).reindex(df.index)

    # OBV — On Balance Volume
    direction = np.sign(df["Return_1d"].fillna(0))
    df["OBV"] = (direction * df["Volume"]).groupby(df["stock_id"]).cumsum()
    # Normalize OBV per stock (z-score rolling)
    df["OBV"] = g["OBV"].transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) /
                  (x.rolling(60, min_periods=20).std() + 1e-9)
    )

    if "Day_Trade_Volume" not in df.columns:
        df["Day_Trade_Volume"] = 0.0
    return df


def _merge_daytrade(df: pd.DataFrame, df_daytrade: pd.DataFrame | None,
                    availability_flags: bool = False) -> pd.DataFrame:
    """Merge Day_Trade_Volume ratio from daytrade_raw.parquet."""
    if df_daytrade is None or df_daytrade.empty:
        if availability_flags:
            df = _mark_avail(df, "Avail_Daytrade", False)
        return df
    dt = df_daytrade[["Date", "stock_id", "Day_Trade_Volume"]].copy()
    dt["Date"] = pd.to_datetime(dt["Date"])
    dt["Day_Trade_Volume"] = pd.to_numeric(dt["Day_Trade_Volume"], errors="coerce").clip(0, 1)
    df = df.merge(dt, on=["Date", "stock_id"], how="left", suffixes=("", "_dt"))
    if "Day_Trade_Volume_dt" in df.columns:
        if availability_flags:
            # 必須看 _dt 欄：`Day_Trade_Volume` 已被 _add_technical_b_features
            # 預先建成整欄 0.0，直接對它取 notna 會恆為 True
            df = _mark_avail(df, "Avail_Daytrade", df["Day_Trade_Volume_dt"].notna())
        df["Day_Trade_Volume"] = df["Day_Trade_Volume_dt"].fillna(df["Day_Trade_Volume"])
        df.drop(columns=["Day_Trade_Volume_dt"], inplace=True)
    else:
        if availability_flags:
            df = _mark_avail(df, "Avail_Daytrade", df["Day_Trade_Volume"].notna())
        df["Day_Trade_Volume"] = df["Day_Trade_Volume"].fillna(0.0)
    return df


def _merge_per_pbr(df: pd.DataFrame, df_per: pd.DataFrame | None,
                   fundamentals_v2: bool = False) -> pd.DataFrame:
    """
    Merge PER/PBR/DY from per_raw.parquet (daily, direct join).

    fundamentals_v2 時額外產生 `PER__obs` / `PBR__obs` 兩個暫時欄位，記錄
    **ffill 之前**哪些列是真正觀測到的官方值。`_derive_valuation_fallback`
    需要這個資訊才能分辨「官方新值」與「凍結三個月的舊值」——後者應該被自算值
    取代，前者不應該。兩個暫時欄位由該函式用完即刪，不會進入特徵矩陣。
    """
    if df_per is None or df_per.empty:
        return df
    p = df_per.copy()
    p["Date"] = pd.to_datetime(p["Date"])
    for c in ["PER", "PBR", "DY", "dividend_yield"]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    if "dividend_yield" in p.columns and "DY" not in p.columns:
        p = p.rename(columns={"dividend_yield": "DY"})
    keep = ["Date", "stock_id"] + [c for c in ["PER", "PBR", "DY"] if c in p.columns]
    df = df.merge(p[keep], on=["Date", "stock_id"], how="left", suffixes=("", "_per"))
    for c in ["PER", "PBR"]:
        dup = c + "_per"
        if dup in df.columns:
            df[c] = df[dup].combine_first(df.get(c))
            df.drop(columns=[dup], inplace=True)
    # Forward-fill PER/PBR within each stock (not updated every day)
    for c in ["PER", "PBR"]:
        if c in df.columns:
            if fundamentals_v2:
                # ffill 之前先記錄「這一列是真的有官方值」
                df[c + "__obs"] = pd.to_numeric(df[c], errors="coerce").notna()
            df[c] = df.groupby("stock_id")[c].transform(lambda x: x.ffill())
    return df


def _derive_valuation_fallback(df: pd.DataFrame,
                               fundamentals_v2: bool = False) -> pd.DataFrame:
    """
    用已有資料自行推算 PER/PBR，補上 `per_raw` 涵蓋不到的股票（B-1，2026-07-28）。

    【為什麼做】per_raw 的新來源 TWSE BWIBBU_ALL **只涵蓋上市**（2026-07-27 當日
    1,080 支），而宇宙有 1,942 支——865 支缺 PER/PBR，其中絕大多數是上櫃。
    但這不是「找不到來源只好將就」：本益比與股價淨值比本來就能從已有的 EPS、
    權益、市值推導出來，自算是**更完整**的做法，不是替代品。

        PER = Close / EPS_TTM              （近四季 EPS 合計）
        PBR = market_value / Book_Value    （權益歸屬於母公司業主）

    PBR 刻意用「市值 ÷ 權益」而非「股價 ÷ 每股淨值」，可完全避開股數換算，
    因此對減資與現金增資造成的股數變動免疫（本專案已知 20 筆減資事件）。

    【PIT 保證】EPS_TTM 與 Book_Value 都是在季頻算好、再走
    `_merge_financial_statements` 的 as-of join 帶進來，各季受各自的
    `available_from` 保護（Q4 年報 +90 天、其餘 +45 天），不會提前使用未公告季度。

    【三層優先序】這是本函式最容易做錯的地方。初版只用 `combine_first` 填 NaN，
    結果 PER 只補了 308 列（+0.5%）幾乎沒效果——因為那 865 支**不是缺值**，
    而是被 `_merge_per_pbr` 的 ffill 凍結在 2026-04-24 的舊值一路延用至今。
    真正該做的是讓自算值**取代凍結值**：

        ① 官方當日觀測值（`PER__obs` 為 True）  ← 最高優先，絕不覆蓋
        ② 自算值                                ← 取代凍結的 ffill
        ③ 凍結的 ffill 官方值                    ← 兩者皆無時才留

    【交叉驗證只用 ①】用 ffill 值當分母會污染結果：實測全部列一起比時
    PER 比值 median 為 0.9498（看起來像有 5% 系統偏差），但那其實是
    「我們用當季 EPS、官方那欄卻凍結了三個月」造成的假象，不是公式錯。
    """
    obs_cols = [c for c in ("PER__obs", "PBR__obs") if c in df.columns]
    if not fundamentals_v2 or "Close" not in df.columns:
        if obs_cols:
            df = df.drop(columns=obs_cols)
        return df

    # ── PER 的分子必須是**當日的原始收盤價**，不是還原價 ─────────────
    #
    # 2026-07-30 實測：用 `Close`（官方還原價）算出來的 PER，
    # 與交易所公布的 PER 比值 median = **0.7653**，±10% 內只有 21.2%。
    # 根因不是水位差，是分子用錯價格——交易所的 PER 用**當日實際成交價**，
    # 而 `Close` 是以今天為基準、把歷史除息全部還原後的價格。
    #
    # 決定性證據：2011–2026 累積還原乘數 median = **0.7729**，與 0.7653 吻合到 1% 內。
    # 對照組：PBR 走 `market_value`（未還原市值）→ 比值 1.0016 完全正常。
    #
    # 語意上也是這樣才對：時點 t 的本益比，本來就該用 t 當時的實際股價。
    #
    # 還原公式 adjusted(t) = raw(t) × Π{f(e) : e > t}
    # → raw(t) = adjusted(t) / Π{f(e) : e > t}
    close_adj = pd.to_numeric(df["Close"], errors="coerce")
    close = _unadjusted_close(df, close_adj)

    # ── PER = 原始收盤價 / EPS_TTM ───────────────────────────────
    per_calc = None
    if "EPS_TTM" in df.columns:
        eps = pd.to_numeric(df["EPS_TTM"], errors="coerce")
        # 負值或近零的 EPS 算出來的本益比沒有經濟意義（交易所也不公布）
        per_calc = (close / eps.where(eps > 0)).replace([np.inf, -np.inf], np.nan)

    # ── PBR = market_value / Book_Value ─────────────────────────
    pbr_calc = None
    if "Book_Value" in df.columns and "Market_Cap_Log" in df.columns:
        bv = pd.to_numeric(df["Book_Value"], errors="coerce")
        mv = np.expm1(pd.to_numeric(df["Market_Cap_Log"], errors="coerce"))
        pbr_calc = (mv / bv.where(bv > 0)).replace([np.inf, -np.inf], np.nan)

    for name, calc, lo, hi in [("PER", per_calc, 0.0, 1000.0),
                               ("PBR", pbr_calc, 0.0, 100.0)]:
        if calc is None:
            logger.warning(f"[valuation_v2] {name} 缺推算原料，跳過")
            continue
        official = pd.to_numeric(df[name], errors="coerce") if name in df.columns \
            else pd.Series(np.nan, index=df.index)
        obs = (df[name + "__obs"].fillna(False).astype(bool)
               if name + "__obs" in df.columns
               else pd.Series(False, index=df.index))

        # ── 交叉驗證：只拿「官方當日觀測值」當分母 ──────────────
        both = obs & official.notna() & (official > 0) & calc.notna()
        if int(both.sum()) >= 100:
            r = (calc[both] / official[both]).replace([np.inf, -np.inf], np.nan).dropna()
            logger.info(
                f"[valuation_v2] {name} 交叉驗證（僅官方觀測列）n={len(r):,}"
                f"｜自算/官方 median {r.median():.4f}"
                f"｜±10% 內 {(r.sub(1).abs() <= 0.10).mean():.1%}"
                f"｜±25% 內 {(r.sub(1).abs() <= 0.25).mean():.1%}"
            )
        else:
            logger.warning(f"[valuation_v2] {name} 交叉驗證樣本不足（n={int(both.sum())}）")

        calc = calc.where((calc > lo) & (calc < hi))

        # ── 橫斷面校準：消掉自算與官方之間的系統性水位差 ────────
        # 實測 PER 的 自算/官方 median = 0.9494（僅用官方觀測列比對，n=36,079），
        # 是真實差異而非假象。原因未完全查明（候選：基本 vs 稀釋 EPS、
        # 交易所對股本變動的追溯調整），但**後果明確**：自算值與官方值
        # 共存於同一個橫斷面，5% 的水位差會讓「被自算的那群股票」整體偏移，
        # 製造出假的排名差異——而 PER/PBR 正是要拿來做橫斷面比較的特徵。
        #
        # 用「同一天同時有官方值與自算值」的股票算當日校準係數。
        # 只用當日資訊、不跨日、不看未來，與本專案其他 PIT 要求一致
        # （對照：全樣本算一個常數會用到 test 期資料，不可接受）。
        both_day = obs & official.notna() & (official > 0) & calc.notna()
        if int(both_day.sum()) >= 100:
            ratio_s = (calc / official).where(both_day)
            k = ratio_s.groupby(df["Date"]).transform("median")
            # 當日重疊樣本不足 → k 為 NaN；係數離譜 → 多半是當日資料有問題。
            # 兩種情況都退回 1.0（不校準），寧可保留已知的水位差也不亂調。
            k_ok = k.between(0.5, 2.0) & k.notna()
            n_cal = int((k_ok & calc.notna()).sum())
            calc = calc / k.where(k_ok, 1.0)
            logger.info(f"[valuation_v2] {name} 橫斷面校準：{n_cal:,} 列套用當日係數"
                        f"（係數 median {k[k_ok].median():.4f}，"
                        f"未校準 {int((~k_ok & calc.notna()).sum()):,} 列）")
        else:
            logger.warning(f"[valuation_v2] {name} 重疊樣本不足，未做校準")

        # ── 三層優先序：官方觀測 > 自算 > 凍結 ffill ────────────
        n_obs = int((obs & official.notna()).sum())
        n_stale = int(((~obs) & official.notna()).sum())
        n_replaced = int(((~obs) & official.notna() & calc.notna()).sum())
        n_filled = int((official.isna() & calc.notna()).sum())
        df[name] = official.where(obs).combine_first(calc).combine_first(official)
        logger.info(
            f"[valuation_v2] {name} 官方觀測 {n_obs:,} 列保留｜"
            f"凍結 ffill {n_stale:,} 列中 {n_replaced:,} 列改用自算"
            f"（{n_replaced / max(n_stale, 1):.1%}）｜"
            f"自算另補原本空值 {n_filled:,} 列｜"
            f"最終非空 {int(df[name].notna().sum()):,} / {len(df):,}"
        )
    if obs_cols:
        df = df.drop(columns=obs_cols)
    return df


def _unadjusted_close(df: pd.DataFrame, close_adj: pd.Series) -> pd.Series:
    """
    由官方還原價反推**當日的原始收盤價**（供 PER 分子使用，見呼叫處說明）。

        adjusted(t) = raw(t) × Π{ adj_factor(e) : e 為除權息/減資日, e > t }
        →  raw(t)   = adjusted(t) / Π{ ... }

    因子來源是 `ex_rights_raw.parquet`（TWSE TWT49U + TPEX exDailyQ + 減資恢復買賣表，
    26,385 筆 / 2,143 支），與當初重建還原價用的是同一份，所以能精確反推。

    取不到因子表時退回還原價並警告——寧可留下已知的偏差，也不要靜默用錯的值。
    """
    p = Path(PROCESSED_DIR) / "ex_rights_raw.parquet"
    if not p.exists():
        logger.warning("[valuation_v2] 找不到 ex_rights_raw，PER 分子退回還原價"
                       "（會有約 −23% 的系統性偏差）")
        return close_adj
    try:
        ex = pd.read_parquet(p)
    except Exception as e:                                        # noqa: BLE001
        logger.warning(f"[valuation_v2] ex_rights_raw 讀取失敗（{e}），PER 分子退回還原價")
        return close_adj

    dcol = next((c for c in ("date", "Date", "ex_date") if c in ex.columns), None)
    if dcol is None or "adj_factor" not in ex.columns:
        logger.warning("[valuation_v2] ex_rights_raw 欄位不符，PER 分子退回還原價")
        return close_adj

    ex = ex[["stock_id", dcol, "adj_factor"]].copy()
    ex["stock_id"] = ex["stock_id"].astype(str)
    ex[dcol] = pd.to_datetime(ex[dcol], errors="coerce")
    ex["adj_factor"] = pd.to_numeric(ex["adj_factor"], errors="coerce")
    ex = ex.dropna().sort_values(["stock_id", dcol])

    sid = df["stock_id"].astype(str).to_numpy()
    dts = pd.to_datetime(df["Date"]).to_numpy()
    mult = np.ones(len(df), dtype="float64")

    for s, grp in ex.groupby("stock_id", sort=False):
        m = sid == s
        if not m.any():
            continue
        e = grp[dcol].to_numpy()
        f = grp["adj_factor"].to_numpy()
        # suffix_prod[i] = f[i] * … * f[k-1]；長度 k+1，最後一項 1.0
        suffix = np.append(np.cumprod(f[::-1])[::-1], 1.0)
        # 事件日嚴格大於 t 的第一個位置
        idx = np.searchsorted(e, dts[m], side="right")
        mult[m] = suffix[idx]

    n_adj = int((mult != 1.0).sum())
    logger.info(f"[valuation_v2] PER 分子改用原始收盤價："
                f"{n_adj:,}/{len(df):,} 列套用還原乘數"
                f"（median {np.median(mult[mult != 1.0]) if n_adj else 1.0:.4f}）")
    return close_adj / pd.Series(mult, index=df.index)


def _add_group_c_avail_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group C 的兩個可得性旗標（V6.3 / 決策1）。

    刻意放在 build_features 的 Group C 全部合併完之後，而不是塞進個別 `_merge_*`：
      - `Avail_Valuation` 要等 `_derive_valuation_fallback` 把自算 PER/PBR 補完，
        否則會把「官方沒有但自算補上」的列誤標成不可得（2026 年上櫃股全是這種）
      - `Avail_Financials` 要等 `_merge_fundamentals` 的 as-of join 完成

    此時這些欄位仍是 NaN（最終的 fillna(0) 在 `clean_and_scale`），
    所以 `notna()` 正是「有真實觀測」的正確判準。
    """
    # `Avail_Valuation` 已於 2026-07-30 依全量結果移除
    # （訓練窗 mean 0.9966 / std 0.0585，自算 PER/PBR 幾乎補滿所有列 → 常數欄）。
    fin = pd.Series(False, index=df.index)
    for c in ("EPS", "Book_Value", "ROE", "Gross_Margin"):
        if c in df.columns:
            fin = fin | pd.to_numeric(df[c], errors="coerce").notna()
    df = _mark_avail(df, "Avail_Financials", fin)

    n = len(df)
    logger.info(f"[availability] Avail_Financials 可得 {int(fin.sum()):,}/{n:,}"
                f"（{fin.mean():.1%}）")
    return df


def _merge_market_value_feature(df: pd.DataFrame, df_mv: pd.DataFrame | None) -> pd.DataFrame:
    """Compute Market_Cap_Log from market_value_raw.parquet."""
    if df_mv is None or df_mv.empty:
        return df
    mv = df_mv[["Date", "stock_id", "market_value"]].copy()
    mv["Date"] = pd.to_datetime(mv["Date"])
    mv["market_value"] = pd.to_numeric(mv["market_value"], errors="coerce").clip(lower=0)
    mv["Market_Cap_Log"] = np.log1p(mv["market_value"])
    df = df.merge(mv[["Date", "stock_id", "Market_Cap_Log"]],
                  on=["Date", "stock_id"], how="left", suffixes=("", "_mv"))
    if "Market_Cap_Log_mv" in df.columns:
        df["Market_Cap_Log"] = df["Market_Cap_Log_mv"].combine_first(df.get("Market_Cap_Log"))
        df.drop(columns=["Market_Cap_Log_mv"], inplace=True)
    df["Market_Cap_Log"] = df.groupby("stock_id")["Market_Cap_Log"].transform(
        lambda x: x.ffill().fillna(0.0))
    return df


# ============================================================
# Group C — Fundamentals (10 dims)
# ============================================================

def _merge_fundamentals(
    df: pd.DataFrame,
    df_rev:          pd.DataFrame | None = None,
    df_fin:          pd.DataFrame | None = None,
    df_balance_sheet: pd.DataFrame | None = None,
    fundamentals_v2: bool = False,
) -> pd.DataFrame:
    """
    Merge monthly revenue and quarterly financial statements.
    Uses 'as-of' join to avoid look-ahead bias:
      - Revenue: published on 10th of following month → safe after that date
      - Financials: published ~45 days after quarter end → safe after that date
        (fundamentals_v2=True 時 Q4/年報改 +90 天，見 build_features docstring)
    """
    fund_defaults = {
        "PER": 15.0, "PBR": 1.5,
        "Revenue_MoM": 0.0, "Revenue_YoY": 0.0,
        "EPS": 0.0, "EPS_Surprise": 0.0,
        "Gross_Margin": 0.3, "ROE": 0.1,
        "Market_Cap_Log": 0.0, "Book_Value": 0.0,
    }
    for col, default in fund_defaults.items():
        if col not in df.columns:
            df[col] = default

    if df_rev is not None and not df_rev.empty:
        df = _merge_revenue(df, df_rev)

    if df_fin is not None and not df_fin.empty:
        df = _merge_financial_statements(df, df_fin, df_balance_sheet,
                                         fundamentals_v2=fundamentals_v2)

    return df


def _merge_revenue(df: pd.DataFrame, df_rev: pd.DataFrame) -> pd.DataFrame:
    df_rev = df_rev.copy()
    # Support both 'date' (raw FinMind) and 'Date' (already renamed by merger)
    date_col = "Date" if "Date" in df_rev.columns else "date"
    df_rev["date"] = pd.to_datetime(df_rev[date_col])
    # Revenue is published on the 10th of following month; safe to use from 11th onward
    df_rev["available_from"] = df_rev["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=11)

    df_rev = df_rev.sort_values(["stock_id", "date"])
    df_rev["Revenue_MoM"] = df_rev.groupby("stock_id")["revenue"].pct_change(1).fillna(0)
    df_rev["Revenue_YoY"] = df_rev.groupby("stock_id")["revenue"].pct_change(12).fillna(0)

    # As-of merge: for each (stock_id, Date) in df, use latest revenue available
    merged_rows = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_rev = df_rev[df_rev["stock_id"] == sid].sort_values("available_from")
        sub_df = sub_df.copy()
        sub_df["Revenue_MoM"] = _asof_lookup(sub_df["Date"], sub_rev["available_from"], sub_rev["Revenue_MoM"])
        sub_df["Revenue_YoY"] = _asof_lookup(sub_df["Date"], sub_rev["available_from"], sub_rev["Revenue_YoY"])
        merged_rows.append(sub_df)

    if merged_rows:
        df = pd.concat(merged_rows, ignore_index=True)
    return df


def _balance_sheet_equity(df_bs: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    從 balance_sheet_raw 取「歸屬於母公司業主之權益合計」當 Book_Value。

    為什麼不能用 financials_raw 的同名 type：那份是綜合損益表，
    `EquityAttributableToOwnersOfParent` 的 origin_name 實際是
    「淨利（淨損）歸屬於母公司業主」（FinMind 英文名標錯），拿它當權益會讓 ROE ≈ 1。

    ⚠️ 必須排除 `*_per` 結尾的列——那是同一科目的百分比版本
    （2330 於 2025-03-31：本尊 4.5642e12、`_per` 版 63.98）。
    覆蓋率：93,323 列 / 2,168 支 / 58 季。
    """
    if df_bs is None or df_bs.empty or "type" not in df_bs.columns:
        return None
    bs = df_bs.copy()
    date_col = "Date" if "Date" in bs.columns else "date"
    if date_col not in bs.columns:
        return None
    bs["Date"] = pd.to_datetime(bs[date_col])
    PREF = ["EquityAttributableToOwnersOfParent", "Equity"]   # 前者優先（母公司業主）
    for t in PREF:
        sub = bs[bs["type"] == t]                             # 精確比對，天然排除 *_per
        if sub.empty:
            continue
        out = (sub.groupby(["stock_id", "Date"])["value"]
               .last().rename("Book_Value").reset_index())
        out["Book_Value"] = pd.to_numeric(out["Book_Value"], errors="coerce")
        logger.info(f"[fundamentals_v2] Book_Value 來源 balance_sheet_raw.{t}："
                    f"{len(out):,} 筆 (股,季)")
        return out
    logger.warning("[fundamentals_v2] balance_sheet_raw 找不到權益科目，Book_Value 將退回預設值")
    return None


def _merge_financial_statements(
    df: pd.DataFrame,
    df_fin: pd.DataFrame,
    df_balance_sheet: pd.DataFrame | None = None,
    fundamentals_v2: bool = False,
) -> pd.DataFrame:
    """Merge quarterly EPS, Gross_Margin, ROE with look-ahead protection.
    Handles FinMind long format: columns = [Date, stock_id, type, value, origin_name]"""
    df_fin = df_fin.copy()
    date_col = "Date" if "Date" in df_fin.columns else "date"
    df_fin["Date"] = pd.to_datetime(df_fin[date_col])
    if fundamentals_v2:
        # 台股法定申報期限：Q1 5/15、Q2 8/14、Q3 11/14（皆 ≈ 期末 +45 天），
        # 但 Q4/年報是次年 3/31（≈ 期末 +90 天）。舊行為一律 +45，使 12-31 財報
        # 在 2/14 就可見，每年 2/14–3/31 之間存在 45 天的年報 look-ahead 窗。
        _lag_days = np.where(df_fin["Date"].dt.month == 12, 90, 45)
        df_fin["available_from"] = df_fin["Date"] + pd.to_timedelta(_lag_days, unit="D")
        logger.info("[fundamentals_v2] 財報 available_from：Q1–Q3 +45 天、Q4/年報 +90 天")
    else:
        df_fin["available_from"] = df_fin["Date"] + pd.Timedelta(days=45)

    # -- Detect wide vs long format --
    is_long = "type" in df_fin.columns and "value" in df_fin.columns

    if is_long:
        # Pivot long → wide, extracting key financial items
        TYPE_MAP = {
            # EPS variants
            "EPS": "EPS", "AfterTax_EPS": "EPS", "BasicEPS": "EPS",
            # Revenue / Gross
            "Operating_Revenue": "Revenue", "OperatingRevenue": "Revenue",
            "Gross_Profit": "GrossProfit", "GrossProfit": "GrossProfit",
            # ROE
            "ROE": "ROE",
            # Book value
            "Total_Equity": "Book_Value", "TotalEquity": "Book_Value",
            "StockholdersEquity": "Book_Value",
        }
        if fundamentals_v2:
            # ⚠️ 2026-07-27 修正：上面的鍵是照猜測的欄名寫的，與 financials_raw 實際的
            # 58 種 type 值域對不上，導致三個特徵**自 2005 年起就是死常數**
            # （實測今日橫斷面 std 皆為 0.0000）：
            #   Operating_Revenue/OperatingRevenue → 實際是 "Revenue"
            #       → Revenue 欄不存在 → Gross_Margin = GrossProfit/Revenue 永遠算不出
            #       → 退回 fund_defaults 的常數 0.3
            #   ROE → financials_raw 的 58 種 type 中根本沒有 ROE → 永遠是常數 0.1
            #   Total_Equity/StockholdersEquity → 實際是
            #       "EquityAttributableToOwnersOfParent" → Book_Value 永遠是 0.0
            # 只有 EPS 與 GrossProfit 原本就命中，所以只壞這三個。
            TYPE_MAP.update({
                "Revenue": "Revenue",
                # ROE 分子＝稅後淨利。覆蓋率實測（與權益同 (股,季) 鍵的交集）：
                #   IncomeAfterTaxes 97.7% / NetIncome 45.4%
                #   （TotalConsolidatedProfitForThePeriod 98.6% 但含其他綜合損益，語意不符）
                "IncomeAfterTaxes": "IncomeAfterTaxes",
                "NetIncome": "NetIncome",
            })
            # ⚠️ 絕對不要在這裡把 EquityAttributableToOwnersOfParent 對到 Book_Value。
            # financials_raw 是**綜合損益表**，該 type 的 origin_name 實際是
            # 「淨利（淨損）歸屬於母公司業主」——FinMind 的英文 type 名稱標錯了。
            # 2330 於 2025-03-31：該值 3.6156e11，與 IncomeAfterTaxes 的 3.6073e11
            # 幾乎相同，所以拿它當分母算 ROE 等於「淨利除以淨利」→ 得到 ≈1.0。
            # 真正的股東權益在 balance_sheet_raw（見下方 _balance_sheet_equity）：
            # 同一 (股,季) 的 EquityAttributableToOwnersOfParent = 4.5642e12
            # （origin_name「歸屬於母公司業主之權益合計」）。
            # 驗算 3.607e11 / 4.564e12 = 單季 0.079 → 年化約 31.6%，與台積電實際相符。
        df_fin["mapped"] = df_fin["type"].map(TYPE_MAP)
        df_fin = df_fin[df_fin["mapped"].notna()].copy()
        if df_fin.empty:
            logger.warning("financial_statements: no recognisable type values found")
            return df
        df_fin["value"] = pd.to_numeric(df_fin["value"], errors="coerce")
        # Take last value per (stock_id, Date, mapped)
        df_wide = (df_fin.groupby(["stock_id", "Date", "available_from", "mapped"])["value"]
                   .last().unstack("mapped").reset_index())
        # Compute derived columns
        if "GrossProfit" in df_wide.columns and "Revenue" in df_wide.columns:
            df_wide["Gross_Margin"] = df_wide["GrossProfit"] / df_wide["Revenue"].replace(0, np.nan)
        if fundamentals_v2:
            # Book_Value 從 balance_sheet_raw 取（df_balance_sheet 在此之前是個
            # 從未被讀取的參數——資產負債表資料一直沒被用到）
            _eq = _balance_sheet_equity(df_balance_sheet)
            if _eq is not None:
                df_wide = df_wide.merge(_eq, on=["stock_id", "Date"], how="left")
            # ROE = 稅後淨利 / 母公司業主權益（**單季**，非年化非 TTM；
            # 橫斷面標準化後只看相對排序，一致即可）
            _ni = None
            for _c in ("IncomeAfterTaxes", "NetIncome"):
                if _c in df_wide.columns:
                    _ni = df_wide[_c] if _ni is None else _ni.fillna(df_wide[_c])
            if _ni is not None and "Book_Value" in df_wide.columns:
                df_wide["ROE"] = _ni / df_wide["Book_Value"].replace(0, np.nan)
                _m = df_wide["ROE"].median()
                # 合理性守門：單季 ROE 中位數若 >0.5（年化 >200%）代表分子分母錯配，
                # 寧可讓它退回預設常數也不要上線錯 40 倍的特徵。
                if pd.notna(_m) and abs(_m) > 0.5:
                    logger.warning(
                        f"[fundamentals_v2] ROE 單季中位數 {_m:.4f} 不合理（年化 "
                        f"{_m*4:.0%}），判定分子分母錯配，捨棄 ROE 改用預設常數"
                    )
                    df_wide.drop(columns=["ROE"], inplace=True)
                else:
                    logger.info(f"[fundamentals_v2] ROE 單季中位數 {_m:.4f}"
                                f"（年化約 {_m*4:.1%}）")
            # 對映失敗不可再靜默退回預設常數——那正是這三維死了二十年沒被發現的原因
            for _c in ("Gross_Margin", "ROE", "Book_Value"):
                if _c not in df_wide.columns:
                    logger.warning(
                        f"[fundamentals_v2] {_c} 仍無法從 financials_raw 導出，"
                        f"將退回預設常數（此為死特徵，請檢查 TYPE_MAP 與實際 type 值域）"
                    )
        df_fin = df_wide
        fin_cols = [c for c in ["EPS", "Gross_Margin", "ROE", "Book_Value"] if c in df_fin.columns]
    else:
        fin_cols = [c for c in ["EPS", "Gross_Margin", "ROE", "Book_Value"] if c in df_fin.columns]

    if not fin_cols:
        return df

    df_fin = df_fin.sort_values(["stock_id", "available_from"])

    if fundamentals_v2 and "EPS" in df_fin.columns:
        # EPS_Surprise 必須在「季頻」的 df_fin 上算：pct_change(4) = 與去年同季相比。
        # 舊行為（見本函式結尾的 else 分支）是在 as-of merge **之後** 對日頻列做
        # pct_change(4)，實際只比 4 個交易日。實測 2330 / 2317 / 1301 皆為
        # 「EPS 換值 N 次 → 非零 4N 列」，93.8% 的列恆為 0，只有換值後 4 個交易日
        # 出現脈衝，完全不是「本季 vs 去年同季」的驚喜值。
        # 註：若某公司缺季，pct_change(4) 是「4 筆之前」而非嚴格 4 季，
        #     與 _merge_revenue 的 pct_change(12) 同款近似。
        df_fin["EPS_Surprise"] = (
            df_fin.groupby("stock_id")["EPS"]
                  .pct_change(4)
                  .replace([np.inf, -np.inf], np.nan)
        )
        fin_cols = fin_cols + ["EPS_Surprise"]

        # EPS_TTM（近四季合計）供 PER 自行推算使用（B-1，2026-07-28）。
        # 必須在**季頻**上算、算完才隨同一個 as-of join 帶進日頻——這樣 TTM 的
        # 每一季都受各自的 available_from 保護（Q4 年報 +90 天、其餘 +45 天），
        # 不會把還沒公告的季度 EPS 提前用進去。
        # 若改在日頻上做 rolling，會像 EPS_Surprise 舊 bug 那樣變成「4 個交易日」。
        df_fin["EPS_TTM"] = (
            df_fin.groupby("stock_id")["EPS"]
                  .rolling(4, min_periods=4).sum()
                  .reset_index(level=0, drop=True)
        )
        fin_cols = fin_cols + ["EPS_TTM"]

    # Vectorised as-of per stock
    result_rows = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_fin = df_fin[df_fin["stock_id"] == sid].sort_values("available_from")
        sub_df = sub_df.copy()
        for col in fin_cols:
            sub_df[col] = _asof_lookup(sub_df["Date"],
                                        sub_fin["available_from"].reset_index(drop=True),
                                        sub_fin[col].reset_index(drop=True))
        result_rows.append(sub_df)
    if result_rows:
        df = pd.concat(result_rows, ignore_index=True)

    if fundamentals_v2:
        # 已在季頻算好、隨 as-of join 帶進來，這裡只補 NaN（首 4 季無前值）
        if "EPS_Surprise" in df.columns:
            df["EPS_Surprise"] = df["EPS_Surprise"].fillna(0)
    elif "EPS" in df.columns:
        # V6.1 舊行為（保留以維持與現行 checkpoint 的訓練/推論一致）
        df["EPS_Surprise"] = df.groupby("stock_id")["EPS"].pct_change(4).fillna(0)

    return df


# ============================================================
# V6.1 — Group B Additions (Holdings, Securities, Foreign Holding)
# ============================================================

def _merge_holdings(df: pd.DataFrame, df_holdings: pd.DataFrame | None,
                    availability_flags: bool = False) -> pd.DataFrame:
    """Merge 大戶持股分級 data → Holdings_Large_Pct + Holdings_Large_Change.

    holdings_raw.parquet contains weekly data (集保戶股權分散表).
    Columns: [Week, stock_id, Whale_Hold_Ratio, Retail_Hold_Ratio]

    DATA NOTE: Whale_Hold_Ratio is all-zero in the current parquet (data quality issue
    from the fetcher storing an empty column). We use the available Retail_Hold_Ratio
    as a proxy instead:
      Holdings_Large_Pct = 1.0 - (Retail_Hold_Ratio / 100.0)
    This gives the non-retail (i.e. large holder) fraction, which is the
    economically meaningful signal we want.
    """
    if df_holdings is None or df_holdings.empty:
        df["Holdings_Large_Pct"] = 0.0
        df["Holdings_Large_Change"] = 0.0
        if availability_flags:
            df = _mark_avail(df, "Avail_Holdings", False)
        return df

    h = df_holdings.copy()
    # holdings_raw uses 'Week' column instead of 'Date'
    if "Week" in h.columns and "Date" not in h.columns:
        h = h.rename(columns={"Week": "Date"})
    h["Date"] = pd.to_datetime(h["Date"])

    # ------------------------------------------------------------------
    # Strategy: use Retail_Hold_Ratio if available (Whale_Hold_Ratio is
    # all-zero due to a data fetcher issue). Large_Pct = 1 - Retail/100.
    # Fallback: scan for any usable numeric ratio column.
    # ------------------------------------------------------------------
    if "Retail_Hold_Ratio" in h.columns:
        h["Retail_Hold_Ratio"] = pd.to_numeric(h["Retail_Hold_Ratio"], errors="coerce")
        # Aggregate per (Date, stock_id) — should already be unique but guard anyway
        retail_agg = (
            h.groupby(["Date", "stock_id"])["Retail_Hold_Ratio"]
            .mean()
            .reset_index()
        )
        retail_agg["Holdings_Large_Pct"] = (1.0 - retail_agg["Retail_Hold_Ratio"] / 100.0).clip(0.0, 1.0)
        h_large = retail_agg[["Date", "stock_id", "Holdings_Large_Pct"]]
    else:
        # Generic fallback: pick first numeric ratio/percent column that isn't all zero
        ratio_cols = [c for c in h.columns if any(k in c.lower() for k in ["ratio", "pct", "percent"])]
        num_cols = h.select_dtypes(include="number").columns.tolist()
        candidate_cols = [c for c in (ratio_cols + num_cols) if h[c].fillna(0).abs().sum() > 0]
        if not candidate_cols:
            logger.warning("_merge_holdings: no usable ratio column found — setting Holdings_Large_Pct=0")
            df["Holdings_Large_Pct"] = 0.0
            df["Holdings_Large_Change"] = 0.0
            if availability_flags:
                df = _mark_avail(df, "Avail_Holdings", False)
            return df
        col = candidate_cols[0]
        h[col] = pd.to_numeric(h[col], errors="coerce")
        # Assume the column is already a 'large holder' percentage (0-100 scale)
        h_agg = h.groupby(["Date", "stock_id"])[col].mean().reset_index()
        h_agg["Holdings_Large_Pct"] = (h_agg[col] / 100.0).clip(0.0, 1.0)
        h_large = h_agg[["Date", "stock_id", "Holdings_Large_Pct"]]
        logger.info(f"_merge_holdings: using fallback column '{col}' for Holdings_Large_Pct")

    # Left merge (weekly → daily), then forward fill per stock
    h_large = h_large.sort_values(["stock_id", "Date"])
    df = df.merge(h_large, on=["Date", "stock_id"], how="left", suffixes=("", "_hld"))
    if "Holdings_Large_Pct_hld" in df.columns:
        df["Holdings_Large_Pct"] = df["Holdings_Large_Pct_hld"].combine_first(df.get("Holdings_Large_Pct"))
        df.drop(columns=["Holdings_Large_Pct_hld"], inplace=True)
    df = df.sort_values(["stock_id", "Date"])
    if availability_flags:
        df = _mark_avail(
            df, "Avail_Holdings",
            df.groupby("stock_id")["Holdings_Large_Pct"].transform(lambda x: x.ffill()).notna())
    df["Holdings_Large_Pct"] = df.groupby("stock_id")["Holdings_Large_Pct"].transform(
        lambda x: x.ffill().fillna(0.0))

    # Week-over-week change (5 trading days ≈ 1 week)
    df["Holdings_Large_Change"] = df.groupby("stock_id")["Holdings_Large_Pct"].diff(5).fillna(0.0)

    return df


def _merge_securities(df: pd.DataFrame, df_sec: pd.DataFrame | None,
                      availability_flags: bool = False) -> pd.DataFrame:
    """Merge 借券餘額 → Securities_Balance (daily short lending balance)."""
    if df_sec is None or df_sec.empty:
        df["Securities_Balance"] = 0.0
        if availability_flags:
            df = _mark_avail(df, "Avail_Securities", False)
        return df

    s = df_sec.copy()
    s["Date"] = pd.to_datetime(s["Date"])

    # Find the balance column
    bal_col = [c for c in s.columns if "balance" in c.lower() or "volume" in c.lower()]
    if not bal_col:
        num_cols = s.select_dtypes(include="number").columns.tolist()
        bal_col = num_cols[:1] if num_cols else []

    if not bal_col:
        df["Securities_Balance"] = 0.0
        if availability_flags:
            df = _mark_avail(df, "Avail_Securities", False)
        return df

    s["Securities_Balance"] = pd.to_numeric(s[bal_col[0]], errors="coerce").fillna(0)
    df = df.merge(s[["Date", "stock_id", "Securities_Balance"]],
                  on=["Date", "stock_id"], how="left", suffixes=("", "_sec"))
    if "Securities_Balance_sec" in df.columns:
        df["Securities_Balance"] = df["Securities_Balance_sec"].combine_first(df.get("Securities_Balance"))
        df.drop(columns=["Securities_Balance_sec"], inplace=True)
    if availability_flags:
        # 這個旗標本身帶有經濟意義：借券餘額表只收錄有借券活動的股票，
        # 「出現在表上」= 可被借券/放空，是獨立於數值的資訊
        df = _mark_avail(
            df, "Avail_Securities",
            df.groupby("stock_id")["Securities_Balance"].transform(lambda x: x.ffill()).notna())
    df["Securities_Balance"] = df.groupby("stock_id")["Securities_Balance"].transform(
        lambda x: x.ffill().fillna(0.0))
    return df


def _merge_foreign_shareholding(df: pd.DataFrame, df_fs: pd.DataFrame | None,
                                availability_flags: bool = False) -> pd.DataFrame:
    """Merge 外資持股比例 → Foreign_Holding_Pct (cumulative %).

    DATA NOTE: foreign_shareholding_raw.parquet contains multiple ratio columns.
    We MUST use ForeignInvestmentSharesRatio (外資實際持股%) NOT
    ForeignInvestmentRemainRatio (剩餘可買空間%) — they are semantically opposite.
    A generic 'first ratio column' heuristic would pick the wrong one.
    """
    if df_fs is None or df_fs.empty:
        df["Foreign_Holding_Pct"] = 0.0
        if availability_flags:
            df = _mark_avail(df, "Avail_ForeignShare", False)
        return df

    fs = df_fs.copy()
    if "date" in fs.columns and "Date" not in fs.columns:
        fs.rename(columns={"date": "Date"}, inplace=True)
    fs["Date"] = pd.to_datetime(fs["Date"])

    # Explicit column priority — DO NOT use generic "ratio" heuristic here:
    # ForeignInvestmentSharesRatio  = 外資實際持股比例 (0-100%) ← CORRECT
    # ForeignInvestmentRemainRatio  = 剩餘投資空間比例 (0-100%) ← WRONG (opposite concept)
    PREFERRED_COLS = [
        "ForeignInvestmentSharesRatio",   # 外資實際持股% (primary)
        "ForeignInvestmentSharesRatio".lower(),
        "foreign_investment_shares_ratio",
    ]
    chosen_col = None
    for cname in PREFERRED_COLS:
        if cname in fs.columns:
            chosen_col = cname
            break

    # Fallback: if column not found, warn and use generic heuristic but exclude "Remain"
    if chosen_col is None:
        candidates = [
            c for c in fs.select_dtypes(include="number").columns
            if "remain" not in c.lower() and "upper" not in c.lower()
               and "limit" not in c.lower() and "chinese" not in c.lower()
               and ("ratio" in c.lower() or "pct" in c.lower() or "percent" in c.lower())
        ]
        if candidates:
            chosen_col = candidates[0]
            logger.warning(
                f"_merge_foreign_shareholding: ForeignInvestmentSharesRatio not found, "
                f"falling back to '{chosen_col}'"
            )
        else:
            logger.warning("_merge_foreign_shareholding: no usable shareholding ratio column found")
            df["Foreign_Holding_Pct"] = 0.0
            if availability_flags:
                df = _mark_avail(df, "Avail_ForeignShare", False)
            return df

    logger.info(f"_merge_foreign_shareholding: using '{chosen_col}' as Foreign_Holding_Pct")
    fs["Foreign_Holding_Pct"] = pd.to_numeric(fs[chosen_col], errors="coerce")
    df = df.merge(fs[["Date", "stock_id", "Foreign_Holding_Pct"]],
                  on=["Date", "stock_id"], how="left", suffixes=("", "_fsp"))
    if "Foreign_Holding_Pct_fsp" in df.columns:
        df["Foreign_Holding_Pct"] = df["Foreign_Holding_Pct_fsp"].combine_first(df.get("Foreign_Holding_Pct"))
        df.drop(columns=["Foreign_Holding_Pct_fsp"], inplace=True)
    if availability_flags:
        df = _mark_avail(
            df, "Avail_ForeignShare",
            df.groupby("stock_id")["Foreign_Holding_Pct"].transform(lambda x: x.ffill()).notna())
    df["Foreign_Holding_Pct"] = df.groupby("stock_id")["Foreign_Holding_Pct"].transform(
        lambda x: x.ffill().fillna(0.0))
    return df


# ============================================================
# V6.1 — Group C Additions (Dividend, Free Cash Flow)
# ============================================================

def _merge_dividend_feature(df: pd.DataFrame, df_div: pd.DataFrame | None) -> pd.DataFrame:
    """Compute Dividend_Yield_Fwd from dividend announcements.

    Uses as-of join: dividends are available once announced (before ex-date).
    Dividend_Yield_Fwd = latest announced cash dividend / current Close price.
    """
    if df_div is None or df_div.empty:
        if "Dividend_Yield_Fwd" not in df.columns:
            df["Dividend_Yield_Fwd"] = 0.0
        return df

    div = df_div.copy()
    if "date" in div.columns and "Date" not in div.columns:
        div.rename(columns={"date": "Date"}, inplace=True)
    div["Date"] = pd.to_datetime(div["Date"])

    # Find cash dividend column
    cash_col = [c for c in div.columns if "cash" in c.lower() and ("earning" in c.lower() or "dividend" in c.lower())]
    if not cash_col:
        cash_col = [c for c in div.columns if "dividend" in c.lower()][:1]

    if not cash_col:
        df["Dividend_Yield_Fwd"] = 0.0
        return df

    div["_cash_div"] = pd.to_numeric(div[cash_col[0]], errors="coerce").fillna(0)
    # Keep latest dividend per stock per date
    div = div.sort_values(["stock_id", "Date"])
    div_latest = div.groupby(["stock_id", "Date"])["_cash_div"].sum().reset_index()

    # As-of merge: for each trading day, use the latest announced dividend
    merged = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_div = div_latest[div_latest["stock_id"] == sid].sort_values("Date")
        sub_df = sub_df.copy()
        if not sub_div.empty:
            sub_df["_latest_div"] = _asof_lookup(
                sub_df["Date"],
                sub_div["Date"].reset_index(drop=True),
                sub_div["_cash_div"].reset_index(drop=True),
            )
        else:
            sub_df["_latest_div"] = 0.0
        merged.append(sub_df)

    if merged:
        df = pd.concat(merged, ignore_index=True)

    # Compute forward yield = latest cash dividend / current price
    df["_latest_div"] = df.get("_latest_div", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["Dividend_Yield_Fwd"] = df["_latest_div"] / df["Close"].replace(0, np.nan)
    df["Dividend_Yield_Fwd"] = df["Dividend_Yield_Fwd"].clip(0, 0.2).fillna(0.0)  # cap at 20%
    df.drop(columns=["_latest_div"], inplace=True, errors="ignore")
    return df


def _add_free_cash_flow(df: pd.DataFrame, df_cashflow: pd.DataFrame | None,
                        fundamentals_v2: bool = False) -> pd.DataFrame:
    """Compute Free_Cash_Flow from cashflow_raw.parquet.

    FCF = Operating Cash Flow + Investing Cash Flow（投資活動含資本支出、通常為負）
    Uses as-of join (45-day lag) to prevent look-ahead bias.

    ⚠️ 2026-08-04：舊路徑（`fundamentals_v2=False`）有**兩個**會靜默算錯的問題，
       兩者都只在 v2 下修正，預設維持原行為以免動到 V6.1 的特徵語意。

       ① **投資活動的 type 名稱猜錯**（與 `Gross_Margin`/`ROE`/`Book_Value`
          在 2026-07-27 修掉的完全同型）：程式找的是
          `CashFlowsFromInvestingActivities` / `InvestingActivities`
          ——實測 `cashflow_raw` 裡**各 0 筆**；真正的 type 是
          `CashProvidedByInvestingActivities`（100,967 筆）。
          → `investing` 永遠取到預設的 0，**`Free_Cash_Flow` 其實恆等於
             營業活動現金流、從來沒有減過資本支出**。
          實測影響：22.2% 的 (股,季) 符號改變、橫斷面排名 Spearman 僅 0.654。

       ② **沒有處理累計慣例**：`cashflow_raw` 是**年初至今累計**
          （與 `financials_raw` 的單季不同——見 `fetcher.py` 的 MOPS 區塊陷阱 ④；
          證據：台積電 2025 四季營業現金流 6,256/11,226/15,495/22,750 億單調遞增、
          `期初現金餘額` 四季完全相同）。直接 as-of join 會讓特徵在年內
          Q1→Q4 遞增再重置，形成**與公司現金生成能力無關的鋸齒**。
          實測影響：累計 vs 單季的橫斷面排名相關 Q2 0.65 / Q3 0.58 / Q4 0.54
          ——原本以為「同一天大家都在同一累計階段、會抵銷」，**實測不成立**。
          → v2 下還原為**單季**，與 Group C 其餘欄位（Revenue/GrossProfit/EPS
            皆為單季）同義。
    """
    if df_cashflow is None or df_cashflow.empty:
        if "Free_Cash_Flow" not in df.columns:
            df["Free_Cash_Flow"] = 0.0
        return df

    cf = df_cashflow.copy()
    date_col = "Date" if "Date" in cf.columns else "date"
    cf["Date"] = pd.to_datetime(cf[date_col])
    cf["available_from"] = cf["Date"] + pd.Timedelta(days=45)

    # FinMind long format: [Date, stock_id, type, value]
    if "type" in cf.columns and "value" in cf.columns:
        cf["value"] = pd.to_numeric(cf["value"], errors="coerce")
        # Look for operating CF and capex
        TYPE_MAP = {
            "CashFlowsFromOperatingActivities": "OperatingCF",
            "OperatingActivities": "OperatingCF",
            "CashFlowsFromInvestingActivities": "InvestingCF",
            "InvestingActivities": "InvestingCF",
        }
        if fundamentals_v2:
            # 問題 ①：加上實際存在於值域的 type（上面那兩個都是 0 筆）
            TYPE_MAP["CashProvidedByInvestingActivities"] = "InvestingCF"
        cf["mapped"] = cf["type"].map(TYPE_MAP)
        cf = cf[cf["mapped"].notna()].copy()

        if cf.empty:
            df["Free_Cash_Flow"] = 0.0
            return df

        cf_wide = cf.pivot_table(
            index=["stock_id", "Date", "available_from"],
            columns="mapped", values="value", aggfunc="last"
        ).reset_index()

        if fundamentals_v2:
            # 問題 ②：累計 → 單季（Q1 本身即單季；Q2~Q4 減同年上一季）
            # ⚠️ 必須在算 FCF **之前**逐欄還原，不能對 FCF 事後相減——
            #    某一季若缺 OperatingCF 或 InvestingCF 其中之一，事後相減會把
            #    「缺值」變成「跳動」；逐欄處理時 NaN 會自然傳染、由下游 fillna 處理。
            cf_wide = cf_wide.sort_values(["stock_id", "Date"])
            _q = cf_wide["Date"].dt.quarter
            _y = cf_wide["Date"].dt.year
            for _c in ("OperatingCF", "InvestingCF"):
                if _c not in cf_wide.columns:
                    continue
                prev = cf_wide.groupby(["stock_id", _y])[_c].shift(1)
                cf_wide[_c] = cf_wide[_c].where(_q == 1, cf_wide[_c] - prev)

        if "OperatingCF" in cf_wide.columns:
            investing = cf_wide.get("InvestingCF", pd.Series(0.0, index=cf_wide.index))
            # FCF ≈ Operating CF + Investing CF (investing is negative for capex)
            cf_wide["Free_Cash_Flow"] = cf_wide["OperatingCF"].fillna(0) + investing.fillna(0)
        else:
            df["Free_Cash_Flow"] = 0.0
            return df
    else:
        df["Free_Cash_Flow"] = 0.0
        return df

    # As-of merge per stock
    cf_wide = cf_wide.sort_values(["stock_id", "available_from"])
    result_rows = []
    for sid, sub_df in df.groupby("stock_id"):
        sub_cf = cf_wide[cf_wide["stock_id"] == sid].sort_values("available_from")
        sub_df = sub_df.copy()
        if not sub_cf.empty:
            sub_df["Free_Cash_Flow"] = _asof_lookup(
                sub_df["Date"],
                sub_cf["available_from"].reset_index(drop=True),
                sub_cf["Free_Cash_Flow"].reset_index(drop=True),
            )
        else:
            sub_df["Free_Cash_Flow"] = 0.0
        result_rows.append(sub_df)

    if result_rows:
        df = pd.concat(result_rows, ignore_index=True)
    df["Free_Cash_Flow"] = df["Free_Cash_Flow"].fillna(0.0)
    return df



def _merge_macro(
    df: pd.DataFrame,
    df_macro: pd.DataFrame | None,
    df_fear_greed: pd.DataFrame | None = None,
    df_business_indicator: pd.DataFrame | None = None,
    df_fed_rate: pd.DataFrame | None = None,
    df_futures_inst: pd.DataFrame | None = None,
    df_options_inst: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge macro data including V6.1 additions.
    Handles our macro_raw.parquet column names:
    TWII_Close, US_SOX, US_QQQ, US_VIX, US_TNX, Gold, Oil, USD_TWD, FED_Rate,
    CNN_FearGreed, TW_Biz_Signal.

    DATA NOTE: FED_Rate is read directly from macro_raw (complete 2005-2026 data).
    The standalone fed_rate.parquet is broken (only 8 rows, all 2004-01-01) and
    is used only as a fallback if macro_raw does not contain a FED_Rate column.
    Similarly, CNN_FearGreed and TW_Biz_Signal in macro_raw are preferred over
    the separate fear_greed.parquet / business_indicator.parquet.
    """
    DEFAULTS = {
        "TWII_Return": 0.0, "SPX_Return": 0.0,
        "VIX": 20.0, "TNX": 4.0,
        "Gold_Return": 0.0, "Oil_Return": 0.0,
        "USD_TWD": 30.0,
        # V6.1 macro features
        "Futures_OI_Foreign": 0.0,
        "Options_PC_Ratio": 1.0,
        "Fear_Greed": 50.0,
        "Business_Signal": 23.0,   # green light default
        "FED_Rate": 4.0,
    }
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    if df_macro is not None and not df_macro.empty:
        m = df_macro.copy()
        m["Date"] = pd.to_datetime(m["Date"])

        # Rename to canonical names used in features
        rename_map = {
            "TWII_Close":    "TWII",
            "US_QQQ":        "SPX",       # QQQ as SPX proxy
            "US_VIX":        "VIX",
            "US_TNX":        "TNX",
            # V6.1 macro fields embedded in macro_raw — prefer these over
            # the separate broken parquet files (fed_rate, fear_greed, business_indicator)
            "CNN_FearGreed": "Fear_Greed",
            "TW_Biz_Signal": "Business_Signal",
            # FED_Rate is already named correctly in macro_raw
        }
        m.rename(columns=rename_map, inplace=True)

        # Compute pct-change returns
        for raw, ret in [("TWII", "TWII_Return"), ("SPX", "SPX_Return"),
                         ("Gold", "Gold_Return"), ("Oil", "Oil_Return")]:
            if raw in m.columns:
                m[ret] = m[raw].pct_change(1).fillna(0)

        # V6.2: multi-period TWII returns for RS feature computation.
        # These are intermediate columns merged into df so _add_rs_features() can use them;
        # they are NOT in FEATURE_COLS and will be excluded by the final column reorder.
        if "TWII" in m.columns:
            m["TWII_Return_5d"]  = m["TWII"].pct_change(5).fillna(0)
            m["TWII_Return_20d"] = m["TWII"].pct_change(20).fillna(0)
            m["TWII_Return_60d"] = m["TWII"].pct_change(60).fillna(0)

        # Columns to merge from macro_raw (now includes FED_Rate, Fear_Greed, Business_Signal)
        # TWII_Return_5d/20d/60d are intermediate helpers for RS features (V6.2), not FEATURE_COLS.
        want = [c for c in [
            "TWII_Return", "SPX_Return", "VIX", "TNX",
            "Gold_Return", "Oil_Return", "USD_TWD",
            "FED_Rate", "Fear_Greed", "Business_Signal",
            "TWII_Return_5d", "TWII_Return_20d", "TWII_Return_60d",
        ] if c in m.columns]
        df = df.merge(m[["Date"] + want], on="Date", how="left", suffixes=("", "_m"))
        for col in want:
            dup = col + "_m"
            if dup in df.columns:
                df[col] = df[dup].combine_first(df[col])
                df.drop(columns=[dup], inplace=True)
            df[col] = df[col].fillna(DEFAULTS.get(col, 0.0))

        # Log which V6.1 macro fields were sourced from macro_raw
        sourced = [c for c in ["FED_Rate", "Fear_Greed", "Business_Signal"] if c in want]
        if sourced:
            logger.info(f"macro_raw supplied V6.1 fields: {sourced} (skipping broken separate parquets)")

    # Track whether Business_Signal was successfully set by macro_raw
    # (i.e., not all still at default value 23.0). Used below as fallback guard.
    _business_from_macro = (
        df_macro is not None
        and not df_macro.empty
        and "TW_Biz_Signal" in df_macro.columns
        and "Business_Signal" in df.columns
        and not (df["Business_Signal"] == DEFAULTS["Business_Signal"]).all()
    )

    # -- V6.1: Futures OI Foreign --
    if df_futures_inst is not None and not df_futures_inst.empty:
        fi = df_futures_inst.copy()
        fi["Date"] = pd.to_datetime(fi["Date"])
        # Compute net OI for foreign investors
        fi_cols = fi.columns.tolist()
        if "institutional_investors" in fi_cols:
            fi_foreign = fi[fi["institutional_investors"].str.contains("外資|Foreign", case=False, na=False)].copy()
            if not fi_foreign.empty:
                oi_long_col = [c for c in fi_cols if "long_open_interest" in c.lower() or "long_oi" in c.lower()]
                oi_short_col = [c for c in fi_cols if "short_open_interest" in c.lower() or "short_oi" in c.lower()]
                if oi_long_col and oi_short_col:
                    fi_foreign["Futures_OI_Foreign"] = (
                        pd.to_numeric(fi_foreign[oi_long_col[0]], errors="coerce").fillna(0) -
                        pd.to_numeric(fi_foreign[oi_short_col[0]], errors="coerce").fillna(0)
                    )
                    fi_agg = fi_foreign.groupby("Date")["Futures_OI_Foreign"].sum().reset_index()
                    df = df.merge(fi_agg, on="Date", how="left", suffixes=("", "_fut"))
                    if "Futures_OI_Foreign_fut" in df.columns:
                        df["Futures_OI_Foreign"] = df["Futures_OI_Foreign_fut"].combine_first(df["Futures_OI_Foreign"])
                        df.drop(columns=["Futures_OI_Foreign_fut"], inplace=True)
                    df["Futures_OI_Foreign"] = df["Futures_OI_Foreign"].fillna(0.0)

    # -- V6.1: Options Put/Call Ratio --
    if df_options_inst is not None and not df_options_inst.empty:
        oi = df_options_inst.copy()
        oi["Date"] = pd.to_datetime(oi["Date"])
        oi_cols = oi.columns.tolist()
        # Try to compute Put/Call ratio from volume columns
        if "option_type" in oi_cols or "call_put" in oi_cols:
            type_col = "option_type" if "option_type" in oi_cols else "call_put"
            vol_col = [c for c in oi_cols if "volume" in c.lower() or "deal_volume" in c.lower()]
            if vol_col:
                oi["_vol"] = pd.to_numeric(oi[vol_col[0]], errors="coerce").fillna(0)
                put_vol = oi[oi[type_col].str.contains("put|P|賣", case=False, na=False)].groupby("Date")["_vol"].sum()
                call_vol = oi[oi[type_col].str.contains("call|C|買", case=False, na=False)].groupby("Date")["_vol"].sum()
                pc_ratio = (put_vol / call_vol.replace(0, np.nan)).rename("Options_PC_Ratio").reset_index()
                df = df.merge(pc_ratio, on="Date", how="left", suffixes=("", "_opt"))
                if "Options_PC_Ratio_opt" in df.columns:
                    df["Options_PC_Ratio"] = df["Options_PC_Ratio_opt"].combine_first(df["Options_PC_Ratio"])
                    df.drop(columns=["Options_PC_Ratio_opt"], inplace=True)
                df["Options_PC_Ratio"] = df["Options_PC_Ratio"].fillna(1.0)

    # -- V6.1: Fear & Greed Index --
    if df_fear_greed is not None and not df_fear_greed.empty:
        fg = df_fear_greed.copy()
        fg["Date"] = pd.to_datetime(fg["Date"])
        # Find the value column (varies by FinMind format)
        val_col = [c for c in fg.columns if c not in ["Date", "date"] and "value" in c.lower() or "index" in c.lower()]
        if not val_col:
            val_col = [c for c in fg.columns if c not in ["Date", "date"]][:1]
        if val_col:
            fg["Fear_Greed"] = pd.to_numeric(fg[val_col[0]], errors="coerce")
            df = df.merge(fg[["Date", "Fear_Greed"]], on="Date", how="left", suffixes=("", "_fg"))
            if "Fear_Greed_fg" in df.columns:
                df["Fear_Greed"] = df["Fear_Greed_fg"].combine_first(df["Fear_Greed"])
                df.drop(columns=["Fear_Greed_fg"], inplace=True)
            df["Fear_Greed"] = df["Fear_Greed"].ffill().fillna(50.0)

    # -- V6.1: Business Indicator (景氣燈號) — FALLBACK ONLY --
    # PRIMARY source: macro_raw.TW_Biz_Signal (dates = actual publication dates, correctly aligned).
    # This block is a fallback: only runs if macro_raw did NOT supply Business_Signal.
    #
    # LOOK-AHEAD FIX: business_indicator.parquet uses month-start dates (e.g. 2026-01-01),
    # but NDEV publishes the report ~60 days after period end (Jan data → late March).
    # We shift the available date by +60 days to prevent look-ahead bias.
    if not _business_from_macro and df_business_indicator is not None and not df_business_indicator.empty:
        bi = df_business_indicator.copy()
        bi["Date"] = pd.to_datetime(bi["Date"])

        # Detect score column (monitoring score or composite)
        score_col = [c for c in bi.columns if "score" in c.lower() or "composite" in c.lower()
                     or "signal" in c.lower() or "monitoring" in c.lower()]
        if not score_col:
            score_col = [c for c in bi.columns if c not in ["Date", "date"]][:1]

        if score_col:
            bi["Business_Signal"] = pd.to_numeric(bi[score_col[0]], errors="coerce")
            bi = bi.sort_values("Date")
            bi_dedup = bi.drop_duplicates(subset=["Date"], keep="last").copy()

            # Apply 60-day publication lag: Jan data (Date=Jan 1) not available until ~Mar 2
            bi_dedup["available_from"] = (
                bi_dedup["Date"] + pd.DateOffset(months=2)
            ).dt.normalize()  # keeps day=1, adds 2 months: Jan→Mar, Feb→Apr, ...

            bi_dedup["available_from"] = bi_dedup["available_from"].astype("datetime64[ns]")
            df["Date"] = df["Date"].astype("datetime64[ns]")

            # merge_asof on available_from (not Date) to enforce publication lag
            df = pd.merge_asof(
                df.sort_values("Date"),
                bi_dedup[["available_from", "Business_Signal"]].rename(
                    columns={"available_from": "Date"}
                ).sort_values("Date"),
                on="Date",
                direction="backward",
                suffixes=("", "_bi"),
            )
            if "Business_Signal_bi" in df.columns:
                df["Business_Signal"] = df["Business_Signal_bi"].combine_first(df["Business_Signal"])
                df.drop(columns=["Business_Signal_bi"], inplace=True)
            df["Business_Signal"] = df["Business_Signal"].ffill().fillna(23.0)
            logger.info("business_indicator.parquet used as fallback (60-day lag applied)")
    elif _business_from_macro:
        logger.debug("Business_Signal already sourced from macro_raw — skipping business_indicator.parquet")

    # -- V6.1: FED Rate --
    # PRIMARY: FED_Rate is already sourced from macro_raw above (complete 2005-2026).
    # FALLBACK only: if macro_raw didn't have FED_Rate, try the separate fed_rate parquet.
    if "FED_Rate" not in df.columns or (df["FED_Rate"] == DEFAULTS["FED_Rate"]).all():
        if df_fed_rate is not None and not df_fed_rate.empty:
            fr = df_fed_rate.copy()
            fr["Date"] = pd.to_datetime(fr["Date"])
            rate_col = [c for c in fr.columns if "rate" in c.lower() or "value" in c.lower()]
            if not rate_col:
                rate_col = [c for c in fr.columns if c not in ["Date", "date"]][:1]
            if rate_col:
                fr["FED_Rate"] = pd.to_numeric(fr[rate_col[0]], errors="coerce")
                fr = fr.sort_values("Date")
                fr_dedup = fr.drop_duplicates(subset=["Date"], keep="last")
                # Only use if the fallback has more than one unique date (i.e. not the broken 8-row file)
                if fr_dedup["Date"].nunique() > 1:
                    fr_dedup["Date"] = fr_dedup["Date"].astype("datetime64[ns]")
                    df["Date"] = df["Date"].astype("datetime64[ns]")
                    df = pd.merge_asof(df.sort_values("Date"), fr_dedup[["Date", "FED_Rate"]],
                                       on="Date", direction="backward", suffixes=("", "_fr"))
                    if "FED_Rate_fr" in df.columns:
                        df["FED_Rate"] = df["FED_Rate_fr"].combine_first(df["FED_Rate"])
                        df.drop(columns=["FED_Rate_fr"], inplace=True)
                    df["FED_Rate"] = df["FED_Rate"].ffill().fillna(DEFAULTS["FED_Rate"])
                else:
                    logger.warning("fed_rate.parquet appears broken (<=1 unique date) — keeping macro_raw values")

    return df


# ============================================================
# Alpha Targets (Multi-Horizon)
# ============================================================

def _add_alpha_targets(df: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward Alpha relative to TWII (benchmark) for 5d, 20d, 60d horizons.
    Alpha_Nd = stock cumulative return over N days - TWII cumulative return over N days

    Important: these are FUTURE returns, so they must only be used as training labels.
    They should never leak into the feature columns during inference.
    """
    g = df.groupby("stock_id", sort=False)

    def _fwd_cum_return(series: pd.Series, n: int) -> pd.Series:
        """Forward n-day cumulative return (look-ahead, safe for labels only)."""
        fwd = series.shift(-n) / series - 1
        return fwd

    df["Fwd_5d"]  = g["Close"].transform(lambda x: _fwd_cum_return(x, 5))
    df["Fwd_10d"] = g["Close"].transform(lambda x: _fwd_cum_return(x, 10))   # V6.2 雙模型：短線 10d 標籤
    df["Fwd_20d"] = g["Close"].transform(lambda x: _fwd_cum_return(x, 20))
    df["Fwd_60d"] = g["Close"].transform(lambda x: _fwd_cum_return(x, 60))

    # TWII benchmark returns
    if df_macro is not None and not df_macro.empty and "TWII" in df_macro.columns:
        df_macro = df_macro.copy()
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])
        df_macro["TWII_Fwd_5d"]  = _fwd_cum_return(df_macro["TWII"], 5)
        df_macro["TWII_Fwd_10d"] = _fwd_cum_return(df_macro["TWII"], 10)
        df_macro["TWII_Fwd_20d"] = _fwd_cum_return(df_macro["TWII"], 20)
        df_macro["TWII_Fwd_60d"] = _fwd_cum_return(df_macro["TWII"], 60)
        df = df.merge(
            df_macro[["Date", "TWII_Fwd_5d", "TWII_Fwd_10d", "TWII_Fwd_20d", "TWII_Fwd_60d"]],
            on="Date", how="left",
        )
        df["Alpha_5d"]  = df["Fwd_5d"]  - df["TWII_Fwd_5d"].fillna(0)
        df["Alpha_10d"] = df["Fwd_10d"] - df["TWII_Fwd_10d"].fillna(0)
        df["Alpha_20d"] = df["Fwd_20d"] - df["TWII_Fwd_20d"].fillna(0)
        df["Alpha_60d"] = df["Fwd_60d"] - df["TWII_Fwd_60d"].fillna(0)
        df.drop(columns=["TWII_Fwd_5d", "TWII_Fwd_10d", "TWII_Fwd_20d", "TWII_Fwd_60d"], inplace=True)
    else:
        # ⚠️ 這條路徑是**實際在走的**（2026-08-03 實測確認，非讀碼推論）：
        # `macro_raw` 的欄位叫 `TWII_Close`，把它改名成 `TWII` 的 `_merge_macro`
        # 是在 `m = df_macro.copy()` 上做的 → 呼叫端的 `df_macro` 永遠沒有 `TWII`
        # 欄 → 上面那個 if 從來不成立。所以 `Alpha_Nd` 其實是**原始前瞻報酬**，
        # 不是相對大盤的超額報酬（實測 `Alpha_5d` vs `Fwd_5d` 的 max|Δ| = 0.000e+00，
        # 對照組差 0.183）。
        #
        # **刻意不修**：標籤最後都轉成每日橫斷面 rank，而大盤前瞻報酬是**當日常數**，
        # 減一個當日常數不改變當日排序；IC 又是 Spearman，同樣免疫。改了會讓
        # 所有既有結果無法重現，卻換不到任何實質差異。
        # 但**必須讓它出聲**——原本這個退化是完全靜默的，名字叫 Alpha 卻不是 Alpha。
        _cols = list(df_macro.columns) if df_macro is not None else []
        logger.warning(
            "[targets] df_macro 沒有 'TWII' 欄（實際欄位：%s…）→ Alpha_Nd 退化為"
            "**原始前瞻報酬**，未減去大盤。這是目前的實際行為，且對 rank 標籤與"
            " Spearman IC 無影響（減當日常數不改排序）；詳見 "
            "docs/label-horizon-vs-holding-period-2026-08-03.md §7.1",
            _cols[:6],
        )
        df["Alpha_5d"]  = df["Fwd_5d"]
        df["Alpha_10d"] = df["Fwd_10d"]
        df["Alpha_20d"] = df["Fwd_20d"]
        df["Alpha_60d"] = df["Fwd_60d"]

    df.drop(columns=["Fwd_5d", "Fwd_10d", "Fwd_20d", "Fwd_60d"], inplace=True)
    return df


# ============================================================
# Utility
# ============================================================

def _asof_lookup(dates: pd.Series, ref_dates: pd.Series, values: pd.Series) -> pd.Series:
    """
    For each date in `dates`, find the most recent value in `values`
    where ref_dates <= date (as-of join, no look-ahead bias).

    VECTORISED: uses numpy searchsorted for O(N log M) performance
    instead of a Python loop which would be O(N * M).
    """
    if ref_dates.empty or values.empty:
        return pd.Series(np.nan, index=dates.index)

    ref_arr = ref_dates.values   # sorted array of reference timestamps
    val_arr = values.values

    # For each date, find the rightmost ref_date <= date
    # searchsorted(..., 'right') gives the insertion point after all matching elements
    positions = np.searchsorted(ref_arr, dates.values, side="right") - 1

    # positions < 0 means no ref_date is available yet (before first publish date)
    result = np.where(positions >= 0, val_arr[positions], np.nan)
    return pd.Series(result.astype(np.float64), index=dates.index)


def _neutralize_cross_section(df: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
    """
    橫斷面產業／市值中性化（F3）。在 winsorize 之後、z-score 之前執行。

    【解決什麼】未中性化的因子會偷偷變成產業或規模的賭注：一個「價值因子」
    可能只是「一直押傳產」，一個「動能因子」可能只是「一直押小型股」。
    深度學習**不會**自動處理這件事——模型沒有理由把 beta 和 alpha 分開，
    它只會學到「能預測報酬的東西」，包含那些你其實不想賭的系統性成分。

    mode:
      "industry"        產業內去均值。只有產業 dummy 時，對 dummy 迴歸取殘差
                        在數學上等同於減去組內均值，用 groupby 快很多。
      "industry_mktcap" 對 [產業 dummies, log 市值] 做 OLS 取殘差。
                        設計矩陣每天只有一份，59 個特徵一次批次求解。

    【PIT 限制，明確揭露】產業分類來自 `stock_info` 的現況快照，不是逐日的
    PIT 歷史。實測（2026-07-29）2023→2026 的標籤變動幾乎全是交易所分類改版
    （觀光事業→觀光餐旅、創新「版」→創新「板」是錯字修正、其他→運動休閒
    是新增類別），真正的公司重新分類只有個位數，所以用最新分類回推歷史可接受。
    但這是量測結果，不是假設，也不代表沒有影響。
    """
    from marketmamba.data.feature_spec import resolve_sector
    from marketmamba.data.hygiene import load_stock_info

    sec_map = resolve_sector(load_stock_info(latest_only=False))
    if sec_map.empty:
        logger.warning("[neutralize] 取不到產業分類，略過中性化")
        return df

    sector = df["stock_id"].astype(str).map(
        dict(zip(sec_map["stock_id"], sec_map["sector"]))).fillna("Unknown")
    n_unknown = int((sector == "Unknown").sum())
    logger.info(f"[neutralize] mode={mode}｜{sector.nunique()} 個產業｜"
                f"無產業別 {n_unknown:,} 列（{n_unknown / max(len(df), 1):.1%}，"
                f"自成一組不與他人混合）")

    if mode == "industry":
        # grouper 建一次、55 個欄位一次 transform。
        # 逐欄 `df.groupby([Date, sector])[col]` 會讓 pandas 每次重建分組索引，
        # 實測在 141K 列 × 55 欄下慢一個數量級。
        g = df.groupby([df["Date"], sector], sort=False)
        df[cols] = df[cols].to_numpy() - g[cols].transform("mean").to_numpy()
        return df

    # ── industry_mktcap：逐日 OLS ──────────────────────────────
    if "Market_Cap_Log" not in df.columns:
        logger.warning("[neutralize] 缺 Market_Cap_Log，退回 industry-only")
        return _neutralize_cross_section(df, cols, "industry")

    mc = pd.to_numeric(df["Market_Cap_Log"], errors="coerce")
    codes, _ = pd.factorize(sector)
    vals = df[cols].to_numpy(dtype="float64", copy=True)
    mc_v = mc.to_numpy(dtype="float64")
    date_codes, _ = pd.factorize(df["Date"], sort=True)

    order = np.argsort(date_codes, kind="stable")
    bounds = np.searchsorted(date_codes[order], np.arange(date_codes.max() + 2))
    n_skipped = 0
    for d in range(len(bounds) - 1):
        idx = order[bounds[d]:bounds[d + 1]]
        if idx.size < 10:          # 橫斷面太小，迴歸沒有意義
            n_skipped += 1
            continue
        s = codes[idx]
        uniq = np.unique(s)
        # 設計矩陣：產業 one-hot（含截距效果）+ log 市值
        X = np.zeros((idx.size, uniq.size + 1), dtype="float64")
        X[np.arange(idx.size), np.searchsorted(uniq, s)] = 1.0
        m = mc_v[idx]
        X[:, -1] = np.nan_to_num(m - np.nanmean(m), nan=0.0)

        Y = vals[idx]
        bad = ~np.isfinite(Y)
        if bad.any():
            Y = np.where(bad, 0.0, Y)   # NaN 暫代 0 參與迴歸，殘差後還原
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        if bad.any():
            resid[bad] = np.nan
        vals[idx] = resid

    df[cols] = vals
    if n_skipped:
        logger.info(f"[neutralize] {n_skipped} 個交易日橫斷面 <10 支，未中性化")
    return df


def clean_and_scale(df: pd.DataFrame, macro_norm: str = "cross",
                    neutralize: str = "none") -> pd.DataFrame:
    """
    Final cleaning step:
      1. Drop rows with too many NaN features
      2. Winsorize at [1%, 99%] cross-sectionally (per Date)
      3. Z-score standardise cross-sectionally (per Date)

    This is intentionally cross-sectional, not time-series — we want relative
    ranks within a day, not absolute magnitudes.

    Args:
        macro_norm:
          "cross"（預設）：V6.1 行為——所有特徵 per-date cross-sectional z-score。
            ⚠️ D1 已知問題：Group D macro 特徵同日全股票同值 → std=0 → 恆為 0。
            V6.1 checkpoint 即以全 0 macro 訓練（proj_D 權重未訓練），
            **推論端在 V6.2 checkpoint 部署前必須維持 "cross"**。
          "ts"（V6.2 重訓起）：Group D 改 expanding time-series z-score——
            每日用「前一日為止」的歷史均值/標準差標準化（shift(1) 無 look-ahead），
            至少 252 天歷史才輸出非零值，z 值截斷 ±3σ。其餘特徵維持 cross-sectional。

        neutralize（F3，V6.3）：
          "none"（預設）：不做中性化，與 V6.1/V6.2 行為完全相同。
          "industry"：產業內去均值。
          "industry_mktcap"：對 [產業 dummies, log 市值] 取 OLS 殘差。
          ⚠️ 中性化會改變 2005 年起所有歷史特徵值，必須與對應的 checkpoint 綁定，
             理由同 fundamentals_v2 / macro_norm。預設關閉，
             先用便宜模型（Ridge/GBDT）量出 IC delta 再決定要不要帶進重訓。
    """
    assert macro_norm in ("cross", "ts"), f"macro_norm must be 'cross' or 'ts', got {macro_norm!r}"

    # Drop rows where > 30% of feature columns are NaN
    #
    # 判斷基準刻意**排除可得性旗標**：旗標永遠是 0/1、不可能是 NaN，
    # 算進去會讓門檻變寬（67 維時 int(0.7×67)=46，8 個旗標白送 8 分），
    # 於是在 59 維會被剔除的列到了 67 維反而存活——同一份原始資料、
    # 兩種協定版本留下不同的列，之後比較 v1/v2 的 IC 就不是同一批樣本了。
    _qual_cols = [c for c in FEATURE_COLS if c not in AVAIL_COLS]
    threshold = int(0.7 * len(_qual_cols))
    df = df.dropna(subset=_qual_cols, thresh=threshold).copy()

    macro_cols = set(FEATURE_GROUPS["macro_environment"]) if macro_norm == "ts" else set()

    # V6.3：可得性旗標一律**不做**任何標準化，保持原始 0/1。
    # 若逐日 z-score：全市場都有資料的那天 std=0 → 整欄變 0；
    # 全市場都沒資料的那天 std=0 → 也變 0。兩個相反的狀態變成同一個值，
    # 而「整欄全缺」正是旗標要表達的事，等於自我抵銷。
    avail_cols = [c for c in AVAIL_COLS if c in df.columns]
    if avail_cols:
        _stat = {c: (float(df[c].mean()), int(df[c].nunique())) for c in avail_cols}
        logger.info(
            "[clean_and_scale] 可得性旗標維持原始 0/1（不 winsorize、不 z-score）：\n  "
            + "\n  ".join(f"{c}: 可得比例 {m:.1%}，相異值 {n}" for c, (m, n) in _stat.items())
        )

    cross_cols = [c for c in FEATURE_COLS if c not in macro_cols and c not in AVAIL_COLS]

    # 順序：winsorize → （可選）中性化 → z-score
    # winsorize 先做，是為了不讓離群值主導中性化迴歸的係數；
    # z-score 最後做，才能讓殘差回到可比較的尺度。
    for col in cross_cols:
        df[col] = df.groupby("Date")[col].transform(
            lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
        )

    if neutralize != "none":
        assert neutralize in ("industry", "industry_mktcap"), \
            f"neutralize must be none/industry/industry_mktcap, got {neutralize!r}"
        neu_cols = [c for c in cross_cols if c not in NEUTRALIZE_EXCLUDE]
        logger.info(f"[clean_and_scale] neutralize={neutralize}："
                    f"{len(neu_cols)}/{len(cross_cols)} 欄參與"
                    f"（排除 {len(cross_cols) - len(neu_cols)} 欄：旗標／原始價量／市值／macro）")
        df = _neutralize_cross_section(df, neu_cols, neutralize)

    for col in cross_cols:
        df[col] = df.groupby("Date")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )

    if macro_cols:
        MIN_HIST = 252   # 至少一年歷史才開始輸出非零 z 值
        latest_dt = df["Date"].max()
        latest_vals = {}
        for col in sorted(macro_cols):
            # macro 同日全股票同值 → 以日為單位的序列計算 expanding 統計
            s  = df.groupby("Date")[col].first().sort_index()
            mu = s.expanding(min_periods=MIN_HIST).mean().shift(1)   # shift(1)：只用過去資料
            sd = s.expanding(min_periods=MIN_HIST).std().shift(1)
            z  = ((s - mu) / (sd + 1e-9)).clip(-3.0, 3.0)
            df[col] = df["Date"].map(z)
            latest_vals[col] = float(z.get(latest_dt, float("nan")))
        # 規則 7：數值明確輸出（最後交易日的 macro z 值）
        logger.info(
            f"[clean_and_scale] macro_norm=ts（expanding z-score, min {MIN_HIST}d, shift(1), clip ±3σ）"
        )
        logger.info(
            "[clean_and_scale] 最後交易日 macro z 值: "
            + ", ".join(f"{k}={v:+.3f}" for k, v in latest_vals.items())
        )

    # Fill any remaining NaNs with 0 (cross-sectional mean after z-score)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    return df
