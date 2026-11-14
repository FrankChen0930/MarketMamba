"""
macro_ts_full.py — Group D 的 expanding z-score 用全歷史算，再貼回尾端窗
========================================================================
每日推論只建「尾端窗」（3~6 年），但 `clean_and_scale(macro_norm="ts")` 的
Group D 是 **expanding** 統計量（min 252 天、shift(1)），算在**傳進去的
日期範圍**上 → 窗長不同，12 個 macro 欄的數值就不同。

2026-08-08 實測（`experimental/diag_window_panel.py`，兩個不同窗長互比、
同資料同一天、唯一變因是窗長）：

    非 macro 欄  51/54 完全相同（ρ=1.000000、max|Δ|=0）
    macro 欄     **全部偏很大**：Oil_Return max|Δ|=2.27、TNX 1.07、VIX 0.87

⚠️ Mamba 的上線 arm（`v2_kg_nomacro` 系）把 Group D 整個歸零，所以**量不到這件事**；
   但 `v2_kg` / `v3_kg` / `old_kg` / `no_gat` 與 B 類的 Ridge/GBDT/GRU 都吃 Group D，
   躲不掉。餵錯的 Group D 給訓練時吃正確值的 checkpoint ＝ train/serve skew。

為什麼可以只補 12 欄
--------------------
macro 是**每日橫斷面常數**（實測抽 5 天，12 維的當日相異值數都是 1），
所以只需要「每日一列」的序列，不需要任何個股列——成本約 5,700 列，可忽略。

⚠️ **兩個不可妥協的實作紀律**
  ① z-score 一律呼叫 `feature_engineer.macro_ts_zscore`，**不在這裡複製那五行**
     （2026-08-08 為此把它從 `clean_and_scale` 抽出來，逐位元回歸通過）。
  ② 原始值取自 `baseline_cache_v2/chunks/base_chunk_*.parquet`——那是
     `build_base_matrix` 在 **clean_and_scale 之前**寫出的未標準化值。
     取 clean 之後的檔會拿到「已經被標準化過的東西再標準化一次」。

本模組**刻意不 import 任何 protocol 相關的東西**，因為 Mamba 線（59 維）與
baseline 線（66 維）在不同 process、不同 `MM_PROTOCOL` 下都要用它。
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd


def full_history_macro_z(macro_cols: list[str],
                         recent_raw: pd.DataFrame | None = None,
                         chunk_dir: Path | None = None) -> pd.DataFrame | None:
    """回 (index=Date, columns=macro_cols) 的全歷史 expanding z-score；失敗回 None。

    `recent_raw`：(index=Date, columns=macro_cols) 的**未標準化** macro，
    來自當前尾端窗（`clean_and_scale` 之前）。

    ⚠️ **這個參數不是可有可無的**。chunk 是 `build_base_matrix` 當時寫的，
       停在 2026-07-29；而每日推論跑的是「今天」。少了它，今天的日期在
       expanding 序列裡根本不存在 → `map` 回來全是 **NaN**，
       比窗內自算還糟（窗內自算至少有值、只是偏）。
       所以歷史段取 chunk、**今天這段一定要由呼叫端補上**，兩段接起來再算 expanding。
    """
    from marketmamba.config import PROCESSED_DIR
    from marketmamba.data.feature_engineer import macro_ts_zscore

    cols = list(macro_cols)
    chunk_dir = chunk_dir or (Path(PROCESSED_DIR) / "baseline_cache_v2" / "chunks")
    chunks = sorted(Path(chunk_dir).glob("base_chunk_*.parquet"))

    hist = None
    if chunks:
        try:
            # macro 每日同值 → 讀第一個 chunk 就夠（每個 chunk 都含全部日期）
            src = pd.read_parquet(chunks[0], columns=["Date"] + cols)
            src["Date"] = pd.to_datetime(src["Date"])
            hist = src.groupby("Date")[cols].first().sort_index()
            del src
            gc.collect()
        except Exception:                                       # noqa: BLE001
            hist = None

    if hist is None and recent_raw is None:
        return None

    if hist is None:
        per_day = recent_raw.sort_index()
    elif recent_raw is None:
        per_day = hist
    else:
        # 歷史段以 chunk 為準，只補 chunk 沒有的日子（不覆蓋既有歷史，
        # 否則同一天可能出現兩個不同的原始值）
        add = recent_raw[~recent_raw.index.isin(hist.index)]
        per_day = pd.concat([hist, add]).sort_index()

    return pd.DataFrame({c: macro_ts_zscore(per_day[c]) for c in cols})


def splice(df: pd.DataFrame, macro_cols: list[str], *, logger=None,
           recent_raw: pd.DataFrame | None = None,
           chunk_dir: Path | None = None) -> pd.DataFrame:
    """把 `df` 的 macro 欄換成全歷史版本。**就地修改並回傳同一個 df。**

    抓不到來源時**不 raise**：對 Group D 歸零的 arm 完全沒差，
    為了它讓整條每日流程掛掉不合理。但一定要**大聲講**，
    否則就變成「有做但看不出來」（規則 7）。
    """
    def _say(msg: str, warn: bool = False) -> None:
        # ⚠️ **一律 print**，logger 只是附加。2026-08-08 第一版只走 logger，
        #    結果在 `run_v62_inference` 的驗證 log 裡**一行都沒出現**
        #    （root logger 早被別的 import 設定過，basicConfig 變成 no-op），
        #    害我一度以為貼回沒執行。這一步是「餵給模型的輸入被換掉了」，
        #    看不見等同沒做（規則 7）——不能把它的可見性交給 logging 設定。
        print(("⚠️ " if warn else "") + msg, flush=True)
        if logger is not None:
            (logger.warning if warn else logger.info)(msg)

    cols = [c for c in macro_cols if c in df.columns]
    if not cols:
        _say("[macro] df 沒有 Group D 欄位 → 跳過貼回", True)
        return df

    z = full_history_macro_z(cols, recent_raw=recent_raw, chunk_dir=chunk_dir)
    if z is None or z.empty:
        _say("[macro] 找不到 base_chunk_*.parquet → **沿用尾端窗自算的 macro**"
             "（12 欄會偏；實測 Oil_Return max|Δ| 可達 2.27）。"
             "Group D 歸零的 arm 不受影響，其餘 arm 的分數不可信。", True)
        return df

    # 規則 7：貼回前後拿**同一天**比（不是第一列——窗首那幾天 expanding 還沒暖機，
    # 兩邊都是 NaN/0，看不出差別，第一版就是這樣印出「+0.0000 → +1.3630」的誤導數字）
    probe_date = df["Date"].max()
    before = {c: float(df.loc[df["Date"] == probe_date, c].iloc[0]) for c in cols[:3]}

    for c in cols:
        df[c] = df["Date"].map(z[c]).astype(np.float32)

    after = {c: float(df.loc[df["Date"] == probe_date, c].iloc[0]) for c in cols[:3]}
    cov = float(df[cols[0]].notna().mean())
    _say(f"[macro] 全歷史貼回 {len(cols)} 欄（來源 {len(z):,} 個交易日："
         f"{z.index.min().date()} → {z.index.max().date()}）｜覆蓋率 {cov:.1%}")
    for c in before:
        _say(f"[macro]   {c:<22s} {probe_date.date()}："
             f"窗內自算 {before[c]:+.4f} → 全歷史 {after[c]:+.4f}")
    if cov < 0.999:
        _say(f"[macro] 覆蓋率只有 {cov:.1%}——推論日可能超出 chunk 的日期範圍，"
             f"那幾天的 macro 會是 NaN。chunk 停在 {z.index.max().date()}，"
             f"要往後就得重建 chunk。", True)
    return df
