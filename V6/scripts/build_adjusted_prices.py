"""
MarketMamba — B-3 最終步驟：組出還原後的全歷史價格表（2026-07-28）
=====================================================================
輸入：`_raw_prices/raw_YYYY.parquet`（交易所原始價）+ `ex_rights_raw.parquet`（官方因子）
輸出：`prices_adj_raw.parquet`（**新檔**，驗證通過前不動 `prices_raw`）

還原公式與方向的推導見 `apply_ex_rights_adjustment.py` 的檔頭，此處只組裝。

---------------------------------------------------------------------------
【為什麼直接採用交易所資料，而不是把舊檔缺的列補回來】
  舊 `prices_raw`（yfinance）比交易所多 374,228 列。拆解後：

      93.0%  落在該股「首次出現於交易所資料」之前  → 上市櫃前的**興櫃**期
       7.0%  上市櫃後的零星缺漏（697 支，成交量 median **90 股**，99.7% < 1 萬股）

  兩類都不該補：前者是興櫃（B-2 已決定排除），後者是 90 股的成交日——
  那種「收盤價」不構成有意義的價格。把它們用 yfinance 的還原基準塞進來，
  反而會在零星日子製造與鄰居不同基準的假報酬，正是本次要根治的病。

  **附帶收穫**：「首次出現於交易所資料的日期」＝上市櫃日，是個 **PIT 正確**的判準。
  它自動排除「曾是興櫃、後來轉上市櫃」股票的興櫃期——而那正是 B-2 用
  `stock_info` 現況快照時**無法處理**、被我記為已知限制的部分。改用本判準後該限制消失。

【2007-07 前的上櫃段】
  TPEX 價格端點只到 2007 下半年，該段沒有原始價。做法不是直接沿用舊檔——
  舊檔是 yfinance 基準，與我們的因子基準不同，硬接會在 2007-07-01 造成假跳空。
  改為**接縫等比縮放**：取該股在接縫後第一個共同交易日，
  以 `我方還原價 / 舊檔價` 為係數，把整段 2007-07 前乘上去。
  這樣保留該段的**形狀**（報酬序列不變），同時讓水位與新基準連續。
  這些列以 `src="legacy_scaled"` 標記，下游可自行決定要不要用。

用法（repo 根目錄）：
    python V6/scripts/build_adjusted_prices.py --dry-run
    python V6/scripts/build_adjusted_prices.py --apply
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_ex_rights_adjustment import (  # noqa: E402
    apply_to_frame, build_factor_lookup, spot_check, validate,
)
from marketmamba.config import PROCESSED_DIR  # noqa: E402

RAW_DIR = PROCESSED_DIR / "_raw_prices"
EX_PATH = PROCESSED_DIR / "ex_rights_raw.parquet"
OLD_PATH = PROCESSED_DIR / "prices_raw.parquet"
OUT_PATH = PROCESSED_DIR / "prices_adj_raw.parquet"
TPEX_START = pd.Timestamp("2007-07-01")
PRICE_COLS = ["Open", "High", "Low", "Close"]


def splice_pre_tpex(adj: pd.DataFrame) -> pd.DataFrame:
    """
    把 2007-07 前、只有舊檔才有的列接回來，並做接縫等比縮放。

    只處理**在新資料中確實存在**的股票（否則無從決定縮放係數，也代表該股
    在交易所資料裡從未出現＝興櫃或已下市，本來就不該納入）。
    """
    old = pd.read_parquet(OLD_PATH)
    old["Date"] = pd.to_datetime(old["Date"])
    old["stock_id"] = old["stock_id"].astype(str)
    pre = old[old["Date"] < TPEX_START]
    if pre.empty:
        return pd.DataFrame()

    # 新資料在接縫前已涵蓋上市（TWSE 自 2005 可抓），故只需補新資料缺的 key
    have = set(map(tuple, adj.loc[adj["Date"] < TPEX_START,
                                  ["Date", "stock_id"]]
                   .itertuples(index=False, name=None)))
    pre = pre[~pd.Series(list(map(tuple, pre[["Date", "stock_id"]]
                                  .itertuples(index=False, name=None))),
                         index=pre.index).isin(have)]
    if pre.empty:
        return pd.DataFrame()

    # 每股的接縫係數：接縫後第一個共同交易日的 我方還原價 / 舊檔價
    post_new = (adj[adj["Date"] >= TPEX_START]
                .sort_values(["stock_id", "Date"])
                .groupby("stock_id", as_index=False).first()[["stock_id", "Date", "Close"]]
                .rename(columns={"Date": "join_date", "Close": "new_close"}))
    j = post_new.merge(
        old[["Date", "stock_id", "Close"]].rename(
            columns={"Date": "join_date", "Close": "old_close"}),
        on=["stock_id", "join_date"], how="inner")
    j["k"] = pd.to_numeric(j["new_close"], errors="coerce") / \
        pd.to_numeric(j["old_close"], errors="coerce").replace(0, np.nan)
    j = j[np.isfinite(j["k"]) & (j["k"] > 0)]
    kmap = dict(zip(j["stock_id"], j["k"]))

    pre = pre[pre["stock_id"].isin(kmap)].copy()
    if pre.empty:
        return pd.DataFrame()
    k = pre["stock_id"].map(kmap).astype("float64")
    for c in PRICE_COLS:
        if c in pre.columns:
            pre[c] = pd.to_numeric(pre[c], errors="coerce") * k
    pre["adj_mult"] = k
    print(f"  接縫縮放：{len(pre):,} 列 / {pre['stock_id'].nunique():,} 支"
          f"｜係數 median {pd.Series(list(kmap.values())).median():.4f}"
          f"（{pre['Date'].min().date()} → {pre['Date'].max().date()}）")
    return pre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.print_help()
        return

    files = sorted(RAW_DIR.glob("raw_*.parquet"))
    print(f"[輸入] 原始價年度檔 {len(files)} 個")
    ex = pd.read_parquet(EX_PATH)
    ex["Date"] = pd.to_datetime(ex["Date"])
    ex["stock_id"] = ex["stock_id"].astype(str)
    print(f"[輸入] 還原因子 {len(ex):,} 筆 / {ex['stock_id'].nunique():,} 支")

    lut = build_factor_lookup(ex)
    parts_adj, parts_raw = [], []
    for p in files:
        d = pd.read_parquet(p)
        parts_raw.append(d.assign(Date=pd.to_datetime(d["Date"])))
        parts_adj.append(apply_to_frame(d, lut))
    adj = pd.concat(parts_adj, ignore_index=True)
    raw = pd.concat(parts_raw, ignore_index=True)
    del parts_adj, parts_raw
    adj["src"] = "exchange"
    print(f"[交易所段] {len(adj):,} 列｜{adj['stock_id'].nunique():,} 支"
          f"｜{adj['Date'].min().date()} → {adj['Date'].max().date()}")

    print("[接縫] 2007-07 前的上櫃段（無原始價來源）")
    pre = splice_pre_tpex(adj)
    if not pre.empty:
        pre["src"] = "legacy_scaled"
        adj = pd.concat([adj, pre[adj.columns]], ignore_index=True)

    adj = adj.drop_duplicates(subset=["Date", "stock_id"], keep="first")
    adj = adj.sort_values(["stock_id", "Date"]).reset_index(drop=True)
    print(f"[合計] {len(adj):,} 列｜{adj['stock_id'].nunique():,} 支"
          f"｜來源分布 {dict(adj['src'].value_counts())}")

    validate(adj[adj["src"] == "exchange"], raw, ex)
    spot_check(adj[adj["src"] == "exchange"], raw, ex)

    if a.apply:
        adj.drop(columns=["adj_mult"]).to_parquet(OUT_PATH, index=False)
        print(f"\n✅ 已寫入 {OUT_PATH.name}（新檔，`prices_raw` 未被改動）")
        print("   下一步：以此檔跑一次特徵/推論比對，確認無誤後才切換")
    else:
        print("\n--dry-run：未寫檔。")


if __name__ == "__main__":
    main()
