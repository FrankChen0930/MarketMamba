"""
MarketMamba — 全歷史「未還原原始價」重抓（2026-07-28）
========================================================
【為什麼要重抓】
  現有 `prices_raw.parquet` 的歷史段是 yfinance `auto_adjust` 的還原價
  （以最後一次全量同步、約 2026-04-24 為基準），之後的增量段是交易所原始價。
  要用我們自己的官方還原因子重建全歷史，必須有**乾淨的未還原原始價**當基底。

  另一個不能省的理由：除權息參考價公式裡的現金股利是**絕對金額（元）**，
  不具尺度不變性——拿被縮放過的價格當前收盤價去反推因子，結果會是錯的。
  這也是為什麼 B-3 的順序必然是「先重抓原始價 → 再算上櫃因子 → 最後還原」。

【端點歷史深度（2026-07-28 實測）】
  TWSE  2005-01-04 起可抓（該日 694 檔）              ✓ 全期涵蓋
  TPEX  2007 上半年才開始（2007-01-04 無、2007-07-02 有 528 檔）
        → 2007-07 以前的上櫃原始價**無來源**，該段維持現況並標記為已知限制
          （使用者 2026-07-28 已同意此限制）

【可恢復設計】
  逐年寫 `_raw_prices/raw_YYYY.parquet`，中斷後重跑會自動跳過已完成的年份。
  單年檔案小、記憶體佔用低（本機記憶體吃緊，不可整份 8.7M 列在記憶體裡累積）。

【與每日推論的互斥】
  每日 19:30 的推論也會打同一批交易所端點，長時間輪詢可能互相干擾
  （交易所對高頻請求會回異常資料——2026-06-19 端午節回 1,075 筆假資料即為前例）。
  預設在 19:15–20:15 自動暫停，用 --no-pause 可關閉。

用法（repo 根目錄）：
    python V6/scripts/refetch_raw_prices_full.py --run
    python V6/scripts/refetch_raw_prices_full.py --status
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402
from marketmamba.data.fetcher import (  # noqa: E402
    fetch_prices_tpex_direct,
    fetch_prices_twse_direct,
)

OUT_DIR = PROCESSED_DIR / "_raw_prices"
PRICE_PATH = PROCESSED_DIR / "prices_raw.parquet"

TWSE_START = "2005-01-01"
TPEX_START = "2007-07-01"          # 實測最早可用約在此
SLEEP = 1.1                        # 每次請求間隔
PAUSE_FROM, PAUSE_TO = dtime(19, 15), dtime(20, 15)


def _trading_days() -> list[pd.Timestamp]:
    """交易日曆取自現有 prices_raw（已清過非交易日假資料）。"""
    d = pd.read_parquet(PRICE_PATH, columns=["Date"])
    return sorted(pd.to_datetime(d["Date"]).unique())


def _fetch_retry(fn, ds: str, tries: int = 3):
    """
    帶重試的單日抓取。

    為什麼需要：建 `ex_rights_raw` 時，一次性的網路失敗讓整個 2022–2024 區塊落空，
    腳本照樣跑完並產出一張缺三年的表，而總筆數看起來完全合理。
    對外部來源的每一次請求都要假設它會偶爾失敗，
    **「聚合統計看起來合理」不能當作沒失敗的證據**。

    這裡不把最終失敗當致命錯誤（真的有非交易日或該市場當日無資料的情況），
    但會回報給呼叫端計入 `空回應`，逐年印出供人工複查。
    """
    for i in range(tries):
        df = fn(ds)
        if df is not None and not df.empty:
            return df
        if i < tries - 1:
            time.sleep(2 + i * 3)
    return None


def _maybe_pause(no_pause: bool) -> None:
    if no_pause:
        return
    while True:
        now = datetime.now().time()
        if not (PAUSE_FROM <= now <= PAUSE_TO):
            return
        print(f"  [暫停] 現在 {now:%H:%M:%S} 落在每日推論時段 "
              f"{PAUSE_FROM:%H:%M}–{PAUSE_TO:%H:%M}，等待 5 分鐘…", flush=True)
        time.sleep(300)


def run(no_pause: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = _trading_days()
    by_year: dict[int, list[pd.Timestamp]] = {}
    for d in days:
        by_year.setdefault(pd.Timestamp(d).year, []).append(pd.Timestamp(d))

    years = sorted(by_year)
    done = {int(p.stem.split("_")[1]) for p in OUT_DIR.glob("raw_*.parquet")}
    todo = [y for y in years if y not in done]
    total_days = sum(len(by_year[y]) for y in todo)
    print(f"[計畫] 交易日 {len(days):,} 天 / {years[0]}–{years[-1]}")
    print(f"[計畫] 已完成年份 {sorted(done) or '無'}｜待處理 {len(todo)} 年 "
          f"/ {total_days:,} 天｜預估 {total_days * 2 * SLEEP / 3600:.1f} 小時")
    print(f"[計畫] TWSE 自 {TWSE_START}｜TPEX 自 {TPEX_START}（更早無來源）\n",
          flush=True)

    t0 = time.time()
    n_done = 0
    for y in todo:
        rows, n_tw, n_tp, n_fail = [], 0, 0, 0
        ys = time.time()
        for d in by_year[y]:
            _maybe_pause(no_pause)
            ds = d.strftime("%Y-%m-%d")
            if ds >= TWSE_START:
                tw = _fetch_retry(fetch_prices_twse_direct, ds)
                time.sleep(SLEEP)
                if tw is not None and not tw.empty:
                    rows.append(tw)
                    n_tw += len(tw)
                else:
                    n_fail += 1
            if ds >= TPEX_START:
                tp = _fetch_retry(fetch_prices_tpex_direct, ds)
                time.sleep(SLEEP)
                if tp is not None and not tp.empty:
                    rows.append(tp)
                    n_tp += len(tp)
                else:
                    n_fail += 1
            n_done += 1

        if not rows:
            print(f"  {y}: ❌ 無任何資料（跳過，不寫檔以便重跑）", flush=True)
            continue
        df = pd.concat(rows, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates(subset=["Date", "stock_id"], keep="first")
        df.to_parquet(OUT_DIR / f"raw_{y}.parquet", index=False)

        el = time.time() - t0
        eta = (total_days - n_done) / max(n_done, 1) * el / 3600
        print(f"  {y}: {len(df):>7,} 列｜TWSE {n_tw:>7,} + TPEX {n_tp:>7,}"
              f"｜{df['stock_id'].nunique():>5,} 支｜空回應 {n_fail:>3}"
              f"｜本年 {(time.time()-ys)/60:5.1f} 分｜ETA {eta:4.1f} 小時", flush=True)

    print(f"\n✅ 完成，總耗時 {(time.time()-t0)/3600:.2f} 小時")
    status()


def status() -> None:
    files = sorted(OUT_DIR.glob("raw_*.parquet"))
    if not files:
        print("尚未開始（無 _raw_prices/raw_*.parquet）")
        return
    tot = 0
    print(f"\n[進度] 已完成 {len(files)} 個年度檔：")
    for p in files:
        d = pd.read_parquet(p, columns=["Date", "stock_id"])
        tot += len(d)
        print(f"  {p.stem}: {len(d):>7,} 列｜{d['stock_id'].nunique():>5,} 支"
              f"｜{d['Date'].min()} → {d['Date'].max()}")
    print(f"[進度] 合計 {tot:,} 列（現有 prices_raw 為 8,761,018 列可作對照）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-pause", action="store_true",
                    help="不在 19:15–20:15 每日推論時段暫停")
    a = ap.parse_args()
    if a.status:
        status()
    elif a.run:
        run(a.no_pause)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
