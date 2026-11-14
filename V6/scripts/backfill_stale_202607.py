"""
MarketMamba — 停更資料回補（2026-07-24）
==========================================
緣起：market_value_raw / per_raw / securities_raw / daytrade_raw / margin_raw
五個檔案在使用者 FinMind VIP 到期後（約 2026-04-24）陸續停更三個月。詳細診斷見
planing/資料基礎升級計畫_baseline_common扶正.md 第 6 節、CLAUDE.md 決策紀錄。

隔離原則（情境 C，會動到 production 資料源，但不直接覆蓋）：
  - per/securities/market_value 三個新來源（TWSE/MOPS 官方直連，見 fetcher.py
    2026-07-24 新增的三個函式）先寫到 _staging_202607_backfill/，不動 production
    的 per_raw.parquet / securities_raw.parquet / market_value_raw.parquet
  - margin/daytrade 用 FinMind 單股迴圈（免費額度不支援整批查詢），一樣先寫暫存區
  - macro_raw 是唯一直接覆蓋 production 的一步，因為只新增 TWII 欄位、其餘欄位
    行為不變，執行前會自動備份成 macro_raw_backup_<today>.parquet
  - 所有暫存檔案跑完後由使用者人工核對，再另外執行 merge 步驟寫回 production
    （merge 腳本待這次回補驗證過資料品質後再補，不在本次範圍內）

FinMind 免費額度 600 次/小時（官方文件確認），margin/daytrade 兩個資料集共用同一
額度，本腳本用 6.5 秒/次的保守間隔（約 550次/小時，留緩衝），並且逐股 checkpoint，
中斷後重新執行會自動跳過已完成的股票，可以放著跑一整晚不用看著。

用法（WSL、repo 根目錄執行）：
    python V6/scripts/backfill_stale_202607.py --step per
    python V6/scripts/backfill_stale_202607.py --step securities
    python V6/scripts/backfill_stale_202607.py --step market_value
    python V6/scripts/backfill_stale_202607.py --step macro
    python V6/scripts/backfill_stale_202607.py --step margin       # 長時間，可斷點續傳
    python V6/scripts/backfill_stale_202607.py --step daytrade     # 長時間，可斷點續傳
    python V6/scripts/backfill_stale_202607.py --all               # 依序全部執行
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402
from marketmamba.data.fetcher import (  # noqa: E402
    fetch_per_twse_direct,
    fetch_securities_lending_twse_direct,
    fetch_shares_outstanding_mops,
    _finmind_fetch_chunked,
    _sync_macro_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_202607")

STAGING_DIR = PROCESSED_DIR / "_staging_202607_backfill"
STAGING_DIR.mkdir(parents=True, exist_ok=True)

BACKFILL_START = "2026-04-25"   # 上次確認正常更新的隔天（04-24 是最後一筆已知正常資料）
FINMIND_SLEEP  = 6.5             # 秒/次，600 req/hr 額度留緩衝（約 550/hr）


def _missing_trading_days() -> list[str]:
    """從 prices_raw（本來就每天正常更新）取出 BACKFILL_START 起的交易日清單。"""
    prices = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet", columns=["Date"])
    prices["Date"] = pd.to_datetime(prices["Date"])
    today = date.today().strftime("%Y-%m-%d")
    days = sorted(prices.loc[
        (prices["Date"] >= pd.Timestamp(BACKFILL_START)) &
        (prices["Date"] <= pd.Timestamp(today)),
        "Date",
    ].dt.strftime("%Y-%m-%d").unique())
    return days


def _full_stock_universe() -> list[str]:
    prices = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet", columns=["stock_id"])
    ids = sorted(prices["stock_id"].astype(str).unique())
    return [s for s in ids if s.isdigit() and len(s) == 4]


# ============================================================
# Step: per_raw（TWSE BWIBBU_ALL，逐日迴圈，快）
# ============================================================
def backfill_per() -> None:
    days = _missing_trading_days()
    out_path = STAGING_DIR / "per_raw_backfill.parquet"
    frames = []
    n_ok, n_empty = 0, 0
    t0 = time.time()
    for i, d in enumerate(days):
        df = fetch_per_twse_direct(d)
        if df is not None and not df.empty:
            frames.append(df)
            n_ok += 1
        else:
            n_empty += 1
        if (i + 1) % 10 == 0:
            print(f"[per] {i+1}/{len(days)} 天完成 | ok={n_ok} empty={n_empty} "
                  f"| {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.3)   # 對 TWSE 官方端點禮貌性間隔，非嚴格額度限制

    if not frames:
        print("[per] 沒有抓到任何資料，中止", flush=True)
        return
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(out_path, index=False)
    print(f"[per] 完成：{len(result):,} 列 | {result['stock_id'].nunique()} 檔 | "
          f"{n_ok}/{len(days)} 天有資料 | 耗時 {(time.time()-t0)/60:.1f} 分 | 寫入 {out_path}",
          flush=True)


# ============================================================
# Step: securities_raw（TWSE SBL/TWT96U，逐日迴圈，快）
# ============================================================
def backfill_securities() -> None:
    days = _missing_trading_days()
    out_path = STAGING_DIR / "securities_raw_backfill.parquet"
    frames = []
    n_ok, n_empty = 0, 0
    t0 = time.time()
    for i, d in enumerate(days):
        df = fetch_securities_lending_twse_direct(d)
        if df is not None and not df.empty:
            frames.append(df)
            n_ok += 1
        else:
            n_empty += 1
        if (i + 1) % 10 == 0:
            print(f"[securities] {i+1}/{len(days)} 天完成 | ok={n_ok} empty={n_empty} "
                  f"| {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.3)

    if not frames:
        print("[securities] 沒有抓到任何資料，中止", flush=True)
        return
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(out_path, index=False)
    print(f"[securities] 完成：{len(result):,} 列 | {result['stock_id'].nunique()} 檔 | "
          f"{n_ok}/{len(days)} 天有資料 | 耗時 {(time.time()-t0)/60:.1f} 分 | 寫入 {out_path}",
          flush=True)


# ============================================================
# Step: market_value_raw（收盤價 × 已發行股數，向量化計算，不用迴圈）
# ============================================================
def backfill_market_value() -> None:
    t0 = time.time()
    shares = fetch_shares_outstanding_mops()
    if shares is None or shares.empty:
        print("[market_value] 抓不到股本資料，中止", flush=True)
        return

    days = _missing_trading_days()
    prices = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet",
                              columns=["Date", "stock_id", "Close"])
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices[prices["Date"].isin(pd.to_datetime(days))]
    prices = prices[prices["stock_id"].astype(str).str.match(r"^\d{4}$")]
    prices = prices.drop_duplicates(subset=["stock_id", "Date"], keep="last")

    merged = prices.merge(shares[["stock_id", "shares_outstanding"]], on="stock_id", how="inner")
    merged["market_value"] = merged["Close"] * merged["shares_outstanding"]
    result = merged[["Date", "stock_id", "market_value"]].dropna()

    out_path = STAGING_DIR / "market_value_raw_backfill.parquet"
    result.to_parquet(out_path, index=False)
    n_unmatched = prices["stock_id"].nunique() - merged["stock_id"].nunique()
    print(f"[market_value] 完成：{len(result):,} 列 | {result['stock_id'].nunique()} 檔（股本資料涵蓋 "
          f"{shares['stock_id'].nunique()} 檔，{n_unmatched} 檔 prices_raw 有但股本查無資料）| "
          f"耗時 {(time.time()-t0):.0f}s | 寫入 {out_path}", flush=True)


# ============================================================
# Step: macro_raw（直接重建，唯一動 production 檔案的一步，先備份）
# ============================================================
def backfill_macro() -> None:
    prod_path = PROCESSED_DIR / "macro_raw.parquet"
    backup_path = PROCESSED_DIR / f"macro_raw_backup_{date.today().strftime('%Y%m%d')}.parquet"
    if prod_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(prod_path, backup_path)
        print(f"[macro] 備份既有 macro_raw.parquet → {backup_path}", flush=True)

    before = pd.read_parquet(prod_path) if prod_path.exists() else None
    _sync_macro_data(start="2005-01-01", end=date.today().strftime("%Y-%m-%d"), force=True)
    after = pd.read_parquet(prod_path)

    print(f"[macro] 重建完成：{len(after):,} 列 | 欄位 {list(after.columns)} | "
          f"最後日期 {after['Date'].max()}", flush=True)
    if before is not None:
        missing_cols = set(before.columns) - set(after.columns)
        new_cols = set(after.columns) - set(before.columns)
        print(f"[macro] 跟舊版比較：新增欄位 {new_cols or '無'} | 消失欄位 {missing_cols or '無'}"
              "（消失的欄位如果只是 US_SOX/CNN_FearGreed/TW_Biz_Signal/FED_Rate，"
              "屬於已確認低優先度、暫不補的欄位，見 CLAUDE.md 決策紀錄）", flush=True)


# ============================================================
# Step: margin_raw / daytrade_raw（FinMind 單股迴圈，長時間、可斷點續傳）
# ============================================================
def _finmind_backfill_loop(dataset: str, out_name: str, checkpoint_name: str) -> None:
    stocks = _full_stock_universe()
    ckpt_path = STAGING_DIR / checkpoint_name
    out_path = STAGING_DIR / out_name
    done: set[str] = set()
    if ckpt_path.exists():
        done = set(json.loads(ckpt_path.read_text())["done"])
        print(f"[{dataset}] 從 checkpoint 繼續，已完成 {len(done)}/{len(stocks)} 檔", flush=True)

    today = date.today().strftime("%Y-%m-%d")
    t0 = time.time()
    n_ok, n_empty = 0, 0
    buffer = []

    def _flush():
        if not buffer:
            return
        new_df = pd.concat(buffer, ignore_index=True)
        if out_path.exists():
            old_df = pd.read_parquet(out_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_parquet(out_path, index=False)
        buffer.clear()

    # 2026-07-24 新增、同日修正：連續失敗提早中止，防止悶頭跑好幾小時才發現是被
    # FinMind 封鎖。但第一版只用「連續 None」判斷，誤殺了「這批股票剛好真的沒有
    # 融資融券/當沖資料」的正常情況（例如近期新股，掛牌後要等一段時間才有信用交易
    # 資格）——_finmind_fetch 對「封鎖」跟「查得到但沒資料」都同樣回傳 None，兩者
    # 混在一起。修正：連續失敗達門檻時，不直接放棄，而是額外送一次診斷請求，直接
    # 看 HTTP 狀態碼與訊息內容區分「真的被擋」（400/403 + register/banned 字樣）
    # 還是「單純沒資料」（200 + success + 空陣列），只有前者才真的中止。
    ABORT_AFTER_CONSECUTIVE_FAILS = 20
    consecutive_fails = 0

    def _diagnose_block(sid: str) -> bool:
        """回傳 True 代表真的被擋（該中止），False 代表只是單純沒資料（該繼續）。"""
        import requests
        from marketmamba.data.fetcher import FINMIND_BASE
        from marketmamba.config import FINMIND_TOKEN
        try:
            r = requests.get(FINMIND_BASE, params={
                "dataset": dataset, "start_date": BACKFILL_START, "end_date": today,
                "token": FINMIND_TOKEN, "data_id": sid,
            }, timeout=20)
            j = r.json()
        except Exception as e:
            print(f"[{dataset}] 診斷請求本身失敗（{e}），保守起見視為真的有問題，中止", flush=True)
            return True
        # status==200（即使 rows=0）都算正常；其餘（400/403/其他）算真的被擋
        really_blocked = j.get("status") != 200
        print(f"[{dataset}] 診斷請求（{sid}）：status={j.get('status')} msg={j.get('msg')!r} "
              f"rows={len(j.get('data', []) or [])} → {'真的被擋，中止' if really_blocked else '只是沒資料，繼續'}",
              flush=True)
        return really_blocked

    remaining = [s for s in stocks if s not in done]
    for i, sid in enumerate(remaining):
        df = _finmind_fetch_chunked(dataset, start_date=BACKFILL_START, end_date=today, stock_id=sid)
        if df is not None and not df.empty:
            if "date" in df.columns:
                df = df.rename(columns={"date": "Date"})
            df["stock_id"] = sid
            buffer.append(df)
            n_ok += 1
            consecutive_fails = 0
        else:
            n_empty += 1
            consecutive_fails += 1
            if consecutive_fails >= ABORT_AFTER_CONSECUTIVE_FAILS:
                if _diagnose_block(sid):
                    print(f"[{dataset}] ⚠️ 連續 {consecutive_fails} 檔都失敗且診斷確認被擋，提早中止。"
                          f"已處理 {i+1}/{len(remaining)} 檔，不寫入 checkpoint（這批不算數，"
                          f"下次重跑會從頭開始這批，不會誤判成「已完成」）。",
                          flush=True)
                    return
                else:
                    # 診斷確認只是沒資料，不是被擋：重設計數器，正常繼續跑下去
                    consecutive_fails = 0
        done.add(sid)

        if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
            _flush()
            ckpt_path.write_text(json.dumps({"done": sorted(done)}))
            elapsed_min = (time.time() - t0) / 60
            rate = (i + 1) / max(elapsed_min, 0.01)
            eta_min = (len(remaining) - (i + 1)) / max(rate, 0.01)
            print(f"[{dataset}] {i+1}/{len(remaining)} 檔（總進度 {len(done)}/{len(stocks)}）| "
                  f"ok={n_ok} empty={n_empty} | 已耗時 {elapsed_min:.0f} 分 | "
                  f"預估剩餘 {eta_min:.0f} 分", flush=True)

        time.sleep(FINMIND_SLEEP)

    print(f"[{dataset}] 全部完成：{len(done)}/{len(stocks)} 檔 | 寫入 {out_path}", flush=True)


def backfill_margin() -> None:
    _finmind_backfill_loop(
        "TaiwanStockMarginPurchaseShortSale",
        "margin_raw_backfill.parquet",
        "margin_backfill_checkpoint.json",
    )


def backfill_daytrade() -> None:
    _finmind_backfill_loop(
        "TaiwanStockDayTrading",
        "daytrade_raw_backfill.parquet",
        "daytrade_backfill_checkpoint.json",
    )


# ============================================================
# CLI
# ============================================================
STEPS = {
    "per": backfill_per,
    "securities": backfill_securities,
    "market_value": backfill_market_value,
    "macro": backfill_macro,
    "margin": backfill_margin,
    "daytrade": backfill_daytrade,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS.keys()))
    ap.add_argument("--all", action="store_true", help="依序執行全部步驟（per→securities→market_value→macro→margin→daytrade）")
    args = ap.parse_args()

    if args.all:
        for name, fn in STEPS.items():
            print(f"\n{'='*70}\n[STEP] {name}\n{'='*70}", flush=True)
            fn()
    elif args.step:
        STEPS[args.step]()
    else:
        ap.print_help()
