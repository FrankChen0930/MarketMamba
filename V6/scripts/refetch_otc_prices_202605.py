"""
MarketMamba — 重抓覆蓋 2026-05-25 起的上櫃股價（2026-07-27）
============================================================
緣起：舊的 `fetch_prices_tpex_direct` 打 `openapi/v1/tpex_mainboard_daily_close_quotes`，
該端點**完全忽略 date 參數**（五種格式含不傳，回傳逐位元相同的當天資料），而舊程式
又把 `"Date": date_str` 硬寫進每一列 → 「抓今天的報價、蓋上你要求的日期」寫進
prices_raw，且不報錯。該路徑是 run_daily_update 在 yfinance OTC 覆蓋率 <30% 時的
fallback，2026-05 起持續觸發。

實測污染程度（prices_raw 的 OTC 列 vs 交易所真值，2026-07-24）：
    Volume 比值 median 0.93（p10 0.38 / p90 2.20），248/835 檔偏差 >2 倍
    Close 偏差 >2% 214/835 檔
對照組（同日 TWSE MI_INDEX 來源）：Volume median 1.0011、Close 零偏差 → 只有 OTC 要修。

本腳本用已修好的 `fetch_prices_tpex_direct`（改打 `/www/zh-tw/afterTrading/otc`，
會正確遵守日期、含回傳日期核對）逐日重抓並覆蓋 prices_raw 的對應列。

範圍與界線：
  - 只覆蓋「該日交易所回傳的上櫃股票」，TWSE 來源的列一律不動（已驗證乾淨）
  - 交易所回傳的是**未還原價**。但實測 prices_raw 近期（07-16/24/27）本來就與
    交易所原始價完全一致（Close 零偏差），代表增量段本來就是未還原 →
    覆蓋不會讓還原基準變得更糟，只是把日期錯置修正。還原接縫要另案處理
    （全量重抓或自建還原因子表）。
  - **不會**把 2026-05-25 消失的 353 支救回來：實測（依 distinct stock_id）其中
    341 支是興櫃（stock_info type='emerging'）、6 支 tpex、1 支 twse、5 支 stock_info
    查無，合計 353。原因是 yfinance 原本涵蓋興櫃、直連端點不涵蓋，與本 bug 無關，
    需另案決定要不要把興櫃排除得一致。
    ⚠️ stock_info 每股可能有多列（一列一個產業分類，全檔 4,097 列 / 3,086 支），
       統計時務必先 drop_duplicates("stock_id")，否則算到的是列數而非股票數。

用法（repo 根目錄）：
    python V6/scripts/refetch_otc_prices_202605.py --dry-run
    python V6/scripts/refetch_otc_prices_202605.py --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402
from marketmamba.data.fetcher import fetch_prices_tpex_direct  # noqa: E402

PRICE_PATH = PROCESSED_DIR / "prices_raw.parquet"
START = "2026-05-25"
SLEEP_SEC = 1.1
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.print_help()
        return

    pr = pd.read_parquet(PRICE_PATH)
    pr["Date"] = pd.to_datetime(pr["Date"])
    end = pr["Date"].max()
    days = [d.strftime("%Y-%m-%d")
            for d in pd.date_range(START, end) if d.dayofweek < 5]
    if args.limit:
        days = days[: args.limit]
    print(f"[範圍] {START} → {end.date()}，候選平日 {len(days)} 天"
          f"（非交易日由 fetcher 回 None 自動跳過）", flush=True)

    frames, empty = [], []
    t0 = time.time()
    for i, d in enumerate(days, 1):
        df = fetch_prices_tpex_direct(d)
        if df is None or df.empty:
            empty.append(d)
        else:
            frames.append(df)
        if i % 10 == 0 or i == len(days):
            print(f"  進度 {i}/{len(days)}｜取得 {sum(len(f) for f in frames):,} 列"
                  f"｜非交易日 {len(empty)}｜{time.time()-t0:.0f}s", flush=True)
        time.sleep(SLEEP_SEC)

    if not frames:
        print("❌ 沒有取得任何資料")
        return
    new = pd.concat(frames, ignore_index=True)
    new["Date"] = pd.to_datetime(new["Date"])
    new = new.drop_duplicates(subset=["Date", "stock_id"], keep="last")

    print(f"\n[取得] {len(new):,} 列｜{new['Date'].nunique()} 個交易日｜"
          f"{new['stock_id'].nunique()} 支上櫃股")
    print(f"[取得] 非交易日 {len(empty)} 天：{empty}")

    # ── 比對：現有值 vs 重抓值，量化污染程度 ──
    key = ["Date", "stock_id"]
    cmp_ = pr.merge(new, on=key, how="inner", suffixes=("_old", "_new"))
    print(f"\n[比對] 可比對 {len(cmp_):,} 列（prices_raw 已有、且本次重抓到）")
    for c in OHLCV:
        o, n = cmp_[f"{c}_old"].astype(float), cmp_[f"{c}_new"].astype(float)
        ok = (o > 0) & (n > 0)
        rel = (n[ok] / o[ok] - 1).abs()
        print(f"    {c:<7} 完全相同 {int((o == n).sum()):>6,} ({float((o == n).mean()):>6.1%})"
              f"｜偏差>2% {int((rel > 0.02).sum()):>6,}"
              f"｜偏差>50% {int((rel > 0.5).sum()):>6,}"
              f"｜median |Δ| {float(rel.median()):.4%}")

    only_new = len(new) - len(cmp_)
    print(f"[比對] 重抓到但 prices_raw 沒有的列 = {only_new:,}（會新增）")
    # prices_raw 有、但該日重抓不到的上櫃股（可疑列：可能是錯日期寫進去的）
    otc_universe = set(new["stock_id"])
    win = pr[(pr["Date"] >= pd.Timestamp(START)) & (pr["Date"] <= end)]
    suspect = win[win["stock_id"].isin(otc_universe)].merge(
        new[key].assign(_hit=1), on=key, how="left")
    n_suspect = int(suspect["_hit"].isna().sum())
    print(f"[比對] prices_raw 內屬上櫃宇宙、但該日交易所無資料的列 = {n_suspect:,}"
          f"（該股當日未交易或停牌，保留不動）")

    if args.dry_run:
        print("\n--dry-run：未寫檔。", flush=True)
        return

    backup = PROCESSED_DIR / f"prices_raw_backup_otc_refetch_{date.today():%Y%m%d}.parquet"
    if not backup.exists():
        pr.to_parquet(backup, index=False)
        print(f"\n[備份] → {backup.name}", flush=True)

    # 覆蓋：刪掉同 (Date, stock_id) 的舊列，再併入重抓列
    idx_new = pd.MultiIndex.from_frame(new[key])
    idx_old = pd.MultiIndex.from_frame(pr[key])
    keep = ~idx_old.isin(idx_new)
    out = pd.concat([pr[keep], new[pr.columns]], ignore_index=True)
    n_replaced = int((~keep).sum())
    out = out.drop_duplicates(subset=key, keep="last")
    out = out.sort_values(["stock_id", "Date"]).reset_index(drop=True)
    out.to_parquet(PRICE_PATH, index=False)
    print(f"[寫回] 覆蓋 {n_replaced:,} 列、新增 {len(new) - n_replaced:,} 列"
          f"｜{len(pr):,} → {len(out):,} 列", flush=True)
    print("✅ 完成", flush=True)


if __name__ == "__main__":
    main()
