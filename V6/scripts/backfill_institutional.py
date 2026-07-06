"""
一次性回補 institutional_raw.parquet 的機構資料缺口
=====================================================
背景（2026-07-06 診斷）：
  - TWSE T86 缺 selectType 參數 → 2026-04-25 起每日更新只寫進水泥類 7 支
  - TPEX 舊端點改版只回 HTML → 上櫃機構資料完全沒進來
  fetcher.py 已修復；本腳本用修復後的 fetcher 重抓缺口區間。

設計：**批次模式**——先把所有日期抓進記憶體，最後對 parquet 做「單次」
讀＋去重＋寫。不逐日重寫 148MB 檔案（逐日版曾因超時被砍在寫檔中途）。

用法（Windows 或 WSL 皆可，需網路）：
    python V6/scripts/backfill_institutional.py [--start 2026-04-25] [--end YYYY-MM-DD]

執行前會自動備份 institutional_raw.parquet → *_backup_YYYYMMDD.parquet。
"""
import argparse
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

V6_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V6_DIR))

import pandas as pd

from marketmamba.config import PROCESSED_DIR
from marketmamba.data.fetcher import (
    fetch_institutional_tpex,
    fetch_institutional_twse,
)

INST_PATH = PROCESSED_DIR / "institutional_raw.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-04-25")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if not INST_PATH.exists():
        print(f"❌ {INST_PATH} 不存在，中止")
        sys.exit(1)

    backup = INST_PATH.with_name(
        f"institutional_raw_backup_{date.today().strftime('%Y%m%d')}.parquet"
    )
    if not backup.exists():
        shutil.copy2(INST_PATH, backup)
        print(f"備份 → {backup.name}")

    # ── 階段 1：全部抓進記憶體（不動 parquet）─────────────────────────────
    frames: list[pd.DataFrame] = []
    fetched_dates: list[str] = []
    n_days = n_skip = 0
    d = start
    while d <= end:
        if d.weekday() >= 5:   # 週末直接跳過
            d += timedelta(days=1)
            continue
        n_days += 1
        ds = d.strftime("%Y-%m-%d")
        df_tse = fetch_institutional_twse(ds)
        df_otc = fetch_institutional_tpex(ds)
        day_frames = [x for x in (df_tse, df_otc) if x is not None and not x.empty]
        if day_frames:
            inst = pd.concat(day_frames, ignore_index=True)
            inst["Date"] = ds
            frames.append(inst)
            fetched_dates.append(ds)
            print(f"  {ds}: TWSE {len(df_tse) if df_tse is not None else 0} + "
                  f"TPEX {len(df_otc) if df_otc is not None else 0} = {len(inst)} rows ✓", flush=True)
        else:
            n_skip += 1
            print(f"  {ds}: 無資料（休市日或尚未發布）", flush=True)
        time.sleep(2.0)   # 對交易所 API 保持禮貌
        d += timedelta(days=1)

    if not frames:
        print("沒有抓到任何資料，parquet 未變動")
        return

    # ── 階段 2：單次讀＋去重＋寫 ───────────────────────────────────────────
    new_data = pd.concat(frames, ignore_index=True)
    print(f"\n抓取完成：{len(fetched_dates)} 天 / {len(new_data):,} rows，開始合併 parquet…", flush=True)

    df_old = pd.read_parquet(INST_PATH)
    df_old["Date"] = pd.to_datetime(df_old["Date"]).dt.strftime("%Y-%m-%d")
    df_old = df_old[~df_old["Date"].isin(set(fetched_dates))]   # 移除將被替換的日期（含水泥股殘料）
    merged = pd.concat([df_old, new_data], ignore_index=True)

    tmp = INST_PATH.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp)
    tmp.replace(INST_PATH)   # 原子替換，寫一半被砍也不會弄壞原檔
    print(f"寫入完成：{len(merged):,} rows（原 {len(df_old):,} + 新 {len(new_data):,}）")

    # ── 健檢 ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(INST_PATH, columns=["Date", "stock_id"])
    df["Date"] = pd.to_datetime(df["Date"])
    recent = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=14)]
    print(f"健檢：嘗試 {n_days} 個平日、跳過 {n_skip} 天；"
          f"最近 14 天 unique stocks = {recent['stock_id'].nunique()}（修復前只有 7），"
          f"max date = {recent['Date'].max().date()}")


if __name__ == "__main__":
    main()
