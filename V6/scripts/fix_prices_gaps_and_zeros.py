"""
MarketMamba — prices_raw 缺漏交易日回補 + Close<=0 損壞列清除（2026-07-28）
==========================================================================
兩件事一起做，因為都只動 prices_raw、共用同一次備份。

【1】回補缺漏的真實交易日 2026-04-27、2026-04-28
     發現經過：回補 margin 時用 prices_raw 當交易日曆，發現缺口起點是 04-29。
     交叉確認這兩天是真交易日——交易所的 margin/daytrade 端點都有資料、非國定假日。
     （對照：2026-07-10 經多端點確認**確實不是**交易日，prices_raw 缺它是正確的。）

     來源選擇：用**交易所直連**而非 yfinance。
     判定依據（2026-07-28 實測）：prices_raw 的 2026-04-24 資料與交易所原始價
     **逐檔相同**（1,081 檔比值 median 1.0000、無任何偏差），例如 2412 中華電
     為 135.50 而它 07-09 才除息跌 6 元——若已按現在還原，4 月價應約 129.5。
     → 該區段是「以最後一次全量同步（約 2026-04-24）為準」之後的**原始價**，
       用交易所回補才與鄰居同基準；用 yfinance（auto_adjust 以今天為準）反而會
       在那兩天造成人為的價格斷階。

【2】清除 Close<=0 的 329 列（2026-04-30 ~ 05-22、122 支）
     零成交日被寫入 0 價。留著會讓 pct_change 產生 ±inf 並在任何跨越該區間的
     rolling 窗口擴散——2026 年 515 筆 |單日報酬|>40% 中有 422 筆（82%）源自此。
     推論端 `_sanitize()` 已在讀取時剔除，但訓練/回測若直接讀 raw 仍會踩到。

     ⚠️ Volume==0 的 229 列**不清**：無成交但有參考價是合法狀態（停牌/無量），
     刪掉會製造假的交易日缺口。只清定義上不可能為真的 Close<=0。

用法（repo 根目錄）：
    python V6/scripts/fix_prices_gaps_and_zeros.py --dry-run
    python V6/scripts/fix_prices_gaps_and_zeros.py --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
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

PRICE_PATH = PROCESSED_DIR / "prices_raw.parquet"
MISSING_DAYS = ["2026-04-27", "2026-04-28"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.print_help()
        return

    pr = pd.read_parquet(PRICE_PATH)
    pr["Date"] = pd.to_datetime(pr["Date"])
    print(f"[讀入] {len(pr):,} 列｜{pr['Date'].min().date()} → {pr['Date'].max().date()}",
          flush=True)

    # ── 1. 回補缺漏交易日 ────────────────────────────────────────────────
    have = set(pr["Date"].dt.strftime("%Y-%m-%d"))
    todo = [d for d in MISSING_DAYS if d not in have]
    print(f"\n[1] 缺漏交易日：{MISSING_DAYS}｜實際待補 {todo or '無'}")
    frames = []
    for d in todo:
        tw = fetch_prices_twse_direct(d)
        time.sleep(1.2)
        tp = fetch_prices_tpex_direct(d)
        time.sleep(1.2)
        got = [x for x in (tw, tp) if x is not None and not x.empty]
        if not got:
            print(f"    {d}: ❌ 交易所無資料（可能不是交易日，請複查）")
            continue
        day = pd.concat(got, ignore_index=True)
        day = day.drop_duplicates(subset=["Date", "stock_id"], keep="first")
        print(f"    {d}: TWSE {0 if tw is None else len(tw)} + "
              f"TPEX {0 if tp is None else len(tp)} → 去重後 {len(day)} 檔")
        frames.append(day)

    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not new.empty:
        new["Date"] = pd.to_datetime(new["Date"])
        # 與鄰近交易日的檔數對照（規則 7：數值明確輸出）
        near = pr[pr["Date"].isin(pd.to_datetime(["2026-04-24", "2026-04-29"]))]
        cnt = near.groupby(near["Date"].dt.strftime("%Y-%m-%d"))["stock_id"].nunique()
        print(f"    鄰近日檔數對照：{dict(cnt)}｜本次回補 "
              f"{dict(new.groupby(new['Date'].dt.strftime('%Y-%m-%d'))['stock_id'].nunique())}")

    # ── 2. Close<=0 損壞列 ──────────────────────────────────────────────
    bad = pd.to_numeric(pr["Close"], errors="coerce") <= 0
    n_bad = int(bad.sum())
    print(f"\n[2] Close<=0 損壞列：{n_bad} 列 / {pr.loc[bad, 'stock_id'].nunique()} 支")
    if n_bad:
        rng = pr.loc[bad, "Date"]
        print(f"    日期範圍 {rng.min().date()} → {rng.max().date()}")
    n_vol0 = int((pd.to_numeric(pr["Volume"], errors="coerce") == 0).sum())
    print(f"    （Volume==0 的 {n_vol0} 列刻意保留：無成交但有參考價是合法狀態）")

    if args.dry_run:
        print("\n--dry-run：未寫檔。", flush=True)
        return

    backup = PROCESSED_DIR / f"prices_raw_backup_gapfix_{date.today():%Y%m%d}.parquet"
    if not backup.exists():
        pr.to_parquet(backup, index=False)
        print(f"\n[備份] → {backup.name}", flush=True)

    out = pr[~bad]
    if not new.empty:
        out = pd.concat([out, new[pr.columns]], ignore_index=True)
    out = out.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    out = out.sort_values(["stock_id", "Date"]).reset_index(drop=True)
    out.to_parquet(PRICE_PATH, index=False)
    print(f"[寫回] {len(pr):,} − {n_bad}（損壞）+ {len(new):,}（回補） = {len(out):,} 列",
          flush=True)
    print(f"[驗證] Close<=0 剩 "
          f"{int((pd.to_numeric(out['Close'], errors='coerce') <= 0).sum())} 列"
          f"｜交易日數 {out['Date'].nunique():,}", flush=True)
    print("✅ 完成", flush=True)


if __name__ == "__main__":
    main()
