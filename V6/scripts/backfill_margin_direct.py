"""
MarketMamba — margin_raw 缺口回補（交易所直連，2026-07-27）
============================================================
背景：margin_raw 自 2026-04-24 停更（FinMind VIP 到期）。本腳本用 2026-07-27
新增的 `fetch_margin_direct()`（TWSE MI_MARGN + TPEX margin/balance）逐日回補。

為什麼不用既有的 FinMind staging 檔（`_staging_202607_backfill/margin_raw_backfill.parquet`）：
該檔帶著 FinMind 對上櫃股「券賣/券買標反」的 bug（見 `fix_margin_short_swap.py`），
直接 merge 會把已修好的欄位語意再污染回去。直連版本恆等式 100% 成立。

速度：交易所端點是「逐日整批」，一天約 2.7 秒（TWSE + TPEX 兩次請求 + 禮貌間隔），
60 個交易日約 3 分鐘——比 FinMind 免費層的逐股迴圈（3.6 小時）快兩個數量級。

用法（repo 根目錄）：
    python V6/scripts/backfill_margin_direct.py --dry-run
    python V6/scripts/backfill_margin_direct.py --apply
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
from marketmamba.data.fetcher import fetch_margin_direct  # noqa: E402

MARGIN_PATH = PROCESSED_DIR / "margin_raw.parquet"
PRICE_PATH = PROCESSED_DIR / "prices_raw.parquet"
SLEEP_SEC = 1.2          # 對交易所端點的禮貌間隔（每天兩次請求）


def _target_days() -> list[str]:
    """
    缺口 = margin_raw 最後一天之後的所有「平日」，扣掉 margin_raw 已有的。

    ⚠️ 刻意**不用 prices_raw 當交易日曆**：2026-07-27 查證發現 prices_raw 自己就漏了
       2026-04-27、04-28、07-10 三個真實交易日（兩個獨立來源在那幾天都有資料，
       且非國定假日）。拿它當基準會把洞原封不動複製到每一個下游資料源。
       改成掃所有平日、讓交易所 API 自己判斷——非交易日 fetcher 會回 None，
       多打幾次請求的成本可忽略，而且日曆會自我修正。
    """
    mg = pd.read_parquet(MARGIN_PATH, columns=["Date"])
    mg_days = set(pd.to_datetime(mg["Date"]).dt.strftime("%Y-%m-%d"))
    start = pd.to_datetime(max(mg_days)) + pd.Timedelta(days=1)
    pr = pd.read_parquet(PRICE_PATH, columns=["Date"])
    price_days = set(pd.to_datetime(pr["Date"]).dt.strftime("%Y-%m-%d"))
    end = pd.to_datetime(max(price_days))

    weekdays = [d.strftime("%Y-%m-%d")
                for d in pd.date_range(start, end) if d.dayofweek < 5]
    days = [d for d in weekdays if d not in mg_days]
    holes = [d for d in days if d not in price_days]
    if holes:
        print(f"[注意] 這些平日 prices_raw 也沒有（其中非國定假日者為 prices_raw 的缺漏，"
              f"需另外回補）：{holes}", flush=True)
    return days


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 天（測試用）")
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.print_help()
        return

    days = _target_days()
    if args.limit:
        days = days[: args.limit]
    print(f"[缺口] margin_raw 缺 {len(days)} 個交易日"
          f"{f'：{days[0]} → {days[-1]}' if days else ''}", flush=True)
    if not days:
        print("沒有缺口，結束。")
        return

    frames, failed = [], []
    t0 = time.time()
    for i, d in enumerate(days, 1):
        df = fetch_margin_direct(d)
        if df is None or df.empty:
            failed.append(d)
        else:
            frames.append(df)
        if i % 10 == 0 or i == len(days):
            print(f"  進度 {i}/{len(days)}｜已取得 {sum(len(f) for f in frames):,} 列"
                  f"｜失敗 {len(failed)} 天｜{time.time()-t0:.0f}s", flush=True)
        time.sleep(SLEEP_SEC)

    if not frames:
        print("❌ 完全沒有取得資料，結束。")
        return

    new = pd.concat(frames, ignore_index=True)
    new["Date"] = pd.to_datetime(new["Date"])

    # 健檢（規則 7：數值明確輸出）
    per_day = new.groupby("Date").size()
    print(f"\n[健檢] 取得 {len(new):,} 列 | {new['Date'].nunique()} 個交易日 | "
          f"{new['stock_id'].nunique()} 支")
    print(f"[健檢] 每日檔數 min/median/max = "
          f"{per_day.min()}/{int(per_day.median())}/{per_day.max()}")
    d_bal = new.sort_values(["stock_id", "Date"]).groupby("stock_id")["Short_Balance"].diff()
    net = (new.sort_values(["stock_id", "Date"])["Short_Sale"]
           - new.sort_values(["stock_id", "Date"])["Short_Cover"])
    ok = d_bal.notna() & ((d_bal != 0) | (net != 0))
    print(f"[健檢] corr(ΔShort_Balance, Sale-Cover) = {d_bal[ok].corr(net[ok]):+.4f}"
          f"（應為正，代表欄位語意與修好的歷史一致）")
    print(f"[健檢] NaN 總數 = {int(new.isna().sum().sum())} | 負值列數 = "
          f"{int((new[['Margin_Purchase','Margin_Repay','Short_Sale','Short_Cover',
                        'Margin_Balance','Short_Balance']] < 0).any(axis=1).sum())}")
    if failed:
        print(f"[健檢] ⚠️ 未取得的日期（{len(failed)} 天）：{failed}")

    if args.dry_run:
        print("\n--dry-run：未寫檔。", flush=True)
        return

    old = pd.read_parquet(MARGIN_PATH)
    old["Date"] = pd.to_datetime(old["Date"])
    backup = PROCESSED_DIR / f"margin_raw_backup_before_backfill_{date.today():%Y%m%d}.parquet"
    if not backup.exists():
        old.to_parquet(backup, index=False)
        print(f"\n[備份] → {backup.name}", flush=True)

    merged = pd.concat([old, new[old.columns]], ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    merged = merged.sort_values(["stock_id", "Date"]).reset_index(drop=True)
    merged.to_parquet(MARGIN_PATH, index=False)
    print(f"[寫回] {len(old):,} + {len(new):,} → {len(merged):,} 列"
          f"（去重 {n_before - len(merged):,}）| 最新 {merged['Date'].max().date()}", flush=True)
    print("✅ 完成", flush=True)


if __name__ == "__main__":
    main()
