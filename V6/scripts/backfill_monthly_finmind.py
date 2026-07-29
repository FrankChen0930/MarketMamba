"""
MarketMamba — 月營收 / 季財報的初始回補（2026-07-29）
======================================================
`revenue_raw` 停更 118 天、`financials_raw` 停更 119 天。
每日更新已加入**滾動逐股補齊**（`_catch_up_monthly`，每天 120/60 支），
但要從停更狀態追回來需要約 16 天；本腳本一次跑完，之後交給每日更新維持。

【為什麼是逐股，不能一次抓全市場】
  2026-07-29 實測 FinMind 免費層（register）對這兩個 dataset 的限制是**形狀**不是速率：

      不帶 data_id（全市場）→ HTTP 400 "Your level is register"
      帶 data_id（單股）    → HTTP 200 success

  所以只能逐股查 ~1,900 次。以 0.6 秒間隔估約 20 分鐘/資料源。
  這也是「不訂閱 FinMind VIP」這個決定的實際代價所在——
  但月/季頻資料一年只更新 12/4 次，過夜跑一次完全可接受。

【只補現役宇宙】已下市股票的資料永遠停在下市那天，補了也不會變，
  且會佔住「最舊」的名次。以 `prices_raw` 最後一個交易日的宇宙為準。

用法（repo 根目錄）：
    python V6/scripts/backfill_monthly_finmind.py --run
    python V6/scripts/backfill_monthly_finmind.py --run --only revenue
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402
from marketmamba.data.fetcher import _catch_up_monthly  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("backfill_monthly")

SPECS = {
    "revenue": ("revenue_raw", "TaiwanStockMonthRevenue"),
    "financials": ("financials_raw", "TaiwanStockFinancialStatements"),
}
BATCH = 150          # 每輪處理的股票數（_catch_up_monthly 內部會挑最舊的）
MAX_ROUNDS = 30      # 安全上限，避免無限迴圈


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", choices=list(SPECS))
    a = ap.parse_args()
    if not a.run:
        ap.print_help()
        return

    today = date.today().strftime("%Y-%m-%d")
    todo = [a.only] if a.only else list(SPECS)

    for key in todo:
        name, dataset = SPECS[key]
        path = PROCESSED_DIR / f"{name}.parquet"
        backup = PROCESSED_DIR / f"{name}_backup_{date.today():%Y%m%d}.parquet"
        if path.exists() and not backup.exists():
            pd.read_parquet(path).to_parquet(backup, index=False)
            log.info(f"[備份] → {backup.name}")

        log.info("=" * 74)
        log.info(f"■ {name}（dataset={dataset}）")
        log.info("=" * 74)
        t0 = time.time()
        total = 0
        for rnd in range(1, MAX_ROUNDS + 1):
            n = _catch_up_monthly(name, dataset, today, max_stocks=BATCH)
            total += n
            log.info(f"  第 {rnd} 輪：淨增 {n:,} 列｜累計 {total:,}"
                     f"｜已耗時 {(time.time() - t0) / 60:.1f} 分")
            if n == 0:
                # 連續一輪毫無新增 = 最舊的那批已經是最新的，全部追平
                log.info("  本輪無新增，視為已追平，結束")
                break
        d = pd.read_parquet(path)
        dcol = "date" if "date" in d.columns else "Date"
        d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
        log.info(f"✅ {name}: {len(d):,} 列｜{d['stock_id'].nunique():,} 支"
                 f"｜最新 {d[dcol].max().date()}｜總耗時 "
                 f"{(time.time() - t0) / 60:.1f} 分")


if __name__ == "__main__":
    main()
