r"""
fetch_trading_status.py — 處置股 / 注意股（風控 C 類的剩餘缺口）
=================================================================
為什麼
------
《風控層基本版檢核表》C 類要求回測要能反映台股的執行限制。
2026-08-01 已用既有 OHLC 量化了**漲跌停**（影響 ±1pp 內、低頻下為正）。
剩下的三項（處置股 / 注意股 / 全額交割 / 下市）需要另接資料源，本檔補上前兩項。

**四項裡只有「處置」有真正的交易限制**：
  - 分盤集合競價（每 5 或 20 分鐘撮合一次）→ 流動性大幅下降、價差擴大
  - 預收款券 → 實務上很難照收盤價成交
  「注意」只是警示、沒有交易限制，但常是處置的前兆，一併抓來當診斷。

端點（2026-08-01 實測）
-----------------------
| 市場 | 處置 | 注意 |
|---|---|---|
| TWSE | `/rwd/zh/announcement/punish` | `/rwd/zh/announcement/notice` |
| TPEX | `/www/zh-tw/bulletin/disposal` | `/www/zh-tw/bulletin/attention` |

四個都要 `startDate`/`endDate`（**`date` 參數會回 0 列**——雷區 #14b：
宣告「無來源」之前要掃過參數名 × 格式的組合），
且 TWSE 是 `YYYYMMDD`、TPEX 是 `YYYY/MM/DD`（雷區 #2）。

實測踩到的三個坑
----------------
1. **分隔符號不同**：TWSE 的處置起迄用**全形** `～`(U+FF5E)、TPEX 用**半形** `~`(U+007E)。
   只寫一種的話另一邊會整批解析失敗，而且不會報錯（只會是 0 列）。
2. **TPEX 會回骨架列**：無資料的日子回「本日無處置資料」、代號與起訖皆為空字串
   （雷區 #4：「列數 > N」不能當有效性判準）。
3. **代號不一定是 4 位數**：實測有 `074123`（權證/ETN 之類），須套 `^\d{4}$`。

輸出
----
`Data/processed_v6/trading_status_raw.parquet`
欄位：`Date`(str) / `stock_id` / `status`（`disposal` | `attention`）/ `market` / `announced`
**已展開成逐日**——處置是一段期間（通常 10~12 個營業日），展開後下游只要用
`(Date, stock_id)` 查表就知道當天有沒有限制，不必再解析期間。

⚠️ 2026-08-04 重構：抓取與解析邏輯已移至 `marketmamba/data/fetcher.py`
----------------------------------------------------------------------
原因是每日流程需要**增量**更新，而本檔的 `build()` 是整檔重建
（逐年抓 11 年後全檔覆寫），不能直接排進 `run_daily_update`。
解析邏輯（含上述所有坑）現在住在 `fetcher.py` 當單一來源，本檔改為 import：

    fetcher.fetch_trading_status_direct(market, kind, start, end)
    fetcher.expand_trading_status_daily(recs, status, calendar)
    fetcher._catch_up_trading_status(today)      ← 每日增量，已接進 run_daily_update

本檔保留 `build()` 供**整檔重建**（首次建立、或懷疑歷史有缺時），行為與重構前相同。

用法
----
    python V6/experimental/fetch_trading_status.py --years 2015 2026   # 整檔重建
    python V6/experimental/fetch_trading_status.py --summary
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_V6 = Path(__file__).resolve().parent.parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

from marketmamba.config import PROCESSED_DIR                     # noqa: E402
from marketmamba.data.fetcher import (                           # noqa: E402
    expand_trading_status_daily,
    fetch_trading_status_direct,
)

OUT = Path(PROCESSED_DIR) / "trading_status_raw.parquet"


def build(years: range) -> None:
    from experimental.baseline_common import _filter_universe, _load_raw
    pr = _load_raw("prices_raw")
    pr = _filter_universe(pr)
    calendar = pd.DatetimeIndex(sorted(pr["Date"].unique()))     # 真實交易日曆
    print(f"[ts] 交易日曆 {len(calendar)} 天（{calendar[0].date()} → {calendar[-1].date()}）",
          flush=True)

    frames = []
    for y in years:
        y0, y1 = f"{y}-01-01", f"{y}-12-31"
        for kind in ("disposal", "attention"):
            a = fetch_trading_status_direct("twse", kind, y0, y1)
            b = fetch_trading_status_direct("tpex", kind, y0, y1)
            print(f"[ts] {y} {kind:10s} twse {len(a):>4} 筆｜tpex {len(b):>4} 筆", flush=True)
            for recs in (a, b):
                if recs:
                    frames.append(expand_trading_status_daily(recs, kind, calendar))

    if not frames:
        raise SystemExit("❌ 一筆都沒抓到")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "stock_id", "status"], keep="first")
    df = df.sort_values(["Date", "stock_id", "status"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"\n✅ [ts] {len(df):,} 列 → {OUT}", flush=True)
    summary(df)


def summary(df: pd.DataFrame | None = None) -> None:
    if df is None:
        if not OUT.exists():
            raise SystemExit(f"❌ {OUT} 不存在")
        df = pd.read_parquet(OUT)
    print("\n" + "=" * 66)
    for st, g in df.groupby("status"):
        yr = pd.to_datetime(g["Date"]).dt.year
        print(f"[{st}] {len(g):,} 個「股票×日」｜{g['stock_id'].nunique():,} 支｜"
              f"{g['Date'].min()} → {g['Date'].max()}")
        print(f"    逐年：" + " ".join(f"{y}:{n:,}" for y, n in yr.value_counts().sort_index().items()))
        print(f"    市場：{dict(g['market'].value_counts())}")
        # 處置期間長度健檢——制度上是 10~12 個營業日，明顯偏離代表解析錯了
        if st == "disposal":
            per = g.groupby(["stock_id", "announced"]).size()
            print(f"    每次處置的天數 min/median/max = {per.min()}/{int(per.median())}/{per.max()}"
                  f"（制度上約 10~12 個營業日）")
    print("=" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2015, 2026],
                    help="起訖年（含）")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        build(range(a.years[0], a.years[1] + 1))
