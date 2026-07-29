"""
MarketMamba — daytrade_raw 全歷史重建（交易所直連，2026-07-27）
================================================================
為什麼是「重建」而不是「回補缺口」：
production 的 `daytrade_raw.parquet` 的 `Day_Trade_Volume` 在 **2014–2026 全部
4,263,330 列都是 0**（非零列數 0，逐年檢查皆 min=max=0）。整欄自始就沒有資料，
不是 2026-04-24 停更造成的——FinMind `TaiwanStockDayTrading` 的 `BuyAfterSale`
欄位為空，舊管線一路寫 0 進去。因此要重抓全歷史，不是補三個月。

特徵定義：
    Day_Trade_Volume = 當日沖銷交易成交股數 / 該股當日成交股數
已驗證（2026-07-24 / 07-09 兩天）：
  - 個股比率 TWSE max 0.9305、TPEX max 1.0000，**沒有任何一檔 > 1**
    → 單一個股的當沖股數不會超過其成交股數，單位與計算方式一致
  - median 0.23–0.27，符合台股當沖比的已知水位
  - prices_raw 的 Volume 單位已對 TWSE MI_INDEX 成交股數驗證（median 比值 0.9989）
未能精確對上的部分（誠實揭露）：交易所公布的「當沖占市場比重」約為本計算的一半，
其分母採雙邊計算慣例，屬市場總計的記帳定義差異，不影響個股比率的有效性。

⚠️ V6.1 影響：這一維在 V6.1 訓練時**恆為 0**，對應的 proj_B 權重從未被有效訓練。
   填入真值後，推論會把真實數值乘上一個未訓練的權重 → 會注入噪音。
   目標是「資料先修對」，但這一維在重訓前建議留意；若要讓 V6.1 維持原行為，
   可在推論端暫時把該欄歸零（一行 mask），不需要動資料。

用法（repo 根目錄；約 1.5–2 小時，可中斷續傳）：
    python V6/scripts/backfill_daytrade_direct.py --probe        # 先驗 5 天
    python V6/scripts/backfill_daytrade_direct.py --apply
"""
from __future__ import annotations

import argparse
import json
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
    daytrade_shares_to_ratio,
    fetch_daytrade_direct,
)

DAYTRADE_PATH = PROCESSED_DIR / "daytrade_raw.parquet"
PRICE_PATH = PROCESSED_DIR / "prices_raw.parquet"
STAGING = PROCESSED_DIR / "_staging_daytrade_rebuild"
CKPT = STAGING / "checkpoint.json"
START_DATE = "2014-01-06"        # 現股當沖上路；production 舊資料也是從這天開始
SLEEP_SEC = 1.0                  # 每天兩次請求；對交易所保持禮貌但不過慢
FLUSH_EVERY = 50                 # 每 N 天落盤一次，中斷可續


def _load_ckpt() -> dict:
    if CKPT.exists():
        return json.loads(CKPT.read_text(encoding="utf-8"))
    return {"done": [], "empty": []}


def _save_ckpt(ck: dict) -> None:
    CKPT.write_text(json.dumps(ck, ensure_ascii=False), encoding="utf-8")


def _candidate_days(start: str, end: str) -> list[str]:
    """所有平日；非交易日由 fetcher 回 None 自然跳過（刻意不用 prices_raw 當日曆，
    因為它自己就漏了 2026-04-27/04-28 兩個真實交易日）。"""
    return [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, end) if d.dayofweek < 5]


def _run(days: list[str], probe: bool = False) -> pd.DataFrame:
    STAGING.mkdir(parents=True, exist_ok=True)
    ck = _load_ckpt()
    done, empty = set(ck["done"]), set(ck["empty"])
    todo = [d for d in days if d not in done and d not in empty]
    print(f"[計畫] 候選平日 {len(days)}｜已完成 {len(done)}｜已知非交易日 {len(empty)}"
          f"｜待處理 {len(todo)}", flush=True)

    buf: list[pd.DataFrame] = []
    part_idx = len(list(STAGING.glob("part_*.parquet")))
    t0 = time.time()
    for i, d in enumerate(todo, 1):
        df = fetch_daytrade_direct(d)
        if df is None or df.empty:
            empty.add(d)
        else:
            buf.append(df)
            done.add(d)
        if i % FLUSH_EVERY == 0 or i == len(todo):
            if buf:
                pd.concat(buf, ignore_index=True).to_parquet(
                    STAGING / f"part_{part_idx:04d}.parquet", index=False)
                part_idx += 1
                buf = []
            _save_ckpt({"done": sorted(done), "empty": sorted(empty)})
            el = time.time() - t0
            eta = el / i * (len(todo) - i) / 60
            print(f"  進度 {i}/{len(todo)}｜交易日 {len(done)}｜非交易日 {len(empty)}"
                  f"｜{el/60:.1f} 分｜ETA {eta:.0f} 分", flush=True)
        time.sleep(SLEEP_SEC)

    parts = sorted(STAGING.glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--probe", action="store_true", help="只抓最近 5 個平日驗證")
    args = ap.parse_args()
    if not (args.apply or args.probe):
        ap.print_help()
        return

    pr = pd.read_parquet(PRICE_PATH, columns=["Date"])
    end = pd.to_datetime(pr["Date"]).max().strftime("%Y-%m-%d")
    days = _candidate_days(START_DATE, end)
    if args.probe:
        days = days[-5:]
        print(f"[probe] 只驗最近 5 個平日：{days}", flush=True)

    shares = _run(days, probe=args.probe)
    if shares.empty:
        print("❌ 沒有取得任何資料")
        return

    ratio = daytrade_shares_to_ratio(shares)
    print(f"\n[健檢] 當沖股數 {len(shares):,} 列 → 換算比率後 {len(ratio):,} 列"
          f"（差額 = prices_raw 查無成交量或成交量為 0 的列，刻意剔除不補 0）")
    print(f"[健檢] 交易日數 {ratio['Date'].nunique()}｜股票數 {ratio['stock_id'].nunique()}")
    r = ratio["Day_Trade_Volume"]
    print(f"[健檢] Day_Trade_Volume min={r.min():.4f} median={r.median():.4f} "
          f"max={r.max():.4f}｜>1 的列數={int((r > 1).sum())}"
          f"｜非零比例={float((r > 0).mean()):.2%}")

    # 理論上界守門：當沖股數不可能超過該股當日成交股數，>1 代表分母（prices_raw
    # 的 Volume）有問題。剔除而非 clip——feature_engineer._merge_daytrade 的
    # .clip(0,1) 會把 247 這種值靜默變成看似合理的 1.0（編造值比缺值更糟）。
    n_over = int((ratio["Day_Trade_Volume"] > 1).sum())
    if n_over:
        by_year = (ratio[ratio["Day_Trade_Volume"] > 1]
                   .assign(y=lambda d: d["Date"].dt.year).groupby("y").size())
        print(f"[守門] 剔除比率 >1 的 {n_over:,} 列（{n_over/len(ratio):.3%}），依年："
              f"{dict(by_year)}")
        ratio = ratio[ratio["Day_Trade_Volume"] <= 1]
    yearly = ratio.assign(y=ratio["Date"].dt.year).groupby("y")["Day_Trade_Volume"].agg(
        ["count", "median", "max"])
    print("[健檢] 依年（median 應隨當沖制度放寬而上升）：")
    print(yearly.to_string())

    if args.probe:
        print("\n--probe：未寫檔。", flush=True)
        return

    old = pd.read_parquet(DAYTRADE_PATH)
    backup = PROCESSED_DIR / f"daytrade_raw_backup_{date.today():%Y%m%d}.parquet"
    if not backup.exists():
        old.to_parquet(backup, index=False)
        print(f"\n[備份] → {backup.name}（舊檔 Day_Trade_Volume 全為 0，僅留稽核用）",
              flush=True)
    out = ratio.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    out = out.sort_values(["stock_id", "Date"]).reset_index(drop=True)
    out.to_parquet(DAYTRADE_PATH, index=False)
    print(f"[寫回] {len(old):,}（全 0）→ {len(out):,} 列，"
          f"{out['Date'].min().date()} → {out['Date'].max().date()}", flush=True)
    print("✅ 完成", flush=True)


if __name__ == "__main__":
    main()
