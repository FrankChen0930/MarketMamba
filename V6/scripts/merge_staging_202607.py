"""
MarketMamba — 合併 2026-07 停更回補的 staging 檔進 production（2026-07-28）
==========================================================================
對象：`per_raw` / `securities_raw` / `market_value_raw` 三個檔。
staging 由使用者 2026-07-24 寫的 `backfill_stale_202607.py` 產生（TWSE/MOPS 直連），
當時註明「merge 腳本待驗證過資料品質後再補」——本檔就是那個 merge 步驟。

【已完成的驗證】（2026-07-28）
  結構：三個檔皆**無非交易日假資料、無重複鍵**
        （對照組：daytrade staging 在非交易日 2026-07-10 有 1,710 列假資料）
  接縫連續性（production 2026-04-24 FinMind vs staging 2026-04-27 交易所）：
        PER median 0.9806｜PBR median 1.0000｜market_value median 0.9897 → 皆連續
  market_value 零值率 0.18%，反而優於 production 的 0.48%

【schema 對齊】staging 與 production 欄名不同，必須改名後才能併：
  per_raw          : dividend_yield      → DY
  securities_raw   : Securities_Balance  → Securities_Lending
  market_value_raw : 相同，不需改

【已知限制（待使用者決定，見檔尾 TODO）】
  per_raw 的新來源 TWSE BWIBBU_ALL **只涵蓋上市（1,080 支）**，
  而 production 歷史含上櫃（2,201 支）。合併後上櫃股的 PER/PBR 會停在 2026-04-24
  並由 ffill 延用。目前找不到 TPEX 的對應端點（已試四個路徑皆非 JSON）。
  → 仍比「全部凍結」好，但會在 2026-04-27 形成上市/上櫃的覆蓋落差。

用法（repo 根目錄）：
    python V6/scripts/merge_staging_202607.py --dry-run
    python V6/scripts/merge_staging_202607.py --apply
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
    fetch_per_twse_direct,
    fetch_securities_lending_twse_direct,
)

STAGING = PROCESSED_DIR / "_staging_202607_backfill"
PRICE_PATH = PROCESSED_DIR / "prices_raw.parquet"

SPECS = {
    "per_raw": {
        "staging": "per_raw_backfill.parquet",
        "rename": {"dividend_yield": "DY"},
        "fetcher": fetch_per_twse_direct,
    },
    "securities_raw": {
        "staging": "securities_raw_backfill.parquet",
        "rename": {"Securities_Balance": "Securities_Lending"},
        "fetcher": fetch_securities_lending_twse_direct,
    },
    "market_value_raw": {
        "staging": "market_value_raw_backfill.parquet",
        "rename": {},
        "fetcher": None,     # 需 MOPS 股本 × 收盤價，日更另由 fetcher 端處理
    },
}


def _trading_days_after(d0: pd.Timestamp) -> list[str]:
    pr = pd.read_parquet(PRICE_PATH, columns=["Date"])
    days = sorted(set(pd.to_datetime(pr["Date"]).dt.strftime("%Y-%m-%d")))
    return [d for d in days if pd.Timestamp(d) > d0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.print_help()
        return

    for name, spec in SPECS.items():
        prod_path = PROCESSED_DIR / f"{name}.parquet"
        stg_path = STAGING / spec["staging"]
        print("=" * 78)
        print(f"■ {name}")
        print("=" * 78)
        if not stg_path.exists():
            print(f"  staging 不存在，跳過：{stg_path.name}")
            continue

        prod = pd.read_parquet(prod_path)
        prod["Date"] = pd.to_datetime(prod["Date"])
        stg = pd.read_parquet(stg_path)
        stg["Date"] = pd.to_datetime(stg["Date"])
        stg = stg.rename(columns=spec["rename"])

        missing_cols = [c for c in prod.columns if c not in stg.columns]
        extra_cols = [c for c in stg.columns if c not in prod.columns]
        print(f"  production {len(prod):,} 列（最新 {prod['Date'].max().date()}）｜"
              f"staging {len(stg):,} 列（{stg['Date'].min().date()} → {stg['Date'].max().date()}）")
        print(f"  schema 對齊後：staging 缺 {missing_cols or '無'}｜多 {extra_cols or '無'}")
        for c in missing_cols:
            stg[c] = pd.NA
        stg = stg[prod.columns]

        # ── 補 staging 尾端到最新交易日 ──
        tail = _trading_days_after(stg["Date"].max())
        extra = []
        if tail and spec["fetcher"] is not None:
            print(f"  staging 尾端缺 {len(tail)} 個交易日 {tail} → 直連補齊")
            for d in tail:
                df = spec["fetcher"](d)
                time.sleep(1.2)
                if df is None or df.empty:
                    continue
                df = df.rename(columns=spec["rename"])
                for c in prod.columns:
                    if c not in df.columns:
                        df[c] = pd.NA
                extra.append(df[prod.columns])
            print(f"    補到 {sum(len(x) for x in extra):,} 列")
        elif tail:
            print(f"  staging 尾端缺 {len(tail)} 個交易日 {tail}"
                  f"（本資料源需 MOPS 股本，改由每日更新處理，此處不補）")

        new = pd.concat([stg] + extra, ignore_index=True) if extra else stg
        new["Date"] = pd.to_datetime(new["Date"])

        merged = pd.concat([prod, new], ignore_index=True)
        n_before = len(merged)
        merged = merged.drop_duplicates(subset=["Date", "stock_id"], keep="last")
        merged = merged.sort_values(["stock_id", "Date"]).reset_index(drop=True)
        print(f"  合併：{len(prod):,} + {len(new):,} → {len(merged):,} 列"
              f"（去重 {n_before - len(merged):,}）｜最新 {merged['Date'].max().date()}")

        if args.apply:
            backup = PROCESSED_DIR / f"{name}_backup_{date.today():%Y%m%d}.parquet"
            if not backup.exists():
                prod.to_parquet(backup, index=False)
                print(f"  [備份] → {backup.name}")
            merged.to_parquet(prod_path, index=False)
            print(f"  ✅ 已寫回 {prod_path.name}")
        print()

    if args.dry_run:
        print("--dry-run：未寫檔。", flush=True)


if __name__ == "__main__":
    main()
