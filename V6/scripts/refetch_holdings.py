"""
MarketMamba V6.1 — Re-fetch Holdings (大戶持股分級) v3 - FINAL
================================================================
已確認 FinMind API 行為：
  - TaiwanStockHoldingSharesPer 無 data_id 時回傳空（即使有 VIP）
  - 必須用 data_id = stock_id 逐支股票查詢
  - 資料範圍：2018-01-05 起（FinMind 最早可用）
  - 欄位：date, stock_id, HoldingSharesLevel, people, percent, unit
  - 大戶判斷：HoldingSharesLevel != "1-999"

策略（省記憶體 + 斷點重啟）：
  - 每次處理 1 支股票（無記憶體問題）
  - 每 50 支股票存一個 checkpoint parquet
  - 下次執行自動跳過已完成的股票
  - 估計：2511 支 × 1.0s/支 = ~42 分鐘

執行方式：
  python V6/scripts/refetch_holdings.py              # 正常執行（斷點繼續）
  python V6/scripts/refetch_holdings.py --force      # 全部重抓
  python V6/scripts/refetch_holdings.py --merge-only # 只合併 checkpoint
  python V6/scripts/refetch_holdings.py --dry-run    # 預覽
"""

import os
import sys
import gc
import re
import time
import logging
import argparse
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Setup ────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT.parent / "Data"
PROC_DIR  = DATA_DIR / "processed_v6"
CKPT_DIR  = PROC_DIR / "_holdings_checkpoints"
LOG_DIR   = ROOT / "logs"
PROC_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")
FINMIND_BASE  = "https://api.finmindtrade.com/api/v4/data"

# FinMind TaiwanStockHoldingSharesPer 可用起始日期（實測確認）
START_DATE = "2018-01-01"
END_DATE   = date.today().strftime("%Y-%m-%d")

# 大戶判斷：集保戶股權分散表 HoldingSharesLevel == "1-999" 為散戶
RETAIL_LEVEL = "1-999"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "refetch_holdings.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("holdings")


# ================================================================
# API helpers
# ================================================================

def _fetch_one_stock(stock_id: str, retries: int = 3) -> pd.DataFrame | None:
    """Fetch full history for one stock (data_id mode)."""
    params = {
        "dataset":    "TaiwanStockHoldingSharesPer",
        "data_id":    stock_id,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "token":      FINMIND_TOKEN,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(FINMIND_BASE, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == 200:
                    rows = data.get("data", [])
                    if rows:
                        df = pd.DataFrame(rows)
                        # Add stock_id col if missing
                        if "stock_id" not in df.columns:
                            df["stock_id"] = stock_id
                        return df
                    return None  # empty = no data (legitimate)
            elif resp.status_code == 402:
                log.error("  FinMind 402: API 點數不足！請確認 VIP 狀態")
                return None
            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"  Rate limited (429), waiting {wait}s ...")
                time.sleep(wait)
                continue
        except requests.Timeout:
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            log.debug(f"  {stock_id} Error: {e}")
            time.sleep(3)
    return None


# ================================================================
# Ratio calculation
# ================================================================

def _compute_ratios(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format data to weekly Whale/Retail ratios.

    Input : date, stock_id, HoldingSharesLevel, people, percent, unit
    Output: Week, stock_id, Whale_Hold_Ratio, Retail_Hold_Ratio

    percent 欄位 = 該級距持股占總流通股的百分比（FinMind 直接提供）
    Whale  = sum of percent for all non-"1-999" levels
    Retail = percent for "1-999" level
    """
    df = df_raw.copy()
    df["Week"]    = pd.to_datetime(df["date"])
    df["percent"] = pd.to_numeric(df["percent"], errors="coerce").fillna(0.0)
    df["_retail"] = df["HoldingSharesLevel"].astype(str).str.strip() == RETAIL_LEVEL

    # Total per (Week, stock_id)
    total  = df.groupby(["Week", "stock_id"])["percent"].sum()
    retail = (df[df["_retail"]]
              .groupby(["Week", "stock_id"])["percent"]
              .sum())

    result = pd.concat([total.rename("_total"), retail.rename("Retail_Hold_Ratio")], axis=1).reset_index()
    result["Retail_Hold_Ratio"] = result["Retail_Hold_Ratio"].fillna(0.0)
    result["Whale_Hold_Ratio"]  = (result["_total"] - result["Retail_Hold_Ratio"]).clip(0, 100)

    return result[["Week", "stock_id", "Whale_Hold_Ratio", "Retail_Hold_Ratio"]]


# ================================================================
# Progress tracking via checkpoint index file
# ================================================================

INDEX_FILE = CKPT_DIR / "_completed_stocks.txt"

def get_completed_stocks() -> set[str]:
    if INDEX_FILE.exists():
        return set(INDEX_FILE.read_text(encoding="utf-8").splitlines())
    return set()

def mark_completed(stock_id: str) -> None:
    with open(INDEX_FILE, "a", encoding="utf-8") as f:
        f.write(stock_id + "\n")


# ================================================================
# Main
# ================================================================

def get_stock_universe() -> list[str]:
    prices_path = PROC_DIR / "prices_raw.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"prices_raw.parquet not found at {prices_path}")
    df = pd.read_parquet(prices_path, columns=["stock_id"])
    all_ids = df["stock_id"].unique()
    return sorted([s for s in all_ids if re.match(r"^\d{4}$", str(s).strip())])


def fetch_all(stock_ids: list[str], batch_size: int = 50, force: bool = False) -> None:
    completed = get_completed_stocks() if not force else set()
    if force and INDEX_FILE.exists():
        INDEX_FILE.unlink()

    remaining = [s for s in stock_ids if s not in completed]
    log.info(f"Universe: {len(stock_ids)} | Done: {len(completed)} | Remaining: {len(remaining)}")

    if not remaining:
        log.info("All done. Use --merge-only or --force.")
        return

    batch_rows = []
    ok_count = err_count = 0
    batch_idx = len([f for f in CKPT_DIR.glob("ckpt_batch_*.parquet")])

    for i, sid in enumerate(remaining, 1):
        df_raw = _fetch_one_stock(sid)
        if df_raw is not None and not df_raw.empty:
            df_ratio = _compute_ratios(df_raw)
            if not df_ratio.empty:
                batch_rows.append(df_ratio)
                ok_count += 1
            del df_raw, df_ratio
        else:
            err_count += 1

        mark_completed(sid)

        # Progress log every 25 stocks
        if i % 25 == 0 or i == len(remaining):
            log.info(f"  [{i:4d}/{len(remaining)}] ok:{ok_count} empty:{err_count}")

        # Save batch checkpoint every `batch_size` stocks
        if len(batch_rows) >= batch_size or (i == len(remaining) and batch_rows):
            batch_df = pd.concat(batch_rows, ignore_index=True)
            ckpt = CKPT_DIR / f"ckpt_batch_{batch_idx:04d}.parquet"
            batch_df.to_parquet(ckpt, index=False)
            log.info(f"  → Saved checkpoint: {ckpt.name} ({len(batch_df):,} rows)")
            del batch_df
            batch_rows = []
            gc.collect()
            batch_idx += 1

        time.sleep(0.8)  # ~1.25 req/s — respectful to API


def merge_checkpoints() -> pd.DataFrame:
    ckpt_files = sorted(CKPT_DIR.glob("ckpt_batch_*.parquet"))
    if not ckpt_files:
        log.error("No checkpoints found!")
        return pd.DataFrame()

    log.info(f"Merging {len(ckpt_files)} checkpoints ...")
    frames = [pd.read_parquet(f) for f in ckpt_files]
    merged = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()

    merged = (merged
              .drop_duplicates(subset=["Week", "stock_id"])
              .sort_values(["stock_id", "Week"])
              .reset_index(drop=True))

    whale_zero  = (merged["Whale_Hold_Ratio"].fillna(0) == 0).mean() * 100
    retail_zero = (merged["Retail_Hold_Ratio"].fillna(0) == 0).mean() * 100

    log.info(f"\n{'='*60}")
    log.info(f"  MERGE RESULT")
    log.info(f"{'='*60}")
    log.info(f"  Total rows      : {len(merged):,}")
    log.info(f"  Unique stocks   : {merged['stock_id'].nunique():,}")
    log.info(f"  Date range      : {merged['Week'].min()} → {merged['Week'].max()}")
    log.info(f"  Whale Zero%     : {whale_zero:.1f}%  (was 100%, target: <15%)")
    log.info(f"  Retail Zero%    : {retail_zero:.1f}%")
    log.info(f"\n  Whale_Hold_Ratio stats:")
    log.info(merged["Whale_Hold_Ratio"].describe().to_string())

    if whale_zero < 15:
        log.info("\n  ✅ Whale_Hold_Ratio looks healthy!")
    else:
        log.warning(f"\n  ⚠️  Whale Zero% still {whale_zero:.0f}% — check data")

    return merged


def save_final(df: pd.DataFrame) -> None:
    out = PROC_DIR / "holdings_raw.parquet"
    if out.exists():
        bak = PROC_DIR / "holdings_raw_BACKUP.parquet"
        out.rename(bak)
        log.info(f"  Backed up → {bak.name}")
    df.to_parquet(out, index=False)
    log.info(f"  ✅ Saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ================================================================
# CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Re-fetch holdings_raw.parquet (Whale_Hold_Ratio fix v3)")
    p.add_argument("--force",       action="store_true")
    p.add_argument("--merge-only",  action="store_true")
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--batch-size",  type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("  MarketMamba — Re-fetch Holdings v3 (per-stock, data_id)")
    log.info(f"  Range  : {START_DATE} → {END_DATE}")
    log.info(f"  Data   : {PROC_DIR}")
    log.info(f"  Ckpt   : {CKPT_DIR}")
    log.info("=" * 60)

    if not FINMIND_TOKEN:
        log.error("FINMIND_TOKEN not set!")
        sys.exit(1)

    stock_ids = get_stock_universe()
    completed = get_completed_stocks() if not args.force else set()
    remaining = [s for s in stock_ids if s not in completed]

    if args.dry_run:
        log.info(f"[DRY RUN] Universe  : {len(stock_ids)}")
        log.info(f"[DRY RUN] Completed : {len(completed)}")
        log.info(f"[DRY RUN] Remaining : {len(remaining)}")
        log.info(f"[DRY RUN] Est. time : ~{len(remaining) * 0.8 / 60:.0f} min")
        return

    if not args.merge_only:
        fetch_all(stock_ids, batch_size=args.batch_size, force=args.force)

    final = merge_checkpoints()
    if final.empty:
        log.error("Nothing to save!")
        sys.exit(1)

    save_final(final)

    log.info("\n  NEXT STEPS:")
    log.info("  1. python V6/scripts/verify_fixes.py")
    log.info("  2. python V6/scripts/build_feature_matrix.py")


if __name__ == "__main__":
    main()
