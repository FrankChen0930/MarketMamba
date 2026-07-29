"""
MarketMamba — 把 `prices_raw` 切換成官方還原價（2026-07-29，使用者確認後執行）
================================================================================
把 `prices_adj_raw.parquet`（B-3 產出）寫進 production 的 `prices_raw.parquet`。

【設計原則：只改值、不改型別】
  切換前實測兩檔 schema 有兩處差異，都會製造風險：

      Date:  production `large_string`（'YYYY-MM-DD'） vs 新檔 `timestamp[ns]`
      src :  production 無                            vs 新檔 有

  **`Date` 型別絕對不能改。** 每日更新的 fetcher 產出的是字串日期，
  若 parquet 存成 timestamp，concat 後會變成混型 object，
  `drop_duplicates` 靜默失效——那正是問題 1（10,591 列重複、2432 的 Return_5d
  正負號相反）的根因。同一個坑不踩第二次。

  `src` 欄位同理不進 production：每日新增的列不會有這欄，會產生 NaN，
  且下游沒有任何程式需要它。完整版（含 `src`）保留在 `prices_adj_raw.parquet`
  供查閱；`legacy_scaled` 的列本來就能用「Date < 2007-07-01 且為上櫃」識別。

  → production schema 與切換前**逐欄逐型別完全相同**，只有價格數值改變。

【可回復】切換前備份到 `prices_raw_backup_before_adj_YYYYMMDD.parquet`。
  回復方式：把備份檔複製回 `prices_raw.parquet` 即可。

用法（repo 根目錄）：
    python V6/scripts/switch_to_adjusted_prices.py --dry-run
    python V6/scripts/switch_to_adjusted_prices.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402

P = Path(PROCESSED_DIR)
PROD = P / "prices_raw.parquet"
ADJ = P / "prices_adj_raw.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.print_help()
        return

    old_schema = pq.read_schema(PROD)
    old_cols = [f.name for f in old_schema if not f.name.startswith("__")]
    old_types = {f.name: str(f.type) for f in old_schema}
    print(f"[現況] production schema：{ {c: old_types[c] for c in old_cols} }")

    adj = pd.read_parquet(ADJ)
    print(f"[新檔] {len(adj):,} 列｜欄 {list(adj.columns)}")

    # ── 對齊 production schema ──────────────────────────────────
    out = adj.drop(columns=[c for c in adj.columns if c not in old_cols])
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")   # ← 關鍵
    out["stock_id"] = out["stock_id"].astype(str)
    missing = [c for c in old_cols if c not in out.columns]
    if missing:
        print(f"❌ 新檔缺 production 既有欄位 {missing}，中止")
        return
    out = out[old_cols].sort_values(["stock_id", "Date"]).reset_index(drop=True)

    old = pd.read_parquet(PROD, columns=["Date", "stock_id"])
    print(f"[比對] 舊 {len(old):,} 列 → 新 {len(out):,} 列"
          f"（{len(out) - len(old):+,}）")
    print(f"[比對] 舊 {old['stock_id'].nunique():,} 支 → "
          f"新 {out['stock_id'].nunique():,} 支")
    print(f"[比對] 日期型別：寫入前確認為 {out['Date'].dtype}"
          f"（必須是 object/字串，與 production 一致）")

    # 寫入前健檢（規則 7：數值明確輸出）
    n_dup = int(out.duplicated(subset=["Date", "stock_id"]).sum())
    n_bad = int((pd.to_numeric(out["Close"], errors="coerce") <= 0).sum())
    n_nan = int(out[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
    last = out["Date"].max()
    n_last = int((out["Date"] == last).sum())
    print(f"[健檢] 重複 {n_dup:,}｜Close<=0 {n_bad:,}｜OHLC NaN {n_nan:,}"
          f"｜最新 {last} 共 {n_last:,} 支")
    if n_dup or n_bad or n_nan:
        print("❌ 健檢未過，中止切換")
        return

    if a.dry_run:
        print("\n--dry-run：未寫檔。")
        return

    backup = P / f"prices_raw_backup_before_adj_{date.today():%Y%m%d}.parquet"
    if not backup.exists():
        shutil.copy(PROD, backup)
        print(f"\n[備份] → {backup.name}（回復方式：複製回 prices_raw.parquet）")
    out.to_parquet(PROD, index=False)

    chk = pq.read_schema(PROD)
    new_types = {f.name: str(f.type) for f in chk if not f.name.startswith("__")}
    same = all(old_types[c].replace("large_", "") == new_types[c].replace("large_", "")
               for c in old_cols)
    print(f"[驗證] 寫入後 schema：{new_types}")
    print(f"[驗證] 與切換前型別一致：{'✓' if same else '❌'}")
    print("✅ 切換完成")


if __name__ == "__main__":
    main()
