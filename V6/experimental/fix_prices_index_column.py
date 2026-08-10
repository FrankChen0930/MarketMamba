"""
fix_prices_index_column.py — 移除 prices_raw.parquet 裡多出來的 __index_level_0__ 欄
================================================================================
2026-08-10：當天三次執行（PersonalOS_Daily 21:30、MarketMamba_V62 22:15、
手動 V6.1 22:30）全部在 `fetcher.py:3857` 的
`pd.read_parquet(price_path, columns=["Date"])` 當場拋

    ArrowInvalid: Multiple matches for FieldRef.Name(__index_level_0__)

因為 schema 長成這樣（**兩個同名欄**，pyarrow 無法決定要哪一個）：

    Date, stock_id, Open, High, Low, Close, Volume,
    __index_level_0__: double,      ← ① 資料欄（08-08 回補留下的 RangeIndex）
    __index_level_0__: int64        ← ② 索引（今天 21:34 那次 append 寫出來的）

**怎麼形成的**（逐項量測，不是推測）
------------------------------------
① 依「欄位位置」讀出 double 欄的值 = `0 … 8,737,250`、單調遞增、
   非 NaN 共 **8,737,251** 個 —— 正好是 2026-08-08 回補後的列數，
   且**只有 2026-08-10 的 1,959 列是 NaN**。
   → 它是回補那次寫入留下的 RangeIndex，今天之前就在檔案裡了。
   ⚠️ 但 committed 版的 `backfill_prices.py:220` 寫的是 `to_parquet(..., index=False)`，
      照那份程式碼不會產出這個欄；中間檔已被覆蓋，**無法再重現**。
      實際跑的很可能是 commit 之前的工作區版本 —— 這點沒有定論。

② 今天 21:34 的每日更新走 `_append_to_parquet`：
   讀進來（stray 欄被當成一般資料欄）→ concat 今天的新價格（沒有這欄 → NaN → double）
   → `drop_duplicates` 讓索引變得**不連續** → `df.to_parquet(path)` 沒有 `index=False`
   → 把不連續索引又寫成第二個 `__index_level_0__`。撞名。

**為什麼這個潛伏 bug 撐到今天才爆**
`to_parquet` 不加 `index=False` 時，索引**連續**的話 pandas 只寫進 metadata、
不會落成實體欄；**只有不連續時才會實體化**。同一批被寫的 `institutional_raw` /
`margin_raw` 都乾淨，就是因為它們沒觸發 `drop_duplicates`。
→ 根因已在 `fetcher._append_to_parquet` 補上 `index=False` + 寫入前守門。

**做法**
--------
只做一件事：**依欄位位置**刪掉第 7 欄（那個 double stray 欄），其餘原封不動。
不能用欄名刪 —— 欄名重複，pandas / pyarrow 的 name-based API 都會失敗。

紀律（CLAUDE.md）
  · 只改值、不改型別 → 保留下來的 7 欄，型別逐欄比對必須相同
  · **守門放在寫入之前** → 寫完再檢查只能發現、不能防止
  · 先寫暫存檔、驗過才 `os.replace` 換上 → 中途失敗不會留下半殘的正式檔

用法（WSL）
-----------
    python V6/experimental/fix_prices_index_column.py            # 只檢查，不寫
    python V6/experimental/fix_prices_index_column.py --apply    # 實際修復
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROD = Path("/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/prices_raw.parquet")

EXPECTED_KEEP = ["Date", "stock_id", "Open", "High", "Low", "Close", "Volume"]
STRAY_PREFIX = "__index_level_"


def _describe(names: list[str]) -> str:
    return ", ".join(names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="實際寫入（預設只檢查）")
    ap.add_argument("--path", type=Path, default=PROD)
    args = ap.parse_args()

    path: Path = args.path
    if not path.exists():
        print(f"[錯誤] 找不到檔案：{path}")
        return 1

    pf = pq.ParquetFile(path)
    names = [f.name for f in pf.schema_arrow]
    types = [str(f.type) for f in pf.schema_arrow]
    print(f"[現況] {path.name}")
    print(f"  列數   : {pf.metadata.num_rows:,}")
    print(f"  欄位   : {_describe(names)}")
    for n, t in zip(names, types):
        print(f"           {n}: {t}")

    stray_pos = [i for i, n in enumerate(names) if n.startswith(STRAY_PREFIX)]
    keep_pos = [i for i, n in enumerate(names) if not n.startswith(STRAY_PREFIX)]

    if not stray_pos:
        print("\n[結論] 沒有 __index_level_* 欄，不需要修復。")
        return 0

    print(f"\n[判定] stray 欄位置 = {stray_pos}（{[names[i] for i in stray_pos]}）")
    print(f"       要保留的位置 = {keep_pos}（{[names[i] for i in keep_pos]}）")

    # ── 寫入前守門 ①：保留下來的欄名必須就是預期的 7 欄，順序也要一致 ──────────
    keep_names = [names[i] for i in keep_pos]
    if keep_names != EXPECTED_KEEP:
        print(f"\n[中止] 保留欄與預期不符：\n  實際 {keep_names}\n  預期 {EXPECTED_KEEP}")
        return 1

    # ── 讀進來（一次讀全表；欄名重複時 pandas 讀不了，pyarrow 可以）─────────────
    # ⚠️ 不能用 pq.read_table() —— 它走 dataset API，欄名重複時直接
    #    `ArrowInvalid: Can't unify schema with duplicate field names.`
    #    ParquetFile.read() 是單檔路徑，不做 schema unify，讀得動。
    print("\n[讀取] 載入全表…")
    tbl = pf.read()
    n_rows = tbl.num_rows

    # stray 欄的內容摘要——留一份證據在 log 裡（規則 7：數值要看得見）
    for i in stray_pos:
        col = tbl.column(i)
        vals = col.to_pandas().to_numpy()
        nn = int((~pd.isna(vals)).sum())
        print(f"  欄[{i}] {names[i]}: {types[i]}｜非 NaN {nn:,}／{n_rows:,}"
              f"｜min={np.nanmin(vals) if nn else float('nan')}"
              f" max={np.nanmax(vals) if nn else float('nan')}")

    tbl_new = tbl.select(keep_pos)
    # pandas metadata 仍然指向 __index_level_0__，留著會讓下游還原出幽靈索引 → 整個拿掉
    tbl_new = tbl_new.replace_schema_metadata(None)

    # ── 寫入前守門 ②：型別逐欄相同、列數不變 ─────────────────────────────────
    new_types = [str(f.type) for f in tbl_new.schema]
    old_types = [types[i] for i in keep_pos]
    if new_types != old_types:
        print(f"\n[中止] 型別改變了：\n  改後 {new_types}\n  原本 {old_types}")
        return 1
    if tbl_new.num_rows != n_rows:
        print(f"\n[中止] 列數改變：{n_rows:,} → {tbl_new.num_rows:,}")
        return 1
    print(f"\n[守門] 欄名 ✓｜型別逐欄相同 ✓｜列數不變 {n_rows:,} ✓")

    if not args.apply:
        print("\n（--check 模式，未寫入。要寫入請加 --apply）")
        return 0

    # ── 備份 ────────────────────────────────────────────────────────────────
    bkp = path.with_name(f"{path.stem}_backup_before_indexfix_"
                         f"{pd.Timestamp.now():%Y%m%d}{path.suffix}")
    shutil.copy2(path, bkp)
    print(f"\n[寫入] 已備份 → {bkp.name}")

    # ── 先寫暫存檔，驗過才換上 ────────────────────────────────────────────────
    tmp = path.with_suffix(".parquet.tmp")
    compression = (pf.metadata.row_group(0).column(0).compression or "SNAPPY").lower()
    pq.write_table(tbl_new, tmp, compression=compression)
    print(f"[寫入] 暫存檔完成 → {tmp.name}（compression={compression}）")

    # ── 驗收：暫存檔的每一欄，與原表對應欄逐值相同 ─────────────────────────────
    tbl_chk = pq.read_table(tmp)
    ok = True
    print("\n[驗收] 逐欄比對（原表 vs 修復後）")
    if tbl_chk.num_rows != n_rows:
        print(f"  ✗ 列數 {n_rows:,} → {tbl_chk.num_rows:,}")
        ok = False
    for j, i in enumerate(keep_pos):
        a = tbl.column(i).to_pandas().to_numpy()
        b = tbl_chk.column(j).to_pandas().to_numpy()
        if a.dtype.kind == "f":
            same = np.array_equal(a, b, equal_nan=True)
            delta = np.nanmax(np.abs(a - b)) if same else float("nan")
            print(f"  {'✓' if same else '✗'} {names[i]}: max|Δ| = {0.0 if same else delta:.3e}")
        else:
            same = bool((a == b).all())
            print(f"  {'✓' if same else '✗'} {names[i]}: 逐值相同 = {same}")
        ok = ok and same

    # 鍵集合（Date, stock_id）不變
    ka = set(zip(tbl.column(0).to_pandas(), tbl.column(1).to_pandas()))
    kb = set(zip(tbl_chk.column(0).to_pandas(), tbl_chk.column(1).to_pandas()))
    print(f"  {'✓' if ka == kb else '✗'} (Date, stock_id) 鍵集合相同"
          f"：{len(ka):,} vs {len(kb):,}")
    ok = ok and (ka == kb)

    # pandas 讀得動了嗎（這才是原本壞掉的那件事）
    try:
        probe = pd.read_parquet(tmp, columns=["Date"])
        print(f"  ✓ pd.read_parquet(columns=['Date']) 成功：{len(probe):,} 列")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ pd.read_parquet 仍失敗：{e}")
        ok = False

    if not ok:
        tmp.unlink(missing_ok=True)
        print("\n[中止] 驗收未通過，暫存檔已刪除，正式檔未動。")
        return 1

    os.replace(tmp, path)
    print(f"\n[完成] 已換上修復後的 {path.name}")
    print(f"        欄位：{_describe([f.name for f in pq.ParquetFile(path).schema_arrow])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
