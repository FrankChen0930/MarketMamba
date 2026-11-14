"""
MarketMamba — 建立除權除息還原因子表（2026-07-28）
====================================================
來源：TWSE 除權除息計算結果表（TWT49U），支援日期區間查詢，
全歷史只需數次請求（實測一次可拉 3 年 / 2,540 筆）。

用途一：**還原因子的權威口徑**
    adj_factor = 除權息參考價 / 除權息前收盤價
  交易所的參考價已內含現金股利、股票股利與現金增資的全部影響，
  比從 `dividend_raw` 的現金股利反推準確（後者處理不了無償配股與增資）。

用途二：判定 `prices_raw` 各段是否已還原、以及接縫在哪。

【已驗證】2412 中華電 2026-07-09：
    官方 前收 139.5 → 參考價 134.3（息值 5.2），adj_factor = 0.962724
    prices_raw 實際 07-08 139.5 → 07-09 133.5（−4.30%）
    官方參考價隱含 −3.73%，差額為當日真實漲跌
  → 證實 prices_raw 該段**未還原**，且本表足以修正。

【2026-07-28 更新：上櫃已納入】
  初版只涵蓋上市，因為 TPEX `bulletin/exDailyQ` 傳 `date=...` 只回「即將除權息」
  的前瞻窗口，被判定為無歷史來源。**那個判定是錯的**——參數名要用
  `startDate`/`endDate` 且必須是 `YYYY/MM/DD`（compact 格式被靜默忽略）。
  這是本專案雷區清單第 2 條的又一次重演，差點讓上櫃退回用 `dividend_raw` 公式自算。

  現在兩市場都用官方口徑，`market` 欄標示來源：
      twse  TWT49U      2005-01 起
      tpex  exDailyQ    2007 下半年起（與 TPEX 價格端點的 2007-07 起點一致）

【本檔只建表，不改任何既有資料】
  要不要拿它去還原 `prices_raw` 是另一個步驟（B-3 步驟 3），
  因為那會改變全歷史的價格序列。

用法：
    python V6/scripts/build_ex_rights_table.py --build
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402
from marketmamba.data.fetcher import (  # noqa: E402
    fetch_capital_reduction_tpex_direct,
    fetch_capital_reduction_twse_direct,
    fetch_ex_rights_tpex_direct,
    fetch_ex_rights_twse_direct,
)

OUT = PROCESSED_DIR / "ex_rights_raw.parquet"
START_YEAR = 2005
TPEX_START_YEAR = 2007          # 2007 H1 = 0 筆、H2 起才有
CHUNK_YEARS = 3


def _fetch_retry(fn, a: str, b: str, label: str, tries: int = 3):
    """
    帶重試的抓取。

    為什麼需要：2026-07-28 首次建表時，TPEX 的 2022–2024 區塊因**暫時性失敗**回 None，
    腳本照樣印「0 筆」繼續跑完，產出一張缺了整整三年的表（該區間實際有 3,070 筆）。
    重跑同一個請求就成功了。
    → 抓取失敗必須重試並在最終仍失敗時**明確警示**，不能靜默當成「該區間本來就沒資料」。
    """
    for i in range(1, tries + 1):
        df = fn(a, b)
        if df is not None and not df.empty:
            return df
        if i < tries:
            print(f"    {label} 第 {i} 次回空，{i * 5} 秒後重試…", flush=True)
            time.sleep(i * 5)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if not args.build:
        ap.print_help()
        return

    end_year = pd.Timestamp.today().year
    frames = []

    print("■ 上市（TWSE TWT49U）")
    y = START_YEAR
    while y <= end_year:
        y2 = min(y + CHUNK_YEARS - 1, end_year)
        df = _fetch_retry(fetch_ex_rights_twse_direct,
                          f"{y}0101", f"{y2}1231", f"TWSE {y}–{y2}")
        n = 0 if df is None else len(df)
        print(f"  {y}–{y2}: {n:>5} 筆" + ("   ⚠️ 重試後仍為空" if not n else ""),
              flush=True)
        if df is not None and not df.empty:
            df["market"] = "twse"
            frames.append(df)
        time.sleep(1.5)
        y = y2 + 1

    print("■ 上櫃（TPEX exDailyQ，注意參數要 startDate/endDate + 斜線格式）")
    y = TPEX_START_YEAR
    while y <= end_year:
        y2 = min(y + CHUNK_YEARS - 1, end_year)
        df = _fetch_retry(fetch_ex_rights_tpex_direct,
                          f"{y}/01/01", f"{y2}/12/31", f"TPEX {y}–{y2}")
        n = 0 if df is None else len(df)
        print(f"  {y}–{y2}: {n:>5} 筆" + ("   ⚠️ 重試後仍為空" if not n else ""),
              flush=True)
        if df is not None and not df.empty:
            df["market"] = "tpex"
            frames.append(df)
        time.sleep(1.5)
        y = y2 + 1

    # ── 減資恢復買賣（2026-07-29 新增）────────────────────────────────
    # 彌補虧損減資**不經過除權除息程序**，完全不在 TWT49U / exDailyQ 裡。
    # 不處理的話會留下假的 +900% 單日報酬：實測全歷史 |報酬|>100% 的 273 筆中
    # 有 203 筆是這類事件（前收 1–3 元 → 停牌 12–23 天 → 復牌 10–40 元）。
    print("■ 減資恢復買賣（TWSE reducation/TWTAUU + TPEX bulletin/revivt）")
    for label, fn, fmt in (
        ("TWSE", fetch_capital_reduction_twse_direct, "{y}0101|{y2}1231"),
        ("TPEX", fetch_capital_reduction_tpex_direct, "{y}/01/01|{y2}/12/31"),
    ):
        y = START_YEAR
        while y <= end_year:
            y2 = min(y + CHUNK_YEARS - 1, end_year)
            a, b = fmt.format(y=y, y2=y2).split("|")
            df = _fetch_retry(fn, a, b, f"{label} 減資 {y}–{y2}", tries=2)
            n = 0 if df is None else len(df)
            print(f"  {label} {y}–{y2}: {n:>4} 筆", flush=True)
            if df is not None and not df.empty:
                df["market"] = label.lower()
                frames.append(df)
            time.sleep(1.5)
            y = y2 + 1

    if not frames:
        print("❌ 沒有取得任何資料")
        return
    out = pd.concat(frames, ignore_index=True)
    if "why" not in out.columns:
        out["why"] = pd.NA
    # 同一支股票不可能同日同時出現在兩個市場；若有重複以先抓到的（上市）為準
    n0 = len(out)
    out = out.drop_duplicates(subset=["Date", "stock_id"], keep="first")
    if len(out) < n0:
        print(f"\n[去重] 跨市場重複 {n0 - len(out):,} 筆（保留上市）")
    # 兩個交易所的 kind 用語不同（TWSE「息/權息/權」vs TPEX「除息/除權息/除權」），
    # 不統一的話下游任何 groupby("kind") 都會把同一類事件拆成兩組。
    out["kind"] = (out["kind"].astype(str).str.strip()
                   .str.replace("^除", "", regex=True))
    out = out.sort_values(["stock_id", "Date"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    print()
    print(f"[健檢] {len(out):,} 筆｜{out['stock_id'].nunique():,} 支｜"
          f"{out['Date'].min().date()} → {out['Date'].max().date()}")
    print(f"[健檢] 依市場：{dict(out['market'].value_counts())}"
          f"｜各自涵蓋 "
          + "、".join(f"{m} {g['stock_id'].nunique():,} 支 "
                      f"({g['Date'].min().date()}→{g['Date'].max().date()})"
                      for m, g in out.groupby("market")))
    print(f"[健檢] adj_factor min={out['adj_factor'].min():.4f} "
          f"median={out['adj_factor'].median():.4f} max={out['adj_factor'].max():.4f}"
          f"｜>1 的筆數={int((out['adj_factor'] > 1).sum())}（>1 代表**減資**：參考價高於前收，屬合法事件）")
    _cut = out[out["kind"] == "減資"]
    if len(_cut):
        print(f"[健檢] 減資事件 {len(_cut):,} 筆｜倍率 median {_cut['adj_factor'].median():.3f}"
              f"｜max {_cut['adj_factor'].max():.2f}｜原因 {dict(_cut['why'].value_counts().head(4))}")
    print(f"[健檢] 事件類型：{dict(out['kind'].value_counts())}")

    # ── 年度 × 市場缺口檢查 ──────────────────────────────────────
    # 首建時 TPEX 2022–2024 因暫時性失敗整段落空，而總筆數看起來仍「合理」。
    # 逐年逐市場列出並自動標記缺口，讓這種洞不可能再無聲通過。
    piv = (out.assign(y=out["Date"].dt.year)
              .pivot_table(index="y", columns="market", values="stock_id",
                           aggfunc="count").fillna(0).astype(int))
    print("[健檢] 依年 × 市場:")
    print(piv.to_string())
    holes = []
    for mkt, first_y in [("twse", START_YEAR), ("tpex", TPEX_START_YEAR + 1)]:
        if mkt not in piv.columns:
            holes.append(f"{mkt} 整個缺席")
            continue
        s = piv[mkt]
        holes += [f"{mkt} {y} 年為 0" for y in s.index if y >= first_y and s[y] == 0]
    if holes:
        print(f"[健檢] ❌ 發現缺口：{'、'.join(holes)}")
        print("[健檢]    → 這通常是抓取暫時性失敗，重跑本腳本即可（已內建重試）")
    else:
        print("[健檢] ✅ 無年度缺口")
    print(f"\n✅ 已寫入 {OUT.name}（新檔，未改動任何既有資料）")


if __name__ == "__main__":
    main()
