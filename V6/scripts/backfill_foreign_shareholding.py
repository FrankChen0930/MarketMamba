"""
backfill_foreign_shareholding.py — 回補外資持股的 9xxx 歷史缺口（D1）
========================================================================
2026-07-29 由可得性旗標（決策1）挖出的資料缺口。

【問題】
`foreign_shareholding_raw` 缺 **43 支仍在交易的股票**的 2018–2026-04 歷史，
全部是 9 開頭的連續區間（9802、9902、9904–9912、9914、9917–9919、9921、
9924–9934…＝寶成、豐泰、巨大、中鼎、美利達、櫻花、新保、中視…）。
影響 85,938 列（2018-01~2026-04 的 2.29%）。

七個逐日來源的「首碼 1–8 vs 首碼 9」覆蓋率實測，**只有這一個異常**：
    foreign_shareholding  95.1% vs 8.5%（差 86.6）
    其餘六個來源差距皆在 ±16 以內，9xxx 甚至多半略高
→ 不是 FinMind 的通病，是這一個資料集的孤立缺口。

【為什麼要修，而不是靠旗標擋掉就好】
可得性旗標已經擋掉最直接的錯誤（不會再把捏造的 0 當成「外資持股 0%」的訊號）。
但有一件事旗標解不掉：

    **旗標的語意會在 2026-05-06 翻轉。**
    那 43 支在 2013–2026-04 的 Avail_ForeignShare 全是 0，
    2026-05-06 直連回補開始變成 1。模型在 2013–2023 訓練時，
    這個旗標實質上是「這是不是 9xxx 傳產股」的代理變數
    （缺失剛好是連續代號、9xxx 集中在紡織/食品/營建/觀光）；
    到 2026 上線推論，同一批股票的旗標卻是 1。
    → 訓練/推論語意不一致，與 D1 macro_norm、fundamentals_v2 同一類陷阱。

【可行性已實測】TWSE MI_QFIIS 歷史完整：2018-01-15 / 2020-06-15 / 2023-03-15 /
2026-04-15 四個日期都回 925–1,083 支，且 9904 / 9910 / 9921 / 9933 每天都在。
資料在交易所一直都有，是 FinMind 沒抓到。

---------------------------------------------------------------------------
設計要點

1. **只補缺的 (date, stock_id)，不動任何既有列。**
   直連端點回的是全市場，但若整批覆蓋上去，等於把既有的 FinMind 序列
   換成 TWSE 序列——兩個來源的口徑差異會在每一支股票中間製造接縫。
   只補洞的話，那 43 支的整段歷史都來自 TWSE、內部一致，
   且與 2026-05-06 之後的直連段完美銜接。

2. **只改值、不改型別**（2026-07-29 換 prices_raw 的教訓）。
   既有 parquet 的 `date` 是 timestamp[ns]、`stock_id` 是 large_string。
   寫入前逐欄對齊，否則混型會讓 drop_duplicates 靜默失效。

3. **逐年 checkpoint**，中斷可續跑（比照 refetch_raw_prices_full.py）。

4. **不要與其他打 TWSE 的作業同時跑**（2026-07-28 的教訓：
   外資持股回補與價格重抓同時打 TWSE 造成 HTTP 307 限流，
   該年度檔缺了 4 天才發現）。

用法（repo 根目錄）：
    python V6/scripts/backfill_foreign_shareholding.py --dry-run   # 只列要補什麼
    python V6/scripts/backfill_foreign_shareholding.py --fetch     # 抓（逐年落檔，可中斷續跑）
    python V6/scripts/backfill_foreign_shareholding.py --merge     # 驗收後併回 production（會先備份）
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from marketmamba.config import PROCESSED_DIR                       # noqa: E402
from marketmamba.data.fetcher import (                             # noqa: E402
    FS_COLS,
    fetch_foreign_shareholding_direct,
)
from marketmamba.data.hygiene import filter_tradable_universe      # noqa: E402

P = Path(PROCESSED_DIR)
FS_PATH = P / "foreign_shareholding_raw.parquet"
STAGING = P / "_staging_fs_backfill"
SLEEP_SEC = 1.2          # 對交易所客氣一點；2026-07-28 有過限流前例
START = "2018-01-02"     # foreign_shareholding_raw 的既有起點
END = "2026-05-05"       # 2026-05-06 起已由每日直連覆蓋


# ============================================================
# 缺口盤點
# ============================================================

def find_gaps() -> tuple[pd.DataFrame, list[str], list[pd.Timestamp]]:
    """
    回傳 (缺口明細, 受影響股票清單, 需要抓取的交易日清單)。

    「缺口」的定義刻意收斂到**仍在交易的股票**：已下市股在交易所端沒有資料，
    抓再多次也補不到（那是存活者偏差，協定 §7 已揭露），
    把它們算進來只會讓進度數字永遠達不到 100%。
    """
    px = pd.read_parquet(P / "prices_raw.parquet", columns=["Date", "stock_id"])
    px["Date"] = pd.to_datetime(px["Date"])
    px["stock_id"] = px["stock_id"].astype(str)

    last = px["Date"].max()
    alive = set(px.loc[px["Date"] >= last - pd.Timedelta(days=10), "stock_id"])
    uni = set(filter_tradable_universe(
        pd.DataFrame({"stock_id": sorted(px["stock_id"].unique())}))["stock_id"]) & alive

    fs = pd.read_parquet(FS_PATH, columns=["date", "stock_id"])
    fs["date"] = pd.to_datetime(fs["date"])
    fs["stock_id"] = fs["stock_id"].astype(str)

    win = (px["Date"] >= START) & (px["Date"] <= END) & px["stock_id"].isin(uni)
    want = px.loc[win, ["Date", "stock_id"]].rename(columns={"Date": "date"})
    have = set(map(tuple, fs[["date", "stock_id"]].to_numpy()))
    miss = want[[t not in have for t in map(tuple, want.to_numpy())]]

    stocks = sorted(miss["stock_id"].unique())
    days = sorted(miss["date"].unique())
    return miss, stocks, [pd.Timestamp(d) for d in days]


# ============================================================
# 抓取
# ============================================================

_TRIAL: list[pd.DataFrame] = []      # --limit 試跑的結果（不落檔，只在記憶體驗收）


def fetch(days: list[pd.Timestamp], limit: int | None = None) -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    by_year: dict[int, list[pd.Timestamp]] = {}
    for d in days:
        by_year.setdefault(d.year, []).append(d)

    t0 = time.time()
    n_done = 0
    for year in sorted(by_year):
        out = STAGING / f"fs_{year}.parquet"
        if out.exists():
            print(f"[{year}] 已存在，跳過（要重抓請先刪除 {out.name}）", flush=True)
            continue

        parts, n_empty = [], 0
        yd = by_year[year]
        for i, d in enumerate(yd, 1):
            ds = d.strftime("%Y-%m-%d")
            try:
                df = fetch_foreign_shareholding_direct(ds)
            except Exception as e:                                  # noqa: BLE001
                print(f"  {ds} 例外：{e}", flush=True)
                df = None
            if df is None or df.empty:
                n_empty += 1
            else:
                parts.append(df)
            time.sleep(SLEEP_SEC)
            n_done += 1
            if i % 25 == 0 or i == len(yd):
                el = time.time() - t0
                eta = el / max(n_done, 1) * (len(days) - n_done)
                print(f"  [{year}] {i}/{len(yd)}｜累計 {n_done}/{len(days)}｜"
                      f"已耗 {el/60:.1f} 分｜ETA {eta/60:.1f} 分｜空回應 {n_empty}",
                      flush=True)
            if limit and n_done >= limit:
                print(f"  已達 --limit {limit}，停止", flush=True)
                break

        if parts:
            y = pd.concat(parts, ignore_index=True)
            if limit:
                # 試跑模式**刻意不落檔**：落了半年的檔，正式跑時會因為
                # 「已存在，跳過」而永遠補不齊那一年，而且不會有任何警告。
                print(f"[{year}] 試跑取得 {len(y):,} 列 / {y['stock_id'].nunique()} 支"
                      f"（--limit 模式不落檔）", flush=True)
                _TRIAL.append(y)
            else:
                y.to_parquet(out, index=False)
                print(f"[{year}] 落檔 {len(y):,} 列 / {y['stock_id'].nunique()} 支 → {out.name}",
                      flush=True)
        else:
            print(f"[{year}] ⚠ 完全沒抓到資料，未落檔", flush=True)

        # 空回應比例過高＝被限流或端點改版，不要悶著頭跑完
        if yd and n_empty / len(yd) > 0.5:
            print(f"[{year}] ⚠ 空回應 {n_empty}/{len(yd)} 超過一半，中止。"
                  f"請確認是否被限流，或該年度端點版面不同。", flush=True)
            return
        if limit and n_done >= limit:
            return


# ============================================================
# 驗收
# ============================================================

def verify(staged: pd.DataFrame, stocks: list[str]) -> bool:
    """
    收尾一定要印分布，不能只印總數與最大值（資料層修復的教訓 2）。
    """
    ok = True
    print("\n" + "=" * 74)
    print("驗收")
    print("=" * 74)

    print(f"  抓到 {len(staged):,} 列 / {staged['stock_id'].nunique()} 支 / "
          f"{staged['date'].nunique()} 個交易日")
    print(f"  日期範圍 {staged['date'].min().date()} ~ {staged['date'].max().date()}")

    # ① 逐日列數：中位數附近才正常。2026-07-28 就是靠這個抓到 3 天只拿到上櫃
    daily = staged.groupby("date").size()
    thin = daily[daily < daily.median() * 0.5]
    print(f"  逐日列數 median {daily.median():.0f}｜min {daily.min()}｜max {daily.max()}")
    if len(thin):
        print(f"  ⚠ {len(thin)} 天列數不足中位數一半（可能只拿到單一市場）："
              f"{[str(d.date()) for d in thin.index[:8]]}")
        ok = False
    else:
        print("  ✓ 無列數異常偏低的日期")

    # ② 恆等式：持股比率 ≈ 持股數 / 發行股數 × 100
    #    這個檔有 SharesRatio（實際持股）與 RemainRatio（尚可投資空間）兩個
    #    語意相反的欄位，不驗恆等式會抓不出取錯欄的問題
    s = staged.dropna(subset=["ForeignInvestmentShares", "NumberOfSharesIssued",
                              "ForeignInvestmentSharesRatio"])
    s = s[s["NumberOfSharesIssued"] > 0]
    if len(s):
        implied = s["ForeignInvestmentShares"] / s["NumberOfSharesIssued"] * 100.0
        diff = (implied - s["ForeignInvestmentSharesRatio"]).abs()
        pass_rate = float((diff < 0.05).mean())
        print(f"  恆等式（持股數/發行股數×100 vs 持股比率）通過率 {pass_rate:.2%}"
              f"｜誤差 median {diff.median():.4f}")
        if pass_rate < 0.98:
            print("  ⚠ 通過率偏低，可能取到了 RemainRatio")
            ok = False
    else:
        print("  ⚠ 無法驗恆等式（缺欄位）")
        ok = False

    # ③ 目標股票的覆蓋
    got = set(staged["stock_id"])
    covered = [s2 for s2 in stocks if s2 in got]
    print(f"  目標股票覆蓋 {len(covered)}/{len(stocks)} 支")
    if len(covered) < len(stocks) * 0.9:
        print(f"  ⚠ 未覆蓋：{sorted(set(stocks) - got)[:15]}")
        ok = False

    # ④ 抽驗實際數值（規則 7：看得到數字才算做完）
    print("\n  抽驗（外資持股比率 %）：")
    for sid in ("9904", "9910", "9921", "9933"):
        sub = staged[staged["stock_id"] == sid].sort_values("date")
        if sub.empty:
            print(f"    {sid}: 無資料")
            continue
        head, tail = sub.iloc[0], sub.iloc[-1]
        print(f"    {sid}: {head['date'].date()} {head['ForeignInvestmentSharesRatio']:.2f}%"
              f"  →  {tail['date'].date()} {tail['ForeignInvestmentSharesRatio']:.2f}%"
              f"（{len(sub):,} 筆）")
    return ok


# ============================================================
# 併回 production
# ============================================================

def merge(staged: pd.DataFrame) -> None:
    prod = pd.read_parquet(FS_PATH)
    n0 = len(prod)

    # 只補既有沒有的鍵，既有列一律不動
    key_prod = set(map(tuple, prod[["date", "stock_id"]].astype(
        {"stock_id": str}).assign(date=pd.to_datetime(prod["date"])).to_numpy()))
    st = staged.copy()
    st["date"] = pd.to_datetime(st["date"])
    st["stock_id"] = st["stock_id"].astype(str)
    new = st[[t not in key_prod for t in map(tuple, st[["date", "stock_id"]].to_numpy())]]
    print(f"\n  staging {len(st):,} 列 → 扣掉既有已有的，實際新增 {len(new):,} 列")
    if new.empty:
        print("  沒有新列，不需合併")
        return

    # 「只改值、不改型別」：逐欄對齊 production 的 dtype
    for c in FS_COLS:
        if c not in new.columns:
            new[c] = pd.NA
    new = new[FS_COLS]
    for c in FS_COLS:
        if c == "date":
            continue
        try:
            new[c] = new[c].astype(prod[c].dtype)
        except (TypeError, ValueError):
            new[c] = new[c].astype(object)

    stamp = datetime.now().strftime("%Y%m%d")
    backup = FS_PATH.with_name(f"foreign_shareholding_raw_backup_{stamp}.parquet")
    if not backup.exists():
        prod.to_parquet(backup, index=False)
        print(f"  已備份 → {backup.name}")

    out = pd.concat([prod, new], ignore_index=True)
    out = out.drop_duplicates(subset=["date", "stock_id"], keep="first")
    out = out.sort_values(["date", "stock_id"], kind="mergesort").reset_index(drop=True)
    out.to_parquet(FS_PATH, index=False)
    print(f"  {n0:,} → {len(out):,} 列（+{len(out) - n0:,}）")
    print(f"  最新日期 {pd.to_datetime(out['date']).max().date()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只盤點缺口，不連網")
    ap.add_argument("--fetch", action="store_true", help="抓取（逐年落檔，可中斷續跑）")
    ap.add_argument("--merge", action="store_true", help="驗收並併回 production")
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 天（試跑用）")
    args = ap.parse_args()

    print("=" * 74)
    print("外資持股 9xxx 歷史缺口回補")
    print("=" * 74)
    miss, stocks, days = find_gaps()
    print(f"  缺口：{len(miss):,} 列 / {len(stocks)} 支 / {len(days)} 個交易日")
    print(f"  期間：{days[0].date()} ~ {days[-1].date()}" if days else "  無缺口")
    import collections
    pref = collections.Counter(s[0] for s in stocks)
    print(f"  首碼分布：{dict(sorted(pref.items()))}")
    # 實測每天約 2.6 秒（1.2s 間隔 + ~1.4s 網路往返 × 上市/上櫃兩個端點），
    # 不是只有 SLEEP_SEC。用實測值估才不會給出樂觀一倍的數字。
    print(f"  預估耗時：{len(days) * 2.6 / 60:.0f} 分鐘（實測每天約 2.6s）")

    if args.dry_run or not (args.fetch or args.merge):
        print("\n  （--dry-run：未連網。加 --fetch 開始抓取）")
        return

    if args.fetch:
        print(f"\n⚠ 執行期間請勿同時跑其他會打 TWSE 的作業"
              f"（2026-07-28 曾因此觸發 HTTP 307 限流）\n")
        fetch(days, limit=args.limit)

    if args.limit and _TRIAL:
        staged = pd.concat(_TRIAL, ignore_index=True)
        print(f"\n  （試跑模式：以記憶體中的 {len(staged):,} 列驗收，未寫入 staging）")
    else:
        files = sorted(STAGING.glob("fs_*.parquet"))
        if not files:
            print("\n  staging 無資料，結束")
            return
        staged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    staged["date"] = pd.to_datetime(staged["date"])
    staged["stock_id"] = staged["stock_id"].astype(str)
    staged = staged.drop_duplicates(subset=["date", "stock_id"], keep="first")

    ok = verify(staged, stocks)
    if args.merge:
        if not ok:
            print("\n  ✗ 驗收未通過，不合併。請先查清楚上面的警告。")
            return
        merge(staged)
    else:
        print("\n  （驗收完成。確認無誤後加 --merge 併回 production）")


if __name__ == "__main__":
    main()
