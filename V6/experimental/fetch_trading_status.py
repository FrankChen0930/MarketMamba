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

用法
----
    python V6/experimental/fetch_trading_status.py --years 2015 2026
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

OUT = Path(PROCESSED_DIR) / "trading_status_raw.parquet"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP = 0.8
# 起迄期間的分隔符號：TWSE 全形、TPEX 半形。兩種都要收（實測踩過）。
SEP = re.compile(r"[～~﹏－-]+")
STOCK_RE = re.compile(r"^\d{4}$")


def _roc_to_ad(s: str) -> str | None:
    """民國 `114/07/07` 或 `114.07.04` → 西元 `2025-07-07`。格式不符回 None（不猜）。

    ⚠️ **分隔符號兩種都要收**：TWSE 的 punish 用 `/`、notice 用 `.`（實測 2026-08-01）。
       只寫 `/` 的話 notice 會整批解析失敗、回 0 列，而且不會報錯。
    """
    m = re.match(r"^\s*(\d{2,3})[/.](\d{1,2})[/.](\d{1,2})\s*$", str(s))
    if not m:
        return None
    y, mo, d = int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))
    try:
        return f"{pd.Timestamp(year=y, month=mo, day=d).date()}"
    except ValueError:
        return None


def _get(url: str, params: dict) -> list | None:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        j = r.json()
    except Exception as e:                                        # noqa: BLE001
        print(f"    ⚠ {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        time.sleep(SLEEP)
    if isinstance(j, dict):
        if j.get("data") is not None:
            return j["data"]
        t = j.get("tables") or []
        if t and t[0].get("data") is not None:
            return t[0]["data"]
    return None


def fetch_twse(kind: str, y0: str, y1: str) -> list[dict]:
    """kind: disposal → punish（含起迄期間）／attention → notice（單日）。"""
    path = "punish" if kind == "disposal" else "notice"
    rows = _get(f"https://www.twse.com.tw/rwd/zh/announcement/{path}",
                {"response": "json", "startDate": y0.replace("-", ""),
                 "endDate": y1.replace("-", "")}) or []
    out = []
    for r in rows:
        try:
            if kind == "disposal":
                sid, ann, span = str(r[2]).strip(), _roc_to_ad(r[1]), str(r[6])
                parts = [p for p in SEP.split(span) if p.strip()]
                if len(parts) != 2:
                    continue
                s, e = _roc_to_ad(parts[0]), _roc_to_ad(parts[1])
            else:
                sid, ann = str(r[1]).strip(), _roc_to_ad(r[5])
                s = e = ann                                   # 注意股是單日公告
            if not STOCK_RE.match(sid) or not s or not e:
                continue
            out.append({"stock_id": sid, "start": s, "end": e,
                        "announced": ann, "market": "twse"})
        except (IndexError, TypeError):
            continue
    return out


def fetch_tpex(kind: str, y0: str, y1: str) -> list[dict]:
    path = "disposal" if kind == "disposal" else "attention"
    rows = _get(f"https://www.tpex.org.tw/www/zh-tw/bulletin/{path}",
                {"response": "json", "startDate": y0.replace("-", "/"),
                 "endDate": y1.replace("-", "/")}) or []
    out = []
    for r in rows:
        try:
            # ⚠️ 欄序兩種公告不同（實測 2026-08-01）：
            #   disposal : 編號 / 公布日期 / **證券代號** / 名稱 / 累計 / 起訖 / ...
            #   attention: 編號 / **證券代號** / 名稱 / 累計 / 注意資訊 / 公告日期 / ...
            # 兩邊都用 r[2] 的話，attention 會拿到「證券名稱」→ 過不了 ^\d{4}$ → 0 列。
            if kind == "disposal":
                sid, ann = str(r[2]).strip(), _roc_to_ad(r[1])
            else:
                sid, ann = str(r[1]).strip(), _roc_to_ad(r[5])
            if kind == "disposal":
                parts = [p for p in SEP.split(str(r[5])) if p.strip()]
                if len(parts) != 2:
                    continue                                  # 含「本日無處置資料」骨架列
                s, e = _roc_to_ad(parts[0]), _roc_to_ad(parts[1])
            else:
                s = e = ann
            if not STOCK_RE.match(sid) or not s or not e:
                continue
            out.append({"stock_id": sid, "start": s, "end": e,
                        "announced": ann, "market": "tpex"})
        except (IndexError, TypeError):
            continue
    return out


def expand_to_daily(recs: list[dict], status: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """把 [start, end] 期間展開成逐日列，只保留真實交易日。"""
    out = []
    for r in recs:
        s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        if e < s:
            continue
        days = calendar[(calendar >= s) & (calendar <= e)]
        for d in days:
            out.append((str(d.date()), r["stock_id"], status, r["market"], r["announced"]))
    return pd.DataFrame(out, columns=["Date", "stock_id", "status", "market", "announced"])


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
            a, b = fetch_twse(kind, y0, y1), fetch_tpex(kind, y0, y1)
            print(f"[ts] {y} {kind:10s} twse {len(a):>4} 筆｜tpex {len(b):>4} 筆", flush=True)
            for recs in (a, b):
                if recs:
                    frames.append(expand_to_daily(recs, kind, calendar))

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
