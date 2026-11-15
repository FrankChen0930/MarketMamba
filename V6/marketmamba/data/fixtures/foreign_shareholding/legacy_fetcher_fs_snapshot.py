"""LEGACY / TEST-ONLY verbatim snapshot of the foreign-shareholding functions from
V6/marketmamba/data/fetcher.py at git baseline 258ee09 (before the parser extraction).

Used only by test_foreign_shareholding_parity.py as the OLD reference. Do not import from
production code and do not edit (the body below is copied unchanged; only this header and the
minimal names the functions need -- logging/pandas/requests/HEADERS/Optional -- are added).
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketMamba/6.0)"}
logger = logging.getLogger("legacy_fetcher_fs_snapshot")


FS_COLS = [
    "date", "stock_id", "stock_name", "InternationalCode",
    "ForeignInvestmentRemainingShares", "ForeignInvestmentShares",
    "ForeignInvestmentRemainRatio", "ForeignInvestmentSharesRatio",
    "ForeignInvestmentUpperLimitRatio", "ChineseInvestmentUpperLimitRatio",
    "NumberOfSharesIssued", "RecentlyDeclareDate", "note",
]


def _fs_num(s) -> Optional[float]:
    """'25,219,056' / '87.86%' → float。"""
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _fs_identity_check(df: pd.DataFrame, tag: str) -> None:
    """
    恆等式驗證：`持股比率 ≈ 持有股數 / 發行股數 × 100`。

    ⚠️ 為什麼一定要驗這個：`foreign_shareholding_raw` 有兩個長得很像但**語意相反**
       的比率欄——`ForeignInvestmentSharesRatio`（外資實際持股%）與
       `ForeignInvestmentRemainRatio`（尚可投資空間%）。
       `feature_engineer._merge_foreign_shareholding` 用的是前者，對映錯會讓
       整個特徵符號顛倒且不會報錯。用恆等式驗過才知道自己對到哪一欄。
    """
    a = pd.to_numeric(df["ForeignInvestmentShares"], errors="coerce")
    b = pd.to_numeric(df["NumberOfSharesIssued"], errors="coerce")
    r = pd.to_numeric(df["ForeignInvestmentSharesRatio"], errors="coerce")
    m = (b > 0) & a.notna() & r.notna()
    if int(m.sum()) < 10:
        logger.warning(f"foreign_shareholding {tag}: 恆等式樣本不足")
        return
    calc = a[m] / b[m] * 100.0
    diff = (calc - r[m]).abs()
    ok = float((diff <= 0.05).mean())
    logger.info(f"foreign_shareholding {tag}: 恆等式 持股比率≈持股數/發行股數 "
                f"通過率 {ok:.1%}（n={int(m.sum()):,}，median 差 {diff.median():.4f}）")
    if ok < 0.9:
        logger.warning(f"foreign_shareholding {tag}: 恆等式通過率偏低，欄位對映可能有誤")


def fetch_foreign_shareholding_twse_direct(date: str) -> Optional[pd.DataFrame]:
    """上市外資及陸資持股（TWSE MI_QFIIS）。需 `selectType=ALLBUT0999` 才回全市場。"""
    ds = date.replace("-", "")
    try:
        resp = requests.get("https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS",
                            params={"date": ds, "selectType": "ALLBUT0999",
                                    "response": "json"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TWSE QFIIS fetch failed {date}: {e}")
        return None
    if str(data.get("stat", "")).upper() != "OK":
        return None
    if str(data.get("date", "")) != ds:            # 硬性核對回傳日期
        logger.warning(f"TWSE QFIIS: 回傳日期 {data.get('date')!r} != 請求 {ds!r}")
        return None

    rows, fields = data.get("data") or [], data.get("fields") or []
    if not rows:
        for t in data.get("tables", []):
            if t.get("data"):
                rows, fields = t["data"], t.get("fields") or []
                break
    if not rows or not fields:
        return None

    def _col(*names) -> int:
        for nm in names:
            for i, f in enumerate(fields):
                if nm in str(f):
                    return i
        return -1

    ix = {
        "stock_id": _col("證券代號"), "stock_name": _col("證券名稱"),
        "InternationalCode": _col("國際證券編碼"),
        "NumberOfSharesIssued": _col("發行股數"),
        "ForeignInvestmentRemainingShares": _col("尚可投資股數"),
        "ForeignInvestmentShares": _col("持有股數"),
        "ForeignInvestmentRemainRatio": _col("尚可投資比率"),
        "ForeignInvestmentSharesRatio": _col("持股比率"),
        "ForeignInvestmentUpperLimitRatio": _col("共用法令投資上限比率", "法令投資上限比率"),
        "ChineseInvestmentUpperLimitRatio": _col("陸資法令投資上限比率"),
    }
    if min(ix["stock_id"], ix["ForeignInvestmentShares"],
           ix["ForeignInvestmentSharesRatio"]) < 0:
        logger.warning(f"TWSE QFIIS: 版面無法辨識，fields={fields}")
        return None

    recs = []
    for r in rows:
        sid = str(r[ix["stock_id"]]).strip() if ix["stock_id"] < len(r) else ""
        if not sid.isdigit() or len(sid) != 4:
            continue
        rec = {"date": pd.Timestamp(date), "stock_id": sid, "note": ""}
        for k, i in ix.items():
            if k == "stock_id" or i < 0 or i >= len(r):
                continue
            rec[k] = str(r[i]).strip() if k in ("stock_name", "InternationalCode") \
                else _fs_num(r[i])
        recs.append(rec)
    if not recs:
        return None
    df = pd.DataFrame(recs)
    for c in FS_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[FS_COLS]
    _fs_identity_check(df, f"TWSE {date}")
    return df


def fetch_foreign_shareholding_tpex_direct(date: str) -> Optional[pd.DataFrame]:
    """
    上櫃外資及陸資持股（TPEX `insti/qfii`）。

    端點名不直觀（不是 forgnHold / foreignHold，那些都回 404 HTML），
    日期需 `YYYY/MM/DD`。實測正確認日期（2026/07/27、05/06、07/24 各回對應值）。
    """
    ds = date.replace("-", "/")
    try:
        resp = requests.get("https://www.tpex.org.tw/www/zh-tw/insti/qfii",
                            params={"date": ds, "response": "json"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TPEX qfii fetch failed {date}: {e}")
        return None
    if str(data.get("date", "")) != date.replace("-", ""):
        logger.warning(f"TPEX qfii: 回傳日期 {data.get('date')!r} != 請求 {date!r}")
        return None

    rows, fields = [], []
    for t in data.get("tables", []):
        if t.get("data"):
            rows, fields = t["data"], t.get("fields") or []
            break
    if not rows or not fields:
        return None

    def _col(*names) -> int:
        for nm in names:
            for i, f in enumerate(fields):
                if nm in str(f):
                    return i
        return -1

    ix = {
        "stock_id": _col("代號"), "stock_name": _col("名稱"),
        "NumberOfSharesIssued": _col("發行股數"),
        "ForeignInvestmentRemainingShares": _col("尚可投資股數"),
        "ForeignInvestmentShares": _col("持有股數"),
        "ForeignInvestmentRemainRatio": _col("尚可投資比率"),
        "ForeignInvestmentSharesRatio": _col("持股比率"),
        "ForeignInvestmentUpperLimitRatio": _col("法令投資上限比率"),
    }
    if min(ix["stock_id"], ix["ForeignInvestmentShares"],
           ix["ForeignInvestmentSharesRatio"]) < 0:
        logger.warning(f"TPEX qfii: 版面無法辨識，fields={fields}")
        return None

    recs = []
    for r in rows:
        sid = str(r[ix["stock_id"]]).strip() if ix["stock_id"] < len(r) else ""
        if not sid.isdigit() or len(sid) != 4:
            continue
        rec = {"date": pd.Timestamp(date), "stock_id": sid, "note": ""}
        for k, i in ix.items():
            if k == "stock_id" or i < 0 or i >= len(r):
                continue
            rec[k] = str(r[i]).strip() if k == "stock_name" else _fs_num(r[i])
        recs.append(rec)
    if not recs:
        return None
    df = pd.DataFrame(recs)
    for c in FS_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[FS_COLS]
    _fs_identity_check(df, f"TPEX {date}")
    return df


def fetch_foreign_shareholding_direct(date: str) -> Optional[pd.DataFrame]:
    """上市 + 上櫃外資持股合併。任一市場成功即回傳。"""
    parts = []
    for fn in (fetch_foreign_shareholding_twse_direct,
               fetch_foreign_shareholding_tpex_direct):
        d = fn(date)
        if d is not None and not d.empty:
            parts.append(d)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=["date", "stock_id"], keep="first")
