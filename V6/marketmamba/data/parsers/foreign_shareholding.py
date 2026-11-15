"""Foreign-shareholding (TWSE MI_QFIIS / TPEX insti/qfii) parser + normalization.

Extracted verbatim (behavior-preserving) from ``fetcher.py``.  Pure: no network, no config.
HTTP request / response-raise handling stays in ``fetcher``; this module only turns an already
decoded JSON payload into the ``FS_COLS`` frame.

``log`` (optional ``logging.Logger``) lets the fetcher keep emitting messages through its own
logger (log text/levels are unchanged); default is this module's logger.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FS_COLS = [
    "date", "stock_id", "stock_name", "InternationalCode",
    "ForeignInvestmentRemainingShares", "ForeignInvestmentShares",
    "ForeignInvestmentRemainRatio", "ForeignInvestmentSharesRatio",
    "ForeignInvestmentUpperLimitRatio", "ChineseInvestmentUpperLimitRatio",
    "NumberOfSharesIssued", "RecentlyDeclareDate", "note",
]


def fs_num(s) -> Optional[float]:
    """'25,219,056' / '87.86%' → float。"""
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def identity_check(df: pd.DataFrame, tag: str, log: Optional[logging.Logger] = None) -> None:
    """
    恆等式驗證：`持股比率 ≈ 持有股數 / 發行股數 × 100`。

    ⚠️ 為什麼一定要驗這個：`foreign_shareholding_raw` 有兩個長得很像但**語意相反**
       的比率欄——`ForeignInvestmentSharesRatio`（外資實際持股%）與
       `ForeignInvestmentRemainRatio`（尚可投資空間%）。
       `feature_engineer._merge_foreign_shareholding` 用的是前者，對映錯會讓
       整個特徵符號顛倒且不會報錯。用恆等式驗過才知道自己對到哪一欄。
    """
    lg = log or logger
    a = pd.to_numeric(df["ForeignInvestmentShares"], errors="coerce")
    b = pd.to_numeric(df["NumberOfSharesIssued"], errors="coerce")
    r = pd.to_numeric(df["ForeignInvestmentSharesRatio"], errors="coerce")
    m = (b > 0) & a.notna() & r.notna()
    if int(m.sum()) < 10:
        lg.warning(f"foreign_shareholding {tag}: 恆等式樣本不足")
        return
    calc = a[m] / b[m] * 100.0
    diff = (calc - r[m]).abs()
    ok = float((diff <= 0.05).mean())
    lg.info(f"foreign_shareholding {tag}: 恆等式 持股比率≈持股數/發行股數 "
            f"通過率 {ok:.1%}（n={int(m.sum()):,}，median 差 {diff.median():.4f}）")
    if ok < 0.9:
        lg.warning(f"foreign_shareholding {tag}: 恆等式通過率偏低，欄位對映可能有誤")


def _finish(recs, tag: str, date: str, log) -> Optional[pd.DataFrame]:
    if not recs:
        return None
    df = pd.DataFrame(recs)
    for c in FS_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[FS_COLS]
    identity_check(df, f"{tag} {date}", log)
    return df


def parse_twse_qfiis(payload: dict, date: str, log: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Parse a decoded TWSE MI_QFIIS JSON payload for ``date`` ('YYYY-MM-DD')."""
    lg = log or logger
    data = payload
    ds = date.replace("-", "")
    if str(data.get("stat", "")).upper() != "OK":
        return None
    if str(data.get("date", "")) != ds:            # 硬性核對回傳日期
        lg.warning(f"TWSE QFIIS: 回傳日期 {data.get('date')!r} != 請求 {ds!r}")
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
        lg.warning(f"TWSE QFIIS: 版面無法辨識，fields={fields}")
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
                else fs_num(r[i])
        recs.append(rec)
    return _finish(recs, "TWSE", date, lg)


def parse_tpex_qfiis(payload: dict, date: str, log: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Parse a decoded TPEX insti/qfii JSON payload for ``date`` ('YYYY-MM-DD')."""
    lg = log or logger
    data = payload
    if str(data.get("date", "")) != date.replace("-", ""):
        lg.warning(f"TPEX qfii: 回傳日期 {data.get('date')!r} != 請求 {date!r}")
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
        lg.warning(f"TPEX qfii: 版面無法辨識，fields={fields}")
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
            rec[k] = str(r[i]).strip() if k == "stock_name" else fs_num(r[i])
        recs.append(rec)
    return _finish(recs, "TPEX", date, lg)
