"""
MarketMamba V6 — Data Fetcher
==============================
Hybrid data-source strategy:
  Layer 1 (fastest) : yfinance  → OHLCV price/volume (available 15 min after close)
  Layer 2 (fast)    : TWSE/TPEX direct → institutional investors (~16:30–17:00)
  Layer 3 (fallback): FinMind   → margin/short, 8 major banks, etc. (18:00–19:00)
                                  Forward-Fill used if not yet updated.

Target: inference can start at ~17:00 instead of 19:00+.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from marketmamba.config import (
    DATA_DIR,
    DATA_END_DATE,
    DATA_SOURCE_PRIORITY,
    DATA_START_DATE,
    FINMIND_TOKEN,
    MARGIN_FORWARD_FILL,
    PROCESSED_DIR,
    TPEX_INSTITUTIONAL_URL,
    TWSE_INSTITUTIONAL_URL,
)

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
RAW_DIR     = DATA_DIR / "raw_v6"
CACHE_DIR   = DATA_DIR / "cache_v6"
for _d in [RAW_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketMamba/6.0)"}
FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"

# FinMind 免費層（level=1 "Free"）的實測額度（2026-07-29 由 /v2/user_info 查得）：
#     api_request_limit_hour = 600
#     api_request_limit_day  = 600
# 也就是**每日 600 次**。逐股補齊 ~1,900 支需要跨多天，這是「不訂閱 VIP」的實際節奏限制。
FINMIND_FREE_DAILY_LIMIT = 600


class FinMindQuotaExceeded(RuntimeError):
    """
    FinMind 拒絕服務：HTTP 402（當日額度用盡）或 403（IP 被封鎖）。

    刻意用例外而不是回傳 None：這與「這支股票沒有新資料」是**完全不同的事**，
    但兩者在舊版都表現為 `None`。2026-07-29 的回補因此連跑六輪、試了 900 支、
    每輪淨增 0 列，日誌卻一切正常——問題被完美地偽裝成「已經追平了」。

    ⚠️ 2026-07-29 教訓：短時間內密集打 ~1,000 次請求後，FinMind 直接回
    `403 ip banned`（而非只是 402）。免費層的 600 次/日不是可以一次用完的預算，
    **請求速率本身也會觸發封鎖**。因此每日滾動批量刻意設得遠低於額度，
    且遇到本例外一律立即中止、不重試。
    """

# ============================================================
# Ticker Universe Helpers
# ============================================================

def load_ticker_universe() -> tuple[list[str], list[str]]:
    """
    Load TSE (TWSE) and OTC (TPEX) stock ID lists.
    Returns (tse_ids, otc_ids) as plain 4-digit strings, e.g. ['2330', '2317', ...]
    Falls back to fetching from FinMind if local cache is missing.
    """
    cache_path = CACHE_DIR / "ticker_universe.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = _fetch_universe_from_finmind()
        df.to_parquet(cache_path)

    tse = df[df["market"] == "TSE"]["stock_id"].tolist()
    otc = df[df["market"] == "OTC"]["stock_id"].tolist()
    logger.info(f"Universe: {len(tse)} TSE + {len(otc)} OTC stocks")
    return tse, otc


def _fetch_universe_from_finmind() -> pd.DataFrame:
    """Fetch full stock list from FinMind (TaiwanStockInfo)."""
    logger.info("Fetching ticker universe from FinMind...")
    params = {
        "dataset": "TaiwanStockInfo",
        "token": FINMIND_TOKEN,
    }
    resp = requests.get(FINMIND_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != 200:
        raise RuntimeError(f"FinMind error: {data.get('msg')}")
    df = pd.DataFrame(data["data"])
    logger.info(f"  TaiwanStockInfo columns: {list(df.columns)}")

    # Filter: only ordinary 4-digit stock IDs
    df = df[df["stock_id"].str.match(r"^\d{4}$")].copy()

    # --- Determine exchange type ---
    # FinMind uses 'type' column (not 'market'), with values 'twse' / 'tpex'
    # Possible column names across different FinMind API versions:
    _market_col = None
    for _candidate in ["type", "market", "market_category", "exchange"]:
        if _candidate in df.columns:
            _market_col = _candidate
            break

    if _market_col:
        # Map FinMind exchange codes → our TSE / OTC convention
        _market_map = {
            "twse": "TSE", "TSE": "TSE", "sii": "TSE", "上市": "TSE",
            "tpex": "OTC", "OTC": "OTC", "otc": "OTC", "上櫃": "OTC",
        }
        df["market"] = df[_market_col].map(_market_map).fillna("TSE")
        logger.info(f"  Market column '{_market_col}' mapped → TSE/OTC")
    else:
        # No exchange column found — default everything to TSE
        # yfinance will fail on OTC suffix and we'll catch it via the missing list
        logger.warning("  No market/type column in TaiwanStockInfo — defaulting all to TSE")
        df["market"] = "TSE"

    tse_count = (df["market"] == "TSE").sum()
    otc_count  = (df["market"] == "OTC").sum()
    logger.info(f"  Universe: {tse_count} TSE + {otc_count} OTC stocks")

    return df[["stock_id", "stock_name", "industry_category", "market"]].reset_index(drop=True)


# ============================================================
# Layer 1: yfinance — Price/Volume
# ============================================================

def fetch_prices_yfinance(
    tse_ids: list[str],
    otc_ids: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Batch-download OHLCV for all stocks via yfinance.
    Returns long-format DataFrame with columns:
      [Date, stock_id, Open, High, Low, Close, Volume]
    Stocks missing from yfinance are flagged for FinMind fallback.
    """
    all_tickers = [f"{s}.TW" for s in tse_ids] + [f"{s}.TWO" for s in otc_ids]
    id_map = {f"{s}.TW": s for s in tse_ids}
    id_map.update({f"{s}.TWO": s for s in otc_ids})

    logger.info(f"yfinance: downloading {len(all_tickers)} tickers [{start} -> {end}]")

    # yfinance batch download — split into chunks to avoid rate limits
    BATCH_SIZE = 200  # yfinance starts rate-limiting above ~300 tickers at once
    all_records = []
    all_missing = []

    for batch_start in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[batch_start: batch_start + BATCH_SIZE]
        logger.info(f"  Batch {batch_start // BATCH_SIZE + 1}: {len(batch)} tickers...")

        for attempt in range(3):  # retry up to 3 times on rate limit
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                break
            except Exception as e:
                if "Too Many Requests" in str(e) or "429" in str(e):
                    wait = 30 * (attempt + 1)
                    logger.warning(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"  yfinance batch error: {e}")
                    raw = None
                    break
        else:
            # All 3 retries failed
            all_missing.extend(batch)
            continue

        if raw is None or raw.empty:
            all_missing.extend(batch)
            continue

        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    # Detect which level contains ticker names
                    # Old yfinance (group_by='ticker'): level 0 = Ticker, level 1 = Price
                    # New yfinance (>= 0.2.18):         level 0 = Price,  level 1 = Ticker
                    lvl0_vals = raw.columns.get_level_values(0).unique()
                    lvl1_vals = raw.columns.get_level_values(1).unique()
                    if ticker in lvl0_vals:
                        df_t = raw[ticker]           # old API
                    elif ticker in lvl1_vals:
                        df_t = raw.xs(ticker, axis=1, level=1)  # new API
                    else:
                        all_missing.append(ticker)
                        continue
                    df_t = df_t.dropna(subset=["Close"])
                else:
                    # Single-level columns (single ticker batch)
                    df_t = raw.dropna(subset=["Close"]) if len(batch) == 1 else pd.DataFrame()
            except (KeyError, ValueError):
                all_missing.append(ticker)
                continue
            if df_t.empty:
                all_missing.append(ticker)
                continue
            df_t = df_t.reset_index()
            df_t["stock_id"] = id_map[ticker]
            df_t["Date"] = pd.to_datetime(df_t["Date"]).dt.date
            all_records.append(df_t[["Date", "stock_id", "Open", "High", "Low", "Close", "Volume"]])

        if batch_start + BATCH_SIZE < len(all_tickers):
            time.sleep(3)  # polite pause between batches

    # Deduplicate: some delisted tickers appear in both TSE and OTC lists
    delisted_count  = sum(1 for t in all_missing if t in ["YFTzMissingError", "YFPricesMissingError"])
    rate_limit_count = len(all_missing) - delisted_count
    if all_missing:
        logger.warning(
            f"yfinance: {len(all_missing)} tickers unavailable "
            f"(many are delisted — this is expected for 2012+ historical data)"
        )
    if all_records:
        df_prices = pd.concat(all_records, ignore_index=True)
    else:
        df_prices = pd.DataFrame()
        logger.warning("yfinance returned no usable data")

    return df_prices, all_missing


# ============================================================
# Layer 2: TWSE / TPEX Direct — Prices + Institutional Investors
# ============================================================

def fetch_prices_twse_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetch all TWSE stocks' daily OHLCV from TWSE MI_INDEX.
    Available ~15:00 on trading day. No auth required.
    """
    date_compact = date_str.replace("-", "")
    url = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
    params = {"response": "json", "date": date_compact, "type": "ALLBUT0999"}
    try:
        resp = requests.get(url, params=params, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            logger.warning(f"TWSE price: HTTP {resp.status_code}")
            return None
        data = resp.json()
        if data.get("stat") != "OK":
            logger.warning(f"TWSE price: stat={data.get('stat')}")
            return None

        # MI_INDEX returns a 'tables' list; the individual stock table is the
        # one whose fields include 開盤價 (Open).
        tables = data.get("tables", [])
        target = next(
            (t for t in tables
             if isinstance(t.get("fields"), list)
             and any("開盤" in f for f in t["fields"])
             and len(t.get("data", [])) > 100),
            None,
        )
        if target is None:
            logger.warning("TWSE price: no price table found in response")
            return None

        fields = target["fields"]
        rows   = target["data"]

        def _col(name: str) -> int:
            return next((i for i, f in enumerate(fields) if name in f), -1)

        i_id    = _col("代號")
        i_open  = _col("開盤")
        i_high  = _col("最高")
        i_low   = _col("最低")
        i_close = _col("收盤")
        i_vol   = _col("成交股數")

        if any(x < 0 for x in [i_id, i_open, i_high, i_low, i_close]):
            logger.warning(f"TWSE price: unexpected fields {fields}")
            return None

        def _num(s: str) -> Optional[float]:
            try:
                return float(str(s).replace(",", "").strip())
            except (ValueError, TypeError):
                return None

        records = []
        for row in rows:
            try:
                sid = str(row[i_id]).strip()
                if not sid.isdigit():
                    continue
                c = _num(row[i_close])
                if c is None:
                    continue  # suspended / no trade
                records.append({
                    "Date":     date_str,
                    "stock_id": sid,
                    "Open":     _num(row[i_open])  or c,
                    "High":     _num(row[i_high])  or c,
                    "Low":      _num(row[i_low])   or c,
                    "Close":    c,
                    "Volume":   _num(row[i_vol])   or 0.0 if i_vol >= 0 else 0.0,
                })
            except (IndexError, ValueError):
                continue

        if not records:
            return None
        df = pd.DataFrame(records)
        logger.info(f"TWSE direct prices: {len(df)} stocks for {date_str}")
        return df

    except Exception as e:
        logger.warning(f"TWSE direct price fetch failed: {e}")
        return None


def fetch_prices_tpex_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    上櫃個股日 OHLCV（TPEX /www/zh-tw/afterTrading/otc，全市場單一天一次請求）。

    ⚠️⚠️ 2026-07-27 重寫——舊版有嚴重的靜默資料污染
    ------------------------------------------------------
    舊版打 `openapi/v1/tpex_mainboard_daily_close_quotes`，該端點**完全忽略 date
    參數**：實測傳 '2026-07-16' / '2026-07-24' / '2026/07/16' / '20260716' / 不傳，
    五種情況回傳的資料**逐位元相同**，首列 Date 永遠是當天（民國 '1150727'）。
    而舊版又把 `"Date": date_str` 硬寫進每一列 → 等於「抓今天的報價、蓋上你要求的
    日期」寫進 prices_raw，且不會報任何錯。
    這條路徑在 run_daily_update 裡是 yfinance OTC 覆蓋率 <30% 時的 fallback，
    2026-05 起一直在觸發。實測後果：prices_raw 的 OTC 列對交易所真值
    Volume 比值 median 0.93（p10 0.38 / p90 2.20，248/835 檔偏差 >2 倍），
    且偏差隨日期距今越遠越大——正是「回傳當天資料」的指紋。
    （對照組：TWSE MI_INDEX 來源的列 Volume 比值 median 1.0011、Close 零偏差。）

    現版改用 `/www/zh-tw/afterTrading/otc`，實測會正確遵守日期
    （傳 2026/07/16 回 date='20260716'，傳 2026/07/24 回 '20260724'），
    並加上回傳日期硬性核對。

    版面（2026-07-27 實測，17 欄）—— ⚠️ 收盤在 idx 2、開盤在 idx 4，**收盤在開盤之前**，
    與一般 OHLC 順序及舊 OpenAPI 的欄名對映都不同：
      0 代號 | 1 名稱 | 2 收盤 | 3 漲跌 | 4 開盤 | 5 最高 | 6 最低
      7 成交股數 | 8 成交金額 | 9 成交筆數 | ...
    價格為交易所原始價（未還原），與 yfinance 的 auto_adjust 還原價不同基準，
    因此 run_daily_update 的多來源合併採 keep="first"（yfinance 優先）。
    """
    date_slash = date_str.replace("-", "/")          # 此端點只認斜線格式
    date_compact = date_str.replace("-", "")
    url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc"
    try:
        # ⚠️ type 參數不可省：不傳時 stat 仍為 'ok'、date 也正確，但資料表是空的
        #    （2026-07-27 實測 maxn=0）。EW = 上櫃主板；AL 會混入權證等 10,128 筆。
        resp = requests.get(url,
                            params={"date": date_slash, "type": "EW", "response": "json"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TPEX direct price fetch failed for {date_str}: {e}")
        return None

    if str(data.get("stat", "")).lower() not in ("ok", "success"):
        logger.info(f"TPEX price {date_str}: stat={data.get('stat')}（非交易日或尚未公布）")
        return None
    got = str(data.get("date", "")).replace("-", "").replace("/", "")
    if got != date_compact:
        logger.warning(
            f"TPEX price: 回傳日期 {got or None} != 請求 {date_compact}，捨棄"
            f"（此類端點會靜默回傳當天資料，見 docstring）"
        )
        return None

    def _num(s) -> Optional[float]:
        try:
            return float(str(s).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    records = []
    for table in data.get("tables", []):
        fields = table.get("fields") or []
        if len(fields) < 8 or not any("代號" in f for f in fields):
            continue
        for row in table.get("data") or []:
            if len(row) < 8:
                continue                     # 骨架列（非交易日）
            sid = str(row[0]).strip()
            if not sid.isdigit() or len(sid) != 4:
                continue
            c = _num(row[2])                 # 收盤（注意：在開盤之前）
            if c is None or c <= 0:
                continue
            records.append({
                "Date":     date_str,
                "stock_id": sid,
                "Open":     _num(row[4]) or c,
                "High":     _num(row[5]) or c,
                "Low":      _num(row[6]) or c,
                "Close":    c,
                "Volume":   _num(row[7]) or 0.0,
            })

    if not records:
        logger.info(f"TPEX price {date_str}: 無有效數值列")
        return None
    df = pd.DataFrame(records)
    logger.info(f"TPEX direct prices: {len(df)} stocks for {date_str}")
    return df


def fetch_institutional_twse(date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetch TSE three-institutional investors (三大法人) directly from TWSE.
    Usually available by 16:30–17:00, ~30–60 min ahead of FinMind.

    Args:
        date_str: 'YYYY-MM-DD'
    Returns:
        DataFrame with columns [stock_id, Foreign_Net, Investment_Trust_Net, Dealer_Net]
        or None if data not yet published.
    """
    date_compact = date_str.replace("-", "")
    # selectType=ALLBUT0999 = 全部（不含權證/ETN）。沒有這個參數時 TWSE 只回
    # 預設類股（水泥工業 7 支）——2026-04-25 起的每日更新就是因此只寫進 7 支股票。
    params = {"date": date_compact, "selectType": "ALLBUT0999", "response": "json"}
    try:
        resp = requests.get(
            TWSE_INSTITUTIONAL_URL,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"TWSE institutional fetch failed: {e}")
        return None

    if data.get("stat") != "OK":
        logger.info(f"TWSE institutional: data not ready for {date_str} (stat={data.get('stat')})")
        return None

    rows = data.get("data", [])
    if not rows:
        return None

    records = []
    for row in rows:
        # TWSE T86 (selectType=ALLBUT0999) 19 欄：
        # [0]代號 [1]名稱 [2]外陸資買(不含外資自營) [3]外陸資賣 [4]外陸資淨
        # [5-7]外資自營買/賣/淨 [8]投信買 [9]投信賣 [10]投信淨
        # [11]自營商淨 [12-17]自營自行/避險明細 [18]三大法人淨
        try:
            if len(row) < 19:
                continue
            stock_id = str(row[0]).strip()
            if not stock_id.isdigit() or len(stock_id) != 4:
                continue
            def _parse_int(s: str) -> int:
                return int(str(s).replace(",", "").replace("--", "0") or "0")

            records.append({
                "stock_id":              stock_id,
                "Foreign_Buy":           _parse_int(row[2]),
                "Foreign_Sell":          _parse_int(row[3]),
                "Foreign_Net":           _parse_int(row[4]),
                "Investment_Trust_Buy":  _parse_int(row[8]),
                "Investment_Trust_Sell": _parse_int(row[9]),
                "Investment_Trust_Net":  _parse_int(row[10]),
                "Dealer_Net":            _parse_int(row[11]),
            })
        except (IndexError, ValueError):
            continue

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info(f"TWSE direct: {len(df)} stocks institutional data for {date_str}")
    return df


# TPEX 網站 2024 改版後舊端點（.../3insti/daily_trade/3itrade_hedge.php）只回 HTML。
# 新端點常數放這裡而非 config.py：本機 config.py 是刻意保持 56 維、不 commit 的
# dirty 檔，改它無法安全推上 remote。
TPEX_INSTITUTIONAL_URL_V2 = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"


def fetch_institutional_tpex(date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetch OTC (TPEX) institutional investor data directly from TPEX (新版 API).
    Returns same schema as fetch_institutional_twse.
    """
    params = {
        "type": "Daily",
        "sect": "EW",
        "date": date_str.replace("-", "/"),   # YYYY/MM/DD
        "response": "json",
    }
    try:
        resp = requests.get(
            TPEX_INSTITUTIONAL_URL_V2,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"TPEX institutional fetch failed: {e}")
        return None

    tables = data.get("tables", [])
    rows = tables[0].get("data", []) if tables else []
    if not rows:
        return None

    records = []
    for row in rows:
        # TPEX dailyTrade 24 欄（已用「買-賣=淨、分項加總=合計」數值驗證）：
        # [2-4]外資及陸資買/賣/淨(不含外資自營) [5-7]外資自營 [8-10]外資合計
        # [11-13]投信買/賣/淨 [14-16]自營自行 [17-19]自營避險 [20-22]自營合計 [23]三大法人淨
        try:
            if len(row) < 24:
                continue
            stock_id = str(row[0]).strip()
            if not stock_id.isdigit() or len(stock_id) != 4:
                continue
            def _p(s: str) -> int:
                return int(str(s).replace(",", "") or "0")
            records.append({
                "stock_id":              stock_id,
                "Foreign_Buy":           _p(row[2]),
                "Foreign_Sell":          _p(row[3]),
                "Foreign_Net":           _p(row[4]),
                "Investment_Trust_Buy":  _p(row[11]),
                "Investment_Trust_Sell": _p(row[12]),
                "Investment_Trust_Net":  _p(row[13]),
                "Dealer_Net":            _p(row[22]),
            })
        except (IndexError, ValueError):
            continue

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info(f"TPEX direct: {len(df)} stocks institutional data for {date_str}")
    return df


# ============================================================
# Layer 2b: TWSE Direct — PER/PBR、借券、股本（2026-07-24 新增）
# ============================================================
# 緣起：per_raw/securities_raw/market_value_raw 三個檔案原本靠 FinMind 整批查詢
# 建置，使用者 FinMind VIP 到期後整批查詢被降級擋掉（免費 register 等級只開放
# 帶 data_id 的單股查詢）。這三個改用 TWSE 官方公開端點，免費、無 FinMind
# 額度限制，且跟現有 fetch_prices_twse_direct/fetch_institutional_twse 是同一種
# 「官方直連」模式。三個函式均為純新增，不影響任何既有函式。

def fetch_per_twse_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    上市個股日本益比、殖利率及股價淨值比（TWSE BWIBBU_ALL，全市場單一天一次請求）。
    只涵蓋上市（TSE），上櫃（OTC）目前沒有找到對應官方端點，仍缺。
    """
    date_compact = date_str.replace("-", "")
    url = "https://www.twse.com.tw/exchangeReport/BWIBBU_ALL"
    try:
        resp = requests.get(url, params={"date": date_compact, "response": "json"},
                             headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"TWSE PER fetch failed: {e}")
        return None

    rows = data.get("data", [])
    if not rows:
        return None

    def _num(s: str) -> Optional[float]:
        try:
            return float(str(s).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    records = []
    for row in rows:
        try:
            sid = str(row[0]).strip()
            if not sid.isdigit() or len(sid) != 4:
                continue
            records.append({
                "Date":              date_str,
                "stock_id":          sid,
                "PER":               _num(row[2]),
                "dividend_yield":    _num(row[3]),
                "PBR":               _num(row[4]),
            })
        except (IndexError, ValueError):
            continue

    if not records:
        return None
    df = pd.DataFrame(records)
    logger.info(f"TWSE direct: {len(df)} stocks PER data for {date_str}")
    return df


def fetch_securities_lending_twse_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    上市上櫃股票當日可借券賣出股數（TWSE SBL/TWT96U，全市場單一天一次請求）。
    回傳資料是「雙欄並排」格式（每列含兩支股票），且股票代號欄位包在
    <a href=...>代號</a> 裡，需要拆開還原成兩筆記錄並清掉 HTML tag。
    """
    date_compact = date_str.replace("-", "")
    url = "https://www.twse.com.tw/SBL/TWT96U"
    try:
        resp = requests.get(url, params={"date": date_compact, "response": "json"},
                             headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"TWSE securities lending fetch failed: {e}")
        return None

    rows = data.get("data", [])
    if not rows:
        return None

    import re
    _tag_re = re.compile(r"<a[^>]*>([^<]+)</a>")

    def _strip_tag(s: str) -> str:
        m = _tag_re.match(str(s).strip())
        return m.group(1) if m else str(s).strip()

    def _num(s: str) -> Optional[float]:
        try:
            return float(str(s).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    records = []
    for row in rows:
        # 每列兩組 (代號, 可借券股數)：[0:2] 與 [2:4]
        for i in (0, 2):
            try:
                sid = _strip_tag(row[i])
                if not sid.isdigit() or len(sid) != 4:
                    continue
                bal = _num(row[i + 1])
                if bal is None:
                    continue
                records.append({
                    "Date": date_str,
                    "stock_id": sid,
                    "Securities_Balance": bal,
                })
            except (IndexError, ValueError):
                continue

    if not records:
        return None
    df = pd.DataFrame(records)
    logger.info(f"TWSE direct: {len(df)} stocks securities lending data for {date_str}")
    return df


# ============================================================
# 融資融券 — 交易所直連（2026-07-27 新增，取代 FinMind 路徑）
# ============================================================
# 為什麼改直連：FinMind 免費層強迫逐股查詢（~2,000 次呼叫 / 3.6 小時），
# 交易所端點是「逐日整批」（1 次呼叫回全市場，60 個交易日約 1.5 分鐘）。
# 直連不只免費，回補還比 FinMind VIP 快，因為差別在 API 形狀而非速率上限。
#
# 輸出欄位直接用 production margin_raw.parquet 的名稱，避免再經過
# feature_engineer._merge_margin 的 legacy_map 轉換。
MARGIN_COLS = ["Margin_Purchase", "Margin_Repay", "Short_Sale",
               "Short_Cover", "Margin_Balance", "Short_Balance"]


def _margin_num(s) -> float:
    """交易所回傳的數字含千分位逗號；無法解析（'--'、空白）一律視為 0。"""
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _margin_identity_check(src: str, date_str: str, rows: list[dict]) -> None:
    """
    數值驗證（規則 7：數值明確輸出）：
      融資今日餘額 = 融資前日餘額 + 融資買進 - 融資賣出 - 現金償還
      融券今日餘額 = 融券前日餘額 + 融券賣出 - 融券買進 - 現券償還
    欄序若對映錯誤（T86 曾發生過），這兩條恆等式會大量不成立。
    不成立不阻擋寫入，只告警——資券互抵等情況本來就有少數例外。
    """
    bad = 0
    for r in rows:
        m_ok = abs((r["_m_prev"] + r["Margin_Purchase"] - r["Margin_Repay"]
                    - r["_m_repay"]) - r["Margin_Balance"]) <= 1
        s_ok = abs((r["_s_prev"] + r["Short_Sale"] - r["Short_Cover"]
                    - r["_s_repay"]) - r["Short_Balance"]) <= 1
        if not (m_ok and s_ok):
            bad += 1
    rate = bad / max(len(rows), 1)
    msg = (f"{src} margin {date_str}：{len(rows)} 檔，恆等式不成立 {bad} 檔 "
           f"({rate:.2%})")
    if rate > 0.05:
        logger.warning(msg + "  ⚠️ 超過 5%，欄位對映可能有誤")
    else:
        logger.info(msg + "  ✓")


def fetch_margin_twse_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    上市個股融資融券餘額（TWSE MI_MARGN，全市場單一天一次請求）。

    版面（2026-07-27 實測，tables[1]「融資融券彙總」，16 欄）：
      0 代號 | 1 名稱
      2 融資買進 | 3 融資賣出 | 4 融資現金償還 | 5 融資前日餘額 | 6 融資今日餘額 | 7 融資限額
      8 融券買進 | 9 融券賣出 | 10 融券現券償還 | 11 融券前日餘額 | 12 融券今日餘額 | 13 融券限額
      14 資券互抵 | 15 註記
    單位：張（交易單位），與 FinMind TaiwanStockMarginPurchaseShortSale 一致。

    ⚠️ TPEX 的融券欄序是「券賣、券買」，與此處 TWSE 的「買進、賣出」**相反**，
       兩個 fetcher 不可共用索引。
    """
    date_compact = date_str.replace("-", "")
    url = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
    try:
        resp = requests.get(url, params={"date": date_compact, "selectType": "ALL",
                                         "response": "json"},
                            headers=HEADERS, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TWSE margin fetch failed for {date_str}: {e}")
        return None

    if str(data.get("stat", "")).upper() != "OK":
        logger.info(f"TWSE margin {date_str}: stat={data.get('stat')}（非交易日或尚未公布）")
        return None
    # 硬性核對回傳日期——避免端點忽略參數而回到別的日期（TPEX 已實證會這樣）
    if str(data.get("date", "")).replace("-", "") != date_compact:
        logger.warning(f"TWSE margin: 回傳日期 {data.get('date')} != 請求 {date_compact}，捨棄")
        return None

    tables = data.get("tables", [])
    target = next((t for t in tables
                   if isinstance(t.get("fields"), list) and len(t["fields"]) >= 16
                   and any("代號" in f for f in t["fields"])), None)
    if target is None or not target.get("data"):
        logger.warning(f"TWSE margin {date_str}: 找不到個股彙總表")
        return None

    records = []
    for row in target["data"]:
        try:
            sid = str(row[0]).strip()
            if not sid.isdigit() or len(sid) != 4:
                continue
            records.append({
                "Date": date_str, "stock_id": sid,
                "Margin_Purchase": _margin_num(row[2]),    # 融資買進
                "Margin_Repay":    _margin_num(row[3]),    # 融資賣出
                "Margin_Balance":  _margin_num(row[6]),    # 融資今日餘額
                "Short_Cover":     _margin_num(row[8]),    # 融券買進（回補）
                "Short_Sale":      _margin_num(row[9]),    # 融券賣出
                "Short_Balance":   _margin_num(row[12]),   # 融券今日餘額
                "_m_prev":  _margin_num(row[5]),  "_m_repay": _margin_num(row[4]),
                "_s_prev":  _margin_num(row[11]), "_s_repay": _margin_num(row[10]),
            })
        except (IndexError, ValueError):
            continue

    if not records:
        return None
    _margin_identity_check("TWSE", date_str, records)
    return pd.DataFrame(records)[["Date", "stock_id"] + MARGIN_COLS]


def fetch_margin_tpex_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    上櫃個股融資融券餘額（TPEX /www/zh-tw/margin/balance，全市場單一天一次請求）。

    ⚠️⚠️ 日期格式必須是 **YYYY/MM/DD**。2026-07-27 實測：用 YYYYMMDD 或
       YYYY-MM-DD 時此端點會**靜默忽略參數、回傳當天資料，且 stat 仍為 'ok'**。
       若照 TWSE 的 compact 格式寫，整段回補會變成「今天的數字蓋上歷史日期」
       而完全不報錯。因此除了用正確格式，下方另有回傳日期的硬性核對。

    版面（2026-07-27 實測，tables[0]，20 欄）：
      0 代號 | 1 名稱
      2 前資餘額 | 3 資買 | 4 資賣 | 5 現償 | 6 資餘額 | 7 資屬證金 | 8 資使用率 | 9 資限額
      10 前券餘額 | 11 券賣 | 12 券買 | 13 券償 | 14 券餘額 | 15 券屬證金 | 16 券使用率 | 17 券限額
      18 資券相抵 | 19 備註
      → 11/12 是「券賣、券買」，與 TWSE 的 8/9「買進、賣出」順序相反。
    單位：張。
    """
    date_slash = date_str.replace("-", "/")                   # 必須用斜線格式
    date_compact = date_str.replace("-", "")
    url = "https://www.tpex.org.tw/www/zh-tw/margin/balance"
    try:
        resp = requests.get(url, params={"date": date_slash, "response": "json"},
                            headers=HEADERS, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TPEX margin fetch failed for {date_str}: {e}")
        return None

    if str(data.get("stat", "")).lower() not in ("ok", "success"):
        logger.info(f"TPEX margin {date_str}: stat={data.get('stat')}（非交易日或尚未公布）")
        return None
    if str(data.get("date", "")).replace("-", "").replace("/", "") != date_compact:
        logger.warning(
            f"TPEX margin: 回傳日期 {data.get('date')} != 請求 {date_compact}，捨棄"
            f"（此端點會靜默回傳當天資料，見 docstring）"
        )
        return None

    tables = data.get("tables", [])
    target = next((t for t in tables
                   if isinstance(t.get("fields"), list) and len(t["fields"]) >= 20
                   and any("代號" in f for f in t["fields"])), None)
    if target is None or not target.get("data"):
        logger.warning(f"TPEX margin {date_str}: 找不到個股表")
        return None

    records = []
    for row in target["data"]:
        try:
            sid = str(row[0]).strip()
            if not sid.isdigit() or len(sid) != 4:
                continue
            records.append({
                "Date": date_str, "stock_id": sid,
                "Margin_Purchase": _margin_num(row[3]),    # 資買
                "Margin_Repay":    _margin_num(row[4]),    # 資賣
                "Margin_Balance":  _margin_num(row[6]),    # 資餘額
                "Short_Sale":      _margin_num(row[11]),   # 券賣  ← 注意與 TWSE 相反
                "Short_Cover":     _margin_num(row[12]),   # 券買
                "Short_Balance":   _margin_num(row[14]),   # 券餘額
                "_m_prev":  _margin_num(row[2]),  "_m_repay": _margin_num(row[5]),
                "_s_prev":  _margin_num(row[10]), "_s_repay": _margin_num(row[13]),
            })
        except (IndexError, ValueError):
            continue

    if not records:
        return None
    _margin_identity_check("TPEX", date_str, records)
    return pd.DataFrame(records)[["Date", "stock_id"] + MARGIN_COLS]


def fetch_margin_direct(date_str: str) -> Optional[pd.DataFrame]:
    """上市 + 上櫃融資融券合併（供 run_daily_update 與回補共用）。"""
    frames = [f for f in (fetch_margin_twse_direct(date_str),
                          fetch_margin_tpex_direct(date_str))
              if f is not None and not f.empty]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "stock_id"], keep="first")
    logger.info(f"margin 直連合併 {date_str}：{len(df)} 檔")
    return df


# ============================================================
# 現股當沖 — 交易所直連（2026-07-27 新增）
# ============================================================
# 為什麼需要：production 的 daytrade_raw.parquet 的 Day_Trade_Volume
# **2014–2026 全部 4,263,330 列都是 0**（非零列數 0）——整欄自始就沒有資料，
# 不是停更問題。FinMind 的 TaiwanStockDayTrading 回傳的 BuyAfterSale 欄位為空，
# 原本的管線也就一路寫 0 進去。
#
# ⚠️ 「列數 > N」不能當有效性判準：TWSE TWTB4U 對非交易日 2026-07-10 會回
#    1,209 列**骨架列**（只有代號、沒有數值欄），總計表則為空。必須檢查
#    「實際解析出數值的列數」，否則會把空殼當成有效資料寫進 parquet。

def fetch_daytrade_twse_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    上市現股當沖成交股數（TWSE /rwd/zh/dayTrading/TWTB4U，全市場單一天一次請求）。

    版面（2026-07-27 實測，個股表 6 欄）：
      0 證券代號 | 1 證券名稱 | 2 暫停現股賣出後現款買進當沖註記
      3 當日沖銷交易成交股數 | 4 當日沖銷交易買進成交金額 | 5 當日沖銷交易賣出成交金額
    單位：股。回溯測試可用到 2018-01-03（925 檔）。
    """
    return _fetch_daytrade_generic(
        date_str,
        url="https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U",
        params={"date": date_str.replace("-", ""), "selectType": "All",
                "response": "json"},
        src="TWSE",
    )


def fetch_daytrade_tpex_direct(date_str: str) -> Optional[pd.DataFrame]:
    """
    上櫃現股當沖成交股數（TPEX /www/zh-tw/intraday/stat，全市場單一天一次請求）。

    ⚠️ 日期格式必須是 YYYY/MM/DD——與 margin/balance 同一個坑：用 YYYYMMDD 會
       靜默回傳當天資料且 stat='ok'（2026-07-27 實測請求 20260724 回 date=20260727）。

    版面（2026-07-27 實測，tables[1] 個股表 6 欄，與 TWSE 同序）：
      0 證券代號 | 1 證券名稱 | 2 註記 | 3 當日沖銷交易成交股數
      4 買進成交金額 | 5 賣出成交金額
    """
    return _fetch_daytrade_generic(
        date_str,
        url="https://www.tpex.org.tw/www/zh-tw/intraday/stat",
        params={"date": date_str.replace("-", "/"), "type": "Daily",
                "response": "json"},
        src="TPEX",
    )


def _fetch_daytrade_generic(date_str: str, url: str, params: dict,
                            src: str) -> Optional[pd.DataFrame]:
    """TWSE / TPEX 的當沖個股表版面相同（代號在 0、當沖股數在 3），共用解析。"""
    date_compact = date_str.replace("-", "")
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=25)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"{src} daytrade fetch failed for {date_str}: {e}")
        return None

    if str(data.get("stat", "")).lower() not in ("ok", "success"):
        logger.info(f"{src} daytrade {date_str}: stat={data.get('stat')}")
        return None
    got = str(data.get("date", "")).replace("-", "").replace("/", "")
    if got != date_compact:
        logger.warning(f"{src} daytrade: 回傳日期 {got or None} != 請求 {date_compact}，捨棄")
        return None

    records = []
    for table in data.get("tables", []):
        rows = table.get("data") or []
        fields = table.get("fields") or []
        if len(fields) < 4 or not any("代號" in f for f in fields):
            continue
        for row in rows:
            if len(row) < 4:
                continue                        # 骨架列（非交易日）→ 直接跳過
            sid = str(row[0]).strip()
            if not sid.isdigit() or len(sid) != 4:
                continue
            try:
                shares = float(str(row[3]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
            records.append({"Date": date_str, "stock_id": sid,
                            "DayTrade_Shares": shares})

    if not records:
        logger.info(f"{src} daytrade {date_str}: 無有效數值列（非交易日或版面異動）")
        return None
    logger.info(f"{src} daytrade {date_str}: {len(records)} 檔")
    return pd.DataFrame(records)


def fetch_daytrade_direct(date_str: str) -> Optional[pd.DataFrame]:
    """上市 + 上櫃當沖股數合併。回傳 [Date, stock_id, DayTrade_Shares]（單位：股）。"""
    frames = [f for f in (fetch_daytrade_twse_direct(date_str),
                          fetch_daytrade_tpex_direct(date_str))
              if f is not None and not f.empty]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "stock_id"], keep="first")
    return df


def daytrade_shares_to_ratio(df_shares: pd.DataFrame) -> pd.DataFrame:
    """
    把當沖股數換成 production daytrade_raw 的 `Day_Trade_Volume`
    ＝ 當沖成交股數 / 該股當日成交股數（feature_engineer 會再 clip 到 [0,1]）。

    成交股數取自 prices_raw 的 Volume（yfinance / TWSE 皆為「股」，單位一致）。
    找不到對應成交量的列直接剔除，不用 0 補——0 會被模型當成「沒有當沖」的真值。
    """
    price_path = PROCESSED_DIR / "prices_raw.parquet"
    days = set(pd.to_datetime(df_shares["Date"]).dt.strftime("%Y-%m-%d"))
    pr = pd.read_parquet(price_path, columns=["Date", "stock_id", "Volume"])
    pr["Date"] = pd.to_datetime(pr["Date"])
    pr = pr[pr["Date"].dt.strftime("%Y-%m-%d").isin(days)]
    pr = pr.drop_duplicates(subset=["Date", "stock_id"], keep="last")

    out = df_shares.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.merge(pr, on=["Date", "stock_id"], how="inner")
    out = out[out["Volume"] > 0]
    out["Day_Trade_Volume"] = out["DayTrade_Shares"] / out["Volume"]
    return out[["Date", "stock_id", "Day_Trade_Volume"]]


# ============================================================
# 除權除息計算結果表 — 還原因子的權威來源（2026-07-28 新增）
# ============================================================

def fetch_ex_rights_twse_direct(start: str, end: str) -> Optional[pd.DataFrame]:
    """
    上市股票除權除息計算結果表（TWSE TWT49U）。**支援日期區間查詢**，
    一次請求可涵蓋整個月，是本專案少數不必逐日迴圈的端點。

    為什麼重要：這張表同時給「除權息前收盤價」與「除權息參考價」，
    兩者相除就是**交易所官方口徑的還原因子**：

        adj_factor = 除權息參考價 / 除權息前收盤價

    它一次涵蓋現金股利、股票股利與現金增資（參考價的計算已內含全部），
    比從 `dividend_raw` 的現金股利反推準確——後者處理不了無償配股與增資。

    ⚠️ 版面隨年份改變，**必須用 `fields` 陣列對映、不可硬編索引**：
      2008 起（15 欄）：… 3 除權息前收盤價 | 4 除權息參考價 | 5 權值+息值 | 6 權/息 …
      2005–2007（17 欄）：… 3 除權息前收盤價 | 4 除權息參考價 |
                          5 **權值** | 6 **息值** | 7 權值+息值 | 8 權/息 …
      兩者的索引 3/4 相同（所以 adj_factor 不受影響），但 5/6 語意完全不同——
      初版硬編索引導致 2005–2007 的 1,592 列把「權值」當成 value、「息值」當成 kind。

    註：`adj_factor > 1` 是合法的，代表**減資**（參考價高於前收盤價），
        因此本表同時是減資事件的來源（檢查表 C3）。

    ⚠️ TPEX 的對應公告 `/www/zh-tw/bulletin/exDailyQ` **忽略 date 參數**
       （傳 2026/07/09 與 2026/06/24 都回 date='20260728~20260729'，
       是「即將除權息」的前瞻公告而非歷史查詢），因此上櫃歷史除權息尚無來源。
    """
    url = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"
    try:
        resp = requests.get(url, params={"startDate": start.replace("-", ""),
                                         "endDate": end.replace("-", ""),
                                         "response": "json"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TWSE ex-rights fetch failed {start}~{end}: {e}")
        return None
    if str(data.get("stat", "")).upper() != "OK":
        return None

    rows = data.get("data") or []
    fields = data.get("fields") or []
    if not rows:
        for t in data.get("tables", []):
            if len(t.get("data") or []) > 0:
                rows, fields = t["data"], t.get("fields") or []
                break
    if not rows or not fields:
        return None

    def _col(*names) -> int:
        """依欄名找索引——版面 2005–2007 為 17 欄、2008 起 15 欄，硬編會錯位。"""
        for nm in names:
            for i, f in enumerate(fields):
                if nm in str(f):
                    return i
        return -1

    i_date, i_id = _col("資料日期"), _col("股票代號")
    i_before, i_ref = _col("除權息前收盤價"), _col("除權息參考價")
    i_val, i_kind = _col("權值+息值"), _col("權/息")
    if min(i_date, i_id, i_before, i_ref) < 0:
        logger.warning(f"TWSE ex-rights: 版面無法辨識，fields={fields}")
        return None

    def _num(s):
        try:
            return float(str(s).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    def _roc(s):
        """民國 '115年06月01日' → Timestamp。"""
        t = str(s).strip()
        try:
            y = int(t[:t.index("年")]) + 1911
            m = int(t[t.index("年") + 1:t.index("月")])
            d = int(t[t.index("月") + 1:t.index("日")])
            return pd.Timestamp(y, m, d)
        except Exception:                                     # noqa: BLE001
            return pd.NaT

    recs = []
    for r in rows:
        if len(r) <= max(i_date, i_id, i_before, i_ref):
            continue
        sid = str(r[i_id]).strip()
        if not sid.isdigit() or len(sid) != 4:
            continue
        dt, before, ref = _roc(r[i_date]), _num(r[i_before]), _num(r[i_ref])
        if pd.isna(dt) or not before or not ref or before <= 0:
            continue
        recs.append({"Date": dt, "stock_id": sid,
                     "close_before": before, "ref_price": ref,
                     "value": _num(r[i_val]) if 0 <= i_val < len(r) else None,
                     "kind": str(r[i_kind]).strip() if 0 <= i_kind < len(r) else "",
                     "adj_factor": ref / before})
    if not recs:
        return None
    return pd.DataFrame(recs)


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


def fetch_ex_rights_tpex_direct(start: str, end: str) -> Optional[pd.DataFrame]:
    """
    上櫃股票除權除息計算結果（TPEX `bulletin/exDailyQ`）。TWT49U 的上櫃對應版，
    欄位語意相同，因此兩市場可合併成同一張還原因子表。

    ⚠️ **踩過的雷（2026-07-28）**：這個端點乍看「忽略 date 參數、只回前瞻窗口」——
       傳 `date=2025/08/15` 會回 `date='20260728~20260729'`（即將除權息的公告），
       因此一度被判定為「上櫃歷史除權息無官方來源」，退而用 `dividend_raw` 自算公式。
       真正的原因是 **參數名要用 `startDate`/`endDate`，且必須是 `YYYY/MM/DD`**：
           startDate=20250801  → 靜默忽略，回前瞻窗口
           startDate=2025/08/01 → 正確回傳 20250801~20250831 共 150 筆
       這正是本專案雷區清單第 2 條（TPEX 只認斜線格式）的又一次重演。

    區間查詢實測可一次拉 5 年（2020–2024 回 4,976 筆），全歷史約 3–4 次請求。

    【歷史深度】2007 下半年起（2007 H1 = 0 筆、2007 H2 = 3 筆、2008 H2 = 374 筆），
       與 TPEX 價格端點的 2007-07 起點一致，不額外增加限制區段。

    日期格式與 TWSE 版**不同**：TPEX 是 `114/08/01`（民國年/月/日），
    TWSE TWT49U 是 `115年06月01日`。不可共用 parser。
    """
    url = "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ"
    try:
        resp = requests.get(url, params={"startDate": start, "endDate": end,
                                         "response": "json"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TPEX ex-rights fetch failed {start}~{end}: {e}")
        return None

    # 硬性核對回傳區間，避免又拿到前瞻窗口卻誤以為是歷史資料
    want = f"{start.replace('/', '')}~{end.replace('/', '')}"
    got = str(data.get("date", ""))
    if got != want:
        logger.warning(f"TPEX ex-rights: 回傳區間 {got!r} != 請求 {want!r}，丟棄")
        return None

    tables = data.get("tables") or []
    rows, fields = [], []
    for t in tables:
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

    i_date, i_id = _col("除權息日期"), _col("代號")
    i_before, i_ref = _col("除權息前收盤價"), _col("除權息參考價")
    i_val, i_kind = _col("權值+息值"), _col("權/息")
    if min(i_date, i_id, i_before, i_ref) < 0:
        logger.warning(f"TPEX ex-rights: 版面無法辨識，fields={fields}")
        return None

    def _num(s):
        try:
            return float(str(s).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    def _roc_slash(s):
        """民國 '114/08/01' → Timestamp。"""
        try:
            y, m, d = str(s).strip().split("/")
            return pd.Timestamp(int(y) + 1911, int(m), int(d))
        except Exception:                                     # noqa: BLE001
            return pd.NaT

    recs = []
    for r in rows:
        if len(r) <= max(i_date, i_id, i_before, i_ref):
            continue
        sid = str(r[i_id]).strip()
        if not sid.isdigit() or len(sid) != 4:
            continue
        dt = _roc_slash(r[i_date])
        before, ref = _num(r[i_before]), _num(r[i_ref])
        if pd.isna(dt) or not before or not ref or before <= 0:
            continue
        recs.append({"Date": dt, "stock_id": sid,
                     "close_before": before, "ref_price": ref,
                     "value": _num(r[i_val]) if 0 <= i_val < len(r) else None,
                     "kind": str(r[i_kind]).strip() if 0 <= i_kind < len(r) else "",
                     "adj_factor": ref / before})
    if not recs:
        return None
    return pd.DataFrame(recs)


def _cap_reduction_parse(rows, fields, roc_parser) -> Optional[pd.DataFrame]:
    """減資恢復買賣表的共用解析（TWSE / TPEX 欄名幾乎相同）。"""
    def _col(*names) -> int:
        for nm in names:
            for i, f in enumerate(fields):
                if nm in str(f):
                    return i
        return -1

    i_date = _col("恢復買賣日期")
    i_id = _col("股票代號")
    i_before = _col("停止買賣前收盤價格", "最後交易日之收盤價格")
    i_ref = _col("恢復買賣參考價", "減資恢復買賣開始日參考價格")
    i_why = _col("減資原因")
    if min(i_date, i_id, i_before, i_ref) < 0:
        logger.warning(f"減資表版面無法辨識，fields={fields}")
        return None

    def _num(s):
        try:
            return float(str(s).replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    recs = []
    for r in rows:
        if len(r) <= max(i_date, i_id, i_before, i_ref):
            continue
        sid = str(r[i_id]).strip()
        if not sid.isdigit() or len(sid) != 4:
            continue
        dt = roc_parser(r[i_date])
        before, ref = _num(r[i_before]), _num(r[i_ref])
        if pd.isna(dt) or not before or not ref or before <= 0:
            continue
        recs.append({"Date": dt, "stock_id": sid,
                     "close_before": before, "ref_price": ref,
                     "value": None,
                     "kind": "減資",
                     "why": str(r[i_why]).strip() if 0 <= i_why < len(r) else "",
                     "adj_factor": ref / before})
    return pd.DataFrame(recs) if recs else None


def fetch_capital_reduction_twse_direct(start: str, end: str) -> Optional[pd.DataFrame]:
    """
    上市減資恢復買賣（TWSE `reducation/TWTAUU`）。⚠️ 路徑拼字就是 `reducation`。

    【為什麼需要這張表——TWT49U 涵蓋不到】
      彌補虧損減資（減資換股）**不經過除權除息程序**，所以完全不在 TWT49U 裡。
      實測：全歷史 |單日報酬|>100% 的 273 筆中，203 筆是這類事件——
      典型指紋是「前收 1–3 元 → 停牌 12–23 天 → 復牌 10–40 元」，倍率集中在 ~10×
      （減資九成）。不處理的話這些會變成假的 +900% 單日報酬。

    還原因子與除權息同一個形式：

        adj_factor = 恢復買賣參考價 / 停止買賣前收盤價格

    減資時 > 1（復牌價高於停牌前），與既有「adj_factor > 1 代表減資」的慣例一致。
    事件日取**恢復買賣日**——那天的價格已是新基準，語意與除權息日相同。
    """
    try:
        resp = requests.get("https://www.twse.com.tw/rwd/zh/reducation/TWTAUU",
                            params={"startDate": start.replace("-", ""),
                                    "endDate": end.replace("-", ""),
                                    "response": "json"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TWSE capital reduction fetch failed {start}~{end}: {e}")
        return None
    if str(data.get("stat", "")).upper() != "OK":
        return None

    rows, fields = data.get("data") or [], data.get("fields") or []
    if not rows:
        for t in data.get("tables", []):
            if t.get("data"):
                rows, fields = t["data"], t.get("fields") or []
                break
    if not rows or not fields:
        return None

    def _roc(s):
        """民國 '107年01月02日' 或 '107/01/02' 皆可。"""
        t = str(s).strip()
        try:
            if "年" in t:
                y = int(t[:t.index("年")]) + 1911
                m = int(t[t.index("年") + 1:t.index("月")])
                d = int(t[t.index("月") + 1:t.index("日")])
            else:
                y, m, d = t.split("/")
                y, m, d = int(y) + 1911, int(m), int(d)
            return pd.Timestamp(y, m, d)
        except Exception:                                     # noqa: BLE001
            return pd.NaT

    return _cap_reduction_parse(rows, fields, _roc)


def fetch_capital_reduction_tpex_direct(start: str, end: str) -> Optional[pd.DataFrame]:
    """
    上櫃減資恢復買賣（TPEX `bulletin/revivt`）。參數需 `startDate`/`endDate` + 斜線格式。
    欄名與 TWSE 略有不同（「最後交易日之收盤價格」/「減資恢復買賣開始日參考價格」），
    解析器用欄名比對吸收差異。
    """
    try:
        resp = requests.get("https://www.tpex.org.tw/www/zh-tw/bulletin/revivt",
                            params={"startDate": start, "endDate": end,
                                    "response": "json"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TPEX capital reduction fetch failed {start}~{end}: {e}")
        return None

    want = f"{start.replace('/', '')}~{end.replace('/', '')}"
    if str(data.get("date", "")) != want:
        logger.warning(f"TPEX revivt: 回傳區間 {data.get('date')!r} != 請求 {want!r}")
        return None

    rows, fields = [], []
    for t in data.get("tables", []):
        if t.get("data"):
            rows, fields = t["data"], t.get("fields") or []
            break
    if not rows or not fields:
        return None

    def _roc_any(s):
        """
        TPEX 這張表的日期是 **7 碼緊湊民國格式** `'1070102'`，不是斜線格式。
        （TPEX 的日期格式已經咬過三次：兩次在請求端要斜線、這次在回應端不給斜線。）
        兩種都接受，避免下次版面又換。
        """
        t = str(s).strip().replace("年", "/").replace("月", "/").replace("日", "")
        try:
            if "/" in t:
                y, m, d = [x for x in t.split("/") if x]
            elif t.isdigit() and len(t) in (6, 7):
                y, m, d = t[:-4], t[-4:-2], t[-2:]
            else:
                return pd.NaT
            return pd.Timestamp(int(y) + 1911, int(m), int(d))
        except Exception:                                     # noqa: BLE001
            return pd.NaT

    return _cap_reduction_parse(rows, fields, _roc_any)


# ============================================================
# TAIFEX 三大法人（期貨 / 選擇權）直連
# ============================================================
#
# 為什麼改直連：FinMind 免費層對 `TaiwanFuturesInstitutionalInvestors` 與
# `TaiwanOptionsInstitutionalInvestors` 回 HTTP 400 "Your level is register"
# （2026-07-29 實測），也就是這兩個源在不訂閱的前提下**永遠拿不到**。
# TAIFEX 自己的下載端點是公開的，而且是**不同主機**——不與 TWSE/TPEX 的
# 速率限制互相排擠，也不受 FinMind 額度影響。

TAIFEX_FUT_URL = "https://www.taifex.com.tw/cht/3/futContractsDateDown"
TAIFEX_OPT_URL = "https://www.taifex.com.tw/cht/3/callsAndPutsDateDown"
TAIFEX_HEADERS = {**HEADERS, "Referer": "https://www.taifex.com.tw/"}

# 商品名稱 → 代碼。既有 parquet 用的是 FinMind 的代碼（TX / TXO），
# 對不到的一律保留原始中文名（不猜、不丟）。
TAIFEX_PRODUCT_CODE = {
    "臺股期貨": "TX", "台股期貨": "TX",
    "小型臺指期貨": "MTX", "小型台指期貨": "MTX",
    "電子期貨": "TE", "金融期貨": "TF",
    "臺指選擇權": "TXO", "台指選擇權": "TXO",
    "電子選擇權": "TEO", "金融選擇權": "TFO",
}

# 中文欄名 → production 欄名。用**欄名對映**而非索引：
# 雷區清單第 14 條（同一端點的版面會隨年份改變）在 TWT49U 上已經咬過一次。
# 2026-07-29 實測的真實欄名（value 為 production 欄名，list 為候選中文欄名）。
# ⚠️ 成交金額欄實際叫「多方**交易**契約金額(千元)」，中間有「交易」二字；
#    初版少寫了它、又因為子字串比對方向相反而對不上，會靜默留下整欄 NaN。
# 2026-07-29 實測的真實欄名。**期貨與選擇權用詞不同**（同一個交易所、同一組概念）：
#     期貨    多方交易口數 / 空方交易口數 / 多方未平倉口數 …
#     選擇權  買方交易口數 / 賣方交易口數 / 買方未平倉口數 …
# 初版只寫了期貨那組，選擇權整整 8 個數值欄全部落空（且靜默留 NaN）。
_TAIFEX_COLMAP = {
    "long_deal_volume":                   ["多方交易口數", "買方交易口數"],
    "long_deal_amount":                   ["多方交易契約金額(千元)", "買方交易契約金額(千元)"],
    "short_deal_volume":                  ["空方交易口數", "賣方交易口數"],
    "short_deal_amount":                  ["空方交易契約金額(千元)", "賣方交易契約金額(千元)"],
    "long_open_interest_balance_volume":  ["多方未平倉口數", "買方未平倉口數"],
    "long_open_interest_balance_amount":  ["多方未平倉契約金額(千元)", "買方未平倉契約金額(千元)"],
    "short_open_interest_balance_volume": ["空方未平倉口數", "賣方未平倉口數"],
    "short_open_interest_balance_amount": ["空方未平倉契約金額(千元)", "賣方未平倉契約金額(千元)"],
}

# TAIFEX 回 CALL/PUT，既有 parquet 是 買權/賣權。統一成既有值，
# 否則下游任何 groupby("call_put") 會把同一類拆成兩組。
_TAIFEX_CP = {"CALL": "買權", "PUT": "賣權", "call": "買權", "put": "賣權"}

# TAIFEX 回「外資及陸資」，既有 parquet 是「外資」。統一成既有值。
_TAIFEX_INST = {"外資及陸資": "外資"}

# 既有 parquet 只涵蓋 TX / TXO（FinMind 的子集），每個交易日固定 3 列（三類法人）。
# TAIFEX 一次回**所有商品**（單日 69 列）。若不過濾就併進去，
# `Futures_OI_Foreign`（外資淨未平倉，跨商品加總）的**加總範圍會在接續日無聲改變**——
# 特徵值突然跳一個量級，而且不會有任何錯誤訊息。
TAIFEX_KEEP_FUTURES = {"TX"}
TAIFEX_KEEP_OPTIONS = {"TXO"}


def _taifex_post(url: str, start: str, end: str) -> Optional[str]:
    """TAIFEX 下載端點一律走 POST + 表單。回傳 CSV 文字。"""
    data = {"firstDate": start, "lastDate": end,
            "queryStartDate": start, "queryEndDate": end,
            "commodityId": ""}
    try:
        resp = requests.post(url, data=data, headers=TAIFEX_HEADERS, timeout=45)
        resp.raise_for_status()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TAIFEX fetch failed {start}~{end}: {e}")
        return None
    # ⚠️ TAIFEX 回的是 **MS950（Big5）**，且 header 有正確宣告。
    #    用 `apparent_encoding` 猜測在中文資料上不可靠（會把 Big5 猜成 GB2312 之類），
    #    一律以 header 宣告為準、猜測只當退路。
    ct = resp.headers.get("content-type", "")
    enc = None
    if "charset=" in ct.lower():
        enc = ct.lower().split("charset=")[-1].strip()
    resp.encoding = enc or resp.apparent_encoding or "utf-8"
    text = resp.text
    # 非交易日 / 無資料時回的是 HTML 錯誤頁（實測 2026-07-10、602 bytes），不是空 CSV。
    if text.lstrip().lower().startswith(("<!doctype", "<html")):
        return None
    return text


def _taifex_parse(text: str, date_str: str, is_option: bool) -> Optional[pd.DataFrame]:
    """
    解析 TAIFEX 三大法人 CSV。

    刻意做三件事（全部來自本專案踩過的雷）：
      ① 用**欄名**建立對映，缺任何必要欄就明講並放棄，不硬編索引
      ② 硬性核對每一列的日期等於請求日期，不符者剔除
      ③ 回傳前印出實際欄名，讓第一次執行就能發現版面不如預期
    """
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TAIFEX CSV 解析失敗 {date_str}: {e}")
        return None
    if df.empty:
        return None
    df.columns = [str(c).strip() for c in df.columns]

    def _find(*cands) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        for c in cands:                                       # 退而求其次：包含即可
            for col in df.columns:
                if c in col:
                    return col
        return None

    c_date = _find("日期")
    c_prod = _find("商品名稱", "商品代號")
    c_inst = _find("身份別")
    c_cp = _find("買賣權別", "權別") if is_option else None
    if not all([c_date, c_prod, c_inst]) or (is_option and not c_cp):
        logger.warning(f"TAIFEX 版面無法辨識（{date_str}），實際欄名："
                       f"{list(df.columns)}")
        return None

    out = pd.DataFrame()
    d = pd.to_datetime(df[c_date], errors="coerce")
    want = pd.to_datetime(date_str)
    ok = d.notna()
    if date_str:                                              # 單日查詢才核對
        n_bad = int((ok & (d != want)).sum())
        if n_bad:
            logger.warning(f"TAIFEX {date_str}: {n_bad} 列日期不符，已剔除")
            ok &= (d == want)
    df, d = df[ok], d[ok]
    if df.empty:
        return None

    out["Date"] = d.values
    prod = df[c_prod].astype(str).str.strip()
    out["option_id" if is_option else "futures_id"] = prod.map(
        lambda x: TAIFEX_PRODUCT_CODE.get(x, x)).values
    if is_option:
        _cp = df[c_cp].astype(str).str.strip()
        out["call_put"] = _cp.map(lambda x: _TAIFEX_CP.get(x, x)).values
    _inst = df[c_inst].astype(str).str.strip()
    out["institutional_investors"] = _inst.map(
        lambda x: _TAIFEX_INST.get(x, x)).values

    n_mapped, missing = 0, []
    for en, cands in _TAIFEX_COLMAP.items():
        col = _find(*cands)
        if col is None:
            out[en] = pd.NA
            missing.append(en)
        else:
            out[en] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce").values
            n_mapped += 1
    if missing:
        logger.warning(f"TAIFEX {date_str}: 只對映到 {n_mapped}/"
                       f"{len(_TAIFEX_COLMAP)} 個數值欄，缺 {missing}"
                       f"｜實際欄名={list(df.columns)}")
    return out


def fetch_futures_institutional_direct(date_str: str) -> Optional[pd.DataFrame]:
    """期貨三大法人（TAIFEX，逐日）。供 `Futures_OI_Foreign` 特徵使用。"""
    t = _taifex_post(TAIFEX_FUT_URL, date_str.replace("-", "/"),
                     date_str.replace("-", "/"))
    return _taifex_parse(t, date_str, is_option=False) if t else None


def fetch_options_institutional_direct(date_str: str) -> Optional[pd.DataFrame]:
    """選擇權三大法人（TAIFEX，逐日）。供 `Options_PC_Ratio` 特徵使用。"""
    t = _taifex_post(TAIFEX_OPT_URL, date_str.replace("-", "/"),
                     date_str.replace("-", "/"))
    return _taifex_parse(t, date_str, is_option=True) if t else None


def fetch_holdings_tdcc_direct() -> Optional[pd.DataFrame]:
    """
    集保戶股權分散表（TDCC 開放資料 `getOD.ashx?id=1-5`）。

    【為什麼改直連】FinMind 的 `TaiwanStockHoldingSharesPer` 已需付費層
    （2026-07-29 實測回 400 "Your level is register"）。TDCC 開放資料免費、
    一次請求就是全市場（68,238 列 / 約 2,400 支 × 17 個分級）。

    ⚠️ **只有最新一週**：`getOD.ashx` 忽略任何日期參數
       （`date` / `DATE` / `d` 三種寫法實測都回同一週）。
       TDCC 的查詢頁（`smWeb/qryStock`）雖有 51 週歷史，但**逐股查詢**——
       2,000 支 × 11 週 = 22,000 次請求，不成比例。
       → 歷史缺口（2026-05-08 之後）補不回來，只能從現在起逐週累積。

    【修掉一個死常數 bug】舊的聚合把**分級 17 =「合計」**（恆為 100.00%）
       也加進總和，於是 `總計 ≈ 200`、`Whale = 200 − 散戶 ≈ 199.67`，
       被 `clip(0, 100)` 壓成 100 → **`Whale_Hold_Ratio` 在全部 848,269 列
       都是 100.0（只有 2 個相異值）**，是個死特徵。
       本函式改用分級 17 當總計、`Whale = 100 − Retail`。

       ✅ 對下游無影響且無接縫問題：`_merge_holdings` 早就繞過 `Whale_Hold_Ratio`、
          改用 `Holdings_Large_Pct = 1 − Retail/100`，而 `Retail` 一直是對的。
          修好的 Whale 恰等於該式 ×100，語意一致。

    【順帶記錄】分級 15（>1,000,000 股 ≈ >1000 張）才是真正的「大戶持股比例」
       （2330 為 84.70%），比「100 − 散戶」有意義得多。但歷史 parquet 只存了
       Whale/Retail 兩欄、沒有分級明細，無法回算，故先不新增該欄位。
    """
    from io import StringIO
    try:
        resp = requests.get("https://opendata.tdcc.com.tw/getOD.ashx",
                            params={"id": "1-5"}, headers=HEADERS, timeout=90)
        resp.raise_for_status()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TDCC 集保股權分散抓取失敗：{e}")
        return None
    resp.encoding = "utf-8-sig"
    try:
        df = pd.read_csv(StringIO(resp.text))
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"TDCC 集保股權分散解析失敗：{e}")
        return None
    df.columns = [str(c).strip() for c in df.columns]

    def _col(*names) -> str | None:
        for n in names:
            for c in df.columns:
                if n in c:
                    return c
        return None

    c_date, c_id = _col("資料日期"), _col("證券代號")
    c_lv, c_pct = _col("持股分級"), _col("占集保庫存數比例")
    if not all([c_date, c_id, c_lv, c_pct]):
        logger.warning(f"TDCC 版面無法辨識：{list(df.columns)}")
        return None

    df["_sid"] = df[c_id].astype(str).str.strip()
    df["_lv"] = pd.to_numeric(df[c_lv], errors="coerce")
    df["_pct"] = pd.to_numeric(df[c_pct], errors="coerce")
    df["_week"] = pd.to_datetime(df[c_date].astype(str).str.strip(),
                                 format="%Y%m%d", errors="coerce")
    df = df[df["_sid"].str.match(r"^\d{4}$") & df["_week"].notna()]
    if df.empty:
        return None

    LV_RETAIL, LV_TOTAL = 1, 17
    retail = (df[df["_lv"] == LV_RETAIL]
              .groupby(["_week", "_sid"])["_pct"].sum())
    total = (df[df["_lv"] == LV_TOTAL]
             .groupby(["_week", "_sid"])["_pct"].sum())
    out = pd.concat([total.rename("_total"), retail.rename("Retail_Hold_Ratio")],
                    axis=1).reset_index()
    out = out.rename(columns={"_week": "Week", "_sid": "stock_id"})
    out["Retail_Hold_Ratio"] = out["Retail_Hold_Ratio"].fillna(0.0)
    # 分級 17 是合計（實測 median 100.00）；若某支缺該列就退回 100
    out["_total"] = out["_total"].fillna(100.0)
    out["Whale_Hold_Ratio"] = (out["_total"] - out["Retail_Hold_Ratio"]).clip(0, 100)

    n_sat = int((out["Whale_Hold_Ratio"] >= 99.999).sum())
    logger.info(f"TDCC 集保股權分散：{out['Week'].iloc[0].date()}｜"
                f"{len(out):,} 支｜Retail median {out['Retail_Hold_Ratio'].median():.3f}%"
                f"｜Whale median {out['Whale_Hold_Ratio'].median():.3f}%"
                f"｜Whale 飽和於 100 的 {n_sat:,} 支"
                f"（舊聚合是 100%，此數應接近 0）")
    return out[["Week", "stock_id", "Whale_Hold_Ratio", "Retail_Hold_Ratio"]]


def _catch_up_holdings() -> int:
    """集保股權分散補齊（TDCC 直連，週頻）。只能取最新一週，見 fetcher docstring。"""
    path = PROCESSED_DIR / "holdings_raw.parquet"
    if not path.exists():
        return 0
    new = fetch_holdings_tdcc_direct()
    if new is None or new.empty:
        return 0
    old = pd.read_parquet(path)
    old["Week"] = pd.to_datetime(old["Week"])
    out = pd.concat([old, new[old.columns]], ignore_index=True)
    out = out.drop_duplicates(subset=["Week", "stock_id"], keep="last")
    out = out.sort_values(["stock_id", "Week"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    added = len(out) - len(old)
    logger.info(f"holdings_raw: 淨增 {added:,} 列｜{old['Week'].max().date()} → "
                f"{out['Week'].max().date()}")
    return added


def fetch_dividends_mops_direct() -> Optional[pd.DataFrame]:
    """
    股利分派情形（MOPS 開放資料 `t187ap45_L`／`_O`，含**上市與上櫃**）。

    【為什麼改直連】FinMind 的 `TaiwanStockDividend` 是**逐股查詢**，
    ~2,000 次請求撞爆免費層的 600 次/日額度；MOPS 是兩個 CSV、兩次請求。

    【`date` 欄的語意差異——必須明確記錄】
      既有 `dividend_raw`（FinMind）的 `date` 實測是**除息交易日 + 6 天**
      （median 6 天、p10/p90 = 6/8），也就是股利**發放後**才出現，
      而真正的公告日（`AnnouncementDate`）比它早約 22 天。

      MOPS 這張表沒有除息交易日，只有「董事會（擬議）股利分派日」＝真正的公告日。
      本函式以它填 `date`。**後果**：新資料讓股利比歷史資料早約 28 天被特徵「看到」。

      兩者都**不含未來資訊**（董事會決議當日即為公開資訊），差別只在時效——
      歷史段是偏保守的遲鈍，新資料段較貼近實際可得時點。
      `_merge_dividend_feature` 的 docstring 原本就寫「available once announced
      (before ex-date)」，新做法其實才符合原意。接縫處會有一次時效性的變化，
      這是刻意的取捨而非 bug。

    【涵蓋範圍】MOPS 只提供**當年度**的申報快照（上市 1,142 列、上櫃 913 列），
      不能回補歷史。歷史仍靠既有 parquet（FinMind 抓的 2005 起）。
    """
    from io import StringIO
    frames = []
    for sfx, market in (("L", "上市"), ("O", "上櫃")):
        url = f"https://mopsfin.twse.com.tw/opendata/t187ap45_{sfx}.csv"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45)
            resp.raise_for_status()
        except Exception as e:                                # noqa: BLE001
            logger.warning(f"MOPS 股利分派 {market} 抓取失敗：{e}")
            continue
        resp.encoding = "utf-8-sig"
        try:
            df = pd.read_csv(StringIO(resp.text))
        except Exception as e:                                # noqa: BLE001
            logger.warning(f"MOPS 股利分派 {market} 解析失敗：{e}")
            continue
        df.columns = [str(c).strip() for c in df.columns]

        def _col(*names) -> str | None:
            for n in names:
                if n in df.columns:
                    return n
            for n in names:
                for c in df.columns:
                    if n in c:
                        return c
            return None

        c_id = _col("公司代號")
        c_date = _col("董事會（擬議）股利分派日", "董事會")
        c_year = _col("股利年度")
        if not all([c_id, c_date]):
            logger.warning(f"MOPS 股利分派 {market} 版面無法辨識：{list(df.columns)}")
            continue

        def _num(col_name: str | None) -> pd.Series:
            if col_name is None:
                return pd.Series(0.0, index=df.index)
            return pd.to_numeric(
                df[col_name].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0.0)

        out = pd.DataFrame({
            "stock_id": df[c_id].astype(str).str.strip(),
            # 民國 YYYMMDD（如 1150311）→ 西元
            "date": pd.to_datetime(
                df[c_date].astype(str).str.strip().map(_roc_compact_to_ad),
                errors="coerce"),
            "year": (df[c_year].astype(str).str.strip() + "年"
                     if c_year else pd.NA),
            # ⚠️ 欄位語意不可照欄名直譯。MOPS 把股利**依來源拆成三欄**
            #    （盈餘分配／法定盈餘公積／資本公積），而既有 parquet（FinMind）
            #    是把**總額全放在 `*EarningsDistribution`**、`*StatutorySurplus` 恆為 0
            #    （實證：1101 台泥 113 年度 FinMind 記 1.00 全在第一欄）。
            #    若照欄名對映，像台泥這種「全部由資本公積配發」的公司
            #    會變成 0 元股利——而且不會有任何錯誤訊息。
            "CashEarningsDistribution": (
                _num(_col("盈餘分配之現金股利"))
                + _num(_col("法定盈餘公積發放之現金"))
                + _num(_col("資本公積發放之現金"))),
            "CashStatutorySurplus": 0.0,
            "StockEarningsDistribution": (
                _num(_col("盈餘轉增資配股"))
                + _num(_col("法定盈餘公積轉增資配股"))
                + _num(_col("資本公積轉增資配股"))),
            "StockStatutorySurplus": 0.0,
        })
        out = out[out["stock_id"].str.match(r"^\d{4}$") & out["date"].notna()]
        if not out.empty:
            frames.append(out)
            logger.info(f"MOPS 股利分派 {market}：{len(out):,} 筆"
                        f"（{out['date'].min().date()} → {out['date'].max().date()}）")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["stock_id", "date"], keep="last")


def _roc_compact_to_ad(s) -> str | None:
    """民國緊湊格式 '1150311' / '990311' → '2026-03-11'。"""
    t = str(s).strip()
    if not t.isdigit() or len(t) not in (6, 7):
        return None
    try:
        return f"{int(t[:-4]) + 1911:04d}-{t[-4:-2]}-{t[-2:]}"
    except Exception:                                         # noqa: BLE001
        return None


def fetch_shares_outstanding_mops() -> Optional[pd.DataFrame]:
    """
    上市/上櫃/興櫃公司基本資料（MOPS 公開資訊觀測站開放資料，含已發行普通股數）。
    不分日期，抓的是目前最新一份快照——股本變動不頻繁（現金增資/減資才會變），
    拿來乘上每日已有的收盤價估算市值，精度已足夠（現金增資/減資本身另有己知
    的極端報酬複驗項目在追蹤，見 planing/資料基礎升級計畫 2c）。
    """
    frames = []
    for suffix, market in [("L", "TSE"), ("O", "OTC")]:
        url = f"https://mopsfin.twse.com.tw/opendata/t187ap03_{suffix}.csv"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"MOPS shares outstanding fetch failed ({market}): {e}")
            continue
        try:
            df = pd.read_csv(pd.io.common.BytesIO(resp.content), encoding="utf-8")
        except Exception as e:
            logger.warning(f"MOPS CSV parse failed ({market}): {e}")
            continue
        col = next((c for c in df.columns if "已發行普通股數" in c), None)
        if col is None or "公司代號" not in df.columns:
            logger.warning(f"MOPS CSV missing expected columns ({market}): {list(df.columns)[:5]}")
            continue
        sub = df[["公司代號", col]].rename(
            columns={"公司代號": "stock_id", col: "shares_outstanding"})
        sub["stock_id"] = sub["stock_id"].astype(str).str.strip()
        sub = sub[sub["stock_id"].str.match(r"^\d{4}$")]
        sub["shares_outstanding"] = pd.to_numeric(sub["shares_outstanding"], errors="coerce")
        sub["market"] = market
        frames.append(sub)

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).dropna(subset=["shares_outstanding"])
    logger.info(f"MOPS shares outstanding: {len(df)} companies")
    return df


# ============================================================
# Layer 3: FinMind — Margin, Short, Banks, etc.
# ============================================================

def _finmind_fetch(
    dataset:     str,
    start_date:  str,
    end_date:    str,
    stock_id:    str | None = None,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Generic FinMind API call with exponential backoff retry. Returns None on failure."""
    if not FINMIND_TOKEN:
        logger.warning("FINMIND_TOKEN not set; skipping FinMind fetch")
        return None
    params = {
        "dataset":    dataset,
        "start_date": start_date,
        "end_date":   end_date,
        "token":      FINMIND_TOKEN,
    }
    if stock_id:
        # 2026-07-24 修正：FinMind API 實際參數名稱是 data_id，不是 stock_id
        # （對照 V6/scripts/fetch_v61_data.py 既有正確用法確認）。舊的 "stock_id"
        # 參數名稱不被 FinMind 辨識，導致單股查詢實質上一直被當成整批查詢送出，
        # 免費額度下整批查詢被擋、單股查詢改用 data_id 才會生效——這個 bug 原本
        # 就存在於 fetch_prices_finmind() 的 yfinance fallback 路徑，這次因為
        # margin/daytrade 回補大量使用單股查詢才被發現。
        params["data_id"] = stock_id

    for attempt in range(max_retries):
        try:
            resp = requests.get(FINMIND_BASE, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != 200:
                logger.debug(f"FinMind {dataset}: {data.get('msg')} (stock={stock_id})")
                return None
            return pd.DataFrame(data["data"])
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            # ⚠️ 402 = 當日額度用盡（免費層 600 次/日）。這**不是**「這支沒資料」，
            #    但舊版把它丟進下面的 debug 分支回 None，於是額度耗盡完全靜默：
            #    2026-07-29 的回補因此連跑六輪、試了 900 支、每輪淨增 0，
            #    而日誌看起來一切正常。額度問題必須大聲中止，不能偽裝成空資料。
            if code in (402, 403):
                raise FinMindQuotaExceeded(
                    f"FinMind 拒絕服務（HTTP {code}"
                    f"{'：當日額度用盡，免費層 600 次/日' if code == 402 else '：IP 被封鎖'}）："
                    f"{dataset} stock={stock_id}"
                ) from e
            if code in (429, 503) and attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"FinMind rate limit ({code}), retry {attempt+1}/{max_retries} in {wait}s")
                time.sleep(wait)
            else:
                logger.debug(f"FinMind {dataset} HTTP error: {e}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.debug(f"FinMind {dataset} attempt {attempt+1} failed ({e}), retry in {wait}s")
                time.sleep(wait)
            else:
                logger.debug(f"FinMind {dataset} chunk error: {e}")
                return None
    return None


def _finmind_fetch_chunked(
    dataset:    str,
    start_date: str,
    end_date:   str,
    stock_id:   str | None = None,
    chunk_years: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Fetch FinMind data in yearly chunks to respect free tier date limits.
    FinMind free tier rejects requests spanning > ~1825 days.
    Splits the [start_date, end_date] range into 'chunk_years'-year windows.
    """
    from datetime import datetime as _dt
    _start = _dt.strptime(start_date, "%Y-%m-%d")
    _end   = _dt.strptime(end_date,   "%Y-%m-%d")

    frames = []
    current = _start
    while current <= _end:
        chunk_end = min(
            _dt(current.year + chunk_years - 1, 12, 31),
            _end,
        )
        df = _finmind_fetch(
            dataset,
            start_date=current.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
            stock_id=stock_id,
        )
        if df is not None and not df.empty:
            frames.append(df)
        current = _dt(current.year + chunk_years, 1, 1)
        time.sleep(0.6)  # ~60 req/min free tier

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def fetch_margin_finmind(
    date_str: str,
    allow_forward_fill: bool = True,
) -> tuple[Optional[pd.DataFrame], bool]:
    """
    Fetch margin purchase / short sale data from FinMind.

    Returns:
        (df, is_forward_filled)
        df=None if data is unavailable; is_forward_filled=True when yesterday's
        cached data was substituted.
    """
    cache_path = CACHE_DIR / f"margin_{date_str}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path), False

    df = _finmind_fetch(
        "TaiwanStockMarginPurchaseShortSale",
        start_date=date_str,
        end_date=date_str,
    )

    if df is None or df.empty:
        if allow_forward_fill and MARGIN_FORWARD_FILL:
            yesterday = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday_cache = CACHE_DIR / f"margin_{yesterday}.parquet"
            if yesterday_cache.exists():
                logger.warning(f"Margin: FinMind not ready for {date_str}, using Forward Fill from {yesterday}")
                return pd.read_parquet(yesterday_cache), True
        return None, False

    df.to_parquet(cache_path)
    return df, False


def fetch_prices_finmind(
    stock_ids: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Per-stock fallback price fetch for rate-limited tickers.
    Splits into YEARLY chunks to respect FinMind free tier 1825-day limit.
    Skips any stock where FinMind returns an error (likely delisted).
    """
    from datetime import datetime as _dt

    def _yearly_fetch(sid: str, y_start: str, y_end: str) -> pd.DataFrame | None:
        df = _finmind_fetch("TaiwanStockPrice", start_date=y_start, end_date=y_end, stock_id=sid)
        return df if (df is not None and not df.empty) else None

    start_year = int(start[:4])
    end_year   = int(end[:4])

    frames = []
    skipped = 0
    for i, sid in enumerate(stock_ids):
        stock_frames = []
        for yr in range(start_year, end_year + 1):
            y_start = f"{yr}-01-01" if yr > start_year else start
            y_end   = f"{yr}-12-31" if yr < end_year   else end
            chunk = _yearly_fetch(sid, y_start, y_end)
            if chunk is not None:
                stock_frames.append(chunk)
            time.sleep(0.3)  # ~60 req/min free tier

        if stock_frames:
            df_stock = pd.concat(stock_frames, ignore_index=True)
            # Normalise column names to V6 standard
            if "date" in df_stock.columns:
                df_stock = df_stock.rename(columns={"date": "Date"})
            if "stock_id" not in df_stock.columns:
                df_stock["stock_id"] = sid
            # Map FinMind OHLCV column names
            col_map = {"open": "Open", "max": "High", "min": "Low",
                       "close": "Close", "Trading_Volume": "Volume"}
            df_stock.rename(columns={k: v for k, v in col_map.items()
                                     if k in df_stock.columns}, inplace=True)
            keep = [c for c in ["Date", "stock_id", "Open", "High", "Low", "Close", "Volume"]
                    if c in df_stock.columns]
            frames.append(df_stock[keep])
        else:
            skipped += 1

        if i % 10 == 9:
            time.sleep(1.0)  # extra pause every 10 stocks

    logger.info(f"FinMind price fallback: {len(frames)} fetched, {skipped} skipped (likely delisted)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================
# Orchestrator — Full Historical Sync
# ============================================================

def run_full_data_sync(
    start: str = DATA_START_DATE,
    end:   str | None = DATA_END_DATE,
    force_rebuild: bool = False,
) -> list[str]:
    """
    Main entry point. Pulls all data needed for V6 training.

    Returns:
        List of trading day strings 'YYYY-MM-DD' that were synced.

    Strategy:
        1. Price/Volume: yfinance batch (fast) + FinMind fallback
        2. Institutional: TWSE/TPEX direct + FinMind fallback
        3. Margin/Short: FinMind (or Forward Fill if not ready)
        4. Monthly Revenue + Fundamentals: FinMind (monthly, cached)
        5. Macro (VIX, SPX, Gold, Oil): yfinance
    """
    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"=== V6 Full Data Sync: {start} → {end} ===")

    tse_ids, otc_ids = load_ticker_universe()

    # --- Step 1: Price/Volume ---
    price_cache = PROCESSED_DIR / "prices_raw.parquet"
    if force_rebuild or not price_cache.exists():
        df_prices, missing_tickers = fetch_prices_yfinance(tse_ids, otc_ids, start, end)
        # Fallback for missing
        if missing_tickers:
        # Fallback for missing — MUST replace .TWO before .TW to avoid '6516O' artefact
            missing_ids = [t.replace(".TWO", "").replace(".TW", "") for t in missing_tickers]
            df_fallback = fetch_prices_finmind(missing_ids, start, end)
            if not df_fallback.empty:
                df_prices = pd.concat([df_prices, df_fallback], ignore_index=True)
        df_prices.to_parquet(price_cache)
        logger.info(f"Prices saved: {len(df_prices):,} rows")
    else:
        df_prices = pd.read_parquet(price_cache)
        logger.info(f"Prices loaded from cache: {len(df_prices):,} rows")

    # --- Step 2: Institutional Investors (chunked — 14 years > FinMind free tier limit) ---
    inst_cache = PROCESSED_DIR / "institutional_raw.parquet"
    if force_rebuild or not inst_cache.exists():
        logger.info("Fetching institutional data via FinMind (chunked yearly)...")
        df_inst = _finmind_fetch_chunked(
            "TaiwanStockInstitutionalInvestors",
            start_date=start,
            end_date=end,
        )
        if df_inst is not None and not df_inst.empty:
            col_map = {
                "date":                         "Date",
                "Foreign_Investor_Buy":          "Foreign_Buy",
                "Foreign_Investor_Sell":         "Foreign_Sell",
                "Foreign_Investor_Buy__Sell":    "Foreign_Net",
                "Investment_Trust_Buy":          "Investment_Trust_Buy",
                "Investment_Trust_Sell":         "Investment_Trust_Sell",
                "Investment_Trust_Buy__Sell":    "Investment_Trust_Net",
                "Dealer_proprietary_Buy__Sell":  "Dealer_Net",
            }
            df_inst.rename(columns={k: v for k, v in col_map.items() if k in df_inst.columns}, inplace=True)
            df_inst.to_parquet(inst_cache)
            logger.info(f"Institutional saved: {len(df_inst):,} rows")
        else:
            logger.warning("FinMind institutional data unavailable")
    else:
        logger.info(f"Institutional loaded from cache: {(PROCESSED_DIR / 'institutional_raw.parquet').stat().st_size // 1024:,} KB")

    # --- Step 3: Margin / Short Sale (chunked — 14 years > FinMind free tier limit) ---
    margin_cache = PROCESSED_DIR / "margin_raw.parquet"
    if force_rebuild or not margin_cache.exists():
        logger.info("Fetching margin data via FinMind (chunked yearly)...")
        df_margin = _finmind_fetch_chunked(
            "TaiwanStockMarginPurchaseShortSale",
            start_date=start,
            end_date=end,
        )
        if df_margin is not None and not df_margin.empty:
            if "date" in df_margin.columns:
                df_margin = df_margin.rename(columns={"date": "Date"})
            df_margin.to_parquet(margin_cache)
            logger.info(f"Margin saved: {len(df_margin):,} rows")
        else:
            logger.warning("Margin data unavailable from FinMind")
            df_margin = pd.DataFrame()
    else:
        df_margin = pd.read_parquet(margin_cache)
        logger.info(f"Margin loaded from cache: {len(df_margin):,} rows")

    # --- Step 4: Monthly Revenue & Fundamentals ---
    _sync_monthly_data(force_rebuild)

    # --- Step 5: Macro ---
    _sync_macro_data(start, end, force_rebuild)

    trading_days = _get_trading_days(df_prices)
    logger.info(f"=== Sync complete: {len(trading_days)} trading days ===")
    return trading_days


# ============================================================
# Daily Update — Inference Mode (Fast Path)
# ============================================================

def _catch_up_margin(today: str, max_days: int = 15) -> tuple[int, list[str]]:
    """
    把 margin_raw.parquet 補齊到 today（含），回傳 (寫入列數, 實際補到的日期清單)。

    為什麼要「補缺口」而不是只抓今天：交易所的融資融券餘額公布時間可能晚於推論
    觸發時刻（2026-07-27 實測 21:27 已有當日資料，19:30 是否已有未確認）。若每日
    更新只抓當天、抓不到就算了，那一天就永遠是洞——這正是 margin_raw 停更三個月
    沒被發現的模式。改成每次執行都把缺口補齊，對公布延遲自我修復。

    max_days 是保險絲：缺口大於此值時只補最近 max_days 天並告警，避免每日排程
    意外變成長時間任務（大缺口請用 V6/scripts/backfill_margin_direct.py）。
    """
    path = PROCESSED_DIR / "margin_raw.parquet"
    if path.exists():
        existing = set(
            pd.to_datetime(pd.read_parquet(path, columns=["Date"])["Date"])
            .dt.strftime("%Y-%m-%d")
        )
        start = pd.to_datetime(max(existing)) + pd.Timedelta(days=1)
    else:
        existing, start = set(), pd.to_datetime(today)

    days = [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, pd.to_datetime(today))
            if d.dayofweek < 5 and d.strftime("%Y-%m-%d") not in existing]
    if len(days) > max_days:
        logger.warning(
            f"Margin gap is {len(days)} weekdays — only filling the most recent "
            f"{max_days}. Use V6/scripts/backfill_margin_direct.py for the rest."
        )
        days = days[-max_days:]

    frames, done = [], []
    for d in days:
        df = fetch_margin_direct(d)
        if df is None or df.empty:
            continue          # 非交易日或尚未公布，下次執行會再試
        frames.append(df)
        done.append(d)
        if len(days) > 1:
            time.sleep(1.2)   # 對交易所端點的禮貌間隔
    if not frames:
        return 0, []

    # 一次性讀改寫，刻意不用 _append_to_parquet：
    #   ① 後者逐日各重寫一次 7.95M 列的 parquet（15 天缺口就重寫 15 次）
    #   ② 後者會把 Date 正規化成字串，與 margin_raw 既有的 timestamp[ns] 及
    #      backfill_margin_direct.py 寫入的型別不一致 —— 混型 Date 正是
    #      2026-07-27 修掉的 prices_raw 重複列根因，不要在這裡重新製造一次
    new = pd.concat(frames, ignore_index=True)
    new["Date"] = pd.to_datetime(new["Date"])
    if path.exists():
        old = pd.read_parquet(path)
        old["Date"] = pd.to_datetime(old["Date"])
        old = old[~old["Date"].isin(set(new["Date"]))]      # 同日資料以新抓的為準
        out = pd.concat([old, new[old.columns]], ignore_index=True)
    else:
        out = new
    out = out.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    out.sort_values(["stock_id", "Date"]).reset_index(drop=True).to_parquet(path, index=False)
    return len(new), done


_TRADING_DAY_CACHE: dict[str, bool] = {}


def is_trading_day(date_str: str) -> bool:
    """
    該日台股是否開盤。權威判定來自 TWSE MI_INDEX（非交易日回
    「很抱歉，沒有符合條件的資料!」），週末直接判 False 不打 API。

    為什麼需要：2026-06-07（週日）與 2026-06-19（端午）都曾被寫入整日假資料
    （前者 5,194 筆、後者 1,075 筆），因為寫入端完全沒有交易日曆概念。
    非交易日不存在合法資料，所以這是「拒絕寫入」而非「事後清理」的閘門。

    每次執行只會對同一天打一次（結果快取）。
    """
    if date_str in _TRADING_DAY_CACHE:
        return _TRADING_DAY_CACHE[date_str]
    if pd.Timestamp(date_str).dayofweek >= 5:
        _TRADING_DAY_CACHE[date_str] = False
        return False
    try:
        resp = requests.get(
            "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
            params={"response": "json", "date": date_str.replace("-", ""),
                    "type": "ALLBUT0999"},
            headers=HEADERS, timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        # 有效交易日：stat=OK 且個股表真的有數值列（非交易日會回骨架或空表）
        ok = str(data.get("stat", "")).upper() == "OK" and any(
            len(t.get("data") or []) > 100 for t in data.get("tables", [])
        )
    except Exception as e:                                    # noqa: BLE001
        # 判不出來時保守視為交易日：寧可讓下游的日期核對去擋，
        # 也不要因為網路問題就整天不更新。
        logger.warning(f"is_trading_day({date_str}) 判定失敗，保守視為交易日：{e}")
        ok = True
    _TRADING_DAY_CACHE[date_str] = ok
    logger.info(f"[交易日閘門] {date_str} → {'交易日' if ok else '非交易日，跳過所有寫入'}")
    return ok


def _check_universe_coverage(today: str) -> str:
    """
    當日檔數對前一交易日的變動率檢查。回傳空字串表示正常，否則回警告字串。

    刻意只告警不阻擋 prices 寫入：股票池變動可能是合法的（新上市/下市），
    擋掉會讓當天完全沒有訊號。但要讓它顯眼——2026-05-25 一日少 353 支
    （2,321 → 1,968）就是因為沒有這個檢查而過了兩個月沒被發現。
    """
    path = PROCESSED_DIR / "prices_raw.parquet"
    if not path.exists():
        return ""
    df = pd.read_parquet(path, columns=["Date", "stock_id"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    per_day = df.groupby("Date")["stock_id"].nunique().sort_index()
    if today not in per_day.index or len(per_day) < 2:
        return ""
    pos = per_day.index.get_loc(today)
    if pos == 0:
        return ""
    n_now, n_prev = int(per_day.iloc[pos]), int(per_day.iloc[pos - 1])
    delta = (n_now - n_prev) / max(n_prev, 1)
    if abs(delta) > 0.05:
        msg = (f"股票池單日變動 {delta:+.1%}（{per_day.index[pos-1]} {n_prev} 支 → "
               f"{today} {n_now} 支）")
        logger.warning(f"[覆蓋率閘門] ⚠️ {msg}")
        return msg
    logger.info(f"[覆蓋率閘門] {today} {n_now} 支，對前一交易日 {delta:+.2%} ✓")
    return ""


def _catch_up_daytrade(today: str, max_days: int = 15) -> tuple[int, list[str]]:
    """
    把 daytrade_raw.parquet 補齊到 today（含）。設計與 `_catch_up_margin` 相同：
    補缺口而非只抓今天，對交易所公布延遲自我修復。

    存在理由：`Day_Trade_Volume` 原本 2014–2026 全部 4,263,330 列都是 0
    （FinMind 的 BuyAfterSale 欄位為空，舊管線一路寫 0）。全歷史已用
    V6/scripts/backfill_daytrade_direct.py 重建；若不把它接進每日更新，
    重建完隔天就會再度停止更新。
    """
    path = PROCESSED_DIR / "daytrade_raw.parquet"
    if path.exists():
        existing = set(
            pd.to_datetime(pd.read_parquet(path, columns=["Date"])["Date"])
            .dt.strftime("%Y-%m-%d")
        )
        start = pd.to_datetime(max(existing)) + pd.Timedelta(days=1)
    else:
        existing, start = set(), pd.to_datetime(today)

    days = [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, pd.to_datetime(today))
            if d.dayofweek < 5 and d.strftime("%Y-%m-%d") not in existing]
    if len(days) > max_days:
        logger.warning(
            f"Daytrade gap is {len(days)} weekdays — only filling the most recent "
            f"{max_days}. Use V6/scripts/backfill_daytrade_direct.py for the rest."
        )
        days = days[-max_days:]

    frames, done = [], []
    for d in days:
        df = fetch_daytrade_direct(d)
        if df is None or df.empty:
            continue
        frames.append(df)
        done.append(d)
        if len(days) > 1:
            time.sleep(1.2)
    if not frames:
        return 0, []

    ratio = daytrade_shares_to_ratio(pd.concat(frames, ignore_index=True))
    if ratio.empty:
        return 0, []
    # 比率型特徵的理論上界：當沖股數不可能超過該股成交股數。超界代表分母有問題，
    # 剔除而非 clip——feature_engineer 的 .clip(0,1) 會把超界值變成看似合理的 1.0。
    n_over = int((ratio["Day_Trade_Volume"] > 1).sum())
    if n_over:
        logger.warning(f"Daytrade: 剔除比率 >1 的 {n_over} 列（分母 Volume 可疑）")
        ratio = ratio[ratio["Day_Trade_Volume"] <= 1]

    if path.exists():
        old = pd.read_parquet(path)
        old["Date"] = pd.to_datetime(old["Date"])
        old = old[~old["Date"].isin(set(ratio["Date"]))]
        out = pd.concat([old, ratio[old.columns]], ignore_index=True)
    else:
        out = ratio
    out = out.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    out.sort_values(["stock_id", "Date"]).reset_index(drop=True).to_parquet(path, index=False)
    return len(ratio), done


def _catch_up_generic(name: str, fetcher, today: str,
                      rename: dict | None = None,
                      max_days: int = 15,
                      date_col: str = "Date") -> tuple[int, list[str]]:
    """
    通用「補缺口」寫入器，供只需要單一逐日 fetcher 的資料源使用
    （per_raw / securities_raw / foreign_shareholding_raw ...）。
    語意與 `_catch_up_margin` 一致：每次執行都把 parquet 補齊到 today，
    對交易所公布延遲自我修復。

    rename：fetcher 的輸出欄名 → production parquet 欄名
            （例：per 的 dividend_yield → DY；securities 的
              Securities_Balance → Securities_Lending）。
            schema 不一致而硬併會產生新欄位並讓下游 ffill 失效。

    date_col：`foreign_shareholding_raw` 的日期欄是小寫 `date` 而非 `Date`。
              沿用既有 schema、不改動歷史檔，故在此以參數吸收差異。
    """
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        return 0, []
    prod = pd.read_parquet(path)
    prod[date_col] = pd.to_datetime(prod[date_col])
    existing = set(prod[date_col].dt.strftime("%Y-%m-%d"))
    start = pd.to_datetime(max(existing)) + pd.Timedelta(days=1)

    days = [d.strftime("%Y-%m-%d")
            for d in pd.date_range(start, pd.to_datetime(today))
            if d.dayofweek < 5 and d.strftime("%Y-%m-%d") not in existing]
    if len(days) > max_days:
        logger.warning(f"{name} gap is {len(days)} weekdays — only filling the most "
                       f"recent {max_days}.")
        days = days[-max_days:]
    if not days:
        return 0, []

    frames, done = [], []
    for d in days:
        df = fetcher(d)
        if df is None or df.empty:
            continue
        if rename:
            df = df.rename(columns=rename)
        for c in prod.columns:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[prod.columns])
        done.append(d)
        if len(days) > 1:
            time.sleep(1.2)
    if not frames:
        return 0, []

    new = pd.concat(frames, ignore_index=True)
    new[date_col] = pd.to_datetime(new[date_col])
    out = pd.concat([prod, new], ignore_index=True)
    out = out.drop_duplicates(subset=[date_col, "stock_id"], keep="last")
    (out.sort_values(["stock_id", date_col]).reset_index(drop=True)
        .to_parquet(path, index=False))
    return len(new), done


def _catch_up_taifex(today: str, max_months: int = 6) -> tuple[int, int]:
    """
    期貨/選擇權三大法人的補缺口（TAIFEX 直連，2026-07-29 納入每日更新）。

    【為什麼直連】FinMind 免費層對這兩個 dataset 回 400 "Your level is register"，
    不訂閱就永遠拿不到。TAIFEX 端點公開、且是**不同主機**——不與 TWSE/TPEX
    搶速率，也不受 FinMind 額度影響。

    【區間查詢】TAIFEX 支援 `queryStartDate`/`queryEndDate`，一次一個月
    （實測 2026/05 回 1,242 列 / 18 個交易日），比逐日省 20 倍請求。

    【只保留 TX / TXO】既有 parquet 是 FinMind 的子集（每日 3 列），
    而 TAIFEX 一次回所有商品（單日 69 列）。不過濾的話
    `Futures_OI_Foreign`（跨商品加總）會在接續日無聲跳一個量級。

    【歷史深度】TAIFEX 這兩個端點約只有 2.5 年（2023 下半年起），
    更早的歷史仍靠既有 parquet（FinMind 抓的 2018-06 起），兩者在此銜接。
    """
    specs = [
        ("futures_institutional_raw", TAIFEX_FUT_URL, False,
         ["Date", "futures_id", "institutional_investors"], TAIFEX_KEEP_FUTURES,
         "futures_id"),
        ("options_institutional_raw", TAIFEX_OPT_URL, True,
         ["Date", "option_id", "call_put", "institutional_investors"],
         TAIFEX_KEEP_OPTIONS, "option_id"),
    ]
    totals = []
    for name, url, is_opt, key, keep, idcol in specs:
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            totals.append(0)
            continue
        old = pd.read_parquet(path)
        old["Date"] = pd.to_datetime(old["Date"])
        last = old["Date"].max()
        start = last + pd.Timedelta(days=1)
        end = pd.to_datetime(today)
        if start > end:
            logger.info(f"{name}: 已最新（{last.date()}）")
            totals.append(0)
            continue

        frames = []
        cur, n_month = start, 0
        while cur <= end and n_month < max_months:
            chunk_end = min(cur + pd.offsets.MonthEnd(0), end)
            t = _taifex_post(url, cur.strftime("%Y/%m/%d"),
                             chunk_end.strftime("%Y/%m/%d"))
            d = _taifex_parse(t, "", is_option=is_opt) if t else None
            if d is not None and not d.empty:
                d = d[d[idcol].isin(keep)]
                if not d.empty:
                    frames.append(d)
            cur = chunk_end + pd.Timedelta(days=1)
            n_month += 1
            time.sleep(1.5)

        if not frames:
            logger.warning(f"{name}: {last.date()} 之後無新資料")
            totals.append(0)
            continue
        new = pd.concat(frames, ignore_index=True)
        for c in old.columns:
            if c not in new.columns:
                new[c] = pd.NA
        out = pd.concat([old, new[old.columns]], ignore_index=True)
        out = out.drop_duplicates(subset=key, keep="last")
        out = out.sort_values(key).reset_index(drop=True)
        out.to_parquet(path, index=False)
        added = len(out) - len(old)
        logger.info(f"{name}: 抓到 {len(new):,} 列 → 淨增 {added:,} 列"
                    f"｜{last.date()} → {out['Date'].max().date()}")
        totals.append(added)
    return tuple(totals)                                      # type: ignore[return-value]


def _catch_up_dividends() -> int:
    """
    股利分派補齊（MOPS 直連，2026-07-29 納入每日更新）。

    MOPS 提供的是**當年度申報快照**，所以做法是每天把快照併進來、
    以 (stock_id, date) 去重——新的董事會決議會自然出現，已存在的不會重複。
    不能回補歷史（歷史仍靠既有 parquet，FinMind 抓的 2005 起）。

    ⚠️ 新列的除權息日期等欄位為 NA（MOPS 這張表沒有）。
       目前無下游依賴：`_merge_dividend_feature` 只用 `date` 與現金股利欄，
       而 B-3 的除權息因子已改用交易所官方表（`ex_rights_raw`），
       不再需要從 `dividend_raw` 反推。
    """
    path = PROCESSED_DIR / "dividend_raw.parquet"
    if not path.exists():
        return 0
    new = fetch_dividends_mops_direct()
    if new is None or new.empty:
        logger.warning("dividend_raw: MOPS 無資料")
        return 0
    old = pd.read_parquet(path)
    old["date"] = pd.to_datetime(old["date"], errors="coerce")
    for c in old.columns:
        if c not in new.columns:
            new[c] = pd.NA
    out = pd.concat([old, new[old.columns]], ignore_index=True)
    out = out.drop_duplicates(subset=["stock_id", "date"], keep="last")
    out = out.sort_values(["stock_id", "date"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    added = len(out) - len(old)
    logger.info(f"dividend_raw: MOPS {len(new):,} 筆 → 淨增 {added:,} 列"
                f"｜最新 {out['date'].max().date()}")
    return added


# ══════════════════════════════════════════════════════════════════════════════
# MOPS 財報三表 + 月營收 整批直連
# ══════════════════════════════════════════════════════════════════════════════
#
# 【為什麼需要】FinMind 免費層對財報只能**逐股查詢**（~1,900 次/季），
#   實測滾動補齊速率約 32 支/天，補完 2,300 支要 70 天，而每季持續進來
#   → 追不上。2026-08-04 實測覆蓋率：
#       financials_raw   2025-12-31 = 2,475 支 → 2026-03-31 = 176 支（7%）
#       balance_sheet_raw / cashflow_raw       → 2026-03-31 =  16 支（0.7%）
#       revenue_raw      2026-04 = 1,958 支 → 2026-05 起 = 871 支（44%）
#   MOPS 的整批端點是**一季兩次請求**（上市 + 上櫃）拿到 1,929 家。
#
# 【host 必須是 mopsov】`mops.twse.com.tw` 只回 686 bytes，`mopsov` 才是資料主機。
#
# ── 四個對映陷阱（都是「照欄名直譯會靜默算錯」，實測後才定案）────────────────
#
#  ① `本期淨利（淨損）` → `IncomeAfterTaxes`，**不是 `NetIncome`**。
#     FinMind 用**全形/半形括號**區分兩個 type：
#         `本期淨利（淨損）`（全形）→ IncomeAfterTaxes  ← 2024 起 17,963 筆
#         `本期淨利(淨損)`  （半形）→ NetIncome         ← 2024 起      0 筆
#     MOPS 給的是全形。對到 NetIncome 會讓 ROE 分子落在近年無資料的 type 上
#     （`_merge_fundamentals` 的優先序是 IncomeAfterTaxes 97.7% > NetIncome 45.4%）。
#
#  ② 毛利有兩欄，取 `營業毛利（毛損）`、**不是 `營業毛利（毛損）淨額`**。
#     兩者都對到 GrossProfit，但 2024 起 FinMind 實際使用
#     `營業毛利（毛損）` 17,385 筆 vs `淨額` 18 筆。「淨額」聽起來像最終值，
#     選錯會與 99.9% 的歷史不同義。
#
#  ③ 單位：MOPS 是**千元**、FinMind 是**元** → 需 ×1000。
#     但 EPS 與「股數」欄是原始單位，**不可乘**（見 `_MOPS_NO_SCALE`）。
#     驗證：台泥 2026-05 月營收 MOPS 12,612,013 千元 ×1000
#          = FinMind 12,612,013,000 逐位元相同。
#
#  ④ MOPS 第 2/3/4 季是**累計數**。但 FinMind 三張表的慣例**並不一致**——
#     這是本輪最不直觀、也只有實測才看得出來的一項：
#
#         financials_raw    → **單季**   ∴ 需逐季相減
#         cashflow_raw      → **累計**   ∴ 維持原值、不可相減
#         balance_sheet_raw → 時點值     ∴ 不可相減
#
#     證據（2026-08-04，V1 驗證）：
#       · 台積電 2025 四季 EPS 13.95/15.36/17.44/19.51，加總 ≈ 66 才是全年 → 單季
#       · 台積電 2025 四季營業活動現金流 6,256/11,226/15,495/22,750 億
#         **單調遞增**，且 `期初現金餘額` 四季完全相同（21,276 億＝年初）→ 累計
#     第一版把現金流量表也拿去相減，V1 量到 median 比值 Q2≈0.53、Q3≈0.32
#     （正好是 1/2、1/3）才抓到。**若照直覺「三張表都是單季」寫下去，
#     現金流特徵會整批只剩實際值的三分之一，而且不會有任何錯誤訊息。**
#
# ── 對映表怎麼來的（不是猜的）──────────────────────────────────────────────
#   FinMind 的 `origin_name` 欄保留了中文科目名，與 MOPS 的欄名幾乎逐字相同。
#   對映是從既有 parquet 的 (type, origin_name) 配對反推、再用近年使用次數決定
#   歧義項。語意不唯一或 FinMind 沒有對應 type 的欄位一律**不寫入**
#   （列在 `_MOPS_SKIPPED_*`），寧可少一欄，不要寫進一個名字對、值錯的欄。
# ══════════════════════════════════════════════════════════════════════════════

MOPS_HOST = "https://mopsov.twse.com.tw"

_MOPS_STATEMENT_ENDPOINT = {
    "financials":    "ajax_t163sb04",     # 綜合損益表
    "balance_sheet": "ajax_t163sb05",     # 資產負債表
    "cashflow":      "ajax_t163sb20",     # 現金流量表
}

# 需要「累計 → 單季」相減的表。
# ⚠️ 只有 financials。cashflow 在 FinMind 端本來就是累計值（見陷阱 ④），
#    相減會讓它變成實際值的 1/2~1/3；balance_sheet 是時點值。
_MOPS_FLOW_KINDS = {"financials"}

# ── 對映表：`MOPS 欄名 → (FinMind type, 優先序)` ──────────────────────────────
#
# 【怎麼建的】絕大多數項目是**程式從既有 parquet 的 (origin_name → type) 反推**，
#   不是手寫。優先序 = FinMind 2023 年起使用該 origin_name 的次數。
#
# 【優先序拿來做什麼】MOPS 同一張表裡有多個欄位會對到同一個 type，例如
#   `營業毛利（毛損）`(25,698) 與 `營業毛利（毛損）淨額`(55) 都是 GrossProfit。
#   與其手動挑一個（挑錯就是陷阱 ②），不如兩個都收、由優先序決定勝出者——
#   判準因此與 FinMind 自己的取捨一致，而不是與我的猜測一致。
#
# 【⚠️ MOPS 有兩種版面：一般業 vs 金融保險業】這是與 TAIFEX 那次
#   「期貨用多方/空方、選擇權用買方/賣方」完全同型的陷阱。
#   金融業用的是 `權益總額`／`資產總額`／`歸屬於母公司業主權益合計`，
#   只寫一般業的欄名會讓**整個金融保險業拿不到 `Book_Value`**（而且不會報錯）。
#   同一家公司只會出現其中一種版面，故兩種都收不會互相干擾。
#
# 【標 (語意) 者】= FinMind 的 origin_name 字面不同但語意確定
#   （MOPS「流動資產」= FinMind「流動資產合計」）。這些是人工判斷，已逐項標註。
_MOPS_TYPE_MAP: dict[str, dict[str, tuple[str, int]]] = {
    "financials": {
        # ── 一般業版面 ──
        "營業收入": ("Revenue", 25989),
        "營業成本": ("CostOfGoodsSold", 25940),
        "營業毛利（毛損）": ("GrossProfit", 25698),
        "營業毛利（毛損）淨額": ("GrossProfit", 55),          # 優先序低，讓上面那個勝出
        "營業費用": ("OperatingExpenses", 26649),
        "營業利益（損失）": ("OperatingIncome", 26046),
        "營業外收入及支出": ("TotalNonoperatingIncomeAndExpense", 26046),
        "其他收益及費損淨額": ("OTHNOE", 1910),
        "已實現銷貨（損）益": ("RealizedGain", 1001),
        "未實現銷貨（損）益": ("UnrealizedGain", 1220),
        "稅前淨利（淨損）": ("PreTaxIncome", 26314),
        "所得稅費用（利益）": ("TAX", 26600),
        "繼續營業單位本期淨利（淨損）": ("IncomeFromContinuingOperations", 26533),
        "繼續營業單位稅前淨利（淨損）": ("IncomeBeforeTaxFromContinuingOperations", 480),
        "停業單位損益": ("IncomeLossFromDiscontinuedOperation", 778),
        "本期淨利（淨損）": ("IncomeAfterTaxes", 26600),        # ← 陷阱 ①：不是 NetIncome
        "本期綜合損益總額": ("TotalConsolidatedProfitForThePeriod", 26771),
        "其他綜合損益（淨額）": ("OtherComprehensiveIncome", 23079),
        "基本每股盈餘（元）": ("EPS", 2477),
        # FinMind 把「淨利歸屬母公司業主」標成 EquityAttributableToOwnersOfParent
        # （英文名其實錯了，那是損益不是權益）。為與歷史相容必須沿用同一個 type；
        # 真正的股東權益在 balance_sheet（`_balance_sheet_equity` 已處理）。
        "淨利（淨損）歸屬於母公司業主": ("EquityAttributableToOwnersOfParent", 22917),
        "綜合損益總額歸屬於母公司業主": ("EquityAttributableToOwnersOfParent", 97),
        "淨利（淨損）歸屬於非控制權益": ("NoncontrollingInterests", 14355),
        "綜合損益總額歸屬於非控制權益":
            ("ComprehensiveIncomeConsolidatedNetIncomeAttributedNonControllingInterest", 14689),
        # 生物資產（農林漁牧業專用）
        "原始認列生物資產及農產品之利益（損失）":
            ("GainsOnInitialRecognitionOfBiologicalAssetsForCurrentPeriod", 76),
        "生物資產當期公允價值減出售成本之變動利益（損失）":
            ("GainsOnChangesInFairValueLessCosts2SellOfBiologicalAssetsForCurrentPeriod", 97),
        # ── 金融保險業版面 ──
        "淨收益": ("Revenue", 171),
        "收入": ("Revenue", 48),
        "支出": ("CostOfGoodsSold", 48),
        "收益": ("Income", 506),
        "支出及費用": ("Expense", 506),
        "利息淨收益": ("NetInterestIncome", 603),
        "利息以外淨收益": ("NetNonInterestIncome", 171),
        "利息以外淨損益": ("NetNonInterestIncome", 432),
        "呆帳費用、承諾及保證責任準備提存": ("BadDebts", 600),
        "營業利益": ("OperatingIncome", 506),
        "營業外損益": ("TotalNonbusinessIncome", 506),
        "所得稅（費用）利益": ("TAX", 171),
        "繼續營業單位稅前損益": ("PreTaxIncome", 171),
        "繼續營業單位稅前純益（純損）": ("PreTaxIncome", 238),
        "繼續營業單位本期稅後淨利（淨損）": ("IncomeFromContinuingOperations", 432),
        "繼續營業單位本期純益（純損）": ("IncomeFromContinuingOperations", 238),
        "本期稅後淨利（淨損）": ("IncomeAfterTax", 603),
        "本期綜合損益總額（稅後）": ("OtherComprehensiveIncomeAfterTaxThePeriod", 432),
        "本期其他綜合損益（稅後淨額）": ("OtherComprehensiveIncomeAfterTaxThePeriod", 673),
        "其他綜合損益（稅後）": ("OtherComprehensiveIncome", 432),
        "其他綜合損益（稅後淨額）": ("OtherComprehensiveIncomeAfterTax", 238),
        "其他綜合損益": ("OtherComprehensiveIncome", 48),
        "基本每股盈餘": ("EPS", 24592),
        "淨利（損）歸屬於母公司業主": ("EquityAttributableToOwnersOfParent", 596),
        "淨利（損）歸屬於非控制權益": ("NoncontrollingInterests", 336),
    },
    "balance_sheet": {
        # ── 一般業版面（FinMind origin 多帶「合計」二字 → 語意對映）──
        "流動資產": ("CurrentAssets", 22000),                  # (語意)「流動資產合計」
        "非流動資產": ("NoncurrentAssets", 22000),             # (語意)「非流動資產合計」
        "資產總計": ("TotalAssets", 621),
        "流動負債": ("CurrentLiabilities", 22000),             # (語意)「流動負債合計」
        "非流動負債": ("NoncurrentLiabilities", 22000),        # (語意)「非流動負債合計」
        "負債總計": ("Liabilities", 621),                      # (語意)「負債總額」
        "股本": ("CapitalStock", 22000),                       # (語意)「股本合計」
        "資本公積": ("CapitalSurplus", 22000),                 # (語意)「資本公積合計」
        "保留盈餘": ("RetainedEarnings", 22000),               # (語意)「保留盈餘合計」
        "其他權益": ("OtherEquityInterest", 22000),            # (語意)「其他權益合計」
        "歸屬於母公司業主之權益合計": ("EquityAttributableToOwnersOfParent", 22564),  # ★
        "非控制權益": ("NoncontrollingInterests", 14864),
        "權益總計": ("Equity", 649),                                                 # ★
        "預收股款（權益項下）之約當發行股數（單位：股）":
            ("EquivalentIssueSharesOfAdvanceReceiptsForOrdinaryShare", 100),
        "母公司暨子公司所持有之母公司庫藏股股數（單位：股）":
            ("NumberOfSharesInEntityHeldByEntityAndByItsSubsidiaries", 100),
        # ── 金融保險業版面 ──（漏掉會讓整個金融業沒有 Book_Value）
        "資產總額": ("TotalAssets", 24854),
        "負債總額": ("Liabilities", 24862),
        "權益總額": ("Equity", 24826),                                               # ★
        "歸屬於母公司業主權益合計": ("EquityAttributableToOwnersOfParent", 20000),    # (語意) ★
        "歸屬於母公司業主之權益": ("EquityAttributableToOwnersOfParent", 20000),      # (語意) ★
        "保留盈餘（或累積虧損）": ("RetainedEarnings", 20000),                        # (語意)
        "母公司暨子公司持有之母公司庫藏股股數（單位：股）":
            ("NumberOfSharesInEntityHeldByEntityAndByItsSubsidiaries", 100),
        # ── 兩種版面共有 ──
        "現金及約當現金": ("CashAndCashEquivalents", 25475),
        "無形資產": ("IntangibleAssets", 22358),
        "使用權資產": ("RightOfUseAsset", 24658),
        "應付公司債": ("BondsPayable", 5296),
        "本期所得稅資產": ("CurrentIncomeTaxAssets", 14676),
        "本期所得稅負債": ("CurrentTaxLiabilities", 22284),
        "遞延所得稅資產": ("DeferredTaxAssets", 22984),
    },
    "cashflow": {
        "營業活動之淨現金流入（流出）": ("CashFlowsFromOperatingActivities", 25394),  # ★ FCF
        "投資活動之淨現金流入（流出）": ("CashProvidedByInvestingActivities", 25380),
        "籌資活動之淨現金流入（流出）": ("CashFlowsProvidedFromFinancingActivities", 25350),
        "本期現金及約當現金增加（減少）數": ("CashBalancesIncrease", 25401),
        "期初現金及約當現金餘額": ("CashBalancesBeginningOfPeriod", 25401),
        "期末現金及約當現金餘額": ("CashBalancesEndOfPeriod", 25401),
    },
}

# 刻意不寫入的欄位。記錄理由是為了讓「為什麼少這幾欄」有據可查，
# 而不是看起來像漏掉——也讓版面新增欄位時的警告不會被這些常態項淹沒。
_MOPS_SKIPPED = {
    "financials": {
        "合併前非屬共同控制股權損益": "FinMind 該 origin_name 的 type 是『-』（無英文代碼）",
        "合併前非屬共同控制股權綜合損益淨額": "同上",
        "淨利（損）歸屬於共同控制下前手權益": "同上",
        "淨利（淨損）歸屬於共同控制下前手權益": "同上",
        "綜合損益總額歸屬於共同控制下前手權益": "同上",
        "保險服務結果": "FinMind 無此 origin_name（IFRS 17 新科目）",
        "保險其他營業成本": "同上",
        "財務結果": "同上",
        "其他營業結果": "同上",
    },
    "balance_sheet": {
        # 金融保險業的細項科目：FinMind 值域沒有，且下游未消費
        "權益─具證券性質之虛擬通貨": "FinMind 無對應 type",
        "權益－具證券性質之虛擬通貨": "FinMind 無對應 type（全形連字號版本）",
        "庫藏股票": "FinMind 無對應 type",
        "共同控制下前手權益": "FinMind 無對應 type",
        "合併前非屬共同控制股權": "FinMind 無對應 type",
        "待註銷股本股數（單位：股）": "FinMind 無對應 type",
        "每股參考淨值": "FinMind 無對應 type（下游用 Equity/股數自算）",
        "Unnamed:12": "read_html 產生的空欄",
    },
    "cashflow": {
        "匯率變動對現金及約當現金之影響": "FinMind 無對應 type",
    },
}

def _mops_needs_scaling(col: str) -> bool:
    """該欄是否需要「千元 → 元」的 ×1000。

    ⚠️ 用**樣式**判斷而不是硬編欄名清單：金融保險業版面的欄名有變體
    （`基本每股盈餘` vs `基本每股盈餘（元）`、`母公司暨子公司持有…` vs
    `…所持有…`），硬編清單漏掉任何一個，那一欄就會整組差 1000 倍。
    """
    if "單位：股" in col or "單位:股" in col:      # 股數欄，本來就是「股」
        return False
    if "每股" in col:                              # EPS、每股參考淨值：本來就是「元」
        return False
    return True


def _mops_norm_col(c) -> str:
    """把 read_html 的欄名正規化：攤平 MultiIndex、去空白、去重複層級。

    read_html 對 MOPS 會回 `'公司 代號'`（中間有空白）與
    `('營業收入', '當月營收')` 這種兩層欄名。
    """
    if isinstance(c, tuple):
        parts = [re.sub(r"\s+", "", str(x)) for x in c
                 if not str(x).startswith("Unnamed")]
        ded = []
        for p in parts:
            if not ded or ded[-1] != p:
                ded.append(p)
        return "|".join(ded)
    return re.sub(r"\s+", "", str(c))


def _mops_to_number(v) -> float:
    """MOPS 數值解析：`--`／空白／`-` 一律視為缺值。

    ⚠️ 不可用 `pd.to_numeric(errors='coerce')` 一把梭就算了——MOPS 用 `--`
    表示「本科目不適用」，若被靜默轉成 0 會讓「沒有這個科目」變成「這個科目是 0」，
    在毛利率／ROE 這類比率上是完全不同的意思。
    """
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v) if pd.notna(v) else float("nan")
    s = str(v).strip().replace(",", "")
    if s in ("", "--", "-", "—", "nan", "None"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _mops_season_end(year: int, season: int) -> pd.Timestamp:
    """季別 → 財報基準日（與 FinMind `financials_raw.Date` 的慣例一致）。"""
    return pd.Timestamp({1: f"{year}-03-31", 2: f"{year}-06-30",
                         3: f"{year}-09-30", 4: f"{year}-12-31"}[season])


def _mops_pick_company_tables(tables: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """從 read_html 回傳的一堆表裡挑出「公司資料表」。

    ⚠️ 不可用「列數最多」或固定索引來挑：
      · 財報三表的主表確實只有一張，但月營收是**依產業拆成數十張表**
        （實測上市 32 張、合計 1,023 家），只取最大的一張會漏掉九成。
      · 版面表（頁首說明、統計摘要）也可能有不少列。
    判準改為「欄位含公司代號」，對兩種版面都成立、且對未來改版較穩健。
    """
    out = []
    for t in tables:
        cols = [_mops_norm_col(c) for c in t.columns]
        if any(c == "公司代號" or c.endswith("|公司代號") for c in cols):
            t = t.copy()
            t.columns = cols
            out.append(t)
    return out


def _mops_post(endpoint: str, payload: dict, what: str,
               retries: int = 3) -> Optional[str]:
    """對 MOPS 發 POST 並回傳 HTML 文字（失敗回 None、不拋例外）。

    ⚠️ **必須重試**：實測 MOPS 會偶發回 `502 Bad Gateway`。
    這類暫時性失敗若不重試，上層會拿到「這一季這個市場沒資料」，
    而那與「真的沒有」無法區分——本專案已經在 TPEX 2022–2024 整段落空上
    踩過一次同型的坑（`data-source-implementation-traps.md`）。
    """
    url = f"{MOPS_HOST}/mops/web/{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, data=payload, headers=HEADERS, timeout=90)
            resp.raise_for_status()
        except Exception as e:                                # noqa: BLE001
            if attempt < retries:
                logger.warning(f"MOPS {what} 第 {attempt}/{retries} 次失敗：{e}"
                               f"｜{attempt * 5} 秒後重試")
                time.sleep(attempt * 5)
                continue
            logger.warning(f"MOPS {what} 連續 {retries} 次抓取失敗：{e}")
            return None
        resp.encoding = resp.apparent_encoding or "utf-8"
        # 非資料頁（查無資料／季別未公布）會回很短的內容
        if len(resp.content) < 5000:
            logger.info(f"MOPS {what}：內容僅 {len(resp.content)} bytes，視為尚未公布")
            return None
        return resp.text
    return None


def fetch_mops_statement_wide(kind: str, year: int, season: int,
                              typek: str) -> Optional[pd.DataFrame]:
    """抓單一市場、單一季別的財報寬表（原始欄名、原始單位、**累計數**）。

    `typek`: `"sii"` 上市 / `"otc"` 上櫃。
    """
    if kind not in _MOPS_STATEMENT_ENDPOINT:
        raise ValueError(f"未知的 kind：{kind}")
    what = f"{kind}/{typek}/{year}Q{season}"
    html = _mops_post(
        _MOPS_STATEMENT_ENDPOINT[kind],
        {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1,
         "TYPEK": typek, "year": str(year - 1911), "season": f"{season:02d}"},
        what,
    )
    if html is None:
        return None
    try:
        tables = pd.read_html(StringIO(html))
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"MOPS {what} 解析失敗：{e}")
        return None
    comp = _mops_pick_company_tables(tables)
    if not comp:
        logger.warning(f"MOPS {what}：找不到含『公司代號』的表")
        return None
    df = pd.concat(comp, ignore_index=True) if len(comp) > 1 else comp[0]
    df = df[df["公司代號"].notna()].copy()
    df["公司代號"] = df["公司代號"].astype(str).str.strip()
    # 表頭列有時會被 read_html 當成資料列吃進來
    df = df[df["公司代號"].str.fullmatch(r"\d{4}")]
    return df.reset_index(drop=True)


def _mops_wide_to_long(wide: pd.DataFrame, kind: str,
                       date: pd.Timestamp) -> pd.DataFrame:
    """MOPS 寬表 → FinMind 長格式 `[Date, stock_id, type, value, origin_name]`。

    同時完成單位換算（千元 → 元，見 `_mops_needs_scaling`）與**同 type 去重**。

    去重是必要的：MOPS 一張表裡多個欄位會對到同一個 FinMind type
    （例：`營業毛利（毛損）` 與 `營業毛利（毛損）淨額` 都是 GrossProfit），
    不處理會讓 `[Date, stock_id, type]` 出現重複鍵，寫進 parquet 後
    下游的 `groupby(...).last()` 拿到哪一個要看排序 → 不可重現。
    """
    tmap = _MOPS_TYPE_MAP[kind]
    rows, unmapped = [], []
    for col in wide.columns:
        if col in ("公司代號", "公司名稱"):
            continue
        hit = tmap.get(col)
        if hit is None:
            if col not in _MOPS_SKIPPED.get(kind, {}):
                unmapped.append(col)
            continue
        t, prio = hit
        vals = wide[col].map(_mops_to_number)
        if _mops_needs_scaling(col):
            vals = vals * 1000.0                              # 千元 → 元
        sub = pd.DataFrame({
            "Date": date,
            "stock_id": wide["公司代號"].values,
            "type": t,
            "value": vals.values,
            "origin_name": col,
            "_prio": prio,
        })
        rows.append(sub[sub["value"].notna()])
    if unmapped:
        # 版面新增欄位時要看得見，不能靜默忽略——TAIFEX 那次就是靜默留 NaN
        logger.warning(f"MOPS {kind}：{len(unmapped)} 個欄位未在對映表中，已略過："
                       f"{unmapped}")
    cols = ["Date", "stock_id", "type", "value", "origin_name"]
    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.concat(rows, ignore_index=True)
    key = ["Date", "stock_id", "type"]
    n_dup = int(out.duplicated(subset=key).sum())
    if n_dup:
        # 依優先序（＝FinMind 自己近年的使用次數）留下勝出者，並印出實際發生的
        # 對映衝突——這是「同一個 type 有多個來源欄」時唯一能事後查核的線索
        winners = (out.sort_values("_prio", ascending=False)
                   .drop_duplicates(subset=key, keep="first"))
        losers = (out.merge(winners[key + ["origin_name"]], on=key,
                            suffixes=("", "_win"))
                  .query("origin_name != origin_name_win"))
        pairs = sorted({(r.origin_name, r.origin_name_win)
                        for r in losers.itertuples()})
        logger.info(f"MOPS {kind}：{n_dup:,} 列同 type 重複，依優先序保留；"
                    f"落選→勝出：{pairs[:6]}{' …' if len(pairs) > 6 else ''}")
        out = winners
    return out.drop(columns=["_prio"]).reset_index(drop=True)


def fetch_financial_statement_mops_direct(kind: str, year: int,
                                          season: int) -> Optional[pd.DataFrame]:
    """財報三表整批直連，回傳 **FinMind 長格式、單季值**。

    上市 + 上櫃各一次請求；流量表（損益／現金流）會自動用
    「本季累計 − 上季累計」還原成單季值（見陷阱 ④）。
    """
    def _both(sea: int) -> Optional[pd.DataFrame]:
        """抓上市 + 上櫃並合併。**兩個市場缺任一就整季放棄。**

        ⚠️ 不可「有幾個算幾個」：實測 MOPS 對 otc 回過一次 502，
        當時上層拿到只有上市的 1,046 家、看起來完全正常，
        接著被當成「上一季」拿去相減 → 上櫃 883 家全部匹配不到 →
        走 fallback 保留累計數 → **54.2% 的資料是錯的且不會報錯**
        （54.2% 正好 = 1046/1929）。缺一個市場一定是失敗、不是常態。
        """
        parts = []
        for typek in ("sii", "otc"):
            w = fetch_mops_statement_wide(kind, year, sea, typek)
            if w is None or w.empty:
                logger.warning(f"MOPS {kind} {year}Q{sea}：{typek} 無資料 → "
                               f"整季放棄（寧可沒有，不要只有一半）")
                return None
            parts.append(w)
            time.sleep(1.0)
        wide = pd.concat(parts, ignore_index=True)
        wide = wide.drop_duplicates(subset=["公司代號"], keep="first")
        return _mops_wide_to_long(wide, kind, _mops_season_end(year, sea))

    cur = _both(season)
    if cur is None or cur.empty:
        return None

    # 資產負債表是時點值、Q1 本來就是單季 → 直接回
    if kind not in _MOPS_FLOW_KINDS or season == 1:
        logger.info(f"MOPS {kind} {year}Q{season}：{cur['stock_id'].nunique():,} 支"
                    f"／{len(cur):,} 列（單季值，無需相減）")
        return cur

    prev = _both(season - 1)
    if prev is None or prev.empty:
        logger.warning(f"MOPS {kind} {year}Q{season}：抓不到上一季累計數，"
                       f"無法還原單季值 → 放棄本季（不寫入累計數，避免與歷史不同義）")
        return None

    key = ["stock_id", "type"]
    m = cur.merge(prev[key + ["value"]], on=key, how="left",
                  suffixes=("", "_prev"))

    # ⚠️ 上季無對應的列一律**丟棄**，絕不「保留累計數」。
    #    第一版就是保留原值，結果 MOPS 一次 502 讓上櫃整批匹配不到 →
    #    13,340 列悄悄變成累計數混進單季資料裡，V1 量到「median 仍是 1.000000
    #    但只有 54% 落在容差內」才抓到。**缺值看得見，錯值看不見。**
    n_unmatched = int(m["value_prev"].isna().sum())
    frac = n_unmatched / max(len(m), 1)
    if frac > 0.05:
        # 少量無對應是新上市造成的常態；大量無對應一定是抓取失敗。
        # 這種情況連「丟棄」都不夠——整季都不可信。
        logger.warning(f"MOPS {kind} {year}Q{season}：{n_unmatched:,}/{len(m):,} 列"
                       f"（{frac:.1%}）在上一季無對應，遠高於新上市能解釋的比例"
                       f" → 判定為上一季抓取不完整，整季放棄")
        return None
    if n_unmatched:
        logger.info(f"MOPS {kind} {year}Q{season}：{n_unmatched:,} 列"
                    f"（{frac:.2%}）在上一季無對應（新上市／科目變動），已丟棄")
    m["value"] = m["value"] - m["value_prev"]                 # NaN 會傳染 → 自然丟棄
    out = m.drop(columns=["value_prev"])
    out = out[out["value"].notna()].reset_index(drop=True)
    logger.info(f"MOPS {kind} {year}Q{season}：{out['stock_id'].nunique():,} 支"
                f"／{len(out):,} 列（已由累計數還原為單季值）")
    return out


def fetch_revenue_mops_direct(year: int, month: int) -> Optional[pd.DataFrame]:
    """月營收整批直連，回傳 FinMind `revenue_raw` schema。

    端點 `nas/t21/{sii,otc}/t21sc03_{民國年}_{月}_0.html`，編碼 **big5**。

    ⚠️ 兩個必須照做的細節：
      · 表格**依產業拆成數十張**，要全部 concat（只取最大的一張會漏掉九成）。
      · FinMind 的 `Date` 是**營收月份的下個月 1 日**
        （`revenue_month=5, revenue_year=2026` → `Date=2026-06-01`），
        不是營收月份本身。對錯會讓整批資料錯開一個月且不會報錯。
    """
    frames = []
    for typek in ("sii", "otc"):
        url = f"{MOPS_HOST}/nas/t21/{typek}/t21sc03_{year - 1911}_{month}_0.html"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=90)
            resp.raise_for_status()
        except Exception as e:                                # noqa: BLE001
            logger.warning(f"MOPS 月營收 {typek} {year}-{month:02d} 抓取失敗：{e}")
            continue
        resp.encoding = "big5"
        try:
            tables = pd.read_html(StringIO(resp.text))
        except Exception as e:                                # noqa: BLE001
            logger.warning(f"MOPS 月營收 {typek} {year}-{month:02d} 解析失敗：{e}")
            continue
        comp = _mops_pick_company_tables(tables)
        if not comp:
            logger.warning(f"MOPS 月營收 {typek} {year}-{month:02d}：無公司資料表")
            continue
        frames.extend(comp)
        time.sleep(1.0)

    if not frames:
        return None
    wide = pd.concat(frames, ignore_index=True)
    col_rev = next((c for c in wide.columns if c.endswith("|當月營收")), None)
    if col_rev is None:
        logger.warning(f"MOPS 月營收 {year}-{month:02d}：找不到『當月營收』欄"
                       f"（欄位：{list(wide.columns)[:8]}）")
        return None
    wide["公司代號"] = wide["公司代號"].astype(str).str.strip()
    wide = wide[wide["公司代號"].str.fullmatch(r"\d{4}")]
    wide = wide.drop_duplicates(subset=["公司代號"], keep="first")

    rev = wide[col_rev].map(_mops_to_number) * 1000.0         # 千元 → 元
    out = pd.DataFrame({
        "Date": pd.Timestamp(year, month, 1) + pd.offsets.MonthBegin(1),
        "stock_id": wide["公司代號"].values,
        "country": "Taiwan",
        "revenue": rev.values,
        "revenue_month": month,
        "revenue_year": year,
        "create_time": "",
    })
    out = out[out["revenue"].notna()].reset_index(drop=True)
    out["revenue"] = out["revenue"].round().astype("int64")
    logger.info(f"MOPS 月營收 {year}-{month:02d}：{len(out):,} 支"
                f"｜Date={out['Date'].iloc[0].date() if len(out) else '—'}")
    return out


def _mops_quarter_due(year: int, season: int) -> pd.Timestamp:
    """財報的法定申報期限（用來判斷「這一季現在該有資料了嗎」）。

    台股規定：Q1 5/15、Q2 8/14、Q3 11/14、Q4（年報）隔年 3/31。
    多留 3 天緩衝，避免在期限當天就開始每天空打。
    """
    due = {1: (year, 5, 15), 2: (year, 8, 14),
           3: (year, 11, 14), 4: (year + 1, 3, 31)}[season]
    return pd.Timestamp(*due) + pd.Timedelta(days=3)


def _catch_up_mops_quarterly(today: str, max_quarters: int = 6,
                             min_coverage: float = 0.80) -> dict[str, int]:
    """財報三表的季頻補齊（MOPS 整批直連）。

    【與 `_catch_up_monthly` 的分工】那支是 FinMind **逐股**滾動，實測約 32 支/天、
    補完 2,300 支要 70 天，**追不上每季新進來的量**（2026-08-04 稽核：
    `financials` 2026Q1 只有 7%、`balance_sheet`/`cashflow` 只有 0.7%）。
    本函式改用 MOPS 整批端點，**一季兩次請求拿到 1,929 家**。

    【策略】只補「法定申報期限已過、但覆蓋率不足」的季別，且**每次執行最多補一季**——
    財報是季頻資料，沒有必要在同一天把好幾季都抓完，分散開來也不會給 MOPS 壓力。

    【MOPS 優先】重疊鍵以 MOPS 為準（`keep="last"`）。理由不只是 FinMind 有標錯的
    英文 type 名稱，更根本的是 **MOPS 是原始申報來源、FinMind 是轉手方**。
    實測重疊區 median 比值為 1.000000，覆寫風險低。
    ⚠️ 但**只覆寫本次抓取的那一季**，不回頭重寫歷史——那是另一個決策。
    """
    added: dict[str, int] = {}
    t = pd.Timestamp(today)
    live = _live_universe()
    n_live = len(live) if live else 2000

    # 候選季別：由新到舊，只看申報期限已過的
    cands: list[tuple[int, int]] = []
    y, s = t.year, (t.month - 1) // 3 + 1
    for _ in range(max_quarters + 4):
        s -= 1
        if s == 0:
            y, s = y - 1, 4
        if _mops_quarter_due(y, s) <= t:
            cands.append((y, s))
        if len(cands) >= max_quarters:
            break

    for kind in ("financials", "balance_sheet", "cashflow"):
        path = PROCESSED_DIR / f"{kind}_raw.parquet"
        if not path.exists():
            continue
        old = pd.read_parquet(path)
        old["Date"] = pd.to_datetime(old["Date"], errors="coerce")
        old["stock_id"] = old["stock_id"].astype(str)
        have = old.groupby("Date")["stock_id"].nunique()

        target = None
        for (yy, ss) in cands:                                # 由新到舊，補最新的缺口
            d = _mops_season_end(yy, ss)
            cov = int(have.get(d, 0)) / max(n_live, 1)
            if cov < min_coverage:
                target = (yy, ss, d, cov)
                break
        if target is None:
            logger.info(f"{kind}_raw: 近 {len(cands)} 季覆蓋率皆 ≥{min_coverage:.0%}，無需補齊")
            added[kind] = 0
            continue

        yy, ss, d, cov = target
        logger.info(f"{kind}_raw: {d.date()}（{yy}Q{ss}）覆蓋 {cov:.1%} "
                    f"< {min_coverage:.0%} → 由 MOPS 補齊")
        try:
            new = fetch_financial_statement_mops_direct(kind, yy, ss)
        except Exception as e:                                # noqa: BLE001
            logger.warning(f"{kind}_raw: MOPS 抓取例外（不影響推論）：{e}")
            added[kind] = 0
            continue
        if new is None or new.empty:
            logger.warning(f"{kind}_raw: MOPS {yy}Q{ss} 無可用資料")
            added[kind] = 0
            continue

        for c in old.columns:
            if c not in new.columns:
                new[c] = pd.NA
        key = ["Date", "stock_id", "type"]
        # MOPS 放後面 + keep="last" ＝ 重疊鍵以 MOPS 為準
        out = pd.concat([old, new[old.columns]], ignore_index=True)
        out = out.drop_duplicates(subset=key, keep="last")
        out = out.sort_values(key).reset_index(drop=True)
        out.to_parquet(path, index=False)
        n_add = len(out) - len(old)
        added[kind] = n_add
        logger.info(f"{kind}_raw: MOPS {len(new):,} 列 → 淨增 {n_add:,} 列"
                    f"｜{d.date()} 覆蓋 {int(have.get(d, 0)):,} → "
                    f"{out[out['Date'] == d]['stock_id'].nunique():,} 支")
    return added


def _catch_up_mops_revenue(today: str, max_months: int = 6,
                           min_coverage: float = 0.80) -> int:
    """月營收補齊（MOPS 整批直連）。

    月營收的公告期限是**次月 10 日**。同樣只補「期限已過但覆蓋率不足」的月份，
    每次執行最多補一個月。

    ⚠️ FinMind `revenue_raw` 的 `Date` 是**營收月份的下個月 1 日**
    （`revenue_month=5, revenue_year=2026` → `Date=2026-06-01`），
    `fetch_revenue_mops_direct` 已照此慣例產出，此處不再轉換。
    """
    path = PROCESSED_DIR / "revenue_raw.parquet"
    if not path.exists():
        return 0
    t = pd.Timestamp(today)
    old = pd.read_parquet(path)
    old["Date"] = pd.to_datetime(old["Date"], errors="coerce")
    old["stock_id"] = old["stock_id"].astype(str)
    have = old.groupby("Date")["stock_id"].nunique()
    live = _live_universe()
    n_live = len(live) if live else 2000

    target = None
    cur = pd.Timestamp(t.year, t.month, 1)
    for _ in range(max_months):
        cur = cur - pd.offsets.MonthBegin(1)                   # 營收月份
        due = cur + pd.offsets.MonthBegin(1) + pd.Timedelta(days=12)
        if due > t:
            continue
        d = cur + pd.offsets.MonthBegin(1)                     # parquet 的 Date 慣例
        cov = int(have.get(d, 0)) / max(n_live, 1)
        if cov < min_coverage:
            target = (cur.year, cur.month, d, cov)
            break
    if target is None:
        logger.info(f"revenue_raw: 近 {max_months} 個月覆蓋率皆 ≥{min_coverage:.0%}，無需補齊")
        return 0

    yy, mm, d, cov = target
    logger.info(f"revenue_raw: {yy}-{mm:02d} 營收（Date={d.date()}）"
                f"覆蓋 {cov:.1%} < {min_coverage:.0%} → 由 MOPS 補齊")
    try:
        new = fetch_revenue_mops_direct(yy, mm)
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"revenue_raw: MOPS 抓取例外（不影響推論）：{e}")
        return 0
    if new is None or new.empty:
        logger.warning(f"revenue_raw: MOPS {yy}-{mm:02d} 無可用資料")
        return 0

    for c in old.columns:
        if c not in new.columns:
            new[c] = pd.NA
    key = ["Date", "stock_id"]
    out = pd.concat([old, new[old.columns]], ignore_index=True)
    out = out.drop_duplicates(subset=key, keep="last")         # MOPS 優先
    out = out.sort_values(key).reset_index(drop=True)
    out.to_parquet(path, index=False)
    n_add = len(out) - len(old)
    logger.info(f"revenue_raw: MOPS {len(new):,} 列 → 淨增 {n_add:,} 列"
                f"｜{d.date()} 覆蓋 {int(have.get(d, 0)):,} → "
                f"{out[out['Date'] == d]['stock_id'].nunique():,} 支")
    return n_add


# ══════════════════════════════════════════════════════════════════════════════
# 處置股 / 注意股（交易狀態）
# ══════════════════════════════════════════════════════════════════════════════
#
# 【為什麼在這裡】原本只有 `V6/experimental/fetch_trading_status.py`，而那支的
#   `build()` 是**整檔重建**（逐年抓 11 年後 `to_parquet` 全檔覆寫），
#   不能排進每日流程。解析邏輯（含下面四個實測踩過的坑）搬到這裡當**單一來源**，
#   experimental 那支改為 import，避免兩份實作日後分歧。
#
# 【四項裡只有「處置」有真正的交易限制】分盤集合競價（每 5 或 20 分鐘撮合一次）
#   → 流動性大幅下降、價差擴大；預收款券 → 很難照收盤價成交。
#   「注意」只是警示、無交易限制，但常是處置前兆，一併抓來當診斷。
#
# 【端點與四個坑（2026-08-01 實測）】
#   TWSE 處置 `/rwd/zh/announcement/punish`｜注意 `/rwd/zh/announcement/notice`
#   TPEX 處置 `/www/zh-tw/bulletin/disposal`｜注意 `/www/zh-tw/bulletin/attention`
#   ① 四個都要 `startDate`/`endDate`（`date` 參數回 0 列），
#      且 TWSE 是 `YYYYMMDD`、TPEX 是 `YYYY/MM/DD`
#   ② 起迄分隔符 TWSE 用**全形** `～`(U+FF5E)、TPEX 用**半形** `~`
#   ③ 民國日期分隔符 punish 用 `/`、notice 用 `.`
#   ④ TPEX 無資料時回**骨架列**（「本日無處置資料」、代號與起訖皆空字串）
#   另：TPEX 的 disposal 與 attention **欄序不同**（代號分別在 r[2] / r[1]）
_TS_SEP = re.compile(r"[～~﹏－-]+")
_TS_STOCK_RE = re.compile(r"^\d{4}$")
_TS_ENDPOINTS = {
    ("twse", "disposal"):  "https://www.twse.com.tw/rwd/zh/announcement/punish",
    ("twse", "attention"): "https://www.twse.com.tw/rwd/zh/announcement/notice",
    ("tpex", "disposal"):  "https://www.tpex.org.tw/www/zh-tw/bulletin/disposal",
    ("tpex", "attention"): "https://www.tpex.org.tw/www/zh-tw/bulletin/attention",
}


def _ts_roc_to_ad(s) -> Optional[str]:
    """民國 `114/07/07` 或 `114.07.04` → 西元 `'2025-07-07'`。格式不符回 None（不猜）。

    ⚠️ 分隔符**兩種都要收**：TWSE 的 punish 用 `/`、notice 用 `.`。
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


def _ts_get(url: str, params: dict) -> Optional[list]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        j = r.json()
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"trading_status {url.rsplit('/', 1)[-1]}：{type(e).__name__}: {e}")
        return None
    finally:
        time.sleep(0.8)
    if isinstance(j, dict):
        if j.get("data") is not None:
            return j["data"]
        t = j.get("tables") or []
        if t and t[0].get("data") is not None:
            return t[0]["data"]
    return None


def fetch_trading_status_direct(market: str, kind: str,
                                start: str, end: str) -> list[dict]:
    """抓單一市場 × 單一類型的公告，回傳 `[{stock_id, start, end, announced, market}]`。

    `start`/`end` 為 `YYYY-MM-DD`；市場間的格式差異在此函式內處理。
    """
    url = _TS_ENDPOINTS[(market, kind)]
    if market == "twse":
        params = {"response": "json", "startDate": start.replace("-", ""),
                  "endDate": end.replace("-", "")}
    else:
        params = {"response": "json", "startDate": start.replace("-", "/"),
                  "endDate": end.replace("-", "/")}
    rows = _ts_get(url, params) or []

    out = []
    for r in rows:
        try:
            if market == "twse":
                if kind == "disposal":
                    sid, ann, span = str(r[2]).strip(), _ts_roc_to_ad(r[1]), str(r[6])
                else:
                    sid, ann, span = str(r[1]).strip(), _ts_roc_to_ad(r[5]), None
            else:
                # TPEX 兩種公告欄序不同：disposal 代號在 r[2]、attention 在 r[1]。
                # 兩邊都用 r[2] 的話 attention 會拿到「證券名稱」→ 過不了 ^\d{4}$ → 0 列。
                if kind == "disposal":
                    sid, ann, span = str(r[2]).strip(), _ts_roc_to_ad(r[1]), str(r[5])
                else:
                    sid, ann, span = str(r[1]).strip(), _ts_roc_to_ad(r[5]), None

            if span is not None:                              # 處置：一段期間
                parts = [p for p in _TS_SEP.split(span) if p.strip()]
                if len(parts) != 2:
                    continue                                  # 含 TPEX 的骨架列
                s, e = _ts_roc_to_ad(parts[0]), _ts_roc_to_ad(parts[1])
            else:                                             # 注意：單日公告
                s = e = ann

            if not _TS_STOCK_RE.match(sid) or not s or not e:
                continue
            out.append({"stock_id": sid, "start": s, "end": e,
                        "announced": ann, "market": market})
        except (IndexError, TypeError):
            continue
    return out


def expand_trading_status_daily(recs: list[dict], status: str,
                                calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """把 `[start, end]` 期間展開成逐日列，只保留真實交易日。

    展開後下游只要用 `(Date, stock_id)` 查表就知道當天有無限制，不必再解析期間。
    """
    out = []
    for r in recs:
        s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        if e < s:
            continue
        for d in calendar[(calendar >= s) & (calendar <= e)]:
            out.append((str(d.date()), r["stock_id"], status,
                        r["market"], r["announced"]))
    return pd.DataFrame(out, columns=["Date", "stock_id", "status",
                                      "market", "announced"])


def _trading_calendar() -> pd.DatetimeIndex:
    """真實交易日曆（取自 `prices_raw` 的相異日期）。"""
    path = PROCESSED_DIR / "prices_raw.parquet"
    if not path.exists():
        return pd.DatetimeIndex([])
    d = pd.read_parquet(path, columns=["Date"])
    return pd.DatetimeIndex(sorted(pd.to_datetime(d["Date"].unique())))


def _catch_up_trading_status(today: str, lookback_days: int = 120) -> int:
    """處置股 / 注意股的**增量**補齊。

    【與 experimental 那支 `build()` 的分工】`build()` 是整檔重建（逐年抓 11 年、
    全檔覆寫），適合首次建立或重整；本函式只抓最近 `lookback_days` 天的公告後
    **合併**進既有檔，適合每日跑。

    【為什麼回看窗要夠長，不能只抓「昨天到今天」】處置是**一段期間**
    （通常 10~12 個營業日），一則今天公布的處置會涵蓋未來十幾天；
    反過來，前幾週公布的處置其期間可能延伸到今天。只抓當天公告會漏掉
    「期間仍在進行中、但公告日不在窗內」的那些。120 天遠大於單次處置期間，
    也順帶把偶爾漏抓的日子自我修復。

    【去重鍵】`[Date, stock_id, status]`，與 `build()` 完全相同（`keep="first"`）
    ——同一天同一支可能同時有處置與注意，status 必須進鍵。
    """
    path = PROCESSED_DIR / "trading_status_raw.parquet"
    cal = _trading_calendar()
    if len(cal) == 0:
        logger.warning("trading_status: 取不到交易日曆（prices_raw 不存在），跳過")
        return 0

    end = pd.Timestamp(today)
    start = end - pd.Timedelta(days=lookback_days)
    frames = []
    for kind in ("disposal", "attention"):
        for market in ("twse", "tpex"):
            try:
                recs = fetch_trading_status_direct(
                    market, kind, str(start.date()), str(end.date()))
            except Exception as e:                            # noqa: BLE001
                logger.warning(f"trading_status {market}/{kind} 抓取失敗：{e}")
                continue
            if recs:
                frames.append(expand_trading_status_daily(recs, kind, cal))

    if not frames:
        logger.info("trading_status: 近期無公告（或四個端點皆無回應）")
        return 0
    new = pd.concat(frames, ignore_index=True)

    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=new.columns)
    key = ["Date", "stock_id", "status"]
    out = pd.concat([old, new], ignore_index=True)
    out = out.drop_duplicates(subset=key, keep="first")       # 既有優先，與 build() 一致
    out = out.sort_values(key).reset_index(drop=True)
    out.to_parquet(path, index=False)
    added = len(out) - len(old)
    n_disp = int((new["status"] == "disposal").sum())
    logger.info(f"trading_status: 近 {lookback_days} 天抓到 {len(new):,} 個「股票×日」"
                f"（處置 {n_disp:,}／注意 {len(new) - n_disp:,}）→ 淨增 {added:,} 列"
                f"｜最新 {out['Date'].max()}")
    return added


def _live_universe() -> set[str]:
    """
    目前仍在交易的股票代號（取 `prices_raw` 最後一個交易日）。

    用途：月/季頻滾動補齊時排除已下市股票。它們的資料永遠停在下市那天，
    若不排除會一直佔住「最舊」的名次、把額度耗光，活躍股反而永遠輪不到。
    """
    try:
        path = PROCESSED_DIR / "prices_raw.parquet"
        if not path.exists():
            return set()
        d = pd.read_parquet(path, columns=["Date", "stock_id"])
        last = d["Date"].max()
        return set(d.loc[d["Date"] == last, "stock_id"].astype(str))
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"_live_universe 讀取失敗，不做現役過濾：{e}")
        return set()


def _catch_up_monthly(name: str, dataset: str, today: str,
                      max_stocks: int = 120, sleep_s: float = 0.6,
                      skip: set[str] | None = None,
                      attempted_out: set[str] | None = None) -> int:
    """
    月/季頻資料源的**滾動逐股**增量補齊（`revenue_raw` 月營收、`financials_raw` 季財報）。

    【為什麼需要】原本這兩個源只在 `run_full_data_sync(force_rebuild=True)` 時整份重抓，
    平時走 `if not force and cache.exists(): 用快取`——也就是**永遠不會自己更新**。
    revenue 因此停在 2026-04-01（落後 118 天）。這與 margin 的第二個根因同型：
    「有 fetcher、但沒被接進每日流程」＝遲早停更。

    【為什麼是逐股滾動，而不是一次抓全市場】
    2026-07-29 實測 FinMind 免費層（register）對這兩個 dataset 的限制是**形狀**而非速率：

        不帶 data_id（全市場）→ HTTP 400 "Your level is register"
        帶 data_id（單股）    → HTTP 200 success

    也就是說必須逐股查詢 ~2,000 次。那放不進每日推論路徑（會拖上數小時），
    但月頻資料一個月只變一次，**不需要每天全抓**。
    故改成每天只補「資料最舊的 N 支」，約 `2000/N` 天輪完一輪：
    N=120 → 約 17 天輪一輪，遠短於月頻的更新週期，且每天只多花約 1–2 分鐘。

    這也順帶讓補齊具備自我修復性：某天失敗的股票下次自然會因為「最舊」而排到前面。

    大範圍初始回補請用 `V6/scripts/backfill_monthly_finmind.py`（過夜跑），
    本函式只負責讓它**不再停更**。
    """
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        return 0
    old = pd.read_parquet(path)
    dcol = "date" if "date" in old.columns else "Date"
    old[dcol] = pd.to_datetime(old[dcol], errors="coerce")
    old["stock_id"] = old["stock_id"].astype(str)

    # 依「該股最新資料日期」排序，最舊的先補。
    # ⚠️ 必須先限制在**現役宇宙**內：直接排序會把已下市股票排到最前面
    #    （實測最舊是 2002-02-01），而它們永遠不會再有新資料，
    #    每輪都會浪費在同一批身上、真正需要更新的活躍股永遠排不到。
    per = old.groupby("stock_id")[dcol].max().sort_values()
    live = _live_universe()
    if live:
        n_all = len(per)
        per = per[per.index.isin(live)]
        logger.info(f"{name}: 現役宇宙過濾 {n_all:,} → {len(per):,} 支"
                    f"（排除已下市，它們不會再更新）")
    # `skip` 讓呼叫端（回補腳本）記住本次執行已試過誰。
    # ⚠️ 沒有這個機制時，回補腳本只能用「本輪淨增 0 就結束」判斷是否追平——
    #    但有些公司已停止申報、抓了本來就回 0，於是整個回補在還剩上千支
    #    沒處理時就提前中止（實測 financials 只跑 2.3 分鐘、1,925/1,942 支仍停在 2025-12）。
    #    「這一輪沒收穫」≠「全部都追平了」。
    if skip:
        per = per[~per.index.isin(skip)]
    targets = per.index[:max_stocks].tolist()
    if attempted_out is not None:
        attempted_out.update(targets)
    if not targets:
        return 0
    logger.info(f"{name}: 最新 {per.max().date()}｜最舊 {per.min().date()}"
                f"｜本輪補最舊 {len(targets)} 支（共 {len(per):,} 支，"
                f"約 {int(np.ceil(len(per) / max(max_stocks, 1)))} 天輪一輪）")

    frames, n_fail = [], 0
    quota_hit = False
    for sid in targets:
        start = (per[sid] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        try:
            d = _finmind_fetch(dataset, start_date=start, end_date=today, stock_id=sid)
        except FinMindQuotaExceeded as e:
            # 額度用盡就立刻停，繼續打只是空轉（而且會讓日誌看起來像「沒有新資料」）
            logger.warning(f"{name}: {e}｜本輪在第 {len(frames)} 支後中止，"
                           f"已抓到的仍會寫入；明日額度重置後再續")
            quota_hit = True
            break
        except Exception:                                     # noqa: BLE001
            d, n_fail = None, n_fail + 1
        if d is not None and not d.empty:
            frames.append(d)
        time.sleep(sleep_s)
    if quota_hit and attempted_out is not None:
        # 沒真的試到的股票要還回去，否則下次會被 skip 跳過而永遠補不到
        attempted_out.difference_update(set(targets[len(frames):]))

    if not frames:
        logger.warning(f"{name}: 本輪 {len(targets)} 支皆無新資料"
                       f"（失敗 {n_fail} 支）")
        return 0

    new = pd.concat(frames, ignore_index=True)
    # FinMind 回傳一律小寫 `date`，但 production parquet 可能存成 `Date`
    # （revenue_raw 就是）。不對齊會 KeyError，且若靜默新增一欄會讓下游 ffill 失效。
    if dcol not in new.columns:
        for cand in ("date", "Date"):
            if cand in new.columns:
                new = new.rename(columns={cand: dcol})
                break
    if dcol not in new.columns:
        logger.warning(f"{name}: FinMind 回傳無日期欄（{list(new.columns)[:6]}），跳過")
        return 0
    new[dcol] = pd.to_datetime(new[dcol], errors="coerce")
    new["stock_id"] = new["stock_id"].astype(str)
    for c in old.columns:
        if c not in new.columns:
            new[c] = pd.NA
    key = [dcol, "stock_id"] + (["type"] if "type" in old.columns else [])
    out = pd.concat([old, new[old.columns]], ignore_index=True)
    out = out.drop_duplicates(subset=key, keep="last")
    out = out.sort_values(key).reset_index(drop=True)
    out.to_parquet(path, index=False)
    added = len(out) - len(old)
    logger.info(f"{name}: 抓到 {len(new):,} 列 → 淨增 {added:,} 列"
                f"｜失敗 {n_fail} 支｜最新 {out[dcol].max().date()}")
    return added


def _catch_up_market_value(today: str, max_days: int = 15) -> tuple[int, list[str]]:
    """
    market_value = 當日收盤價 × 已發行普通股數（MOPS 股本快照）。
    沒有逐日的市值端點，所以用價格 × 股本自行計算——與
    `V6/scripts/backfill_stale_202607.py` 的 backfill_market_value() 同一套算法。

    ⚠️ 股本用的是 MOPS 的**當前快照**（非 point-in-time）。對「今天」而言正確，
       但若用來回補較久以前的日期，遇到期間內辦過增減資的公司會有偏差。
       此函式只補最近數日，影響可忽略；大範圍回補請用 backfill 腳本並註明此限制。
    """
    path = PROCESSED_DIR / "market_value_raw.parquet"
    price_path = PROCESSED_DIR / "prices_raw.parquet"
    if not path.exists() or not price_path.exists():
        return 0, []
    prod = pd.read_parquet(path)
    prod["Date"] = pd.to_datetime(prod["Date"])
    existing = set(prod["Date"].dt.strftime("%Y-%m-%d"))
    start = pd.to_datetime(max(existing)) + pd.Timedelta(days=1)

    pr = pd.read_parquet(price_path, columns=["Date", "stock_id", "Close"])
    pr["Date"] = pd.to_datetime(pr["Date"])
    pr = pr[(pr["Date"] >= start) & (pr["Date"] <= pd.to_datetime(today))]
    pr = pr[pr["Date"].dt.strftime("%Y-%m-%d").map(lambda d: d not in existing)]
    if pr.empty:
        return 0, []
    days = sorted(pr["Date"].dt.strftime("%Y-%m-%d").unique())
    if len(days) > max_days:
        days = days[-max_days:]
        pr = pr[pr["Date"].dt.strftime("%Y-%m-%d").isin(days)]

    shares = fetch_shares_outstanding_mops()
    if shares is None or shares.empty or "shares_outstanding" not in shares.columns:
        logger.warning("market_value: MOPS 股本抓取失敗，跳過")
        return 0, []
    shares = shares[["stock_id", "shares_outstanding"]].drop_duplicates("stock_id")

    mv = pr.merge(shares, on="stock_id", how="inner")
    mv["market_value"] = (pd.to_numeric(mv["Close"], errors="coerce")
                          * pd.to_numeric(mv["shares_outstanding"], errors="coerce"))
    mv = mv[mv["market_value"] > 0][["Date", "stock_id", "market_value"]]
    if mv.empty:
        return 0, []

    out = pd.concat([prod, mv[prod.columns]], ignore_index=True)
    out = out.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    out.sort_values(["stock_id", "Date"]).reset_index(drop=True).to_parquet(path, index=False)
    return len(mv), days


def run_daily_update(
    target_date: str | None = None,
    allow_forward_fill: bool = True,
) -> dict:
    """
    Lightweight update for daily inference.
    Only fetches today's data; uses cache for history.

    Args:
        target_date       : Target trading date 'YYYY-MM-DD'. Defaults to today.
        allow_forward_fill: If True, use yesterday's margin data when FinMind is
                            not yet updated. If False, margin is treated as missing.

    Returns:
        {
          "missing"       : list[str]  — sources with no data for today at all
          "forward_filled": list[str]  — sources where yesterday's data was substituted
        }
    """
    _MIN_PRICE_ROWS = 2000   # below this, treat the price cache as incomplete

    today = target_date or date.today().strftime("%Y-%m-%d")
    logger.info(f"=== V6 Daily Update: {today} (forward_fill={allow_forward_fill}) ===")

    missing: list[str] = []
    forward_filled: list[str] = []
    warnings_: list[str] = []

    # ── 交易日閘門（最前面，非交易日一律不寫任何檔）──────────────────────────
    if not is_trading_day(today):
        logger.info(f"=== {today} 非交易日，跳過所有資料寫入 ===")
        return {"missing": [], "forward_filled": [], "warnings": [],
                "skipped": "not_a_trading_day"}

    tse_ids, otc_ids = load_ticker_universe()

    # ── Price / Volume ────────────────────────────────────────────────────────
    price_path = PROCESSED_DIR / "prices_raw.parquet"

    _today_price_rows = 0
    if price_path.exists():
        _existing = pd.read_parquet(price_path, columns=["Date"])
        _existing["Date"] = pd.to_datetime(_existing["Date"]).dt.strftime("%Y-%m-%d")
        _today_price_rows = int((_existing["Date"] == today).sum())

    if _today_price_rows >= _MIN_PRICE_ROWS:
        logger.info(
            f"Prices for {today} already in parquet ({_today_price_rows:,} rows) — skipping download"
        )
    else:
        # yfinance end is exclusive — pass tomorrow to guarantee today is included.
        # Passing start==end returns 0 days for past dates (the most common cause
        # of "Price data incomplete" on re-runs or early-morning retries).
        _yf_end = (date.fromisoformat(today) + timedelta(days=1)).isoformat()
        df_prices, _missing_tickers = fetch_prices_yfinance(tse_ids, otc_ids, today, _yf_end)

        # Even when yfinance returns some TSE data, OTC (.TWO) stocks may be
        # entirely missing.  Check OTC coverage and supplement with TPEX direct
        # if more than 70 % of OTC universe is absent.
        _otc_got = set(df_prices["stock_id"].unique()) & set(otc_ids) if not df_prices.empty else set()
        _otc_missing_ratio = 1.0 - len(_otc_got) / max(len(otc_ids), 1)

        if df_prices.empty or _otc_missing_ratio > 0.70:
            if df_prices.empty:
                logger.info("yfinance empty → TWSE/TPEX direct price fetch...")
            else:
                logger.warning(
                    f"yfinance OTC coverage too low ({len(_otc_got)}/{len(otc_ids)} stocks) "
                    f"→ supplementing with TWSE/TPEX direct..."
                )
            frames = [] if df_prices.empty else [df_prices]
            df_twse = fetch_prices_twse_direct(today)
            if df_twse is not None and not df_twse.empty:
                frames.append(df_twse)
            df_tpex = fetch_prices_tpex_direct(today)
            if df_tpex is not None and not df_tpex.empty:
                frames.append(df_tpex)
            if frames:
                # ⚠️ 各來源的 Date 型別不同：yfinance 走 datetime index → Timestamp；
                #    fetch_prices_twse_direct / fetch_prices_tpex_direct 直接塞 date_str → str。
                #    concat 後欄位變成 object 混型，Timestamp('2026-07-22') 與 '2026-07-22'
                #    被視為不同值 → drop_duplicates 完全失效，同股同日兩列都會寫進 parquet。
                #    實測後果（2026-07-06 起）：每日約 1,550 列重複；同一天同時存在 yfinance
                #    還原價與 TWSE 未還原價（1459 於 2026-07-22 兩列比值 1.188）；Volume 單位
                #    也不一致（1213 於 2026-07-15 為 3000 vs 1）。
                #    因此務必先把 Date 統一成 YYYY-MM-DD 字串再去重。
                _src_rows = [len(f) for f in frames]
                for _f in frames:
                    _f["Date"] = pd.to_datetime(_f["Date"]).dt.strftime("%Y-%m-%d")
                df_prices = pd.concat(frames, ignore_index=True).drop_duplicates(
                    # keep="first" → yfinance（auto_adjust 還原價）優先，與歷史序列的還原
                    # 基準一致；TWSE/TPEX direct 只用來補 yfinance 抓不到的股票。
                    subset=["Date", "stock_id"], keep="first",
                )
                logger.info(
                    f"多來源價格合併 {today}：各源列數 {_src_rows} → 去重後 {len(df_prices)} 列"
                    f"（移除跨來源重疊 {sum(_src_rows) - len(df_prices)} 列）"
                )
            else:
                logger.warning("TWSE/TPEX direct also returned no price data")

        if not df_prices.empty:
            _append_to_parquet(price_path, df_prices, today)
            _today_price_rows += len(df_prices)

    if _today_price_rows < _MIN_PRICE_ROWS:
        logger.error(
            f"Price data incomplete for {today}: only {_today_price_rows} rows "
            f"(need ≥ {_MIN_PRICE_ROWS})"
        )
        missing.append("prices")
    else:
        logger.info(f"Price data OK: {_today_price_rows:,} rows for {today}")

    # ── Institutional Investors ───────────────────────────────────────────────
    df_tse = fetch_institutional_twse(today)
    df_otc = fetch_institutional_tpex(today)
    inst_frames = [x for x in [df_tse, df_otc] if x is not None]
    if inst_frames:
        inst_today = pd.concat(inst_frames, ignore_index=True)
        if not inst_today.empty:
            inst_today["Date"] = today
            _append_to_parquet(PROCESSED_DIR / "institutional_raw.parquet", inst_today, today)
            logger.info(f"Institutional data OK: {len(inst_today)} rows for {today}")
    else:
        logger.warning(
            f"No institutional data for {today} "
            f"(market may be closed or data not yet published)"
        )
        missing.append("institutional")

    # ── Margin / Short Sale ───────────────────────────────────────────────────
    # 2026-07-27 重寫。舊版有兩個獨立的問題，疊加造成 margin_raw 停更三個月無人察覺：
    #   ① 走 fetch_margin_finmind() —— FinMind VIP 到期後失效，且該資料集對**上櫃股**
    #      把券賣/券買標反（見 V6/scripts/fix_margin_short_swap.py 的證據與修復）
    #   ② 更根本：它只寫到 CACHE_DIR 的單日暫存檔，**從未 append 到 margin_raw.parquet**
    #      —— 這才是 margin_raw 停在「最後一次全量同步日」的真正原因
    # 現在改交易所直連（恆等式 100% 成立）並真正寫回 margin_raw.parquet。
    n_margin, margin_days = _catch_up_margin(today)
    if n_margin == 0:
        logger.warning(f"Margin data unavailable for {today} — skipping (non-critical)")
    else:
        logger.info(f"Margin data OK: {n_margin:,} rows for {len(margin_days)} day(s) "
                    f"{margin_days}")

    # ── Day Trading（2026-07-27 納入每日更新）──────────────────────────────────
    n_dt, dt_days = _catch_up_daytrade(today)
    if n_dt == 0:
        logger.warning(f"Daytrade data unavailable for {today} — skipping (non-critical)")
    else:
        logger.info(f"Daytrade data OK: {n_dt:,} rows for {len(dt_days)} day(s) {dt_days}")

    # ── PER/PBR、借券餘額、市值（2026-07-28 納入每日更新）─────────────────────
    # 這三個源原本完全不在每日更新裡，靠手動全量同步，因此自 2026-04-24 停更三個月。
    # ⚠️ per 的來源 TWSE BWIBBU_ALL 只涵蓋上市；上櫃 PER/PBR 仍缺（見 TODO）。
    for _nm, _fn, _ren in (
        ("per_raw", fetch_per_twse_direct, {"dividend_yield": "DY"}),
        ("securities_raw", fetch_securities_lending_twse_direct,
         {"Securities_Balance": "Securities_Lending"}),
    ):
        try:
            _n, _d = _catch_up_generic(_nm, _fn, today, rename=_ren)
            logger.info(f"{_nm}: {_n:,} rows for {len(_d)} day(s) {_d}" if _n
                        else f"{_nm}: 無新增（已最新或今日尚未公布）")
        except Exception as _e:                               # noqa: BLE001
            logger.warning(f"{_nm} 更新失敗（不影響推論）：{_e}")
    try:
        _n_mv, _d_mv = _catch_up_market_value(today)
        logger.info(f"market_value_raw: {_n_mv:,} rows for {len(_d_mv)} day(s)" if _n_mv
                    else "market_value_raw: 無新增")
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"market_value_raw 更新失敗（不影響推論）：{_e}")

    # ── 外資持股（2026-07-28 納入；上市 MI_QFIIS + 上櫃 insti/qfii）────────────
    # 同樣是原本不在每日更新裡的源，自 2026-05-05 停更 84 天，
    # 讓 Foreign_Holding_Pct 這一維變成凍結值。
    try:
        _n_fs, _d_fs = _catch_up_generic(
            "foreign_shareholding_raw", fetch_foreign_shareholding_direct,
            today, date_col="date")
        logger.info(f"foreign_shareholding_raw: {_n_fs:,} rows for {len(_d_fs)} day(s)"
                    if _n_fs else "foreign_shareholding_raw: 無新增")
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"foreign_shareholding_raw 更新失敗（不影響推論）：{_e}")

    # ── 期貨/選擇權三大法人（2026-07-29 納入；TAIFEX 直連）────────────────────
    # FinMind 免費層對這兩個 dataset 回 400，不訂閱就永遠拿不到。
    # TAIFEX 是不同主機，不與 TWSE/TPEX 搶速率，也不受 FinMind 額度影響。
    try:
        _n_fut, _n_opt = _catch_up_taifex(today)
        logger.info(f"TAIFEX 三大法人：期貨 +{_n_fut:,} 列｜選擇權 +{_n_opt:,} 列"
                    if (_n_fut or _n_opt) else "TAIFEX 三大法人：無新增")
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"TAIFEX 三大法人更新失敗（不影響推論）：{_e}")

    # ── 集保股權分散（2026-07-29 納入；TDCC 直連，週頻）──────────────────────
    # FinMind 的 TaiwanStockHoldingSharesPer 已需付費層。TDCC 開放資料免費、
    # 一次請求即全市場，但**只有最新一週**（見 fetcher docstring）。
    # 週頻資料每天跑無妨——去重會處理，且對公布延遲自我修復。
    try:
        _n_hold = _catch_up_holdings()
        logger.info(f"holdings_raw: +{_n_hold:,} 列" if _n_hold else "holdings_raw: 無新增")
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"holdings_raw 更新失敗（不影響推論）：{_e}")

    # ── 股利分派（2026-07-29 納入；MOPS 直連，含上市與上櫃）────────────────────
    # FinMind 的 TaiwanStockDividend 是逐股查詢（~2,000 次），撞爆免費層額度；
    # MOPS 是兩個 CSV、兩次請求。
    try:
        _n_div = _catch_up_dividends()
        logger.info(f"dividend_raw: +{_n_div:,} 列" if _n_div else "dividend_raw: 無新增")
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"dividend_raw 更新失敗（不影響推論）：{_e}")

    # ── 處置股 / 注意股（2026-08-04 納入）──────────────────────────────────────
    # 組合建構系統要用它排除處置股（分盤集合競價 → 很難照收盤價成交）。
    # 原本只有 experimental 的整檔重建腳本、靠手動跑，停在 2026-07-31。
    try:
        _n_ts = _catch_up_trading_status(today)
        if not _n_ts:
            logger.info("trading_status_raw: 無新增")
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"trading_status_raw 更新失敗（不影響推論）：{_e}")

    # ── 財報三表 + 月營收：MOPS 整批直連（2026-08-04 納入）────────────────────
    # 【為什麼要加在 FinMind 之前】2026-08-04 稽核發現財報有覆蓋斷崖：
    #   financials 2026Q1 只有 176 支（7%）、balance_sheet/cashflow 只有 16 支（0.7%）、
    #   月營收 2026-05 起只剩 871 支（44%）。根因是 FinMind 免費層只能逐股查詢，
    #   實測滾動速率約 32 支/天，補完 2,300 支要 70 天——**追不上每季新進來的量**。
    #   MOPS 是一季兩次請求拿到 1,929 家，且 balance_sheet/cashflow 本來就不在
    #   任何每日流程裡（`_catch_up_monthly` 只涵蓋 revenue + financials）。
    # 【與下方 FinMind 的分工】MOPS 只有現存公司（1,972 支），FinMind 的歷史含
    #   已下市股（2,475 支）。MOPS 先把主體補滿之後，FinMind 的「補最舊 N 支」
    #   會自然改去補 MOPS 涵蓋不到的那批，兩者互補、不重工。
    # 【已驗證】`V6/scripts/validate_mops_financials.py` 四項全過（2026-08-04）：
    #   差分規則、量級交叉（median 比值 1.000000）、指標核對、接縫連續性。
    try:
        _mops_q = _catch_up_mops_quarterly(today)
        if any(_mops_q.values()):
            logger.info("MOPS 財報補齊：" + "｜".join(
                f"{k} +{v:,}" for k, v in _mops_q.items() if v))
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"MOPS 財報補齊失敗（不影響推論）：{_e}")
    try:
        _mops_r = _catch_up_mops_revenue(today)
        if _mops_r:
            logger.info(f"MOPS 月營收補齊：+{_mops_r:,} 列")
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"MOPS 月營收補齊失敗（不影響推論）：{_e}")

    # ── 月/季頻資料源（2026-07-29 納入；FinMind 免費層仍可用）──────────────────
    # 這兩個源原本只在 force_rebuild 時整份重抓，平時走快取分支＝永遠不會更新，
    # revenue 因此停更 118 天。距上次資料未滿門檻天數會自動跳過，不會每天空打。
    # 逐股滾動：每天補「現役宇宙中資料最舊的 N 支」，約 16 天輪完一輪。
    # 月/季頻資料一個月才變一次，不需要每天全抓；這樣每天只多花約 1–2 分鐘。
    # ⚠️ 批量刻意設得遠低於免費層額度（600 次/日）：
    #    2026-07-29 實測，短時間密集打 ~1,000 次後 FinMind 直接回 403 ip banned，
    #    也就是**請求速率本身就會觸發封鎖**，不是把 600 次用完才擋。
    #    80+40=120 次/日約占額度 20%，留足空間給臨時查詢，也不致觸發速率封鎖。
    #    代價是追平較慢（~24 天），但月/季頻資料本來就不急。
    for _nm, _ds, _n in (("revenue_raw", "TaiwanStockMonthRevenue", 80),
                         ("financials_raw", "TaiwanStockFinancialStatements", 40)):
        try:
            _catch_up_monthly(_nm, _ds, today, max_stocks=_n, sleep_s=1.0)
        except FinMindQuotaExceeded as _e:
            logger.warning(f"{_nm}: {_e}｜今日不再嘗試")
        except Exception as _e:                               # noqa: BLE001
            logger.warning(f"{_nm} 更新失敗（不影響推論）：{_e}")

    # ── 覆蓋率閘門（只告警不阻擋，見函式 docstring）────────────────────────────
    cov = _check_universe_coverage(today)
    if cov:
        warnings_.append(cov)

    logger.info(
        f"=== Daily update done: {today} | "
        f"missing={missing or 'none'} | fwd_fill={forward_filled or 'none'} | "
        f"warnings={warnings_ or 'none'} ==="
    )
    return {"missing": missing, "forward_filled": forward_filled,
            "warnings": warnings_}



# ============================================================
# Macro Data (VIX, SPX, Gold, Oil, USD/TWD)
# ============================================================

def _sync_macro_data(start: str, end: str, force: bool = False) -> None:
    macro_cache = PROCESSED_DIR / "macro_raw.parquet"
    if not force and macro_cache.exists():
        logger.info("Macro loaded from cache")
        return

    # 2026-07-24 補上 TWII：_merge_macro() 把 TWII_Close 重新命名成 TWII 餵模型
    # （對 V6.1 因 Group D 橫斷面 z-score bug 而被歸零、對 V6.2 有默認值 0.0 兜底，
    # 兩者都不緊急），但 signal_conditions.py 的 regime 閘門（TWII vs MA60）是直接
    # 讀 macro_raw 的 TWII_Close 判斷保守模式，不經過模型特徵管線、不受歸零 bug
    # 影響，需要真的新鮮資料。US_SOX/CNN_FearGreed/TW_Biz_Signal/FED_Rate 確認
    # 對現有模型無實質影響（Fear_Greed 見決策紀錄；其餘同屬 Group D 同一個 bug），
    # 暫不在此補齊，避免陷入用不到的資料源。
    macro_tickers = {
        "^VIX":    "VIX",
        "^GSPC":   "SPX",
        "^TWII":   "TWII_Close",
        "GC=F":    "Gold",
        "CL=F":    "Oil",
        "^TNX":    "TNX",
        "TWD=X":   "USD_TWD",
    }
    frames = []
    for ticker, col in macro_tickers.items():
        df_t = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df_t.empty:
            continue
        # --- FIX: flatten MultiIndex from new yfinance API ---
        # New yfinance (>= 0.2.18) returns MultiIndex (Price, Ticker) for all downloads
        if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
            df_t = df_t.loc[:, ~df_t.columns.duplicated()]  # drop duplicate col names
        # ---
        df_t = df_t[["Close"]].rename(columns={"Close": col}).reset_index()
        if "Date" not in df_t.columns and "index" in df_t.columns:
            df_t.rename(columns={"index": "Date"}, inplace=True)
        df_t["Date"] = pd.to_datetime(df_t["Date"])
        frames.append(df_t)

    if not frames:
        logger.warning("No macro data fetched")
        return

    df_macro = frames[0]
    for df_t in frames[1:]:
        df_macro = df_macro.merge(df_t, on="Date", how="outer")
    df_macro.sort_values("Date", inplace=True)
    df_macro.to_parquet(macro_cache)
    logger.info(f"Macro saved: {df_macro.shape}")


def _sync_monthly_data(force: bool = False) -> None:
    """Fetch monthly revenue and quarterly financials from FinMind in yearly chunks."""
    today = date.today().strftime("%Y-%m-%d")

    revenue_cache = PROCESSED_DIR / "revenue_raw.parquet"
    if not force and revenue_cache.exists():
        logger.info("Monthly revenue loaded from cache")
    else:
        logger.info("Fetching monthly revenue via FinMind (chunked yearly)...")
        df_rev = _finmind_fetch_chunked(
            "TaiwanStockMonthRevenue",
            start_date=DATA_START_DATE,
            end_date=today,
        )
        if df_rev is not None and not df_rev.empty:
            df_rev.to_parquet(revenue_cache)
            logger.info(f"Revenue saved: {df_rev.shape}")
        else:
            logger.warning("Revenue data unavailable from FinMind")

    fin_cache = PROCESSED_DIR / "financials_raw.parquet"
    if not force and fin_cache.exists():
        logger.info("Financial statements loaded from cache")
    else:
        logger.info("Fetching financial statements via FinMind (chunked yearly)...")
        df_fin = _finmind_fetch_chunked(
            "TaiwanStockFinancialStatements",
            start_date=DATA_START_DATE,
            end_date=today,
        )
        if df_fin is not None and not df_fin.empty:
            df_fin.to_parquet(fin_cache)
            logger.info(f"Financials saved: {df_fin.shape}")
        else:
            logger.warning("Financial statements unavailable from FinMind")


# ============================================================
# Utility Helpers
# ============================================================

def _get_trading_days(df_prices: pd.DataFrame) -> list[str]:
    """Extract sorted unique trading day strings from a price DataFrame."""
    days = pd.to_datetime(df_prices["Date"]).dt.strftime("%Y-%m-%d").unique().tolist()
    return sorted(days)


def _append_to_parquet(path: Path, df_new: pd.DataFrame, date_str: str) -> None:
    """
    Append new rows to an existing parquet, replacing rows for date_str if present.
    Always normalizes the Date column to YYYY-MM-DD string to prevent schema conflicts
    (e.g., existing parquet may have Date as int64/datetime, new data as string).
    """
    if df_new.empty:
        return

    # Normalize df_new Date to string
    df_new = df_new.copy()
    if "Date" in df_new.columns:
        df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.strftime("%Y-%m-%d")

    if path.exists():
        df_old = pd.read_parquet(path)
        # Normalize df_old Date to string (it might be int64/datetime from original sync)
        if "Date" in df_old.columns:
            df_old["Date"] = pd.to_datetime(df_old["Date"]).dt.strftime("%Y-%m-%d")
        # Remove any existing rows for this date before appending
        df_old = df_old[df_old["Date"] != date_str]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # 保險絲：本函式的兩個呼叫端（prices_raw / institutional_raw）都應該是
    # (Date, stock_id) 唯一。上游任何來源合併疏漏都不該讓 parquet 長出重複列，
    # 因為重複列會直接污染時序特徵（rolling 名義 N 天、實際回溯不足 N 天）。
    if {"Date", "stock_id"} <= set(df.columns):
        _n_before = len(df)
        df = df.drop_duplicates(subset=["Date", "stock_id"], keep="last")
        if len(df) < _n_before:
            logger.warning(
                f"{path.name}: 寫入前偵測到 {_n_before - len(df):,} 列 (Date, stock_id) "
                f"重複，已去除（請追查上游來源合併邏輯）"
            )

    df.to_parquet(path)


# ============================================================
# V6.1 New Data Sources (Phase 2)
# ============================================================

def fetch_futures_institutional(
    start: str = DATA_START_DATE,
    end:   str | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch 期貨三大法人 (Futures Institutional Investors) from FinMind.
    Dataset: TaiwanFuturesInstitutionalInvestors

    Key output columns:
      - Date, institutional_investors (外資/投信/自營商)
      - long_deal_volume, short_deal_volume  (多空成交口數)
      - long_open_interest, short_open_interest  (多空未平倉口數)

    For the model, we compute:
      - Futures_OI_Foreign = 外資 (long_open_interest - short_open_interest)
    """
    cache_path = PROCESSED_DIR / "futures_institutional_raw.parquet"
    if not force and cache_path.exists():
        logger.info(f"Futures institutional loaded from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching futures institutional data [{start} → {end}]...")

    df = _finmind_fetch_chunked(
        "TaiwanFuturesInstitutionalInvestors",
        start_date=start,
        end_date=end,
    )
    if df is None or df.empty:
        logger.warning("Futures institutional data unavailable")
        return None

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter for 台指期 (TX) which is the most relevant futures contract
    if "futures_id" in df.columns:
        df = df[df["futures_id"].str.contains("TX", case=False, na=False)].copy()

    df.to_parquet(cache_path)
    logger.info(f"Futures institutional saved: {df.shape}")
    return df


def fetch_options_institutional(
    start: str = DATA_START_DATE,
    end:   str | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch 選擇權三大法人 (Options Institutional Investors) from FinMind.
    Dataset: TaiwanOptionInstitutionalInvestors

    For the model, we compute:
      - Options_PC_Ratio = Put volume / Call volume (Put/Call ratio)
    """
    cache_path = PROCESSED_DIR / "options_institutional_raw.parquet"
    if not force and cache_path.exists():
        logger.info(f"Options institutional loaded from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching options institutional data [{start} → {end}]...")

    df = _finmind_fetch_chunked(
        "TaiwanOptionInstitutionalInvestors",
        start_date=start,
        end_date=end,
    )
    if df is None or df.empty:
        logger.warning("Options institutional data unavailable")
        return None

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    df.to_parquet(cache_path)
    logger.info(f"Options institutional saved: {df.shape}")
    return df


def fetch_total_return_index(
    start: str = DATA_START_DATE,
    end:   str | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch 加權股價報酬指數 (Total Return Index, includes dividends).
    Dataset: TaiwanStockTotalReturnIndex

    This is the correct benchmark for Alpha computation — includes
    reinvested dividends, unlike the raw TAIEX/TWII.
    """
    cache_path = PROCESSED_DIR / "total_return_index_raw.parquet"
    if not force and cache_path.exists():
        logger.info(f"Total return index loaded from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching total return index [{start} → {end}]...")

    df = _finmind_fetch_chunked(
        "TaiwanStockTotalReturnIndex",
        start_date=start,
        end_date=end,
    )
    if df is None or df.empty:
        logger.warning("Total return index data unavailable")
        return None

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    df.to_parquet(cache_path)
    logger.info(f"Total return index saved: {df.shape}")
    return df


def fetch_dividends(
    start: str = DATA_START_DATE,
    end:   str | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch 股利政策 (Dividend announcements) from FinMind.
    Dataset: TaiwanStockDividend

    Key output columns:
      - stock_id, date, CashEarningsDistribution (現金股利)
      - StockEarningsDistribution (股票股利)

    For the model: Dividend_Yield_Fwd = announced cash dividend / current price
    """
    cache_path = PROCESSED_DIR / "dividend_raw.parquet"
    if not force and cache_path.exists():
        logger.info(f"Dividends loaded from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching dividend data [{start} → {end}]...")

    df = _finmind_fetch_chunked(
        "TaiwanStockDividend",
        start_date=start,
        end_date=end,
    )
    if df is None or df.empty:
        logger.warning("Dividend data unavailable")
        return None

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    df.to_parquet(cache_path)
    logger.info(f"Dividends saved: {df.shape}")
    return df


def fetch_foreign_shareholding(
    start: str = DATA_START_DATE,
    end:   str | None = None,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Fetch 外資持股比例 (Foreign Investor Shareholding %) from FinMind.
    Dataset: TaiwanStockShareholding

    Unlike institutional_raw (which shows daily BUY/SELL flow),
    this shows the CUMULATIVE holding percentage — a more stable
    signal of foreign investor conviction.
    """
    cache_path = PROCESSED_DIR / "foreign_shareholding_raw.parquet"
    if not force and cache_path.exists():
        logger.info(f"Foreign shareholding loaded from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    end = end or date.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching foreign shareholding data [{start} → {end}]...")

    df = _finmind_fetch_chunked(
        "TaiwanStockShareholding",
        start_date=start,
        end_date=end,
    )
    if df is None or df.empty:
        logger.warning("Foreign shareholding data unavailable")
        return None

    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    df.to_parquet(cache_path)
    logger.info(f"Foreign shareholding saved: {df.shape}")
    return df

