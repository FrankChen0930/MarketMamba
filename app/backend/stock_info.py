"""
Taiwan Stock Info Cache
========================
Priority:
  1. Bundled ticker_mapping.json  (fast, always available, offline-safe)
  2. TWSE / TPEX open API         (fresher data, used to fill gaps and refresh)

ticker_mapping.json format: {"2330.TW": "台積電", "2317.TW": "鴻海", ...}
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Industry code → Chinese name mapping ─────────────────────────────────────
_SECTOR_MAP = {
    "01": "水泥",        "02": "食品",        "03": "塑膠",
    "04": "紡織纖維",    "05": "電機機械",    "06": "電器電纜",
    "07": "化學生技醫療","08": "玻璃陶瓷",    "09": "造紙",
    "10": "鋼鐵",        "11": "橡膠",        "12": "汽車",
    "13": "電子",        "14": "建材營造",    "15": "航運",
    "16": "觀光餐旅",    "17": "金融保險",    "18": "貿易百貨",
    "19": "綜合",        "20": "其他",        "21": "化學",
    "22": "生技醫療",    "23": "油電燃氣",    "24": "半導體",
    "25": "電腦周邊",    "26": "光電",        "27": "通信網路",
    "28": "電子零組件",  "29": "電子通路",    "30": "資訊服務",
    "31": "其他電子",    "32": "文化創意",    "33": "農業科技",
    "34": "電子商務",    "35": "綠能環保",    "36": "數位雲端",
    "37": "運動休閒",    "38": "居家生活",    "39": "新零售",
    "40": "管理股票",    "41": "存託憑證",
    "80": "ETF",         "81": "受益憑證",
    "AA": "電子",        "AB": "半導體",      "AC": "其他電子",
}

# ── Static fallback: ticker_mapping.json ─────────────────────────────────────
_STATIC_MAP: dict[str, str] = {}   # {"2330": "台積電", ...}

def _load_static_map() -> dict[str, str]:
    """Load bundled ticker_mapping.json once at startup."""
    global _STATIC_MAP
    if _STATIC_MAP:
        return _STATIC_MAP

    # Search for ticker_mapping.json relative to this file
    candidates = [
        Path(__file__).parent / "ticker_mapping.json",
        Path(__file__).parent.parent / "ticker_mapping.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                # Normalize: strip ".TW" / ".TWO" suffix
                _STATIC_MAP = {
                    k.split(".")[0]: v
                    for k, v in raw.items()
                    if isinstance(v, str) and v.strip()
                }
                logger.info(f"Loaded {len(_STATIC_MAP)} stocks from static ticker_mapping.json")
                return _STATIC_MAP
            except Exception as e:
                logger.warning(f"Failed to load ticker_mapping.json: {e}")

    logger.warning("ticker_mapping.json not found — will rely on API only")
    return {}

# Load at import time (synchronous, fast)
_load_static_map()

# ── In-memory API cache ───────────────────────────────────────────────────────
_stock_info: dict[str, dict] = {}   # { "2330": {"name": "台積電", "sector": "半導體"} }
_cache_time: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=24)


async def _fetch_one(url: str, market: str) -> dict[str, dict]:
    """Fetch one exchange's company list from open API."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            if r.status_code != 200:
                logger.warning(f"Stock info fetch failed ({market}): {r.status_code}")
                return {}
            data = r.json()

        result = {}
        for row in data:
            ticker = str(row.get("公司代號", "")).strip()
            name   = str(row.get("公司簡稱", "")).strip()
            code   = str(row.get("產業別", "")).strip().lstrip("0") or "20"
            sector_code = code.zfill(2) if code.isdigit() else code
            sector = _SECTOR_MAP.get(sector_code, "其他")
            if ticker and name:
                result[ticker] = {"name": name, "sector": sector}
        logger.info(f"Loaded {len(result)} stocks from {market} API")
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch stock info ({market}): {e}")
        return {}


async def get_stock_info() -> dict[str, dict]:
    """
    Return {ticker: {name, sector}} for all stocks.

    Strategy:
    - Always seed from static ticker_mapping.json (instant, offline-safe)
    - Try to enrich with live TWSE/TPEX API (adds sector info)
    - API failure is non-fatal: static map still covers names
    """
    global _stock_info, _cache_time

    # Return fresh API cache if available
    if _stock_info and _cache_time and datetime.now() - _cache_time < _CACHE_TTL:
        return _stock_info

    # Seed from static map (name only, sector defaults to "其他")
    base = {
        ticker: {"name": name, "sector": "其他"}
        for ticker, name in _STATIC_MAP.items()
    }

    # Try to enrich with live API (adds sector info)
    try:
        tse_url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        otc_url = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        tse = await _fetch_one(tse_url, "TSE")
        otc = await _fetch_one(otc_url, "OTC")
        api_data = {**tse, **otc}

        if api_data:
            # Merge: API data takes precedence (has sector), static fills gaps
            merged = {**base, **api_data}
            _stock_info = merged
            _cache_time = datetime.now()
            logger.info(f"Stock info: {len(api_data)} from API + {len(base)} static = {len(_stock_info)} total")
            return _stock_info
    except Exception as e:
        logger.warning(f"API enrichment failed, using static map only: {e}")

    # Fallback: static map only (still has all names)
    _stock_info = base
    _cache_time = datetime.now()
    logger.info(f"Using static stock info only: {len(_stock_info)} stocks")
    return _stock_info


def get_stock_name(ticker: str, info: dict[str, dict]) -> str:
    """Get stock short name. Falls back to static map then ticker."""
    # Check passed-in info first
    if ticker in info:
        name = info[ticker].get("name", "")
        if name and name != ticker:
            return name
    # Direct static map fallback
    static_name = _STATIC_MAP.get(ticker, "")
    return static_name if static_name else ticker


def get_stock_sector(ticker: str, info: dict[str, dict]) -> str:
    """Get stock sector name, fallback to '其他'."""
    return info.get(ticker, {}).get("sector", "其他")
