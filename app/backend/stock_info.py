"""
Taiwan Stock Info Cache
========================
Fetches company name + sector from TWSE & TPEX public APIs.
Cached in-memory for 24 hours to avoid repeated API calls.

TSE API: https://openapi.twse.com.tw/v1/opendata/t187ap03_L
OTC API: https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Industry code → Chinese name mapping ────────────────────────────────────
_SECTOR_MAP = {
    "01": "水泥",
    "02": "食品",
    "03": "塑膠",
    "04": "紡織纖維",
    "05": "電機機械",
    "06": "電器電纜",
    "07": "化學生技醫療",
    "08": "玻璃陶瓷",
    "09": "造紙",
    "10": "鋼鐵",
    "11": "橡膠",
    "12": "汽車",
    "13": "電子",
    "14": "建材營造",
    "15": "航運",
    "16": "觀光餐旅",
    "17": "金融保險",
    "18": "貿易百貨",
    "19": "綜合",
    "20": "其他",
    "21": "化學",
    "22": "生技醫療",
    "23": "油電燃氣",
    "24": "半導體",
    "25": "電腦周邊",
    "26": "光電",
    "27": "通信網路",
    "28": "電子零組件",
    "29": "電子通路",
    "30": "資訊服務",
    "31": "其他電子",
    "32": "文化創意",
    "33": "農業科技",
    "34": "電子商務",
    "35": "綠能環保",
    "36": "數位雲端",
    "37": "運動休閒",
    "38": "居家生活",
    "39": "新零售",
    "40": "管理股票",
    "41": "存託憑證",
    "80": "ETF",
    "81": "受益憑證",
    "AA": "電子",        # OTC 電子
    "AB": "半導體",      # OTC 半導體
    "AC": "其他電子",
}

# ── In-memory cache ──────────────────────────────────────────────────────────
_stock_info: dict[str, dict] = {}   # { "2330": {"name": "台積電", "sector": "半導體"} }
_cache_time: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=24)


async def _fetch_one(url: str, market: str) -> dict[str, dict]:
    """Fetch one exchange's company list and parse into {ticker: {name, sector}}."""
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
            # Normalize code: pad to 2 chars
            sector_code = code.zfill(2) if code.isdigit() else code
            sector = _SECTOR_MAP.get(sector_code, "其他")
            if ticker and name:
                result[ticker] = {"name": name, "sector": sector}
        logger.info(f"Loaded {len(result)} stocks from {market}")
        return result
    except Exception as e:
        logger.error(f"Failed to fetch stock info ({market}): {e}")
        return {}


async def get_stock_info() -> dict[str, dict]:
    """
    Return {ticker: {name, sector}} for all TSE + OTC stocks.
    Result is cached for 24 hours.
    """
    global _stock_info, _cache_time

    if _stock_info and _cache_time and datetime.now() - _cache_time < _CACHE_TTL:
        return _stock_info

    tse_url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    otc_url = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"

    tse = await _fetch_one(tse_url, "TSE")
    otc = await _fetch_one(otc_url, "OTC")

    _stock_info = {**tse, **otc}
    _cache_time = datetime.now()
    logger.info(f"Stock info cache updated: {len(_stock_info)} total stocks")
    return _stock_info


def get_stock_name(ticker: str, info: dict[str, dict]) -> str:
    """Get stock short name, fallback to ticker if not found."""
    return info.get(ticker, {}).get("name", ticker)


def get_stock_sector(ticker: str, info: dict[str, dict]) -> str:
    """Get stock sector name, fallback to '其他' if not found."""
    return info.get(ticker, {}).get("sector", "其他")
