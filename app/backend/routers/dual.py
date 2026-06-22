"""
Dual-model signals router — Phase 2 步驟 5
==========================================
讀 GitHub raw 的 df_short.csv / df_trend.csv（1h TTL cache + asyncio.Lock，
完全比照 signals.py 的 df_kelly 模式），一次回短線 + 趨勢兩組。

⚠️ 附加 router，完全不動既有 signals/market/performance 等。
⚠️ 輸出是 rank-score 語意（非報酬）：Score 越高=預測排名越前；SQ = Score / Uncertainty。
"""
import os
import io
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dual", tags=["Dual"])

# 重用既有 GITHUB_RESULTS_URL（指向 df_kelly.csv），換檔名即得 short/trend
GITHUB_RESULTS_URL = os.getenv("GITHUB_RESULTS_URL", "")
SHORT_URL = GITHUB_RESULTS_URL.replace("df_kelly.csv", "df_short.csv") if GITHUB_RESULTS_URL else ""
TREND_URL = GITHUB_RESULTS_URL.replace("df_kelly.csv", "df_trend.csv") if GITHUB_RESULTS_URL else ""
CACHE_TTL = timedelta(hours=1)

_cache: Optional[dict] = None
_cache_time: Optional[datetime] = None
_cache_lock: asyncio.Lock = asyncio.Lock()


async def _fetch_csv(url: str) -> Optional[pd.DataFrame]:
    if not url:
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
        if r.status_code != 200:
            logger.warning(f"Dual fetch failed {url}: {r.status_code}")
            return None
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        logger.error(f"Dual fetch error {url}: {e}")
        return None


def _rows(df: pd.DataFrame, score_keys, unc_keys, sq_key, name_fn) -> list:
    out = []
    for i, row in df.iterrows():
        sid = str(row["stock_id"])
        item = {"rank": i + 1, "stock_id": sid, "name": name_fn(sid)}
        for k in score_keys:
            item[k] = round(float(row.get(k, 0)), 4)
        for k in unc_keys:
            item[k] = round(float(row.get(k, 0)), 4)
        item[sq_key] = round(float(row.get(sq_key, 0)), 3)
        out.append(item)
    return out


async def _load() -> Optional[dict]:
    df_s = await _fetch_csv(SHORT_URL)
    df_t = await _fetch_csv(TREND_URL)
    if df_s is None and df_t is None:
        return None

    # 股票名稱（沿用既有 stock_info）
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    try:
        from stock_info import get_stock_info, get_stock_name
        info = await get_stock_info()
        name_fn = lambda sid: get_stock_name(sid, info)
    except Exception:
        name_fn = lambda sid: sid

    date = None
    short, trend = [], []
    if df_s is not None and len(df_s):
        date = str(df_s["Date"].iloc[0]) if "Date" in df_s.columns else date
        short = _rows(df_s, ["Score_5d", "Score_10d"], ["Unc_5d", "Unc_10d"], "SQ_5d", name_fn)
    if df_t is not None and len(df_t):
        date = str(df_t["Date"].iloc[0]) if "Date" in df_t.columns else date
        trend = _rows(df_t, ["Score_20d", "Score_60d"], ["Unc_20d", "Unc_60d"], "SQ_20d", name_fn)

    logger.info(f"Loaded dual signals: short={len(short)} trend={len(trend)} ({date})")
    return {"date": date, "model_version": "V6.2-dual", "short": short, "trend": trend}


@router.get("/signals")
async def get_dual_signals(top: Optional[int] = 100):
    """雙模型訊號（短線 5d/10d + 趨勢 20d/60d），rank-score 語意，依 SQ 排序。"""
    global _cache, _cache_time
    data = None
    if _cache and _cache_time and datetime.now() - _cache_time < CACHE_TTL:
        data = _cache
    else:
        async with _cache_lock:
            if _cache and _cache_time and datetime.now() - _cache_time < CACHE_TTL:
                data = _cache
            else:
                data = await _load()
                if data:
                    _cache, _cache_time = data, datetime.now()

    if not data:
        return {"date": None, "model_version": "V6.2-dual", "short": [], "trend": [],
                "note": "雙模型推論資料尚未產生（run_dual_inference + push df_short/trend.csv 後出現）"}

    if top:
        data = {**data, "short": data["short"][:top], "trend": data["trend"][:top]}
    return data


@router.post("/cache/refresh")
async def refresh_dual_cache():
    """清除雙模型快取，下次請求會重新從 GitHub 抓 df_short/trend.csv。

    每日自動化在 push 完 df_short/trend.csv 後呼叫，讓前端立即看到當日資料
    （不必等 1h TTL 自然過期）。比照 signals.py 的 /cache/refresh。
    """
    global _cache, _cache_time
    async with _cache_lock:
        _cache = None
        _cache_time = None
    logger.info("Dual cache cleared via /dual/cache/refresh")
    return {"status": "ok", "message": "dual cache cleared"}
