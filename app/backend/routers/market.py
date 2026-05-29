"""
Market router — Real data sources
===================================
TAIEX + macro data from Yahoo Finance v8 JSON API (direct HTTP, no yfinance library)
VIX fallback: FRED CSV API
run_status + date derived from df_kelly.csv on GitHub
"""
import asyncio
import io
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
from fastapi import APIRouter

from schemas import MarketStatusResponse, TaiexStatus, TickerResponse, TickerItem

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["Market"])

GITHUB_RESULTS_URL = os.getenv("GITHUB_RESULTS_URL", "")

# ── In-memory cache (5-min TTL for market data) ───────────────────────────────
_market_cache: Optional[MarketStatusResponse] = None
_market_cache_time: Optional[datetime] = None
_MARKET_TTL = timedelta(minutes=5)

_ticker_cache: Optional[TickerResponse] = None
_ticker_cache_time: Optional[datetime] = None

_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}
_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"


async def _yf_price_and_pct(symbol: str) -> Tuple[float, float]:
    """Fetch regularMarketPrice and % change from Yahoo Finance v8 JSON API."""
    import httpx
    url = f"{_YF_BASE}/{symbol}?interval=1d&range=5d"
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url, headers=_YF_HEADERS)
        if r.status_code == 200:
            meta = r.json()["chart"]["result"][0]["meta"]
            price = float(meta.get("regularMarketPrice") or 0)
            prev  = float(meta.get("chartPreviousClose") or 0)
            pct   = (price / prev - 1) * 100 if prev else 0.0
            return price, pct
    except Exception as e:
        logger.warning(f"yf_v8 {symbol}: {e}")
    return 0.0, 0.0


async def _fred_vix() -> float:
    """Fetch latest VIX closing price from FRED CSV (free, no auth)."""
    import httpx
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            df = df[df["value"] != "."].tail(5)
            return float(df["value"].iloc[-1])
    except Exception as e:
        logger.warning(f"FRED VIX: {e}")
    return 0.0


async def _load_kelly_meta() -> dict:
    """Read run date and signal stats from GitHub df_kelly.csv."""
    if not GITHUB_RESULTS_URL:
        return {}
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(GITHUB_RESULTS_URL)
            if r.status_code != 200:
                return {}
        df = pd.read_csv(io.StringIO(r.text))
        date_str  = str(df["Date"].iloc[0]) if "Date" in df.columns else ""
        advancing = int((df["Exp_Alpha_20d"] > 0).sum()) if "Exp_Alpha_20d" in df.columns else 0
        declining = int((df["Exp_Alpha_20d"] <= 0).sum()) if "Exp_Alpha_20d" in df.columns else 0
        return {"date": date_str, "advancing": advancing, "declining": declining, "total": len(df)}
    except Exception as e:
        logger.warning(f"Kelly meta load failed: {e}")
        return {}


async def _build_market() -> MarketStatusResponse:
    # Parallel fetch: Kelly meta + all market prices
    meta, (twii_price, twii_pct), (vix_price, _), (_, spx_pct), (_, gold_pct), (usd_twd, _) = \
        await asyncio.gather(
            _load_kelly_meta(),
            _yf_price_and_pct("%5ETWII"),   # TAIEX
            _yf_price_and_pct("%5EVIX"),    # VIX (price only)
            _yf_price_and_pct("%5EGSPC"),   # S&P 500
            _yf_price_and_pct("GC%3DF"),    # Gold futures
            _yf_price_and_pct("TWD%3DX"),   # USD/TWD
        )

    # FRED as VIX fallback if Yahoo returned 0
    vix = vix_price if vix_price else await _fred_vix()

    run_date    = meta.get("date", datetime.today().strftime("%Y-%m-%d"))
    twii_change = round(twii_price * twii_pct / 100, 2) if twii_price else 0.0

    return MarketStatusResponse(
        taiex=TaiexStatus(
            value=round(twii_price, 1),
            change=twii_change,
            change_pct=round(twii_pct, 2),
        ),
        advancing=meta.get("advancing", 0),
        declining=meta.get("declining", 0),
        model_ic=0.1235,
        last_run=run_date,
        run_status="completed" if run_date else "not_ready",
        training_epoch=None,
        training_status="completed",
        spx_change=round(spx_pct, 2),
        vix=round(vix, 2),
        gold_change=round(gold_pct, 2),
        usd_twd=round(usd_twd, 3),
    )


@router.get("", response_model=MarketStatusResponse)
async def get_market_status():
    """大盤狀態：TAIEX 即時 + 宏觀指標 + 模型資訊"""
    global _market_cache, _market_cache_time
    if _market_cache and _market_cache_time and datetime.now() - _market_cache_time < _MARKET_TTL:
        return _market_cache
    _market_cache = await _build_market()
    _market_cache_time = datetime.now()
    return _market_cache


@router.get("/ticker", response_model=TickerResponse)
async def get_ticker():
    """Ticker bar — top 8 stocks from signals + TAIEX"""
    global _ticker_cache, _ticker_cache_time
    if _ticker_cache and _ticker_cache_time and datetime.now() - _ticker_cache_time < timedelta(minutes=5):
        return _ticker_cache

    twii_price, twii_pct = await _yf_price_and_pct("%5ETWII")
    items = [TickerItem(
        id="TAIEX", name="加權",
        price=f"{twii_price:,.0f}" if twii_price else "—",
        change=f"{twii_price * twii_pct / 100:+.1f}" if twii_price else "—",
        pct=f"{twii_pct:+.2f}%" if twii_pct else "—",
        up=twii_pct >= 0,
    )]

    # Top stocks from GitHub signals
    if GITHUB_RESULTS_URL:
        try:
            import httpx, sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
            from stock_info import get_stock_info, get_stock_name
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(GITHUB_RESULTS_URL)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                df = df.sort_values("Signal_Quality", ascending=False).head(7)
                info = await get_stock_info()
                for _, row in df.iterrows():
                    ticker = str(row["Ticker"])
                    items.append(TickerItem(
                        id=ticker,
                        name=get_stock_name(ticker, info),
                        price="—",
                        change="—",
                        pct="—",
                        up=True,
                    ))
        except Exception as e:
            logger.warning(f"Ticker build failed: {e}")

    _ticker_cache = TickerResponse(items=items)
    _ticker_cache_time = datetime.now()
    return _ticker_cache
