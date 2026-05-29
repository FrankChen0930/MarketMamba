"""
Market router — Real data sources
===================================
TAIEX : TWSE mis real-time API (gov, no IP restriction)
VIX / SPX / Gold : FRED CSV API (US Fed, open data)
USD/TWD : open.er-api.com (free, no auth)
advancing/declining + run_date : df_kelly.csv on GitHub
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

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketMamba/1.0)"}


async def _fetch_taiex() -> Tuple[float, float]:
    """Return (current_price, pct_change) from TWSE real-time index API."""
    import httpx
    url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=_HEADERS)
        if r.status_code == 200:
            msg = r.json()["msgArray"][0]
            price = float(msg.get("z") or 0)
            prev  = float(msg.get("y") or 0)
            pct   = (price / prev - 1) * 100 if prev else 0.0
            return price, pct
    except Exception as e:
        logger.warning(f"TWSE TAIEX: {e}")
    return 0.0, 0.0


async def _fred_latest_and_pct(series_id: str) -> Tuple[float, float]:
    """Return (latest_value, pct_change_from_prev) from FRED CSV API."""
    import httpx
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(url)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text), header=0, names=["date", "value"])
            df = df[df["value"] != "."].tail(5)
            if df.empty:
                return 0.0, 0.0
            latest = float(df["value"].iloc[-1])
            prev   = float(df["value"].iloc[-2]) if len(df) >= 2 else 0.0
            pct    = (latest / prev - 1) * 100 if prev else 0.0
            return latest, pct
    except Exception as e:
        logger.warning(f"FRED {series_id}: {e}")
    return 0.0, 0.0


async def _fetch_usd_twd() -> float:
    """Return USD/TWD rate from open.er-api.com (free, no auth)."""
    import httpx
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
        if r.status_code == 200:
            return float(r.json().get("rates", {}).get("TWD", 0))
    except Exception as e:
        logger.warning(f"ExchangeRate USD/TWD: {e}")
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
    # Parallel fetch all data sources
    (
        meta,
        (twii_price, twii_pct),
        (vix, _),
        (_, spx_pct),
        (_, gold_pct),
        usd_twd,
    ) = await asyncio.gather(
        _load_kelly_meta(),
        _fetch_taiex(),
        _fred_latest_and_pct("VIXCLS"),    # VIX
        _fred_latest_and_pct("SP500"),     # S&P 500
        _fred_latest_and_pct("GOLDAMGBD228NLBM"),  # Gold
        _fetch_usd_twd(),
    )

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

    twii_price, twii_pct = await _fetch_taiex()
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
