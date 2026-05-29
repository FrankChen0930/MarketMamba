"""
Market router — Real data sources
===================================
TAIEX + macro data from stooq.com (free, no auth required, Render-friendly)
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

# stooq symbol map
_STOOQ = {
    "twii":   "^twii",     # TAIEX 加權指數
    "vix":    "^vix",      # VIX 恐慌指數
    "spx":    "^spx",      # S&P 500
    "gold":   "xauusd",    # 黃金現貨 (USD/oz)
    "usdtwd": "usdtwd",    # USD/TWD 匯率
}


async def _stooq_df(symbol: str) -> pd.DataFrame:
    """Fetch last 5 daily OHLC rows from stooq.com. Returns empty DF on failure."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "Date" in r.text:
            df = pd.read_csv(io.StringIO(r.text))
            return df.dropna(subset=["Close"]).tail(5)
    except Exception as e:
        logger.warning(f"stooq {symbol}: {e}")
    return pd.DataFrame()


async def _stooq_price(symbol: str) -> float:
    df = await _stooq_df(symbol)
    return float(df["Close"].iloc[-1]) if not df.empty else 0.0


async def _stooq_price_and_pct(symbol: str) -> Tuple[float, float]:
    df = await _stooq_df(symbol)
    if df.empty:
        return 0.0, 0.0
    price = float(df["Close"].iloc[-1])
    pct = (price / float(df["Close"].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0.0
    return price, pct


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
        date_str = str(df["Date"].iloc[0]) if "Date" in df.columns else ""
        advancing = int((df["Exp_Alpha_20d"] > 0).sum()) if "Exp_Alpha_20d" in df.columns else 0
        declining = int((df["Exp_Alpha_20d"] <= 0).sum()) if "Exp_Alpha_20d" in df.columns else 0
        return {"date": date_str, "advancing": advancing, "declining": declining, "total": len(df)}
    except Exception as e:
        logger.warning(f"Kelly meta load failed: {e}")
        return {}


async def _build_market() -> MarketStatusResponse:
    # Parallel fetch: Kelly meta + all stooq prices
    meta, (twii_price, twii_pct), vix, (_, spx_pct), (_, gold_pct), usd_twd = await asyncio.gather(
        _load_kelly_meta(),
        _stooq_price_and_pct(_STOOQ["twii"]),
        _stooq_price(_STOOQ["vix"]),
        _stooq_price_and_pct(_STOOQ["spx"]),
        _stooq_price_and_pct(_STOOQ["gold"]),
        _stooq_price(_STOOQ["usdtwd"]),
    )

    run_date = meta.get("date", datetime.today().strftime("%Y-%m-%d"))
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

    twii_price, twii_pct = await _stooq_price_and_pct(_STOOQ["twii"])
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
