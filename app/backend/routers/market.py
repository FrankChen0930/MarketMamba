"""
Market router — Real data sources
===================================
TAIEX + macro data from yfinance (^TWII, ^GSPC, ^VIX, GC=F, TWD=X)
run_status + date derived from df_kelly.csv on GitHub
"""
import io
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

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


def _safe_yf_pct(ticker: str) -> float:
    """Get today's % change for a yfinance ticker. Returns 0.0 on failure."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="2d", auto_adjust=True, progress=False, timeout=8)
        if len(df) >= 2:
            return float((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100)
    except Exception:
        pass
    return 0.0


def _safe_yf_price(ticker: str) -> float:
    """Get latest closing price for a yfinance ticker."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period="2d", auto_adjust=True, progress=False, timeout=8)
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
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
        date_str = str(df["Date"].iloc[0]) if "Date" in df.columns else ""
        advancing = int((df["Exp_Alpha_20d"] > 0).sum()) if "Exp_Alpha_20d" in df.columns else 0
        declining = int((df["Exp_Alpha_20d"] <= 0).sum()) if "Exp_Alpha_20d" in df.columns else 0
        return {"date": date_str, "advancing": advancing, "declining": declining, "total": len(df)}
    except Exception as e:
        logger.warning(f"Kelly meta load failed: {e}")
        return {}


async def _build_market() -> MarketStatusResponse:
    meta = await _load_kelly_meta()
    run_date = meta.get("date", datetime.today().strftime("%Y-%m-%d"))

    # TAIEX real-time
    twii_price = _safe_yf_price("^TWII")
    twii_pct   = _safe_yf_pct("^TWII")
    twii_change = round(twii_price * twii_pct / 100, 2) if twii_price else 0.0

    # Macro
    spx_pct  = _safe_yf_pct("^GSPC")
    vix      = _safe_yf_price("^VIX")
    gold_pct = _safe_yf_pct("GC=F")
    usd_twd  = _safe_yf_price("TWD=X")

    return MarketStatusResponse(
        taiex=TaiexStatus(
            value=round(twii_price, 1) if twii_price else 0.0,
            change=twii_change,
            change_pct=round(twii_pct, 2),
        ),
        advancing=meta.get("advancing", 0),
        declining=meta.get("declining", 0),
        model_ic=0.1235,          # Best WF fold IC (F03, real data)
        last_run=run_date,
        run_status="completed" if run_date else "not_ready",
        training_epoch=None,
        training_status="completed",
        # Macro fields
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

    items = []
    # TAIEX first
    twii_price = _safe_yf_price("^TWII")
    twii_pct   = _safe_yf_pct("^TWII")
    items.append(TickerItem(
        id="TAIEX", name="加權",
        price=f"{twii_price:,.0f}" if twii_price else "—",
        change=f"{twii_price * twii_pct / 100:+.1f}" if twii_price else "—",
        pct=f"{twii_pct:+.2f}%" if twii_pct else "—",
        up=twii_pct >= 0,
    ))

    # Try to grab top stocks from GitHub signals
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
                    pct = _safe_yf_pct(f"{ticker}.TW")
                    items.append(TickerItem(
                        id=ticker,
                        name=get_stock_name(ticker, info),
                        price="—",
                        change=f"{pct:+.2f}%" if pct != 0 else "—",
                        pct=f"{pct:+.2f}%" if pct != 0 else "—",
                        up=pct >= 0,
                    ))
        except Exception as e:
            logger.warning(f"Ticker build failed: {e}")


    _ticker_cache = TickerResponse(items=items)
    _ticker_cache_time = datetime.now()
    return _ticker_cache
