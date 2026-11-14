"""
Market router — Real data sources
===================================
TAIEX    : TWSE mis API (gov) → Yahoo Finance v8 fallback
SPX/Gold : Yahoo Finance v8 JSON API (direct HTTP, no library)
VIX      : FRED CSV API
USD/TWD, JPY/TWD : open.er-api.com
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

# K-line cache: {f"{ticker}:{range}": (timestamp, payload)}
_kline_cache: dict = {}
_KLINE_TTL = timedelta(minutes=15)
_KLINE_RANGES = {"3mo", "6mo", "1y", "2y"}

_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


# ── TAIEX (TWSE real-time → Yahoo Finance v8 fallback) ────────────────────────

async def _fetch_taiex() -> Tuple[float, float]:
    """Return (current_price, pct_change). Tries TWSE first, then Yahoo v8."""
    import httpx

    # Primary: TWSE real-time index API
    try:
        url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1&delay=0"
        headers = {
            "User-Agent": _BROWSER_UA,
            "Referer": "https://www.twse.com.tw",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, headers=headers)
        if r.status_code == 200:
            msg = r.json().get("msgArray", [{}])[0]
            z = msg.get("z", "-")
            y = msg.get("y", "0")
            # z can be "-" when market is closed → fall back to yesterday's close
            price = float(z) if z and z != "-" else float(y or 0)
            prev  = float(y or 0)
            if price and prev:
                pct = (price / prev - 1) * 100
                return price, pct
    except Exception as e:
        logger.warning(f"TWSE TAIEX: {e}")

    # Fallback: Yahoo Finance v8 JSON API
    return await _yf_v8("%5ETWII")


# ── Yahoo Finance v8 JSON API (direct HTTP, no yfinance library) ───────────────

async def _yf_v8(symbol: str) -> Tuple[float, float]:
    """Return (price, pct_change) from Yahoo Finance v8 chart API."""
    import httpx
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
        if r.status_code == 200:
            meta  = r.json()["chart"]["result"][0]["meta"]
            price = float(meta.get("regularMarketPrice") or 0)
            prev  = float(meta.get("chartPreviousClose") or 0)
            pct   = (price / prev - 1) * 100 if prev else 0.0
            return price, pct
    except Exception as e:
        logger.warning(f"Yahoo v8 {symbol}: {e}")
    return 0.0, 0.0



# ── Yahoo Finance v8 OHLCV (daily K-line for the frontend chart) ───────────────

async def _yf_v8_ohlcv(symbol: str, range_: str) -> list:
    """Return daily OHLCV candles from Yahoo Finance v8 chart API.

    Each candle: {date, open, high, low, close, volume}. Empty list on failure.
    """
    import httpx
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range={range_}"
    )
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
        if r.status_code != 200:
            return []
        result = r.json()["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        candles = []
        for i, ts in enumerate(timestamps):
            o, h, l, c = (quote["open"][i], quote["high"][i],
                          quote["low"][i], quote["close"][i])
            if c is None or o is None:
                continue   # 停牌 / 缺值日
            candles.append({
                "date":   datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                "open":   round(float(o), 2),
                "high":   round(float(h), 2),
                "low":    round(float(l), 2),
                "close":  round(float(c), 2),
                "volume": int(quote["volume"][i] or 0),
            })
        return candles
    except Exception as e:
        logger.warning(f"Yahoo v8 OHLCV {symbol}: {e}")
        return []


# ── Exchange rates (USD/TWD + JPY/TWD from single call) ──────────────────────

async def _fetch_fx() -> Tuple[float, float]:
    """Return (usd_twd, jpy_twd) from open.er-api.com. jpy_twd = 100 JPY in TWD."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://open.er-api.com/v6/latest/USD")
        if r.status_code == 200:
            rates   = r.json().get("rates", {})
            usd_twd = float(rates.get("TWD", 0))
            jpy_usd = float(rates.get("JPY", 0))
            jpy_twd = round(usd_twd / jpy_usd * 100, 3) if jpy_usd else 0.0
            return usd_twd, jpy_twd
    except Exception as e:
        logger.warning(f"ExchangeRate FX: {e}")
    return 0.0, 0.0


# ── Kelly metadata from GitHub ────────────────────────────────────────────────

async def _load_kelly_meta() -> dict:
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
        return {"date": date_str, "advancing": advancing, "declining": declining}
    except Exception as e:
        logger.warning(f"Kelly meta: {e}")
        return {}


# ── Build market response ─────────────────────────────────────────────────────

async def _build_market() -> MarketStatusResponse:
    (
        meta,
        (twii_price, twii_pct),
        (vix, _),
        (_, spx_pct),
        (_, gold_pct),
        (usd_twd, jpy_twd),
    ) = await asyncio.gather(
        _load_kelly_meta(),
        _fetch_taiex(),
        _yf_v8("%5EVIX"),       # VIX — Yahoo Finance v8 (FRED fails on Render)
        _yf_v8("%5EGSPC"),      # S&P 500
        _yf_v8("GC%3DF"),       # Gold futures
        _fetch_fx(),
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
        jpy_twd=round(jpy_twd, 3),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=MarketStatusResponse)
async def get_market_status():
    """大盤狀態：TAIEX 即時 + 宏觀指標 + 模型資訊"""
    global _market_cache, _market_cache_time
    if _market_cache and _market_cache_time and datetime.now() - _market_cache_time < _MARKET_TTL:
        return _market_cache
    _market_cache = await _build_market()
    _market_cache_time = datetime.now()
    return _market_cache


@router.get("/kline/{ticker}")
async def get_kline(ticker: str, range: str = "6mo"):
    """個股日 K 線（OHLCV）— Yahoo v8 proxy，上市 .TW 查不到 fallback 上櫃 .TWO。

    回傳：{ticker, symbol, range, candles: [{date, open, high, low, close, volume}]}
    per-ticker 15 分鐘記憶體快取，避免 Yahoo rate limit。
    """
    if range not in _KLINE_RANGES:
        range = "6mo"
    ticker = ticker.strip()
    cache_key = f"{ticker}:{range}"

    cached = _kline_cache.get(cache_key)
    if cached and datetime.now() - cached[0] < _KLINE_TTL:
        return cached[1]

    symbol = f"{ticker}.TW"
    candles = await _yf_v8_ohlcv(symbol, range)
    if not candles:
        symbol = f"{ticker}.TWO"
        candles = await _yf_v8_ohlcv(symbol, range)

    payload = {
        "ticker":  ticker,
        "symbol":  symbol if candles else None,
        "range":   range,
        "candles": candles,
    }
    if candles:   # 查無資料不快取，讓下次請求重試
        _kline_cache[cache_key] = (datetime.now(), payload)
    return payload


@router.get("/ticker", response_model=TickerResponse)
async def get_ticker():
    """Ticker bar — TAIEX + top stocks from signals"""
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
                tickers = [str(row["Ticker"]) for _, row in df.iterrows()]

                async def _fetch_stock_quote(ticker: str) -> Tuple[float, float]:
                    """嘗試上市（.TW），若查不到再試上櫃（.TWO）"""
                    price, pct = await _yf_v8(f"{ticker}.TW")
                    if not price:
                        price, pct = await _yf_v8(f"{ticker}.TWO")
                    return price, pct

                quotes = await asyncio.gather(*[_fetch_stock_quote(t) for t in tickers])

                for ticker, (price, pct) in zip(tickers, quotes):
                    if price:
                        items.append(TickerItem(
                            id=ticker,
                            name=get_stock_name(ticker, info),
                            price=f"{price:,.2f}",
                            change=f"{price * pct / 100:+.2f}",
                            pct=f"{pct:+.2f}%",
                            up=pct >= 0,
                        ))
                    else:
                        items.append(TickerItem(
                            id=ticker,
                            name=get_stock_name(ticker, info),
                            price="—", change="—", pct="—", up=True,
                        ))
        except Exception as e:
            logger.warning(f"Ticker build failed: {e}")

    _ticker_cache = TickerResponse(items=items)
    _ticker_cache_time = datetime.now()
    return _ticker_cache
