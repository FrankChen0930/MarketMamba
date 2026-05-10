"""
Signals router
==============
Data source priority:
  1. GitHub raw URL (df_kelly.csv) — written by PersonalOS after daily inference
  2. Mock data fallback — used while model is still training

PersonalOS workflow:
  run_daily_inference.py → saves V6/results/df_kelly.csv → git push → backend reads here
"""
import os
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Query

from schemas import SignalItem, SignalsResponse, InferenceStatus
from mock_data import MOCK_SIGNALS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/signals", tags=["Signals"])

# ── GitHub Results Config ─────────────────────────────────────────────────────
# Set GITHUB_RESULTS_URL in Render env vars to enable live data.
# Example: https://raw.githubusercontent.com/FrankChen0930/MarketMamba/main/V6/results/df_kelly.csv
GITHUB_RESULTS_URL = os.getenv("GITHUB_RESULTS_URL", "")
CACHE_TTL = timedelta(hours=1)

_cache: Optional[SignalsResponse] = None
_cache_time: Optional[datetime] = None

# History index cache
_history_cache: Optional[dict] = None
_history_cache_time: Optional[datetime] = None

GITHUB_HISTORY_URL = GITHUB_RESULTS_URL.replace("df_kelly.csv", "history_index.json") if GITHUB_RESULTS_URL else ""


async def _load_from_github() -> Optional[SignalsResponse]:
    """Fetch df_kelly.csv from GitHub raw URL and parse into SignalsResponse."""
    if not GITHUB_RESULTS_URL:
        return None
    try:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        import httpx
        from stock_info import get_stock_info, get_stock_name, get_stock_sector


        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(GITHUB_RESULTS_URL)
            if r.status_code != 200:
                logger.warning(f"GitHub fetch failed: {r.status_code}")
                return None

        df = pd.read_csv(io.StringIO(r.text))
        df = df.sort_values("Sharpe_Score", ascending=False).reset_index(drop=True)

        # Load name + sector lookup (cached 24h)
        info = await get_stock_info()

        signals = []
        for i, row in df.iterrows():
            try:
                ticker = str(row["Ticker"])
                signal_str = "BUY" if row.get("Sharpe_Score", 0) > 0.5 else (
                    "SELL" if row.get("Sharpe_Score", 0) < -0.5 else "HOLD"
                )
                signals.append(SignalItem(
                    rank=i + 1,
                    stock_id=ticker,
                    name=get_stock_name(ticker, info),
                    sector=get_stock_sector(ticker, info),
                    alpha_5d=float(row.get("Exp_Alpha_5d", 0)),
                    alpha_20d=float(row.get("Exp_Alpha_20d", 0)),
                    alpha_60d=float(row.get("Exp_Alpha_60d", 0)),
                    uncertainty=float(row.get("Uncertainty", 0)),
                    vol_ratio=float(row.get("vol_ratio", 1.0)),
                    signal=signal_str,
                    suggested_weight=float(row.get("Suggested_Weight", 0)),
                    confidence=str(row.get("Confidence", "中信心")),
                ))
            except Exception as e:
                logger.debug(f"Skipping row {i}: {e}")

        date_str = str(df["Date"].iloc[0]) if "Date" in df.columns else datetime.today().strftime("%Y-%m-%d")
        logger.info(f"Loaded {len(signals)} signals from GitHub ({date_str})")
        return SignalsResponse(
            date=date_str,
            model_version="V6",
            total_stocks=len(signals),
            signals=signals,
        )

    except Exception as e:
        logger.error(f"Failed to load from GitHub: {e}")
        return None



async def _get_signals() -> SignalsResponse:
    """Return signals with cache. Falls back to mock data."""
    global _cache, _cache_time

    # Return cache if still fresh
    if _cache and _cache_time and datetime.now() - _cache_time < CACHE_TTL:
        return _cache

    # Try GitHub
    result = await _load_from_github()
    if result:
        _cache = result
        _cache_time = datetime.now()
        return _cache

    # Fallback: mock data (training in progress)
    return MOCK_SIGNALS


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=SignalsResponse)
async def get_signals(
    top: Optional[int] = Query(None, description="Return top-N stocks only"),
    signal: Optional[str] = Query(None, description="Filter: BUY / HOLD / SELL"),
):
    """今日選股訊號 — 依 Alpha_20d / Sharpe 排序"""
    data = await _get_signals()
    signals = data.signals

    if signal:
        signals = [s for s in signals if s.signal == signal.upper()]
    if top:
        signals = signals[:top]

    return SignalsResponse(
        date=data.date,
        model_version=data.model_version,
        total_stocks=data.total_stocks,
        signals=signals,
    )


@router.get("/scanner")
async def get_scanner_signals():
    """交易訊號掃描 — 入場/退場/觀察信號"""
    global _scanner_cache, _scanner_cache_time

    if (_scanner_cache and _scanner_cache_time
            and datetime.now() - _scanner_cache_time < CACHE_TTL):
        return _scanner_cache

    data = None

    # Try GitHub first (production)
    if GITHUB_SCANNER_URL:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(GITHUB_SCANNER_URL)
                if r.status_code == 200:
                    data = r.json()
                    logger.info(f"Scanner signals loaded from GitHub: {len(data.get('buy_signals',[]))} BUY")
                else:
                    logger.warning(f"action_signals.json fetch failed: {r.status_code}")
        except Exception as e:
            logger.warning(f"Scanner GitHub load error: {e}")

    # Fallback: local file (dev mode)
    if data is None and _LOCAL_SCANNER_PATH.exists():
        try:
            import json as _json
            with open(_LOCAL_SCANNER_PATH, encoding="utf-8") as f:
                data = _json.load(f)
            logger.info(f"Scanner signals loaded from local: {len(data.get('buy_signals',[]))} BUY")
        except Exception as e:
            logger.warning(f"Scanner local load error: {e}")

    if data is None:
        return {"buy_signals": [], "exit_signals": [], "watch_list": [],
                "date": None, "market_regime": "UNKNOWN"}

    # Enrich with stock names
    data = await _enrich_scanner_names(data)
    _scanner_cache = data
    _scanner_cache_time = datetime.now()
    return data


async def _enrich_scanner_names(data: dict) -> dict:
    """Add stock names to scanner signal entries."""
    try:
        from stock_info import get_stock_info, get_stock_name, get_stock_sector
        info = await get_stock_info()

        for section in ("buy_signals", "exit_signals", "watch_list"):
            for item in data.get(section, []):
                ticker = item.get("ticker", "")
                if ticker and "name" not in item:
                    item["name"] = get_stock_name(ticker, info)
                    item["sector"] = get_stock_sector(ticker, info)
    except Exception as e:
        logger.warning(f"Scanner name enrichment failed (non-fatal): {e}")
    return data


@router.get("/{date}", response_model=SignalsResponse)
async def get_signals_by_date(date: str):
    """指定日期訊號（目前只保存最新一次）"""
    return await _get_signals()


@router.post("/run-inference", response_model=InferenceStatus)
async def run_inference():
    """觸發推論 — 由 PersonalOS 在本機執行，此端點只供狀態查詢"""
    return InferenceStatus(
        status="local_only",
        message="Inference runs locally via PersonalOS (RTX 3060). "
                "Results are pushed to GitHub and auto-loaded here.",
    )


@router.post("/cache/refresh", response_model=InferenceStatus)
async def refresh_cache():
    """強制重新從 GitHub 載入最新結果（PersonalOS push 完後呼叫）"""
    global _cache, _cache_time, _history_cache, _history_cache_time
    _cache = None
    _cache_time = None
    _history_cache = None
    _history_cache_time = None
    result = await _load_from_github()
    if result:
        return InferenceStatus(status="ok", message=f"Refreshed: {result.total_stocks} signals for {result.date}")
    return InferenceStatus(status="fallback", message="GitHub not available, using mock data")


@router.get("/history")
async def get_rebalance_history():
    """調倉紀錄 — 從 GitHub history_index.json 讀取過去每日 rebalancing 結果"""
    global _history_cache, _history_cache_time

    # Serve from cache
    if (_history_cache and _history_cache_time
            and datetime.now() - _history_cache_time < CACHE_TTL):
        return _history_cache

    if not GITHUB_HISTORY_URL:
        return {"history": [], "note": "GITHUB_RESULTS_URL not configured"}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(GITHUB_HISTORY_URL)
            if r.status_code == 200:
                data = r.json()
                _history_cache = data
                _history_cache_time = datetime.now()
                logger.info(f"History index loaded: {len(data.get('history',[]))} entries")
                return data
            logger.warning(f"history_index.json fetch failed: {r.status_code}")
    except Exception as e:
        logger.warning(f"History load error: {e}")

    return {"history": [], "last_updated": None}


# ── Scanner Signals ────────────────────────────────────────────────────────────

GITHUB_SCANNER_URL = GITHUB_RESULTS_URL.replace("df_kelly.csv", "action_signals.json") if GITHUB_RESULTS_URL else ""

# Local fallback for dev mode
import pathlib as _pathlib
_LOCAL_SCANNER_PATH = _pathlib.Path(__file__).resolve().parent.parent.parent.parent / "V6" / "results" / "action_signals.json"

_scanner_cache: Optional[dict] = None
_scanner_cache_time: Optional[datetime] = None


