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


async def _load_from_github() -> Optional[SignalsResponse]:
    """Fetch df_kelly.csv from GitHub raw URL and parse into SignalsResponse."""
    if not GITHUB_RESULTS_URL:
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(GITHUB_RESULTS_URL)
            if r.status_code != 200:
                logger.warning(f"GitHub fetch failed: {r.status_code}")
                return None

        df = pd.read_csv(io.StringIO(r.text))

        # df_kelly.csv columns: Ticker, Date, Exp_Alpha_5d, Exp_Alpha_20d,
        #   Exp_Alpha_60d, Uncertainty, Sharpe_Score, Confidence, Suggested_Weight, ...
        df = df.sort_values("Sharpe_Score", ascending=False).reset_index(drop=True)

        signals = []
        for i, row in df.iterrows():
            try:
                signal_str = "BUY" if row.get("Sharpe_Score", 0) > 0.5 else (
                    "SELL" if row.get("Sharpe_Score", 0) < -0.5 else "HOLD"
                )
                signals.append(SignalItem(
                    rank=i + 1,
                    stock_id=str(row["Ticker"]),
                    name=str(row.get("Name", row["Ticker"])),
                    sector=str(row.get("Sector", "未分類")),
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
    global _cache, _cache_time
    _cache = None
    _cache_time = None
    result = await _load_from_github()
    if result:
        return InferenceStatus(status="ok", message=f"Refreshed: {result.total_stocks} signals for {result.date}")
    return InferenceStatus(status="fallback", message="GitHub not available, using mock data")
