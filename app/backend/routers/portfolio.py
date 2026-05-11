"""
Portfolio Router — 持倉資料
==============================
優先從 Supabase 讀 shioaji 同步的真實持倉。
若 Supabase 無資料或未設定，降級為 mock data。
"""
import os
import logging
from datetime import datetime
from fastapi import APIRouter
from schemas import PortfolioItem, PortfolioResponse
from mock_data import MOCK_PORTFOLIO

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


async def _fetch_from_supabase() -> PortfolioResponse | None:
    """Fetch real positions from Supabase portfolio_positions table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None

    import httpx
    url = f"{SUPABASE_URL}/rest/v1/portfolio_positions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    params = {"select": "*", "order": "market_value.desc"}

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            rows = resp.json()

        if not rows:
            return None

        positions = []
        total_pnl = 0.0
        total_value = 0.0
        last_updated = ""

        for r in rows:
            stock_id = r.get("stock_id", "")
            name = r.get("name", stock_id)
            qty = int(r.get("quantity", 0))
            cost = float(r.get("cost_price", 0))
            current = float(r.get("current_price", cost))
            pnl = (current - cost) * qty * 1000  # 1張=1000股
            pnl_pct = float(r.get("pnl_pct", 0))
            market_val = float(r.get("market_value", 0))

            total_pnl += pnl
            total_value += market_val
            if r.get("synced_at", "") > last_updated:
                last_updated = r["synced_at"]

            positions.append(PortfolioItem(
                stock_id=stock_id,
                name=name,
                qty=qty,
                avg_price=cost,
                current_price=current,
                pnl=pnl,
                pnl_pct=pnl_pct,
                model_signal="N/A",  # will be enriched if needed
            ))

        return PortfolioResponse(
            total_pnl=total_pnl,
            total_value=total_value,
            positions=positions,
            data_source="shioaji",
            last_updated=last_updated[:19].replace("T", " ") if last_updated else "—",
        )

    except Exception as e:
        logger.warning(f"Supabase fetch failed, falling back to mock: {e}")
        return None


@router.get("", response_model=PortfolioResponse)
async def get_portfolio():
    """持倉列表 — 優先 Supabase（shioaji），降級 mock"""
    real = await _fetch_from_supabase()
    if real and real.positions:
        return real
    return MOCK_PORTFOLIO
