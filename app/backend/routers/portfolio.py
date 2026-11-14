"""
Portfolio Router — 持倉資料（Supabase / Shioaji）
"""
import os
import logging
from fastapi import APIRouter
from schemas import PortfolioItem, PortfolioResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

EMPTY_PORTFOLIO = PortfolioResponse(
    total_pnl=0, total_value=0, positions=[],
    data_source="shioaji", last_updated="—",
)


async def _fetch_from_supabase() -> PortfolioResponse:
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL / SUPABASE_KEY not set")
        return EMPTY_PORTFOLIO

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
    except Exception as e:
        logger.warning(f"Supabase fetch failed: {e}")
        return EMPTY_PORTFOLIO

    if not rows:
        return EMPTY_PORTFOLIO

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
        pnl = (current - cost) * qty * 1000
        pnl_pct = float(r.get("pnl_pct", 0))
        market_val = float(r.get("market_value", 0))

        total_pnl += pnl
        total_value += market_val
        if r.get("synced_at", "") > last_updated:
            last_updated = r["synced_at"]

        positions.append(PortfolioItem(
            stock_id=stock_id, name=name, qty=qty,
            avg_price=cost, current_price=current,
            pnl=pnl, pnl_pct=pnl_pct, model_signal="N/A",
        ))

    return PortfolioResponse(
        total_pnl=total_pnl, total_value=total_value,
        positions=positions, data_source="shioaji",
        last_updated=last_updated[:19].replace("T", " ") if last_updated else "—",
    )


@router.get("", response_model=PortfolioResponse)
async def get_portfolio():
    """持倉列表 — 從 Supabase 讀 shioaji 同步的真實持倉"""
    return await _fetch_from_supabase()
