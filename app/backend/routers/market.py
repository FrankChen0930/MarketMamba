from fastapi import APIRouter
from schemas import MarketStatusResponse, TickerResponse
from mock_data import MOCK_MARKET, MOCK_TICKER

router = APIRouter(prefix="/market", tags=["Market"])


@router.get("", response_model=MarketStatusResponse)
async def get_market_status():
    """大盤狀態：TAIEX、漲跌家數、model IC、訓練進度"""
    return MOCK_MARKET


@router.get("/ticker", response_model=TickerResponse)
async def get_ticker():
    """Ticker bar 用的股票跑馬燈資料"""
    return MOCK_TICKER
