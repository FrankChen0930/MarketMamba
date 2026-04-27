from fastapi import APIRouter
from schemas import PortfolioResponse
from mock_data import MOCK_PORTFOLIO

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])


@router.get("", response_model=PortfolioResponse)
async def get_portfolio():
    """持倉列表（mock — 之後接永豐 shioaji）"""
    return MOCK_PORTFOLIO
