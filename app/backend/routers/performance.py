from fastapi import APIRouter
from schemas import PerformanceResponse
from mock_data import MOCK_PERFORMANCE

router = APIRouter(prefix="/performance", tags=["Performance"])


@router.get("", response_model=PerformanceResponse)
async def get_performance():
    """完整量化績效：IC 歷史 + Walk-Forward 結果 + 累積報酬"""
    return MOCK_PERFORMANCE


@router.get("/ic", response_model=list)
async def get_ic_history():
    """僅回傳 IC 歷史（Dashboard 小圖用）"""
    return [p.model_dump() for p in MOCK_PERFORMANCE.ic_history]
