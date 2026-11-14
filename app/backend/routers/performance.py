"""
Performance router — 真實訓練/線上 IC 資料
============================================
資料來源（GitHub raw，與 signals.py 同一機制）：
  - V6/results/training_status.json : Colab 訓練逐 epoch 記錄（學習曲線、scale_gates、config）
    （訓練完成後由使用者從 Drive 複製進 repo 並 push）
  - V6/results/ic_analysis.json     : 每日推論的線上 IC 統計（已存在的真實資料）

無資料時回傳空欄位，前端顯示「尚無訓練紀錄」。不再使用任何寫死/合成資料。
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter
from schemas import (
    ICPoint,
    OnlineICPoint,
    OnlineICSummary,
    PerformanceResponse,
    ScaleGatePoint,
    TrainingInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/performance", tags=["Performance"])

GITHUB_RESULTS_URL = os.getenv("GITHUB_RESULTS_URL", "")  # 指向 df_kelly.csv
TRAINING_STATUS_URL = (
    GITHUB_RESULTS_URL.replace("df_kelly.csv", "training_status.json")
    if GITHUB_RESULTS_URL else ""
)
IC_ANALYSIS_URL = (
    GITHUB_RESULTS_URL.replace("df_kelly.csv", "ic_analysis.json")
    if GITHUB_RESULTS_URL else ""
)

CACHE_TTL = timedelta(minutes=30)
_cache: Optional[PerformanceResponse] = None
_cache_time: Optional[datetime] = None
_cache_lock: asyncio.Lock = asyncio.Lock()


async def _fetch_json(url: str) -> Optional[dict]:
    if not url:
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code != 200:
                logger.warning(f"GitHub fetch failed ({r.status_code}): {url}")
                return None
            return r.json()
    except Exception as e:
        logger.error(f"Fetch error {url}: {e}")
        return None


def _parse_training(ts: Optional[dict]) -> tuple[Optional[TrainingInfo], list, list]:
    """training_status.json → (TrainingInfo, ic_history, scale_gates)"""
    if not ts:
        return None, [], []

    info = TrainingInfo(
        model_version=ts.get("model_version", "V6"),
        status=ts.get("status", "unknown"),
        started_at=ts.get("started_at"),
        updated_at=ts.get("updated_at"),
        epoch=int(ts.get("epoch", 0)),
        epochs_max=int(ts.get("epochs_max", 0)),
        best_val_ic=ts.get("best_val_ic"),
        best_ic_epoch=int(ts.get("best_ic_epoch", 0)),
        best_val_loss=ts.get("best_val_loss"),
        early_stop_patience=ts.get("early_stop_patience"),
        config=ts.get("config", {}) or {},
    )

    h = ts.get("history", {}) or {}
    tl, vl, vic = h.get("train_loss", []), h.get("val_loss", []), h.get("val_ic", [])
    n = min(len(tl), len(vl), len(vic))
    ic_history = [
        ICPoint(epoch=i + 1, train_loss=float(tl[i]), val_loss=float(vl[i]), val_ic=float(vic[i]))
        for i in range(n)
    ]

    scale_gates = [
        ScaleGatePoint(epoch=i + 1, short=float(g[0]), mid=float(g[1]), long=float(g[2]))
        for i, g in enumerate(h.get("scale_gates", []))
        if isinstance(g, (list, tuple)) and len(g) == 3
    ]
    return info, ic_history, scale_gates


def _parse_online_ic(ia: Optional[dict]) -> tuple[list, list, Optional[str]]:
    """ic_analysis.json → (online_ic 5d 時序, summary 列表, period 字串)"""
    if not ia:
        return [], [], None

    online_ic = [
        OnlineICPoint(
            pred_date=str(p.get("pred_date", "")),
            future_date=str(p.get("future_date", "")),
            ic=float(p.get("ic", 0.0)),
            n_stocks=int(p.get("n_stocks", 0)),
        )
        for p in ia.get("ic_series_5d", [])
    ]

    summary = []
    for hz, s in (ia.get("horizon_summary", {}) or {}).items():
        try:
            summary.append(OnlineICSummary(
                horizon=str(hz),
                n_days=int(s.get("n_days", 0)),
                mean_ic=float(s.get("mean_ic", 0.0)),
                icir=float(s.get("icir", 0.0)),
                t_stat=float(s.get("t_stat", 0.0)),
                ic_gt0_pct=float(s.get("ic_gt0_pct", 0.0)),
            ))
        except Exception as e:
            logger.debug(f"Skip horizon {hz}: {e}")

    period = None
    if ia.get("period_start") and ia.get("period_end"):
        period = f"{ia['period_start']} ~ {ia['period_end']}"
    return online_ic, summary, period


async def _build_response() -> PerformanceResponse:
    ts, ia = await asyncio.gather(
        _fetch_json(TRAINING_STATUS_URL),
        _fetch_json(IC_ANALYSIS_URL),
    )
    training, ic_history, scale_gates = _parse_training(ts)
    online_ic, online_summary, online_period = _parse_online_ic(ia)

    return PerformanceResponse(
        training=training,
        ic_history=ic_history,
        scale_gates=scale_gates,
        online_ic=online_ic,
        online_summary=online_summary,
        online_period=online_period,
        data_sources={
            "training_status": "loaded" if ts else "missing",
            "ic_analysis": "loaded" if ia else "missing",
        },
    )


async def _get_performance() -> PerformanceResponse:
    global _cache, _cache_time
    if _cache and _cache_time and datetime.now() - _cache_time < CACHE_TTL:
        return _cache
    async with _cache_lock:
        if _cache and _cache_time and datetime.now() - _cache_time < CACHE_TTL:
            return _cache
        resp = await _build_response()
        # 兩個來源都失敗且有舊快取 → 沿用舊快取，避免暫時性網路問題清空頁面
        if (resp.training is None and not resp.online_ic) and _cache:
            logger.warning("Performance sources unavailable; serving stale cache")
            return _cache
        _cache, _cache_time = resp, datetime.now()
        return _cache


@router.get("", response_model=PerformanceResponse)
async def get_performance():
    """模型狀態：訓練學習曲線（training_status.json）+ 線上 IC（ic_analysis.json）"""
    return await _get_performance()


@router.post("/cache/refresh")
async def refresh_cache():
    """強制清空 performance 快取（訓練結果 push 後立即看到新資料）"""
    global _cache, _cache_time
    _cache, _cache_time = None, None
    return {"status": "ok", "message": "performance cache cleared"}
