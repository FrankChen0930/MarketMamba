"""
V6.2 組合層 router — 規格 `5d/20`
=================================
讀 GitHub raw 的 `v62_state_{arm}.json`（持股狀態）與 `df_v62*.csv`（當日分數），
1h TTL cache + asyncio.Lock，完全比照 `dual.py` / `signals.py` 的模式。

⚠️ **附加 router，不動既有 signals / dual / market / performance 任何一支。**

⚠️ 語意（前端必須照這個講，不可簡化成「今日選股」）
----------------------------------------------------
本模型**每 20 個交易日才換一次股**。中間 19 天的分數變動**不是交易訊號**——
回測實測每日再平衡是 −19.9%、每 20 日是 +38.1%，兩者差在成本與換手。
所以 API 一律同時回：
  `is_rebalance_day`（今天是不是換股日）
  `days_to_next`（距下次還幾個交易日）
  `in_top_n`（現有持股還有幾檔在模型當前 Top50 —— 中間 19 天唯一有參考價值的數字）
  `data_complete`（當日資料完整性；缺漏時前端要顯示，否則事後無法區分
                   「模型不好」與「那幾天資料缺了」）
"""
import asyncio
import io
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v62", tags=["V6.2"])

GITHUB_RESULTS_URL = os.getenv("GITHUB_RESULTS_URL", "")


def _url(name: str) -> str:
    return GITHUB_RESULTS_URL.replace("df_kelly.csv", name) if GITHUB_RESULTS_URL else ""


# arm → (state 檔, 分數檔, 顯示名)。與 `V6/run_v62_inference.py` 的 ARMS 對應。
ARMS: dict[str, tuple[str, str, str]] = {
    "v2_kg_nomacro": ("v62_state_v2_kg_nomacro.json", "df_v62.csv",    "5 日頭（上線規格）"),
    "head10d":       ("v62_state_head10d.json",       "df_v62_h10.csv", "10 日頭"),
    "head20d":       ("v62_state_head20d.json",       "df_v62_h20.csv", "20 日頭"),
}
DEFAULT_ARM = "v2_kg_nomacro"
SPEC = {"n": 50, "k": 1.5, "rebalance_days": 20, "weight": "equal",
        "backtest": {"window": "2024-01-02 ~ 2026-06-02（582 天）",
                     "ann_return": 0.381, "sharpe": 1.697, "max_drawdown": -0.257,
                     "decile_spread_sharpe": 5.007, "mean_ic": 0.1139},
        "caveats": ["單一 seed（年化帶 ±2.7pp 的 run-to-run 雜訊）",
                    "單一 582 天多頭窗、無 walk-forward",
                    "優勢集中在大盤上升段；下跌段比 v2_kg 差 8.4pp"]}

CACHE_TTL = timedelta(hours=1)
_cache: dict[str, dict] = {}
_cache_time: dict[str, datetime] = {}
_lock = asyncio.Lock()


async def _fetch(url: str, as_json: bool):
    if not url:
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
        if r.status_code != 200:
            logger.warning(f"v62 fetch {url}: {r.status_code}")
            return None
        return r.json() if as_json else pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        logger.error(f"v62 fetch error {url}: {e}")
        return None


async def _name_fn():
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    try:
        from stock_info import get_stock_info, get_stock_name
        info = await get_stock_info()
        return lambda sid: get_stock_name(sid, info)
    except Exception:
        return lambda sid: sid


async def _load(arm: str) -> Optional[dict]:
    state_file, score_file, label = ARMS[arm]
    st = await _fetch(_url(state_file), as_json=True)
    sc = await _fetch(_url(score_file), as_json=False)
    if st is None:
        return None

    nm = await _name_fn()
    rank_of: dict[str, int] = {}
    scores = []
    if sc is not None and len(sc):
        sc["stock_id"] = sc["stock_id"].astype(str)
        for i, row in sc.iterrows():
            sid = str(row["stock_id"])
            rank_of[sid] = i + 1
            if i < 100:
                scores.append({"rank": i + 1, "stock_id": sid, "name": nm(sid),
                               "score": round(float(row.get("score", 0)), 4)})

    holdings = [{"stock_id": s, "name": nm(s),
                 "weight": round(float(st.get("weights", {}).get(s, 0)), 4),
                 "current_rank": rank_of.get(s),
                 "in_top_n": bool(rank_of.get(s, 10**9) <= SPEC["n"])}
                for s in st.get("holdings", [])]

    return {
        "arm": arm, "label": label, "spec": SPEC,
        "date": st.get("last_date"),
        "is_rebalance_day": st.get("last_rebalance") == st.get("last_date"),
        "last_rebalance": st.get("last_rebalance"),
        "n_rebalances": st.get("n_rebalances", 0),
        "days_since_rebalance": st.get("days_since_rebalance"),
        "days_to_next": st.get("days_to_next"),
        "in_top_n": st.get("in_top_n"),
        "data_complete": st.get("data_complete", {}),
        "holdings": holdings,
        "scores": scores,
    }


@router.get("/portfolio")
async def get_v62_portfolio(arm: str = DEFAULT_ARM):
    """
    V6.2 現行持股 + 再平衡倒數。

    ⚠️ `holdings` 是**現在持有的**，不是「今天建議買進的」。
       只有 `is_rebalance_day=true` 那天的變動才是交易動作。
    """
    if arm not in ARMS:
        return {"error": f"unknown arm {arm}", "available_arms": list(ARMS)}

    now = datetime.now()
    if arm in _cache and now - _cache_time.get(arm, now) < CACHE_TTL:
        return _cache[arm]
    async with _lock:
        if arm in _cache and now - _cache_time.get(arm, now) < CACHE_TTL:
            return _cache[arm]
        data = await _load(arm)
        if data:
            _cache[arm], _cache_time[arm] = data, datetime.now()
            return data

    return {"arm": arm, "label": ARMS[arm][2], "spec": SPEC, "date": None,
            "is_rebalance_day": False, "holdings": [], "scores": [],
            "note": "V6.2 尚未產生資料（run_v62_inference + v62_portfolio --step 並 push 後出現）"}


@router.get("/arms")
async def list_arms():
    """並行跑的模型清單。**這個集合在開始累積實戰紀錄那天就定案**——
    晚加入的模型少了那段紀錄，無法與其他模型公平並列。"""
    return {"default": DEFAULT_ARM,
            "arms": [{"arm": k, "label": v[2]} for k, v in ARMS.items()]}


@router.post("/cache/refresh")
async def refresh_v62_cache():
    """清 V6.2 快取（每日自動化 push 完後呼叫，不必等 1h TTL）。"""
    async with _lock:
        _cache.clear()
        _cache_time.clear()
    logger.info("v62 cache cleared")
    return {"status": "ok", "message": "v62 cache cleared"}
