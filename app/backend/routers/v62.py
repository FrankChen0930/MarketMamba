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


# ── arm 清單：讀 `v62_arms.json`（由 `V6/v62_portfolio.write_manifest()` 產生）──
#
# ⚠️ **刻意不在這裡自帶一份 arm 表。** 後端跑在 Render（rootDir=app/backend），
#    import 不到 `V6/v62_portfolio.py`，所以兩邊的表**連 assert 都對不起來**
#    ——不同 process、不同機器。本專案已經因為「同一套規則兩份實作、各自維護」
#    出過 bug（scanner 的進場條件 vs sim_engine 的進場分數）。
#    → 唯一真相是 `PORTFOLIOS`，每天隨結果一起 push 成 `v62_arms.json`。
#
# 下面的 fallback 只在 manifest 還沒 push 上去時用（例如第一次部署），
# 且**只放主線一個**——放多了就等於偷偷把重複的表又寫回來。
MANIFEST_FILE = "v62_arms.json"
FALLBACK_DEFAULT = "v2_kg_nomacro_f20"
FALLBACK_ARMS = [{
    "arm": FALLBACK_DEFAULT, "score_arm": "v2_kg_nomacro", "freq": 20, "n": 50,
    "k": 1.5, "tier": "primary", "backtest_ann": 0.380, "head": "5d 頭",
    "label": "5d 頭 / 20 日", "state_file": f"v62_state_{FALLBACK_DEFAULT}.json",
    "score_file": "df_v62.csv",
    "note": "manifest 尚未推送，這是硬編的主線 fallback",
}]

# 回測脈絡：全 arm 共通的警語（每個 arm 自己的數字從 manifest / state 讀）
CAVEATS = ["單一 seed（年化帶 ±2.7pp 的 run-to-run 雜訊）",
           "單一 582 天多頭窗、無 walk-forward",
           "優勢集中在大盤上升段；下跌段比 v2_kg 差 8.4pp",
           "回測年化來自 582 天單一窗，**不是**前瞻紀錄"]
TIER_DESC = {
    "primary":    "主線規格（回測最佳）",
    "equivalent": "研究用；與主線差距在雜訊底線（±6pp）內，分不出優劣",
    "inferior":   "研究用；已知明確劣於主線，請勿照做",
}

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


async def _manifest() -> dict:
    """讀 `v62_arms.json`；抓不到就退回只有主線的 fallback（1h 快取）。"""
    now = datetime.now()
    if "__manifest__" in _cache and now - _cache_time.get("__manifest__", now) < CACHE_TTL:
        return _cache["__manifest__"]
    m = await _fetch(_url(MANIFEST_FILE), as_json=True)
    if not m or not m.get("arms"):
        logger.warning("v62: 讀不到 v62_arms.json → 用 fallback（只有主線）")
        m = {"default": FALLBACK_DEFAULT, "arms": FALLBACK_ARMS,
             "tier_desc": TIER_DESC, "stale": True}
    _cache["__manifest__"], _cache_time["__manifest__"] = m, now
    return m


async def _arm_map() -> dict[str, dict]:
    return {a["arm"]: a for a in (await _manifest())["arms"]}


async def _load(arm: str, meta: dict) -> Optional[dict]:
    state_file = meta.get("state_file") or f"v62_state_{arm}.json"
    score_file = meta.get("score_file") or "df_v62.csv"
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

    # 規格以 **state 檔裡的 spec 為準**——那是當天真的用來換股的參數。
    # manifest 只補上顯示用的欄位（label / 回測數字）。兩者若不一致，
    # 相信 state：manifest 可能是後來改的，而 state 是當下的事實。
    spec = dict(st.get("spec") or {})
    spec.setdefault("n", meta.get("n", 50))
    spec.setdefault("k", meta.get("k", 1.5))
    spec.setdefault("freq", meta.get("freq", 20))
    spec.setdefault("tier", meta.get("tier", "equivalent"))
    n_hold = int(spec.get("n", 50))

    holdings = [{"stock_id": s, "name": nm(s),
                 "weight": round(float(st.get("weights", {}).get(s, 0)), 4),
                 "current_rank": rank_of.get(s),
                 "in_top_n": bool(rank_of.get(s, 10**9) <= n_hold)}
                for s in st.get("holdings", [])]

    tier = spec.get("tier", "equivalent")
    return {
        "arm": arm, "label": meta.get("label", arm),
        "spec": spec, "tier": tier,
        "tier_desc": TIER_DESC.get(tier, ""),
        "is_primary": tier == "primary",
        "backtest_ann": meta.get("backtest_ann"),
        "caveats": CAVEATS,
        "note_arm": meta.get("note", ""),
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
async def get_v62_portfolio(arm: str | None = None):
    """
    V6.2 現行持股 + 再平衡倒數。

    ⚠️ `holdings` 是**現在持有的**，不是「今天建議買進的」。
       只有 `is_rebalance_day=true` 那天的變動才是交易動作。

    ⚠️ 只有 `tier="primary"` 那個 arm 是上線規格。其餘是**研究用的並行組合**
       （不同預測頭 / 不同再平衡率），前端必須把 `tier_desc` 顯示出來。
    """
    amap = await _arm_map()
    arm = arm or (await _manifest()).get("default", FALLBACK_DEFAULT)
    if arm not in amap:
        return {"error": f"unknown arm {arm}", "available_arms": list(amap)}

    now = datetime.now()
    if arm in _cache and now - _cache_time.get(arm, now) < CACHE_TTL:
        return _cache[arm]
    async with _lock:
        if arm in _cache and now - _cache_time.get(arm, now) < CACHE_TTL:
            return _cache[arm]
        data = await _load(arm, amap[arm])
        if data:
            _cache[arm], _cache_time[arm] = data, datetime.now()
            return data

    meta = amap[arm]
    return {"arm": arm, "label": meta.get("label", arm),
            "spec": {"n": meta.get("n"), "k": meta.get("k"), "freq": meta.get("freq"),
                     "tier": meta.get("tier")},
            "tier": meta.get("tier"), "tier_desc": TIER_DESC.get(meta.get("tier"), ""),
            "is_primary": meta.get("tier") == "primary",
            "caveats": CAVEATS, "date": None,
            "is_rebalance_day": False, "holdings": [], "scores": [],
            "note": "V6.2 尚未產生資料（run_v62_daily.py 跑完並 push 後出現）"}


@router.get("/arms")
async def list_arms():
    """並行跑的組合清單（模型 × 預測頭 × 再平衡率）。

    **這個集合在開始累積實戰紀錄那天就定案**——晚加入的少了那段紀錄，
    無法與其他組合公平並列。

    來源是 `v62_arms.json`（由 `V6/v62_portfolio.PORTFOLIOS` 產生），
    後端不自帶一份表。`stale=true` 代表 manifest 還沒 push、目前用 fallback。
    """
    m = await _manifest()
    return {"default": m.get("default", FALLBACK_DEFAULT),
            "tier_desc": m.get("tier_desc", TIER_DESC),
            "caveats": CAVEATS,
            "stale": bool(m.get("stale")),
            "generated_at": m.get("generated_at"),
            "arms": [{"arm": a["arm"], "label": a.get("label", a["arm"]),
                      "tier": a.get("tier"), "freq": a.get("freq"),
                      "head": a.get("head"), "backtest_ann": a.get("backtest_ann"),
                      "note": a.get("note", "")} for a in m["arms"]]}


@router.post("/cache/refresh")
async def refresh_v62_cache():
    """清 V6.2 快取（每日自動化 push 完後呼叫，不必等 1h TTL）。"""
    async with _lock:
        _cache.clear()
        _cache_time.clear()
    logger.info("v62 cache cleared")
    return {"status": "ok", "message": "v62 cache cleared"}
