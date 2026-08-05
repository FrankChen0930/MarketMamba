"""
v62_portfolio.py — V6.2 組合層狀態機（規格 `5d/20`）
=====================================================
把每日分數變成**實際持股名單**：N=50 / k=1.5 緩衝 / 每 20 個交易日再平衡。

分數本身不是持股名單——中間 19 天不換股，所以必須有狀態。

規格來源：`docs/portfolio-construction-baseline-v1.md` + 2026-08-05 定案
（CLAUDE.md「★ V6.2 上線規格」）。回測對照：N=50/k=1.5/20日 年化 +38.1%。

七個設計決定（2026-08-05）
--------------------------
 ① **一個 arm 一份 state**（`v62_state_{arm}.json`）→ 多模型並行時互不干擾。
 ② **再平衡觸發用交易日曆算，不用「跑了幾次」**：距上次再平衡 ≥20 個交易日就換。
    推論失敗一天不會讓排程漂掉——用「跑了幾次」的話，漏跑一天就永遠差一天。
 ③ **緩衝逐行照抄 `portfolio_lab.run_config`**：保留 rank ≤ k×N(=75) 的持股，
    不足 N 的部分從 Top-N 依序補。
 ④ **權重：再平衡日等權，期間自然漂移**（不做期中再平衡）——與回測一致。
 ⑤ **可交易性：照規格格產生名單（不擋漲跌停／處置），但逐檔標註旗標。**
    擋掉會讓實戰與回測不同義（實測代價 −5.5pp ~ −7.3pp，是既有記載的例外）。
    **不偷改規則**，讓使用者看得到、自己決定要不要手動略過。
 ⑥ **每日落檔含資料完整性**：幾個月後看到某段表現差，要能區分「模型不好」
    與「那幾天 margin 缺了」。事後補不回來，所以當天就要記。
 ⑦ **驗證：把狀態機 replay 582 天，必須重現 portfolio_lab 的那一格。**
    這是獨立實作 → 兩邊算出同一個年化，才代表兩邊都對。

用法
----
    # 驗證（先跑這個）
    MM_PROTOCOL=v2 python V6/v62_portfolio.py --replay v2_kg_nomacro__live

    # 每日（在 run_v62_inference 之後）
    python V6/v62_portfolio.py --step --scores V6/results/df_v62.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

# ── 規格常數（改這裡等於改上線規格，不要散落在別處）──────────────────
N_HOLD      = 50      # 持股檔數
BUFFER_K    = 1.5     # 緩衝：rank ≤ k×N 就續抱
REBAL_DAYS  = 20      # 每 20 個交易日再平衡
WEIGHT_MODE = "equal"

RESULTS_DIR = _V6 / "results"
STATE_FMT   = "v62_state_{arm}.json"
LOG_FMT     = "v62_portfolio_{arm}.jsonl"   # 逐日 append，事後不得改


# ============================================================
# 1. 核心：緩衝再平衡（純函式，可單獨驗證）
# ============================================================
def rebalance(holdings: list[str], rank: dict[str, float],
              n: int = N_HOLD, k: float = BUFFER_K) -> dict:
    """
    `rank`: {stock_id: 當日排名}（1 = 分數最高；不可交易者不要放進來）。

    邏輯與 `portfolio_lab.run_config` 逐行相同：
      keep = 現有持股中 rank ≤ k*n 者（緩衝：還在前 75 名就不動）
      adds = Top-n 裡不在 keep 的，依序補到 n 檔
    """
    kn = k * n
    keep = [s for s in holdings if s in rank and rank[s] <= kn]
    keep_set = set(keep)
    top_n = [s for s, _ in sorted(rank.items(), key=lambda kv: kv[1])[:n]]
    adds = [s for s in top_n if s not in keep_set][:max(n - len(keep), 0)]
    new = keep + adds
    return {"holdings": new, "kept": keep, "added": adds,
            "dropped": [s for s in holdings if s not in set(new)]}


def equal_weights(holdings: list[str]) -> dict[str, float]:
    if not holdings:
        return {}
    w = 1.0 / len(holdings)
    return {s: w for s in holdings}


# ============================================================
# 2. 驗證：replay 582 天，必須重現 portfolio_lab
# ============================================================
def replay(model: str, n: int = N_HOLD, k: float = BUFFER_K,
           freq: int = REBAL_DAYS) -> bool:
    """
    用同一份分數檔，讓**本檔的狀態機**獨立算出年化/Sharpe/換手，
    再對 `portfolio_lab_result.json` 同一格。兩個獨立實作算出同一個數字，
    才代表狀態機的緩衝與再平衡邏輯真的與回測一致。

    判準（跑之前定死）：年化差 < 0.001（0.1pp）、換手差 < 0.01。
    """
    from experimental.portfolio_lab import COST_BUY, COST_SELL, Market, RESULT_DIR

    sc = pd.read_parquet(_V6 / "experimental" / "result" / "scores" / f"{model}.parquet")
    sc["Date"] = pd.to_datetime(sc["Date"])
    rank_df = sc.pivot(index="Date", columns="stock_id", values="score") \
                .rank(axis=1, ascending=False, method="first")
    mkt = Market(rank_df.index.to_numpy(), list(rank_df.columns))
    rank_df = rank_df.reindex(index=mkt.dates, columns=mkt.stocks)
    rank_df = rank_df.where(mkt.px.notna())        # 沒有價格 → 不可交易

    ret = mkt.ret.copy()
    ret_np = np.nan_to_num(ret.to_numpy(np.float64))
    dates = mkt.dates
    reb_idx = set(range(0, len(dates), freq))

    holdings: list[str] = []
    w: dict[str, float] = {}
    port_ret = np.zeros(len(dates))
    turnovers: list[float] = []
    col = {s: i for i, s in enumerate(mkt.stocks)}

    for t in range(len(dates)):
        # ① 昨日權重吃今天的報酬（與 run_config 相同順序）
        if holdings:
            g = {s: w[s] * (1.0 + ret_np[t, col[s]]) for s in holdings}
            tot = sum(g.values())
            port_ret[t] = tot - sum(w[s] for s in holdings)
            if tot > 0:
                w = {s: v / tot for s, v in g.items()}
        # ② 收盤後再平衡（下一日才吃到新組合的報酬）
        if t in reb_idx:
            row = rank_df.iloc[t]
            rk = {s: float(v) for s, v in row.items() if np.isfinite(v)}
            if len(rk) < n:
                continue
            r = rebalance(holdings, rk, n, k)
            new_w = equal_weights(r["holdings"])
            delta = {s: new_w.get(s, 0.0) - w.get(s, 0.0)
                     for s in set(new_w) | set(w)}
            buy = sum(v for v in delta.values() if v > 0)
            sell = -sum(v for v in delta.values() if v < 0)
            port_ret[t] -= buy * COST_BUY + sell * COST_SELL
            # 換手定義以 portfolio_lab 為準：只記**買進側**（`turnovers.append(buy_frac)`，
            # portfolio_lab.py:378）。用 (buy+sell)/2 會在首次建倉那次差 0.5，
            # 分攤到 30 次再平衡剛好是 1.67pp——第一版就是這樣對不上的。
            turnovers.append(buy)
            holdings, w = r["holdings"], new_w

    ann = float((1 + port_ret).prod() ** (252 / len(dates)) - 1)
    sharpe = float(port_ret.mean() / port_ret.std() * np.sqrt(252))
    turn = float(np.mean(turnovers))

    ref_path = RESULT_DIR / "portfolio_lab_result.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))["models"][model]["grid"]
    cell = next(g for g in ref if g["n"] == n and abs(g["k"] - k) < 1e-9
                and g["freq"] == freq and g["liq"] is None)

    d_ann = abs(ann - cell["ann_return"])
    d_turn = abs(turn - cell["avg_turnover"])
    ok = d_ann < 0.001 and d_turn < 0.01
    print(f"\n{'='*70}\n[replay] {model}｜N={n} k={k} freq={freq}｜{len(dates)} 天\n{'='*70}")
    print(f"{'指標':16s}{'狀態機':>12s}{'portfolio_lab':>16s}{'差':>10s}")
    print(f"{'年化':16s}{ann*100:11.2f}%{cell['ann_return']*100:15.2f}%{d_ann*100:9.3f}pp")
    print(f"{'Sharpe':16s}{sharpe:12.3f}{cell['ann_sharpe']:16.3f}"
          f"{abs(sharpe-cell['ann_sharpe']):10.3f}")
    print(f"{'換手':16s}{turn*100:11.1f}%{cell['avg_turnover']*100:15.1f}%{d_turn*100:9.2f}pp")
    print(f"[replay] 再平衡 {len(turnovers)} 次（回測 {cell['n_rebalances']} 次）")
    print(f"[replay] {'✅ 兩個獨立實作一致 → 狀態機邏輯正確' if ok else '❌ 不一致，狀態機有問題'}")
    print(f"{'='*70}")
    return ok


# ============================================================
# 3. 每日推進
# ============================================================
def _trading_days_between(a: str, b: str) -> int:
    """用 prices_raw 的實際交易日曆算（不是曆日、也不是「跑了幾次」）。"""
    from marketmamba.config import PROCESSED_DIR
    pr = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet", columns=["Date"])
    d = pd.to_datetime(pr["Date"].astype(str)).drop_duplicates().sort_values()
    return int(((d > pd.Timestamp(a)) & (d <= pd.Timestamp(b))).sum())


def step(arm: str, scores_path: Path, data_complete: dict | None = None,
         force_rebalance: bool = False) -> dict:
    """讀今日分數 → 決定要不要再平衡 → 更新 state → append 逐日紀錄。"""
    sc = pd.read_csv(scores_path, dtype={"stock_id": str})
    date = str(pd.to_datetime(sc["Date"].iloc[0]).date())
    sc = sc.sort_values("score", ascending=False).reset_index(drop=True)
    rk = {r.stock_id: float(i + 1) for i, r in enumerate(sc.itertuples())}

    st_path = RESULTS_DIR / STATE_FMT.format(arm=arm)
    st = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {
        "arm": arm, "holdings": [], "weights": {}, "last_rebalance": None,
        "n_rebalances": 0, "spec": {"n": N_HOLD, "k": BUFFER_K, "freq": REBAL_DAYS,
                                    "weight": WEIGHT_MODE},
    }
    if st.get("last_date") == date:
        print(f"[state] {date} 已處理過，跳過（重跑不會重複計算）", flush=True)
        return st

    since = (_trading_days_between(st["last_rebalance"], date)
             if st["last_rebalance"] else REBAL_DAYS)
    due = force_rebalance or st["last_rebalance"] is None or since >= REBAL_DAYS

    if due:
        r = rebalance(st["holdings"], rk)
        st["holdings"] = r["holdings"]
        st["weights"] = equal_weights(r["holdings"])
        st["last_rebalance"] = date
        st["n_rebalances"] += 1
        print(f"[state] ★ {date} 再平衡 #{st['n_rebalances']}｜"
              f"續抱 {len(r['kept'])}｜買進 {len(r['added'])}｜賣出 {len(r['dropped'])}",
              flush=True)
        if r["added"]:
            print(f"  買進：{', '.join(r['added'][:15])}"
                  f"{' …' if len(r['added']) > 15 else ''}", flush=True)
        if r["dropped"]:
            print(f"  賣出：{', '.join(r['dropped'][:15])}"
                  f"{' …' if len(r['dropped']) > 15 else ''}", flush=True)
        since = 0
    else:
        print(f"[state] {date} 不換股（距上次再平衡 {since}/{REBAL_DAYS} 個交易日）",
              flush=True)

    # 漂移度：目前持股還有幾檔在模型當前 Top-N（中間 19 天唯一有參考價值的數字）
    drift = sum(1 for s in st["holdings"] if rk.get(s, 1e9) <= N_HOLD)
    st.update({"last_date": date, "days_since_rebalance": since,
               "days_to_next": max(REBAL_DAYS - since, 0),
               "in_top_n": drift, "data_complete": data_complete or {}})
    print(f"[state] 持股 {len(st['holdings'])} 檔｜其中 {drift} 檔仍在模型 Top{N_HOLD}"
          f"｜距下次再平衡 {st['days_to_next']} 個交易日", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    st_path.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    # 逐日 append，事後不得修改——這份才是「不可竄改的前瞻紀錄」
    rec = {"date": date, "arm": arm, "rebalanced": bool(due),
           "holdings": st["holdings"], "in_top_n": drift,
           "data_complete": data_complete or {},
           "written_at": datetime.now().isoformat(timespec="seconds")}
    with (RESULTS_DIR / LOG_FMT.format(arm=arm)).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", metavar="MODEL",
                    help="用該分數檔 replay，對 portfolio_lab 驗證狀態機")
    ap.add_argument("--step", action="store_true", help="每日推進一步")
    ap.add_argument("--arm", default="v2_kg_nomacro")
    ap.add_argument("--scores", default=str(RESULTS_DIR / "df_v62.csv"))
    ap.add_argument("--force-rebalance", action="store_true",
                    help="強制再平衡（第一天上線用）")
    a = ap.parse_args()

    if a.replay:
        sys.exit(0 if replay(a.replay) else 1)
    if a.step:
        step(a.arm, Path(a.scores), force_rebalance=a.force_rebalance)
    else:
        ap.print_help()
