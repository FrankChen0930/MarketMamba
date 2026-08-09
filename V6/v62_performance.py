"""
v62_performance.py — 把每日持股紀錄變成「實際賺了多少」
========================================================
`v62_portfolio.py` 每天寫 `v62_portfolio_{arm}.jsonl`（持股 + 權重 + 完整性），
但**沒有任何程式把它變成報酬**。並行跑 15 個組合的目的就是比較，
而比較的工具在 2026-08-08 之前不存在——這支就是補這個洞。

與回測同義，是這支唯一重要的事
------------------------------
成本、報酬順序、換手定義**全部 import 自 `portfolio_lab` / `v62_portfolio`**，
不另寫一套。實戰數字若用不同口徑算出來，跟 `+38.0%` 那些回測值就不可比，
那整個「累積實戰紀錄」的意義就沒了。

  · 成本      `portfolio_lab.COST_BUY / COST_SELL`
  · 報酬順序  昨日權重吃今天的報酬 → 收盤後再平衡（同 `replay()` 的迴圈）
  · 換手      只記買進側（`portfolio_lab.py:378` 的定義）
  · 基準      等權 eligible 宇宙（規格 §1；**不是 TAIEX**）

⚠️ 前瞻紀錄的性質
-----------------
這裡算的是**真實 out-of-sample**：每一天的持股都是那天收盤後決定的，
沒有任何事後資訊。這正是回測買不到、只能用時間換的東西。
但**樣本數會很小**——起跑後幾週的數字幾乎全是雜訊。
本檔一律印出 `n_days`，並在不足時明講「還不能下結論」。

判讀紀律（沿用 CLAUDE.md）
-------------------------
  · 組合層 N=50 的雜訊底線約 **±6pp 年化** → 小於這個的差距不該當數
  · 比較模型優劣用 decile spread，不用 Top50 年化——但**前瞻紀錄算不出 decile**
    （只有持股、沒有全市場分數的逐日排序），所以這裡只能給 Top50 那條線，
    **判讀時要記得它天生比較吵**
  · **年化一律附 ±標準誤**（`ann_stderr_pp`）。實測（2026-08-09 合成紀錄）：
    30 天算出「年化 **−52.3%**」，而它的標準誤是 **±126pp**——沒有誤差棒的話，
    那個數字會被擺在 37.3% 的回測值旁邊，讀起來像「模型崩了」。

用法
----
    python V6/v62_performance.py                    # 全部 arm，印表
    python V6/v62_performance.py --arms v2_kg_nomacro_f20 ridge_f20
    python V6/v62_performance.py --json             # 另存 v62_performance.json（給後端）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

# ⚠️ 本檔會 import `portfolio_lab`（取 `Market` 與成本常數），而那條線在 import 期
#    就依 `MM_PROTOCOL` 綁定協定。沒設的話會退回 **v1**，載到 2026-07-12 建的舊
#    baseline_cache（**除權息還原與資料修復之前**）——而且只印一行警告、不會失敗。
#    這裡直接設定，不依賴呼叫端記得帶環境變數。
if os.environ.get("MM_PROTOCOL") != "v2":
    os.environ["MM_PROTOCOL"] = "v2"

import v62_portfolio as VP                                       # noqa: E402

RESULTS_DIR = VP.RESULTS_DIR
OUT_JSON = RESULTS_DIR / "v62_performance.json"

# ⚠️ 門檻原本是 20 個交易日，**那擋不住問題**（2026-08-09 用合成紀錄實測）：
#    30 天的紀錄算出「年化 −52.3%」，而 20 天的門檻會放行它，前端就會把那個
#    數字擺在 37.3% 的回測值旁邊——看起來像「模型崩了」，其實只是外推的假象。
#
#    年化報酬的標準誤 ≈ 252 × s_daily / √n。以 N=50 等權組合的日波動
#    **假設** s ≈ 0.012 代入（⚠️ 這是估的；合成紀錄實測那籃子 s ≈ 0.027，
#    誤差棒是這裡的兩倍多 → 真實值只會比下表更寬，不會更窄）：
#        n = 20  → ±68pp     n = 60  → ±39pp
#        n = 126 → ±27pp     n = 252 → ±19pp
#    對照組合層的雜訊底線 **±6pp** —— 也就是說**第一年之內，年化都不精確到
#    足以拿來排名**。所以：
#      ① 門檻拉到 60（一季）：低於此連年化都不顯示，只給累積報酬
#      ② 超過門檻也**一律附上 ±ann_stderr_pp**，讓數字自己說明它有多不準
#    ①是判斷、②是事實。只做①會讓 61 天那天突然冒出一個看似精確的數字。
MIN_DAYS_FOR_ANY_CLAIM = 60
NOISE_FLOOR_PP = 6.0             # 組合層 N=50 的雜訊底線（年化 pp）


def _costs() -> tuple[float, float]:
    """成本常數一律 import，不在這裡寫死——回測改了這裡要跟著改。"""
    from experimental.portfolio_lab import COST_BUY, COST_SELL
    return float(COST_BUY), float(COST_SELL)


def _price_returns(dates: list[pd.Timestamp], stocks: list[str]) -> pd.DataFrame:
    """日報酬矩陣——**直接用 `portfolio_lab.Market`，不自己 pivot**。

    ⚠️ 第一版自己寫了一份 `px.pivot(...).pct_change()`，結果年化 38.20% vs
       回測 38.02%（差 0.18pp）。原因是漏了 `Market` 的兩件事：
         ① `px.where(px > 0)` —— Close<=0 視為缺值（parquet 裡還有 329 列這種資料）
         ② 明確 `ffill().pct_change()` —— 停牌視為持平
       這正是「數值 import 自既有實作、不另寫一套」那條紀律要防的事：
       自己寫的第二份看起來一樣，但在邊角上悄悄分岔，而且**只差 0.18pp、
       很容易被當成捨入誤差放過**。
    """
    from experimental.portfolio_lab import Market
    mkt = Market(np.array(dates, dtype="datetime64[ns]"), sorted(stocks))
    return mkt.ret


def load_log(arm: str) -> pd.DataFrame:
    p = RESULTS_DIR / VP.LOG_FMT.format(arm=arm)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(ln) for ln in p.open(encoding="utf-8") if ln.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # 同一天可能因為重跑而有多筆（step() 有冪等保護，但 jsonl 是 append）
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def evaluate(arm: str, log: pd.DataFrame, ret: pd.DataFrame) -> dict | None:
    """從逐日持股紀錄算實際報酬。迴圈順序**逐行對照 `v62_portfolio.replay()`**。"""
    if log.empty:
        return None
    cb, cs = _costs()
    dates = list(log["date"])
    port = np.zeros(len(dates))
    cost_paid = np.zeros(len(dates))
    turnovers: list[float] = []

    w: dict[str, float] = {}            # 進入當日時持有的權重（= 前一天收盤後的組合）
    for t, d in enumerate(dates):
        # ① 昨日權重吃今天的報酬
        if w and d in ret.index:
            r = ret.loc[d]
            g = {s: w[s] * (1.0 + (float(r.get(s)) if pd.notna(r.get(s)) else 0.0))
                 for s in w}
            tot = sum(g.values())
            port[t] = tot - sum(w.values())
            if tot > 0:
                w = {s: v / tot for s, v in g.items()}
        # ② 收盤後再平衡（下一日才吃到新組合的報酬）
        row = log.iloc[t]
        if bool(row.get("rebalanced")):
            new_w = dict(row.get("weights") or {})
            if not new_w:                       # 舊紀錄沒存 weights → 等權回推
                h = list(row.get("holdings") or [])
                new_w = {s: 1.0 / len(h) for s in h} if h else {}
            delta = {s: new_w.get(s, 0.0) - w.get(s, 0.0) for s in set(new_w) | set(w)}
            buy = sum(v for v in delta.values() if v > 0)
            sell = -sum(v for v in delta.values() if v < 0)
            c = buy * cb + sell * cs
            port[t] -= c
            cost_paid[t] = c
            turnovers.append(buy)               # 換手只記買進側（同 portfolio_lab）
            w = new_w

    n = len(dates)
    cum = float((1 + port).prod() - 1)
    ann = float((1 + port).prod() ** (252 / n) - 1) if n > 0 else 0.0
    sd = float(port.std())
    sharpe = float(port.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    eq = (1 + port).cumprod()
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    # 年化報酬的標準誤（pp）：252 × s_daily / √n。
    # **一定要跟年化一起出現**——年化是把 n 天外推成 252 天，樣本小的時候
    # 那個外推的誤差比任何模型間的差距都大好幾倍，而數字本身看不出來。
    ann_se_pp = float(252 * sd / np.sqrt(n) * 100) if n > 1 and sd > 0 else None
    return {
        "arm": arm,
        "n_days": n,
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "cum_return": round(cum, 4),
        "ann_return": round(ann, 4),
        "ann_stderr_pp": round(ann_se_pp, 1) if ann_se_pp is not None else None,
        "ann_sharpe": round(sharpe, 3),
        "max_drawdown": round(mdd, 4),
        "n_rebalances": len(turnovers),
        "avg_turnover": round(float(np.mean(turnovers)), 4) if turnovers else None,
        "total_cost": round(float(cost_paid.sum()), 4),
        "days_with_incomplete_data": int(sum(
            1 for x in log.get("data_complete", []) if isinstance(x, dict)
            and any(v is False for v in x.values()))),
        "enough_sample": n >= MIN_DAYS_FOR_ANY_CLAIM,
        "daily_returns": [round(float(x), 6) for x in port],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--json", action="store_true", help="另存 v62_performance.json")
    a = ap.parse_args()

    arms = a.arms or list(VP.PORTFOLIOS)
    logs = {x: load_log(x) for x in arms}
    have = {x: v for x, v in logs.items() if not v.empty}
    if not have:
        print("尚無任何前瞻紀錄——V6.2 還沒開始跑，或 jsonl 還沒產生。\n"
              f"（找的是 {RESULTS_DIR}/v62_portfolio_*.jsonl）")
        return 0

    all_dates = sorted({d for lg in have.values() for d in lg["date"]})
    all_stocks = {s for lg in have.values() for hs in lg["holdings"] for s in (hs or [])}
    ret = _price_returns(all_dates, sorted(all_stocks))
    res = {}
    for x, lg in have.items():
        r = evaluate(x, lg, ret)
        if r:
            p = VP.PORTFOLIOS.get(x)
            if p:
                r.update({"tier": p.tier, "freq": p.freq, "head": p.head,
                          "backtest_ann": p.bt_ann, "score_arm": p.score_arm})
            res[x] = r

    n_days = max(v["n_days"] for v in res.values())
    print("=" * 104)
    print(f"V6.2 前瞻績效（真實 out-of-sample）｜{len(res)} 個組合｜最長 {n_days} 個交易日")
    print("=" * 104)
    if n_days < MIN_DAYS_FOR_ANY_CLAIM:
        print(f"⚠️ 只有 {n_days} 個交易日（門檻 {MIN_DAYS_FOR_ANY_CLAIM}）——"
              f"**這些數字現在不能拿來下任何結論**，只用來確認管線有在跑。\n"
              f"   年化欄位一併印出 ±標準誤，看一眼就知道它有多不準。")
    print(f"{'arm':26s}{'天':>4s}{'累積':>9s}{'年化 ±標準誤':>20s}{'Sharpe':>8s}"
          f"{'MDD':>8s}{'換手':>7s}{'回測年化':>10s}{'差':>9s}  分級")
    print("-" * 112)
    for x, v in sorted(res.items(), key=lambda kv: -kv[1]["ann_return"]):
        bt = v.get("backtest_ann")
        diff = (v["ann_return"] - bt) * 100 if bt is not None else None
        se = v.get("ann_stderr_pp")
        ann_s = f"{v['ann_return']*100:.1f}%" + (f" ±{se:.0f}pp" if se else "")
        # 「差」小於標準誤時不該被讀成差距 → 標記出來，不要讓它看起來像結論
        d_s = "—" if diff is None else (
            f"{diff:+.1f}pp" + ("*" if se and abs(diff) < se else ""))
        print(f"{x:26s}{v['n_days']:4d}{v['cum_return']*100:8.1f}%"
              f"{ann_s:>20s}{v['ann_sharpe']:8.2f}"
              f"{v['max_drawdown']*100:7.1f}%"
              f"{(v['avg_turnover'] or 0)*100:6.0f}%"
              f"{(f'{bt*100:.1f}%' if bt is not None else '—'):>10s}"
              f"{d_s:>9s}  {v.get('tier','')}")
    print("-" * 112)
    print(f"⚠️ 「差」是前瞻 vs 回測。組合層 N=50 的雜訊底線約 ±{NOISE_FLOOR_PP:.0f}pp，"
          f"**小於這個的差距不該當數**；標 `*` 者連自己的標準誤都跨不過。")
    print("⚠️ ±標準誤 = 252 × s_daily / √n。它通常比模型之間的差距大好幾倍——"
          "**第一年之內年化都排不出名次**，這不是工具的問題，是樣本量的問題。")
    print("⚠️ 前瞻紀錄算不出 decile spread（只有持股、沒有全市場逐日排序），"
          "所以這張表天生比 decile 吵——判讀時要記得。")
    inc = {x: v["days_with_incomplete_data"] for x, v in res.items()
           if v["days_with_incomplete_data"]}
    if inc:
        print(f"⚠️ 有資料缺漏的天數：{inc}——那幾天的分數可能不可靠。")

    if a.json:
        OUT_JSON.write_text(json.dumps(
            {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
             "n_days": n_days, "noise_floor_pp": NOISE_FLOOR_PP,
             "min_days_for_any_claim": MIN_DAYS_FOR_ANY_CLAIM,
             "models": res}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ → {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
