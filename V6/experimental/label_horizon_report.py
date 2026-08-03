"""
label_horizon_report.py — 標籤 horizon vs 持有期：彙整與大盤區間切分
====================================================================
回答使用者 2026-08-03 的問題：「再平衡 20 天卻只預測 5 天，不是很怪嗎？」

本檔只做**彙整與再切分**，不重跑任何模型：
  A. 訊號層：每份分數對 5d / 10d / 20d 三個評估 horizon 的逐日 Spearman IC
  B. 組合層：從 `portfolio_lab_result.json` 取毛報酬 / 成本 / 淨年化 × 再平衡頻率
  C. 大盤上升段 / 下跌段切分（`portfolio_lab` 不存逐日序列，需重跑少數幾格）

C 的切分定義沿用 `docs/portfolio-lab-results-2026-08-01.md` §7c：
**依「等權 eligible 宇宙過去 20 日累積報酬」的正負分段**，不是前後半——
§7c 已證明前後半切分測不到這件事（切點兩側都是多頭）。

用法
----
    MM_PROTOCOL=v2 python V6/experimental/label_horizon_report.py
輸出：`V6/experimental/result/label_horizon_result.json`
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

if os.environ.get("MM_PROTOCOL") != "v2":
    raise SystemExit("❌ 請設 MM_PROTOCOL=v2 再跑")

from experimental.portfolio_lab import (          # noqa: E402
    Market, RESULT_DIR, SCORE_DIR, TRADING_DAYS, equal_weight_universe, run_config,
)

OUT = RESULT_DIR / "label_horizon_result.json"
LAB_JSON = RESULT_DIR / "portfolio_lab_result.json"
FREQS = [1, 3, 5, 10, 20]
CELL = {"n": 50, "k": 1.5, "liq": None}          # 跨模型比較用的那一格
SEG_WINDOW = 20                                   # §7c：以等權宇宙過去 20 日累積報酬定段


# ============================================================
# A. 訊號層：一份分數 × 三個評估 horizon
# ============================================================
def signal_layer(models: list[str]) -> dict:
    from marketmamba.config import PROCESSED_DIR
    lp = Path(PROCESSED_DIR) / "baseline_cache_v2" / "baseline_label_10d.parquet"
    if not lp.exists():
        raise SystemExit(f"❌ 找不到 {lp.name}，先跑 label_10d.py")
    lab = pd.read_parquet(lp)
    lab["Date"] = pd.to_datetime(lab["Date"]).astype(str).str.slice(0, 10)
    lab["stock_id"] = lab["stock_id"].astype(str)

    out = {}
    for m in models:
        f = SCORE_DIR / f"{m}.parquet"
        if not f.exists():
            continue
        sc = pd.read_parquet(f)
        sc["Date"] = pd.to_datetime(sc["Date"]).astype(str).str.slice(0, 10)
        sc["stock_id"] = sc["stock_id"].astype(str)
        mg = sc.merge(lab, on=["Date", "stock_id"], how="left")
        assert len(mg) == len(sc), f"{m}: 併標籤後列數變了"
        row = {}
        for h in ("5d", "10d", "20d"):
            ics = []
            for _, g in mg.groupby("Date", sort=True):
                y = g[f"Alpha_{h}"].to_numpy(np.float64)
                s = g["score"].to_numpy(np.float64)
                ok = np.isfinite(y) & np.isfinite(s)
                if ok.sum() >= 30:
                    ics.append(pd.Series(s[ok]).corr(pd.Series(y[ok]), method="spearman"))
            a = np.array([x for x in ics if np.isfinite(x)])
            row[h] = {"mean_ic": round(float(a.mean()), 4),
                      "icir": round(float(a.mean() / a.std()), 3) if a.std() else None,
                      "pct_pos": round(float((a > 0).mean()), 3), "n_days": int(len(a))}
        out[m] = row
        print(f"[訊號層] {m:26s}｜" + "｜".join(
            f"vs {h} {row[h]['mean_ic']:+.4f}(ICIR {row[h]['icir']})"
            for h in ("5d", "10d", "20d")), flush=True)
    return out


# ============================================================
# B. 組合層：毛/淨拆解 × 再平衡頻率（讀既有 JSON，不重跑）
# ============================================================
def portfolio_layer(models: list[str]) -> dict:
    if not LAB_JSON.exists():
        raise SystemExit(f"❌ 找不到 {LAB_JSON.name}，先跑 portfolio_lab --sweep")
    lab = json.loads(LAB_JSON.read_text(encoding="utf-8"))["models"]
    out = {}
    for m in models:
        if m not in lab:
            continue
        rows = {}
        for r in lab[m]["grid"]:
            if r["n"] == CELL["n"] and r["k"] == CELL["k"] and r["liq"] == CELL["liq"] \
                    and r["freq"] in FREQS:
                rows[r["freq"]] = {
                    "gross": round(r["ann_return"] + r["ann_cost_drag"], 4),
                    "cost": r["ann_cost_drag"], "net": r["ann_return"],
                    "sharpe": r["ann_sharpe"], "mdd": r["max_drawdown"],
                    "turnover": r["avg_turnover"], "excess": r["excess_vs_ew"],
                }
        out[m] = {"by_freq": rows,
                  "benchmark_ew": lab[m].get("benchmark_ew_universe_ann"),
                  "decile_spread_sharpe": (lab[m].get("decile") or {}).get("decile_spread_sharpe")}
    return out


# ============================================================
# C. 大盤上升段 / 下跌段（需要逐日序列 → 重跑少數幾格）
# ============================================================
def regime_split(models: list[str]) -> dict:
    ref = None
    for m in models:
        f = SCORE_DIR / f"{m}.parquet"
        if f.exists():
            ref = pd.read_parquet(f)
            break
    if ref is None:
        raise SystemExit("❌ 沒有任何分數檔")
    ref["Date"] = pd.to_datetime(ref["Date"])
    dates = np.sort(ref["Date"].unique())
    stocks = sorted(ref["stock_id"].astype(str).unique())
    mkt = Market(dates, stocks)

    out, bench_seg = {}, None
    for m in models:
        f = SCORE_DIR / f"{m}.parquet"
        if not f.exists():
            continue
        t0 = time.time()
        sig = pd.read_parquet(f)
        sig["Date"] = pd.to_datetime(sig["Date"])
        rank = sig.pivot(index="Date", columns="stock_id", values="score") \
                  .rank(axis=1, ascending=False, method="first")
        rank = rank.reindex(index=mkt.dates, columns=mkt.stocks).where(mkt.px.notna())
        bench = equal_weight_universe(mkt, rank)
        if bench_seg is None:
            # 分段依據：等權宇宙過去 SEG_WINDOW 日累積報酬的正負（§7c 定義）
            cum = pd.Series(bench).rolling(SEG_WINDOW).apply(lambda x: (1 + x).prod() - 1)
            bench_seg = (cum > 0).to_numpy()
            valid = np.isfinite(cum.to_numpy())
            print(f"[區間] 上升段 {int((bench_seg & valid).sum())} 天 / "
                  f"下跌段 {int((~bench_seg & valid).sum())} 天"
                  f"（前 {SEG_WINDOW-1} 天無定義）", flush=True)
            out["_segments"] = {"up_days": int((bench_seg & valid).sum()),
                                "down_days": int((~bench_seg & valid).sum()),
                                "window": SEG_WINDOW}
            for tag, sel in (("up", bench_seg & valid), ("down", ~bench_seg & valid)):
                b = bench[sel]
                out["_segments"][f"benchmark_{tag}_ann"] = round(
                    float((1 + pd.Series(b)).prod() ** (TRADING_DAYS / max(len(b), 1)) - 1), 4)
        r = run_config(mkt, rank, CELL["n"], CELL["k"], 20, CELL["liq"])
        daily = r.pop("_daily")
        valid = np.isfinite(pd.Series(bench).rolling(SEG_WINDOW).apply(
            lambda x: (1 + x).prod() - 1).to_numpy())
        row = {"all_ann": r["ann_return"], "sharpe": r["ann_sharpe"]}
        for tag, sel in (("up", bench_seg & valid), ("down", ~bench_seg & valid)):
            d = np.asarray(daily)[sel]
            d = d[np.isfinite(d)]
            row[f"{tag}_ann"] = round(
                float((1 + pd.Series(d)).prod() ** (TRADING_DAYS / max(len(d), 1)) - 1), 4)
        out[m] = row
        print(f"[區間] {m:26s}｜全期 {row['all_ann']:+.1%}｜上升段 {row['up_ann']:+.1%}"
              f"｜下跌段 {row['down_ann']:+.1%}（{time.time()-t0:.0f}s）", flush=True)
    return out


if __name__ == "__main__":
    horizon_models = [f"{m}__lab{h}_p20" for m in ("ridge", "gbdt") for h in ("5d", "10d", "20d")]
    mamba_models = ["v2_kg_nomacro", "v2_kg_nomacro__head10d", "v2_kg", "v2_kg__head10d"]
    regime_models = ["v2_kg_nomacro", "v2_kg", "gru", "ridge", "gbdt"]

    res = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "cell": CELL, "freqs": FREQS, "seg_window": SEG_WINDOW}
    print("=" * 78 + "\nA. 訊號層：一份分數 × 三個評估 horizon\n" + "=" * 78, flush=True)
    res["signal"] = signal_layer(horizon_models + mamba_models)
    print("\n" + "=" * 78 + "\nB. 組合層：毛/淨 × 再平衡頻率\n" + "=" * 78, flush=True)
    res["portfolio"] = portfolio_layer(horizon_models + mamba_models)
    print("\n" + "=" * 78 + "\nC. 大盤上升段 / 下跌段（N=50 / k=1.5 / 20 日）\n" + "=" * 78,
          flush=True)
    res["regime"] = regime_split(regime_models)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 結果 → {OUT.name}", flush=True)
