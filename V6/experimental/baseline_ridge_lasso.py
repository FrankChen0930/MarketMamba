"""
baseline_ridge_lasso.py — 方向二 Step 2：階 1 線性 baseline（Ridge / Lasso）
=============================================================================
協定：docs/baseline-experiment-protocol-draft-2026-07-11.md（v1.0 凍結）
  - 單一切分（主）：train 2012-01-01~2023-12-31（尾端 15% 交易日為 val 選 α）
                    test 2024-01-01~2026-06-02（與 Phase 3 harness 同窗）
  - Expanding WF（輔，協定方案 B）：2015 起、測試窗 6 個月、步進 3 個月（Ridge only）
  - label：per-date pct-rank 置中的 Alpha_5d（主）/ Alpha_20d（副）
  - 特徵：300 維（59 base + 241 lag/rolling/momentum，見 baseline_common 附錄規格）
  - 評估：日 Spearman IC（vs 實際 Alpha）+ Newey-West t + Top50 等權 5 日再平衡含成本

實作說明（誠實記錄，寫進輸出 JSON 的 notes）：
  - 訓練列用 day_stride=2（隔日抽樣）：5d 前瞻窗高度重疊、相鄰日樣本冗餘，
    抽樣減記憶體一半、資訊損失小；**test 的 IC 評估用全部交易日（stride=1）**。
  - Ridge 用標準化代數 Gram 閉式解（G、Xᵀy 一次累積，α 掃描免重算），
    scaler 永遠來自該次 fit 的列（fold 內），無跨期洩漏。
  - Lasso 用 sklearn coordinate descent（precompute Gram），α 由 val IC 選。
  - 與 Phase 3 harness 一致：train/test 邊界不做 5d purge（邊界 5 天 label 重疊，
    佔 12 年訓練期比例可忽略；為可比性維持同慣例）。

執行（repo 根目錄）：
  python V6/experimental/baseline_ridge_lasso.py            # 需先跑 baseline_common.py --build
  python V6/experimental/baseline_ridge_lasso.py --skip-wf  # 跳過 walk-forward 輔助
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from experimental.baseline_common import (   # noqa: E402（內含 59 維 config 自切）
    BASE_PATH, PROTOCOL, all_feature_names, build_base_matrix, build_derived,
    daily_spearman_ic, ic_summary, load_xy, portfolio_backtest,
)

RESULT_PATH = _THIS_DIR / "result" / "baseline_ridge_lasso_result.json"
RIDGE_ALPHAS = np.logspace(1, 7, 13)          # 標準化 Gram 尺度下（diag≈n），對應 α/n ∈ [~1e-5, ~3]
LASSO_ALPHAS = np.logspace(-5, -2.5, 8)       # sklearn 1/(2n) 目標函數尺度
TRAIN_STRIDE = 2
BLOCK = 500_000                                # Gram 累積的分塊列數（float64 轉換的暫存上限）


# ============================================================
# Ridge：標準化代數 Gram 閉式解
# ============================================================
def gram_stats(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> dict:
    """累積 G=XᵀX、s=Σx、b=Xᵀy、sy=Σy、n（float64、分塊，避免整份 float64 複製）。"""
    p = X.shape[1]
    G = np.zeros((p, p), dtype=np.float64)
    s = np.zeros(p, dtype=np.float64)
    b = np.zeros(p, dtype=np.float64)
    sy, n = 0.0, 0
    idx = np.flatnonzero(rows)
    for i in range(0, len(idx), BLOCK):
        blk = X[idx[i:i + BLOCK]].astype(np.float64)
        yb = y[idx[i:i + BLOCK]].astype(np.float64)
        G += blk.T @ blk
        s += blk.sum(axis=0)
        b += blk.T @ yb
        sy += yb.sum()
        n += len(blk)
    return {"G": G, "s": s, "b": b, "sy": sy, "n": n}


def stats_add(a: dict, c: dict) -> dict:
    return {"G": a["G"] + c["G"], "s": a["s"] + c["s"], "b": a["b"] + c["b"],
            "sy": a["sy"] + c["sy"], "n": a["n"] + c["n"]}


def ridge_solve(st: dict, alpha: float) -> tuple[np.ndarray, float, np.ndarray]:
    """在「fit 列自己的標準化空間」解 ridge。回傳 (w_raw, intercept, w_std)：
    預測 = X @ w_raw + intercept（等價於 ((x-μ)/σ)·w_std + ȳ）。"""
    n, mu = st["n"], st["s"] / st["n"]
    var = np.maximum(np.diag(st["G"]) / n - mu ** 2, 1e-12)
    sigma = np.sqrt(var)
    Gc = (st["G"] - np.outer(st["s"], st["s"]) / n) / np.outer(sigma, sigma)
    bc = (st["b"] - st["s"] * (st["sy"] / n)) / sigma
    w_std = np.linalg.solve(Gc + alpha * np.eye(len(mu)), bc)
    w_raw = w_std / sigma
    intercept = float(st["sy"] / n - mu @ w_raw)
    return w_raw.astype(np.float64), intercept, w_std


def mean_daily_ic(dates, scores, realized) -> float:
    ic = daily_spearman_ic(dates, scores, realized)
    return float(ic.mean()) if len(ic) else float("nan")


# ============================================================
# 主流程
# ============================================================
def run(skip_wf: bool = False) -> dict:
    t0 = time.time()
    names = all_feature_names()
    P = PROTOCOL

    # ── 載入：訓練+WF 用 stride-2（2012 → 2026-06），test 評估用全解析度 ──
    print(f"[load] train span（stride={TRAIN_STRIDE}）{P['TRAIN_START']} → {P['TEST_END']} ...", flush=True)
    tr = load_xy(P["TRAIN_START"], P["TEST_END"], day_stride=TRAIN_STRIDE)
    print(f"[load] {tr['X'].shape[0]:,} 列 × {tr['X'].shape[1]} 維 "
          f"({tr['X'].nbytes/2**30:.2f} GB, float32)", flush=True)
    print(f"[load] test（stride=1）{P['TEST_START']} → {P['TEST_END']} ...", flush=True)
    te = load_xy(P["TEST_START"], P["TEST_END"], day_stride=1)
    print(f"[load] {te['X'].shape[0]:,} 列 × {te['X'].shape[1]} 維 "
          f"({te['X'].nbytes/2**30:.2f} GB)", flush=True)

    dates_tr = pd.DatetimeIndex(tr["dates"])
    train_mask_all = dates_tr <= pd.Timestamp(P["TRAIN_END"])
    train_days = np.sort(dates_tr[train_mask_all].unique())
    val_days = set(train_days[int(len(train_days) * (1 - P["VAL_RATIO"])):])
    val_mask = train_mask_all & dates_tr.isin(val_days)
    fit_mask = train_mask_all & ~dates_tr.isin(val_days)
    print(f"[split] train 天數 {len(train_days)}（fit {len(train_days)-len(val_days)} + val {len(val_days)}，"
          f"val 自 {pd.Timestamp(min(val_days)).date()} 起）", flush=True)

    results = {"models": {}}
    for horizon, label_col, alpha_col in ((5, "rank_5d", "alpha_5d"), (20, "rank_20d", "alpha_20d")):
        y = tr[label_col]
        ok = ~np.isnan(y)
        fit_m, trn_m, val_m = fit_mask & ok, train_mask_all & ok, val_mask & ok
        print(f"\n===== horizon {horizon}d（label={label_col}）=====", flush=True)
        print(f"[rows] fit {fit_m.sum():,} | val {val_m.sum():,} | full-train {trn_m.sum():,}", flush=True)

        # ---------- Ridge ----------
        tg = time.time()
        st_fit = gram_stats(tr["X"], y, fit_m)
        print(f"[ridge] fit Gram 累積完成（{time.time()-tg:.0f}s）", flush=True)
        Xv, dv, av = tr["X"][val_m], tr["dates"][val_m], tr[alpha_col][val_m]
        val_ics = {}
        for a in RIDGE_ALPHAS:
            w_raw, c, _ = ridge_solve(st_fit, a)
            val_ics[a] = mean_daily_ic(dv, Xv @ w_raw + c, av)
        best_a = max(val_ics, key=lambda k: (val_ics[k] if not np.isnan(val_ics[k]) else -9))
        print("[ridge] α 掃描（val mean IC）: "
              + " | ".join(f"{a:.0e}:{v:+.4f}" for a, v in val_ics.items()), flush=True)
        print(f"[ridge] best α = {best_a:.0e}（val IC {val_ics[best_a]:+.4f}）", flush=True)

        st_full = stats_add(st_fit, gram_stats(tr["X"], y, val_m))
        w_raw, c, w_std = ridge_solve(st_full, best_a)
        te_ok = ~np.isnan(te[alpha_col])
        scores_te = te["X"] @ w_raw + c
        ic = daily_spearman_ic(te["dates"][te_ok], scores_te[te_ok], te[alpha_col][te_ok])
        ridge_res = {
            "best_alpha": float(best_a),
            "val_ic_by_alpha": {f"{a:.0e}": (round(v, 4) if not np.isnan(v) else None)
                                for a, v in val_ics.items()},
            "test_ic": ic_summary(ic, horizon=horizon),
            "top_coefficients": _top_coefs(w_std, names, k=20),
        }
        if horizon == 5:
            print("[ridge] 組合回測（Top50 等權、5 日再平衡、含成本）...", flush=True)
            ridge_res["test_portfolio"] = portfolio_backtest(te["dates"], te["stock_ids"], scores_te)
        print(f"[ridge] test：{_fmt_ic(ridge_res['test_ic'])}", flush=True)

        # ---------- Lasso ----------
        from sklearn.linear_model import Lasso
        tg = time.time()
        mu = st_full["s"] / st_full["n"]
        sigma = np.sqrt(np.maximum(np.diag(st_full["G"]) / st_full["n"] - mu ** 2, 1e-12))
        Zf = ((tr["X"][fit_m] - mu.astype(np.float32)) / sigma.astype(np.float32))
        yf = y[fit_m].astype(np.float64)
        Zv = ((Xv - mu.astype(np.float32)) / sigma.astype(np.float32))
        print(f"[lasso] 標準化 fit 副本 {Zf.nbytes/2**30:.2f} GB（{time.time()-tg:.0f}s）", flush=True)
        lasso_val, lasso_models = {}, {}
        model = Lasso(alpha=1.0, fit_intercept=False, precompute=True, max_iter=3000,
                      warm_start=True, tol=1e-5)
        for a in sorted(LASSO_ALPHAS, reverse=True):
            model.alpha = float(a)
            model.fit(Zf, yf)
            lasso_val[a] = mean_daily_ic(dv, Zv @ model.coef_, av)
            lasso_models[a] = (model.coef_.copy(), int((model.coef_ != 0).sum()))
            print(f"[lasso] α={a:.1e} 非零 {lasso_models[a][1]:3d}/300 val IC {lasso_val[a]:+.4f} "
                  f"({time.time()-tg:.0f}s 累計)", flush=True)
        del Zf, Zv
        gc.collect()
        best_la = max(lasso_val, key=lambda k: (lasso_val[k] if not np.isnan(lasso_val[k]) else -9))

        # 用選定 α 在 full-train 上重 fit（含 val）
        Zt = ((tr["X"][trn_m] - mu.astype(np.float32)) / sigma.astype(np.float32))
        final = Lasso(alpha=float(best_la), fit_intercept=False, precompute=True,
                      max_iter=5000, tol=1e-5)
        final.fit(Zt, y[trn_m].astype(np.float64))
        del Zt
        gc.collect()
        w_l = final.coef_ / sigma                       # 還原到原始特徵尺度
        c_l = float(-(mu @ w_l))
        scores_te_l = te["X"] @ w_l + c_l
        ic_l = daily_spearman_ic(te["dates"][te_ok], scores_te_l[te_ok], te[alpha_col][te_ok])
        lasso_res = {
            "best_alpha": float(best_la),
            "n_nonzero": int((final.coef_ != 0).sum()),
            "val_ic_by_alpha": {f"{a:.1e}": (round(v, 4) if not np.isnan(v) else None)
                                for a, v in lasso_val.items()},
            "test_ic": ic_summary(ic_l, horizon=horizon),
            "top_coefficients": _top_coefs(final.coef_, names, k=20),
        }
        if horizon == 5:
            lasso_res["test_portfolio"] = portfolio_backtest(te["dates"], te["stock_ids"], scores_te_l)
        print(f"[lasso] best α={best_la:.1e}（非零 {lasso_res['n_nonzero']}/300）| "
              f"test：{_fmt_ic(lasso_res['test_ic'])}", flush=True)

        results["models"][f"{horizon}d"] = {"ridge": ridge_res, "lasso": lasso_res}

    # ── Walk-Forward 輔助（協定方案 B，Ridge only、α 沿用主切分選定值）──
    if not skip_wf:
        results["walk_forward_ridge_5d"] = _walk_forward_ridge(
            tr, alpha=results["models"]["5d"]["ridge"]["best_alpha"])

    results.update(_meta(tr, te))
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 結果已存 {RESULT_PATH}（總耗時 {(time.time()-t0)/60:.1f} 分）", flush=True)
    _print_summary(results)
    return results


def _walk_forward_ridge(tr: dict, alpha: float) -> dict:
    """Expanding WF（config：2012 起、test 窗 6 個月、步進 3 個月、min train 3 年）。
    以「季」為 Gram 累積單位（fold 邊界對齊季首），每 fold 閉式重解、零額外記憶體。"""
    print(f"\n[WF] expanding walk-forward（Ridge, α={alpha:.0e}，stride={TRAIN_STRIDE} 評估）...", flush=True)
    y = tr["rank_5d"]
    ok = ~np.isnan(y)
    dates = pd.DatetimeIndex(tr["dates"])
    quarters = dates.to_period("Q")
    q_list = sorted(quarters.unique())
    q_stats = {}
    for q in q_list:
        rows = ok & np.asarray(quarters == q)
        if rows.sum() > 0:
            q_stats[q] = gram_stats(tr["X"], y, rows)
    print(f"[WF] 季 Gram 完成：{len(q_stats)} 季", flush=True)

    folds = []
    ts = pd.Timestamp("2015-01-01")                    # 2012 + min 3 年訓練
    end_limit = pd.Timestamp(PROTOCOL["TEST_END"])
    while ts < end_limit:
        te_end = min(ts + pd.DateOffset(months=6), end_limit)
        train_qs = [q for q in q_stats if q.start_time < ts]
        st = None
        for q in train_qs:
            st = q_stats[q] if st is None else stats_add(st, q_stats[q])
        w_raw, c, _ = ridge_solve(st, alpha)
        m = ok & np.asarray(dates >= ts) & np.asarray(dates < te_end)
        if m.sum() > 0:
            ic = daily_spearman_ic(tr["dates"][m], tr["X"][m] @ w_raw + c, tr["alpha_5d"][m])
            folds.append({"test_start": str(ts.date()), "test_end": str(te_end.date()),
                          "n_train_rows": int(st["n"]), "n_days": int(len(ic)),
                          "mean_ic": round(float(ic.mean()), 4)})
            print(f"[WF] {ts.date()}~{te_end.date()}  train {st['n']:>9,} 列  "
                  f"IC {folds[-1]['mean_ic']:+.4f}（{len(ic)} 天）", flush=True)
        ts += pd.DateOffset(months=3)

    fold_ics = np.array([f["mean_ic"] for f in folds])
    return {
        "alpha": float(alpha), "n_folds": len(folds),
        "mean_fold_ic": round(float(fold_ics.mean()), 4),
        "fold_ic_std": round(float(fold_ics.std()), 4),
        "pct_folds_positive": round(float((fold_ics > 0).mean()), 3),
        "min_fold_ic": round(float(fold_ics.min()), 4),
        "max_fold_ic": round(float(fold_ics.max()), 4),
        "folds": folds,
        "note": "IC 以 stride-2 抽樣日評估（每 fold 約 60 天）；Mamba 無對應 WF 數字（協定 §6 方案 B）",
    }


def _top_coefs(w: np.ndarray, names: list[str], k: int = 20) -> list[dict]:
    order = np.argsort(-np.abs(w))[:k]
    return [{"feature": names[i], "coef_std": round(float(w[i]), 5)} for i in order if w[i] != 0]


def _fmt_ic(s: dict) -> str:
    return (f"mean IC {s['mean_ic']:+.4f} | ICIR {s['icir']} | IC>0 {s['pct_pos']:.0%} | "
            f"t(NW) {s['t_newey_west']} | {s['n_days']} 天")


def _meta(tr: dict, te: dict) -> dict:
    return {
        "protocol": {k: PROTOCOL[k] for k in ("TRAIN_START", "TRAIN_END", "TEST_START", "TEST_END",
                                              "VAL_RATIO", "MIN_HISTORY_DAYS", "TOP_N",
                                              "REBALANCE_DAYS", "COST_BUY", "COST_SELL")},
        "n_features": len(all_feature_names()),
        "train_rows_stride2": int(tr["X"].shape[0]),
        "test_rows_full": int(te["X"].shape[0]),
        "notes": [
            f"訓練列以 day_stride={TRAIN_STRIDE}（隔日）抽樣（5d 重疊窗冗餘高）；test IC 用全部交易日",
            "與 Phase 3 harness 一致：train/test 邊界無 5d purge（邊界 label 重疊 5 天，佔比可忽略）",
            "Alpha label 實為 raw forward return：macro_raw 只有 TWII_Close 欄、"
            "_add_alpha_targets 的 TWII 減基準分支從未觸發（Colab 訓練 Mamba 亦同路徑）——"
            "per-date rank 對當日常數平移免疫，label 與 Mamba 完全一致、日 IC 不受影響；"
            "組合層 excess_vs_twii 另以 macro TWII_Close 計算、只算到 2026-04-24（macro 停更日）",
            "matrix 自 2010 起建（非 2005 全歷史）：macro ts z-score 以 2010 起 expanding——"
            "macro 為橫斷面常數，對排名類模型無影響（線性模型完全免疫、每日 IC 不變）",
            "lag/rolling 以『列』為單位 shift（clean 後偶發缺日近似交易日 shift），衍生特徵 NaN→0",
            "存活者偏差 D3 未修復：四階模型同資料相對比較公平，絕對數字偏高（協定 §2）",
        ],
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }


def _print_summary(r: dict) -> None:
    print("\n" + "=" * 72, flush=True)
    print("Baseline 階 1（線性）結果摘要 — 對照 Mamba v6_short 同 harness 5d IC 0.0870", flush=True)
    print("=" * 72, flush=True)
    for h in ("5d", "20d"):
        for m in ("ridge", "lasso"):
            s = r["models"][h][m]["test_ic"]
            extra = f" | 非零 {r['models'][h][m]['n_nonzero']}/300" if m == "lasso" else ""
            print(f"  {h:>3} {m:<6} {_fmt_ic(s)}{extra}", flush=True)
    for m in ("ridge", "lasso"):
        pf = r["models"]["5d"][m].get("test_portfolio")
        if pf:
            print(f"  5d {m} 組合：年化 {pf['ann_return']:+.1%} | Sharpe {pf['ann_sharpe']} | "
                  f"MDD {pf['max_drawdown']:.1%} | 換手 {pf['avg_turnover_per_rebalance']:.0%}/次"
                  + (f" | 對 TWII 超額 {pf['excess_vs_twii']:+.1%}（{pf.get('excess_window','')}）"
                     if "excess_vs_twii" in pf else ""), flush=True)
    wf = r.get("walk_forward_ridge_5d")
    if wf:
        print(f"  WF（Ridge 5d）：{wf['n_folds']} folds | mean IC {wf['mean_fold_ic']:+.4f} | "
              f"正 fold {wf['pct_folds_positive']:.0%} | 範圍 [{wf['min_fold_ic']:+.4f}, "
              f"{wf['max_fold_ic']:+.4f}]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wf", action="store_true")
    ap.add_argument("--build", action="store_true", help="缺快取時先建（等同 baseline_common --build）")
    args = ap.parse_args()
    if args.build or not BASE_PATH.exists():
        build_base_matrix()
        build_derived()
    run(skip_wf=args.skip_wf)
