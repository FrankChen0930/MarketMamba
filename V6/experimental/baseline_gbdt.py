"""
baseline_gbdt.py — 方向二 Step 3：階 2 GBDT baseline（LightGBM）
================================================================
協定：docs/baseline-experiment-protocol-draft-2026-07-11.md（v1.0 凍結）
  - 與階 1 完全同 harness：300 維特徵（附錄 A）、rank label、
    train 2012–2023（尾端 15% 交易日為 val）、test 2024-01~2026-06-02
  - 階 2 規格：LightGBM regressor 對 rank_5d / rank_20d（同 label 鐵律，
    不用 lambdarank——換目標函數就不是同一場對照）
  - 可解釋性：LightGBM 原生 pred_contrib（SHAP 值）+ gain importance

超參數（誠實記錄）：
  - 小網格 num_leaves {31,127} × min_data_in_leaf {500,2000}（共 4 組，
    控制多重測試負擔），val 日 IC 選組；其餘固定：lr=0.05、
    feature/bagging_fraction=0.8、lambda_l2=1.0、early stopping 100 輪（val l2）
  - 選定後以 best_iteration 在 full-train（fit+val）重訓，test 全程未參與

評估（含 2026-07-13 排查報告 §5 建議的標準輸出）：
  - 日 Spearman IC + ICIR + Newey-West t（vs 實際 Alpha）
  - 分層 IC（流動性/市值三分組，headline 必須配分層數字引用）
  - Top50 組合（協定 §7）成本 ×1/×2 + 等權宇宙基準對照

執行（repo 根目錄、Windows）：
  python V6/experimental/baseline_gbdt.py              # 主切分（含 5d/20d）
  python V6/experimental/baseline_gbdt.py --wf-only    # 之後另跑 expanding WF（耗時）
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import experimental.baseline_common as bc                      # noqa: E402
from experimental.baseline_common import (                     # noqa: E402
    PROTOCOL, all_feature_names, daily_spearman_ic, ic_summary, load_xy,
    portfolio_backtest,
)
from experimental.baseline_ic_diagnosis import _grouped_ic, _liquidity_lut  # noqa: E402

RESULT_PATH = _THIS_DIR / "result" / "baseline_gbdt_result.json"
TRAIN_STRIDE = 2

FIXED_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "verbosity": -1,
    "seed": 20260713,
    "num_threads": 0,
}
GRID = [{"num_leaves": nl, "min_data_in_leaf": md}
        for nl in (31, 127) for md in (500, 2000)]
MAX_ROUNDS = 2000
EARLY_STOP = 100


def _mean_ic(dates, scores, realized) -> float:
    ic = daily_spearman_ic(dates, scores, realized)
    return float(ic.mean()) if len(ic) else float("nan")


def _fmt_ic(s: dict) -> str:
    return (f"mean IC {s['mean_ic']:+.4f} | ICIR {s['icir']} | IC>0 {s['pct_pos']:.0%} | "
            f"t(NW) {s['t_newey_west']} | {s['n_days']} 天")


# ============================================================
# 主切分
# ============================================================
def run_main(skip_20d: bool = False) -> dict:
    t0 = time.time()
    P = PROTOCOL
    print(f"[load] train {P['TRAIN_START']} → {P['TRAIN_END']}（stride={TRAIN_STRIDE}）...", flush=True)
    tr = load_xy(P["TRAIN_START"], P["TRAIN_END"], day_stride=TRAIN_STRIDE)
    print(f"[load] {tr['X'].shape[0]:,} 列 × {tr['X'].shape[1]} 維", flush=True)
    print(f"[load] test {P['TEST_START']} → {P['TEST_END']}（stride=1）...", flush=True)
    te = load_xy(P["TEST_START"], P["TEST_END"], day_stride=1)
    print(f"[load] {te['X'].shape[0]:,} 列 × {te['X'].shape[1]} 維", flush=True)

    dates_tr = pd.DatetimeIndex(tr["dates"])
    train_days = np.sort(dates_tr.unique())
    val_days = set(train_days[int(len(train_days) * (1 - P["VAL_RATIO"])):])
    val_mask = dates_tr.isin(val_days)
    fit_mask = ~val_mask
    print(f"[split] train 天數 {len(train_days)}（val 自 {pd.Timestamp(min(val_days)).date()} 起，"
          f"{len(val_days)} 天）", flush=True)

    names = all_feature_names()
    results = {"models": {}}
    horizons = [(5, "rank_5d", "alpha_5d")] + ([] if skip_20d else [(20, "rank_20d", "alpha_20d")])
    for horizon, label_col, alpha_col in horizons:
        y = tr[label_col]
        ok = ~np.isnan(y)
        fit_m, val_m, trn_m = fit_mask & ok, val_mask & ok, ok
        print(f"\n===== horizon {horizon}d（label={label_col}）=====", flush=True)
        print(f"[rows] fit {fit_m.sum():,} | val {val_m.sum():,} | full-train {trn_m.sum():,}", flush=True)

        Xf, yf = tr["X"][fit_m], y[fit_m].astype(np.float64)
        Xv = tr["X"][val_m]
        dv, av = tr["dates"][val_m], tr[alpha_col][val_m]

        # ── 網格：val l2 early-stop，val 日 IC 選組 ──
        sweep, best = [], None
        for gi, gp in enumerate(GRID):
            tg = time.time()
            params = {**FIXED_PARAMS, **gp}
            ds_fit = lgb.Dataset(Xf, label=yf, free_raw_data=True)
            ds_val = lgb.Dataset(Xv, label=y[val_m].astype(np.float64), reference=ds_fit)
            booster = lgb.train(params, ds_fit, num_boost_round=MAX_ROUNDS,
                                valid_sets=[ds_val], valid_names=["val"],
                                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            n_best = booster.best_iteration
            val_ic = _mean_ic(dv, booster.predict(Xv, num_iteration=n_best), av)
            rec = {"num_leaves": gp["num_leaves"], "min_data_in_leaf": gp["min_data_in_leaf"],
                   "best_iteration": int(n_best),
                   "val_l2": round(float(booster.best_score["val"]["l2"]), 6),
                   "val_ic": round(val_ic, 4)}
            sweep.append(rec)
            print(f"[gbdt] grid {gi+1}/{len(GRID)} leaves={gp['num_leaves']:>3} "
                  f"min_leaf={gp['min_data_in_leaf']:>4} → {n_best:>4} 輪 | "
                  f"val IC {val_ic:+.4f} | val l2 {rec['val_l2']:.5f} "
                  f"({time.time()-tg:.0f}s)", flush=True)
            del booster, ds_fit, ds_val
            gc.collect()
        best = max(sweep, key=lambda r: (r["val_ic"] if not np.isnan(r["val_ic"]) else -9))
        print(f"[gbdt] best：leaves={best['num_leaves']} min_leaf={best['min_data_in_leaf']} "
              f"iters={best['best_iteration']}（val IC {best['val_ic']:+.4f}）", flush=True)
        del Xf, yf, Xv
        gc.collect()

        # ── full-train（fit+val）以 best_iteration 重訓 ──
        tg = time.time()
        params = {**FIXED_PARAMS, "num_leaves": best["num_leaves"],
                  "min_data_in_leaf": best["min_data_in_leaf"]}
        ds_full = lgb.Dataset(tr["X"][trn_m], label=y[trn_m].astype(np.float64), free_raw_data=True)
        final = lgb.train(params, ds_full, num_boost_round=best["best_iteration"])
        print(f"[gbdt] full-train 重訓完成（{best['best_iteration']} 輪，{time.time()-tg:.0f}s）", flush=True)
        del ds_full
        gc.collect()

        # ── test 評估 ──
        scores = final.predict(te["X"])
        okt = ~np.isnan(te[alpha_col])
        ic = daily_spearman_ic(te["dates"][okt], scores[okt], te[alpha_col][okt])
        res = {"best_params": {k: best[k] for k in ("num_leaves", "min_data_in_leaf", "best_iteration")},
               "grid_sweep": sweep,
               "test_ic": ic_summary(ic, horizon=horizon)}
        print(f"[gbdt] test：{_fmt_ic(res['test_ic'])}", flush=True)

        # 分層 IC（排查 §5 標準輸出）
        df = pd.DataFrame({"Date": te["dates"], "stock_id": te["stock_ids"],
                           "s": scores, "r": te[alpha_col]}).dropna(subset=["r"])
        liq = _liquidity_lut(P["TEST_START"], P["TEST_END"])
        df = df.merge(liq, on=["Date", "stock_id"], how="left")
        df["liq_bin"] = df.groupby("Date")["adv20"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 3, labels=["L1_小量", "L2_中量", "L3_大量"]))
        res["test_ic_by_liquidity"] = _grouped_ic(df.dropna(subset=["liq_bin"]), "liq_bin")
        print("[gbdt] IC by 流動性：" + " | ".join(
            f"{k}: {v['mean_ic']:+.4f}" for k, v in sorted(res["test_ic_by_liquidity"].items())), flush=True)

        # 可解釋性：gain importance + 原生 SHAP（抽樣 30k test 列）
        gain = final.feature_importance(importance_type="gain")
        order = np.argsort(-gain)[:20]
        res["top_gain_importance"] = [{"feature": names[i], "gain_pct": round(float(gain[i] / gain.sum()), 4)}
                                      for i in order]
        rng = np.random.default_rng(0)
        samp = rng.choice(len(te["X"]), size=min(30_000, len(te["X"])), replace=False)
        contrib = final.predict(te["X"][samp], pred_contrib=True)[:, :-1]   # 末欄為期望值
        mean_abs = np.abs(contrib).mean(axis=0)
        order = np.argsort(-mean_abs)[:20]
        res["top_shap"] = [{"feature": names[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
                           for i in order]
        print("[gbdt] SHAP top10：" + ", ".join(names[i] for i in order[:10]), flush=True)
        del contrib
        gc.collect()

        # 組合層（5d only）：成本 ×1/×2 + 等權宇宙基準
        if horizon == 5:
            print("[gbdt] 組合回測（Top50 等權、5 日再平衡）...", flush=True)
            res["test_portfolio"] = portfolio_backtest(te["dates"], te["stock_ids"], scores)
            cb, cs = PROTOCOL["COST_BUY"], PROTOCOL["COST_SELL"]
            bc.PROTOCOL["COST_BUY"], bc.PROTOCOL["COST_SELL"] = cb * 2, cs * 2
            res["test_portfolio_cost_x2"] = portfolio_backtest(te["dates"], te["stock_ids"], scores)
            bc.PROTOCOL["COST_BUY"], bc.PROTOCOL["COST_SELL"] = cb, cs
            res["benchmark_equal_weight_all"] = portfolio_backtest(
                te["dates"], te["stock_ids"], np.zeros(len(te["dates"])), top_n=10 ** 6)
            pf = res["test_portfolio"]
            print(f"[gbdt] 組合：年化 {pf['ann_return']:+.1%} | Sharpe {pf['ann_sharpe']} | "
                  f"MDD {pf['max_drawdown']:.1%} | 換手 {pf['avg_turnover_per_rebalance']:.0%}/次"
                  + (f" | 對 TWII 超額 {pf['excess_vs_twii']:+.1%}" if "excess_vs_twii" in pf else ""),
                  flush=True)
            print(f"[gbdt] 成本×2：年化 {res['test_portfolio_cost_x2']['ann_return']:+.1%} | "
                  f"等權宇宙基準：總報酬 {res['benchmark_equal_weight_all']['total_return']:+.1%}", flush=True)
            final.save_model(str(_THIS_DIR / "result" / "gbdt_5d.txt"))

        results["models"][f"{horizon}d"] = res
        del final, scores
        gc.collect()

    results["protocol"] = {k: PROTOCOL[k] for k in ("TRAIN_START", "TRAIN_END", "TEST_START",
                                                    "TEST_END", "VAL_RATIO", "TOP_N",
                                                    "REBALANCE_DAYS", "COST_BUY", "COST_SELL")}
    results["fixed_params"] = FIXED_PARAMS
    results["notes"] = [
        f"訓練列 day_stride={TRAIN_STRIDE}（同階 1）；test IC 用全部交易日",
        "label=rank（同一 label 鐵律）、objective=regression L2——不用 lambdarank",
        "網格僅 4 組（控制多重測試負擔），val 日 IC 選組、early stopping 用 val l2",
        "分層 IC / 成本敏感度 / 等權宇宙基準為 2026-07-13 排查後的標準輸出（引用需分層）",
        "存活者偏差 D3 未修復：絕對數字偏高，四階相對比較公平（協定 §2）",
    ]
    results["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 結果已存 {RESULT_PATH}（總耗時 {(time.time()-t0)/60:.1f} 分）", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("階 2（GBDT）vs 已知數字（5d test IC）", flush=True)
    print("=" * 72, flush=True)
    print(f"  Mamba v6_short（同 harness） +0.0870", flush=True)
    print(f"  Ridge 300 維                +0.1015", flush=True)
    s = results["models"]["5d"]["test_ic"]
    print(f"  GBDT  300 維                {s['mean_ic']:+.4f}（{_fmt_ic(s)}）", flush=True)
    return results


# ============================================================
# Expanding Walk-Forward（協定方案 B；耗時，另跑）
# ============================================================
def run_wf() -> None:
    if not RESULT_PATH.exists():
        raise SystemExit("請先跑主切分（無 --wf-only）取得 best params")
    main_res = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    bp = main_res["models"]["5d"]["best_params"]
    params = {**FIXED_PARAMS, "num_leaves": bp["num_leaves"],
              "min_data_in_leaf": bp["min_data_in_leaf"]}
    n_rounds = bp["best_iteration"]
    print(f"[WF] expanding walk-forward（GBDT, {params['num_leaves']} leaves, "
          f"{n_rounds} 輪, stride={TRAIN_STRIDE}）...", flush=True)

    P = PROTOCOL
    tr = load_xy(P["TRAIN_START"], P["TEST_END"], day_stride=TRAIN_STRIDE)
    y = tr["rank_5d"]
    ok = ~np.isnan(y)
    dates = pd.DatetimeIndex(tr["dates"])

    folds = []
    ts = pd.Timestamp("2015-01-01")
    end_limit = pd.Timestamp(P["TEST_END"])
    t0 = time.time()
    while ts < end_limit:
        te_end = min(ts + pd.DateOffset(months=6), end_limit)
        m_tr = ok & np.asarray(dates < ts)
        m_te = ok & np.asarray(dates >= ts) & np.asarray(dates < te_end)
        if m_te.sum() > 0 and m_tr.sum() > 100_000:
            tg = time.time()
            ds = lgb.Dataset(tr["X"][m_tr], label=y[m_tr].astype(np.float64), free_raw_data=True)
            booster = lgb.train(params, ds, num_boost_round=n_rounds)
            ic = daily_spearman_ic(tr["dates"][m_te], booster.predict(tr["X"][m_te]), tr["alpha_5d"][m_te])
            folds.append({"test_start": str(ts.date()), "test_end": str(te_end.date()),
                          "n_train_rows": int(m_tr.sum()), "n_days": int(len(ic)),
                          "mean_ic": round(float(ic.mean()), 4)})
            print(f"[WF] {ts.date()}~{te_end.date()}  train {int(m_tr.sum()):>9,} 列  "
                  f"IC {folds[-1]['mean_ic']:+.4f}（{len(ic)} 天，{time.time()-tg:.0f}s）", flush=True)
            del booster, ds
            gc.collect()
        ts += pd.DateOffset(months=3)

    fold_ics = np.array([f["mean_ic"] for f in folds])
    main_res["walk_forward_gbdt_5d"] = {
        "params": {k: params[k] for k in ("num_leaves", "min_data_in_leaf")},
        "num_boost_round": n_rounds, "n_folds": len(folds),
        "mean_fold_ic": round(float(fold_ics.mean()), 4),
        "fold_ic_std": round(float(fold_ics.std()), 4),
        "pct_folds_positive": round(float((fold_ics > 0).mean()), 3),
        "min_fold_ic": round(float(fold_ics.min()), 4),
        "max_fold_ic": round(float(fold_ics.max()), 4),
        "folds": folds,
        "note": "params/輪數沿用主切分 val 選定值（同階 1 慣例：早期 fold 對此有輕微樂觀性，已知妥協）",
    }
    RESULT_PATH.write_text(json.dumps(main_res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[WF] 完成：{len(folds)} folds | mean IC {fold_ics.mean():+.4f} | "
          f"正 fold {float((fold_ics > 0).mean()):.0%} | 範圍 [{fold_ics.min():+.4f}, {fold_ics.max():+.4f}] "
          f"| 總耗時 {(time.time()-t0)/60:.1f} 分", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf-only", action="store_true", help="只跑 expanding WF（需先有主切分結果）")
    ap.add_argument("--skip-20d", action="store_true")
    args = ap.parse_args()
    if args.wf_only:
        run_wf()
    else:
        run_main(skip_20d=args.skip_20d)
