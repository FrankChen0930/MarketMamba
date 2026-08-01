"""
f5_r_series.py — F5：一次一變因量 IC delta（R0–R5）
=============================================================================
目的：在把任何改動帶進 F6 重訓之前，先量出每一項**單獨**值多少 IC。
這一階是閘門——只有量出正貢獻的改動才會進入重訓。

為什麼不是照原計畫的 R0→R5 直線疊加
-----------------------------------------------------------------------------
原表把 R1/R2/R3 當成可以逐級疊加，但 **v1→v2 是一次跳四個變因**：
`fundamentals_v2`、宇宙過濾（v1 混進 23 支 ETF + 12 支興櫃）、矩陣起點、
以及**外資持股 9xxx 缺口修復**（v1 快取建於 2026-07-12，在修復之前）。
只靠 v1/v2 兩份快取量不出單一變因。

所以梯子改成**全部在 v2 內部爬**，只有真正寫進矩陣值的變因才另建快取：

    R0   v1 59d  train2012  旗標-  fund✗ neu-none  purge✗   harness 橋接（已完成 +0.0977）
    R0b  v2      train2012  旗標✗  fund✓ neu-none  purge✗   v1→v2 整包差
    R0c  v1like  train2012  旗標✗  fund✗ neu-none  purge✗   v1 規格**跑在新資料上**
    R0d  v1univ  train2012  旗標✗  fund✓ neu-none  purge✗   v2 規格 + v1 宇宙規則
    R1   v2      train2013  旗標✗  fund✓ neu-none  purge✗   Δ起點  = R1 − R0b
    R2   v2      train2013  旗標✓  fund✓ neu-none  purge✗   Δ旗標  = R2 − R1
    R3   v2_nofund train2013 旗標✗ fund✗ neu-none  purge✗   Δfund  = R1 − R3
    R3b  v2_nofund train2013 旗標✓ fund✗ neu-none  purge✗   診斷：旗標在舊財報下的增量
    R4a  v2_neuind train2013 旗標✓ fund✓ industry  purge✗   Δ中性化 = R4a − R2
    R5   v2      train2013  旗標✓  fund✓ neu-none  purge✓   Δpurge = R5 − R2（誠實基準）

R0c / R0d：把 −0.0079 拆開（2026-08-01 新增）
-----------------------------------------------------------------------------
`R0b − R0 = −0.0079` 是全 F5 最大的單項效應，原本無法歸因。原以為殘差是
「宇宙過濾 + 外資持股 9xxx 回補」，但實際查檔期後發現**還漏了更大的一項**：

    baseline_cache/baseline_base_59d.parquet（R0 的矩陣）   建於 2026-07-12
    prices_raw 切換成全歷史除權息還原                        2026-07-29

R0 的矩陣建立之後，raw 幾乎被翻修了一遍——B-3 全歷史除權息還原（報酬 std
0.0817→0.0349）、margin 券賣/券買 swap 修正、daytrade 全歷史重建、
holdings/dividends/TAIFEX/foreign_shareholding 直連回補、revenue/financials 回補。
（哪一項才是主因，見下面「實測後的修正」——**不是**還原。）
（`baseline_derived_roll` 雖於 07-27 重建，但它讀 07-12 的 chunk，只有 `Mom_*`
取自 07-27 的 prices_raw——仍在還原之前，故 v1 側口徑一致為「還原前」。）

所以三段拆解是：

    Δ資料修復   = R0c − R0    v1 規格、v1 宇宙，**唯一差別＝raw 資料**
    Δ宇宙過濾   = R0b − R0d   v2 規格，**唯一差別＝ETF/興櫃排除與否**
    Δfund+起點  = R0d − R0c   fund_v2 + MATRIX_START 2010→2011
                              （可與已量的 R1 − R3 = −0.0014 對照）

三者相加**必須等於** R0b − R0 → `report()` 尾端自動檢查（同 R3-all 的完整性檢查
精神）。對不上就代表某一級的設定跑錯了，而那種錯誤不會自己報出來。

⚠️ `R0c − R0` 是**整包**資料修復，不是單獨的外資持股回補。要再往下拆需要逐源
   回退舊 parquet 重建矩陣，成本不成比例，目前不做。

實測後的修正：主因**不是**除權息還原（2026-08-01）
-----------------------------------------------------------------------------
原本假設 −0.0053 主要來自價格還原（未還原時除息日的假跌幅是可預測的 → 製造
虛假 IC）。**兩個檢查都不支持這個假設**：

  ① 把 580 個測試日按「前瞻 5 日窗內的除權息事件數」分組，逐日 Δ 沒有單調關係，
     corr(事件數, Δ) = **+0.059**（方向還相反）。
  ② 逐欄比對舊/新矩陣（2024-01、39,754 共同列，每欄跨列 Spearman）：
     **價格類全部 ρ ≈ 0.985–0.989**（Close/Open/High/Low/MA_20/MA_60/ATR_14）——
     除權息還原是逐股的單調重新縮放，橫斷面**排名**幾乎不動。

真正變動大的是**籌碼欄**：
     Day_Trade_Volume  相異值 1 → 30,906（整維從死值變活值）
     Short_Sale ρ=0.78 / Short_Cover ρ=0.86 （margin 券賣/券買 swap 修正，33.2% 的列）
     Holdings_Large_Change ρ=0.91、Foreign_Holding_Pct ρ=0.95
Group C 的 Gross_Margin(0.05)/ROE(0.19) 雖然 ρ 很低，但在 fund_v2=False 下相異值
只有 10–14 個、近乎死值，實務影響小。Group D 那 8 個變動欄是**每日橫斷面常數**，
對 Ridge 的日內 rank IC **零影響**。

→ 所以 −0.0053 幾乎確定來自 **Group B 籌碼欄的正確性修正**，而那些是真的把錯的
   資料改對（券賣/券買標反、當沖量整欄為 0）。與 purge、Q4 延遲同類：
   **誠實化的代價，不是「這個改動不好」**。
   要指認到單一資料源，得逐源回退重建矩陣（換欄法在此不適用——兩份矩陣的列
   只有約 85% 對得上，不像 v2/nofund 那樣幾乎完全重疊）。
⚠️ R0 原本是從 `baseline_ridge_lasso_result.json` 匯入的、**沒有逐日 IC**，
   所有對 R0 的 Δ 都算不出 t 值。用 `MM_PROTOCOL=v1 --rung R0` 重跑一次即可補上
   （會覆蓋匯入的那筆），順便驗證能否重現 +0.0977。

R3 為什麼是「旗標關閉」
-----------------------------------------------------------------------------
`Avail_Financials` 是在 `_merge_fundamentals(fundamentals_v2=...)` **之後**才算的，
判準是 EPS/Book_Value/ROE/Gross_Margin 的 `notna()`——所以旗標會跟著 fund_v2 開關走。
但舊路徑下 Book_Value/ROE/Gross_Margin 是**死常數而不是 NaN**，`notna()` 仍為 True
→ 旗標會把「有一份壞掉的財報」判成「可得」，語意退化成近乎恆為 1。
把 R3 的旗標關掉、對照同樣關旗標的 R1，fund_v2 才是唯一變因。
R3b（同一份矩陣、旗標開）保留下來當診斷：比較 (R2−R1) 與 (R3b−R3)
就能直接量出上面那個語意失真有多大。

兩把尺的取捨（**跑之前先講定，避免看到數字才選規則**）
-----------------------------------------------------------------------------
  實務門檻 |Δ| ≥ 0.009  ← 效應大小（沿用 Phase 3-A 對訓練雜訊校準出的門檻）
  配對 NW t  |t| ≥ 2     ← 統計可靠度（同一批交易日的**配對**日 IC 差，比各自的 SE 靈敏）

  兩者都過          → 採納
  兩者都沒過        → 不採納（留較簡單的那一版）
  Δ大但 t 不顯著    → **不採納，列待覆核**：效應可能被少數幾天主導。
                       另看逐日 Δ>0 比例與去掉最極端 5 天後的 Δ_trim，過了才採納
  Δ小但 t 顯著      → 技術上為真，但**不採納進規格**：0.003 的 IC 換不到實務價值，
                       而每多一個變因就多一份訓練/推論不一致的風險（本專案已有
                       macro_norm、fundamentals_v2 兩個前例）。
                       例外：零成本項（不增維、不改推論路徑，例如切分方式）則採納

  R5（purge）不適用採納與否——它是正確性修正，一律採用，只誠實報出代價。

閘門看 **5d**（協定 §3 主 horizon、也對齊短線操作）；20d 每一級都會一起算（同一次
載入，零額外成本）當佐證。兩個 horizon 方向相反時標記 `horizon 不一致`，不自動採納。

IC 定義：每日橫斷面 Spearman（同 `baseline_common.daily_spearman_ic`），與 production
`ic_analyzer.py` / `trainer.compute_ic` 同一個統計量；差別只在最少股票數門檻
（此處 30、ic_analyzer 50、trainer 5）與相關的對象（此處 Alpha、dual_ic_analyzer 是 SQ）。

執行（repo 根目錄，PowerShell）
-----------------------------------------------------------------------------
    $env:MM_PROTOCOL="v2"; python V6/experimental/f5_r_series.py --rung R1 `
        --train-start 2013-01-01 --flags off
    $env:MM_VARIANT="nofund"; python V6/experimental/f5_r_series.py --rung R3 `
        --train-start 2013-01-01 --flags off --build
    python V6/experimental/f5_r_series.py --report        # 印梯子 + 配對檢定

結果按 rung 併進同一份 JSON → 中斷不用重跑已完成的級。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from experimental.baseline_common import (   # noqa: E402（內含 config 自切）
    BASE_PATH, CACHE_DIR, PROTOCOL, PROTOCOL_VERSION, VARIANT,
    all_feature_names, build_base_matrix, build_derived,
    daily_spearman_ic, ic_summary, load_xy, newey_west_t, portfolio_backtest,
)
# 數值一律沿用 R0 那支腳本的實作：確保級間差異來自變因、不是我另寫了一套 Ridge
from experimental.baseline_ridge_lasso import (  # noqa: E402
    RIDGE_ALPHAS, TRAIN_STRIDE, _top_coefs, gram_stats, mean_daily_ic, ridge_solve, stats_add,
)
from experimental.splitters import train_val_split_dates  # noqa: E402

RESULT_PATH = _THIS_DIR / "result" / "f5_r_series_result.json"

# 決策規則的兩個門檻（pre-registered，見檔頭）
DELTA_FLOOR = 0.009
T_FLOOR = 2.0

# 梯子的配對比較：(A, B, 說明) → Δ = A − B，A 是「有該變因」的那一級
COMPARISONS: list[tuple[str, str, str]] = [
    ("R0b", "R0",  "v1→v2 整包（資料修復 + 宇宙過濾 + fund_v2）→ 下面三項拆解，相加應等於它"),
    ("R0c", "R0",  "  ├ Δ 資料修復整包（唯一差別＝raw 資料 07-12 → 07-31；主因是籌碼欄，見檔頭）"),
    ("R0b", "R0d", "  ├ Δ 宇宙過濾（排除 ETF + 興櫃）"),
    ("R0d", "R0c", "  └ Δ fundamentals_v2 + MATRIX_START（對照 R1−R3 = −0.0014）"),
    ("R1",  "R0b", "Δ 訓練起點 2012 → 2013"),
    ("R2",  "R1",  "Δ 可得性旗標（在 fundamentals_v2 下）"),
    ("R1",  "R3",  "Δ fundamentals_v2（兩側皆關旗標）"),
    ("R3b", "R3",  "診斷：可得性旗標在**舊財報**下的增量（與 R2−R1 對照看語意失真）"),
    # fund_v2 子修正拆解：Δ = R1 − R3x，即「把該組欄位換回舊版」損失多少
    # （與 R1−R3 同方向：正 = 該子修正有貢獻，負 = 該子修正讓 IC 變差）
    ("R1",  "R3-q4delay", "  ├ 子修正 (a) Q4/年報 available_from 45→90 天（look-ahead 移除）"),
    ("R1",  "R3-fin3",    "  ├ 子修正 (b) Gross_Margin/ROE/Book_Value 取真值"),
    ("R1",  "R3-epssurp", "  ├ 子修正 (c) EPS_Surprise 改季頻"),
    ("R1",  "R3-perpbr",  "  ├ 子修正 (d) PER/PBR 自算 + 橫斷面校準"),
    ("R1",  "R3-all",     "  └ 四項一起換回（**應 ≈ R1−R3 = 拆解完整性檢查**）"),
    ("R4a", "R2",  "Δ 中性化 industry"),
    ("R4b", "R2",  "Δ 中性化 industry_mktcap"),
    ("R5",  "R2",  "Δ purge 60 + embargo 20（正確性修正，非採納與否）"),
    # GBDT 裁決：Ridge 對「旗標」與「中性化」都分不出雜訊，而旗標本來就要靠
    # 交互作用才有用（線性模型結構上用不到）→ 由樹模型定案這兩項規格
    ("G-R2",  "G-R1", "【GBDT】可得性旗標 → 決定 INPUT_DIM 66 還是 59"),
    ("G-R4a", "G-R2", "【GBDT】中性化 industry → 決定 NEUTRALIZE"),
]


# ============================================================
# 0) 換欄：把 `fundamentals_v2` 的整包差拆成子修正
# ============================================================
# `fundamentals_v2` 其實 gate 了四個獨立的修正，但只有一個布林旗標：
#   (a) Q4/年報 available_from 45 → 90 天（**移除 look-ahead**）
#   (b) Gross_Margin / ROE / Book_Value 改從 balance_sheet 取真值（原為死常數）
#   (c) EPS_Surprise 由日頻 pct_change(4) 改季頻
#   (d) PER / PBR 自算 + 當日橫斷面校準
# 要分開量，正規作法是各建一份矩陣（每份 27 分鐘）。但 v2 與 nofund 兩份矩陣
# **可按 (Date, stock_id) 對齊**（v2 是 nofund 的子集，只少 7 列 NaN），
# 所以直接拿 v2、把某一組欄位換成 nofund 的值就等價，且零重建。
#
# 欄位 → 子修正的對應（實測差異列比例佐證，見 docstring）：
#   (a) EPS 單獨 —— EPS 在舊路徑本來就正常，它的 12.2% 差異純粹來自 Q4 延遲
#   (b) Gross_Margin / ROE / Book_Value —— 舊路徑下 Book_Value 只有 1 個相異值
#   (c) EPS_Surprise
#   (d) PER / PBR
# ⚠️ (a) 也會影響三維的**時序**，但那三欄在舊路徑是死常數、時序無意義，
#    所以 (a) 與 (b) 在那三欄上本質不可分；EPS 是唯一乾淨的 (a) 觀測點。
SWAP_GROUPS: dict[str, list[str]] = {
    "q4delay":   ["EPS"],
    "fin3":      ["Gross_Margin", "ROE", "Book_Value"],
    "epssurp":   ["EPS_Surprise"],
    "perpbr":    ["PER", "PBR"],
    "all":       ["EPS", "Gross_Margin", "ROE", "Book_Value", "EPS_Surprise", "PER", "PBR"],
}


def apply_column_swap(out: dict, swap_from: str, cols: list[str],
                      date_from: str, date_to: str) -> None:
    """就地把 `out["X"]` 中 `cols`（含其 lag1/5/20）換成另一份快取的值。

    lag 欄位一起換：只換 base 而留著 v2 的 lag 等於同一個特徵有兩種來源混在一起。
    rolling 不必動——`ROLL_CORE` 不含任何 Group C 欄位。
    """
    from marketmamba.config import PROCESSED_DIR
    src_dir = PROCESSED_DIR / f"baseline_cache_v2_{swap_from}"
    if not src_dir.exists():
        raise SystemExit(f"❌ 換欄來源不存在：{src_dir}")

    names = out["feature_names"]
    keys = pd.DataFrame({"Date": pd.to_datetime(out["dates"]), "stock_id": out["stock_ids"]})
    filt = [("Date", ">=", pd.Timestamp(date_from)), ("Date", "<=", pd.Timestamp(date_to))]

    # base + 三個 lag part，各自取需要的欄位
    plan = [(src_dir / "baseline_base_66d.parquet", {c: c for c in cols})]
    for n in (1, 5, 20):
        plan.append((src_dir / f"baseline_derived_lag{n}.parquet",
                     {f"{c}_lag{n}": f"{c}_lag{n}" for c in cols}))

    n_swapped = 0
    for path, colmap in plan:
        want = [c for c in colmap if c in names]
        if not want:
            continue
        src = pd.read_parquet(path, columns=["Date", "stock_id"] + want, filters=filt)
        merged = keys.merge(src, on=["Date", "stock_id"], how="left", validate="one_to_one")
        miss = merged[want[0]].isna().sum()
        for c in want:
            j = names.index(c)
            v = merged[c].to_numpy(np.float32)
            np.nan_to_num(v, copy=False)           # 同 load_xy 慣例
            out["X"][:, j] = v
            n_swapped += 1
        print(f"[swap] {path.name}：{len(want)} 欄 ← {swap_from}"
              f"（對齊缺 {miss:,}/{len(merged):,} 列）", flush=True)
        del src, merged
        gc.collect()
    print(f"[swap] 共換掉 {n_swapped} 個欄位（{cols} 及其 lag1/5/20）", flush=True)


# ============================================================
# 0b) GBDT 裁決器
# ============================================================
# 用途：Ridge 定不了的兩項規格（可得性旗標、中性化）。旗標要靠**交互作用**
# 才有用，線性模型結構上用不到它——樹天生做交互作用，才是對的裁判。
#
# **刻意不重掃網格**：沿用 Step 3 在 v1 上選定的 leaves=127 / min_leaf=2000。
# 每個 config 各自掃網格的話，超參數會把特徵差異吸收掉，就不再是一次一變因。
# 輪數仍用 val early-stopping（那是資料的函數，不是自由度），並把 best_iteration
# 記進結果，任何差異都看得見。
GBDT_PARAMS = {
    "objective": "regression", "metric": "l2", "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_bin": 63, "verbosity": -1,
    "seed": 20260713, "num_threads": 0,
    "num_leaves": 127, "min_data_in_leaf": 2000,      # Step 3 定案值
}
GBDT_MAX_ROUNDS, GBDT_EARLY_STOP = 2000, 100


def _fit_gbdt(tr: dict, te: dict, y: np.ndarray, fit_m, val_m, trn_m,
              alpha_col: str, horizon: int, names: list[str]) -> tuple[dict, np.ndarray]:
    import lightgbm as lgb
    tg = time.time()
    ds_fit = lgb.Dataset(tr["X"][fit_m], label=y[fit_m].astype(np.float64), free_raw_data=True)
    ds_val = lgb.Dataset(tr["X"][val_m], label=y[val_m].astype(np.float64), reference=ds_fit)
    booster = lgb.train(GBDT_PARAMS, ds_fit, num_boost_round=GBDT_MAX_ROUNDS,
                        valid_sets=[ds_val], valid_names=["val"],
                        callbacks=[lgb.early_stopping(GBDT_EARLY_STOP, verbose=False)])
    n_best = int(booster.best_iteration)
    val_ic = mean_daily_ic(tr["dates"][val_m], booster.predict(tr["X"][val_m], num_iteration=n_best),
                           tr[alpha_col][val_m])
    print(f"[gbdt] early-stop {n_best} 輪 | val IC {val_ic:+.4f} "
          f"| val l2 {booster.best_score['val']['l2']:.5f}（{time.time()-tg:.0f}s）", flush=True)
    del ds_fit, ds_val, booster
    gc.collect()

    tg = time.time()
    ds_full = lgb.Dataset(tr["X"][trn_m], label=y[trn_m].astype(np.float64), free_raw_data=True)
    final = lgb.train(GBDT_PARAMS, ds_full, num_boost_round=n_best)
    print(f"[gbdt] full-train 重訓完成（{n_best} 輪，{time.time()-tg:.0f}s）", flush=True)
    del ds_full
    gc.collect()

    scores = final.predict(te["X"])
    okt = ~np.isnan(te[alpha_col])
    ic = daily_spearman_ic(te["dates"][okt], scores[okt], te[alpha_col][okt])

    # 旗標的 gain 佔比：直接回答「樹到底有沒有用到這些旗標」——
    # 比單看 IC delta 更有診斷力（IC 可能沒動，但旗標其實在當閘門用）
    gain = final.feature_importance(importance_type="gain")
    tot = float(gain.sum()) or 1.0
    avail_idx = [i for i, n in enumerate(names) if n.startswith("Avail_")]
    flag_gain = {names[i]: round(float(gain[i]) / tot, 5) for i in avail_idx}
    order = np.argsort(gain)[::-1][:15]
    if avail_idx:
        print(f"[gbdt] 旗標 gain 佔比合計 {sum(flag_gain.values()):.3%}："
              + " ".join(f"{k.replace('Avail_','')}={v:.3%}" for k, v in flag_gain.items()), flush=True)

    return {
        "model": "gbdt",
        "gbdt_params": {k: GBDT_PARAMS[k] for k in ("num_leaves", "min_data_in_leaf", "learning_rate")},
        "best_iteration": n_best,
        "val_ic": round(val_ic, 4),
        "test_ic": ic_summary(ic, horizon=horizon),
        "ic_by_day": {str(k)[:10]: round(float(v), 6) for k, v in ic.items()},
        "avail_flag_gain_share": flag_gain or None,
        "avail_flag_gain_total": round(sum(flag_gain.values()), 5) if flag_gain else None,
        "top_gain": [{"feature": names[i], "gain_share": round(float(gain[i]) / tot, 5)}
                     for i in order],
    }, scores


# ============================================================
# 1) 單級跑階
# ============================================================
def run_rung(rung: str, train_start: str, flags: bool, purge: bool,
             skip_portfolio: bool = False, swap_from: str | None = None,
             swap_group: str | None = None, model: str = "ridge") -> dict:
    t0 = time.time()
    names_all = all_feature_names()
    P = PROTOCOL
    train_start = train_start or P["TRAIN_START"]

    print(f"\n{'='*70}\n[rung {rung}] protocol={PROTOCOL_VERSION} variant={VARIANT or '-'} "
          f"cache={CACHE_DIR.name}\n           train_start={train_start} flags={'on' if flags else 'off'} "
          f"purge={'on' if purge else 'off'} fund_v2={P.get('FUNDAMENTALS_V2', False)} "
          f"neutralize={P.get('NEUTRALIZE', 'none')}\n{'='*70}", flush=True)

    # ── 特徵欄位遮罩（旗標開/關）──
    keep_col = np.array([not n.startswith("Avail_") or flags for n in names_all])
    names = [n for n, k in zip(names_all, keep_col) if k]
    n_avail = sum(1 for n in names if n.startswith("Avail_"))
    print(f"[cols] {len(names_all)} → {len(names)} 維（Avail_* 保留 {n_avail} 個）", flush=True)

    # ── 載入 ──
    print(f"[load] train span（stride={TRAIN_STRIDE}）{train_start} → {P['TEST_END']} ...", flush=True)
    tr = load_xy(train_start, P["TEST_END"], day_stride=TRAIN_STRIDE)
    if not keep_col.all():
        tr["X"] = np.ascontiguousarray(tr["X"][:, keep_col])
        gc.collect()
    print(f"[load] {tr['X'].shape[0]:,} 列 × {tr['X'].shape[1]} 維 "
          f"({tr['X'].nbytes/2**30:.2f} GB)", flush=True)
    te = load_xy(P["TEST_START"], P["TEST_END"], day_stride=1)
    if not keep_col.all():
        te["X"] = np.ascontiguousarray(te["X"][:, keep_col])
        gc.collect()
    print(f"[load] test {te['X'].shape[0]:,} 列 × {te['X'].shape[1]} 維", flush=True)

    # ── 換欄（子修正拆解）：train 與 test 都要換，否則訓練/測試特徵定義不一致 ──
    if swap_group:
        cols = SWAP_GROUPS[swap_group]
        print(f"[swap] 群組 {swap_group} ← {swap_from}：{cols}", flush=True)
        tr["feature_names"] = names
        te["feature_names"] = names
        apply_column_swap(tr, swap_from, cols, train_start, P["TEST_END"])
        apply_column_swap(te, swap_from, cols, P["TEST_START"], P["TEST_END"])

    # ── 切分 ──
    dates_tr = pd.DatetimeIndex(tr["dates"])
    in_train = dates_tr <= pd.Timestamp(P["TRAIN_END"])
    train_days = [str(d)[:10] for d in np.sort(dates_tr[in_train].unique())]
    ph, emb = int(P.get("PURGE_HORIZON", 60)), int(P.get("EMBARGO_DAYS", 20))

    # val 切點用與 baseline_ridge_lasso 完全相同的算式（int(n×0.85) 起算），
    # 差一天都會讓 R0b 對 R0 不可比。
    if purge:
        kept, _ = train_val_split_dates(train_days, P["TRAIN_END"], horizon=ph,
                                        embargo_days=emb, label="train↔test")
    else:
        kept = train_days
    cut = int(len(kept) * (1 - P["VAL_RATIO"]))
    val_days, fit_days = kept[cut:], kept[:cut]
    if purge:
        fit_days, _ = train_val_split_dates(fit_days, fit_days[-1], horizon=ph,
                                            embargo_days=emb, label="fit↔val")

    d_str = pd.Series(dates_tr.astype(str).str.slice(0, 10))
    val_mask = d_str.isin(set(val_days)).to_numpy()
    fit_mask = d_str.isin(set(fit_days)).to_numpy()
    trn_mask = fit_mask | val_mask
    print(f"[split] fit {len(fit_days)} 天 | val {len(val_days)} 天（{val_days[0]}…）| "
          f"full-train {len(fit_days)+len(val_days)}/{len(train_days)} 天", flush=True)

    out: dict = {}
    for horizon, label_col, alpha_col in ((5, "rank_5d", "alpha_5d"), (20, "rank_20d", "alpha_20d")):
        y = tr[label_col]
        ok = ~np.isnan(y)
        fit_m, trn_m, val_m = fit_mask & ok, trn_mask & ok, val_mask & ok
        print(f"\n----- horizon {horizon}d | fit {fit_m.sum():,} val {val_m.sum():,} "
              f"full {trn_m.sum():,} 列 -----", flush=True)

        if model == "gbdt":
            res, scores_te = _fit_gbdt(tr, te, y, fit_m, val_m, trn_m, alpha_col,
                                       horizon, names)
            if horizon == 5 and not skip_portfolio:
                print("[gbdt] 組合回測（Top50 等權、5 日再平衡、含成本）...", flush=True)
                res["test_portfolio"] = portfolio_backtest(te["dates"], te["stock_ids"], scores_te)
            s = res["test_ic"]
            print(f"[gbdt] test IC {s['mean_ic']:+.4f}（ICIR {s['icir']}, NW t {s['t_newey_west']}, "
                  f"{s['n_days']} 天, 正比例 {s['pct_pos']:.1%}）", flush=True)
            out[f"{horizon}d"] = res
            continue

        tg = time.time()
        st_fit = gram_stats(tr["X"], y, fit_m)
        Xv, dv, av = tr["X"][val_m], tr["dates"][val_m], tr[alpha_col][val_m]
        val_ics = {}
        for a in RIDGE_ALPHAS:
            w_raw, c, _ = ridge_solve(st_fit, a)
            val_ics[a] = mean_daily_ic(dv, Xv @ w_raw + c, av)
        best_a = max(val_ics, key=lambda k: (val_ics[k] if not np.isnan(val_ics[k]) else -9))
        print("[ridge] α 掃描（val mean IC）: "
              + " | ".join(f"{a:.0e}:{v:+.4f}" for a, v in val_ics.items()), flush=True)
        print(f"[ridge] best α = {best_a:.0e}（val IC {val_ics[best_a]:+.4f}，{time.time()-tg:.0f}s）",
              flush=True)

        st_full = stats_add(st_fit, gram_stats(tr["X"], y, val_m))
        w_raw, c, w_std = ridge_solve(st_full, best_a)
        te_ok = ~np.isnan(te[alpha_col])
        scores_te = te["X"] @ w_raw + c
        ic = daily_spearman_ic(te["dates"][te_ok], scores_te[te_ok], te[alpha_col][te_ok])

        res = {
            "best_alpha": float(best_a),
            "val_ic_by_alpha": {f"{a:.0e}": (round(v, 4) if not np.isnan(v) else None)
                                for a, v in val_ics.items()},
            "test_ic": ic_summary(ic, horizon=horizon),
            # 逐日 IC：級間比較要用**配對**日 IC 差才靈敏，不存就只能比兩個平均值
            "ic_by_day": {str(k)[:10]: round(float(v), 6) for k, v in ic.items()},
            "top_coefficients": _top_coefs(w_std, names, k=15),
        }
        if horizon == 5 and not skip_portfolio:
            print("[ridge] 組合回測（Top50 等權、5 日再平衡、含成本）...", flush=True)
            res["test_portfolio"] = portfolio_backtest(te["dates"], te["stock_ids"], scores_te)
        s = res["test_ic"]
        print(f"[ridge] test IC {s['mean_ic']:+.4f}（ICIR {s['icir']}, NW t {s['t_newey_west']}, "
              f"{s['n_days']} 天, 正比例 {s['pct_pos']:.1%}）", flush=True)
        out[f"{horizon}d"] = res

    entry = {
        "rung": rung,
        "config": {
            "protocol_version": PROTOCOL_VERSION,
            "variant": VARIANT or None,
            "cache_dir": CACHE_DIR.name,
            "base_matrix": BASE_PATH.name,
            "train_start": train_start,
            "train_end": P["TRAIN_END"],
            "test_start": P["TEST_START"],
            "test_end": P["TEST_END"],
            "availability_flags": bool(flags),
            "fundamentals_v2": bool(P.get("FUNDAMENTALS_V2", False)),
            "neutralize": P.get("NEUTRALIZE", "none"),
            "purge": bool(purge),
            "purge_horizon": ph if purge else None,
            "embargo_days": emb if purge else None,
            "n_features": len(names),
            "n_avail_flags": n_avail,
            "train_stride": TRAIN_STRIDE,
            "swap_from": swap_from,
            "swap_group": swap_group,
            "swap_cols": SWAP_GROUPS[swap_group] if swap_group else None,
            "model": model,
        },
        "rows": {"train_span": int(tr["X"].shape[0]), "test": int(te["X"].shape[0]),
                 "fit_days": len(fit_days), "val_days": len(val_days),
                 "train_days_available": len(train_days)},
        "models": out,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    _save(rung, entry)
    print(f"\n✅ [rung {rung}] 完成（{entry['elapsed_min']} 分）→ {RESULT_PATH.name}", flush=True)
    return entry


def _save(rung: str, entry: dict) -> None:
    """按 rung 併進同一份 JSON：中斷後不必重跑已完成的級。"""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_res = {}
    if RESULT_PATH.exists():
        all_res = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    all_res.setdefault("rungs", {})[rung] = entry
    all_res["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
    RESULT_PATH.write_text(json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# 2) R0 匯入（v1 的既有結果）
# ============================================================
def import_r0() -> None:
    """把 baseline_ridge_lasso 的 v1 結果收進梯子當 R0。

    ⚠️ 只有 test_ic 的**平均值**能匯入——那支腳本沒存逐日 IC，
    所以 R0 無法參與配對檢定（R0b−R0 只會有 Δ、沒有 t）。
    """
    src = _THIS_DIR / "result" / "baseline_ridge_lasso_result.json"
    if not src.exists():
        print(f"❌ 找不到 {src}", flush=True)
        return
    d = json.loads(src.read_text(encoding="utf-8"))
    entry = {
        "rung": "R0",
        "config": {
            "protocol_version": "v1", "variant": None, "cache_dir": "baseline_cache",
            "base_matrix": "baseline_base_59d.parquet",
            "train_start": d["protocol"]["TRAIN_START"], "train_end": d["protocol"]["TRAIN_END"],
            "test_start": d["protocol"]["TEST_START"], "test_end": d["protocol"]["TEST_END"],
            "availability_flags": False, "fundamentals_v2": False, "neutralize": "none",
            "purge": False, "purge_horizon": None, "embargo_days": None,
            "n_features": d["n_features"], "n_avail_flags": 0, "train_stride": 2,
        },
        "rows": {"train_span": d["train_rows_stride2"], "test": d["test_rows_full"]},
        "models": {h: {"best_alpha": d["models"][h]["ridge"]["best_alpha"],
                       "test_ic": d["models"][h]["ridge"]["test_ic"],
                       "ic_by_day": None,
                       **({"test_portfolio": d["models"][h]["ridge"]["test_portfolio"]}
                          if "test_portfolio" in d["models"][h]["ridge"] else {})}
                   for h in ("5d", "20d")},
        "notes": ["v1 快取建於 2026-07-12，早於整批資料修復（外資持股 9xxx 回補、"
                  "**prices_raw 全歷史除權息還原 2026-07-29**、margin swap、daytrade 重建…）"
                  " → 與 v2 各級並列時資料基礎不同（核對結果 F2、協定 §7）",
                  "來源腳本未存逐日 IC → 無法參與配對顯著性檢定。"
                  "用 `MM_PROTOCOL=v1 --rung R0 --train-start 2012-01-01 --flags off` "
                  "重跑可覆蓋這筆並補上逐日 IC"],
        "generated_at": d.get("generated_at"),
    }
    _save("R0", entry)
    print(f"✅ R0 已匯入：5d {entry['models']['5d']['test_ic']['mean_ic']:+.4f} / "
          f"20d {entry['models']['20d']['test_ic']['mean_ic']:+.4f}", flush=True)


# ============================================================
# 3) 報表：梯子 + 配對檢定 + 決策
# ============================================================
def paired_delta(a: dict, b: dict, horizon: int) -> dict | None:
    """同一批交易日的配對日 IC 差：Δ、NW t、Δ>0 比例、去掉最極端 5 天的 Δ_trim。"""
    ia, ib = a.get("ic_by_day"), b.get("ic_by_day")
    if not ia or not ib:
        return None
    common = sorted(set(ia) & set(ib))
    if len(common) < 30:
        return None
    d = np.array([ia[k] - ib[k] for k in common], dtype=np.float64)
    order = np.argsort(np.abs(d))[:-5]                 # 去掉 |Δ| 最大的 5 天
    return {
        "n_days": len(common),
        "delta": round(float(d.mean()), 4),
        "t_newey_west": round(newey_west_t(d, lag=horizon), 2),
        "pct_days_positive": round(float((d > 0).mean()), 3),
        "delta_trim5": round(float(d[order].mean()), 4),
    }


def verdict(delta: float, t: float | None, correctness_fix: bool = False) -> str:
    """檔頭 pre-registered 的決策規則。

    `correctness_fix=True`（purge）不套採納與否——它修的是洩漏，一律採用，
    只誠實報出代價。印成「不進規格」會在半年後被誤讀成「purge 不用做」。
    """
    if correctness_fix:
        return f"正確性修正，一律採用（代價 {delta:+.4f}）"
    big = abs(delta) >= DELTA_FLOOR
    sig = t is not None and not np.isnan(t) and abs(t) >= T_FLOOR
    if big and sig:
        return "採納" if delta > 0 else "拒絕（顯著變差）"
    if big and not sig:
        return "待覆核（幅度夠但不顯著 → 看 Δ_trim5 與正比例）"
    if sig and not big:
        return "顯著但幅度不足 → 不進規格（零成本項例外）"
    return "無效應"


def report() -> None:
    if not RESULT_PATH.exists():
        print(f"❌ 尚無結果：{RESULT_PATH}", flush=True)
        return
    all_res = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    rungs = all_res.get("rungs", {})

    print(f"\n{'='*104}\nF5 梯子（Ridge，每日橫斷面 Spearman IC；閘門看 5d，20d 為佐證）\n{'='*104}")
    hdr = (f"{'級':11s} {'快取／換欄':26s} {'起點':11s} {'旗標':5s} {'fund':5s} {'中性化':9s} "
           f"{'purge':6s} {'維':4s} {'5d IC':>8s} {'20d IC':>8s} {'年化':>7s}")
    print(hdr + "\n" + "-" * 104)
    for name in ["R0", "R0c", "R0d", "R0b", "R1", "R2", "R3", "R3b",
                 "R3-q4delay", "R3-fin3", "R3-epssurp", "R3-perpbr", "R3-all",
                 "R4a", "R4b", "R5", "G-R1", "G-R2", "G-R4a"]:
        e = rungs.get(name)
        if not e:
            continue
        c, m = e["config"], e["models"]
        pf = m["5d"].get("test_portfolio") or {}
        ann = f"{pf['ann_return']:+.1%}" if "ann_return" in pf else "—"
        tag = c["cache_dir"] if not c.get("swap_group") else \
            f"v2 ← {c['swap_from']}:{c['swap_group']}"
        print(f"{name:11s} {tag:26s} {c['train_start']:11s} "
              f"{'開' if c['availability_flags'] else '關':5s} "
              f"{'✓' if c['fundamentals_v2'] else '✗':5s} {c['neutralize']:9s} "
              f"{'開' if c['purge'] else '關':6s} {c['n_features']:4d} "
              f"{m['5d']['test_ic']['mean_ic']:+8.4f} {m['20d']['test_ic']['mean_ic']:+8.4f} {ann:>7s}")

    print(f"\n{'='*104}\n配對比較（Δ = A − B，同一批交易日的逐日 IC 差）\n{'='*104}")
    for A, B, desc in COMPARISONS:
        if A not in rungs or B not in rungs:
            continue
        print(f"\n▸ {A} − {B}：{desc}")
        for h, hz in (("5d", 5), ("20d", 20)):
            ma, mb = rungs[A]["models"][h], rungs[B]["models"][h]
            raw = ma["test_ic"]["mean_ic"] - mb["test_ic"]["mean_ic"]
            p = paired_delta(ma, mb, horizon=hz)
            if p is None:
                print(f"   {h:4s} Δ={raw:+.4f}（無逐日 IC，無法配對檢定）")
                continue
            v = verdict(p["delta"], p["t_newey_west"],
                        correctness_fix=(A == "R5")) if h == "5d" else ""
            print(f"   {h:4s} Δ={p['delta']:+.4f}  NW t={p['t_newey_west']:+6.2f}  "
                  f"Δ>0 {p['pct_days_positive']:.1%}  Δ_trim5={p['delta_trim5']:+.4f}  "
                  f"n={p['n_days']}  {v}")
        # horizon 一致性
        d5 = paired_delta(rungs[A]["models"]["5d"], rungs[B]["models"]["5d"], 5)
        d20 = paired_delta(rungs[A]["models"]["20d"], rungs[B]["models"]["20d"], 20)
        # 只在至少一個 horizon 的效應大到有意義時才警告：兩邊都是 ±0.0000 的
        # 雜訊翻號去示警，等於製造永遠不會消的假警報。
        if (d5 and d20 and np.sign(d5["delta"]) != np.sign(d20["delta"])
                and max(abs(d5["delta"]), abs(d20["delta"])) >= DELTA_FLOOR):
            print("   ⚠️ horizon 不一致（5d 與 20d 方向相反）→ 不自動採納，需人工判讀")

    decomposition_check(rungs)

    print(f"\n門檻：|Δ| ≥ {DELTA_FLOOR}（實務）且 |NW t| ≥ {T_FLOOR}（統計）→ 採納。"
          f"完整規則見本檔檔頭（跑之前已定）。")
    if "R0" in rungs and "R0b" in rungs and rungs["R0"]["models"]["5d"].get("ic_by_day") is None:
        print("⚠️ R0 是匯入的舊結果、沒有逐日 IC → 對 R0 的 Δ 都算不出 t。"
              "跑 `MM_PROTOCOL=v1 --rung R0 --train-start 2012-01-01 --flags off` 可補上。")


def decomposition_check(rungs: dict) -> None:
    """R0b−R0 的三段拆解完整性檢查。

    三段（資料修復 / 宇宙過濾 / fund_v2+起點）是把同一段路切成三刀，
    相加**必須**還原成整包差。對不上就代表某一級的設定跑錯了——而那類錯誤
    （例如變體沒吃到、train_start 打錯）**不會自己報出來**，只會安靜地
    給出一個看起來合理的數字。這一行就是專門擋它的。
    """
    need = ("R0", "R0b", "R0c", "R0d")
    if not all(r in rungs for r in need):
        missing = [r for r in need if r not in rungs]
        if "R0c" in rungs or "R0d" in rungs:
            print(f"\n（拆解完整性檢查：尚缺 {missing}，跑齊後自動顯示）")
        return

    print(f"\n{'='*104}\nR0b − R0 三段拆解完整性檢查\n{'='*104}")
    for h in ("5d", "20d"):
        def ic(r: str) -> float:
            return rungs[r]["models"][h]["test_ic"]["mean_ic"]
        whole = ic("R0b") - ic("R0")
        parts = [("資料修復（raw 07-12 → 07-31）", ic("R0c") - ic("R0")),
                 ("宇宙過濾（排除 ETF + 興櫃）",   ic("R0b") - ic("R0d")),
                 ("fund_v2 + MATRIX_START",       ic("R0d") - ic("R0c"))]
        tot = sum(p for _, p in parts)
        print(f"\n[{h}] 整包 R0b−R0 = {whole:+.4f}")
        for lbl, v in parts:
            share = f"{abs(v)/abs(whole):.0%}" if abs(whole) > 1e-9 else "—"
            print(f"        {lbl:32s} {v:+.4f}   （佔 {share}）")
        gap = tot - whole
        flag = "✓ 一致" if abs(gap) < 5e-4 else f"❌ 對不上（差 {gap:+.4f}）→ 先懷疑某一級的設定"
        print(f"        {'三段相加':32s} {tot:+.4f}   {flag}")

    r13 = None
    if "R1" in rungs and "R3" in rungs:
        r13 = rungs["R1"]["models"]["5d"]["test_ic"]["mean_ic"] - \
              rungs["R3"]["models"]["5d"]["test_ic"]["mean_ic"]
        print(f"\n對照：R1 − R3（fund_v2 在 train2013、兩側皆關旗標）= {r13:+.4f}。"
              f"\n      與上面「fund_v2 + MATRIX_START」的差額即 MATRIX_START 2010→2011 的效應"
              f"\n      （macro ts 暖機期不同；macro 是每日橫斷面常數，對 Ridge 的日內 rank IC"
              f"\n       理論上零影響，實際差額應接近 0，只剩 eligible 邊界的微小差異）。")


# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", help="級代號，例如 R1 / R2 / R3b")
    ap.add_argument("--train-start", default=None, help="覆寫 TRAIN_START（預設用 PROTOCOL）")
    ap.add_argument("--flags", choices=("on", "off"), default="on", help="可得性旗標是否納入特徵")
    ap.add_argument("--purge", action="store_true", help="套用 purge + embargo 切分")
    ap.add_argument("--swap-from", default="nofund", help="換欄來源變體（預設 nofund）")
    ap.add_argument("--swap-group", choices=list(SWAP_GROUPS),
                    help="把該組欄位（含 lag）換成來源變體的值，用於拆解 fundamentals_v2")
    ap.add_argument("--model", choices=("ridge", "gbdt"), default="ridge",
                    help="gbdt 用於裁決 Ridge 定不了的規格（旗標／中性化）")
    ap.add_argument("--build", action="store_true", help="快取不存在時先建矩陣")
    ap.add_argument("--skip-portfolio", action="store_true")
    ap.add_argument("--import-r0", action="store_true", help="匯入 v1 既有結果當 R0")
    ap.add_argument("--report", action="store_true", help="印梯子 + 配對檢定")
    a = ap.parse_args()

    if a.import_r0:
        import_r0()
    if a.rung:
        if a.build:
            build_base_matrix()
            build_derived()
        if not BASE_PATH.exists():
            raise SystemExit(f"❌ 矩陣不存在：{BASE_PATH}\n   加 --build 或先跑 "
                             f"MM_PROTOCOL={PROTOCOL_VERSION} MM_VARIANT={VARIANT or ''} "
                             f"python V6/experimental/baseline_common.py --build")
        run_rung(a.rung, a.train_start, flags=(a.flags == "on"), purge=a.purge,
                 skip_portfolio=a.skip_portfolio,
                 swap_from=a.swap_from, swap_group=a.swap_group, model=a.model)
    if a.report or not (a.rung or a.import_r0):
        report()
