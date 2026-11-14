"""
baseline_ic_diagnosis.py — Baseline IC 0.1015 異常值排查（診斷 D0–D5）
=======================================================================
對應：docs/baseline-ic-suspicion-review-2026-07-13.md 的診斷任務 1–5
     + 追加 D0（價格資料錯誤 → 機械性反轉）檢查
隔離原則：只讀 Data/processed_v6/（含 baseline_cache/），輸出只寫
         V6/experimental/result/baseline_ic_diagnosis_result.json。
         不動 production、不動 V6/models/、不改 baseline_common/baseline_ridge_lasso。

診斷項目：
  D0  價格資料錯誤掃描：漲跌幅超限（2015-06 前 ±7%、後 ±10%）、V 型單日對翻，
      並計算「排除可疑 (股,日) ±5 天窗」後的 test IC —— 若 IC 大跌，資料錯誤是主因
  D1  IC 依流動性（20 日均成交金額）三分組 + 依市值三分組（懷疑清單任務 1）
  D2  反轉特徵消融（Gram 子矩陣重解，免重算）：
      AB1 移除短窗價格反轉核心 / AB2 移除全部價格技術特徵（任務 2）
      + 單因子樸素 IC（-Return_5d 等，量出「純反轉」在這份資料裡值多少）
  D3  樸素基準：同一批股票等權持有 vs TWII（任務 3，拆 3a/3b）
  D4  摩擦成本敏感度：成本 ×2 / ×3、Top50 限流動性前 1/3（任務 4）
  D5  籌碼 point-in-time 抽驗：matrix Foreign_Net(t) 對 raw(t) vs raw(t-1)
      的橫斷面 Spearman —— 若對 t-1 更像，代表特徵被提前一天（任務 5）

執行（repo 根目錄、Windows）：
  python V6/experimental/baseline_ic_diagnosis.py
"""
from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import experimental.baseline_common as bc                      # noqa: E402
from experimental.baseline_common import (                     # noqa: E402
    BASE_PATH, PROTOCOL, all_feature_names, daily_spearman_ic, ic_summary,
    load_xy, portfolio_backtest, _load_raw, _load_twii,
)
from experimental.baseline_ridge_lasso import gram_stats, ridge_solve  # noqa: E402

RESULT_PATH = _THIS_DIR / "result" / "baseline_ic_diagnosis_result.json"
RIDGE_ALPHA = 10.0          # baseline_ridge_lasso_result.json 的 best_alpha（5d）
TRAIN_STRIDE = 2            # 與原 baseline 相同


# ============================================================
# 共用：載入 prices（同 baseline_common 的過濾慣例）
# ============================================================
def _load_prices(cols=("Date", "stock_id", "Close", "Volume")) -> pd.DataFrame:
    pr = _load_raw("prices_raw")
    pr = pr[pr["stock_id"].astype(str).str.match(r"^\d{4}$")]
    pr = pr.drop_duplicates(subset=["stock_id", "Date"], keep="last")
    pr = pr.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)
    return pr[list(cols)]


# ============================================================
# D0：價格資料錯誤掃描
# ============================================================
def diag0_price_errors(te: dict, scores: np.ndarray) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("D0 價格資料錯誤掃描（漲跌幅超限 / V 型對翻 / 排除後 IC）", flush=True)
    print("=" * 72, flush=True)
    pr = _load_prices()
    bad = pr[~(pr["Close"] > 0)]
    by_month = bad.groupby(bad["Date"].dt.to_period("M")).size()
    print(f"[D0] Close ≤ 0 或 NaN 的損壞列：{len(bad):,} 筆", flush=True)
    if len(bad):
        print("[D0] 損壞列 by 月：" + " | ".join(f"{m}:{n}" for m, n in by_month.tail(12).items()), flush=True)
    g = pr.groupby("stock_id", sort=False)["Close"]
    pr["ret"] = g.pct_change()

    limit = np.where(pr["Date"] < pd.Timestamp("2015-06-01"), 0.07, 0.10)
    tol = 0.008                                  # 容忍 tick 進位 / 微小誤差
    pr["over_limit"] = pr["ret"].abs() > (limit + tol)

    # V 型對翻：|r_t| 超限 且 隔日 |r_{t+1}| 也超限且方向相反（壞 tick 的典型足跡）
    r_next = g.pct_change().groupby(pr["stock_id"], sort=False).shift(-1)
    pr["v_spike"] = pr["over_limit"] & (r_next.abs() > (limit + tol)) & (np.sign(pr["ret"]) != np.sign(r_next))

    flagged = pr[pr["over_limit"]]
    by_year = flagged.groupby(flagged["Date"].dt.year).size()
    print(f"[D0] 全歷史超限報酬筆數：{len(flagged):,}（占 {len(flagged)/max(len(pr),1):.3%}）| "
          f"其中 V 型對翻 {int(pr['v_spike'].sum()):,} 筆", flush=True)
    print("[D0] 超限筆數 by 年：" + " | ".join(f"{y}:{n}" for y, n in by_year.items()), flush=True)

    top = flagged.reindex(flagged["ret"].abs().sort_values(ascending=False).index).head(15)
    print("[D0] |報酬| 最大的 15 筆（人工抽驗用）：", flush=True)
    for _, row in top.iterrows():
        print(f"     {row['stock_id']} {row['Date'].date()}  ret={row['ret']:+.1%}  Close={row['Close']}", flush=True)

    # ── 排除可疑窗後的 test IC ──
    # 對 test 每列 (s, t)：若該股在 [t-5, t+5] 交易日窗內有任一超限報酬 → 排除
    t0, t1 = pd.Timestamp(PROTOCOL["TEST_START"]), pd.Timestamp(PROTOCOL["TEST_END"])
    w = pr[(pr["Date"] >= t0 - pd.Timedelta(days=15)) & (pr["Date"] <= t1 + pd.Timedelta(days=15))]
    piv = w.pivot_table(index="Date", columns="stock_id", values="over_limit", aggfunc="max").fillna(False)
    piv = piv.astype(float).rolling(11, center=True, min_periods=1).max().astype(bool)  # ±5 交易日窗

    keys = pd.DataFrame({"Date": te["dates"], "stock_id": te["stock_ids"]})
    lut = piv.stack()
    lut = lut[lut]                                # 只留 True，查表 miss 即 False
    flag_rows = pd.MultiIndex.from_frame(keys[["Date", "stock_id"]]).isin(lut.index)
    n_flag = int(flag_rows.sum())

    ok = ~np.isnan(te["alpha_5d"])
    ic_full = daily_spearman_ic(te["dates"][ok], scores[ok], te["alpha_5d"][ok])
    keep = ok & ~flag_rows
    ic_clean = daily_spearman_ic(te["dates"][keep], scores[keep], te["alpha_5d"][keep])
    print(f"[D0] test 列被標可疑：{n_flag:,}/{len(keys):,}（{n_flag/len(keys):.2%}）", flush=True)
    print(f"[D0] test IC 全樣本 {ic_full.mean():+.4f} → 排除可疑後 {ic_clean.mean():+.4f}", flush=True)

    return {
        "n_over_limit_all_history": int(len(flagged)),
        "pct_over_limit": round(len(flagged) / max(len(pr), 1), 5),
        "n_v_spike": int(pr["v_spike"].sum()),
        "over_limit_by_year": {str(y): int(n) for y, n in by_year.items()},
        "top15_extreme": [{"stock_id": r["stock_id"], "date": str(r["Date"].date()),
                           "ret": round(float(r["ret"]), 4)} for _, r in top.iterrows()],
        "test_rows_flagged": n_flag,
        "test_rows_flagged_pct": round(n_flag / len(keys), 4),
        "test_ic_full": round(float(ic_full.mean()), 4),
        "test_ic_excl_flagged": round(float(ic_clean.mean()), 4),
    }


# ============================================================
# D1：IC 依流動性 / 市值三分組
# ============================================================
def _liquidity_lut(t0: str, t1: str) -> pd.DataFrame:
    pr = _load_prices()
    pr["dollar_vol"] = pr["Close"] * pr["Volume"]
    pr["adv20"] = pr.groupby("stock_id", sort=False)["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    m = (pr["Date"] >= pd.Timestamp(t0)) & (pr["Date"] <= pd.Timestamp(t1))
    return pr.loc[m, ["Date", "stock_id", "adv20"]]


def _grouped_ic(df: pd.DataFrame, group_col: str) -> dict:
    out = {}
    for gname, sub in df.groupby(group_col):
        ic = daily_spearman_ic(sub["Date"].to_numpy(), sub["s"].to_numpy(), sub["r"].to_numpy())
        out[str(gname)] = {"mean_ic": round(float(ic.mean()), 4), "n_days": int(len(ic)),
                           "avg_names_per_day": int(len(sub) / max(len(ic), 1))}
    return out


def diag1_ic_by_group(te: dict, scores: np.ndarray) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("D1 IC 依流動性 / 市值三分組", flush=True)
    print("=" * 72, flush=True)
    df = pd.DataFrame({"Date": te["dates"], "stock_id": te["stock_ids"],
                       "s": scores, "r": te["alpha_5d"]}).dropna(subset=["r"])

    liq = _liquidity_lut(PROTOCOL["TEST_START"], PROTOCOL["TEST_END"])
    df = df.merge(liq, on=["Date", "stock_id"], how="left")
    df["liq_bin"] = df.groupby("Date")["adv20"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["L1_小量", "L2_中量", "L3_大量"]))

    mc = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id", "Market_Cap_Log"],
                         filters=[("Date", ">=", pd.Timestamp(PROTOCOL["TEST_START"])),
                                  ("Date", "<=", pd.Timestamp(PROTOCOL["TEST_END"]))])
    df = df.merge(mc, on=["Date", "stock_id"], how="left")
    df["mc_bin"] = df.groupby("Date")["Market_Cap_Log"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["M1_小市值", "M2_中市值", "M3_大市值"]))

    by_liq = _grouped_ic(df.dropna(subset=["liq_bin"]), "liq_bin")
    by_mc = _grouped_ic(df.dropna(subset=["mc_bin"]), "mc_bin")
    for tag, d in (("流動性", by_liq), ("市值", by_mc)):
        print(f"[D1] IC by {tag}：" + " | ".join(f"{k}: {v['mean_ic']:+.4f}" for k, v in sorted(d.items())),
              flush=True)
    return {"ic_by_liquidity": by_liq, "ic_by_market_cap": by_mc}


# ============================================================
# D2：反轉特徵消融（Gram 子矩陣）+ 單因子樸素 IC
# ============================================================
_SUFFIX = re.compile(r"^(?P<base>.+?)(?:_lag(?P<lag>\d+)|_rmean(?P<rm>\d+)|_rstd(?P<rs>\d+))?$")


def _classify(name: str) -> tuple[str, str]:
    """回傳 (base_feature, transform)。Mom_5d 這類本身就是 base。"""
    if re.match(r"^Mom_\d+d$", name):
        return name, "base"
    m = _SUFFIX.match(name)
    base = m.group("base")
    if m.group("lag"):
        return base, f"lag{m.group('lag')}"
    if m.group("rm"):
        return base, f"rmean{m.group('rm')}"
    if m.group("rs"):
        return base, f"rstd{m.group('rs')}"
    return name, "base"


SHORT_REV_BASES = {"Open", "High", "Low", "Close", "Return_1d", "Return_5d",
                   "RSI_14", "KD_K", "KD_D", "RS_5d"}
SHORT_REV_TRANS = {"base", "lag1", "lag5", "rmean5"}
ALL_PRICE_TECH = SHORT_REV_BASES | {"Volume", "Return_20d", "MA_20", "MA_60", "ATR_14",
                                    "RS_20d", "RS_60d", "OBV", "Volatility_20d"}


def _ablation_indices(names: list[str]) -> dict[str, np.ndarray]:
    keep = {}
    ab1_drop, ab2_drop = [], []
    for i, n in enumerate(names):
        base, trans = _classify(n)
        is_mom_short = n in ("Mom_5d", "Mom_10d")
        is_mom = bool(re.match(r"^Mom_\d+d$", n))
        if (base in SHORT_REV_BASES and trans in SHORT_REV_TRANS) or is_mom_short:
            ab1_drop.append(i)
        if base in ALL_PRICE_TECH or is_mom:
            ab2_drop.append(i)
    keep["AB1_去短窗價格反轉"] = np.setdiff1d(np.arange(len(names)), np.array(ab1_drop))
    keep["AB2_去全部價格技術"] = np.setdiff1d(np.arange(len(names)), np.array(ab2_drop))
    return keep


def diag2_ablation(st_full: dict, te: dict, scores_full: np.ndarray) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("D2 反轉特徵消融 + 單因子樸素 IC", flush=True)
    print("=" * 72, flush=True)
    names = all_feature_names()
    ok = ~np.isnan(te["alpha_5d"])
    ic_full = daily_spearman_ic(te["dates"][ok], scores_full[ok], te["alpha_5d"][ok])
    out = {"full_300": {"n_features": 300, "mean_ic": round(float(ic_full.mean()), 4)}}
    print(f"[D2] full 300 維：IC {ic_full.mean():+.4f}（重現檢查，應 ≈ +0.1015）", flush=True)

    for tag, ix in _ablation_indices(names).items():
        sub = {"G": st_full["G"][np.ix_(ix, ix)], "s": st_full["s"][ix],
               "b": st_full["b"][ix], "sy": st_full["sy"], "n": st_full["n"]}
        w_raw, c, _ = ridge_solve(sub, RIDGE_ALPHA)
        sc = te["X"][:, ix] @ w_raw + c
        ic = daily_spearman_ic(te["dates"][ok], sc[ok], te["alpha_5d"][ok])
        dropped = [names[i] for i in range(len(names)) if i not in set(ix.tolist())]
        out[tag] = {"n_features": int(len(ix)), "n_dropped": len(names) - int(len(ix)),
                    "mean_ic": round(float(ic.mean()), 4),
                    "icir": round(float(ic.mean() / ic.std()), 3)}
        print(f"[D2] {tag}：留 {len(ix)}/300 維 → IC {ic.mean():+.4f}（ICIR {out[tag]['icir']}）", flush=True)
        if len(dropped) <= 60:
            out[tag]["dropped_features"] = dropped

    # 單因子樸素 IC（正值代表「因子本身方向」的 IC；反轉因子取負號）
    singles = {"-Return_5d": ("Return_5d", -1), "-Return_1d": ("Return_1d", -1),
               "-RS_5d": ("RS_5d", -1), "-RS_60d": ("RS_60d", -1),
               "-Close_z": ("Close", -1), "Mom_60d": ("Mom_60d", +1),
               "-RSI_14": ("RSI_14", -1), "Foreign_Net": ("Foreign_Net", +1)}
    out["single_factor_ic"] = {}
    for tag, (col, sign) in singles.items():
        j = names.index(col)
        ic = daily_spearman_ic(te["dates"][ok], sign * te["X"][ok, j], te["alpha_5d"][ok])
        out["single_factor_ic"][tag] = round(float(ic.mean()), 4)
        print(f"[D2] 單因子 {tag:<14} IC {ic.mean():+.4f}", flush=True)
    return out


# ============================================================
# D3：樸素基準（等權全 eligible）vs TWII
# ============================================================
def diag3_naive_benchmark(te: dict) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("D3 樸素基準：等權全 eligible / 隨機 Top50 vs TWII", flush=True)
    print("=" * 72, flush=True)
    zeros = np.zeros(len(te["dates"]), dtype=np.float64)
    ew_all = portfolio_backtest(te["dates"], te["stock_ids"], zeros, top_n=10 ** 6)
    rng = np.random.default_rng(20260713)
    rand50 = portfolio_backtest(te["dates"], te["stock_ids"], rng.standard_normal(len(zeros)))
    for tag, r in (("等權全 eligible", ew_all), ("隨機 Top50（含成本）", rand50)):
        print(f"[D3] {tag}：年化 {r['ann_return']:+.1%} | 總報酬 {r['total_return']:+.1%}"
              + (f" | 對 TWII 超額 {r['excess_vs_twii']:+.1%}（{r.get('excess_window','')}）"
                 if "excess_vs_twii" in r else ""), flush=True)
    twii = _load_twii(PROTOCOL["TEST_START"], PROTOCOL["TEST_END"])
    if twii is not None and len(twii) > 2:
        print(f"[D3] TWII 同期：{twii.iloc[0]:.0f} → {twii.iloc[-1]:.0f}"
              f"（{twii.iloc[-1]/twii.iloc[0]-1:+.1%}，至 {twii.index[-1].date()}）", flush=True)
    return {"equal_weight_all": ew_all, "random_top50": rand50}


# ============================================================
# D4：摩擦成本敏感度
# ============================================================
def diag4_cost_sensitivity(te: dict, scores: np.ndarray) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("D4 摩擦成本敏感度（成本 ×1/×2/×3、Top50 限流動性前 1/3）", flush=True)
    print("=" * 72, flush=True)
    out = {}
    cb, cs = PROTOCOL["COST_BUY"], PROTOCOL["COST_SELL"]
    for k in (1, 2, 3):
        bc.PROTOCOL["COST_BUY"], bc.PROTOCOL["COST_SELL"] = cb * k, cs * k
        r = portfolio_backtest(te["dates"], te["stock_ids"], scores)
        out[f"cost_x{k}"] = r
        print(f"[D4] 成本 ×{k}：年化 {r['ann_return']:+.1%} | Sharpe {r['ann_sharpe']} | "
              f"MDD {r['max_drawdown']:.1%}", flush=True)
    bc.PROTOCOL["COST_BUY"], bc.PROTOCOL["COST_SELL"] = cb, cs

    liq = _liquidity_lut(PROTOCOL["TEST_START"], PROTOCOL["TEST_END"])
    keys = pd.DataFrame({"Date": te["dates"], "stock_id": te["stock_ids"]})
    keys = keys.merge(liq, on=["Date", "stock_id"], how="left")
    thr = keys.groupby("Date")["adv20"].transform(lambda x: x.quantile(2 / 3))
    masked = np.where(keys["adv20"].to_numpy() >= thr.to_numpy(), scores, -np.inf)
    r = portfolio_backtest(te["dates"], te["stock_ids"], masked)
    out["top50_high_liquidity_only"] = r
    print(f"[D4] Top50 限流動性前 1/3（成本 ×1）：年化 {r['ann_return']:+.1%} | "
          f"Sharpe {r['ann_sharpe']} | MDD {r['max_drawdown']:.1%}"
          + (f" | 對 TWII 超額 {r['excess_vs_twii']:+.1%}" if "excess_vs_twii" in r else ""), flush=True)
    return out


# ============================================================
# D5：籌碼 point-in-time 抽驗
# ============================================================
def diag5_pit_check(n_dates: int = 8) -> dict:
    print("\n" + "=" * 72, flush=True)
    print("D5 籌碼 point-in-time 抽驗（matrix Foreign_Net vs raw 同日 / 前一日）", flush=True)
    print("=" * 72, flush=True)
    mtx = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id", "Foreign_Net"],
                          filters=[("Date", ">=", pd.Timestamp(PROTOCOL["TEST_START"])),
                                   ("Date", "<=", pd.Timestamp(PROTOCOL["TEST_END"]))])
    days = np.sort(mtx["Date"].unique())
    rng = np.random.default_rng(42)
    sample_days = np.sort(rng.choice(days[1:], size=min(n_dates, len(days) - 1), replace=False))
    prev_map = {d: days[np.searchsorted(days, d) - 1] for d in sample_days}
    need = sorted(set(sample_days) | set(prev_map.values()))

    from marketmamba.config import PROCESSED_DIR
    raw_path = PROCESSED_DIR / "institutional_raw.parquet"
    # institutional_raw 的 Date 欄為字串型別 → 以字串日期做 pyarrow filter
    need_str = [str(pd.Timestamp(d).date()) for d in need]
    try:
        raw = pd.read_parquet(raw_path, filters=[("Date", "in", need_str)])
    except Exception:
        raw = pd.read_parquet(raw_path, filters=[("Date", "in", [pd.Timestamp(d) for d in need])])
    if "date" in raw.columns and "Date" not in raw.columns:
        raw = raw.rename(columns={"date": "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw[raw["Date"].isin(need)]
    fcol = next((c for c in ("Foreign_Net", "foreign_net") if c in raw.columns), None)
    if fcol is None:
        buy = next((c for c in raw.columns if "Foreign" in c and "Buy" in c), None)
        sell = next((c for c in raw.columns if "Foreign" in c and "Sell" in c), None)
        raw["Foreign_Net"] = raw[buy] - raw[sell]
        fcol = "Foreign_Net"
    raw = raw.drop_duplicates(subset=["stock_id", "Date"], keep="last")

    rows = []
    for d in sample_days:
        m = mtx[mtx["Date"] == d][["stock_id", "Foreign_Net"]]
        r_t = raw[raw["Date"] == d][["stock_id", fcol]].rename(columns={fcol: "raw_t"})
        r_p = raw[raw["Date"] == prev_map[d]][["stock_id", fcol]].rename(columns={fcol: "raw_prev"})
        j = m.merge(r_t, on="stock_id").merge(r_p, on="stock_id").dropna()
        if len(j) < 50:
            continue
        c_t = j["Foreign_Net"].corr(j["raw_t"], method="spearman")
        c_p = j["Foreign_Net"].corr(j["raw_prev"], method="spearman")
        rows.append({"date": str(pd.Timestamp(d).date()), "n": len(j),
                     "corr_same_day": round(float(c_t), 3), "corr_prev_day": round(float(c_p), 3)})
        print(f"[D5] {pd.Timestamp(d).date()}  n={len(j):>4}  同日 ρ={c_t:+.3f}  前一日 ρ={c_p:+.3f}", flush=True)
    avg_t = float(np.mean([r["corr_same_day"] for r in rows])) if rows else float("nan")
    avg_p = float(np.mean([r["corr_prev_day"] for r in rows])) if rows else float("nan")
    verdict = ("matrix 特徵對應『同日』籌碼（語意正確：t 日盤後資料標在 t 日）"
               if avg_t > avg_p + 0.2 else "⚠️ 特徵疑似被提前/位移，需檢查 feature_engineer 合併")
    print(f"[D5] 平均：同日 ρ={avg_t:+.3f} vs 前一日 ρ={avg_p:+.3f} → {verdict}", flush=True)
    return {"samples": rows, "avg_corr_same_day": round(avg_t, 3),
            "avg_corr_prev_day": round(avg_p, 3), "verdict": verdict,
            "note": ("T86 於 t 日收盤後（約 16:00）公布；回測假設 t 日收盤價成交，"
                     "同日籌碼特徵對『收盤成交』假設有 ~2.5 小時樂觀性——四階模型同受影響，"
                     "且 5d 訊號主力是價格反轉、籌碼係數小，對本次結論影響有限")}


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    t0 = time.time()
    P = PROTOCOL
    print(f"[load] train {P['TRAIN_START']} → {P['TRAIN_END']}（stride={TRAIN_STRIDE}）...", flush=True)
    tr = load_xy(P["TRAIN_START"], P["TRAIN_END"], day_stride=TRAIN_STRIDE)
    print(f"[load] {tr['X'].shape[0]:,} 列 × {tr['X'].shape[1]} 維", flush=True)

    y = tr["rank_5d"]
    ok = ~np.isnan(y)
    print("[fit] full-train Gram 累積 ...", flush=True)
    st_full = gram_stats(tr["X"], y, ok)
    w_raw, c, _ = ridge_solve(st_full, RIDGE_ALPHA)
    del tr
    gc.collect()

    print(f"[load] test {P['TEST_START']} → {P['TEST_END']}（stride=1）...", flush=True)
    te = load_xy(P["TEST_START"], P["TEST_END"], day_stride=1)
    scores = te["X"] @ w_raw + c
    okt = ~np.isnan(te["alpha_5d"])
    ic = daily_spearman_ic(te["dates"][okt], scores[okt], te["alpha_5d"][okt])
    print(f"[fit] 重現檢查：test 5d IC {ic.mean():+.4f}（原報告 +0.1015；"
          f"本次 fit 含 val 期、與原 st_full 相同）", flush=True)

    results = {"reproduced_test_ic": ic_summary(ic, horizon=5)}
    steps = [("D0_price_errors", lambda: diag0_price_errors(te, scores)),
             ("D1_ic_by_group", lambda: diag1_ic_by_group(te, scores)),
             ("D2_ablation", lambda: diag2_ablation(st_full, te, scores)),
             ("D3_naive_benchmark", lambda: diag3_naive_benchmark(te)),
             ("D4_cost_sensitivity", lambda: diag4_cost_sensitivity(te, scores)),
             ("D5_pit_check", diag5_pit_check)]
    for key, fn in steps:
        try:
            results[key] = fn()
        except Exception as e:                         # 單項失敗不丟掉其他結果
            import traceback
            results[key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{key}] ❌ 失敗：{e}", flush=True)
            traceback.print_exc()
    results["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    results["ridge_alpha"] = RIDGE_ALPHA

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                           encoding="utf-8")
    print(f"\n✅ 診斷完成，結果已存 {RESULT_PATH}（總耗時 {(time.time()-t0)/60:.1f} 分）", flush=True)


if __name__ == "__main__":
    main()
