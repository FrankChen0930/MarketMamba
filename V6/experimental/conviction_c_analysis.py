# -*- coding: utf-8 -*-
"""
方向三-C：不確定性驅動集中（Conviction 萃取）分析
====================================================
研究問題：模型的「信心」（低 MC-Dropout Uncertainty）是否校準到「準確度」？
「訊號強且不確定性低」的子集，是否表現得像一個高信念組合？

對應計畫：planing/研究計畫_方向三_Conviction萃取實驗.md 的 C2/C3
前置盤點（C1，2026-07-11）：
  - V6/results/{date}/df_kelly.csv 共 50 個交易日（2026-04-29 → 2026-07-09）
  - schema 三段：04-29~05-22 Sharpe_Score（16天）/ 05-24~06-12 Signal_Quality（16天）
    / 06-15~07-09 + Signal_Quality_Raw（18天）
  - P0 修復（GAT 全圖推論）於 2026-06-12 上線，該日起 Uncertainty 中位數 0.045→0.032
    → post-P0 樣本 = 06-12 起，pre-P0 與 post-P0 必須分開統計

方法（每個交易日、每個 horizon 各算一次，再跨日彙總）：
  子集定義（C2）：
    all              全體 cross-section（基準；超額報酬以全體均值為 0 點）
    top50_alpha      Exp_Alpha_{h} 前 50 名（控制組：只看訊號強度、不用不確定性）
    top50_sq         Signal_Quality 前 50 名（現行 dashboard 排序）
    conviction       Alpha 前 20% ∩ Uncertainty 後 20%（訊號強 + 模型有信心）
    strong_uncertain Alpha 前 20% ∩ Uncertainty 前 20%（訊號強 + 模型沒把握，對照組）
  校準曲線（C3）：
    以當日 Uncertainty 五分位（Q1=最低 U）分箱：
      (a) 箱內 Spearman(Exp_Alpha, 實際報酬) —— 模型有信心時預測是否更準
      (b) Alpha 前 20% 內按 U 分箱的平均超額報酬/命中率 —— 信心是否改善選股
    另算每日 Spearman(U, |超額報酬|) —— U 是否抓到「意外幅度」
    （注意：此項可能只是反映個股波動度，不等於方向準確度校準，判讀時要分開）

隔離原則：純唯讀分析（df_kelly 歸檔 + prices_raw），不動 production、不動模型。
輸出：V6/experimental/result/conviction_c_analysis.json + 兩張 PNG + stdout 明細表。

執行（Windows 本機即可，無需 WSL/GPU）：
    python V6/experimental/conviction_c_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "V6" / "results"
PRICES_PATH = REPO / "Data" / "processed_v6" / "prices_raw.parquet"
OUT_DIR = REPO / "V6" / "experimental" / "result"

P0_DATE = "2026-06-12"          # GAT 全圖推論修復上線日（含當日屬 post-P0）
HORIZONS = {"5d": 5, "20d": 20}  # 前瞻交易日數
TOP_N = 50
QUANTILE = 0.20                  # conviction 定義用的分位（前/後 20%）
MIN_SUBSET = 10                  # 子集當日至少要有幾檔才納入統計
MIN_STOCKS_FOR_IC = 50

SUBSET_LABELS = {
    "all":              "全體",
    "top50_alpha":      "Top50 by Alpha（無U控制組）",
    "top50_sq":         "Top50 by SQ（現行排序）",
    "conviction":       "Conviction（Alpha前20%∩U後20%）",
    "strong_uncertain": "強訊號高U（Alpha前20%∩U前20%）",
}


def _resolve_sq_col(df: pd.DataFrame) -> str | None:
    """SQ 欄位向下相容：Raw（未截斷）> Signal_Quality > 舊名 Sharpe_Score"""
    for c in ("Signal_Quality_Raw", "Signal_Quality", "Sharpe_Score"):
        if c in df.columns:
            return c
    return None


def load_price_wide() -> pd.DataFrame:
    pr = pd.read_parquet(PRICES_PATH, columns=["stock_id", "Date", "Close"])
    pr["stock_id"] = pr["stock_id"].astype(str)
    # 已知污染：2026-06-07（週日）被寫入 4,317 筆非 4 位數代碼；一律過濾
    pr = pr[pr["stock_id"].str.match(r"^\d{4}$")]
    pr = pr.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    wide = pr.pivot_table(index="Date", columns="stock_id", values="Close", aggfunc="last")
    return wide.sort_index()


def _future_date(pred_date: str, n: int, price_dates: list[str]) -> str | None:
    try:
        idx = price_dates.index(pred_date)
    except ValueError:
        return None
    tgt = idx + n
    return price_dates[tgt] if tgt < len(price_dates) else None


def analyze() -> dict:
    price_wide = load_price_wide()
    price_dates = list(price_wide.index)

    archive_dates = sorted(
        p.name for p in RESULTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("2026") and (p / "df_kelly.csv").exists()
    )
    print(f"歸檔天數: {len(archive_dates)} ({archive_dates[0]} → {archive_dates[-1]})")
    print(f"價格資料: {price_dates[0]} → {price_dates[-1]}\n")

    daily_rows: list[dict] = []          # C2 子集 × 日
    calib_rows: list[dict] = []          # C3 U 分箱 × 日
    ic_rows: list[dict] = []             # 每日 IC / U-|excess| 相關

    for pred_date in archive_dates:
        df = pd.read_csv(RESULTS_DIR / pred_date / "df_kelly.csv", dtype={"Ticker": str})
        sq_col = _resolve_sq_col(df)
        if sq_col is None or "Uncertainty" not in df.columns:
            print(f"  [{pred_date}] 缺 SQ/Uncertainty 欄位，跳過")
            continue
        era = "post_p0" if pred_date >= P0_DATE else "pre_p0"

        for h_label, n_days in HORIZONS.items():
            alpha_col = f"Exp_Alpha_{h_label}"
            if alpha_col not in df.columns:
                continue
            fut = _future_date(pred_date, n_days, price_dates)
            if fut is None or pred_date not in price_wide.index:
                continue

            p0 = price_wide.loc[pred_date].reindex(df["Ticker"].values)
            p1 = price_wide.loc[fut].reindex(df["Ticker"].values)
            valid = (
                p0.notna().values & p1.notna().values & (p0.values > 0)
                & np.isfinite(df[alpha_col].values)
                & np.isfinite(df["Uncertainty"].values)
                & np.isfinite(df[sq_col].values)
            )
            if valid.sum() < MIN_STOCKS_FOR_IC:
                continue

            ret = (p1.values[valid] / p0.values[valid] - 1.0)
            excess = ret - ret.mean()
            alpha = df[alpha_col].values[valid]
            unc = df["Uncertainty"].values[valid]
            sq = df[sq_col].values[valid]
            n_all = int(valid.sum())

            # ---- 每日 IC ----
            ic_alpha, _ = stats.spearmanr(alpha, ret)
            ic_sq, _ = stats.spearmanr(sq, ret)
            u_abs_corr, _ = stats.spearmanr(unc, np.abs(excess))
            ic_rows.append({
                "date": pred_date, "era": era, "horizon": h_label, "n": n_all,
                "ic_alpha": float(ic_alpha), "ic_sq": float(ic_sq),
                "u_vs_absexcess": float(u_abs_corr),
            })

            # ---- C2 子集 ----
            alpha_rank = stats.rankdata(-alpha)          # 1 = alpha 最高
            unc_rank = stats.rankdata(unc)               # 1 = U 最低
            top_alpha_mask = alpha_rank <= n_all * QUANTILE
            low_u_mask = unc_rank <= n_all * QUANTILE
            high_u_mask = unc_rank > n_all * (1 - QUANTILE)

            subsets = {
                "all":              np.ones(n_all, dtype=bool),
                "top50_alpha":      alpha_rank <= TOP_N,
                "top50_sq":         stats.rankdata(-sq) <= TOP_N,
                "conviction":       top_alpha_mask & low_u_mask,
                "strong_uncertain": top_alpha_mask & high_u_mask,
            }
            for name, mask in subsets.items():
                n_sub = int(mask.sum())
                if n_sub < MIN_SUBSET:
                    continue
                daily_rows.append({
                    "date": pred_date, "era": era, "horizon": h_label,
                    "subset": name, "n": n_sub,
                    "mean_excess_pct": float(excess[mask].mean() * 100),
                    "hit_rate": float((excess[mask] > 0).mean()),
                })

            # ---- C3 U 五分位校準 ----
            u_quintile = np.ceil(unc_rank / n_all * 5).astype(int).clip(1, 5)
            for q in range(1, 6):
                qm = u_quintile == q
                if qm.sum() < MIN_STOCKS_FOR_IC:
                    continue
                q_ic, _ = stats.spearmanr(alpha[qm], ret[qm])
                inb = qm & top_alpha_mask   # 該 U 箱內的強訊號股
                calib_rows.append({
                    "date": pred_date, "era": era, "horizon": h_label,
                    "u_quintile": q, "n": int(qm.sum()),
                    "ic_in_bin": float(q_ic),
                    "topalpha_n": int(inb.sum()),
                    "topalpha_excess_pct": float(excess[inb].mean() * 100) if inb.sum() >= 5 else None,
                    "topalpha_hit_rate": float((excess[inb] > 0).mean()) if inb.sum() >= 5 else None,
                })

    return {
        "daily": pd.DataFrame(daily_rows),
        "calib": pd.DataFrame(calib_rows),
        "ic": pd.DataFrame(ic_rows),
    }


def _tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12))


def summarize(frames: dict) -> dict:
    daily, calib, ic = frames["daily"], frames["calib"], frames["ic"]
    summary: dict = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                     "p0_date": P0_DATE, "quantile": QUANTILE, "top_n": TOP_N,
                     "subsets": {}, "calibration": {}, "ic_overview": {}}

    for (h, era), g in ic.groupby(["horizon", "era"]):
        summary["ic_overview"][f"{h}/{era}"] = {
            "n_days": len(g),
            "mean_ic_alpha": round(g.ic_alpha.mean(), 4),
            "mean_ic_sq": round(g.ic_sq.mean(), 4),
            "icir_sq": round(g.ic_sq.mean() / (g.ic_sq.std(ddof=1) + 1e-12), 3) if len(g) > 1 else None,
            "ic_sq_gt0_pct": round((g.ic_sq > 0).mean() * 100, 1),
            "mean_u_vs_absexcess": round(g.u_vs_absexcess.mean(), 4),
        }

    for (h, era, sub), g in daily.groupby(["horizon", "era", "subset"]):
        summary["subsets"][f"{h}/{era}/{sub}"] = {
            "n_days": len(g),
            "avg_n_stocks": round(g.n.mean(), 1),
            "mean_excess_pct_per_period": round(g.mean_excess_pct.mean(), 3),
            "t_stat": round(_tstat(g.mean_excess_pct.values), 2),
            "mean_hit_rate": round(g.hit_rate.mean(), 3),
            "days_positive_pct": round((g.mean_excess_pct > 0).mean() * 100, 1),
        }

    for (h, era, q), g in calib.groupby(["horizon", "era", "u_quintile"]):
        ta = g.topalpha_excess_pct.dropna()
        summary["calibration"][f"{h}/{era}/Q{q}"] = {
            "n_days": len(g),
            "mean_ic_in_bin": round(g.ic_in_bin.mean(), 4),
            "topalpha_mean_excess_pct": round(ta.mean(), 3) if len(ta) else None,
            "topalpha_mean_hit_rate": round(g.topalpha_hit_rate.dropna().mean(), 3) if g.topalpha_hit_rate.notna().any() else None,
        }
    return summary


def print_report(summary: dict) -> None:
    print("=" * 88)
    print("C2 子集比較（跨日平均超額報酬 %/期；excess 以當日全體均值為 0 點）")
    print("=" * 88)
    order = ["all", "top50_alpha", "top50_sq", "conviction", "strong_uncertain"]
    for h in HORIZONS:
        for era in ("post_p0", "pre_p0"):
            keys = [k for k in summary["subsets"] if k.startswith(f"{h}/{era}/")]
            if not keys:
                continue
            print(f"\n--- horizon={h} | era={era} ---")
            print(f"{'子集':<34}{'天數':>4}{'均檔數':>8}{'超額%/期':>10}{'t值':>7}{'命中率':>8}{'正天數%':>8}")
            for sub in order:
                k = f"{h}/{era}/{sub}"
                if k not in summary["subsets"]:
                    continue
                s = summary["subsets"][k]
                print(f"{SUBSET_LABELS[sub]:<30}{s['n_days']:>6}{s['avg_n_stocks']:>9}"
                      f"{s['mean_excess_pct_per_period']:>+10.3f}{s['t_stat']:>7.2f}"
                      f"{s['mean_hit_rate']:>8.3f}{s['days_positive_pct']:>8.1f}")

    print("\n" + "=" * 88)
    print("C3 校準曲線（U 五分位；Q1=模型最有信心）")
    print("=" * 88)
    for h in HORIZONS:
        for era in ("post_p0", "pre_p0"):
            keys = [k for k in summary["calibration"] if k.startswith(f"{h}/{era}/")]
            if not keys:
                continue
            print(f"\n--- horizon={h} | era={era} ---")
            print(f"{'U分位':<8}{'天数':>4}{'箱內IC':>10}{'強訊號超額%':>12}{'強訊號命中率':>12}")
            for q in range(1, 6):
                k = f"{h}/{era}/Q{q}"
                if k not in summary["calibration"]:
                    continue
                c = summary["calibration"][k]
                ex = c["topalpha_mean_excess_pct"]
                hr = c["topalpha_mean_hit_rate"]
                print(f"Q{q:<7}{c['n_days']:>5}{c['mean_ic_in_bin']:>+10.4f}"
                      f"{(f'{ex:+.3f}' if ex is not None else 'N/A'):>12}"
                      f"{(f'{hr:.3f}' if hr is not None else 'N/A'):>12}")

    print("\n" + "=" * 88)
    print("每日 IC 總覽（Spearman）")
    print("=" * 88)
    print(f"{'horizon/era':<16}{'天數':>4}{'IC_alpha':>10}{'IC_SQ':>8}{'ICIR_SQ':>9}{'IC>0%':>8}{'U~|excess|':>12}")
    for k, s in summary["ic_overview"].items():
        print(f"{k:<16}{s['n_days']:>4}{s['mean_ic_alpha']:>+10.4f}{s['mean_ic_sq']:>+8.4f}"
              f"{(s['icir_sq'] if s['icir_sq'] is not None else float('nan')):>9.3f}"
              f"{s['ic_sq_gt0_pct']:>8.1f}{s['mean_u_vs_absexcess']:>+12.4f}")


def make_charts(frames: dict, summary: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Microsoft JhengHei", "figure.dpi": 150,
        "axes.unicode_minus": False,  # JhengHei 沒有 U+2212 負號字元
        "axes.edgecolor": "#d9d8d3", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#eceae5", "grid.linewidth": 0.7,
        "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
    })
    BLUE, AQUA, RED, GRAY = "#2a78d6", "#1baf7a", "#e34948", "#9b9a94"
    saved: list[str] = []

    # 圖 1：C3 校準曲線（5d）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for ax, metric, title, fmt in (
        (axes[0], "mean_ic_in_bin", "U 分箱內 IC（Exp_Alpha_5d vs 實際 5d 報酬）", "{:+.3f}"),
        (axes[1], "topalpha_mean_excess_pct", "Alpha 前 20% 內：各 U 分箱平均超額報酬（%/5d）", "{:+.2f}"),
    ):
        for era, color, style, label in (("post_p0", BLUE, "-", "post-P0（06-12 起）"),
                                          ("pre_p0", GRAY, "--", "pre-P0")):
            ys = [summary["calibration"].get(f"5d/{era}/Q{q}", {}).get(metric) for q in range(1, 6)]
            xs = [q for q, y in zip(range(1, 6), ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if not ys:
                continue
            ax.plot(xs, ys, style, color=color, linewidth=2, marker="o", markersize=6, label=label)
            for x, y in zip(xs, ys):
                ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color="#52514e")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f"Q{q}" for q in range(1, 6)])
        ax.set_xlabel("Uncertainty 五分位（Q1 = 模型最有信心）")
        ax.set_title(title, fontsize=10.5, color="#0b0b0b")
        ax.axhline(0, color="#d9d8d3", linewidth=0.8)
        ax.set_facecolor("#fcfcfb")
        ax.legend(fontsize=8.5, frameon=False)
    fig.suptitle("方向三-C3：信心－準確度校準曲線（5d 前瞻）", fontsize=12, y=1.02)
    fig.tight_layout()
    p = OUT_DIR / "conviction_c3_calibration_5d.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 圖 2：C2 子集平均超額報酬（5d, post-P0）
    order = ["top50_alpha", "top50_sq", "conviction", "strong_uncertain"]
    vals, labels, colors = [], [], []
    for sub in order:
        s = summary["subsets"].get(f"5d/post_p0/{sub}")
        if s is None:
            continue
        vals.append(s["mean_excess_pct_per_period"])
        labels.append(SUBSET_LABELS[sub].replace("（", "\n（"))
        colors.append({"conviction": AQUA, "strong_uncertain": RED}.get(sub, BLUE))
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor="#fcfcfb")
    bars = ax.bar(labels, vals, color=colors, width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:+.2f}%", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6 if v >= 0 else -14),
                    ha="center", fontsize=9.5, color="#0b0b0b")
    ax.axhline(0, color="#52514e", linewidth=0.9)
    ax.set_ylabel("平均超額報酬（%/5d，去當日全體均值）")
    n_days = summary["subsets"].get("5d/post_p0/conviction", {}).get("n_days", "?")
    ax.set_title(f"方向三-C2：子集平均超額報酬（5d 前瞻，post-P0，n={n_days} 天）",
                 fontsize=11, color="#0b0b0b")
    ax.set_facecolor("#fcfcfb")
    ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    p = OUT_DIR / "conviction_c2_subsets_5d.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))
    return saved


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = analyze()
    if frames["daily"].empty:
        print("沒有任何可計算的日子，中止")
        sys.exit(1)
    summary = summarize(frames)
    print_report(summary)

    out = dict(summary)
    out["daily_detail"] = frames["daily"].to_dict(orient="records")
    out["calib_detail"] = frames["calib"].to_dict(orient="records")
    out["ic_detail"] = frames["ic"].to_dict(orient="records")
    json_path = OUT_DIR / "conviction_c_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON 已存 → {json_path}")
    for p in make_charts(frames, summary):
        print(f"✅ 圖表已存 → {p}")


if __name__ == "__main__":
    main()
