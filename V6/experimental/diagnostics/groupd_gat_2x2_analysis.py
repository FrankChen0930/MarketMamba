"""Group D × GAT 2×2 對照 + 配對 Newey-West 檢定（純讀 JSON）。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DL = Path(r"D:\Downloads")
kg = json.load(open(DL / "kg_ablation_result.json", encoding="utf-8"))["arms"]
gd = json.load(open(DL / "groupd_ablation_result.json", encoding="utf-8"))["arms"]
g2 = json.load(open(DL / "groupd_ablation_result_gatv2.json", encoding="utf-8"))["arms"]

CELL = {
    "A_nogat_macro":   kg["no_gat"],
    "B_nogat_nomacro": gd["no_macro"],
    "C_v2_macro":      kg["v2_kg"],
    "D_v2_nomacro":    g2["no_macro"],
}

print("=" * 104)
print("[0] harness 一致性核對（不一致就不能放同一張表）")
print("=" * 104)
rows = []
for k, v in CELL.items():
    rows.append({"cell": k, "params": v.get("n_parameters"), "seed": v.get("seed"),
                 "use_gat": v.get("use_gat"), "n_days": v.get("eval_n_days"),
                 "epochs": v.get("epochs_param"), "peak_ep": v.get("peak_epoch"),
                 "epochs_ran": v.get("epochs_ran")})
print(pd.DataFrame(rows).to_string(index=False))

ic = {k: pd.Series(v["ic_by_day"], dtype=float) for k, v in CELL.items()}
for k, s in ic.items():
    s.index = pd.to_datetime(s.index)
    ic[k] = s.sort_index()
common = None
for s in ic.values():
    common = s.index if common is None else common.intersection(s.index)
print(f"\n共同交易日 {len(common)} 天（各 arm 原始天數 "
      f"{[len(s) for s in ic.values()]}）")
IC = pd.DataFrame({k: s.reindex(common) for k, s in ic.items()})


def nw_t(x, lags=10):
    x = np.asarray(x, float)
    n, mu = len(x), x.mean()
    e = x - mu
    v = (e @ e) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * ((e[L:] @ e[:-L]) / n)
    return float(mu / np.sqrt(v / n))


print()
print("=" * 104)
print("[1] 2×2 訊號層（重評 mean IC，582 天）")
print("=" * 104)
m = {k: IC[k].mean() for k in CELL}
tbl = pd.DataFrame(
    [[m["A_nogat_macro"], m["B_nogat_nomacro"]],
     [m["C_v2_macro"], m["D_v2_nomacro"]]],
    index=["no GAT", "v2 圖"], columns=["with macro", "no macro"])
tbl["Δ 移除 macro"] = tbl["no macro"] - tbl["with macro"]
tbl.loc["Δ 加 GAT"] = [tbl.loc["v2 圖", "with macro"] - tbl.loc["no GAT", "with macro"],
                       tbl.loc["v2 圖", "no macro"] - tbl.loc["no GAT", "no macro"],
                       np.nan]
print(tbl.round(4).to_string())

add = m["A_nogat_macro"] + (m["B_nogat_nomacro"] - m["A_nogat_macro"]) \
      + (m["C_v2_macro"] - m["A_nogat_macro"])
print(f"\n可加預測 D = A + (B−A) + (C−A) = {add:+.4f}")
print(f"實際 D                              = {m['D_v2_nomacro']:+.4f}")
print(f"交互作用（實際 − 可加）              = {m['D_v2_nomacro'] - add:+.4f}")

print()
print("=" * 104)
print("[2] 配對 Newey-West 檢定（同日相減）")
print("=" * 104)
pairs = [
    ("macro 效應 @no GAT", "B_nogat_nomacro", "A_nogat_macro", "乾淨（同架構同參數同 RNG）"),
    ("macro 效應 @v2 圖", "D_v2_nomacro", "C_v2_macro", "乾淨（同架構同參數同 RNG）"),
    ("GAT 效應 @with macro", "C_v2_macro", "A_nogat_macro", "髒（參數量差 264,704、RNG 分岔）"),
    ("GAT 效應 @no macro", "D_v2_nomacro", "B_nogat_nomacro", "髒（同上）"),
    ("D vs A（兩者一起）", "D_v2_nomacro", "A_nogat_macro", "髒"),
]
print(f"{'比較':<22} {'Δ mean IC':>10} {'NW t':>8} {'正比例':>8} {'std(Δ)':>9}  說明")
for lab, hi, lo, note in pairs:
    d = (IC[hi] - IC[lo]).dropna()
    print(f"{lab:<22} {d.mean():>+10.4f} {nw_t(d):>8.2f} {(d>0).mean():>8.1%} "
          f"{d.std():>9.4f}  {note}")

d_int = (IC["D_v2_nomacro"] - IC["C_v2_macro"]) - (IC["B_nogat_nomacro"] - IC["A_nogat_macro"])
print(f"\n交互作用逐日：Δ {d_int.mean():+.4f}｜NW t {nw_t(d_int):+.2f}｜"
      f"（負值＝兩個效應互相搶訊號）")

print()
print("── 穩健性：前後半 / 去極端 5 天 ──")
h = len(IC) // 2
for lab, hi, lo, _ in pairs[:4]:
    d = (IC[hi] - IC[lo]).dropna()
    trim = d.drop(d.abs().nlargest(5).index)
    print(f"  {lab:<22} 前半 {d[:h].mean():+.4f}(t={nw_t(d[:h]):+.2f})｜"
          f"後半 {d[h:].mean():+.4f}(t={nw_t(d[h:]):+.2f})｜去極端5天 {trim.mean():+.4f}")

print()
print("=" * 104)
print("[3] 其他指標 2×2（ICIR / 正比例 / 組合層，組合層為 portfolio_backtest 舊口徑 Top50 5日）")
print("=" * 104)
rows = []
for k, v in CELL.items():
    p = v.get("test_portfolio", {})
    rows.append({"cell": k, "mean_IC": v["eval_mean_ic_5d"], "ICIR": v["eval_icir_5d"],
                 "正比例": v["eval_pct_pos"], "年化": p.get("ann_return"),
                 "Sharpe": p.get("ann_sharpe"), "MDD": p.get("max_drawdown"),
                 "換手": p.get("avg_turnover_per_rebalance"),
                 "峰值ep": v.get("peak_epoch"), "峰值IC": v.get("peak_ic_5d")})
print(pd.DataFrame(rows).to_string(index=False))

print()
print("── val IC 曲線（看有沒有被排程截斷）──")
for k, v in CELL.items():
    c = v.get("val_ic_5d_curve")
    if c:
        print(f"  {k:<18} {[f'{x:+.4f}' for x in c]}  峰值 ep{int(np.argmax(c))+1}")
