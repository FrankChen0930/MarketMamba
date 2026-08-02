"""同一把尺重算所有模型的 IC + 十分位報酬輪廓（純讀）。

A. 用完全相同的程式路徑，對同一份 Alpha_5d 算逐日 Spearman IC
B. 十分位平均前瞻 5 日報酬 —— 看 GRU 的優勢是不是集中在兩端
C. 用 Kendall tau / 兩端子集 IC 佐證
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba")
sys.path.insert(0, str(ROOT / "V6"))
SCORES = ROOT / "V6/experimental/result/scores"
MODELS = ["ridge", "gbdt", "no_gat", "old_kg", "v2_kg", "v3_kg", "gru"]

from experimental.baseline_common import BASE_PATH, PROTOCOL  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 60)

lab = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id", "Alpha_5d", "eligible"],
                      filters=[("Date", ">=", pd.Timestamp(PROTOCOL["TEST_START"])),
                               ("Date", "<=", pd.Timestamp(PROTOCOL["TEST_END"]))])
lab["Date"] = pd.to_datetime(lab["Date"])
lab = lab.dropna(subset=["Alpha_5d"])
print(f"[label] {len(lab):,} 列有 Alpha_5d｜{lab['Date'].nunique()} 天")

print()
print("=" * 100)
print("[A] 同一把尺：逐日 Spearman IC（對同一份 Alpha_5d、同一段日期）")
print("=" * 100)
rows = []
dec_profiles = {}
for m in MODELS:
    d = pd.read_parquet(SCORES / f"{m}.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    j = d.merge(lab, on=["Date", "stock_id"], how="inner")
    ics, decs = [], []
    for dt, g in j.groupby("Date", sort=True):
        if len(g) < 100:
            continue
        ics.append(spearmanr(g["score"], g["Alpha_5d"]).statistic)
        q = pd.qcut(g["score"].rank(method="first"), 10, labels=False)
        decs.append(g.groupby(q)["Alpha_5d"].mean().reindex(range(10)).to_numpy())
    ics = np.array(ics)
    dec_profiles[m] = np.nanmean(np.vstack(decs), axis=0)
    rows.append({"model": m, "n_days": len(ics), "n_rows": len(j),
                 "mean_IC": ics.mean(), "ICIR": ics.mean() / ics.std(),
                 "IC>0比例": (ics > 0).mean(),
                 "NW_t": ics.mean() / (ics.std() / np.sqrt(len(ics)))})
ic_df = pd.DataFrame(rows).set_index("model")
print(ic_df.round(4).to_string())

print()
print("=" * 100)
print("[B] 十分位平均 Alpha_5d（0=分數最低 … 9=分數最高）；單位為 rank 標籤")
print("=" * 100)
prof = pd.DataFrame(dec_profiles, index=[f"D{i}" for i in range(10)]).T
print(prof.round(4).to_string())
print("\n兩端 vs 中段的貢獻：")
summ = pd.DataFrame({
    "D9-D0(兩端差)": prof["D9"] - prof["D0"],
    "D8-D1(次兩端)": prof["D8"] - prof["D1"],
    "D7-D2": prof["D7"] - prof["D2"],
    "D6-D3(中段)": prof["D6"] - prof["D3"],
})
summ["兩端/中段比"] = summ["D9-D0(兩端差)"] / summ["D6-D3(中段)"]
print(summ.round(4).to_string())

print()
print("=" * 100)
print("[C] 只在兩端子集內算 IC（各取分數最極端的 20%），對照全市場 IC")
print("=" * 100)
rows = []
for m in MODELS:
    d = pd.read_parquet(SCORES / f"{m}.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    j = d.merge(lab, on=["Date", "stock_id"], how="inner")
    full, ext, mid = [], [], []
    for dt, g in j.groupby("Date", sort=True):
        if len(g) < 200:
            continue
        full.append(spearmanr(g["score"], g["Alpha_5d"]).statistic)
        r = g["score"].rank(pct=True)
        e = g[(r <= 0.10) | (r >= 0.90)]
        mm = g[(r > 0.30) & (r < 0.70)]
        if len(e) > 50:
            ext.append(spearmanr(e["score"], e["Alpha_5d"]).statistic)
        if len(mm) > 50:
            mid.append(spearmanr(mm["score"], mm["Alpha_5d"]).statistic)
    rows.append({"model": m, "全市場IC": np.mean(full),
                 "兩端20%內IC": np.mean(ext), "中段40%內IC": np.mean(mid)})
print(pd.DataFrame(rows).set_index("model").round(4).to_string())
print("  註：『兩端內 IC』衡量在最極端那批股票裡還能不能繼續排序。")
