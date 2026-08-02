"""GRU decile 優勢的性質診斷（純讀）。

1. 兩端 decile 的流動性 / 價格 / 波動度輪廓
2. 限制在高流動半場後，spread 是否還在
3. 時間集中度：逐月 spread、去掉最好的 5 天
4. GRU 與 v2_kg 的兩端成員重疊
5. 分數與「過去報酬」的關係 —— GRU 是不是在兩端下更重的反轉賭注
6. 兩端是否富集「無前瞻標籤」（下市/停牌）的列
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba")
sys.path.insert(0, str(ROOT / "V6"))
SCORES = ROOT / "V6/experimental/result/scores"
MODELS = ["ridge", "gbdt", "v2_kg", "gru"]

from experimental.portfolio_lab import Market, TRADING_DAYS  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 60)

# 共用 market（ridge 與 gru 的 panel 逐對相同 → 可共用；v2_kg 少 2 支，另建）
base = pd.read_parquet(SCORES / "ridge.parquet")
base["Date"] = pd.to_datetime(base["Date"])
_r = base.pivot(index="Date", columns="stock_id", values="score")
MKT = Market(_r.index.to_numpy(), list(_r.columns))
del _r, base


def get_rank(m, mkt):
    d = pd.read_parquet(SCORES / f"{m}.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    rk = d.pivot(index="Date", columns="stock_id", values="score") \
          .rank(axis=1, ascending=False, method="first")
    return rk.reindex(index=mkt.dates, columns=mkt.stocks).where(mkt.px.notna())


def decile_run(mkt, rank, freq=5, liq_min=None):
    """回傳 (long_r, short_r, 持有紀錄)。liq_min 給定時只在高流動子集裡選。"""
    rk = rank.to_numpy(np.float64).copy()
    if liq_min is not None:
        rk[~(mkt.liq_pct.to_numpy(np.float64) >= liq_min)] = np.nan
    ret = mkt.ret.to_numpy(np.float64).copy()
    np.nan_to_num(ret, copy=False)
    T = len(mkt.dates)
    lr, sr = np.zeros(T), np.zeros(T)
    hl = hs = None
    picks = []
    for t in range(T):
        if hl is not None:
            lr[t] = float(ret[t][hl].mean())
            sr[t] = float(ret[t][hs].mean())
        if t % freq == 0:
            ok = np.isfinite(rk[t])
            m = int(ok.sum())
            if m < 50:
                continue
            d = max(m // 10, 1)
            order = np.argsort(np.where(ok, rk[t], np.inf))
            hl, hs = order[:d], order[m - d:m]
            picks.append((t, hl, hs))
    return lr, sr, picks


def sharpe(x):
    return float(np.mean(x) / np.std(x) * np.sqrt(TRADING_DAYS)) if np.std(x) > 0 else np.nan


def ann(x):
    return float((1 + pd.Series(x)).prod() ** (TRADING_DAYS / len(x)) - 1)


ranks = {m: get_rank(m, MKT) for m in MODELS}

print("=" * 96)
print("[1] 兩端 decile 的輪廓（流動性百分位 / 收盤價 / 近60日波動度，取中位數）")
print("=" * 96)
liqp = MKT.liq_pct.to_numpy(np.float64)
pxv = MKT.px.to_numpy(np.float64)
volv = MKT.vol.to_numpy(np.float64)
prof = {}
for m in MODELS:
    _, _, picks = decile_run(MKT, ranks[m])
    def med(arrs, mat):
        v = np.concatenate([mat[t][idx] for t, idx in arrs])
        return float(np.nanmedian(v))
    L = [(t, hl) for t, hl, _ in picks]
    S = [(t, hs) for t, _, hs in picks]
    prof[m] = {
        "long_liq_pct": med(L, liqp), "short_liq_pct": med(S, liqp),
        "long_px": med(L, pxv), "short_px": med(S, pxv),
        "long_vol60": med(L, volv), "short_vol60": med(S, volv),
    }
print(pd.DataFrame(prof).T.round(4).to_string())

print()
print("=" * 96)
print("[2] 限制在高流動半場（liq_pct >= 0.5）後的 decile spread")
print("=" * 96)
rows = []
for m in MODELS:
    for tag, lm in [("全市場", None), ("高流動半場", 0.5), ("高流動前1/3", 0.667)]:
        lr, sr, _ = decile_run(MKT, ranks[m], liq_min=lm)
        ls = lr - sr
        rows.append({"model": m, "子集": tag, "long_ann": ann(lr), "short_ann": ann(sr),
                     "spread_ann": float(np.mean(ls) * TRADING_DAYS), "spread_sharpe": sharpe(ls)})
print(pd.DataFrame(rows).pivot(index="model", columns="子集").round(3).to_string())

print()
print("=" * 96)
print("[3] 時間集中度：去掉 spread 最好的 N 天後的 Sharpe")
print("=" * 96)
rows = []
for m in MODELS:
    lr, sr, _ = decile_run(MKT, ranks[m])
    ls = lr - sr
    d = {"model": m, "原始": sharpe(ls)}
    for n in (5, 10, 20):
        idx = np.argsort(ls)[::-1][:n]
        d[f"去最好{n}天"] = sharpe(np.delete(ls, idx))
    d["正日比例"] = float((ls > 0).mean())
    # 逐半年
    yh = pd.Series(MKT.dates).dt.to_period("Q").astype(str).to_numpy()
    for q in ["2024Q1", "2025Q1", "2026Q1"]:
        pass
    rows.append(d)
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n逐季 spread 年化（看是否集中在某段）")
lsq = {}
for m in MODELS:
    lr, sr, _ = decile_run(MKT, ranks[m])
    s = pd.Series(lr - sr, index=MKT.dates)
    lsq[m] = s.groupby(s.index.to_period("Q")).mean() * TRADING_DAYS
print(pd.DataFrame(lsq).round(3).to_string())

print()
print("=" * 96)
print("[4] GRU 與 v2_kg / ridge 的兩端成員重疊率")
print("=" * 96)
pk = {m: decile_run(MKT, ranks[m])[2] for m in MODELS}
for other in ["ridge", "gbdt", "v2_kg"]:
    ol = np.mean([len(set(a) & set(b)) / len(a)
                  for (_, a, _), (_, b, _) in zip(pk["gru"], pk[other])])
    os_ = np.mean([len(set(a) & set(b)) / len(a)
                   for (_, _, a), (_, _, b) in zip(pk["gru"], pk[other])])
    print(f"  gru vs {other:<6} 前10%重疊 {ol:.1%} ｜ 後10%重疊 {os_:.1%}")

print()
print("=" * 96)
print("[5] 分數 vs 過去報酬：GRU 是不是在兩端下更重的反轉賭注")
print("=" * 96)
pxf = MKT.px.ffill()
past5 = (pxf / pxf.shift(5) - 1).to_numpy(np.float64)
past20 = (pxf / pxf.shift(20) - 1).to_numpy(np.float64)
res = {}
for m in MODELS:
    rk = ranks[m].to_numpy(np.float64)
    c5, c20, l5, s5 = [], [], [], []
    for t in range(20, len(MKT.dates)):
        ok = np.isfinite(rk[t]) & np.isfinite(past5[t])
        if ok.sum() < 100:
            continue
        c5.append(np.corrcoef(rk[t][ok], past5[t][ok])[0, 1])
        ok20 = np.isfinite(rk[t]) & np.isfinite(past20[t])
        c20.append(np.corrcoef(rk[t][ok20], past20[t][ok20])[0, 1])
    for t, hl, hs in pk[m]:
        if t >= 20:
            l5.append(np.nanmedian(past5[t][hl]))
            s5.append(np.nanmedian(past5[t][hs]))
    res[m] = {"corr(排名,過去5d報酬)": float(np.mean(c5)),
              "corr(排名,過去20d報酬)": float(np.mean(c20)),
              "前10%的過去5d報酬中位": float(np.mean(l5)),
              "後10%的過去5d報酬中位": float(np.mean(s5))}
print(pd.DataFrame(res).T.round(4).to_string())
print("  註：排名 1=最好。corr 為正 → 好排名給了『過去跌的』＝反轉；為負 → 追動能。")

print()
print("=" * 96)
print("[6] 兩端是否富集『無前瞻標籤』的列（下市/停牌邊緣）")
print("=" * 96)
from experimental.baseline_common import BASE_PATH, PROTOCOL  # noqa: E402
lab = pd.read_parquet(BASE_PATH, columns=["Date", "stock_id", "Alpha_5d"],
                      filters=[("Date", ">=", pd.Timestamp(PROTOCOL["TEST_START"])),
                               ("Date", "<=", pd.Timestamp(PROTOCOL["TEST_END"]))])
lab["Date"] = pd.to_datetime(lab["Date"])
nanmat = lab.pivot(index="Date", columns="stock_id", values="Alpha_5d") \
            .reindex(index=MKT.dates, columns=MKT.stocks).isna().to_numpy()
base_rate = float(nanmat[[t for t, _, _ in pk["gru"]]].mean())
print(f"  基準（全市場、再平衡日）NaN 標籤比例 {base_rate:.2%}")
for m in MODELS:
    l = np.mean([nanmat[t][hl].mean() for t, hl, _ in pk[m]])
    s = np.mean([nanmat[t][hs].mean() for t, _, hs in pk[m]])
    print(f"  {m:<6} 前10% {l:.2%} ｜ 後10% {s:.2%}")
