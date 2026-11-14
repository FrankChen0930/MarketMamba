"""GRU 兩端優勢的穩健性 + 配對檢定（純讀）。

A. 換 decile 定義（前後 5% / 10% / 20%）與再平衡頻率（1/5/20 日）
B. GRU 對各模型的「逐日 spread 差」配對 Newey-West t
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba")
sys.path.insert(0, str(ROOT / "V6"))
SCORES = ROOT / "V6/experimental/result/scores"
MODELS = ["ridge", "gbdt", "v2_kg", "v3_kg", "gru"]

from experimental.portfolio_lab import Market, TRADING_DAYS  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 60)

base = pd.read_parquet(SCORES / "ridge.parquet")
base["Date"] = pd.to_datetime(base["Date"])
_r = base.pivot(index="Date", columns="stock_id", values="score")
MKT = Market(_r.index.to_numpy(), list(_r.columns))
del _r, base


def get_rank(m):
    d = pd.read_parquet(SCORES / f"{m}.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    rk = d.pivot(index="Date", columns="stock_id", values="score") \
          .rank(axis=1, ascending=False, method="first")
    return rk.reindex(index=MKT.dates, columns=MKT.stocks).where(MKT.px.notna())


def spread_series(rank, freq=5, pct=0.10):
    rk = rank.to_numpy(np.float64)
    ret = MKT.ret.to_numpy(np.float64).copy()
    np.nan_to_num(ret, copy=False)
    T = len(MKT.dates)
    lr, sr = np.zeros(T), np.zeros(T)
    hl = hs = None
    for t in range(T):
        if hl is not None:
            lr[t] = float(ret[t][hl].mean())
            sr[t] = float(ret[t][hs].mean())
        if t % freq == 0:
            ok = np.isfinite(rk[t])
            m = int(ok.sum())
            if m < 50:
                continue
            d = max(int(m * pct), 1)
            order = np.argsort(np.where(ok, rk[t], np.inf))
            hl, hs = order[:d], order[m - d:m]
    return lr - sr


def sharpe(x):
    return float(np.mean(x) / np.std(x) * np.sqrt(TRADING_DAYS)) if np.std(x) > 0 else np.nan


def nw_t(x, lags=10):
    x = np.asarray(x, float)
    n, mu = len(x), x.mean()
    e = x - mu
    g0 = (e @ e) / n
    v = g0
    for L in range(1, lags + 1):
        g = (e[L:] @ e[:-L]) / n
        v += 2 * (1 - L / (lags + 1)) * g
    return float(mu / np.sqrt(v / n))


ranks = {m: get_rank(m) for m in MODELS}

print("=" * 92)
print("[A] 穩健性：spread Sharpe 在不同 decile 寬度 × 再平衡頻率下")
print("=" * 92)
rows = []
for m in MODELS:
    for pct in (0.05, 0.10, 0.20):
        for fq in (1, 5, 20):
            rows.append({"model": m, "寬度": f"{int(pct*100)}%", "freq": fq,
                         "Sharpe": sharpe(spread_series(ranks[m], fq, pct))})
df = pd.DataFrame(rows)
print(df.pivot(index="model", columns=["寬度", "freq"], values="Sharpe").round(2).to_string())

print()
print("=" * 92)
print("[B] 配對檢定：GRU 的逐日 spread − 對手的逐日 spread（同日、同市場，預設 10%/5日）")
print("=" * 92)
sg = spread_series(ranks["gru"])
print(f"{'對手':<8} {'GRU年化':>9} {'對手年化':>9} {'Δ年化':>9} {'配對NW t':>9} {'GRU較優日比例':>13}")
for m in MODELS:
    if m == "gru":
        continue
    so = spread_series(ranks[m])
    d = sg - so
    print(f"{m:<8} {np.mean(sg)*TRADING_DAYS:>8.1%} {np.mean(so)*TRADING_DAYS:>9.1%} "
          f"{np.mean(d)*TRADING_DAYS:>+9.1%} {nw_t(d):>9.2f} {(d>0).mean():>13.1%}")

print()
print("── 前後半切分（同一檢定分兩段）──")
h = len(sg) // 2
for m in MODELS:
    if m == "gru":
        continue
    d = sg - spread_series(ranks[m])
    print(f"  vs {m:<7} 前半 Δ {np.mean(d[:h])*TRADING_DAYS:+6.1%} (t={nw_t(d[:h]):+5.2f}) ｜ "
          f"後半 Δ {np.mean(d[h:])*TRADING_DAYS:+6.1%} (t={nw_t(d[h:]):+5.2f})")
