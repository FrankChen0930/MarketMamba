"""GRU decile spread Sharpe = 2.846 異常診斷（純讀，不動任何既有檔案）。

量六件事：
  1. panel 是否對齊（日期集合 / 股票集合 / (Date,stock_id) 對）
  2. 重現各模型的 decile spread（確認 2.846 不是筆誤）
  3. 排名 lag-1 自相關（訊號持續性 → 換手 → Sharpe）
  4. decile 兩端的成員換手率
  5. long / short 兩腳分開看
  6. 分數本身的分布（相異值數、是否有大量 ties）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Desktop\work\ProjectForMe\MarketMamba")
sys.path.insert(0, str(ROOT / "V6"))

SCORES = ROOT / "V6/experimental/result/scores"
MODELS = ["ridge", "gbdt", "no_gat", "old_kg", "v2_kg", "v3_kg", "gru"]

sig = {}
for m in MODELS:
    d = pd.read_parquet(SCORES / f"{m}.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    sig[m] = d

print("=" * 78)
print("[1] panel 對齊檢查")
print("=" * 78)
ref = sig["ridge"]
ref_dates = set(ref["Date"].unique())
ref_stocks = set(ref["stock_id"].unique())
ref_pairs = set(map(tuple, ref[["Date", "stock_id"]].to_numpy()))

print(f"{'model':<8} {'rows':>9} {'days':>6} {'stocks':>7} "
      f"{'日交集':>7} {'股交集':>7} {'(D,S)對與ridge相同':>18}")
for m in MODELS:
    d = sig[m]
    ds, ss = set(d["Date"].unique()), set(d["stock_id"].unique())
    pairs = set(map(tuple, d[["Date", "stock_id"]].to_numpy()))
    print(f"{m:<8} {len(d):>9,} {len(ds):>6} {len(ss):>7} "
          f"{len(ds & ref_dates):>7} {len(ss & ref_stocks):>7} "
          f"{str(pairs == ref_pairs):>18}")

g, r = sig["gru"], sig["ridge"]
print(f"\ngru 日期範圍  {g['Date'].min().date()} → {g['Date'].max().date()}")
print(f"ridge 日期範圍 {r['Date'].min().date()} → {r['Date'].max().date()}")
gp = set(map(tuple, g[["Date", "stock_id"]].to_numpy()))
print(f"gru 有而 ridge 沒有的 (Date,stock) 對：{len(gp - ref_pairs):,}")
print(f"ridge 有而 gru 沒有的 (Date,stock) 對：{len(ref_pairs - gp):,}")

# 每日檔數分布
print("\n每日檔數（median / min / max）：")
for m in MODELS:
    c = sig[m].groupby("Date").size()
    print(f"  {m:<8} median {c.median():>6.0f}  min {c.min():>5}  max {c.max():>5}")

print()
print("=" * 78)
print("[2]~[6] 逐模型指標")
print("=" * 78)

from experimental.portfolio_lab import Market, decile_spread, TRADING_DAYS  # noqa: E402


def decile_detail(mkt, rank, freq=5):
    """decile_spread 的展開版：多回傳兩腳各自的統計 + 成員換手。"""
    rk = rank.to_numpy(np.float64)
    ret = mkt.ret.to_numpy(np.float64).copy()
    np.nan_to_num(ret, copy=False)
    T = len(mkt.dates)
    long_r, short_r = np.zeros(T), np.zeros(T)
    hold_l = hold_s = None
    turn_l, turn_s, sizes = [], [], []
    for t in range(T):
        if hold_l is not None:
            long_r[t] = float(ret[t][hold_l].mean())
            short_r[t] = float(ret[t][hold_s].mean())
        if t % freq == 0:
            ok = np.isfinite(rk[t])
            m = int(ok.sum())
            if m < 50:
                continue
            d = max(m // 10, 1)
            order = np.argsort(np.where(ok, rk[t], np.inf))
            nl, ns = order[:d], order[m - d:m]
            if hold_l is not None:
                turn_l.append(1 - len(set(nl) & set(hold_l)) / len(nl))
                turn_s.append(1 - len(set(ns) & set(hold_s)) / len(ns))
            hold_l, hold_s = nl, ns
            sizes.append(d)
    ls = long_r - short_r

    def ann(x):
        return (1 + pd.Series(x)).prod() ** (TRADING_DAYS / len(x)) - 1

    def sharpe(x):
        return float(np.mean(x) / np.std(x) * np.sqrt(TRADING_DAYS)) if np.std(x) > 0 else np.nan

    return {
        "decile_size": int(np.median(sizes)),
        "long_ann": ann(long_r), "long_sharpe": sharpe(long_r),
        "short_ann": ann(short_r), "short_sharpe": sharpe(short_r),
        "spread_ann": float(np.mean(ls) * TRADING_DAYS),
        "spread_sharpe": sharpe(ls),
        "spread_daily_std": float(np.std(ls)),
        "spread_daily_mean": float(np.mean(ls)),
        "turnover_long": float(np.mean(turn_l)), "turnover_short": float(np.mean(turn_s)),
    }


out = {}
for m in MODELS:
    d = sig[m]
    rank = d.pivot(index="Date", columns="stock_id", values="score") \
            .rank(axis=1, ascending=False, method="first")
    mkt = Market(rank.index.to_numpy(), list(rank.columns))
    rank = rank.reindex(index=mkt.dates, columns=mkt.stocks).where(mkt.px.notna())

    det = decile_detail(mkt, rank)
    # 排名 lag-1 自相關（逐日 Spearman，用 rank 直接算 Pearson 即等價）
    rk = rank.to_numpy(np.float64)
    acs = []
    for t in range(1, len(rk)):
        a, b = rk[t - 1], rk[t]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 50:
            acs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    det["rank_ac1"] = float(np.mean(acs))
    # 分數分布
    nuq = d.groupby("Date")["score"].nunique()
    cnt = d.groupby("Date").size()
    det["uniq_ratio"] = float((nuq / cnt).median())
    det["score_std"] = float(d.groupby("Date")["score"].std().median())
    det["n_days"] = len(mkt.dates)
    det["n_stocks"] = len(mkt.stocks)
    out[m] = det
    print(f"  [{m}] 完成", flush=True)

df = pd.DataFrame(out).T
pd.set_option("display.width", 220, "display.max_columns", 50)
print()
print("── decile 兩腳拆解 ──")
print(df[["n_days", "n_stocks", "decile_size", "long_ann", "long_sharpe",
          "short_ann", "short_sharpe", "spread_ann", "spread_sharpe"]].round(4).to_string())
print()
print("── 為什麼 Sharpe 高：分子 vs 分母 + 持續性 ──")
print(df[["spread_daily_mean", "spread_daily_std", "spread_sharpe",
          "rank_ac1", "turnover_long", "turnover_short",
          "uniq_ratio", "score_std"]].round(4).to_string())

df.to_csv(Path(__file__).parent / "gru_decile_diag.csv")
print("\n已存 gru_decile_diag.csv")
