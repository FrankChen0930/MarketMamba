"""
portfolio_lab.py — 組合建構基準版 v1.0 的網格掃描
==================================================
對應規格：`docs/portfolio-construction-baseline-v1.md`（跑之前已凍結）

為什麼要這支
------------
現行口徑（Top50 等權、5 日再平衡、無緩衝）實測換手 70–77%／次：

    每次成本 = 0.71 × 0.15%(買) + 0.71 × 0.45%(賣) = 0.426%
    每年 50.4 次 → 年化成本拖累 ≈ 19.4%

而 `v2_kg` 的**淨**年化只有 +8.3% → **被成本吃掉的比留下的還多**。
（獨立驗證：D4 測成本 ×2 時 Ridge 從 +18.7% 掉到 −5.4%，差 20.3pp，與上式吻合。）

→ 能砍半換手的設計約值 +10pp 年化；而 IC 上量過最大的一次只值 1–2pp。
**這是在修「年化」這把量尺，不是改做組合層。**

隔離原則
--------
**純附加**：不動 `baseline_common.portfolio_backtest`——方向二已發表的所有組合層
數字都引用它，改了那些數字就無法重現。本檔自成一套，輸出獨立的 JSON。

設計要點
--------
1. **價格／流動性／產業只載入一次**，240 組共用。`portfolio_backtest` 每次呼叫都
   重讀 prices_raw，跑 240 組會是災難。
2. **每日排名預先算好**（582 × N_stocks 的 rank 矩陣），之後每組只剩集合運算。
3. 緩衝（banding）的精確定義：

       每個再平衡日 d：
         1. 賣出：現有持股中 rank(d) > k×N 者
         2. 買進：從 rank(d) ≤ N 且未持有者中，依 rank 由小到大補滿到 N 檔

   候選永遠足夠：`keep ⊆ holdings` 且 `|holdings| ≤ N`；需要補的槽位是
   `N − |keep|`，而 top-N 中未被 keep 佔用的名額是 `N − |keep ∩ topN| ≥ N − |keep|`。

用法
----
    # 1) 產生分數（Ridge / GBDT 本機可跑；GRU 需 WSL torch；Mamba 需 Colab）
    python V6/experimental/portfolio_lab.py --export-scores ridge
    python V6/experimental/portfolio_lab.py --export-scores gbdt

    # 2) 掃網格（讀 result/scores_*.parquet，全部模型一起跑）
    python V6/experimental/portfolio_lab.py --sweep

    # 3) 只印報表
    python V6/experimental/portfolio_lab.py --report

Mamba 三組的分數請在 Colab 產生（checkpoint 在 Drive，前向一次不用重訓），
見本檔尾端 `COLAB_SNIPPET`。
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

RESULT_DIR = _THIS_DIR / "result"
SCORE_DIR = RESULT_DIR / "scores"
OUT_PATH = RESULT_DIR / "portfolio_lab_result.json"

# ── 基準版：固定，不掃描（規格 §1）────────────────────────────────
COST_BUY = 0.0015          # 法定 0.1425%，取整偏保守
COST_SELL = 0.0045         # 0.1425% + 證交稅 0.30% = 0.4425%，取整偏保守
LIQ_WINDOW = 20            # 流動性用近 20 日成交金額中位數
# 非等權配權的單股權重上限（= WEIGHT_CAP × 等權）。台股成交金額極度偏斜，
# 不設上限的話流動性加權會退化成「整個組合押一檔台積電」（實測最大權重 84.95%）。
WEIGHT_CAP = 3.0
GRID_WEIGHT = ["equal", "inv_vol", "liquidity"]     # v1.1 新增維度

# ── 掃描網格：240 組（規格 §2）────────────────────────────────────
GRID_N = [10, 25, 50, 100, 224]          # 224 ≈ 宇宙 2,245 的前 10%
GRID_K = [1.0, 1.3, 1.5, 2.0]            # 1.0 = 無緩衝（現行行為）
GRID_FREQ = [1, 3, 5, 10, 20]            # 10 日為 2026-08-01 使用者要求補上（v1.1 提案 C）
GRID_LIQ = [None, 0.333, 0.667]          # 保留成交金額百分位 ≥ 此值；None = 不過濾

# ── Headline：跑之前定死，依慣例選，不看數字（規格 §4）────────────
HEADLINES = {
    # v1.1 提案 A（2026-08-01 使用者批准）：「貼近實際操作」的頻率 每日 → 20 日。
    # 失誤在把「每天看 dashboard」誤讀成「每天換股」；實測每日再平衡 Ridge −16.3%、
    # GBDT −31.6%，根因是 Top50 名單隔天只留存 34–47%。
    # ⚠️ 這是**看到數字之後**才提的修訂，所有模型都在 v1.1 下重跑，新舊數字不得混用。
    "標準因子研究口徑": {"n": 224, "k": 1.5, "freq": 5, "liq": None},
    "貼近實際操作":     {"n": 25,  "k": 1.5, "freq": 20, "liq": 0.667},
}

# ── v1.1 擴充網格（108 組）─────────────────────────────────────────────
# 主網格（300 組）維持 v1.0 不動，擴充維度另開一張表：
#   配權（提案 B 的同伴）、分數平滑（提案 B）、交易限制（風控 C）
# **取值範圍的縮減有獨立依據**：11 年 WF 已驗證低頻（10/20 日）+ 緩衝（1.5/2.0）
# 是有效區間，不是看 582 天窗的結果挑的。
GRID_EXT = {
    "n":      [25, 50, 100],
    "k":      [1.5, 2.0],
    "freq":   [10, 20],
    "weight": GRID_WEIGHT,
    "smooth": [1, 5, 10],          # 分數先做 w 日移動平均再排名
}

TRADING_DAYS = 252


# ============================================================
# 1) 市場資料（載入一次，240 組共用）
# ============================================================
class Market:
    """收盤價 / 流動性百分位 / 產業別，全部對齊到同一組 (dates × stocks)。"""

    def __init__(self, dates: np.ndarray, stocks: list[str]):
        from experimental.baseline_common import _filter_universe, _load_raw

        self.dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
        d0, d1 = self.dates[0], self.dates[-1]
        print(f"[market] 載入 {d0.date()} → {d1.date()} 的價格 ...", flush=True)

        pr = _load_raw("prices_raw")
        pr = pr[(pr["Date"] >= d0) & (pr["Date"] <= d1 + pd.Timedelta(days=30))]
        pr = _filter_universe(pr)
        pr = pr.drop_duplicates(subset=["stock_id", "Date"], keep="last")
        pr = pr[pr["stock_id"].isin(set(stocks))]

        px = pr.pivot(index="Date", columns="stock_id", values="Close").sort_index()
        px = px.where(px > 0)                       # Close<=0 視為缺值
        # 成交金額（元）：Close × Volume。Volume 單位是股。
        dv = pr.pivot(index="Date", columns="stock_id", values="Volume").sort_index()
        dv = (dv * px).where(lambda x: x > 0)

        self.px_full = px                            # 含 val 窗之後的日子（算最後一段報酬用）
        self.px = px.reindex(self.dates)
        # 流動性：近 20 日成交金額中位數 → 當日橫斷面百分位（0=最差, 1=最好）
        liq = dv.rolling(LIQ_WINDOW, min_periods=5).median().reindex(self.dates)
        self.liq_pct = liq.rank(axis=1, pct=True)
        self.dvol = liq                                    # 流動性加權用（成交金額中位數）
        # 波動度倒數加權用：近 60 日報酬標準差（在 self.ret 之後才算，見下）

        # 每日報酬（用 ffill 後的價格，停牌視為持平——與舊 portfolio_backtest 一致）
        pxf = self.px_full.ffill()
        self.ret = pxf.pct_change().reindex(self.dates)
        # 波動度倒數加權：近 60 日報酬標準差（shift(1) 避免用到當日資訊）
        self.vol = pxf.pct_change().rolling(60, min_periods=20).std().shift(1).reindex(self.dates)

        # ── 漲跌停（風控 C 類）──────────────────────────────────────────
        # **不需要新資料源**：台股漲跌幅上限 2015-06-01 起 ±10%、之前 ±7%，
        # 直接由還原收盤價的日報酬推導。門檻取 0.095 / 0.065（留 tick 進位餘裕）。
        #   收盤達漲停 → 買不到（買方排隊）
        #   收盤達跌停 → 賣不掉（賣方排隊）
        _lim = np.where(self.dates < pd.Timestamp("2015-06-01"), 0.065, 0.095)[:, None]
        _r = self.ret.to_numpy(np.float64)
        self.at_limit_up = _r >= _lim
        self.at_limit_down = _r <= -_lim

        # ── 處置股（風控 C 類，2026-08-01 補）────────────────────────────
        # 處置 = 分盤集合競價（5 或 20 分鐘一次）+ 預收款券 → 實務上很難照收盤價成交。
        # 「注意股」只是警示、無交易限制，故**不擋**（仍讀進來供診斷）。
        self.under_disposal = np.zeros((len(self.dates), len(self.px.columns)), dtype=bool)
        from marketmamba.config import PROCESSED_DIR as _PD
        _ts = Path(_PD) / "trading_status_raw.parquet"
        if _ts.exists():
            t = pd.read_parquet(_ts)
            t = t[t["status"] == "disposal"]
            t["Date"] = pd.to_datetime(t["Date"])
            di = {d: i for i, d in enumerate(self.dates)}
            ci = {c: i for i, c in enumerate(self.px.columns)}
            hit = 0
            for d, sid in zip(t["Date"], t["stock_id"].astype(str)):
                i, j = di.get(d), ci.get(sid)
                if i is not None and j is not None:
                    self.under_disposal[i, j] = True
                    hit += 1
            print(f"[market] 處置股：窗內命中 {hit:,} 個「股票×日」"
                  f"（{self.under_disposal.any(axis=0).sum()} 支）", flush=True)
        else:
            print(f"[market] ⚠ 找不到 {_ts.name} → 處置限制不可用（block_disposal 將無效果）",
                  flush=True)

        self.stocks = list(self.px.columns)
        print(f"[market] {len(self.dates)} 天 × {len(self.stocks)} 支｜"
              f"價格缺值 {self.px.isna().mean().mean():.1%}｜"
              f"流動性可用 {self.liq_pct.notna().mean().mean():.1%}", flush=True)

        self.sector = self._load_sector()

    @staticmethod
    def _load_sector() -> dict[str, str]:
        try:
            from marketmamba.data.feature_spec import canonical_sector
            from marketmamba.data.hygiene import load_stock_info
            info = load_stock_info(latest_only=True)
            col = next((c for c in ("industry_category", "industry", "sector")
                        if c in info.columns), None)
            if col is None:
                return {}
            # canonical_sector 吃的是 Series 不是 str（逐列呼叫會 AttributeError）
            sec = canonical_sector(info[col])
            return dict(zip(info["stock_id"].astype(str), sec))
        except Exception as e:                                # noqa: BLE001
            print(f"[market] ⚠ 產業別載入失敗（產業集中度將略過）：{type(e).__name__}: {e}",
                  flush=True)
            return {}


# ============================================================
# 2) 單一組合設定的回測
# ============================================================
def _weights(mkt: Market, t: int, names: list[str], col_idx: dict, scheme: str) -> np.ndarray:
    """
    配權方式（v1.1 新增；基準版仍是 equal）。

    - equal    ：等權。baseline，不引入「分數大小可不可信」這個變因
    - inv_vol  ：1 / 近 60 日報酬標準差。風險平價的簡化版，壓低小型高波動股
    - liquidity：∝ 近 20 日成交金額中位數。直接對應滑價/可執行性
    三者都只用 t 之前的資訊（vol 已 shift(1)、dvol 是 rolling median）。
    權重異常（NaN / 全 0）一律退回等權，並不會靜默產生 0 部位。

    ⚠️ **必須設權重上限**（2026-08-01 實測）：台股成交金額極度偏斜，
    原始比例配權會退化成單一持股——實測 `liquidity` 的最大單股權重達 **84.95%**
    （等於整個組合押一檔台積電）、`inv_vol` 也有 28.5%。
    故一律套 `WEIGHT_CAP × (1/N)` 的上限並反覆重新正規化（等權不受影響）。
    """
    m = len(names)
    if scheme == "equal":
        return np.full(m, 1.0 / m)
    idx = [col_idx[s] for s in names]
    if scheme == "inv_vol":
        v = mkt.vol.to_numpy(np.float64)[t][idx]
        raw = 1.0 / np.where(np.isfinite(v) & (v > 1e-6), v, np.nan)
    elif scheme == "liquidity":
        raw = mkt.dvol.to_numpy(np.float64)[t][idx]
        raw = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
    else:
        raise ValueError(f"未知配權方式 {scheme!r}")
    if not np.isfinite(raw).any():
        return np.full(m, 1.0 / m)
    raw = np.where(np.isfinite(raw), raw, np.nanmedian(raw))   # 缺值補中位數，不丟部位
    s = raw.sum()
    if s <= 0:
        return np.full(m, 1.0 / m)
    w = raw / s
    cap = WEIGHT_CAP / m                                       # 上限 = WEIGHT_CAP 倍的等權
    for _ in range(50):                                        # 反覆截頂 + 重新正規化
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            return np.full(m, 1.0 / m)
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def run_config(mkt: Market, rank: pd.DataFrame, n: int, k: float, freq: int,
               liq: float | None, cost_mult: float = 1.0,
               weight: str = "equal", block_limit: bool = False,
               block_disposal: bool = False) -> dict:
    """
    rank: (dates × stocks) 的每日排名（1 = 分數最高；不可交易者為 NaN）
    回傳年化/Sharpe/MDD/換手/年化成本拖累/產業集中度/最大權重。
    """
    dates = mkt.dates
    reb_idx = list(range(0, len(dates), freq))
    kn = k * n

    holdings: list[str] = []
    w = np.zeros(len(mkt.stocks))
    col_idx = {s: i for i, s in enumerate(mkt.stocks)}
    ret_np = mkt.ret.to_numpy(np.float64)
    np.nan_to_num(ret_np, copy=False)               # 缺值日視為 0 報酬（同停牌持平）

    port_ret = np.zeros(len(dates))
    cost_series = np.zeros(len(dates))
    turnovers: list[float] = []
    maxw: list[float] = []
    sector_top3: list[float] = []
    sector_max: list[float] = []
    reb_set = set(reb_idx)

    rank_np = rank.to_numpy(np.float64)
    liq_np = mkt.liq_pct.to_numpy(np.float64) if liq is not None else None

    for t in range(len(dates)):
        # ① 先讓昨日的權重吃今天的報酬
        if holdings:
            r = ret_np[t]
            idx = [col_idx[s] for s in holdings]
            gross = 1.0 + r[idx]
            v = w[idx] * gross
            tot = v.sum()
            port_ret[t] = tot - w[idx].sum()        # w 已正規化成合計 1
            if tot > 0:
                w[idx] = v / tot                    # 重新正規化（權重自然漂移）
            maxw.append(float(w[idx].max()) if len(idx) else 0.0)

        # ② 再平衡（收盤後決定，下一日才吃到新組合的報酬）
        if t in reb_set:
            rk = rank_np[t].copy()
            if liq_np is not None:
                rk = np.where(liq_np[t] >= liq, rk, np.nan)
            ok = np.isfinite(rk)
            if ok.sum() < n:
                continue
            order = np.argsort(np.where(ok, rk, np.inf))

            if block_disposal and not block_limit:
                # 只擋處置：處置中的候選不買（已持有者仍可賣——處置限制的是成交難度，
                # 不是禁止交易；賣出的一方在預收款券下仍可執行）
                dz = mkt.under_disposal[t]
                top_n, i_ = [], 0
                while len(top_n) < n and i_ < len(order):
                    if ok[order[i_]] and not dz[order[i_]]:
                        top_n.append(mkt.stocks[order[i_]])
                    i_ += 1
                keep = [s for s in holdings
                        if np.isfinite(rk[col_idx[s]]) and rk[col_idx[s]] <= kn]
                keep_set = set(keep)
                adds = [s for s in top_n if s not in keep_set][:max(n - len(keep), 0)]
            elif block_limit:
                # 風控 C：買不到 / 賣不掉。**順延而不是留空**——留空等於偷偷降低曝險。
                lu, ld = mkt.at_limit_up[t], mkt.at_limit_down[t]
                if block_disposal:
                    lu = lu | mkt.under_disposal[t]        # 處置中也視為買不到
                # 賣出受阻：本來要賣（跌出緩衝）但收在跌停 → 繼續持有
                keep = [s for s in holdings
                        if (np.isfinite(rk[col_idx[s]]) and rk[col_idx[s]] <= kn)
                        or ld[col_idx[s]]]
                keep_set = set(keep)
                need = n - len(keep)
                # 買入受阻：收在漲停的候選跳過，往後順延取下一名
                adds, i_ = [], 0
                while len(adds) < max(need, 0) and i_ < len(order):
                    c = mkt.stocks[order[i_]]
                    i_ += 1
                    if not ok[order[i_ - 1]] or c in keep_set or lu[order[i_ - 1]]:
                        continue
                    adds.append(c)
            else:
                top_n = [mkt.stocks[i] for i in order[:n]]
                keep = [s for s in holdings
                        if np.isfinite(rk[col_idx[s]]) and rk[col_idx[s]] <= kn]
                keep_set = set(keep)
                need = n - len(keep)
                adds = [s for s in top_n if s not in keep_set][:max(need, 0)]
            new = keep + adds

            # ── 換手改用「權重變動」而非「檔數比例」（v1.1 修訂 E）──
            # 非等權下檔數比例會嚴重失真（例如流動性加權會把權重集中在少數大型股，
            # 換掉 10 檔小部位的實際成本遠低於檔數比例算出來的）。
            # 權重法對等權也更正確——它把窗內漂移造成的再平衡量算進去。
            w_new = np.zeros(len(mkt.stocks))
            wt = _weights(mkt, t, new, col_idx, weight)
            for s, ww in zip(new, wt):
                w_new[col_idx[s]] = ww
            delta = w_new - w
            buy_frac = float(delta[delta > 0].sum())     # 首次建倉時 = 1.0、賣出 = 0
            sell_frac = float(-delta[delta < 0].sum())
            c = (buy_frac * COST_BUY + sell_frac * COST_SELL) * cost_mult
            cost_series[t] = c
            port_ret[t] -= c
            turnovers.append(buy_frac)

            if mkt.sector and new:
                # 風控 A：產業集中度。以**權重**計（不是檔數）——非等權下檔數會低估集中度
                sw: dict[str, float] = {}
                for s_, ww in zip(new, wt):
                    sw[mkt.sector.get(s_, "Unknown")] = sw.get(mkt.sector.get(s_, "Unknown"), 0.0) + ww
                order_s = sorted(sw.values(), reverse=True)
                sector_top3.append(float(sum(order_s[:3])))
                sector_max.append(float(order_s[0]))

            w = w_new
            holdings = new

    return _summarize(port_ret, cost_series, turnovers, maxw, sector_top3, sector_max,
                      len(reb_idx))


def _ann(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    c = float((1 + x).prod())
    return c ** (TRADING_DAYS / len(x)) - 1 if c > 0 else -1.0


def _sh(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    s = x.std()
    return round(float(x.mean() / s * np.sqrt(TRADING_DAYS)), 3) if s > 0 else None


def _summarize(port_ret, cost_series, turnovers, maxw, sector_top3, sector_max,
               n_reb) -> dict:
    r = pd.Series(port_ret)
    r = r[np.isfinite(r)]
    n = len(r)
    cum = float((1 + r).prod())
    ann = cum ** (TRADING_DAYS / n) - 1 if n > 0 and cum > 0 else -1.0
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else None
    curve = (1 + r).cumprod()
    mdd = float((curve / curve.cummax() - 1).min())
    # 年化成本拖累：把實際扣掉的成本複利年化
    tot_cost = float(np.sum(cost_series))
    ann_cost = 1 - (1 - tot_cost / max(n, 1)) ** TRADING_DAYS if n else 0.0
    # 子期間（前半／後半）：2026-08-01 加。單一 2.3 年窗口 + 240 格網格必然有選擇偏誤，
    # 而「20 日最好」很可能與這段行情的趨勢延續性混在一起（使用者 2026-08-01 指出）。
    # 直接切日報酬序列即可——策略是連續的，切的是量測窗不是策略。
    h = n // 2
    rv = r.to_numpy()

    # ── 風控 B（檢核表 B 類）：波動、回撤深度**與持續期間**、觀察型警戒線 ──
    curve = (1 + r).cumprod()
    dd = (curve / curve.cummax() - 1).to_numpy()
    under = dd < -1e-9                       # 是否處於回撤中
    longest, cur_len = 0, 0
    for u in under:
        cur_len = cur_len + 1 if u else 0
        longest = max(longest, cur_len)
    roll_vol = r.rolling(20).std() * np.sqrt(TRADING_DAYS)
    return {
        "ann_return": round(float(ann), 4),
        "ann_sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown": round(mdd, 4),
        "avg_turnover": round(float(np.mean(turnovers)), 3) if turnovers else 0.0,
        "ann_cost_drag": round(float(ann_cost), 4),
        "n_rebalances": n_reb,
        "n_days": n,
        "max_weight": round(float(np.max(maxw)), 4) if maxw else None,
        "sector_top3_share": round(float(np.mean(sector_top3)), 3) if sector_top3 else None,
        "ann_1h": round(_ann(rv[:h]), 4), "ann_2h": round(_ann(rv[h:]), 4),
        "sharpe_1h": _sh(rv[:h]), "sharpe_2h": _sh(rv[h:]),
        # 風控 B
        "ann_vol": round(float(r.std() * np.sqrt(TRADING_DAYS)), 4),
        "roll_vol20_median": round(float(roll_vol.median()), 4),
        "roll_vol20_max": round(float(roll_vol.max()), 4),
        "max_dd_duration_days": int(longest),
        "pct_days_underwater": round(float(under.mean()), 3),
        "n_days_dd_below_15pct": int((dd < -0.15).sum()),      # 觀察型警戒線（不自動執行）
        # 風控 A
        "sector_max_share": round(float(np.mean(sector_max)), 3) if sector_max else None,
        "sector_max_worst": round(float(np.max(sector_max)), 3) if sector_max else None,
        "sector_over_40pct_pct": round(float(np.mean(np.array(sector_max) > 0.40)), 3)
        if sector_max else None,
        # 風控 E：換手穩定性（無預警飆升 = 排序變不穩）
        "turnover_std": round(float(np.std(turnovers)), 3) if turnovers else None,
        "turnover_max": round(float(np.max(turnovers)), 3) if turnovers else None,
        "_daily": rv,
    }


# ============================================================
# 3) 基準與診斷
# ============================================================
def equal_weight_universe(mkt: Market, rank: pd.DataFrame) -> np.ndarray:
    """主基準：等權 eligible 宇宙、每日再平衡、不計成本（規格 §1）。"""
    ok = np.isfinite(rank.to_numpy(np.float64))
    ret = mkt.ret.to_numpy(np.float64).copy()
    np.nan_to_num(ret, copy=False)
    out = np.zeros(len(mkt.dates))
    for t in range(len(mkt.dates)):
        m = ok[t]
        if m.sum() > 0:
            out[t] = float(ret[t][m].mean())
    return out


def signal_health(mkt: Market, rank: pd.DataFrame, horizon: int = 5) -> dict:
    """
    風控 E（檢核表 E 類）：訊號失效監控。

    只看整段回測期的平均 IC，看不出訊號**何時**開始壞掉。這裡算：
      - 逐日 rank IC（分數排名 vs 未來 `horizon` 日報酬）
      - **滾動 20 / 60 日 IC** —— 訊號系統性衰退會先出現在這裡
      - 滾動 60 日 IC 轉負的期間佔比 + 最長連續轉負天數

    前瞻報酬直接由 `mkt.px_full`（還原收盤價）算，與組合層同一套價格序列，
    不另外維護一份（檢核表 C 類最後一項的要求）。
    """
    pxf = mkt.px_full.ffill()
    fwd = (pxf.shift(-horizon) / pxf - 1.0).reindex(index=mkt.dates, columns=mkt.stocks)
    rk = rank.to_numpy(np.float64)
    fw = fwd.to_numpy(np.float64)

    ics: list[float] = []
    for t in range(len(mkt.dates)):
        m = np.isfinite(rk[t]) & np.isfinite(fw[t])
        if m.sum() < 30:
            ics.append(np.nan)
            continue
        a = pd.Series(rk[t][m]).rank()
        b = pd.Series(fw[t][m]).rank()
        ics.append(float(np.corrcoef(a, b)[0, 1]))
    s = pd.Series(ics, index=mkt.dates)
    # 排名是「1 = 最好」，與報酬方向相反 → 取負號讓 IC 的正負符合直覺
    s = -s
    r20, r60 = s.rolling(20).mean(), s.rolling(60).mean()

    neg = (r60 < 0).fillna(False).to_numpy()
    longest, cur = 0, 0
    for x in neg:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    return {
        "mean_ic": round(float(s.mean()), 4),
        "ic_std": round(float(s.std()), 4),
        "icir": round(float(s.mean() / s.std()), 3) if s.std() > 0 else None,
        "pct_days_ic_positive": round(float((s > 0).mean()), 3),
        "roll20_ic_min": round(float(r20.min()), 4),
        "roll20_ic_max": round(float(r20.max()), 4),
        "roll60_ic_min": round(float(r60.min()), 4),
        "pct_roll60_negative": round(float(np.mean(neg)), 3),
        "longest_roll60_negative_days": int(longest),
        "roll60_by_year": {str(y): round(float(v), 4)
                           for y, v in r60.groupby(r60.index.year).mean().items()},
    }


def decile_spread(mkt: Market, rank: pd.DataFrame, freq: int = 5) -> dict:
    """診斷用（非可交易策略）：前 10% − 後 10%，等權、不計成本。"""
    rk = rank.to_numpy(np.float64)
    ret = mkt.ret.to_numpy(np.float64).copy()
    np.nan_to_num(ret, copy=False)
    long_r, short_r = np.zeros(len(mkt.dates)), np.zeros(len(mkt.dates))
    hold_l, hold_s = None, None
    for t in range(len(mkt.dates)):
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
            hold_l, hold_s = order[:d], order[m - d:m]
    ls = long_r - short_r
    ann_l = (1 + pd.Series(long_r)).prod() ** (TRADING_DAYS / len(long_r)) - 1
    return {
        "decile_long_ann": round(float(ann_l), 4),
        "decile_spread_ann_simple": round(float(ls.mean() * TRADING_DAYS), 4),
        "decile_spread_sharpe": round(float(ls.mean() / ls.std() * np.sqrt(TRADING_DAYS)), 3)
        if ls.std() > 0 else None,
    }


def tracking_error(strat: np.ndarray, bench: np.ndarray) -> tuple[float, float]:
    d = np.asarray(strat) - np.asarray(bench)
    d = d[np.isfinite(d)]
    te = float(d.std() * np.sqrt(TRADING_DAYS))
    excess = float((1 + pd.Series(d)).prod() ** (TRADING_DAYS / len(d)) - 1) if len(d) else 0.0
    return round(excess, 4), round(te, 4)


# ============================================================
# 4) 分數產生（Ridge / GBDT 本機可跑）
# ============================================================
def export_scores(model: str, label: str = "5d", purge_days: int = 0) -> Path:
    """
    產生 test 窗（2024-01-01 → 2026-06-02）的逐股分數 → result/scores/{model}.parquet

    設定固定沿用 F5 的 R1／G-R1（v2 協定、train 2013、旗標關、fund_v2 開、
    neutralize none、purge 關），與 F6 的 val 窗同一批交易日。

    label / purge_days（2026-08-03 新增，標籤 horizon 實驗用）
    ---------------------------------------------------------
    - `label`："5d"（預設）／"10d"／"20d"，決定訓練用哪個 rank 標籤、
      以及選超參數時對哪個 horizon 的 alpha 算 val IC。
    - `purge_days`：訓練集尾端剔除幾個**交易日**。現行協定是
      `TRAIN_END=2023-12-31` 緊接 `TEST_START=2024-01-01`、**中間沒有間隔**，
      所以標籤有 h 天的重疊洩漏。在四階對照裡那是共用的缺陷（無害於相對比較），
      **但一旦改變 label horizon，洩漏量就跟著改變**（5d 洩 5 天、20d 洩 20 天），
      會系統性偏袒長 horizon——正是本實驗要檢定的那一組。
      → 標籤 horizon 實驗一律三組同用 `purge_days=20`（= 最長 horizon）。

    **預設參數（"5d", 0）逐位元維持既有行為**，輸出檔名也維持 `{model}.parquet`；
    只要任一參數非預設，就改用 `{model}__lab{label}_p{purge}.parquet`，
    既有分數檔絕不被覆蓋。
    """
    import os
    if os.environ.get("MM_PROTOCOL") != "v2":
        raise SystemExit("❌ 請設 MM_PROTOCOL=v2 再跑（分數要用 v2 矩陣）")
    if label not in ("5d", "10d", "20d"):
        raise SystemExit(f"❌ label 只能是 5d / 10d / 20d（收到 {label!r}）")

    from experimental.baseline_common import PROTOCOL, all_feature_names, load_xy
    from experimental.baseline_ridge_lasso import (
        RIDGE_ALPHAS, TRAIN_STRIDE, gram_stats, mean_daily_ic, ridge_solve, stats_add,
    )
    t0 = time.time()
    default_run = (label == "5d" and purge_days == 0)
    # 非預設 run 一律載入統一標籤快照（三個 horizon 同源），預設 run 不碰 → 行為不變
    extra = not default_run
    names_all = all_feature_names()
    keep = np.array([not n.startswith("Avail_") for n in names_all])   # 旗標關（同 R1）
    print(f"[scores] model={model} label={label} purge={purge_days} 交易日"
          f"｜{'預設 run（與既有分數檔同設定）' if default_run else '標籤 horizon 實驗 run'}",
          flush=True)

    print(f"[scores] 載入 train span 2013-01-01 → {PROTOCOL['TEST_END']} ...", flush=True)
    tr = load_xy("2013-01-01", PROTOCOL["TEST_END"], day_stride=TRAIN_STRIDE, extra_labels=extra)
    tr["X"] = np.ascontiguousarray(tr["X"][:, keep])
    gc.collect()
    te = load_xy(PROTOCOL["TEST_START"], PROTOCOL["TEST_END"], day_stride=1, extra_labels=extra)
    te["X"] = np.ascontiguousarray(te["X"][:, keep])
    gc.collect()
    print(f"[scores] train {tr['X'].shape} | test {te['X'].shape}", flush=True)

    dates_tr = pd.DatetimeIndex(tr["dates"])
    train_days = [str(d)[:10] for d in np.sort(dates_tr[dates_tr <= pd.Timestamp(PROTOCOL["TRAIN_END"])].unique())]
    if purge_days:
        # `purge_days` 的語意是**交易日**。但 train_days 已被 TRAIN_STRIDE 抽樣過
        # （每 stride 個交易日取一天），所以要剔除的抽樣日數 = purge_days / stride，
        # 無條件進位（寧可多隔離一點，不可少）。兩個數字都印出來供判讀。
        n_drop = int(np.ceil(purge_days / TRAIN_STRIDE))
        n0 = len(train_days)
        dropped = train_days[-n_drop:]
        train_days = train_days[:-n_drop]
        print(f"[scores] purge：目標隔離 {purge_days} 個交易日｜stride={TRAIN_STRIDE} → "
              f"剔除 {n_drop} 個抽樣訓練日（{dropped[0]} → {dropped[-1]}）｜"
              f"訓練日 {n0} → {len(train_days)}｜"
              f"新的訓練尾端 {train_days[-1]}，TEST_START={PROTOCOL['TEST_START']}", flush=True)
    cut = int(len(train_days) * (1 - PROTOCOL["VAL_RATIO"]))
    val_days, fit_days = set(train_days[cut:]), set(train_days[:cut])
    ds = pd.Series(dates_tr.astype(str).str.slice(0, 10))
    val_m0, fit_m0 = ds.isin(val_days).to_numpy(), ds.isin(fit_days).to_numpy()

    y = tr[f"rank_{label}"]
    ok = ~np.isnan(y)
    fit_m, val_m, trn_m = fit_m0 & ok, val_m0 & ok, (fit_m0 | val_m0) & ok

    if model == "ridge":
        st = gram_stats(tr["X"], y, fit_m)
        best_a, best_v = None, -9
        for a in RIDGE_ALPHAS:
            w_raw, c, _ = ridge_solve(st, a)
            v = mean_daily_ic(tr["dates"][val_m], tr["X"][val_m] @ w_raw + c, tr[f"alpha_{label}"][val_m])
            if not np.isnan(v) and v > best_v:
                best_a, best_v = a, v
        print(f"[scores] ridge best α={best_a:.0e}（val IC {best_v:+.4f}）", flush=True)
        w_raw, c, _ = ridge_solve(stats_add(st, gram_stats(tr["X"], y, val_m)), best_a)
        scores = te["X"] @ w_raw + c
    elif model == "gbdt":
        import lightgbm as lgb
        from experimental.f5_r_series import GBDT_EARLY_STOP, GBDT_MAX_ROUNDS, GBDT_PARAMS
        ds_fit = lgb.Dataset(tr["X"][fit_m], label=y[fit_m].astype(np.float64))
        ds_val = lgb.Dataset(tr["X"][val_m], label=y[val_m].astype(np.float64), reference=ds_fit)
        b = lgb.train(GBDT_PARAMS, ds_fit, num_boost_round=GBDT_MAX_ROUNDS,
                      valid_sets=[ds_val], valid_names=["val"],
                      callbacks=[lgb.early_stopping(GBDT_EARLY_STOP, verbose=False)])
        n_best = int(b.best_iteration)
        print(f"[scores] gbdt early-stop {n_best} 輪", flush=True)
        del ds_fit, ds_val, b
        gc.collect()
        final = lgb.train(GBDT_PARAMS, lgb.Dataset(tr["X"][trn_m],
                                                   label=y[trn_m].astype(np.float64)),
                          num_boost_round=n_best)
        scores = final.predict(te["X"])
    else:
        raise SystemExit(f"❌ 本機只能產生 ridge / gbdt 的分數（收到 {model!r}）；"
                         f"GRU 需 WSL torch、Mamba 三組需 Colab（見檔尾 COLAB_SNIPPET）")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORE_DIR / (f"{model}.parquet" if default_run
                       else f"{model}__lab{label}_p{purge_days}.parquet")
    pd.DataFrame({"Date": te["dates"], "stock_id": te["stock_ids"],
                  "score": np.asarray(scores, dtype=np.float32)}).to_parquet(out, index=False)
    print(f"✅ [scores] {model}：{len(scores):,} 列 → {out.name}（{(time.time()-t0)/60:.1f} 分）",
          flush=True)
    return out


# ============================================================
# 5) 網格掃描
# ============================================================
def sweep(models: list[str] | None = None) -> dict:
    files = sorted(SCORE_DIR.glob("*.parquet")) if SCORE_DIR.exists() else []
    if models:
        files = [f for f in files if f.stem in models]
    if not files:
        raise SystemExit(f"❌ 找不到分數檔：{SCORE_DIR}\n"
                         f"   先跑 MM_PROTOCOL=v2 python V6/experimental/portfolio_lab.py "
                         f"--export-scores ridge")

    all_res = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    all_res.setdefault("models", {})
    all_res["spec"] = {
        "doc": "docs/portfolio-construction-baseline-v1.md",
        "cost_buy": COST_BUY, "cost_sell": COST_SELL,
        "grid_n": GRID_N, "grid_k": GRID_K, "grid_freq": GRID_FREQ, "grid_liq": GRID_LIQ,
        "headlines": HEADLINES, "liq_window": LIQ_WINDOW,
    }

    for f in files:
        name = f.stem
        t0 = time.time()
        print(f"\n{'='*72}\n[sweep] {name}\n{'='*72}", flush=True)
        sig = pd.read_parquet(f)
        sig["Date"] = pd.to_datetime(sig["Date"])
        rank = sig.pivot(index="Date", columns="stock_id", values="score") \
                  .rank(axis=1, ascending=False, method="first")
        mkt = Market(rank.index.to_numpy(), list(rank.columns))
        rank = rank.reindex(index=mkt.dates, columns=mkt.stocks)
        # 沒有價格的股票不可交易 → 排名設 NaN
        rank = rank.where(mkt.px.notna())

        bench = equal_weight_universe(mkt, rank)
        rows = []
        total = len(GRID_N) * len(GRID_K) * len(GRID_FREQ) * len(GRID_LIQ)
        done = 0
        for n in GRID_N:
            for k in GRID_K:
                for fq in GRID_FREQ:
                    for lq in GRID_LIQ:
                        r = run_config(mkt, rank, n, k, fq, lq)
                        ex, te_ = tracking_error(r.pop("_daily"), bench)
                        r.update({"n": n, "k": k, "freq": fq,
                                  "liq": lq, "excess_vs_ew": ex, "tracking_error": te_})
                        rows.append(r)
                        done += 1
                        if done % 40 == 0:
                            print(f"  ... {done}/{total}（{time.time()-t0:.0f}s）", flush=True)
        # ── v1.1 擴充網格：配權 × 平滑 × 交易限制 ──────────────────────
        ext_rows = []
        te0 = time.time()
        for sm in GRID_EXT["smooth"]:
            raw = sig.pivot(index="Date", columns="stock_id", values="score")
            if sm > 1:
                raw = raw.rolling(sm, min_periods=1).mean()
            rk_s = raw.rank(axis=1, ascending=False, method="first")                       .reindex(index=mkt.dates, columns=mkt.stocks).where(mkt.px.notna())
            for n in GRID_EXT["n"]:
                for k in GRID_EXT["k"]:
                    for fq in GRID_EXT["freq"]:
                        for wt in GRID_EXT["weight"]:
                            r = run_config(mkt, rk_s, n, k, fq, None, weight=wt)
                            ex, te_ = tracking_error(r.pop("_daily"), bench)
                            r.update({"n": n, "k": k, "freq": fq, "weight": wt,
                                      "smooth": sm, "excess_vs_ew": ex, "tracking_error": te_})
                            ext_rows.append(r)
        print(f"  ext {len(ext_rows)} 組（{time.time()-te0:.0f}s）", flush=True)

        # 交易限制（風控 C）：只對兩個 headline 做——它是「可執行性」不是「可調參數」
        constraints = {}
        for hl, cfg in HEADLINES.items():
            row = {}
            for lab, bl, bd in (("無限制", False, False), ("擋漲跌停", True, False),
                                ("擋處置", False, True), ("兩者都擋", True, True)):
                r = run_config(mkt, rank, cfg["n"], cfg["k"], cfg["freq"], cfg["liq"],
                               block_limit=bl, block_disposal=bd)
                r.pop("_daily")
                row[lab] = {kk: r[kk] for kk in ("ann_return", "ann_sharpe",
                                                 "max_drawdown", "avg_turnover")}
            constraints[hl] = row

        # 成本 ×2 敏感度：只對兩個 headline 做（240 組全做太慢且無必要）
        cost2 = {}
        for hl, cfg in HEADLINES.items():
            r2 = run_config(mkt, rank, cfg["n"], cfg["k"], cfg["freq"], cfg["liq"],
                            cost_mult=2.0)
            r2.pop("_daily")
            cost2[hl] = r2
        hb = len(bench) // 2
        all_res["models"][name] = {
            "n_days": len(mkt.dates), "n_stocks": len(mkt.stocks),
            "benchmark_ew_universe_ann": round(_ann(bench), 4),
            # 子期間的市場環境：判斷「20 日最好」是不是被行情帶起來的，先得知道行情長怎樣
            "benchmark_ann_1h": round(_ann(bench[:hb]), 4),
            "benchmark_ann_2h": round(_ann(bench[hb:]), 4),
            "subperiod_split_date": str(mkt.dates[hb].date()),
            "decile": decile_spread(mkt, rank),
            "signal_health": signal_health(mkt, rank),      # 風控 E：訊號失效監控
            "grid": rows,
            "grid_ext": ext_rows,                      # v1.1：配權 × 平滑
            "constraints_headline": constraints,       # 風控 C：漲跌停 / 處置
            "cost_x2_headline": cost2,
            "elapsed_min": round((time.time() - t0) / 60, 1),
        }
        print(f"[sweep] {name} 完成（{(time.time()-t0)/60:.1f} 分）"
              f"｜等權宇宙基準年化 {_ann(bench):+.1%}"
              f"（前半 {_ann(bench[:hb]):+.1%} / 後半 {_ann(bench[hb:]):+.1%}）", flush=True)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(all_res, indent=1, ensure_ascii=False), encoding="utf-8")
        del mkt, rank, sig
        gc.collect()
    return all_res


# ============================================================
# 6) 報表
# ============================================================
def report() -> None:
    if not OUT_PATH.exists():
        print(f"❌ 尚無結果：{OUT_PATH}", flush=True)
        return
    res = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    for name, m in res.get("models", {}).items():
        g = pd.DataFrame(m["grid"])
        print(f"\n{'='*100}\n{name}｜{m['n_days']} 天 × {m['n_stocks']} 支"
              f"｜等權宇宙基準年化 {m['benchmark_ew_universe_ann']:+.1%}\n{'='*100}")

        print("\n▸ Headline（跑之前定死的兩組，不是最好看的兩組）")
        for hl, cfg in HEADLINES.items():
            row = g[(g.n == cfg["n"]) & (g.k == cfg["k"]) & (g.freq == cfg["freq"])
                    & (g.liq.isna() if cfg["liq"] is None else (g.liq == cfg["liq"]))]
            if row.empty:
                continue
            r = row.iloc[0]
            c2 = m["cost_x2_headline"].get(hl, {})
            print(f"  {hl:16s} N={cfg['n']:<4} k={cfg['k']} 頻率={cfg['freq']}日 "
                  f"流動性={cfg['liq'] or '無'}")
            print(f"    年化 {r.ann_return:+7.1%} ｜ Sharpe {r.ann_sharpe:>6} ｜ "
                  f"MDD {r.max_drawdown:+7.1%} ｜ 換手 {r.avg_turnover:>5.0%} ｜ "
                  f"**年化成本 {r.ann_cost_drag:>6.1%}**")
            print(f"    超額(對等權宇宙) {r.excess_vs_ew:+6.1%} ｜ TE {r.tracking_error:>5.1%} ｜ "
                  f"前三大產業 {r.sector_top3_share} ｜ 最大單股權重 {r.max_weight} ｜ "
                  f"成本×2 年化 {c2.get('ann_return', float('nan')):+.1%}")

        print(f"\n▸ decile 診斷：前10% 年化 {m['decile']['decile_long_ann']:+.1%}"
              f" ｜ 前10%−後10% 年化(簡單加總) {m['decile']['decile_spread_ann_simple']:+.1%}"
              f" ｜ spread Sharpe {m['decile']['decile_spread_sharpe']}")

        print("\n▸ 緩衝 k 的效果（固定 N=50、5 日、無流動性過濾）")
        sub = g[(g.n == 50) & (g.freq == 5) & (g.liq.isna())].sort_values("k")
        print(f"    {'k':>5} {'換手':>7} {'年化成本':>9} {'年化':>8} {'Sharpe':>7} {'MDD':>8}")
        for _, r in sub.iterrows():
            print(f"    {r.k:>5} {r.avg_turnover:>7.0%} {r.ann_cost_drag:>9.1%} "
                  f"{r.ann_return:>+8.1%} {str(r.ann_sharpe):>7} {r.max_drawdown:>+8.1%}")

        print("\n▸ 再平衡頻率 × 緩衝（N=50、無流動性過濾）：年化 / 換手")
        piv_a = g[(g.n == 50) & (g.liq.isna())].pivot(index="freq", columns="k",
                                                      values="ann_return")
        piv_t = g[(g.n == 50) & (g.liq.isna())].pivot(index="freq", columns="k",
                                                      values="avg_turnover")
        print("    " + "".join(f"{'k='+str(c):>16}" for c in piv_a.columns))
        for f_ in piv_a.index:
            print(f"    {f_:>2}日" + "".join(
                f"{piv_a.loc[f_, c]:>+9.1%}/{piv_t.loc[f_, c]:>5.0%}" for c in piv_a.columns))

        print("\n▸ 持股數 N 的效果（k=1.5、5 日、無流動性過濾）— 跨 N 比較看 Sharpe 不看年化")
        sub = g[(g.k == 1.5) & (g.freq == 5) & (g.liq.isna())].sort_values("n")
        print(f"    {'N':>5} {'年化':>8} {'Sharpe':>7} {'MDD':>8} {'換手':>7} "
              f"{'年化成本':>9} {'超額':>8} {'TE':>7}")
        for _, r in sub.iterrows():
            print(f"    {r.n:>5} {r.ann_return:>+8.1%} {str(r.ann_sharpe):>7} "
                  f"{r.max_drawdown:>+8.1%} {r.avg_turnover:>7.0%} {r.ann_cost_drag:>9.1%} "
                  f"{r.excess_vs_ew:>+8.1%} {r.tracking_error:>7.1%}")

        print("\n▸ 流動性門檻（N=50、k=1.5、5 日）")
        sub = g[(g.n == 50) & (g.k == 1.5) & (g.freq == 5)]
        for _, r in sub.iterrows():
            lab = "無" if pd.isna(r.liq) else f"前 {1-r.liq:.0%}"
            print(f"    {lab:>8} 年化 {r.ann_return:>+7.1%} ｜ Sharpe {str(r.ann_sharpe):>6} ｜ "
                  f"換手 {r.avg_turnover:>5.0%}")

        best = g.loc[g.ann_sharpe.astype(float).idxmax()]
        print(f"\n▸ 全網格 Sharpe 最高（**僅供參考，不是 headline**）："
              f"N={best.n} k={best.k} 頻率={best.freq}日 "
              f"流動性={'無' if pd.isna(best.liq) else best.liq} → "
              f"年化 {best.ann_return:+.1%} Sharpe {best.ann_sharpe} 換手 {best.avg_turnover:.0%}")


COLAB_SNIPPET = r'''
# ── 在 Colab 產生 Mamba 三組的分數（不用重訓，只是前向一次）──────────
import numpy as np, pandas as pd, torch
from pathlib import Path
import experimental.short_model as sm
from experimental.kg_ablation import ARMS, DROPOUT, build_dates
from marketmamba.config import AMP_ENABLED, MODELS_DIR, PROCESSED_DIR
from marketmamba.models.trainer import (TemporalCrossSectionDataset, build_kg_csr,
                                        get_batch_edges_csr, make_dataloader)

_, val_dates = build_dates(df)          # 與消融同一套切分（582 天）
out_dir = Path("/content/drive/MyDrive/MarketMamba_V6/scores"); out_dir.mkdir(exist_ok=True)
dev = torch.device("cuda")

for arm, (use_gat, kg_file) in ARMS.items():
    ck = MODELS_DIR / f"v6_short_KG_{arm}.pt"
    if not ck.exists():
        print(f"跳過 {arm}（找不到 {ck}）"); continue
    model = sm.ShortModelV6(use_gat=use_gat, dropout=DROPOUT).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev)["state_dict"]); model.eval()
    if kg_file:
        import marketmamba.models.trainer as T
        _o = T.KG_CACHE_PATH; T.KG_CACHE_PATH = Path(PROCESSED_DIR) / kg_file
        try: kg, s2i = build_kg_csr()
        finally: T.KG_CACHE_PATH = _o
    else:
        kg, s2i = build_kg_csr()
    ds = TemporalCrossSectionDataset(df, val_dates, mode="val", n_sample=None)
    rows = []
    with torch.no_grad():
        for i, (X, Y, stks, _m) in enumerate(make_dataloader(ds, shuffle=False)):
            if X.shape[0] <= 1: continue
            ei, ea = get_batch_edges_csr(stks, kg, s2i, dev)
            with torch.amp.autocast('cuda', enabled=AMP_ENABLED):
                p = model(X.to(dev), ei, ea)
            d = str(ds.valid_dates[i])[:10]
            rows.append(pd.DataFrame({"Date": d, "stock_id": [str(x) for x in stks],
                                      "score": p[:, 0].float().cpu().numpy()}))
    pd.concat(rows).to_parquet(out_dir / f"{arm}.parquet", index=False)
    print(f"✅ {arm} → {out_dir / (arm + '.parquet')}")
# 下載這幾個 .parquet 放到 V6/experimental/result/scores/ 再跑 --sweep
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-scores", choices=("ridge", "gbdt"),
                    help="產生該模型的 test 窗分數（需 MM_PROTOCOL=v2）")
    ap.add_argument("--label", choices=("5d", "10d", "20d"), default="5d",
                    help="訓練用哪個 horizon 的 rank 標籤（標籤 horizon 實驗）")
    ap.add_argument("--purge", type=int, default=0,
                    help="訓練集尾端剔除幾個交易日（標籤 horizon 實驗一律 20）")
    ap.add_argument("--sweep", action="store_true", help="掃 240 組網格")
    ap.add_argument("--models", nargs="*", help="只掃這幾個模型（預設全部）")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--colab", action="store_true", help="印出 Colab 產生 Mamba 分數的程式碼")
    a = ap.parse_args()

    if a.colab:
        print(COLAB_SNIPPET)
    if a.export_scores:
        export_scores(a.export_scores, label=a.label, purge_days=a.purge)
    if a.sweep:
        sweep(a.models)
    if a.report or not (a.export_scores or a.sweep or a.colab):
        report()
