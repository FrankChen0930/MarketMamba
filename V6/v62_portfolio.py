"""
v62_portfolio.py — V6.2 組合層狀態機（主線規格 `5d/20`）
=========================================================
把每日分數變成**實際持股名單**：N=50 / k=1.5 緩衝 / 每 freq 個交易日再平衡。

分數本身不是持股名單——中間 19 天不換股，所以必須有狀態。

規格來源：`docs/portfolio-construction-baseline-v1.md` + 2026-08-05 定案
（CLAUDE.md「★ V6.2 上線規格」）。回測對照：N=50/k=1.5/20日 年化 **+37.3%**
（2026-08-09 在回補後的新面板上重評；舊面板是 +38.0%，差 0.7pp、在雜訊內）。

多頻率並行（2026-08-08 新增）
-----------------------------
主線 20 日中間有 19 天不換股，dashboard 沒東西可看 → 同一份分數同時餵給
多個頻率的狀態機，中間天數可以看較高頻的**參考組合**。

**再平衡率是組合層參數、不是模型參數**——一份分數可同時跑 1/3/5/10/20 日，
GPU 前向零額外成本。所以「推論 arm」（模型 × 預測頭 → 分數檔）與
「組合 arm」（分數檔 × n/k/freq → 持股）是分開的兩張表：
前者在 `run_v62_inference.ARMS`，後者是本檔的 `PORTFOLIOS`。

⚠️ 高頻組合的分級依據見 `PORTFOLIOS` 的 `tier`。
**`bt_ann` 全部是 2026-08-09 在回補後的新面板上重跑的**（11 個 arm 一致口徑），
不是沿用 `docs/label-horizon-vs-holding-period-2026-08-03.md` §2 的舊值——
舊值算在缺 15% 列的面板上，與現在的線上資料不同義。

八個設計決定（2026-08-05；⑧ 為 08-08 新增）
--------------------------
 ① **一個 arm 一份 state**（`v62_state_{arm}.json`）→ 多模型並行時互不干擾。
    多頻率並行時 arm 名稱帶頻率後綴（`..._f05`），所以彼此也不會撞。
 ⑧ **state 的 spec 一旦寫下就不許改**：讀到既有 state 但 n/k/freq 與傳入的
    不同 → 直接 raise。中途改規格會讓那份「不可竄改的前瞻紀錄」變成兩段
    不同東西接在一起，而且**完全不會報錯**——正是本專案反覆踩到的那類坑。
 ② **再平衡觸發用交易日曆算，不用「跑了幾次」**：距上次再平衡 ≥20 個交易日就換。
    推論失敗一天不會讓排程漂掉——用「跑了幾次」的話，漏跑一天就永遠差一天。
 ③ **緩衝逐行照抄 `portfolio_lab.run_config`**：保留 rank ≤ k×N(=75) 的持股，
    不足 N 的部分從 Top-N 依序補。
 ④ **權重：再平衡日等權，期間自然漂移**（不做期中再平衡）——與回測一致。
 ⑤ **可交易性：照規格格產生名單（不擋漲跌停／處置），但逐檔標註旗標。**
    擋掉會讓實戰與回測不同義（實測代價 −5.5pp ~ −7.3pp，是既有記載的例外）。
    **不偷改規則**，讓使用者看得到、自己決定要不要手動略過。
 ⑥ **每日落檔含資料完整性**：幾個月後看到某段表現差，要能區分「模型不好」
    與「那幾天 margin 缺了」。事後補不回來，所以當天就要記。
 ⑦ **驗證：把狀態機 replay 582 天，必須重現 portfolio_lab 的那一格。**
    這是獨立實作 → 兩邊算出同一個年化，才代表兩邊都對。

用法
----
    # 驗證（先跑這個）
    MM_PROTOCOL=v2 python V6/v62_portfolio.py --replay v2_kg_nomacro__live

    # 每日（在 run_v62_inference 之後）
    python V6/v62_portfolio.py --step --scores V6/results/df_v62.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

# ── 規格常數（改這裡等於改上線規格，不要散落在別處）──────────────────
N_HOLD      = 50      # 持股檔數
BUFFER_K    = 1.5     # 緩衝：rank ≤ k×N 就續抱
REBAL_DAYS  = 20      # 主線：每 20 個交易日再平衡
WEIGHT_MODE = "equal"

RESULTS_DIR = _V6 / "results"
STATE_FMT   = "v62_state_{arm}.json"
LOG_FMT     = "v62_portfolio_{arm}.jsonl"   # 逐日 append，事後不得改


# ============================================================
# 0. 組合 arm 表（分數檔 × 再平衡率）
# ============================================================
@dataclass(frozen=True)
class Portfolio:
    """一個組合層狀態機。`score_arm` 指向 `run_v62_inference.ARMS` 的某個 key。"""
    score_arm: str            # 分數來源（推論 arm）
    freq:      int            # 再平衡間隔（交易日）
    tier:      str            # primary / equivalent / inferior — 前端警告分級用
    bt_ann:    float | None   # 回測淨年化（582 天窗），前端顯示用；None = 沒跑過
    head:      str            # 顯示用的頭名稱。**明確寫死，不從 score_arm 字串推導**
                              # （`head10d` 不含子字串 `h10` → 推導會靜默標成 5d 頭）
    family:    str            # 分組（前端用；見 `_FAMILY_DESC`）。**與 tier 正交**：
                              # tier 說「能不能照做」，family 說「這是什麼東西」。
                              # 19 個 arm 平鋪在同一排按鈕裡看不出結構，
                              # 而「主線的另一個頻率」與「壞掉的 KG 對照組」
                              # 是完全不同性質的東西，混在一起會誤導。
    n:         int = N_HOLD
    k:         float = BUFFER_K
    note:      str = ""

    @property
    def label(self) -> str:
        return f"{self.head} / {self.freq} 日"


# 雜訊底線：組合層 N=50 約 **±6pp**（CLAUDE.md「判讀數字之前」）。
# 分級是拿與主線的差距對這條線量出來的，不是憑感覺分的。詳細規則見下方。
_TIER_DESC = {
    "primary":      "主線規格",
    "equivalent":   "研究用；與主線差距在雜訊底線（±6pp）內，分不出優劣",
    "inferior":     "研究用；已知明確劣於主線，請勿照做",
    "incomparable": "研究用；**與主線不可並列**——出自不同訓練輪（隔離天數不同），"
                    "回測數字只在該輪內部有意義",
}

# ⚠️ 分級規則（2026-08-09 修過一次，原本是錯的）
#    第一版只看 `|Δ| > 6pp` 就標 inferior → `head10d_f20` 比主線**好 8.9pp**
#    卻被標成「已知明確劣於主線，請勿照做」。**方向搞反了。**
#    而更根本的問題是：`head10d`/`head20d` 出自**隔離 40 天**那一輪、
#    主線是隔離 30 天，CLAUDE.md 明寫「兩輪之間隔離天數不同，不可跨輪並列」
#    → 那個 +8.9pp 的比較本身就不成立，不該拿來分級。故新增 `incomparable`。
#
#    正確規則：
#      primary       = 上線規格本身
#      incomparable  = 不同訓練輪，不做比較
#      equivalent    = 同輪且 |Δ| < 6pp
#      inferior      = 同輪且 Δ ≤ −6pp（**只有更差才叫 inferior**）

# ── 分組（family）：**與 tier 正交，兩者都要顯示** ───────────────────────
#    tier  = 「能不能照這個做」（拿回測差距對雜訊底線量出來的）
#    family= 「這是什麼東西」（同一顆模型換頻率 vs 完全不同的模型）
#
#    為什麼需要 family：`v2_kg_nomacro_f03`（主線換成 3 日再平衡）與
#    `old_kg_f20`（KG 壞掉的對照組）**tier 都是 inferior**，但前者是
#    「同一個訊號、換個用法」、後者是「一個已知有缺陷的模型」。
#    只給 tier 的話，這兩個在前端長得一模一樣——那是誤導。
_FAMILY_DESC = {
    "main_5d":   "上線模型 · 5d 頭（不同再平衡率）",
    "main_10d":  "上線模型 · 10d 頭（同一顆 checkpoint 的第二欄）",
    "ckpt":      "獨立訓練的研究 checkpoint",
    "ablation":  "F6 消融對照組",
    "baseline":  "B 類經典模型對照組",
}
# 顯示順序（前端照這個排；dict 的宣告順序不保證前端會照用，所以明確給一份）
FAMILY_ORDER = ["main_5d", "main_10d", "ckpt", "ablation", "baseline"]

PORTFOLIOS: dict[str, Portfolio] = {
    # ── 主線：5d 頭 / 20 日（已定案的上線規格）────────────────────
    "v2_kg_nomacro_f20": Portfolio("v2_kg_nomacro",     20, "primary",    0.373, "5d 頭",
                                   "main_5d", note="★ 上線規格 5d/20"),
    # ── 5d 頭的其他頻率（中間天數的參考組合）──────────────────────
    "v2_kg_nomacro_f10": Portfolio("v2_kg_nomacro",     10, "equivalent", 0.374, "5d 頭", "main_5d"),
    "v2_kg_nomacro_f05": Portfolio("v2_kg_nomacro",      5, "equivalent", 0.324, "5d 頭", "main_5d"),
    "v2_kg_nomacro_f03": Portfolio("v2_kg_nomacro",      3, "inferior",   0.259, "5d 頭", "main_5d"),
    "v2_kg_nomacro_f01": Portfolio("v2_kg_nomacro",      1, "inferior",   0.250, "5d 頭", "main_5d"),
    # ── 10d 頭（同一顆 checkpoint 的第二欄）────────────────────────
    #    ⚠️ 高頻端**要用這顆頭**：與 5d 頭的差距隨頻率變高而擴大
    #    （20 日 +1.2pp 在雜訊內 → 1 日 +9.6pp 超出雜訊底線）。
    #    機制在成本欄：10d 頭分數變動慢 → 換手低 → 1 日那格成本 31.1% vs 42.6%。
    "v2_kg_nomacro_h10_f20": Portfolio("v2_kg_nomacro_h10", 20, "equivalent", 0.421,
                                       "10d 頭", "main_10d",
                                       note="回測比主線高 1.2pp，但在雜訊內"),
    "v2_kg_nomacro_h10_f10": Portfolio("v2_kg_nomacro_h10", 10, "equivalent", 0.427,
                                       "10d 頭", "main_10d"),
    "v2_kg_nomacro_h10_f05": Portfolio("v2_kg_nomacro_h10",  5, "equivalent", 0.390,
                                       "10d 頭", "main_10d"),
    "v2_kg_nomacro_h10_f03": Portfolio("v2_kg_nomacro_h10",  3, "equivalent", 0.329,
                                       "10d 頭", "main_10d"),
    "v2_kg_nomacro_h10_f01": Portfolio("v2_kg_nomacro_h10",  1, "equivalent", 0.326,
                                       "10d 頭", "main_10d",
                                       note="新面板下 −4.7pp，落回雜訊內（舊面板是 −9.7pp）"),
    # ── 獨立訓練的研究 checkpoint（各只跑主線頻率）─────────────────
    "head10d_f20": Portfolio("head10d", 20, "incomparable", 0.462, "h10 ckpt", "ckpt",
                             note="不同 checkpoint；隔離 40 天那一輪，不可與上面並列"),
    "head20d_f20": Portfolio("head20d", 20, "incomparable", 0.385, "h20 ckpt", "ckpt",
                             note="不同 checkpoint；隔離 40 天那一輪，不可與上面並列"),
    # ── F6 消融的四個 Mamba arm（2026-08-08 加入）──────────────────
    #    ⚠️ 這四個**吃 Group D**（zero_macro=False）→ 依賴 `build_feature_df()`
    #    的 macro 全歷史貼回。實測窗內自算 vs 全歷史：TWII_Return −0.1264 → −0.8985。
    #    定位是對照組，不是候選上線規格。
    "v3_kg_f20":  Portfolio("v3_kg", 20, "inferior", 0.275, "v3 圖", "ablation",
                            note="加 4,504 條相關性邊，對 v2_kg 無效應（decile 1.928）"),
    "v2_kg_f20":  Portfolio("v2_kg", 20, "inferior", 0.270, "v2 圖", "ablation",
                            note="Group D 照常；與主線只差 Group D（decile 1.905）"),
    "old_kg_f20": Portfolio("old_kg", 20, "inferior", 0.146, "舊圖", "ablation",
                            note="壞掉的 KG——2330 的鄰居是電器電纜（decile 1.231）"),
    "no_gat_f20": Portfolio("no_gat", 20, "inferior", 0.162, "無 GAT", "ablation",
                            note="架構少 graph_layer/gate/norm_fuse（decile 1.664）"),
    # ── B 類經典模型（`run_v62_baselines.py` 產分數，另一個 process）──
    #    定位是**對照組**，不是候選上線規格：八模型表裡它們都輸給 Mamba
    #    （ridge decile 1.088 / gbdt 1.735 / gru 2.388 vs v2_kg_nomacro 5.005）。
    #    留著跑是為了「同一段真實 OOS 期間、同一把尺」的並列紀錄。
    "ridge_f20": Portfolio("ridge", 20, "inferior", 0.216, "Ridge 307維", "baseline",
                           note="線性 baseline；重建 vs 參考 ρ=0.9970"),
    "gru_f20":   Portfolio("gru", 20, "inferior", 0.206, "GRU 60×59", "baseline",
                           note="checkpoint 即原始那顆，未重訓"),
    # ⚠️ GBDT 的可重現性要**分兩層講**（2026-08-08 實測）：
    #    訊號層**不可重現**（重建 vs 參考 ρ=0.9203、Top50 重疊只有 25/50；
    #    我自己兩次跑彼此也才 0.9434）——樹的切點是離散決策，訓練窗動 30 天就翻。
    #    但組合層**幾乎一樣**：11.0% vs 11.2%、Sharpe 0.653 vs 0.639、換手 79% vs 81%。
    #    → 換掉的那半個 Top50 與被換掉的一樣好。所以「持股名單對不上」≠「策略不同」。
    #    bt_ann 用重建模型自己跑出來的 11.0%（`gbdt__p30fix_20260808`），不借用參考值。
    "gbdt_f20":  Portfolio("gbdt", 20, "inferior", 0.141, "GBDT 307維", "baseline",
                           note="訊號層不可重現（ρ=0.9203）但組合層一致"
                                "（11.0% vs 參考 11.2%）；decile Sh 1.714 vs 1.806"),
}
DEFAULT_PORTFOLIO = "v2_kg_nomacro_f20"

# 守門：family 打錯字會讓前端整組消失（分組時對不上就不會被 render），
# 而且**完全不會報錯**——正是本專案反覆踩到的靜默失敗型態。
_bad_fam = {n: p.family for n, p in PORTFOLIOS.items() if p.family not in _FAMILY_DESC}
if _bad_fam:
    raise SystemExit(f"❌ 未知的 family：{_bad_fam}（可用：{sorted(_FAMILY_DESC)}）")
if set(FAMILY_ORDER) != set(_FAMILY_DESC):
    raise SystemExit(f"❌ FAMILY_ORDER 與 _FAMILY_DESC 不一致："
                     f"{set(FAMILY_ORDER) ^ set(_FAMILY_DESC)}")


# ============================================================
# 1. 核心：緩衝再平衡（純函式，可單獨驗證）
# ============================================================
def rebalance(holdings: list[str], rank: dict[str, float],
              n: int = N_HOLD, k: float = BUFFER_K) -> dict:
    """
    `rank`: {stock_id: 當日排名}（1 = 分數最高；不可交易者不要放進來）。

    邏輯與 `portfolio_lab.run_config` 逐行相同：
      keep = 現有持股中 rank ≤ k*n 者（緩衝：還在前 75 名就不動）
      adds = Top-n 裡不在 keep 的，依序補到 n 檔
    """
    kn = k * n
    keep = [s for s in holdings if s in rank and rank[s] <= kn]
    keep_set = set(keep)
    top_n = [s for s, _ in sorted(rank.items(), key=lambda kv: kv[1])[:n]]
    adds = [s for s in top_n if s not in keep_set][:max(n - len(keep), 0)]
    new = keep + adds
    return {"holdings": new, "kept": keep, "added": adds,
            "dropped": [s for s in holdings if s not in set(new)]}


def equal_weights(holdings: list[str]) -> dict[str, float]:
    if not holdings:
        return {}
    w = 1.0 / len(holdings)
    return {s: w for s in holdings}


# ============================================================
# 2. 驗證：replay 582 天，必須重現 portfolio_lab
# ============================================================
def replay(model: str, n: int = N_HOLD, k: float = BUFFER_K,
           freq: int = REBAL_DAYS) -> bool:
    """
    用同一份分數檔，讓**本檔的狀態機**獨立算出年化/Sharpe/換手，
    再對 `portfolio_lab_result.json` 同一格。兩個獨立實作算出同一個數字，
    才代表狀態機的緩衝與再平衡邏輯真的與回測一致。

    判準（跑之前定死）：年化差 < 0.001（0.1pp）、換手差 < 0.01。
    """
    from experimental.portfolio_lab import COST_BUY, COST_SELL, Market, RESULT_DIR

    sc = pd.read_parquet(_V6 / "experimental" / "result" / "scores" / f"{model}.parquet")
    sc["Date"] = pd.to_datetime(sc["Date"])
    rank_df = sc.pivot(index="Date", columns="stock_id", values="score") \
                .rank(axis=1, ascending=False, method="first")
    mkt = Market(rank_df.index.to_numpy(), list(rank_df.columns))
    rank_df = rank_df.reindex(index=mkt.dates, columns=mkt.stocks)
    rank_df = rank_df.where(mkt.px.notna())        # 沒有價格 → 不可交易

    ret = mkt.ret.copy()
    ret_np = np.nan_to_num(ret.to_numpy(np.float64))
    dates = mkt.dates
    reb_idx = set(range(0, len(dates), freq))

    holdings: list[str] = []
    w: dict[str, float] = {}
    port_ret = np.zeros(len(dates))
    turnovers: list[float] = []
    col = {s: i for i, s in enumerate(mkt.stocks)}

    for t in range(len(dates)):
        # ① 昨日權重吃今天的報酬（與 run_config 相同順序）
        if holdings:
            g = {s: w[s] * (1.0 + ret_np[t, col[s]]) for s in holdings}
            tot = sum(g.values())
            port_ret[t] = tot - sum(w[s] for s in holdings)
            if tot > 0:
                w = {s: v / tot for s, v in g.items()}
        # ② 收盤後再平衡（下一日才吃到新組合的報酬）
        if t in reb_idx:
            row = rank_df.iloc[t]
            rk = {s: float(v) for s, v in row.items() if np.isfinite(v)}
            if len(rk) < n:
                continue
            r = rebalance(holdings, rk, n, k)
            new_w = equal_weights(r["holdings"])
            delta = {s: new_w.get(s, 0.0) - w.get(s, 0.0)
                     for s in set(new_w) | set(w)}
            buy = sum(v for v in delta.values() if v > 0)
            sell = -sum(v for v in delta.values() if v < 0)
            port_ret[t] -= buy * COST_BUY + sell * COST_SELL
            # 換手定義以 portfolio_lab 為準：只記**買進側**（`turnovers.append(buy_frac)`，
            # portfolio_lab.py:378）。用 (buy+sell)/2 會在首次建倉那次差 0.5，
            # 分攤到 30 次再平衡剛好是 1.67pp——第一版就是這樣對不上的。
            turnovers.append(buy)
            holdings, w = r["holdings"], new_w

    ann = float((1 + port_ret).prod() ** (252 / len(dates)) - 1)
    sharpe = float(port_ret.mean() / port_ret.std() * np.sqrt(252))
    turn = float(np.mean(turnovers))

    ref_path = RESULT_DIR / "portfolio_lab_result.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))["models"][model]["grid"]
    cell = next(g for g in ref if g["n"] == n and abs(g["k"] - k) < 1e-9
                and g["freq"] == freq and g["liq"] is None)

    d_ann = abs(ann - cell["ann_return"])
    d_turn = abs(turn - cell["avg_turnover"])
    ok = d_ann < 0.001 and d_turn < 0.01
    print(f"\n{'='*70}\n[replay] {model}｜N={n} k={k} freq={freq}｜{len(dates)} 天\n{'='*70}")
    print(f"{'指標':16s}{'狀態機':>12s}{'portfolio_lab':>16s}{'差':>10s}")
    print(f"{'年化':16s}{ann*100:11.2f}%{cell['ann_return']*100:15.2f}%{d_ann*100:9.3f}pp")
    print(f"{'Sharpe':16s}{sharpe:12.3f}{cell['ann_sharpe']:16.3f}"
          f"{abs(sharpe-cell['ann_sharpe']):10.3f}")
    print(f"{'換手':16s}{turn*100:11.1f}%{cell['avg_turnover']*100:15.1f}%{d_turn*100:9.2f}pp")
    print(f"[replay] 再平衡 {len(turnovers)} 次（回測 {cell['n_rebalances']} 次）")
    print(f"[replay] {'✅ 兩個獨立實作一致 → 狀態機邏輯正確' if ok else '❌ 不一致，狀態機有問題'}")
    print(f"{'='*70}")
    return ok


# ============================================================
# 3. 每日推進
# ============================================================
@lru_cache(maxsize=1)
def _trading_calendar() -> pd.DatetimeIndex:
    """prices_raw 的交易日曆，**一個 process 只讀一次**。

    ⚠️ 這裡的快取不是可有可無的優化：`step()` 每呼叫一次就要問一次日曆，
    而多頻率並行後一天會呼叫 12 次（12 個組合 arm）。沒有快取＝每天重讀
    12 次 127 MB 的 parquet。多加一個組合 arm 的成本本該是零，
    沒快取的話它會變成「零 GPU 成本、但每次多一次大檔 IO」。
    """
    from marketmamba.config import PROCESSED_DIR
    pr = pd.read_parquet(Path(PROCESSED_DIR) / "prices_raw.parquet", columns=["Date"])
    d = pd.to_datetime(pr["Date"].astype(str)).drop_duplicates().sort_values()
    return pd.DatetimeIndex(d)


def _trading_days_between(a: str, b: str) -> int:
    """用 prices_raw 的實際交易日曆算（不是曆日、也不是「跑了幾次」）。"""
    d = _trading_calendar()
    return int(((d > pd.Timestamp(a)) & (d <= pd.Timestamp(b))).sum())


@lru_cache(maxsize=8)
def _tradable_on(date: str) -> frozenset[str]:
    """當日有有效收盤價（Close > 0）的股票 —— **當天沒有價格就是買不到**。

    ⚠️ 這個過濾是 2026-08-08 補的，補的是一個**真實的線上 vs 回測口徑分歧**。
    `portfolio_lab` 與 `replay()` 都有 `rank_df.where(mkt.px.notna())`
    （`Market` 另有 `px.where(px > 0)`），但 `step()` 原本完全沒做
    → 線上可能選到當天停牌／無成交的股票，而回測從來不會。

    **怎麼發現的**：`v62_performance.py` 把 582 天的持股紀錄算成報酬，
    得到 38.20% vs 回測 38.02%（差 0.18pp）。逐日 diff 後定位到
    **2024-10-01 一檔股票**（線上選 6206、回測選 3035），持股往後帶
    就造成整段系統性差異。**0.18pp 很容易被當成捨入誤差放過** ——
    這正是「前瞻績效工具」除了記錄績效之外的第二個價值。
    """
    from marketmamba.config import PROCESSED_DIR

    p = Path(PROCESSED_DIR) / "prices_raw.parquet"
    # ⚠️ `Date` 在 production 的 prices_raw 是 **large_string**、不是 timestamp
    #    （CLAUDE.md「換 production 資料檔：只改值、不改型別」記過這件事）。
    #    用 `pd.Timestamp` 當 filter 值會 ArrowNotImplementedError。
    #    先試字串，型別哪天真的改了再退回 Timestamp——兩種都不成立才放棄。
    key = pd.Timestamp(date).strftime("%Y-%m-%d")
    cols = ["Date", "stock_id", "Close"]
    try:
        pr = pd.read_parquet(p, columns=cols, filters=[("Date", "==", key)])
    except Exception:                                           # noqa: BLE001
        pr = pd.read_parquet(p, columns=cols,
                             filters=[("Date", "==", pd.Timestamp(date))])
    if pr.empty:
        return frozenset()
    c = pd.to_numeric(pr["Close"], errors="coerce")
    return frozenset(pr.loc[c > 0, "stock_id"].astype(str))


def step(arm: str, scores_path: Path, data_complete: dict | None = None,
         force_rebalance: bool = False, n: int = N_HOLD, k: float = BUFFER_K,
         freq: int = REBAL_DAYS, tier: str = "primary") -> dict:
    """讀今日分數 → 決定要不要再平衡 → 更新 state → append 逐日紀錄。

    `arm` 同時是 state/log 的檔名 key，多頻率並行時必須各自唯一
    （`PORTFOLIOS` 的 key 已經帶 `_fNN` 後綴）。
    """
    sc = pd.read_csv(scores_path, dtype={"stock_id": str})
    date = str(pd.to_datetime(sc["Date"].iloc[0]).date())
    # ⚠️ **並列必須用 stock_id 打破，而且要穩定排序**（2026-08-08 修）。
    #
    #    分數是 float32，同一天實測有 **183 組完全相等的分數**（2024-10-01）。
    #    原本是 `sort_values("score", ascending=False)`：
    #      ① pandas 預設 quicksort 是**不穩定排序** → `step()` 本身不具決定性
    #      ② 回測那邊是 `rank(axis=1, method="first")`，並列依 **pivot 欄序**
    #         （= stock_id 字典序）決定 → 兩邊對同一組並列會給出不同名次
    #
    #    實測後果：2024-10-01 線上選了 `6206`、回測選了 `3035`（兩者 score 相同、
    #    rank 47/48），持股往後帶 → **整段 582 天年化差 0.18pp**（38.20 vs 38.02）。
    #    這是「不會報錯、看起來像捨入誤差」的那類分歧。
    #
    #    `kind="mergesort"` = 穩定排序；次鍵 stock_id 遞增 = 對齊 `method="first"`。
    sc["stock_id"] = sc["stock_id"].astype(str)
    sc = sc.sort_values(["score", "stock_id"], ascending=[False, True],
                        kind="mergesort").reset_index(drop=True)
    rk = {r.stock_id: float(i + 1) for i, r in enumerate(sc.itertuples())}
    n_tie = int(sc["score"].duplicated().sum())
    if n_tie:
        print(f"[state] 並列分數 {n_tie} 組 → 一律以 stock_id 遞增打破"
              f"（與回測 rank(method='first') 同口徑）", flush=True)

    # 當日無收盤價 → 買不到 → 逐出候選池（與 portfolio_lab / replay 同口徑）。
    # ⚠️ 排名數字**不重算**：`replay()` 是先 rank 全體、再 `.where(px.notna())`
    #    把不可交易者設成 NaN，名次不會往前遞補。這裡照抄那個語意——
    #    改成重算名次會讓緩衝門檻（rank ≤ k×N）的意義跟著變。
    tradable = _tradable_on(date)
    if tradable:
        n_before = len(rk)
        rk = {s: v for s, v in rk.items() if s in tradable}
        dropped = n_before - len(rk)
        print(f"[state] 可交易過濾：{n_before} → {len(rk)} 檔"
              f"（剔除 {dropped} 檔當日無收盤價）", flush=True)
    else:
        print(f"[state] ⚠️ {date} 讀不到任何收盤價 → **不做可交易過濾**"
              f"（回測會濾，這天的線上口徑與回測不同，判讀時要排除）", flush=True)

    spec = {"n": n, "k": k, "freq": freq, "weight": WEIGHT_MODE, "tier": tier}
    st_path = RESULTS_DIR / STATE_FMT.format(arm=arm)
    st = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {
        "arm": arm, "holdings": [], "weights": {}, "last_rebalance": None,
        "n_rebalances": 0, "spec": spec,
    }
    # 設計 ⑧：規格一旦寫下就不許改。中途改了會讓 jsonl 變成兩段不同的東西
    # 接在一起，而且完全不會報錯——那份紀錄的價值就是「不可竄改」。
    old = {kk: st.get("spec", {}).get(kk) for kk in ("n", "k", "freq")}
    if st.get("spec") and old != {kk: spec[kk] for kk in ("n", "k", "freq")}:
        raise SystemExit(
            f"❌ arm={arm} 的既有 state 規格是 {old}，但這次傳入 "
            f"{{'n': {n}, 'k': {k}, 'freq': {freq}}}。\n"
            f"   規格不可中途更改（會污染前瞻紀錄）。要換規格請用新的 arm 名稱，\n"
            f"   或先刪掉 {st_path.name} 與對應的 jsonl 重新開始。")

    if st.get("last_date") == date:
        print(f"[state] {date} 已處理過，跳過（重跑不會重複計算）", flush=True)
        return st

    since = (_trading_days_between(st["last_rebalance"], date)
             if st["last_rebalance"] else freq)
    due = force_rebalance or st["last_rebalance"] is None or since >= freq

    if due:
        r = rebalance(st["holdings"], rk, n, k)
        st["holdings"] = r["holdings"]
        st["weights"] = equal_weights(r["holdings"])
        st["last_rebalance"] = date
        st["n_rebalances"] += 1
        print(f"[state] ★ {date} 再平衡 #{st['n_rebalances']}｜"
              f"續抱 {len(r['kept'])}｜買進 {len(r['added'])}｜賣出 {len(r['dropped'])}",
              flush=True)
        if r["added"]:
            print(f"  買進：{', '.join(r['added'][:15])}"
                  f"{' …' if len(r['added']) > 15 else ''}", flush=True)
        if r["dropped"]:
            print(f"  賣出：{', '.join(r['dropped'][:15])}"
                  f"{' …' if len(r['dropped']) > 15 else ''}", flush=True)
        since = 0
    else:
        print(f"[state] {date} 不換股（距上次再平衡 {since}/{freq} 個交易日）",
              flush=True)

    # 漂移度：目前持股還有幾檔在模型當前 Top-N（不換股那幾天唯一有參考價值的數字）
    drift = sum(1 for s in st["holdings"] if rk.get(s, 1e9) <= n)
    st.update({"last_date": date, "days_since_rebalance": since,
               "days_to_next": max(freq - since, 0), "spec": spec,
               "in_top_n": drift, "data_complete": data_complete or {}})
    print(f"[state] 持股 {len(st['holdings'])} 檔｜其中 {drift} 檔仍在模型 Top{n}"
          f"｜距下次再平衡 {st['days_to_next']} 個交易日", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    st_path.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    # 逐日 append，事後不得修改——這份才是「不可竄改的前瞻紀錄」
    #
    # `weights` = **上次再平衡當下的權重**（非再平衡日不更新，期間的漂移不寫回）。
    #
    # ⚠️ 誠實說明：這一欄**不是「補不回來」的資料**——現行規格是等權，
    #    再平衡日的權重必然是 1/N，漂移也能由價格推回。記它的理由是
    #    **讓紀錄自我描述**：哪天 `WEIGHT_MODE` 不再是 "equal"（規格已留這個參數），
    #    所有「假設等權」的重建就會靜默算錯，而舊紀錄裡沒有任何線索能發現。
    #    多一欄的成本近乎零，換掉一個未來才會爆的靜默假設。
    rec = {"date": date, "arm": arm, "rebalanced": bool(due),
           "holdings": st["holdings"], "weights": st["weights"],
           "in_top_n": drift, "spec": spec, "data_complete": data_complete or {},
           "written_at": datetime.now().isoformat(timespec="seconds")}
    with (RESULTS_DIR / LOG_FMT.format(arm=arm)).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return st


def write_manifest(path: Path | None = None) -> Path:
    """把 `PORTFOLIOS` 發布成 `v62_arms.json`，供後端 router 讀取。

    **為什麼要有這個檔**：後端跑在 Render（rootDir=`app/backend`），
    import 不到本檔。若在 router 裡再寫一份 arm 表，就是跨部署邊界的重複實作
    ——而且**連 assert 都擋不住**（兩份程式碼不在同一個 process 裡）。
    本專案已經因為「重複實作 + 各自維護」出過 bug（scanner vs sim 的進場標準）。

    → 唯一真相是本檔的 `PORTFOLIOS`，發布成資料讓後端讀。
      後端只需要知道「有哪些 arm、檔名是什麼」，其餘規格（n/k/freq/tier）
      每天都會隨 state 檔一起落檔，router 直接從 state 讀就好。
    """
    path = path or (RESULTS_DIR / "v62_arms.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 分數檔名從 `run_v62_inference.ARMS` 取，**不在這裡也不在 router 裡再寫一份**。
    # 順帶當守門：`PORTFOLIOS` 指到不存在的 score_arm 會在這裡當場炸，
    # 而不是等到後端抓不到檔案、前端顯示空白（那是靜默失敗）。
    # Mamba 線（59 維）與 B 類 baseline 線（66 維）是**兩個獨立 process**
    # ——config patch 是 module 級全域，混在同一個 process 會靜默算錯。
    # 但兩邊的分數檔名都要進 manifest，所以這裡只 import 表、不執行推論。
    import run_v62_inference as _R
    score_file = {k: f"{v.out_name}.csv" for k, v in _R.ARMS.items()}
    try:
        import run_v62_baselines as _BL       # 只讀 ARMS，不觸發 MM_PROTOCOL 檢查
        score_file.update({k: f"{v.out_name}.csv" for k, v in _BL.ARMS.items()})
    except SystemExit:
        # `run_v62_baselines` 在 import 期就要求 MM_PROTOCOL=v2；沒設時
        # 退回硬編（只有三個名字，且與該檔的 ARMS 對齊）。不 raise——
        # Mamba 線不該因為 baseline 線沒設環境變數就整條掛掉。
        score_file.update({"ridge": "df_v62_ridge.csv", "gbdt": "df_v62_gbdt.csv",
                           "gru": "df_v62_gru.csv"})
        print("[manifest] 未設 MM_PROTOCOL=v2 → baseline 分數檔名用硬編 fallback",
              flush=True)

    missing = sorted({p.score_arm for p in PORTFOLIOS.values()} - set(score_file))
    if missing:
        raise SystemExit(f"❌ PORTFOLIOS 指到不存在的分數 arm：{missing}"
                         f"（可用：{sorted(score_file)}）")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "default": DEFAULT_PORTFOLIO,
        "tier_desc": _TIER_DESC,
        "family_desc": _FAMILY_DESC,
        "family_order": FAMILY_ORDER,
        "arms": [
            {"arm": name, "score_arm": p.score_arm, "freq": p.freq, "n": p.n,
             "k": p.k, "tier": p.tier, "backtest_ann": p.bt_ann, "note": p.note,
             "head": p.head, "family": p.family,
             "state_file": STATE_FMT.format(arm=name),
             "score_file": score_file[p.score_arm], "label": p.label}
            for name, p in PORTFOLIOS.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] {len(PORTFOLIOS)} 個組合 arm → {path}", flush=True)
    return path


def print_portfolios() -> None:
    """列出組合 arm 表（規則 7：設定要看得見，不是只能讀程式碼推論）。"""
    print(f"\n{'='*88}\n組合 arm 表（{len(PORTFOLIOS)} 個）\n{'='*88}")
    for fam in FAMILY_ORDER:
        members = [(n, p) for n, p in PORTFOLIOS.items() if p.family == fam]
        if not members:
            continue
        print(f"\n【{fam}】{_FAMILY_DESC[fam]}（{len(members)} 個）")
        print(f"{'arm':26s}{'分數來源':22s}{'freq':>5s}{'N':>4s}{'k':>5s}"
              f"{'回測年化':>10s}  分級")
        print("-" * 88)
        for name, p in members:
            bt = f"{p.bt_ann*100:.1f}%" if p.bt_ann is not None else "—"
            print(f"{name:26s}{p.score_arm:22s}{p.freq:5d}{p.n:4d}{p.k:5.1f}{bt:>10s}"
                  f"  {p.tier}")
    print("-" * 88)
    for t, d in _TIER_DESC.items():
        print(f"  {t:12s} {d}")
    print("\n⚠️ family 與 tier 正交：family 說「這是什麼」，tier 說「能不能照做」。"
          "\n   例：`v2_kg_nomacro_f03`（主線換 3 日）與 `old_kg_f20`（壞掉的 KG）"
          "tier 都是 inferior，\n   但前者是同一個訊號換用法、後者是已知有缺陷的模型。")
    # ⚠️ 這行原本寫「回測年化來自 docs/label-horizon-vs-holding-period-2026-08-03.md §2」
    #    ——那已經過時：2026-08-09 回補資料後 11 個 arm 全部在新面板上重跑過。
    #    比較基準寫死在說明文字裡也會過時，這是同一類坑的第四次。
    print(f"{'='*88}\n⚠️ 回測年化 = 2026-08-09 在**回補後的新面板**上重跑的 582 天單一窗"
          f"（單一 seed）。**不是前瞻紀錄** —— 前瞻看 `v62_performance.py`。\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", metavar="MODEL",
                    help="用該分數檔 replay，對 portfolio_lab 驗證狀態機")
    ap.add_argument("--replay-freq", type=int, default=REBAL_DAYS,
                    help="replay 用的再平衡間隔（預設 20）")
    ap.add_argument("--step", action="store_true", help="每日推進一步")
    ap.add_argument("--arm", default=DEFAULT_PORTFOLIO,
                    help="組合 arm（見 --list）")
    ap.add_argument("--scores", default=str(RESULTS_DIR / "df_v62.csv"))
    ap.add_argument("--force-rebalance", action="store_true",
                    help="強制再平衡（第一天上線用）")
    ap.add_argument("--list", action="store_true", help="列出所有組合 arm")
    a = ap.parse_args()

    if a.list:
        print_portfolios()
        sys.exit(0)
    if a.replay:
        sys.exit(0 if replay(a.replay, freq=a.replay_freq) else 1)
    if a.step:
        p = PORTFOLIOS.get(a.arm)
        if p is None:
            sys.exit(f"❌ 未知的組合 arm：{a.arm}（用 --list 看可用的）")
        step(a.arm, Path(a.scores), force_rebalance=a.force_rebalance,
             n=p.n, k=p.k, freq=p.freq, tier=p.tier)
    else:
        ap.print_help()
