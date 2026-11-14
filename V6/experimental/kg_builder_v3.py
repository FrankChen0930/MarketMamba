"""
kg_builder_v3.py — 在 v2 圖上加「相關性邊」（PIT 安全的靜態版）
================================================================
為什麼做這個
------------
F6 + 組合層實測給出一個**機制上的線索**：

    20 日再平衡、N=50   上升段      下跌段
      no_gat           +57.9%      −40.4%
      v2_kg            +58.3%      **−17.3%**

**上升段幾乎相同，下跌段差 23 個百分點。**
→ GAT 帶進來的關係資訊，作用是**在下跌段避開一起崩的股票**，
不是在多頭挑到更會漲的（與 MDD −26.9% vs −33.4% 一致）。

而「誰跟誰一起動」正是**相關性邊**直接編碼的東西——比產業鏈邊（Phase 4-A）
少隔一層，工程量也小得多（不用爬蟲、不用公司↔節點對應表）。

舊圖本來就設計了這一層，但 `build_rolling_correlation_edges()` **從未成功產出**
（642,451 條邊裡相關性邊 **0 條**，例外被 try/except 吞掉）。所以「相關性邊有沒有用」
這個問題其實從來沒被回答過。

PIT 安全性（本檔的核心設計）
----------------------------
`kg_builder_v2.py` 檔頭已註明「現行寫法有 look-ahead」——那是因為相關性用了全歷史。
本檔的作法是：**只用截至 `CORR_END` 的報酬計算相關性**，預設 `2023-12-31`
＝ F6 的訓練切點。於是：

  - 對 2024-01 之後的 val/test 窗，圖完全不含未來資訊 ✅
  - 圖仍是**單一靜態快照**，不需要改動架構（`build_kg_csr` 只讀一個 npz）✅
  - 代價：圖的關係是「訓練期的關係」，套用到之後的期間。與 v2 圖用「最新產業別
    套全歷史」是同一類近似，但方向相反（v2 是未來資訊往回套，本檔是過去資訊往後套，
    **後者才是 PIT 正確的方向**）

⚠️ 真正的時變圖（每期一張）需要架構改動（dataset 要能每期換圖），不在本檔範圍。

設計
----
- 報酬：`prices_raw` 還原收盤價的日報酬，取 `CORR_START` ~ `CORR_END`
- 只保留該區間內有效報酬 ≥ `MIN_OBS` 天的股票
- **先去市場因子**：每日橫斷面減去等權平均報酬 → 相關性才是「殘差同動」而不是
  「大家都跟大盤走」。不做這步的話幾乎所有股票兩兩相關都是 0.3~0.6，選出來的
  鄰居只是高 beta 股，沒有資訊
- 每支股票取殘差相關性最高的 `TOP_K` 個鄰居，門檻 `MIN_CORR`
- 權重 `W_CORR`；與 v2 的既有邊合併後，每節點仍受 `MAX_NEIGHBORS_GAT` 上限

用法
----
    python V6/experimental/kg_builder_v3.py            # 建 knowledge_graph_v3.npz
    python V6/experimental/kg_builder_v3.py --compare  # 只印 v2 vs v3 對照

之後在 Colab/WSL 用與 F6 相同的乾淨配對量增量（同架構、同 seed、只換圖）：
    from experimental.kg_ablation import run_kg_ablation, ARMS
    ARMS["v3_kg"] = (True, "knowledge_graph_v3.npz")
    run_kg_ablation(df, arms=("v3_kg",), drive_dir=...)
→ 與已完成的 `v2_kg`（+0.0991）比。**兩者架構相同、seed 相同 → 是乾淨的配對比較**，
與 C−B 那組同級（那組配對 NW t = 5.17）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent.parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

from marketmamba.config import MAX_NEIGHBORS_GAT, PROCESSED_DIR       # noqa: E402

KG_V2_PATH = Path(PROCESSED_DIR) / "knowledge_graph_v2.npz"
KG_V3_PATH = Path(PROCESSED_DIR) / "knowledge_graph_v3.npz"

CORR_START = "2018-01-01"     # 6 年殘差相關，夠穩定又不會太陳舊
CORR_END = "2023-12-31"       # ★ F6 訓練切點 → 對 2024+ 的評估窗零 look-ahead
MIN_OBS = 750                 # 該區間內至少 750 個有效報酬日（約 3 年）
TOP_K = 5                     # 每股取相關性最高的 5 個鄰居
W_CORR = 0.6                  # 邊權重（v2：產業 0.5、集團 0.8）

# 殘差相關門檻。**依分布定，不依結果定**：實測非對角分位數
# p50 −0.003 / p90 0.091 / p99 0.259 → 0.20 約在 p98，
# 代表「所有配對中前 2% 的同動程度」才連邊。
# （原本的 0.35 在 p99 以上，只覆蓋 33.5% 的股票，劑量太小到量不出東西。）
MIN_CORR = 0.20

# ── 合併模式（2026-08-01 新增）────────────────────────────────────────
# `replace`  ：與 v2 邊一起排序、每節點取前 MAX_NEIGHBORS_GAT 條
#              → 相關性邊（權重 0.6）會把產業邊（0.5）擠掉，**淨增只有 +281**，
#                而且「加了相關性邊」與「拿掉部分產業邊」兩個變因混在一起
# `additive` ：**v2 的邊完全不動**，相關性邊加在上面（每節點最多 EXTRA_K 條）
#              → 與 v2 的差異**只有「多了相關性邊」這一個變因**，是乾淨的配對比較
#
# `build_kg_csr()` 原樣載入 npz、**不會再截斷**（已確認 trainer.py），
# 所以超過 MAX_NEIGHBORS_GAT 的邊會真的被用到，不需要改 config。
EXTRA_K = 5


def load_residual_returns() -> pd.DataFrame:
    """日報酬去掉每日橫斷面均值（＝去市場因子），回傳 (dates × stocks)。"""
    from experimental.baseline_common import _filter_universe, _load_raw
    pr = _load_raw("prices_raw")
    pr = pr[(pr["Date"] >= pd.Timestamp(CORR_START)) & (pr["Date"] <= pd.Timestamp(CORR_END))]
    pr = _filter_universe(pr).drop_duplicates(subset=["stock_id", "Date"], keep="last")
    px = pr.pivot(index="Date", columns="stock_id", values="Close").sort_index()
    px = px.where(px > 0)
    ret = px.pct_change()
    ret = ret.where(ret.abs() < 0.5)                      # 剔除還原殘留的極端值
    keep = ret.notna().sum() >= MIN_OBS
    ret = ret.loc[:, keep]
    resid = ret.sub(ret.mean(axis=1), axis=0)             # ★ 去市場因子
    print(f"[v3] 殘差報酬：{resid.shape[0]} 天 × {resid.shape[1]} 支"
          f"（{CORR_START} ~ {CORR_END}，門檻 ≥{MIN_OBS} 個有效日）", flush=True)
    return resid


def build_corr_edges(resid: pd.DataFrame, stock_ids: list[str]) -> tuple[list, list]:
    idx = {s: i for i, s in enumerate(stock_ids)}
    cols = [c for c in resid.columns if c in idx]
    R = resid[cols]
    C = R.corr(min_periods=MIN_OBS // 2).to_numpy(np.float32)
    np.fill_diagonal(C, -9.0)
    print(f"[v3] 相關矩陣 {C.shape}｜非對角 median "
          f"{np.nanmedian(C[C > -8]):.3f}（去市場因子後應接近 0）", flush=True)

    edges, weights, n_hit = [], [], 0
    for j, s in enumerate(cols):
        row = np.nan_to_num(C[j], nan=-9.0)
        order = np.argsort(row)[::-1][:TOP_K]
        for o in order:
            if row[o] < MIN_CORR:
                break
            edges.append((idx[s], idx[cols[o]]))
            weights.append(W_CORR)
            n_hit += 1
    print(f"[v3] 相關性邊（單向）：{n_hit:,} 條｜"
          f"有鄰居的股票 {len({e[0] for e in edges}):,} / {len(cols):,} 支"
          f"（門檻 ρ≥{MIN_CORR}、每股上限 {TOP_K}）", flush=True)
    return edges, weights


def build(out_path: Path = KG_V3_PATH, mode: str = "additive") -> None:
    v2 = np.load(KG_V2_PATH, allow_pickle=True)
    stock_ids = [str(s) for s in v2["stock_ids"]]
    ei2, ea2 = v2["edge_index"], v2["edge_attr"]
    print(f"[v3] v2 底圖：{len(stock_ids):,} 節點 / {ei2.shape[1]:,} 邊", flush=True)

    resid = load_residual_returns()
    e_new, w_new = build_corr_edges(resid, stock_ids)

    # 合併（含 v2 既有邊），去重取權重最大者，再套每節點上限
    from collections import defaultdict
    best: dict[tuple[int, int], float] = {}
    for a, b, w in zip(ei2[0], ei2[1], ea2):
        k = (int(a), int(b))
        best[k] = max(best.get(k, 0.0), float(w))
    n_dup = 0
    for (a, b), w in zip(e_new, w_new):
        if a == b:
            continue
        if (a, b) in best:
            n_dup += 1
        best[(a, b)] = max(best.get((a, b), 0.0), w)

    if mode == "replace":
        per_node = defaultdict(list)
        for (a, b), w in best.items():
            per_node[a].append((w, b))
        ei, ea = [], []
        for a, lst in per_node.items():
            for w, b in sorted(lst, key=lambda x: -x[0])[:MAX_NEIGHBORS_GAT]:
                ei.append((a, b))
                ea.append(w)
    else:                                    # additive：v2 邊完全不動，相關性邊加在上面
        v2_set = {(int(a), int(b)) for a, b in zip(ei2[0], ei2[1])}
        ei = [(int(a), int(b)) for a, b in zip(ei2[0], ei2[1])]
        ea = [float(w) for w in ea2]
        added = defaultdict(int)
        n_add = 0
        for (a, b), w in zip(e_new, w_new):
            if a == b or (a, b) in v2_set or added[a] >= EXTRA_K:
                continue
            ei.append((a, b))
            ea.append(w)
            added[a] += 1
            n_add += 1
        print(f"[v3] additive：v2 的 {len(v2_set):,} 條邊原樣保留，"
              f"新增相關性邊 {n_add:,} 條（每節點上限 {EXTRA_K}）"
              f"→ **與 v2 的差異只有『多了相關性邊』一個變因**", flush=True)
    edge_index = np.array(ei, dtype=np.int32).T
    edge_attr = np.array(ea, dtype=np.float32)

    print(f"[v3] 相關性邊與 v2 既有邊重複 {n_dup:,} 條"
          f"（重複＝同產業/同集團本來就高相關，是合理的）", flush=True)
    print(f"[v3] 合併後：{edge_index.shape[1]:,} 邊（v2 是 {ei2.shape[1]:,}，"
          f"淨增 {edge_index.shape[1]-ei2.shape[1]:+,}）｜每節點上限 {MAX_NEIGHBORS_GAT}", flush=True)
    deg = np.bincount(edge_index[0], minlength=len(stock_ids))
    u, c = np.unique(np.round(edge_attr, 2), return_counts=True)
    print(f"[v3] 度分布 min/median/max = {deg.min()}/{int(np.median(deg))}/{deg.max()}"
          f"｜孤立節點 {(deg == 0).sum():,}", flush=True)
    print(f"[v3] 權重分布：" + "｜".join(f"{a}:{b:,}" for a, b in zip(u, c)), flush=True)

    np.savez_compressed(out_path, stock_ids=np.array(stock_ids),
                        edge_index=edge_index, edge_attr=edge_attr)
    print(f"✅ [v3] 已寫入 {out_path}", flush=True)
    spot_check(stock_ids, edge_index, edge_attr)


def spot_check(stock_ids, edge_index, edge_attr, probes=("2330", "2317", "1301", "2891")):
    """抽驗鄰居是否合理——舊圖就是敗在沒人看過 2330 的鄰居是誰（電器電纜、綠能環保）。"""
    idx = {s: i for i, s in enumerate(stock_ids)}
    try:
        from marketmamba.data.feature_spec import canonical_sector
        from marketmamba.data.hygiene import load_stock_info
        info = load_stock_info(latest_only=True)
        col = next((c for c in ("industry_category", "industry", "sector") if c in info.columns), None)
        name = dict(zip(info["stock_id"].astype(str), info.get("stock_name", info["stock_id"])))
        sec = dict(zip(info["stock_id"].astype(str), canonical_sector(info[col]))) if col else {}
    except Exception:                                     # noqa: BLE001
        name, sec = {}, {}
    print("\n[v3] 鄰居抽驗：")
    for p in probes:
        if p not in idx:
            continue
        m = edge_index[0] == idx[p]
        nb = [(stock_ids[b], w) for b, w in zip(edge_index[1][m], edge_attr[m])]
        nb.sort(key=lambda x: -x[1])
        s = "、".join(f"{b}{name.get(b,'')}({sec.get(b,'?')},{w})" for b, w in nb[:8])
        print(f"  {p}{name.get(p,'')}（{sec.get(p,'?')}）→ {s}", flush=True)


def compare() -> None:
    for tag, p in (("v2", KG_V2_PATH), ("v3", KG_V3_PATH)):
        if not p.exists():
            print(f"{tag}: 不存在")
            continue
        d = np.load(p, allow_pickle=True)
        u, c = np.unique(np.round(d["edge_attr"], 2), return_counts=True)
        deg = np.bincount(d["edge_index"][0], minlength=len(d["stock_ids"]))
        print(f"{tag}: {len(d['stock_ids']):,} 節點 / {d['edge_index'].shape[1]:,} 邊"
              f"｜度 median {int(np.median(deg))}｜孤立 {(deg == 0).sum():,}"
              f"｜權重 " + "｜".join(f"{a}:{b:,}" for a, b in zip(u, c)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--mode", choices=("additive", "replace"), default="additive")
    ap.add_argument("--out", default=None, help="輸出檔名（預設依 mode 決定）")
    a = ap.parse_args()
    if a.compare:
        compare()
    else:
        out = Path(PROCESSED_DIR) / (a.out or
                                     ("knowledge_graph_v3.npz" if a.mode == "additive"
                                      else "knowledge_graph_v3_replace.npz"))
        build(out, mode=a.mode)
        print()
        compare()
