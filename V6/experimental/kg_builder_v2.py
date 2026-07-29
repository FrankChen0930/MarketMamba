"""
kg_builder_v2.py — 知識圖譜 v2（決策2：先重建乾淨的產業圖）
==============================================================
隔離：**純新增**。不改 `marketmamba/knowledge/graph_builder.py`，
      輸出寫到新檔名 `knowledge_graph_v2.npz`，
      V6.1 每日推論讀的仍是原本的 `knowledge_graph_cache.npz`，零影響。

---------------------------------------------------------------------------
為什麼要重建：現行的圖不是「可以更好」，是壞的

2026-07-29 實測 `knowledge_graph_cache.npz`：

  ① 42,864 個節點，其中**只有 2,510 個是真股票**
     其餘是 ETF / 權證 / 特別股（00400A、00631L…），來自污染的 ticker_universe。
     它們永遠不會出現在訓練橫斷面裡，佔掉 94% 的邊卻毫無作用。

  ② 642,451 條邊中，**滾動相關性邊 0 條**
     `build_rolling_correlation_edges()` 從未成功產出（例外被 try/except 吞掉），
     動態層實質不存在。

  ③ 供應鏈邊是垃圾
     `_parse_tpex_html()` 的作法是 **regex 抓 HTML 裡所有 4 位數字**當關聯股票。
     實測 2330 的「供應鏈鄰居」包含電器電纜、化學工業、綠能環保，
     而且 2330 / 2317 / 2303 的鄰居尾巴完全相同——抓到的是同一批頁面樣板數字。

  ④ 2330 的產業邊一條都沒有
     每節點上限 15 條、依權重排序，於是 4 條集團邊（0.8）+ 11 條爬蟲垃圾（0.6）
     把名額佔滿，真正的同業（0.5）全被擠掉。台股最大的股票，圖上的鄰居是雜訊。

  ⑤ 產業邊用**未設 seed 的 `random.sample`** → 每次重建結果都不一樣，無法重現。

  ⑥ `CONGLOMERATE_TABLE` 有一筆字面錯誤：`"2遠東新世紀"`（不是股票代號）。

---------------------------------------------------------------------------
v2 做了什麼

  1. 宇宙改用 `hygiene.filter_tradable_universe()` → 只有真股票
  2. 產業別先過 `feature_spec.canonical_sector()`
     （實測 TPEX 用「運動休閒類」、TWSE 用「運動休閒」，不正規化會把同一個產業
       沿上市/上櫃切成兩個互不相連的群，而那條界線在經濟上毫無意義）
  3. 產業邊改**決定性**連法：sector leaders + 市值鄰近同業（見 `build_sector_edges_v2`）
  4. **刪掉 regex 爬蟲那層**——真正的產業鏈邊留給 Phase 4-A 用正式節點對應表做
  5. 集團表修掉錯誤代號
  6. **不含相關性邊**（那是選項 C，且現行寫法有 look-ahead：
     用整份資料最後 60 天算好、套用到 2005 年的訓練樣本）

---------------------------------------------------------------------------
已知限制（誠實揭露，不假裝解決）

  - 圖是**單一靜態快照**：產業別與市值都取自最新資料，套用到全歷史。
    產業成員與相對規模是慢變數，比相關性邊的 look-ahead 輕微得多，但仍不是 PIT。
    實測「真正的公司產業重新分類」在 2023→2026 只有個位數（多數變動是交易所改分類名），
    所以這個近似可接受——但這是量測結果，不是假設。
  - 市值用最新值排序 → 早期歷史的「同規模同業」其實是用今天的規模定義的。

用法（repo 根目錄）：
    python V6/experimental/kg_builder_v2.py            # 建圖 + 印抽驗
    python V6/experimental/kg_builder_v2.py --compare  # 另外與舊圖並排比較
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import KG_EDGE_WEIGHTS, MAX_NEIGHBORS_GAT, PROCESSED_DIR  # noqa: E402
from marketmamba.data.feature_spec import resolve_sector  # noqa: E402
from marketmamba.data.hygiene import filter_tradable_universe, load_stock_info  # noqa: E402

KG_V2_PATH = Path(PROCESSED_DIR) / "knowledge_graph_v2.npz"

W_SECTOR = KG_EDGE_WEIGHTS["twse_sector"]        # 0.5
W_CONGLOM = KG_EDGE_WEIGHTS["conglomerate"]      # 0.8

# 每支股票在同產業內的連邊配置（合計 <= MAX_NEIGHBORS_GAT）
N_SECTOR_LEADERS = 5     # 產業龍頭：捕捉產業 beta 的傳導
N_SECTOR_PEERS = 10      # 市值鄰近同業：規模相近者最具可比性

# ============================================================
# 集團表（大幅縮減，只保留可確認的從屬關係）
# ============================================================
#
# ⚠️ 舊表（`graph_builder.py:CONGLOMERATE_TABLE`）**有大量事實錯誤**，
#    而且這些是權重最高（0.8）的邊，會優先擠掉正確的同業邊。抽驗發現的問題：
#
#      鴻海集團   放了 2353 宏碁（獨立公司）、6005 群益證（群益金鼎證券，無從屬）
#      國泰集團   放了 2884 玉山金（獨立金控）、9910 豐泰（製鞋，完全無關）
#      富邦集團   放了 2883 開發金、5880 合庫金（皆為獨立金控）
#      遠東集團   放了 2401 凌陽科技（與遠東系無關）
#      友達集團   放了 3481 群創（是競爭對手，不是同集團）
#      廣達集團   放了 3376 新日興（獨立鉸鏈廠）
#      「台積電生態圈」「聯發科生態圈」根本不是集團，是人為湊的「相關個股」，
#      用 0.8 的權重表達「從屬關係」在語意上就是錯的
#
# 本表只保留**公開且明確的集團從屬**。寧可少連幾條邊，也不要用最高權重
# 灌入錯誤關係——那正是舊圖 2330 連不到任何同業的原因。
#
# 【待補】完整的集團／產業鏈關係應該由 Phase 4-A 用正式的節點對應表建立
#         （見 docs/phase4-industry-chain-fusion-plan-2026-06-27.md），
#         不要再靠手寫表累積錯誤。
CONGLOMERATE_TABLE: dict[str, list[str]] = {
    "台塑集團": ["1301", "1303", "1326", "6505"],          # 台塑/南亞/台化/台塑化
    "遠東集團": ["1402", "1710", "2845", "2903", "4904"],  # 遠東新世紀/東聯/遠東銀/遠百/遠傳
    "統一集團": ["1216", "2912"],                           # 統一企業/統一超商
    "鴻海集團": ["2317", "2354"],                           # 鴻海/鴻準
}

# ── 交叉比對結果（2026-07-29，來源：Yahoo 股市「集團股」分類頁）────────────
#
# 上面那份保守表與 Yahoo 分類比對：**0 個誤收、37 個漏收**（13 vs 50 檔）。
# 保守的方向是對的——漏連邊只是少了資訊，誤連邊會讓 GAT 沿著錯誤關係傳播訊號。
#
# ⚠️ 順帶驗證出舊表另一個錯誤：3532 台勝科被舊表放在「友達集團」，
#    實際上它是**台塑集團**成員（台塑九寶之一）。
#
# 以下是 Yahoo 的完整名單，等 GAT 消融證實圖有貢獻後再決定要不要採用。
# 注意 Yahoo 的集團分類本身是**供應商的判斷**、不是官方登記，
# 部分成員關係較鬆（例如統一集團的「網家」、鴻海集團的「康聯生醫」），
# 採用前建議再逐一過目。
CONGLOMERATE_TABLE_FULL: dict[str, list[str]] = {
    # 台塑九寶
    "台塑集團": ["1301", "1303", "1326", "1434", "2408", "3532", "6505", "8046", "8131"],
    # 遠東/亞東（Yahoo 分類名為「遠東/亞東」）
    "遠東集團": ["1102", "1402", "1460", "1710", "2606", "2845", "2903", "4904", "6997"],
    "統一集團": ["1216", "1232", "1789", "2511", "2855", "2912", "5902", "6558", "8044", "9907"],
    "鴻海集團": ["2258", "2314", "2317", "2328", "2354", "3062", "3149", "3413", "3437",
                 "3498", "4958", "5243", "6196", "6414", "6416", "6451", "6456", "6638",
                 "6665", "6668", "6933", "7455"],
}


# ============================================================
# 宇宙與屬性
# ============================================================

def load_universe() -> pd.DataFrame:
    """回傳 [stock_id, sector, market_cap]，只含可交易的真股票。"""
    pr = pd.read_parquet(Path(PROCESSED_DIR) / "prices_raw.parquet",
                         columns=["Date", "stock_id"])
    pr["Date"] = pd.to_datetime(pr["Date"])
    uni = pd.DataFrame({"stock_id": sorted(pr["stock_id"].astype(str).unique())})
    uni = filter_tradable_universe(uni)
    del pr

    # 用 resolve_sector（吃**完整累積表**）而不是 load_stock_info(latest_only=True)：
    # 後者在同一快照日期有多個標籤時是任意挑一個，會讓 2330 落在「電子工業」
    # 這個 250 支的早年大類，而不是「半導體業」。
    info = load_stock_info(latest_only=False)
    if not info.empty:
        uni = uni.merge(resolve_sector(info), on="stock_id", how="left")
        uni["sector"] = uni["sector"].fillna("Unknown")
    else:
        uni["sector"] = "Unknown"

    # 市值：取每支股票最後一筆
    mv_path = Path(PROCESSED_DIR) / "market_value_raw.parquet"
    if mv_path.exists():
        mv = pd.read_parquet(mv_path, columns=["Date", "stock_id", "market_value"])
        mv["Date"] = pd.to_datetime(mv["Date"])
        mv = (mv.sort_values("Date").drop_duplicates("stock_id", keep="last")
                [["stock_id", "market_value"]])
        mv["stock_id"] = mv["stock_id"].astype(str)
        uni = uni.merge(mv, on="stock_id", how="left")
    else:
        uni["market_value"] = np.nan
    # 缺市值者排到最後（不影響 leader 選擇，只影響它自己的 peer 挑選）
    uni["market_value"] = pd.to_numeric(uni["market_value"], errors="coerce").fillna(0.0)
    return uni.sort_values("stock_id").reset_index(drop=True)


# ============================================================
# 邊：產業（決定性）
# ============================================================

def build_sector_edges_v2(
    uni: pd.DataFrame,
    stock_ids: list[str],
    n_leaders: int = N_SECTOR_LEADERS,
    n_peers: int = N_SECTOR_PEERS,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    同產業連邊，**完全決定性**（同一份輸入永遠得到同一張圖）。

    每支股票 i 連到同產業中的：
      (a) 市值最大的 n_leaders 支（產業龍頭）
          → 龍頭的動向會傳導到整個產業，這是台股很明確的現象（台積電之於電子）
      (b) 市值排序上最鄰近的 n_peers 支（前後各半）
          → 規模相近的同業最具可比性

    取代原本的 `random.sample`：那個不設 seed，每次重建的圖都不一樣，
    實驗無法重現，也無法回答「IC 變化是來自改動還是來自重抽的圖」。
    """
    id_to_idx = {sid: i for i, sid in enumerate(stock_ids)}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for sector, grp in uni.groupby("sector", sort=True):
        # 市值由大到小；同市值時用 stock_id 決勝，確保完全決定性
        grp = grp.sort_values(["market_value", "stock_id"],
                              ascending=[False, True]).reset_index(drop=True)
        members = [s for s in grp["stock_id"].tolist() if s in id_to_idx]
        n = len(members)
        if n < 2:
            continue

        leaders = members[:min(n_leaders, n)]
        for pos, sid in enumerate(members):
            i = id_to_idx[sid]
            nbrs: list[str] = [s for s in leaders if s != sid]
            # 市值排序上的鄰居：以 pos 為中心取 n_peers 個，
            # **碰到邊界時把窗口整段平移**而不是截斷——否則產業龍頭（pos=0）
            # 只會拿到單邊的一半鄰居（初版實測 2330 只有 9 個鄰居而非 15）
            half = max(n_peers // 2, 1)
            lo = max(0, min(pos - half, n - n_peers - 1))
            hi = min(n, lo + n_peers + 1)
            lo = max(0, hi - n_peers - 1)
            nbrs += [s for s in members[lo:hi] if s != sid]

            seen: set[str] = set()
            for s in nbrs:
                if s in seen:
                    continue
                seen.add(s)
                edges.append((i, id_to_idx[s]))
                weights.append(W_SECTOR)

    return edges, weights


def build_conglomerate_edges_v2(
    stock_ids: list[str],
) -> tuple[list[tuple[int, int]], list[float]]:
    """集團內全連接。權重最高（0.8），因為集團從屬是最強的結構訊號。"""
    id_to_idx = {sid: i for i, sid in enumerate(stock_ids)}
    edges, weights = [], []
    missing: list[str] = []
    for _, members in CONGLOMERATE_TABLE.items():
        idxs = []
        for sid in members:
            if sid in id_to_idx:
                idxs.append(id_to_idx[sid])
            else:
                missing.append(sid)
        for i in idxs:
            for j in idxs:
                if i != j:
                    edges.append((i, j))
                    weights.append(W_CONGLOM)
    if missing:
        print(f"  [集團表] {len(missing)} 個代號不在宇宙內（已略過）：{sorted(set(missing))}")
    return edges, weights


# ============================================================
# 合併
# ============================================================

def merge_edges(
    all_edges: list[tuple[int, int]],
    all_weights: list[float],
    n_stocks: int,
    max_neighbors: int = MAX_NEIGHBORS_GAT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    重複的 (i,j) 取最大權重；每個節點保留至多 max_neighbors 條。

    排序鍵是 (權重 desc, 鄰居 index asc)——第二個鍵是為了決定性：
    只用權重排序時，同權重邊的先後取決於 dict 迭代順序，
    在不同 Python 版本/插入順序下可能不同。
    """
    pair_weight: dict[tuple[int, int], float] = {}
    for (i, j), w in zip(all_edges, all_weights):
        if w > pair_weight.get((i, j), -1.0):
            pair_weight[(i, j)] = w

    out: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (i, j), w in pair_weight.items():
        out[i].append((j, w))

    final_e, final_w = [], []
    for i in range(n_stocks):
        for j, w in sorted(out[i], key=lambda x: (-x[1], x[0]))[:max_neighbors]:
            final_e.append((i, j))
            final_w.append(w)

    if not final_e:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.float32)
    return (np.array(final_e, dtype=np.int32).T,
            np.array(final_w, dtype=np.float32))


# ============================================================
# 主流程
# ============================================================

def build(out_path: Path = KG_V2_PATH) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    print("=" * 78)
    print("知識圖譜 v2 建構")
    print("=" * 78)

    uni = load_universe()
    stock_ids = uni["stock_id"].tolist()
    n = len(stock_ids)
    n_unknown = int((uni["sector"] == "Unknown").sum())
    print(f"  宇宙：{n:,} 支（可交易、已排除 ETF/興櫃）")
    print(f"  產業：{uni['sector'].nunique()} 類｜無產業別 {n_unknown} 支")
    print(f"  市值：median {uni['market_value'].median():,.0f}｜"
          f"缺市值 {int((uni['market_value'] == 0).sum())} 支")

    e_sec, w_sec = build_sector_edges_v2(uni, stock_ids)
    print(f"  產業邊：{len(e_sec):,} 條（決定性，龍頭 {N_SECTOR_LEADERS} + "
          f"市值鄰近 {N_SECTOR_PEERS}）")
    e_con, w_con = build_conglomerate_edges_v2(stock_ids)
    print(f"  集團邊：{len(e_con):,} 條")

    edge_index, edge_attr = merge_edges(e_sec + e_con, w_sec + w_con, n)
    print(f"  合併後：{edge_index.shape[1]:,} 條（每節點上限 {MAX_NEIGHBORS_GAT}）")

    deg = np.bincount(edge_index[0], minlength=n)
    print(f"  degree：median {np.median(deg):.0f}｜min {deg.min()}｜max {deg.max()}｜"
          f"孤立節點 {int((deg == 0).sum())} 支")
    uniq_w, cnt_w = np.unique(np.round(edge_attr, 2), return_counts=True)
    print("  權重分布：" + "｜".join(f"{w}: {c:,}" for w, c in zip(uniq_w, cnt_w)))

    np.savez_compressed(out_path,
                        stock_ids=np.array(stock_ids),
                        edge_index=edge_index,
                        edge_attr=edge_attr)
    print(f"  已存：{out_path}（{out_path.stat().st_size / 1e6:.2f} MB）")
    return edge_index, edge_attr, stock_ids, uni


def spot_check(edge_index: np.ndarray, edge_attr: np.ndarray,
               stock_ids: list[str], uni: pd.DataFrame,
               probes: tuple[str, ...] = ("2330", "2317", "1101", "2412", "2454")) -> None:
    """
    鄰居抽驗——**這一關必須人工看得懂才算做完**（規則 7）。

    2026-07-29 正是靠這個抽驗發現舊圖是壞的：2330 的鄰居裡有電器電纜與綠能環保。
    只印節點數與邊數的話，舊圖看起來一切正常。
    """
    print()
    print("=" * 78)
    print("鄰居抽驗（人工可讀性檢查）")
    print("=" * 78)
    idx = {s: i for i, s in enumerate(stock_ids)}
    sec = dict(zip(uni["stock_id"], uni["sector"]))
    name = {}
    info = load_stock_info(latest_only=True)
    if not info.empty and "stock_name" in info.columns:
        name = dict(zip(info["stock_id"].astype(str), info["stock_name"].astype(str)))

    for t in probes:
        i = idx.get(t)
        if i is None:
            print(f"\n  {t} 不在宇宙內")
            continue
        m = edge_index[0] == i
        nbrs, ws = edge_index[1][m], edge_attr[m]
        order = np.argsort(-ws, kind="stable")
        print(f"\n  {t} {name.get(t, '')}（{sec.get(t, '?')}）→ {m.sum()} 個鄰居")
        for j, w in zip(nbrs[order], ws[order]):
            s = stock_ids[j]
            tag = "集團" if w >= 0.8 else "同業"
            print(f"      {w:.1f} {tag}  {s} {name.get(s, ''):<8s} {sec.get(s, '?')}")


def compare_with_old() -> None:
    """與舊圖並排，量化改了什麼。"""
    old_p = Path(PROCESSED_DIR) / "knowledge_graph_cache.npz"
    if not old_p.exists():
        print("  找不到舊圖，略過比較")
        return
    old = np.load(old_p, allow_pickle=True)
    new = np.load(KG_V2_PATH, allow_pickle=True)
    o_ids = [str(s) for s in old["stock_ids"]]
    n_ids = [str(s) for s in new["stock_ids"]]
    import re
    real = re.compile(r"^\d{4}$")
    print()
    print("=" * 78)
    print("新舊圖對照")
    print("=" * 78)
    print(f"  節點：{len(o_ids):,} → {len(n_ids):,}"
          f"（真股票 {sum(bool(real.match(s)) for s in o_ids):,} → "
          f"{sum(bool(real.match(s)) for s in n_ids):,}）")
    print(f"  邊　：{old['edge_index'].shape[1]:,} → {new['edge_index'].shape[1]:,}")
    for tag, d in (("舊", old), ("新", new)):
        u, c = np.unique(np.round(d["edge_attr"], 2), return_counts=True)
        print(f"  {tag}圖權重：" + "｜".join(f"{w}: {n:,}" for w, n in zip(u, c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="與舊圖並排比較")
    args = ap.parse_args()

    ei, ea, sids, uni = build()
    spot_check(ei, ea, sids, uni)
    if args.compare:
        compare_with_old()


if __name__ == "__main__":
    main()
