"""
compare_panel_ic.py — 舊面板 vs 新面板：同一份標籤下的新舊 IC 對照
==================================================================
2026-08-08 回補 `prices_raw`（+441,543 列）並重建 base matrix（+356,900 列）之後，
每個 arm 都在新面板上重評分成 `{arm}__live.parquet`。這支負責回答：

    **同一個 checkpoint，餵它比較完整的資料，訊號層變好還是變壞？**

為什麼需要這支（而不是看 `--score-window` 的 log）
--------------------------------------------------
`run_v62_inference.score_window()` 原本會印一行
`對照（舊資料基礎）：mean IC +0.1145`——**那是 `v2_kg_nomacro` 一個 arm 的舊值，
卻對每個 arm 都印同一個數字**。重評分八個 arm 時 log 長成
「head10d 從 0.1145 掉到 0.1094」，但 head10d 的舊值根本不是 0.1145。
2026-08-09 已移除那行，改由本檔做真正的對照。

★ 兩種口徑都報，因為它們回答不同問題
------------------------------------
  **own**    ：各自用自己的宇宙算 IC → 「總效應」（含宇宙變大本身）
  **common** ：只取兩邊都有的 (Date, stock_id) → 「純訊號效應」

宇宙變大不是雜訊、是這次修正的一部分，所以 own 才是「實際會拿到的」；
但把它跟 common 並列，才分得出「IC 變化有多少來自多了那些股票」。
**只報一個都會誤導。**

判準沿用 CLAUDE.md：mean IC 的實務門檻 ±0.009。

用法
----
    MM_PROTOCOL=v2 python V6/experimental/compare_panel_ic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent.parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

from marketmamba.config import PROCESSED_DIR                       # noqa: E402

SCORE_DIR = _V6 / "experimental" / "result" / "scores"
LABEL = Path(PROCESSED_DIR) / "baseline_cache_v2" / "baseline_label_10d.parquet"
IC_THRESHOLD = 0.009

# arm → (舊分數檔, 新分數檔, 評估標籤欄)
# ⚠️ 標籤欄要對應該 arm 讀的那顆頭：讀第 1 欄（10d 頭）的要用 Alpha_10d，
#    用錯欄位會讓「新舊比較」變成「不同 horizon 的比較」。
PAIRS = [
    ("v2_kg_nomacro",      "v2_kg_nomacro",            "Alpha_5d"),
    ("v2_kg_nomacro_h10",  "v2_kg_nomacro__head10d",   "Alpha_10d"),
    ("head10d",            "h20abl_h10__head10d",      "Alpha_10d"),
    ("head20d",            "h20abl_h20__head10d",      "Alpha_20d"),
    ("v2_kg",              "v2_kg",                    "Alpha_5d"),
    ("v3_kg",              "v3_kg",                    "Alpha_5d"),
    ("old_kg",             "old_kg",                   "Alpha_5d"),
    ("no_gat",             "no_gat",                   "Alpha_5d"),
    ("ridge",              "ridge__lab5d_p30",         "Alpha_5d"),
    ("gbdt",               "gbdt__p30fix_20260808",    "Alpha_5d"),
    ("gru",                "gru__p30_s20260713",       "Alpha_5d"),
]


def _load(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    d["stock_id"] = d["stock_id"].astype(str)
    return d[["Date", "stock_id", "score"]]


def daily_ic(sc: pd.DataFrame, lab: pd.DataFrame, col: str,
             keys: set | None = None) -> np.ndarray:
    m = sc.merge(lab[["Date", "stock_id", col]], on=["Date", "stock_id"], how="inner")
    if keys is not None:
        m = m[[(d, s) in keys for d, s in zip(m["Date"], m["stock_id"])]]
    out = []
    for _, g in m.groupby("Date", sort=True):
        y, s = g[col].to_numpy(np.float64), g["score"].to_numpy(np.float64)
        ok = np.isfinite(y) & np.isfinite(s)
        if ok.sum() >= 30:
            out.append(pd.Series(s[ok]).corr(pd.Series(y[ok]), method="spearman"))
    return np.array([x for x in out if np.isfinite(x)])


def main() -> int:
    if not LABEL.exists():
        raise SystemExit(f"❌ 找不到 {LABEL.name}（先跑 label_10d.py --force）")
    lab = pd.read_parquet(LABEL)
    lab["Date"] = pd.to_datetime(lab["Date"]).dt.strftime("%Y-%m-%d")
    lab["stock_id"] = lab["stock_id"].astype(str)
    print(f"標籤：{LABEL.name}（{len(lab):,} 列，重建於新面板）\n")

    print("=" * 104)
    print(f"{'arm':22s}{'標籤':>10s}"
          f"{'舊IC(own)':>11s}{'新IC(own)':>11s}{'Δown':>9s}"
          f"{'舊IC(com)':>11s}{'新IC(com)':>11s}{'Δcom':>9s}  判定")
    print("-" * 104)
    rows = []
    for arm, old_name, col in PAIRS:
        new = _load(SCORE_DIR / f"{arm}__live.parquet")
        old = _load(SCORE_DIR / f"{old_name}.parquet")
        if new is None or old is None:
            print(f"{arm:22s}{col:>10s}   （缺 "
                  f"{'新' if new is None else ''}{'舊' if old is None else ''} 分數檔）")
            continue
        common = set(map(tuple, old[["Date", "stock_id"]].values)) & \
                 set(map(tuple, new[["Date", "stock_id"]].values))
        o_own, n_own = daily_ic(old, lab, col), daily_ic(new, lab, col)
        o_com, n_com = daily_ic(old, lab, col, common), daily_ic(new, lab, col, common)
        d_own, d_com = n_own.mean() - o_own.mean(), n_com.mean() - o_com.mean()
        verdict = "—" if abs(d_own) < IC_THRESHOLD else ("✅ 變好" if d_own > 0 else "⚠️ 變差")
        print(f"{arm:22s}{col:>10s}"
              f"{o_own.mean():+11.4f}{n_own.mean():+11.4f}{d_own:+9.4f}"
              f"{o_com.mean():+11.4f}{n_com.mean():+11.4f}{d_com:+9.4f}  {verdict}")
        rows.append((arm, d_own, d_com, len(common), len(old), len(new)))
    print("-" * 104)
    print(f"判準：|Δ mean IC| ≥ {IC_THRESHOLD}（CLAUDE.md 的實務門檻）才算有變化")
    print("own = 各自的宇宙（總效應，含宇宙變大）｜common = 兩邊共同的 (Date, stock_id)（純訊號效應）")
    if rows:
        big = [r for r in rows if abs(r[1]) >= IC_THRESHOLD]
        print(f"\n超出門檻的 arm：{[r[0] for r in big] or '無'}"
              f"（共 {len(rows)} 個）")
        print(f"共同鍵佔比：{rows[0][3]/rows[0][5]:.1%}（新分數有多少落在舊宇宙內）")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
