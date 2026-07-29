"""
MarketMamba — B-3 步驟 3：用官方還原因子重建全歷史還原價（2026-07-28）
========================================================================
輸入：
  `Data/processed_v6/_raw_prices/raw_YYYY.parquet`  ← 步驟 2 重抓的**未還原原始價**
  `Data/processed_v6/ex_rights_raw.parquet`         ← 官方還原因子（上市 TWT49U + 上櫃 exDailyQ）
輸出：
  `Data/processed_v6/prices_adj_raw.parquet`（**新檔**，驗證通過前不動 `prices_raw`）

---------------------------------------------------------------------------
【還原公式與方向】

    adjusted(t) = raw(t) × Π{ adj_factor(e) : e 為除權息日, e > t }

即「回頭還原」：最新價格維持原始價（乘數為 1），歷史價格往下縮。
方向錯了會讓整條序列的報酬反向——這是本步驟最容易犯且最難察覺的錯，
所以下方有一個數學上的自我檢查：

  設除權息日 e 當天無真實漲跌，即 raw(e) = raw(e-1) × f(e)（剛好等於參考價），
  則還原後報酬 =
      adjusted(e) / adjusted(e-1) − 1
    = [raw(e) × Π_{x>e}] / [raw(e-1) × Π_{x>e-1}] − 1
    = [raw(e) / raw(e-1)] / f(e) − 1
    = f(e) / f(e) − 1 = 0     ✓ 除權息不再製造假跌幅

【除權息當日本身的歸屬】
  除權息日 e 當天的價格**已經**是還原後基準（開盤參考價就是 ref_price），
  所以 e 當天的乘數必須**排除** f(e)、只含之後的事件。
  程式用 `searchsorted(dates, t, side="right")` 達成——
  side="right" 讓「等於 t 的事件」被算進「t 之前」，正好排除自己。

【Volume 刻意不還原】
  無償配股會讓股數增加，理論上歷史成交量該乘上配股倍數才可比。
  但 `adj_factor` 是**現金股利 + 股票股利 + 現金增資合併**的結果，
  拿它去調整股數會把純現金股利（股數不變）也錯誤地放大。
  要正確處理需要單獨的「無償配股率」欄位，本表未存。
  → 依照本專案原則「缺值比編造值好」，Volume 維持原始值並在此明確記錄。
    （後續若要做：建表時多存 TWSE「權值」/ TPEX「每仟股無償配股」即可推導。）

【記憶體策略】
  乘數只跟「該股所有未來事件」有關，與價格資料無關，因此可以**先建好每股的
  因子查表，再逐年套用**——全程不需要把 8.7M 列同時放進記憶體。

用法（repo 根目錄）：
    python V6/scripts/apply_ex_rights_adjustment.py --dry-run      # 用已完成的年度檔試算
    python V6/scripts/apply_ex_rights_adjustment.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V6_DIR = Path(__file__).resolve().parent.parent
if str(_V6_DIR) not in sys.path:
    sys.path.insert(0, str(_V6_DIR))

from marketmamba.config import PROCESSED_DIR  # noqa: E402

RAW_DIR = PROCESSED_DIR / "_raw_prices"
EX_PATH = PROCESSED_DIR / "ex_rights_raw.parquet"
OUT_PATH = PROCESSED_DIR / "prices_adj_raw.parquet"
PRICE_COLS = ["Open", "High", "Low", "Close"]


# ============================================================
# 因子查表
# ============================================================

def build_factor_lookup(ex: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    每支股票 → (事件日期 int64 陣列, 後綴累積乘積陣列)。

    後綴累積乘積 `suffix[i] = Π_{j>=i} f_j`，長度 n+1，`suffix[n] = 1.0`。
    查詢時 `k = searchsorted(dates, t, "right")`，乘數即 `suffix[k]`
    ——涵蓋所有嚴格晚於 t 的事件。
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ex = ex.sort_values(["stock_id", "Date"])
    for sid, g in ex.groupby("stock_id", sort=False):
        f = pd.to_numeric(g["adj_factor"], errors="coerce").to_numpy(dtype="float64")
        d = g["Date"].to_numpy(dtype="datetime64[ns]").astype("int64")
        ok = np.isfinite(f) & (f > 0)
        f, d = f[ok], d[ok]
        if len(f) == 0:
            continue
        suffix = np.ones(len(f) + 1, dtype="float64")
        suffix[:-1] = np.cumprod(f[::-1])[::-1]
        out[str(sid)] = (d, suffix)
    return out


def apply_to_frame(df: pd.DataFrame,
                   lut: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """對單一年度的價格表套用還原乘數。回傳含 `adj_mult` 欄的新表。"""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    t = df["Date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    mult = np.ones(len(df), dtype="float64")

    sid_arr = df["stock_id"].astype(str).to_numpy()
    order = np.argsort(sid_arr, kind="stable")
    sid_sorted = sid_arr[order]
    bounds = np.searchsorted(sid_sorted, np.unique(sid_sorted), side="left")
    bounds = np.append(bounds, len(sid_sorted))
    uniq = np.unique(sid_sorted)

    for i, sid in enumerate(uniq):
        ent = lut.get(sid)
        if ent is None:
            continue                                  # 無除權息紀錄 → 乘數 1
        d, suffix = ent
        idx = order[bounds[i]:bounds[i + 1]]
        k = np.searchsorted(d, t[idx], side="right")  # 排除除權息日自己
        mult[idx] = suffix[k]

    df["adj_mult"] = mult
    for c in PRICE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * mult
    return df


# ============================================================
# 驗證
# ============================================================

def validate(adj: pd.DataFrame, raw: pd.DataFrame, ex: pd.DataFrame) -> None:
    """
    核心驗證：**除權息日的假跌幅應該消失**。

    做法是比較「除權息日當天的報酬」在還原前後的分布。
    還原前，該日報酬會系統性地包含 −(股利/股價)；還原後應該收斂到 0 附近
    且與「非除權息日」的一般報酬分布相似。
    """
    print()
    print("=" * 78)
    print("■ 驗證 1：除權息日的假跌幅是否消失")
    print("=" * 78)

    def _ret(df: pd.DataFrame) -> pd.DataFrame:
        d = df[["Date", "stock_id", "Close"]].sort_values(["stock_id", "Date"])
        d["ret"] = d.groupby("stock_id")["Close"].pct_change()
        return d

    r_raw, r_adj = _ret(raw), _ret(adj)
    ev = ex[["Date", "stock_id"]].copy()
    ev["is_ex"] = True
    for tag, r in [("還原前（原始價）", r_raw), ("還原後", r_adj)]:
        m = r.merge(ev, on=["Date", "stock_id"], how="left")
        m["is_ex"] = m["is_ex"].fillna(False)
        a = m.loc[m["is_ex"], "ret"].dropna()
        b = m.loc[~m["is_ex"], "ret"].dropna()
        if len(a) < 10:
            print(f"  {tag}: 除權息日樣本不足（n={len(a)}）")
            continue
        print(f"  {tag}:")
        print(f"    除權息日   n={len(a):>7,}｜median {a.median():+.4%}"
              f"｜mean {a.mean():+.4%}｜<−2% 的比例 {(a < -0.02).mean():6.2%}")
        print(f"    一般交易日 n={len(b):>7,}｜median {b.median():+.4%}"
              f"｜mean {b.mean():+.4%}｜<−2% 的比例 {(b < -0.02).mean():6.2%}")

    print()
    print("=" * 78)
    print("■ 驗證 2：乘數的合理性")
    print("=" * 78)
    mm = adj["adj_mult"]
    print(f"  乘數 min {mm.min():.6f}｜median {mm.median():.6f}｜max {mm.max():.6f}")
    print(f"  乘數 == 1 的列（無後續除權息）：{int((mm == 1.0).sum()):,} / {len(mm):,}"
          f"（{(mm == 1.0).mean():.1%}）")
    print(f"  乘數 > 1 的列：{int((mm > 1.0).sum()):,}"
          f"（來自減資事件 adj_factor>1，屬合法）")
    # ── 單調性：方向要想清楚，寫反了會誤判正確的結果 ──────────────
    # mult(t) = Π{f(e) : e > t}。t 越過事件 e 時 f(e) 從乘積掉出，
    # 故 mult(後) = mult(前) / f(e)：
    #     除權息 f < 1  → 乘數**遞增**（歷史被壓低、最新為 1）
    #     減資   f > 1  → 乘數遞減（唯一合法的下降情形）
    # 初版把斷言寫成「應遞減」，於是把 4,383 筆完全正確的遞增報成違反。
    # 驗算：9105 於 2005-04-12（f=0.1）乘數 0.026144 → 0.261443 = 0.026144/0.1 ✓
    chk = adj[["Date", "stock_id", "adj_mult"]].sort_values(["stock_id", "Date"])
    chk["prev_date"] = chk.groupby("stock_id")["Date"].shift(1)
    d = chk.groupby("stock_id")["adj_mult"].diff()
    n_up = int((d > 1e-12).sum())
    n_down = int((d < -1e-12).sum())

    # 下降只該發生在減資（adj_factor > 1）。
    # ⚠️ 不可用「事件日 == 價格日」精確比對：官方「恢復買賣日」那天該股不一定有成交，
    #    乘數下降會出現在**復牌後第一個實際交易日**（實測落差 1–14 天，最長 89 天）。
    #    初版用精確比對，把 22 筆完全正確的結果報成「無法解釋」。
    #    語意正確的判準是：存在減資事件落在 (前一個價格日, 本價格日] 這個區間內。
    cut = ex[ex["adj_factor"] > 1][["Date", "stock_id"]]
    n_unexplained = 0
    if n_down:
        down = chk.loc[d.index[d < -1e-12], ["Date", "stock_id", "prev_date"]]
        by_sid: dict[str, np.ndarray] = {
            sid: g["Date"].to_numpy(dtype="datetime64[ns]")
            for sid, g in cut.groupby("stock_id")
        }
        for r in down.itertuples():
            ev = by_sid.get(str(r.stock_id))
            if ev is None:
                n_unexplained += 1
                continue
            lo = np.datetime64(r.prev_date) if pd.notna(r.prev_date) else \
                np.datetime64("1900-01-01")
            hi = np.datetime64(r.Date)
            if not ((ev > lo) & (ev <= hi)).any():
                n_unexplained += 1
    print(f"  乘數變動：上升 {n_up:,} 次（= 除權息事件，正確）｜"
          f"下降 {n_down:,} 次")
    print(f"  下降中無法用減資解釋的：{n_unexplained:,} 次 "
          f"{'✓' if n_unexplained == 0 else '❌ 需追查'}")

    print()
    print("=" * 78)
    print("■ 驗證 3：價格完整性")
    print("=" * 78)
    for c in PRICE_COLS:
        v = pd.to_numeric(adj[c], errors="coerce")
        print(f"  {c}: <=0 {int((v <= 0).sum()):,} 列｜NaN {int(v.isna().sum()):,} 列")
    ok = ((adj["Low"] <= adj[["Open", "Close"]].min(axis=1) + 1e-9)
          & (adj["High"] >= adj[["Open", "Close"]].max(axis=1) - 1e-9)
          & (adj["High"] >= adj["Low"] - 1e-9))
    print(f"  OHLC 順序（乘法還原應完全保序）：違反 {int((~ok).sum()):,} 列"
          f"｜還原前違反 "
          f"{int((~((raw['Low'] <= raw[['Open','Close']].min(axis=1) + 1e-9) & (raw['High'] >= raw[['Open','Close']].max(axis=1) - 1e-9))).sum()):,} 列")


def spot_check(adj: pd.DataFrame, raw: pd.DataFrame, ex: pd.DataFrame,
               n: int = 5) -> None:
    """挑幾個「股利佔股價比例最大」的事件逐筆列出，人工可讀（規則 7）。"""
    print()
    print("=" * 78)
    print("■ 驗證 4：單筆事件逐檔對照（挑還原幅度最大的）")
    print("=" * 78)
    days = set(pd.to_datetime(raw["Date"]).unique())
    cand = ex[ex["Date"].isin(days)].nsmallest(n * 4, "adj_factor")
    shown = 0
    for _, e in cand.iterrows():
        sid, dt = str(e["stock_id"]), e["Date"]
        for tag, src in [("原始", raw), ("還原", adj)]:
            s = src[(src["stock_id"].astype(str) == sid)].sort_values("Date")
            s = s[["Date", "Close"]].reset_index(drop=True)
            pos = s.index[s["Date"] == dt]
            if len(pos) == 0 or pos[0] == 0:
                break
            i = pos[0]
            prev, cur = s.loc[i - 1, "Close"], s.loc[i, "Close"]
            print(f"  {sid} {str(dt.date())} {e['kind']} "
                  f"(官方 adj_factor={e['adj_factor']:.6f})  {tag}："
                  f"{prev:.4f} → {cur:.4f} = {cur/prev - 1:+.2%}")
        else:
            print(f"    └ 官方參考價隱含 {e['adj_factor'] - 1:+.2%}，"
                  f"差額才是當日真實漲跌")
            shown += 1
            if shown >= n:
                break


# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.print_help()
        return

    files = sorted(RAW_DIR.glob("raw_*.parquet"))
    if not files:
        print(f"❌ 找不到重抓的原始價：{RAW_DIR}")
        return
    years = [int(p.stem.split("_")[1]) for p in files]
    print(f"[輸入] 原始價年度檔 {len(files)} 個：{years[0]}–{years[-1]}")
    if a.dry_run:
        print("       （--dry-run：只用目前已完成的年份試算，重抓仍在進行中屬正常）")

    ex = pd.read_parquet(EX_PATH)
    ex["Date"] = pd.to_datetime(ex["Date"])
    ex["stock_id"] = ex["stock_id"].astype(str)
    print(f"[輸入] 還原因子 {len(ex):,} 筆｜{ex['stock_id'].nunique():,} 支"
          f"｜{ex['Date'].min().date()} → {ex['Date'].max().date()}")

    lut = build_factor_lookup(ex)
    print(f"[因子] 建立查表：{len(lut):,} 支股票")

    parts_adj, parts_raw = [], []
    for p in files:
        d = pd.read_parquet(p)
        parts_raw.append(d.assign(Date=pd.to_datetime(d["Date"])))
        parts_adj.append(apply_to_frame(d, lut))
        print(f"  {p.stem}: {len(d):>7,} 列 → 已套用", flush=True)

    adj = pd.concat(parts_adj, ignore_index=True)
    raw = pd.concat(parts_raw, ignore_index=True)
    del parts_adj, parts_raw
    print(f"[輸出] 合計 {len(adj):,} 列｜{adj['stock_id'].nunique():,} 支"
          f"｜{adj['Date'].min().date()} → {adj['Date'].max().date()}")

    validate(adj, raw, ex)
    spot_check(adj, raw, ex)

    if a.apply:
        adj.drop(columns=["adj_mult"]).to_parquet(OUT_PATH, index=False)
        print(f"\n✅ 已寫入 {OUT_PATH.name}（新檔，`prices_raw` 未被改動）")
    else:
        print("\n--dry-run：未寫檔。")


if __name__ == "__main__":
    main()
