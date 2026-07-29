"""
splitters.py — Purged Walk-Forward + Embargo（F4）
====================================================
隔離：純新增模組，不改 `marketmamba/evaluation/walk_forward.py` 的既有行為。

---------------------------------------------------------------------------
為什麼需要：現行切分每個 fold 邊界都在洩漏

2026-07-29 全專案 grep 確認：**沒有任何 purge / embargo 程式碼**。
`walk_forward.py` 是 expanding window、train 與 test **直接相鄰**，
而 label 是「未來 N 天報酬」（N = 5 / 20 / 60）。

於是訓練集最後 N 天的樣本，其 label 期間**整段落在測試區間內**：

    train ────────────────────┤├──────────── test
                         t=T-60│t=T
                              └──── 這些樣本的 label 是 [T-60, T] 之後的報酬，
                                    與 test 期完全重疊

模型在訓練時就看過測試期的價格走勢。這不是理論風險——60 天的 horizon
在 6 個月（約 120 交易日）的 test window 下，等於半個 test 期被偷看過。

參考 López de Prado《Advances in Financial Machine Learning》第 7 章
（本專案 `quant_rigor_checklist.md` 的 B2 / B3）。

---------------------------------------------------------------------------
兩個機制

  **Purge**：剔除訓練集中「label 期間與測試區間重疊」的樣本。
    樣本 t 的 label 覆蓋 [t, t+horizon]，只要它碰到 test 區間就剔除。
    對 expanding window（訓練永遠在測試之前）而言，
    等價於砍掉訓練集尾端的 `horizon` 個交易日。

  **Embargo**：purge 之外再留一段空白緩衝。
    處理的是 purge 管不到的殘餘關聯——特徵本身有自相關
    （本專案有 60 天的 rolling 特徵、252 天的序列窗），
    訓練集最後一天的特徵與測試期第一天的特徵共用大量原始資料。

---------------------------------------------------------------------------
參數選擇（協定 v2.0，凍結後不中途改）

  horizon      = 該模型 label 的最長天數（多 horizon 模型取 max，即 60）
  embargo_days = 20

  embargo 取 20 的理由：本專案最長的 rolling 特徵窗是 60 天
  （`baseline_common.ROLL_MEAN_WINDOWS` 到 60、`Volatility_20d`、`MA_60`），
  完全消除特徵重疊要留滿 60 天，但那樣每個 fold 要砍掉 80 個交易日的訓練資料。
  20 天（1/3 個最長窗）是嚴謹度與樣本量的折衷。
  **這是個選擇不是定理**，所以寫進協定並在報告中揭露。

⚠️ 誠實預期：加上 purge/embargo 之後報出來的 IC **會下降**。
   那是對的方向——現行數字含邊界洩漏。新舊都要留，對照表明講兩者的差異。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """一個切分。所有欄位都是實際的交易日陣列，不是索引。"""
    fold_id: int
    train_dates: np.ndarray
    test_dates: np.ndarray
    n_purged: int          # 因 label 重疊被剔除的訓練日數
    n_embargoed: int       # 因 embargo 被剔除的訓練日數

    @property
    def train_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_dates[-1]) if len(self.train_dates) else pd.NaT

    @property
    def test_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_dates[0]) if len(self.test_dates) else pd.NaT

    def describe(self) -> str:
        return (f"fold {self.fold_id:>2d}｜train {len(self.train_dates):>5,} 日 "
                f"(…{str(self.train_end)[:10]})｜"
                f"test {len(self.test_dates):>4,} 日 "
                f"({str(self.test_start)[:10]}…)｜"
                f"purge -{self.n_purged}｜embargo -{self.n_embargoed}")


def purged_split(
    all_dates: np.ndarray | pd.Series,
    test_start,
    test_end,
    horizon: int,
    embargo_days: int = 20,
    train_start=None,
) -> Split:
    """
    單一切分（給「單一切分為主」的 Phase 3 harness 用）。

    Args:
        all_dates: 全部交易日（會自動排序去重）
        horizon:   label 覆蓋的交易日數；多 horizon 模型傳最長的那個
        embargo_days: purge 之外額外的緩衝

    訓練集 = 所有 < (test_start - horizon - embargo) 的交易日。
    以**交易日**計算而非日曆日——與 label 用 `.shift(-n)` 的語意一致。
    """
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(all_dates).unique())).sort_values()
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)

    test_mask = (d >= ts) & (d <= te)
    test_dates = d[test_mask]

    before = d[d < ts]
    n_cut = horizon + embargo_days
    keep = before[:-n_cut] if n_cut > 0 and len(before) > n_cut else (
        before if n_cut == 0 else d[:0])
    if train_start is not None:
        keep = keep[keep >= pd.Timestamp(train_start)]

    n_dropped = len(before) - len(keep) - (
        0 if train_start is None else int((before < pd.Timestamp(train_start)).sum()))
    n_purged = min(horizon, max(n_dropped, 0))
    n_embargoed = max(n_dropped - n_purged, 0)

    return Split(0, keep.to_numpy(), test_dates.to_numpy(), n_purged, n_embargoed)


def train_val_split_dates(
    all_dates,
    cutoff_train_end: str,
    horizon: int,
    embargo_days: int = 20,
    label: str = "",
) -> tuple[list[str], list[str]]:
    """
    給訓練腳本用的便利函式：把 `train ≤ cutoff / val > cutoff` 這種**零 purge**
    的切分，換成有 purge + embargo 的版本。回傳 (train_dates, val_dates)，
    型別是 `str` 清單，與既有訓練腳本的用法相容。

    這支存在的理由：全專案的訓練腳本都是
        train_dates = [d for d in all_dates if d <= cutoff]
        val_dates   = [d for d in all_dates if d >  cutoff]
    train 尾端 `horizon` 天的 label 整段落在 val 區間內，模型訓練時就看過了。
    改用本函式只需換一行，且會**明確印出砍掉幾天**——不印的話沒人會發現差別。

    Args:
        horizon: 該次訓練 label 的最長天數。短線模型（5d/10d）傳 10；
                 多 horizon（5/20/60）傳 60。傳錯會 purge 不足或過度。
    """
    d = sorted({str(x)[:10] for x in pd.Series(all_dates).astype(str)})
    cut = str(cutoff_train_end)[:10]
    before = [x for x in d if x <= cut]
    val = [x for x in d if x > cut]

    n_cut = horizon + embargo_days
    train = before[:-n_cut] if n_cut > 0 and len(before) > n_cut else []
    tag = f"[{label}] " if label else ""
    print(f"{tag}[purged split] cutoff={cut}｜horizon={horizon}＋embargo={embargo_days}"
          f" → 砍掉 train 尾端 {len(before) - len(train)} 天", flush=True)
    print(f"{tag}[purged split] train {len(before)} → {len(train)} 天"
          f"（…{train[-1] if train else '—'}）｜val {len(val)} 天"
          f"（{val[0] if val else '—'}…）", flush=True)
    if not train:
        raise ValueError(f"purge 後訓練集為空（before={len(before)}, n_cut={n_cut}）")
    return train, val


def purged_walk_forward(
    all_dates: np.ndarray | pd.Series,
    horizon: int,
    embargo_days: int = 20,
    test_window_days: int = 126,
    step_days: int = 63,
    min_train_days: int = 756,
    first_test_start=None,
) -> list[Split]:
    """
    Expanding-window walk-forward，每個 fold 都套 purge + embargo。

    預設值對應現行 `config.py` 的設定，但單位改成**交易日**：
      test_window_days 126 ≈ 6 個月（WF_TEST_WINDOW_MONTHS）
      step_days         63 ≈ 3 個月（WF_STEP_MONTHS）
      min_train_days   756 ≈ 3 年（WF_MIN_TRAIN_YEARS）

    改用交易日而非日曆月，是為了讓 purge 的天數（也是交易日）與切分邊界同單位——
    混用會讓 purge 在長假多的年份少砍幾天，形成難以察覺的不一致。
    """
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(all_dates).unique())).sort_values()
    n = len(d)
    cut = horizon + embargo_days

    start_i = min_train_days + cut
    if first_test_start is not None:
        start_i = max(start_i, int(np.searchsorted(d, pd.Timestamp(first_test_start))))

    splits: list[Split] = []
    fid = 0
    i = start_i
    while i + test_window_days <= n:
        test_dates = d[i:i + test_window_days]
        before = d[:i]
        keep = before[:-cut] if cut > 0 else before
        if len(keep) >= min_train_days:
            splits.append(Split(fid, keep.to_numpy(), test_dates.to_numpy(),
                                min(horizon, len(before) - len(keep)),
                                max(len(before) - len(keep) - horizon, 0)))
            fid += 1
        i += step_days
    return splits


def assert_no_leakage(split: Split, horizon: int,
                      calendar: np.ndarray | pd.Series) -> None:
    """
    斷言訓練集中沒有任何樣本的 label 窗口伸進測試區間。

    這是驗收的硬性檢查——不是印個訊息就算數。之所以要獨立成一個函式，
    是因為「切分看起來對」與「切分真的沒洩漏」是兩件事，
    而洩漏不會有任何執行期徵兆，只會讓 IC 變好看。

    ⚠️ `calendar` 必須是**完整的交易日曆**，不可以只傳 train ∪ test。
       初版就是拿 train+test 拼起來當日曆，結果被 purge/embargo 砍掉的那段
       在陣列裡不存在，`+horizon` 個位置直接跨過缺口落進測試期，
       把一個正確的切分誤報成洩漏。斷言本身抓到了這個錯，但方向是反的——
       這正好說明日曆的定義有多容易寫錯。
    """
    if not len(split.train_dates) or not len(split.test_dates):
        return
    cal = pd.DatetimeIndex(pd.to_datetime(pd.Series(calendar).unique())).sort_values()
    last_train = pd.Timestamp(split.train_dates[-1])
    test_start = pd.Timestamp(split.test_dates[0])

    last_i = int(np.searchsorted(cal, last_train))
    label_end = cal[min(last_i + horizon, len(cal) - 1)]

    assert label_end < test_start, (
        f"fold {split.fold_id} 洩漏：訓練集最後一天 {last_train.date()} 的 "
        f"{horizon} 日 label 延伸到 {label_end.date()}，"
        f"已進入測試區間（起 {test_start.date()}）"
    )


def summarize(splits: list[Split], horizon: int, embargo_days: int,
              calendar: np.ndarray | pd.Series) -> None:
    """印出切分摘要 + 逐 fold 洩漏檢查結果（規則 7：數值要看得到）。"""
    print(f"[purged WF] horizon={horizon} 交易日｜embargo={embargo_days} 交易日｜"
          f"共 {len(splits)} 個 fold")
    for s in splits:
        print("  " + s.describe())
        assert_no_leakage(s, horizon, calendar)
    if splits:
        tot_cut = sum(s.n_purged + s.n_embargoed for s in splits)
        avg_tr = np.mean([len(s.train_dates) for s in splits])
        print(f"  合計剔除 {tot_cut:,} 個訓練日（每 fold {tot_cut / len(splits):.0f} 日）｜"
              f"平均訓練集 {avg_tr:,.0f} 日")
    print("  ✓ 全部 fold 通過無洩漏斷言")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    _V6 = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_V6))
    from marketmamba.config import PROCESSED_DIR

    px = pd.read_parquet(Path(PROCESSED_DIR) / "prices_raw.parquet", columns=["Date"])
    dates = pd.to_datetime(px["Date"]).unique()
    print(f"交易日曆：{len(pd.unique(dates)):,} 日\n")

    print("── 單一切分（Phase 3 harness 同窗）──")
    for h in (5, 60):
        s = purged_split(dates, "2024-01-01", "2026-06-02", horizon=h,
                         embargo_days=20, train_start="2013-01-01")
        print("  horizon", h, "→", s.describe())
        assert_no_leakage(s, h, dates)
    print()
    print("── 對照組：不做 purge/embargo（現行 walk_forward.py 的行為）──")
    naive = purged_split(dates, "2024-01-01", "2026-06-02", horizon=0,
                         embargo_days=0, train_start="2013-01-01")
    try:
        assert_no_leakage(naive, 60, dates)
        print("  ⚠ 未偵測到洩漏（不預期）")
    except AssertionError as e:
        print(f"  ✓ 斷言正確抓到現行切分的洩漏：\n    {e}")
    print()
    print("── Walk-forward（horizon=60）──")
    summarize(purged_walk_forward(dates, horizon=60, embargo_days=20,
                                  first_test_start="2016-01-01"),
              horizon=60, embargo_days=20, calendar=dates)
