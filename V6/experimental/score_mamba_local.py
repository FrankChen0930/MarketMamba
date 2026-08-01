"""
score_mamba_local.py — 在本機 WSL 產生 Mamba 三組的組合層分數
==============================================================
用途：`portfolio_lab.py` 需要每個模型在 test 窗（582 天）的逐股分數。
Ridge/GBDT 本機可算，Mamba 需要 GPU 前向——**但不需要重訓**，
所以把 Colab 的 checkpoint 載下來、在 WSL 的 RTX 3060 上前向一次即可。

⚠️ 一個必須先講清楚的差異
--------------------------
F6 訓練/驗證用的特徵矩陣是在 Colab 建的（**自 2005 起**、59 維、無 Avail 旗標）。
本機沒有那一份，只有 `baseline_cache_v2/baseline_base_66d.parquet`（**自 2011 起**、
66 維 = 59 + 7 個 Avail 旗標）。兩者的 59 個特徵欄**建構方式完全相同**
（`fundamentals_v2=True`、`macro_norm="ts"`、同一套宇宙過濾），唯一的差別是：

    `macro_norm="ts"` 是 **expanding** z-score → 起點不同會讓 Group D 的 12 維數值不同。

這正是 `docs/f6-training-log-and-readout.md` §4.4 已經記錄的二階差異。
Group D 是每日橫斷面常數，但 Mamba 是非線性模型，**不能假設它免疫**。

→ 所以本檔**內建一個決定性的驗證**：把算出來的逐日 IC 拿去對
`kg_ablation_result.json` 裡 Colab 記錄的 `ic_by_day`。三組的 `eval_mean_ic_5d`
分別是 +0.0884 / +0.0939 / +0.0991。**對得上就代表本機矩陣可用；對不上就不可用，
必須改回 Colab 產生分數**（`portfolio_lab.py --colab`）。

用法（WSL）
-----------
    wsl -d Ubuntu -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && \\
      conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && \\
      python V6/experimental/score_mamba_local.py --arms no_gat --max-days 40"

先用 `--max-days` 小樣本驗證 + 測速，過了再跑全部 582 天。
輸出：`V6/experimental/result/scores/{arm}.parquet`（欄位 Date / stock_id / score）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_V6 = Path(__file__).resolve().parent.parent
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))

# ── 59 維 config patch：必須在 import 任何 marketmamba.models.* 之前 ──────
# `architecture.py` 在 import 當下就把 GROUP_DIMS/INPUT_DIM 綁進
# `FactorGroupedEmbedding.__init__` 的預設參數（def 執行時求值）。
import marketmamba.config as cfg                                   # noqa: E402

_RS = ["RS_5d", "RS_20d", "RS_60d"]
if not all(r in cfg.FEATURE_GROUPS["price_momentum"] for r in _RS):
    cfg.FEATURE_GROUPS["price_momentum"] = cfg.FEATURE_GROUPS["price_momentum"] + _RS
cfg.INPUT_DIM = 59
cfg.FEATURE_COLS = (cfg.FEATURE_GROUPS["price_momentum"] + cfg.FEATURE_GROUPS["institutional_flow"]
                    + cfg.FEATURE_GROUPS["fundamentals"] + cfg.FEATURE_GROUPS["macro_environment"])
cfg.GROUP_DIMS = {k: len(v) for k, v in cfg.FEATURE_GROUPS.items()}
assert len(cfg.FEATURE_COLS) == 59, f"expected 59, got {len(cfg.FEATURE_COLS)}"
# RS 必須在 group A 的**末端**（位置 12–14）——放錯 proj_A 會吃到錯欄位且不會報錯
assert cfg.FEATURE_COLS[12:15] == _RS, f"RS 位置錯誤：{cfg.FEATURE_COLS[9:16]}"

from marketmamba.config import PROCESSED_DIR                        # noqa: E402

FEATURE_COLS = list(cfg.FEATURE_COLS)
BASE_MATRIX = PROCESSED_DIR / "baseline_cache_v2" / "baseline_base_66d.parquet"
CKPT_DIRS = [Path("/mnt/d/Downloads"), Path("D:/Downloads"), PROCESSED_DIR.parent / "models"]
RESULT_DIR = Path(__file__).resolve().parent / "result"
SCORE_DIR = RESULT_DIR / "scores"
KG_RESULT = Path("/mnt/d/Downloads/kg_ablation_result.json")

ARMS = {                       # arm -> (use_gat, KG 檔名)
    "no_gat": (False, None),
    "old_kg": (True, "knowledge_graph_cache.npz"),
    "v2_kg":  (True, "knowledge_graph_v2.npz"),
}
DROPOUT = 0.2                  # 與 kg_ablation 相同（eval 模式下不生效，但架構要一致）
HISTORY_START = "2022-01-01"   # val 起點前需 ≥252 個交易日；2022-01 給約 490 天緩衝


def _find_ckpt(arm: str) -> Path:
    name = f"v6_short_KG_{arm}.pt"
    for d in CKPT_DIRS:
        p = d / name
        if p.exists():
            return p
    raise SystemExit(f"❌ 找不到 {name}，找過：{[str(d) for d in CKPT_DIRS]}")


def _val_dates() -> list[str]:
    """直接沿用 Colab 記錄的 582 個 val 日期，避免本機重算切分時差一天。"""
    for p in (KG_RESULT, Path("D:/Downloads/kg_ablation_result.json")):
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            arm = next(iter(d["arms"].values()))
            return sorted(arm["ic_by_day"].keys())
    raise SystemExit(f"❌ 找不到 kg_ablation_result.json（需要它取 val 日期與驗證基準）")


def _recorded_ic() -> dict[str, dict[str, float]]:
    for p in (KG_RESULT, Path("D:/Downloads/kg_ablation_result.json")):
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            return {k: v["ic_by_day"] for k, v in d["arms"].items()}
    return {}


def load_df(val_dates: list[str]) -> pd.DataFrame:
    """從 v2 base matrix 取 59 維 + 標籤，組成 Dataset 需要的 df。"""
    t0 = time.time()
    cols = ["Date", "stock_id"] + FEATURE_COLS + ["Alpha_5d", "Alpha_20d"]
    filt = [("Date", ">=", pd.Timestamp(HISTORY_START)),
            ("Date", "<=", pd.Timestamp(val_dates[-1]))]
    df = pd.read_parquet(BASE_MATRIX, columns=cols, filters=filt)
    df["Date"] = pd.to_datetime(df["Date"])
    df["stock_id"] = df["stock_id"].astype(str)
    # ShortModelV6 是 2 頭 [5d, 10d]；矩陣沒有 Alpha_10d，用 Alpha_20d 佔位。
    # **評分只取 preds[:,0]，Y 完全不參與**，佔位不影響分數；
    # 只有下面的 IC 驗證會用到 Alpha_5d，那一欄是真值。
    df["Alpha_10d"] = df["Alpha_20d"]
    n_hist = df[df["Date"] < pd.Timestamp(val_dates[0])]["Date"].nunique()
    print(f"[score] 矩陣 {len(df):,} 列 × {df['stock_id'].nunique()} 支｜"
          f"{df['Date'].min().date()} → {df['Date'].max().date()}｜"
          f"val 起點前有 {n_hist} 個交易日（需 ≥252）｜{time.time()-t0:.0f}s", flush=True)
    assert n_hist >= 252, f"歷史不足（{n_hist} < 252），請把 HISTORY_START 往前調"
    return df


def score_arm(df, val_dates: list[str], arm: str, recorded: dict) -> dict:
    import torch
    import experimental.short_model as sm
    import marketmamba.models.trainer as T
    from marketmamba.config import AMP_ENABLED
    from marketmamba.models.trainer import (
        TemporalCrossSectionDataset, build_kg_csr, compute_ic, get_batch_edges_csr,
        make_dataloader,
    )

    use_gat, kg_file = ARMS[arm]
    ck = _find_ckpt(arm)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*66}\n[score] arm={arm} use_gat={use_gat} ckpt={ck.name} dev={dev}\n{'='*66}",
          flush=True)

    # dataset 的標籤欄：ShortModelV6 是 2 頭
    _orig = (T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS)
    T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = ["Alpha_5d", "Alpha_10d"], [5, 10], [5, 10]
    try:
        model = sm.ShortModelV6(use_gat=use_gat, dropout=DROPOUT).to(dev)
        state = torch.load(ck, map_location=dev)
        model.load_state_dict(state.get("state_dict", state))   # strict=True：載錯當場失敗
        model.eval()
        print(f"[score] checkpoint ep{state.get('epoch')} val_ic_5d={state.get('val_ic_5d')}"
              f"｜參數 {model.n_parameters:,}", flush=True)

        if kg_file:
            _o = T.KG_CACHE_PATH
            T.KG_CACHE_PATH = Path(PROCESSED_DIR) / kg_file
            try:
                kg, s2i = build_kg_csr()
            finally:
                T.KG_CACHE_PATH = _o
        else:
            kg, s2i = build_kg_csr()

        ds = TemporalCrossSectionDataset(df, val_dates, mode="val", n_sample=None)
        loader = make_dataloader(ds, shuffle=False)

        rows, ic_by_day = [], {}
        t0 = time.time()
        with torch.no_grad():
            for i, (X, Y, stks, _m) in enumerate(loader):
                if X.shape[0] <= 1:
                    continue
                d = str(ds.valid_dates[i])[:10]
                ei, ea = get_batch_edges_csr(stks, kg, s2i, dev)
                with torch.amp.autocast('cuda', enabled=AMP_ENABLED and dev.type == "cuda"):
                    p = model(X.to(dev), ei, ea)
                s = p[:, 0].float().cpu().numpy()
                ic_by_day[d] = float(compute_ic(s, Y[:, 0].float().cpu().numpy()))
                rows.append(pd.DataFrame({"Date": d, "stock_id": [str(x) for x in stks],
                                          "score": s.astype(np.float32)}))
                if (i + 1) % 100 == 0:
                    el = time.time() - t0
                    print(f"  {i+1}/{len(ds.valid_dates)}｜{el:.0f}s｜"
                          f"ETA {el/(i+1)*(len(ds.valid_dates)-i-1):.0f}s", flush=True)
    finally:
        T.TARGET_COLS, T.PRED_HORIZONS, cfg.PRED_HORIZONS = _orig

    out = pd.concat(rows, ignore_index=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    dst = SCORE_DIR / f"{arm}.parquet"
    out.to_parquet(dst, index=False)

    ics = np.array([v for v in ic_by_day.values() if np.isfinite(v)])
    mine = float(ics.mean())
    print(f"[score] {arm}：{len(out):,} 列 / {len(ic_by_day)} 天 → {dst.name}"
          f"｜本機 mean IC {mine:+.4f}（{(time.time()-t0)/60:.1f} 分）", flush=True)

    # ── 決定性驗證：對 Colab 記錄的逐日 IC ──
    ref = recorded.get(arm, {})
    v = {"arm": arm, "n_days": len(ic_by_day), "local_mean_ic": round(mine, 4)}
    common = sorted(set(ic_by_day) & set(ref))
    if common:
        a = np.array([ic_by_day[k] for k in common])
        b = np.array([ref[k] for k in common])
        v.update({
            "colab_mean_ic": round(float(b.mean()), 4),
            "diff": round(float(a.mean() - b.mean()), 4),
            "corr_daily": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "max_abs_daily_diff": round(float(np.abs(a - b).max()), 4),
            "n_common": len(common),
        })
        ok = abs(v["diff"]) < 0.005 and v["corr_daily"] > 0.95
        v["verdict"] = "✅ 一致，本機矩陣可用" if ok else "❌ 不一致 → 改用 Colab 產生分數"
        print(f"[驗證] {arm}｜本機 {mine:+.4f} vs Colab {v['colab_mean_ic']:+.4f}"
              f"（差 {v['diff']:+.4f}）｜逐日相關 {v['corr_daily']:.4f}"
              f"｜最大單日差 {v['max_abs_daily_diff']:.4f}｜n={len(common)}\n"
              f"[驗證] {v['verdict']}", flush=True)
    else:
        v["verdict"] = "⚠ 無 Colab 逐日 IC 可對照"
        print(f"[驗證] {v['verdict']}", flush=True)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--max-days", type=int, default=0,
                    help="只跑前 N 個 val 日（小樣本驗證＋測速用；0 = 全部 582 天）")
    a = ap.parse_args()

    vd = _val_dates()
    print(f"[score] Colab 記錄的 val 窗：{len(vd)} 天（{vd[0]} → {vd[-1]}）", flush=True)
    if a.max_days:
        vd = vd[:a.max_days]
        print(f"[score] ⚠ 小樣本模式：只跑前 {len(vd)} 天（分數檔不可用於正式掃描）", flush=True)

    df = load_df(vd)
    rec = _recorded_ic()
    ver = [score_arm(df, vd, arm, rec) for arm in a.arms]

    print(f"\n{'='*66}\n驗證彙總\n{'='*66}")
    for v in ver:
        print(f"  {v['arm']:8s} 本機 {v['local_mean_ic']:+.4f}"
              f"｜Colab {v.get('colab_mean_ic', float('nan')):+.4f}"
              f"｜差 {v.get('diff', float('nan')):+.4f}"
              f"｜逐日相關 {v.get('corr_daily', float('nan'))}｜{v['verdict']}")
    if a.max_days:
        print("\n⚠ 小樣本模式產生的 parquet 只有部分日期，跑正式掃描前要用全量重跑。")
    (RESULT_DIR / "score_mamba_validation.json").write_text(
        json.dumps(ver, indent=1, ensure_ascii=False), encoding="utf-8")
