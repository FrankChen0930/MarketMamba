"""
baseline_rnn.py — 方向二 Step 4：階 3 序列模型 baseline（GRU）
================================================================
協定：docs/baseline-experiment-protocol-draft-2026-07-11.md（v1.0 凍結）
  - 階 3 作用：隔離「Mamba 贏是架構本身，還是任何序列模型皆可」
  - 與階 1/2 完全同 harness：同 universe/label(rank)/切分/成本模型
  - 2026-07-14 使用者拍板兩決定：
    ① 循環單元 = GRU（LSTM 同級替代，參數少、3060 更快；擇一控制多重測試）
    ② 回看窗 = 60（對齊 5d 對照對象 v6_short 的 window 60；Phase 1 deep
      supervision 已實證長窗對 5d 冗餘。偏離協定 §3 字面的 (N,252,59)，
      理由記錄於此與結果報告）
  - loss = MSE on rank label（同階 1/2 純 L2 慣例；v6_short 另有 listnet_5d，
    誠實註記差異、不跟進以免引入新自由度）

資料層：直接重用 baseline_base_59d.parquet（clean 後 59 維 + rank label +
  eligible），每股按日連續排列後以「列」為單位切 (60, 59) 視窗——與扁平特徵
  的 lag 同一慣例（列 shift ≈ 交易日 shift）。eligible（≥202 天歷史）保證
  60 步視窗無 padding；另加「載入後股內 cumcount ≥ 60」防護（停牌斷檔邊角）。

超參數（誠實記錄）：小網格 hidden {64,128} × lr {1e-3, 3e-4}（4 組），
  val 日 IC 選組 + early stopping（patience 3、max 15 epochs）；選定後以
  best epoch 數在 full-train（fit+val）重訓，test 全程未參與。

執行（WSL2 mamba_env、RTX 3060）：
  wsl -d Ubuntu -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && \
    python -u V6/experimental/baseline_rnn.py 2>&1 | tee V6/experimental/result/rnn_run.log"
選項：--skip-20d 只跑 5d；--batch 4096 若 GPU OOM；--cell lstm 換 LSTM
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
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import experimental.baseline_common as bc                      # noqa: E402
from experimental.baseline_common import (                     # noqa: E402
    BASE_PATH, PROTOCOL, daily_spearman_ic, ic_summary, portfolio_backtest,
)
from experimental.baseline_ic_diagnosis import _grouped_ic, _liquidity_lut  # noqa: E402
from experimental.baseline_common import FEATURE_COLS          # noqa: E402

RESULT_PATH = _THIS_DIR / "result" / "baseline_rnn_result.json"
WINDOW = 60
TRAIN_STRIDE = 2
LOAD_FROM = "2011-01-01"            # train 2012 起，前留 ≥60 列視窗緩衝
GRID = [{"hidden": h, "lr": lr} for h in (64, 128) for lr in (1e-3, 3e-4)]
MAX_EPOCHS = 15
PATIENCE = 3
SEED = 20260713


# ============================================================
# 資料層
# ============================================================
class Panel:
    """每股按 (stock_id, Date) 連續排列的全域陣列 + (股,日) 樣本索引。"""

    def __init__(self):
        cols = ["Date", "stock_id", "eligible", "rank_5d", "rank_20d",
                "Alpha_5d", "Alpha_20d"] + FEATURE_COLS
        df = pd.read_parquet(BASE_PATH, columns=cols,
                             filters=[("Date", ">=", pd.Timestamp(LOAD_FROM)),
                                      ("Date", "<=", pd.Timestamp(PROTOCOL["TEST_END"]))])
        df = df.sort_values(["stock_id", "Date"], kind="mergesort").reset_index(drop=True)
        self.F = np.ascontiguousarray(df[FEATURE_COLS].to_numpy(np.float32))
        self.dates = df["Date"].to_numpy()
        self.sids = df["stock_id"].to_numpy()
        self.eligible = df["eligible"].to_numpy()
        self.labels = {5: df["rank_5d"].to_numpy(np.float32),
                       20: df["rank_20d"].to_numpy(np.float32)}
        self.alphas = {5: df["Alpha_5d"].to_numpy(np.float32),
                       20: df["Alpha_20d"].to_numpy(np.float32)}
        cum = df.groupby("stock_id", sort=False).cumcount().to_numpy()
        self.window_ok = cum >= (WINDOW - 1)           # 載入後股內至少 60 列（防停牌斷檔邊角）
        del df
        gc.collect()
        print(f"[panel] {len(self.F):,} 列 × {self.F.shape[1]} 維 | "
              f"{pd.Series(self.sids).nunique()} 支 | window_ok {self.window_ok.mean():.1%}", flush=True)

    def sample_rows(self, date_from: str, date_to: str, horizon: int,
                    day_stride: int = 1, require_label: bool = True) -> np.ndarray:
        m = (self.dates >= np.datetime64(date_from)) & (self.dates <= np.datetime64(date_to))
        m &= self.eligible & self.window_ok
        if require_label:
            m &= ~np.isnan(self.labels[horizon])
        if day_stride > 1:
            days = np.sort(pd.unique(self.dates[m]))
            m &= pd.Series(self.dates).isin(set(days[::day_stride])).to_numpy()
        return np.flatnonzero(m)

    def gather(self, rows: np.ndarray) -> np.ndarray:
        """(B,) 列索引 → (B, 60, 59)。排序保證前 59 列屬同一股票（window_ok 已過濾）。"""
        offs = rows[:, None] + np.arange(-(WINDOW - 1), 1)[None, :]
        return self.F[offs]                            # fancy index → 直接得 (B, 60, 59)


# ============================================================
# 模型
# ============================================================
class RNNReg(nn.Module):
    def __init__(self, cell: str, hidden: int, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(len(FEATURE_COLS), hidden, num_layers=layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def _predict(model, panel, rows, device, batch: int) -> np.ndarray:
    model.eval()
    preds = np.empty(len(rows), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            xb = torch.from_numpy(panel.gather(rows[i:i + batch])).to(device, non_blocking=True)
            preds[i:i + batch] = model(xb).float().cpu().numpy()
    return preds


def _train_one(panel, fit_rows, val_rows, horizon, cfg, device, batch,
               max_epochs=MAX_EPOCHS, patience=PATIENCE, tag="") -> dict:
    """訓練一組設定，回傳 {best_val_ic, best_epoch, state_dict, history}。
    val_rows=None 時為 full-train 重訓（跑滿 max_epochs、不 early stop）。"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = RNNReg(ARGS.cell, cfg["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    y = panel.labels[horizon]
    best = {"val_ic": -9.0, "epoch": 0, "state": None}
    history, bad = [], 0
    rng = np.random.default_rng(SEED)
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(fit_rows))
        tot, nb = 0.0, 0
        for i in range(0, len(order), batch):
            rows = fit_rows[order[i:i + batch]]
            xb = torch.from_numpy(panel.gather(rows)).to(device, non_blocking=True)
            yb = torch.from_numpy(y[rows]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        rec = {"epoch": ep, "train_loss": round(tot / nb, 5), "sec": round(time.time() - t0)}
        if val_rows is not None:
            p = _predict(model, panel, val_rows, device, batch)
            ic = daily_spearman_ic(panel.dates[val_rows], p, panel.alphas[horizon][val_rows])
            rec["val_ic"] = round(float(ic.mean()), 4)
            print(f"[{tag}] ep{ep:>2} train_loss {rec['train_loss']:.5f} "
                  f"val IC {rec['val_ic']:+.4f} ({rec['sec']}s)", flush=True)
            if rec["val_ic"] > best["val_ic"]:
                best = {"val_ic": rec["val_ic"], "epoch": ep,
                        "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"[{tag}] early stop @ ep{ep}（best ep{best['epoch']} "
                          f"val IC {best['val_ic']:+.4f}）", flush=True)
                    history.append(rec)
                    break
        else:
            print(f"[{tag}] ep{ep:>2} train_loss {rec['train_loss']:.5f} ({rec['sec']}s)", flush=True)
        history.append(rec)
    if val_rows is None:
        best = {"val_ic": None, "epoch": max_epochs,
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
    best["history"] = history
    del model
    torch.cuda.empty_cache()
    return best


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} | torch {torch.__version__} | cell={ARGS.cell} | "
          f"window={WINDOW} | batch={ARGS.batch}", flush=True)
    if device == "cpu":
        print("⚠️ 沒抓到 CUDA——請確認在 WSL2 mamba_env 執行；CPU 跑會慢一個數量級", flush=True)
    P = PROTOCOL
    panel = Panel()

    # 切分（同階 1/2：train 尾端 15% 交易日為 val）
    tr_rows_all = panel.sample_rows(P["TRAIN_START"], P["TRAIN_END"], 5,
                                    day_stride=TRAIN_STRIDE, require_label=False)
    train_days = np.sort(pd.unique(panel.dates[tr_rows_all]))
    val_start = train_days[int(len(train_days) * (1 - P["VAL_RATIO"]))]
    print(f"[split] train 天數 {len(train_days)}（val 自 {pd.Timestamp(val_start).date()} 起）", flush=True)

    results = {"models": {}}
    horizons = [5] + ([] if ARGS.skip_20d else [20])
    for h in horizons:
        rows = panel.sample_rows(P["TRAIN_START"], P["TRAIN_END"], h, day_stride=TRAIN_STRIDE)
        fit_rows = rows[panel.dates[rows] < val_start]
        val_rows = rows[panel.dates[rows] >= val_start]
        te_rows = panel.sample_rows(P["TEST_START"], P["TEST_END"], h,
                                    day_stride=1, require_label=False)
        print(f"\n===== horizon {h}d =====", flush=True)
        print(f"[rows] fit {len(fit_rows):,} | val {len(val_rows):,} | test {len(te_rows):,}", flush=True)

        # ── 網格 ──
        sweep = []
        for gi, cfg in enumerate(GRID):
            tag = f"{h}d g{gi+1}/{len(GRID)} h{cfg['hidden']} lr{cfg['lr']:g}"
            r = _train_one(panel, fit_rows, val_rows, h, cfg, device, ARGS.batch, tag=tag)
            sweep.append({**cfg, "best_val_ic": r["val_ic"], "best_epoch": r["epoch"],
                          "history": r["history"]})
        best_i = int(np.argmax([s["best_val_ic"] for s in sweep]))
        bcfg = sweep[best_i]
        print(f"[{h}d] best：hidden={bcfg['hidden']} lr={bcfg['lr']:g} "
              f"ep{bcfg['best_epoch']}（val IC {bcfg['best_val_ic']:+.4f}）", flush=True)

        # ── full-train 重訓（fit+val、跑 best epoch 數）──
        full = _train_one(panel, rows, None, h, {"hidden": bcfg["hidden"], "lr": bcfg["lr"]},
                          device, ARGS.batch, max_epochs=bcfg["best_epoch"], tag=f"{h}d full")
        model = RNNReg(ARGS.cell, bcfg["hidden"]).to(device)
        model.load_state_dict(full["state"])

        # ── test 評估 ──
        scores = _predict(model, panel, te_rows, device, ARGS.batch)
        okt = ~np.isnan(panel.alphas[h][te_rows])
        ic = daily_spearman_ic(panel.dates[te_rows][okt], scores[okt],
                               panel.alphas[h][te_rows][okt])
        res = {"cell": ARGS.cell, "window": WINDOW,
               "best_params": {"hidden": bcfg["hidden"], "lr": bcfg["lr"],
                               "epochs": bcfg["best_epoch"]},
               "grid_sweep": [{k: s[k] for k in ("hidden", "lr", "best_val_ic", "best_epoch")}
                              for s in sweep],
               "epoch_history_best_config": bcfg["history"],
               "test_ic": ic_summary(ic, horizon=h)}
        s = res["test_ic"]
        print(f"[{h}d] test：mean IC {s['mean_ic']:+.4f} | ICIR {s['icir']} | "
              f"IC>0 {s['pct_pos']:.0%} | t(NW) {s['t_newey_west']} | {s['n_days']} 天", flush=True)

        # 分層 IC（排查後標準輸出）
        df = pd.DataFrame({"Date": panel.dates[te_rows], "stock_id": panel.sids[te_rows],
                           "s": scores, "r": panel.alphas[h][te_rows]}).dropna(subset=["r"])
        liq = _liquidity_lut(P["TEST_START"], P["TEST_END"])
        df = df.merge(liq, on=["Date", "stock_id"], how="left")
        df["liq_bin"] = df.groupby("Date")["adv20"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 3, labels=["L1_小量", "L2_中量", "L3_大量"]))
        res["test_ic_by_liquidity"] = _grouped_ic(df.dropna(subset=["liq_bin"]), "liq_bin")
        print(f"[{h}d] IC by 流動性：" + " | ".join(
            f"{k}: {v['mean_ic']:+.4f}" for k, v in sorted(res["test_ic_by_liquidity"].items())), flush=True)

        # 組合層（5d only）：成本 ×1/×2
        if h == 5:
            print("[5d] 組合回測（Top50 等權、5 日再平衡）...", flush=True)
            res["test_portfolio"] = portfolio_backtest(panel.dates[te_rows],
                                                       panel.sids[te_rows], scores)
            cb, cs = P["COST_BUY"], P["COST_SELL"]
            bc.PROTOCOL["COST_BUY"], bc.PROTOCOL["COST_SELL"] = cb * 2, cs * 2
            res["test_portfolio_cost_x2"] = portfolio_backtest(panel.dates[te_rows],
                                                               panel.sids[te_rows], scores)
            bc.PROTOCOL["COST_BUY"], bc.PROTOCOL["COST_SELL"] = cb, cs
            pf = res["test_portfolio"]
            print(f"[5d] 組合：年化 {pf['ann_return']:+.1%} | Sharpe {pf['ann_sharpe']} | "
                  f"MDD {pf['max_drawdown']:.1%} | 換手 {pf['avg_turnover_per_rebalance']:.0%}/次"
                  + (f" | 對 TWII 超額 {pf['excess_vs_twii']:+.1%}" if "excess_vs_twii" in pf else ""),
                  flush=True)
            print(f"[5d] 成本×2：年化 {res['test_portfolio_cost_x2']['ann_return']:+.1%}", flush=True)
            torch.save(full["state"], _THIS_DIR / "result" / f"{ARGS.cell}_5d.pt")

        results["models"][f"{h}d"] = res
        del model, scores
        torch.cuda.empty_cache()
        gc.collect()

    results["protocol"] = {k: P[k] for k in ("TRAIN_START", "TRAIN_END", "TEST_START", "TEST_END",
                                             "VAL_RATIO", "TOP_N", "REBALANCE_DAYS",
                                             "COST_BUY", "COST_SELL")}
    results["notes"] = [
        f"cell={ARGS.cell}、window={WINDOW}（2026-07-14 使用者拍板：GRU + 60，"
        "偏離協定 §3 字面 252——對齊 v6_short 的 window 60、Phase 1 DS 已證長窗對 5d 冗餘）",
        "loss = 純 MSE on rank label（同階 1/2；v6_short 另有 listnet_5d，此處不跟進）",
        f"訓練列 day_stride={TRAIN_STRIDE}（同階 1/2）；test IC 用全部交易日",
        "網格 4 組（hidden×lr）val 日 IC 選組、early stopping patience 3；full-train 以 best epoch 重訓",
        "存活者偏差 D3 未修復：絕對數字偏高，四階相對比較公平（協定 §2）；引用需配分層 IC",
    ]
    results["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 結果已存 {RESULT_PATH}（總耗時 {(time.time()-t0)/60:.1f} 分）", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("階 3（GRU）vs 已知數字（5d test IC，全市場/高流動組）", flush=True)
    print("=" * 72, flush=True)
    print("  Mamba v6_short（同 harness） +0.0870 / —", flush=True)
    print("  Ridge 300 維                +0.1015 / +0.0705", flush=True)
    print("  GBDT  300 維                +0.1098 / +0.0802", flush=True)
    s = results["models"]["5d"]["test_ic"]
    lg = results["models"]["5d"]["test_ic_by_liquidity"].get("L3_大量", {})
    print(f"  GRU   60×59 序列            {s['mean_ic']:+.4f} / {lg.get('mean_ic', float('nan')):+.4f}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-20d", action="store_true")
    ap.add_argument("--batch", type=int, default=8192, help="GPU OOM 時降 4096")
    ap.add_argument("--cell", choices=["gru", "lstm"], default="gru")
    ARGS = ap.parse_args()
    main()
