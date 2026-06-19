"""
run_dual_inference.py — Phase 2 步驟 4：雙模型並行推論（隔離，不碰 V6.1）
==========================================================================
載入 v6_short.pt(單尺度 5d/10d) + v6_trend.pt(多尺度 20d/60d)，對最新交易日
cross-section 做 MC-Dropout 推論，輸出 df_short.csv / df_trend.csv。

⚠️ 輸出是 **rank-score 語意**（模型用 cross-section rank 目標訓練）：分數越高=預測
   當日排名越前面，直接拿來選股；不是報酬率。SQ = Score / Uncertainty（風險調整後排名）。

⚠️ 完全獨立於 V6.1：本檔在自己的 process 內把 config 切成 59 維（V6.1 的 56 維每日推論
   是另一個 process，互不影響）；輸出獨立檔名，不覆蓋 df_kelly.csv；失敗也不影響線上。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── 1) 先把 config 切 59 維（必須在 import 任何讀 FEATURE_COLS 的模組之前）──
import marketmamba.config as cfg
_RS = ["RS_5d", "RS_20d", "RS_60d"]
if not all(r in cfg.FEATURE_GROUPS["price_momentum"] for r in _RS):
    cfg.FEATURE_GROUPS["price_momentum"] = cfg.FEATURE_GROUPS["price_momentum"] + _RS
cfg.INPUT_DIM = 59
cfg.FEATURE_COLS = (cfg.FEATURE_GROUPS["price_momentum"] + cfg.FEATURE_GROUPS["institutional_flow"]
                    + cfg.FEATURE_GROUPS["fundamentals"] + cfg.FEATURE_GROUPS["macro_environment"])
cfg.GROUP_DIMS = {k: len(v) for k, v in cfg.FEATURE_GROUPS.items()}
assert len(cfg.FEATURE_COLS) == 59, f"expected 59 features, got {len(cfg.FEATURE_COLS)}"

# ── 2) 切完才 import 重模組（它們會綁定 patched 後的 59 維設定）──
from marketmamba.config import MODELS_DIR
from marketmamba.data.merger import merge_all_data
from marketmamba.data.feature_engineer import build_features, clean_and_scale
from marketmamba.models.trainer import (
    TemporalCrossSectionDataset, build_kg_csr, get_batch_edges_csr, TrainingHistory,
)
from marketmamba.models.architecture import MarketMambaV6
from experimental.short_model import ShortModelV6

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_dual_inference")

RESULTS_DIR = MODELS_DIR.parent / "results"
N_MC        = 30
INFER_BATCH = 128


def _build_feature_df() -> pd.DataFrame:
    """載入快取 raw → 建 59 維特徵 → clean_and_scale(macro_norm='ts'，與雙模型訓練對齊)。"""
    data = merge_all_data()
    df = build_features(
        df_price=data["prices"], df_inst=data["inst"], df_margin=data["margin"],
        df_per=data["per"], df_securities=data["securities"], df_market_value=data["market_value"],
        df_daytrade=data["daytrade"], df_holdings=data["holdings"], df_rev=data["revenue"],
        df_fin=data["financials"], df_balance_sheet=data["balance_sheet"], df_cashflow=data["cashflow"],
        df_macro=data["macro"], df_futures_inst=data["futures_inst"], df_options_inst=data["options_inst"],
        df_dividend=data["dividend"], df_foreign_shareholding=data["foreign_shareholding"],
        df_fear_greed=data["fear_greed"], df_business_indicator=data["business_indicator"],
        df_fed_rate=data["fed_rate"],
    )
    df = clean_and_scale(df, macro_norm="ts")     # 雙模型用 ts 訓練 → 對齊（與 V6.1 的 "cross" 無關）
    df = df.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    return df


@torch.no_grad()
def _mc_infer(model, X, padding_mask, edge_index, edge_attr, device, multiscale, n_out, seed):
    """兩段式 MC-Dropout 推論（Mamba 分批 + 完整圖 GAT，與 V6.1 run_inference 一致）。
    回傳 (mean, std)，shape (N, n_out)。"""
    N = X.shape[0]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    pred_mc = torch.zeros(N_MC, N, n_out)
    model.train()    # 開 dropout 做 MC 抽樣
    for mc in range(N_MC):
        h_parts = []
        for bs in range(0, N, INFER_BATCH):
            x_b = X[bs: bs + INFER_BATCH].to(device)
            h_emb = model.embedding(x_b)
            if multiscale:
                pm_b = padding_mask[bs: bs + INFER_BATCH].to(device) if padding_mask is not None else None
                h_parts.append(model.encoder(h_emb, pm_b))
            else:
                h_parts.append(model.encoder(h_emb))          # 單尺度，無 padding_mask
        h_temporal = torch.cat(h_parts, dim=0)
        h_graph = model.graph_layer(h_temporal, edge_index, edge_attr)
        gate_w  = model.gate(torch.cat([h_temporal, h_graph], dim=-1))
        h_fused = model.norm_fuse(gate_w * h_temporal + (1 - gate_w) * h_graph)
        h_fused = model.dropout(h_fused)
        if multiscale:
            pred_mc[mc] = model.head(h_fused).cpu()                                   # (N,3)
        else:
            pred_mc[mc] = torch.cat([model.head_5d(h_fused), model.head_10d(h_fused)], dim=-1).cpu()  # (N,2)
    model.eval()
    return pred_mc.mean(dim=0).numpy(), pred_mc.std(dim=0).numpy()


def run_dual_inference(out_dir: Path | None = None, device_str: str | None = None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 資料 + 最新交易日 cross-section ──
    df = _build_feature_df()
    df["Date"] = pd.to_datetime(df["Date"])
    latest = df["Date"].max()
    latest_str = latest.strftime("%Y-%m-%d")
    logger.info(f"Dual inference date: {latest_str}")

    ds = TemporalCrossSectionDataset(df, [latest_str], mode="test", n_sample=None)
    if len(ds) == 0:
        raise ValueError(f"No valid cross-section for {latest_str}")
    X, _, valid_stocks, padding_mask = ds[0]
    N = X.shape[0]
    kg_csr, stock_to_idx = build_kg_csr()
    edge_index, edge_attr = get_batch_edges_csr(valid_stocks, kg_csr, stock_to_idx, device)
    seed = int(latest.strftime("%Y%m%d"))
    logger.info(f"[cross-section] N={N} | KG edges={edge_index.shape[1]:,} | MC={N_MC} seed={seed}")

    torch.serialization.add_safe_globals([TrainingHistory])

    # ── 短線模型（單尺度，5d/10d）──
    sck = torch.load(MODELS_DIR / "v6_short.pt", map_location=device, weights_only=False)
    scfg = sck.get("config", {"window": 60, "n_layers": 3})
    short = ShortModelV6(window=scfg.get("window", 60), n_layers=scfg.get("n_layers", 3))
    short.load_state_dict(sck["state_dict"])
    short.to(device)
    s_mean, s_std = _mc_infer(short, X, padding_mask, edge_index, edge_attr, device,
                              multiscale=False, n_out=2, seed=seed)
    df_short = pd.DataFrame({
        "stock_id": valid_stocks[:len(s_mean)], "Date": latest_str,
        "Score_5d": s_mean[:, 0], "Score_10d": s_mean[:, 1],
        "Unc_5d": np.clip(s_std[:, 0], 0, None), "Unc_10d": np.clip(s_std[:, 1], 0, None),
    })
    df_short["SQ_5d"] = df_short["Score_5d"] / (df_short["Unc_5d"] + 1e-6)
    df_short = df_short.sort_values("SQ_5d", ascending=False).reset_index(drop=True)
    df_short.to_csv(out_dir / "df_short.csv", index=False)
    logger.info(f"✅ df_short.csv（{len(df_short)} 支，依 SQ_5d 排序）→ {out_dir / 'df_short.csv'}")

    # ── 趨勢模型（多尺度，20d/60d）──
    tck = torch.load(MODELS_DIR / "v6_trend.pt", map_location=device, weights_only=False)
    trend = MarketMambaV6()
    trend.load_state_dict(tck["state_dict"])
    trend.to(device)
    t_mean, t_std = _mc_infer(trend, X, padding_mask, edge_index, edge_attr, device,
                              multiscale=True, n_out=3, seed=seed)
    df_trend = pd.DataFrame({
        "stock_id": valid_stocks[:len(t_mean)], "Date": latest_str,
        "Score_20d": t_mean[:, 1], "Score_60d": t_mean[:, 2],
        "Unc_20d": np.clip(t_std[:, 1], 0, None), "Unc_60d": np.clip(t_std[:, 2], 0, None),
    })
    df_trend["SQ_20d"] = df_trend["Score_20d"] / (df_trend["Unc_20d"] + 1e-6)
    df_trend = df_trend.sort_values("SQ_20d", ascending=False).reset_index(drop=True)
    df_trend.to_csv(out_dir / "df_trend.csv", index=False)
    logger.info(f"✅ df_trend.csv（{len(df_trend)} 支，依 SQ_20d 排序）→ {out_dir / 'df_trend.csv'}")

    return df_short, df_trend


if __name__ == "__main__":
    run_dual_inference()
