"""
MarketMamba V6 — Inference Module
====================================
Standalone inference entrypoint for:
  - Daily inference (called by run_daily_inference.py)
  - One-off batch inference (for backtesting, research)
  - Colab inference verification after training

This module is GPU-agnostic: works on CUDA (RTX 3060 local / A100 Colab) or CPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from marketmamba.config import (
    FEATURE_COLS,
    MODELS_DIR,
    PRED_HORIZONS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEQ_LEN,
)

logger = logging.getLogger(__name__)


# ============================================================
# Model Loader
# ============================================================

def load_model(
    checkpoint_path: Path | None = None,
    device_str:      str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.nn.Module, dict]:
    """
    Load a V6 checkpoint and return (model, metadata).

    Args:
        checkpoint_path : path to .pt file; defaults to MODELS_DIR/v6_best.pt
        device_str      : 'cuda' or 'cpu'

    Returns:
        (model, ckpt_meta)
        model      : MarketMambaV6 with loaded weights, in eval mode
        ckpt_meta  : dict with epoch, val_loss, val_ic, etc.
    """
    from marketmamba.models.architecture import MarketMambaV6

    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / "v6_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train V6 first using notebooks/V6_Training.py on Colab."
        )

    device = torch.device(device_str)
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = MarketMambaV6()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    meta = {
        "epoch":    ckpt.get("epoch", "?"),
        "val_loss": ckpt.get("val_loss", float("nan")),
        "val_ic":   ckpt.get("val_ic",  float("nan")),
        "path":     str(checkpoint_path),
    }
    logger.info(
        f"Model loaded: {checkpoint_path.name} "
        f"(epoch={meta['epoch']}, val_loss={meta['val_loss']:.5f}, val_ic={meta['val_ic']:.4f})"
    )
    return model, meta


# ============================================================
# Core Inference (one cross-section)
# ============================================================

def infer_cross_section(
    model:       torch.nn.Module,
    X:           torch.Tensor,    # (N, SEQ_LEN, INPUT_DIM)
    edge_index:  torch.Tensor,    # (2, E)
    edge_attr:   torch.Tensor,    # (E,)
    n_mc:        int = 30,        # MC-Dropout samples for uncertainty
    device:      torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MC-Dropout inference on one cross-section.

    Args:
        model       : loaded MarketMambaV6
        X           : input tensor
        edge_index  : graph connectivity
        edge_attr   : edge weights
        n_mc        : number of stochastic forward passes

    Returns:
        pred_mean : (N, 3) — [pred_5d, pred_20d, pred_60d]
        pred_std  : (N, 3) — uncertainty estimates
    """
    if device is None:
        device = next(model.parameters()).device

    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.to(device)

    model.train()   # Enable Dropout for MC sampling
    preds = []
    with torch.no_grad():
        for _ in range(n_mc):
            p = model(X, edge_index, edge_attr)   # (N, 3)
            preds.append(p.cpu())

    model.eval()
    stack = torch.stack(preds, dim=0)   # (n_mc, N, 3)
    return stack.mean(dim=0).numpy(), stack.std(dim=0).numpy()


# ============================================================
# Full Inference Pipeline
# ============================================================

def run_inference(
    df:              pd.DataFrame,
    checkpoint_path: Optional[Path] = None,
    device_str:      str = "cuda" if torch.cuda.is_available() else "cpu",
    date_str:        Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run V6 inference on the most recent (or specified) trading day.

    Args:
        df              : complete feature DataFrame
        checkpoint_path : model checkpoint (.pt file)
        device_str      : 'cuda' or 'cpu'
        date_str        : specific date to infer ('YYYY-MM-DD'), default: latest

    Returns:
        df_kelly : ranked stock predictions with weights
        df_traj  : multi-horizon trajectories
    """
    from marketmamba.models.trainer import TemporalCrossSectionDataset, load_kg_edges

    device = torch.device(device_str)
    model, meta = load_model(checkpoint_path, device_str)

    # Select target date
    df["Date"] = pd.to_datetime(df["Date"])
    if date_str:
        target_date = pd.Timestamp(date_str)
    else:
        target_date = df["Date"].max()
    target_str = target_date.strftime("%Y-%m-%d")

    # Build dataset for this single day
    test_ds = TemporalCrossSectionDataset(df, [target_str], mode="test", n_sample=None)
    if len(test_ds) == 0:
        raise ValueError(f"No valid cross-section for {target_str}")

    # KG edges
    stock_ids = df["stock_id"].unique().tolist()
    edge_index, edge_attr = load_kg_edges(stock_ids, device)

    # Inference
    X, _ = test_ds[0]
    pred_mean, pred_std = infer_cross_section(
        model, X, edge_index, edge_attr, device=device
    )

    # Stock IDs for this cross-section
    mask = df["Date"] == target_date
    stocks_today = df.loc[mask, "stock_id"].values[:len(pred_mean)]

    # Build df_kelly
    df_kelly = _build_kelly_frame(stocks_today, pred_mean, pred_std, target_str)

    # Attach price data (Close, Volume) for liquidity filter
    price_cols = ["stock_id", "Close", "Volume"]
    df_price_today = df.loc[mask, [c for c in price_cols if c in df.columns]].copy()
    df_kelly = df_kelly.merge(
        df_price_today.rename(columns={"stock_id": "Ticker"}),
        on="Ticker", how="left",
    )
    df_kelly["Turnover_5D"] = df_kelly.get("Volume", 0) * df_kelly.get("Close", 0)

    # Slippage & confidence
    df_kelly = _add_derived_columns(df_kelly)

    # Build df_traj
    df_traj = _build_traj_frame(stocks_today, pred_mean, pred_std, target_str)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_kelly.to_csv(RESULTS_DIR / "df_kelly.csv", index=False, encoding="utf-8-sig")
    df_traj.to_csv(RESULTS_DIR  / "df_traj.csv",  index=False, encoding="utf-8-sig")

    return df_kelly, df_traj


# ============================================================
# Output Frame Builders
# ============================================================

def _build_kelly_frame(
    stock_ids: np.ndarray,
    pred_mean: np.ndarray,
    pred_std:  np.ndarray,
    date_str:  str,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "Ticker":        stock_ids,
        "Date":          date_str,
        "Exp_Alpha_5d":  pred_mean[:, 0],
        "Exp_Alpha_20d": pred_mean[:, 1],
        "Exp_Alpha_60d": pred_mean[:, 2],
        "Uncertainty_5d":  pred_std[:, 0],
        "Uncertainty_20d": pred_std[:, 1],
        "Uncertainty_60d": pred_std[:, 2],
    })
    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Slippage estimate by liquidity tier
    HIGH_LIQ = 5e8
    MID_LIQ  = 1e8
    df["Slippage"] = 0.008
    df.loc[df.get("Turnover_5D", 0) > MID_LIQ,  "Slippage"] = 0.004
    df.loc[df.get("Turnover_5D", 0) > HIGH_LIQ, "Slippage"] = 0.002

    df["Net_Alpha_20d"] = (df["Exp_Alpha_20d"] - df["Slippage"] - 0.003).clip(lower=-1.0)

    # Liquidity hard filter
    MIN_TURNOVER = 1e7
    df.loc[df.get("Turnover_5D", pd.Series(dtype=float)).fillna(0) < MIN_TURNOVER, "Net_Alpha_20d"] = -999.0

    # Sharpe score
    df["Signal_Quality"] = (
        df["Net_Alpha_20d"] / (df["Uncertainty_20d"].clip(lower=1e-6))
    ).clip(lower=-10.0, upper=10.0)

    # Confidence label
    df["Confidence"] = pd.cut(
        df["Uncertainty_20d"],
        bins=[0, 0.02, 0.05, 1.0],
        labels=["高信心", "中信心", "低信心"],
        right=False,
    )

    # Suggested weight (proportional to positive Sharpe)
    pos = df["Signal_Quality"].clip(lower=0)
    total = pos.sum()
    df["Suggested_Weight"] = (pos / (total + 1e-9)).round(4)

    # Sort by Sharpe
    df = df.sort_values("Signal_Quality", ascending=False).reset_index(drop=True)
    return df


def _build_traj_frame(
    stock_ids: np.ndarray,
    pred_mean: np.ndarray,
    pred_std:  np.ndarray,
    date_str:  str,
) -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker":           stock_ids,
        "Date":             date_str,
        "Pred_5d":          pred_mean[:, 0],
        "Pred_20d":         pred_mean[:, 1],
        "Pred_60d":         pred_mean[:, 2],
        "Uncertainty_5d":   pred_std[:, 0],
        "Uncertainty_20d":  pred_std[:, 1],
        "Uncertainty_60d":  pred_std[:, 2],
    })
