"""
MarketMamba V6 — Daily Inference Pipeline
==========================================
Main entry point for local daily inference on RTX 3060 + WSL2.

Execution flow:
  [17:00] Step 1 — Hybrid data update (yfinance + TWSE direct)
  [17:05] Step 2 — Feature engineering for today's cross-section
  [17:10] Step 3 — Mamba+GAT inference → df_kelly.csv, df_traj.csv
  [17:15] Step 4 — LLM report generation → market_summary.json
  [17:20] Step 5 — Push results to GitHub → Streamlit auto-refresh

Run via Windows Task Scheduler:
  wsl -d Ubuntu -e bash -c "cd /mnt/d/Desktop/work/MarketMamba && conda run -n mamba_env python V6/run_daily_inference.py"
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make package importable when run as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

from marketmamba.config import (
    FEATURE_COLS,
    KG_CACHE_PATH,
    LLM_REPORT_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEQ_LEN,
)
from marketmamba.data.fetcher import run_daily_update, load_ticker_universe
from marketmamba.data.feature_engineer import build_features, clean_and_scale
from marketmamba.llm.report_generator import generate_market_report, build_market_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("V6.Inference")


# ============================================================
# Inference Core
# ============================================================

def run_inference(
    df:        pd.DataFrame,
    model_path: Path | None = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run V6 cross-sectional inference on the most recent trading day.

    Args:
        df         : feature DataFrame (all history, latest day will be used)
        model_path : .pt checkpoint; defaults to MODELS_DIR/v6_best.pt
        device_str : 'cuda' or 'cpu'

    Returns:
        df_kelly   : stock ranking with Alpha, Sharpe, Kelly weights
        df_traj    : multi-horizon predicted trajectories (5d, 20d, 60d)
    """
    from marketmamba.models.architecture import MarketMambaV6
    from marketmamba.models.trainer import TemporalCrossSectionDataset, load_kg_edges

    device = torch.device(device_str)

    # -- Load checkpoint --
    if model_path is None:
        model_path = MODELS_DIR / "v6_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Please train V6 first using V6/notebooks/V6_Training.py on Colab."
        )

    ckpt = torch.load(model_path, map_location=device)
    model = MarketMambaV6()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    logger.info(
        f"Loaded checkpoint: {model_path.name} "
        f"(epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss', 'N/A'):.5f})"
    )

    # -- Latest date's cross-section --
    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    latest_str  = latest_date.strftime("%Y-%m-%d")
    logger.info(f"Inference date: {latest_str}")

    # Build one-day dataset
    test_ds = TemporalCrossSectionDataset(
        df, [latest_str], mode="test", n_sample=None  # always use full cross-section
    )
    if len(test_ds) == 0:
        raise ValueError(f"No valid cross-section found for {latest_str}")

    # -- Load KG edges --
    edge_index, edge_attr = load_kg_edges(df["stock_id"].unique().tolist(), device)

    # -- Inference with MC-Dropout uncertainty --
    X, _ = test_ds[0]
    X = X.to(device)

    N_MC = 30   # MC-Dropout samples
    preds_mc = []
    model.train()   # enable dropout for MC
    with torch.no_grad():
        for _ in range(N_MC):
            p = model(X, edge_index, edge_attr)   # (N, 3)
            preds_mc.append(p.cpu())
    model.eval()

    preds_stack = torch.stack(preds_mc, dim=0)       # (N_MC, N, 3)
    pred_mean   = preds_stack.mean(dim=0).numpy()    # (N, 3)
    pred_std    = preds_stack.std(dim=0).numpy()     # (N, 3)  ← uncertainty

    # -- Get stock IDs for this cross-section --
    mask = df["Date"] == latest_date
    stocks_today = df.loc[mask, "stock_id"].values[:len(pred_mean)]

    # -- Build df_kelly --
    df_kelly = pd.DataFrame({
        "Ticker":        stocks_today,
        "Date":          latest_str,
        "Exp_Alpha_5d":  pred_mean[:, 0],
        "Exp_Alpha_20d": pred_mean[:, 1],   # primary
        "Exp_Alpha_60d": pred_mean[:, 2],
        "Uncertainty":   pred_std[:, 1],    # 20d uncertainty
    })

    # Liquidity filter: remove stocks below minimum daily volume
    df_today = df[mask][["stock_id", "Volume", "ATR_14"]].copy()
    df_kelly  = df_kelly.merge(
        df_today.rename(columns={"stock_id": "Ticker"}),
        on="Ticker", how="left",
    )
    # 1000萬台幣流動性門檻（FinMind Volume 單位: 股；假設均價 ~Close 可從 df 取）
    df_close = df[mask][["stock_id", "Close"]].rename(columns={"stock_id": "Ticker"})
    df_kelly = df_kelly.merge(df_close, on="Ticker", how="left")
    df_kelly["Turnover_5D"] = df_kelly["Volume"] * df_kelly["Close"]  # approx

    # Hard liquidity filter
    MIN_TURNOVER = 1e7   # 1000萬台幣
    low_liq_mask = df_kelly["Turnover_5D"].fillna(0) < MIN_TURNOVER
    if low_liq_mask.sum() > 0:
        logger.info(f"Liquidity filter: removed {low_liq_mask.sum()} illiquid stocks")
    df_kelly.loc[low_liq_mask, "Exp_Alpha_20d"] = -999.0

    # Slippage penalty (rough estimate: 0.4% for small-mid cap)
    df_kelly["Slippage_Est"] = 0.004
    df_kelly["Net_Alpha_20d"] = (
        df_kelly["Exp_Alpha_20d"] - df_kelly["Slippage_Est"]
    ).clip(lower=-1.0)

    # Sharpe score = Net Alpha / Uncertainty (proxy for risk-adjusted rank)
    df_kelly["Sharpe_Score"] = (
        df_kelly["Net_Alpha_20d"] / (df_kelly["Uncertainty"] + 1e-6)
    ).clip(lower=-10.0, upper=10.0)

    # Confidence label
    df_kelly["Confidence"] = pd.cut(
        df_kelly["Uncertainty"],
        bins=[0, 0.02, 0.05, 1.0],
        labels=["高信心", "中信心", "低信心"],
        right=False,
    )

    # Kelly weight (simplified, proportional to Sharpe clipped at 0)
    positive = df_kelly["Sharpe_Score"].clip(lower=0)
    total = positive.sum()
    df_kelly["Suggested_Weight"] = (positive / (total + 1e-9)).round(4)

    # Sort by Sharpe
    df_kelly = df_kelly.sort_values("Sharpe_Score", ascending=False).reset_index(drop=True)

    # -- Build df_traj: multi-horizon trajectory --
    df_traj = pd.DataFrame({
        "Ticker":        stocks_today,
        "Date":          latest_str,
        "Pred_5d":       pred_mean[:, 0],
        "Pred_20d":      pred_mean[:, 1],
        "Pred_60d":      pred_mean[:, 2],
        "Uncertainty_5d":  pred_std[:, 0],
        "Uncertainty_20d": pred_std[:, 1],
        "Uncertainty_60d": pred_std[:, 2],
    })

    return df_kelly, df_traj


# ============================================================
# Main Pipeline
# ============================================================

def main(target_date: str | None = None, skip_push: bool = False) -> None:
    today = target_date or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"\n{'='*55}")
    logger.info(f"  MarketMamba V6 — Daily Inference  [{today}]")
    logger.info(f"{'='*55}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device_str}")
    if device_str == "cuda":
        import torch
        logger.info(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # -- Step 1: Data update --
    logger.info("\n[Step 1/5] Hybrid data update...")
    try:
        run_daily_update(target_date=today)
    except Exception as e:
        logger.error(f"Data update failed: {e}")
        raise

    # -- Step 2: Feature matrix --
    logger.info("\n[Step 2/5] Building feature matrix...")
    prices  = pd.read_parquet(PROCESSED_DIR / "prices_raw.parquet")
    inst    = pd.read_parquet(PROCESSED_DIR / "institutional_raw.parquet") \
              if (PROCESSED_DIR / "institutional_raw.parquet").exists() else None
    margin  = pd.read_parquet(PROCESSED_DIR / "margin_raw.parquet") \
              if (PROCESSED_DIR / "margin_raw.parquet").exists() else None
    revenue = pd.read_parquet(PROCESSED_DIR / "revenue_raw.parquet") \
              if (PROCESSED_DIR / "revenue_raw.parquet").exists() else None
    financials = pd.read_parquet(PROCESSED_DIR / "financials_raw.parquet") \
                 if (PROCESSED_DIR / "financials_raw.parquet").exists() else None
    macro   = pd.read_parquet(PROCESSED_DIR / "macro_raw.parquet") \
              if (PROCESSED_DIR / "macro_raw.parquet").exists() else None

    df = build_features(prices, inst, margin, revenue, financials, macro)
    df = clean_and_scale(df)
    logger.info(f"Feature matrix: {df.shape}")

    # -- Step 3: Inference --
    logger.info("\n[Step 3/5] Running V6 inference...")
    df_kelly, df_traj = run_inference(df, device_str=device_str)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kelly_path = RESULTS_DIR / "df_kelly.csv"
    traj_path  = RESULTS_DIR / "df_traj.csv"
    df_kelly.to_csv(kelly_path, index=False, encoding="utf-8-sig")
    df_traj.to_csv(traj_path,  index=False, encoding="utf-8-sig")

    logger.info(f"\n🎯 Top 10 Alpha Stocks [{today}]:")
    print(
        df_kelly.head(10)[
            ["Ticker", "Exp_Alpha_20d", "Sharpe_Score", "Confidence", "Suggested_Weight"]
        ].to_string(index=False)
    )

    # -- Step 4: LLM Report --
    logger.info("\n[Step 4/5] Generating LLM market report...")
    try:
        market_data = build_market_data()
        report = generate_market_report(df_kelly, market_data, save=True)
        logger.info(f"\n📝 LLM Report preview:\n{report['summary'][:300]}...")
    except Exception as e:
        logger.warning(f"LLM report skipped: {e}")

    # -- Step 5a: Archive (rolling 90-day history) --
    logger.info("\n[Step 5/5] Archiving + pushing results...")
    _archive_results(df_kelly, today)

    # -- Step 5b: Push to GitHub (backend auto-updates) --
    if not skip_push:
        pushed = _push_to_github(RESULTS_DIR, today)
        if pushed:
            logger.info("  Backend will serve real data on next cache refresh (≤1h)")
    else:
        logger.info("  --skip-push: skipping git push (dry run)")

    logger.info(f"\n{'='*55}")
    logger.info(f"  ✅ V6 Inference complete [{today}]")
    logger.info(f"  Results: {RESULTS_DIR}")
    logger.info(f"{'='*55}\n")


def _push_to_github(results_dir: Path, date_str: str) -> bool:
    """
    Git add + commit + push results to GitHub.
    Render backend will pick up the new df_kelly.csv on next cache refresh (≤1h).
    Returns True on success, False on failure (non-fatal — results saved locally).
    """
    import subprocess

    repo_root = results_dir.parent.parent.parent   # MarketMamba/
    try:
        subprocess.run(
            ["git", "add", "V6/results/df_kelly.csv", "V6/results/df_traj.csv"],
            cwd=repo_root, check=True, capture_output=True,
        )
        # Check if there's anything to commit
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root,
        )
        if diff.returncode == 0:
            logger.info("No changes in results — git push skipped (already up to date)")
            return True

        subprocess.run(
            ["git", "commit", "-m", f"inference: daily results {date_str}"],
            cwd=repo_root, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_root, check=True, capture_output=True,
        )
        logger.info("✅ Results pushed to GitHub (branch: main)")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"⚠️  git push failed — results saved locally but NOT on GitHub.\n"
            f"   Error: {e.stderr.decode() if e.stderr else str(e)}\n"
            f"   Manual fix: git add V6/results/ && git push"
        )
        return False


def _archive_results(df_kelly: pd.DataFrame, date_str: str) -> None:
    """Save today's results to a dated subdirectory (rolling 90-day archive)."""
    import shutil
    from datetime import timedelta

    dated_dir = RESULTS_DIR / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    df_kelly.to_csv(dated_dir / "df_kelly.csv", index=False, encoding="utf-8-sig")

    # Remove entries older than 90 days
    cutoff = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name < cutoff:
            shutil.rmtree(d)
            logger.info(f"Removed old archive: {d.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketMamba V6 Daily Inference")
    parser.add_argument("--date",      type=str,  default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--skip-push", action="store_true",     help="Skip git push (dry run / test mode)")
    args = parser.parse_args()
    main(target_date=args.date, skip_push=args.skip_push)
