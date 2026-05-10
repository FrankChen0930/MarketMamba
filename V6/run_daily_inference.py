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
import json
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

    from marketmamba.models.trainer import TrainingHistory
    torch.serialization.add_safe_globals([TrainingHistory])
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
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

    # -- Inference with MC-Dropout uncertainty (mini-batch to fit 6 GB VRAM) --
    X, _, valid_stocks = test_ds[0]   # __getitem__ returns (X, Y, stock_ids)

    N = X.shape[0]
    INFER_BATCH = 128   # stocks per GPU step (tune down to 64 if still OOM)
    N_MC        = 30    # MC-Dropout samples

    pred_mean_acc = torch.zeros(N, 3)
    pred_std_acc  = torch.zeros(N, 3)

    model.train()   # enable dropout for MC
    with torch.no_grad():
        for batch_start in range(0, N, INFER_BATCH):
            x_b = X[batch_start: batch_start + INFER_BATCH].to(device)
            mc_preds = []
            for _mc in range(N_MC):
                p = model(x_b, edge_index, edge_attr)   # (B, 3)
                mc_preds.append(p.cpu())
            mc_stack = torch.stack(mc_preds, dim=0)      # (N_MC, B, 3)
            pred_mean_acc[batch_start: batch_start + INFER_BATCH] = mc_stack.mean(0)
            pred_std_acc [batch_start: batch_start + INFER_BATCH] = mc_stack.std(0)
    model.eval()

    pred_mean = pred_mean_acc.numpy()   # (N, 3)
    pred_std  = pred_std_acc.numpy()    # (N, 3)

    # Use valid_stocks from dataset (correctly ordered to match X rows)
    stocks_today = valid_stocks[:len(pred_mean)]


    # -- Build df_kelly --
    df_kelly = pd.DataFrame({
        "Ticker":        stocks_today,
        "Date":          latest_str,
        "Exp_Alpha_5d":  pred_mean[:, 0],
        "Exp_Alpha_20d": pred_mean[:, 1],   # primary
        "Exp_Alpha_60d": pred_mean[:, 2],
        "Uncertainty":   pred_std[:, 1],    # 20d uncertainty
    })

    # Liquidity filter — use RAW prices (Volume/Close are z-scored in feature matrix)
    mask = df["Date"] == latest_date
    try:
        _raw = pd.read_parquet(
            PROCESSED_DIR / "prices_raw.parquet",
            columns=["Date", "stock_id", "Volume", "Close"],
        )
        _raw["Date"] = pd.to_datetime(_raw["Date"])
        df_today = (_raw[_raw["Date"] == latest_date]
                    .drop_duplicates("stock_id", keep="last")[["stock_id", "Volume", "Close"]])
    except Exception:
        # Fallback: use z-scored values (threshold must be relaxed to 0)
        df_today = (df[mask][["stock_id", "Volume", "Close"]]
                    .drop_duplicates("stock_id", keep="last").copy())
    df_kelly = df_kelly.merge(
        df_today.rename(columns={"stock_id": "Ticker"}),
        on="Ticker", how="left",
    )
    df_kelly["Turnover_5D"] = df_kelly["Volume"].fillna(0) * df_kelly["Close"].fillna(0)

    # Hard liquidity filter
    MIN_TURNOVER = 1e7   # 1000萬台幣
    low_liq_mask = df_kelly["Turnover_5D"] < MIN_TURNOVER
    n_filtered = int(low_liq_mask.sum())
    if n_filtered > 0:
        logger.info(f"Liquidity filter: removed {n_filtered} illiquid stocks "
                    f"out of {len(df_kelly)}")
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

    # Sort by Sharpe — exclude penalised stocks from Top 10 display
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
    # ── Time-aware date selection ─────────────────────────────────────────────
    # Taiwan market closes at 13:30; data is reliable after ~14:00.
    # Running before 14:00 would fetch incomplete/empty data → use yesterday.
    if target_date is not None:
        today = target_date
    else:
        from zoneinfo import ZoneInfo
        from datetime import timedelta
        now_twn = datetime.now(ZoneInfo("Asia/Taipei"))
        if now_twn.hour < 14:
            today = (now_twn - timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(
                f"⏰ Before 14:00 Taiwan time ({now_twn.strftime('%H:%M')}) — "
                f"using yesterday's date: {today}"
            )
        else:
            today = now_twn.strftime("%Y-%m-%d")

    logger.info(f"\n{'='*55}")
    logger.info(f"  MarketMamba V6 — Daily Inference  [{today}]")
    logger.info(f"{'='*55}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device_str}")
    if device_str == "cuda":
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

    def _read(name: str):
        path = PROCESSED_DIR / name
        return pd.read_parquet(path) if path.exists() else None

    # V6.0 core sources
    prices       = _read("prices_raw.parquet")
    inst         = _read("institutional_raw.parquet")
    margin       = _read("margin_raw.parquet")
    per          = _read("per_raw.parquet")
    market_value = _read("market_value_raw.parquet")
    daytrade     = _read("daytrade_raw.parquet")
    revenue      = _read("revenue_raw.parquet")
    financials   = _read("financials_raw.parquet")
    balance_sheet = _read("balance_sheet_raw.parquet")
    macro        = _read("macro_raw.parquet")

    # V6.1 additional sources
    securities   = _read("securities_raw.parquet")
    holdings     = _read("holdings_raw.parquet")
    cashflow     = _read("cashflow_raw.parquet")
    dividend     = _read("dividend_raw.parquet")
    foreign_shareholding = _read("foreign_shareholding_raw.parquet")
    fear_greed   = _read("fear_greed.parquet")
    business_indicator = _read("business_indicator.parquet")
    fed_rate     = _read("fed_rate.parquet")
    futures_inst = _read("futures_institutional_raw.parquet")
    options_inst = _read("options_institutional_raw.parquet")

    if prices is None:
        raise FileNotFoundError(f"prices_raw.parquet not found in {PROCESSED_DIR}")

    # For inference, only the last ~2 years is needed to cover all rolling windows
    # (MA_60, ATR_14, SEQ_LEN=60, etc.). Loading 14 years causes OOM on local machine.
    INFERENCE_LOOKBACK_DAYS = 730  # 2 calendar years ≈ 500 trading days
    prices["Date"] = pd.to_datetime(prices["Date"])
    cutoff = prices["Date"].max() - pd.Timedelta(days=INFERENCE_LOOKBACK_DAYS)
    prices = prices[prices["Date"] >= cutoff].copy()
    logger.info(f"Prices trimmed to last 2y: {len(prices):,} rows "
                f"({prices['Date'].min().date()} → {prices['Date'].max().date()})")

    # Trim other time-series data to same window (save memory)
    def _trim(df_src):
        if df_src is None: return None
        if "Date" in df_src.columns:
            df_src["Date"] = pd.to_datetime(df_src["Date"])
            return df_src[df_src["Date"] >= cutoff].copy()
        return df_src

    inst     = _trim(inst)
    margin   = _trim(margin)
    daytrade = _trim(daytrade)
    holdings = _trim(holdings)
    securities = _trim(securities)
    foreign_shareholding = _trim(foreign_shareholding)

    df = build_features(
        df_price         = prices,
        df_inst          = inst,
        df_margin        = margin,
        df_per           = per,
        df_securities    = securities,
        df_market_value  = market_value,
        df_daytrade      = daytrade,
        df_holdings      = holdings,
        df_rev           = revenue,
        df_fin           = financials,
        df_balance_sheet = balance_sheet,
        df_cashflow      = cashflow,
        df_macro         = macro,
        # V6.1 new data sources
        df_futures_inst  = futures_inst,
        df_options_inst  = options_inst,
        df_dividend      = dividend,
        df_foreign_shareholding = foreign_shareholding,
        df_fear_greed    = fear_greed,
        df_business_indicator = business_indicator,
        df_fed_rate      = fed_rate,
    )
    df = clean_and_scale(df)
    # Deduplicate: institutional_raw is long-format (4 rows/stock/date) → 4x duplication
    n_before = len(df)
    df = df.drop_duplicates(subset=["Date", "stock_id"], keep="last")
    if len(df) < n_before:
        logger.info(f"Deduped feature matrix: {n_before:,} → {len(df):,} rows")
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
    top10 = df_kelly[df_kelly["Exp_Alpha_20d"] > -999].head(10)
    print(
        top10[["Ticker", "Exp_Alpha_20d", "Sharpe_Score", "Confidence", "Suggested_Weight"]]
        .to_string(index=False)
    )
    logger.info(f"Total investable stocks: {(df_kelly['Exp_Alpha_20d'] > -999).sum()} "
                f"/ {len(df_kelly)} total")


    # -- Step 4: LLM Report --
    logger.info("\n[Step 4/5] Generating LLM market report...")
    try:
        market_data = build_market_data()
        report = generate_market_report(df_kelly, market_data, save=True)
        logger.info(f"\n📝 LLM Report preview:\n{report['summary'][:300]}...")
    except Exception as e:
        logger.warning(f"LLM report skipped: {e}")

    # -- Step 5a: Archive (rolling 90-day history) --
    logger.info("\n[Step 5/6] Archiving results...")
    _archive_results(df_kelly, today)

    # -- Step 5b: Signal Scanner --
    logger.info("\n[Step 6/6] Running Signal Scanner...")
    try:
        from marketmamba.signals.scanner import run_scan
        scan_result = run_scan(df_kelly_path=RESULTS_DIR / "df_kelly.csv")
        n_buy = len(scan_result.get("buy_signals", []))
        n_exit = len(scan_result.get("exit_signals", []))
        n_watch = len(scan_result.get("watch_list", []))
        logger.info(f"  Scanner: {n_buy} BUY · {n_exit} EXIT · {n_watch} WATCH")
    except Exception as e:
        logger.warning(f"Signal Scanner failed (non-fatal): {e}")

    # -- Push to GitHub (backend auto-updates) --
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
    import os
    import subprocess

    repo_root = results_dir.parent.parent   # V6/results → V6 → MarketMamba (.git is here)
    # WSL2: git can't find repo across /mnt filesystem boundary without this
    git_env = {**os.environ, "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1"}
    try:
        subprocess.run(
            ["git", "add",
             "V6/results/df_kelly.csv",
             "V6/results/df_traj.csv",
             "V6/results/market_summary.json",
             "V6/results/action_signals.json",
             "V6/results/history_index.json",
             "V6/results/archive/"],   # push archived history too

            cwd=repo_root, check=True, capture_output=True, env=git_env,
        )

        # Check if there's anything to commit
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root, env=git_env,
        )
        if diff.returncode == 0:
            logger.info("No changes in results — git push skipped (already up to date)")
            return True

        subprocess.run(
            ["git", "commit", "-m", f"inference: daily results {date_str}"],
            cwd=repo_root, check=True, capture_output=True, env=git_env,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_root, check=True, capture_output=True, env=git_env,
        )
        logger.info("✅ Results pushed to GitHub (branch: main)")
        # Notify Render backend to refresh its cache immediately
        _refresh_render_cache()
        return True


    except subprocess.CalledProcessError as e:
        logger.warning(
            f"⚠️  git push failed — results saved locally but NOT on GitHub.\n"
            f"   Error: {e.stderr.decode() if e.stderr else str(e)}\n"
            f"   Manual fix: git add V6/results/ && git push"
        )
        return False


def _refresh_render_cache() -> None:
    """
    POST to Render backend to invalidate the 1-hour signals cache immediately
    after a git push so the dashboard shows fresh results within seconds.
    Set RENDER_BACKEND_URL in .env, e.g. https://marketmamba-api.onrender.com
    """
    import os, urllib.request
    url = os.getenv("RENDER_BACKEND_URL", "").rstrip("/")
    if not url:
        return
    endpoint = f"{url}/api/signals/cache/refresh"
    try:
        req = urllib.request.Request(endpoint, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(f"📡 Render cache refreshed: {resp.status}")
    except Exception as e:
        logger.warning(f"Cache refresh failed (non-fatal): {e}")





def _archive_results(df_kelly: pd.DataFrame, date_str: str) -> None:
    """Save today's results to a dated subdirectory (rolling 90-day archive)."""
    import shutil
    from datetime import timedelta

    dated_dir = RESULTS_DIR / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    df_kelly.to_csv(dated_dir / "df_kelly.csv", index=False, encoding="utf-8-sig")

    # Update lightweight history index for the dashboard
    _update_history_index(df_kelly, date_str)

    # Remove entries older than 90 days
    cutoff = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name < cutoff:
            shutil.rmtree(d)
            logger.info(f"Removed old archive: {d.name}")


def _update_history_index(df_kelly: pd.DataFrame, date_str: str) -> None:
    """
    Maintain V6/results/history_index.json — a lightweight log of each
    rebalancing day (top 50 positions). Used by the Signal Scanner for
    rank stability tracking and by the Investment Sim page.
    """
    history_path = RESULTS_DIR / "history_index.json"

    # Load existing entries
    history: list = []
    if history_path.exists():
        try:
            with open(history_path, encoding="utf-8") as f:
                history = json.load(f).get("history", [])
        except Exception:
            history = []

    # Remove today if already present (idempotent)
    history = [h for h in history if h.get("date") != date_str]

    # Build today's entry — top 50 by Sharpe for scanner rank stability
    investable = df_kelly[df_kelly["Exp_Alpha_20d"] > -999]
    top50 = investable.head(50)

    cols_needed = ["Ticker", "Exp_Alpha_20d", "Sharpe_Score", "Confidence", "Suggested_Weight", "Uncertainty"]
    available   = [c for c in cols_needed if c in top50.columns]
    portfolio   = []
    for i, (_, row) in enumerate(top50[available].iterrows()):
        portfolio.append({
            "rank":       i + 1,
            "ticker":     str(row.get("Ticker", "")),
            "alpha":      round(float(row.get("Exp_Alpha_20d", 0)), 6),
            "sharpe":     round(float(row.get("Sharpe_Score",  0)), 3),
            "confidence": str(row.get("Confidence", "")),
            "weight":     round(float(row.get("Suggested_Weight", 0)), 4),
            "uncertainty":round(float(row.get("Uncertainty", 0)), 6),
        })

    entry = {
        "date":             date_str,
        "total_investable": int(len(investable)),
        "portfolio":        portfolio,
    }
    history.insert(0, entry)
    history = history[:60]   # keep last 60 trading days

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"last_updated": date_str, "history": history}, f,
                  ensure_ascii=False, indent=2)
    logger.info(f"History index updated → {history_path} ({len(history)} entries)")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketMamba V6 Daily Inference")
    parser.add_argument("--date",      type=str,  default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--skip-push", action="store_true",     help="Skip git push (dry run / test mode)")
    args = parser.parse_args()
    main(target_date=args.date, skip_push=args.skip_push)
