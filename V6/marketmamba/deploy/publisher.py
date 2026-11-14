"""
MarketMamba V6 — GitHub Publisher / Deploy
============================================
Handles publishing results to GitHub so the Streamlit frontend can auto-refresh.

Responsibilities:
  1. Copy result files (df_kelly.csv, df_traj.csv, market_summary.json) to repo root
  2. Git add → commit → push
  3. Archive today's results to rolling 90-day history
  4. Optional: send notification (can be extended with Telegram/Discord webhook)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from marketmamba.config import RESULTS_DIR, ROOT_DIR

logger = logging.getLogger(__name__)

# Root of the git repo (one level up from V6/)
REPO_ROOT = ROOT_DIR.parent


# ============================================================
# Git Helpers
# ============================================================

def _git(cmd: str, cwd: Path = REPO_ROOT) -> tuple[int, str]:
    """Run a git command and return (returncode, output)."""
    result = subprocess.run(
        f"git {cmd}",
        shell=True,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() + result.stderr.strip()
    return result.returncode, output


def _check_git_available() -> bool:
    rc, _ = _git("status")
    return rc == 0


# ============================================================
# File Staging
# ============================================================

def stage_results(date_str: str | None = None) -> list[Path]:
    """
    Copy today's result files into the repo's public results directory.
    Returns list of files that were staged.
    """
    date_str = date_str or datetime.today().strftime("%Y-%m-%d")

    # Files to publish
    source_files = {
        "df_kelly.csv":       RESULTS_DIR / "df_kelly.csv",
        "df_traj.csv":        RESULTS_DIR / "df_traj.csv",
        "market_summary.json": RESULTS_DIR / "market_summary.json",
    }

    # Public results folder (what the frontend reads)
    public_dir = REPO_ROOT / "results"
    public_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    for fname, src_path in source_files.items():
        if not src_path.exists():
            logger.warning(f"Result file missing: {src_path}")
            continue
        dst = public_dir / fname
        shutil.copy2(src_path, dst)
        staged.append(dst)

    # Also write update timestamp
    ts_path = public_dir / "update_time.txt"
    ts_path.write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    staged.append(ts_path)

    logger.info(f"Staged {len(staged)} files to {public_dir}")
    return staged


# ============================================================
# Main Push Function
# ============================================================

def push_to_github(
    date_str: str | None = None,
    commit_msg: str | None = None,
) -> bool:
    """
    Publish today's results to GitHub.

    Returns:
        True if push succeeded, False otherwise.
    """
    date_str = date_str or datetime.today().strftime("%Y-%m-%d")

    if not _check_git_available():
        logger.error("Git not available or not in a git repository")
        return False

    # Stage files
    staged = stage_results(date_str)
    if not staged:
        logger.warning("No files to stage — push aborted")
        return False

    # Git add
    staged_paths = " ".join(f'"{str(p)}"' for p in staged)
    rc, out = _git(f"add results/")
    if rc != 0:
        logger.error(f"git add failed: {out}")
        return False

    # Git commit
    msg = commit_msg or f"V6 daily results [{date_str}]"
    rc, out = _git(f'commit -m "{msg}"')
    if rc != 0 and "nothing to commit" in out:
        logger.info("Nothing new to commit")
        return True
    if rc != 0:
        logger.error(f"git commit failed: {out}")
        return False

    # Git push
    rc, out = _git("push origin main")
    if rc != 0:
        logger.error(f"git push failed: {out}")
        return False

    logger.info(f"✅ Results pushed to GitHub [{date_str}]")
    return True


# ============================================================
# Shutdown Helper (Colab)
# ============================================================

def shutdown_colab() -> None:
    """Safely disconnect Colab runtime to stop billing GPU quota."""
    try:
        from google.colab import runtime
        runtime.unassign()
        logger.info("Colab runtime disconnected")
    except ImportError:
        # Not in Colab environment — no-op
        pass
    except Exception as e:
        logger.warning(f"Colab shutdown error: {e}")
