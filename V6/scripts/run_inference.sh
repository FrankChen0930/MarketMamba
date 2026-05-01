#!/usr/bin/env bash
# =============================================================================
# MarketMamba V6 — Daily Inference Runner
# =============================================================================
# PersonalOS 每日自動呼叫這個腳本，時間建議設在台股收盤後 16:00-17:00。
#
# 使用方式：
#   bash V6/scripts/run_inference.sh              # 正式執行
#   bash V6/scripts/run_inference.sh --dry-run    # 不 push，測試用
#
# Windows Task Scheduler 呼叫方式（WSL2）：
#   wsl -d Ubuntu -e bash /mnt/d/Desktop/work/ProjectForMe/MarketMamba/V6/scripts/run_inference.sh
# =============================================================================

set -euo pipefail

# ── 設定 ──────────────────────────────────────────────────────────────────────
REPO_DIR="/mnt/d/Desktop/work/ProjectForMe/MarketMamba"
CONDA_ENV="mamba_env"
LOG_DIR="$REPO_DIR/V6/results/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/inference_${DATE}.log"

# 是否 dry run（不 push）
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--skip-push"
    echo "⚠️  DRY RUN MODE — results will NOT be pushed to GitHub"
fi

# ── 建立 log 目錄 ──────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── 日誌 tee（同時輸出到 terminal 和 log 檔）──────────────────────────────────
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "════════════════════════════════════════════════"
echo "  MarketMamba V6 Daily Inference"
echo "  Date: $DATE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════"

# ── 切換到 repo 目錄 ───────────────────────────────────────────────────────────
cd "$REPO_DIR"

# ── 啟動 conda 環境並執行推論 ──────────────────────────────────────────────────
echo ""
echo "[INFO] Activating conda environment: $CONDA_ENV"

# 確保 conda 可用（WSL2 環境）
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "/opt/conda/etc/profile.d/conda.sh" 2>/dev/null || true

conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "[ERROR] Failed to activate conda env: $CONDA_ENV"
    exit 1
}

echo "[INFO] Python: $(python --version)"
echo "[INFO] Starting inference..."
echo ""

# ── 執行主推論腳本 ─────────────────────────────────────────────────────────────
python V6/run_daily_inference.py $DRY_RUN

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "════════════════════════════════════════════════"
    echo "  ✅ Inference COMPLETED successfully"
    echo "  Log saved: $LOG_FILE"
    echo "════════════════════════════════════════════════"
else
    echo "════════════════════════════════════════════════"
    echo "  ❌ Inference FAILED (exit code: $EXIT_CODE)"
    echo "  Check log: $LOG_FILE"
    echo "════════════════════════════════════════════════"
fi

exit $EXIT_CODE
