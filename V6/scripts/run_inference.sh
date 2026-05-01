#!/usr/bin/env bash
# =============================================================================
# MarketMamba V6 — Daily Inference Runner (Robust Edition)
# =============================================================================
# 功能：
#   1. 自動判斷今天是否為台股交易日（週末 + 國定假日跳過）
#   2. 資料抓取失敗 → Windows 跳通知
#   3. 推論成功/失敗 → Windows 跳通知
#   4. 每次執行完整 log 存檔
#
# 使用方式：
#   bash V6/scripts/run_inference.sh              # 正式執行
#   bash V6/scripts/run_inference.sh --dry-run    # 不 push，測試用
#   bash V6/scripts/run_inference.sh --force      # 強制執行（跳過交易日檢查）
#
# PersonalOS / Task Scheduler 設定：
#   wsl -d Ubuntu -e bash /mnt/d/Desktop/work/ProjectForMe/MarketMamba/V6/scripts/run_inference.sh
# =============================================================================

set -uo pipefail   # 注意：去掉 -e，讓我們自行控制 exit

# ── 設定 ──────────────────────────────────────────────────────────────────────
REPO_DIR="/mnt/d/Desktop/work/ProjectForMe/MarketMamba"
CONDA_ENV="mamba_env"
LOG_DIR="$REPO_DIR/V6/results/logs"
DATE=$(date +%Y-%m-%d)
DAY_OF_WEEK=$(date +%u)   # 1=Mon, 7=Sun
LOG_FILE="$LOG_DIR/inference_${DATE}.log"
STATUS_FILE="$LOG_DIR/last_run_status.json"

# ── 參數解析 ────────────────────────────────────────────────────────────────────
DRY_RUN=""
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--skip-push" ;;
        --force)   FORCE=true ;;
    esac
done

# ── 建立 log 目錄 ──────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── 日誌同步輸出 ──────────────────────────────────────────────────────────────
exec > >(tee -a "$LOG_FILE") 2>&1

# ── Windows 通知函式（從 WSL2 呼叫 PowerShell）────────────────────────────────
notify() {
    local TITLE="$1"
    local MSG="$2"
    local ICON="${3:-Information}"   # Information / Warning / Error
    # 嘗試用 PowerShell 跳 Toast 通知
    powershell.exe -NoProfile -NonInteractive -Command "
        \$notif = New-Object -ComObject WScript.Shell
        \$notif.Popup('$MSG', 5, 'MarketMamba: $TITLE', 64)
    " >/dev/null 2>&1 || true
    # 備援：寫到 Windows 事件日誌
    # powershell.exe -Command "Write-EventLog -LogName Application -Source 'MarketMamba' -EventId 1 -Message '$MSG'" >/dev/null 2>&1 || true
}

# ── 台股交易日判斷 ─────────────────────────────────────────────────────────────
is_trading_day() {
    # 週六 (6) 或週日 (7) → 不開盤
    if [ "$DAY_OF_WEEK" -ge 6 ]; then
        echo "SKIP_WEEKEND"
        return
    fi

    # 2026 台股國定假日（YYYY-MM-DD 格式）
    # 來源：勞動部行事曆 + TWSE 公告
    local HOLIDAYS=(
        # 2026
        "2026-01-01"  # 元旦
        "2026-01-27"  # 農曆除夕
        "2026-01-28"  # 春節
        "2026-01-29"  # 春節
        "2026-01-30"  # 春節
        "2026-02-02"  # 春節
        "2026-02-28"  # 228 紀念日
        "2026-04-03"  # 兒童節補假
        "2026-04-04"  # 兒童節
        "2026-04-05"  # 清明節
        "2026-05-01"  # 勞動節
        "2026-06-19"  # 端午節
        "2026-09-25"  # 中秋節
        "2026-10-09"  # 國慶日補假
        "2026-10-10"  # 國慶日
        "2026-12-25"  # 聖誕節（非固定，但常見補假）
        # 2027（如果推論跨年用到）
        "2027-01-01"
    )

    for h in "${HOLIDAYS[@]}"; do
        if [ "$DATE" = "$h" ]; then
            echo "SKIP_HOLIDAY"
            return
        fi
    done

    # 額外驗證：用 yfinance 確認今天 TAIEX 有沒有資料（最可靠）
    # 如果 Python 環境可用的話
    export PATH="$HOME/miniconda3/bin:$PATH"
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        local RESULT
        RESULT=$(python3 -c "
import yfinance as yf, sys
from datetime import datetime, timedelta
today = '${DATE}'
# 抓今天的 TAIEX 資料
df = yf.download('^TWII', start=today, end=today, progress=False, auto_adjust=True)
if df.empty:
    print('NO_DATA')
else:
    print('HAS_DATA')
" 2>/dev/null) || RESULT="UNKNOWN"
        echo "$RESULT"
    else
        echo "UNKNOWN"
    fi
}

# ── 主程式開始 ─────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  MarketMamba V6 Daily Inference"
echo "  Date: $DATE  (Day: $DAY_OF_WEEK)"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  DryRun: ${DRY_RUN:-none}  Force: $FORCE"
echo "════════════════════════════════════════════════"

# ── 交易日檢查 ────────────────────────────────────────────────────────────────
if [ "$FORCE" = false ]; then
    echo ""
    echo "[CHECK] Is today a trading day?"
    TRADE_STATUS=$(is_trading_day)
    echo "  → $TRADE_STATUS"

    case "$TRADE_STATUS" in
        SKIP_WEEKEND)
            echo "📅 Today is weekend — skipping inference"
            echo '{"date":"'"$DATE"'","status":"skipped","reason":"weekend"}' > "$STATUS_FILE"
            exit 0
            ;;
        SKIP_HOLIDAY)
            echo "📅 Today is a public holiday — skipping inference"
            notify "跳過推論" "今天 ($DATE) 是國定假日，台股休市，推論已跳過"
            echo '{"date":"'"$DATE"'","status":"skipped","reason":"holiday"}' > "$STATUS_FILE"
            exit 0
            ;;
        NO_DATA)
            echo "⚠️  TAIEX has no data for today — market may be closed or data not ready"
            notify "⚠️ 無交易資料" "今天 ($DATE) TAIEX 沒有資料，可能是休市或資料尚未更新。推論已跳過。" "Warning"
            echo '{"date":"'"$DATE"'","status":"skipped","reason":"no_market_data"}' > "$STATUS_FILE"
            exit 0
            ;;
        HAS_DATA)
            echo "  ✅ Market is open — proceeding"
            ;;
        UNKNOWN)
            echo "  ⚠️  Cannot verify trading day (yfinance unavailable) — proceeding anyway"
            ;;
    esac
fi

# ── 啟動 conda 環境 ────────────────────────────────────────────────────────────
echo ""
echo "[ENV] Activating conda: $CONDA_ENV"
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "/opt/conda/etc/profile.d/conda.sh" 2>/dev/null || {
    notify "❌ 環境錯誤" "無法找到 conda，請確認 miniconda 已安裝" "Error"
    exit 1
}

conda activate "$CONDA_ENV" 2>/dev/null || {
    notify "❌ 環境錯誤" "無法啟動 conda 環境: $CONDA_ENV" "Error"
    exit 1
}
echo "  Python: $(python --version 2>&1)"

# ── 切換到 repo 目錄 ──────────────────────────────────────────────────────────
cd "$REPO_DIR"

# ── 模型檔案檢查 ───────────────────────────────────────────────────────────────
MODEL_FILE="$REPO_DIR/V6/models/v6_best.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found: $MODEL_FILE"
    notify "❌ 模型檔案缺失" "找不到 v6_best.pt，推論無法進行！請確認模型檔案位置。" "Error"
    echo '{"date":"'"$DATE"'","status":"error","reason":"model_not_found"}' > "$STATUS_FILE"
    exit 1
fi
echo "[CHECK] Model file: ✅ $(du -h "$MODEL_FILE" | cut -f1)"

# ── 執行推論 ────────────────────────────────────────────────────────────────────
echo ""
echo "[RUN] Starting inference..."
INFERENCE_START=$(date +%s)

python V6/run_daily_inference.py $DRY_RUN
EXIT_CODE=$?

INFERENCE_END=$(date +%s)
ELAPSED=$(( INFERENCE_END - INFERENCE_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "════════════════════════════════════════════════"
    echo "  ✅ Inference COMPLETED  (${ELAPSED_MIN}m${ELAPSED#*m}s)"
    echo "  Log: $LOG_FILE"
    echo "════════════════════════════════════════════════"
    notify "✅ 推論完成" "MarketMamba V6 今日推論成功完成！耗時 ${ELAPSED_MIN} 分鐘。結果已推送至 GitHub。"
    echo '{"date":"'"$DATE"'","status":"success","elapsed_sec":'"$ELAPSED"'}' > "$STATUS_FILE"
else
    echo "════════════════════════════════════════════════"
    echo "  ❌ Inference FAILED  (exit code: $EXIT_CODE)"
    echo "  Log: $LOG_FILE"
    echo "════════════════════════════════════════════════"
    notify "❌ 推論失敗" "MarketMamba V6 推論發生錯誤！Exit: $EXIT_CODE 請查看 Log: inference_${DATE}.log" "Error"
    echo '{"date":"'"$DATE"'","status":"error","exit_code":'"$EXIT_CODE"'}' > "$STATUS_FILE"
fi

exit $EXIT_CODE
