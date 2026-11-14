@echo off
:: ============================================================
:: MarketMamba V6 — Daily Inference Trigger
:: Scheduled via Windows Task Scheduler at 17:00 on weekdays
:: ============================================================
:: ⚠ Task Scheduler CRITICAL SETTINGS:
::   - "Run only when user is logged on" (NOT "Run whether user is logged on or not")
::     → WSL2 requires an active user session; SYSTEM account cannot run WSL.
::   - "Configure for: Windows 10" or later
::   - "Do not start a new instance" if already running
::
:: Setup:
::   1. Open Task Scheduler → Create Basic Task
::   2. Trigger: Daily, 17:00, Mon-Fri only
::   3. Action: Start a Program
::      Program/script : cmd.exe
::      Arguments      : /c "D:\Desktop\work\ProjectForMe\MarketMamba\V6\scripts\daily_inference.bat"
::   4. General tab → select "Run only when user is logged on"

SET LOGDIR=D:\Desktop\work\ProjectForMe\MarketMamba\V6\logs
IF NOT EXIST "%LOGDIR%" MKDIR "%LOGDIR%"

:: 可選參數轉傳給 Python（例如手動重啟時想加 --skip-push 測試）
:: 用法： daily_inference.bat --skip-push
SET EXTRA_ARGS=%~1

echo [%DATE% %TIME%] Starting V6 daily inference... >> "%LOGDIR%\scheduler.log"

:: Use absolute conda path (~ can fail in non-interactive Task Scheduler sessions)
:: PIPESTATUS trick: write Python exit code to temp file so tee doesn't swallow it
wsl -d Ubuntu -- bash -lc ^
  "source /home/frank/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && python V6/run_daily_inference.py %EXTRA_ARGS% 2>&1 | tee -a V6/logs/inference.log; exit ${PIPESTATUS[0]}"

SET PYTHON_EXIT=%ERRORLEVEL%

IF %PYTHON_EXIT% EQU 0 (
    echo [%DATE% %TIME%] Inference completed successfully >> "%LOGDIR%\scheduler.log"
) ELSE (
    echo [%DATE% %TIME%] Inference FAILED with exit code %PYTHON_EXIT% >> "%LOGDIR%\scheduler.log"
)

exit /b %PYTHON_EXIT%
