@echo off
:: ============================================================
:: MarketMamba V6 — Daily Inference Trigger
:: Scheduled via Windows Task Scheduler at 17:00 on weekdays
:: ============================================================
:: Setup: 
::   1. Open Task Scheduler → Create Basic Task
::   2. Trigger: Daily, 17:00, repeat Mon-Fri
::   3. Action: Start a Program
::      Program: D:\Desktop\work\ProjectForMe\MarketMamba\V6\scripts\daily_inference.bat

SET LOGDIR=D:\Desktop\work\ProjectForMe\MarketMamba\V6\logs
IF NOT EXIST "%LOGDIR%" MKDIR "%LOGDIR%"

echo [%DATE% %TIME%] Starting V6 daily inference... >> "%LOGDIR%\scheduler.log"

wsl -d Ubuntu -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && python V6/run_daily_inference.py 2>&1 | tee -a V6/logs/inference.log"

IF %ERRORLEVEL% EQU 0 (
    echo [%DATE% %TIME%] Inference completed successfully >> "%LOGDIR%\scheduler.log"
) ELSE (
    echo [%DATE% %TIME%] Inference FAILED with exit code %ERRORLEVEL% >> "%LOGDIR%\scheduler.log"
)
