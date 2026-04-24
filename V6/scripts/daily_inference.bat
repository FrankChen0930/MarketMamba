@echo off
:: ============================================================
:: MarketMamba V6 — Daily Inference Trigger
:: Scheduled via Windows Task Scheduler at 17:00 on weekdays
:: ============================================================
:: Setup: 
::   1. Open Task Scheduler → Create Basic Task
::   2. Trigger: Daily, 17:00, repeat Mon-Fri
::   3. Action: Start a Program
::      Program: C:\Windows\System32\wsl.exe
::      Arguments: -d Ubuntu -e bash -c "cd /mnt/d/Desktop/work/MarketMamba && conda run -n mamba_env python V6/run_daily_inference.py >> V6/logs/inference.log 2>&1"

SET LOGDIR=D:\Desktop\work\MarketMamba\V6\logs
IF NOT EXIST "%LOGDIR%" MKDIR "%LOGDIR%"

echo [%DATE% %TIME%] Starting V6 daily inference... >> "%LOGDIR%\scheduler.log"

wsl -d Ubuntu -e bash -c "cd /mnt/d/Desktop/work/MarketMamba && conda run -n mamba_env python V6/run_daily_inference.py >> V6/logs/inference.log 2>&1"

IF %ERRORLEVEL% EQU 0 (
    echo [%DATE% %TIME%] ✅ Inference completed successfully >> "%LOGDIR%\scheduler.log"
) ELSE (
    echo [%DATE% %TIME%] ❌ Inference FAILED with exit code %ERRORLEVEL% >> "%LOGDIR%\scheduler.log"
)
