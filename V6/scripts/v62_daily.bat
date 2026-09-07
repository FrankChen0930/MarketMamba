@echo off
:: ============================================================
:: MarketMamba V6.2 — 每日組合層（獨立於 V6.1，失敗不影響線上）
:: ============================================================
:: ⚠ 執行時間必須在 **21:30 之後**
::   實測 2026-08-05：19:35 時 TWSE 的 margin 與 daytrade 都還「尚未公布」，
::   21:11 才有當日資料。跑太早不會報錯——`_merge_margin` 會 ffill 昨天的值，
::   訓練端有當日 margin 而推論端沒有，形成 train/serve 不對稱。
::
:: V6.2 會自己呼叫既有 run_daily_update 更新 raw parquet，不依賴 V6.1。
:: 若只需要保留每日資料 ingestion，使用 --fetch-only；該模式在抓取與
:: 當日完整性檢查後立即結束，不建特徵矩陣、不載模型、不推論、不發布。
::
:: Task Scheduler 設定（與 daily_inference.bat 相同的關鍵項）：
::   - "Run only when user is logged on"（WSL2 需要有使用者 session，SYSTEM 帳號跑不動）
::   - "Do not start a new instance" if already running
::
:: 用法：
::   v62_daily.bat                 平日例行
::   v62_daily.bat --first-day     上線第一天（強制建倉）
::   v62_daily.bat --fetch-only    只更新 raw parquet，不建矩陣/推論/發布

SET LOGDIR=D:\Desktop\work\ProjectForMe\MarketMamba\V6\logs
IF NOT EXIST "%LOGDIR%" MKDIR "%LOGDIR%"

SET EXTRA_ARGS=%~1

echo [%DATE% %TIME%] Starting V6.2 daily... >> "%LOGDIR%\scheduler.log"

wsl -d Ubuntu -- bash -lc ^
  "source /home/frank/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env && cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba && python V6/run_v62_daily.py %EXTRA_ARGS% 2>&1 | tee -a V6/logs/v62_daily.log; exit ${PIPESTATUS[0]}"

SET PYTHON_EXIT=%ERRORLEVEL%

IF %PYTHON_EXIT% EQU 0 (
    echo [%DATE% %TIME%] V6.2 completed >> "%LOGDIR%\scheduler.log"
) ELSE (
    echo [%DATE% %TIME%] V6.2 FAILED exit=%PYTHON_EXIT% >> "%LOGDIR%\scheduler.log"
)

exit /b %PYTHON_EXIT%
