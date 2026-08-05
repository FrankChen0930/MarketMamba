@echo off
:: ============================================================
:: MarketMamba V6.2 — 每日組合層（獨立於 V6.1，失敗不影響線上）
:: ============================================================
:: ⚠ 執行時間必須在 **21:30 之後**
::   實測 2026-08-05：19:35 時 TWSE 的 margin 與 daytrade 都還「尚未公布」，
::   21:11 才有當日資料。跑太早不會報錯——`_merge_margin` 會 ffill 昨天的值，
::   訓練端有當日 margin 而推論端沒有，形成 train/serve 不對稱。
::
:: ⚠ 必須排在 V6.1（PersonalOS_Daily）**完成之後**
::   V6.2 不自己抓資料，它吃 V6.1 的 run_daily_update 更新好的 raw parquet。
::   實測 V6.1 + 雙模型全鏈路約 33 分鐘（19:30 → 20:03）。
::
:: Task Scheduler 設定（與 daily_inference.bat 相同的關鍵項）：
::   - "Run only when user is logged on"（WSL2 需要有使用者 session，SYSTEM 帳號跑不動）
::   - "Do not start a new instance" if already running
::
:: 用法：
::   v62_daily.bat                 平日例行
::   v62_daily.bat --first-day     上線第一天（強制建倉）

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
