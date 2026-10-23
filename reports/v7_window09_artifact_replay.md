# Window09 Durable Artifacts & Offline Replay

狀態：UNMERGED / NOT_DEPLOYED；main 僅整合固定 W08 bdea30d，前後101tests、
19hashes、24cases 及 independent34tests GO。Window09 從同一 SHA 建隔離分支，
closure 文件不先推進 main。現有29工作樹無同範圍工具，沿用 experimental/research/
reports 與唯一 Backlog，無新 Registry/Orchestrator。

U1 使用 E1 sha256/atomic_json/fsync，新增小型合成 Bundle Schema 與 publisher。
Profile 明列必要產物，未訓練案例不假造 training logs 或 checkpoint。所有附件均驗
bytes/hash，JSON 再驗 schema ID。stage 與失敗 attempt 診斷保留；同 Run 的 flock
及同檔案系統 rename 為發布邊界，重試驗同一內容，衝突或已完成損壞均不覆寫。
External references 必須 allowlist、拒 symlink/parent traversal、每次核驗真實 bytes；
只有 hash 沒有 bytes 回報 NOT_RESTORABLE。範圍是 local POSIX filesystem，不承諾
網路磁碟／斷電硬體或對抗可直接改寫 storage 的外部使用者。

U1 18focused/independent18PASS；第一輪 review3Required 已用 RED evidence 修復：
路徑 .. 越界、manifest/publication 葉 symlink、JSON 1e400 overflow。Critical0/
Required0，1bounded rework。U2 replay/restore 是下一已授權單元。

Artifacts verified、training complete、research admitted 分開。generated_at 是發布時
本機實際 UTC，logical observation/evaluation 時間是合成研究情境，不能當歷史
實際預測發行證據。既有 E1 store 使用 weights_only=False；本輪不呼叫其 load/save
或 orphan recovery，不載入未知 PyTorch payload。永久位置／保存年限／GC／資料與
研究 Admission／完整 GPU runtime acceptance 留給 Human。

恢復：cd .worktrees/v7-window09-artifact-replay，核對 HEAD/clean，讀 Current State、
Authority、Charter、research-program Window09 與 window09-evidence.json。
StrictSTOP/E1matrixBLOCKED/H2unapproved/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED；不讀正式 payload、不搜尋 E1 Matrix、不開始 Window10。
