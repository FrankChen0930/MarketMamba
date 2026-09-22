# Market Mamba 使用 MAS：接入與故障交接

## 固定入口
共用交接目錄：/mnt/c/Users/Master/.codex/visualizations/2026/09/13/01a09bc5-6a7e-70f2-bfc8-f99cc83f00e4/mas-coordination
接收對象以該目錄 routing.json 的 mas_maintenance_thread_id 為準。Market Mamba 新對話 ID 尚未知；由該對話第一次接入時登記於自己的 registration 檔，再通知 MAS 接收對話核對並更新 routing。不要靠標題猜 ID。

本文件交給 Market Mamba 原對話，由它建立新開發對話並攜入需求、程式位置、分支、測試方式與 V7 排除範圍。此 MAS 對話不代替它挑功能或建立產品對話。

## 初次接入（功能寫入前）
1. 讀 routing.json、本文件與 README 的 production entrypoint 章節。使用 orchestrator.production_cli；orchestrator.cli 是另一個 legacy demo。
2. 建立 incidents 以外的 registrations/<Market Mamba thread id>.json，包含 thread_id、專案路徑、隔離工作目錄、產品 commit、MAS 完整 commit、Python 路徑、測試工具、計畫中的 exact paths。未知填 null；不要捏造。
3. 用已核准的獨立 Market Mamba 工作目錄做小型配套功能；模型、V7、訓練、實驗輸出均排除。不要讀寫模型內容來做試用。產品原對話提供必要的確切排除路徑。
4. MAS 執行版本需固定為一個乾淨 commit。現在已測試的功能基準是 c51befb；交接文件提交後會更新 routing 的 reference_commit。該 reference 不是每個任務自動升級指令。MAS 維護對話從自己的 repo 準備獨立 pinned checkout；Market Mamba 執行時 PYTHONPATH 指向它，不指向持續修改的維護分支。每個進行中任務記住完整 MAS SHA，不中途切版。
5. 核對官方 provider CLI 可用、專案 runtime 設定只針對選定 worktree、Python/測試工具與 MAS 命令目錄相容。不得輸出 credentials。Python 合成任務已有真實證據；網頁 build/browser checks 的支援必須實測，不能把 unittest PASS 當網頁驗收。
6. 先核對唯讀狀態，再進行規劃與 exact Plan 正常批准。一般進度授權不是偽造 Human Plan/WorkEnvelope/final acceptance 紀錄的理由。

唯讀狀態的已存在入口（替換成實際路徑與任務識別；沒有任務/DB時省略對應參數）：
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pinned-mas>/src <python> -m orchestrator.production_cli --project-root <isolated-product-worktree> --current-state --durable-store <selected-store> --task-id <source-task-id>

--resume-durable 及各 continue/approve 指令不是本文件授權的診斷操作。只能依當前 checkpoint 的合法下一步執行。source task id 與 machine-issued remediation id 要分別保存，不能互換。

## 出問題時
- 產品邏輯/測試錯誤由產品對話處理；MAS 派工/權限/保存/恢復/命令相容問題交 MAS 對話。分類不明標 UNKNOWN，一起附重現資料，不先大改框架。
- 先停止該工作單元的新寫入/重試；確認 MAS/Builder 沒有仍在寫相同檔案。不要 kill 未確認歸屬的程序。
- 建立 incidents/<UUID>/report.json，依 incident-template.json 填入。記錄 source task、issued task、revision/cycle、MAS SHA、產品 baseline/current SHA、DB路徑、最後 head/digest、命令與 exit、已套用變更與未知事項。log先去除credentials/私人資料；完整log留檔，用位置+摘要，不傾倒整段聊天。
- 保存產品現場與原始 DB。若需DB副本，使用 SQLite backup API／一致性backup方式，不單拷貝正在寫入的.sqlite檔。只讀快照診斷；不得手改正式 lifecycle SQL。
- 發送給 routing 指定的 MAS 對話：「MAS問題 <incident UUID>；report=<絕對路徑>；blocked=<是/否>；期待=<診斷/修復>」。工具使用 send_message_to_thread，threadId用routing的實際ID。訊息是通知，文件才是持久交接。
- 工具不可用時保留文件並告知使用者需轉交的精簡通知；不假裝送達。沒有配置定時監看，不承諾無訊息時自動巡檢。
- 不依賴讀另一對話完整歷史。接收端需要更多資料，用同incident下 requests.md 提出，並通知reporting_thread_id。

## MAS維護與恢復
MAS對話寫 incidents/<UUID>/response.json；產品對話寫自己的 resume-result.json，避免兩邊改同一檔。各檔先寫temp再rename替換，訊息重送同UUID不得重複修復/套用。

MAS回覆需含：問題歸屬、重現結果、修正commit、測試、適用版本、status，以及 resume_policy：
- SAFE_RESUME：已證明此狀態可用指定版本繼續，列明精確前置條件及現有CLI下一步。
- NEW_TASK_REQUIRED：保留可確認的產品成果，另建剩餘工作；舊state不強行搬移。
- CODEX_TAKEOVER：先確保MAS停止相同寫入，再由產品Codex接手並記錄一次救援。
- NEEDS_SEMANTIC_DECISION：列具體未定產品/風險問題，不代行使用者判斷。
- UNDETERMINED：尚未證明可恢復，不提供猜測的重跑命令。

MAS修復不得直接修改產品工作目錄。修好後通知reporting_thread_id；對方核對現場 hashes/head仍一致，建立/切換指定pinned版本，才執行明確恢復步驟。修復測試PASS不等於任何舊state都相容。產品迴報恢復結果；MAS據此close事件。
非阻擋問題可由產品繼續其他互不相依工作；不讓同一產品寫入同時交給MAS及Codex。

## 職責與授權
使用者已要求建立兩個對話的故障交接。允許兩者發送本範圍的問題、補件及修復結果；不授權發信/Slack或建立更多任務。產品對話不得自行改MAS核心；MAS對話不得改模型/V7。
用量 reset 只由MAS維護對話協調：weekly嚴格超過95%時，原授權最多使用一次。Market Mamba對話不可另消耗同一授權；shared usage-reset.json 與MAS memory都要讀，送出前存idempotency key，不確定結果用同key查明。
