---
name: ruflo-project
description: MarketMamba 的 Claude 主入口協作。需要背景工作、持久記憶、Claude/Codex 多代理或恢復既有工作時使用。
---

# Ruflo 專案協作

先讀 AGENTS.md、Current_State.md、Authority_Map.md。遵守使用者已核准的範圍，不重問已核准事項；新研究、資料、GPU、production、merge/push/deploy 不由任務派遣取得授權。

你是互動主控；沿用 builder / verifier / archivist 分工。先用 `node tools/project_agents/runtime.mjs doctor`。不要執行 `ruflo init --force`、`@latest`、直接 `dual run`、預設 daemon 或週期任務，也不要改全域 plugins。此接入沿用官方 DualModeOrchestrator，但用專案 adapter 管理程序、版本與證據。

1. 先完成必要 reconnaissance，確認 source clean/committed；主控將已核准任務寫入 `.artifacts/ruflo/<task>.json`。格式見 `docs/operations/ruflo.md`，每個 worker 明訂 prompt、platform、role、readOnly。`model` 可省略，沿用官方 CLI 預設；不能猜不存在的 model id。
2. `node tools/project_agents/runtime.mjs submit .artifacts/ruflo/<task>.json`：回傳 job id 後即在背景繼續。最多一個 active job，job 內最多 3 workers、1 writer、每 worker 最長 1800 秒。按依賴分批執行。
3. 每個 worker 在獨立 retained worktree；主控收集其 `.result.json`、`.patch`、`.files/`，檢查 source commit 與 hashes。Reviewer 的 prompt 必須要求讀 dependency artifact 中的 snapshot 與 patch，獨立驗證而非信任前一位 worker 的口頭結論。
4. `status <id>` 觀察；`cancel <id>` 停止；parent 中斷後用 `recover <id>` 對帳並停止有正確程序 identity 的 orphan。中斷不自動重跑。絕不以 registered、exit=0、記憶中的 completed 當作驗證通過。
5. 主控評估產物與測試後，才在已授權範圍套用指定檔案；不自動 merge。既有 worktree/history/artifact 不刪除。
6. 持久記憶使用專案 MCP 的 memory 工具或 `runtime.mjs memory ...`，namespace 含 task/branch，寫入 source commit、artifact paths、evidence class。結果與程序狀態由 adapter 自動保存；語意摘要由主控保存。Session hooks 僅記錄 session checkpoint metadata，不讀全文 transcript、不啟動模型或排程。

官方 worker 的 CLI 認證沿用既有登入。不能因可以讀認證而推定帳單或 quota；不傳入 API key 環境。API 執行後端 `agent_execute` 不在此 worker 路徑。

專案 DB 在 Git common root 的 `.artifacts/ruflo/memory.db`，跨工作樹共享；這是輔助協作記憶，不是第二套 project map，也不提供正式資料或模型 admission。
