# Ruflo：Claude 主入口開發協作

狀態：UNMERGED / NOT_DEPLOYED；僅開發工具接入，不是 production rollout。

## Identity 與邊界

固定 Ruflo/CLI 3.77.1、官方 `@claude-flow/codex` 3.0.8。沿用官方 DualModeOrchestrator 的依賴排序、併發分批與失敗取消；專案 adapter 只補程序群管理、產物驗證及持久 job state。不是 Core fork，也不是第二套 MAS。

Canonical authority 不變：AGENTS → Current_State → Authority_Map → contracts/tests/domain。`.claude/agents/` 原角色不覆寫。Ruflo memory 和 worker summary 均為非權威協作證據。

`tools/project_agents/integration-contract.json` 定義 acceptance bounds。主控每次明確提交一個已授權 job；同時一個 active job、job 內最多 3 workers、最多 1 writer、1–1800 秒/worker、最多 8 workers/job、stdout+stderr 最多 2 MiB/worker。預設 900 秒。失敗停止下游；無定期 audit/optimize、無自動重試、無 GPU/資料抓取或 production 權限。

此工作使用已有 tools/tasks/docs/knowledge；新增 tools/project_agents 是程序邊界，既有 project_knowledge 不應兼任程序管理。外部評估目錄只提供已安裝 dependencies 及歷史線索，驗收必須本次重跑。

## 安裝與主入口

已安裝 dependencies 的預設位置是 `/home/frank/projects/ruflo-evaluation/tools`。其他機器應先獨立安裝相同 exact versions，再將該 tools root 作為 install 的參數；installer 不下載、不修改全域 plugins。

```bash
node tools/project_agents/install.mjs
node tools/project_agents/runtime.mjs doctor
```

installation.json 保存 observedAt、source commit、contract hash、Node/CLI/adapter/官方 host binaries 的選定入口 hashes。升級任一 pinned 入口會 STOP，需重新核對／安裝；此清單不是所有 transitive dependencies 的完整 supply-chain attestation。

重新啟動專案 Claude Code；在 MCP 清單確認 `claude-flow`。專案 `.mcp.json` 只暴露 memory 工具；派遣經明確的 Bash command，不把 unrestricted agent_execute 當 worker。可用 `/ruflo-project` 或要求 Claude 使用該 skill。

獨立 Codex 主入口讀 `.codex/config.toml`；worker 由官方 `codex exec` 啟動，不需要再跑 `init --dual`。模型可省略，沿用既有 CLI 設定。登入／用量仍是各官方 CLI 的實際狀態；本整合不保證帳單渠道、額度或免費。

SessionStart 提示專案流程；PreCompact/SessionEnd 保存 metadata checkpoint 到本機與 Ruflo memory。不讀 transcript 全文、不產生模型摘要、不自行產生新任務。主控另外保存有語意的 handoff 摘要；worker 完成結果由 adapter 保存。

## 提交背景任務

先提交 source/config 變更，確保工作樹 clean；job JSON 放 ignored `.artifacts/ruflo/`，不得將密鑰放在 prompt。

```json
{
  "taskContext": "使用者已核准的局部開發任務",
  "timeoutSeconds": 900,
  "maxConcurrent": 3,
  "workers": [
    {"id":"builder","platform":"codex","role":"builder","prompt":"實作指定變更並執行 focused tests；回報檔案與實際結果。","readOnly":false},
    {"id":"verifier","platform":"claude","role":"verifier","prompt":"讀取 dependency artifact 中的 patch 與 snapshot，獨立檢查需求與正確性；不可僅信任摘要。","readOnly":true,"dependsOn":["builder"]}
  ]
}
```

```bash
node tools/project_agents/runtime.mjs submit .artifacts/ruflo/task.json
node tools/project_agents/runtime.mjs list
node tools/project_agents/runtime.mjs status JOB_ID
node tools/project_agents/runtime.mjs cancel JOB_ID
node tools/project_agents/runtime.mjs recover JOB_ID
```

submit 啟動 detached supervisor 並回傳 id；關閉互動主 session 後可繼續，但作業系統關閉不會自動重啟。沒有 OS service／cron／Windows Scheduler 安裝。

每個 worker 使用 `.worktrees/ruflo-workers/<job>/<worker>` 的 detached source worktree，固定在 submission source commit，不會讀到主控未提交的變更。不自動 merge、不自動清理。job state 記錄 purpose、source commit、owner job、workspaces、created/finished time；保留期／移除須按 GOVERNANCE 人工確認。

Reviewer 收到 predecessor 的文字及不可變 snapshot/patch 路徑；其 worktree 本身是原始 source。唯讀 Claude reviewer 僅有 Read/Glob/Grep，不能執行測試；需要實際測試時，將該步指定給 Codex worker，或由主控在受核准範圍重跑。

## 記憶與證據

Git common root `.artifacts/ruflo/` 保存：

- installation.json：已觀測的 runtime identity。
- memory.db：Ruflo durable memory，跨 worktree 顯式使用同一 DB path。
- jobs/<id>/job.json、submission.json、state.json、result.json、*.out、*.err、*.result.json、*.patch、*.files/：原始輸出、程序 identity、模型 session/thread、hash 與 retained 產物。
- sessions/：不含 transcript 內容的 session checkpoint metadata。

```bash
node tools/project_agents/runtime.mjs memory store --namespace project-handoff --key TASK_ID --value '包含 source commit、產物、實際驗證、限制的摘要'
node tools/project_agents/runtime.mjs memory retrieve --namespace project-handoff --key TASK_ID
node tools/project_agents/runtime.mjs memory search --namespace project-handoff --query '查詢內容'
```

本機 AgentDB 的 better-sqlite3 binding 缺失，實測 MCP bridge 寫入未能跨程序讀回。採官方 `CLAUDE_FLOW_DISABLE_BRIDGE=1`，使用 Ruflo SQLite 持久路徑；不宣稱 AgentDB controllers、神經學習或 HNSW 效能已啟用。SQLite 持久化／讀回與語意搜尋品質是不同驗證；不因輸出顯示 HNSW 就推定 HNSW performance 或 recall 已驗證。模型 exit=0 尚需完整 model-completion event／非空 final output。completed 是開發流程完成，不是獨立 correctness acceptance，更不是研究、交易或部署通過。

status 對 completed result 驗 hash；記錄腐損／缺失 STOP。cancel 驗證 supervisor PID/start identity 後送 SIGTERM；程序群在逾時／取消／完成後清理。parent 遺失時 recover 停止 identity 相符的已知子程序並標 interrupted，不自動重跑。若 leader 已退出而同 group 尚有無法驗證的 descendants，回報 UNKNOWN 並保留鎖，需人工檢查，不放行新 job。重新開一個 job 是新的執行決策。

限制：官方 CLI sandbox、工具限制與 project policy 共同約束 workers；此 adapter 不是防惡意程式的容器。沒有新的正式資料掛載，但現有系統可讀路徑並未全面隱藏。禁止將 credentials、正式資料或 protected models 當作測試輸入。kill/restart 保證只限已記錄程序 identity；不是機器斷電或跨機備援認證。
