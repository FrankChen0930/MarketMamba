# Ruflo Claude 主入口整合

核准：使用者於 2026-11-19 表示「我覺得可以，就照這樣做吧」。範圍為完整開發協作，非 production／資料／訓練／定期排程授權。

## 結構與 identity

Base f501a7eb28fab26c86f1bc776ad80a0ff13cfe81；隔離工作樹 `.worktrees/ruflo-integration`。
沿用既有 tools/tasks/docs/knowledge，不新增 top-level tree。`tools/project_agents/` 是必要的程序執行邊界；`tools/project_knowledge/` 只做治理檢查，不能承擔 worker 啟停。既有外部評估為歷史參考，不複製評估結果為正式驗證。
固定 Ruflo/CLI 3.77.1 與官方 Codex adapter 3.0.8；沿用 DualModeOrchestrator 的依賴／併發，不 fork Core。

## 執行項目

1. 先建立 `tools/project_agents/runtime.test.mjs`：限制、失敗依賴、真實子程序取消／逾時、日誌上限、輸出契約、記憶與恢復。先 RED 再實作。
2. `runtime.mjs`、`process.mjs`、`integration-contract.json`、`install.mjs`：固定入口／雜湊、官方 CLI、共享 durable Ruflo DB；背景 job manifests、狀態／取消／中斷對帳，不自動重跑不確定任務。寫入任務採隔離 Git worktree，審查固定產物。
3. 專案 `.mcp.json`、`.claude/settings.json`、`.agents/config.toml`、`.codex/config.toml` 與 `.claude/skills/ruflo-project/SKILL.md`。保留 AGENTS 與角色；CLAUDE 加入簡短協作入口。
4. 真實有界模型 smoke：Claude 派遣 Codex，獨立審查、輸出與記憶跨程序讀回。無 GPU／正式資料／API keys／排程。失敗不升格 PASS。
5. `docs/operations/ruflo.md`、Current_State、Authority_Map、Operations、Worktree_Register 同 change 更新；正常 tests → knowledge impact → knowledge health → review → commit。將乾淨完成分支切換至主入口供使用，不 merge/push/deploy。

## Review Focus

- PID reuse／重啟後不能任意 kill 其他程序；cancel 使用程序 start identity。
- 失敗／空輸出／不完整事件不能標完成；未知只作 interrupted。
- 不以記憶文字代替 artifacts／獨立驗證；跨 worktree DB identity 明確。
- 預設啟動不下載 @latest、不安裝全域 plugin、不觸發週期任務。
- 來源版本／hash 不符 STOP，工作樹 dirty 或未知來源 STOP；歷史證據與資料不修改。
