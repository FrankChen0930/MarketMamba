# V7 系統可觀測基礎：執行紀錄

> Historical verification record for 2026-09-16. On 2026-09-22, source preservation added public API field projection and stricter schema/decision validation; tests were rerun. The hashes and test counts below describe the original run, not the later source revision. Current source/deployment status is in canonical Current State and Authority Map.

日期：2026-09-16
狀態：P2.1 完成；P2.0 仍未完成，因元件由核准後的 Codex takeover 交付，沒有執行 MAS Builder／Verifier。

## 交付

- `v7-health-summary-v1` builder：完整解析後才以 temp、fsync、replace 原子發布。
- 唯讀 `GET /api/v7/status`：missing 回 `not_ready`；壞檔或 schema 不符回 503/`error`，不沿用 stale 值。
- `/v7/status` 資料健康頁、payload normalization 與 `RunStatusSummary` 元件。
- 頁面明確不是模型訓練、推論或部署狀態；unknown fail closed、缺值顯示 `—`、真實 0 保留。
- R1：A/B 共 3×648 chunks、3,734,190 rows、0 SHA failure；集成年度 6/6 勝過單 seed，季度 21/22。

## Smoke

`.artifacts/v7-local-matrix/prepared-smoke/data_health.json` 發布結果：`degraded`、publish allowed；992 entries、336 dates、2 stocks；BLOCK 0、QUARANTINE 318、WARN 672、INFO 2。摘要可發布，但不得標成 healthy。

## 最終驗證

- health 4 tests、backend 3 tests、frontend contract/component 7 tests：passed
- scoped lint、production build、`git diff --check`：passed
- protected-path audit：沒有本批模型、權重、Data、parquet、A/B/C 或 deliveries 變更
- 真實瀏覽器：desktop 與 390 CSS px 無右側 overflow、console 空白
- degraded、not_ready、malformed error、healthy/真實零值狀態：passed
- status live region、標題、重新整理按鈕及鍵盤焦點：passed

完整 repo frontend lint 仍有 59 個既有錯誤；本批沒有擴大處理。npm audit 報告 11 個既有依賴風險（1 low、1 moderate、9 high），未自動改依賴。build 只有既有 bundle >500 kB 警告。

## MAS disposition

READ_ONLY Advisor 產生 `run-status-summary-minimal-v1`：

- source task：`01a0a57e-9b6c-7831-bc65-82c65eeff419`
- occurrence：`invocation-16ac081f6f9043a399840a771a7e0caa`
- checkpoint：seq1 `ADVISOR_PLAN_AVAILABLE`
- digest：`159a0f943adab0f687b8c201195e7329cee51af2410e3f1de6a7ee7fa17f98b9`
- 使用者已核准 occurrence；Builder／Verifier 未執行

linked worktree 的 `.git` pointer 觸發 MAS ENOTDIR。事故 `ab8b241a-941c-4449-9d8f-9e8d0cc5ea31` 採 `CODEX_TAKEOVER`；確認無 active writer、baseline 與 DB digest 後，只在三檔 allowlist 完成一次 Codex rescue。原 lifecycle DB、pending Plan、checkpoint 未修改，也未恢復舊 checkpoint。

三檔 SHA256：

- `RunStatusSummary.jsx`：`5af162772c1c820d04165864f7efe4435a07e7d78ff762ad39f5911c37cdeb03`
- `runStatusSummary.mjs`：`ac58c0278ebcce3b2c670b168fefb4be160e06e733ee2a25f6e89b67cf7d35dd`
- `runStatusSummary.test.mjs`：`27beac3042165c3ae9191aaed588b48755cc3db1ed44b321a8ff4ba6ab1a59dc`

因此 P2.1 完成；P2.0 仍需另一次真正成功的 MAS Builder／Verifier 閉環。

## 時間校準

- 開始：2026-09-16 01:30:50 +08:00
- health/API checkpoint：20 分 30 秒
- 核准後元件接管、整合、瀏覽器驗收與最終查核：約 35 分 56 秒
- 全批次：約 56 分 26 秒

下一個離席批次應規劃三個可獨立驗收薄切片，或一個約 90–150 分鐘的核心系統批次。
