# V7 System Observability Foundation Implementation Plan

> Historical plan dated 2026-09-16. The implementation was reviewed for Git preservation on 2026-09-22; current source and deployment status are governed by `knowledge/00_Project_Map/Current_State.md` and `Authority_Map.md`. This checklist is not a live task authority.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立不依賴 C 結果的 V7 系統可觀測基礎：分級資料健康摘要、唯讀狀態 API、瀏覽器狀態頁，並完成一次受限 MAS Builder／Verifier 試用。

**Architecture:** 既有 `data_health.json` 先正規化成小型、版本化 publication JSON；FastAPI 只讀 publication，不掃 parquet；React 只呈現 API 真實狀態。MAS 僅在獨立 product worktree 製作通用狀態元件，外層 Codex 驗證後才精確整合。

**Tech Stack:** Python 3.12、FastAPI、React 19、Vite 8、Node 24、MAS `orchestrator.production_cli`

**Spec:** `tasks/plan.md` P2.0/P2.1；`research/v7/product-architecture-alignment.md` 第 4–7 節

## Global Constraints

- 禁止修改模型目錄、權重、Data、parquet、A/B/C 與 deliveries。
- 不部署、不 push、不新增排程、不改既有 V6.1/V6.2 endpoint。
- unknown 不得當 healthy；null 不得當 0；所有數字須有來源與日期。
- 保留主工作樹所有未提交內容，不 reset/clean/stash。
- MAS 固定 commit `3e61bcdc2e10ac89006eec498097d65f9fb95c1a`，不得改 MAS 核心／DB。
- MAS 出現 occurrence-specific Human gate 就停，不代批；Codex 繼續獨立工作。
- 所有 production change 採 TDD，完成前 fresh verification。

## Task 1：隔離與 authority preflight

**Files:** 建立 `/home/frank/projects/MarketMamba-mas-p2-run-status` worktree；更新本 task registration 的 `isolated_product_worktree` 與 `planned_exact_paths`。

**MAS exact allowlist:**
- `app/frontend/src/components/runStatusSummary.mjs`
- `app/frontend/src/components/RunStatusSummary.jsx`
- `app/frontend/src/components/runStatusSummary.test.mjs`

- [ ] 核對 main HEAD、dirty state、routing 與 MAS commit。
- [ ] 依 `superpowers:using-git-worktrees` 建立乾淨 worktree。
- [ ] 記錄 exact allowlist；denylist 保持 handoff 原規格。
- [ ] 兩個 worktree 分別跑 `git status --short`，確認互不污染。

**Acceptance:** 主工作樹 byte state 不變；隔離 worktree clean；registration 精確可審。

## Task 2：V7 data-health publication

**Files:**
- Create `V6/experimental/v7_health_summary.py`
- Create `V6/experimental/v7_health_summary_test.py`

**Interfaces:**
- `build_health_summary(document, *, data_id, generated_at) -> dict`
- `write_health_summary(source, output, *, data_id, generated_at) -> dict`
- schema `v7-health-summary-v1`

- [ ] 先寫 RED tests：QUARANTINE→degraded/可發布；BLOCK→blocked/不可發布；缺 `blocking` 必須失敗。
- [ ] 實作 BLOCK > QUARANTINE > WARN > INFO、日期／股票去重排序、reason 統計、graph/provenance。
- [ ] 寫入採 temp + fsync + replace；壞來源不得覆蓋舊好檔。
- [ ] 用 smoke `data_health.json` 驗證並輸出人類可讀數字。

**Verification:**
```bash
.runtimes/colab-2026.04/.venv/bin/python -m unittest V6.experimental.v7_health_summary_test
```

## Task 3：唯讀 V7 status API

**Files:**
- Create `app/backend/routers/v7.py`
- Create `app/backend/test_v7.py`
- Modify `app/backend/main.py`

**Interface:** `GET /api/v7/status` 讀取 `V7_RESULTS_DIR/health-summary.json`。

- [ ] RED tests：missing→200/not_ready/不可發布；valid→200；malformed/schema mismatch→503/error，禁止 stale fallback。
- [ ] 最小 router + additive include；不改既有 router。
- [ ] 驗證 API route 與既有 app import。

**Verification:**
```bash
.runtimes/colab-2026.04/.venv/bin/python -m unittest app.backend.test_v7
.runtimes/colab-2026.04/.venv/bin/python -c "from app.backend.main import app; print([r.path for r in app.routes if r.path.startswith('/api/v7')])"
```

## Task 4：MAS 受限 RunStatusSummary

**Files:** 僅 Task 1 exact allowlist 三檔。

**Interface:** props `{state,title,updatedAt,message,counts}`；狀態僅 healthy/degraded/blocked/not_ready/error；unknown→error；missing number→「—」。

- [ ] 用 production_cli 做 preflight／Advisor plan，禁止 legacy CLI。
- [ ] 若有新 Human gate，保存 occurrence 並停 MAS writes。
- [ ] Builder 先寫 Node built-in tests，再實作 accessible `role="status"` 元件。
- [ ] Verifier 獨立檢查 allowlist、語意、無假 0、無模型／結果存取。
- [ ] Codex 覆核 diff 並跑 Node tests、lint、build；任何越界變更拒收。

## Task 5：整合 V7 狀態頁

**Files:**
- 整合通過驗證的三個 component files
- Create `app/frontend/src/api/v7.js`
- Create `app/frontend/src/pages/V7Status.jsx`
- Modify `app/frontend/src/App.jsx`

**Interface:** route `/v7/status`；顯示來源日期、資料 ID、發布決策、影響計數與明確空／錯誤狀態。

- [ ] 先寫純 normalization contract tests。
- [ ] 使用既有 Axios client；頁面不得宣稱是 production V7 inference。
- [ ] additive route，不改 legacy route。
- [ ] 使用 browser-testing skill 檢查 desktop/narrow、鍵盤、healthy/degraded/not_ready/error、console。
- [ ] 跑 Node tests、lint/build、`git diff --check`、protected-path audit。

## Task 6：證據與進度

**Files:**
- Modify after verified work: `tasks/todo.md`
- Create `research/v7/system-observability-foundation-review-20260916.md`
- `AGENTS.md` 僅在使用者事後確認交付後更新。

- [ ] R1 記錄：3×648 chunks、3,734,190 rows、0 SHA failure、年度 6/6、季度 21/22。
- [ ] P2.0 僅在真 MAS Builder+Verifier 完成時勾選。
- [ ] P2.1 僅在 publication/API/page 全部驗證時勾選；P2.2 排名頁仍保持未完成。
- [ ] 列完整 changed-file/test/MAS disposition；不 commit/push/deploy。

## Time checkpoints / stop rules

- **1–2 小時：**完成 Tasks 1–3，health publication + API 可獨立驗收。
- **3–4 小時：**在 authority/browser 可用時完成 Tasks 4–5。
- MAS blocked 時不空等，改強化獨立 API/tests；絕不繞過 approval。
- 遇到既有 lint/build 問題只隔離記錄，不擴張成 legacy cleanup。
- 任何動作將觸及 protected/model/data/result 路徑時立即停該單元，保留已完成切片。
