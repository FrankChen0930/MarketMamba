# MarketMamba — Project Agent Policy

> 本檔只保存穩定的安全、協作、驗證與知識維護規則。
> 專案現況與 authority 不在此重複；請使用 canonical project map。

## 開工順序

進行 substantive work 前，依序讀取：

1. `knowledge/00_Project_Map/Current_State.md`
2. `knowledge/00_Project_Map/Authority_Map.md`
3. 任務相關的 machine contract、manifest、artifact 與 executable tests
4. [`knowledge/01_Domains/`](knowledge/01_Domains/) 中的任務相關 domain note

先確認 repository root、branch/HEAD、`git status --short --branch`、worktrees、實際 runtime 與資料 realpath。`AGENTS.md` 是操作契約，Current State 是目前狀態，Authority Map 決定衝突時誰有權威；README、檔名、未追蹤檔與聊天記憶不能單獨升格成 authority。

## Reconnaissance before creation

建立 top-level directory、implementation、workflow、experiment/report tree、project memory、architecture、replacement version、backup folder、ZIP snapshot 或 worktree 前，先搜尋現有 canonical implementation、同用途目錄與流程，閱讀 Authority Map，判斷能否沿用既有結構。仍需新增時，在 coherent change 中記錄既有結構不能承載的原因與 artifact identity。禁止以 `final/`、`final2/`、`new/`、`backup-new/`、`v2-new/` 逃避正式 experiment/artifact contract。新 worktree 只用於邊界清楚的隔離工作，先查現有 worktree 是否已服務同一目標。

Project map 是 current-state／navigation authority。實作語義仍以 versioned machine contracts、manifests、hashes 與 executable tests 優先；不要用 explanatory prose 覆蓋 machine evidence。

## 協作方式

- 一律以繁體中文溝通。
- 改動前先說明會改哪些檔案、原因與驗證方式；取得核准後可自主完成核准範圍，不必每個小步驟重問。
- 診斷先行，一次只改一個可歸因變因。正式實作應避免不必要的反覆重訓與外部成本。
- 結果異常地好時，主動檢查 leakage、PIT、universe、label timing、source provenance、selection bias 與 evaluation window。
- 誠實呈現 `STOP`、限制、未知與 evidence class；不得為了繼續流程放寬 gate。
- 長任務提供簡短進度；最終回覆必須自包含，列出變更、驗證、未處理範圍與風險。
- 除非明確授權，不 push、merge、deploy、啟用排程、啟動 GPU、購買資料、消耗 usage reset、增加通知對象或改變 production authority。

## Authority 與狀態語言

- 文件衝突時，依 `Authority_Map.md` 的 hierarchy 判定；不得依檔名、日期或 `final` 字樣猜測。
- Canonical authority 缺失或互相矛盾時，記錄 ambiguity 並 fail closed，不得靜默選用 stale prose。
- Branch-local implementation 必須標記 `UNMERGED`、`NOT_DEPLOYED`，除非有可驗證 evidence 證明相反。Code presence 不等於 production authority。
- 外部 scheduler、deployment、service 等 runtime observation 必須包含 `observed_at`，並遵守 project map 的 freshness rule。
- Research evidence、historical simulation、paper/forward evidence 與 verified trading evidence 必須分開表達，不得互相升格。
- 歷史 artifacts 不修改、不以新名稱重標，也不得因結果漂亮而升格成 current evidence。

## 不可破壞的安全邊界

- Production 路徑是紅線。新能力優先採 additive、shadow 或 feature flag，不得順手改既有 endpoint、排程、資料流或 success decision。
- `V6/models/` 與正式模型權重不可修改或刪除，除非使用者逐次明確授權。
- 不得降低 zero-tolerance、acceptance threshold，或為單一案例加入特例。
- Execution、tradability、OHLCV、universe、fundamentals 與 PIT evidence 一律 fail closed；缺少合格證據時不可推定可交易或正確。
- 不得把 historical-simulation readiness 解讀成 strict executable/trading readiness。
- Line Notify 已停止服務，不新增或建議 Line Notify。

## 資料與 artifact 保護

- Artifact 分類與 manifest 規則見 `knowledge/GOVERNANCE.md`：`CANONICAL`、`GENERATED`、`TEMPORARY`、`ARCHIVE`、`EXTERNAL_LARGE_DATA`、`UNKNOWN`。新實驗產物記錄 source commit、contract/config、輸入 identity/hash、環境、split/support、model identity、outputs 與 evidence class；未知即標 `UNKNOWN`。

- 刪除前先建立精確 inventory，確認 authority、manifest references、唯一副本與重建方式。
- 永久保留 source/history、contracts、tests、manifests、reports、PIT/provenance/universe/labels、正式 predictions、必要 logs/checkpoints 與歷史研究證據。
- Runtime、venv、temporary extraction、partial、stale cache、重複 matrix 與被取代 bundle 只有在確認可重建且無 authority reference 後才可清理。
- 備份是否可刪要比較鍵集合／內容覆蓋，不只比較列數、檔名或大小；檢查工具本身也要驗證。
- 不覆寫共用輸出；smoke/partial/experimental outputs 必須使用獨立 namespace。

## Git 與 worktree

- Git 是 source、config、contract、governance 文件的版本紀錄。完成 coherent substantive unit 後 commit；push 與 deploy 各需獨立授權，不能由「已完成」推得。
- 開工先看 `git status --short --branch`、worktrees 與實際 branch context。
- 主工作樹可能包含使用者既有變更；保留它們，不使用 destructive reset/checkout 覆蓋。
- 只 stage 指定檔案，禁止 `git add -A`。
- Correctness、research experiment、production-system work 與 knowledge migration 優先使用隔離 worktree。
- 移除 worktree 前確認 clean、commits 已保留且 branch/history 不會遺失。
- Commit 保持 atomic；禁止未授權 push、merge、rebase 或 deploy。
- Worktree 的 purpose、branch、base、owner/task、created date、current state、authority、merge/artifact dependency、retired date 與 safe-removal preconditions 依 `knowledge/GOVERNANCE.md` 記錄；既有八個 worktree 不因本次治理工作移除。

## Runtime contract

- Canonical development/research checkout 是 WSL-native `/home/frank/projects/MarketMamba`。既有 Windows operational boundary 是 `D:\Desktop\work\ProjectForMe\MarketMamba`，其 Scheduler/WSL action 只採用有 `observed_at` 的 runtime observation；不自行猜兩個 checkout 等價。
- `MARKETMAMBA_DATA_ROOT`、`Data/` symlink/realpath、source commit 與環境 identity 必須顯式記錄。Colab 是獨立訓練 runtime，使用 exact lock 與 bundle identity；本治理契約不搬動 runtime/data。

## 實作與測試

- 先建立可失敗的驗證，再修實作；修正後執行 focused tests、相關 regression 與風險相稱的完整驗證。
- 驗證輸出必須包含實際數值與 artifact identity，不能只顯示「成功」。
- Source/data semantic change 要測 PIT、publication time、universe、label horizon、provenance 與 fail-closed behavior。
- Resume/checkpoint change 要測 round-trip、kill/restart、corrupt/missing pointer、orphan recovery、contract mismatch、matrix mismatch 與 seed isolation。
- 排序／Top-N 必須有 deterministic tie-break；線上與研究模擬使用相同口徑。
- 無法計算的風險／不確定性使用保守語義，不能以 0 或 truthiness 讓未知值通過 gate。
- 完成前執行 syntax/schema、tests、artifact/hash、link/reference、`git diff --check` 與 clean-worktree 驗證；不要只相信先前輸出。

## 訓練與 Colab 穩定契約

- Bundle 必須 self-contained、safe-extract、逐檔 size/SHA-256 驗證，並保留 source commit、environment、contract 與 matrix identity。
- Matrix 採 build-once/reuse：identity 不變時持久化重用；partial 不得視為 complete。
- Checkpoint 先 local serialize/fsync/hash，再 durable copy/reverify，最後 atomic pointer；保留 latest、previous、best。
- Resume 必須驗 contract/matrix identity。Validated orphan 可依明確 policy 恢復 latest，但不得自行升格 best 或 formal result。
- 使用 BEST-not-LAST，selection/secondary metric、split、purge、maturity、paired evaluation 與 thresholds 由 frozen contract 決定。
- Stage completion 必須驗證 artifacts，不以「檔案存在」作唯一判準。
- Durable predictions、metrics、telemetry、logs、manifests 與 evaluation report 驗證完成後才可釋放 GPU。

## Knowledge-impact Gate

完成 substantive work 前逐項詢問：

- Semantics 是否改變？
- Authority 是否改變？
- Subsystem boundary 是否改變？
- Active runtime state 是否改變？
- Correctness gate 是否改變？
- 是否新增或 supersede canonical artifact？
- 是否 deprecate 任何能力？
- Major blocker 或 next decision 是否改變？

任一答案為 YES：在同一 change 更新相關 canonical knowledge，至少檢查 `Current_State.md` 與 `Authority_Map.md`。全部為 NO：通常不需更新 knowledge。Pure formatting、comments、pure refactor、tests-only 與不改 public semantics 的 internal rename 通常不觸發。

Current State、active objective、architecture decision、experiment conclusion、blocker、next action、canonical implementation、authority boundary、runtime assumption 或 worktree/branch responsibility 有變化時，memory update 是同一 coherent change 的完成條件。沒有 knowledge 影響時，在 handoff 明寫 `knowledge_update: not_required` 及理由，不得沉默省略。Runtime observation 只能在實際觀測後更新 `observed_at`。

完成 substantive work 時依序：正常 tests → `check_knowledge_impact.py` → 更新受影響 knowledge，或以 `NO_KNOWLEDGE_CHANGE_REQUIRED` 附非空白理由 → `knowledge_health.py --ci` → 確認目標 worktree clean。Acknowledgment 只記錄判斷，不會略過 structural FAIL。

詳細 severity、authority supersession、runtime observation 與 generated-index 規則見 [`knowledge/GOVERNANCE.md`](knowledge/GOVERNANCE.md)。治理工具通過仍不取代人工檢查新語義與 evidence class。

## Handoff contract

每次 substantial task 結束，在 final response 並視需要在 canonical state 留下：`Completed`、`Changed`、`Decisions`、`Evidence`、`Remaining`、`Blockers`、`Next recommended action`、`Project memory updated`、`Authority map updated`、`Git status`、`Commit`、`Push`、`Worktree`、`Runtime/environment`。不適用寫 `N/A`。既有 dirty/untracked/staged 狀態要明列，不為 handoff 建立重複文件樹。
