# Worktree Register

## 三份DRAFT複審文件修補 spec_fixer2 — 2026-11-24

- Path `.worktrees/ruflo-workers/run-mwuu88xi-da927f/spec_fixer2`；branch detached HEAD，base `abc12f50a8246bbece3f187636f697c9a536a235`；owner Codex builder，task 五項文件修訂；created date UNKNOWN（指派日2026-11-24），retired N/A。既有worktree，起始clean，未新增worktree。
- State：DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；僅三規格及必要knowledge的未stage patch，Human明令no-commit／push／merge。無研究程式、訓練或外部操作，未攜新資料artifact；凍結彙整沿用Current State，不新增檔案樹。
- Dependencies：獨立複審與Human凍結／後續執行授權；patch未提交即尚無retained commit。Safe removal須patch已由Human保留／整合且確認無唯一產物、clean及明確授權；本輪不清理worktree。

## 專案清單 skill_writer — 2026-11-24

- Purpose/task／owner：Human／Codex builder，`job-run-mwuchypx-211f4f/skill_writer`，洩漏稽核與 Colab preflight 兩份 skill、共用指標及 knowledge 導航；沿用 `.worktrees/ruflo-workers/run-mwuchypx-211f4f/skill_writer`，detached branch、base/HEAD `8eff9fef3386d5b852b79f5c66d09a0893891122`。本輪接手2026-11-24，實際建立時間 UNKNOWN，retired N/A。起始 clean，僅文件 patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；不 commit／push／merge。
- Authority／dependency：Human EXECUTE_APPROVED bounded task；清單連回既有 contracts／reports／tests／Human 單一需求檔，沒有新研究或 production authority，沒有 runtime observation。無外部 artifact 產生或執行中 Colab／Drive 依賴變更；候選須後續獨立 session／Human 驗收，不改既有研究 NO_GO。新增 skill 被既有 ignore 規則涵蓋，僅兩個指定檔案使用 intent-to-add 供 patch 檢閱，未暫存內容。
- Safe-removal preconditions：先保存未提交六檔 patch、核對引用與完成 retirement review；目前 dirty 不可移除，既有其他 worktrees 不動。Runtime：WSL-native、`/usr/bin/python3` 3.12.3；文件 pytest 借既有 pilot09 venv 但不借其研究 evidence，CPU／CUDA_VISIBLE_DEVICES=''。MARKETMAMBA_DATA_ROOT unset；Data 為本 worktree 一般目錄、realpath 為同 worktree/Data，不代表 canonical formal data；未讀改其中資料。

## P4a 第 13 次複審 schema／metadata builder — 2026-11-24

- Purpose/task：Q5 完整巢狀 schema、marker 溢位／解析失敗重排練與 metadata symlink 防護；owner Human/Codex builder，job-run-mwtn0pir-48e95f。沿用 `.worktrees/ruflo-workers/run-mwtn0pir-48e95f/builder_p4a_gate3`，detached HEAD/base `40d621cc9ac586e45fff1e095f59465b388094a1`；created 2026-11-24（本輪接手，非宣稱新建），retired N/A。起始 clean；依 no-commit 指示保留本任務 dirty patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：Human EXECUTE_APPROVED bounded task；沿用既有 runner／tests／source-compat／requirements／工程報告與 knowledge，未建立新實作樹或 worktree。不改 frozen research／environment／budget、36-fit 採用與64-fit SHA；未動 rolling_seq_scores／portfolio_eval／Data／V6/models。無 GPU／網路／Colab／production／scheduler／additional agents／wakeup／recurring jobs／commit／push／merge／deploy。
- Merge/artifact dependency：獨立複審與 Human 現場排練／acceptance 待完成；GENERATED 宣告綁定 working bytes，後續 deliver 重算；CPU logs／合成資料／in-memory mutations 在 `/tmp/p4-gate3-*`／pytest 獨立暫存目錄，完整輸出收入[工程報告](../../reports/v7_p4a_colab_engineering.md)。Safe-removal preconditions：先保存未提交 patch／必要證據、處理引用並完成 retirement review；目前 dirty 不可移除，其他 worktrees 不動。

## P4a 正式前自動排練閘門 builder — 2026-11-24

- Purpose/task：notebook 正式前 Q3 W120／W250 mamba2＋Q5 自動排練閘門、W≠60 前置檢查與來源相容證據；owner Human/Codex builder，job-run-mwtk8k4r-a80ee8。沿用 `.worktrees/ruflo-workers/run-mwtk8k4r-a80ee8/builder_p4a_gate`，detached HEAD/base `bf0114dad9dc2d9bef7502c21d6fd36737eb0458`，created 2026-11-24、retired N/A。起始 clean；依 no-commit 指示保留 dirty patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：Human EXECUTE_APPROVED bounded task；不改 frozen research／environment／budget、36 完成採用條件與64-fit queue。不改 rolling_seq_scores／portfolio_eval／正式 Data／V6/models；無 GPU／網路／Colab／production／scheduler／additional agents／wakeup／commit／push／merge／deploy。沿用既有 source／tests／docs／report／knowledge 容器，沒有新 worktree 或平行 artifact tree。
- Merge/artifact dependency：獨立複審、Human 真實 CUDA／官方 mixer 排練及 acceptance 未完成；GENERATED 相容宣告由既有工具重產於原候選路徑。TEMPORARY CPU logs／合成 notebook／mutations 在 `/tmp/p4-gate-*`，完整輸出見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Safe-removal preconditions：先保存未提交 patch／必要證據、處理引用並完成 retirement review；目前 dirty 不可移除，其他既有 worktrees 不動。

## P4a Q3 事故 fixer／排練 builder — 2026-11-23

- Purpose/task：尾端 NaN 列身分修補、真實資料 single-fit 排練入口、36 完成接續與來源相容宣告；owner Human/Codex builder，job-run-mwt8u5js-7f212c。沿用 `.worktrees/ruflo-workers/run-mwt8u5js-7f212c/fixer_p4a_q3`，detached HEAD/base `685f6ece1faa7ed39ff7ebff7056b2c4ad7eb4b2`，created 2026-11-23、retired N/A。起始 clean；依 no-commit 指示保留 dirty patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：Human EXECUTE_APPROVED bounded task；研究／queue／matrix／環境／預算契約與正式 completed gate 維持，新增本事故 36 完成採用檢查及 TEMPORARY 排練。不寫正式 Data／V6/models；無 GPU／網路／Colab／production／scheduler／additional agents／wakeup／commit／push／merge／deploy。沿用既有 implementation／tests／report／knowledge 容器，無新 worktree／artifact tree。
- Merge/artifact dependency：獨立審查、Human 真實 CUDA 排練與 acceptance 未完成；GENERATED 宣告維持既有 `tasks/p4a-source-compat-20e6b54.json` 路徑，由工具重產綁定工作 bytes。CPU logs `/tmp/p4-q3-*`、合成資料與 rehearsal 目錄為 TEMPORARY，不是研究 evidence。詳見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Safe-removal preconditions：先保存未提交 patch／必要證據、處理引用、完成 retirement review；目前 dirty 不可移除，既有其他 worktrees 不動。

## P4a 來源相容升級 builder — 2026-11-22

- Purpose/task：4e56e9d → HEAD 來源相容宣告、舊 lineage 接續與 CPU RED/GREEN；owner Human/Codex builder，job-run-mwrxutx3-d28fa4。沿用 `.worktrees/ruflo-workers/run-mwrxutx3-d28fa4/builder_p4a_upgrade`，detached HEAD/base `20e6b5408d9f9b13d56290a72a2dc313c35d1b94`，created 2026-11-22、retired N/A。起始 clean，依 no-commit 保留 dirty patch；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：Human 明確核准本次 bounded task；FROZEN P4a／OBSERVED runtime bytes 與原 ledger／checkpoint／completed-fit gate 不變。既有容器新增來源證據 helper 與 GENERATED 宣告，訓練與評估不複製；無 GPU／網路／Colab／additional agents／資料取得／正式 Data／V6/models／production／scheduler 操作；不 commit／push／merge／deploy。
- Merge/artifact dependency：獨立審查及 Human acceptance；原 02_formal ZIP 與 budget 供 runtime lineage 驗證，實際 Drive 現況 UNVERIFIED。`tasks/p4a-source-compat-20e6b54.json` 為本輪產生的候選來源證據；TEMPORARY logs／合成 bundles 在 `/tmp/p4-upgrade-*` 與 pytest 暫存目錄，實際輸出見 [工程報告](../../reports/v7_p4a_colab_engineering.md)。Safe-removal preconditions：保存未提交 patch／必要證據、處理引用並完成 retirement review；目前 dirty 不可移除。

## P4a 進度 builder — 2026-11-22

- Purpose/task：後續 Colab 交付串流／時間進度快照／speed-test 獨立帳本及 CPU RED/GREEN；owner Human/Codex builder，job-run-mwrof3ay-8d0c43。沿用 `.worktrees/ruflo-workers/run-mwrof3ay-8d0c43/builder_p4a_progress`，detached HEAD/base `f6a173eab03332bc3c50d78885a1d865366d2518`；created2026-11-22、retired N/A，起始clean，依 no-commit 保留dirty patch。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：Human 持續需求及既有 FROZEN P4a／OBSERVED runtime 契約；只新增觀測與模式帳本路由，研究／production authority、runtime observed_at 不變。沿用既有 implementation／tests／docs／reports／knowledge 容器，不新建平行流程或 worktree；無GPU／網路／Colab／additional agents／commit／push／merge／deploy／正式Data／模型／scheduler。
- Merge/artifact dependency：獨立審查與 Human acceptance；後續交付重新綁 source identity，正在跑的原始02_formal不可換版。CPU logs `/tmp/p4-progress-*.log`、合成pytest artifacts與 `/tmp/p4-progress-notebook.ipynb` 是 TEMPORARY／ENGINEERING_ONLY，詳見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Safe-removal preconditions：先保存未提交patch與必要證據、處理引用並完成retirement review；目前dirty不可移除。

## P4a 三項 REQUIRED fixer — 2026-11-22

- Purpose/task：獨立複審三項 REQUIRED：A/B 列支援、C 交集後排名、completed-fit 終止／配方／history驗證；owner Human/Codex builder，job-run-mwric0jj-32d745。沿用 `.worktrees/ruflo-workers/run-mwric0jj-32d745/fixer_p4a_full`，detached HEAD/base `d82d0209cba6686c10fc1a01c78172446a6796fa`；created2026-11-22、retired N/A，起始clean，依 no-commit 保留dirty patch。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：既有 FROZEN P4a contract、OBSERVED runtime與CPU合成tests；研究契約、P3 adapter、portfolio engine、正式Data、V6/models、production、scheduler不改。無GPU／網路／Colab／additional agents／commit／push／merge／deploy。
- Merge/artifact dependency：獨立複審與Human acceptance，後續交付須重新綁定source identity；TEMPORARY證據在 `/tmp/p4-required-*` 與pytest隔離目錄，不是研究artifact。Runtime與實際輸出見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Safe-removal preconditions：保存未提交patch及必要證據、處理引用並完成retirement review；目前dirty不可移除。

## P4a 完整固定佇列 builder — 2026-11-22

- Purpose/task：Human核准一次完成Drive資料外置、epoch接續、固定64fits/Q5、P4 evaluator與兩檔交付；owner Human/Codex builder，job-run-mwrgy396-cf1213。沿用 `.worktrees/ruflo-workers/run-mwrgy396-cf1213/builder_p4a_full`；detached branch/HEAD/base `f333421fafdcb02f716b16376ff65717021edfa4`，created2026-11-22、retired N/A。起始clean，依no-commit指示保留dirty patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：研究語義為 [FROZEN P4a contract](../../tasks/p4a-colab-contract-20261121.json) 與既有OBSERVED環境契約；不改P3/protected Data/V6 models/production/scheduler authority。無GPU/網路/Colab、無額外agents、無commit/push/merge/deploy。
- Merge/artifact dependency：獨立審查、Human真實資料上傳／deliver與Colab現場驗證仍待；TEMPORARY logs `/tmp/p4-*.log`，合成artifacts在pytest隔離目錄；CPU環境與實際測試見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Safe-removal preconditions：先保存未提交patch與必要evidence，處理引用並完成retirement review；目前dirty不可移除。

## P4a Drive 分次交付 builder — 2026-11-22

- Purpose/task：既有deliver／notebook的Drive路徑、共用budget、wheel回退、ZIP大小／identity預檢及CPU RED/GREEN；owner Human/Codex builder，job-run-mwqscvk1-a3db0f。沿用 `.worktrees/ruflo-workers/run-mwqscvk1-a3db0f/builder_p4a_drive`，detached base/HEAD `df78d6cb3bb706f6f4d58b94310c3d352aab8385`；created2026-11-22，retired N/A，起始clean；依no-commit指示保留dirty patch。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：P4 DRAFT機器契約、既有Human OBSERVED runtime契約與本輪CPU合成tests；無新GPU／Colab／研究／production authority。沿用research_lab、docs、reports及knowledge容器；不新建implementation／artifact tree或worktree。無protected source、formal Data、V6/models、scheduler變更。
- Merge/artifact dependency：獨立審查、Human預算／正式契約凍結及另行授權現場Colab；TEMPORARY logs為 `/tmp/p4-drive-*`、合成zip／notebook／README在pytest隔離目錄，依賴既有離線 `/tmp/p4-notebook-deps`，無網路／未改venv。Safe-removal preconditions：先保存未提交patch與必要證據、處理引用並完成retirement review，目前dirty不可移除；不commit/push/merge/deploy。

## P4a 兩檔交付 builder — 2026-11-22

- Purpose/task：擴充既有bundle產生器為ZIP＋notebook、X.npy去重、安全bootstrap／cp313自檢與CPU反證；owner Human/Codex builder。沿用 `.worktrees/ruflo-workers/run-mwqr9yiu-8c464e/builder_p4a_delivery`，detached HEAD/base `74795a4a56f25585000d26d281e305381de6230a`；created2026-11-22，retired N/A，起始clean，依明確no-commit指示保留dirty patch。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：DRAFT P4研究machine contract、既有Human OBSERVED Colab runtime契約與本輪CPU合成tests；無新GPU／Colab／production authority。沿用research_lab、docs、reports與knowledge容器，既有pack可擴充，未建立平行implementation／artifact tree／worktree；沒有改動protected source、V6/models或formal Data。
- Merge/artifact dependency：獨立審查、Human研究凍結及另行授權Colab ZIP／編譯／Drive現場驗證；本輪TEMPORARY logs、synthetic ZIP／notebook在 `/tmp/p4-delivery-*` 與pytest隔離目錄。nbformat及相依套件從本機既有Anaconda／pip cache複製到 `/tmp/p4-notebook-deps`，無網路／未改任何venv。Safe-removal preconditions：先保存未提交patch與必要證據、處理引用並完成retirement review，目前dirty不可移除。不commit/push/merge/deploy。

## P4a Colab 操作修訂 builder — 2026-11-21

- Purpose/task：可選外部 SHA、Drive wheel-dir、獨立 TBD Colab runtime 契約與 speed mismatch 記錄；owner Human/Codex builder。沿用既有 `.worktrees/ruflo-workers/run-mwqi30jf-aa0d7f/builder_p4a_colab_fix`，detached HEAD/base `36aa27da6e26c1877332c9cdea927e6409c3d105`；created2026-11-21，retired N/A，起始 clean，現在依明確 no-commit 指示保留 dirty patch。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。
- Authority：DRAFT P4 machine contract／candidate runtime contract（TBD）／CPU tests；沒有 GPU／Colab／production authority。未新建 worktree、未修改 P3 gate、正式 Data 或 V6/models。
- Merge/artifact dependency：獨立審查、Human pip list／runtime 自檢、研究凍結與另行授權現場驗證；本輪 TEMPORARY logs／合成 artifacts 在 `/tmp/p4-colab-fix-*`。Safe-removal preconditions：先保存未提交 patch／必要證據、處理引用並完成 retirement review，目前 dirty 不可移除。不 commit/push/merge/deploy。

## P4a Colab builder — 2026-11-21

- Existing path `.worktrees/ruflo-workers/run-mwqd0vuy-7d9826/builder_p4a`；detached HEAD/base `e4f21c3747124f4c6cc46a873eae891697910067`；owner Human/Codex builder；task P4 bundle／Colab executor／GRU／CPU tests／DRAFT規格；created2026-11-21，retired N/A。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；原候選已提交於 `8cb61c4`；本輪不 commit/push/merge/GPU/Colab/network/agents。
- Reuse既有research_lab、tasks、docs/operations、environments與knowledge結構；新增P4模組原因在[草案](../../tasks/spec-p4a-colab-20261121.md)。沒有新增worktree／top-level artifact tree；驗證logs及合成bundle在`/tmp`，不修改P3既有結果。Authority：DRAFT候選machine contract／CPU tests，無研究結果。
- Merge/artifact dependency：獨立審查、Human／Claude凍結與GPU／wheel／Drive現場驗證；safe removal須先保存未提交patch及必要證據、處理引用並明確retirement review，候選 `8cb61c4` 與第一輪修補 `05cb840` 已保存；第二輪修補已提交為 `a76e98d`，仍不可直接移除。


- Review fixer：既有 `.worktrees/ruflo-workers/run-mwqe7o12-d33ef8/fixer_p4a`，created2026-11-21，detached base/HEAD `8cb61c48b2b302338600a91b3ad59dc350d09a6c`，owner Human/Codex，task 審查 REQUIRED／MINOR 修補。起始 clean，第一輪 patch 已保存為 `05cb840`；retired N/A；authority 仍 DRAFT / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。Merge/artifact dependency 為獨立審查、CPU RED/GREEN／mutation logs 及後續 Human 決策；safe removal 需保存 patch／證據、確認引用與 retirement review。沒有新建 worktree。

- Second review fixer：沿用 `.worktrees/ruflo-workers/run-mwqew35j-886363/fixer2_p4a`；detached base/HEAD `05cb840b5f4309844794d30bd4762abbcde49a13`；owner Human/Codex builder，created2026-11-21，retired N/A。Task：第二輪 REQUIRED／MINOR bounded 修補；起始 clean，依 no-commit 保留第二輪 dirty patch，後續獨立 commit 尚無 hash。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。Authority：DRAFT candidate／CPU tests；merge/artifact dependency：獨立複審、RED/GREEN／mutation logs、Human 研究與現場環境決策。Safe removal：先保存 patch／必要證據、處理引用及 retirement review；不得直接移除。未新建 worktree。

## P3 修訂3 fixer — 2026-11-20

- Existing path `.worktrees/ruflo-workers/run-mwoibr4r-44988a/fixer`；detached HEAD/base `39320a04512c446fb75aa993690a7496e434e57b`；owner Human/Codex builder；task 修訂3預算錨點、顯式重建與回歸／反證；created2026-11-20，retired N/A。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；no commit/push/merge/GPU/agents。
- 沿用既有 adapter/tests/report 與 `.artifacts/p3/`；revision3-verification 為本輪 GENERATED CPU logs／hash inventory，避免覆寫先前 artifacts；無新 worktree／top-level tree。Authority：frozen P3修訂3與candidate executable tests；GPU_SMOKE_NOT_RUN。
- Merge dependency：獨立審查與Human acceptance；artifact dependency：本輪合成CPU tests／in-memory mutation logs。Safe removal：先保存未提交patch與必要證據、處理引用與明確retirement review；目前dirty不得移除。

## P3 修訂2 fixer — 2026-11-20

- Existing path `.worktrees/ruflo-workers/run-mwnv3018-83e5c9/fixer`；detached HEAD/base `a91f38e86dab0d332fc63c6dc8a6ab7361a19b99`；owner Human/Codex builder；task 修訂2准入／全域帳本／凍結與報告；created2026-11-20，retired N/A。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；no commit/push/merge/GPU/agents。
- 沿用既有 adapter/evaluator/tests/report 與 `.artifacts/p3/`；revision2-verification 隔離 GENERATED CPU logs，避免覆寫舊證據。Authority：P3 frozen 修訂2及candidate executable tests；正式GPU准入仍未解鎖，沒有新增worktree或runtime observation。
- Merge dependency：獨立審查與Human acceptance；artifact dependency：本輪CPU logs、合成fixtures與既有唯讀六折metadata。Safe removal：先保存未提交patch與必要證據、處理引用、明確retirement review；目前dirty不得移除。

## P3 修訂1 fixer — 2026-11-20

- Existing path `.worktrees/ruflo-workers/run-mwnu9iri-e7a54a/fixer`；branch detached HEAD，base `69dc5baaee2d8e5c35d36458335c29fb4f26b619`；owner Human/Codex builder；task 修訂1項2/3/5/6/7/8；created2026-11-20，retired N/A。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；no commit/push/merge/GPU/agents。
- 沿用既有adapter/evaluator/tests/report與`.artifacts/p3/`，`revision1-verification`隔離本輪GENERATED CPU反證／pytest logs，不新增worktree或top-level流程。Authority：P3 frozen spec修訂1與candidate executable tests；正式GPU准入未解鎖。
- Merge dependency：獨立審查與Human acceptance；artifact dependency：唯讀pilot09六折splits hashes、合成CPU fixtures與本輪logs。Safe removal：先保存未提交patch及必要證據、處理引用與明確retirement review；目前dirty不得移除。

## P2 amendment 1 bounded repair worker — 2026-11-19

- Existing path `.worktrees/ruflo-workers/run-mwnpp2vy-e42792/fixer`; detached HEAD/base `1920fde0abfead53e84a2a9354a695e59a03764c`; owner Human/Codex builder, task approved seven amendment-1 repairs; created 2026-11-19; retired N/A. UNMERGED / NOT_DEPLOYED; explicit no commit/push/merge/agents.
- Reuses existing portfolio engine/evaluator, tests/report and `.artifacts/p2/` container; `job-run-mwnpp2vy-e42792` namespace isolates generated smoke/logs and avoids overwriting earlier evidence. No new worktree, training or formal data changes. Authority: candidate source/tests under frozen P2 registration amendment 1; no research or production promotion.
- Merge dependency: independent review and Human acceptance. Artifact dependency: local generated verification and read-only canonical full P1 scores/prices/caps, identities in smoke report. Safe removal requires preserved source patch, retained artifacts/resolved references and explicit retirement review; dirty worktree is not safe to remove.


## P1 review repair worker — 2026-11-19

- Existing path `.worktrees/ruflo-workers/run-mwmgdq5i-f34fa7/fixer`; detached HEAD/base `abf0368d709b83bc916144292da8b5930cc0ff26`; owner Human/Codex builder, task R1–R3/M1–M4/M6; created 2026-11-19; retired N/A. UNMERGED / NOT_DEPLOYED, retained with uncommitted bounded repair per explicit no-commit instruction.
- Reuses existing P1 implementation, tests and `.artifacts/p1/` container; separate `review-2023-s17-r500` smoke namespace preserves previous runs. Authority: candidate source/tests only. Merge dependency: independent review/Human acceptance; no merge/push authorized. Artifact dependencies: local smoke parts/logs/manifests, read-only pilot09 fold_2023 matrix and canonical prices/market-value parquet paths and hashes.
- Safe removal requires preserved source patch/history, resolved artifact/path references and explicit retirement review; current dirty worktree is not safe to remove.


## Ruflo integration — 2026-11-19

- Primary checkout: `development/ruflo-primary`, same completed integration history; prior `fix/macro-sync-fetch-hardening` retained at f501a7e. Local source activation only, no merge/push/production deployment.
- Purpose: complete Claude-primary Ruflo development integration; owner Human/Codex; branch `development/ruflo-integration`; base `f501a7eb28fab26c86f1bc776ad80a0ff13cfe81`; created 2026-11-19; retired N/A; UNMERGED / NOT_DEPLOYED.
- Native worktree created at `/home/frank/.codex/worktrees/2377/MarketMamba`, relocated to `.worktrees/ruflo-integration` for writable-root compliance. Existing pre-ruflo checkout is historical preservation, not this feature workspace; no existing checkout serves this integration.
- Authority: versioned development contract/tests only. Dependencies: pinned external npm installations; local ignored `.artifacts/ruflo` installation/memory/evidence. No production dependency or authorization.
- Runtime worker worktrees are scoped to `.worktrees/ruflo-workers/<job>/<worker>`, retained detached at fixed submission commit. Each job state/submission record owns purpose/task, source/base, created/finished time, status and paths; no automatic merge/removal. Safe removal requires clean status, preserved commits/patches/snapshots, resolved memory/artifact references and explicit review. Original worktrees remain untouched.


## Run status count patch adoption — 2026-11-13

- Path `.worktrees/run-status-count-validation`; branch `feat/run-status-count-validation`; base `8b0786018d763942d0ea68eb9ce537cb33f8b1fd`; owner Human/Codex, task independent MAS patch adoption; created 2026-11-13; retired N/A. LOCAL_MAIN_INTEGRATED / NOT_DEPLOYED at `e06fb256211f21612a25af4b10f709a756540d32` via Human-authorized fast-forward; feature worktree retained, clean at that commit.
- Existing worktrees do not serve this independent adoption. Reusing the MAS goal worktree would alter protected historical evidence; reuse the existing `.worktrees/` container with a separate bounded checkout. Native worktree creation was relocated here before source changes.
- Authority: display-only implementation/tests and adoption knowledge; merge dependency: satisfied by Human-authorized local main integration of `e06fb256211f21612a25af4b10f709a756540d32`; no push/deploy authority. Artifact dependency: original MAS goal `19459dd37dab44748c88a69b469eccad` delivery.patch (2,542 bytes, SHA-256 `d8c25509dab82400f212e803a8c44a92d5b5d854bf926c8b41f0e60f898d20ad`) and tracked portfolio-history JSON fixture; no private data dependency.
- Safe removal requires a clean checkout, retained branch/commit history, resolved path/artifact references and explicit retirement review. No existing worktree or MAS evidence is retired or modified.

## Window37 W34 bounded integration — 2026-11-02

- Path `.worktrees/v7-window37-w34-integration`; branch `integration/v7-window37-w34-models`; base72d35e78a3372d79373064a31dc6891d5ca3db83;owner HumanW37/Codex;created2026-11-02;retiredN/A. VERIFIED_CANDIDATE / NOT_DEPLOYED.
- Bounded port needed because W34 diverged and its model.py includes exhausted Fit/fetch/budget entrypoints; no whole-branch merge. Readonly loader preserves original score semantics and model/scaler/recipe/feature artifact identities.
- Artifact dependency W34 originals and161restore files, W11 isolated environment; no data moved/deleted. Safe removal requires clean retained source/history, resolved references and review; BACKUP_PENDING.

## Window35 collector closure / recovery — 2026-11-02

- Path `.worktrees/v7-window35-recovery-v04`; branch `research/v7-window35-recovery-v04`; base `c57ac3d7822f229ae08849e56581ebb3c969441c`; owner Codex/HumanWindow35;created2026-11-02;retiredN/A. COMPLETE / STOP / UNMERGED / NOT_DEPLOYED.
- W19 diverged; no branch merge/runtime port. Reuse mainSnapshot/QC/Restore and add minimal offline classifier/decision/attempt records, no secondcollector/platform.
- Private `.artifacts/window35-recovery-v04/` and `.artifacts/window35-restore/` hold analysis/offline evidence only; originalW19/W34 retained. No liveGET/Fit/GPU/Taskchanges. BACKUP_PENDING. Safe-removal requires retained cleanhistory and resolved bytes/pins/references plus explicit retirement review.

## Window33 recent shadow — 2026-10-25

- Path `.worktrees/v7-window33-recent-shadow`; branch `research/v7-window33-recent-shadow`; base `8fa41869a2583f0b5db698db83eed98611f24a7e`; owner Codex/HumanWindow33;created2026-10-25;retiredN/A. DATA_PRESERVED / FIT_BLOCKED / STOP / UNMERGED / NOT_DEPLOYED.
- Reuse W29 bounded source driver, W31 clocks/maturity/common resource guards, W27 safety, W30 fixed LGBM, W16/W18 bytes. Thin new dated query guard needed; no second collector/platform.
- Private `.artifacts/window33-recent-shadow/` and `.artifacts/window33-restore/`; preserve all previous evidence/W11 environment. Safe-removal needs clean retained Git, resolved bytes/pins/dependencies and explicit review. BACKUP_PENDING.

## Window32 shadow preparation — 2026-10-25

- Path `.worktrees/v7-window32-shadow-preparation`; branch `research/v7-window32-shadow-preparation`; base `67e2d25a710273646d41c0ad861b4a993593ba4d`. Owner Codex/HumanWindow32;created2026-10-25;retiredN/A. PREPARATION_COMPLETE / STOP / LOCAL_INTEGRATED_FIXED8fa4186 / NOT_DEPLOYED.
- Bounded W31 closure and Synthetic-only prediction-first publication/late-label eligibility; existing W09 complete bundle needs metrics, so reuse its primitives without fabricating metrics or a new registry. No actual inference/Fit/network/schedule.
- Private `.artifacts/window32-shadow-preparation/` and `.artifacts/window32-shadow-restore/` contain11 tiny Synthetic fixture files, not market data; retained W31/W29/W30 references remain protected. BACKUP_PENDING.
- Safe-removal requires retained clean Git history, resolved evidence dependencies and explicit retirement review. W19 untouched.

## Window31 expanding development — 2026-10-25

- Path `.worktrees/v7-window31-expanding`; branch `research/v7-window31-expanding`; base `b64ac353833a9248121f12ebe61e1333decb2dbf`. Owner Codex/HumanWindow31;created2026-10-25;retiredN/A. COMPLETE / STOP / LOCAL_INTEGRATED_FIXED67e2d25 / NOT_DEPLOYED.
- Three fixed origins reuse W07 geometry/W29 data/W30 LGBM/W10 Ridge/W08 evaluator; thin origin-neutral label/view adapter required because prior masks hardcode old yearly splits. No new framework.
- Private `.artifacts/window31-expanding/`, restore `.artifacts/window31-restore/`; W29/W30 source evidence and W11 environment retained. BACKUP_PENDING.
- Limits12Ridge+12LGBMfits,CPU1800s/artifact512MiB,0API/GPU/W19 operations. Safe-removal requires clean retained commit, resolved data/model/prediction/restore dependencies and explicit retirement review.

## Window30 real LightGBM — 2026-10-25

- Path `.worktrees/v7-window30-real-lightgbm`; branch `research/v7-window30-real-lightgbm`; base `e83dfe68e4bc73b59cb6390ec130ba967138e7f1`.
- Owner Codex/HumanWindow30;created2026-10-25;retiredN/A. COMPLETE / STOP / LOCAL_INTEGRATED_FIXEDb64ac35 / NOT_DEPLOYED; see window30-run/replay/review.
- One bounded unit; existing W11 LightGBM/wheel/lib/CPU env, W29 frozen feature-only rows/labels/support, W08 metrics and existing atomic/lock primitives. Existing synthetic guard is unchanged; Human authorizes one real-data research adapter.
- Private `.artifacts/window30-real-lightgbm/`, restore `.artifacts/window30-restore/`; W29 private raw/predictions/dataset and W11 env must remain available. Code/navigation restored via retained Git checkout. BACKUP_PENDING.
- Limits2LGBM fits, CPU1200s/fit300s/artifacts256MiB;zero Ridge/API/GPU/W19 operations. Safe-removal requires retainedcleancommit, allsource/model/prediction/restore dependencies resolved and explicitretirementreview.

## Window29 expanded development — 2026-10-25

- Path `.worktrees/v7-window29-expanded-development`; branch `research/v7-window29-expanded-development`; base `1d9a2a7a1ee57c455ba58250f63681efbbdd9050`.
- Owner Codex/HumanWindow29;created2026-10-25;retiredN/A. COMPLETE / STOP / LOCAL_INTEGRATED_FIXEDe83dfe6 / NOT_DEPLOYED; review window29-review.json.
- Reuse official W22bytes, existing source snapshots/atomic locks/W27 safety/W10 Ridge/W08 evaluator; minimal feature-only split is necessary because W24 wrapper gates future Entry/Exit before returning features. No new platform/V6 change.
- Private `.artifacts/window29-expanded-development/`, restore `.artifacts/window29-restore/`; local shared FinMind rate ledger only, not dataset registry.
- Authorized ≤80requests/70price,2Ridgefits,CPU3600s/artifacts1GiB; W19 protected/W12HALTED. Safe-removal requires retained cleancommit, resolved raw/model/prediction/restore dependencies and explicit retirement review.

## Window28 clean training / causality — 2026-10-25

- Path `.worktrees/v7-window28-clean-refit`; branch `research/v7-window28-clean-refit`; base `39e2fedeb550a08d00f4abddc9a6caa7b1d570f8`.
- Owner Codex / Human Window28; created 2026-10-25; retired N/A. COMPLETE / STOP / LOCAL_INTEGRATED_FIXED1d9a2a7 / NOT_DEPLOYED. 2fits completed; see window28-run/replay/review.
- Reuses W24 numeric Ridge/feature pipeline, W27 mask/support, W25/W08 evaluator; no new framework or V6 change.
- Private `.artifacts/window28-clean-refit/`; preserve W24/W27 originals. Exact two CPU fits authorized, API/GPU/W19 operations zero.
- Safe removal requires retained clean commit, resolved raw/model/prediction/replay dependencies and explicit retirement review.

## Window27 label safety — 2026-10-25

- Path `.worktrees/v7-window27-label-safety`; branch `research/v7-window27-label-safety`; base `367005dfaba0988fca5e294d9b3e281408bbfce8`.
- Owner Codex/Human Window27; created 2026-10-25; retired N/A. COMPLETE / STOP / LOCAL_INTEGRATED_FIXED39e2fed / NOT_DEPLOYED. Freeze commit e77f32d; four-group diagnostic and replay completed; review evidence in window27-review.json.
- Private `.artifacts/window27-label-safety/`, references W24/W25/W26 immutable bytes. Reuse W25 paired helper/W08 primitive; no new framework.
- No API/fit/production/scheduler/GPU authority. W19 protected, W12 HALTED. Safe removal requires retained clean commit and resolved artifact dependencies, explicit retirement review.

## Window26 label integrity / next-study proposal — 2026-10-25

- Path `.worktrees/v7-window26-label-contract`; branch `research/v7-window26-label-contract`; base `27055f7ec34cd4aa6d47f67bd89e5b6616244bc8`.
- Owner Codex / Human Window26; created 2026-10-25; retired N/A. COMPLETE / STOP / LOCAL_INTEGRATED_FIXED367005d / NOT_DEPLOYED.
- Reuses W24/W25 immutable bytes and source snapshots; new report/spec/forensic scripts only, no new platform or V6 implementation.
- Authority: bounded event forensics and PROPOSED study preparation; no new model fit, predictions, RankIC, formal admission, GPU or scheduler operations.
- Artifacts `.artifacts/window26-label-integrity/`; same-host restore `.artifacts/window26-restore/`; BACKUP_PENDING.
- Dependencies W24/W25 source bytes and W26 event/document snapshots; preserve originals. W19 protected, W12 HALTED, W06 not adopted.
- Safe removal requires clean retained commit, resolved raw/source/restore references and explicit retirement review; no cleanup authorized here.

## Window25 fixed-prediction audit — 2026-10-25

- Path `.worktrees/v7-window25-robustness-audit`; branch `research/v7-window25-robustness-audit`;
  base90b1327560f717727ad3fd53dfe86cb421c70f5b; owner Codex/HumanWindow25;created2026-10-25;retiredN/A.
- COMPLETE/STOP/LOCAL_INTEGRATED_BY_W26_FIXED27055f7/NOT_DEPLOYED. W24executedlimitedpilot; thisdistinctauthorization only audits existing evidence.
- Reuse experimental/research/reports/Decisions, W08metric/E1circularhelper/W09verify/W24artifacts;no newplatform.
- Root.artifacts/window25-robustness-audit and samehostwindow25-audit-restore own generateddiagnostics/source,
  BACKUP_PENDING. W24source191files immutable; no oldartifact/worktreecleanup.
- Safe removal requires retainedcleancommit, allsource/raw/result/restore references resolved and explicitretirementreview.
- W24fixed90b1327 localintegrated; W19/task/artifacts/budget/runtimePROTECTED, nooperations.

## Window24 expanded limited pilot — 2026-10-25

- Path `.worktrees/v7-window24-expanded-pilot`; branch `research/v7-window24-expanded-pilot`;
  base87af76b038fcbbda9e1e4b3a4349e693a3b1ba48; owner Codex/HumanWindow24; created2026-10-25; retiredN/A.
- COMPLETE/STOP/LOCAL_INTEGRATED_BY_W25_FIXED90b1327/NOT_DEPLOYED. W23onlyfive-stocksmoke; W24hasdistinct bounded expanded data/pilot authorization.
- Reuse experimental/research/reports/Decisions, existingrequestguard/numeric/evaluator/publisher; no newplatform.
- Root.artifacts/window24-expanded-pilot unique raw/three-masks/bundle/predictions/sourcecopies; samehost
  window24-price-restore and window24-pilot-restore are not independent backups(BACKUP_PENDING).
- Safe removal requires retainedcleancommit, all source/raw/restore references resolved and explicitretirementreview.
  Noartifactworktreecleanupauthorized. W23fixed87af76b integrated; W19/task/artifacts/budget/runtime PROTECTED.

## Window23 calendar and engineering smoke — 2026-10-25

- Path `.worktrees/v7-window23-calendar-smoke`; branch `research/v7-window23-calendar-smoke`;
  base7027f03ebdb7b7b3bd90bc2bf5ce97a4c267c470; owner Codex/HumanWindow23; created2026-10-25; retiredN/A.
- COMPLETE/STOP/LOCAL_INTEGRATED_BY_W24_FIXED87af76b/NOT_DEPLOYED. W22 covered proof but no allowed real-data smoke; distinct bounded authorization.
- Reuse existing experimental, research/reports, W09publisher/W08kernel/W10numeric core; no new platform.
- Root.artifacts/window23-calendar-smoke and same-host window23-restore own unique docs/masks/bundle/source-at-run;
  BACKUP_PENDING. Original immutable bundle and raw preserved; safe removal needs retained clean commit,
  resolved artifact refs and explicit retirement review. No cleanup authorized.
- W22 fixed7027f03 locally integrated; W19/task/artifacts/budget/runtime PROTECTED and untouched.

## Window22 historical panel — 2026-10-25

- Path `.worktrees/v7-window22-historical-panel`; branch `research/v7-window22-historical-panel`;
  base428a5fa5cb7b771db39a053f88c7046501399f7e; owner Codex/HumanWindow22; created2026-10-25; retiredN/A.
- COMPLETE/STOP/LOCAL_INTEGRATED_BY_W23_FIXED7027f03/NOT_DEPLOYED. W21 only short windows; this isolated two-unit scope adds bounded continuous proof.
- Existing experimental/research/reports/Decisions and root.artifacts reused; no second platform.
- Root.artifacts/window22-historical-panel and restore own unique raw/panel/mask/source; BACKUP_PENDING, no GC.
- Safe removal requires retained clean branch/commit, resolved artifact/source references and explicit retirement review.
- W19 protected; no operations. W21 locally integrated fixed428a5fa, original bytes remain untouched.

## Window21 FinMind proof — 2026-10-25

- Path `.worktrees/v7-window21-finmind-proof`;branch `research/v7-window21-finmind-proof`;
  base `6a34520d8f9c1166172af3503cad531e23f5ecec`;owner Codex/HumanWindow21;
  created2026-10-25;retiredN/A;COMPLETE/STOP/LOCAL_INTEGRATED_BY_W22/NOT_DEPLOYED。
- ExistingW20onlydocumentation，無同目標真實proof；沿用experimental/research/reports/Decisions及root.artifacts，無新平台。
- Actualraw/snapshot/record/claims/source/closedmarker位於根目錄.artifacts/window21-finmind-proof，
  root.artifacts/window21-finmind-restore是同機copy，BACKUP_PENDING；source/與source-reviewed/分開留存，18originalrecords不改。
- Safe removal需clean retainedcommit、resolvedrootbyte/restore/source/license refs與獨立退役review；本輪不刪任何artifacts。
- W19 worktree/task/artifact/budget PROTECTED，不操作、不觸發；main只ffW20，不merge其他branch。

## Window20 historical source decision — 2026-10-25

- Path `.worktrees/v7-window20-historical-sources`;branch `research/v7-window20-historical-sources`;
  base `87695e4d5dc6e047f5ab39ca3cd8294b971bbc3a`;owner Codex/HumanWindow20;
  created2026-10-25;retiredN/A;COMPLETE/STOP/LOCAL_INTEGRATED_BY_WINDOW21/NOT_DEPLOYED。
- Prior40treesclean/distinctscope；本tree隔離文件調查，沿用research/reports/maps，沒有platform或實作。
- `.artifacts/window20-documents`保存8官方HTML及Snapshotmanifest，不含行情；Git只有URLs/hash/report，不能靠hash還原bytes。
- Safe removal需clean retainedcommit、resolvedHTMLbyte/dependencyrefs及separate退役review；不清理W14/16/18或W19。
- W19獨立Task/root/codepins不修改或觸發；本輪只readfixedcf1a84f/policy/armhash，main未前進。

## Window18 manual collector pilot — 2026-10-24

- Path `.worktrees/v7-window18-manual-collector`;branch `development/v7-window18-manual-collector`;
  base `96e8c0c4f2f7f78c508048e31f0f163d6c707128`;owner Codex/HumanWindow18;
  created2026-10-24;retiredN/A;WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED。
- Prior38treesclean，無同目標工作區；沿用experimental/research/reports/maps與根目錄.artifacts。
- 唯一manualpilot已完成；兩exactCSV各1GET，budgetroot `.artifacts/window18-network-budget` 已用完。
- Raw與first/source-reviewed code於根目錄`.artifacts/window18-manual-pilot`，初次Restore於
  `.artifacts/window18-manual-restore`，修後Restore於`.artifacts/window18-reviewed-restore`；同機非獨立backup。
- W14/W16/W17原bytes與W12HALTED保留；NO_GC。Safe removal須retainedcommits／resolvedbytes/runtime refs／
  separatelyreviewedretirement；本輪無清理。Dailyactivation／researchadmission仍未批准，W06/W12未整合。

## Window17 forward operational readiness — 2026-10-24

- Path `.worktrees/v7-window17-forward-readiness`;branch `research/v7-window17-forward-readiness`;
  base `000b0b9c40448b11a41d05396dffc6e0f3ea1235`;owner Codex/HumanWindow17;
  created2026-10-24;retiredN/A;WINDOW_COMPLETE/STOP/LOCAL_INTEGRATED_BY_WINDOW18/NOT_DEPLOYED。
- 37 existingclean trees distinctscope；new isolation only for two boundedunits；既有reports/research/experimental/maps沿用。
- W16source/restorebytes保留不動；根目錄`.artifacts/window16-durable-evidence`持久副本獨立於本tree，
  仍同機同filesystem BACKUP_PENDING；`.artifacts/window17-official-documents`於根目錄保存官方文件bytes。
- Safe removal requires clean retainedcommit、resolvedbyte/runtime/dependencyrefs及separate retirement review；本輪不清理。
- Authority僅離線readiness/proposal；正式collector／研究admission未批准；main固定W16，W06/W12未整合。

## Window16 real resource coverage — 2026-10-24

- Path `.worktrees/v7-window16-resource-proof`;branch `research/v7-window16-resource-proof`;
  base `27dcb36b5431b70e8aa831edb6cb9234cb3a40b3`;owner Codex/HumanWindow16;
  created2026-10-24;retiredN/A. WINDOW_COMPLETE/STOP/LOCAL_INTEGRATED_BY_WINDOW17/NOT_DEPLOYED;single reviewGO0Critical/Required,0repair.
- Prior36trees clean/distinctscope; no samepurpose. Existing Snapshot/metadata/report/maps
  reused; no newsourceplatform or testsframework. Mainff27dcb held;no W06/W12merge.
- Preserve `.artifacts/window16-resource-proof` (requests ledger+2raw/manifests) and
  `.artifacts/window16-resource-restore` (independent2bytecopies). Sameworktree not
  crossdevicebackup; loss of tree canloseboth. W14/W15 originalbyte refs unchanged.
- Safe removal only after clean retainedcommits/bytesrefs resolved and separate review;
  no cleanup/move authorized. No furtherfetch/retry/fallback/collector execution.

## Window15 licensed dual-track — 2026-10-24

- Path `.worktrees/v7-window15-dual-track`;branch `research/v7-window15-dual-track`;
  base `4da1bcfa81c8ff786de82d29ec461c0851d411d5`;owner Codex/HumanWindow15;
  created2026-10-24;retiredN/A. WINDOW_COMPLETE/STOP/LOCAL_INTEGRATED_BY_WINDOW16/NOT_DEPLOYED.
- Prior35trees clean/distinctscope; none serves this licensed dualtrack task.
  Existing experimental/research/Decisions/reports/maps reused; no platform.
- Preserve .artifacts/window15-license-evidence (3officialHTML/raw/manifests) and
  .artifacts/window15-synthetic-vintage (records/intents/snapshots/attempts and old
  observedsource). Hash-onlymetadata cannot restore lostbytes; no GC/move authorized.
- W14originalrawstill onlyW14tree; mainff4daheld, W06/W12notmerged. Old92files and
  4Frozencontracts retainedunchanged. Safe removal requires clean retainedcommits,
  resolved byte/runtime refs and separate reviewedretirement. FinalHEAD containingclosure.

## Window14 bounded data admission pilot — 2026-10-24

- Path `.worktrees/v7-window14-data-admission`;branch `research/v7-window14-data-admission`;
  base `c78098f3fdddb74e461bc54d700bde5baac0fcc1`;owner Codex/HumanWindow14;
  created2026-10-24;retiredN/A. WINDOW_COMPLETE/STOP/LOCAL_INTEGRATED_BY_WINDOW15_SOURCE_ONLY/NOT_DEPLOYED.
- Prior34trees clean/distinctscope; no sameW14pilot. Existing experimental/research/
  reports/Decisions reused, no platform. MainffW13heldc78098f; W06/W12notmerged.
- Preserve `.artifacts/window14-historical-pilot` immutable raw/manifests (337221rawbytes),
  no hash-only restore claim; preserve W11 83CPUfiles and W12 HALTED5diagnostics.
  Safe removal requires clean retainedcommits/bytes/refs resolved+separate review.

- U1/U2independent0Critical/Required,1boundedrepair each;150CPUtestsPASS.
  Futurestudy/collection rights/data/policy approval pending;finalHEAD containingclosure.

## Window13 thermal/admission preparation — 2026-10-24

- Path `.worktrees/v7-window13-admission-preparation`;branch `research/v7-window13-admission-preparation`;
  base `8db5b9111b98d4965a9f56f10a51c5157284b900`;owner Codex/HumanWindow13;
  created2026-10-24;retiredN/A. WINDOW_COMPLETE/STOP/LOCAL_INTEGRATED_BY_WINDOW14/NOT_DEPLOYED.
- U1sourcef26dbf5; final containingU2closurecommit. CPUverification140+13tests,
  reviews0Critical/Required/0rework; no implementation/GPU/dataadmission.
- Prior33trees clean/distinct scopes; no W13 same-purpose tree. Existing Decisions/
  reports/research/maps suffice; new isolation avoids changing main/pendingcandidates.
  W12/W06/protocol/recovery read at fixedSHA only, no adoption. Preserve W12 rawHALTED
  budget/diagnostics and W11 83CPUbytes files; no cleanup. Safe removal only after
  clean retainedcommits/bytesrefs resolved and separate retirement review.

## Window11 L2/CPU resume — 2026-10-23

- Path `.worktrees/v7-window11-l2-cpu-resume`;branch `development/v7-window11-l2-cpu-resume`;
  base `f25181637e40138c84dc3722292647d4325882fc`;owner Codex/HumanWindow11;
  created2026-10-23;retiredN/A. WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
  SourceU1 f00bebab33cdeaa75b8401c6a6988dfaa3d3e0e7;U2 e28ac62ed95007e246185c5c2a134d17a5d0e08b;finalcontainingclosurecommit.
  Preserve .artifacts/window11-cpu-resume-proof bytes (~1.79MB) and exactinventory;
  not restorable from Git hashes alone. No safe removal until bytes/env refs resolved.
- Existing31trees clean/distinct scope; none serves W11. Existing source/evidence/maps
  reused. Dedicated `.runtimes/window11-cpu` overlay + cached officialwheels are GENERATED,
  reference existing CPUbasevenv; no shared/global/Windows env changed. Preserve these
  runtime refs until separately reviewed retirement; no cleanup authorized.
- Mainf251816 held. Safe removal requires clean retained commits/bytes/evidence and
  resolved venv/artifact references plus separate retirement review.

## Window10 classical synthetic study — 2026-10-23

- Path `.worktrees/v7-window10-classical-e2e`; branch `development/v7-window10-classical-e2e`;
  base `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9`; owner Codex/HumanWindow10;
  created2026-10-23;retiredN/A. WINDOW_COMPLETE/LOCAL_INTEGRATED_BY_WINDOW11/NOT_DEPLOYED.
  U1 4ec22a4330ed6db7df0b3dafb4106cea820516e2; final U2/closure checkpoint is containing commit.
- Existing30trees clean/distinct scope, none serves W10 adapters. Existing directories
  experimental/research/reports/maps sufficient; no new platform. Reuses integrated
  W07/08/09 + NumPy2.0.2; sklearn/lightgbm missing; no external data access/install.
- Main91fa82c held; W09 originaltree/artifacts retained. Safe removal requires clean
  preserved commits/evidence/resolved references and separate retirement review.

## Window09 artifact/replay — 2026-10-23

- Path `.worktrees/v7-window09-artifact-replay`; branch `development/v7-window09-artifact-replay`;
  base `bdea30d877fa2c5d02308f343fa5591a35616aec`; owner Codex/HumanWindow09;
  created2026-10-23; retiredN/A. WINDOW_COMPLETE/LOCAL_INTEGRATED_BY_WINDOW10/NOT_DEPLOYED.
  U1 79d6c556d1bb5648a40a02c3b6f0e628c6512b4a; final container repair/U2 are in containing closure commit.
- Existing29trees clean/distinct objectives, none serves W09. Existing experimental,
  research/reports/maps suffice; no new platform or memory tree. Dependencies E1
  safe filesystem/hash/state validator, integrated W08 outputs; synthetic bytes only.
- Main bdea30d held; W08 originaltree/history retained. Safe removal requires clean
  preserved commits/evidence and resolved artifact refs/separate retirement review.

## Window08 ranking evaluation — 2026-10-23

- Path `.worktrees/v7-window08-ranking-evaluation`; branch `development/v7-window08-ranking-evaluation`;
  base `1631020bce8223ac4337c7e97473a2a3d9a555f0`; owner Codex/HumanWindow08;
  created2026-10-23; retiredN/A. WINDOW_COMPLETE / LOCAL_INTEGRATED_BY_WINDOW09 / NOT_DEPLOYED.
  Final HEAD is containing U2/closure commit; U1 826b8ddf798e752584d180181d24cc1db967e019.
- Existing28trees checked clean/distinct purposes; none serves W08 evaluation.
  Isolate new work; reuse existing experimental/research/reports/maps directories,
  no alternate platform. Depends only integrated E5/W07 primitives, synthetic fixtures,
  existing CPUvenv; no real payload. W07 originaltree/history retained (now integrated).
- Main held1631020, pending closure kept here. Safe removal requires clean preserved
  commits/evidence, resolved artifact references and separate retirement review.

## Window07 temporal/isolation — 2026-10-23

- Path `.worktrees/v7-window07-temporal-isolation`; branch `development/v7-window07-temporal-isolation`;
  base `465dca229fc1452aaf1b46572b06886a7d8a2c8f`; owner Codex/HumanWindow07;created2026-10-23;retiredN/A.
- Existing27trees hold fixed/source/audit roles, no same-scope activeunit; new isolation
  protects main/pending branches. Existing experimental/research/reports/maps suffice.
- WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED. Final containing Unit2/closure commit.
  Dependencies maincalendar/hash primitives, pinned W06/audit
  reference only; syntheticfixtures, no manageddata/modelpayload. Sourceunit commits
  retained. Safe removal: clean/preservedcommits/evidence/resolvedrefs and separate
  retirement approval. No existingtree/history/data cleaned.


## Window05 feature integrity — 2026-10-23

- Path `.worktrees/v7-window05-feature-integrity`; branch `research/v7-window05-feature-integrity`;
  base `2ee4ca83e709af64be4dd63032ba4fb47903d769`; owner Codex/current Human Window05; created2026-10-23; retired N/A.
- Purpose bounded F0/F1/F2. Existing23trees hold distinct frozen/source/audit roles,
  none serves this new objective; isolation protects main and historical payloads.
  Existing experimental/research/reports paths suffice; no new architecture/memory tree.
- WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED; final containing F2/closure commit.
  Dependencies frozenmanifest/policy, pinned historical
  source metadata, existing CPUvenv and limited official-document reads. E1 matrix
  absent; no processed_v6 mount, inherited3trackedDatafiles unchanged/unread.
- Safe removal requires clean tree, retained commits/artifacts and resolved
  references plus separate retirement authorization. No existing tree removed.


## W04B-LOG integrated — 2026-10-21

- Fixed `development/v7-window04-cpu-continuation@97fb4c9d669f3aa237ef5545a0ef322a5cc2d6a9`
  is now LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Original worktree retained
  clean at the fixed SHA; older UNMERGED observations below describe prior state.
- Main ff-only31f7acc→97fb4c9, then this docs-only closure. No new worktree needed;
  all23 prior trees and historical artifacts retained. Retired N/A; no cleanup
  authorized; existing safe-removal requirements remain. No pending development.


## Window04 CPU continuation — 2026-10-21

- Path `.worktrees/v7-window04-cpu-continuation`; branch
  `development/v7-window04-cpu-continuation`; base `31f7accc475b5b735684d5a06725a237d85d5693`.
- Owner Codex/current Human Window04 continuation. Created2026-10-21; retired N/A.
  Purpose bounded CPU reliability work after approved integration. Existing22trees
  retain fixed/audit/source roles; none serves this new development window.
- UNMERGED / NOT_DEPLOYED. Dependencies existing runtime contracts, fixtures and
  recovery venv. Source/memory/closure stay here, no main navigation updates.
- Safe removal requires clean tree, retained commits/artifacts, resolved references
  and separate retirement approval. No cleanup authorized.


## Window04 reconciliation — 2026-10-21

- Path `.worktrees/v7-window04-reconciliation`, branch `integration/v7-window04-reconciliation`;
  base c733a53f81ff7bd572739b5d75f8baabf1189976; owner Codex/current Human reconciliation.
- Created 2026-10-21; retired N/A. Purpose preserve both histories and verify new
  candidate, not Stage B. Existing 21 trees hold distinct fixed/source/audit purposes;
  none may be repointed for this newly authorized candidate. Existing document paths reused.
- UNMERGED_IN_MAIN / NOT_DEPLOYED; dependencies fixed489a127 and retained audit3330427.
  Safe removal requires clean checkout, retained commits/artifacts, resolved references
  and separate retirement authorization. No cleanup or main merge authorized.


## Portfolio H0 local integration closure — 2026-10-21

- WSL primary checkout `/home/frank/projects/MarketMamba`, branch `main`, now includes
  exact approved4a66928 from83a6a2e via ff-only. Owner Human-authorized Codex H0;
  this containing commit adds only closure documentation/evidence.
- CPU/roadmap/PORT/UX/font source branches below are preserved historical checkpoints
  whose source ancestry is now LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Their
  earlier UNMERGED state describes pre-H0 observations, superseded by this entry.
- All16 worktrees remain; no checkout removed, reset, repointed or cleaned. Audit
  `audit/v7-h0-preflight@7ecb920` is retained separately and was not merged.
- Candidate remains `fix/v7-timeline-font-visibility@4a66928`; retire dates remain N/A.
  Existing safe-removal conditions and separate authorization still apply.
- Main is the next authorized CPU-work source baseline; this does not grant a new
  Program. No active fixture services, production deployment or operational change.
- [Closure](../../reports/v7_portfolio_h0_integration_closure.md),
  [evidence](../../research/v7/portfolio-h0-integration-closure.json).


## Timeline font blocker repair — 2026-10-21

- Path `.worktrees/v7-timeline-font-visibility`; branch `fix/v7-timeline-font-visibility`;
  base exact `6c89479416ad9fa429db7818cc645f729367eac9`; owner Codex under explicit
  Human late-font blocker scope. Created2026-10-21; retired N/A.
- Purpose minimal frontend repair and bounded H0 preflight. Existing product branch
  owns fixed old candidate; audit branch owns failure evidence. Neither can serve
  this new candidate without losing its frozen responsibility, so isolation is required.
- COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED; new SHA awaits explicit H0 approval.
  Depends full CPU/PORT/UX ancestry; audit7ecb920 is evidence only, not ancestor.
- No worktree removal authorized. Safe removal requires clean state, retained commits/
  artifacts and resolved references. Temporary fixture services stopped after review.
- [Report](../../reports/v7_portfolio_font_visibility.md), [sole backlog](../../research/v7/research-program.md).

## Preserved H0 audit source — 2026-10-21

- `.worktrees/v7-h0-preflight`, `audit/v7-h0-preflight@7ecb92016557af135ffd28c80892aa6be9d727df`,
  base6c894794; owner Codex/Human H0 preflight; created2026-10-21; retired N/A.
- Evidence-only STOP checkpoint for failed old-candidate font acceptance; no main
  integration, not a replacement product candidate. Original evidence remains immutable.
  No removal authorized; retain clean commit/artifacts and references before cleanup.


## Portfolio History UX — 2026-10-21

- Path `.worktrees/v7-portfolio-ux`; branch `development/v7-portfolio-ux`; base
  `5a2bc6fcee4c0bf78190a6ac74d11f53f8becbd4`; owner Codex under Human UX-01–04.
- Created 2026-10-21; purpose presentation-only refinement, preserving prior PORT
  checkout and original acceptance. Existing worktrees own frozen prior evidence.
- Final closure continues this same candidate from `1553a59`; owner Codex under
  explicit Human two-task closure scope. Both tasks COMPLETED_VERIFIED in containing
  closure commit; [closure report](../../reports/v7_portfolio_ux_closure.md).
- State COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED; authority UI engineering only.
  API/Vite/dedicated Chrome stopped; no production observation refreshed.
- Depends full PORT/CPU ancestry, existing Node dependencies and temporary API venv;
  no market/model data. Retired N/A; no removal approved. Safe removal requires clean
  tree, preserved commits, artifact audit and resolved references.
- [Current backlog](../../research/v7/research-program.md);
  [report](../../reports/v7_portfolio_ux_acceptance.md).

## H1 Portfolio History slice — 2026-10-21

- Path `.worktrees/v7-portfolio-history`; branch `development/v7-portfolio-history`;
  base `bebf19ee937a2237641e09701d0564590ed30fc6`; owner Codex under Human H1.
- Existing roadmap candidate is a frozen review checkpoint, CPU windows are closed;
  this isolated worktree owns new fixture product code. Created 2026-10-21; retired N/A.
- State COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED; authority fixture engineering only.
  Local API/Vite/Chrome stopped after acceptance; ignored generated artifacts retained.
- Dependencies: full roadmap/CPU ancestry, no model/market inputs; unit state in
  [evidence](../../research/v7/portfolio-history-evidence.json), sole backlog in research-program.
- Safe removal requires clean tree, preserved commits, audit of unique ignored fixtures
  and browser/test evidence, resolved references and authorization. No merge/push/removal now.

## Strategic recovery and CPU integration candidate — 2026-10-21

- Path `.worktrees/v7-cpu-roadmap`; branch `integration/v7-cpu-roadmap`.
- Base `4c388ded57062194114ea3e221d00c177eb96892`; owner Human-authorized Codex,
  task Strategic Roadmap Recovery & Autonomous Development Planning.
- Purpose: isolated candidate jointly verifying five CPU fixes and preserving
  current roadmap/backlog proposal. Existing CPU checkout remains a fixed completed
  window checkpoint; other checkouts serve separate evidence/approval lineages.
- Created 2026-10-21 Taipei; state PLANNING_COMPLETE / GO_CANDIDATE_ONLY /
  UNMERGED / NOT_DEPLOYED. Authority: engineering evidence/planning, no new
  research or production authority. Retired date N/A.
- Dependencies: main83a6a2e → E1 inventory11ae71d → CPU windows4c388de; Python
  stdlib/recovery venv, fixture-only tests; no candidate Data or model weights used.
- Merge needs explicit exact-SHA Human approval of full inherited evidence and
  planning; removal needs preserved commits, clean/unique-artifact audit, resolved
  references and explicit approval. No removal/merge/push authorized now.
- Current queue [research program](../../research/v7/research-program.md);
  [candidate report](../../reports/v7_strategic_roadmap_recovery.md).

Window 02 reuses the window 01 checkout below from `7eb9cd9`, owner Codex under
current Human CPU authorization; WINDOW_COMPLETE, UNMERGED / NOT_DEPLOYED.
Two units verified; U1 `735ae6f`, U2 containing closure commit; no further READY
unit queued. Prior window and E1 lineage remain integration dependencies.
[Task/resume](../../tasks/v7-cpu-development-window-02.md). Existing creation,
authority, dependencies and safe-removal requirements remain unchanged.

## CPU-only window 01 — 2026-10-21

`.worktrees/v7-cpu-window-01`, branch `development/v7-cpu-window-01`, base
`11ae71d1a3ea7f68ed10d5623bc9d639689e4e8b`; owner Human-authorized Codex.
Purpose: three bounded existing V7 portfolio contract fixes/verification units;
existing checkouts are completed evidence or other subsystem lineages and are
preserved. State WINDOW_COMPLETE / UNMERGED / NOT_DEPLOYED. Authority: engineering only;
dependencies: inherited E1 evidence and main `83a6a2e`, Python stdlib,
temporary synthetic fixtures, no external data. Created 2026-10-21 Taipei;
retired N/A. Safe removal requires preserved commits, clean/artifact audit and
explicit Human approval. No removal or integration is authorized here.

## E1 evidence recovery — 2026-10-21

New `.worktrees/e1-frozen-evidence`, branch `recovery/e1-frozen-evidence`,
base `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e`, owner Human-authorized Codex.
Purpose: bounded E1 artifact inventory/identity and gated offline inspection.
Existing research/recovery checkouts carry distinct historical work; this new
checkout isolates current evidence from integrated main and preserved sources.
State: UNMERGED evidence/documentation only, STOP_REPLAY_MATRIX_IDENTITY_UNAVAILABLE.
Dependencies: fixed eec63c5/ef020e5 Git artifacts and read-only local deliveries.
No execution/training authority transferred. Retired N/A; removal requires
retained commit/evidence, clean checkout and explicit decision.

## Local Phase A integration closure — 2026-10-21

WSL main fast-forwarded `50c5b78` → fixed `89f9496` under explicit Human
authorization. `.worktrees/phase-a-task5` remains clean at the approved candidate;
all original source worktrees are retained unchanged. No checkout is removed.
Subsequent main closure commit updates memory/report only. E1 new work will use
a separate isolated checkout, not the historical candidate or Windows operations.

## Phase A integration — 2026-10-21

New `.worktrees/phase-a-task5`, `integration/phase-a-task5`, base
`63a040d306e402dbc22e8592f57f034fd4a9a863`, owner Human-authorized Codex.
Purpose: isolated joint validation plus Task 5 offline evidence. A new checkout
is explicitly requested to preserve both existing source candidates.
Seventeen Phase A-only commits from `9efed7b` through `1e119ef` are cherry-picked
with source commit provenance; unrelated research lineage is excluded.
State UNMERGED / NOT_DEPLOYED. Source checkouts and main retained unchanged.
Task 5 limited offline engineering approved at `1c5250e` in the 2026-10-21
Final Engineering Review request. Final Phase A/integration decision pending.
This checkout is reused for the bounded decision/governance update and final
review; no new worktree or production implementation is needed.
Resume from the branch's committed evidence index and report, not the older
source checkouts' historical pending-Task4 notes. Do not start further work.
Dependencies: stdlib SQLite tests and existing E1 recovery CPU .venv for PIT;
all replay data temporary synthetic files. No execution/production authority.
Retirement N/A; requires preserved commits, clean/artifact audit and Human approval.

## Bounded recovery addition — observed 2026-10-20

New checkout `.worktrees/v7-snapshot-immutability`, branch
`fix/v7-snapshot-immutability`, base
`50c5b785e8c373913c3301a056497eb955f36681`, created 2026-10-20; owner/task:
Codex under Human's bounded V7 engineering recovery. Purpose: one source-snapshot
immutability fix and offline recovery tests. State: `UNMERGED / NOT_DEPLOYED`,
no research or execution authority carried; retirement date N/A.
Existing worktrees serve Phase A, E1 research/recovery/diagnostics or memory
hardening, not this main-based snapshot fix, so a separate checkout avoids
overwriting or importing their unmerged work. Source and main were clean at
start; existing checkouts were retained. The additional memory-hardening
checkout is `fix/memory-lifecycle-hardening@91db7d1`, clean at observation.
Test runtime depends on the retained E1 recovery `.venv`; fixtures are temporary,
not formal data artifacts. Adoption requires separate integration approval.
Safe removal requires retained commits/branch, clean state, artifact/dependency
check and explicit Human approval; this entry authorizes no removal.

Same-day continuation from `8309ee3`: reuse this clean isolated checkout for
Unit A reader-contract validation and its tests; no new checkout needed for
the same snapshot subsystem. Phase A evidence recovery is a separate unit on
the operations lineage; its original worktree must not be overwritten.

Unit B added `.worktrees/phase-a-task4-evidence`, branch
`fix/phase-a-task4-evidence@46df7b3402f85877a5fc55600d53b15a6f3a7849`,
direct base `f5b0ff3df7645098cc01a9eeb36f5df22c4a3e41`, created 2026-10-20
under the same Human bounded objective. Original operations checkout was clean
and retained unchanged. Scope: Task 4 artifact-summary fix, synthetic tests and
evidence index; `UNMERGED / NOT_DEPLOYED / CORRECTNESS_APPROVAL_NOT_ESTABLISHED`.
Runtime: system Python 3.12.3 stdlib; no production data dependency. This checkout
retains the historical ops AGENTS file; current project authority remains the
canonical map, not its stale runtime narrative. Existing canonical memory is
updated in the snapshot candidate, not duplicated into the old lineage.
Both candidates require explicit integration decisions; no merge/cleanup grant.

Task 4 isolation follow-up continued in this clean existing isolated worktree,
now `fix/phase-a-task4-evidence@1e119ef855f8fd319f774162beca0bd729d4b921`.
Only tests/evidence changed from `46df7b3`; original ops checkout remains at
`f5b0ff3`. Snapshot candidate remains a separate lineage; no combined candidate
or merge was needed for independent subsystem tests. Task 4 Human review is
pending, no authority carried. Raw synthetic test evidence is tracked, not a
runtime database. Canonical-memory checkpoint remains on the snapshot branch.

> Updated from `git worktree list --porcelain` and branch verification on 2026-09-22 after the four authorized K checkouts were retired. This is a registry of checkouts and responsibility, not authority to merge or remove them. `UNKNOWN` means creation/ownership was not verified. Update in the same coherent change when a branch or responsibility changes; a fresh runtime/worktree observation needs its own date.

| Checkout under `.worktrees/` | Branch / HEAD | Purpose and owner/task | Base | Current state | Authority carried / dependencies | Created / retired |
|---|---|---|---|---|---|---|
| `project-knowledge-k0-discovery` | `docs/project-knowledge-k0-discovery` / `c802f52` | K0 discovery; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | K0 inventory/report; K1/K2 derived from it. Verify later branches retain references before removal. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `project-knowledge-k1-bootstrap` | `docs/project-knowledge-k1-bootstrap` / `31dd74a` | K1 map/bootstrap; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | Current/Authority map bootstrap and legacy Claude archive; K2 successor. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `project-knowledge-k1-domains` | `docs/project-knowledge-k1-domains` / `2c6c6f3` | domain notes; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | Twelve domain notes; K2 successor. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `project-knowledge-k2-governance` | `integration/project-knowledge-k1` / `18152d0` | K2 governance; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | Knowledge checks, authority/history lineage. Adopted in main by governance consolidation, but compare source commits and generated refs before retiring checkout. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `v7-corrected-e5-2026-diagnostic` | `feature/v7-e1-rolling-origin-refresh` / `044eef7` | corrected E5/E1 bundle; owner UNKNOWN | `main@7bdf2e4` merge base | clean; research reference | E1 frozen contract, runner, notebooks; bundle and research evidence depend on branch. | UNKNOWN / N/A |
| `v7-e1-colab-recovery` | `fix/v7-e1-colab-recovery` / `0f06e36` | Colab recovery/shared Drive; owner UNKNOWN | `main@7bdf2e4` merge base | clean; recovery reference | Recovery logic and acceptance evidence; validate artifact dependencies before retirement. | UNKNOWN / N/A |
| `v7-e1-postrun-diagnostics` | `research/v7-e1-postrun-diagnostics` / `1d8e23f` | E1 admitted results/postrun diagnostics; owner UNKNOWN | `main@7bdf2e4` merge base | clean; latest research evidence | E1 result manifest, month/epoch diagnostic, K2-derived knowledge. Branch-qualified result authority persists. | UNKNOWN / N/A |
| `v7-historical-pit-reconstruction` | `feature/production-phase-a-shadow-ledger` / `f5b0ff3` | Phase A shadow ledger; owner UNKNOWN | `main@7bdf2e4` merge base | clean; unmerged, not deployed | Ops code/tests, Task 4 review and Task 5 replay pending. Directory name is stale; branch is authoritative identity. | UNKNOWN / N/A |

**Safe-removal preconditions for every row:** record owner approval and intended retirement date; confirm checkout clean; retain branch/commit/history; verify no unique uncommitted or ignored artifact is required; verify all authority/manifest absolute-path references and merge dependencies; inventory rebuildable runtime separately. This register grants no cleanup authorization. New worktrees require a bounded objective and a check that an existing checkout does not already serve it.

## Phase 3B-2 retirement preparation (observed 2026-09-22)

All eight checkouts were clean in `git status --short --branch`; their branch refs and commits remain present. A clean checkout alone never authorizes removal. The readiness labels below are planning labels only.

| Checkout | Readiness | Remaining authority / local dependency | Before any removal |
|---|---|---|---|
| `project-knowledge-k0-discovery` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | K0 history/report retained in Git and main navigation; no ignored payload observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `project-knowledge-k1-bootstrap` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | Bootstrap and legacy Claude history in Git; main governance supersedes active rules; no ignored payload observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `project-knowledge-k1-domains` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | Domain note lineage in Git/main; ignored Python bytecode observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `project-knowledge-k2-governance` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | K2 tools/rules adopted and evolved in main; ignored Python bytecode observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `v7-corrected-e5-2026-diagnostic` | `KEEP_AUTHORITY` | E5/E1 bundles, frozen result/evidence and branch-only diagnostics remain; ignored bytecode present. | Resolve bundle/result and artifact links before retirement review. |
| `v7-e1-colab-recovery` | `KEEP_AUTHORITY` | Drive rescue/migration and recovery evidence remain branch-qualified; local `.venv` and ignored review records require separate reproducibility/retention assessment. | Preserve evidence and verify environment/artifact identity; obtain owner approval. |
| `v7-e1-postrun-diagnostics` | `KEEP_ACTIVE` | Matrix recovery/replay and diagnostic authority remain open. | Complete task and preserve result evidence first. |
| `v7-historical-pit-reconstruction` | `KEEP_ACTIVE` | Actual branch is `feature/production-phase-a-shadow-ledger`; Task 4 correctness approval and Task 5 acceptance remain open; ignored review records exist. | Complete independent Phase A approval and artifact review first. |

A NUL-safe path comparison against main found branch-tracked paths absent at the same main path: K0=1, K1 bootstrap=1 (historical archive path), K1 domains=3 (archive plus two K1 task notes), K2=5 (archive plus K1/K2 task notes). These remain preserved in their branch commits; retirement requires retaining branch refs/history and confirming no checkout-path consumer. No checkout, branch, ignored file or artifact was removed in Phase 3B-2.

## Phase 3B-3A completed retirement (2026-09-22)

Human authorization in the Phase 3B-3A task covered only the four K checkouts. Pre-removal verification found each clean, its branch ref pointing to the recorded HEAD commit, no untracked user source, no symlink, no active absolute-path consumer in AGENTS/knowledge/docs/scripts/tests/tasks, and no required unique local artifact. `archive/V5.5_repo` was an empty checkout directory for a historical gitlink at `f9f5924`; that pointer remains in each branch tree, but its target object is unavailable in the current repository (`git cat-file -t` failed). No local target contents were present to lose. K1 domains had four and K2 twelve ignored `.pyc` files; normal `git worktree remove` removed them with their checkouts. No force option was used.

| Retired checkout | Retired at (+08:00) | Branch and preserved HEAD | Remaining authority | Local disposition |
|---|---|---|---|---|
| `project-knowledge-k0-discovery` | 2026-09-22 14:58–15:00 | `docs/project-knowledge-k0-discovery@c802f52` | Historical K0 discovery in branch Git history and main project map | Checkout removed normally; no ignored files |
| `project-knowledge-k1-bootstrap` | 2026-09-22 14:58–15:00 | `docs/project-knowledge-k1-bootstrap@31dd74a` | Historical K1 bootstrap/legacy guide in branch Git history | Checkout removed normally; no ignored files |
| `project-knowledge-k1-domains` | 2026-09-22 14:58–15:00 | `docs/project-knowledge-k1-domains@2c6c6f3` | Historical domain bootstrap and two task notes in branch Git history | Checkout removed normally; four reproducible `.pyc` files removed with checkout |
| `project-knowledge-k2-governance` | 2026-09-22 14:58–15:00 | `integration/project-knowledge-k1@18152d0` | Historical K2 governance and four K1/K2 task notes in branch Git history; current governance remains in main | Checkout removed normally; twelve reproducible `.pyc` files removed with checkout |

Each preserving branch ref and commit remained after removal. The `archive/V5.5_repo` **pointer** and branch-only task-note contents remain durable through those commits; the gitlink target content is not locally available and its external provenance remains UNKNOWN. Same-path copies in main were not required. Four research/operations worktrees remain active/preserved and were not touched. Exact observed completion time per individual remove command was not captured; interval above bounds this session and avoids false precision.

## Autonomous Charter window — 2026-10-21

- Path: `.worktrees/v7-autonomous-charter`; branch `development/v7-autonomous-charter`.
- Base: `6d74d7384caa5fa90b1b050ce507bfb8282f19df`; owner: Codex / Human first-window task.
- Created: 2026-10-21; retired: N/A; current state: WINDOW_COMPLETE, retained for review; UNMERGED / NOT_DEPLOYED.
- Purpose: Charter and first CPU-only units. Existing 15 linked worktrees preserve
  prior H0/CPU/research evidence; none serves this new main-based bounded window.
- Authority: Human charter decision; no new research/runtime authority.
- Merge dependency: separate Human approval; artifact dependency: existing recovery
  venv read-only for CPU tests, no data mounted. Main and old trees untouched.
- Safe removal: clean committed history retained, evidence/path dependencies
  checked and explicit retirement review; no removal authorized now.

Closure: five Charter-window units complete, final identity is the containing
snapshot/window commit; earlier units be7998d/3f07f80/5847afc/2c60415. Source and
tracked evidence committed; no data/GPU artifacts created. No active execution
responsibility remains. A future window must not infer automatic authorization.

## Charter fixed-candidate integration — 2026-10-21

`development/v7-autonomous-charter@f7af348` is now reachable from WSL main via
explicit Human ff-only grant; its worktree/HEAD remain untouched and clean.
All17 existing trees retained. New cross-domain window has separate authorization;
its checkout must start from verified main and be registered before execution.

## Cross-domain window02 — 2026-10-21

Path `.worktrees/v7-cross-domain-window-02`; branch `development/v7-cross-domain-window-02`;
base `c1179fe1ee8b66c768391b5f22f8aaebfc42d646`; owner Codex / current Human window02.
Created2026-10-21, retiredN/A; WINDOW_COMPLETE_STOP, UNMERGED/NOT_DEPLOYED. Purpose: four selected
cross-domain CPU units, not the fixed Charter integration. Existing17 trees serve
completed/historical work and cannot be repurposed without moving fixed evidence.
Authority: current Human decision; no research/production authority. Dependencies:
main closure, existing recovery venv read-only, temp fixtures; no data mount.
Safe removal requires clean preserved commits, no unique artifacts/path consumers,
resolved integration dependencies and separate retirement approval. No removal now.

## Window03 isolated candidate — 2026-10-21

Path `.worktrees/v7-data-evaluation-preparation`; branch
`development/v7-data-evaluation-preparation`; base9956446, integrates fixedf34f9f3
only within candidate. Owner Codex/current Human grant. Created2026-10-21;
retiredN/A; WINDOW_COMPLETE_STOP/UNMERGED/NOT_DEPLOYED. Purpose: integration readiness, deterministic
parser fixes and fairness preparation; other18trees retained with original purposes.
Dependencies: fixed parent commits, existing recovery venv read-only; no data mount.
Authority scoped engineering only. Safe removal requires clean preserved commits,
no unique artifacts/path consumers and explicit retirement; no removal authorized.

## Window02/03 integration — 2026-10-21

Fixedf34f9f3/eba6b649 trees preserved; now ancestors of local main. Their UNMERGED
status above is historical, superseded solely for local integration. No tree removal
or deployment. New fair-evaluation work needs separate isolated scope/worktree.

## Window04 fixed checkpoint — 2026-10-21

Purpose: bounded fair evaluation decision package and read-only metadata audit.
Owner Codex/current Human grant; created2026-10-21, retiredN/A. Branch
`development/v7-fair-evaluation-decision`, path `.worktrees/v7-fair-evaluation-decision`,
base`f31fcbba4d82d260915d7b640d80abc9c9ba8fc1`, HEAD`489a127cf951f93f7bfc873e9333f05113f5b8e9`. WINDOW_COMPLETE_STOP/UNMERGED/NOT_DEPLOYED.
Dependencies: pinned Git metadata, existing recovery venv; original matrix blocked.
No data mount. Main records pointer only; all20trees retained. Safe removal requires
preserved clean commits, no unique artifacts/consumers, resolved dependencies and
explicit retirement approval. No cleanup or tool integration authorized here.

## Fair evaluation decision window04 — 2026-10-21

Path `.worktrees/v7-fair-evaluation-decision`; branch
`development/v7-fair-evaluation-decision`; basef31fcbb after fixedeba6b649 integration.
Owner Codex/current Human grant; created2026-10-21, retiredN/A. WINDOW_COMPLETE_STOP/UNMERGED/
NOT_DEPLOYED. Purpose: bounded decision package and repeatable metadata audit.
Prior19trees preserved fixed purposes; no equivalent current window exists.
Dependencies: Git pinned metadata, existing recovery venv read-only, original
matrix unavailable. No data mount. Safe removal requires preserved clean commits,
no unique consumers/artifacts, resolved dependencies and separate retirement approval.
