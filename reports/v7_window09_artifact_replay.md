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

U1 初版18focused/independent18PASS；第一輪 review3Required 已用 RED evidence 修復：
路徑 .. 越界、manifest/publication 葉 symlink、JSON 1e400 overflow。Critical0/
Required0。自查 publication=null 的例外另以 RED 證據完成第二輪、也是最後一輪
容器修復；另驗證並行重複發布。U1 最終20testsPASS。U2 最終20testsPASS、1輪
修復；獨立40testsPASS，Critical0/Required0，已完成。

Artifacts verified、training complete、research admitted 分開。generated_at 是發布時
本機實際 UTC，logical observation/evaluation 時間是合成研究情境，不能當歷史
實際預測發行證據。既有 E1 store 使用 weights_only=False；本輪不呼叫其 load/save
或 orphan recovery，不載入未知 PyTorch payload。永久位置／保存年限／GC／資料與
研究 Admission／完整 GPU runtime acceptance 留給 Human。

恢復：cd .worktrees/v7-window09-artifact-replay，核對 HEAD/clean，讀 Current State、
Authority、Charter、research-program Window09 與 window09-evidence.json。
StrictSTOP/E1matrixBLOCKED/H2unapproved/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED；不讀正式 payload、不搜尋 E1 Matrix、不開始 Window10。

## U2 Replay 與 Restore 結果

Replay adapter 僅從 U1 verified Bundle 取得固定 schema 的合成輸入，核對六個關鍵
W08 source bytes、已記錄的 Python/torch/numpy/pandas 版本、config/data/fold/
calendar/feature/label/universe/policy 身份。使用原 evaluate/compare，未建立第二套
RankIC。再比對完整 daily metrics、paired comparison、rejection ledger 的 canonical
hash；exposure ledger 僅保留，不因此取得當 Fold selection/refit 權限。

初次研究輸出中的 signal/evaluation clocks 為2024合成情境；publication.generated_at
與 replayed_at 是本機2026實際 UTC。historical_live_prediction_proven=False。
重算成功不代表當年真的產生預測，也不代表模型能重新訓練。

| 能力 | 本輪證據 |
|---|---|
| METADATA_VERIFIED | TRUE：完整 schema、身份、實際附件 bytes/hash |
| EVALUATION_REPLAYABLE | TRUE：獨立 CPU process，daily/pair/rejections 完全一致 |
| TRAINING_RESUMABLE | FALSE：未驗證實際模型、optimizer、RNG/sampler/AMP runtime roundtrip |
| FULL_REPRODUCTION_READY | FALSE：未驗證完整 training pipeline、所有相依套件／資料來源與正式准入 |

四種能力互不自動推得。六個來源 hashes 是本 adapter 的關鍵來源集，並非完整
transitive runtime/dependency archive。Code commit 與內容 pins 不是來源真實性的
簽章；Synthetic 身份（含seed0、mock model hashes）不得當正式研究設定或模型權重。

Restore adapter 僅處理 SYNTHETIC_JSON_NOT_PYTORCH，重用 E1 pointer schema 與
validate_state，查每個 BEST/LATEST/PREVIOUS 的 bytes/hash/contract/matrix/seed/
run/model；不呼叫 torch.load。LATEST 損壞可報明原因並選合法 PREVIOUS；BEST
單獨回報，Orphan 僅列出，不改 pointer、不升格 BEST。Dataset bytes 不存在或
損壞會 NOT_RESTORABLE；hash-only 不構成還原方法。此處 matrix identity 為具名
合成 dataset 身份，不冒充缺失 E1 Frozen Matrix。合成模型/RNG字典只是 contract
測試，不是實際 PyTorch 狀態。既有 E1 Trainer/Adapter 不重寫，無任何真實訓練。

U2 首輪獨立 review1Required：scores=null 可拋例外；第一輪修復補巢狀容器與
restore identity 驗證，並讓缺失/延遲 Label 測試通過 rawref 檢查後實際進入 W08
maturity gate，保留 LABEL_ELIGIBILITY_RECHECK_MISMATCH 及拒絕詳情。

## 保存範圍與測試

本輪僅兩個工程單元；U1 source 79d6c556d1bb5648a40a02c3b6f0e628c6512b4a，最後容器修復與 U2 在 containing
closure commit。141聯合 tests PASS（既有101+U1 20+U2 20），14必要失敗情境
映射至 executable assertions。兩份新 JSON schemas／三份示例輸出通過 Draft202012。
所有 Frozen E5/E1 hashes 不變，原 Windows/Data/models/scheduler 與 production 未動。

既有 W08 範例沒有 transaction completion schema，故新增唯一、小型、可從 Git
直接還原的 window09-synthetic-bundle / synthetic-restore fixtures；只存此次合成
bytes，不要求大型 Matrix/checkpoint/predictions 入 Git，也不決定正式永久位置。
completed/ 內容與必要 source/proof 為 CANONICAL_SYNTHETIC_EVIDENCE；attempts/
staging/locks 是 GENERATED_LOCAL_DIAGNOSTICS，不清理既有產物，僅以 scoped
.gitignore 排除 runtime 工作檔。測試 TemporaryDirectory 只持有當次自建合成資料。

[Offline 證據](../research/v7/window09-offline-evidence.json)、
[Replay 結果](../research/v7/window09-replay-evidence.json)、
[失敗測試對照](../research/v7/window09-failure-coverage.json)。
Bundle publication 不自動成為 research admitted；必要大型資料沒 bytes 時明確拒絕。
本輪沒有 formal checkpoint/training log，因此不偽造這些附件；generic manifest
支援它們的保存或明確 reference，Replay 目前使用兩個具名 model alias a/b 的
合成 profile，並非完整 Registry 或任意 Trainer adapter。

## 後續限制與恢復

尚待真正 run/artifact 整合驗收：實際 immutable dataset bytes、完整 feature/label/
universe/provenance、依賴 lock、模型和所有 runtime states、exact resume roundtrip、
正式 prediction/label maturity policy 與正式 GPU/Colab acceptance。本輪不能以合成
成功替代上述證據，也不重做已存在 E1 runtime 功能或歷史研究規劃。

Human 必須另外決定永久儲存位置、保存年限/清理、正式資料使用與研究 Admission、
GPU runtime 驗收；這些尚未批准。Window09 無剩餘 READY 工作；下一步僅為取得
新的有界授權後，按既有工具補驗真實 run 的 artifact/runtime binding。

最小恢復：在 development/v7-window09-artifact-replay 核對 git status/HEAD，讀
Current State/Authority/Charter、唯一 Backlog Window09、window09-evidence.json。
window09-offline-evidence.json 提供 bundle root/run key/manifest SHA 及 CLI 參數；
使用既有 CPUvenv 呼叫 -m V6.experimental.v7_research_replay，output 選新的空路徑，
不可覆寫既存 evidence。Manifest 及所有附件 bytes 已保留，小型 fixture 可由 Git
取回；沒有將 hash-only 當還原證據。main 固定 bdea30d；新成果未整合、未push、
未deploy。本輪完成 STOP，不啟動 Window10。
