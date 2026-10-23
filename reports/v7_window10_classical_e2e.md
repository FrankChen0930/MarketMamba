# Window10 Classical Synthetic End-to-End Foundation

Status UNMERGED/NOT_DEPLOYED. W09 fixed91fa82c integrated from bdea30d ff-only,
pre/post141tests/39hashes/independent40tests GO; main held91fa82c. W10 isolated
development/v7-window10-classical-e2e. Two units only; toy fitting explicitly
authorized, no formal research or production changes.

U1: L0 uses explicit named toy factor/direction; Ridge NumPy64 closed-form solve
with means/std estimated exclusively from W07-eligible fit view. Constant features
use named UNIT_SCALE fixture rule. No sklearn dependency. Legacy ridge solver was
inspected but its wrapper imports/mutates historical feature configuration and
data-path globals; this minimal solve avoids that wrapper, reusing W07 role gates
and W08 metrics instead of legacy split/evaluation semantics. No old source edited.

L2 small CPU adapter present but lightgbm missing; actual fit UNVERIFIED/BLOCKED.
Candidate requirement lightgbm>=4,<5, exact version lock/actual test pending; no
network installation or global environment changes. Neither proposed dependency
range nor toy seed0/config/factor/selection metric is formal research approval.

U1 13focused/independent13PASS,0Critical/Required after1rework: full selection-run
hash binding and null-container rejection. Training/forward views remain W07 guarded;
prediction API accepts no Forward Label Store. Train-only scaler freshly computed
for each fit/refit, no checkpoint/resume input. Actual tiny fit status lives in
model artifact, separate from immutable W07 initial DECLARED authorization intent.

U2 complete: guarded selection/refit/forward, named synthetic policy, W08 comparison,
W09 v2 truthful fit-status profile and independent validation/forward replay.
13focusedtests/independent53 incl40oldW09PASS,0Critical/Required after1rework.

Resume named tree HEAD/clean, Current State/Authority/Charter, sole Backlog W10,
window10-evidence.json. StrictSTOP/E1BLOCKED/H2unapproved/FairEvaluation and
PreNextOpenPROPOSED/RecoveryDEFERRED. No real data read/search/train/GPU/production.

## 實際合成生命週期與身份

僅4檔 toy securities、32筆 feature samples、x/z兩個人工特徵。Label 值由 x/z
刻意線性生成，僅用 W07 交易日曆檢查 next-entry/5d/10d 成熟區間，不是歷史報酬。
每模型 selection train12筆、validation8筆、fresh refit24筆、forward8筆。
本次模型 head 為5d；監督樣本仍通過 W07 required5d/10d eligibility。沒有官方
48-feature/label/admission變更，沒有真正歷史 PIT 或金融效益證據。

| 階段 | L0 | Ridge |
|---|---|---|
| 初始宣告/Selection Fit | selection-L0 | selection-L1 |
| Fresh Refit | refit-L0 | refit-L1 |
| Forward | forward-L0 (alias b) | forward-L1 (alias a) |

Study=`study-fixture`、Fold=`fold-A`（具名合成fixture，完整身份由manifest pins固定）。
Toy validation兩模型 Mean RankIC=0.9999999999999998，具名 tie-break 選 L0；兩日
paired delta(b-a)=0。此結果來自刻意生成的合成排序，不支持任何金融模型結論。
兩模型都 refit 是固定工程比較參與者，並非 hyperparameter search 或正式 baseline
admission。Selection decision hash在Forward前後相同，Forward labels 只在 fit/
refit/prediction 完成後交給 evaluation capability；更改它們不改當次 selection/
fit/refit。CPU truth statuses與 W07初始授權意圖分開記錄；toy2024 ready/deadline
時間不是2024真的執行或Human readiness approval的證明，actual publication/replay
時間為2026本機UTC。

Ridge每次從合法 fit view重算mean/std和係數；Refit新Run且無resume checkpoint，
不使用前一Fit/其他Fold的scaler。L0只讀顯式x與方向+1；均非正式Factor決策。
L2僅依現環境回報DEPENDENCY_BLOCKED，不生成假的fit/evaluation/forward結果。
Candidate >=4,<5不是已認證依賴；精確版本/lock/實測尚缺，未安裝或改動全域環境。

## Artifact Completion 與 Replay

沿用同一 W09 publisher，新增明確 `v7-study-run-bundle-v2` /
`SYNTHETIC_CLASSICAL_FIT` profile，training_status=`SYNTHETIC_CPU_FIT_COMPLETED`；
checkpoint_status=`NOT_GENERATED_TINY_CLASSICAL_NO_RESUME`。原v1的NOT_RUN_SYNTHETIC
與NO_TRAINING enum保持原義，不以舊值冒稱沒有擬合。V2需要training log附件。

V2保存模型JSON/coefficients、recipe、config、fit metadata、initial/refit/forward
宣告、validation prediction/metrics/snapshot、forward predictions/daily/pair、
rejection/exposure ledger、environment/code manifests及completion evidence。
不製造不存在的PyTorch checkpoint或訓練runtime狀態，不設正式保存年限。

初版Review發現僅status的空模型ledger仍可Replay；第一輪RED/rework修復，現核對
config==training_contract_sha、prediction-model hash、recipe/run/fold/phase、
selection->newRefit->forward lineage及fresh/noresume pins。再使用W08原evaluate
重算validation與forward metrics，用具名toy rule核對selection，不新增RankIC實作。
每日、paired delta、rejections完全一致，Validation/Selection replay亦一致。
V1舊W09案例保持相容，未把engineer toy selection metric當成正式研究政策。

Bundle內保存真實小型合成bytes；partial/interrupted save不會完成，重複同內容
ALREADY_PUBLISHED；模型coverage不同使用共同support，排除原因明列。
新 `window10-synthetic-bundle` 命名空間必要，因舊W09不可變no-fit fixture不能
代表本次fitted producer；只存小型可重驗fixture，非新Registry/平台，不要求
大型Matrix/weights/predictions提交Git。completed為CANONICAL_SYNTHETIC_EVIDENCE，
attempts/staging/locks為GENERATED_LOCAL_DIAGNOSTICS，scopedignore保留本機、不GC。

獨立CPU process重讀Bundle並Replay，METADATA_VERIFIED/EVALUATION_REPLAYABLE為真；
TRAINING_RESUMABLE/FULL_REPRODUCTION_READY仍未證明。保存Ridge係數不等於證明
PyTorch/Mamba Exact Resume。Code manifest固定7個producer/evaluator關鍵來源，
不是完整transitive environment/archive或整個Study重新擬合的認證。

## 驗收、依賴與後續恢復

167joint CPUtests PASS（既有141+U1 13+U2 13）。U1 1輪修復2Required，U2 1輪
修復1Required；最終0Critical/Required。所有必要failure場景有executable assertions。
新v2schema/2outputs validated；Frozen E5/E1四hash不變，未動正式data/models/
Windows/scheduler/production。Main91fa82c held；source U1 4ec22a4330ed6db7df0b3dafb4106cea820516e2;
U2/final checkpoint為containing closure commit。UNMERGED/NOT_DEPLOYED/STOP。

真正研究仍缺：合格real dataset/feature/label/universe/PIT/availability證據、正式
evaluation/selection/split/seed/baseline admission；L2精確隔離依賴及CPU驗證；
完整runtime lock與actual optimizer/RNG/sampler/AMP/checkpoint roundtrip和GPU驗收。
沿用已有E1機制補驗binding，不提前重構Trainer，不猜測/重建Missing E1 Matrix。
進入Local GPU/Mamba-2須另有有界授權與明確acceptance/成本配置，本輪未執行。

Human仍決定資料範圍/decision-time與evaluation/selection參數、正式baseline研究方案、
artifact永久位置/保存/GC、dependency環境與真正GPU/Colab runtime admission。
下一高價值CPU候選為L2隔離依賴鎖定與tiny pipeline驗證，僅PROPOSED_NEXT_GRANT，
不自動安裝、研究、訓練或啟動Window11。

最小恢復：cd .worktrees/v7-window10-classical-e2e；核對git status/HEAD，讀首段
Current State/Authority、Charter、唯一research-program Window10及window10-evidence。
[End-to-end證據](../research/v7/window10-e2e-evidence.json)保存bundle root/run-key/
manifest SHA和獨立CLI參數；[Replay](../research/v7/window10-replay-evidence.json)。
呼叫既有CPUvenv -m V6.experimental.v7_research_replay時output必選新空路徑，
不得覆寫舊evidence。新成果等待Human整合批准，本輪完成後STOP。
