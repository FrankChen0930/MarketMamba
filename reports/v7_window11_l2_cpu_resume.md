# Window11 LightGBM Completion & Actual PyTorch CPU Resume

狀態：WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED。Main 僅 ff-only 整合固定
W10f251816，before91fa82c；pre/post167tests、40hashes、獨立26tests GO。W11自同一
main建立隔離branch，無額外main文件commit，其他未批准分支未合併。

## U1 真正 LightGBM

LightGBM4.6.0/sklearn1.6.1/joblib1.6.0/threadpoolctl3.6.0/cloudpickle3.1.2，五個
官方PyPI wheels逐檔size/SHA與官方JSON核對、require-hashes安裝專用venv。既有repo
lock已有sklearn/joblib/threadpool/cloudpickle版本；joblib缺cloudpickle的實際import
錯誤經診斷後補3.1.2，setupcorrection1。BaseTorch2.10.0+cpu/NumPy2.0.2/SciPy1.16.3
未變，無全域/Windows/Torch/CUDA升級。來源 https://pypi.org/project/lightgbm/4.6.0/ 。
Wheelhash證明下載bytes，非獨立安全審查。Overlay .pth依賴保留的CPUbasevenv，非完整
hermetic/archive；lock/cache/provenance在environments/v7-window11-cpu和.runtimes。

沿W10同一W07guard/W08evaluator/W09publisher，顯式comparison_model=L2，L1預設
保持。L2實際fit12/refit24toyrows，forward8rows；固定4trees/CPU1thread/seed0，
無search。Forwardlabel更改不改selection/refit；coverage不同使用commonvalidsupport。
模型boostertext、validation/forward輸出及lineage均保留，獨立process replay驗證。
Toy結果不是金融證據，正式features/labels/selection/admission未變。

## U2 實際 CPU 訓練與中斷

完全重用train_seed實際Forward/Loss/Backward/AdamW/clip/scheduler/validation/
selection loop，TinyCPUModel只有Linear2x2+Dropout(.25)，6參數。Config是具名toy
副本，2features/lookback2/3dailytrainbatches/4epochs；不改Frozen E5/E1 JSON。
E1 DurableCheckpointStore/LifecycleCheckpointAdapter重用原local serialize/fsync/
hash/durablecopy/reverify/atomicpointer。新增_read_payload codec seam，舊default
維持；ProofStore僅weights_only=True，NumPyRNG轉primitive，無allowlist/Falsefallback。
只讀本輪具身份/hash且新建的payload，未讀未知歷史PyTorch檔。完整generations不GC。

Core僅加optional step_observer(defaultNone)及engineering_probe(defaultFalse)；
CPU工程flag必須搭配明確CPU smoke入口，未放寬正式CUDA gate。輸出如實標
SYNTHETIC_CPU_ENGINEERING／historical readiness NOT_EVALUATED，並非研究准入。
Loss/LR observer使用W04 append_event/fsync，寫入故障直接中止。不存在另一套Trainer。

三個實際process：reference退出0；candidate在step4、epoch1/nextbatch1（warmup3
steps之後）完成optimizer+scheduler及pointer fsync後os._exit75；新process resume
退出0，最終12steps/4epochs。Comparision contract在執行前固定：CPU FP32/1thread、
同環境/seed17、所有非telemetrystate與tensorbytes exact，floating tolerance0。
LATEST/PREVIOUS/BEST各23欄位完全一致，含model/optimizer/scheduler/RNG/sampler/
epoch/globalstep/trace/selection與stopstate。Loss、LR、參數及earlyselection一致。
排除wall/CPU/I/O耗時、UUID檔名和publicationtimestamps，不假稱它們可逐位元相同。

證明等級：CPU_STEP_BOUNDARY_RESUMABLE + CPU_MID_EPOCH_RESUMABLE，僅本tiny模型、
固定daily-index/mapstyle順序、0workers、完整optimizerstep。未另測Epoch-boundary
interrupt，沒有GPU/AMP/shuffled或multiworkerDataLoader/部分batch/跨kernel/precision
承諾。CPU證明不是Mamba-2 exactResume或FullReproductionReady，也不授權GPU。

第一輪Review Required：空AdamWmoments、weightdecay999仍可接受；RED後以相同
meta-model/adamw_parameter_groups/AdamW模板核固定hyperparameters、params/groups、
step、moment shape/dtype/finite/variance，避免靜默重建或改config。第二輪Required：
source snapshot正向路徑header變數被覆蓋；RED後更名source_hash並測成功/失敗路徑。
U2已用兩輪修復，最終0Critical/Required。沒有再擴修新工程。

失敗測試包含corruptLatest/validPrevious（數值仍同reference）、missingcheckpoint、
wrongrun/data/model/config、afterwarmup、concurrent/sequentialduplicateResume、invalid
scheduler、brokenlog、interruptedpublication、missingSampler/incompleteoptimizer、
unsafe pickle。失敗明確FAILED，不重新從零偽裝resume；不auto-promote orphan/BEST。

## Source/Artifact 完整性

ActualCPU proof執行時HEAD為U1且有本任務dirtysource，完整sourcehash已保存。U2commit
e28ac62ed95007e246185c5c2a134d17a5d0e08b 精確收納那些bytes；source-witness明確區分observedHEAD與
之後sourcecommit，不假造當時Git狀態。U1source f00bebab33cdeaa75b8401c6a6988dfaa3d3e0e7。

新增CPUobserver使整份RankIC所在sourcefile的hash改變。Initial L2bundle/report保留
原來源，需原snapshot重播。另以顯式source_commit核實7個Gitblobs，freshsource-suffix
RunID發布current-runtime L2bundle；無覆寫舊Run。保留source/evidence完整歷史，closure
文件和新artifact留候選branch。不要把初版快照直接拿到不匹配runtime宣稱Replay成功。

[Current L2](../research/v7/window11-l2-runtime-evidence.json)、
[CPU proof](../research/v7/window11-cpu-resume-evidence.json)、
[Source witness](../research/v7/window11-source-witness.json)。CPU checkpoint/dataset/logs
共83檔、1,788,382bytes，實際保留 .artifacts/window11-cpu-resume-proof，逐檔inventory
含SHA/size。這些GENERATED_PROTECTED_PROOF bytes未提交Git，不得GC或移除worktree；
若bytes遺失，Git hashes不能還原，必須NOT_RESTORABLE。新toy dataset不冒充FrozenE1。
永久位置/保存年限未自行決定，E1matrix仍BLOCKED，無重新搜尋/重建。

## 驗證與剩餘條件

211jointtestsPASS，涵蓋原167、L2/source5、CPU9及E1checkpoint/lifecycle/train/W04log
必要回歸。U1setupcorrection1、reviewrework0；U2rework2，最終Critical0/Required0。
既有knowledgeHealth Governance/AuthorityDrift/LegacyDocs WARNING保留，無structuralFAIL。
正式data/models/Windows/scheduler未動，Frozen4hashes不變，main固定f251816。

LocalGPU仍缺專用GPU runtime exactlock、模型/kernel/precision/GradScaler/CUDA RNG/
loader-worker或shuffle state驗收，以及真正Mamba-2 state/compatibility證據。需要Human
批准下一有界syntheticLocalGPU窗口、device/VRAM/steps/walltime/resource條件；Colab/
付費資源另批。真實研究另外需要資料/PIT/labelmaturity/split/selection/decision-time/
admission規格；本輪均未批准。既有E1trainer不重建，CPU6參數證明不外推到GPU模型。

## 恢復與停止

cd .worktrees/v7-window11-l2-cpu-resume；git status/HEAD；讀首段CurrentState/
Authority/Charter、唯一research-program Window11及window11-evidence。先核CPUbytes
inventory與basevenv/overlay存在；已完成的CPUproof再次--resume應ALREADY_COMPLETED，
不應重新訓練。Current L2evidence提供sourcecommit/runkey/manifestSHA與ReplayCLI，
output必選新空路徑。新來源/U2已commit，本containingclosurecommit是最終checkpoint。
STOP；不自行合併新branch、push、deploy、啟動GPU或下一窗口。
