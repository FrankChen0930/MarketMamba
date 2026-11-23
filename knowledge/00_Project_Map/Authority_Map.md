## P4a 自動排練閘門的 authority 邊界 — 2026-11-24

`job-run-mwtk8k4r-a80ee8`／detached `bf0114d` 是 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED 候選。依 [Human 需求第 9 節](../../docs/operations/colab-human-requirements.md)，[notebook](../../V6/research_lab/p4_bundle.py) 正式前自動排練與 [runner](../../V6/research_lab/p4_colab.py) W≠60 前置檢查是工程 gate；`progress/*/rehearsal_passed.json` 僅觀測工程排練成功，不能取代正式 completed-fit／checkpoint／研究 gate。標記綁 run command、已驗 bundle source、prepared data／rankgauss 與實際 environment；無效即重做。TEMPORARY 排練 outputs 只在 `/tmp`。

[工具重產宣告](../../tasks/p4a-source-compat-20e6b54.json) 擴充的理由僅限自動排練閘門與長窗前置檢查，AST pin 與 CPU 位元／控制流程測試不等於獨立審查。36 完成 fit 採用條件、64-fit queue SHA、frozen research／runtime／environment／budget authority 維持；現場 CUDA／官方 mixer／W250 峰值／Colab／Drive 均 UNVERIFIED，原 NO_GO／Human acceptance 未解除。沒有 supersede 既有研究或 production authority，也沒有 runtime observed_at 更新。驗證與剩餘風險見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a Q3 標籤／排練候選的 authority 邊界 — 2026-11-23

本輪 `job-run-mwt8u5js-7f212c`／detached `685f6ec` 是 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；[p4_eval](../../V6/research_lab/p4_eval.py) 的 NaN 列身分／Q5 成熟 IC 修正與 [p4_colab](../../V6/research_lab/p4_colab.py) 排練入口是候選工程語義，非研究結果或 deployment authority。[Human 需求第 9 節](../../docs/operations/colab-human-requirements.md)是新增的上線前真實資料排練規則；CPU 測試不取代現場 gate。原 36 completed fits／Q3 事故屬 Human-reported evidence，非本輪 Drive observation。

[來源相容工具](../../V6/research_lab/p4_source_compat.py)與[工具重產宣告](../../tasks/p4a-source-compat-20e6b54.json)（GENERATED／CPU_ENGINEERING_ONLY）綁定工作 bytes、固定 queue SHA／研究／矩陣／環境／預算；新增採用前全部 36 個 Q1／Q2 resume_fit 驗證，不重跑、不改歷史 artifacts。AST pin 只是候選範圍限制，仍需獨立審查。排練僅 TEMPORARY／REHEARSAL，正式 resume_fit 不承認其完成檔。[工程報告](../../reports/v7_p4a_colab_engineering.md)與[Training note](../01_Domains/Training.md)提供驗證及剩餘現場風險；無既有研究／production authority 或 runtime observed_at supersession。

## P4a 來源相容升級候選 — 2026-11-22

`job-run-mwrxutx3-d28fa4` builder，沿用既有 detached worktree `builder_p4a_upgrade`，base/HEAD `20e6b5408d9f9b13d56290a72a2dc313c35d1b94`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 指示保留 patch。依 [Human 需求第 8 節](../../docs/operations/colab-human-requirements.md)新增 [來源驗證模組](../../V6/research_lab/p4_source_compat.py)，沿用 [runner](../../V6/research_lab/p4_colab.py)／[交付工具](../../V6/research_lab/p4_bundle.py)，沒有新 worktree／實驗樹。現有 runner 沒有跨來源證據層，因此在相同 research_lab 新增單一 helper；不複製訓練／完成驗證實作。

目前只認證 `4e56e9d` → 本工作樹逐檔 bytes。`deliver --resume-from-commit 4e56e9d` 由 git objects 重算全部舊來源與逐檔 diff／分類，綁定未變的契約／64-fit queue／DATASET_MANIFEST／環境／record_only 預算，宣告隨 ZIP 打包；[本輪 GENERATED 宣告](../../tasks/p4a-source-compat-20e6b54.json)供審查，不是新研究 authority。以舊 run 身分綁定的原 ZIP manifest 補足舊 run_identity 缺少的來源清單，完整核對 from／to、差異集合及非來源輸入後才採用舊 run_hash。run 目錄與帳本追加 SOURCE_UPGRADE，歷史 complete／pointer／checkpoint／run_identity 不改寫。缺證據、差異或多 lineage 一律 STOP；無舊 run 才新建，明設 allow-new-run 才允許新批次，既有帳本／STOP gate 不略過。

本節取代先前「任何換版只能重新綁身分開新 run」的操作限制：執行中的 kernel 仍不可換碼；斷線後才用 `03_formal_resume` 新 notebook，同一 EXPERIMENT_DIR／budget，保留原 02_formal ZIP 及原 notebook 保底。只限數值中立的進度／日誌／speed-test 帳本分離／文件；每個新來源基準先補跨版測試與獨立審查。研究、production、環境 authority 與 runtime observed_at 均未改動；本機唯讀原 ZIP 的 229 個來源與 git objects 完全一致，不能推定 Drive 現況。

最終完整 CPU 回歸 **774 passed／1 skipped／5 warnings（382.55s）**，4 項 in-memory mutation 皆攔截。CPU 跨版 GRU／crossstock（mixer 替身）權重／預測／RNG、舊 epoch pointer 接續、舊完成 fit 驗證及 12 完成＋52 剩餘的 runner／ledger 整合證據見 [工程報告](../../reports/v7_p4a_colab_engineering.md)。Colab／T4／Drive FUSE 現場 UNVERIFIED；無 GPU／網路／資料取得／production／scheduler／通知／額外代理。Next：獨立審查 patch、宣告與測試，接受後重新 deliver；現場測試與 Human acceptance 仍未完成，不解除原 NO_GO。

## P4a 進度兩項 REQUIRED 修補候選 — 2026-11-22

`job-run-mwrpdfba-d4ad9d` builder，既有 detached worktree `fixer_p4a_progress`，base/HEAD `5156087da0e2aaba5d0cf703db3cdd67200cfbe9`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 授權保留 patch。本節補正前節進度語義，沿用 [runner](../../V6/research_lab/p4_colab.py) 與 [進度測試](../../V6/research_lab/tests/test_p4_progress.py)：驗證 checkpoint 後設定本次 step 起點，至少 10 新 steps／1 秒才估 fit ETA；3900 → 4000／5000、10 秒的接續案例為 100 秒。佇列目前 fit 優先用本次 step ETA，後續 fit 用本次完成 fit 固定實測耗時均值（至少 1 秒），cached skips 不作樣本；無可靠量測即 null／預估中，Q5 仍額外未知。

略過已驗證 fit 也更新編號、Q、arm／year／seed／window／lr，清除前筆訓練欄位，進度行與 TXT／JSON 標示「已完成（略過）」並保留至正常退出；全部快取與 mixed skip/train 均逐 fit 驗證。中途接續恢復已提交 epoch／step，最後 fit 完成時兩個 ETA=0；不更改 checkpoint／完成 gate、64-fit 順序、模型／RNG、凍結研究與環境契約、帳本或 production authority。

CPU 合成完整測試 **745 passed／1 skipped／5 warnings（273.22s）**，四項 in-memory 變異皆被新測試攔截；notebook nbformat 與 8 個程式格 compile 通過。工具 knowledge-impact=PASS／DEFAULT，人工 gate 判定進度語義與 next decision 有影響，因此同步更新本節與 Training note。沒有 runtime observed_at 更新；Colab／T4／Drive FUSE 仍 UNVERIFIED。獨立複審仍待進行，不以 builder 完成取代原 NO_GO 或 Human acceptance。Next：獨立複審此 patch；接受後另案重新產生綁定新 source identity 的交付，現場驗證另行授權。實際輸出、artifact hashes 與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a 進度與 speed-test 帳本候選 — 2026-11-22

`job-run-mwrof3ay-8d0c43` builder，既有 detached HEAD/base `f6a173eab03332bc3c50d78885a1d865366d2518`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 授權保留 patch。依[Human 持續需求](../../docs/operations/colab-human-requirements.md)沿用 [notebook 產生器](../../V6/research_lab/p4_bundle.py)、[P4 runner](../../V6/research_lab/p4_colab.py)與 `seq_runner.train_one`：串流／flush、每 50 steps／epoch、60 秒時間心跳，Drive `progress/formal/` 與 `progress/speedtest/` 的 TXT／JSON、可行動錯誤及離線摘要格。快照是觀測，不是 checkpoint／完成／研究 authority；FORMAL 才顯示驗證 IC。ETA 未知即預估中，佇列以本次完成 fit 的實測耗時外推，不假設 Q5 耗時。

本節取代舊候選「speed 與 formal 共用 budget」的操作描述：speed 自動使用 `budget_speedtest/`，FORMAL 保持 `budget/`；各自跨交付延續，既有 chain／identity／缺帳 gate 不變，不遷移或重置歷史帳本。正在跑的原始 02_formal 不可中途換版；後續交付須重新綁定 source identity。FROZEN 研究契約、OBSERVED runtime bytes／observed_at、P3 adapter 與 portfolio engine 不變，無 production／scheduler／正式 Data／模型操作。只有 CPU 合成工程證據（權重／預測 bytes 與 torch RNG 一致、真子程序串流、帳本先 speed 後 FORMAL、in-memory 變異），Colab／T4／Drive FUSE 仍 UNVERIFIED；不等同 Human acceptance。Next：獨立審查 patch，接受後另案重新 deliver 與現場驗證。實際輸出與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a 三項 REQUIRED 修補候選 — 2026-11-22

`job-run-mwric0jj-32d745`，既有 detached HEAD/base `d82d0209cba6686c10fc1a01c78172446a6796fa`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 授權保留 patch。修補範圍限獨立複審的三項 REQUIRED：[evaluator](../../V6/research_lab/p4_eval.py) 對 Family A/B 列集合不一致明確拒絕並報差異，只有 C 取共同交集且先限制每個 seed 再重算 percentile rank／平均；保留交集與排除比例。[completed-fit validator](../../V6/research_lab/p4_colab.py) 綁定 frozen queue／每 fit 的 window、lr 有效配方，驗證連續 history、重算 BEST／bad、8 epochs 或有效早停、metrics／pointer／預測 fit 設定。發布與續跑共用驗證；舊缺 recipe／history 證據的完成標記 fail closed，不遷移或重標歷史 artifacts。

機器研究契約與 OBSERVED runtime bytes 不變，未更新 observed_at；P3 adapter／portfolio engine 原樣，無 production／scheduler／正式 Data／模型／GPU／網路／Colab變更。只有 CPU 合成工程證據；Colab／T4／Drive FUSE 現場仍 UNVERIFIED，工程完成不等於 Human acceptance。實際驗證與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。Next：獨立複審本 patch，再決定接受與重新產生綁定新 source identity 的交付；本輪不執行現場作業。

## P4a FROZEN 研究契約與工程候選 — 2026-11-22

[機器契約](../../tasks/p4a-colab-contract-20261121.json) 與 [凍結規格](../../tasks/spec-p4a-colab-20261121.md) 於2026-11-22、任何P4a結果出現前依Human本次核准凍結；同時綁定既有 [OBSERVED cp313 runtime契約](../../environments/p4a-colab-runtime.json)，不改其 observed_at。這取代下列歷史P4a的DRAFT研究狀態、時間/CU/25%reserve、partial不接續與內含矩陣交付描述，不取代P3的准入、14h預算、21/24結果或production authority。

`job-run-mwrgy396-cf1213`，detached base `f333421fafdcb02f716b16376ff65717021edfa4`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED、no-commit patch。[bundle](../../V6/research_lab/p4_bundle.py)／[executor](../../V6/research_lab/p4_colab.py)／[P4 evaluator](../../V6/research_lab/p4_eval.py)／[合成測試](../../V6/research_lab/tests/test_p4_full.py)決定此候選工程邊界；新增P4 evaluator以重用不可修改的P3/P2公開函式。matrix_v1逐檔SHA與X指定identity、bundle/source/environment identity共同決定run身份，code presence不代表Colab accepted。原`rolling_seq_scores.py`及`portfolio_eval.py`bytes保持不變；`seq_runner`僅新增opt-in精確resume與等值schedule helper。

固定64fits＋Q5，P4記錄帳本無caps，85°C硬體STOP保留；CPU位元一致不推定CUDA可重現或Drive FUSE原子持久性。歷史A/B/C各自Holm3/2/4不給獨立確認權；P3曝光、replication與C0+調參不對稱必須揭露。只有未曝光P2前瞻資料可循獨立確認路徑。無P4結果、無runtime observation更新、無deploy/scheduler/data/model authority變更。驗證與未驗項見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

# MarketMamba Authority Map

## P4a Drive 分次交付路徑候選 — 2026-11-22

`job-run-mwqscvk1-a3db0f` builder，base/HEAD `df78d6cb3bb706f6f4d58b94310c3d352aab8385`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依本輪 no-commit 授權保留 patch。沿用 [p4_bundle](../../V6/research_lab/p4_bundle.py) 的 deliver／notebook；新增 `--run-name`（01_speedtest）／`--drive-root`（MyDrive/MarketMamba）與兩行 README.txt，不建立平行流程或新 worktree。

第0格預填 `RUN_DIR=.../MarketMamba/p4a/01_speedtest`、`EXPERIMENT_DIR=RUN_DIR.parent`；runner／摘要／匯出都使用 `EXPERIMENT_DIR/budget`，speed 與 formal 跨交付共用同一本帳。ZIP 依 RUN_DIR → EXPERIMENT_DIR/inputs → /content 搜尋；殘留 /content SHA 不符明確 WARNING 並忽略。notebook 綁定實際 ZIP bytes/SHA，解壓前另核對 manifest `total_bytes`（含 aliases 邏輯大小）及實體成員大小，不混比壓縮與還原大小；不符 STOP 並提示上傳不完整。WHEEL_DIR 預設 MarketMamba/wheels，空／不存在才回退 Mamba_Wheels_T4_new，均無 wheel 才編譯寫回新目錄；部分 wheel／不相容仍 STOP。

SPEED_TEST 上限改1.0小時／15 CU；FORMAL 無預設，須 Human 明確填入。其餘0.5小時每fit/stage、10 CU/hour、25% reserve 保留；全新三臂 speed 計畫至少需2.5小時／25 CU加已花時間，因此小上限下計畫檢查會STOP，需Human依實際保守估計另行決策，不能放寬gate。既有舊根帳本不自動搬移／重置；若已有消耗，須保留原 EXPERIMENT_DIR 或先獨立盤點遷移。DRAFT 正式 gate、研究／production authority、runtime observed_at 與 lock 未變。

本輪僅本機 CPU 合成工程證據；沒有上傳 Drive、網路、GPU、Colab、資料取得或研究結果，現場行為仍 UNVERIFIED。此節取代前節的6小時／100 CU、/content優先與舊預設wheel路徑操作描述，保留歷史驗證紀錄。既有 fold resolved-root 邊界與壓縮比兩項 OPEN 限制未處理。Next：獨立審查 patch 與預算設定，核准真實輸入後重新 deliver；現場 Colab 驗證及 Human acceptance 另行處理。詳見 [指南](../../docs/operations/colab-p4.md)、[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a 兩檔 Colab 交付候選 — 2026-11-22

`job-run-mwqr9yiu-8c464e` builder，既有 detached HEAD/base `74795a4a56f25585000d26d281e305381de6230a`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，明確 no-commit 授權保留 patch。[產生器](../../V6/research_lab/p4_bundle.py) 的 `deliver` 一次輸出 `p4a-bundle.zip` 與 `p4a-colab.ipynb`，沿用既有 pack／executor／研究契約，沒有平行流程或新 worktree。ZIP 以內容 identity 去重跨 fold 的 X.npy，manifest 完整邏輯檔案清單加 aliases；安全解壓還原各 fold 的一般檔案並逐份驗 size/SHA。fold 輸入 symlink 只在打包端解析，解壓仍禁止 links，tar 相容與白名單／資源／cache gate 維持。

Notebook第0格集中預算與路徑，預設SPEED_TEST；找 /content 再找Drive、標準庫取內含bootstrap後完整驗證，不需手貼SHA或另傳bootstrap。先自檢環境，已符合契約則略過安裝；cp313 Drive wheel路徑 `MyDrive/Mamba_Wheels_T4_new`，cp312不符仍STOP。兩wheel均缺才使用官方sdist編譯備援（mamba C++20、MAX_JOBS=4）及Mamba2小型前後向冒煙。CUDA gate、DRAFT正式拒絕、既有resume／預算帳本與speed輸出隔離不變。

[Colab runtime契約](../../environments/p4a-colab-runtime.json)已是Human提供的OBSERVED Python3.13.16／torch2.14.1+cu130／CUDA13.0／T4／mamba2.3.2.post1／causal1.7.0；本節取代舊候選「等待填TBD版本」的next action，但不修改原觀測時間，不把Human smoke升格為專案正式GPU證據。本輪只有CPU合成測試／nbformat解析／三項in-memory反證，無網路、GPU、資料取得、正式Data、模型、production、scheduler或研究結果。Colab現場ZIP解壓、編譯儲存格、Drive掛載／持久寫入／下載仍 **UNVERIFIED**。Next：獨立審查patch；另行授權真實bundle打包及Colab現場驗證；FORMAL仍需凍結研究契約。細節與限制見[指南](../../docs/operations/colab-p4.md)、[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a Colab 操作修訂候選 — 2026-11-21

`job-run-mwqi30jf-aa0d7f` builder，既有 detached HEAD/base `36aa27da6e26c1877332c9cdea927e6409c3d105`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依本輪明確 no-commit 授權留 patch。無 GPU／網路／Colab／正式 Data／模型／production／scheduler 變更，未 commit/push/merge。

Human 已核准 archive／bootstrap 外部 SHA 不再必填：extract 自動印出 archive SHA，提供時嚴格核對；bundle 內逐檔 size/SHA、資源上限、symlink 防護與白名單仍強制。這取代本頁舊輪次「外部 SHA 必填」操作描述，不重標歷史證據。Drive `--wheel-dir` 模式只裝兩個 wheel（no-deps／no-index／find-links），記錄檔名／cp與平台標籤／SHA，標籤不相容或缺失／多版本 STOP；不改 torch 或其他套件。

新增獨立 [Colab runtime 契約](../../environments/p4a-colab-runtime.json)，初始 TBD／UNOBSERVED，待 Human pip list 與 runtime 自檢填實際觀測版本，不猜 torch／CUDA 組合。本機 [P3 參考 lock](../../environments/p4a-colab.requirements.lock.txt) bytes 不改。speed-test 保存逐項 environment_mismatch 與「結果不可與正式環境比較」，正式模式要求 OBSERVED 契約完全相符，CUDA 不可用仍一律 STOP；DRAFT 研究 gate 不變。P4a 三臂須在同一 Colab 環境 identity；與本機 P3 的比較需揭露環境差異。程式／測試／[指南](../../docs/operations/colab-p4.md)／[報告](../../reports/v7_p4a_colab_engineering.md)是工程候選，沒有新研究結果或 Colab acceptance。Next：獨立審查 patch；Human 提供 pip list／runtime 自檢，完成契約觀測後重新 pack，仍須研究凍結與現場 GPU／Drive 驗證。


## P4a Colab 候選邊界 — 2026-11-21

[P4機器契約](../../tasks/p4a-colab-contract-20261121.json)、[規格草案](../../tasks/spec-p4a-colab-20261121.md)目前DRAFT / NOT_FROZEN；[bundle](../../V6/research_lab/p4_bundle.py)、[executor](../../V6/research_lab/p4_colab.py)及[測試](../../V6/research_lab/tests/test_p4.py)只屬 `job-run-mwqd0vuy-7d9826` detached base `e4f21c3` 的 UNMERGED_IMPLEMENTATION / NOT_DEPLOYED / NOT_ACCEPTED，原候選與第一輪修補已分別提交於 `8cb61c4`、`05cb840`；第二次修補已提交為 `a76e98d`（此後提交狀態以 git log 為準）；正式P4仍需凍結草案與現場GPU／環境驗證。沒有P4結果可supersede。

P4沿用`train_one`、P3 purge／row semantics，但獨立Drive預算根目錄／bundle identity；不得重用或重置P3 `.artifacts/p3`帳本、放寬P3 admission，或重標21/24結果。P4的exact lock原樣保存在[獨立檔](../../environments/p4a-colab.requirements.lock.txt)（SHA `2c39a3fb9567d1f9fc8ee83369c7b6c90cd19a8e679e32e4c8a6c1d75c5782c9`），不取代舊E1環境authority。Colab wheel安裝、kernel／Drive durable semantics尚未驗證；CU為Human提供費率估計，不是provider實扣evidence。CPU fixtures只能證明工程邊界。詳見[工程報告](../../reports/v7_p4a_colab_engineering.md)。


第一輪 `job-run-mwqe7o12-d33ef8` fixer 以 `8cb61c4` 為 base 進行審查修補，已由後續流程保存為 `05cb840`，仍 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。修補 speed 全輸出隔離、解壓資源 gate、executable-cache 排除與 pre-import 清理、completed-fit 必要 schema。規格保留 OPEN 研究決策，已曝光 2021–2025 只作描述性；GPU／ABI／Drive 持久性仍 UNVERIFIED，CPU 工程完成不授予 Colab 或研究接受。

第二輪 `job-run-mwqew35j-886363` 修補以 `05cb840` 為 base，依 no-commit 留 patch，仍 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。新增 raw header／解壓讀取雙上限、Linux descriptor 路徑寫入與 exact bundle 路徑白名單，只屬工程安全 gate，沒有凍結研究契約或升格 runtime authority。當輪外部 SHA 必填策略已由上節 Human 操作修訂取代；manifest 仍不能自證來源。

## P3 修訂3 預算完整性候選 — 2026-11-20

[凍結修訂3](../../tasks/spec-mamba-portfolio-p3-20261120.md) 與[adapter／帳本](../../V6/research_lab/rolling_seq_scores.py)、[回歸測試](../../V6/research_lab/tests/test_p3.py) 決定本 bounded candidate 的預算語義。`job-run-mwoibr4r-44988a`，detached `39320a0`，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，未提交；更新下節修訂2之完整性／重建邊界，不 supersede 歷史 artifacts 或研究／production authority。全域 budget-global.jsonl 與 budget-anchor.json 必須完全一致，fit sidecar／manifest 記錄當時 sequence／head_sha256／spent；合法前綴或清空並非有效恢復。缺 state 必須顯式 REBUILT，以所有 sidecar 最大 spent 及存活 anchor／可驗帳本保守扣帳，保留舊 bytes／hashes，重建鎖 STOP；REBUILT 不提供舊 fit 的 completed binding／確認性 authority。此機制防疏失，不防刻意刪除全部 ledger／anchor／sidecar。詳見[工程報告](../../reports/v7_p3_portfolio_engineering.md)。GPU_SMOKE_NOT_RUN，工程完成不代表Human acceptance。

## P3 bounded builder candidate authority — 2026-11-20

[P3 frozen registration 修訂2](../../tasks/spec-mamba-portfolio-p3-20261120.md) owns research assumptions；修訂1的官方 temporal 預設卷積／cross d_conv2 仍有效。`job-run-mwnv3018-83e5c9` detached base `a91f38e` 的[adapter](../../V6/research_lab/rolling_seq_scores.py)、[evaluator](../../V6/research_lab/portfolio_eval.py)、[tests](../../V6/research_lab/tests/test_p3.py) 為未提交工程候選，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，不 supersede P1/P2、歷史 artifacts 或 production authority。

只有 executor 產生的 p3-official-gpu-smoke-v2、獨立 review job 的結構化 reviewed_by 與可稽核 standing_authorization，且所有 source/spec/environment hashes／frozen cfg／官方 defaults 符合，才可准入。`p3_main` 重新驗 evidence、全域帳本 SHA/spent/STOP、run_identity 與24fit run_hash/sidecars；手寫 RESOLVED 不具 authority。全域 `.artifacts/p3/budget-global.jsonl` 是本候選唯一預算 authority，不依 out-dir；固定目錄 inode flock、hash chain、STOP/CLEARED 事件由 adapter 管理，直接刪 STOP 不授予續跑權。缺歷史帳本不自動遷移或當0。詳見[工程報告](../../reports/v7_p3_portfolio_engineering.md)／[Current State](Current_State.md#p3-序列分數組合工程候選--2026-11-20)。GPU_SMOKE_NOT_RUN；CPU boundary doubles／測試通過均非官方 GPU evidence、Human acceptance 或交易准入。

## P2 amendment 1 candidate boundary — 2026-11-19

[Frozen P2 registration including amendment 1](../../tasks/spec-lowturnover-p2-20261119.md) owns research assumptions; [portfolio](../../V6/research_lab/portfolio.py)、[evaluator](../../V6/research_lab/portfolio_eval.py) 與 [P2 tests](../../V6/research_lab/tests/test_p2.py) 是 detached `1920fde` bounded repair candidate，UNMERGED / NOT_DEPLOYED。修訂 1 supersede 原 P2 全因子網格及主要基準最低費用假設；不 supersede P1／歷史 artifacts。主要基準只收比例成本，最低費用基準敏感度 descriptive／confirmation_eligible=false，不進 H-P2a/b。OFAT＋三指定組合去重為14格／fold×system、168 P2格＋12 P1 references。P2全格（含buffer=top_k）明確啟用preserve_frozen_overflow，P1預設關閉；凍結overflow保留全部既有股數，不新增／不強制賣出可成交持股，terminal不變。預設P1算式不改；位元一致僅預設逐日策略／基準報酬，以pre-P2引擎golden及舊P1 part比對驗證；schema與run_hash不同、舊P1 parts無法續跑。--fold／--system子集必grid_complete=false，各自run_hash不得拼成確認性結果。

P2僅--mode p2啟用，使用`.artifacts/p2/`、input/spec/implementation hashes、source commit、hash-bound immutable parts/retry chain；signal-close排名與initial-NAV notional×initial_capital換TWD語義不變。H-P2b只配對共同日期，尾端明列；H-P2a/b共同Holm family2，LightGBM描述性。2023單折與完整估時是工程 evidence，不升格研究確認／production authority或Human acceptance。詳見 [Current State](Current_State.md#p2-amendment-1-bounded-repair-candidate--2026-11-19) 與 [工程報告](../../reports/v7_p2_portfolio_engineering.md)。

## P1 bounded diagnostic/retry follow-up — 2026-11-19

`job-run-mwmgvttf-837c02` 的 detached `d2390b5` fixer 為 UNMERGED / NOT_DEPLOYED、uncommitted candidate；[evaluator](../../V6/research_lab/portfolio_eval.py) 與 [P1 executable tests](../../V6/research_lab/tests/test_p1.py) 決定本輪缺價分類、volume magnitude diagnostics、part fold/system binding 與 explicit ERROR retry 語義。[Frozen P1 spec](../../tasks/spec-portfolio-eval-20261119.md) 的 fitting／主要假設與引擎 accounting 不變。只有 fold 內末端連續至少 10 個 observed sessions 無有效 close 的類別對應 termination exposure；pre-fold／transient 分列，未知下市原因不升格。容量只是 NAV=1、未調 NAV drift 的 approximation。

ERROR 預設持久保留；`--retry-errors` 的連續 immutable attempt／previous-payload-hash 鏈只可恢復最新通過驗證的狀態，成功 cells 不重新計算，原始 parts／報告不覆寫。所有部分均綁 run identity 與 fold/system；舊實作 parts 不得混用。缺 paired primary 的 fold/system 必須明列，部分支援不稱完整。此處更新的是上一節 P1 review-repair 的 candidate durability/reporting 邊界；未 supersede 歷史 artifacts、資料來源 authority 或 production。Volume 的 2330／8069／1236 及全宇宙年度／每日量級一致性只是可重現工程證據，UNKNOWN／不一致 fail closed，不等於每列 source verification。範圍與驗證見 [Current State](Current_State.md#p1-bounded-evaluator-follow-up--2026-11-19)；Human acceptance 另行決定。


## P1 review-repair boundary — 2026-11-19

The [registered P1 assumptions](../../tasks/spec-portfolio-eval-20261119.md) remain frozen. Repair candidate at detached `abf0368` in `.worktrees/ruflo-workers/run-mwmgdq5i-f34fa7/fixer` changes [evaluation durability/reporting](../../V6/research_lab/portfolio_eval.py) and additive [position diagnostics](../../V6/research_lab/portfolio.py), governed by [regression tests](../../V6/research_lab/tests/test_p1.py). UNMERGED / NOT_DEPLOYED; no production/data/PIT admission or historical-artifact supersession. A part is reusable only when its run identity and payload SHA-256 match; ERROR parts are durable evidence, not successful inference. A fresh namespace is required for a new attempt/configuration. Final clean-fold completeness owns confirmatory promotion; partial cells stay candidate_confirmatory. Unit magnitude evidence is explicitly weaker than per-row exchange provenance.


## P1 candidate implementation boundary — 2026-11-19

The [registered P1 spec](../../tasks/spec-portfolio-eval-20261119.md) owns the frozen research assumptions. [rolling_scores](../../V6/research_lab/rolling_scores.py), [portfolio_eval](../../V6/research_lab/portfolio_eval.py), optional universe masking in [portfolio](../../V6/research_lab/portfolio.py), and [P1 tests](../../V6/research_lab/tests/test_p1.py) are uncommitted builder candidates at detached base `a0e1f00`, UNMERGED / NOT_DEPLOYED. Existing rolling splits, C0+ fitting and portfolio execution/cost semantics remain their implementation authorities; no historical result or production authority is superseded. Ignored `.artifacts/p1/` hashes/manifests identify local GENERATED E2 RESEARCH_ONLY evidence, not accepted results. Smoke completion is not H-P1 confirmation or Human acceptance. Full clean-fold/seeds training, grid execution and independent acceptance remain outstanding; missing PIT/delisted/adjusted-price evidence cannot be inferred from code or positive returns.

## Imported research code boundary — 2026-11-19

`V6/research_lab/` and the `tasks/roadmap-20261115.md` research log are imported from pilot09 `rd/hybrid` `65d0b68` as RESEARCH_ONLY candidate code; they carry no production, PIT-admission or deployment authority. The roadmap is a historical research log whose numbers must not be compared across different data/label/validation protocols. See [Current State](Current_State.md#research-code-import-from-pilot09--2026-11-19).

## Version-control and executor policy — 2026-11-19

[AGENTS.md](../../AGENTS.md) now records Human authorization to commit and push branches/`main` to `origin` (no force-push, history rewrite, or remote-branch deletion), the Codex-first executor policy, local-first compute rule, and the pilot09 research repo as read-only reference. Deploy, scheduler/`v62_daily.bat` edits, Colab spend and production authority are unchanged and still require explicit approval. See [Current State](Current_State.md#human-policy-update-v7-direction-version-control-model-split--2026-11-19).


## Ruflo development collaboration boundary — 2026-11-19

`development/ruflo-integration` (base f501a7e), UNMERGED / NOT_DEPLOYED: [versioned integration contract](../../tools/project_agents/integration-contract.json), [adapter/tests](../../tools/project_agents/runtime.mjs) and [operator guide](../../docs/operations/ruflo.md) own only the development collaboration capability. Official DualModeOrchestrator schedules workers; project adapter pins runtime and validates processes/results. Runtime observations remain timestamped local evidence. Ruflo memory, agent registration, worker completion and session summaries never supersede project map, contracts, accepted research artifacts or production authority. No new scheduler, data, GPU, merge/push/deploy authority. [Bounded acceptance](../../reports/ruflo_integration_acceptance.json) is local engineering evidence only; primary checkout branch development/ruflo-primary does not promote this to main or production. SQLite backend uses official disable-bridge policy after actual cross-process validation.


## V6 fetch hardening candidate — 2026-11-18

`fix/macro-sync-fetch-hardening` from `bc88a181a15ad20e41daf29ec68a623d99ccf5d8`
is UNMERGED / NOT_DEPLOYED. [Fetcher](../../V6/marketmamba/data/fetcher.py)
and [offline tests](../../V6/scripts/test_fetcher_date_gaps.py) define candidate
single-session date/title binding and observed-price-calendar internal-gap
repair. Follow-up to `721eb84`: default 30-calendar-day lookback (positive override),
latest 10 missing sessions per source by default, missing-weekday-today candidate
independent of price-calendar presence, and market-value candidate age <=10 calendar
days with explicit non-PIT warnings for older skipped gaps. Market value requires
actual price rows and reports only written dates. These candidate semantics
supersede the unbounded scan/calendar-only today behavior in `721eb84`.
`_trading_calendar()`
remains the existing V6 availability convention, not a qualified PIT calendar.
No formal dataset, historical artifact, production or research authority is
superseded. Current MOPS shares remain non-PIT for historical market value;
correct date binding does not prove historical publication time or full universe.
See [Current State](Current_State.md#v6-fetch-date-binding-and-internal-gaps--2026-11-18).

## Run status count adoption boundary — 2026-11-13

`feat/run-status-count-validation` carries the independently adopted MAS count
formatting patch from base `8b0786018d763942d0ea68eb9ce537cb33f8b1fd`,
LOCAL_MAIN_INTEGRATED / NOT_DEPLOYED at `e06fb256211f21612a25af4b10f709a756540d32`
via Human-authorized fast-forward. Implementation and executable tests at
`app/frontend/src/components/runStatusSummary.mjs` and
`app/frontend/src/components/runStatusSummary.test.mjs` define this display-only
behavior. Patch identity and verification are recorded in [Current State](Current_State.md#run-status-count-patch-adoption--2026-11-13).
No production/research authority is superseded. The original MAS goal's historical
Git Delivery remains incomplete; this adoption does not amend its evidence/state.

> Status: `CANONICAL_CURRENT_PROJECT_MAP`
> Owner: `Human + project agents`
> Semantic change owner must update this map in the same change.
> Prior research/Phase A map accepted `2026-09-20`; 2026-09-22 runtime additions are agent-observed, pending separate human review. Canonical means current navigation authority, not immutable final state.

## Window37 W34 Read-only Model Integration — VERIFIED CANDIDATE

Main72d35e7/W34 source82cd49c diverged atc57ac3d. Isolated `integration/v7-window37-w34-models` ports only readonly loader and original scores function, plus byte-identical model/recipe/feature/replay evidence. No new Fit/GET, no old training/fetch entrypoints, W35 Recovery unchanged. W34 original161files+restore verified; necessary JSON model/scaler/recipe/serving-schema pins retained. [Integration](../../research/v7/window37-integration.json), [adoption](../../research/v7/window37-model-adoption.json). Independent Gate GO Critical0 Required0；2models tests＋15W35 regression PASS，512原分數exact Replay；Human已授權必要驗證後main ff-only，實際整合identity由W37 closure補記。
W36 remains separate protected branch b2bc041; no Task/runtime/artifact/budget mutation. Its limited tasks are independently active under HumanW36 only, not a Shadow authorization. W34 models OFFLINE_VALIDATED, no Production/StrictPIT admission. BACKUP_PENDING deferred by Human; no independent backup claim. W37 manual Shadow implementation only after integration, no schedule or waiting; current Nov2 night is not Preopen.

## Window35 Five-session Collector Closure / Recovery v0.4 — COMPLETE / STOP

W35 branch `research/v7-window35-recovery-v04`，base main `c57ac3d7822f229ae08849e56581ebb3c969441c`，UNMERGED / NOT_DEPLOYED。W19cf1與main分岔於87695，無合併/Runtime移植；main既有verified_record/QC/snapshot/restore可驗證既有scheduled records。W34僅exactGit82cd固定64ID只讀，無模型載入/修改/整合。
五日5/5runs、10/10sources、原10GET證據確認12031rows/11814basicPASS/217missing/Fatal0。完整Raw TWSE72 vs69exceptions、TPEx145 vs100，補48明細；121個market/security缺1/2/3/4/5天69/23/17/9/3。217全部OUT_OF_W34_TWSE64_UNIVERSE；cause仍UNKNOWN，已確認合法無價/停牌0、SourceFailure0，不因持續性或商品樣式推定。64股每day64有效OHLCV/兩Featureinput，共320，0RankingScores。
Task唯讀觀測2026-11-02T21:27:33+08：Ready/enabled/last19:00:01result0/NextRun=null（不據此稱停用），原End23:59:59觀測時未到，EXPIRING_PROTECTED_EVIDENCE。W19Task/政策/10GETbudget/135原檔hash均保留未改；不延長或重啟。
最小v7_data_recovery.py僅offline分類/決策/immutable claims/outcomes；resource/session整份CSV計數非per-stock，1content與候選2network retries分開，offline physical參考cap4，正式營運Retry參數未凍結。來源停止跨Session保留，wrongSession/sameContent/stillMissing停止，Backfill不冒充Forward。
主要Review原NO_GO/Required1保留；R1 failure-parent retry分叉已單輪修復與[Independent Closure](../../research/v7/window35-review-closure.json) GO Critical0／Required0。15tests PASS；同event冪等、每失敗parent一次、下一retry只能跟新failure。原replay保留，postrepair獨立[Replay](../../research/v7/window35-replay-closure.json) 重現12031/217與Synthetic修復/limit-stop，0GET/Fit/GPU/Taskchanges。
10個W19只讀replay、41offline檔9409116bytes/份restore，135原W19files未變；BACKUP_PENDING，同機非獨立備份。[Evidence](../../research/v7/window35-evidence.json)、[完整Missing](../../research/v7/window35-missing-distribution.json)、[Report](../../reports/v7_window35_collector_recovery.md)。
[Activation Candidate](../../research/v7/window35-activation-candidate.json) 僅PROPOSED：Normal19／Conditional07、Interactive限制/原W19期滿隔離/新限期Task/保存備份監測。真正live Normal/Recovery dispatcher仍未移植接合，不能直接啟用；下一Human最少批准W35整合、有界live adapter/驗收及限期操作設定、calendar/retry/update-evidence與獨立備份位置。
恢復：W35 worktree → Current_State → 唯一Backlog → window35-evidence.json → closure/report。無剩餘本輪READY，STOP；StrictSTOP/FormalAdmissionNOT_APPROVED/W34OFFLINE_PREPARATION_ONLY/W12SAFETY_HALTED。不得自整合W35/新GET/長期Task/Shadow/下一窗口。

## Window33 Recent Data & Initial Shadow Model — DATA PRESERVED / FIT BLOCKED / STOP

W32 fixed `8fa41869a2583f0b5db698db83eed98611f24a7e` 已由main67e2d25本機ff-only整合；Review GO/11Synthetic原檔及restore/13tests/治理核對。W33 branch `research/v7-window33-recent-shadow`，base同固定SHA，worktree `.worktrees/v7-window33-recent-shadow`；UNMERGED / NOT_DEPLOYED。GET前91d7cca固定W29同64檔/日期/guard。
64 FinMind price GET全200、0metadata/retry/token/fallback，7604573rawbytes；最小間隔5.001249s、最高known prior hour63<100，額度CLOSED/剩6不重用。43437rows，2024-01-02～2026-10-22、679OBSERVED sessions（242/243/194）；58zero、19union-relative missing，不補值或推定停牌。
[Data](../../research/v7/window33-data-evidence.json)：43319數值Feature候選，5d42720／10d42173數值Label候選；近期Calendar沒有合格官方/例外休市/時鐘證據，W07回CALENDAR_QUALIFICATION_UNKNOWN。全部新mask fitFalse；未因Vintage未知全面拒絕資料，也不把價格union升成qualified。8422於2025-11-17疑似單位不連續，事件原因UNKNOWN，沿用既有flags不修價。
[Compatibility](../../research/v7/window33-compatibility.json)：W16/W18同Oct22原始CSV；64檔OHLCV精確相等，官方current＋明示FinMind prior的兩Feature hybrid全相等；非連續兩官方Snapshots獨立parity。Feature order/shape2保持。
實際Training N0／Ridge0／LGBM0／Offline Scores NOT_RUN_DEPENDENCY_BLOCKED；沒有新模型、Prediction或RankIC。Prediction Replay N/A，不能冒充完整Shadow準備成功。W30Recipe與W27 safety不變；2024–2026 EXPOSED。
264files58910660bytes/份獨立程序Raw/Derived restore，68old rawpins未變；CPU28.817576＋60reserve<1800，artifact+restore117822087<1GiB。Private `.artifacts/window33-recent-shadow/`／`.artifacts/window33-restore/`，BACKUP_PENDING同機非獨立備份。
8targetedtests PASS；獨立Review GO（資料保存與阻塞收尾）Critical0／Required0，0輪review rework；見window33-review.json，不代表訓練Gate已通過。[Evidence](../../research/v7/window33-evidence.json)、[Report](../../reports/v7_window33_recent_shadow.md)。初期interpreter/import/probe shape失敗trace與CPU計帳保留，無額外GET或輸出覆寫。
下一Human：有界補足近期合格calendar/休市證據，或批准另具身份的觀測Calendar代理語義；不得只改qualification字串。解除Gate後另授權固定4fits及Offline驗收；真正PREOPEN另需模型/source/time/missing/recovery及操作授權。本輪STOP，不自動續期Collector或開下一窗口。
恢復：W33 worktree → Current_State → 唯一Backlog → window33-evidence.json；不得重跑已CLOSED source。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED無任何讀寫/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED，無Scheduler/GPU/Push/Deploy操作。

## Window32 Expanding Closure & Prospective Shadow — PREPARATION COMPLETE / STOP

W31 fixed `67e2d25a710273646d41c0ad861b4a993593ba4d` 已由main b64ac35本機ff-only整合；72原檔/restore、24views/24model identities、maturity及10tests核對。Refit−Train−Validation差額5d300/305/310、10d597/610/620均為TRAIN邊界排除後已成熟的舊labels，非Forward。未重Fit/重算舊RankIC，W31歷史Warning保留。
W32 branch `research/v7-window32-shadow-preparation`，base同固定SHA；worktree `.worktrees/v7-window32-shadow-preparation`，LOCAL_INTEGRATED_FIXED8fa4186 / NOT_DEPLOYED。
[Shadow Contract](../../research/v7/window32-shadow-contract.json) 為PROPOSED；正常目標07–08Asia/Taipei，持久發布嚴格早於合格Entry開盤才PREOPEN，晚到MISSED_PREOPEN/RESEARCH_ONLY。未凍結精確source freeze/wait/missing政策或啟用排程。
Synthetic-only薄入口重用W07時鐘/日曆、W09 strict JSON/lock/safe path及既有atomic-no-clobber；無真實模型載入、Fit或Inference CLI。Feature-only keys先保存，5d/10d成熟後才Join，Backfill不冒充Forward。13targetedtests PASS；11個Synthetic檔16123bytes/份同機restore；BACKUP_PENDING。
[Readiness](../../research/v7/window32-readiness.json)：已核immutable訓練資料僅2021–2023，不能宣稱2026FreshRefit就緒。W31兩Feature schema可重用，Daily CSV尚需units/security/market/feature parity及合格Universe/Calendar/events；V6的56/59維Inference不可直接沿用。
W19僅讀固定Git policy物件作參考，沒有Task/worktree/artifact/runtime存取或續期；五次單日snapshot不能取代多年訓練資料，實際Task狀態未觀測。W06/W12/W19均未整合。
[Evidence](../../research/v7/window32-evidence.json)、[Report](../../reports/v7_window32_shadow_preparation.md)、window32-review.json GO Critical0／Required0，0輪review rework。下一Human：初始資料/事件/Label准入與限額FreshFit、Shadow時間/missing/evaluation契約、模型pins、起訖/環境/儲存備份/監測及操作授權。無剩餘已授權READY；STOP，不產生真實排名或開下一窗口。
恢復：W32 worktree → Current_State → 唯一Backlog → window32-evidence.json；只讀檢視，不啟用任何Collector或Shadow。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED；InferenceRecoveryImplementation仍DEFERRED。

## Window31 First Expanding CPU Development — COMPLETE / STOP

W30 fixed `b64ac353833a9248121f12ebe61e1333decb2dbf` 已由main e83dfe6本機ff-only整合；38pins/restore及6mock-only tests通過。970 daily＝940 defined＋30 None；2910 paired＝2820 defined＋90 None；原模型351/379 split nodes、預測非退化，未重Fit。見 [Integration](../../research/v7/window31-integration.json)。
W31 branch `research/v7-window31-expanding`，base同固定SHA，worktree `.worktrees/v7-window31-expanding`；LOCAL_INTEGRATED_FIXED67e2d25 / NOT_DEPLOYED。首次Fit前 `ddcf5d835b43578ed9fc5567ee677d88e928e203` 固定 [Contract](../../research/v7/window31-contract.json)／[Prepared](../../research/v7/window31-prepared.json)。
三Origin2023-01-03／04-06／07-03，Forward54／59／63sessions。5d TRAIN14113/17559/21321，REFIT29229/32574/36276；10d TRAIN13711/17143/20890，REFIT28743/32046/35715。持有窗跨界/未成熟Label排除，額外purge/embargo0不是E530sessions。每Origin Validation Fit與Fresh Refit不同run/scaler/state；Validation診斷後refit，全部Forward評分延至三Origin完成，無Forward-feedback tuning。
Ridge12fits＋LGBM12fits已達本輪上限；L0±固定。453192完整Feature-only Predictions先保存後Label Join；7264daily/10896paired，含Validation明確None尾端不當0。Forward352個session×horizon全部可算，Daily N59–64，min40未降。
Forward Ridge5d O1/O2/O3 Mean .028606/.035161/.024427；10d −.012394/.008584/−.002463。LGBM5d .023592/.058135/.065496、10d −.021556/.026531/.000766。全部為EXPOSED DEVELOPMENT，不選winner/不宣稱獨立統計或收益。
10純數值/mock tests PASS；獨立程序Replay453192pred/7264daily/10896paired完全吻合，0追加Fit。72files91000410bytes/份同機Restore，310 prior W29/W30pins未變；CPU87.690630＋60reserve<1800，artifact+restore182002029<512MiB；BACKUP_PENDING。
初始導航authority_class缺漏使healthFAIL；prefit shell未set-e，未攔住O1。已修導航，保留原FAIL與修後WARNING證據，數值code/contract未變；不可宣稱所有preflight從未失敗。獨立Review GO Critical0／Required0，0輪review rework；見window31-review.json。無額外API/GPU/W19或V6操作。
[Run](../../research/v7/window31-run.json)、[Results](../../research/v7/window31-results.json)、[Evidence](../../research/v7/window31-evidence.json)、[Report](../../reports/v7_window31_expanding_cpu.md)。W31當輪main保持b64ac35、成果未合併；W32已獲Human授權ff-only整合至67e2d25，無其他未授權整合。
下一Human：正式Dataset/Universe與UNKNOWN事件政策、Decision-time/Label/maturity/purge、selection/refit/evaluation/statistics、Exposure/prospective評估、運算/獨立備份。無剩餘已授權READY；不得重跑已用完額度的O1/O2/O3。
恢復只讀入口：上述worktree → Current_State → 唯一Backlog → window31-evidence.json；W29/W30 raw/models/restore、W11環境依賴保留。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED。STOP，不開下一窗口。

## Window30 Fixed Real-data LightGBM — COMPLETE / STOP

W29 fixed `e83dfe68e4bc73b59cb6390ec130ba967138e7f1` 已由main1d9a2a7本機ff-only整合；340pins/restore、4tests、ReviewGO核對。
W30 branch `research/v7-window30-real-lightgbm`，base同固定SHA，worktree `.worktrees/v7-window30-real-lightgbm`；LOCAL_INTEGRATED_FIXEDb64ac35 / NOT_DEPLOYED。
任何Fit前60e7ca7固定唯一Recipe/版本/native-lib SHA/feature order/seed/TRAIN view/planned ModelSpec IDs；weights SHA僅Fit後產生。
W11環境LightGBM4.6.0/NumPy2.0.2/SciPy1.16.3/sklearn1.6.1不變；L2/64trees/7leaves/depth3/lr.05/minleaf100/L2=1/maxbin63/seed0/thread1，無eval_set/earlystop/search。
W29 Cohort/Data/兩Features/Labels/Fold/support完全沿用；TRAIN14113/13711，最新exit2021-12-30；identity preprocessing，bin mapper只學TRAIN。
2個fresh LGBM fits各horizon一次，皆64trees，fit約.02777/.02791秒，modeltext52369/54849bytes；Ridge/API/GPU0。
60726完整Feature-only Predictions先immutable保存後Label Join；同W29 support/AVERAGE/min40，970groups940defined N58–64，30tail/maturity保留拒絕。
LGBM Mean/Median：2022/5d .018814/.021627、2022/10d −.012617/−.016793、2023/5d .021182/.026036、2023/10d −.013374/−.013470。
5d正但四組Mean均低於既有Ridge，10d兩年負；沒有改Recipe重Fit、選winner或批准Mamba2。2022/2023 EXPOSED，UNKNOWN actions/asof/vintage與Cohort bias仍保留。
6 mock-only tests PASS；獨立Replay60726pred/970daily/2910paired PASS，0Fit；1主要Review GO Critical0 Required0／0 review rework。
Reviewer不能只由stdout獨立重證shell timeout wrapper；已保存實際tool command/session/start/complete reference，未重跑Fit；OS CPU guard及實測fit保留。
11files34874071bytes/份同機restore、W29原299files hash未變；artifact+restore69749181<256MiB，CPU30.723458＋60reserve<1200；BACKUP_PENDING。
[Recipe](../../research/v7/window30-recipe.json)、[Prepared](../../research/v7/window30-prepared.json)、[Run](../../research/v7/window30-run.json)、
[Results](../../research/v7/window30-results.json)、[Evidence](../../research/v7/window30-evidence.json)、[Report](../../reports/v7_window30_fixed_lightgbm.md)、
[唯一Backlog](../../research/v7/research-program.md#window30-fixed-real-data-lightgbm-baseline)。
下一Human：Expanding origins/model-ready cutoffs/train/validation/forward/maturity/purge/selection/refit/evaluation與data/proxy/universe/events/exposure/CPU/storage上限。
固定CPU E2E基礎已驗，但W29 masks為固定split，各新Origin需重新核對maturity/purge，不能直接宣稱Expanding實驗已批准或認證；無剩餘已授權READY，STOP。
StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED；不mergeW30/push/deploy/nextwindow。

## Window29 Expanded Cross-sectional Development — COMPLETE / STOP

W28 fixed `1d9a2a7a1ee57c455ba58250f63681efbbdd9050` 已由main39e2fed本機ff-only整合；35pins/restore、6tests、ReviewGO核對。
W29 branch `research/v7-window29-expanded-development`，base同固定SHA；worktree `.worktrees/v7-window29-expanded-development`；LOCAL_INTEGRATED_FIXEDe83dfe6 / NOT_DEPLOYED。
GET前746ca04固定68新Cohort（W24重疊0）：60pre2021 current issuers＋4periodlisting＋4delisted start/typeUNKNOWN；64檔進Prediction，仍HINDSIGHT/存活者偏差，非HistoricalUniverseVerified。
68真實FinMind price GET／metadata0／8387840rawbytes，全部HTTP200；單並發、實際起始間隔min5.001297秒，known-local100/hour（外部匿名用量UNKNOWN），unused預算CLOSED。
一次本機mutable counter修復：seq1保留不重送、seq2由Trace證明尚未GET；E1atomic_json只修counter，失敗CPU保守扣600秒未reset；不清除providerSTOP、不擴預算。
最小feature-only介面分離原W24未來Label gate；只讀當前/前一session和當下membership，Predictions先保存後Label Join；未來Label刪除/eligibility變動不能改PredictionUniverse。
48061raw rows／44899feature-only rows；940核心groups N58–64，min40未降；30tail/maturity groups/model N0保留拒絕證據，尾端Predictions仍保存。
原兩Features，L0正／反（反向post-W25 control）、Ridgealpha1兩次freshfit、TRAIN-only scaling，TRAIN N14113/13711。LGBM於GET前略過既有synthetic≤64/toy guard，沒有新Recipe或繞過。
5d Ridge Mean/Median：2022 .037333/.033014、2023 .047138/.050515，正向延續；10d2022 .012073/−.004079、2023 −.008580/−.002417，一致正向未延續。
不能宣稱跨Cohort泛化/優勢/正式統計成立：來源/Universe/support/protocol不同，2022/2023皆EXPOSED；UNKNOWN actions/suspension/asof/vintage保留。
182178predictions，3模型相同可評估Support；18targetedtests，獨立Replay Raw-derived feature/masks＋2910daily/2910paired PASS，0額外Fit。獨立Review GO，Critical0／Required0，1主要輪／0 review rework。
299files／74612661bytes同機restore；總artifact+restore149226490bytes<1GiB，CPU計帳647.740401秒（含failed600）＋60reserve<3600；BACKUP_PENDING。
[Plan](../../research/v7/window29-plan.json)、[Cohort](../../research/v7/window29-cohort.json)、[Coverage](../../research/v7/window29-coverage.json)、
[Run](../../research/v7/window29-run.json)、[Results](../../research/v7/window29-results.json)、[Evidence](../../research/v7/window29-evidence.json)、[Report](../../reports/v7_window29_expanded_development.md)、
[唯一Backlog](../../research/v7/research-program.md#window29-expanded-cross-sectional-ranking-development)。
下一Human：補強ordinary/historical membership與事件未知政策，事先固定新資料/新時間或prospective的5d primary/10dsecondary＋L0±/Ridge研究與精確資料/CPU/storage/backup上限；不因本結果調參或選勝者。
本輪無剩餘已授權READY，STOP；StrictSTOP/FormalAdmissionNOT_APPROVED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED/W19完全隔離。

## Window28 Clean Training Refit & Temporal Causality — COMPLETE / STOP

W27 fixed `39e2fedeb550a08d00f4abddc9a6caa7b1d570f8` 已本機 ff-only 整合（main前367005d）；35pins＋原計算source/restore、14tests、ReviewGO核對。
W28 branch `research/v7-window28-clean-refit`，base同固定SHA，worktree `.worktrees/v7-window28-clean-refit`；LOCAL_INTEGRATED_FIXED1d9a2a7 / NOT_DEPLOYED。
Plan/code/prefit先commit `ec0b072` 再執行；原W24兩Features／Ridgealpha1／W27Policy與Mask不變，2021各horizon只fresh fit一次，總2fits，0resume/search。
5d TRAIN 7338→7164、10d 7188→6841；mean/std僅在clean TRAIN重估並逐值驗證。2022/2023完全不參與Fit/Scaling。
同一W27 Eval support、940日期×horizon、N>=20未降：Ridge舊→新 2022/5d .033873→.034682、2022/10d .045471→.044561、
2023/5d .080008→.080407、2023/10d .053166→.054248；2022/2023皆EXPOSED Development，非模型選擇或獨立效能證明。
18個future有效OHLCV數值擾動案例，Features/固定模型Predictions exact-equal、當日positive control有效，Raw不改。
只驗數值因果性：既有feature wrapper也有未來Label資格gate，不宣稱row eligibility為as-of線上接口，亦非歷史publication/vintage認證。
原July1之2321不在W24資料；任何Fit前依原Mask第一共同合格日固定2022-07-18，不改Mask或樣本池。
6targetedtests；獨立Replay1880daily/940paired/29143newRidge scores/18cases PASS，Replay0Fit。獨立Review GO，Critical0／Required0，1輪主要審查／0 review rework。
227 prior private source identities核對；W28 11files／28170524bytes同機restore，BACKUP_PENDING，原W24/W27 models/predictions/metrics均保留。
[Plan](../../research/v7/window28-plan.json)、[Run](../../research/v7/window28-run.json)、[Results](../../research/v7/window28-results.json)、
[Causality](../../research/v7/window28-causality.json)、[Evidence](../../research/v7/window28-evidence.json)、[Report](../../reports/v7_window28_clean_refit.md)、
[唯一Backlog](../../research/v7/research-program.md#window28-clean-training-refit-and-temporal-causality)。
UNKNOWN actions/suspension/asof/vintage、hindsight/universe/已曝光仍保留；不宣稱完整污染已去除或投資收益。
下一Human：限定新資料/universe/事件未知政策及Label用途，凍結Exposure-aware Study/Fold/Recipe/evaluation與資料/CPU/儲存/備份上限；無剩餘已授權READY，STOP。
0 API/GPU/Colab/Bootstrap/LOSO/V6/W19操作。StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED。

## Window27 Pragmatic Label Safety & Frozen-Prediction Sensitivity — COMPLETE / STOP

W26 fixed `367005dfaba0988fca5e294d9b3e281408bbfce8` 已本機 ff-only 整合（main 前27055f7）；38 pins／3 tests／獨立 Closure GO。
W27 branch `research/v7-window27-label-safety`，base367005d，worktree `.worktrees/v7-window27-label-safety`；LOCAL_INTEGRATED_FIXED39e2fed / NOT_DEPLOYED。
Policy `W27_RAW_PRICE_DIAGNOSTIC_DEVELOPMENT_ONLY_v1`／Predicate／Mask 先在 `e77f32d` commit 固定，再重評既有 W24 預測。
原44266→新43148樣本；排除1118（Training521僅診斷，evaluation597），disjoint primary：confirmed event3／missing33／zero1082。
UNKNOWN event/suspension/asof 對所有樣本保留，不升格 CLEAN；新 Mask 不修改原 W24 mask，沒有 refit 或消除原模型 Training 風險。
四組原940日期×horizon仍全可計算；N>=20未降，Ridge均值：2022/5d .037016→.033873、2022/10d .043930→.045471、
2023/5d .080310→.080008、2023/10d .054056→.053166。這是 post-hoc development sensitivity，非正式模型選擇／獨立驗證。
2023/5d排除68筆持有路徑零價，沒有本輪已確認減資跨界；不得用三筆10d事件解釋5d。無 event-only 追加評分。
14 targeted tests PASS；獨立 Replay1880 daily／940 paired，原44266 Mask 由 raw 重建吻合；W24/W25/W26 218 prior byte pins 未變。
一次 import namespace 修正在任何 metric 前；Replay int-key JSON 比較問題以 canonical hash 修復，原計算source留存、calculate AST相同，metrics不覆寫。
獨立 Review GO，Critical0／Required0，1輪主要審查／0輪 review rework；新成果留候選。0 API／fit/refit／new prediction／Bootstrap／LOSO／GPU／V6或W19修改。
Private `.artifacts/window27-label-safety/` 與同機 `.artifacts/window27-restore/`；BACKUP_PENDING，不宣稱獨立備份。
[Policy](../../research/v7/window27-policy.json)、[Mask](../../research/v7/window27-mask-manifest.json)、[Results](../../research/v7/window27-results.json)、
[Report](../../reports/v7_window27_label_safety.md)、[Evidence](../../research/v7/window27-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window27-pragmatic-label-safety-and-frozen-prediction-sensitivity)。
下一 Human：更大研究 Universe/事件覆蓋/未知政策/Label路線、Fold/maturity/selection/refit/evaluation/exposure及精確資料/CPU/保存上限；無剩餘已授權 READY，STOP。
StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED；W19完全隔離。

## Window26 Label Integrity & Next Study — COMPLETE / STOP

W25 固定候選 `27055f7ec34cd4aa6d47f67bd89e5b6616244bc8` 已由 main `90b1327` 本機 ff-only 整合；
4 focused tests、12 private artifacts／21 deliverable pins 與 Review GO 已核對。W26 base 等於該固定 SHA。
W26 branch `research/v7-window26-label-contract`，worktree `.worktrees/v7-window26-label-contract`；LOCAL_INTEGRATED_FIXED367005d / NOT_DEPLOYED。
2321 已確認 68% 虧損減資；2023-02-17 市場參考價單位變更與 2022-12-08 會計減資基準日分離。
三筆既有 10d Label 算式／市場索引吻合，但跨股數單位變更，不可作未調整持有期報酬。
六個缺行情日期是市場交易日，非市場休市；原始官方停牌公告尚未核實，精確停牌區間保留 UNKNOWN。
其他公司行動覆蓋／historical availability／vintage 未認證；不重寫 W24／W25 歷史結果，不重新評分。
兩路 Label Policy、新 Study、L0 正反向對照與所有日期／閾值／資源額度均為 PROPOSED，待 Human 決定。
曝光帳本核對 E5／E1 machine contracts；2022／2023 已曝光，未列期間為 EXPOSURE_UNKNOWN，不自動成為獨立測試。
FinMind 非價格 1/6、行情 0；官方文件 loads/attempts 4/4；全部預算 CLOSED。0 fit/refit/new prediction/RankIC/GPU。
W26 private root `.artifacts/window26-label-integrity/`；15 files／4,785,564 bytes 獨立同機 Restore PASS；
W24 191＋W25 12 files 原 Hash 保持。BACKUP_PENDING，不宣稱正式備份或完整重現。
[Report](../../reports/v7_window26_label_integrity.md)、[Evidence](../../research/v7/window26-evidence.json)、
[事件矩陣](../../research/v7/window26-event-matrix.json)、[Study Candidate](../../research/v7/window26-next-study-candidate.json)、
[Exposure Ledger](../../research/v7/window26-exposure-ledger.json)、[唯一 Backlog](../../research/v7/research-program.md#window26-label-integrity-and-next-study-contract)。
Review／tests 狀態以 window26-evidence.json／window26-review.json 為準；本窗口無剩餘已授權工程，STOP。
下一 Human：Label 路線及未知／停牌／除息／下市政策、資料與 Universe 資格、Development-only 或 prospective、
Fold／Selection／Refit／評估／資源上限；本次未批准其執行。Strict STOP／E1 BLOCKED／H2 NOT APPROVED／W12 SAFETY_HALTED，W19 不操作。

## Window25 Ranking Robustness & Leakage Audit — complete / STOP

W24fixed `90b1327560f717727ad3fd53dfe86cb421c70f5b` 已LOCAL_INTEGRATED；main由87af76b ff-only，
26tests／191sourcefiles+33deliverablepins／GO與rawrestore驗證。W25 `.worktrees/v7-window25-robustness-audit`，
branch `research/v7-window25-robustness-audit`，base90b1327；LOCAL_INTEGRATED_BY_W26_FIXED27055f7/NOT_DEPLOYED。
單一read-onlyanalysis，0GET/fit/refit/newpredictions/GPU；既有W24policy/predictions/masks/data不改。
44266sourceformulas/timing檢查0確認跨期數值錯誤；TRAINhash/mean/std吻合，historicalavailability/maturity/vintage仍UNKNOWN。
四組RidgeMean/Median正，positive約59–62%；LOSO33×4=132cases/31020paired-days不翻負，非排除群組或TRAIN影響。
固定10/20circularblocks敏感度：2022relativecomparison脆弱/部分bounds跨0；2023direction一致但EXPOSED。
公司行動未知；2321約+224%10dproxy跨6缺行情session，來源/停牌/公司行動原因UNKNOWN，無發明過濾門檻或改寫結果。
分類ENGINEERING_VALID_WITH_EXPLORATORY_SIGNAL僅條件rawproxy；2022comparisonSIGNAL_FRAGILE，actions/asofINSUFFICIENT_EVIDENCE；非Humancorrectness/admissiongate。
Root.artifacts/window25-robustness-audit保存LOSO/casetrace/source；12files獨立同機restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window25_robustness_audit.md)、[matrix](../../research/v7/window25-leakage-matrix.json)、
[evidence](../../research/v7/window25-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window25-ranking-signal-robustness-and-leakage-audit)。
10focused/26mainpost/4independent新testsPASS；ReviewGO0CriticalRequired，0rework；16CBBbounds/264LOSOdailycase獨立核對。
下一Human：公司行動/停牌/adjustment與2321有界補證、universe/PIT或proxy政策、matchedbaseline方向與新未曝光資料/正式統計設計。
本輪無剩餘READY；STOP。StrictSTOP/FormalAdmissionNOT_APPROVED/E1BLOCKED/H2NOT_APPROVED/W12HALTED，W19完全不操作。

## Window24 Expanded Retrospective Pilot — complete / STOP

W23fixed `87af76b038fcbbda9e1e4b3a4349e693a3b1ba48` 已LOCAL_INTEGRATED；main由7027f03 ff-only，
26tests／55priorlocalfiles+25deliverables／GO與rawrestore重驗。W24 `.worktrees/v7-window24-expanded-pilot`，
branch `research/v7-window24-expanded-pilot`，base87af76b；LOCAL_INTEGRATED_BY_W25_FIXED90b1327/NOT_DEPLOYED。
下載前固定36檔HINDSIGHT_SELECTED cohort（currentcapital非historicalmarketcap）；31新FinMindGET+5舊raw，
0metadata/retry/redirect/token/fallback，unused14CLOSED。24511rows，QC24160VALID/346zero/5UNKNOWN；不fill或調整價混用。
TWSE2021–2023既有729calendar；52488三分mask，engineering46291／本輪exploratoryrawproxy44266／strict0。
33current/new股票入比較；3下市startUNKNOWN僅保留診斷，不能宣稱已消除存活者偏誤。
固定L0／Ridge各horizon一次（2realRidgefits），Train2021/Val2022/exposed2023；0search/selection/refit。
940core日期×horizon全部N27–33，預先固定min20未降；1880daily/940paired與scores獨立ReplayPASS。
公司行動/publication/vintage/可交易資格仍UNKNOWN，所有Label為獨立rawdiagnosticproxy，非收益或模型優劣證據。
本輪LIMITED_EXPLORATORY授權已行使；FormalResearchAdmission仍NOT_APPROVED／StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED。
Root.artifacts/window24-expanded-pilot/private原始與完整predictions，193files+5舊raw獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window24_expanded_pilot.md)、[evidence](../../research/v7/window24-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window24-expanded-retrospective-ranking-pilot)。
50focused/26mainpost/44independenttestsPASS；ReviewGO0CriticalRequired，0rework；191pins/restore驗證。
下一Human：正式raw/adjusted/actions政策與historicaluniverse/calendar/Label/eval/compute批准；本輪無剩餘READY，STOP。
W19工作樹/task/artifacts/budget/runtime不操作不干擾，不新增runtime觀測；不mergeW06/W12/W19或W24，不push/deploy。

## Window23 Historical Calendar & CPU Smoke — complete / STOP

W22 fixed `7027f03ebdb7b7b3bd90bc2bf5ce97a4c267c470` 已LOCAL_INTEGRATED，main由428a5fa ff-only；
23tests／78localfiles+23deliverablehashes／既有GO重驗。W23 `.worktrees/v7-window23-calendar-smoke`，
branch `research/v7-window23-calendar-smoke`，base7027f03；LOCAL_INTEGRATED_BY_W24_FIXED87af76b/NOT_DEPLOYED。
官方documentloads10已CLOSED（8directGET+2webopen），FinMind0。TWSE2019/20tableempty，2021/22schedule244/246，
2023schedule加官方0803臨時休市239，與observedreference相同；不提升strictPIT／完整TPEx日曆資格。
十檔日期membership僅listinginterval/knownboundary；8272不套用2024TPEx身份，停牌/可交易資格UNKNOWN保留。
W22原19728strictmask不改，Stricteligible0；A195價格幾何失敗／B7009獨立工程診斷／C12524其他阻塞。
Human本輪LIMITED_ENGINEERING_SMOKE_ONLY：五檔TWSE，Train範圍masked到2021、Validation2022、Forward-style2023；
固定L0／兩horizons各一次Ridge，0search/selection/refit；1880dailyrows，1864defined、16minN不足跳過，非模型性能結論。
公式/feature/fold/policy於fit前封存；rawreturn非可執行收益，actions/publication/vintage仍UNKNOWN，不是研究admission。
唯一rework修復JSONmap排序造成replaylist順序不同；原bundle/models/metrics及run-code留存，無重新fit，獨立ReplayPASS。
Root.artifacts/window23-calendar-smoke及同機window23-restore保存資料，56files獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window23_calendar_smoke.md)、[evidence](../../research/v7/window23-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window23-historical-calendar-and-cpu-smoke)。
正式study仍PROPOSED/DataAdmissionNOT_APPROVED；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED；W19完全不操作。
58focused/23mainpost/26independenttestsPASS；獨立GO0CriticalRequired，唯一replay修復無refit。
下一Human集中決定正式historicalcalendar/universe/action/label/availability/evaluation/compute；無本輪剩餘工程READY，STOP。

## Window22 Continuous Historical Panel — complete / STOP

W21兩Required獨立複核CLOSED，15testsPASS/Critical0Required0；133localfiles驗Hash。
main由6a34520 ff-only至 `428a5fa5cb7b771db39a053f88c7046501399f7e`；沒有提前main文件commit。
W22 `.worktrees/v7-window22-historical-panel` / `research/v7-window22-historical-panel`，base428a5fa；LOCAL_INTEGRATED_BY_W23_FIXED7027f03/NOT_DEPLOYED。
10FinMindpriceGET、0metadata/diagnostic，均HTTP200，unused20CLOSED；10檔9864rows/1762931rawbytes，2019–2023。
6檔同1216reference日期（不是合格officialcalendar），9725structuralVALID/131UNVERIFIED/8zero，不fill或算return/RankIC。
官方18419/25036/11543CSV各1GET；TPExdelisted403立即停，共4resourceattempts；cataloguedocs另3GET。
6515TWSE上市20210120、8272TPEx上櫃20241023、2499終止上市20201110有officialCSV證據；逐日suspension/membership未知。
實際5d/10d eligible0：calendar/actions/availability/labelmaturity仍UNKNOWN，historicalstudyPROPOSED，不作admission。
Root.artifacts/window22-historical-panel與同機restore保存raw/panel/mask；15產物獨立程序restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window22_historical_panel.md)、[coverage](../../research/v7/window22-panel-summary.json)、
[唯一Backlog](../../research/v7/research-program.md#window22-continuous-historical-panel)。
23focusedtestsPASS；一次主要獨立ReviewGO/Critical0Required0，0rework；15份restore皆PASS。
下一步只建議精準calendar/lifecycle補證與Human統一data/label/eval/CPU授權；本輪完成後STOP。
W19工作樹/排程/Artifacts/budget不讀寫不觸發不整合；W12SAFETY_HALTED/StrictSTOP/E1BLOCKED/H2NOT_APPROVED不變。

## Window21 FinMind Historical Proof — complete / STOP

W20fixed `6a34520d8f9c1166172af3503cad531e23f5ecec` 已LOCAL_INTEGRATED，main由87695e4 ff-only；
11deliverable／8HTML+manifest／既有ReviewGO驗證。W21 research/v7-window21-finmind-proof，base同上，
`.worktrees/v7-window21-finmind-proof`，COMPLETE/STOP/LOCAL_INTEGRATED_BY_W22/NOT_DEPLOYED；固定428a5fa。
18anonymousHTTP200＝14price+3metadata+1repeat；4名額未用且CLOSED，無Token/retry/redirect/fallback。
6檔116price／48dates／2010-01-04..2025-08-01；100structuralVALID、3NO_PUBLISHED_PRICE（含有量零價）、
2documented-date市場例外、11UNVERIFIED；零ffill／PriceAdj混用／研究准入。2499有2010price及provider下市身份，近/後窗口空不推論漏資料。
Snapshot收到838854bytes，去重後836995rawbytes。root.artifacts/window21-finmind-proof獨立於worktree，
18份初次／修後獨立程序Restore與reviewerRestorePASS；原records/code保留。Backup仍同機BACKUP_PENDING。
VendorCalendar含future及已知20261023休市日期，禁止當actualcalendar；StockInfoupdated不等於active/listing，Universe/lifecycle仍BLOCKED。
來源僅CONDITIONAL_RETROSPECTIVE_CANDIDATE，多年份小窗口≠全市場連續性／StrictPIT／Executable證據。
W21原一次Review 2Required及單輪修復證據保留；W22已針對性獨立複核CLOSED/GO，見Window22integration記錄。
[Proof](../../reports/v7_window21_finmind_proof.md)、[evidence](../../research/v7/window21-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window21-finmind-historical-data-proof)。
下一Human：獨立少特徵retrospective dataset身份、合格calendar/lifecycle、missing/actions/Label/maturity與eval/CPUcompute批准；
本輪無剩餘READY。W19工作樹/排程/artifact/budget完全不操作、不整合；無新runtime觀測。
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/W12SAFETY_HALTED。

## Window20 Historical Source Decision Sprint — complete / STOP

Branch `research/v7-window20-historical-sources`，base/main `87695e4d5dc6e047f5ab39ca3cd8294b971bbc3a`；
`.worktrees/v7-window20-historical-sources`，UNMERGED/NOT_DEPLOYED。僅最多三類來源的兩單元文件調查，沒有行情API或工程實作。
FinMind TaiwanStockPrice為有條件主要候選；富邦HistoricalCandles為帳號/使用權BLOCKED的備用；
W14官方歷史route使用資格未釐清，未套用W15CSV授權。多年份日期皆DOCUMENTED宣稱，不是假定Coverage。
非商業/顯名/禁止raw再散布與匿名300/hr文件已核對；永久保存條件UNKNOWN，0價/興櫃open/調整價/下市生命週期須驗證。
建議下一次無Token資格Pilot至多22requests（18price+3metadata+1重取）、6檔角色×3小窗口，PROPOSED未批准。
不是5d/10d Label／Dataset／研究admission，沒有L0/Ridge/LightGBM訓練或RankIC。
W19固定cf1a84f的限期Task授權獨立保留；本輪不整合/改動/停用/觸發，無新增runtime觀測；以其固定來源及timestamped記錄為準。
[Decision brief](../../reports/v7_window20_source_decision.md)、[matrix](../../research/v7/window20-source-matrix.json)、
[evidence](../../research/v7/window20-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window20-historical-data-source-decision-sprint)。
一次獨立審查GO／Critical0／Required0，8官方HTML924253bytes／manifest及分類預算驗證PASS；無實作或套件重跑。
下一Human只需決定是否批准有限無帳號非商業來源資格驗證及保存範圍；富邦另需帳號/行情權限，付費並非首輪必要。
完成後STOP，沒有本輪剩餘READY；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/GPU_SAFETY_HALTED。

## Window18 手動 Forward Collector Pilot — complete / STOP

W17 fixed `96e8c0c4f2f7f78c508048e31f0f163d6c707128` 已 LOCAL_INTEGRATED/NOT_DEPLOYED，
main 由000b0b9 ff-only至固定候選；38treesclean／19deliverablehashes／9CPUtests確認；W06/W12未整合。
W18 `development/v7-window18-manual-collector`，base同上，`.worktrees/v7-window18-manual-collector`，
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED；finalHEAD為含本收尾commit。僅本輪manualpilot批准，daily仍NOT_APPROVED。
TWSE/TPEx exactlicensedCSV各1GET HTTP200，0retry/redirect/fallback；均2026-10-22：1383/1017rows，
137956/141480bytes，raw合計279436。原record/QC/code保留；TPEx初判27INVALID，修後離線27MISSING，TWSE15MISSING；不補零。
本機根目錄`.artifacts/window18-manual-pilot`保存raw+manifest+provenance，獨立程序初次／修後restore均Hash+rowsPASS。
同機restore不是獨立backup：BACKUP_PENDING。無Content-Length證據明確UNKNOWN，不宣稱來源宣告長度驗證。
Capture MANUAL_RESOURCE_CAPTURE_TIME_UNQUALIFIED，publicationUNKNOWN，非VERIFIED_FORWARD_VINTAGE／ResearchDataAdmission。
W17calendar僅PROVISIONAL TWSEboundedmatch；TPEx日曆未知，不以牆上日期判stale，不將商品全列入股票Universe。
新增manualCLI/CSVQC，共重用v1snapshot/W09safe+lock/W07clock/W17calendar；W15adapter仍Synthetic-only，沒有第二套平台。
25targetedtestsPASS（含原W17 9）、integration9PASS；一次review Critical0/Required3→唯一修復輪全部以RED/GREEN＋
獨立程序實際restore自行驗收。沒有第二次獨立GO，不把首次NO-GO改寫。W14/16/17原bytes與W12HALTED不變。
[Brief](../../reports/v7_window18_manual_pilot.md)、[evidence](../../research/v7/window18-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window18-limited-real-forward-collector-pilot)。
下次Human：daily runtime/scheduler、19:00候選與更新/日曆/QC、有限retry、獨立備份、監測通知與backfill/停止撤銷。
本輪GET額度已用完；無剩餘授權執行。StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/GPU_SAFETY_HALTED，
FairEval/PreNextOpen PROPOSED／InferenceRecovery DEFERRED。不得自動重跑/排程/訓練或開下一窗口。

## Window17 Forward Vintage 操作準備 — complete / STOP

W16 固定 `000b0b9c40448b11a41d05396dffc6e0f3ea1235` 已 LOCAL_INTEGRATED／NOT_DEPLOYED；
main 由 `27dcb36` ff-only 至固定候選，原有 W16 UNMERGED 記錄僅保留歷史意義。W06／W12 未整合。
W17 `research/v7-window17-forward-readiness`，base 同上，`.worktrees/v7-window17-forward-readiness`；
WINDOW_COMPLETE／STOP／UNMERGED／NOT_DEPLOYED，最終 HEAD 為含本收尾的 local commit。
W16 CSV 279436 bytes＋2 manifests＋request ledger 已獨立程序驗證並保留至根目錄
`.artifacts/window16-durable-evidence/`，原工作副本及 Restore 不動；同機同檔案系統，BACKUP_PENDING。
獨立Review一次GO／Critical0／Required0／0rework。32 次必要定點驗收（整合3＋最終29）、6日離線面板；新9測試僅工程／Synthetic，不是歷史 PIT 准入。
TWSE 官方10/23休市標題／ID／發布時刻及10/26年度補假確認；PDF正文未取得。
TPEx直接公告仍缺，不能以TWSE推定；10/27預定交易、非已實際交易。日曆修訂須有鏈與asof身份。
兩資源每1日更新不等於自動存取配額／SLA；定期服務條件與正式啟用仍 BLOCKED。
新元件限離線calendar／storage狀態guard，重用W07/14/15及Snapshot；沒有網路Collector或Calendar Platform。
[Activation brief](../../reports/v7_window17_activation_brief.md)、[evidence](../../research/v7/window17-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window17-forward-vintage-operational-readiness)。
下一Human：先確認兩Resource定期服務條件及各市場有效日曆，再批准runtime／時間頻率／有限重試／
QC缺漏／儲存與獨立備份／監測通知／backfill及停止恢復撤銷。沒有剩餘本輪READY或啟用權限。
StrictSTOP／E1BLOCKED／H2NOT_APPROVED／HistoricalPROPOSED／CollectorNOT_APPROVED／GPU_SAFETY_HALTED；
FairEval／PreNextOpen PROPOSED、InferenceRecovery DEFERRED。不啟動下一窗口。

## Window16 Real Resource Proof — complete / STOP

W15fixed27dcb36b5431b70e8aa831edb6cb9234cb3a40b3 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from4da1bcf;25targetedCPUtestsPASS,source/review/hash/36cleantrees verified.
W14 raw/W15 evidencebytes remain originaltrees;not moved to main. Mainheld27dcb36;
W06/W12/othercandidates notmerged. OlderW15UNMERGED text historical for integration only.
W16 research/v7-window16-resource-proof,base27dcb36,.worktrees/v7-window16-resource-proof,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2units only,no newvalidator/platform. Two exactlicensedCSV GETs,
1each/0retry/redirect/fallback. BothHTTP200 UTF8,only2026-10-22/1session:
TWSE1383rows/1383securities/137956bytes,1368basicQCpass+15missing;
TPEx1017/1017/141480bytes,990basicQCpass+27missing. Nozero-fill orcalendar/universe/
pricecurrency/adjustment/tradabilityadmission. HistoricalTrainingCoverageBLOCKED;
no hiddenhistorysearch. Forwardrawresource CANDIDATE_ONLY, activationNOT_APPROVED.
IndependentPythonrestore2records/279436bytes/hash+rowsPASS;3focusedSnapshot testsPASS.
Raw+restore both localuntrackedcopies in W16tree,not certifiedbackup;tree removalrisk.
[Brief](../../reports/v7_window16_resource_proof.md),[evidence](../../research/v7/window16-evidence.json),
[sole queue](../../research/v7/research-program.md#window16-real-resource-coverage-and-capture-proof).
SingleindependentreviewGO0Critical/Required,0repair;28targetedtests, no fullsuite repeat.
Fallbackpolicy PROPOSED,whitelistempty,no provider switch/collector/fit/RankIC/GPU.
Old120files+4Frozenhashes unchanged. StrictSTOP/E1BLOCKED/H2NOT_APPROVED/
ExploratoryPROPOSED/CollectorNOT_APPROVED/GPU_SAFETY_HALTED. No nextwindow.

## Window15 Licensed Foundations / Dual-track — complete / STOP

W14fixed4da1bcfa81c8ff786de82d29ec461c0851d411d5 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only fromc78098f;pre/post150CPUtestsPASS,independent0Critical/Required.
W14original337221rawbytes/4snapshotfiles remain in W14tree, NOT moved to main.
Mainheld4da1bcf; W06/W12/othercandidates notmerged. OlderW14UNMERGED prose historical.

W15 research/v7-window15-dual-track,base4da1bcf,.worktrees/v7-window15-dual-track,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED. Humanconfirmed dualtrackdirection only.
U1 COMPLETE_REVIEWED:govdatasets11549/11371 exactCSV scopedOGDL1.0+attribution confirmed;
periodicservice/rate/activation conditional; oldhistoricalroutes permissionUNCLEAR.
3officialmetadata/licenseHTML1547650bytes locallyretained/independentverified;0market
payloadrequests. Licensing never implieshistoricalPIT/dataset/researchadmission.
U2 COMPLETE_REVIEWED:independentexploratoryStudy/Dataset/Feature drafts PROPOSED;
7synthetictests. Lag/dates/parameters/universe/calendar/labels/bytes/compute unapproved,
DATA_READY=false; oldsyntheticadapters require separatelyreviewed realdata scope.
U3 COMPLETE_REVIEWED:SYNTHETIC-onlythinarchive adapter reusesSnapshot/W14/W09;
18tests,2boundedrepairs,0remainingCritical/Required. Idempotentraw/newrevision,
currentQC/late vsfirstcapture separated; missingintent/binding/snapshotNOT_RESTORABLE,
no hiddenretryrepair. Initialdryrun+oldsource retained, finalofflinevalidation separate.
175jointCPUtestsPASS; allthree reviews0Critical/Required, U1/U2rework0. No livecollector/
fits/realRankIC/GPU. W11/W12/W14old92files+4Frozenhashes unchanged; no cleanup.
[Brief](../../reports/v7_window15_dual_track_brief.md),
[evidence](../../research/v7/window15-evidence.json),
[sole queue](../../research/v7/research-program.md#window15-licensed-dual-track-foundations).
NextHuman:boundedlicenseddata/coverage/restore qualification, newstudy/eval/admission
policy, separateTrackBoperation/storage/frequency/QC/notice authorization. No READY
execution remains in this completedgrant. FinalHEAD containingclosurecommit.
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/InferenceRecoveryDEFERRED remain. No nextwindow activation.

## Window14 Evidence-first Data Pilot — complete / STOP

W13fixedc78098f3fdddb74e461bc54d700bde5baac0fcc1 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from8db5b911;pre/post140CPUtestsPASS,independentGO0Critical/Required;
12artifact/24source/88protectedhashes verified,34trees preflightclean. Mainheldc78098f;
W06/W12/othercandidates notmerged. OlderW13 UNMERGED text historical for integration only.
W14 research/v7-window14-data-admission,basec78098f,.worktrees/v7-window14-data-admission,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2boundedunits,each1reviewrepair.
U1COMPLETE_REVIEWED:2dailyofficialrequests TWSE20220901/TPEx20220902,337221bytes,
4selectedstocks,currentstructuralQC. Both proposedRETROSPECTIVE_ONLY;0asof/0proxy
qualified. Historicalpublication/revision/calendar/universe/labels and rights remain
unresolved. Contentclass != usagepermission != researchadmission;no moredatarequests.
U2COMPLETE_REVIEWED:PROPOSED forwardcontract+readonlyprototype reusesSnapshot/W07/W09;
10synthetictests,150jointCPUtestsPASS;both0Critical/Required. No collector/scheduler/
newplatform. PROTOTYPE_TIMELY neverauthenticates callerclock/pins oradmitshistoricaldata.
[Brief](../../reports/v7_window14_decision_brief.md),
[evidence](../../research/v7/window14-evidence.json),
[sole queue](../../research/v7/research-program.md#window14-evidence-first-admission-pilot).
NextHuman:permitdata route/rights, qualify asof/vintage or distinctlag-proxy/forward
proposal, then calendar/universe/labels/restore+research/eval/CPUcompute. No trueCPUstudy
ready or remainingauthorizedexecution. Raw4snapshotfiles retained locally;hash-only
restore impossible. ProtectedW11 83bytesfiles/W12 5diagnostics and4Frozenhashes unchanged.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED. FinalHEAD containingclosurecommit;no newwindow.

## Window13 Thermal / Admission Preparation — complete / STOP

Branch research/v7-window13-admission-preparation,base/main8db5b9111b98d4965a9f56f10a51c5157284b900,
.worktrees/v7-window13-admission-preparation,WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
Main unchanged; W11 localintegration verified (olderUNMERGED text historical).
W06 fixed96c60564/ProtocolAudit8601c737/Recoverye181f366/W12 fixed16e444f9 only
UNMERGED_REFERENCE_ONLY; no merge or authority promotion.

U1 COMPLETE_REVIEWED,sourcef26dbf5: HUMAN_CONFIRMED_DESIGN_DIRECTION/GPU_RETEST_DEFERRED;
RTX3060 local-first candidate,WindowsHost target87C reference/futurestop87C;
83C warningPROPOSED, not NVIDIA longtraining certification. OldW12 80C/HALTED/
32.106942s unchanged. GPUkernel/F-B/resume/AMP/fullmodelshape NOT_VERIFIED.
U2 COMPLETE_REVIEWED: [Human Decision Brief](../../reports/v7_window13_decision_brief.md),
DT1–4/EV1–6 options PROPOSED only. W05 audit reused0VERIFIED/21CONDITIONAL/27UNKNOWN;
2TPEx oddlot archives176096bytes currentlyverified, no historicalavailability/QC/
calendar/universe/labels/full48panel proven. No realRankIC or dataadmission.
ENGINEERING_READY scopedW07–11 tools; POLICY_REQUIRED/DATA_BLOCKED/RUNTIME_BLOCKED
separate. 140mainCPUtests+13fixedW06prototypePASS; bothreviewCritical0/Required0,
0rework. ProtectedW11 83files+W12 5diagnostics unchanged; no newsource/framework.
[Decision](../02_History/Decisions/20261024-window13-thermal-admission-preparation.json),
[evidence](../../research/v7/window13-evidence.json),
[sole queue](../../research/v7/research-program.md#window13-thermal-and-admission-preparation).
NextHuman: freeze timing/missing/source/evidenceclass, identify qualified bytes or
approve futurevintage collection, then evaluation/compute contract. No READY execution
in this completedgrant; new boundedauthorization required. Do notclearHALTED/retestGPU.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED unchanged. FinalHEAD containingU2closurecommit.

## Window11 L2 & CPU Resume — complete / STOP

Window10 fixed `f25181637e40138c84dc3722292647d4325882fc` LOCAL_INTEGRATED/NOT_DEPLOYED
ff-only from `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9`;pre/post167testsPASS,
40hashes/independent26 GO,0Critical/Required. Main heldf251816; no main closure commit.
Older Window10 UNMERGED/STOP entries historical, superseded for integration only.

Window11 `development/v7-window11-l2-cpu-resume`,basef251816,
`.worktrees/v7-window11-l2-cpu-resume`, WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
U1 actualLightGBM4.6.0 fit/refit/forward/common-support/independentReplay PASS;
5officialwheels pinned in dedicated overlay, baseCPU Torch/NumPy/SciPy unchanged.
U2 reuses E1 actual train loop/store/adapter, safe weights_only=True probe codec.
Three new-process exits0/75/0,step4(epoch1batch1) afterwarmup ->12steps/4epochs;
23nontelemetryfields exactlyequal forLATEST/PREVIOUS/BEST. CPU_STEP_BOUNDARY_RESUMABLE
and CPU_MID_EPOCH_RESUMABLE only deterministic daily cursor/0workers/FP32/1CPUthread.
No separateepochinterrupt/GPU/AMP/shuffleworker/generalExactResume/full reproduction.
211jointtestsPASS. U1 environment setupcorrection1,reviewrework0;U2rework2,0remaining
Critical/Required. Source U1 f00bebab33cdeaa75b8401c6a6988dfaa3d3e0e7;U2 e28ac62ed95007e246185c5c2a134d17a5d0e08b.
Final HEAD is containing artifact/closure commit; no further engineering/Window12.

CPUbytes83files/~1.79MB retained `.artifacts/window11-cpu-resume-proof` under this
worktree; exactinventory in CPUevidence. Hash-only Git metadata cannot restore lost
checkpoints. Preserve this tree/basevenv/overlay/cache until separately reviewed.
Current runtime-bound L2 source snapshot uses U2commit and fresh run identity;
initialsourcebundle/report retained unchanged, requires original source to replay.
[CPU proof](../../research/v7/window11-cpu-resume-evidence.json),
[current L2](../../research/v7/window11-l2-runtime-evidence.json),
[evidence](../../research/v7/window11-evidence.json),
[sole Backlog](../../research/v7/research-program.md#window11-l2-and-cpu-resume).
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. No realresearch/GPU/production/admission/othermerge.
Resume namedtree HEAD/clean/maps/Charter/evidence; new bounded authorization required.

## Window10 Classical Synthetic Foundation — complete / STOP

Window09 fixed `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `bdea30d877fa2c5d02308f343fa5591a35616aec`.
Pre/post141CPUtests PASS;39hashes/independent40tests GO,0Critical/Required;30trees
clean. Main held91fa82c, all new closure/development documents stay isolated.
Older Window09 UNMERGED/STOP entries superseded for integration only.

Window10 branch `development/v7-window10-classical-e2e`, base91fa82c, worktree
`.worktrees/v7-window10-classical-e2e`. UNMERGED/NOT_DEPLOYED; two units only,
<=2rework/unit. Human now permits tiny CPU synthetic fits, not real-data research.
U1 COMPLETE_REVIEWED:13tests/independent13PASS,1rework (full selection run hash and
null containers). L0 explicit toy factor; L1 NumPy train-only scaled Ridge; L2
adapter code present, dependency BLOCKED (lightgbm absent; no install performed).
U2 COMPLETE_REVIEWED:13tests/independent53 including40oldW09testsPASS,1rework
for fit/prediction/config lineage; validation+toyselection and forward pair replay
verified in independent CPU process. Final167joint testsPASS. U1 source 4ec22a4330ed6db7df0b3dafb4106cea820516e2;
U2+closure final HEAD is this containing commit. WINDOW_COMPLETE/STOP.
W09 v1 no-fit profile unchanged; v2 honest tiny-classical-fit profile added. No new
split/evaluator/platform. METADATA/EVALUATION replay proven; TRAINING_RESUMABLE/
FULL_REPRODUCTION_READY unproven. L2 actual fit remains dependency BLOCKED.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No Frozen E5/E1 or production mutation; no other branch
integration, GPU/Colab/search/real research. [Evidence](../../research/v7/window10-evidence.json),
[sole backlog](../../research/v7/research-program.md#window10-classical-synthetic-foundation).
Resume named branch HEAD/clean/maps/Charter/evidence/queue; final candidate is this
containing U2/closure commit. No READY unit remains in this grant; no Window11.

## Window09 Durable Artifacts & Offline Replay — complete / STOP

Window08 fixed `bdea30d877fa2c5d02308f343fa5591a35616aec` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `1631020bce8223ac4337c7e97473a2a3d9a555f0`.
Pre/post101CPUtests PASS;19hashes/24cases reverified; independent34tests GO,
Critical0/Required0. Main remains bdea30d; no extra main closure commit.
Older Window08 UNMERGED/STOP entries are historical, superseded for this integration only.

Window09 isolated `development/v7-window09-artifact-replay`, base bdea30d, path
`.worktrees/v7-window09-artifact-replay`. UNMERGED / NOT_DEPLOYED. Human authorizes
only two CPU/SYNTHETIC units, <=2evidence-based rework/unit, then STOP.
U1 COMPLETE_REVIEWED:20focusedPASS;2bounded rework rounds corrected traversal,
metadata symlinks/JSON overflow, publication-null/sources containers. Immutable bundle profiles, required
artifacts, staging+diagnostics, fsync/flock/rename, idempotence and verified external
bytes references implemented. U2 COMPLETE_REVIEWED:20testsPASS,1rework for nested
containers and actual maturity recheck; independent final40testsPASS,0Critical/Required.
Final141joint tests;2schemas/3outputs validated. METADATA_VERIFIED and
EVALUATION_REPLAYABLE proven; TRAINING_RESUMABLE/FULL_REPRODUCTION_READY not proven.
WINDOW_COMPLETE/STOP. U1 source 79d6c556d1bb5648a40a02c3b6f0e628c6512b4a; final U1 container repair
and U2 source/evidence are in this containing closure commit.
No tracking platform, retention/GC policy or training/research admission added.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No W06/recovery/audit integration or original matrix search.
[Evidence](../../research/v7/window09-evidence.json),
[backlog](../../research/v7/research-program.md#window09-durable-artifacts-and-offline-replay).
Resume named worktree cleanHEAD/maps/Charter/evidence/sole queue; final HEAD is
containing U2/closure commit. No remaining work in this grant; no Window10 grant.

## Window08 Common Ranking Evaluation — complete / STOP

Window07 fixed `1631020bce8223ac4337c7e97473a2a3d9a555f0` LOCAL_INTEGRATED /
NOT_DEPLOYED by ff-only from `465dca229fc1452aaf1b46572b06886a7d8a2c8f`.
Pre/post67tests PASS;10artifact/frozen hashes and16cases verified; independent24tests,
Critical0/Required0 GO. Main remains exactly1631020; no main closure-doc commit.
Window07 STOP/UNMERGED prose below is historical, superseded only for this integration.

Window08 branch `development/v7-window08-ranking-evaluation`, base1631020, worktree
`.worktrees/v7-window08-ranking-evaluation`. UNMERGED / NOT_DEPLOYED. Human grant
permits only common evaluator and paired comparison, synthetic CPU fixtures, max2
rework/unit. U1 COMPLETE_REVIEWED:826b8ddf798e752584d180181d24cc1db967e019;19tests/independent19PASS,1rework.
U2 COMPLETE_REVIEWED in containing closure commit;15tests,0rework. Both Critical0/
Required0;101joint CPUtests,24explicitcases,2schemas/3outputsPASS. WINDOW_COMPLETE/STOP.
Existing E5 average-tie RankIC, canonical_hash and W07 calendar/fold/role guards reused.
Policy remains named SYNTHETIC_NOT_ADMITTED; minN/weighting explicit, no statistics
or acceptance policy frozen. Metadata-first label gating, raw references and daily
rejection ledger are engineering evidence, not real historical PIT certification.

No W06/recovery/audit branch integrated. Strict Phase0 STOP; E1 Frozen Matrix BLOCKED;
H2 NOT_APPROVED; Fair Evaluation and Pre-next-open PROPOSED; Inference Recovery DEFERRED.
[Evidence](../../research/v7/window08-evidence.json),
[unique backlog](../../research/v7/research-program.md#window08-common-ranking-evaluation).
Final HEAD is this containing U2/closure commit. Resume by HEAD/clean, this entry,
Authority Map, Charter, evidence and backlog; a new bounded grant is required.
No automatic Window09, real-data evaluation, training or main merge of Window08.

## Window07 temporal/isolation foundation — completed / STOP

Human approved two CPU-only synthetic engineering units, no formal study config.
Branch `development/v7-window07-temporal-isolation`, worktree `.worktrees/v7-window07-temporal-isolation`,
base/main `465dca229fc1452aaf1b46572b06886a7d8a2c8f`. UNMERGED / NOT_DEPLOYED.
U1 commit5d736e9:13focused/independent13PASS,1owner-binding repair.
U2 complete:11focused/independent11PASS,
1source-hash repair. Final67related tests/16expectedcasesPASS;0remainingCritical/Required.
WINDOW_COMPLETE / STOP. No further work under this grant. Containing Unit2/closure
commit is final checkpoint. Resume via cleanHEAD here/maps/Charter/uniquequeue;
new bounded authorization required. CommonEvaluator synthetic foundation can be
next scope, not approved policy/data/Window08. [Evidence](../../research/v7/window07-evidence.json),
[decision](../02_History/Decisions/20261023-window07-temporal-isolation.json),
[unique backlog](../../research/v7/research-program.md#window07-temporal-fold--research-isolation).
No fixed validation/forward months/origins/30purge or metrics adopted; unknown policy/
data failclosed. No realPIT certification, no formalFeatureLabel/E5E1/admission change.
Window06/audit/recovery branches only pinned references, not merged or promoted.
StrictSTOP/E1BLOCKED/H2unapproved/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. This grant does not authorize a Window08.

## Window06 — Window05 integrated; decision-time preparation authorized

Fixed `65462ec185783eedb2ba81b66cfa04783f86c6e2` integrated ff-only from main
`2ee4ca83e709af64be4dd63032ba4fb47903d769`; pre/post87 tests PASS,13source/output hashes
verified, independent13tests/0Critical/Required. LOCAL_INTEGRATED / NOT_DEPLOYED.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window06-integration.json).

Same Human grant authorizes isolated `research/v7-window06-decision-time` at
`.worktrees/v7-window06-decision-time` from this closure: VERSIONED PROPOSAL,
<=3source families/~10sessions diagnostic panel and synthetic eligibility tests.
New pre-next-open proposal does not replace frozen E5/E1. No formal cutoff/time,
feature/label/admission or execution policy selected. StrictSTOP/E1BLOCKED/H2unapproved/
FairEvaluationPROPOSED remain. No F2 repeat search/training/production changes.
Read local Current State, Authority Map, Charter and unique Backlog on that branch
for subsequent progress; all pending documents stay isolated. Older STOP entries
are historical for their windows; no indefinite authorization.

## Window05 Feature Integrity and Data Research — complete / STOP

Human authorized F0/F1/F2 only: read-only source audit, synthetic candidate formulas,
and bounded monthly-revenue PIT feasibility. Branch `research/v7-window05-feature-integrity`,
worktree `.worktrees/v7-window05-feature-integrity`, base/main `2ee4ca83e709af64be4dd63032ba4fb47903d769`.
UNMERGED / NOT_DEPLOYED. Source-derived macro Date joins do not prove t-close
availability; missing source values may become defaults. F0 identified material
unquantified risks; no real factor/performance analysis may depend on these
unverified payloads. Historical conclusions are preserved; no result reclassification.

F0 source audit complete:48ordered, VERIFIED0/CONDITIONAL21/UNKNOWN27/INVALID0,
all empirical quality UNKNOWN; F0 commit9fbf1bd,4tests/independent4 PASS.
F1 completed PROPOSED shortlist/formulas,9tests/independent9 PASS,1timezone rework;
source commit1a6a452, real diagnostics BLOCKED. F2 complete NOT_CURRENTLY_FEASIBLE
for broad historical PIT recovery;4historical metadata blobs independently verified,
0newqualified rows,0rework. All independent reviews0Critical/Required;87related tests
PASS. Window complete: STOP / UNMERGED / NOT_DEPLOYED, no automatic continuation.

The containing F2/closure commit on this branch is the final resume checkpoint.
Verify HEAD/clean in the named worktree, then read this section, Authority Map,
`knowledge/GOVERNANCE.md` and the local unique Backlog's Window05 section.
[Backlog](../../research/v7/research-program.md#window05-feature-integrity-and-cost-effective-data-research),
[evidence](../../research/v7/window05-evidence.json). Main stays2ee4ca8; pending
integration documents stay here. No formal manifest/cutoff/label/admission changed.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; Fair Evaluation and all
new feature/timing/forward-vintage candidates PROPOSED. Human timing/data identity/
policy choices precede real-data diagnostics; a new bounded grant is required.
Earlier window execution pointers below are historical; this entry owns latest state.

## Window04 W04B-LOG integration closure — STOP

Human-approved fixed `97fb4c9d669f3aa237ef5545a0ef322a5cc2d6a9` integrated by local ff-only
from `31f7accc475b5b735684d5a06725a237d85d5693`. Engineering source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`;
no new merge commit. This containing main commit adds closure documentation only.
LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED / WINDOW_COMPLETE / STOP.
Pre/post274 tests each PASS; independent runtime6 PASS,0 Critical/Required.
Both source hashes match;23 existing worktrees clean before integration and preserved.
[Closure evidence](../../research/v7/window04-log-integration-closure.json).

Current continuation entry is WSL main: verify HEAD/clean, read this section,
Authority Map, `knowledge/GOVERNANCE.md` and the local unique
`research/v7/research-program.md`. Older branch-local/unmerged pointers below are
historical, superseded only for W04B-LOG local integration. No new work authorized.
No additional high-value READY item was established by the prior bounded discovery;
future concrete CPU defects can be considered under a new bounded grant.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; research options PROPOSED.

## Window04 Stage B — completed / STOP

Branch `development/v7-window04-cpu-continuation`, base/main `31f7accc475b5b735684d5a06725a237d85d5693`,
worktree `.worktrees/v7-window04-cpu-continuation`. UNMERGED / NOT_DEPLOYED.
WINDOW_COMPLETE / STOP: W04B-LOG source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`; final274 tests PASS.
No additional high-value independent READY unit established. A new bounded grant
is required for further development; this grant is exhausted. Resume by verifying
HEAD/clean here, reading this section, Charter and the unique backlog; do not
continue automatically. The containing docs-only closure commit is final checkpoint.
W04B-LOG fixes admitted E1 runtime reader-error propagation: log write/decode
failure reaches caller with original cause/path and stops a still-running child;
a failure observed after child exit also cannot return success. CPU fixtures only,
no Colab/Drive certification or training readiness claim. Focused6 tests PASS,
independent6 PASS,0 Critical/Required,0 review rework rounds (one initial fix).
[Unique backlog](../../research/v7/research-program.md#window04-stage-b--cpu-continuation)
records evaluated alternatives; [evidence](../../research/v7/window04-cpu-continuation-evidence.json).
Strict Phase0 STOP, E1 matrix BLOCKED, H2 NOT_APPROVED and proposals PROPOSED.
This section owns branch-local progress; earlier integration/continuation pointers
below are historical. Stage B source/closure stays here, main remains unchanged.

## Window04 integrated; bounded CPU continuation authorized

Human-approved fixed `98f270dd63f05831b2a42695d8005638fda2bbf5` was ff-only integrated
from main `c733a53f81ff7bd572739b5d75f8baabf1189976`. LOCAL_DEVELOPMENT_INTEGRATED /
NOT_DEPLOYED. Pre/post272 tests each PASS; independent metadata5 PASS,0 Critical/Required.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window04-main-integration.json).

The same Human grant authorizes Stage B, at most four valuable CPU engineering
units with at most two evidenced rework rounds each. Use isolated
`.worktrees/v7-window04-cpu-continuation`, branch `development/v7-window04-cpu-continuation`,
created from this closure; its `research/v7/research-program.md` owns subsequent
selection and completion state. All pending development/closure stays on that branch.
No Stage B merge/push/deploy/training. Strict Phase0 STOP, E1 matrix BLOCKED,
H2 NOT_APPROVED; comparison proposals remain PROPOSED. Earlier awaiting-approval /
no-StageB text below is historical and superseded solely by this explicit grant.

## Window04 Git reconciliation — current candidate

Human authorized isolated version reconciliation only. Branch
`integration/v7-window04-reconciliation`, worktree `.worktrees/v7-window04-reconciliation`,
base/main `c733a53f81ff7bd572739b5d75f8baabf1189976`; second parent fixed
`489a127cf951f93f7bfc873e9333f05113f5b8e9`. Both histories are retained without rewrite.
Original STOP audit `3330427a1d774d30e65216ddab89d668d97c187c` remains separate evidence,
not a third merge parent. This section supersedes older continuation pointers below;
they describe prior checkpoints, not the current working location.

Status: GO_CANDIDATE_ONLY / AWAIT_HUMAN / UNMERGED_IN_MAIN / NOT_DEPLOYED.
Main is unchanged. No Stage B work or research planning is authorized here.
Merge commit `225efa9bb13b67bd5258faba79a47666c8906148` retains exact two parents above.
272 joint tests PASS; independent metadata5 PASS,0 Critical/Required; no unresolved
conflicts. [Integration evidence](../../research/v7/window04-reconciliation-evidence.json). The containing docs-only
closure commit is the final candidate SHA; main and original candidate are ancestors.
Resume by verifying branch/HEAD/clean here, reading this section and the evidence,
then await Human approval of that fixed SHA. Do not start Stage B.
Strict Phase 0 STOP; E1 Frozen Matrix BLOCKED; H2 NOT APPROVED; comparison options
remain PROPOSED. Read the local unique `research/v7/research-program.md` first
for reconciliation checkpoint; original preparation artifacts are unchanged.

## Fair evaluation preparation complete — fixed continuation pointer

Window02/03 source is integrated in local main at approvedeba6b649; this main
commit adds navigation only. New Window04 work remains UNMERGED/NOT_DEPLOYED on
`development/v7-fair-evaluation-decision@489a127cf951f93f7bfc873e9333f05113f5b8e9`, base`f31fcbba4d82d260915d7b640d80abc9c9ba8fc1`.
Worktree `/home/frank/projects/MarketMamba/.worktrees/v7-fair-evaluation-decision`.
WINDOW_COMPLETE / STOP. Readiness tool2a9de32 and decision package489a127 completed;
64 related non-training tests PASS,15 source hashes/cost arithmetic verified,
independent reviews0 Critical/Required. New comparison options remain PROPOSED;
no dates/thresholds/training/H2 approved. Original matrix BLOCKED, Strict Phase0 STOP.

Latest unique backlog is that pinned commit's `research/v7/research-program.md`
Window04 section; report `reports/v7_fair_evaluation_decision_package.md`, evidence
`research/v7/fair-evaluation-preparation-evidence.json`. New audit is not merged
into main. To resume, cd named worktree, verify HEAD/clean, read Current State,
Authority Map and that queue. No full historical rediscovery or repeated matrix
search. New bounded execution needs Human grant; no background continuation.
This pointer supersedes earlier Window04 authorized/READY text below.

## Fair evaluation preparation — completed checkpoint

Isolated `development/v7-fair-evaluation-decision` from integrated main closuref31fcbb.
UNMERGED/NOT_DEPLOYED for new work only. Existing-contract read-only E1 metadata
audit completed,5tests/independent5 PASS,0 Required; always execution_ready=false.
Decision package completed:15 source hashes, exposure ledger, fit/cost formulas and
Colab stop conditions independently verified;0 Critical/Required.64 related tests
PASS. WINDOW_COMPLETE/STOP, all new options PROPOSED. New execution requires
Human bounded grant. Resume at this branch via main pinned closure pointer.
[Unique backlog](../../research/v7/research-program.md#fair-evaluation-preparation-window-04),
[evidence](../../research/v7/fair-evaluation-preparation-evidence.json).

## Window02/03 integrated; fair evaluation preparation authorized

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved fixedeba6b649;
main ff-only9956446→eba6b649, pre/post267 tests each PASS, independent0 Critical/
Required.19 worktrees clean before merge;23 source/2 parser/1 report/7 result hashes
verified, protected contracts/data/models/lock unchanged. Fixed candidates retained.
[Integration evidence](../../research/v7/window0203-integration-20261021.json),
[Human scope](../02_History/Decisions/20261021-engineering-integration-fair-preparation.json).
This supersedes UNMERGED/no-main-merge text below only for admitted Window02/03.
No deployment, policy or research classification change. StrictSTOP/E1BLOCKED/H2
unapproved persist. A separate bounded CPU preparation window is authorized;
its current queue is [the unique program](../../research/v7/research-program.md).

## Data reliability / evaluation preparation window03 — 2026-10-21

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED. Current Human grant superseded the
Window02 STOP solely for this separate window. Main remains9956446; candidate
`development/v7-data-evaluation-preparation` joins main and fixed Window02f34f9f3
in isolation. Five navigation conflicts retain Window02 content, with this new
checkpoint taking precedence; main's previous pointer remains in parent history.
[Decision](../02_History/Decisions/20261021-data-evaluation-window03.json),
[unique backlog](../../research/v7/research-program.md#data-evaluation-window-03),
[evidence](../../research/v7/data-evaluation-window03-evidence.json).
Stage1 isolated integration GO at9bb4247 (259 tests, independent0 Required).
Parser complete: TWSE response date binding and TPEx unique header mapping,
8 tests + independent8 PASS; MOPS source/policy unchanged. No provider certification.
Fairness/readiness completed:12 source identities,7 result hashes,2seed identity
chains; independent0 Required. New designs PROPOSED; NOT_RUN_READY without matrix.
Runtime correction: candidate Data/contains3 inherited tracked historical files,
unchanged and parquet contents unread; no processed_v6 mount. Do not reuse older
Data-absent assumption. Final267 tests PASS, health WARNING/no FAIL. Main stays9956446; next session starts
at `.worktrees/v7-data-evaluation-preparation`, reads this queue/evidence and waits
for a new bounded grant. No main merge authorized. Strict Phase0 STOP/E1 matrix BLOCKED unchanged.

## Cross-domain window02 final checkpoint — STOP

Isolated `development/v7-cross-domain-window-02` from main closurec1179fe,
UNMERGED/NOT_DEPLOYED. W02-DATA/PREP/EVAL/MODEL completed; this bounded Human grant is exhausted;
[unique backlog](../../research/v7/research-program.md#cross-domain-window-02)
owns acceptance and state. No production parser edits or training permitted.
W02-DATA completed7-case offline diagnostic with independent review0 Required;
TWSE/TPEx source fixes and MOPS policy are now explicit Human decision items.
W02-PREP completed frozen contract/matrix metadata guards;34 tests and independent14
PASS. Metadata-only, no training readiness/admission; original matrix remains BLOCKED.
W02-EVAL rejects missing seeds/nonfinite gate values;6 tests + independent6 PASS.
Acceptance thresholds unchanged; historical evidence not recomputed.
W02-MODEL completed source-level selection/evaluation reuse audit;11 pinned sources
independently verified. New research choices remain PROPOSED/Human-only.
Final259 tests PASS, four independent reviews0 Critical/Required. No additional
high-value READY item demonstrated; see queue for P1/P2 decisions and recovery.
[Evidence](../../research/v7/cross-domain-window02-evidence.json).

## Charter integrated and window 02 authorized — 2026-10-21

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved exact
`f7af348710467ac381b2a84e8d192670ced89932`; WSL main ff-only from
`6d74d7384caa5fa90b1b050ce507bfb8282f19df` to that SHA. This containing
closure commit is docs/evidence only. [Integration evidence](../../research/v7/charter-integration-20261021.json)
records pre/post254 tests each (PIT49, non-training E1 31, portfolio67, ops81,
fetch3+governance23), 6 source/4 authority hashes,75 protected blobs unchanged,
17 clean worktrees and independent0 Critical/Required. No tiny training or GPU probe run.
Window01's UNMERGED wording below is historical and superseded only for integration.

[New Human decision](../02_History/Decisions/20261021-charter-integration-window02.json)
authorizes one next isolated CPU window, maximum3–5 meaningful units. Current
selection/checkpoint remains the [unique program](../../research/v7/research-program.md).
Discover Data, Model Research and Experiment Preparation before optional system
work; do not change V6 production. StrictSTOP/E1BLOCKED/Human research gates persist.
The fixed candidate branch remains unchanged. No push/deploy/Windows/MAS activity.

## Autonomous charter first window — 2026-10-21

**WINDOW_COMPLETE / STOP_AFTER_FIRST_WINDOW / UNMERGED / NOT_DEPLOYED.**
Human's [Charter](../GOVERNANCE.md#autonomous-development-charter--2026-10-21)
is persisted in existing governance, and the
[single backlog/checkpoint](../../research/v7/research-program.md#autonomous-charter-window-01)
records the complete first window. Branch `development/v7-autonomous-charter`
starts at unchanged main `6d74d7384caa5fa90b1b050ce507bfb8282f19df`.
Five units: Charter `be7998d`, E1 split completeness `3f07f80`, research decision
preparation `5847afc`, checkpoint fallback `2c60415`, snapshot pre-publication
validation in the containing closure commit. No further work starts in this window.

[Evidence](../../research/v7/autonomous-window-01-evidence.json): E1 33 + PIT49 +
governance23 =105 PASS; independent reviews0 Critical/Required after checkpoint's
second/final repair; snapshot one round; four research sources verified. Source
and output hashes/JSON/syntax/diff checks pass; knowledge health WARNING, no FAIL.
CPU synthetic engineering and PROPOSED research preparation only; no historical
checkpoint payload/matrix/data reads, external network, new training, GPU or MAS.
Existing E1 unittest includes a one-step synthetic CPU smoke. Loader remains
trusted-local and unrestricted: no historical-pickle safety certification.

This supersedes older no-next-program or in-progress statements solely for this
completed window. H0 remains locally integrated; Strict Phase0 STOP, E1 Frozen
Matrix BLOCKED, H2 and production/research gates unchanged. Another session reads
Charter + this backlog + fixed branch evidence, verifies Git/runtime, and waits
for a new bounded window grant before executing. No merge/push/deploy authorized.
No confirmed unexecuted READY item remains in this window; do not infer global
absence of defects or repeat E1 searches without new evidence.

## Portfolio History H0 completed — 2026-10-21

**LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED.** Human accepted the limited
fixture Product Slice and [approved exact4a66928](../02_History/Decisions/20261021-portfolio-final-h0-approval.json).
WSL main fast-forwarded from `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e` to
`4a66928aa4c9bd8a1ddf99a5b42866e9b22efb74`; subsequent containing commit is
**documentation/evidence only**, not a different product candidate. Pre/post CPU,
API/HTTP, frontend and real Chrome font/selection/error checks passed.
[Integration report](../../reports/v7_portfolio_h0_integration_closure.md),
[evidence](../../research/v7/portfolio-h0-integration-closure.json).
CPU01/02, PORT01–04, UX01–04, final UX and font correction are in local main.
This supersedes historical UNMERGED/H0-pending/UX-review-pending prose below for
these units only. Original reports, fixed branches and all16 worktrees are retained;
audit7ecb920 remains separate historical failed-preflight evidence, not merged.
No Required/Critical remains within this tested slice. Main is a valid source
baseline for future separately authorized CPU work; no new Program starts now.
Strict Phase0 STOP, E1 Frozen Matrix BLOCKED, H2 unapproved, research correctness
unapproved. No push/deploy/Windows operational change/training. STOP after closure.
The existing UI's static UNMERGED badge is historical candidate wording, not a live
Git integration indicator; canonical status above owns integration authority.

## Final H0 browser blocker resolved — 2026-10-21

Human [authorized only the late-font visibility repair](../02_History/Decisions/20261021-portfolio-font-blocker-authorization.json).
New branch `fix/v7-timeline-font-visibility` starts from exact `6c894794` product
candidate; `audit/v7-h0-preflight@7ecb920` is evidence only, NOT an ancestor.
True-font RED reproduced desktop3.5px/mobile2.5px clipping. Existing ResizeObserver
now observes all event rows as well as the fixed-height container; GREEN confirms
font-driven layout correction without page movement, focus loss or data change.
[Report](../../reports/v7_portfolio_font_visibility.md),
[evidence index](../../research/v7/portfolio-font-visibility-evidence.json).
Original Required finding RESOLVED within tested fixture scope; bounded preflight
engineering GO, pending Human approval of the new fixed SHA. Final candidate is the
containing commit, not the audit commit. Main remains `83a6a2e`; no merge occurred.
Human limited UX product acceptance remains in force; earlier pending-UX-review
language below is historical. Old H0 authorization does not extend to this new SHA.
UNMERGED / NOT_DEPLOYED; H2 unapproved, Strict Phase0 STOP, E1 Frozen Matrix BLOCKED.
STOP here: no further UX/research work authorized. Native Goal creation was rejected
because the previous H0 goal remains paused; no automatic continuation claim.

## CPU window 02 implementation boundary

`development/v7-cpu-window-01` continues from `7eb9cd9`, UNMERGED / NOT_DEPLOYED.
[Window 02 checkpoint](../../tasks/v7-cpu-development-window-02.md) owns current
engineering planning only. Window complete: normalized collision refusal at
`735ae6f`, duplicate JSON member refusal in this closure commit; both are
reviewed synthetic engineering, not Human research approval. Existing portfolio spec/engine duplicate-refusal and
journal integrity contracts remain authoritative; no research/production promotion,
no replacement of historical window 01 evidence or E1 frozen artifacts.

## Strategic planning ownership — 2026-10-21

The dated current section in [research-program.md](../../research/v7/research-program.md#current-development-program--2026-10-21)
is the **single current development backlog**. This is planning/navigation authority
under the present Human request, not approval of proposed execution/research.
The original umbrella research contract remains historical context in that file;
S5–S9 frozen machine contracts/results retain research authority. Old tasks/plan,
todo and CPU-window queues remain dated evidence, not parallel active roadmaps.

Candidate `integration/v7-cpu-roadmap` from `4c388de` is independently reviewed
GO for local development consideration only, UNMERGED / NOT_DEPLOYED. Main remains
`83a6a2e`; no main merge or source cherry-pick occurred. [Evidence report](../../reports/v7_strategic_roadmap_recovery.md).
PORT-01–04 have [explicit H1 approval](../02_History/Decisions/20261021-portfolio-history-h1.json)
for fixture-only CPU read-only engineering; completion evidence is
[portfolio-history-evidence](../../research/v7/portfolio-history-evidence.json).
PORT-01–04 are COMPLETED_VERIFIED within fixture engineering, with
[acceptance report](../../reports/v7_portfolio_history_acceptance.md).
Human Product Slice Review is pending; engineering completion is not result approval.
New implementation is branch-local UNMERGED / NOT_DEPLOYED. H0 main integration and H2 research
admission remain unapproved and separate. MAS Core Live is not certified or activated.

## UX product direction — 2026-10-21

The subsequent [Human final-closure decision](../02_History/Decisions/20261021-portfolio-ux-closure-decision.json)
accepts the overall dual-mode direction and authorizes only selection visibility
and event readability. Both finishing tasks are engineering-verified; final Human
closure review remains pending. Mobile advanced requires basic access only; AI
should prefer the structured API/JSON. This decision grants no H0/H2 or new research authority.
The following original UX-01–04 record remains historical scope evidence.

[Human UX decision](../02_History/Decisions/20261021-portfolio-ux-human-decision.json)
authorizes only UX-01–04 presentation engineering on `5a2bc6f` ancestry.
The fixture UX slice is engineering-verified; [UX acceptance](../../reports/v7_portfolio_ux_acceptance.md)
is ready for Human review, not Human acceptance or H0/H2 approval.
Pinned project-local frontend skills are subordinate guidance, never research,
production or project authority. No H0/H2 admission or accounting changes.

## Source-of-truth Hierarchy

1. Versioned machine contracts, manifests, hashes, and executable tests.
2. Completed run-local final artifacts and validated checkpoint manifests.
3. Implementation on an explicitly named branch and commit.
4. Timestamped runtime observations.
5. Human reports and guides.
6. Historical notes and legacy artifacts.

An authority class describes evidence strength, not deployment status. Branch-local code and untracked candidates are never promoted to `CANONICAL` merely because they exist.

## Authority by environment and evidence class

- **Operational authority:** implementation plus a timestamped external action/observation. `AGENTS.md` and a runnable script alone do not prove a live schedule. A stale observation becomes `STALE_OBSERVATION`, never a fresh fact by documentation update.
- **Research state:** Current State summarizes admitted branch-qualified results. A main file can lag an accepted worktree result; the frozen experiment contract controls the question, completed run manifest/hashes control the result, and neither promotes production.
- **Artifacts:** a result filename or copied ZIP is insufficient without source commit, contract, input/matrix identity, environment, split/support, model/checkpoint identity, outputs and evidence class. Mutable Parquet caches need separate provenance snapshots. Untracked local files are candidates, not canonical authority.
- **Runtime:** WSL development, Windows Scheduler checkout, Colab Drive and deployed API/UI are distinct. Record `observed_at` for Windows/service facts; use realpath and source commit for cross-environment identity. No inferred equivalence.
- **Branch/worktree:** a versioned `commit:path` remains valid historical/research authority without being merged or deployed. A branch label aids navigation but can move. Worktree checkout name is not evidence of its current branch.

## Freshness Rules

- `RUNTIME_OBSERVATION`: requires `observed_at`; after 7 days it becomes `STALE_OBSERVATION` until rechecked. It is not deleted automatically.
- Repository contracts: no time expiry. They change only through commit identity or explicit supersession.
- Completed-run artifacts: remain evidence for that run; they do not automatically describe a newer run or current production.
- `UNKNOWN`: has no primary authority. Secondary evidence may explain what is known without filling the gap.

## Topic Map

| Topic | Primary Authority | Authority Class | Status | Supersedes | Secondary Evidence | Conflicts / Notes |
|---|---|---|---|---|---|---|
| Worktree responsibility | [Worktree Register](Worktree_Register.md) | `CURRENT_REFERENCE` | `FOUR_ACTIVE_K_BRANCH_HISTORY_RETAINED_2026_09_22` | Ad-hoc worktree cleanup assumptions | Git worktree list, branch HEAD and status | Registration does not authorize removal; ownership/creation dates may be UNKNOWN. |
| Development runtime contract | [Current State](Current_State.md) and [`AGENTS.md`](../../AGENTS.md) | `CURRENT_REFERENCE` | `WSL_NATIVE_DEVELOPMENT` | Old assumption that `/mnt/d` checkout is canonical development | `docs/operations/wsl-migration-2026-09-07.md` | Operational Windows/WSL bridge is separate; no runtime migration this change. |
| Mutable data cache | [`V6/marketmamba/config.py`](../../V6/marketmamba/config.py) and source manifests | `CURRENT_REFERENCE` | `MUTABLE_NON_PIT_CACHE` | Treating cache as frozen research evidence | `MARKETMAMBA_DATA_ROOT`, symlink realpath, V7 source registry | Raw caches can change; authority requires snapshot/identity. |
| Colab runtime | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json` | `CURRENT_REFERENCE` | `SEPARATE_LOCKED_TRAINING_RUNTIME` | Local Python environment inferred as formal Colab | E1 bundle and exact requirements lock | Formal result needs environment/bundle/run identity. |
| Local untracked candidates | None | `UNKNOWN` | `NOT_ADMITTED` | File-existence/current-version inference | Main untracked research files | No branch commit/contract or deployment authority from untracked source. |
| Early V7 local historical source | [`2026-09-22 preservation register`](../../research/v7/historical-local-source-register-20260922.md) plus same-commit source files | `HISTORICAL_ONLY` | `PRESERVED_NOT_CURRENT_E5_OR_E1` | Untracked early V7 source of uncertain ownership | Dated deliveries and research notes linked in register | Registration preserves identity and source; it does not validate results or promote an old architecture. |
| Dated V7 research/task notes | [`2026-09-22 notes register`](../../research/v7/historical-notes-register-20260922.md) plus same-commit notes | `HISTORICAL_ONLY` | `PRESERVED_SUPERSEDED_AS_CURRENT_STATE` | `research/v7/current-state.md` and task plan/todo as active instructions | Original note hashes and branch-local plan/todo blob IDs | Old paths retained for historical Phase 0 references; canonical Current State and frozen contracts control present decisions. |
| Current project state | [Current_State.md](Current_State.md) | `CANONICAL` | `CANONICAL_CURRENT_PROJECT_MAP` | README/OVERVIEW current-state prose and old handoffs | [K0 report](../../reports/marketmamba_k0_project_knowledge_discovery.md) | Accepted 2026-09-20; update or supersede when semantics change. |
| Domain semantics/navigation | [Domain notes](../01_Domains/) | `CURRENT_REFERENCE` | `CANONICAL_DOMAIN_NOTE` / `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN` | Scattered README/guide summaries | [Knowledge index](../README.md), machine contracts linked per note | Notes summarize boundaries and point to authority; they never override machine contracts. |
| Knowledge governance | [GOVERNANCE.md](../GOVERNANCE.md) | `CANONICAL` | `ACTIVE_GOVERNANCE_V1` | K0 governance proposals and manual-only K1 maintenance | `knowledge/knowledge-impact-map.json`, governance tools/tests | Governance detects and records; it never rewrites authority or business semantics automatically. |
| Runtime observation store | [`runtime-observations/`](../runtime-observations/) | `CURRENT_REFERENCE` | `TIMESTAMPED_EVIDENCE_WITH_TTL` | Runtime facts copied only into prose | Runtime status tool and K0 observation evidence | Recording requires an actual observation; stale evidence remains stale. |
| Production runtime | [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) in Windows `main@8b82a68` plus [Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) | `CURRENT_REFERENCE` | `FETCH_ONLY_ACTIVE`; full path `DISABLED` | README/OVERVIEW implication of full scheduled daily inference | D: batch and [operational path](../runtime-observations/operational-path-20260922T132521+0800.json) | Scheduler selects Windows checkout, not WSL development HEAD. |
| Scheduler state | [2026-09-22 Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) | `RUNTIME_OBSERVATION` | Active fetch-only / disabled full V6.2; last run 2026-09-21 result 0 | 2026-09-20 observation and older snapshots | [K0 runtime map](../../research/project-knowledge/k0/runtime-map.json), local logs | Becomes `STALE_OBSERVATION` after 2026-09-29 13:21:35 +08:00. |
| Operational checkout and WSL environment | [Operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) | `RUNTIME_OBSERVATION` | Windows `main@8b82a68` → distro `Ubuntu` → `mamba_env` Python 3.11.15 | Assumed WSL-native development checkout is scheduled | Batch content, `wsl.exe --list --verbose`, Git | Development `Ubuntu-24.04` checkout is distinct `main@3b0df0f` at observation. |
| Operational data root and freshness | [Operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) | `RUNTIME_OBSERVATION` | D: physical `Data/processed_v6`; ten daily sources max 2026-09-21 | Mtime-only freshness or guessed path equivalence | Parquet metadata, `v62_daily.log` | Date check 10/10 does not certify source validity; universe delta warning remains. |
| Data provenance policy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/source-reliability.json` | `CANONICAL` | `ACTIVE_RESEARCH_CONTRACT` | Mixed-source OHLCV admission | Provenance lineage, source index and mismatch inventory in the same directory | Source classification must precede proxy classification. |
| Reusable PIT/provenance code | [`source adoption register`](../../research/v7/source-adoption-register-20260922.json) and its main destinations | `CURRENT_REFERENCE` | `IN_MAIN_RESEARCH_ONLY` | Branch-only implementation for the six adopted primitives | 33 focused CPU tests; source blob IDs in register | Frozen corrected baseline result/readiness and superseded 32/8 architecture draft remain branch-qualified. |
| PIT policy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/policy.json` | `CANONICAL` | `ACTIVE_RESEARCH_CONTRACT` | Current-world/backfilled data assumptions | Executable builders and Phase 0 tests | Missing lifecycle/publication/provenance evidence fails closed. |
| Feature contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json` | `CANONICAL` | `48_FEATURES` | Legacy V6.1 56-feature, V6.2 59-feature and baseline 66-feature contracts | Corrected baseline builder/tests | Counts belong to separate contracts; they are not interchangeable. |
| Feature list | Same versioned feature manifest and its order hash | `CANONICAL` | `ORDER_PINNED` | Legacy feature orders and removed fundamentals | [K0 feature map](../../research/project-knowledge/k0/feature-map.json) | No current-industry backfill, graph features, or industry neutralization. |
| Label contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/label-manifest.json` | `CANONICAL` | `NEXT_OPEN_NO_ROLL` | Legacy close-to-close semantics | Policy plus historical label materializer tests | Entry is next frozen session open; exit is h-th holding-session close. |
| Universe contract | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/universe-manifest.json` | `CANONICAL` | `VERIFIED_LIFECYCLE` | Current constituent backfill | Historical security-master builder/tests | Membership and tradability remain separate concepts. |
| Historical simulation proxy | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/execution-proxy-policy.json` | `CANONICAL` | `P0_FROZEN`; simulation readiness `PASS` | Ad-hoc execution heuristics | Proxy validation and provenance-remediation readiness | Must never be called `VERIFIED_EXECUTABLE`. |
| Strict executable evidence | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` | `CANONICAL` | `STOP`; verified labels 5d=0, 10d=0 | Any claim that proxy PASS implies trading readiness | Corrected Phase 0 audit | Strict STOP and simulation PASS are intentionally simultaneous. |
| Corrected E5 architecture | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-replication-v1/incumbent-contract.json` | `CANONICAL` | `d_model=64`, `d_state=32`, 48 features, no graph | `architecture-config.json` draft values 32/8 | Completed E5 result identity and final runner | Completed result authority outranks the earlier draft. |
| Training recipe | Same incumbent contract; E1 freezes it in `experiment-contract.json` | `CANONICAL` | `FP32_ADAMW_20_EPOCH_MAX` | Earlier notebook-only assumptions | Exact environment lock and final runner source | E1 changes split, not recipe. |
| Reusable corrected E5 and E1 source | [Source adoption register](../../research/v7/source-adoption-register-20260922.json) and main research modules | `CURRENT_REFERENCE` | `IN_MAIN_RESEARCH_ONLY`; E5/E1 core, frozen contract copies, checkpoint telemetry fix, Colab log wrapper | Branch-only source for adopted slices | 33 E5, 22 E1 and 27 recovery focused CPU tests on main | Result manifests, bundles, diagnostic output, operational runtime and deployment remain branch/external authority. |
| Corrected E5 checkpoints | Validated run-local checkpoint manifests for each seed | `RESEARCH_EVIDENCE` | Seeds 17/29 complete; seed43 `INCOMPLETE_EXCLUDED_FROM_FORMAL_CONCLUSIONS` | Filename/pointer-only inference | Diagnostic resume forensics and telemetry | Seed43 pointer metadata named missing generations and recorded `best_epoch=8`; the surviving orphan is epoch 15/step 39,915 and is convergence evidence only, not promotable or directly resumable. |
| Diagnostic v1 | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v1/summary.json` | `RESEARCH_EVIDENCE` | `SUPERSEDED_FOR_CURRENT_CONCLUSION` | Raw partial-run impressions | v1 component analyses | Extended diagnostic v2 is the current conclusion. |
| Extended diagnostic v2 | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v2/diagnostic-summary.json` | `RESEARCH_EVIDENCE` | `STOP_FOR_TRADING_CONTINUE_RESEARCH` | Diagnostic v1 as final conclusion | v2 head, drift, regime and equal-window analyses | Research evidence only; not production/trading authority. |
| E1 experiment specification | `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json` | `CANONICAL` | `FROZEN_EXPERIMENT_SPECIFICATION` | Informal experiment proposals | Contract diff, runner, checkpoint store, generated notebook | The pre-run decision field remains historical specification, not completed-result authority. |
| E1 completed result | `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json` | `RESEARCH_EVIDENCE` | `PASS_REFRESH_HYPOTHESIS` | External-only/unverified result state | Final evaluation, run summary, seed manifests and hashes in the same directory | Historical-simulation proxy only; strict Phase 0 `STOP`; no portfolio admission. |
| E1 post-run diagnostic | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json` | `RESEARCH_EVIDENCE` | `EPOCH_DEGRADATION_INCONCLUSIVE`; `REGIME_DEPENDENCE_INCONCLUSIVE` | Undiagnosed post-run state | Month and leave-one-month-out analyses, checkpoint inventory, logged validation history | July-sensitive 10d absolute level; no final-support epoch replay or causal regime series. |
| Portfolio semantics | [`V6/experimental/v7_portfolio_contract.py`](../../V6/experimental/v7_portfolio_contract.py) plus engine/journal tests | `CURRENT_REFERENCE` | `RESEARCH_ONLY` | V6.2 JSONL as V7 semantic authority | `v7_portfolio_engine.py`, `v7_portfolio_journal.py` | No live broker or scheduler integration was discovered. |
| Production-system architecture | `docs/production-system-architecture-audit@9efed7b:reports/marketmamba_production_system_audit.md` and roadmap | `CURRENT_REFERENCE` | Audit complete; roadmap Phase A–F | Earlier production prose | Production-system machine contracts | Branch implementation may be newer than prose. |
| Phase A implementation | `89f9496:V6/marketmamba/ops/` | `CURRENT_REFERENCE` | `IN_DEVELOPMENT_MAIN`, `NOT_DEPLOYED`; Task 4/5 scoped approvals retained | Earlier candidate-only status; original branches retained | Explicit local integration decision and closure record | Local merge does not grant production or research authority. Shadow evidence does not own legacy fetch success. |
| V7 status API/UI source | [`app/backend/routers/v7.py`](../../app/backend/routers/v7.py), [`app/frontend/src/pages/V7Status.jsx`](../../app/frontend/src/pages/V7Status.jsx), focused tests | `CURRENT_REFERENCE` | `COMMITTED_NOT_DEPLOYED` in WSL development main | Local untracked V7 status candidate | Health summary builder and frontend contract tests | Additive status route only; no model inference, portfolio or scheduler authority. |
| API/UI live deployment | None for live deployment state | `UNKNOWN` | `UNKNOWN` | Claims inferred from source presence | Source paths above | Source capability does not prove deployment; provider revision remains unverified. |
| Agent instructions | [`AGENTS.md`](../../AGENTS.md) | `CURRENT_REFERENCE` | Navigation/policy only | Stale duplicated assumptions in `CLAUDE.md` | K0 agents/CLAUDE audits | Machine contracts and this map outrank project-state prose in agent guides. |

## Authority Classes

CPU-only window 01 is branch-local engineering under the existing approved
portfolio engine contract, not a research authority replacement. See
[scope/backlog](../../tasks/v7-cpu-development-window-01.md).
Main stays `83a6a2e`; E1 evidence remains pinned to `11ae71d`.
Neither test success nor Agent work selection grants research acceptance.

E1 recovery observation (2026-10-21):
[inventory](../../research/v7/e1-rolling-origin-refresh-v1/evidence-recovery-20261021.json)
records surviving byte integrity and limited local discovery. It does not
supersede eec63c5 completed-result or ef020e5 diagnostic research conclusions.
Missing matrix identity and unvalidated payload semantics stop replay.

Subsequent explicit Human integration decision authorized local WSL main only:
`89f9496:research/production-system/phase-a/` is now reachable from main.
The [closure record](../../research/production-system/phase-a/integration-closure.json)
records pre/post identity and verification. This supersedes the candidate's
unmerged status, not research/production authority or either scoped approval.

2026-10-21 explicit Human decision supersedes the prior Task 4 approval-pending
state only: [decision record](../02_History/Decisions/20261021-task4-human-approval.json)
binds limited engineering correctness to fixed `1e119ef` and its verified index.
Agent integration/test judgments do not extend that approval. The subsequent
[Task 5 Human decision](../02_History/Decisions/20261021-task5-human-approval.json)
approves only limited offline engineering at fixed `1c5250e`. Historical
acceptance artifacts remain unmodified. Final Phase A acceptance and integration
are still Human decisions; Agent GO is only an engineering recommendation.
Strict Phase 0 STOP,
research authority and production boundaries remain unchanged.

Historical 2026-10-20 bounded Task 4 follow-up (superseded approval status):
`fix/phase-a-task4-evidence@46df7b3402f85877a5fc55600d53b15a6f3a7849`
is an unmerged artifact-summary fix/evidence candidate derived from `f5b0ff3`,
not a replacement authority or Human correctness decision. Its evidence index
records 70 ops + 3 mocked fetch-only passes and remaining concurrent snapshot
isolation evidence gaps. Task 5 remains not started; legacy fetch owns outcomes.

Same-day follow-up `fix/phase-a-task4-evidence@1e119ef` closes the two listed
isolation evidence gaps with real-process tests (75 ops total). Its Task 4
review package is ready for Human decision, **NOT_APPROVED**. This supersedes
only the missing-isolation-evidence observation, not authority or research state.

Branch-local recovery candidate (2026-10-20): `fix/v7-snapshot-immutability`
changes only the snapshot writer/reader and tests from main `50c5b78`, with its
[checkpoint](Current_State.md#bounded-recovery-checkpoint--2026-10-20).
It is `UNMERGED_IMPLEMENTATION / NOT_DEPLOYED`, not a replacement of frozen
PIT contracts, source-adoption history, research results or Phase A approval.
The 2026-09-22 byte-identical adoption claim describes that adoption, not this
subsequent candidate. No authoritative evidence has been superseded.

- `CANONICAL`: accepted contract or project-map authority for its explicitly stated scope.
- `CURRENT_REFERENCE`: best current implementation/report, but not necessarily immutable or deployed.
- `RESEARCH_EVIDENCE`: valid research conclusion for its run/evidence class.
- `HISTORICAL_ONLY`: retained context that must not drive current decisions alone.
- `RUNTIME_OBSERVATION`: external state valid only at its observation time and freshness window.
- `UNMERGED_IMPLEMENTATION`: branch-local implementation, neither deployed nor mainline authority.
- `UNKNOWN`: no verified primary authority currently exists.
