## 事故狀態機驗收候選 — 2026-11-26

[WP5 報告](../../reports/forward_wp5_20261126.md)與[設計](../../tasks/design-forward-incident-state-machine-20261126.md)：`verdict(sealed_chain)` 統一 attempt、每日與完整鏈開盲；先 OPEN 後驗收再 CLOSE，未關閉／identity／fatal fail closed；跨 session 合法 R+1／R+2 可恢復，無影響代碼保留正常驗收。獨立 oracle 做有限序列窮舉與兩 R_j 等價狀態排列；WP2 20 seeds 全欄／labels／跨欄隨機 missingness oracle 含未成熟 label。此為 `builder_fwd_sm@ed38501` 未提交候選，SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；不改 P2 或 fit 配方，不授正式開盲／production，不解除 seed17／WP6 與原外部 STOP。驗證數值以兩份報告最終補記為準。

## WP5 每版非績效驗收候選 — 2026-11-26

[WP5](../../tools/forward_wp5.py)與[Human靜態清單](../../docs/operations/forward-retraining-checklist.md)以同一source pin驗每版／每次重試，沿WP1 origin−21、快照R−1 18:00與R+2 18:00期限；沿WP4重驗完整三seed pointer／native roundtrip及WP2／WP3合成C0+ reference。兩新程序獨立重做keys／48與185欄／sample／scores／ranks逐bytes相等，非績效、不調參。私有event鏈固定R→R+1→R+2、snapshot／origin／seeds；超時／identity錯立即失效，初始M0／F0失敗不啟動。純合成切換授權不修改正式模型或portfolio state，來源摘要日鏈獨立。

**SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** [報告](../../reports/forward_wp5_20261126.md)記兩mutants與終局checks；合成PASS不是原P1 reference通過、Human source簽署、實際20-session鏈、跨環境lock或WP6正式端點驗收。WP2 seed17與WP1來源／calendar／PIT／lifecycle及WP3 provenance／daily-source STOP保留。無真實forward fit／blind讀取／GPU／排程／production authority或外部observed_at變更。

## WP4 每 fit 環境封存候選 — 2026-11-26

[合成runner](../../tools/forward_wp4.py)每fit前先fsync environment及seed／context／params／target request，完成模型版本內再保存完整快照；fit失敗仍保留已落盤環境。記錄套件／LightGBM版本、Python／compiler／libc、loaded native binary hashes、BLAS／OpenMP與thread env、n_jobs4、CPU flags／cpuinfo／affinity及X dtype／layout／identity。大型binary SHA按完整stat identity在單一程序快取，變更失效；每fit仍寫完整map。這補足日後診斷資料，不保證跨主機重訓逐位元相同，也不證實WP2 seed17的未證根因，不改C0+配方或deterministic參數。

[淨化介面](../../tools/forward_wp4_summary.py)只用合成來源／clock，來源摘要先於private fit決定，Python與native logs私有；正式R+1／R+2／18:00重訓acceptance仍留WP5。證據與限制見[WP4報告](../../reports/forward_wp4_20261126.md)，SYNTHETIC_CPU_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；沒有真實訓練、GPU、blind、OS隔離、scheduler或production authority。

## P4d heartbeat 與資源量測工程候選 — 2026-11-25

`job-run-mwvqcr5w-e515cb/builder_p4d_fix1@d36e709` 的未提交 bounded patch，**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED / CPU_ENGINEERING_ONLY**。[工程報告](../../reports/v7_p4d_engineering.md#排練-heartbeat-required-與最大-fit-量測入口--2026-11-25)記錄排練前啟動共用Progress、REHEARSAL／FORMAL模式與stage、60秒heartbeat替身、寫入失敗繼續及2026全列C0+資源量測入口；不改split／features／selection／queue／threshold或P4a來源policy。僅36列合成特徵＋擬合，未實際量測2026全列RSS，不恢復抽樣cap；CPU回歸不是現場Drive／T4排練證據。原獨立NO_GO仍待複審，完成不等於Human acceptance；沒有研究結果／canonical artifact supersession／production authority變更，不更新authority-map.json或外部observed_at。Next：獨立審查與另案本機最大fit量測。

## P4d stage1b 排練接續候選 — 2026-11-25

獨立tools交付保留原P4a來源inventory。P4d成功標記位於`progress/formal/rehearsal_passed.json`，六個fit與評估產物先複製至獨立publication namespace、重驗後才發布；重連逐bytes／binding／schema／內容重驗，不依賴舊/tmp，不共用P4a帳本。None／空證據STOP，損壞marker重新排練；checkpoint／固定更新／研究threshold不變。詳[工程報告](../../reports/v7_p4d_engineering.md#第一段接續-stage1b--2026-11-25)與[Human需求](../../docs/operations/colab-human-requirements.md#p4d-第一段候選2026-11-25)。CPU替身六fit／notebook驗證與全套回歸僅工程證據；T4顯存／耗時、全列C0+RAM／完整推論、Drive與長排練heartbeat仍待現場核對。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，獨立review與Human acceptance未完成，無GPU／Colab／Drive／外部runtime observation更新。

## P4a 第 13 次複審 REQUIRED＋兩項 MINOR 修補候選 — 2026-11-24

`job-run-mwtn0pir-48e95f`／`builder_p4a_gate3`，沿用既有 detached worktree，base/HEAD `40d621cc9ac586e45fff1e095f59465b388094a1`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依明確授權保留未提交 patch。第 13 次獨立複審為 NO_GO；本輪修補 Q5 巢狀 schema、超大數值／遞迴損壞 marker，以及 metadata 符號連結。完成不解除 NO_GO，不等同 Human acceptance。

[runner](../../V6/research_lab/p4_colab.py) 的階段結果、發布 metadata、Q5 診斷與成功 marker 共用單一封閉遞迴 schema；必要／允許欄位、精確型別、數值有限性／值域、ISO 日期、凍結 rule、R=10 與 permutation/date 支援均 fail closed。`excluded_dates` 依 producer 定義為日期字串清單（可空）；布林只在契約的 bool 欄位合法。Q5 producer 原先傳 session index，現在傳同列 `dates.npy` 的日期，合成證據確認分組與數值診斷不變。marker 的 OverflowError／TypeError／RecursionError 與既有解析錯誤視為無效，顯示期望／實際／下一步後移除並重排練。完成、發布、result JSON 與 marker 拒絕符號連結；產物 bytes、請求、來源與正式完成驗證沿用原 gate。

[工具重產宣告](../../tasks/p4a-source-compat-20e6b54.json) 的 `to.commit` 為本輪可得 HEAD `40d621c`，230 個來源 hashes 綁定 working bytes；後續 commit 可不同，deliver 必須以最終 bytes 重算，不可直接採用舊宣告。36 個 Q1／Q2 W60 完成 fit 全部 resume_fit 驗證／保留 bytes／不重訓的條件與 64-fit SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23` 不變；研究／環境／預算契約、P3 adapter／portfolio engine 及正式 Data／模型未動。AST pins 僅限定候選範圍，非獨立審查證明。

Evidence：本機 CPU 合成與真子程序發布替身；RED→GREEN、逐欄位型別矩陣、三項新 in-memory mutation、實際完整測試與 artifact identity 見[工程報告](../../reports/v7_p4a_colab_engineering.md)。CUDA／官方 mixer／真實 fold／W250 峰值／Colab／Drive FUSE／現場 36 artifacts 與正式接續仍 UNVERIFIED。Next：獨立複審 patch、宣告與測試，通過後由 Human 決定現場排練及 acceptance。無 GPU／網路／Colab／資料取得／production／scheduler／排程／additional agents／wakeup／recurring jobs／commit／push／merge／deploy，未更新 runtime observed_at。

## P4a 第 12 次複審兩項 REQUIRED 修補候選 — 2026-11-24

`job-run-mwtlq2cv-a82151`／`builder_p4a_gate2`，沿用既有 detached worktree，base/HEAD `99b4e0e3547868847262aa305cd5fe7100c3c9f5`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依授權保留未提交 patch。第 12 次複審為 NO_GO：排練只憑返回碼放行、有效 JSON 的成功標記可被清空／竄改仍略過。本候選收緊 [runner](../../V6/research_lab/p4_colab.py) 的工程 gate；不解除原 NO_GO，不取代獨立複審與 Human acceptance。

閘門逐階段讀獨立結果 JSON，核對 Q3 W120 mamba2／Q3 W250 mamba2 各 1 fit、Q5 2 fits 與 validation／evaluation 完成，核對 source／prepared data／reference／environment 與本次 run command、nonce；每個 fit 契約綁定請求，並由 gate 自行重驗既有 completed-output validator 的 scores／reference／checkpoint／complete identity 與 bytes。空輸出、吞例外後正常退出、錯誤 selection／identity 或缺產物一律 STOP，不寫成功 marker、不啟動正式 runner。成功 marker 要求完整 schema、恰好三階段 identity／耗時／REHEARSAL evidence／fit 數及內容 digest；缺欄位、型別錯誤、未知欄位、空 stages 或不符即重排練。digest 只防誤改，非惡意防護；重連快取不依賴 `/tmp` 產物仍存在，marker 不是正式完成或研究 authority。

[工具重產宣告](../../tasks/p4a-source-compat-20e6b54.json) 的 `to.commit` 從舊 `bf0114d` 更新為本任務 HEAD `99b4e0e`，sources hashes 綁定 dirty working bytes；不是已提交或已接受的認證。36 個 Q1／Q2 W60 completed fits 全部 resume_fit 驗證的採用條件、64-fit SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23`、研究／環境／預算契約及 P3 adapter／portfolio engine bytes 均維持。AST pins 為候選範圍限制，仍待獨立複審。

只有本機 CPU 合成工程證據與真子程序替身；CUDA／官方 mixer／W250 峰值／Colab／Drive FUSE／現場 36 artifacts 仍 UNVERIFIED，未更新 runtime observed_at。沒有 GPU／網路／Colab／資料取得／production／scheduler／排程／額外代理／commit／push／merge／部署。實際測試、變異、限制及交接見 [工程報告](../../reports/v7_p4a_colab_engineering.md)。Next：獨立複審 patch、宣告與測試，再由 Human 決定接受及現場驗證；本次完成不等於 Human acceptance。

# Training

## P4a 正式前自動排練閘門候選 — 2026-11-24

`job-run-mwtk8k4r-a80ee8`／`builder_p4a_gate`，沿用既有 detached worktree，base/HEAD `bf0114dad9dc2d9bef7502c21d6fd36737eb0458`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依 no-commit 指示保留 patch。獨立審查 NO_GO 的唯一 REQUIRED 為 notebook 缺正式前自動排練；[產生器](../../V6/research_lab/p4_bundle.py) 現在於正式 runner 同一格前段依序跑 Q3 W120 mamba2、Q3 W250 mamba2、Q5（兩 prerequisite fits），各 1 step，真實資料／官方 mixer 由現場執行；本輪只有 CPU 合成工程證據。

FORMAL／resume 預設啟用、SPEED_TEST 預設關閉，第 0 格可明設 `REHEARSE_FIRST`。任一步失敗／中斷不啟動正式 runner；輸出失敗階段、`/tmp`、期望／實際／下一步。全部成功才寫 progress 的 `rehearsal_passed.json`；同一 run command／已驗 bundle source／prepared matrix＋rankgauss／資料 realpath／實際 environment identity 相同才略過，標記不是研究或完成 fit authority。[runner](../../V6/research_lab/p4_colab.py) 新增 W≠60 的 sessions／eligible dates 前置檢查，W60 不走新路徑；排練成功輸出 CUDA peak allocated bytes。

[來源相容宣告](../../tasks/p4a-source-compat-20e6b54.json) 由工具重產，誠實分類自動閘門與長窗前置檢查；固定 queue SHA `4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23`、36 個 Q1／Q2 completed fits 全部 resume_fit 驗證採用條件與研究／環境／預算契約不變。沒有改 P3 adapter／portfolio engine 或正式 Data／模型。完整實際測試輸出、位元比對、三項 pytest exit=1 的 in-memory 變異與限制見[工程報告](../../reports/v7_p4a_colab_engineering.md)。

Next：獨立複審此 patch，再由 Human 驗證真實 CUDA／官方 mixer／Colab／Drive 與正式接續；本輪完成不等於 Human acceptance，不解除 NO_GO。W250 windows 張量／GPU 峰值、首次全量推論耗時、Drive 真實 36 artifacts 均 UNVERIFIED；無 GPU／網路／Colab／production／scheduler／additional agents／commit／push／merge／deploy，無 runtime observed_at 更新。

## P4a Q3 尾端標籤事故與排練候選 — 2026-11-23

`job-run-mwt8u5js-7f212c`，Codex builder，既有 detached worktree `fixer_p4a_q3`，base/HEAD `685f6ece1faa7ed39ff7ebff7056b2c4ad7eb4b2`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。依 Human bounded EXECUTE_APPROVED／no-commit 指示保留 patch；無 GPU、網路、資料取得、Colab、production、scheduler、additional agents 或正式 Data／模型寫入。

Human 報告舊 `02_formal@4e56e9d` 已完成 Q1＋Q2 共 36 fits，Q3 第一筆 2021/mamba2/17/W120 因尾端未知 label_y5 被 common_rows 當成錯誤而中斷；這是 Human-reported runtime evidence，未自行觀測 Drive，也未改任何 observed_at。本次 [evaluator](../../V6/research_lab/p4_eval.py) 允許 NaN 等價的列身分核對、仍拒絕 ±inf／標籤不一致；組合收益由價格與 score 決定，不使用該標籤。Family C 仍取全部 W60／W120／W250／C0+ 交集後逐 seed 排名。Q5 IC 改用共用成熟標籤列並回報排除量；沒有成熟支援、每日期不足 10 名或常數相關仍 fail closed。沒有變更研究契約、queue、矩陣、環境／lock、預算契約、P3 adapter 或 portfolio engine。

[runner](../../V6/research_lab/p4_colab.py) 新增 `--rehearse`，以真實 data_dir 執行同一 train_fit／train_one／predict／發布與輸出 validator；只寫 `/tmp/p4-REHEARSAL-*`，標示 `REHEARSAL / 不是研究結果`，正式 resume_fit 不識別排練完成檔。Q5 先產生兩個排練檢查點再跑診斷。[Human 規則第 9 節](../../docs/operations/colab-human-requirements.md) 要求各類第一個 fit 與兩種長窗均先現場排練；CPU 合成證據不能代替此 gate。

[來源相容工具](../../V6/research_lab/p4_source_compat.py) 擴充窗≠60 修正／排練／Q5 診斷分類；以 AST pins 限制本候選變動面，原 frozen queue／recipe／ledger／正式 completed validator 的預設 gate 不放寬。[工具重產的 GENERATED 宣告](../../tasks/p4a-source-compat-20e6b54.json) 綁定目前工作樹 bytes（舊檔名不是新 source commit），不升格研究 authority；採用前必須原地 resume_fit 驗證 36 個 Q1／Q2 W60 完成 fit，保留 bytes、不重跑。先前只限進度／日誌的相容範圍僅由本事故專項擴充，不授權任意換配方。

CPU 證據、完整回歸實際輸出、四項 in-memory 變異、來源 identity 與限制見 [工程報告最新節](../../reports/v7_p4a_colab_engineering.md)。涵蓋舊／新各 36 筆 W60 train_fit 呼叫記錄未進 common_rows、原 GRU／crossstock 權重／預測／RNG 位元一致測試，以及 36 完成後只啟動第一個 Q3 的接續整合。這些是合成資料／CPU mixer 或 artifact 替身證據；真實 fold CUDA、官方 mixer、T4／Colab／Drive FUSE、完整正式訓練和 Human acceptance 仍 UNVERIFIED。Next：獨立審查 patch／宣告，Human 於本機 RTX 3060 或 Colab 真實資料排練，再決定正式接續；本輪不產生新交付 ZIP／notebook，不消耗 GPU。

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

## P4a completed-fit 與列支援修補 — 2026-11-22

本節依 [FROZEN contract](../../tasks/p4a-colab-contract-20261121.json)／[規格](../../tasks/spec-p4a-colab-20261121.md) 更新舊候選操作描述：固定64 fits＋Q5、record-only帳本，85°C STOP 保留；partial 可按驗證 pointer 接續。正式 completed-fit 新增每 fit 的有效 frozen recipe identity（長窗／LR 消融使用自身值）、連續 history、重算 BEST／bad、8 epochs 或有效早停，以及 pointer／metrics／預測設定一致性檢查；發布與續跑共用，舊不足證據的標記 fail closed。Family A/B 不得裁列；Family C 先取全比較系統／W60／C0+交集，再逐 seed 重算排名及平均，明報支援與排除比例。研究／環境契約 bytes 未變，本節不重標歷史 artifacts。

`job-run-mwric0jj-32d745`，detached base d82d020，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；CPU 合成 RED/GREEN與三項 in-memory mutation，無 GPU／網路／Colab／production 操作。Colab／Drive FUSE 現場仍 UNVERIFIED，不是 Human acceptance。[工程報告](../../reports/v7_p4a_colab_engineering.md)記錄實際測試與限制。

## P4a Drive 分次交付路徑候選 — 2026-11-22

`job-run-mwqscvk1-a3db0f` builder，base/HEAD `df78d6cb3bb706f6f4d58b94310c3d352aab8385`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依本輪 no-commit 授權保留 patch。沿用 [p4_bundle](../../V6/research_lab/p4_bundle.py) 的 deliver／notebook；新增 `--run-name`（01_speedtest）／`--drive-root`（MyDrive/MarketMamba）與兩行 README.txt，不建立平行流程或新 worktree。

第0格預填 `RUN_DIR=.../MarketMamba/p4a/01_speedtest`、`EXPERIMENT_DIR=RUN_DIR.parent`；runner／摘要／匯出都使用 `EXPERIMENT_DIR/budget`，speed 與 formal 跨交付共用同一本帳。ZIP 依 RUN_DIR → EXPERIMENT_DIR/inputs → /content 搜尋；殘留 /content SHA 不符明確 WARNING 並忽略。notebook 綁定實際 ZIP bytes/SHA，解壓前另核對 manifest `total_bytes`（含 aliases 邏輯大小）及實體成員大小，不混比壓縮與還原大小；不符 STOP 並提示上傳不完整。WHEEL_DIR 預設 MarketMamba/wheels，空／不存在才回退 Mamba_Wheels_T4_new，均無 wheel 才編譯寫回新目錄；部分 wheel／不相容仍 STOP。

SPEED_TEST 上限改1.0小時／15 CU；FORMAL 無預設，須 Human 明確填入。其餘0.5小時每fit/stage、10 CU/hour、25% reserve 保留；全新三臂 speed 計畫至少需2.5小時／25 CU加已花時間，因此小上限下計畫檢查會STOP，需Human依實際保守估計另行決策，不能放寬gate。既有舊根帳本不自動搬移／重置；若已有消耗，須保留原 EXPERIMENT_DIR 或先獨立盤點遷移。DRAFT 正式 gate、研究／production authority、runtime observed_at 與 lock 未變。

本輪僅本機 CPU 合成工程證據；沒有上傳 Drive、網路、GPU、Colab、資料取得或研究結果，現場行為仍 UNVERIFIED。此節取代前節的6小時／100 CU、/content優先與舊預設wheel路徑操作描述，保留歷史驗證紀錄。既有 fold resolved-root 邊界與壓縮比兩項 OPEN 限制未處理。Next：獨立審查 patch 與預算設定，核准真實輸入後重新 deliver；現場 Colab 驗證及 Human acceptance 另行處理。詳見 [指南](../../docs/operations/colab-p4.md)、[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a 兩檔 Colab 交付候選 — 2026-11-22

`job-run-mwqr9yiu-8c464e` builder，既有 detached HEAD/base `74795a4a56f25585000d26d281e305381de6230a`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，明確 no-commit 授權保留 patch。[產生器](../../V6/research_lab/p4_bundle.py) 的 `deliver` 一次輸出 `p4a-bundle.zip` 與 `p4a-colab.ipynb`，沿用既有 pack／executor／研究契約，沒有平行流程或新 worktree。ZIP 以內容 identity 去重跨 fold 的 X.npy，manifest 完整邏輯檔案清單加 aliases；安全解壓還原各 fold 的一般檔案並逐份驗 size/SHA。fold 輸入 symlink 只在打包端解析，解壓仍禁止 links，tar 相容與白名單／資源／cache gate 維持。

Notebook第0格集中預算與路徑，預設SPEED_TEST；找 /content 再找Drive、標準庫取內含bootstrap後完整驗證，不需手貼SHA或另傳bootstrap。先自檢環境，已符合契約則略過安裝；cp313 Drive wheel路徑 `MyDrive/Mamba_Wheels_T4_new`，cp312不符仍STOP。兩wheel均缺才使用官方sdist編譯備援（mamba C++20、MAX_JOBS=4）及Mamba2小型前後向冒煙。CUDA gate、DRAFT正式拒絕、既有resume／預算帳本與speed輸出隔離不變。

[Colab runtime契約](../../environments/p4a-colab-runtime.json)已是Human提供的OBSERVED Python3.13.16／torch2.14.1+cu130／CUDA13.0／T4／mamba2.3.2.post1／causal1.7.0；本節取代舊候選「等待填TBD版本」的next action，但不修改原觀測時間，不把Human smoke升格為專案正式GPU證據。本輪只有CPU合成測試／nbformat解析／三項in-memory反證，無網路、GPU、資料取得、正式Data、模型、production、scheduler或研究結果。Colab現場ZIP解壓、編譯儲存格、Drive掛載／持久寫入／下載仍 **UNVERIFIED**。Next：獨立審查patch；另行授權真實bundle打包及Colab現場驗證；FORMAL仍需凍結研究契約。細節與限制見[指南](../../docs/operations/colab-p4.md)、[工程報告](../../reports/v7_p4a_colab_engineering.md)。

## P4a Colab 工程候選 — 2026-11-21

新增[P4 bundle](../../V6/research_lab/p4_bundle.py)與[executor](../../V6/research_lab/p4_colab.py)，沿用既有`train_one`，新增可選observer與GRU；default P3配方／gate不變。P4 [機器契約](../../tasks/p4a-colab-contract-20261121.json)／[規格](../../tasks/spec-p4a-colab-20261121.md)均DRAFT，正式入口拒絕。Bundle包含source commit及dirty source逐檔identity、原始fold／P1分數manifest／原樣exact lock；排除rankgauss，Colab以既有ensure_rankgauss重建並驗tag／input-output hashes。SAFE extract只允許regular files，拒絕絕對／穿越／符號與hard links。保留fit BEST與epoch latest/previous/best、optimizer/scheduler/RNG；只支援completed-fit resume，partial保留forensic evidence，未知中斷扣至重啟並STOP，不自動升格orphan。

Drive ledger以wall-hours與Human保守費率估計CU雙上限，剩餘完整grid加至少25%餘裕；guard wait不扣active訓練帳，Colab85°C直接STOP而不等待（wait=0）。未知／缺帳本／cache tag／identity不符fail closed，尚無人工clear/rebuild介面。[指南](../../docs/operations/colab-p4.md)與[報告](../../reports/v7_p4a_colab_engineering.md)記錄CPU證據及尚未驗證的wheel／GPU／Drive限制。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，不取代E1或P3歷史authority。


P4a 審查修補（base `8cb61c4`，修補已保存於 `05cb840`／NOT_ACCEPTED）：speed 只保存計時與 telemetry，不保存 epoch state、IC、scores 或結論；正式 completed-fit 需 BEST／latest／previous pointer、weights/hash、scores/reference rows 與 metrics schema。Bundle 在寫入前限制成員及單檔／總量，拒絕 sparse／executable caches；trusted bootstrap 匯入 source 前清除 caches 並停寫 bytecode。規格全部待決選項仍 OPEN；2021–2025 已曝光，無新確認性 authority；ABI／GPU／Drive 仍 UNVERIFIED。

第二輪 bundle gate（base `05cb840`，no-commit patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED）：tarfile 之前逐一讀 physical header，PAX／GNU extended payload 上限64 KiB、拒絕連續 extended headers；gzip 單次請求最多1 MiB，各掃描總量同時受256 GiB＋32 MiB與 manifest 宣告總和＋4 MiB＋32 MiB限制（manifest 出現後立即收緊，末尾宣告則回溯核驗）。三次串流掃描不將 archive 全量放入 RAM 或暫存解壓檔，增加 CPU／I/O 成本，真實大型 bundle 未量測。Linux 每層 ancestor 使用 dir_fd、O_DIRECTORY／O_NOFOLLOW；輸出以 O_NOFOLLOW／O_EXCL 開啟。Source 只接受兩個既有 Python 子樹與指定文件；其他內容依明確 matrix／reference／lock／manifest 清單，拒絕 startup hooks／native libraries／非白名單路徑。當輪外部 SHA 必填策略已由下段 Human 操作修訂取代；內部 manifest 仍不能自證來源。白名單不能替代可信 source 審查，且不是任意惡意本機程序的 sandbox；Colab bootstrap 與目的地 parent 的本機權限仍需可信。

### Colab 操作修訂（job-run-mwqi30jf-aa0d7f）

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。外部 archive／bootstrap SHA 不再必填，extract 自動記錄 archive SHA，提供時嚴格核對；內部逐檔 size/SHA 與安全解壓 gate 全保留。Drive wheel-dir 模式只以 no-deps／no-index 安裝 mamba_ssm、causal_conv1d，列檔名／標籤／SHA並核對 Python／平台；不碰 torch／其餘套件。白名單僅增指定 `source/environments/p4a-colab-runtime.json`，沒有放寬路徑規則。

[Colab 環境契約](../../environments/p4a-colab-runtime.json) 是新的獨立 candidate artifact，由 bundle manifest 綁定逐檔 identity；原樣本機 P3 lock 無法承載不同且尚未知的 Colab runtime，因此沿用 environments 容器新增此檔，未另造環境 tree。初始 TBD，未觀測版本不能填成已知；正式模式要求 OBSERVED 且 runtime／列出套件完全相符。speed-test 允許 mismatch 並逐項記錄與標記結果不可與正式環境比較，CUDA 一律必須可用。P4a 三臂共用同一 Colab identity；P3 跨環境比較須揭露差異。依賴 Human pip list／runtime 自檢及後續現場驗證，沒有本輪 Colab／研究結果。

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: training contract, checkpoints, resume behavior, Colab packaging
- Update triggers: trainer/checkpoint contract, runtime package, or formal run change

## Purpose

Define reproducible training and recovery behavior for corrected E5 and its unchanged-contract E1 refresh.

## Current Boundary

The frozen replication training contract uses 20 epochs, patience 5, minimum 5 epochs, learning rate `1e-4`, weight decay `0.01`, warmup ratio `0.15`, minimum learning-rate factor `1e-4`, gradient clipping 1, fp32, AdamW, and linear-warmup/cosine decay. Selection records IC5 and IC10 at the same best-IC5 checkpoint.

## Authority

The incumbent/experiment contracts and checkpoint loader on `feature/v7-e1-rolling-origin-refresh@044eef7` are authoritative. The byte-identical exact lock is also preserved at `environments/v7-colab-2026.04/requirements.lock.txt` in development main; its 2026-09-12 local verification is not formal E1 GPU-run evidence. Colab notebooks are launch UX, not the training contract.

## Inputs

- Persisted identity-matched matrix and split contract.
- Model/feature/label contracts.
- Explicit seed and device/precision choice.

## Outputs

- Durable generation checkpoints (`latest`, `previous`, `best`).
- State identity, epoch/batch/step/phase/terminal status.
- Per-seed metrics, logs, manifests, and completion markers.

## Current Implementation

Resume must validate contract and matrix identity. Orphan recovery may recover a valid latest generation but must not redefine `best`. Training artifacts must be written durably outside an ephemeral Colab runtime. An unchanged verified matrix is reused across runtimes.

Corrected replication formal evidence currently covers seeds 17 and 29. Seed 43 has a surviving loadable non-terminal state (epoch 15, step 39915 in later forensics); it is not a completed seed. E1 seeds 17 and 29 are complete; both selected epoch 1 and stopped at epoch 6 by early stopping. These are training-dynamics observations, not an accepted new recipe. Epoch 2–4 checkpoints were not retained, and no final-support per-epoch replay has been established.

## Evidence Classes

- Terminal checkpoint plus completion artifact: completed seed.
- Loadable non-terminal checkpoint: resumable or forensic state, subject to identity checks.
- Log alone: supporting evidence, not completion.
- External Drive contents: `UNKNOWN` until imported and validated.

## Current vs Legacy / Research

Older notebooks that assume one uninterrupted runtime or reconstruct matrices every session are historical references. The current requirement is durable checkpoints, idempotent stage reuse, identity validation, and explicit recovery.

## Invariants

- Save enough state to continue optimizer/scheduler/model progress safely.
- Validate checkpoint format, contract, matrix, and stage before resume.
- Never count a non-terminal seed as complete.
- Preserve logs, metrics, hashes, environment metadata, and reason for termination.
- Release GPU only after durable artifacts are flushed and verified.

## Known Limitations

2026-10-21 recovery revalidated best/previous/latest file hashes for both E1
seeds. Pointer epochs are 1/5/6; embedded identities remain unvalidated because
only static ZIP/pickle-opcode inspection was performed. Existing unrestricted
loader was not executed. Frozen matrix is not found locally; no replay/training.

Runtime forensics are inconclusive: matrix wall time, epoch timing, GPU utilization, and Drive I/O were not fully captured. Observed completed-seed runtimes were about 6.94 and 7.82 hours on an A100 80GB fp32, but these are run observations, not forecasts.

## Do Not Use / Do Not Mix

- Do not rebuild a matrix solely after a runtime disconnect.
- Do not use a mismatched checkpoint to “continue.”
- Do not promote partial seed 43; E1 completion now rests on versioned result artifacts, not packaging.
- Do not infer final-support epoch degradation from logged validation curves alone.

## Important Paths

- `feature/v7-e1-rolling-origin-refresh@044eef7:V6/experimental/v7_corrected_e5_train.py`
- `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json`
- Known working Colab reference: `deliveries/V7-Confirmation-20260914/V7-三種子確認與續跑/`

## Related Domains

[Data](Data.md), [Features](Features.md), [Models](Models.md), [Labels](Labels.md), [Operations](Operations.md)

## Update Triggers

Update after checkpoint-format changes, new formal run evidence, Colab bundle changes, or added runtime telemetry.

**2026-09-22 source integration:** Corrected E5/E1 reusable research modules and selected frozen contract copies are now in development main; see `research/v7/source-adoption-register-20260922.json`. Run results, bundles, postrun diagnostics and operational/deployment authority remain separate. No model training or production promotion occurred in Phase 3B-2.

## CPU recovery engineering — 2026-10-21

`development/v7-autonomous-charter` (UNMERGED / NOT_DEPLOYED) validates local
checkpoint entry metadata before loading. Malformed/unreadable latest entries
fall back to valid previous; save/orphan recovery retain that validated generation,
not a corrupt pointer. SIGKILL before pointer, restart, identity/seed mismatch and
second latest loss fixtures pass; best is not promoted by recovery.
[Window evidence](../../research/v7/autonomous-window-01-evidence.json): two repair
rounds, final19 tests, independent19 PASS. Existing trusted-local unrestricted
Torch loader remains unchanged: no historical payload safety/replay certification,
no arbitrary pointer corruption or hostile-filesystem guarantee. E1 stays BLOCKED.

## Charter integration — 2026-10-21

Window01 source/governance is now LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED
at exactf7af348 (supersedes its branch-local status only). Pre/post CPU regression
and authority checks are in [integration evidence](../../research/v7/charter-integration-20261021.json).
A separate Human-authorized cross-domain CPU window may proceed in isolation;
research contracts, formal data, production and StrictSTOP boundaries are unchanged.

## Window02 metadata guard — branch-local

`development/v7-cross-domain-window-02` (UNMERGED/NOT_DEPLOYED) pins the existing
E1 frozen contract and rejects matrix metadata drift, including self-rehashed
changes. This does not verify array bytes or certify training readiness. Existing
recovery evaluator byte/provenance work remains branch-qualified. See
[window02 evidence](../../research/v7/cross-domain-window02-evidence.json) and
[unique backlog](../../research/v7/research-program.md#cross-domain-window-02).

## Window03 fairness and readiness preparation — 2026-10-21

Completed source-level E5/E1 split/selection/final-evaluation review and PROPOSED
comparison designs, without changing existing result classifications or policy.
Existing validators/configs match historical E1 contract/config identities;7 copied
result hashes and2 seed best-pointer metadata chains checked. No matrix/weight
payload validation, training or Colab launch. NOT_RUN_READY; original matrix BLOCKED.
[Report](../../reports/v7_data_evaluation_window03.md),
[unique completed queue](../../research/v7/research-program.md#data-evaluation-window-03).
Main9956446 unchanged; branch `development/v7-data-evaluation-preparation` is
UNMERGED/NOT_DEPLOYED. New bounded execution and formal research decisions need Human.

## Window04 reusable metadata audit

`V6.experimental.v7_e1_readiness` joins existing frozen contract/config and pinned
completed-result metadata checks. Output VERIFIED_METADATA_ONLY never authorizes
execution; matrix payload NOT_VERIFIED, checkpoint payload NOT_LOADED. It does not
re-evaluate predictions or replace recovery evaluator provenance gates. See
[window evidence](../../research/v7/fair-evaluation-preparation-evidence.json).

## Window04 CPU logging reliability

Branch-local W04B-LOG changes `v7_e1_colab_runtime.run_logged` so background log
write/decode failures reach the caller with the original cause and log path,
stop a running child through the existing cleanup, and fail even if the child
already exited. Existing success/nonzero/heartbeat behavior remains covered.
Six CPU fixture tests and independent review verify this scope; no live Colab,
Drive storage durability or research readiness certification. UNMERGED / NOT_DEPLOYED.
See [bounded evidence](../../research/v7/window04-cpu-continuation-evidence.json).

W04B-LOG local integration update (2026-10-21): fixed97fb4c9 now included in
WSL development main; earlier UNMERGED label above is historical. NOT_DEPLOYED;
fixture scope and no live Colab/Drive certification unchanged. See
[integration closure](../../research/v7/window04-log-integration-closure.json).

## Window28 limited clean Ridge / causal check

Human限定批准W27安全Mask下的2021 Ridge fresh fits各horizon一次（2/2已用），原alpha1與兩Features不變；Scaling只取clean TRAIN。
2022/2023同W27 support為已曝光Development；18個有限future OHLCV數值擾動不改Features/固定模型Predictions。
此pass不驗未來Label eligibility gate為as-of，也不認證完整公司行動/vintage。
[Run](../../research/v7/window28-run.json)、[Report](../../reports/v7_window28_clean_refit.md)。

## Window30 fixed real LightGBM development baseline

Human明確批准一組小型real-data recipe，60e7ca7在Fit前固定；W11環境4.6.0不變，2fits/64trees、TRAIN-only bins，無validation earlystop或search。
60726 feature-only predictions /same W29support與獨立Replay通過；5d正、10d負，4組Mean均低於舊Ridge，但不宣告winner/研究准入。
Expanding各Origin仍需另凍結model-ready/maturity/purge/refit/selection與資源；本次不自動批准。
[Evidence](../../research/v7/window30-evidence.json)、[Report](../../reports/v7_window30_fixed_lightgbm.md)。

## W37 readonly adoption

W34 model JSON/scaler/recipe/feature identity is available through `V6/experimental/v7_shadow_models.py`; only readonly scoring port, no fitting. See [integration](../../research/v7/window37-integration.json). Original training evidence remains W34@82cd49c; no new model or research admission.


## P4b 本機專用模組的 bundle 邊界候選 — 2026-11-24

`builder_bundle_exclude`／base 8dc5cc9，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。本機 `p4b_blend.py` 不屬 Colab runtime；[bundle](../../V6/research_lab/p4_bundle.py) 的精確固定摘要清單與 [source compatibility](../../V6/research_lab/p4_source_compat.py) 共用來源集合，宣告列明排除政策；未宣告新來源、必要檔排除、Colab import 依賴或手動注入均拒絕。兩次完整合成 bundle 只改本機模組 bytes，所有成員含宣告／manifest bytes/hash 必須一致；隔離 import 與至少三項 in-memory mutation 依 [測試](../../V6/research_lab/tests/test_p4_source_compat.py) 驗證。明細／最終實測見 [P4b報告](../../reports/v7_p4b_engineering.md)。

from=4e56e9d 不變，to 為 generation-time HEAD／working hashes；matrix 只沿用版本化 identity、未重新認證資料。36 個 W60 完成 fit 全數 resume_fit／不改歷史 bytes／不重訓、64-fit queue SHA、舊 notebook 與 P3 凍結檔均維持。只限本機 CPU 工程，不取代真實官方 mixer／Colab／Drive／現場 artifacts／獨立複審或 Human acceptance。沒有更新外部 observed_at；沒有原研究結論或 production authority supersession。

本輪最終完整CPU回歸（2026-11-25）1204 passed／7 skipped／4 warnings／0 failed（608.24s，exit0）；最終focused 12 passed，3項mutation killed，236成員bytes/hash不變。7 skip為既有套件／nbformat／CUDA環境限制，明細見報告；不等於現場或研究驗收。
