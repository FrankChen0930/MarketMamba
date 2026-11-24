# P4d：對等調參預算與資料／容量／種子曲線

**DRAFT — 待 Human 凍結；不是執行授權。E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 文件識別 `p4d-equal-tuning-20261125`（指定的規格日期；實際撰寫 2026-11-24）。全部歷史資料已曝光，所有比較、CI、p 值與決策都是描述性、形成假設用途。2021–2023 只沿用舊契約「確認性折」的來源名稱，本案不具確認性。完成草案不等於 Human acceptance。

## 1. 目的、建立前盤點與來源

回答有限問題：(a) 在本次新增的相同候選評估次數下，GRU／Mamba2 相對 C0+ 的歷史差距如何；(b) 固定P4a配方，延長最近訓練歷史時，相對差距的曲線如何；(c) 固定 P4a 配方時，更大容量或更多種子能否改善排序及資源效益。不能推論架構普遍優劣，也不能把新增預算相等說成歷來研究投入相等。

先讀 AGENTS.md、CLAUDE.md、Current State／Authority Map、ml-leakage-audit、P4a／P4b、P1／P3 報告、Training／Research domain、GOVERNANCE、Human Colab 需求，以及下列程式、machine contract 與測試。既有 `tasks/` 承載登記文件，`V6/research_lab/` 承載研究工具，`reports/` 承載結果；P4a 固定64 fits／family A/B/C 不包含調參，P4b 是固定混合，所以不能改寫舊契約來容納本案。沿用既有結構新增本文件身分，不建目錄、程式、worktree、輸出樹或替代 pipeline；未來產物才在另案批准後使用 `.artifacts/p4d/<contract_hash>/<run_identity>/`，不得覆寫歷史產物。

本輪延伸工作 `job-run-mwut6qrj-7d9826/spec_writer_p4d_modules`，沿用指派 detached HEAD/base `ffe98b4cb222aaf8011dbdf9f08e38fba03dd03a`，初始 clean，已盤點 worktrees；未新增工作樹。WSL Linux 6.18.58.1、Python3.12.3（`/usr/bin/python3`；`python` 不存在）、`MARKETMAMBA_DATA_ROOT` unset。Data 是普通目錄，realpath 為本 worktree 的 `/Data`，未驗證為正式資料映射，也未讀其內容。本輪只改文件，唯讀主 checkout `.artifacts/p4/p4a-fit-durations-min.json`；不讀 MarketMamba-blind、不動正式 Data／V6/models、GPU／Colab／Drive、scheduler，不 commit／push／merge。Human 已同意設計 A／B 並想嘗試，尚不等於本 DRAFT、計算額度或執行契約已凍結。

### 證據索引與目前配方

- [P4a 規格](spec-p4a-colab-20261121.md)及 [FROZEN machine contract](p4a-colab-contract-20261121.json)：W60、d_model64、2層、expand2、headdim8、d_state16；AdamW lr3e-4、weight_decay .01、8 epochs×400 steps、patience3、8日×256股；rank-gauss、15% warm-up＋cosine、clip1、BEST validation IC5。不能以 `seq_runner.DEFAULT_CFG` 的10 epochs／16日覆蓋 frozen recipe。
- [seq_runner.py](../V6/research_lab/seq_runner.py) 的 `split_rows/train_one/session_loss`、[seq_models.py](../V6/research_lab/seq_models.py)、[p4_colab.py](../V6/research_lab/p4_colab.py) 的 `train_fit`：純 GRU／temporal Mamba2 未啟用 dropout；既有 `.1` 只在 crossstock 路徑生效；y5:y10 固定1:.5、除以1.5，並非已接線的可調 loss 權重。Mamba heads 是 expand*d_model/headdim，非 attention heads；GRU 沒有 heads；本案 S-Mamba 風格指 crossstock，不是 `samba` attention 混合模型。
- [P3 報告](../reports/v7_p3_portfolio_evaluation.md)：21/24、兩種子、受14h上限中斷，不能把缺 fit 補稱完成。[P4b 草案](spec-p4b-blend-20261124.md)維持獨立身分。本輪不修舊報告／舊規格中過時的 next action。
- Human 本次背景：P4a 六折三神經臂均落後、Holm p>.14，W120／250 未改善；Q4 僅2022／seed17，GRU lr1e-4 超額16.8% 對3e-4 的4.9%，Mamba2 8.5% 對5.6%，1e-3較差。這是 **Human supplied exposed evidence**，不是本輪重算或 ledger 驗證；不能據此直接選1e-4、排除1e-3或挑2022調參。

### P1 的 C0+ 實際做了什麼、做了幾次

[P1 報告](../reports/v7_p1_portfolio_evaluation.md)、[rolling_scores.collect](../V6/research_lab/rolling_scores.py)、[c0plus.run/build_features](../V6/research_lab/c0plus.py)與主 checkout 的 `.artifacts/p1/full-scores-ace2eef/manifest.json` 相互對照：

| 項目 | 能核實的事實與界線 |
|---|---|
| LightGBM 配方 | n_estimators200、num_leaves31、lr.05、min_child_samples200、subsample.8／freq1、colsample_bytree.8、reg_lambda10、n_jobs4。P1 呼叫未傳 lgbm override，沒有在 P1 執行一套已登記超參搜尋 |
| C0+ 特徵 | 48原欄＋16 CORE欄×(5個lag1/2/5/10/20＋3個rolling5/20/60)＋3個MARKET欄×mean/std/pos＝185欄。產業／營收 extension 不是 P1 預設。特徵設計與過去研究選擇仍是累積調整，不等於未調過 |
| P1 正式工作數 | 6折×3種子17/29/43＝18 jobs，各 fit_rows500000；每 job 同時擬合 day-t LightGBM 與 C0+，即36次樹模型擬合，**不是18或36個超參 trial**。500k 是訓練列抽樣上限，不是完整歷史全列 |
| manifest identity | source `ace2eef0c580d3484a4c6299de16b1c3d5420898`；manifest SHA256 `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`。前版核 metadata，本輪沿用該歷史索引，未逐份重驗 parquet |
| 歷史調整紀錄 | [匯入 roadmap](roadmap-20261115.md) §11.6、12.10–12.12、12.19、12.36 等有185欄設計、193欄產業 extension、移除市場彙總消融、總經lag／預設值診斷及滾動比較；唯讀舊 repo `.local-data/` 有 c0plus_e5_y5、ablation_market、ind／rev daily 等檔名。這是 historical navigation，不把舊repo結果混作本案 evidence；沒有核到完整 trial ledger／候選總表，歷來超參或特徵選擇總次數 **UNKNOWN**，不得稱 C0+ 已做某個 N 次搜尋 |
| 重跑／評估 | `.artifacts/p1` 另有 full-scores-d2390b5、評估網格與診斷；資料夾數、432個portfolio格或不同seed不等於調參次數，也不證明已採用新配方 |

因此公平主張限於「P4d 新增 N 次候選預算相等」；歷史先驗、特徵工程、搜尋維度與驗證過擬合程度不相等，無法追溯抵銷。

## 2. 調參區間：比較替代方案與採用邊界

| 方案 | 優點 | 洩漏／代價 | 建議 |
|---|---|---|---|
| 各2021–2026折的 outer train 內最近126 sessions | 隨市場更新、易沿用 split_rows | 例如2023折的 validation 可能落於2022測試資料；雖符合一般逐期walk-forward的當時可知性，仍違反本案「任何最終測試資料不得選超參」的較嚴格要求；6倍搜尋且每折配方不同 | 不採；如 Human 日後改用，須另案明示重疊與新預算，不能原案替換 |
| 單一早於2021的 tuning origin | 搜尋及早停資料與六折測試日期完全不交；成本小、單一配方易解釋 | 只有一種市場狀態、較舊，且先前研究已曝光，仍有meta-selection bias | **採用：2020末的126-session validation；全部標籤在2021前成熟** |

不是只檢查樣本 feature date：以 frozen 原始交易 session 索引，令 T 為2021折第一個 evaluation session，候選 validation 最後一個 session `v_end < T-20`，取至此最近126個完整合格 sessions。fit 的最後 session `< v_start-20`。每列 y5／y10 label_end 均須嚴格早於下一區間首日且有限；以完整 outer split 驗 maturity，不能先 finite 篩掉污染列再通過。原碼 `split_rows` 的邊界可能含≤與19個中間session，P4d adapter 必須顯式測本規格的嚴格20-session不等式，不能誤稱逐位元沿用舊切分。日曆日期與實際126日範圍目前 **UNKNOWN**；實作先輸出 split manifest、鍵集合hash、min/max label_end、fit/val數與session數，驗證通過後、任何試驗前凍結。不能為了湊126日放寬成熟度。

Tuning 的訓練起點沿既有 matrix 最早可得、W60完整連續歷史；使用全部合格 fit rows，各臂共同y5/y10有限、相同列身分與日期、min_names64，C0+仍只學y5。不用500k隨機抽列來代表100%歷史，否則縮放會變成抽樣濃度實驗。這與P1 cap不同，必須重訓 P4d C0+，不沿用舊P1分數當主對照；完整列記憶體／耗時未知，是實作前CPU排練的STOP條件，不能靜默恢復cap。舊P1／P4a僅歷史參考。

選好一次全域配方後，六折可以按各自 as-of 使用此前已成熟歷史（包括先前折曾評估過的日期）**擬合固定權重**；不再拿其IC、組合報酬或early stopping選擇epoch、seed、配方。所有超參選擇、BEST epoch與迭代數只能來自2021前的tuning validation。測試label不進該折fit；每折outer train最後session `< eval_first-20`、完整outer split及label_end都驗。這區分「歷史label用於後續固定演算法擬合」與「用測試結果選規則」。

### 預處理隔離契約（D1；待凍結）

**預設採versioned明確macro schema，不從整矩陣偵測。** 綁定48欄order SHA `7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696`；神經rank-gauss恆歸零的零起算欄位為 **36…47**：`TWII_Return, SPX_Return, VIX, TNX, Gold_Return, Oil_Return, USD_TWD, Futures_OI_Foreign, Options_PC_Ratio, Fear_Greed, Business_Signal, FED_Rate`。其餘0…35只用當日已知全截面rank-gauss（average ties、norm.ppf((rank-.5)/N)、非有限值處理依凍結實作）；不隨後續日期、fraction、trial、fold或觀測常數性改mask。此為神經轉換schema，C0+仍用185欄causal特徵，不把macro一併刪除。更動有序欄／schema須新identity，不可沿用快取。

替代（不與預設並跑選優）：只由合格tuning **fit** 日期的完整截面估mask（std<1e-9、每個fit session均滿足才列入），不得讀validation／2021–2026；凍結估計support鍵／日期及輸出mask後，全trial、A/B/D、outer只重用同mask。不能按outer或fraction重新估mask作調參輸入；這會改問題且預設不採。其他學得的scaler／imputer／binning一律僅fit合格fit rows，validation僅transform；context halo只供因果features，不供估計狀態。

rank-gauss／macro mask **cache identity** 必含source／implementation hash、matrix原始X及有序features identity、macro schema完整內容hash、估計方法（明確schema標not_fitted）、tuning fit support／halo／calendar hashes、各session輸入截面及轉換輸出hash、dtype／非有限值／ties政策。不得只用X的size／mtime tag，也不能把只改未來X導致全檔hash變化誤認歷史features變化；可以cache miss重算，但歷史結果必須不變。outer重新擬合的學得狀態另綁該折fit support，禁止反向污染tuning cache。未來只更換cache identity並不證明隔離，必須驗內容。

新增強制驗收：在固定tuning日期／support、seeds及候選表下，任意修改**調參區間之後**的features、labels、日期／股票列、缺值／常數性（包括2021–2026）或追加未來列，tuning features bytes、macro mask、每trial IC／ranking及selected recipe／BEST更新數須完全相同；測cache hit、強制rebuild及獨立新程序。替換為整矩陣抽60日的舊路徑必使專用合成反例失敗（過去120列不變，後段mutation使mask[0]→[]、過去[0,0]→[.67448975,-.67448975]）。本輪只規定驗收、未修程式，舊`ensure_rankgauss`不符合本隔離契約，不可直接作P4d准入。

### 唯讀實證附註：既有各折macro mask（2026-11-24；D1）

本輪離線CPU唯讀，沒有呼叫會建cache的`ensure_rankgauss`／`SeqData`，沒有修改矩陣、cache、程式或既有結果。以AST抽出HEAD `3ed68762a51aa1f62223aa10c354f3a1eae0686f` 的原`macro_columns`函式（整檔SHA `1f4ba6fa9d97a93a5f59413e26b68252c926f071e51dc5465098438af69b088e`）在NumPy namespace執行；X及metadata以`np.load(..., mmap_mode='r')`讀取。兩來源為 `/home/frank/projects/p4-bundles/prepared/folds/fold_2021…fold_2026/` 與 `/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/matrices/rolling/fold_2021…fold_2026/`，依symlink取得實際X分別在prepared/shared與pilot09/matrices/E5_fwd_lagus。每份X實算 **1,425,394,112 bytes／SHA `9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363`**，shape 7,423,927×48；不同realpath不混作不同實驗。

兩來源dates SHA皆 `b0632562242f107855770742762f15fa1f82b6d8e012dce8e4cfd1d68c641016`，end_indices SHA皆 `dd6c6e8e8e4c44ad28f42e9dd266f87c3baebbdc0f986e9aa6865789c9d3516f`；完整5334 sessions（2005-01-03～2026-09-11），2021前3952 sessions（至2020-12-31）。各fold用原split==1的**完整日期截面**（每session end row判斷，非500k抽樣列、非只取finite label）重建訓練日期池。

比較5種mask：①原整矩陣seed0無放回抽60 sessions；②限制該fold train日期、以相同seed0抽60；③限制2021前日期再抽60；④該fold train全部sessions逐日AND；⑤2021前全部sessions逐日AND。②／③是先限制日期池再抽樣，不是把整矩陣抽樣結果截去未來。④／⑤沿原float64 std<1e-9、單日>1列才更新的規則逐日檢查，不改常數容差。所有12個fold／來源組合、5種判定均為 **[36,37,38,39,40,41,42,43,44,45,46,47]**；差異欄位為空。完整欄名同上節schema。下表兩來源的split bytes／hash相同（每檔7,424,055 bytes），train起日皆2013-01-02。

| fold | train sessions | train末日 | splits.npy SHA-256 | 5種mask／兩來源 |
|---|---:|---|---|---|
| 2021 | 1942 | 2020-12-03 | `6c70097cb2f8bbbe27d16b5bc86cacbab506d3fabddc989a7eb28a3a290979d6` | 全相同36…47 |
| 2022 | 2186 | 2021-12-02 | `f55a74081ba43359edff6231aab488b0df9944c607de8489746d399e06e4bf89` | 全相同36…47 |
| 2023 | 2432 | 2022-12-02 | `20a8ad5b9d963130919eafa6c9b425c14bd5cd44f60c4f8c7964d169dc199eac` | 全相同36…47 |
| 2024 | 2671 | 2023-12-01 | `8e6c57aaa8a8611c31440c3e673693d40f879abd708f3615c1f8e7cb9eeec98c` | 全相同36…47 |
| 2025 | 2913 | 2024-12-03 | `281442ccb444f4a60d8b94a1a53ef5ae96a4a9a3d9b5b4662fd93ac91890727d` | 全相同36…47 |
| 2026 | 3156 | 2025-12-02 | `dfd3fa021b5d29447b67c65ecbc602e990cf92778828d151123c714fce57fa6b` | 全相同36…47 |

身分連結：唯讀核對主checkout `.artifacts/p1/full-scores-ace2eef/manifest.json`（既有SHA `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`）與 `.artifacts/p3/scores-c323e79/run_identity.json`（本輪SHA `3f089e92ce7107affa92541ee084f523408071daa2c498a5cdbffe8e1a4ed06c`）inputs，P1／P3各六折X與splits雜湊12/12項均符合上表。P4a本機prepared X符合`p4_bundle.DATASET_X`，prepared.json綁定dataset manifest SHA `bc6401095260e7d4153060e3106fff8dbae88cfb6df3f349b808b7dd762c7d9f`；本輪沒有逐fit或雲端重新驗證，shared/manifest.json內的舊X hash非實際X，不能拿它替代實算或DATASET identity。

**結論：已完成結果不受此問題影響（僅限該mask，限上述identity的P1／P3及P4a本機準備矩陣）**。P1 C0+本身不走神經rank-gauss，這裡核其同源矩陣，不代表另做P1轉換；P3仍21/24、P4a逐fit來源／cache／CUDA未重驗。這不是證明不存在其他洩漏、重算績效或Human acceptance，不能撤銷P4d隔離修補需求；尚不存在的P4d tuning split、逐fraction合格fit／halo及future-mutation試驗仍待實作。可重現的判定方法／輸入hash／數值留在本附註；`/tmp`診斷JSON只是TEMPORARY、非durable authority，不作後續依賴。

## 3. 對等預算、搜尋與選擇

**主要定義：每臂相同 N 個候選配置，每配置同兩個種子17／29，單一 tuning origin。** N 包含基準配置1個，其餘 N−1 是無放回簡單隨機搜尋；搜尋 RNG=20261125，臂名＋trial_id 派生獨立可重建stream，抽樣器版本與先生成的完整候選表先hash／凍結。重複候選重抽但不算新trial；合法組合在launch前驗證。基準神經為P4a配方（dropout0），基準C0+為P1預設；每個種子獨立初始化，不warm-start不同trial。每臂同N、同validation／seeds，不保證相同有效模型容量或過去研究投入。

| 等價定義 | 好處 | 代價／處理 |
|---|---|---|
| 相同試驗數（主要） | 固定候選機會與可重建順序，不因硬體快慢改變search length；符合record-only帳本 | 每trial耗時與搜尋維度不同，神經總費用較高；報告每臂總秒數、GPU/CPU、候選數與學習曲線 |
| 相同牆鐘時間（替代，未執行） | 量測特定硬體下資源效益 | CPU與T4秒數不等價；編譯、I/O、斷線會改候選數；偏好便宜模型。若另案使用，須先凍結硬體、active fit含validation計時、編譯／共享預處理／離線分列、最後完整trial規則與失敗處理；不能事後只挑較有利定義 |

採 random 而非 TPE：小N、離散空間、高噪音，用預先表即可續跑；TPE依序適應也更容易受平行／中斷順序影響。**不用Optuna、不安裝套件**。若日後改Optuna，先經Human批准，僅可裝 pilot09 CPU環境，禁止改GPU環境；這是另案contract，不在本次授權。

### 搜尋空間（提案，凍結前一次決定）

所有離散列表等機率；log-uniform 按自然log等密度。window固定60、不把已曝光W120/250再混入；feature set、label定義、ListNet=0、normalize、clip、warm-up、optimizer、steps400固定。表內單點是明確固定項，不假裝已搜尋。

| 神經項目 | GRU／Mamba2空間與有效性 |
|---|---|
| lr | log-uniform [1e-5,1e-3]，包含但不偏選Q4已知1e-4；baseline3e-4 |
| weight decay | {0,.001,.01,.1}，AdamW；參數衰減分組policy全trial一致、manifest記錄，不順手改Mamba初始化 |
| dropout | {0,.1,.2}；新增各臂相同位置的 embedding後、進temporal前dropout，eval關閉；不把GRU recurrent dropout與Mamba內部norm說成等價；baseline0。目前未接線，必須測非零確實改training forward／梯度、eval確定性及0的舊路徑一致 |
| d_model／層數 | {32,64,128} × {1,2,3}，實際參數量逐trial記錄；相同寬度不是相同容量 |
| head／state | Mamba expand=2、headdim∈{8,16}、d_state∈{16,32}；只枚舉 `(2*d_model)%headdim==0` 且nheads=(2*d_model)/headdim為8倍數的組合。GRU無此項；不搜尋attention head |
| epochs／patience | max_epochs∈{8,16}，patience=3（固定），steps_per_epoch400；BEST mean daily IC5；早停可用，不能事後加epochs。max_epochs不同會改cosine週期，完整記錄 |
| batch | dates_per_batch∈{4,8}，stocks_per_date256固定；報optimizer updates及sample draws，不能以相同epoch宣稱同量unique資料 |
| label weights | y5:y10∈{1:0,1:.5,1:1}，loss除以權重總和；仍全臂以共同finite y5/y10列配對，不能1:0時暗增fit列。現碼硬編1:.5，須opt-in參數化 |

| C0+ LightGBM項目 | 搜尋空間 |
|---|---|
| learning_rate | log-uniform [.01,.15] |
| num_leaves | {15,31,63,127} |
| min_data_in_leaf | {100,200,500,1000}，對應sklearn min_child_samples |
| feature_fraction／bagging_fraction | 各{.6,.8,1.0}，對應colsample_bytree／subsample；bagging_freq=1固定 |
| 正則化 | reg_alpha∈{0,1,10}、reg_lambda∈{0,1,10,100} |
| 迭代 | max_estimators∈{200,500,1000}，每25輪評估validation IC5，patience=75輪；baseline200。保留BEST iteration，非train loss／RMSE早停；只fit C0+，不把額外day-t fit算候選 |
| 固定 | 185欄、y5 rank target、n_jobs4、CPU；不調feature admission／universe／portfolio參數。data binning/scaler等學得狀態僅fit rows |

選擇準則：每種子在固定validation共同列上逐日Spearman IC5（每日至少10股、非常數，否則STOP；報每日有效數），等日權重平均；trial分數為兩種子mean IC5的平均。神經epoch／樹iteration各依自己同一驗證訊號取BEST（同分取最早）；trial分數同至1e-8時取較小trial_id。固定val全列，**不沿用隨機80k validation cap**；validation缺日、NaN分數或任一seed失敗使trial無效。不得以portfolio Sharpe／收益再選配方：126日僅約6次20日再平衡、估計極不穩且成本層overfit風險高；組合指標僅最終外層描述。這也承認IC選出的配方未必最適合淨報酬。

允許上述同trial內早停；**不允許跨trial剪枝、ASHA／median pruner或看測試後延長訓練**。OOM／資料錯／非有限等必記FAILED並STOP該階段；修工程bug另記source lineage，原候選原種子重新執行，保留失敗成本，不補抽較便宜trial。只有所有臂N個合法配置×2seed完成，才標equal_budget_complete；partial不能用較小子集宣稱公平完成。

## 4. 分階段預算與可重現記錄

建議N=12；候選N=6／12／24。工程rehearsal先用synthetic與1-step真實路徑驗形狀，不產生可選的研究結果。研究pilot前4候選／臂（含baseline）計入N，**pilot確實看過validation，是調參的一部分**；可看收斂與seed差異，但不能改空間或依表現挑N。凍結時就選N=12，pilot後只依可行性（記憶體／時間／Human資源）決定繼續原N或停止並完整報partial；如要在pilot後改N／空間，原案封存，另登記DRAFT並揭露曝光與額外累積試驗數。不能在pilot開最終折評估。若Human只批准N=6，就獨立凍結N=6，不事後升級「直到贏」。

### 成本 evidence 與外推（全部估算皆非報價、非本輪訓練）

P1上述manifest的18 jobs各122.35–165.76秒、中位數139.51秒，總執行2572.82秒＝42.88分鐘；這含兩棵模型、特徵／預測與輸出，不是單棵C0+純fit秒數。原環境為pilot09 CPU venv Python3.12.3、LightGBM4.7.0。P4d改全列、最高1000樹與較大leaf，不可沿用「每trial兩分鐘」作保證；初估 **每seed CPU C0+ 10–60分鐘**（規劃假設，未量測全列），1候選兩seed為20–120分鐘。CPU峰值RAM UNKNOWN，先測再由Human定預算；本機CPU可行才執行，不搬CPU工作上Colab。

### P4a 原始 ledger 證據：已取得摘要，不再列為未取得

Human 提供的 `p4a-results (3).zip` 已有 ledger／metrics 耗時整理，指定摘要為主 checkout `.artifacts/p4/p4a-fit-durations-min.json`，本輪實讀 SHA256 `acdde9a7100681abc852c01375984fc26a192584c04e8cf6a250573a72412d03`（975 bytes）。摘要共12個 arm/window/Q 群組、64個 fit 記錄：Q1+Q2共36、Q3共24、Q4共4；Q5不是額外訓練。這與 frozen queue 的計數一致，不能僅由計數證明逐 fit 身分完整。ZIP 內容在本案僅保留「原始 ledger／metrics 的每 fit 耗時摘要」說明，不複製 ZIP、模型、大型 predictions，也不聲稱本輪已打開 ZIP 或重新驗證 hash chain／completed-fit。archive SHA、run／matrix／環境逐 fit identity、p50／p90、active／validation／I/O／離線拆帳目前摘要未提供，仍 UNKNOWN。證據級別為 **Human 提供的 ledger／metrics 派生摘要，本輪核對摘要 bytes 與算術**，不是 P4a 正式結果重新驗收。舊文「原始 ledger 未取得」僅描述前次工作，不再作當前 blocker。

| 臂／窗 | 來源與 n | T4 平均 min/fit | 觀測 min–max min/fit |
|---|---|---:|---:|
| GRU W60 baseline | Q1 6＋Q2 6 | 9.05（約9） | 6.7–10.5 |
| Mamba2 W60 baseline | Q1 6＋Q2 6 | 27.50（約27） | 21.0–29.0 |
| GRU W60 Q4 | 2（不同lr，不併baseline） | 9.0 | 9.0–9.0 |
| Mamba2 W60 Q4 | 2（不同lr，不併baseline） | 26.4 | 26.3–26.4 |
| GRU W120／W250 | Q3各6 | 14.2／24.2 | 9.9–16.0／17.6–27.9 |
| Mamba2 W120／W250 | Q3各6 | 38.4／61.8（約38／62） | 37.6–40.9／58.5–62.8 |
| crossstock W60 | Q1 6／Q2 6 | 29.9／26.3 | 29.5–30.6／14.0–30.9 |

以 baseline W60 Q1／Q2 等 n 加權，GRU 9.05、Mamba2 27.50，兩臂每 seed 共36.55分鐘；觀測最大值相加39.50分鐘。以下「max情境」只是把已觀測單 fit 最大值套用各 fit 的敏感度，**不是未來最大耗時保證**。窗120／250不在本案佇列，不加進總量。搜尋、更大容量、不同資料支援與環境都可能超過此值。

| N／臂 | 搜尋fits（含CPU C0+） | T4搜尋基準等效h／max情境h | CPU搜尋h（未量測） |
|---|---:|---:|---:|
| 6 | 36 | 7.31／7.90 | 2–12 |
| 12（建議） | 72 | 14.62／15.80 | 4–24 |
| 24 | 144 | 29.24／31.60 | 8–48 |

六折最終神經重訓24 fits（每臂12）＝7.31h／max情境7.90h，CPU C0+12 fits＝2–12h。**N12核心，不含三模組：72個神經 fits，21.93h基準等效／23.70h觀測max情境**，CPU6–36h。核心搜尋有寬度／層數／epochs變動，21.93h絕非實測總工時；沿用×1.5–4規劃敏感度為32.90–87.72h，超出仍可能。

### 三模組增量與總量（設計及共享規則見 §6）

模組使用固定 P4a 配方，和「調參後最佳」核心結果不同，**不能默認重用核心六折 fits**。以下先保守不跨核心重用；模組之間只重用完全相同的 baseline／seed／split／更新次數／環境 fits。每臂每一設定均包含2021前 anchor及六個 outer origins，共7 origins；anchor只供驗證曲線與固定 baseline BEST 更新次數，不開 outer 後選容量／種子。全模組採 W60。

| 模組／增量 | 新增神經fits（GRU／Mamba2各半） | 基準等效h | 觀測max情境h |
|---|---:|---:|---:|
| 共用100%／d64×2／seed17,29 baseline：7 origins×2臂×2seed | 28 | 8.53 | 9.22 |
| B 2→8 seeds：7×2×6 | 84 | 25.59 | 27.65 |
| A 六格扣baseline：7×2×2×5 | 140 | 42.64 | 46.08 |
| D 25／50%扣100%：7×2×2×2 | 56 | 17.06 | 18.43 |
| 三模組合計（baseline只算一次） | 308 | 93.81 | 101.38 |
| N12核心＋B（含共用baseline） | 184 | 56.04 | 60.57 |
| N12核心＋B＋A六格 | 324 | 98.69 | 106.65 |
| N12核心＋B＋A六格＋D | 380 | 115.74 | 125.08 |
| A改全12格：較六格再加7×2×2×6 | +168 | +51.17 | +55.30 |

A單獨六格168 fits＝51.17h，全12格336 fits＝102.34h；單獨B112 fits＝34.11h；單獨D84 fits＝25.59h，三者不能直接相加而漏掉共享baseline去重。D如加 C0+ 兩seed三份量，另42個CPU fits約7–42h，仍未量測全列耗時／RAM；與核心合計CPU13–78h。D不是縮短資料就縮短更新：400steps不變，不能將25%工時乘.25。若核心baseline候選與模組anchor逐項identity相同，最多另扣兩臂×2seed＝4 fits（1.22h等效），須以manifest認證；不能預先減帳。P4a重用目前不合格，故上表未扣。

容量未知用公式保留：A五個非baseline格的實測相對倍率為 `r_G(c), r_M(c)`；A增量h為 `14/60 × Σ_c[9.05*r_G(c)+27.5*r_M(c)]`（每臂每格7 origins×2seed）。表中42.64h只是假設全部r=1的歸一化參照；**d256／4層的耗時及顯存 UNKNOWN，沒有可辯護的總工時上界**。完整N12＋三模組若一律套×1.5–4，為173.61–462.97h，僅整體敏感度，不能當排練前已知的預算範圍或上限。核心與模組容量成本應於排練後逐格替換，失敗／重試、編譯、矩陣、推論集成、資料傳輸、斷線閒置另列UNKNOWN；validation若已包含，不重複加。

Colab Pro T4 計算單位採 **估計 r=1.5–2.5 CU/計費h** 作規劃區間，非本次帳單實扣／provider承諾；正式規劃須以Human當次面板觀測替換，不能當現行固定售價。基準等效至觀測max情境交叉乘該r：N12核心約32.9–59.3 CU；核心＋B約84.1–151.4 CU；核心＋B＋A六格約148.0–266.6 CU；全三模組約173.6–312.7 CU。若全三模組工時敏感度173.61–462.97h，約260.4–1157.4 CU；所有區間都未含未知閒置／I/O與大容量超額。CU公式 `實際T4計費h×當次觀測r`；不能把active fit時間當帳單。沒有購買／reset授權；仍為record_only、無runtime時間／CU自動上限或reserve gate，85°C硬體STOP不改。階段停／續由Human依預凍結設計及實測資源決定，不得看到績效才追加試驗。

S-Mamba風格預設不納入。替代案須在任何結果前同N×2seed／Holm3並凍結額外成本及排列驗收；以crossstock Q1+Q2平均28.1、觀測max30.9min估核心 `2N+12` fits，N12再16.86h／18.54h，容量／epochs仍未知。本次A／B限GRU與Mamba2；不因既有crossstock耗時可讀就自行增加第三臂。

算術採精確十進位、各合計先加總後四捨五入到小數2位，勿將已四捨五入各列再相加。

每trial／seed保存完整超參（包含inactive項）、候選／抽樣種子、fit/val鍵與日期support、epoch/iteration history、BEST與早停原因、validation IC5逐日值、wall／active／編譯／I/O／中斷、CPU/GPU/RAM/telemetry、失敗／重試鏈，以及source commit＋dirty逐檔SHA、contract／config／matrix／feature-order／preprocess／環境lock hashes、checkpoint、predictions、metrics與輸出hash。固定seed不保證不同CUDA kernel跨環境位元一致；如實記錄determinism與容差，不混用環境結果。trial目錄與ledger由immutable identity決定，重啟不重抽，不以最佳trial覆蓋全trial總表。

## 5. 六折最終比較與統計

全部臂完成tuning後，先發布selected recipe manifest、兩seed各自BEST epoch／iteration，並鎖定其hash，才開任何2021–2026分數／績效。每臂全六折使用同一配方，各seed用tuning選定的更新次數；神經重訓仍用候選max_epochs決定原cosine總週期、只執行至選定BEST epoch，不能用縮短後的新週期替代。**最終重訓不再看內層IC早停**，不得因某折表現改配置。模型不接續tuning權重；以17／29 fresh fit，rank-average在共同支援上重算。原三seed C0+保留歷史參考，不是本案主對照。

重用 [p4_eval.py](../V6/research_lab/p4_eval.py) 的共同列／排名、[portfolio_eval.py](../V6/research_lab/portfolio_eval.py) 的 `p2_primary_config/p3_pair` 與既有significance函式，由P4d獨立adapter驗證新manifest；不把新trial塞進P4a loader、降低64-fit gate或改family。主要格固定raw、top50、每20sessions、buffer100、成交值2000萬、佣金.1425%×.6、最低20元、資金100萬、稅.3%、單邊滑價10bp；同可交易宇宙等權基準只收比例成本。價格snapshot／volume單位／source provenance需凍結並fail closed，本輪未讀正式價格；不自動採用每日更新cache。

只有兩個登記比較 H-D1=GRU−C0+、H-D2=Mamba2−C0+，在舊2021–2023池做20-session circular block bootstrap1000次、seed20261125、95%CI與Holm **m=2**，凍結為既有shared **雙邊centered p**：令observed_mean=mean(d)，bootstrap_mean為circular抽樣均值，尾端事件為 `|bootstrap_mean − observed_mean| ≥ |observed_mean|`；數值沿實作容差1e-15，即比較右側abs(observed_mean)−1e-15，`p_boot=min(1,(1+命中數)/(1000+1))`，mean=0時p=1；兩個p一次Holm m=2，不直接除二；登記門檻CI下界>0且Holm p<.05仍只代表「已曝光歷史門檻通過」。2024–2026、各折、六折池只描述，不能換到較好pool當主要結果。反向落後可報CI上界<0的方向證據，雙邊Holm通過且CI上界<0才描述此歷史協定下落後，不把方向性科學假設誤稱單邊檢定。若Human選S-Mamba則凍結m=3，不加GRU vs Mamba、scaling slope等正式假設。小樣本、跨折block邊界及bootstrap Monte Carlo誤差沿既有方法揭露。

每折全部系統／seed/date/code列必完整一致；不是先刪不同列再宣稱完整。重訓各臂W60資格相同，label identity相同位置NaN可匹配，±inf／有限與NaN不符拒絕；IC只用共同成熟列，portfolio不把label當收益。seed排名在共同支援重算，選股tie用score降序、code升序；同日期缺價／未知tradability不可填零通過。報input/common/excluded rows與dates、fit／eval成熟數、實際return sessions、逐日配對、換手與成本、年度與pool結果，不只報最佳trial或正向年份。缺任一必要fit或identity即INCOMPLETE／STOP，不從較小子集推升完成。

## 6. 三個可選描述模組：D 資料量、A 容量、B 種子數

三者在看新結果前同時凍結，開關各自獨立；Human已同意加入A／B設計，不再詢問是否可以寫文件。**預設三模組都用固定 P4a 配方，不用P4d調參後最佳配方**：W60、d64、2層、expand2／headdim8／d_state16、AdamW lr3e-4／wd.01、8epochs×400steps上限、8日×256股、rank-gauss、loss y5:y10=1:.5、clip1、15% warm-up＋cosine、ListNet0。純GRU／Mamba有效dropout為0，不把contract的crossstock dropout .1接到temporal路徑。P4a配方帶欠調參混淆，但可隔離新增容量／種子／歷史量，避免三模組各自用不同搜尋勝者。替代為各臂P4d調好配方：較貼近調後效能，卻混入搜尋選擇、width／state／epoch不一致，不能歸因單一縮放軸；若採替代，須事前另凍結完整軸與重算，不能兩套跑完挑勝者。

**固定配方不等於重用舊切分／早停**：三模組統一沿§2的2021前126-session全列validation、嚴格purge／maturity、完整交易年等值100%池；baseline先以17／29取BEST（8epochs上限、patience3），記每臂每seed的BEST更新次數。A所有非baseline容量、D各份量均用同臂同seed的baseline更新次數、原8epoch cosine週期，**不再依各格early stop**；B新種子也用預定對應次數（43／71／101對應17，59／89／113對應29），不能挑較有利次數。anchor各格逐epoch保存train／val IC5但固定更新；outer六折僅固定規則fresh fit、不用折內validation選epoch。這測固定訓練計算配方的反應；大模型可能欠收斂，須誠實記錄，不能事後延長。核心調參後最佳配方仍依§3–5，與三模組分別報告，不將模組額外trial當對等核心預算。

| 模組 | 假設 | 預定格／種子／共同項 | 量測與解讀 | 成本與依賴 |
|---|---|---|---|---|
| A 容量階梯 | 固定資訊與更新配方仍有容量空間 | GRU／Mamba2，預設六格，每格17／29；100%歷史 | 主要anchor驗證IC5及逐seed曲線；train–val IC5 gap診斷過擬合；outer IC5與P2組合層皆描述性。上升＝仍有容量空間的線索；持平／下降＝噪聲或資訊受限的工作假設，亦可能欠收斂／正則不合，非證明 | 單獨168 fits／51.17h等效；先baseline、各寬深shape與記憶體排練。實測大容量倍率未知 |
| B 種子集成曲線 | 訓練隨機性與集成的描述性反應 | 同一d64×2配方，1／2／4／8排名平均；100%歷史 | 主要anchor驗證IC5增量／飽和點；outer IC5、淨超額、換手／成本／回撤描述性；單seed分散與prediction相關。上升＝訓練隨機性與集成可能有用的描述性線索，混合batch抽樣與baseline更新長度選擇，不能歸因初始化；持平＝種子高度相關或信號受限；下降＝互相稀釋／不穩定，非挑seed理由 | 單獨112 fits／34.11h；已有兩seed的28 fits合格時，新增84 fits／25.59h；只依賴baseline，不依賴核心調參 |
| D 資料縮放曲線 | 固定容量／計算下，更多最近歷史提供可用資訊 | 25／50／100%，GRU／Mamba2與C0+參考；17／29，baseline容量與配方不變 | 主要anchor驗證IC5；outer IC5與神經−C0+組合差斜率描述性。上升＝更多歷史可能有用；持平＝此配方未吸收額外資訊；下降＝舊regime／品質或固定步數不足，不能說更多資料有害 | 單獨84神經fits／25.59h＋42 CPU fits假設7–42h；共用100%後新增56神經fits／17.06h。依賴split／history／halo清單，不能按資料比例折工時 |

上述IC5一律同日期／股票／成熟標籤support、日權重；IC10為secondary，不能事後換主指標。train IC5用固定、完整合格train支援（另報計算成本），與val相同評估函式；train−val大且隨容量增長為過擬合線索，不跨不同regime斷言泛化差的唯一原因。anchor曲線不作額外Holm family或顯著性勝出判定；§5核心Holm2不被模組取代。外層組合層不得回流選容量、seed、資料量、epochs或下一格。

### A. 六格預設、12格替代與Mamba2縮放表

全12格為 `{32,64,128,256}×{1,2,4}`，能觀察寬深交互作用及d256×4角落，代價為每臂每origin24 fits，兩臂7 origins共336 fits。預設採寬度掃描2層 `{(32,2),(64,2),(128,2),(256,2)}` ＋深度掃描d64 `{(64,1),(64,4)}`，交點只算一次，**六格**、168 fits；資訊為固定深度的寬度趨勢及固定寬度的深度趨勢，不能估完整交互作用或聲稱已測d256×4。全12格多168 fits、51.17h基準等效，真正昂貴角落可能遠高於此。以一半格數保留兩個主要方向，預設六格；不得看到六格績效後直接擴成12格，若未事前選全12就另案承認曝光。

| d_model | expand | headdim | d_state | d_inner／nheads | 適用深度 |
|---:|---:|---:|---:|---|---|
| 32 | 2 | 8 | 16 | 64／8 | 預設2；全格1/2/4 |
| 64 | 2 | 8 | 16 | 128／16 | 1/2/4 |
| 128 | 2 | 8 | 16 | 256／32 | 預設2；全格1/2/4 |
| 256 | 2 | 8 | 16 | 512／64 | 預設2；全格1/2/4 |

**預先固定headdim8、d_state16、expand2，不隨寬度調整**；以增加head數擴容量，保持每head維度／state及inner比例，所有nheads為8倍數、滿足現有 `seq_models.mamba2_dims` 靜態條件，不等於CUDA已驗證。替代一：headdim=`{8,16,32,64}`、固定8 heads，會同時改每head表示／動態性且d64不再是P4a headdim8；替代二：d_state按寬度成 `{8,16,32,64}`（expand2／headdim8），會同時改記憶深度及state成本；替代三：固定headdim8/state16、expand隨寬度改，會混入inner比例。不採這些方案，因目的為可解釋的容量反應，非搜尋Mamba結構。GRU同寬／層不代表同參數量；逐格記trainable／total參數、相對baseline倍數、fit秒數（含validation與不含的拆項）、epoch／step、峰值allocated／reserved顯存與裝置總用量、推論延遲／吞吐、失敗及重試。以實際參數／秒數作橫軸，不能只用d_model稱等容量。

### B. 訓練隨機性與集成的描述性反應：固定種子順序及重用核對

種子可同時改Python random、NumPy RNG（日期／股票batch抽樣）、torch CPU／CUDA初始化及dropout mask、DataLoader worker／shuffle（若使用）；LightGBM另記bagging／feature_fraction／data seed，搜尋候選RNG固定且不隨B seed變動。manifest列每個stream派生規則、版本、determinism設定與resume RNG狀態，不能只記一個整數便假設只改初始化。有效dropout0的P4a baseline不消除初始化／batch抽樣差異，GPU kernel仍可能非確定。

17／29各自以同一anchor選BEST更新數，新seed依既定交替對應17／29，故K曲線同時混合訓練隨機性、更新長度選擇及集成效應，**不能單獨歸因初始化方差**。anchor用於選更新長度，曲線有選擇樂觀偏向，並非未用於選擇的驗證證據；outer也因整體歷史曝光只描述性。A容量固定更新／正則化不匹配、D歷史與regime混淆維持原限制。三軸表只形成工作假設，不識別機制。

在任何新增結果前固定序列 **[17,29,43,59,71,89,101,113]**，兩臂、所有fold／fraction使用同一序列；K1={17}、K2={17,29}、K4=前4、K8=前8，嚴格巢狀。不是以validation／portfolio挑最佳seed。K1的選擇有單seed偏差，另列8個單seed IC5分散描述，不改主K1或挑最好的K集合。已有P4a 17／29永遠只放前兩位；不能因P4a seed表現換序、拿Q4 lr1e-4或crossstock補數量，也不能把同seed的多次續跑算獨立seed。

先裁所有K／seed與參考系統到相同合法鍵，再逐fold/date計percentile rank、等權平均（不平均raw scores）、code升序解決最終選股同分；保留raw seed scores供重算，缺一seed即INCOMPLETE，不換seed、不降K並稱完成。

| 重用檢查 | 本輪可核實 | 目前結論 |
|---|---|---|
| 配方／架構 | P4a contract確有W60、d64×2、lr3e-4、expand2／head8／state16、17／29；temporal有效dropout0 | 靜態配方相符，Q1／Q2候選；Q3長窗與Q4改lr不符 |
| 環境 | P4a OBSERVED cp313：Python3.13.16、torch2.14.1+cu130、mamba2.3.2.post1、causal1.7.0、T4；摘要無逐fit環境hash，P4d未觀測 | 未證明exact同環境；本機Python3.12.3不能當同一runtime |
| split／validation／更新 | P4a val_max_rows80000、各折BEST早停；P4d採全列2021前選更新、outer固定更新與完整年池 | **目前不同，不能重用P4a fits作本模組前兩seed** |
| artifact identity／來源 | 64-fit摘要計數符合；無逐fit完成manifest／checkpoint／score keys／source lineage重驗 | 尚未合格，不能用filename或n=64代替驗證 |

原則上只有完整有效配方（含scheduler／BEST更新數）、資料／matrix／split／preprocess／support、環境／kernel與determinism、source數值相容及completed-fit hashes全部一致，才可重用P4a作17／29，仍標已曝光的重用來源，不能改寫歷史。若要改用P4a逐折早停協定以促成重用，須事前凍結為獨立描述性替代，不能默默替換P4d禁止outer選epoch的邊界。本預設重訓前兩seed；不同環境不得混入同一ensemble。

同一計算預算比較 **更多seed vs 更大單模型**：按每臂實測累積fit GPU秒與推論秒建立資源曲線。B的K模型總fit成本＝K個baseline fits之和；A各容量報各seed單模型及17／29兩seed平均與ensemble，不能拿A兩seed成本冒稱一個大模型。anchor用seed17的較大單模型對B固定K序列，seed29另列敏感度；兩seed平均僅表示單模型期望分散，非ensemble。比較在預定baseline單fit的1／2／4／8倍預算點：只使用成本≤該點的已凍結配置，全部列出，不為湊點新增訓練；無合格大模型／成本不相當即標無可比點，不插值假裝實測。outer同樣整張描述，不依其結果選Pareto勝者去改規則。若B改善而A無改善，只形成「訓練隨機性／更新長度與集成較容量擴展有利」假設，不能識別初始化方差機制；若A改善且B飽和，較大表示能力值得另案；兩者都升則報各預算邊際收益／推論與儲存代價，不宣稱大模型或ensemble普遍較佳。

### D. 資料量的操作定義與混淆

對每個anchor／outer origin先取全列label成熟且purged的100%訓練日期池。設L為該池首末間的session數，K=floor(L/252)個「交易年等值」；K≥4才執行（不足STOP）。100%為最近K×252個sessions，50%為最近floor(.5K)×252，25%為最近floor(.25K)×252，全部錨在同一purged末日；**不是隨機抽樣／每隔幾日取一筆／按列數切前25%**。此處用252-session年是明確操作定義，非聲稱每曆年恰252日；報實際起迄、K、fraction rounding後比例與列／公司／session數。tuning的100%也使用同一完整交易年等值裁法（不足一年餘數不進fit）；不得主實驗另用較長池再冒稱可重用100%結果。

所有fit端點都在purge之後決定，不先截25%再把purge／validation算進比例。126日validation只在2021前tuning使用，且位於tuning fit之外、比例分母不含validation或20-session gap；縮放最終折不再留126日選模型。縮放fit rows完全由日期後綴與共同合格mask產生，不對股票列做500k隨機cap；單日抽batch是固定optimizer機制，不改history定義。W60及C0+滯後／rolling的特徵上下文允許讀fit起點之前最多59sessions的**特徵**，不得取那段label為fit／val；context halo另列manifest、不能混入訓練年數。新上市W60不足／缺session排除規則固定，eval支援不因history fraction改變；若無法維持共同eval鍵即STOP，不能各自選容易樣本。

主要曲線為固定anchor驗證IC5對資料量；outer比較單位是同fold、同seed、同fraction的神經−C0+，另報兩seed rank-average後的配對日均淨報酬差 `D_f(q)`、IC5差及實際列數／公司覆蓋，全部描述性。D的C0+使用固定P1基準配方（非核心搜尋勝者），100% anchor選BEST iteration後各q固定相同迭代數；這不同於舊P1的500k cap，必須全列重訓。以 `x=log2(實際訓練sessions)`，對每fold三點OLS得β_f（單位日均淨差／訓練歷史倍增）；這等同神經與C0+斜率相減。報六個β、兩pool等折均值、正負數、三點非單調程度與兩seed敏感度。**不做斜率顯著性檢定／不增加p值**：只有六個高度重疊訓練期的fold，seed非獨立市場樣本，三個q也非獨立。可並列既有block bootstrap對每q之配對差的描述CI，但不能把兩端CI相交當slope檢定、不假設各q獨立、不能宣稱因果或無偏外推。

混淆必列：增加歷史同時換了regime與股票／產業組成；公司數與資料品質隨年代改變；容量及P4a配方固定但100%選baseline更新長度；固定400steps可能更少遍歷每筆資料（因此不是訓練到各自最佳的scaling law）；baseline早停長度轉用不同q會偏向100%；rank-gauss以當日全市場作截面、非新增歷史學得scaler；survivorship／PIT／來源缺口仍在。僅可寫「此固定配方與已曝光期間中，差距隨更多最近歷史縮小／擴大，形成後續假設」，不能寫「證明神經只缺資料」「更多資料必勝」或用三點外推所需年份。

### 依賴順序與事前分階段決策

1. **G0 凍結／實作前**：選A六或12格、三模組開關、seed序列／epoch對應／固定P4a配方、D是否含C0+、預算與完整候選表；機器contract、split／身份、CPU全列RAM與合成測試、獨立審查未通過一律STOP。模組不必等核心搜尋勝者；本輪文件完成不啟動任何執行。
2. **G1 baseline與可行性排練**：先本機CPU／小型可行部分；大型完整訓練本機跑不動才另經Human同意Colab。2021前anchor兩臂baseline先固定更新數。逐格shape／forward-backward／完整validation／checkpoint排練；Mamba d256×2為六格最大寬、d64×4為最大深；d256×4只屬12格替代，但若考慮該案須先排練此角落。報顯存及投影工時，不能推定兩個較小格都可跑則角落也可跑。
3. **G2 便宜資訊先行**：先B與A的GRU部分，再B的Mamba baseline種子。GRU B完整7 origins×8seed＝56 fits，8.45h等效；A的GRU額外70 fits＝10.56h等效，共19.01h（容量倍率未知）；B Mamba56 fits＝25.67h等效。B先出曲線不表示可據其績效取消不利A或改K；停止只能按事先資源可行性／Human決定，報PARTIAL與缺格。
4. **G3 昂貴容量決策點**：依排練資源而非勝負決定是否續做原定Mamba大容量；OOM不縮batch／state／窗以維持原身分，保留失敗成本。若資源不足則STOP／PARTIAL；改設計另案，不能把GRU完成稱兩臂完成。
5. **G4 D與完整報告**：D所需history／support可行後按預凍結25／50／100執行，100%可與A/B共用；核心搜尋可獨立排程但不新增scheduler。各階段先凍結全部anchor設計再開outer；不得看驗證趨勢改既定格數／seed／資料比例，outer報告後尤其不能回頭改設計。完整性、metadata／hash、durable輸出及獨立審查完成才由Human決定接受或真正未曝光前瞻候選。

### 三軸結果組合決策表（全部描述性）

「↑／→／↓」是完整預定曲線隨軸增加的方向，anchor IC5為主；逐點、逐seed、共同支援差與不確定性全部附上。沒有事先可辯護的等效界，**→只表示未見一致改善，不代表統計等效**。方向反轉／不同seed不同向標MIXED，不硬分↑或↓；組合層相反另列，不用它改IC5結論。下表逐列涵蓋3³=27種組合（花括號各值各自適用）；不能由一個seed的小幅正差直接下機制結論。

| A容量 | B種子 | D資料量 | 允許的工作假設／下一步（皆需另案，非自動加預算） |
|---|---|---|---|
| ↑ | ↑ | ↑ | 容量、訓練隨機性／集成與歷史均有描述性改善線索，未識別機制；按實測同預算效益提出前瞻候選 |
| ↑ | ↑ | → | 容量／集成有用，更多舊歷史未見增益；檢查regime及固定更新限制 |
| ↑ | ↑ | ↓ | 大模型／多seed改善，舊歷史可能稀釋；核時代組成與品質，不能事後縮history選收益 |
| ↑ | {→,↓} | ↑ | 容量／歷史有線索；集成飽和或稀釋，檢查seed相關，不挑seed |
| ↑ | {→,↓} | → | 容量優先假設；集成／歷史受限，先查大模型收斂及gap |
| ↑ | {→,↓} | ↓ | 容量有空間但歷史延長退化；查regime／資料品質與更新不足 |
| {→,↓} | ↑ | ↑ | 訓練隨機性／集成與歷史曲線改善，容量未改善；若A↓且gap增大則過擬合線索 |
| {→,↓} | ↑ | → | 更多seed優於更大模型的線索；資訊可能受限，查相關與成本 |
| {→,↓} | ↑ | ↓ | 集成穩定化有用，歷史延長不利；查regime，不能靠集成掩蓋source缺口 |
| {→,↓} | {→,↓} | ↑ | 主要為歷史量線索；A↓查過擬合／欠收斂，B↓查稀釋，不能說架構無效 |
| {→,↓} | {→,↓} | → | 無一致擴展收益：噪聲／資訊受限為工作假設，亦可能訓練配方不合；保留強樹基準 |
| {→,↓} | {→,↓} | ↓ | 三軸未改善且資料退化；優先稽核regime／品質／PIT／support與訓練不足，勿追加直到贏 |
| 任意MIXED | 任意 | 任意（或其他軸MIXED） | 無單一方向，報各格與seed異質性，不從最佳端點推整條曲線 |
| UNKNOWN／缺格 | 任意 | 任意（或任一軸缺失） | STOP／INCOMPLETE，不用0或子集填補完整性 |


## 7. 事前決策、不允許事項與後續

| 結果（完整性先通過） | 允許解讀與下一步 |
|---|---|
| 調參後點估仍落後；上界<0或多折方向一致 | 描述此資料／預算／配方仍落後；不說架構無效。優先保留樹基準，若要再研究需新問題與新登記，不繼續加trial直到贏 |
| CI跨0、差距比P4a小 | 「無法區分、方向性縮小」；不能稱等效／非劣，未設非劣界且P4a種子／訓練列不同，不作有校正的進步檢定。以成本／穩定度決定是否值得前瞻候選 |
| CI下界>0且Holm p<.05 | 描述性歷史門檻通過；僅提出候選，另訂真正未曝光前瞻契約、獨立審查與Human批准後才可升級證據，不改production |
| 正向點估但門檻未過 | 未證實，完整報CI／不確定性，不能寫成功 |
| scaling β多為正／負／不一致 | 分別形成「差距可能隨歷史縮小／擴大／未見一致趨勢」假設；不能自動選更多資料／較大模型或聲稱資料量是原因 |
| 任一missing／hash／PIT／支援錯誤 | STOP／INCOMPLETE；不算0、不降低gate、不挑完整年份稱成功 |

禁止看到結果後改搜尋空間／N／種子／metric／family／pool、重用任何2021–2026測試資料選超參或early stop、只報最好trial、依試驗績效丟掉seed／失敗試驗、換價格或universe改善績效、把pilot／rehearsal算確認性結果。即使資料切分通過，原資料source PIT／下市／標籤問題仍不能宣稱消失。沿用限制：raw無股利、現在股數市值非PIT、未驗證下市覆蓋／原因、無漲跌停旗標、觀測日曆、模擬標籤、成本假設與partial2026。

後續方向只列、不納入本規格：跨市場（如美股）預訓練與domain shift控制；取得更早且合格歷史；取得逐公司實際公告日與revision provenance後加入基本面。資料取得／授權／費用／PIT admission均須另案，不能因本文件擅自購買或抓取。Human「神經潛力較大、資料可能不足」僅設計動機，不改門檻與不利結果回報。

## 8. 未來實作需求與驗收（本輪不寫程式）

1. 沿用research_lab，新增薄P4d contract／candidate-table／split-manifest adapter（含module A/B/D、capacity、ensemble K／固定seed序列、fraction與baseline更新數對應）；擴充 `c0plus` opt-in外部fit/val rows、全列訓練與IC5早停，避免訓練額外day-t基準；`seq_models/seq_runner` opt-in dropout／head weights／固定更新重訓入口。舊P1/P3/P4a預設、數值與來源相容保留，不改受保護models或formal Data。
2. `p4_colab/p4_bundle` opt-in P4d候選佇列、ledger與checkpoint identity（含candidate、fraction與split）；重用 `p4_eval` 公開核心但獨立P4d完成閘門／兩假設，不修改P4a64-fit loader。資料matrix build-once/reuse，任何新split只在隔離artifact，不覆寫原splits.npy；共享cache須input／implementation／output hash驗證。
3. 依 [Human操作契約](../docs/operations/colab-human-requirements.md) 交付**一個zip＋一個ipynb＋兩行README.txt**；矩陣一次上傳、保留Drive既有data/wheels與歷史目錄、P4d／speed各自ledger。自動safe-extract逐檔size/SHA，不要求Human手貼hash、查版本或改路徑，不以pip check作gate。當次最新Colab runtime需OBSERVED exact identity，官方T4 wheels缺失才走既有可驗fallback；不假設P4a lock永遠適用。
4. 每epoch local serialize/fsync/hash→durable copy/reverify→atomic pointer，latest/previous/best與optimizer/scheduler/RNG全部保存；trial／seed／fraction／source／matrix／env／contract mismatch STOP。完整fit驗證後跳過、partial從完整epoch接續、不重抽候選；validated orphan只能依明訂policy恢復latest，不升best或formal。每50steps／epoch串流flush、約60秒TXT/JSON心跳、IC／trial進度／fit與queue ETA／溫度時脈／最近checkpoint／可行動錯誤，未知ETA寫未知。做完驗durable artifacts才打包下載並由Human關runtime；不新增通知／wakeups／排程。
5. 保留來源相容升級能力：每次宣告逐檔from/to SHA與差異分類、舊completed-fit／checkpoint／CPU位元一致證明，非中立數值變更須新run；P4d新搜尋與配方不是把P4a原run直接升級。不能清空帳本、刪Drive或只改commit參數繞identity。
6. 強制執行§2未來特徵mutation及cache hit／rebuild／新程序三路隔離測試，確認tuning features、trial ranking、selected recipe全不變，並殺死舊全矩陣macro抽樣mutation。先RED後GREEN的CPU合成測試：2021邊界label_end／嚴格20sessions；fit／val／eval date集合與tuning loader禁止讀eval；跨fraction起點／halo／嵌套鍵；全列沒有500k cap；尾端y5/y10 NaN、±inf、缺session、新上市不足60日、<64股、常數／空成熟支援、極端raw值、重複鍵、缺seed／缺trial；dropout接線、head loss權重有效性、Mamba四寬維度表非法組合、六格去重／12格切換、baseline共享identity、固定種子前綴不挑最佳、P4a重用逐項拒絕、成本帳不得跨module重複計費或漏計；IC／BEST tie、抽樣表重建、排名tie與共同支援，不能只測正常常態矩陣。
7. 統計驗收需符號反轉d→−d時雙邊p相同、正負CI方向相反、mean=0／常數差／尾端等號容差、+1平滑及Holm m=2手算相符。若Human另案要單邊，須另凍結正向尾端 `bootstrap_mean−observed_mean ≥ observed_mean`（同+1平滑與容差）及正／負／零差驗收、重新審查假設；本案不採、不能把雙邊p除二。Resume round-trip、kill/restart、missing/corrupt pointer、orphan、contract/matrix/source mismatch、seed／trial／fraction隔離、失敗成本／重試鏈、cached skips不計速度、全trial表完整性；中立升級舊checkpoint到新版、0dropout與1:.5預設舊路徑位元regression。mutation至少移除purge、偷讀eval選recipe、改fraction為隨機抽樣、忽略dropout／head weight、排名先做再裁、漏seed仍complete、移除hash驗證，全部必被測試攔下；保存mutation結果與source identity。
8. CPU測試不能代替上線前現場排練：最大width／layer／epochs配置、GRU／Mamba、各fraction／新split、resume／全量validation推論以同真實資料與正式路徑排練，報峰值memory與wall；獨立結果schema／identity驗證，不只exit0。必要排練全通過才launch；Colab/T4/Drive現場至實測前一律UNVERIFIED，費用需Human另案同意。本輪禁止GPU所以未執行。獨立審查須不同session，作者自查不等於review，未授權另派agent故本輪不派。

## 9. Open Questions／Human 決定清單、blockers 與知識影響

彙整簽署入口：[三份DRAFT Human凍結決定彙整](../knowledge/00_Project_Map/Current_State.md#三份draft-human凍結決定彙整)。全文與下列重大選擇均未凍結，不代替另案資源／執行授權。

Human已同意加入A容量與B種子並想實際嘗試；本輪只落成可審查設計，不重問文件修改許可，亦未取得GPU／Colab／程式實作授權。重大預設與替代在正式執行前一次凍結：

| 待凍結／重大選擇 | 預設／替代與影響 |
|---|---|
| 預處理隔離／統計 | 預設明確36…47欄schema歸零、cache綁schema及fit support、未來mutation全路徑不變；替代僅合格fit估mask後全案凍結，不能全矩陣抽樣。採雙邊centered公式＋Holm2；單邊僅另案公式／驗收，不除二 |
| B解讀 | 訓練隨機性與集成的描述性反應；保留各RNG與baseline epoch選擇混淆。若要隔離初始化須另案固定batch／dropout stream，不能把本曲線改名因果證據 |
| 核心預算 | N12／臂、17／29、單一2021前126日validation；N6較省、N24覆蓋多但仍已曝光。21.93h是基準等效，非搜尋實測；CPU全列RAM未量測 |
| A六格或12格 | 預設六格，兩臂每格至少2seed，7 origins共168 fits；12格336 fits可估交互作用，多51.17h基準等效，大角落未知。凍結後不依結果擴格 |
| Mamba縮放 | 固定headdim8／state16／expand2，隨width增加heads；替代固定8 heads調headdim、state隨width、或調expand都引入額外混淆，預設不採 |
| 三模組配方 | 全部固定P4a、baseline選定更新長度（B新seed交替對應17／29），非各格BEST。替代P4d調後最佳有搜尋／容量混淆，須事前重定義，不並跑後擇優 |
| B範圍／重用 | 1/2/4/8、固定seed序列，無最佳seed選擇；本預設P4a fits不合格重用。若未來提P4a協定替代，必明示改變split／early-stop問題與來源身分 |
| D是否同案執行 | 可獨立關閉；開啟用25/50/100%最近完整252-session年、全列無cap、P4a配方固定更新，含C0+參考另加42 CPU fits。與A/B共用baseline後新增56神經fits，不能漏算 |
| 執行範圍與階段資源 | 預設全六折＋anchor；先GRU B/A、Mamba B，再依排練資源決定原案Mamba A與D停／續。不能用績效挑階段，停則partial；全模組加N12約115.74h基準等效，真實容量倍率／峰值顯存未知 |
| CU假設 | Colab Pro T4暫估1.5–2.5 CU/h，當次觀測替換；全案基準至觀測max情境約173.6–312.7 CU，大容量敏感度可更高，不是費用核准或上界 |
| 其他核心凍結項 | 完整搜尋表／N／pilot政策、嚴格purge、IC5／BEST tie、固定更新／cosine、全列support／100%定義、Holm2、dropout及loss新接線；S-Mamba預設不納，替代同預算Holm3且補成本 |

Blockers：不再是「P4a ledger未取得」，而是派生摘要未含原ZIP／hash-chain／逐fit source／matrix／環境驗證，不能據此扣重用成本或宣稱P4a重新驗收。split實際日期／hash、train/val support、CPU全列RAM、最大容量時間／顯存與計費率未量測；DRAFT未凍結、machine contract／程式未實作、獨立審查／現場排練未完成。不得自行執行pilot填未知。

ml-leakage-audit自查：look-ahead／source PIT與universe／下市仍UNKNOWN，raw／市值非PIT限制未解；§2／5／6規定label maturity／purge、train-only scaler、共同列及交集後rank、seed／設計預凍結。所有12項僅文件邊界核對，沒有新資料路徑測試或source／CUDA驗證；壞形狀與resume驗收仍列§8未來需求。缺證據仍STOP，不以描述性上升升格為strict executable／trading readiness，獨立review另session待辦。

Knowledge-impact八問：候選研究設計／major blocker／next decision=YES；已生效semantics、authority、subsystem、runtime、correctness gate、canonical supersession、deprecation=NO。前輪同步Current State／Research；本輪另於Authority Map記本修訂的DRAFT邊界，仍未取代machine contract；runtime observed_at不更新。沿用原文件與knowledge，不建重複memory或handoff樹。只交patch、不commit／push／merge，前輪工作樹預期僅三文件dirty（本輪實際範圍見末段），不能宣稱clean或Human acceptance。

本輪驗證紀錄：既有knowledge工具23 tests通過；摘要64 fits與均值／max、共享去重與N6/12/24算術、Markdown本機引用、diff scope與空白檢查另於交接列實際結果。沒有新增可執行schema／程式，無訓練或現場測試；之前版本驗證紀錄不當成本輪通過證據。


## 本輪六項審查修訂身分 — 2026-11-24

`job-run-mwutomlv-e3974c/spec_fixer`，既有worktree、detached HEAD/base `3ed68762a51aa1f62223aa10c354f3a1eae0686f`，起始clean。只改三規格及Current State／Authority Map／Research共6份Markdown，不stage／commit／push／merge，不建立實作或訓練；全數仍DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。F1／F2／D1／D2／B1／D3文件修訂供再審，未解除原NO_GO或取代Human決定。前輪「本輪」字樣為各段歷史紀錄，不當本次驗證證據。

WSL CPU，文件工具`/usr/bin/python3` 3.12.3，唯讀NumPy及shared統計檢查使用`/home/frank/.venvs/marketmamba-gpu/bin/python` 3.12.3（CUDA_VISIBLE_DEVICES空，未載入GPU／訓練）。MARKETMAMBA_DATA_ROOT unset，Data realpath為本worktree的普通Data目錄，未讀正式Data；唯讀實證僅使用Human指定外部prepared／pilot09矩陣及既有公開歷史artifact metadata。沒有讀封盲區、改V6/models、GPU／Colab／Drive、資料取得、production／scheduler、通知／wakeup／recurring jobs或額外agents。

Knowledge-impact：DRAFT設計語義／blocker與next decision細分=YES，同改三份knowledge；已生效machine authority／runtime／subsystem／correctness thresholds／canonical supersession／deprecation=NO。Human仍須選端點、摘要及預處理完整契約並凍結；正式calendar、來源／事件與所有未來adapter／mutation／推導測試及獨立複審未完成。文件完成不等於Human acceptance。
