# P4d：對等調參預算與資料縮放曲線

**DRAFT — 待 Human 凍結；不是執行授權。E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 文件識別 `p4d-equal-tuning-20261125`（指定的規格日期；實際撰寫 2026-11-24）。全部歷史資料已曝光，所有比較、CI、p 值與決策都是描述性、形成假設用途。2021–2023 只沿用舊契約「確認性折」的來源名稱，本案不具確認性。完成草案不等於 Human acceptance。

## 1. 目的、建立前盤點與來源

回答兩個有限問題：(a) 在本次新增的相同候選評估次數下，GRU／Mamba2 相對 C0+ 的歷史差距如何；(b) 固定調好的配方，延長最近訓練歷史時，相對差距的曲線如何。不能推論架構普遍優劣，也不能把新增預算相等說成歷來研究投入相等。

先讀 AGENTS.md、CLAUDE.md、Current State／Authority Map、ml-leakage-audit、P4a／P4b、P1／P3 報告、Training／Research domain、GOVERNANCE、Human Colab 需求，以及下列程式、machine contract 與測試。既有 `tasks/` 承載登記文件，`V6/research_lab/` 承載研究工具，`reports/` 承載結果；P4a 固定64 fits／family A/B/C 不包含調參，P4b 是固定混合，所以不能改寫舊契約來容納本案。沿用既有結構新增本文件身分，不建目錄、程式、worktree、輸出樹或替代 pipeline；未來產物才在另案批准後使用 `.artifacts/p4d/<contract_hash>/<run_identity>/`，不得覆寫歷史產物。

本輪工作樹 `run-mwus8xvs-57bc73/spec_writer_p4d`，detached HEAD/base `aa89bca6030fd4a02e47ba9e52d567a1c08a0425`，初始 clean；已盤點 worktree list，沿用指派工作樹。WSL Linux 6.18.58.1、系統 Python3.12.3，`python` 不存在；`MARKETMAMBA_DATA_ROOT` unset。此樹 Data 是普通目錄，realpath 為本 worktree `/Data`，不是已驗證正式資料映射；本輪只讀主 checkout 的 P1 manifest／P4 評估 metadata 與舊研究記錄，未讀寫正式 Data／V6/models 或 MarketMamba-blind，未接觸 GPU／Colab／Drive。

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
| manifest identity | source `ace2eef0c580d3484a4c6299de16b1c3d5420898`；manifest SHA256 `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`。本輪只核 metadata，不逐份重驗所有 parquet |
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

P4a ledger：本輪在主checkout `.artifacts/p4/` 只找到eval-full／q1q2 JSON與rehearsal logs，未取得原始逐fit hash-chain ledger，沒有讀Drive或封盲區。因此 **無法完成 ledger 實測推導**；這是凍結成本表前的evidence blocker，不能用speed-test步速冒稱ledger。Human提供W60 Mamba2約25–30分鐘／fit、GRU約10分鐘／fit，僅作 provisional 基線。取得可讀本機ledger匯出後，須核contract／source／environment／queue／window／arm、complete驗證、區分cached skip／rehearsal／重啟離線間隔，列每臂n、p50／p90／range及active／validation／I/O秒數與ledger hash；若ledger不能拆離線，列總wall上界，不冒充active T4。

以兩seed、GRU10min＋Mamba2 25–30min推算，每候選兩神經臂共70–80min。這是**基準配方等效時間**；搜尋中d128／3層／16epochs不適用固定耗時，規劃敏感度另列×1.5–4，不保證足夠。

| N／臂／單一tuning origin | 搜尋fit數（3臂） | T4搜尋基準h | 本機CPU搜尋h（假設） |
|---|---:|---:|---:|
| 6 | 36 | 7–8 | 2–12 |
| 12（建議） | 72 | 14–16 | 4–24 |
| 24 | 144 | 28–32 | 8–48 |

兩神經臂六折最終重訓×兩seed＝24 fits，再加基準等效7–8h；C0+12 fits約2–12 CPUh。縮放若併入，100%重用同identity已驗結果，25%／50%新增兩神經臂×6折×2seed×2份量＝48 fits，再加14–16 T4h、C0+24 fits約4–24 CPUh。固定400 steps不隨資料量減少，不能按25%直接打75折工時。

N12總計：不含縮放 T4 **21–24h**；含縮放 **35–40h**；若整套成本為基準的1.5–4倍則分別31.5–96h／52.5–160h，另加編譯、資料準備、現場rehearsal與傳輸（UNKNOWN、不重複算既已包含的validation）。以上CPU總計不含縮放6–36h、含縮放10–60h，亦未保證RAM足夠。計算單位用 `CU = 實際T4計費小時 × Human當次觀測費率 r`；例 **假設** r=10 CU/h，基準含縮放350–400 CU、×1.5–4敏感度525–1600 CU，費率不是provider實扣證據。中斷閒置可能繼續計費，另報，不把active時間當帳單。Human批准的是候選數與規劃額度；依操作契約**不加入runtime時間／CU自動上限或25% reserve gate**，仍只記帳與85°C硬體STOP；可在pilot階段完成後由Human停止／續行，不自動購買或使用reset。

S-Mamba風格建議本案不納入：P3／P4a未見跨股增益，增加排列／分塊超參、成本與第三個比較。若Human在凍結前納入，與其餘臂同N×2seed，固定n_cross1／n_perm2／cross d_conv2、d_ff4*d；共有神經空間照上表（cross FFN dropout跟同一候選值，須另測），Holm m=3，增加tuning `2N`、最終12、縮放24個T4 fits。crossstock單fit ledger未知；僅按P4a speed .462/.456近似Mamba的粗估，額外基準約(2N+12+[縮放24])×25–30min，N12含縮放再25–30h；不是實測保證。正式納入前先凍結其成本與排列診斷驗收，不能結果不好才移出family。

每trial／seed保存完整超參（包含inactive項）、候選／抽樣種子、fit/val鍵與日期support、epoch/iteration history、BEST與早停原因、validation IC5逐日值、wall／active／編譯／I/O／中斷、CPU/GPU/RAM/telemetry、失敗／重試鏈，以及source commit＋dirty逐檔SHA、contract／config／matrix／feature-order／preprocess／環境lock hashes、checkpoint、predictions、metrics與輸出hash。固定seed不保證不同CUDA kernel跨環境位元一致；如實記錄determinism與容差，不混用環境結果。trial目錄與ledger由immutable identity決定，重啟不重抽，不以最佳trial覆蓋全trial總表。

## 5. 六折最終比較與統計

全部臂完成tuning後，先發布selected recipe manifest、兩seed各自BEST epoch／iteration，並鎖定其hash，才開任何2021–2026分數／績效。每臂全六折使用同一配方，各seed用tuning選定的更新次數；神經重訓仍用候選max_epochs決定原cosine總週期、只執行至選定BEST epoch，不能用縮短後的新週期替代。**最終重訓不再看內層IC早停**，不得因某折表現改配置。模型不接續tuning權重；以17／29 fresh fit，rank-average在共同支援上重算。原三seed C0+保留歷史參考，不是本案主對照。

重用 [p4_eval.py](../V6/research_lab/p4_eval.py) 的共同列／排名、[portfolio_eval.py](../V6/research_lab/portfolio_eval.py) 的 `p2_primary_config/p3_pair` 與既有significance函式，由P4d獨立adapter驗證新manifest；不把新trial塞進P4a loader、降低64-fit gate或改family。主要格固定raw、top50、每20sessions、buffer100、成交值2000萬、佣金.1425%×.6、最低20元、資金100萬、稅.3%、單邊滑價10bp；同可交易宇宙等權基準只收比例成本。價格snapshot／volume單位／source provenance需凍結並fail closed，本輪未讀正式價格；不自動採用每日更新cache。

只有兩個登記比較 H-D1=GRU−C0+、H-D2=Mamba2−C0+，在舊2021–2023池做20-session circular block bootstrap1000次、seed20261125、95%CI與Holm **m=2**，按既有P4單邊centered p演算法；登記門檻CI下界>0且Holm p<.05仍只代表「已曝光歷史門檻通過」。2024–2026、各折、六折池只描述，不能換到較好pool當主要結果。反向落後可報CI上界<0的方向證據，不拿正向p當反向顯著性。若Human選S-Mamba則凍結m=3，不加GRU vs Mamba、scaling slope等正式假設。小樣本、跨折block邊界及bootstrap Monte Carlo誤差沿既有方法揭露。

每折全部系統／seed/date/code列必完整一致；不是先刪不同列再宣稱完整。重訓各臂W60資格相同，label identity相同位置NaN可匹配，±inf／有限與NaN不符拒絕；IC只用共同成熟列，portfolio不把label當收益。seed排名在共同支援重算，選股tie用score降序、code升序；同日期缺價／未知tradability不可填零通過。報input/common/excluded rows與dates、fit／eval成熟數、實際return sessions、逐日配對、換手與成本、年度與pool結果，不只報最佳trial或正向年份。缺任一必要fit或identity即INCOMPLETE／STOP，不從較小子集推升完成。

## 6. 資料縮放曲線（建議作可選描述模組）

**選各自P4d調好且凍結的配方**，不選固定P4a：目的為在相同新增搜尋後比較各臂的資料反應；代價是配方在100%調好，可能偏向100%，不能解釋成每個資料量都最佳。固定P4a比較較純粹但帶已知欠調參問題，留作替代，不兩套都跑再挑。25%／50%不另調參、不重新早停或選epoch、容量／batch／更新次數／seed不變，與100%同一推論及成本協定；這測的是固定計算配方下的歷史長度反應。

對每個outer origin先取全列label成熟且purged的100%訓練日期池。設L為該池首末間的session數，K=floor(L/252)個「交易年等值」；K≥4才執行（不足STOP）。100%為最近K×252個sessions，50%為最近floor(.5K)×252，25%為最近floor(.25K)×252，全部錨在同一purged末日；**不是隨機抽樣／每隔幾日取一筆／按列數切前25%**。此處用252-session年是明確操作定義，非聲稱每曆年恰252日；報實際起迄、K、fraction rounding後比例與列／公司／session數。tuning的100%也使用同一完整交易年等值裁法（不足一年餘數不進fit）；不得主實驗另用較長池再冒稱可重用100%結果。

所有fit端點都在purge之後決定，不先截25%再把purge／validation算進比例。126日validation只在2021前tuning使用，且位於tuning fit之外、比例分母不含validation或20-session gap；縮放最終折不再留126日選模型。縮放fit rows完全由日期後綴與共同合格mask產生，不對股票列做500k隨機cap；單日抽batch是固定optimizer機制，不改history定義。W60及C0+滯後／rolling的特徵上下文允許讀fit起點之前最多59sessions的**特徵**，不得取那段label為fit／val；context halo另列manifest、不能混入訓練年數。新上市W60不足／缺session排除規則固定，eval支援不因history fraction改變；若無法維持共同eval鍵即STOP，不能各自選容易樣本。

比較單位是同fold、同seed、同fraction的神經−C0+，主要曲線以兩seed rank-average後的配對日均淨報酬差 `D_f(q)` 表示，另畫IC5差及實際列數／公司覆蓋。以 `x=log2(實際訓練sessions)`，對每fold三點OLS得β_f（單位日均淨差／訓練歷史倍增）；這等同神經與C0+斜率相減。報六個β、兩pool等折均值、正負數、三點非單調程度與兩seed敏感度。**不做斜率顯著性檢定／不增加p值**：只有六個高度重疊訓練期的fold，seed非獨立市場樣本，三個q也非獨立。可並列既有block bootstrap對每q之配對差的描述CI，但不能把兩端CI相交當slope檢定、不假設各q獨立、不能宣稱因果或無偏外推。

混淆必列：增加歷史同時換了regime與股票／產業組成；公司數與資料品質隨年代改變；容量固定且100%選配方；固定400steps可能更少遍歷每筆資料（因此不是訓練到各自最佳的scaling law）；tuning早停長度轉用不同q會偏向100%；rank-gauss以當日全市場作截面、非新增歷史學得scaler；survivorship／PIT／來源缺口仍在。僅可寫「此固定配方與已曝光期間中，差距隨更多最近歷史縮小／擴大，形成後續假設」，不能寫「證明神經只缺資料」「更多資料必勝」或用三點外推所需年份。

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

1. 沿用research_lab，新增薄P4d contract／candidate-table／split-manifest adapter；擴充 `c0plus` opt-in外部fit/val rows、全列訓練與IC5早停，避免訓練額外day-t基準；`seq_models/seq_runner` opt-in dropout／head weights／固定更新重訓入口。舊P1/P3/P4a預設、數值與來源相容保留，不改受保護models或formal Data。
2. `p4_colab/p4_bundle` opt-in P4d候選佇列、ledger與checkpoint identity（含candidate、fraction與split）；重用 `p4_eval` 公開核心但獨立P4d完成閘門／兩假設，不修改P4a64-fit loader。資料matrix build-once/reuse，任何新split只在隔離artifact，不覆寫原splits.npy；共享cache須input／implementation／output hash驗證。
3. 依 [Human操作契約](../docs/operations/colab-human-requirements.md) 交付**一個zip＋一個ipynb＋兩行README.txt**；矩陣一次上傳、保留Drive既有data/wheels與歷史目錄、P4d／speed各自ledger。自動safe-extract逐檔size/SHA，不要求Human手貼hash、查版本或改路徑，不以pip check作gate。當次最新Colab runtime需OBSERVED exact identity，官方T4 wheels缺失才走既有可驗fallback；不假設P4a lock永遠適用。
4. 每epoch local serialize/fsync/hash→durable copy/reverify→atomic pointer，latest/previous/best與optimizer/scheduler/RNG全部保存；trial／seed／fraction／source／matrix／env／contract mismatch STOP。完整fit驗證後跳過、partial從完整epoch接續、不重抽候選；validated orphan只能依明訂policy恢復latest，不升best或formal。每50steps／epoch串流flush、約60秒TXT/JSON心跳、IC／trial進度／fit與queue ETA／溫度時脈／最近checkpoint／可行動錯誤，未知ETA寫未知。做完驗durable artifacts才打包下載並由Human關runtime；不新增通知／wakeups／排程。
5. 保留來源相容升級能力：每次宣告逐檔from/to SHA與差異分類、舊completed-fit／checkpoint／CPU位元一致證明，非中立數值變更須新run；P4d新搜尋與配方不是把P4a原run直接升級。不能清空帳本、刪Drive或只改commit參數繞identity。
6. 先RED後GREEN的CPU合成測試：2021邊界label_end／嚴格20sessions；fit／val／eval date集合與tuning loader禁止讀eval；跨fraction起點／halo／嵌套鍵；全列沒有500k cap；尾端y5/y10 NaN、±inf、缺session、新上市不足60日、<64股、常數／空成熟支援、極端raw值、重複鍵、缺seed／缺trial；dropout接線、head loss權重有效性、Mamba維度表非法組合；IC／BEST tie、抽樣表重建、排名tie與共同支援，不能只測正常常態矩陣。
7. Resume round-trip、kill/restart、missing/corrupt pointer、orphan、contract/matrix/source mismatch、seed／trial／fraction隔離、失敗成本／重試鏈、cached skips不計速度、全trial表完整性；中立升級舊checkpoint到新版、0dropout與1:.5預設舊路徑位元regression。mutation至少移除purge、偷讀eval選recipe、改fraction為隨機抽樣、忽略dropout／head weight、排名先做再裁、漏seed仍complete、移除hash驗證，全部必被測試攔下；保存mutation結果與source identity。
8. CPU測試不能代替上線前現場排練：最大width／layer／epochs配置、GRU／Mamba、各fraction／新split、resume／全量validation推論以同真實資料與正式路徑排練，報峰值memory與wall；獨立結果schema／identity驗證，不只exit0。必要排練全通過才launch；Colab/T4/Drive現場至實測前一律UNVERIFIED，費用需Human另案同意。本輪禁止GPU所以未執行。獨立審查須不同session，作者自查不等於review，未授權另派agent故本輪不派。

## 9. Human 凍結清單、blockers 與知識影響

| 待決定 | 建議／替代 |
|---|---|
| 預算 | N12／臂、2seed、單一tuning origin；N6省一半搜尋、N24增加覆蓋但不能解決歷史曝光；確認T4/CU估計與全列CPU/RAM可行性 |
| S-Mamba風格 | 不納入；若納入必在結果前同預算與Holm3、補現場成本與排列驗收 |
| 調參折 | 採2021前單一126日validation；逐fold較新但與既有test日期重疊，嚴格要求下不採 |
| scaling是否併入 | 建議獨立可選模組、同時凍結；不併入可省48神經fits。若併入用各自調好配方、252-session年與固定計算，不再調各fraction |
| 其他凍結項 | 完整搜尋表／基準點、N與pilot政策、兩seed、IC選擇、嚴格purge、全列無cap／100%定義、固定BEST更新重訓、Holm2與描述性措辭；一併確認dropout新接線屬新配方 |

Blockers：P4a原始ledger尚不可用（耗時只能暫估）；tuning split實際日期／hash／全列RAM／耗時未量測；DRAFT未凍結、machine contract／程式未實作、獨立審查／現場排練未完成。不得自行執行pilot以填估計，亦不把缺ledger寫成結果驗證完成。

Knowledge-impact八問：研究設計／next decision的候選敘述=YES；生效semantics、production/research authority、subsystem boundary、active runtime、correctness gate、canonical artifact supersession、deprecation=NO。同步Current State與Research短條目；已檢查Authority Map hierarchy，**不修改**，因本DRAFT不授權或取代任何machine contract。runtime observed_at不更新。只交文件patch，不commit／push／merge；知識更新不另建記憶或handoff樹。

本輪文件驗證：`python3 -m unittest discover -s tools/project_knowledge -p 'test_*.py'` 為23 tests OK；規格14個本機引用及兩個新增knowledge引用均有效，P1 manifest數值與N6/12/24成本算術核對。impact checker初次PASS、同步knowledge後WARNING（GOVERNANCE review required），已人工檢查治理文件，無治理規則變更需求；`knowledge_health.py --ci --as-of 2026-11-24` exit0／WARNING（authority repin候選、legacy docs），無structural FAIL。`git diff --check`通過；沒有新程式／schema／訓練／現場測試，不能宣稱P4d已可執行。
