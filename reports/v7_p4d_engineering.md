# P4d 第一段工程候選

E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。工程完成不是 Human acceptance。

## 建立前盤點

任務 `job-run-mwvnttp4-a9ebe0/builder_p4d_stage1`，既有 detached HEAD `35cd9b88ddec3bce3849ba0363c0514393d16d47`，初始 clean；已核對 root／worktree list，不新增 worktree。WSL、Python3.12.3，CPU 測試用 `/home/frank/.venvs/marketmamba-gpu/bin/python` 並設 `CUDA_VISIBLE_DEVICES=''`。`MARKETMAMBA_DATA_ROOT` unset；Data realpath 為本 worktree 普通 Data 目錄，未讀其內容，不推定是正式資料映射。

已盤點 p4_colab／p4_bundle／p4_source_compat／seq_runner／seq_models／p4_eval／c0plus、P4b結果及 project map、Training／Research domain、GOVERNANCE。既有 tasks、research_lab、tests、reports 可承載；P4a 契約為64-fit、逐折早停及舊macro偵測，不可承接 P4d 新切分與固定更新語義。故新增獨立 P4d 契約與 adapter，不修改 P4a ledger／queue、rolling_seq_scores.py、portfolio_eval.py 或 seq_runner.py。來源相容能力保留，P4d 是新 run，未宣稱可接續 P4a。

預計改動：附加規格凍結紀錄、Current State凍結表；新增 p4d.py（split／preprocess／queue）、p4d_train.py（固定更新與接續）、p4d_bundle.py（兩檔交付／排練）、本機評估adapter及test_p4d.py；knowledge簡短導航。驗證：RED→GREEN、壞形狀／邊界／未來mutation／queue SHA、CPU合成resume與排練負向、prepared唯讀split盤點、完整research tests、knowledge工具與diff檢查。所有生成物僅本worktree `.artifacts/p4d/` 或 `/tmp`，不向Windows正式交付。

## 分段凍結

Human 2026-11-24 對話「我通過了，可以開始，並且照你建議的順序做」；無數位簽章，確切時刻 UNKNOWN。全文預設規則的凍結身分為原規格 SHA-256 `c36aed96787e2b7e69fc92cd736ef3692c5542f1aecf350c4e0047470790fe54`。只核准第一段 baseline、GRU B前1/2/4/8種子與GRU A六格；C0+ baseline作新配對參考。Mamba2全部、超六格容量、D與N12核心搜尋維持DRAFT待決。原規則bytes作prefix保存，只附加本紀錄，禁止事後改規則。

## 驗證紀錄（持續補錄）

初始 RED：`/tmp/p4d-red.log`，pytest collection 1 error（p4d尚不存在）。尚未進行研究訓練、GPU、Drive或Colab操作。

## 最終結構與重大選擇

前述預計路徑經來源邊界盤點後調整：實際三個新增入口是 `tools/p4d.py`、`tools/p4d_train.py`、`tools/p4d_bundle.py`，本機專用評估器 `tools/p4d_evaluate.py`。不新增top-level目錄或worktree。原因是 `p4_bundle.source_paths()` 全掃 research_lab；把新來源放該處會改P4a相容宣告，即使新程式未被呼叫也不能冒稱相容。替代一是擴充／重認證舊來源宣告，增加不必要範圍；替代二是改舊來源掃描，仍會改其identity；均不採。P4a所有程式、原來源相容JSON與policy、64-fit queue、rolling_seq_scores／portfolio_eval／seq_runner／seq_models均維持HEAD bytes。

P4d重用safe-extract核心的獨立module instance，只有精確的3個tools來源與兩份P4d規格／契約額外路徑可入包；不改全域P4a validator，舊extract會拒絕P4d包。Notebook內嵌相同明列allowlist後才extract，仍驗完整inventory、每檔size／SHA、links／路徑／解壓上限。`p4d_evaluate.py`與p4b本機模組均不入包。新run保留原p4_source_compat來源及功能，但沒有認證P4d跨版升級；不默接不合來源。真實bootstrap已列入本機驗證，雲端仍UNVERIFIED。

固定更新用獨立訓練adapter，沿用SeqRanker、loss、sampler、warmup_cosine及Observer的驗證原語。anchor GRU baseline seeds17／29只用2021前BEST，嚴格改善才更新（同分取較早）；其餘種子按17／29交替對應，六容量各17／29；outer完全fresh、沒有validation早停，固定endpoint權重是該預先選定更新規則的產物。原八epoch cosine週期不縮短。C0+新baseline全列185欄，200上限、每25輪IC5、75輪patience，outer固定anchor選定iteration，沒有額外day-t fit或500k cap。此為預設baseline，非N12搜尋。

Checkpoint包含model／optimizer／scheduler／Python、NumPy、torch CPU／CUDA RNG與BEST／固定endpoint權重；先local serialize／fsync／hash，再durable copy／reverify及atomic pointer。latest／previous／selected保留，missing pointer＋orphans拒絕，不自動升格。完成標記驗bytes、模型／checkpoint binding、BEST或固定endpoint、history、prediction keys及support counts，不能只靠檔案存在略過。C0+未完成的CPU fit重新fresh執行，完整fit驗證後才略過，沒有偽稱iteration級checkpoint接續。帳本只記錄、不設時數／CU上限；85°C gate及推論windows guard保留。

預處理明確歸零36…47，48欄compact-JSON order SHA符合規格；沿用凍結非有限政策（nan/inf先轉0後每日average rank及normal ppf）。cache identity包含implementation bytes、raw X／feature order、schema、原日曆／ends、fit／validation／eval support與halo；逐session輸入／輸出hash及最終output hash，partial不是complete。新cache只在P4d namespace，不讀／寫prepared舊rankgauss cache；新raw prepare只驗證／複製原矩陣並建本機links，不呼叫舊mask偵測。各origin cache identity分開，可能有多份相同轉換副本，是目前明示的磁碟代價，未做跨support去重。

評估器要求完整140 fits、來源／queue／run／完成標記；先共同鍵再逐seed/day percentile-rank平均。B固定K1/2/4/8、8個單seed；A六格兩單seed與ensemble全部列出，與新C0+17／29配對。NaN標籤保留，IC使用共同y5/y10成熟列；全日未知IC為None並揭露，非0收益。主要格沿原P2函式與p3_pair；price snapshot／volume／Config／provenance缺失即不能組合評估。全部pool、成本／換手／回撤與資源／prediction correlation為描述性；20-session circular1000次、seed20261125、centered雙邊公式沿shared函式。**Holm2家族缺未核准Mamba2，第一段不改成Holm1，不宣告家族成功**；A/B不另添正式假設。沒有沿用P1舊分數作主對照。

## 唯讀prepared盤點與凍結

來源 `/home/frank/projects/p4-bundles/prepared`，X shape **7,423,927×48**，X SHA `9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363`。新盤點 [p4d-stage1-prepared-splits-20261125.json](../tasks/p4d-stage1-prepared-splits-20261125.json) 約15KB，記raw input identity、七origin日期／列數／sessions、鍵集合hash、保守label_end、halo、source／runtime，不含資料內容。此為GENERATED／CPU_SPLIT_INVENTORY，非研究績效或PIT證據。交付metadata綁定其完整matrix／split內容；現場準備split後須逐一相符才可排練／正式執行。

| origin | fit列數 | fit sessions | fit末日 | eval列數／sessions |
|---|---:|---:|---|---|
| anchor | 3,935,478 | 3,528 | 2020-05-06 | anchor val185,595／126 |
| 2021 | 4,208,219 | 3,780 | 2020-12-03 | 378,547／244 |
| 2022 | 4,585,264 | 4,032 | 2021-12-02 | 385,419／246 |
| 2023 | 4,966,433 | 4,284 | 2022-12-02 | 390,606／239 |
| 2024 | 5,355,936 | 4,536 | 2023-12-01 | 418,138／242 |
| 2025 | 5,771,347 | 4,788 | 2024-12-03 | 425,682／243 |
| 2026 | 6,190,955 | 5,040 | 2025-12-02 | 302,228／168 |

anchor fit起日2006-01-17，validation **2020-06-04～2020-12-03**，最後保守label_end session3942，2021首eval session3952；fit末session3785，val首3806，嚴格 `< next_first−20`。完整252-session年等值裁切，不把context halo算訓練年數。七origin盤點峰值RSS **2,670,220 KiB（約2.55 GiB）**；這只量metadata／SeqData，不等於完整185欄C0+可行性。label_end沿既有保守session+11，沒有新的逐來源execution標籤證據；strict PIT／tradability仍UNKNOWN／STOP。

## 工時與CU估計（不是實測或費用核准）

只讀主checkout `.artifacts/p4/p4a-fit-durations-min.json`。GRU W60 Q1/Q2各6 fits，均值9.4／8.7min，合併**9.05min/fit**，所列max10.5min。沒有重新驗原ZIP／逐fit帳本／環境，只有既有摘要估時計算，不代表P4d相同耗時。

| 第一段項目 | 去重fits | 9.05min等效T4h | 10.5min情境h |
|---|---:|---:|---:|
| GRU B（含兩baseline種子） | 7×8＝56 | 8.4467 | 9.8 |
| GRU A額外五格×兩seed | 7×5×2＝70 | 10.5583 | 12.25 |
| GRU合計 | 126 | **19.005** | **22.05** |
| 新C0+ baseline | 7×2＝14 CPU fits | UNKNOWN | UNKNOWN |
| 全佇列 | **140** | 不等於140次GPU fit | N/A |

沿規格暫估T4 **1.5–2.5 CU/h**，GRU baseline等效到觀測max情境約 **28.51–55.13 CU**。不是實際計費率或工時上界；不包括warm-up排練、預處理、完整train／validation IC、全量推論、編譯、IO與重試，也未套大容量倍率。**d256×2與d64×4耗時／T4顯存UNKNOWN，須由現場排練量測**；不根據CPU合成耗時外推GPU。

完整C0+最大fit的單一float32矩陣下限約 `6,190,955×185×4＝4,581,306,700 bytes`（約4.27GiB），hstack與LightGBM另有副本／binning／dense panels，實際峰值RAM未知；不得靜默恢復500k cap。全列train IC比P4a probe昂貴許多，可能主導時間，上述19.01h只能稱baseline等效，不能稱總工時保證。

## 自查與未驗證

使用ml-leakage-audit、colab-preflight、test-driven-development、debugging-and-error-recovery、verification-before-completion及code-review-and-quality作作者自查；沒有派agent，非獨立review。RED依序為缺p4d模組的1 collection error、缺train的2 failed／5 passed、缺bundle的2 failed／8 passed。後續實際失敗均保留於本機log：fixture的session差一、股票數不足IC門檻、C0+使用錯誤save_model入口（已改booster_.save_model）、一樹／過少fit列導致constant IC（增加合成支援）、搬移後子程序import舊路徑。沒有降低門檻或刪失敗測試。一次大合成全列推論測試主動中斷（exit130），改為先驗完整split、再獨立smoke支援的CPU單步；真全列推論仍待現場gate。

最新focused實測：**18 passed in45.12s**（後續完整回歸見補錄）。包含壞形狀、邊界、cache hit/rebuild、新程序future mutation、baseline selected epoch／weights一致、固定更新resume位元一致、seed／contract mismatch、corrupt／missing pointer／orphan、真SeqData三容量單步與C0+/outer評估、raw prepare不改來源、bundle safextract／本機來源排除、全部210描述曲線；**5項in-memory mutation killed**，另5項排練證據負向拒絕。CPU smoke使用縮小、重新標記支援（GRU fit130列、val65列，C0+ fit650列），不把它冒稱真全量排練。原P4a排練負向／來源相容測試包含於完整回歸，不取代P4d現場量測。

未驗：任何真研究fit、T4／CUDA bitwise／顯存／最大容量倍率、全列C0+ fit／RAM、完整train IC推論、Drive fsync持久性／斷線、Colab最新runtime相符、正式price snapshot／組合績效、獨立審查與Human acceptance。source PIT、universe／下市與execution provenance未因本工程解除。Mamba2全部、D、N12與超六格容量未實作或執行。

Knowledge-impact：分段凍結狀態、候選入口、split artifact／major blocker及next action＝YES；同步Current State／Authority Map／Research／Worktree Register。既有machine authority／production／active外部runtime／原閾值／deprecation＝NO；沒有更寫observed_at或authority-map.json。只保留未提交patch，非clean-worktree完成宣告。
