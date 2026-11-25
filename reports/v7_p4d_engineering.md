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

## 第一段接續 stage1b — 2026-11-25

任務 `job-run-mwvoxf7h-970cae/builder_p4d_stage1b`，沿用指派 worktree；起始 clean、detached HEAD `b1871a99890f2bfd8c6f2201a770cbbac73eb341`（前輪 WIP），本輪僅交未提交 patch，不 commit／push／merge。原報告的 base35cd9b8／18 passed 為前輪紀錄；本段為本輪新證據。無額外 agents／GPU／Colab／Drive／資料取得／研究 fit／blind／正式 Data／models／production／scheduler／Windows 正式交付。

### 結構決定與改動

保留 `tools/p4d*.py`，不搬回 research_lab。實測 P4a `source_paths()` 為 **226** 個 Python 來源，加四份文件的相容 inventory 為 **230**；收集器只掃 `V6/research_lab`／`V6/experimental`，P4d 不在其中。P4d adapter 是獨立實驗／bundle 邊界，重用 research_lab 原語；三個 tools 精確來源與兩份規格／契約由獨立 allowlist 加入，`p4d_evaluate.py` 不入包。搬移會要求修改／重認證原 P4a 來源邊界；本輪無此必要，不用前綴排除或降低 gate 來維持舊相容宣告。解包後隔離 Python 已實際 import 三個 tools 模組並驗 contract／queue。原 P4a source／policy／相容宣告、rolling_seq_scores.py、portfolio_eval.py 與凍結規格未改。

`tools/p4d_bundle.py`：新增持久化排練標記流程，複製完整六個排練 fit 與評估產物、重驗後原子發布；重連不需舊 /tmp，但仍重驗 durable bytes、完整 schema／job／run／步數／完成與 checkpoint／評估重算。拒絕 None／空閘門結果、metadata symlink、publication 越界與錯 evidence class。副本與 marker 位於 P4d progress namespace，不共用 P4a ledger、不升格研究結果。失敗副本保留而不產生成功 marker，未新增清理行為。`tools/p4d_train.py` 只增加已知 P4d durable publication root 的驗證入口；訓練／選擇語義不改。下載格現在打包帳本、進度／排練證據及 bundle manifest，仍由 Human 自行關閉 runtime。checkpoint 與 same-source resume 沿用既有驗證；P4d 跨版本數值相容仍未認證。

### 實際命令與 RED／GREEN

先執行 Human 指定命令（另設 CUDA_VISIBLE_DEVICES 空）：

```text
PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
1265 passed, 4 skipped, 5 warnings in 645.58s (0:10:45), exit0
```

初始完整回歸 log `/tmp/p4d-stage1b-full.log`，沒有來源相容失敗，沒有放寬 gate。新增 marker／notebook RED 為2 failed；空 gate RED 為1 failed（實際呼叫 formal，正確被反例指出）。第一次修後 focused為24 passed／1 failed（notebook未顯示成功標記檔名）；補上使用說明後 stage1b 四項 **4 passed／21 deselected，3.34s**。完整修後回歸另於下方補錄；不把早期 focused 當最終全套結果。

五項 in-memory mutation 的真實 stdout（`/tmp/p4d-stage1b-green.log`；不改來源 bytes）：

```text
MUTATION split_boundary KILLED
MUTATION macro_schema KILLED
MUTATION seed_sequence KILLED
MUTATION queue_SHA KILLED
MUTATION rehearsal_bypass KILLED
4 passed, 21 deselected in 3.34s
```

切分 mutant 將 outer `>= origin-20` 改成 `>`，邊界列被錯放時原拒絕 oracle 失敗；macro mutant 清空 schema 名單、seed mutant 互換17／29、queue mutant 反序，皆由 frozen contract／queue SHA 攔下。繞過排練 mutant 確實呼叫 formal，負向 oracle 因未 STOP 而殺死該 mutant；不是單純測試被 mock 的函式存在。

### CPU prepared 排練的證據邊界

使用 `/home/frank/projects/p4-bundles/prepared` 唯讀載入真正 SeqData，先確認完整 anchor 與2021 outer split 逐欄符合凍結 inventory。全量推論數百萬列，不適合本輪30分鐘 CPU 工程時限；使用明確 `CPU_SUBSET_REHEARSAL`、`formal_compatible=false` 的最小支援替身，未嘗試完整研究訓練。保留真實原日曆、W60／48欄、原始特徵與標籤；僅65支完整歷史且 fit／validation 兩標籤有限的股票，fit最近10 sessions＝650列、validation 1 session＝65列、evaluation前10 sessions＝650列。outer validation為0。因果 halo 明列，截面 rank 與 C0+ market aggregate 在65股替身上重算，**不等同全宇宙正式預處理**；小矩陣 keys／support 改以獨立 binding 標示，未更寫正式 split／matrix。

harness 只替換 runtime／CUDA telemetry 邊界、prepare 為上述替身與 device=cpu，其餘沿用 `rehearse → execute_job → fit_gru/fit_c0plus → predict → validate_complete → descriptive_pair → validate_gate` 真路徑。六個 fit 各1 epoch／1 step或C0+一輪，未觸發正式 run。先前2次成功路徑約28.73／28.88秒，六fit／20筆daily；最終 source 的新實測身分另補錄。CPU memory telemetry 的0為測試邊界替身，不是CUDA實測0。C0+稀疏日曆的 empty-slice／degrees-of-freedom warnings 如實保留，未修改其實作或門檻。

首次 harness 因縮小支援時未沿用有限標籤資格而出現 `nonfinite training loss`，正確STOP；修正 harness 選股條件後重跑，未放寬 split 或 loss gate。focused另驗持久化副本重驗、移除原暫存 fit／evaluation 後仍可重用標記，以及損壞 durable 評估時移除標記、重新排練失敗則不得放行。

### Notebook 與交付自查

合成資料／無 prepared 的 source-only notebook 於 executable test 逐格 compile；prepared 身分版另在 `/tmp` 打包、解壓並以隔離 Python import 驗 contract／queue。均為一個 zip＋一個 ipynb＋兩行 README，不要求手貼hash；prepared版綁完整 matrix identity與七個frozen origins。**5個 code cells 全部compile**，解包manifest **236** 個成員（不含manifest本身），沒有本機評估器。以假的 google.colab.files 下載邊界執行真實下載格，ZIP內帳本／PROGRESS／marker／bundle-manifest **4個測試成員全部吻合、download呼叫1次**；沒有連線或寫入 /content／Windows／Drive。CPU這些檢查不代表已在Colab執行安裝／mount／T4訓練。

即時進度：正式runner沿用Progress、逐step／epoch訊息與約60秒PROGRESS.txt；140-fit帳本隔離／完整性與cached restart已有CPU orchestration tests。排練目前每個fit完成即flush，長時間排練中的手機heartbeat未新增；完整真資料setup／rankgauss／validation推論、T4排練等待與Drive快照仍需現場核對，不能把短CPU輸出當已實測60秒持久性。

### 未驗證、決策與 knowledge impact

CUDA／T4最大格顯存、耗時／數值、全列C0+185欄fit與RAM、完整train／validation／eval推論、Drive fsync／斷線持久性、最新Colab套件相符、正式price snapshot／組合層、獨立session審查、P4d跨版本接續認證、Human acceptance仍 **UNVERIFIED／Remaining**。Mamba2／D／N12／超六格維持DRAFT且未執行。source PIT／universe／下市／execution provenance仍UNKNOWN／STOP；CPU工程不解除strict executable／trading gate。

八問：交付語義=YES；authority=NO；subsystem boundary=NO；active外部runtime=NO；工程correctness gate（更嚴格）=YES、研究threshold不變；新增或supersede canonical artifact=NO（只有TEMPORARY CPU工程產物）；deprecation=NO；major blocker／next decision=YES（來源相容回歸閉合，現場與獨立審查仍待）。影響工具最初只看到tools／tests而為PASS，不能代替人工判斷；同patch更新Current State／Authority Map／Training／Research／Worktree Register與本需求註記。不改authority-map.json或任何外部observed_at。採code-review-and-quality作者自查，不等於獨立review。

### 最終工程 artifact identity

Local CPU observation：2026-11-25T05:52:43Z；WSL、Python3.12.3／torch2.15.0+cpu／NumPy2.5.4／pandas3.1.1／LightGBM4.7.0／pytest9.2.0。LightGBM與Colab frozen4.6.0不同，未更寫Colab契約、沒有現場相符聲明。

最終CPU summary `/tmp/p4d-stage1b-cpu-_cbw54pw/summary.json`：{"bytes": 112075, "sha256": "102deb56e046c6c98ae15728156fdf443a6dc85e902fea175e29fd0d89d55d8f"}；總計29.133秒。六fit耗時秒：{"gru:256:2:anchor": 14.668, "gru:64:4:anchor": 2.477, "gru:32:2:anchor": 0.592, "c0plus:64:2:anchor": 2.058, "gru:32:2:outer": 0.564, "c0plus:64:2:outer": 1.12}；每個GRU一step，C0+一iteration。20筆daily＝兩系統各10日；durable marker重用已重新驗證。

最終prepared notebook驗證：

```json
{
  "output": "/tmp/p4d-stage1b-notebook-yosdgpsr",
  "code_cells_compiled": 5,
  "bundle_files": 236,
  "frozen_origins": 7,
  "zip": {
    "bytes": 941968,
    "sha256": "4fb9ce0393e4396f8a02e69858d0ed26d2102d270e3c86cfaca4f3e619a52382"
  },
  "notebook": {
    "bytes": 8763,
    "sha256": "fb89f2560c25d090b4930316e84ea6408adb38fcb6ed75314123ed15684be5f9"
  },
  "download_cell_members": 4,
  "download_invocations": 1
}
```

Source bytes（與最終CPU binding逐一相符）：

```json
{
  "p4d.py": {
    "bytes": 13314,
    "sha256": "5df94ced0d62979de44468487229a3e999fe18aeb3340465de21ec67e0deccb1"
  },
  "p4d_train.py": {
    "bytes": 23152,
    "sha256": "39e43cec14c9fedf4af8f7d41adc21fcca73bf12cc8bae9d7957a1c7822b30a7"
  },
  "p4d_bundle.py": {
    "bytes": 18592,
    "sha256": "24440f70d6b43bc7800ff7e03fc0f7966893f071f5f9271f32c4436dcff891d3"
  }
}
```

重現harness僅在`/tmp/p4d_stage1b_cpu.py`與`/tmp/p4d_stage1b_notebook.py`；temporary產物非canonical artifact，未提交資料／權重。TEMPORARY logs：`/tmp/p4d-stage1b-cpu-final2.log`、`/tmp/p4d-stage1b-notebook-final2.log`、`/tmp/p4d-stage1b-green.log`。原始失敗log保留，沒有清理。

Knowledge工具：治理unittest **23 tests OK**；health `--ci --as-of 2026-11-25` **exit0／WARNING**（Governance、Authority Drift、Legacy Docs），Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS，無structural FAIL。impact最終WARNING候選Operations／GOVERNANCE已人工閱讀；本輪未改其production control plane或治理規則，不更新這兩份。受影響的Training／Research及project map已同步；外部observed_at不改。

補充驗證：第一次解包後隔離import的harness未加-B，產生__pycache__，後續verify正確報`executable cache forbidden`；正式notebook bootstrap本已設sys.dont_write_bytecode=True。只修temporary harness加`-I -B`，重新建立隔離包與解包目錄，完整inventory／SHA驗證通過；未放寬bundle gate、未清理原失敗目錄。最終syntax **3個修改Python檔**、contract／queue／frozen prefix SHA、**3個bundle來源hash**均通過；受影響Markdown本機引用 **765個、missing 0**；**8個protected source／contract／split artifacts與HEAD逐bytes相同**。

### 最終完整回歸與固定交接

```text
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
1269 passed, 4 skipped, 5 warnings in 617.86s (0:10:17)
exit0
```

最終log `/tmp/p4d-stage1b-full-final.log`，1273項collection、P4d25項均通過。新增4項，沒有新增skip／刪測試。4個既有skip：`test_delivery_nbformat_parser`、`test_notebook_gate_defaults_compile_and_validation`、`test_delivery_includes_generated_declaration`因離線runtime沒有nbformat（以-rs另核3 skipped／5.95s）；`test_official_mamba2_gpu_smoke`的原skipif為needs CUDA。未安裝套件；另唯讀probe既有marketmamba-gpu venv也沒有nbformat，未載入GPU。故nbformat parser與Colab真kernel執行仍未驗，不能只因code compile稱已過該parser。5個warnings為故意重複ZIP成員1項、stale-audit mutant內Pandas dayofweek deprecation4項，不是P4d gate放行異常。CPU替身單獨的稀疏panel warnings另依前節保留。

重新打包前後manifest逐內容相同，ZIP容器hash可變；使用上列最後一次artifact身份，不把容器差異冒稱source變更。所有實作在最終全套期間凍結；其後只補knowledge／本報告。

- Completed：本輪5項優先工作已完成工程實作／CPU檢查及如實列Remaining；不是Human acceptance，沒有正式交付。
- Changed：下列10個檔案，均為本輪unstaged patch。

```text
tools/p4d_bundle.py
tools/p4d_train.py
V6/research_lab/tests/test_p4d.py
docs/operations/colab-human-requirements.md
reports/v7_p4d_engineering.md
knowledge/00_Project_Map/Current_State.md
knowledge/00_Project_Map/Authority_Map.md
knowledge/00_Project_Map/Worktree_Register.md
knowledge/01_Domains/Research.md
knowledge/01_Domains/Training.md
```

- Decisions：保留tools精確來源邊界；持久化排練產物與嚴格marker重驗；不改研究切分／配方／threshold、不降低P4a來源相容gate。
- Evidence：初始1265／最終1269 passed、4 skipped；5具名mutation killed；CPU6 fits／20 daily、5 code cells compile、解包hash／import／下載格；knowledge23 tests、health WARNING無FAIL；實際hash與輸出見以上各節。
- Remaining：nbformat parser、CUDA／T4最大格與數值／耗時／顯存、全列C0+RAM、完整推論、Drive持久性／斷線、長排練heartbeat、最新Colabruntime、price／組合層、跨版相容認證、獨立review與Human acceptance。Mamba2／D／N12未核准且未實作。
- Blockers：上述現場與獨立審查證據缺口；source PIT／universe／execution仍UNKNOWN／STOP。沒有本輪CPU regression失敗待修。
- Next recommended action：不同session獨立審查patch／契約／實測證據，再由Human決定另案現場排練及正式交付；本輪不啟動任何外部工作。
- Project memory updated：Current State／Worktree Register／Research／Training；knowledge_update: updated。同patch同步，不新增重複memory樹。
- Authority map updated：只記候選工程與evidence邊界，沒有authority升格／authority-map.json或外部observed_at修改。
- Git status：detached HEAD、10 modified／0 staged／0 untracked，初始clean；依no-commit保留patch，因此不是clean-worktree完成宣告。
- Commit：本輪無；HEAD仍b1871a99890f2bfd8c6f2201a770cbbac73eb341（先前WIP）。
- Push：無；merge／deploy亦無。
- Worktree：`/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mwvoxf7h-970cae/builder_p4d_stage1b`；沿用指派worktree，未新增／移除。
- Runtime/environment：WSL／pilot09既有CPU venv，Python3.12.3、torch2.15.0+cpu、LightGBM4.7.0；CUDA_VISIBLE_DEVICES空，最終回歸OMP／MKL threads=1。MARKETMAMBA_DATA_ROOT unset，Data realpath為本worktree普通Data且未讀內容，只唯讀指定prepared；沒有Colab／Drive／Windows正式交付或外部runtime observation。


## 排練 heartbeat REQUIRED 與最大 fit 量測入口 — 2026-11-25

### 範圍、盤點與決策

本輪 `job-run-mwvqcr5w-e515cb/builder_p4d_fix1`，指定既有worktree `/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mwvqcr5w-e515cb/builder_p4d_fix1`，detached HEAD/base `d36e709a21088ce836c6d84e2964518c018cc12e`，起始clean。已讀AGENTS／CLAUDE、Current State／Authority Map、colab-preflight／Human需求第5節、frozen stage1 contract、訓練／bundle／tests與Training／Research domain。既有tools／tests／reports／knowledge與`.artifacts/p4d/measure-*` namespace可承載，不建立新top-level、worktree或替代experiment tree。量測輸出是新鮮隔離的CPU_RESOURCE_MEASUREMENT候選，非canonical研究artifact；測試輸出及logs為TEMPORARY。

此為原獨立審查NO_GO的bounded修補，**UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED / CPU_ENGINEERING_ONLY**，不解除NO_GO。無commit／push／merge／額外agent／GPU／Colab／Drive／網路資料取得／production／scheduler／wakeup／recurring jobs；未讀MarketMamba-blind，未寫formal Data／V6/models或修改rolling_seq_scores.py／portfolio_eval.py。MARKETMAMBA_DATA_ROOT unset；Data realpath仍為本worktree普通Data路徑，未讀內容，不推定等於正式資料。Runtime為WSL；最終指定CPU環境Python3.12.3、torch2.15.0+cpu、NumPy2.5.4、pandas3.1.1、LightGBM4.7.0；與Colab frozen LightGBM4.6.0不同，未改runtime契約／observed_at。

### 實作

- `tools/p4d_bundle.py`：新增launch，在binding／ensure_rehearsal之前啟動發布器，傳入六條排練及正式run；notebook改呼叫launch。rehearse直接呼叫也建立預設publisher，排練傳非持久化CpuLedger guard，不產生正式ledger。嚴格marker／publication gate與6-fit coverage不變。
- `tools/p4d_train.py`：P4d專用Progress繼承舊publisher，只加入mode／stage快照與TXT；不修改P4a共用來源inventory。沿用60秒daemon heartbeat、原子TXT／JSON及寫入失敗繼續。stage覆蓋binding、preprocessing、GRU training／validation／full train inference、C0+ features／fit、full row inference與rehearsal evaluation；正式runner使用同一publisher/context，無重啟heartbeat／帳本重置。排練不finish_fit累積正式throughput，排練ETA未知。
- 同一模組的本機CLI限制origin=2026，走相同load／split與完整fit rows、185欄build_features及固定200輪CPU LightGBM（最大預設iteration成本探針，不是另選anchor更新數）。不抽樣、不推論、不保存模型或績效；C0+ raw features不需rank-gauss，因此僅此量測分支跳過它。`measurement.json`含peak_rss_bytes（resource.getrusage ru_maxrss、明示程序生命週期峰值）、elapsed／prepare／features_and_fit秒數、matrix shape／bytes、seed／iterations、source commit＋working hashes、contract、input hashes、split、data realpath及環境。輸出目錄必須全新，避免覆寫共用產物；失敗不產生成功JSON。完整2026 fit沒有在本任務執行。

供Claude另案執行的本機命令（本輪未執行）：

```bash
PYTHONPATH=V6 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.p4d_train --measure-c0plus --origin 2026 --data-dir /home/frank/projects/p4-bundles/prepared
```

預設輸出`.artifacts/p4d/measure-<uuid>/measurement.json`；須新程序執行，process-lifetime RSS包括imports／讀取／split／features／fit，不是扣除baseline後的fit增量RSS，也不是T4可行性驗收。

### RED → GREEN 與實際證據

初始`/tmp/p4d-fix-red.log`：**4 failed, 25 deselected in 2.59s**，缺stage、launch及measurement_inputs重現。首輪新測試修後`/tmp/p4d-fix-green.log`：**4 passed, 25 deselected in 3.77s**。首輪focused實際為**1 failed, 30 passed in 74.46s**；失敗只因既有notebook文字斷言仍要求guarded_launch直接出現，獨立診斷重跑**1 failed, 30 deselected in 6.95s**確認。將該斷言更新為實際launch入口，gate仍在launch內且負向測試保留；未降低gate。最終程式／tests於完整回歸開始前凍結。

```text
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests/test_p4d.py -q -p no:cacheprovider
31 passed in 75.51s (0:01:15)
exit0

MUTATION rehearsal_heartbeat_disabled KILLED
MUTATION rehearsal_mode_erased KILLED
1 passed, 30 deselected in 2.07s
exit0
```

最終focused `/tmp/p4d-fix-focused-final.log`；兩項in-memory mutants `/tmp/p4d-fix-mutations.log`，只替換暫時namespace/class method，不改source bytes。假時鐘長操作驗60／120／180秒三次REHEARSAL preprocessing快照，並核FORMAL欄位與publisher/thread同一性；六條排練路徑均收到同publisher及非None guard，含預處理兩路。os.replace拋OSError時context與操作繼續，TXT／JSON欄位、原子寫入沿用舊實作。

36列合成fixture使用loader／split邊界double，實際build_features與LightGBM擬合（不是完整2026資料）：matrix_shape=[36,185]、matrix_bytes=26640、iterations=200、peak_rss_bytes=1336168448、elapsed_seconds=0.007365068013314158。iterations=200是配置的n_estimators上限；極小fixture可能因無可分裂節點提早停止，不代表200棵有效樹。此RSS包含該pytest程序先前測試／imports，只用來驗輸出欄位，不能當2026資源估計。JSON位於TEMPORARY `/tmp/pytest-of-frank/pytest-289/test_measure_c0plus_cli_tiny_f0/measure-tiny/measurement.json`，pytest可自行回收；非canonical artifact。

### 限制與knowledge-impact

真實2026全列features＋fit RSS／耗時尚UNKNOWN，Colab／Drive FUSE持久性、T4記憶體／全列推論與native長操作期間實際60秒更新仍UNVERIFIED。Python daemon heartbeat非硬即時保證，OS排程／GIL／Drive阻塞仍需現場排練；未聲稱CPU替身等於現場驗收。strict PIT／source provenance／universe／tradability原UNKNOWN／STOP不變；本輪不新增模型比較、組合收益或research conclusion，不重標歷史artifact。

Knowledge gate：進度語義YES、量測入口boundary YES、next decision YES；authority、production runtime、研究correctness threshold、canonical artifact supersession與deprecation均NO。focused後執行check_knowledge_impact.py自動為PASS／DEFAULT，但人工判定仍需同步Current State、Authority Map、Training、Research及Human需求實作狀態；不靠自動DEFAULT略過記憶更新。完整回歸、health、hash與最終Git交接如下節。


### 最終完整回歸、來源身分與固定交接

```text
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
1275 passed, 4 skipped, 5 warnings in 653.96s (0:10:53)
exit0
```

完整log `/tmp/p4d-fix-full.log`；較原1269項新增6項，0 failed／無新增skip。五warnings為刻意duplicate ZIP member一項、stale-audit mutant內pandas deprecated API四項；與原審查類型一致。focused為31 passed／75.51s；兩項新增in-memory mutants均KILLED，原五項mutation仍在全套內通過。完整回歸期間4份source／test SHA重驗未變，結束後再驗；其後只補文件。初始新測試使用marketmamba-gpu venv但停用CUDA，最終focused及完整回歸均使用上述指定pilot09 CPU venv（torch+cpu），未安裝任何套件或改環境。

來源／tests working bytes SHA-256（HEAD仍d36e709，不以HEAD冒稱本patch身分）：

```json
{
  "tools/p4d.py": "5df94ced0d62979de44468487229a3e999fe18aeb3340465de21ec67e0deccb1",
  "tools/p4d_train.py": "1cc307f3a96ea022fb43b329764b6d75e909b5f635c5fca09e3753b7608dce71",
  "tools/p4d_bundle.py": "76237e4e5cdf10624b7a235026a4458745d31ffc3f50e04bd3d2eb4088bf07cf",
  "V6/research_lab/tests/test_p4d.py": "2fbdcc5015adc46da2bc67c2e0b5df6d5af90c72e882c209ee2145af540bd49e"
}
```

保護核對：`tools/p4d.py`、P4a共用`p4_colab.py`／`p4_bundle.py`／`p4_source_compat.py`、`rolling_seq_scores.py`、`portfolio_eval.py`、`c0plus.py`及P4d frozen contract／prepared split共9份與HEAD逐bytes相同。修改Python3份AST parse通過；focused實際bundle封裝／safe-extract、五個notebook code cells compile通過；6份受影響Markdown本機引用746項missing0（最終新增引用另重驗）。`git diff --check`通過，無staged內容。治理unittest **23 tests OK**；正常測試後再執行impact工具，人工更新受影響knowledge。health `--ci` **exit0／WARNING**：Project Map、Agent Guides、Domains、Runtime Freshness、Generated Index PASS；Governance、Authority Drift、Legacy Docs WARNING，無structural FAIL。不更改既有warning／門檻以求PASS。

- Completed：bounded REQUIRED進度修補、MINOR量測入口、RED→GREEN／focused／完整回歸與記憶更新完成；不是Human acceptance。
- Changed：`tools/p4d_train.py`、`tools/p4d_bundle.py`、`V6/research_lab/tests/test_p4d.py`、`docs/operations/colab-human-requirements.md`、本報告、`knowledge/00_Project_Map/Current_State.md`／`Authority_Map.md`、`knowledge/01_Domains/Training.md`／`Research.md`，共9份unstaged patch。
- Decisions：保留P4d tools獨立來源邊界，共用既有heartbeat原語；排練至正式沿用同publisher，觀測不改研究／帳本／checkpoint authority。資源量測完整rows／200設定上限，不恢復cap。
- Evidence：上列實際RED、31 focused、1275全套、兩項新增mutants、23治理tests、syntax／hash／references／diff與health；evidence class僅CPU工程／合成資源欄位驗證。
- Remaining：真實2026全列量測、T4／Colab／Drive長操作heartbeat及持久性、獨立複審／Human acceptance。未做formal交付或模型／收益結論。
- Blockers：原NO_GO尚未由獨立複審解除；全列RSS／現場證據仍UNKNOWN／UNVERIFIED，不以CPU通過填補。
- Next recommended action：Claude獨立審查patch與原需求，另案於本機新程序量測最大fit；再由Human決定現場排練／交付，不能由本輪完成推得授權。
- Project memory updated：YES，Current State／Training／Research及本報告；不新增memory tree。
- Authority map updated：YES，候選邊界與下一決策；authority-map.json／外部observed_at不改。
- Git status：起始clean，結束detached HEAD、9份modified／unstaged，無staged；依no-commit授權保留patch，**不是clean-worktree**。
- Commit：N/A（禁止）。
- Push：N/A（禁止）；merge／deploy同未執行。
- Worktree：既有`builder_p4d_fix1`，base／HEAD d36e709a21088ce836c6d84e2964518c018cc12e，無新建／移除。
- Runtime/environment：WSL、指定pilot09 CPU venv／Python3.12.3／torch2.15.0+cpu／LightGBM4.7.0，CUDA_VISIBLE_DEVICES空；MARKETMAMBA_DATA_ROOT unset，Data realpath為指派worktree/Data，未讀或寫formal資料。與Colab契約不等價，無runtime observation升格。
