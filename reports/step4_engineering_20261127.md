# 第四步 4a：同業 h20 資料包與無標籤准入

**FROZEN / Q-A / ADMIT（限定歷史工程） / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。**
僅 CPU 工程證據；完成不等於 Human acceptance。歷史 Holm m＝2（1a、1b），季報不建 BQ 歷史模型；known_at UNKNOWN、strict PIT STOP、前瞻／交易 NO_GO。

## 契約與重用

規格 SHA `79a0c24bc57369d03edc709933b44c21a95e81f0c532127f72af93156c8e7a5d`；[Human 凍結紀錄](../tasks/decision-step4-freeze-20261127.md) SHA `18909bacde1cbe4096fb52b66317b6d56d25a51ee4ffc930a6f8fce04f8e7744`。規格前段保留起草時的未授權文字，§8 最後凍結紀錄及 Human 本輪有界指示控制本次執行；不修改凍結文件。

開工先核 project map、authority hierarchy、CLAUDE.md、ml-leakage-audit、第三步各模組／queue／evaluation／工程報告及相關 domain。既有 tools／tests／reports／knowledge 足以承載，不建立替代樹。第三步 admission 會 glob `step3*.py`，故新實作全用 `step4*.py`，不改原檔。既有 `.artifacts/step3/bundle-72` 不能作第四步輸出，因欄序、家族及來源 identity 不同；使用獨立 `.artifacts/step4/bundle/`，不覆寫任何共用來源。

重用 `step3.sha256/verify_file`、SEEDS／YEARS／PARAMS 與第三步 B 的 190 欄 fit／eval、sample／target／keys／label_keys／eval_labels／warmup／calendar。凍結規格要求所有臂重新擬合，因此第三步模型／分數一律不讀、不複用；本輪 fits＝0。完整日截面排名後抽樣的既有 target 原樣複製，不重新抽樣、排名或使用 eval 標籤成熟性裁鍵。

第三步 bundle SHA `97a9d3e8f43bfde984455f2439b0ba9bb22561beb3a65088dfbde27a7ea14613`；peer-v3 manifest SHA `08b6ecb48c5a99cc35c761a0a3a5f606ca6a2c40e7a55ae6000ad5663e76f248`；指定私有獨立來源複核筆記 SHA `d45ca09ddaaa5607900e8e0186b76ba4698da05decc43b8d7486eefebaeffe37`。來源全唯讀；私人逐列來源與完整 manifests 不進 Git。

## 矩陣表示及 gate

每折每 seed 的共同 190 欄基線只複製一份，eval 跨 seed 共用；五欄新區塊另存。108 個 logical jobs 以 component references 指向 fit／eval 基線及區塊，`step4.matrix` 可依臂組成 190／193／192 欄。不逐臂複製全矩陣，沒有產生模型或分數。bundle 自含後續組裝所需陣列、來源 manifests、peer-v3 產物、Human 紀錄及程式 snapshots；原始 provenance 路徑／hash 保留原身份，不將它們重新標為本案取得的資料。

原矩陣逐鍵左接，不因新特徵缺值或未知分類刪基線列。模型家數在分類／在市 UNKNOWN 或缺 join 時記 NaN；已知集合不足三家保留真家數；目標營收已可用時 spillover 回 NaN、已知家數保留。市場對照仍包含目標股。1b 稀疏不另創百分比門檻：凍結要求每個新欄在每個 fit／eval partition 至少兩個有限值且 min＜max；不足 STOP，保留 Holm 名額。

audit 核所有 member size／SHA、來源 pins、目前程式 bytes、欄序／float32／逐欄缺值比例、全部 v3→Step4 block 重播、第三步抽樣與共同鍵／基線／target／eval-label bytes、逐 sample `t+21<origin` 與 warmup。target 及 eval-label 只作 opaque 複製與 SHA 核對，audit 不 `np.load`、不計統計。缺 Human 精確紀錄不得 ADMIT；ADMIT 僅代表本 bundle 的限定歷史工程 gate，不是 fit 執行授權、獨立驗收或 strict PIT。

未來不變性本輪抽查 adapter 的日期／股票鍵連接：擾動未來 v3 列並亂序後重播截止日前抽樣。上游原始價格／營收事件／分類的未來擾動只引用已 pin 的獨立複核筆記，不聲稱本輪重做原始資料審查。

## 執行與驗證

Python：`/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，3.12.3、NumPy 2.5.4、pandas 3.1.1。本 worktree detached HEAD `7a2231e183b02e516be133483fec9de8c26e072e`，開工 clean；MARKETMAMBA_DATA_ROOT unset，Data realpath 為本 worktree 普通目錄，未讀其內容。CPU、CUDA_VISIBLE_DEVICES 空；無網路／GPU／外部觀測／agents。

```bash
PYTHONPATH=V6:V6/research_lab/tests:. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest -q \
  tests/test_step4.py tests/test_step3.py -p no:cacheprovider --tb=short 2>&1 | head -c 3000
```

初次 RED 為新模組不存在；後續整合查出 fixture 的未來擾動破壞最低家數約束，以及合成 output 不符私有 namespace，均修 fixture，未放寬 gate。合成反例包含 UNKNOWN 家數補零與 positional join 兩項實際函式替換 mutants；另覆蓋 sparse constant、重複鍵、少 job、wrong contract、corrupt target、錯共同鍵、缺 implementation。完整 admission fixture 在 np.load wrapper 中禁止 target／eval-label 檔，驗證成功路徑不讀標籤陣列。Builder 依 debugging／code-review-and-quality／verification-before-completion 做根因與五軸自查，沒有獨立 reviewer。

建置前估算 **19,339,037,266 bytes（19.34 GB）**，硬上限 60,000,000,000 bytes，另需 10% free-space margin；當時可用 822,272,286,720 bytes。prepare **232.3019秒、峰值RSS 1,380,792 KiB、manifest列記 artifact bytes 19,185,475,829**；後者為300個唯一member、不含bundle／preflight／admission及logs的小型metadata，全部仍低於60GB。audit **86.2699秒、峰值RSS 1,397,032 KiB**；逐檔size／SHA共300項，**ADMIT、stops=[]、fit_count=0**。兩個CLI均exit0，沒有中止／orphan／背景任務。完整log／manifest位於本worktree `.artifacts/step4/`。

Bundle SHA：`1e0a6b7537c11939b7fecd95232817addd3778735a8fdc1df6d7e6abc84c9ed6`；admission SHA：`43d8ceca7f9d1764489b82836c6ad95cef47185ad7b055b5ec635597356e71d0`。bundle保留 `PREPARED_NOT_ADMITTED` 的immutable建置狀態；准入結論只由另份admission.json綁定其SHA，不回寫建置manifest。

實際原矩陣7,423,927列，48個共用partitions（36 fit＋12 eval），108 jobs，12折共同eval鍵合計 **4,294,004**，signal末日 **2026-09-11**，未以v3新觀測延長。各臂前190欄由同一檔案提供並與第三步B SHA一致。新增五欄每partition全部具變異；下表只統計特徵有限支援，**未讀eval標籤數值或統計成熟標籤**。完整190＋5欄的float32／NaN比例見私有admission。

| 折 | 共同eval鍵 | 1a peer_return有限 | 1b spillover有限 | 三seed fit spillover有限範圍 |
|---|---:|---:|---:|---:|
| 2015 | 312782 | 282512 | 13987 | 9310–9369 |
| 2016 | 313060 | 284670 | 14349 | 13790–13872 |
| 2017 | 332634 | 303779 | 16127 | 16104–16246 |
| 2018 | 336574 | 307857 | 16590 | 18100–18224 |
| 2019 | 337437 | 307873 | 16607 | 19205–19343 |
| 2020 | 360897 | 328211 | 20852 | 20060–20372 |
| 2021 | 378547 | 342982 | 20806 | 21043–21343 |
| 2022 | 385419 | 350613 | 20726 | 21884–22156 |
| 2023 | 390606 | 356546 | 21887 | 22262–22791 |
| 2024 | 418138 | 380805 | 21838 | 23347–23487 |
| 2025 | 425682 | 388035 | 24609 | 23449–23632 |
| 2026 | 302228 | 274919 | 19410 | 23933–24187 |

adapter未來不變性：2015／2021／2026各抽512鍵（共1536），分別擾動191,765／212,359／209,359個未來v3列並亂序，截止日前 **0差異**。抽樣不是上游strict PIT證明；來源raw因果重播仍只引用指定獨立review。

終局 focused：**67 passed／39.84秒／exit0**（Step4 18項＋第三步49項）；相關peer回歸 **24 passed／1.40秒／exit0**。兩個mutants KILLED；五項admission資料／schema mutation均STOP，原始step3 tests與實作均未修改。syntax compile、local links、`git diff --check`通過。knowledge-impact初次PASS、加入worktree／knowledge後WARNING（人工review提示）；knowledge-health `--ci` exit0／Overall WARNING，Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS，Governance／Authority Drift／Legacy Docs WARNING，無structural FAIL。

交付前另一次純hash驗證 **300唯一member＋8個目前程式檔 PASS／30.2914秒**，規格／Human／原第三步與peer manifest pins再驗一致，bundle／admission SHA未變；沒有重跑prepare或audit。4個Python AST、18個新增local link targets、5個新檔whitespace及`git diff --check`終局通過。終局knowledge health仍exit0／WARNING。證據在`.artifacts/step4/final-verification.json`、`links-syntax.json`及`knowledge-health-final.log`；完整來源／outputs與本輪狀態封存在`.artifacts/step4/completion.json`。

## 限制與交接

沒有擬合、IC、eval 標籤統計、MarketMamba-blind、正式 Data／V6/models／TEJ raw 寫入、P4d、production／scheduler／wakeup／部署、commit／push／merge。未執行全 repository regression；依本輪指定跑 Step4＋Step3，相關 peer tests 另記實際結果。季報 Q-A 來源與覆蓋限制沿用凍結規格及既有季度版本報告，不生成 BQ 矩陣或歷史效果。

長期限制：研究分類 known_at UNKNOWN、v3 在市只是研究設計、6161 排除、月營收版本／約23%單位／scope 限制、observed calendar 非官方、universe／公司行動／tradability 未驗、歷史已曝光。Hash 與共同列檢查不能消除上述偏誤或將結果升格為確認性／交易證據。後續需獨立審查與 Human acceptance；4b 才可依後續指示執行 108 fits。

## ml-leakage-audit 逐項判斷

| 項目 | 本輪證據與界線 |
|---|---|
| 可知時間 | v3→Step4 全 block 日期／股票重播與未來列擾動；上游價格／事件／分類擾動沿用指定獨立複核，不聲稱本輪重跑。known_at UNKNOWN，strict PIT STOP。 |
| 標籤成熟／purge | h20／target原樣固定，逐sample依完整 frozen calendar 查 t+21＜origin；不重算或統計eval標籤。 |
| 尾端NaN | eval-label opaque bytes與第三步完全一致；全部W60鍵保留，不用成熟標籤或新欄缺值裁列。 |
| 標準化／rank | 不新增scaler，重用原先完整日rank後抽樣target，未重新排名。 |
| 跨股票時點 | 按同日股票鍵left join；原市場對照含目標、同業排除自身的上游語義不改。 |
| Universe／下市 | 原研究共同鍵不變；v3研究在市不是正式universe／tradability，存活者偏誤未解。 |
| 價源 | 不解析新raw價源、不換調整價，保留原價未含息與公司行動未知。 |
| 共同列 | 所有臂／seed沿用第三步B的keys／sample／target／label_keys／eval_label／baseline SHA；不裁交集。 |
| 種子排名 | 本輪沒有分數或排名；後續評估仍須共同成熟支援上重排種子，不能由本輪工程ADMIT推定已驗。 |
| 凍結／多比較 | 規格及Human SHA固定，歷史Holm2，門檻與前瞻Holm3不變；已曝光歷史不能恢復確認性。 |
| 獨立性 | builder自查，獨立工程審查待辦；既有peer來源複核GO不是本輪整包驗收。 |
| 壞形狀／mutation | UNKNOWN、NaN、稀疏常數、錯鍵、重複、損壞、缺job／code、錯契約、未來亂序均有合成反例；無真實fit或IC。 |

Knowledge-impact：Semantics YES（新adapter UNKNOWN家數）、Authority NO（production不變；新增branch-local契約候選）、Subsystem boundary NO、Active local runtime YES（本地build／audit）、Correctness gate YES（Step4 admission）、canonical artifact supersession NO（新GENERATED候選）、Deprecation NO、Major blocker／next decision YES（工程准入待驗）。Current State／Authority Map／Worktree Register／Features／Research已同change更新；Operations及GOVERNANCE經人工檢查，穩定操作規則無變，不需修改。未更新外部observed_at。

## 固定交接

- Completed：凍結契約、108-job component bundle、真實prepare＋audit各一次，限定歷史工程ADMIT；0 fits、0真實IC、eval標籤零統計。
- Changed：`tools/step4.py`、`tools/step4_prepare.py`、`tools/step4_audit.py`、`tests/test_step4.py`、本報告；Current State／Authority Map／Worktree Register／Features／Research共5份knowledge。
- Decisions：Q-A、Holm2；原190欄／抽樣／target／鍵重用但模型與分數不可重用；UNKNOWN家數NaN，各partition可觀測且變異，缺件STOP。Component representation避免逐臂複製。
- Evidence：300檔SHA／size、108jobs、48partitions、4,294,004 eval鍵、上列資源／support／future checks與67＋24 tests；私有bundle／admission／logs與completion manifest。
- Remaining：獨立工程審查、Human acceptance；4b的108 fits／歷史評估不在本輪scope，全部未執行。無全repo regression；Q-A季報僅沿用既有來源描述，未另做季度事件描述性產物。
- Blockers：本次歷史工程無STOP；strict PIT／前瞻／交易仍受版本、known_at、日曆、lifecycle、公司行動、universe及tradability gate阻擋。
- Next recommended action：獨立審查本patch及component bundle／admission；後續工作包依frozen規格擬合三臂共108個新模型，不沿用第三步scores。
- Project memory updated：YES，同change更新5份knowledge；knowledge_update: required，理由為新契約／adapter／gate／artifact及next action。
- Authority map updated：YES，僅新增branch-local候選及Human凍結導航，未改production machine authority。
- Git status：開工clean；終局5 tracked unstaged＋5 untracked、staged=0；依no-commit保留可審patch，不宣稱clean。
- Commit：N/A，Human禁止，HEAD仍`7a2231e183b02e516be133483fec9de8c26e072e`。
- Push：N/A，禁止；無merge／deployment。
- Worktree：既有`job-run-mwxtq8vx-c1a06e/builder_step4a`，detached；未建立／移除，patch與唯一私有證據保存前不可移除。
- Runtime/environment：WSL CPU／pilot09 Python3.12.3／NumPy2.5.4／pandas3.1.1，CUDA空、OMP／BLAS1；MARKETMAMBA_DATA_ROOT unset、Data realpath本worktree普通目錄。沒有網路／GPU／正式Data／V6/models／raw來源寫入、blind、P4d、production／scheduler／wakeup或agents。

## 主 checkout 重建紀錄（協調者，2026-11-27）

builder 在自己的 worktree 建出的 bundle 記錄的是 worktree 內的路徑，因此在主 checkout 以同一份程式（commit `819fe27`）重新執行 prepare 與 audit。本報告引用的 identity 一律以這次重建為準，已全部替換；先前提到的建置耗時屬於舊建置。

- bundle SHA：`1e0a6b7537c11939b7fecd95232817addd3778735a8fdc1df6d7e6abc84c9ed6`
- admission SHA：`43d8ceca7f9d1764489b82836c6ad95cef47185ad7b055b5ec635597356e71d0`，結果 ADMIT，無 stops
- prepare：339 秒，RSS 1.35 GB
- audit：102 秒，RSS 1.39 GB

獨立複核結果：4b GO（`.artifacts/papers/notes/step4a-review.md`）。重點如下：
- 第三步原始檔 0 筆不符；
- peer 對齊 840 鍵，值差異 0（3 鍵缺值，皆為 NaN）；
- 1,800 萬筆抽樣 fit 列，purge 違規 0；
- 1,536 鍵加入未來擾動後，結果差異 0；
- 4 類合成反例全部 STOP。
