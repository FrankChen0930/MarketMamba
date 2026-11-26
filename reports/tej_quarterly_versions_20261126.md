# 季報版本對帳 — 2026-11-26

**DESCRIPTIVE_SOURCE_VERSION_AUDIT / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。
依 Human 已核准[整合規劃 §A、§E 工作包1](../tasks/plan-fundamentals-supplychain-integration-20261126.md)執行，不讀報酬、價格、標籤或 MarketMamba-blind。完成不等於 Human acceptance。

## 結論與版本邊界

自結 **63,266 列、2,581 公司**，正式單季表 **115,876 列、2,944 公司**（公司數沿用已驗來源報告）。3,898 重複列為 **1,949 公司季度、各 Y/N 兩種合併範圍**；同範圍版本重鍵 **0**。不是同一財季同範圍的兩次修訂證據。1,524 組同日、425 組異日，不能因時間先後把 N 當初版、Y 當更新版。

雙範圍之間數值不同的組數：營收1,372、毛利896、營益1,172、EPS156（含單側缺值差異；分母皆1,949組）。範圍差異本身就足以造成數值差異。

自結金額在乘1,000的診斷假設下，明顯更接近正式單季金額逐季累加的年初至今值。這是**累計口徑線索**，不是每列口徑字典或版本證明；自結資料不可直接當作單季欄位。EPS 不加總、不以累計 EPS 相減生造單季 EPS。

正式表首次／修訂後版本仍不可判定 **115,876 / 115,876＝100%**；自結首次／歷版也不可判定 **63,266 / 63,266＝100%**。可從這兩份快照嚴格重播歷史首次公告數值的版本 **0**。跨來源一致不能降低此比例。

## 實作、來源及環境

沿用既有 `tools/`、`tests/`、`reports/`；[既有 loader](../tools/tej_pit.py)支援單鍵月營收／正式表，不容自結雙範圍，故新增獨立[版本稽核工具](../tools/tej_quarterly_versions.py)，重用其 SHA／UTF-16 TAB／缺值／日期／數值規則；不改共享 loader 或帳本。`.artifacts/tej/quarterly/` 是 Human 指定的私有新用途 namespace，既有 `pit/` 與 `diag/` 證據原樣保留。

- 起始工作樹 clean，detached HEAD `9613273bf1bc2f902e1d856f60aefcfaf55e302e`；本輪禁止 commit／push／merge。
- CPU Python 3.12.3、pandas 3.1.1；執行檔 `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`。fetch venv 沒有 pytest，因此測試與正式稽核統一用既有 pilot09 venv；未安裝套件。
- `MARKETMAMBA_DATA_ROOT` unset；本工作樹 `Data/` 是普通目錄，未讀寫。FinMind 僅讀指定 Windows `Data/processed_v6/financials_raw.parquet`，不讀同目錄其他資料。來源 realpath、bytes、SHA 全記於[JSON](tej_quarterly_versions_20261126.json)。
- 自結 ZIP 使用 **SHA256SUMS.20261126**；原 **SHA256SUMS 未修改**。正式 parquet 與既有 producer manifest 對驗；本機自結 HTML 文件也對本機 docs/SHA256SUMS 驗證；無網路。各輸入前後 SHA 相同。
- 代碼字串保留前置零、空字串／`.` 缺值、非法日期／數值／同範圍重鍵 STOP。保留原始欄位、列號、來源 SHA、發布日及合併範圍。金額×1,000是尚待自結單位字典確認的診斷假設；EPS元／股不縮放；幣別 UNKNOWN，不換匯。
- 自結缺起日、Q/H、幣別與版本識別；FinMind缺合併範圍、幣別、Q/YTD及發布版本資料。正式表保留實際起迄、Q/H、幣別、範圍。所有配對都是**數值探查**，已驗同時點同口徑可比數為0。
- PRIVATE 所有逐列資料只在 `.artifacts/tej/quarterly/`。早期 `run-20261126/`、`validated-20261126/` 是保留的工程中間產物，不能當本報告最終證據；最終為 `release-20261126/`。

## 覆蓋與躍升線索

每季按公司去重；非已上市完整 universe 分母，也不代表可交易。

| 年 | Q1公司 | Q2公司 | Q3公司 | Q4公司 |
|---|---:|---:|---:|---:|
| 2013 | 399 | 417 | 357 | 580 |
| 2014 | 443 | 409 | 395 | 543 |
| 2015 | 459 | 450 | 439 | 594 |
| 2016 | 451 | 461 | 364 | 624 |
| 2017 | 470 | 435 | 426 | 634 |
| 2018 | 514 | 436 | 409 | 646 |
| 2019 | 512 | 432 | 336 | 488 |
| 2020 | 365 | 412 | 787 | 1930 |
| 2021 | 1530 | 1718 | 1523 | 2007 |
| 2022 | 1560 | 1849 | 1571 | 1993 |
| 2023 | 1622 | 1922 | 1775 | 2186 |
| 2024 | 1915 | 2219 | 1934 | 2270 |
| 2025 | 1973 | 2295 | 2003 | 2339 |
| 2026 | 2049 | 2375 | 2072 | N/A |

躍升集中2020Q3／Q4，並於2021各季延續。以2025/11–2026/10基本資料最新快照作**現況代理**，年度公司覆蓋如下；不可回填歷史市場／產業身分。

| 現況市場 | 2019 | 2020 | 2021 |
|---|---:|---:|---:|
| OTC | 241 | 771 | 801 |
| PUB | 32 | 44 | 47 |
| ROTC | 19 | 116 | 148 |
| TIB | 1 | 8 | 9 |
| TSE | 349 | 918 | 954 |
| UNPUB | 11 | 21 | 21 |
| nan | 42 | 78 | 75 |

現況產業代理，依2020對2019新增公司數排序前八類：

| 現況產業 | 2019 | 2020 | 2021 |
|---|---:|---:|---:|
| 生技醫療業 | 42 | 170 | 193 |
| 電子零組件 | 74 | 201 | 207 |
| 半導體業 | 68 | 175 | 185 |
| 光電業 | 41 | 119 | 125 |
| 建材營造 | 14 | 86 | 87 |
| 通訊網路業 | 22 | 85 | 85 |
| 其他 | 34 | 92 | 98 |
| 電機機械 | 43 | 101 | 104 |

增幅跨市場與產業，不像只有某一產業擴張。本機 `self_reported_quarterly_field_doc.html` 說明：自結主要自願，部分公司因特殊原因受要求；來源可包括重大訊息、法說或董事會財務資訊。**未找到逐年供應覆蓋政策／2020–2021強制規則的本機證據**，無網路核查；「董事會資訊揭露增加」「供應商收錄範圍擴張」均只是假說，不下強制原因結論。

## 發布時點

自結發布日缺值0；相對名目季末 n=63,266，最小1、中位40、p95=83、最大1,232日，負延遲0。採同公司、同名目季且**同合併範圍**配正式發布日 n=60,641：早於45,069、同日15,382、晚於190；中位提前3日，範圍 −567～+1,144日。另2,625自結列缺同範圍正式發布日配對。實際正式迄日配對亦無負延遲，完整年度分布見JSON。

這是匯出所載日期，無法證明當日看到的就是這些數值；同日無盤中順序。晚於正式／大於365日者隔離，不截短日期、不倒填。歷史自結日期與原始公告版本尚待工作包2核原件。

## 三方數值對帳

主鍵先用字串公司、名目期別及可得的合併範圍；正式實際期間不符名目期末者另列結構不合格，不升格可比。FinMind無scope，不能聲稱同合併範圍配對。JSON另列 `formal_structure_candidate` 分層、每年／科目／差異及完整雙向未配對分母；`triple_summary` 報三方同時有值與三值完全一致，verified_comparable皆0。

下表「大差異」含相對參考值 >1%、異號及參考零但左值非零；百分比以兩側 finite 為分母。outer列數包括雙向未配對與缺值，不是合格可比分母。金額精度帶≤500元、EPS≤0.005元，只供差異分類；EXACT、精度帶、≤0.1%、≤1%、>1%、異號、零分母與缺值各自互斥。FinMind重鍵排除 0 列。

| 對照 | 科目 | outer列 | 兩側finite | EXACT | 大差異／finite |
|---|---|---:|---:|---:|---:|
| FORMAL_VS_FINMIND | eps | 172,826 | 104,561 | 84,984 | 15,311 (14.64%) |
| FORMAL_VS_FINMIND | gross_profit | 172,826 | 100,127 | 95,498 | 3,776 (3.77%) |
| FORMAL_VS_FINMIND | operating_income | 172,826 | 102,070 | 96,245 | 4,573 (4.48%) |
| FORMAL_VS_FINMIND | revenue | 172,826 | 101,934 | 97,749 | 3,234 (3.17%) |
| SELF_FORMAL_SCOPE_VS_FINMIND | eps | 162,688 | 57,390 | 16,256 | 41,070 (71.56%) |
| SELF_FORMAL_SCOPE_VS_FINMIND | gross_profit | 162,688 | 50,490 | 12,389 | 37,400 (74.07%) |
| SELF_FORMAL_SCOPE_VS_FINMIND | operating_income | 162,688 | 54,284 | 13,430 | 40,345 (74.32%) |
| SELF_FORMAL_SCOPE_VS_FINMIND | revenue | 162,688 | 54,890 | 13,671 | 40,513 (73.81%) |
| SELF_VS_FORMAL | eps | 118,501 | 58,433 | 14,965 | 43,377 (74.23%) |
| SELF_VS_FORMAL | gross_profit | 118,501 | 51,435 | 12,535 | 38,158 (74.19%) |
| SELF_VS_FORMAL | operating_income | 118,501 | 55,164 | 13,385 | 41,186 (74.66%) |
| SELF_VS_FORMAL | revenue | 118,501 | 55,945 | 13,869 | 41,339 (73.89%) |
| SELF_VS_FORMAL_YTD_DIAGNOSTIC | gross_profit | 118,501 | 48,041 | 44,234 | 909 (1.89%) |
| SELF_VS_FORMAL_YTD_DIAGNOSTIC | operating_income | 118,501 | 51,673 | 47,839 | 1,521 (2.94%) |
| SELF_VS_FORMAL_YTD_DIAGNOSTIC | revenue | 118,501 | 52,383 | 49,023 | 638 (1.22%) |

YTD診斷僅對正式Q金額、同公司年／範圍／幣別，從Q1完整連續累加；缺季、缺科目、H或期間不符即缺值。它混有匯出時各期版本，只可辨識可能口徑，不能作歷史已知累計。EPS無YTD診斷。

## 帳本回放與首次／更新傾向

唯讀 `.artifacts/pit-ledger-v2`：`mode=ro&immutable=1`，拒絕live WAL/journal，驗 schema、SQLite、4-run chain、事件digest、不可變base/index hash及原有業務鍵稽核標記。未呼叫snapshot、未寫帳本；DB輸入前後SHA相同。

初始 financials **2,625,875列**；首個本機觀測 `2026-11-25T21:18:06.516630+08:00`，最後 `2026-11-25T22:51:48.668281+08:00`。21 REVISED＋19 FIRST_SEEN共40事件；FIRST_SEEN全屬非本次四科目的綜合損益歸屬母公司，四科目FIRST_SEEN=0。

實際對40業務鍵從不可變base開始，逐事件驗previous_json等於當時state，再套新值，**40/40轉移通過**。私有 `replay_versions.parquet` 留21起始版本＋40事件＝61版本；可按帳本run cutoff回放這些本機觀測，首次觀測之前不回填。完整其他起始存量仍可由既有帳本replay API重播，本輪未另輸出262萬列全表，未宣稱公司首次公告。

EPS修訂與正式表**精確數值**配對：

| 修訂類別 | 正式表符合 | 筆數 |
|---|---|---:|
| FLOAT_REPRESENTATION_ONLY | NEW_ONLY | 6 |
| NAME_ONLY | BOTH | 4 |
| NAME_ONLY | NEITHER | 1 |
| NUMERIC_CHANGE | NEW_ONLY | 10 |

FLOAT_REPRESENTATION_ONLY 指新舊數值差≤1e−12，只是描述分類，不改帳本事件或准入門檻。結果為5名稱變更、6浮點表示微差、10實質數值改變。正式表16筆僅符new、4新舊皆符、1皆不符；其中實質10筆皆偏向new，顯示正式匯出**可能含後來值**，但FinMind快取更動不等於公司更正，不能推廣成整表最新版。自結EPS對這21筆新舊皆不符，且自結可能為累計，無法佐證首次正式值。

## 三欄PIT來源建議

| 欄位 | 正式表候選用途 | 自結提前條件 | 排除／STOP |
|---|---|---|---|
| 毛利率 | 同季Q毛利／營收>0；98,823列只通過日期／結構候選，仍須逐版與當時非金融身分 | 原件證明單季、幣別、單位、同範圍同版及公開日；若為累計，只有同時點可知且已證可比的同期累計金額才可作差 | 目前自結未通過單季證據；H、期間不符、日期缺／長尾、範圍不明、金融身分未知皆缺值 |
| 營益率 | 同上；營業利益可負，營收須正 | 同上；不能拼不同範圍或後來修訂分子分母 | 未驗同版／單位／期間不得提前 |
| EPS年增差 | 兩年同季基本EPS相減，須同幣別與可比股數／追溯調整基礎 | 原件須明確單季基本EPS；兩年比較基礎及當時版本均可證 | 禁止累計EPS相減、跨版湊數；無股數可比證據維持STOP |

**沒有可由本輪直接指定為 strict 歷史可用的期間或公司：0。** 正式表98,823列的候選數未排除歷史產業未知，也未證版本，僅利於安排原件抽核。缺日期4,237、>365日3,298及非曆年／H沿用前報告隔離規則，不修改舊證據。

前瞻可使用經接收與驗證的帳本版本，僅從本機觀測截止往後生效；還須補scope／期間／幣別／股數及當時非金融分類。自結若日後核原件，應建獨立自結事件，正式報告另建事件；公告D嚴格之後首個合格交易日收盤才是最早訊號，實收／驗證較晚再順延，最早下一交易日開盤使用。無日曆或截止證據就不輸出可用時間。下一期應到仍缺即停止沿用，須先查公司法定財年期限，不能自行設固定天數替代。

## 重現、產物與驗證

```bash
CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.tej_quarterly_versions \
 --raw-dir /home/frank/projects/MarketMamba/.artifacts/tej/raw \
 --ifrs /home/frank/projects/MarketMamba/.artifacts/tej/pit/ifrs_quarterly.parquet \
 --financials /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/financials_raw.parquet \
 --ledger /home/frank/projects/MarketMamba/.artifacts/pit-ledger-v2 \
 --out-dir .artifacts/tej/quarterly/release-20261126
```

禁止覆寫已存在namespace；重跑須換季度目錄內明確run identity。manifest SHA `8402c8265870a4e9bb2188d75f0491ef3b7d6238b40ee3f1c1a255a50f269507`。source、config（工具常數／SHA）、環境、輸入、輸出及evidence class已留JSON；model/split N/A，無訓練。

| 私有產物 | 列 | bytes | SHA256 |
|---|---:|---:|---|
| self_reported | 63,266 | 4,667,327 | `667f7d700467973e3c5ecfb30f35d58833956632dca0b13f7ceb24c04a7d995a` |
| timing | 63,266 | 4,958,673 | `5ead202f50927d97671d58dddc5096798fd4e32ec47b27cdc8dfefa9bcc63b73` |
| comparisons | 2,171,563 | 25,932,398 | `648d9b2aa916f5944f2862265a5ad2fca3aa95d2de624b0ef99cba403e9a8cea` |
| ledger_events | 40 | 12,389 | `942de94c6c208746e05ee64a0af9fb4803c22ee522e9a12cfb1ebae3a40017f3` |
| replay_versions | 61 | 6,131 | `79541e7182c91f9fd2fe7b0740adcf5ce3f01fd70eab6007ccdf4a5b436f32ed` |
| formal_admission | 115,876 | 3,492,403 | `f56b57fe04b7407638fddcdeb131aae46068e4352fc6084e03f71d3473b08c16` |

測試與治理最終結果見下方交接；測試皆為合成資料，來源只有本次指定檔案。兩項in-memory loader mutation為發布日缺值倒填、EPS×1,000，皆由數值／缺值斷言攔截，不修改實際檔案。另涵蓋前置零、scope重鍵、非法／缺失／早發布日期、零分母、異號／缺值／非有限值、雙向分母、連續累計缺季、科目隔離as-of及真SQLite合成回放／唯讀雜湊。

## 固定交接

- Completed：工作包1資料層、版本對帳、候選來源建議；工程完成不等於驗收。
- Changed：tools/tej_quarterly_versions.py、tests/test_tej_quarterly_versions.py、本報告／同名JSON、Current_State.md、Authority_Map.md；私有逐列產物只在指定quarterly namespace。
- Decisions：不把scope雙列當修訂；自結累計疑點保留；strict歷史季報STOP，沒有改正式准入門檻。
- Evidence：最終manifest與上列實際數據；最終測試／治理結果在完成前追加。
- Remaining：工作包2原件逐版、單位／Q或YTD／EPS基礎確認、歷史產業、強制揭露政策成因；獨立review。
- Blockers：首次正式數值與版本對應未知、現有自結無同範圍歷版；不能從日期或一致值消除。
- Next recommended action：Human／Claude獨立檢查本patch與彙總，之後依工作包2核自結及正式原件，不讀報酬。
- Project memory updated：Current_State.md；本工作樹candidate狀態。
- Authority map updated：Authority_Map.md；僅新增branch-local導航，不supersede任何machine／production authority。
- Git status：起始clean；結束保留核准no-commit改動，未stage；非clean狀態為明確交付限制。
- Commit：未執行（Human禁止）。
- Push：未執行；merge／deploy亦未執行。
- Worktree：既有builder_quarterly_versions；不建立或移除worktree。
- Runtime/environment：WSL CPU／既有pilot09 Python；無GPU、網路、資料取得、排程、正式Data／V6/models／P4d／第三步bundle／blind變更，無新observed_at runtime宣告。

### 最終驗證與 knowledge-impact 判斷

- `CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest -q tests/test_tej_quarterly_versions.py tests/test_tej_pit.py tests/test_pit_ledger.py`：**86 passed，24.06秒，exit0**。其中本輪16項測試、兩項實際loader函式原碼in-memory mutation均由assertion攔截；其餘70項為既有TEJ／帳本focused regression。初次測試在尚無新模組時collection失敗，之後實作；未執行可能讀真實標籤／矩陣的全repo suite。
- 語法：`python -m py_compile tools/tej_quarterly_versions.py tests/test_tej_quarterly_versions.py`，exit0。
- 最終10個輸入identity、6份逐列輸出與manifest SHA重驗、Parquet列數逐份相符；比較列2,171,563、自結／時點各63,266、帳本事件40、回放版本61、正式候選115,876。loader hash符合manifest；所有來源前後bytes hash相同，原SHA256SUMS與帳本DB未改。
- PUBLIC JSON遞迴檢查無stock_id／key／value_json／previous_json／old_value／new_value逐列欄位，新增Markdown相對連結全部存在；`.artifacts/tej/quarterly/release-20261126/self_reported.parquet` 經 `git check-ignore` 確認忽略。
- `check_knowledge_impact.py`：更新兩份project map前PASS；更新後WARNING（canonical knowledge要求人工review GOVERNANCE.md）。已讀GOVERNANCE.md；僅新增未提交候選導航，不改authority hierarchy、machine contracts、threshold或supersession，故不修改治理規則。
- `knowledge_health.py --ci`：**exit0，Overall WARNING，無FAIL**；Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS。Governance／Authority Drift警告為8個AUTHORITY_REPIN_CANDIDATE；Legacy Docs警告為README／OVERVIEW／PROJECT的既有舊架構、欄數或daily描述，超出本包且不自動repin／改寫。
- Knowledge-impact逐項：新增診斷語義YES；authority變更NO；subsystem boundary變更NO；active runtime state變更NO；既有correctness gate變更NO；新增／supersede canonical artifact NO（僅未驗收GENERATED私有稽核）；deprecate NO；major blocker／next decision資訊YES（scope pairs／YTD線索／首次版未知）。因此本change更新Current_State及Authority_Map，不使用NO_KNOWLEDGE_CHANGE_REQUIRED。
- `git diff --check`通過。起始沒有dirty／untracked／staged；結束2份tracked knowledge修改、4個本輪新增檔案，無stage。Human明確禁止commit，故**未達clean-worktree**；這是交付狀態限制，不以自動commit消除。沒有其他檔案變更。
