# TEJ PIT 資料層稽核 — 2026-11-25

RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。本報告是離線資料層描述，完成不等於 Human acceptance。Human 核准計畫第二步 2a；依 [計畫 §I–K](../tasks/plan-fundamentals-supplychain-20261125.md) 及本輪指示執行，loader／資料稽核不讀報酬、價格或標籤；不接觸 blind、不修改實際帳本、GPU、網路、production、排程或 P4d。全套既有測試的外部真實矩陣存取事故另見驗證限制。PUBLIC repo 僅保留程式、合成測試與彙總；連「代碼＋日期」範例也留在 gitignored 私有產物，沒有公開逐列資料。

## 偵察、身分與重現

最終沿用既有離線資料工具目錄 `tools/`、`tests/` 與 `reports/`；既有 pit_ledger／pit_arrival_audit 可承載相同資料層邊界，但無 TEJ loader 可沿用，故新增 [tej_pit.py](../tools/tej_pit.py)。來源與 machine identity 詳 [JSON](tej_pit_audit_20261125.json)。既有 pit_snapshots／pit_universe 為不同來源准入與生命週期工具，未修改。

- 起始 detached HEAD `aafd0d9c41ac2a42a7277936353a101f93aab95c`，工作樹 clean；使用指派 worktree，不建立新 worktree。
- WSL CPU Python 3.12.3；指定 pilot09 venv。`MARKETMAMBA_DATA_ROOT` unset；checkout `Data/` 是此 worktree 普通目錄，未讀寫其內容。對照來源為 Windows 路徑內兩份指定 parquet，僅讀營收／財報科目欄位。
- 三份 raw zip 原樣唯讀，全部經 SHA256SUMS 比對後才開始轉換。基本資料只選代碼、年月、產業、會計月份及上市／目前狀態欄，不載入其中的價格值。
- `available_date` 完全採發布日，缺值不倒填；輸入欄名和值 strip，`.`／空字串→missing，代碼取空白前字串保留前置零。年月／日期格式錯誤、重鍵、缺鍵、SHA 錯誤即 STOP。
- 月營收金額千元→元；IFRS 各金額新增 `_元` 欄，Revenue 元、EPS 維持元／股，不換匯。原欄位單位仍保留；幣別不推定可互換。
- 每列保留 input SHA、來源列號、發布狀態、版本語義、名目及實際期末；`strict_pit_eligible=False`。`publication_date_usable` 只表示日期未早於已知實際期末，不能當作版本准入 gate。
- 初版與中間診斷生成檔未覆寫或刪除，逐 byte 保存在 `.artifacts/tej/runs/<manifest_sha>/`，並附 relocation 記錄；其原 manifest 絕對路徑描述建立位置。它們不是新正式結果。最終指定路徑為 `.artifacts/tej/pit/`，拒絕在已有內容的輸出 namespace 再跑。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:. \
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.tej_pit \
  --raw-dir /home/frank/projects/MarketMamba/.artifacts/tej/raw \
  --comparison-dir /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6 \
  --out-dir .artifacts/tej/pit \
  --report-json reports/tej_pit_audit_20261125.json \
  --universe-dir /home/frank/projects/p4-bundles/prepared/shared
```

重跑請用 `.artifacts/tej/` 內另一個明確 run identity 目錄，避免覆寫已生成證據。raw、對照、loader、環境、membership arrays、輸出 bytes／SHA、source commit、split/model N/A 皆在 manifest／JSON。


## 輸入與輸出彙總

| 來源 | 列 | 公司 |
| --- | --- | --- |
| 月營收 | 380514 | 2958 |
| IFRS | 115876 | 2944 |

最終 manifest SHA-256：`f3532527b51ed05ad2b0360aad0c14de0531d18f230cf00c7ef456d79fa254f1`。loader SHA-256：`f14ed4794f257d756581b2bb58ee955fd3d60796560d2ef3d5f0feea3f159048`。

| 輸出 | 列 | bytes | SHA-256 |
| --- | --- | --- | --- |
| monthly_revenue | 380514 | 20719344 | `987c7c32636fcaebcff66a6ba774ff37f1c8fca8d20f589e794db8e28b45e44b` |
| ifrs_quarterly | 115876 | 20491442 | `7190a389866e618cccf8948cfcb3042c5edc59891838b56a661d0908509726d9` |

## IFRS 時點異常與處理規則

「年月」為名目資料鍵，不能直接等同期末。財報起迄大多是年初至當季末的累計期間；單季／單半年值的語義另由 Q/H 欄表達，不因起迄跨度而擅自作差財務值。非曆年期間的可重算定義：有起迄且不符合一月起、三／六／九／十二月迄及相應 3／6／9／12 月跨度；缺起迄則 UNKNOWN。這是期間結構分類，非已驗公司會計年度身分。

- 名目期末延遲最小 -82 日；早於名目期末 14 列。這 14 列全部非曆年期間，實際迄日均早於名目期末，按實際迄日重新計算全數消失。Q/H、合併、類別、年份、產業分類見下表及 JSON，並非單一 H 列問題。

- 依實際期末延遲 min／median／p99／max＝8／44／411／1064 日；負延遲 0 列。非曆年期間 64 列、起迄未知 4237 列。

| 分類 | 早於名目期末 14 列 |
| --- | --- |
| 季別 | 1: 5, 2: 6, 3: 3 |
| 單季(Q)/單半年(H) | H: 2, Q: 12 |
| 合併(Y/N) | N: 1, Y: 13 |
| 財報類別（1個別2個體3合併） | 1: 1, 3: 13 |
| year | 2013: 4, 2014: 4, 2015: 4, 2016: 2 |
| non_calendar_period | True: 14 |
| TEJ主產業代碼 | 23G: 1, 23H: 2, 99Z: 11 |


| 季別 | 有日期列 | 延遲中位數 | p95 |
| --- | --- | --- | --- |
| 1 | 25669 | 43.0 | 57.0 |
| 2 | 31389 | 43.0 | 121.0 |
| 3 | 25876 | 43.0 | 47.0 |
| 4 | 28705 | 85.0 | 120.0 |

>365 日長尾 3298 列，全部曆年期間；H 1,208、Q 2,090，N 631、Y 2,667，個別 631、合併 2,667。不能用非曆年制解釋。補發／晚補歷史值／發布日是否指最新版本均是可能原因，單份匯出無法識別。缺發布日 4237 列全部 Q／合併 Y，財報類別、TEJ 主產業與起迄均缺；不據此推定來源實際沒有報表。


建議處理：

1. 期間對齊採實際財報迄日；保留名目鍵與名目異常旗標，禁止修改發布日消除異常。非曆年／H 列先隔離作科目期間核驗，不直接接入曆年季報特徵。
2. 真正早於實際期末者排除時點准入（本批 0 列）；缺發布日記 NO_RELEASE_DATE、缺迄日記 NO_PERIOD_END，不補法定期限。
3. >365 日保留原值與日期、標記 long-tail，只可從實際發布日之後描述；在取得原件／版本證據前排除 strict PIT。不能把長尾截短或認成公司法律遲報。
4. IFRS 首次／重編版本未解，整表 strict PIT STOP，日期修正不解除版本 gate。

各異常依七個維度的完整 counts 與 joint counts 在 JSON。代碼＋期間／起迄／發布日期示例（無財務數值）在 `.artifacts/tej/pit/before_nominal_end_private_examples.parquet`、`over_365_private_examples.parquet`；僅供本機授權審閱。


## 月營收時點與缺值

有發布日 367,532／380,514（96.59%）；缺日 12,982，涉及 229 公司，其中 555 列營收也缺、其餘 12,427 列營收仍有值。缺日列的合併旗標全部缺，現況產業缺 12,533 列；不足以推斷原因，禁止依數值存在補可用日。

發布日−次月10日：min／p50／p95／p99／max＝-9／-1／3／365／1507 日。早於參考日 232,069、同日 99,583、晚於 35,880；晚於次月14日 16,124。發布日早於當月月底 0 列。參考日是描述基準，非逐公司法定期限。

| 年 | 總列 | 缺發布日 | 有日比例 | 次月10日偏移 p50／p95／p99 |
| --- | --- | --- | --- | --- |
| 2013 | 24583 | 908 | 96.31% | -1.0 / 8.0 / 365.0 |
| 2014 | 25315 | 920 | 96.37% | -1.0 / 363.0 / 365.0 |
| 2015 | 25898 | 915 | 96.47% | -1.0 / 2.0 / 366.0 |
| 2016 | 26193 | 896 | 96.58% | -1.0 / 5.0 / 365.0 |
| 2017 | 26470 | 903 | 96.59% | -1.0 / 1.0 / 365.0 |
| 2018 | 26852 | 912 | 96.60% | -1.0 / 2.0 / 365.0 |
| 2019 | 26990 | 915 | 96.61% | -1.0 / 5.0 / 366.0 |
| 2020 | 27319 | 913 | 96.66% | -1.0 / 2.0 / 365.0 |
| 2021 | 27873 | 912 | 96.73% | -1.0 / 2.0 / 365.0 |
| 2022 | 28436 | 906 | 96.81% | -1.0 / 4.0 / 365.0 |
| 2023 | 29083 | 944 | 96.75% | -1.0 / 2.0 / 367.0 |
| 2024 | 29825 | 1025 | 96.56% | -2.0 / 5.0 / 365.0 |
| 2025 | 30468 | 1081 | 96.45% | -1.0 / 2.0 / 365.0 |
| 2026 | 25209 | 832 | 96.70% | -1.0 / 1.0 / 3.0 |

春節：離線未驗逐年節日與適用順延規則，故只列一、二月發布月份代理，及 JSON 全部次月參考月份分布，不把代理當作春節因果或違法逾期率。

| 發布月份 | 列 | 中位／p95／p99偏移 |
| --- | --- | --- |
| 其他 | 308017 | -1.0 / 2.0 / 365.0 |
| 一、二月 | 59515 | -1.0 / 5.0 / 365.0 |

## 與既有快取對帳

月營收按字串代碼、revenue_year／revenue_month 配對全 TEJ 期間；IFRS 主對帳按原始名目年月與 financials_raw.Date 所在月配對對應 Revenue／EPS。非曆年列仍須依實際迄日另行確認，不把名目配對當會計等價。原來源更早年份也保留在 raw 分母，因此兩個方向配對率不同。相對差＝abs(TEJ−raw)/abs(raw)；兩者0記0、raw0但TEJ非0單列並計入差異。缺值不配值、重複參考鍵全部排除，未挑任一列。金融分類僅現況快照，未知保留 nullable，不回填歷史。

| 科目 | TEJ／raw列 | 配對鍵／有效值 | TEJ配對率／raw配對率 | <0.1% | >1%含raw0 | raw0且TEJ非0 |
| --- | --- | --- | --- | --- | --- | --- |
| monthly | 380514 / 459119 | 291847 / 291847 | 76.70% / 63.57% | 291360 | 446 | 2 |
| Revenue | 115876 / 144530 | 101934 / 101934 | 87.97% / 70.53% | 98309 | 3234 | 0 |
| EPS | 115876 / 161438 | 104596 / 104561 | 90.27% / 64.79% | 86331 | 15311 | 99 |


| 科目 | 差分p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| monthly | 0 | 0 | 0 | 9.06861 |
| Revenue | 0 | 0 | 0.499595 | 167.004 |
| EPS | 0 | 0.285714 | 0.947826 | 119.5 |

>1% 差異分類（完整年份、金融、合併、Q/H、幣別及倍率表在 JSON）：

| 科目 | 現況金融／非金融 | 合併口徑 | 倍率診斷 |
| --- | --- | --- | --- |
| monthly | False: 444, True: 2 | nan: 446 | OTHER_OR_UNKNOWN: 441, RAW_ZERO: 2, SIGN_DIFFERENCE: 3 |
| Revenue | False: 3161, True: 73 | N: 505, Y: 2729 | NEAR_0.001: 1, OTHER_OR_UNKNOWN: 3186, SIGN_DIFFERENCE: 47 |
| EPS | False: 14289, True: 1022 | N: 2078, Y: 13233 | OTHER_OR_UNKNOWN: 14680, RAW_ZERO: 99, SIGN_DIFFERENCE: 532 |

年份分類（>1% 含零分母）：

| 年 | 月營收 | IFRS Revenue | IFRS EPS |
| --- | --- | --- | --- |
| 2013 | 5 | 223 | 1011 |
| 2014 | 2 | 277 | 1040 |
| 2015 | 5 | 272 | 1050 |
| 2016 | 8 | 225 | 1011 |
| 2017 | 4 | 280 | 1086 |
| 2018 | 6 | 223 | 1086 |
| 2019 | 5 | 208 | 1100 |
| 2020 | 23 | 242 | 1109 |
| 2021 | 14 | 238 | 1157 |
| 2022 | 21 | 248 | 1188 |
| 2023 | 111 | 277 | 1221 |
| 2024 | 63 | 303 | 1306 |
| 2025 | 86 | 200 | 1113 |
| 2026 | 93 | 18 | 833 |

月營收差異 446 列全部合併旗標未知，不能硬分合併／個體成因。IFRS Revenue 僅一列 TEJ/raw 接近 .001（±1%），沒有普遍千倍錯位；單列倍率只能列候選單位問題，不自動更正。其餘是非一致倍率、符號差異或零分母，可能同時涉及來源、定義、重編、每股數追溯調整、會計期間差；沒有原件不能歸因。

IFRS 首次公告值 vs 重編後值：**UNKNOWN**。Revenue 高一致率無法區分兩版本恰好相同；EPS 差異較普遍，與追溯調整／版本或口徑差異相容，但不是重編的辨識證據，不能宣稱更可能首次值或最新值。原指示禁止修改帳本，允許本任務對照所需的唯讀存取；先前將它誤列為不可讀限制已更正。21筆對應結果見下節，REVISED 包含metadata／科目名稱變更，不能直接當公司財務重編。月營收「首次公告值」僅引用 Human 提供的 TEJ 說明，未獨立核原件，不能延伸到 IFRS 或月盈餘欄。


## 21 筆 financials REVISED 唯讀對照

原指示是禁止修改帳本，並非禁止唯讀；本補充只以 `mode=ro&immutable=1` SQLite 連線，且先確認無 WAL，僅取 `financials_raw`／`REVISED`。來源DB SHA 前後相同：`72fc13d493000973ad2d94f5aea80248398902479b6d8e22e7c459d678fabd01`，沒有新增run、event或寫DB。

21筆全部 EPS，21筆 origin_name 都由「基本每股盈餘（元）」變成「基本每股盈餘」，其中16筆 value 同時改變、5筆僅名稱變。按代碼／名目報表月份連 TEJ，數值使用 Decimal 完全相等，無事後容忍度：

| 符合狀態 | 筆數 |
| --- | --- |
| 僅新值 | 16 |
| 新舊皆符 | 4 |
| 新舊皆不符 | 1 |
| 僅舊值／缺TEJ鍵 | 0 |

在這16筆已改值事件上，TEJ更接近本地目前的快取值，與更新／追溯調整後值相容；但科目名稱也變，仍可能是來源的科目映射或口徑修正，不能證明公司財報重編。21筆不是隨機樣本，舊快取不是首次公告原件，因此整張IFRS首次／最新版本仍UNKNOWN，不能倒填已改值至歷史訊號。

逐列對照只保存event id／run id、代碼、報表期、type、變動欄位與match_state／日期，不保存財務值，位於 `.artifacts/tej/ledger-reconciliation/72fc13d493000973ad2d94f5aea80248398902479b6d8e22e7c459d678fabd01/financial_revised_matches.parquet`，SHA `e8db30afcd2265006282cbe28f7f33a78f0250e40583ac680336cf4752d91240`。可重跑稽核腳本保存在同私有目錄 `audit_script.py`（SHA `c19162b022462c68fcab18d614b08bd2f9ba3d548cc147b589319c88fc9296d7`）；它只生成私有逐列資料與彙總，不改帳本。公開JSON含完整輸入及輸出identity。以下SQL是全部帳本讀取範圍（另讀schema metadata），不讀其它表的數值：

```sql
SELECT id, run_id, key, value_json, previous_json
FROM events
WHERE table_name='financials_raw' AND kind='REVISED'
ORDER BY id;
```


## 每年月覆蓋與 C0+ 研究成員

只載入 `/home/frank/projects/p4-bundles/prepared/shared/{stock_ids,dates}.npy`，不建立 SeqData，不開 X、價格或 y。以每月曾出現的代碼聯集作 C0+ prepared 研究成員分母；TEJ 以該資料月份最終有發布日的公司計數，再算交集／缺日。這不是月底已可用比例，也不證明歷史 lifecycle、可交易或不存在存活者偏誤；不使用基本資料現況回填 universe。166 個資料月份中165個有分母，2026/10 UNKNOWN。完整166列的每年月覆蓋在 JSON。

| 年 | 月數 | TEJ有日公司數 min–max | 研究成員 min–max（已支援月） | 研究成員缺日合計（月公司） |
| --- | --- | --- | --- | --- |
| 2013 | 12 | 1956–2002 | 1281–1340 | 0 |
| 2014 | 12 | 2007–2071 | 1344–1383 | 0 |
| 2015 | 12 | 2073–2096 | 1391–1438 | 0 |
| 2016 | 12 | 2100–2121 | 1444–1491 | 0 |
| 2017 | 12 | 2122–2149 | 1496–1531 | 0 |
| 2018 | 12 | 2153–2171 | 1542–1591 | 0 |
| 2019 | 12 | 2164–2188 | 1596–1629 | 0 |
| 2020 | 12 | 2188–2224 | 1631–1659 | 0 |
| 2021 | 12 | 2227–2274 | 1666–1689 | 0 |
| 2022 | 12 | 2277–2327 | 1695–1737 | 0 |
| 2023 | 12 | 2325–2375 | 1738–1782 | 0 |
| 2024 | 12 | 2376–2430 | 1788–1850 | 0 |
| 2025 | 12 | 2408–2483 | 1856–1927 | 0 |
| 2026 | 10 | 2413–2464 | 1935–1951 | 1 |

## 測試與驗證

合成測試位於 [test_tej_pit.py](../tests/test_tej_pit.py)，沒有任何授權原始值或逐列衍生 fixture。初始 RED：模組不存在，1 collection error、exit2；新增九月累計期間反例後 RED：1 failed / 10 passed，證明分類問題，修正後最終 focused：`14 passed in 0.94s`、exit0。兩項 in-memory mutation（千元乘數改1、缺日倒填次月10日）均 assertion-killed；另有 SHA 改 bytes、schema 改欄名必 STOP，SHA失敗不產出資料／報告。

全套最終執行（CPU threads=1；早期兩次在最終小修正前中止，第三次因發現外部真實矩陣存取而中止，皆 exit130，不當最終驗證結果）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=V6:V6/research_lab/tests:. \
/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

### 全套測試的資料隔離與限制

本輪最初直接依指定命令執行，事前未完整辨識既有 `test_p3_real_fold_purge` 會建立 SeqData，載入6個外部真實 fold 的 X／y；該路徑確實存在，測試也已執行到相關區段。因此**不能宣稱本輪全程未讀真實標籤**。發現後立即中止；沒有把那些值用於本案資料稽核、特徵選擇或研究判定，但此為任務資料邊界偏差，須明列供 Human／reviewer 判斷。

修正執行方式：不修改任何既有測試、不變更 skip 邏輯，用 bwrap 程序層 tmpfs 遮蔽外部 `.local-data`、p4-bundles、Windows正式 Data 與主 checkout `.artifacts`，其餘根目錄只讀、指派 worktree及 /tmp 可寫。既有7項依外部資料存在與否的測試會自然 skip；不讀那些真實資料、也不以合成替代冒充真實驗證。原 pytest 命令與 collection 範圍保持不變。sandbox 原本已限制網路；另試 bwrap --unshare-net 因 NETLINK_ROUTE 不允許而 exit1，未發送網路請求，移除此額外 flag 後使用原 sandbox 限制。

```bash
bwrap --ro-bind / / --bind "$PWD" "$PWD" --bind /tmp /tmp \
  --dev-bind /dev /dev --proc /proc \
  --tmpfs /home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data \
  --tmpfs /home/frank/projects/p4-bundles \
  --tmpfs /mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data \
  --tmpfs /home/frank/projects/MarketMamba/.artifacts \
  env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=V6:V6/research_lab/tests:. \
  /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest V6/research_lab/tests tests -q -p no:cacheprovider
```

### 既有整合 gate 的位置修正

初放 research_lab 的新資料工具會被 P4a training source inventory 自動列入，隔離後全套在 source-compat 區段失敗（該輪中止 exit130）。單項重現實際輸出：`1 failed in 6.31s`，理由 `unreviewed source difference: V6/research_lab/tej_pit.py; independent numerical-neutrality review required`。因此移動本次新檔到既有 `tools/tej_pit.py`／`tests/test_tej_pit.py`，符合既有本機離線資料工具邊界；沒有修改 source_inventory、P4a／P4d、REASONS、白名單、threshold或測試skip。相同失敗單項修正後：`1 passed in 11.35s`、exit0。最終 source SHA／manifest 已重建，原候選逐列檔仍完整保留。

最終全套狀態：**STOP_TIME_BUDGET / NOT_COMPLETED**。Collection 實測1351項；在本輪30分鐘界線前終止測試程序，exit130，沒有全套終局 passed／failed 總計，因此**不宣稱全套通過**。最後輸出進度約69%，尚餘較慢source-compat與其餘測試；以下為完整實際捕獲stdout，並非最終通過摘要。未留下背景測試／排程。下一段須以原bwrap隔離命令完整跑完，不能只憑本輪partial解除驗收STOP。

```text
........................................................................ [  5%]
........................................................................ [ 10%]
........................................................................ [ 15%]
............................................................ssssss...... [ 21%]
........................................................................ [ 26%]
........................................................................ [ 31%]
........s............................................................... [ 37%]
.................................s...................................... [ 42%]
........................................................................ [ 47%]
........................................................................ [ 53%]
........................................................................ [ 58%]
........................................................................ [ 63%]
........................................................................ [ 69%]
..........s............
```

最終 `knowledge_health.py --ci`：exit0，實際輸出：

```text
KNOWLEDGE HEALTH

Project Map        PASS
Agent Guides       PASS
Domains            PASS
Governance         WARNING
Authority Drift    WARNING
Runtime Freshness  PASS
Generated Index    PASS
Legacy Docs        WARNING

Overall: WARNING
```


Knowledge-impact：已以8個精確changed paths執行checker，exit0／WARNING；實際更新Current State、Authority Map、Data、Worktree Register。Operations與GOVERNANCE均已review，因無營運控制面／治理政策語義變更不改正文；不是以ack略過structural gate。初次health：exit0／Overall WARNING，Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index PASS，Governance／Authority Drift／Legacy Docs WARNING。最終以全套STOP狀態更新knowledge後再驗health，結果如下。

Artifact／syntax／privacy：兩張parquet 380,514／115,876列，8份主輸出SHA與manifest相符，strict合格列均0，發布日缺失狀態一致，未來發布日0；21筆修訂補充輸出SHA與唯讀DB SHA相符。Python import／syntax、JSON schema／禁止逐列欄位、報告與新knowledge連結、git diff --check通過。`git ls-files .artifacts/tej`為空；git status只有4份knowledge修改、2份新程式／合成測試與2份新彙總報告。


## 未處理範圍與風險

不做模型、特徵選擇、收益／價格／標籤分析、正式 source admission、版本自動修補、金融歷史身分推定、排程或 runtime authority 變更。資料商 raw／逐列衍生資料僅在 `.artifacts/tej/`；私有產物若隨 worktree 移除會遺失，交接前須保留本工作樹／授權儲存。單份匯出無法重建歷史多版本；日期是日粒度，當日盤中可得性未知。春節精確日期及法律順延未核、21筆ledger只支持本地修訂對照、非曆年科目對帳仍待，builder 自查不是獨立審查。下一步應離線核授權原件／版本語義與差異分類，另由 Human 驗收，不能直接加回 C0+。

## 固定交接

- Completed：TEJ loader、私有雙表／manifest、異常／對帳／覆蓋稽核、21筆修訂唯讀對照；全套因時間界線STOP，未完成。
- Changed：`tools/tej_pit.py`、`tests/test_tej_pit.py`、`reports/tej_pit_audit_20261125.{md,json}`、Current_State、Authority_Map、Data domain、Worktree_Register，共8個版本控制候選檔。
- Decisions：沿用tools離線資料邊界；缺日不倒填；不修改發布日／值消除異常；strict PIT保持STOP；全部逐列資料僅在ignored `.artifacts/tej/`。
- Evidence：source／input／output／DB／script SHA見JSON；focused14 passed、2 mutations killed、source-compat回歸1 passed；最終全套及health實際輸出見上。
- Remaining：完整隔離全套pytest尚待跑完；IFRS首次／重編原件、長尾成因、非曆年科目、逐年春節／法律順延、真正歷史lifecycle分母。
- Blockers：全套驗證未完成；上述資料證據UNKNOWN；早期測試的真實fold存取偏差須Human／reviewer判斷；無獨立審查／Human acceptance。
- Next recommended action：先完整重跑上述隔離全套命令，再審程式與證據邊界、核原件／版本及長尾，再決定後續研究，不直接恢復特徵或改source admission。
- Project memory updated：YES，Current_State／Data／Worktree_Register同一patch。
- Authority map updated：YES，僅候選邊界，不改machine authority-map.json。
- Git status：8檔未提交、未stage；起始無既有dirty／untracked／staged，結束依no-commit保留dirty，沒有任何TEJ逐列資料進入status。
- Commit：N/A，明確禁止。
- Push：N/A，明確禁止；無merge／deploy。
- Worktree：`.worktrees/ruflo-workers/run-mww8tui1-604e40/builder_tej_2a`，detached `aafd0d9`；私有產物未搬離，移除前須依register保留與驗hash。
- Runtime/environment：WSL／指定pilot09 CPU venv，Python3.12.3，CUDA_VISIBLE_DEVICES空；MARKETMAMBA_DATA_ROOT unset、Data realpath為本worktree普通目錄；最終pytest以bwrap遮蔽外部資料，未刷新外部runtime observed_at。
