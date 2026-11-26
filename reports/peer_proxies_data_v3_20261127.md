# 同業代理 v3：TSE／OTC 聯集在市研究決定 — 2026-11-27

**RESEARCH_ONLY / STRICT_PIT_STOP / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。完成不等於 Human acceptance。

本輪依 Human `EXECUTE_APPROVED` 轉述的協調者研究設計決定：同業代理只問當日是否在 TSE 或 OTC 掛牌，不區分交易所。入口為 `min(TSE上市日, OTC上市日 中有值者)`，上市當日納入；出口為下市日，該日及以後排除。ROTC 興櫃與 GISA 創櫃紀錄不能建立入口；僅有首次掛牌日或市場名稱亦不夠。**v3 在市判定完全不讀「變更市場」欄**；讀取該欄只為保留 v2 對照及既有來源診斷，並不推定其方向。無效日期、反向區間及快照衝突仍 fail closed，未降低既有來源 gate。

這是已核准的**研究設計決定**，不是補出了市場變更語義、公告日期、完整歷史 universe 或可成交證據；**known_at 仍 UNKNOWN，strict PIT 仍 STOP**。不改 WP1 或任何 production authority。

沿用 [工具](../tools/peer_proxies.py)、[合成測試](../tests/test_peer_proxies.py)、reports、`.artifacts/peer/`；fresh `replay-v3-20261127/` 為保存 v2 歷史證據而新增 namespace，既有流程可承載，未新增頂層目錄／平行系統／worktree。舊 [v2 報告](peer_proxies_data_v2_20261127.md)／JSON 均未改寫。[本輪 JSON](peer_proxies_data_v3_20261127.json)只含彙總、hash 身分與環境；逐股／逐日資料與抽樣只寫本 worktree `.artifacts/peer/`。

1a／1b 其他規則不變：TEJ 子產業為固定主分類、保留 v2 的產業啟用日分段與衝突排除；排除本股、至少 3 家。1a 用 `close[t−1]/close[t−2]−1`，原 OHLCV／正量／35% 遮罩不變；1b 用上一完整曆月，兩期首次單月值，發布日嚴格之後首 observed session 收盤可用，分母晚到要等候、目標可用即回缺、跨 2013/01 口徑 NaN。沒有讀標籤、prepared matrix、MarketMamba-blind、未來報酬或 IC。

## 覆蓋與家數

5,805,094 唯一候選日股鍵，3,392 sessions（2013-01-02～2026-11-24）；完整 CPU 回放 319.259 秒。同一次執行重算 v2，覆蓋與有效家數 histogram 均精確等於舊 JSON，否則 STOP。逐列保留 `eligible_v2`、v2 代理值、家數與原因欄。

| 量 | v2 | v3 | 增量 | 占同一候選分母 |
|---|---:|---:|---:|---:|
| 候選日股鍵 | 5,805,094 | 5,805,094 | +0 | 100.0000% → 100.0000% |
| 研究在市資格 | 2,043,542 | 5,805,094 | +3,761,552 | 35.2026% → 100.0000% |
| 1a 有值 | 1,386,995 | 5,260,246 | +3,873,251 | 23.8927% → 90.6143% |
| 1b 有值 | 56,481 | 299,486 | +243,005 | 0.9730% → 5.1590% |
| 市場對照有值 | 5,800,203 | 5,800,203 | +0 | 99.9157% → 99.9157% |

新增的 3,761,552 個資格日來自不再要求市場變更方向可解釋，並非新補來源證據。v3 候選分母本來就以明示上市／下市區間定義，因此候選內資格 100% 是設計結果，不能冒稱全體基本資料皆已知。基本資料共 2,958 檔；TEJ 子產業在候選內仍有 9,415 未知日股、5,795,679 已分類日股。TSE 產業未知仍有 2,531,926 日股，不倒填現況。覆蓋大幅增加只反映成員集合改變，未檢視任何預測績效，也不解除選樣、來源、publication、PIT 或公司行動限制。

| 有效其他成員數（全部候選日，含未知；p10／中位／p90／最大） | v2 | v3 |
|---|---|---|
| 1a | 0.0／0.0／7.0／44 | 3.0／10.0／40.0／74 |
| 1b | 0.0／0.0／5.0／44 | 0.0／6.0／28.0／74 |

0 可含資格／分類或價格未知，不能解讀為已知沒有同業；1b 家數達標仍須目標未公告才能有值。完整 histogram 與原因、167 個訊號月早公告比例見 JSON。

TEJ 每日單一成員產業比例（單一成員產業數／當日已識別產業數）中位 **29.7619% → 11.0132%**；v3 p10／p90＝10.0877%／12.8319%，範圍9.1743%～13.2743%。每日產業家數 p10／中位／p90／最大由 **1／2／7／45 → 1／6／16／75**。TSE 對照的每日單一成員比例中位仍為 0%，產業家數中位12→25；完整兩維分布均見 JSON，未依結果換分類。

## 價格交叉檢查與轉市場抽樣

原始 frozen raw `Close` 非缺值（先於 OHLC 品質遮罩）為本節存在性定義，分母分開列示：

- 在市 5,805,094 日股中，有原始價 5,733,559；無原始價 **71,535（1.2323%）**。原因為快照缺值／未收錄，具體是否涵蓋停牌、缺漏或其他來源問題仍 UNKNOWN，不以缺價判定下市，這些列仍保留同業分母。
- 基本資料股票的有價 6,035,642 日股中，不在區間內 **302,083（5.0050%）**：早於明示上市入口271,310、無明示 TSE／OTC 上市日30,771、下市日及以後2。早期價格可能來自其他市場或上市日期涵蓋限制，不能據價格反推掛牌；無明示日期者不能僅依名稱當作 ROTC，也不能補認上市。
- 另有196個價格股票ID／238,138原始價格列不在基本資料股票集合，單列為基本資料覆蓋缺口，不擅自加入成員。
- 原 manifest `price_crosscheck`／逐列 `price_present` 是 **OHLC驗證後 raw close** 的存在性（非還原含息價）：在市缺價71,536，有價但不在市187,990／5,921,548＝3.1747%。它與原始存在性不同；在市有1列 raw價被 OHLC 品質遮罩。兩者都不控制上市／下市區間，JSON 的 `raw_price_crosscheck` 為原始存在性對照。

抽樣事先按 stock ID 排序，取 observed window 內 OTC日早於TSE日的前10家，查TSE轉換日前兩個、當日或首後續 observed session、後兩個 sessions。**10家／50觀測全數 v3 連續在市**；同列留 v2 排除結果與價格存在性。這是原始明示日期的內部一致性抽查，不是官方逐案認證；逐股身分／日期只在私有 `transfer-continuity-samples.json`。

## 實際驗證與限制

標準命令（每個 shell 指令皆以截斷管線結尾，測試結束狀態另核對）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest tests/test_peer_proxies.py tests/test_forward_wp1.py -q -p no:cacheprovider 2>&1 | head -c 3000
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python tools/peer_proxies.py --output .artifacts/peer/replay-v3-20261127 2>&1 | head -c 3000
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python .artifacts/peer/verify-v3-20261127.py 2>&1 | head -c 3000
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python .artifacts/peer/price-presence-v3-20261127.py 2>&1 | head -c 3000
```

新增測試先 RED：4 failed／19 passed；GREEN：指定兩檔 **34 passed**。包含轉市場連續、ROTC／GISA排除、上市日納入、下市日排除、未來轉市場與下市日期／市場名稱 mutation 不改過去，以及直接禁止 v3 呼叫市場 history。兩個 in-memory source mutants（min→max入口、下市日誤納入）均 **KILLED**，沒有改寫正式來源。

8個輸入、6個實作與21個原始產物 hash 全部重驗；5,805,094鍵唯一且無跨年重疊，v2覆蓋／histogram精確重現。三個完整日面板（2013-01-02／2020-01-02／2026-01-02，共5,122列）逐列 scalar v2／v3 市場 oracle相等。來源在回放前後hash一致；未因覆蓋提高調整 gate。作者依code-review技能自查正確性、邊界、私有輸出與成本，**不是獨立review**。語法／JSON／reference／diff與knowledge健康最終結果見末段交接。

環境：既有CPU Python3.12.3、NumPy2.5.4、pandas3.1.1，executable見JSON；只借用既有venv，不讀另一研究repo結果。`MARKETMAMBA_DATA_ROOT=UNSET`、`CUDA_VISIBLE_DEVICES=''`；Data realpath為本worktree `/Data`，不是symlink，未讀寫正式Data。輸入僅主checkout授權raw與frozen price；報告日期是指派交付名，非外部runtime observation。

輸出紀律例外：最初一個混合偵察shell只對末段套截斷，印出較長worktree清單；之後全部改用整組指令截斷。沒有印大檔、陣列或完整diff。首輪沒有GPU／取得資料／額外agent／網路／排程／production／通知／wakeup／commit／push／merge／deploy；未改V6/models、正式Data、raw、P4d、帳本、第三步bundle。

## 固定交接

- Completed：v3研究在市、v2逐列對照、1a／1b重算、原始與品質驗證後缺價核對、轉市場抽樣、測試與彙總。
- Changed：tools/peer_proxies.py、tests/test_peer_proxies.py、本MD／JSON、Current_State／Authority_Map／Universe／Features；私有GENERATED產物在本worktree `.artifacts/peer/`。
- Decisions：只取TSE／OTC明示聯集區間；不辨交易所、不讀市場變更作v3判定。此為本任務研究設計，不能推廣成正式universe authority。
- Evidence：CPU歷史研究回放、合成測試與builder自查；known_at UNKNOWN、strict PIT STOP。manifest及補充證據hash見JSON。
- Remaining：來源publication／歷史版本／官方日曆／公司行動／完整universe、獨立複核與Human acceptance；不算IC或研究績效。
- Blockers：strict PIT來源時間與完整歷史生命週期證據仍不足；不解除其他既有STOP。
- Next recommended action：獨立複核本研究候選與來源差異，再由Human驗收；不得直接啟動研究或production。
- Project memory updated：YES，Current_State、Universe、Features；knowledge_update: required（研究語義／當前候選改變）。
- Authority map updated：YES，只說明本候選邊界，不改machine authority或外部observed_at。
- Git status：開工clean；結束預期僅指定檔案dirty／untracked，因明示no-commit而保留，不能聲稱clean。
- Commit：未執行（Human禁止）。Push：未執行。
- Worktree：builder_peer_v3，detached base `15b809234c9ace5f7271a7c3e28a7a9f2d282264`，job-run-mwxs4cs8-f2d5bf；交接與變更保留前不可移除。
- Runtime/environment：上述CPU環境；未更新外部runtime觀測。

最終驗證：指定標準測試 **34 passed／5.00秒，exit 0**；兩來源mutants killed。AST語法2檔、彙總JSON、33份補充證據hash及15個新本地references通過；`git diff --check`通過。knowledge-impact判斷PASS但人工確認研究語義有變，已更新上述四份knowledge。`knowledge_health.py --ci`為 **WARNING／exit 0**，與開工基線相同：Project Map、Agent Guides、Domains、Runtime Freshness、Generated Index為PASS，Governance／Authority Drift／Legacy Docs保留既有WARNING，沒有新增FAIL。沒有以ack略過structural gate。最終git為6個指定tracked修改＋本MD／JSON兩個untracked；`.artifacts/peer/`為ignored GENERATED。明示no-commit使工作樹刻意保留dirty，未stage、不稱clean。
