# 同業代理歷史分類 v2 — 2026-11-27 交付名稱

**RESEARCH_ONLY / STRICT_PIT_STOP / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED**。實際 CPU 執行日為 2026-11-26（Asia/Taipei）。完成不等於 Human acceptance。

本輪依 Human `EXECUTE_APPROVED` 限定任務，延續[既有資料層](peer_proxies_data_20261126.md)與[整合規劃 §C／D／E](../tasks/plan-fundamentals-supplychain-integration-20261126.md)，只改[同業資料工具](../tools/peer_proxies.py)、[合成測試](../tests/test_peer_proxies.py)，新增本報告、[彙總 JSON](peer_proxies_data_v2_20261127.json)及更新既有 knowledge。沿用 tools／tests／reports／`.artifacts/peer/`，沒有新頂層目錄、平行系統或 worktree；fresh namespace 是為保留舊歷史產物，未覆寫舊報告。

## 文件解析與驗證政策

主 checkout 的 `.artifacts/tej/raw/docs/company_basic_field_doc.html` SHA-256 為 `e49283d66edab4470d683dfe134bb04363cede05776fb3321c77d8aab484af92`，與同目錄唯讀 `SHA256SUMS` 一致。HTML 的欄位標籤與說明有一列錯位，例如「首次掛牌TEJ子產業」列的說明實際談最近變更後的新分類；不能逐 HTML 列直接建立欄位映射。本輪依 Human 明示語義及文件的完整 6161 時序，採 **前一次＝最近一次，變更值＝變更後的新分類，變更日＝啟用基準日**；TSE 產業同義。分類碼只統一 `M` 前綴，不建立任何舊碼／新碼替換表。

先驗每檔快照是否一致、首次／現況分類是否齊備、事件值／日期是否完整、原始最近至最舊日期是否嚴格遞減、事件是否早於首次掛牌、最近事件新分類是否等於現況；不合格整檔分類留缺並記原因。通過後以首次掛牌分類為起點，按事件日期由舊到新，在啟用日當天切換；不把現況分類倒填過去。日期早於首次掛牌列為保守的來源時序衝突，沒有自行修正其含義。

| 維度 | 有變更股票 | 最近值與現況不一致 | 不一致比例 | 日期排序失敗 |
|---|---:|---:|---:|---:|
| TEJ 子產業 | 1,024 | 0 | 0% | 0／1,024 |
| TSE 產業 | 1,336 | 10 | 0.7485% | 0／1,336 |
| 市場（僅測候選方向） | 1,591 | 131 | 8.2338% | 0／1,591 |

上表是獨立條件計數，可與其他原因重疊；下列為按 gate 順序的互斥原因，不可混用分母。全體基本資料 380,514 列／2,958 檔，跨月快照衝突 0。

- TEJ 子產業原始驗證：OK 2,848、首次／現況不一致或缺失 87、變更早於首次掛牌 23。另套文件範例驗證後，OK 2,847、`DOCUMENT_EXAMPLE_MISMATCH` 1，合計未知分類 111 檔；不代表每檔都有市場資格或價格。
- TSE 產業：OK 1,764、首次／現況不一致或缺失 164、變更早於首次掛牌 1,023、歷史不完整 3、最新值不一致 4。另 6 個最新值不一致案例已由較早的時序 gate 拒絕，故上表為 10。
- 市場候選方向：OK 2,336、首次／現況不一致或缺失 424、變更早於首次掛牌 82、最新值不一致 116。另 15 個最新值不一致案例與時序衝突重疊；OK 數包含無變更者，不能當作已驗證有變更者數。

6161 確實在資料中。文件例子為 `23B1 → 23B5 → 23B4 → 23B4A`，啟用日依序為 2002-01-02、2005-03-09、2005-08-16、2007-07-30。首次分類、首次掛牌日、三次變更的分類與日期、事件總數共 **9 項核對，4 通過／5 不符**：四項分類碼與一項變更日期不符。沒有猜測其原因，也未把文件值寫入原資料；該股票 TEJ 分類留缺。私有 `document-example-6161.json` 保存逐項實值、預期與結果，公開報告／JSON只保留彙總。這個差異不推翻 Human 確認的欄位方向，也不能證明舊分類碼可直接映射至新分類碼。

市場文件六個變更欄位沒有文字說明；日期順序雖全過，最新值只有 1,460／1,591 相符，無法由部分一致確認整體方向，故 **value_direction 仍 UNKNOWN；所有有市場變更紀錄者均不使用其歷史市場推定資格**。上市入口只用明示 TSE／OTC 上市日的最早日，下市日期當日及之後排除；不由缺價判定下市，也不以首次掛牌市場相同推定上市入口。下市端點採保守排除，並未取得可成交終日認證。

## 重算規則與覆蓋

主分類仍固定 TEJ 子產業，沒有因覆蓋或報酬換分類。1a 排除本股、至少 3 家，仍用完整前一 session 的 `close[t−1]/close[t−2]−1`；不使用 t 日或未來價差。價格來源、OHLCV／正價／正量／35% 異常遮罩不變；市場等權對照另存。1b 仍用上一完整曆月、兩期首次單月值計同比，兩期各自發布日嚴格之後第一個 observed session 收盤才可用；分母晚公告則等候，目標值可用即回缺，跨 2013 口徑 NaN，至少 3 家。未讀標籤、blind、prepared matrix，未算 IC 或未來報酬。

3,392 sessions，2013-01-02～2026-11-24；完整回放 **201.713 秒**。同一候選鍵分母及市場資格前後一致，改善來自分類重建。

| 量 | v1 | v2 | 增量 | v2／全部候選 |
|---|---:|---:|---:|---:|
| 候選成員日 | 5,805,094 | 5,805,094 | 0 | 100% |
| 可確認市場資格 | 2,043,542 | 2,043,542 | 0 | 35.2026% |
| 1a 有值 | 912,976 | 1,386,995 | +474,019 | 23.8927% |
| 1b 有值 | 33,974 | 56,481 | +22,507 | 0.9730% |
| 市場對照有值 | 5,800,203 | 5,800,203 | 0 | 99.9157% |

1a／1b 舊候選覆蓋為 15.7272%／0.5852%；在 v2 可確認市場且分類已知的 2,043,542 成員日中，覆蓋分別為 67.8721%／2.7639%。這個子集 TEJ 分類未知為 0，**不代表全市場或全部 2,958 檔的分類已知**；市場未知仍有 3,761,552 個候選日。另有 295,898 個可分類成員日（14.4797%）與現況分類不同，沒有現況倒填。

| 有效其他成員數（全部候選日，含 unknown） | v1 p10／中位／p90／最大 | v2 p10／中位／p90／最大 |
|---|---|---|
| 1a | 0／0／4／26 | 0／0／7／44 |
| 1b | 0／0／3／26 | 0／0／5／44 |

0 包含資格／分類未知，不能解讀為已知沒有同業。1b 有效同業至少 3 家的 938,904 列，仍須排除目標已公告者，因此不是全部可出值。完整有效家數 histogram、原因分布、167 個訊號月早公告比例保留於彙總 JSON。

TEJ 每日單一成員產業比例（單一成員產業數／當日已識別產業數）中位由 **32.0313% 降至 29.7619%**；v2 p10／p90＝27.1084%／32.3699%，範圍 26.5060%～34.8066%。每日產業家數分布中位仍為 2、p10＝1、p90 由 6 至 7、最大由 27 至 45。這是市場已知子集的產業分布；TSE 維度分布亦保留於 JSON，未據此更換主分類。

## 驗證與產物 identity

使用既有 CPU 環境 `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，Python 3.12.3／NumPy 2.5.4／pandas 3.1.1；只使用其 runtime，沒有讀該研究 repo 結果。`MARKETMAMBA_DATA_ROOT=UNSET`、`CUDA_VISIBLE_DEVICES=''`。本 worktree `Data` realpath 即本 worktree `/Data`，非 symlink，未讀寫；正式輸入明示來自主 checkout `.artifacts/tej/raw/` 及 frozen `.artifacts/p4b/snapshot-prices-20261125/prices_raw.parquet`。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m pytest tests/test_peer_proxies.py tests/test_forward_wp1.py -q -p no:cacheprovider 2>&1 | head -c 3000
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python tools/peer_proxies.py --output .artifacts/peer/replay-v2-20261127 2>&1 | head -c 3000
CUDA_VISIBLE_DEVICES='' PYTHONPATH=V6:V6/research_lab/tests:. /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python .artifacts/peer/verify-v2-20261127.py 2>&1 | head -c 3000
```

- TDD：新斷言在舊實作失敗；標準 focused／regression **27 passed／3.19 秒**。涵蓋文件 6161 型四段 fixture、舊值方向錯置、非遞增／缺失／錯誤日期、早於掛牌、未來合法變更不影響過去、明示上市入口，以及既有排除本股／門檻／價格與營收未來不變性。
- Source mutation：把生效日 `<=` 換成 `>=`、將事件套用順序反轉，兩個 mutant 都被文件時序測試殺死；只在記憶體執行，沒有覆寫工具。
- **8 輸入／6 實作／20 產物** size／SHA（實作記 SHA）重驗；14 份年度表合計 **5,805,094 唯一日股鍵**、3,392 sessions；全表 strict flag 均 false、known_at 均 UNKNOWN、有效值均滿足至少 3 家且已公告目標 1b 均缺。
- 三日 oracle：2013-01-02／2020-01-02／2026-11-24，**10,324** 次分類及 **5,162** 次市場資格與單日函式核對；另從 frozen OHLCV 重新驗價，逐一排除本股、不使用 group-transform 聚合，**5,162 列** 1a／1b／市場值與有效家數一致（其中有值 1a 823、1b 6），容差 `atol=rtol=1e-12`。
- AST／`git diff --check` 通過。knowledge-impact 與最終 health 結果補記於下方交接；治理不取代資料資格或獨立審查。

全部授權逐列產物與原始比對只在本 worktree `.artifacts/peer/replay-v2-20261127/`；`manifest.json` 固定 source commit、config、輸入、實作、20 產物 identities、環境與 evidence class，SHA 見彙總 JSON 的 `private_manifest_sha256`。`verification.json` 為後續自查證據，驗證腳本位於 `.artifacts/peer/verify-v2-20261127.py`，其 SHA 保留於驗證結果；公開 JSON 的 verification 為該私有結果投影。舊 v1 artifacts／reports 不修改，原 TEJ `SHA256SUMS`、raw/docs 不變。

## 限制與下一步

`known_at` 仍 **UNKNOWN**：啟用基準日不是公告日，快照重建屬 `DOCUMENTED_EFFECTIVE_DATE_HISTORICAL_RESEARCH_REPLAY`，strict PIT 持續 **STOP**。沒有公司逐期首次公告、實收時間、未記錄變更或歷史版本完整性的證明；原始 TEJ 子產業至多四次事件也不能證明沒有截斷。市場方向、6161 文件與資料差異、早於首次掛牌事件均待來源釐清，不能自行修復。日曆仍是 observed price-date union、價格仍為 raw 非含息且公司行動未完整核驗，上市／下市端點不是交易可執行證明；營收版本與合併可比性限制沿用 v1。未訓練、未選模型、未做標籤／IC／績效分析。

作者依 code-review-and-quality 檢視正確性、結構、安全與效能：變更局限資料層、無新依賴／網路 I/O、來源 checksum 與輸出 namespace 仍 fail closed，日內分類矩陣預先分段計算避免逐日重驗所有原始欄位。這不是獨立 review。下一步為獨立審查與 Human 決定是否接受此受限歷史分類；市場語義／6161 差異及公告時間需另案補證，不自動開研究或交易。

## 固定交接

- Completed：歷史產業重建、全體方向／時序稽核、兩代理 CPU 重算、測試、自查與報告；非 Human acceptance。
- Changed：`tools/peer_proxies.py`、`tests/test_peer_proxies.py`、本報告／JSON、Current State／Authority Map／Features；私有產物沿用 `.artifacts/peer/`。
- Decisions：文件確認產業 NEW_VALUE；市場維持 UNKNOWN；衝突留缺；研究分類已改善，strict PIT STOP。
- Evidence：27 passed；8／6／20 identities；5,805,094 唯一鍵；10,324 分類／5,162 市場及代理重算；2 source mutants killed。
- Remaining：獨立審查、Human acceptance、未明市場／分類來源與 publication PIT；未處理模型、正式特徵准入、交易與 deployment。
- Blockers：strict PIT／完整市場歷史、6161 差異與其他來源時序衝突仍未解除；CPU 任務本身無阻塞。
- Next recommended action：獨立 review 彙總與私有比對，再由 Human 決定受限研究用途。
- Project memory updated：YES，Current State／Features；knowledge semantics YES，authority boundary／active production runtime／correctness thresholds NO；新增 GENERATED 候選產物 YES，不 supersede 正式 authority、不 deprecate 既有能力。
- Authority map updated：YES，僅候選 evidence 與方向邊界；machine／production authority 不變。
- Git status：初始 clean；結束保留本任務 dirty／untracked，符合 no-commit，最終精確清單見交接補記。
- Commit：無；source HEAD `cb748759171cca1949fc4d338348060239eb09ae`。
- Push：無；merge／deploy／GPU／資料取得／網路／其他 agents／wakeup／排程均無。
- Worktree：既有指派 `builder_peer_v2`，detached HEAD；owner `job-run-mwxpr1uz-1aaf65/builder_peer_v2`，目的為此 bounded CPU task，created_at UNKNOWN（本輪未建立），active／UNMERGED／NOT_DEPLOYED／NOT_ACCEPTED；退休日 N/A。私有產物依賴此 worktree，未持久保留 dirty source 與 artifacts 前不可移除。
- Runtime/environment：上述 WSL CPU runtime；未改 formal Data、V6/models、P4d、帳本、第三步 bundle、raw/checksums／docs 或任何外部 runtime；未刷新外部 observed_at。


交接補記：最終 knowledge-impact 為 **WARNING**（canonical knowledge 變更要求人工 review `knowledge/GOVERNANCE.md`；已閱讀並確認未改治理規則，非略過 structural gate）；knowledge health **WARNING、0 structural FAIL**，Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index 為 PASS，Governance／Authority Drift／Legacy Docs 為既有 WARNING 類別。AST 2 檔、799 個本地文件連結、JSON 必要欄位及 manifest hash、`git diff --check` returncode 0 均通過。最終 5 modified＋2 untracked、0 staged，都是上列七個任務檔；本次 no-commit，故依授權保留 dirty，沒有為達 clean 覆寫或移除成果。`.artifacts/peer/` 為 ignored 私有 GENERATED 輸出。精確治理輸出及 completion inventory 也保留於該目錄。
