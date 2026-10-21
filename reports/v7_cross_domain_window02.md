# V7 跨領域窗口02 — 工程與研究準備證據

單一 backlog：[research program](../research/v7/research-program.md#cross-domain-window-02)。
基線 main `c1179fe1ee8b66c768391b5f22f8aaebfc42d646`（已含批准的f7af348 Charter）。
分支 `development/v7-cross-domain-window-02`；UNMERGED / NOT_DEPLOYED。
此報告是有界工作的證據，不是第二套 roadmap／memory 或研究批准。

## W02-DATA：官方來源離線診斷

Authority：Human本次Data Engineering方向、Data domain的時間／provenance邊界，
以及既有fetcher的source-specific日期與欄位語意。沒有修改V6 production。
測試直接編譯固定source的六個原始AST function bodies，避開import fetcher會建立
raw/cache目錄的副作用；HTTP與HTML輸入均為synthetic。這是函式診斷，不宣稱
整個production模組或外部網站現在的行為已認證。

| Case | 合法對照 | 異常fixture的觀察 | 判斷與下一步 |
|---|---|---|---|
| TWSE MI_INDEX 日期 | 日期20260716，101列標記2026-07-16 | 回傳20260717仍產出101列，全部標記請求日2026-07-16 | 已重現日期未綁定風險；P1 production修復需另授權，不得修寫歷史正式資料 |
| TPEx daily quote schema | Open10、Close11 | 欄名與值一起重排Open/Close，parser產出Open11、Close10 | 已重現固定index schema drift風險；需在原parser採欄位契約拒絕/解析，不能以新平行pipeline遮掩 |
| TPEx 日期 guard | 同日正常解析 | 異日明確拒絕 | 既有正控制有效；不是所有官方parser都有相同缺陷 |
| MOPS monthly revenue完整性 | 上市＋上櫃2列 | OTC transport失敗時仍回上市1列，沒有完整性欄位 | 覆蓋不足風險，不自行決定monthly應全拒絕；需Human決定domain完整性政策。季度財報已有缺一市場就拒絕，不能靜默套用到monthly |
| MOPS month/PIT | 2026年5月營收→Date2026-06-01，100千元→100000元 | create_time空字串 | Date是既有cache月份口徑，不是publication timestamp；不可當V7 PIT evidence或把它改成新研究日期規則 |

七組對照／異常fixtures通過診斷oracle（不是production correctness PASS）。
外部請求0；正式資料讀寫0。完整數值、source hash與重現命令在
[window02 evidence](../research/v7/cross-domain-window02-evidence.json)。

重現：在該分支root以既有recovery venv執行
`python -B research/v7/cross-domain-provider-probe.py`；只輸出JSON，沒有資料落盤。
Probe是GENERATED diagnostic evidence的generator，不是source certifier。
Source固定於c1179fe；若source hash不同，必須把新觀察另記，不能覆寫本輪歷史。
新真實response/發布時間/資料版本與每個partition provenance仍需合格證據；
本輪不做大量抓取或用mutable cache補成PIT。

## W02-PREP：批准實驗的 metadata 防護

以既有 frozen experiment 的 contract SHA-256
`fea9948f05bb096596cb2cacfc522bad27a7e1f587d0a5a08f704da458d29152`
驗證完整 payload；checked-in 與 build_contract 的內容身份不變。
Matrix metadata 必須有完整 array/source 名稱、合法 SHA、整數列數、固定
feature order、label/P0/horizon、preprocessing 與無 graph 的契約。
變更語意後重新計算自己的 hash 也不能通過。

RED：13 tests、25 failures；GREEN：34 非訓練 tests PASS；獨立14 PASS，
Critical/Required 皆0，修復1輪。微型訓練測試的 source-hash fixture同步為
合法hex，但未執行訓練。此防護只驗 metadata，不認證實際array bytes、
checkpoint loader或完整training readiness；E1 Frozen Matrix仍BLOCKED。
另一 recovery 分支de2b62e已有 evaluator byte/provenance驗證，本輪不重作。

## W02-EVAL：正式 seed 與 acceptance 輸入完整性

固定 SEEDS=17/29；空集合、缺 seed、多 seed 或不同 seed 均拒絕，避免
`all([])` 被當成研究成功。整體與各 seed 的既有 gate 欄位必須為有限實數；
缺值、NaN、Infinity、bool、字串不能通過。合法 JSON 字串 keys 保持相容。
不改變 thresholds、bootstrap 或 PASS/PARTIAL/FAIL 定義。

RED6 tests出現29 failures/13 errors；GREEN6 PASS；獨立6 PASS，0 Required，
1修復輪。這是 aggregate 防護，不是 paired support／provenance認證；
歷史已admit的17/29結果沒有重算或撤銷。另一recovery分支de2b62e也保留
原先decide缺口；其byte/provenance checks仍是不同、未整合的工程成果。

## W02-MODEL：模型、比較公平性與研究設計準備

Authority為既有Research Program、Models/Training domain與Human的research
preparation授權。以下是固定Git artifacts與程式閱讀；沒有讀模型權重、
重新跑 inference、搜尋原矩陣或產生新的研究結果。完整11筆commit/path/
size/SHA-256 inventory在window02 evidence；commit-qualified來源避免把
main早期evaluator誤認成歷史result的產生程式。

### 已有模型與實驗的範圍

| 證據 | 可支持的說法 | 不可升格的說法 |
|---|---|---|
| `c1179fe:research/v7/corrected-e5-replication-v1/incumbent-contract.json` | E5-PIT-Clean-v1固定48 features（15/20/1/12）、sequence60、d_model64/d_state32、temporal1＋cross forward1/reverse1、5d/10d雙head、無graph。FP32/AdamW、best RankIC5 selection，10d讀同一checkpoint | 舊GAT/其他模型草案不是本輪正式架構 |
| 同contract的fields.split與fields.seeds | E5 train2013-01-02..2023-11-17，purge2023-11-20..12-29，research-used evaluation2024-01-02..2026-09-11；原seed計畫17/29/43 | 計畫3seeds不代表3個完成結果；正式分析僅17/29，43依診斷明確excluded |
| `044eef7:research/v7/corrected-e5-diagnostic-v2/diagnostic-summary.json` | 等長窗口RankIC5：2024 .084109、2025 .073829、2026 .029115；RankIC10：.073560/.053531/.003541。截尾放大但不完全造成退化 | 不是近期可交易報酬；strict STOP，verified executable labels=0 |
| 同版本occlusion-policy/results | 原best checkpoints17 epoch16、29 epoch18；四組features置於causal standardized中心0，mask不變，model parameters不變。Replay各324395rows，max error .0000228882 < .0001，rank correlations > .999999 | 這是固定模型occlusion，並非刪feature重訓ablation。效果隨月份反轉，不能據此批准永久刪特徵 |
| `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json`及evaluation | E1保留架構/訓練契約、refresh rolling origin；兩正式seeds，歷史PASS_REFRESH_HYPOTHESIS；paired314780 keys | 不是portfolio admission、研究correctness重新認證或新架構優勢 |
| `c1179fe:research/v7/experiment-hypotheses.md` | 既有Mamba/Mamba2、lookback、GAT等UNTESTED/parked假說可供Human考量 | 不把舊提案當成已完成baseline/ablation或已准訓練 |

E5 split的purge日期為 **2023-11-20..2023-12-29**；30 trading-day
purge包含既有horizon/embargo規則，不在本輪改動。固定研究日曆不等於
observational market的即時可交易資料。E1具體split仍以frozen experiment為準。

### 比較公平性與可重現性

E1的既有evaluation使用new/old各314780 prediction keys的paired support，
不是拿E5全年度summary直接與E1 2026 summary相比。Labels source308462列；
307309 keys匹配、7471 prediction keys materialized為missing labels、1153
source-only keys，coverage為MATERIALIZED_EXACT_PREDICTION_SUPPORT。
因此314780是prediction support，不是每個horizon都有314780有效labels。

歷史E1 ensemble RankIC5=.0634117723、RankIC10=.0520247529；paired bootstrap
lower95分別.0110906193/.0218207175；D10-D1=.0059473002/.0077599392。
同時保留不理想的monotonicity5=.0583379810、10=-.1158419639，避免只報
漂亮數值。數字直接摘錄固定evaluation，沒有重算或改gate。成本、成交、
universe/PIT與ranking/portfolio admission仍需各自證據，不能由RankIC推得。

`de2b62e:V6/experimental/v7_e1_evaluate.py`已有另一recovery分支的matrix
byte identity、label來源與support防護。這些程式不能因main未完整採用而
被宣稱不存在；本輪PREP只處理training metadata，EVAL只補共同decide缺口。
歷史artifact不可用本輪新helper重新命名為current research evidence。

### Selection bias／test reuse：程式可證與不可證

`c1179fe:V6/experimental/v7_corrected_e5_train.py` 的 `train_seed` 建立
split_code=2的同一個`evaluation` dataset（約386行），每epoch在其上取得
RankIC5做`update_selection`與early stopping（約474–522行），再載入best
checkpoint，將同一個dataset交給`evaluate_model`（約525–536行）。
`c1179fe:V6/experimental/v7_e1_run.py`直接呼叫此shared trainer。

這是**程式層級的selection/evaluation support reuse**，與contract所稱
research-used historical evaluation一致。時間purge可避免特定label跨界，
不能消除選模後評估的selection bias。近年退化分析、E1 refresh與postrun
分析也已使用此歷史區間；未來proposal不可再聲稱它是完全未觸碰test。

本輪未開historical bundled payload、原matrix或checkpoint，因此不能量化
bias，也不能斷言各次postrun label-extension的有效mask完全相同。不得由
目前source trace自行撤銷已admit結果；它界定下一次研究比較需要Human決定
的設計。Window01對epoch/regime的INCONCLUSIVE整理仍沿用，不重複計為新研究。

### 僅供Human決策的選項（全部PROPOSED）

| 選項 | Human必須凍結的內容 | 依賴／本輪狀態 |
|---|---|---|
| A：後續未參與選模的時間區間 | exact dates、selection/final-evaluation分離、label maturity、PIT/universe、baseline、paired support與acceptance；不能由Agent補日期 | 新資料admission、可能inference/compute授權；PROPOSED/HUMAN_DECISION_REQUIRED |
| B：nested或walk-forward設計 | folds、purge、selection budget、多重比較政策、固定features/labels/seeds及incumbent；是否訓練ablation需明確批准 | 正式研究規格與GPU/訓練授權；PROPOSED/HUMAN_DECISION_REQUIRED |
| C：維持現有描述性結論、暫不新增實驗 | 確認優先處理資料來源與既有admission依賴 | 可停留於現有STOP；原E1 matrix無新線索保持BLOCKED，不反覆搜尋/生成替代品 |

以上不新增target、label、metric、正式架構或experiment配置。所有新設計均未
READY；本輪完成的是可審查的研究準備。Portfolio H0/Operations目前沒有
比上述已證問題更高價值且獨立的READY缺口，所以不為湊數擴張system work。
