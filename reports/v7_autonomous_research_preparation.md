# V7 下一個研究決策準備 — 2026-10-21

Evidence class: `RESEARCH_PREPARATION / PROPOSED / NOT_APPROVED`。
AUTO-RES 的唯讀決策包；唯一工作佇列仍是
[research program](../research/v7/research-program.md#autonomous-charter-window-01)。
本文件不新增實驗、不解鎖 Phase 1–5、不選定新 Target/Label/比較規則。
基線為 `6d74d738`；來源完整 commit:path 與 SHA-256 見
[本輪證據](../research/v7/autonomous-window-01-evidence.json)。

## 現有證據足以回答與不能回答的問題

| 問題 | 固定來源 | 已知 / 限制 |
|---|---|---|
| E1 是否達成原凍結 refresh 判準？ | `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json` | PASS_REFRESH_HYPOTHESIS，兩種子17/29，historical simulation proxy；不是可交易、portfolio admission 或單一因果機制的證明 |
| 是否只是七月造成改善？ | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json` | 七月占平均 paired delta：5d 0.3018598910、10d 0.2310916090；非多數，但排除七月後10d平均IC低於0.03。這是敏感度診斷，不能事後改正式 acceptance 或排除月份 |
| epoch1 後是否 final-support overfit？ | 同上 + recovery report at `11ae71d` | logged validation 下降；final-support epoch degradation INCONCLUSIVE。保留epoch1/5/6，2–4未留；不可用logged曲線推定最終support因果 |
| 能否立即重播？ | `11ae71d:reports/v7_e1_evidence_recovery.md` | 原 matrix a590b280…947f62 不在已搜索範圍，安全 payload inspection 尚未認證；BLOCKED。這次沒有重搜、讀權重或產生矩陣 |
| 是否可做經濟績效/排名產品？ | result manifest + H1/H0 scope | portfolio_admission=false；H0只接受fixture產品。benchmark、risk/turnover/annualization與真實signal admission仍待Human |

E1 paired_prediction_rows=314780，matched_label_keys=307309，missing_label_keys=7471。
這些是原 manifest 的記錄，不是本輪重算，也不代表每個 horizon 的最終有效樣本數。
目前baseline仍為48 features、64/32、無graph/industry neutralization；seed43不屬正式ensemble。
舊 hypotheses catalog 的三種子/圖模型/架構候選不是已核准 E2。

## 建議 Human 採取的決策順序（尚未批准）

1. **先保留 unresolved，只有新線索才恢復 E1 recovery。** 若有人提供原始
   matrix/manifest 的精確位置，做一次有界 inventory 與逐檔 hash/config/split
   驗證；未知或不符就停。不得以新建資料模仿歷史。沒有新線索時沒有待執行搜尋。
2. **選擇下一個研究問題，而非直接挑模型。** 若要區分 refresh 效果與市場狀態，
   先批准 ex-ante regime 的來源、可用時間、修訂政策及分組方法；若要測訓練
   dynamics，先決定在歷史 matrix 不可回復時是否接受一個明確全新、不同 identity
   的研究方案。兩條路均不能宣稱已回答原 E1 epoch replay。
3. **再凍結一個比較。** 提案候選為固定現有架構/labels/recipe，只改單一
   window或單一訓練因素；對照、評估區間、種子、selection、成本和停止條件均需
   Human逐項批准。此處不指定新數值、方向或預期正結果，不修改既有E1 contract。
4. **最後才批准計算。** 未批准設計不得開始GPU、訓練、大型推論或購買資料。
   GPU需求與預估要用新方案做可驗證的資源規劃；歷史A100時數不是新工作保證。

## 提案選項與資訊價值

| 選項 | 能解除的依賴 | 必要新資訊 / Human decision | CPU準備 / 執行狀態 |
|---|---|---|---|
| A 原E1 retained-checkpoint replay | 對原final-support epoch1/5/6的描述性比較 | 原matrix精確位置、安全loader認證、bounded replay scope與資源確認 | 檢查單可準備；執行BLOCKED；缺epoch2–4仍不能完整描繪曲線 |
| B Ex-ante regime diagnostic | 將month sensitivity與事先可觀察狀態分開描述 | 來源/PIT時點/修訂/分組/比較規則；不得看結果挑regime | 設計PROPOSED；不產生regime labels或計算新結果 |
| C 單因素下一個訓練 | 為未來模型/recipe決策提供可歸因證據 | 正式假說、baseline、target/label、support、seed/selection、threshold、compute | HUMAN_DECISION_REQUIRED；不沿用E1成功替它批准 |
| D Ranking/portfolio admission設計 | 從fixture走向研究產品 | H2、prediction schema、ranking tie/missing/maturity、cost/risk/benchmark規則及strict gate | 優先價值高但本輪不實作、不偷偷增加金融預設 |

建議先 A 的「新線索條件」，沒有線索則把 B/C/D 的決策包交給 Human；
不是要求 Human 現在做全部決定，也不是將 matrix 缺失當作所有工程工作的阻塞。

## 未來方案必填的公平性與 leakage 審查

- Pin baseline、candidate、source commit、contract/config、feature order、matrix、
  predictions 與完整 run manifest；不可將相同檔名視為相同identity。
- 列出按 Date/stock_id 配對的鍵集合、重複/缺失/非finite、每個horizon maturity
  與purge後的有效support；拒絕靜默 zip 截斷或不同support下的勝負比較。
- Publication/retrieval time、universe lifecycle、next-open/no-roll、來源provenance
  和已凍結資料截點都要可驗證；沒有證據就保留UNKNOWN/STOP。
- BEST-not-LAST，兩head使用同一凍結selection；不因看到epoch1最好而改patience、
  事後加seed、回填缺失epoch或改比較窗口。新規則必須在新結果前批准。
- 固定基準與comparison support；區分描述性leave-one-month-out與正式hypothesis。
  不能挑July exclusion或最佳regime事後當作pre-registered outcome。
- 只有一個可歸因變因；多次嘗試、負結果、停止理由與計算成本一起保留，
  不挑最佳run掩蓋selection bias。Human決定多重比較與promotion規則。
- Engineering tests通過只代表程式在fixtures符合contract，不認證市場正確性、
  economic advantage、production readiness或交易能力。

## 驗收與恢復

本輪只驗證上列Git來源存在、JSON語意與記錄數值一致、來源SHA-256，並由獨立
reviewer檢查推論邊界。沒有讀取prediction payload、權重、正式Data或外部來源。
下一session從唯一backlog與本輪evidence讀取固定identity；無新線索不重做recovery。
此包完成的是研究準備，不是新實驗批准或研究結果。Strict Phase0維持STOP。
