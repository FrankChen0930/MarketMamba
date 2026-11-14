# Window32 — Expanding Closure & Prospective Shadow Preparation

狀態：PROPOSED / SYNTHETIC_ONLY_ENTRY / UNMERGED / NOT_DEPLOYED；本輪完成後 STOP。

W31 固定 `67e2d25a710273646d41c0ad861b4a993593ba4d` 已由 main `b64ac353833a9248121f12ebe61e1333decb2dbf` 本機 ff-only 整合。Review GO/0Required、72份原檔與同機restore hashes、24 role views、24 models、保存Replay紀錄、10 targeted tests及治理檢查通過；沒有重新Fit或計算舊RankIC。

| Origin | H | TRAIN + VALIDATION | REFIT | 差額 |
|---|---:|---:|---:|---:|
| 2023-01-03 | 5 | 14113+14816 | 29229 | 300 |
| 2023-01-03 | 10 | 13711+14435 | 28743 | 597 |
| 2023-04-06 | 5 | 17559+14710 | 32574 | 305 |
| 2023-04-06 | 10 | 17143+14293 | 32046 | 610 |
| 2023-07-03 | 5 | 21321+14645 | 36276 | 310 |
| 2023-07-03 | 10 | 20890+14205 | 35715 | 620 |

差額逐筆驗證：signal在Validation開始前，exit在Validation開始後／當天、但嚴格早於Origin；原先因TRAIN邊界排除，Refit時已成熟。未混入Origin後的Label。Training／Validation／Refit每份view hash及模型fresh identity相符。原先W31導航FAIL未阻O1的歷史Warning原樣保留，不改寫歷史為全程無錯。

## 最小接合

Raw Capture → QC/Receipt Availability → Feature-only Universe → Model Inference → Immutable Prediction → 5d/10d Maturity → Evaluation Join → Coverage/Missing/Recovery。

W32只實作中間的Synthetic時間／身份／發布／Join門控；傳入mock分數，不載入真實模型、不做真實排名。重用W07 aware clocks/calendar、W09 strict JSON/safe-path/lock及既有atomic no-clobber bytes。W10 Ridge／W30 LGBM score、W08 evaluator／W09完成與Replay是後續可沿用的元件，W32沒有另建Trainer/Evaluator/Registry。完整W09 Bundle要求DAILY_METRICS，因此不能為了先發布預測而虛構指標；先保存prediction record，成熟評估後再完成study bundle。

`research/v7/window32-shadow-contract.json` 是VERSIONED PROPOSAL，不取代Frozen E5/E1或W06 Proposal。模型全保留L0正／反、Ridge、固定LGBM，5d主／10d次，不根據W31結果選勝者。

## 時間與離線驗收

Asia/Taipei正常目標07:00–08:00；真正PREOPEN需在合格Entry Session實際開盤前完成持久發布。開盤前開始、開盤後才保存仍MISSED_PREOPEN／RESEARCH_ONLY。Prototype用2030合成日曆/時鐘，實際archive creation另存；其PREOPEN結果不代表當時真的執行過排名。

13 focused tests涵蓋Preopen、開盤前恢復、開盤後／恰開盤錯過、QC失敗、缺模型、Feature order錯配、Session/時鐘矛盾、immutable重試／衝突、未成熟與成熟Label分別Join、5d/10d exit錯配、評估早於發布拒絕、Backfill身份不提升、未來Label不影響Prediction keys、拒絕real mode／Feature夾帶Label。模型與原始Synthetic bytes須存在且hash一致；publication time無證據即UNKNOWN，retrieval不冒充publication。

Synthetic本機Root `.artifacts/window32-shadow-preparation/`，restore `.artifacts/window32-shadow-restore/`，均在主Repository而非W32 worktree；11檔16123bytes/份，非真實市場資料。具體pins見window32-offline-evidence.json；同機restore非獨立備份。

## 2026資料與模型缺口

本輪核對的W29/W31 immutable manifests僅2021–2023，不證明2024–2026有合格連續樣本；沒有因未查Mutable Data就斷言全機無資料。W31 weights可供反序列化／scoring工程重用，不是2026 fresh refit。需先批准來源、期間、Universe/lifecycle、Calendar、事件與UNKNOWN政策、Label定義/成熟度、截止與CPU預算，再建立全新Run/Scaler/Model artifacts；L0只需固定方向/配置，不需Fit。不能直接把舊模型標為2026準備完成。

兩Features保持 `close_t/close_prev-1` 與 `(high_t-low_t)/close_t` 順序；Daily CSV至少要前後相鄰合格session、正常OHLCV、不可補零、來源單位/欄名/證券鍵一致，並有當時Universe/事件資格。TWSE-only Cohort不能直接混入TPEx商品。未來需要薄型schema parity/feature derivation及模型載入介面；本輪未啟用。

W19僅以Git固定cf1a84f的policy作source-only參考：原五次機會2026-10-27～11-02；未觀測外部Task是否啟用、未讀操作Artifacts、未觸發或延長。這兩個Resource提供單日／最近snapshot，不能補齊多年歷史或關機遺失的原始Vintage。後續需明確授權read-only receipt handoff、驗證bytes/hash/QC/session/freshness/capture class；Backfill與forward分開。五日Pilot不足以自動構成完整訓練集。

V6/run_daily_inference.py是56-feature legacy路徑，V6.2/dual另有59維；不是兩Features Classical serving。V6/marketmamba/ops/shadow.py觀察fetch-only，也不是Ranking Shadow runner；不得接管或修改V6/W19排程。

## 下一次最少Human決策

1. 初始模型：資料來源／期間／固定股票池、Calendar/事件/UNKNOWN/Label政策、限額資料取得與Fresh Fit授權。
2. Shadow契約：源資料Required/Optional、精確freeze/deadline/bounded wait、missing/late/stale/recovery、preopen合格日曆與Coverage/Evaluation/Exposure規則。
3. 有界啟用：模型/Recipe/Feature Schema pins、執行環境及起訖、是否手動或排程、持久儲存/獨立備份/監測停止方式；W19接合或Collector續期須另行授權。

目前不具備真實Shadow啟動資格。Strict Phase0 STOP、Formal Admission NOT_APPROVED、W19 PROTECTED、W12 SAFETY_HALTED、E1 BLOCKED、H2 NOT_APPROVED；不啟動下一窗口。

獨立Review：GO，Critical0／Required0；13 Synthetic tests、11檔/restore hashes及authority隔離通過，0 Required rework。限制：提供的Synthetic model/score/QC/clocks只驗內部一致性，未認證真實模型的訓練資格、來源真實性或實際preopen持久發布。
