# Production System 實作路線圖

採 additive migration，先包住 active fetch-only。Parquet/JSON/JSONL 保留 business artifacts；SQLite WAL 保存 operational metadata。每階段均有 feature flag 與 rollback。

## Phase A：Shadow operational ledger

新增 `ops/contracts.py`、`ops/store.py`、versioned SQLite schema、`ops/projection.py` 與 tests。Shadow-record run/attempt/stage/artifact/check，投影 current-state，不改現有 success decision。

驗收：retry 共用 logical run；crash 與 success 可區分；能回答最後成功/freshness/artifact identity；至少十次 shadow replay 與既有 outputs 一致。複雜度中、風險低至中；rollback 是關閉 wrapper，既有 artifacts 不動。

## Phase B：Provider registry、snapshots、certification

新增 versioned provider config、adapters、immutable raw snapshots、canonical certifier；每次只遷移一個 domain。

驗收：每個 partition 可追 provider/request/hash/parser/policy；fallback deterministic replay；同內容 dedupe、revision 保留；mixed-source contamination 被拒絕。複雜度高、風險中。

## Phase C：Aging、incidents、notifications

實作 fingerprint/lifecycle、domain-aware freshness；先用 dry-run notifier，再取代 Telegram best-effort。

驗收：重複 incident dedupe；severity escalation/recovery 各通知一次；P0 fail closed；notification failure 不破壞原 incident。複雜度中、風險低。

## Phase D：Resumable production orchestration

把 features、inference、portfolio 納入 state machine；以完整 identity cache artifacts。新 runtime 重用合法矩陣，不因 runtime 變更重建。

驗收：每個 boundary kill/restart；只重做未 commit work；同 date 不產生 conflicting authority；strict Phase 0 仍阻擋 corrected live inference。複雜度／風險高；逐 stage shadow compare。

## Phase E：Manifest、atomic publication、operations UI

新增 immutable deploy manifests、candidate/current pointer、readiness/current-state/incidents API 與唯讀 UI；live/readiness health 分離。

驗收：consumer 不讀半成品；rollback 切回舊 manifest；UI 清楚區分 legacy、historical simulation、paper、verified-live。複雜度中高、風險中。

## Phase F：V7 forward paper portfolio

取得明確批准後，把 certified inputs 接入既有 V7 engine/hash-chain journal，只作 forward paper execution。

驗收：signal/target/intent/fill/action/valuation 可 replay；event idempotence/chain 經 restart 仍有效；tradability 使用 certified evidence；paper 不稱 live。複雜度／風險高。

## 第一個 implementation branch

只做 Phase A，分成 schema/contracts、transactional store、fetch-only shadow adapter、current-state projection、failure/restart tests 五個切片。不可同時 refactor providers、啟用 full V6.2、啟動 GPU 或恢復 Phase 1–5。

未來 production authority/sunset、Phase 0 解鎖、provider license、incident SLA/recipients、paper/live gates、promotion 必須由人決定。排程啟用、push/deploy、data purchase、外部通知變更、GPU、correctness-policy 變更均需另行指示。
