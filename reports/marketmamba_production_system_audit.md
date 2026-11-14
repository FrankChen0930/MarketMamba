# MarketMamba Production-System Architecture Audit

**日期：** 2026-09-18
**基準：** `af8ac8229a1f558370c7f059b537c6e1a6d13723`
**處置：** 僅做 audit；未修改模型、資料、排程、gate 或 legacy A/B/C artifacts。

## 結論

MarketMamba 已有可靠的 scheduled ingestion，以及大量模型、組合、評估、UI 能力，但還不是一個一致的端到端 production system。主要缺口不是模型程式，而是跨元件的 operational control plane：provider provenance、canonical certification、run/stage state、artifact identity、incident lifecycle、atomic promotion。

整體成熟度約 **2.5/5**：ingestion 正式運作；完整 V6.2 inference-to-publication 已實作但排程停用；corrected V7 研究 correctness 更嚴謹，但 strict Phase 0 仍 STOP；operations 主要依賴 file logs、booleans、best-effort 通知。

第一個實作階段應是包住 active fetch-only 的 **shadow operational ledger**，先取得可信的 run/stage/artifact 狀態，而不是重寫 provider、模型或排程。

## 已驗證的 runtime 狀態

| 項目 | 2026-09-18 觀察 | 判讀 |
|---|---|---|
| `MarketMamba_DataFetch` | Enabled/Ready，平日 22:30 | active production ingestion |
| `MarketMamba_V62` | Disabled；last run 2026-08-25，result 255 | full pipeline dormant |
| `PersonalOS_Daily` | Disabled | 非 active path |
| 2026-09-17 fetch-only | exit 0，約 657.9 秒 | 最近成功 |
| daily sources | 10/10 fresh | 現有 ingestion gate 通過 |
| hygiene | 移除 818 duplicate rows | 有檢查，但無 structured audit record |

同一 log 顯示多筆 yfinance unavailable/delisted errors，之後補齊並成功。Fallback 確實有價值，但「最後成功」無法證明每個 partition 的 provenance 與 eligibility。

## Current system map

```text
Windows Scheduler (active)
 -> v62_daily.bat --fetch-only
 -> run_v62_daily.py
 -> multi-provider fetcher
 -> mutable Parquet store + file logs

已實作但停用：
features -> V6.2 inference/baselines -> portfolio -> performance -> direct publish

Research-only：
PIT/universe/label/provenance contracts -> corrected E5 lifecycle
 -> V7 portfolio engine + hash-chain journal
```

目前腳本能完成工作，但系統無法一致回答何時執行、用了哪份證據、停在哪個 stage、昂貴 artifact 能否重用、哪個 publication 具 authority。

## 應保留而非重做

- Scheduled fetching、bounded retry、補齊與 hygiene。
- V6.2 六階段 orchestrator、deterministic portfolio/idempotence、forward OOS uncertainty semantics。
- Corrected V7 PIT provenance、verified-lifecycle universe、48-feature、no-industry-neutralization、no-graph、fail-closed label contracts。
- Corrected E5 build-once/reuse、durable resume、BEST-not-LAST。
- V7 pending targets、market status、partial fills、corporate actions、hash-chain replay journal。
- 現有唯讀 portfolio/performance API/UI 與 regression suites。

## Correctness boundary 不變

Strict Phase 0 = **STOP**；historical-simulation readiness = **PASS**；P0 critical false executable = **0**；corrected verified executable labels 5d=0、10d=0；historical simulated labels 5d=6,889,229、10d=6,790,554。Historical simulation 不是 verified-live evidence；legacy A/B/C artifacts unchanged。

## 五大 production gaps

1. **Provider provenance/certification**：來源決策藏在大型 fetcher；缺 versioned registry 與 partition-level canonical evidence。
2. **Unified run state/resume**：scheduler mutual exclusion 不是 application lease；缺 logical run、stage commit、artifact identity。
3. **Incident lifecycle**：logs/Telegram 缺 fingerprint、aging、ack、recovery、dedupe。
4. **Operational current state**：`/health` 只有 liveness；缺最後成功、freshness、active identities、stage、incidents、publication 的單一視圖。
5. **Research-to-production promotion**：V6.2 有 operational flow，V7 有更強 correctness/journal；缺 manifest、shadow comparison、authority pointer、rollback protocol。

## 分層成熟度

| Layer | 成熟度 | 下一步 |
|---|---:|---|
| Ingestion | 3.5/5 | Registry、snapshot、certification |
| Storage | 3/5 | Immutable evidence、artifact identity |
| Features | 3/5 | Durable production cache identity |
| Inference | 2.5/5 | Explicit authority、shadow promotion |
| Portfolio | 3/5 | Certified inputs、paper integration |
| Orchestration | 2/5 | Ledger、lease、resume、atomic stages |
| Observability | 1.5/5 | Structured state、metrics、readiness |
| Incidents | 1/5 | Lifecycle、aging、dedupe、SLA |
| Delivery/UI | 2.5/5 | Manifest、atomic pointer、ops views |
| Correctness governance | 4/5 | Production promotion protocol |

## Target architecture

```text
Scheduler -> lease/fencing -> resumable modular-monolith orchestrator
  -> provider registry -> immutable raw snapshots -> certification -> canonical
  -> identity/hash/complete-marker artifact cache
  -> explicitly authorised inference -> portfolio -> append-only journal
  -> evaluation -> candidate manifest -> atomic current pointer

SQLite WAL control plane:
runs / attempts / stages / artifacts / checks / incidents / publications
  -> current-state -> readiness/API/UI -> deduplicated notifications
```

這不是微服務重構；模組仍在同一 repository/host，只把現有隱含 contracts 顯式化。

## Storage decision

Parquet/JSON/JSONL 繼續承載市場資料、矩陣、predictions、portfolios、reports。SQLite WAL 保存小型 transactional operational state 與 refs。Plain JSON 對 lease/cross-stage atomic commit 太脆弱；在多主機 concurrent writers 出現前，Postgres 又過度複雜。

## 第一個實作階段

Phase A 定義 immutable operational contracts、versioned SQLite store、fetch-only shadow adapter、atomic current-state projection、crash/concurrency tests。至少十次 shadow replay 與既有 outputs 一致後才進 Phase B。此階段不 refactor providers、不啟用 full V6.2、不接 corrected V7、不跑 GPU。

Human decisions：production authority/sunset、Phase 0 approver、provider licensing、incident SLA/recipients、paper/live start、promotion/rollback。Codex 可實作 shadow infrastructure/tests，但不可 enable schedule、push/deploy、增加通知對象、買資料、改 gate。

## Deliverables

- [System requirements](../research/production-system/system-requirements.md)
- [Component inventory](../research/production-system/component-inventory.json)
- [Data-provider contract](../research/production-system/data-provider-contract.md)
- [Fallback policy](../research/production-system/fallback-policy.md)
- [Daily pipeline state machine](../research/production-system/daily-pipeline-state-machine.md)
- [Observability schema](../research/production-system/observability-schema.md)
- [Incident policy](../research/production-system/incident-policy.md)
- [Implementation roadmap](../research/production-system/implementation-roadmap.md)

## 建議

下一步不要定義成「再寫一支 daily script」或「直接恢復完整排程」。先建立 Phase A，再依序補 provider certification、incidents、resumable orchestration、atomic publication。這能重用既有優點，同時維持 V7 correctness boundary。
