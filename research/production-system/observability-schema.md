# Observability Schema

## 決策

SQLite WAL 只保存小型 transactional operational metadata；市場／模型 artifacts 仍由 Parquet/JSON/JSONL 承載，DB 只存 URI/hash refs。

## Tables

- `logical_runs`：date、mode、authority、evidence class、policy、state、final reason。
- `attempts`：run、attempt no、host/pid、code/dependency identity、heartbeat、exit、state。
- `stage_runs`：stage/input identity、state、timing、retryability、error fingerprint。
- `artifacts`：type、identity/content/schema hashes、URI、size、complete、creator、evidence class。
- `stage_inputs/stage_outputs`：lineage edges。
- `provider_requests`：provider/domain/window、latency/result/retry、snapshot/hash/parser/rows、redacted error。
- `certifications`：partition、policy、candidates、selected hash、eligibility、reason、metrics。
- `checks`：scope、check、status/severity、observed/threshold、policy。
- `incidents/incident_events`：fingerprint、class/severity/state、aging、ack、resolution、append-only transitions。
- `notifications`：incident/channel/dedupe/result。
- `publications`：manifest、candidate/promotion、supersession、approval。
- `leases`：owner、heartbeat、expiry、fencing token。

## Metrics 與 projection

追蹤 run/stage success/latency、provider errors/retries/freshness/coverage、fallback tier、conflicts/schema drift/revisions、artifact reuse/rebuild、incident age/MTTA/MTTR、publication age。

原子產生 `current-state.json`：schema/generated time、last attempted/successful run、active authority/evidence class/stage/lease、provider freshness/certification、active identities/publication、open incidents、reason-coded readiness。它是可重建 projection，不是 source of truth。

使用 WAL、foreign keys、busy timeout；stage success/outputs/checks 同 transaction commit。Artifact 先 write/fsync/rename 再 DB commit。Promotion 需有效 fencing token；schema migration 版本化且先備份。

必要查詢：今天哪裡失敗；某 output 的完整 lineage；provider revision 影響範圍；matrix 為何 rebuild/reuse；incident 歷史；consumer 正讀哪個 manifest。

測試 transaction rollback、kill/reopen、duplicate events、concurrent/stale leases、projection rebuild、migration、secret redaction。
