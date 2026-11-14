# Incident Policy

## Severity

- **P0**：correctness/security/authority 可能被破壞，例如 secret leak、錯誤資料已 promotion、journal chain 破裂、dual writer。立即 fail closed，恢復需人工批准。
- **P1**：當日核心流程無法安全完成，例如 canonical conflict、全來源失敗、publication authority 不明。停止 downstream 並通知。
- **P2**：降級但 policy 仍允許安全路徑。記錄、aging，達條件才通知。
- **P3**：不影響當日結果。列入 daily summary。

Fingerprint 至少含 class、authority、domain/provider、partition、policy version，不含 volatile timestamp/stack address。同一 open incident 只更新 last_seen/count；只有 severity escalation、SLA breach、material evidence change、recovery、resolved 後再發才重送。

Lifecycle：`OPEN -> ACKNOWLEDGED -> MITIGATED -> RESOLVED`，另有 `SUPPRESSED_BY_POLICY`、`FALSE_POSITIVE`。每個 transition append-only 記錄 actor/time/reason/evidence。

初始 aging（正式 SLA 需人類批准）：P0 立即通知、5 分鐘未 ack 升級；P1 立即通知、15 分鐘升級；P2 在 retry/fallback 後，超過 publication SLA 或 60 分鐘才通知；P3 日報彙總，連續三個 expected windows 再評估。Holiday/尚未到發布時間不計 missing age。

立即 fail closed：unknown provenance 進 execution、same identity conflicting authority、model/feature contract mismatch、invalid checksum/marker、stale publisher fencing、strict correctness STOP、疑似 secret exposure。

自動 recovery 只能用 frozen policy。Ack 只表示有人接手，不表示可放寬 gate。通知包含 impact、已採動作、operator action、refs；destination 使用 logical ref，secret 不落 DB。通知失敗開獨立 incident。

Classes：`DATA_PROVIDER`、`SCHEMA_DRIFT`、`PROVENANCE`、`COVERAGE`、`ORCHESTRATION`、`LEASE_CONFLICT`、`ARTIFACT_CORRUPTION`、`MODEL_CONTRACT`、`CORRECTNESS_GATE`、`PORTFOLIO_REPLAY`、`PUBLICATION`、`NOTIFICATION`、`SECURITY`。

測試 dedupe、escalation/recovery 各一次、P2 SLA 內 recovery、holiday、notification failure、以及 process exit 0 時 P0 仍讓 readiness fail。
