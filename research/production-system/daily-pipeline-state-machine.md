# Daily Pipeline State Machine

## Identities 與 DAG

`logical_run_id` = trading date + mode/authority + policy version；attempt、stage run、artifact、publication 各自有 identity。

```text
CALENDAR -> INGEST_RAW -> CERTIFY_CANONICAL -> BUILD_FEATURES
 -> INFER -> BUILD_PORTFOLIO -> EVALUATE
 -> PREPARE_PUBLICATION -> PROMOTE -> NOTIFY
```

未獲 correctness policy 授權的 stage 以 `BLOCKED_BY_POLICY` 結束，不得繞過。

## States

Stage：`PENDING -> RUNNING -> SUCCEEDED | FAILED_RETRYABLE | FAILED_TERMINAL | BLOCKED_BY_POLICY | SKIPPED_REUSED | INTERRUPTED`。

Run：`CREATED, RUNNING, DEGRADED, BLOCKED, FAILED, SUCCEEDED, PUBLISHED`。

只有 transaction commit 才是 SUCCEEDED。Reuse 需 identity/checksum/complete marker 全通過。RUNNING heartbeat stale 後轉 INTERRUPTED；retry 建新 attempt，不改寫歷史。PUBLISHED 只表示 manifest 已 promotion，不等於 verified-live。

## Gates

- CALENDAR 區分交易日、休市與 publication pending。
- INGEST_RAW 保存 immutable evidence；partial success 可 DEGRADED，但不是 canonical complete。
- CERTIFY_CANONICAL 執行 provider/provenance/coverage policy，conflict fail closed。
- BUILD_FEATURES 重用合法完整 artifact；candidate 驗證後原子完成。
- INFER 驗證 model/feature/normalization/checkpoint；strict Phase 0 STOP 時 corrected path BLOCKED_BY_POLICY。
- BUILD_PORTFOLIO 只消費允許 evidence，append deterministic intents/events。
- EVALUATE 保留 period、sample size、evidence class。
- PREPARE/PROMOTE 建 immutable manifest，再以 fencing token 原子切換 pointer。
- NOTIFY 由 incident/current-state change 驅動；失敗不重做已 commit 資料。

## Restart

crash 後可重用合法 raw snapshot；缺 complete marker 的矩陣不可重用；predictions 已 commit 則只重跑 portfolio；promotion 未 commit 時 current pointer 不變；notification crash 不重做 promotion。第二個 concurrent attempt 取得 lease 失敗；stale takeover 增加 fencing token。

Scheduler exit 0 只代表 requested mode 達到 acceptance，不代表 disabled stages 成功。Fetch-only 與 full production 必須是不同 mode/authority。

測試需在每個 boundary kill/restart，並覆蓋 concurrency、stale lease、reuse/rejection、corrupt hash、missing marker、policy STOP、promotion 前後 crash、notifier failure。
