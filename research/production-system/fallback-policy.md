# Deterministic Fallback Policy

## 選擇順序

1. 先依 market calendar 判斷是否應有資料。
2. 使用該 domain/venue/date 最高優先且有效的 provider。
3. bounded retry 後才轉下一候選。
4. 候選必須通過 schema、coverage、semantic、tier gates。
5. 多來源依 frozen policy 選擇，不可挑最像預期的值。
6. 無合格候選即標 unavailable；禁止把 silent forward-fill 當 execution evidence。

Freshness 需 domain-aware：market prices/tradability 依 session；institutional/margin/daytrade 依官方 publication SLA；fundamental/corporate action 依 announcement/effective-time PIT。休市、尚未發布、延遲、缺漏是不同狀態。

## Downstream 權限

Certified T0/T1 依 domain policy 使用。Certified T2 可作有標記的研究／historical simulation，只有明確允許時可進 features，預設不可作 execution authority。T3/unknown、unresolved mixed-source、silent forward-fill 不得進 inference/execution。

## Recovery

Transport failure 使用 bounded exponential backoff + jitter；rate limit 尊重 Retry-After。Schema drift 保存 raw snapshot、隔離 parsed output、開 incident。Partial partition 不覆蓋 canonical。Late official revision 建新 revision、重新 certification 並列出受影響 artifacts。

Failure codes：`PROVIDER_TRANSIENT`、`PROVIDER_RATE_LIMITED`、`PROVIDER_UNAVAILABLE`、`SCHEMA_DRIFT`、`COVERAGE_INCOMPLETE`、`SEMANTIC_MISMATCH`、`PROVENANCE_INELIGIBLE`、`CANONICAL_CONFLICT`、`MARKET_CLOSED_EXPECTED`、`PUBLICATION_PENDING`。

不得為 `TPEX:1264:2022-09-01` 或其他單一案例寫特例；不得因只剩一個 false executable 放寬 zero-tolerance；row count > 0 不代表完整；不得跨 provider 逐欄挑 OHLC。

Fixtures 必須覆蓋 primary failure、eligible/ineligible fallback、holiday、late data、schema drift、mixed source、revision、all-source failure。相同 evidence/policy 必須產生相同 decision。
