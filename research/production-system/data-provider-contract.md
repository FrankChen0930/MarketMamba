# Data Provider Contract

## Provider registry

每筆至少包含 `provider_id`、domains、venue scope、source tier、deterministic priority、不含 secrets 的 endpoint template、calendar/timezone、timeout/retry、parser version、license/retention、validity window 與 owner。

- **T0**：交易所／監管機構 primary official source。
- **T1**：官方或契約上可證明等價的 structured source。
- **T2**：可信 secondary vendor；canonical eligibility 依 domain policy。
- **T3**：public aggregator；預設不可作 execution authority。
- **UNKNOWN**：provenance 不完整；fail closed。

seed inventory 包含 TWSE、TPEX、TAIFEX、TDCC、MOPS、FinMind、yfinance；tier 必須逐 domain 核定，不能只看 provider 名稱。

## Evidence objects

Request evidence 記錄 run/provider/domain/venue、request window/symbols、UTC time、response status、latency、retry no、content hash、snapshot URI、parser/schema version、rows、event-date range、warnings 與 redacted error。

Partition evidence 記錄 request refs、tier、key/schema checks、coverage、duplicates/nulls、event/publication/ingestion times 與 revisions。

Certification 記錄 partition identity、policy/version、candidate hashes、selected hash、eligibility、decision code、cross-provider metrics、decision time 與 code commit。Eligibility 至少區分 research-only、historical-simulation、feature-eligible、execution-eligible。

## 比較與 canonical writer

跨來源檢查 key overlap、OHLCV mismatch distributions、unit/adjustment semantics、corporate actions、session/date/timezone。由 frozen policy 選來源，不以「最接近預期值」挑值。

先寫 content-addressed candidate；schema、uniqueness、coverage、policy、hash 通過後才原子發布 manifest。相同 identity/hash 冪等；相同 identity/不同 hash 建 revision 並開 incident。

## Failure 與 versioning

分類為 transport、rate limit、provider unavailable、schema drift、semantic mismatch、coverage gap、unknown provenance、policy rejection。只有 bounded transient/rate-limit 自動 retry。Registry、parser、schema、certification policy 分別版本化；變更需 fixture replay 與 impact report。

驗收：任一 canonical row 可追至 request snapshot、parser、policy decision、code commit 與完整 fallback history。
