# MarketMamba Production System 需求規格

## 基準與邊界

截至 2026-09-18，唯一啟用中的正式路徑是平日 22:30 的 `V6/run_v62_daily.py --fetch-only`。完整 V6.2 推論、組合、發布已實作但排程停用；corrected V7 strict Phase 0 仍為 STOP。本規格不修改模型、gate、legacy A/B/C artifacts、排程或資料採購。

## 功能需求

1. **來源證據**：每個 provider/domain 有版本化 registry、tier、優先序、timeout/retry、parser version、授權與有效期間。每次 request 記錄範圍、時間、結果、列數、content hash 與 immutable snapshot。
2. **Canonical certification**：擷取與 eligibility 分離。每個 partition 記錄 policy/version、候選來源、selected hash、reason 與 evidence class。未知或 mixed provenance fail closed。
3. **Orchestration**：使用 logical run、attempt、stage、artifact identity。除 scheduler 互斥外，再加 application lease/fencing。crash 後只從已 commit stage 接續。
4. **矩陣重用**：identity 包含 data hashes、universe/feature/label contracts、code、dependency lock 與 date window。新 runtime 驗證 checksum/complete marker 後重用；只有資料/contract 改變或驗證失敗才重建。
5. **推論**：feature/model/normalization identity 必須匹配，採 BEST-not-LAST。legacy、historical simulated、paper、corrected verified-live 是不同 evidence classes。
6. **組合**：deterministic targets、order intents、fills、corporate actions、valuations 可 replay；journal append-only、hash-chained 且 event-id 冪等。
7. **Promotion**：candidate 具有 immutable manifest，列出 data/model/policy/code/artifact hashes 與 gates。promotion 原子切換 current pointer；rollback 切回舊 manifest。
8. **Operations**：能查最後成功 trading date、active stage、provider freshness、model/data/policy identities、publication 與 open incidents；liveness/readiness 分離。

## 非功能需求

- correctness/provenance 邊界 fail closed。
- transactional state、atomic artifact completion、deterministic replay。
- 同 logical identity 不得產生互斥 authority。
- 任一 output 可追至 evidence、policy、code 與 approval。
- secrets 由環境/secret store 注入；URL、header、token、exception log 必須 redaction。
- 保留單機 WSL/Windows Scheduler 形態。
- additive migration，且每階段有 rollback。

## 架構約束

採 modular monolith。大型業務 artifacts 繼續使用 Parquet/JSON/JSONL；SQLite WAL 僅作 transactional operational control plane。現階段不引入微服務、Kubernetes 或雲端依賴。

## 治理邊界

production authority、Phase 0 解鎖、provider 授權、incident SLA/收件人、paper/live 啟動與 promotion 由人決定。Codex 可做 shadow control plane、registry scaffolding、dry-run incidents、cache validation、tests 與唯讀 UI；不可自行 push/deploy、啟用排程、GPU/Phase 1–5、購買資料或修改 gate。
