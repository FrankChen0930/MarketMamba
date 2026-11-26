## 同業代理 v3 聯集在市研究候選 — 2026-11-27

[v3報告](../../reports/peer_proxies_data_v3_20261127.md)／[彙總](../../reports/peer_proxies_data_v3_20261127.json)記Human核准的協調者研究設計：明示TSE／OTC上市日取最早者（當日含）至下市日前（當日不含），不分交易所，v3不用任何市場變更欄。ROTC／GISA／首次掛牌名稱或日期本身不建立資格；保留日期／snapshot衝突fail closed。此段更新下方同業v1／v2的目前研究市場規則；舊結果仍為歷史紀錄，不改WP1或正式universe／48欄／P4d／其他bundle authority。

同一5,805,094候選日股，1a=5,260,246、1b=299,486，TEJ未知9,415；原始在市缺價1.2323%、有價不在市5.0050%，缺價原因UNKNOWN，不以缺價排除成員或推定下市。OTC→TSE十家五十觀測連續在市，v2規則與代理逐列留對照。產業啟用日／≥3家／排除自身、價格與營收可用時鐘不變。**RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；known_at UNKNOWN、strict PIT STOP**。逐列僅本worktree `.artifacts/peer/`；價格日期聯集、來源版本／publication／公司行動／完整universe仍未認證，沒有標籤／blind／IC。驗證見報告，待獨立review與Human acceptance。

## 同業代理工作包3 — 2026-11-26

[資料層報告](../../reports/peer_proxies_data_20261126.md)／[實作](../../tools/peer_proxies.py)是未提交UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED候選；無正式特徵或資料准入。TEJ子產業依粒度及未列變更紀錄比例選主分類，兩套每日分類皆留unknown。首次與現況一致且無變更才重建穩定分類；raw產業／市場變更值方向未知，完整歷史成員及known_at仍STOP，不把現況分類倒填歷史。上市入口與下市端點採WP1保守子集，下市當日排除；價格缺失不移除已識別成員分母。

1a以t當期同業排除本股，聚合t−1／t−2完整session raw收盤價差，至少3家；市場等權另存。1b僅上一完整曆月，首次單月值各自配去年同月，兩期發布日之後首session收盤才可用，目標可用即回缺，至少3家；跨2013/01口徑NaN。逐列、原因及分母只存私有 `.artifacts/peer/`，不修改48欄、正式Data或其他bundle；日曆／公司行動／歷史版本限制見報告。

# Universe

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: historical membership, lifecycle eligibility, universe/tradability separation
- Update triggers: lifecycle source, security classification, or universe policy change

## Purpose

Define which securities exist in the historical research universe without claiming that membership proves daily tradability.

## Current Boundary

The corrected universe includes verified Taiwan common-stock lifecycle intervals represented as half-open ranges `[valid_from, valid_to)`. Unknown lifecycle evidence is excluded. Current classifications are not backfilled into history.

## Authority

The universe manifest and corrected policy at `fix/v7-historical-pit-reconstruction@e3bce91` are authoritative.

## Inputs

Official listing/delisting lifecycle evidence, security type, venue, and canonical identifiers.

## Outputs

Historical membership intervals and per-row eligibility consumed by execution and label construction.

## Current Implementation

The current manifest reports 1,975 stocks and 7,528,865 eligible rows. Of these, 7,528,864 rows have tradability unknown and one is marked suspended under the strict evidence view. Those counts demonstrate why universe membership and execution must remain separate.

## Evidence Classes

- Verified lifecycle interval: historical-universe evidence.
- Present-day classification: current metadata only.
- Provider presence: discovery evidence, not lifecycle proof.

## Current vs Legacy / Research

Legacy full-panel membership may include symbols based on data availability or current classification. The corrected universe is lifecycle-based and fail-closed.

## Invariants

- Membership does not imply tradability or executable open.
- Validity intervals are half-open and date-aware.
- Unknown lifecycle evidence is excluded rather than inferred.
- Identifier/venue changes require explicit lineage.

## Known Limitations

At least one historical listing-start issue remains `UNKNOWN` in the evidence inventory. It must not be filled from present-day metadata without a source.

## Do Not Use / Do Not Mix

- Do not use OHLCV presence as proof of common-stock membership.
- Do not backfill today’s industry/security classification.
- Do not convert `UNKNOWN` tradability into executable status.

## Important Paths

- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/universe-manifest.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/policy.json`

## Related Domains

[Data](Data.md), [Execution](Execution.md), [Labels](Labels.md), [Portfolio](Portfolio.md)

## Update Triggers

Update after lifecycle-source additions, symbol lineage changes, universe-rule changes, or resolution of an unknown listing interval.

## WP1 snapshot-derived lifecycle candidate — 2026-11-26

[共用 WP1](../../tools/forward_wp1.py)保留TEJ補充的掛牌／下市／全額交割及市場／產業變更日期，提供保守as-of查詢；known_at與變更值方向UNKNOWN，不回填今天產業，不認證歷史宇宙或tradability。研究矩陣1,952檔均有snapshot代碼覆蓋，strict PIT合格0；下市終日／停復牌與前瞻准入仍STOP。[工程報告](../../reports/forward_wp1_20261126.md)記來源SHA、私有產物與測試，候選UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，不取代上方corrected universe manifest。
