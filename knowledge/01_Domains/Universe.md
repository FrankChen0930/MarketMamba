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
