# Labels

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: executable label semantics, verified versus simulated labels, readiness gates
- Update triggers: label contract, execution evidence, or readiness decision change

## Purpose

Separate strict verified executable labels from historically simulated labels and legacy return targets.

## Current Boundary

For corrected V7, the signal is observed after close on day `t`; entry is the next calendar session open only when executable; exit is the close after the configured 5- or 10-session holding horizon. There is no rolling to another entry or exit. Unknown execution evidence fails closed.

## Authority

The corrected policy, label manifest, execution-proxy policy, and remediation readiness at `fix/v7-historical-pit-reconstruction@e3bce91` define the semantics and current counts.

## Inputs

Eligible universe membership, exchange calendar, provenance-eligible OHLCV, and daily tradability/executable-open evidence.

## Outputs

- Strict corrected verified executable labels.
- Separately named historical-simulation labels.
- Rejection/reason metadata and maturity boundaries.

## Current Implementation

Strict Phase 0 is `STOP`; corrected verified executable labels remain 5d=`0`, 10d=`0` because direct daily tradability/executable-open evidence is unresolved. Historical-simulation readiness is separately `PASS`, with 5d=`6,889,229` and 10d=`6,790,554` labels in v1. Critical false executable count is zero under the frozen P0/P1/P2 proxy policy.

E1 v2 extends the historical-simulation window and reports 5d=`7,023,920`, 10d=`6,923,464`; these remain simulated labels under that experiment contract.

## Evidence Classes

- Verified executable: direct admissible evidence under the strict contract.
- Historical simulated: proxy-qualified research label, never renamed “verified.”
- Legacy label: old contract preserved for reproducibility only.

## Current vs Legacy / Research

Legacy counts (5d=`7,944,471`, 10d=`7,822,665`) belong to a different contract. Historical-simulation PASS permits research under explicit simulated semantics; it does not clear strict Phase 0 or trading readiness.

## Invariants

- Membership is not tradability.
- Unknown execution evidence remains unknown and fails closed.
- No entry/exit rolling and no synthetic proof of executability.
- Strict and simulated readiness are reported independently.
- Acceptance remains zero-tolerance for critical false executables.

## Known Limitations

Direct historical daily tradability and executable-open evidence is not complete, so strict labels remain empty.

## Do Not Use / Do Not Mix

- Do not relabel simulated targets as corrected verified targets.
- Do not use legacy A/B/C results as tradable evidence.
- Do not lower the acceptance threshold or add case-specific exceptions.

## Important Paths

- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/label-manifest.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/execution-proxy-policy.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json`

## Related Domains

[Universe](Universe.md), [Execution](Execution.md), [Data](Data.md), [Training](Training.md), [Research](Research.md)

## Update Triggers

Update when direct evidence becomes available, the proxy policy is versioned, label counts change, or a new readiness decision is recorded.
