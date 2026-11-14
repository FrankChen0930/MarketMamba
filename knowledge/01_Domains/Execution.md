# Execution

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: executable-open evidence, historical proxy, provenance gate
- Update triggers: official daily status source, proxy policy, or validation inventory change

## Purpose

Define the evidence required to claim that an entry price was executable, and separate strict verification from historical simulation.

## Current Boundary

Strict executable-open status requires direct admissible evidence. Historical simulation uses a frozen P0/P1/P2 proxy policy only after OHLCV source-provenance eligibility. The proxy is not direct verification.

## Authority

The execution-proxy policy, official snapshots, global mismatch inventory, source-reliability policy, and readiness artifact at `fix/v7-historical-pit-reconstruction@e3bce91` are authoritative.

## Inputs

Exchange session calendar, lifecycle membership, daily trading-status evidence, provenance-eligible OHLCV, and official validation snapshots.

## Outputs

Strict evidence classifications, simulated-executable classifications, rejection reasons, mismatch inventories, and readiness decisions.

## Current Implementation

P0/P1/P2 is frozen. Source provenance is checked before applying proxy logic. The historical-simulation validation has zero critical false executables and therefore passes its own acceptance gate. Strict Phase 0 remains `STOP` because daily tradability/executable-open evidence is incomplete.

## Evidence Classes

- Direct official daily status: candidate strict evidence.
- Provenance-eligible OHLCV plus frozen proxy: historical-simulation evidence.
- Mixed/unknown source segment: ineligible, fail closed.

## Current vs Legacy / Research

Legacy price-presence logic did not establish strict executability. The corrected historical proxy is suitable only for explicitly labeled simulation research.

## Invariants

- Apply provenance eligibility before proxy classification.
- Preserve zero tolerance for critical false executables.
- No per-symbol/date exceptions to rescue a validation failure.
- Do not roll a failed entry to a later session.
- Strict and simulated decisions remain separate.

## Known Limitations

The proxy cannot prove actual fill availability, order-book depth, price limits, or trader-specific execution. Direct historical daily status coverage remains insufficient for strict labels.

## Do Not Use / Do Not Mix

- Do not relax thresholds because only one case remains.
- Do not treat non-null OHLCV as executable by default.
- Do not use synthetic or mixed-source values as official status evidence.

## Important Paths

- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/execution-proxy-policy.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/source-reliability.json`

## Related Domains

[Data](Data.md), [Universe](Universe.md), [Labels](Labels.md), [Portfolio](Portfolio.md)

## Update Triggers

Update after new official daily-status evidence, proxy-version changes, provenance mismatch discoveries, or readiness revalidation.
