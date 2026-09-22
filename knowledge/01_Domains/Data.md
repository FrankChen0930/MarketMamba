# Data

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: source provenance, raw/canonical boundaries, PIT eligibility
- Update triggers: provider change, snapshot revision, lineage policy change, new canonical dataset

## Purpose

Define which market data may support research, simulation, and production claims. This note is navigation; machine contracts and source snapshots remain authoritative.

## Current Boundary

`Data/processed_v6` is a mutable, availability-oriented provider dataset. It is not a point-in-time (PIT) authority. Corrected V7 research uses versioned exchange-verified OHLCV, provenance-remediation inventories, and immutable official-source snapshots referenced by the corrected-baseline manifests.

## Authority

1. Machine-readable manifests, hashes, lineage inventories, and tests.
2. Completed run artifacts tied to those identities.
3. Reusable fail-closed PIT/provenance primitives now in development main, with byte-identical branch provenance in `research/v7/source-adoption-register-20260922.json`; remaining materialization builders and result evidence stay at `fix/v7-historical-pit-reconstruction@e3bce91`.
4. Runtime observations with timestamps and TTLs.
5. Reports and historical notes.

## Inputs

- Official TWSE, TPEx, TAIFEX, TDCC, and MOPS publications.
- Provider feeds such as yfinance and FinMind only where an explicit domain policy admits them.
- Historical lifecycle and exchange-session evidence.

## Outputs

- Raw snapshots with source identity and retrieval metadata.
- Canonical tables with provenance eligibility and revision/cutoff semantics.
- Dataset identities consumed by feature, universe, execution, and label contracts.

## Current Implementation

The corrected pipeline separates raw availability from canonical admissibility. OHLCV source eligibility is evaluated per segment and field, not granted provider-wide. Mixed or unknown provenance fails closed for execution claims. Dataset reuse requires matching identity; a new runtime does not imply rebuilding an unchanged matrix.

## Evidence Classes

- Official immutable snapshot: may establish historical facts within its scope.
- Canonical derived table: usable only with lineage and contract identity.
- Mutable provider cache: input evidence, not standalone PIT proof.
- Synthetic or backfilled value: never official execution evidence.

## Current vs Legacy / Research

Legacy V6 datasets remain useful for reproducing historical work, but cannot be relabeled as corrected PIT evidence. Corrected V7 datasets are separate contracts. Research caches such as `v7_prepared` do not override canonical manifests.

## Invariants

- Preserve raw data separately from canonical outputs.
- Record source, observed/publication time where available, revision policy, and hashes.
- Reject unexplained shrinkage or replacement of established source data.
- Provenance eligibility is evaluated before any proxy rule.
- Current-world classifications must not be backfilled into history without evidence.

## Known Limitations

- Some external publication timestamps and historical revisions remain incomplete.
- Provider availability is broader than evidence eligible for strict PIT use.
- External source behavior can change and requires revalidation.

## Do Not Use / Do Not Mix

- Do not treat `Data/processed_v6` as strict PIT evidence.
- Do not mix official snapshots and fallback OHLCV without segment lineage.
- Do not use synthetic prices or present-day metadata to prove historical executability.
- Do not overwrite legacy A/B/C artifacts or reinterpret them as tradable evidence.

## Important Paths

- Correctness authority: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/`
- Provenance remediation: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/`
- Historical execution evidence: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/`
- Legacy provider store: `Data/processed_v6/`

## Related Domains

[Features](Features.md), [Universe](Universe.md), [Execution](Execution.md), [Labels](Labels.md), [Training](Training.md)

## Update Triggers

Update after a source-policy revision, new canonical snapshot, lineage mismatch, changed PIT cutoff rule, or provider migration.
