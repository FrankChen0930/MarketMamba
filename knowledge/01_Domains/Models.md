# Models

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: current reference architecture, model-family boundaries, promotion state
- Update triggers: architecture contract, reference-model, or promotion decision change

## Purpose

Identify the current reference model contract without confusing legacy operational models, corrected replications, and research candidates.

## Current Boundary

The corrected reference is `E5-PIT-Clean-v1`: `d_model=64`, `d_state=32`, `expand=2`, `head_dim=8`, `n_groups=1`, sequence length 60, one temporal block, one forward and one reverse cross-stock block, dropout 0, and 5-day/10-day heads. Explicit graph input is disabled.

## Authority

The incumbent contract on `feature/v7-e1-rolling-origin-refresh@044eef7` and corrected feature/label policies on `fix/v7-historical-pit-reconstruction@e3bce91` jointly define the research baseline. No model is promoted beyond the evidence stated in those artifacts.

## Inputs

The exact corrected 48-feature matrix, eligible universe rows, and a named label contract.

## Outputs

Separate 5-day and 10-day scores, checkpoints, validation metrics, and lineage identities.

## Current Implementation

The architecture retains temporal modeling and implicit forward/reverse cross-stock interaction. It does not consume explicit graph edges. Formal corrected-replication evidence includes seeds 17 and 29. Seed 43 is incomplete and is convergence evidence only.

## Evidence Classes

- Contract + loadable checkpoint + matching matrix identity: model-run evidence.
- Formal completed seed: eligible for the stated analysis scope.
- Non-terminal orphan checkpoint: forensic/convergence evidence only.
- Diagnostic report: guides research, never promotion by itself.

## Current vs Legacy / Research

V6.1/V6.2 Mamba/GAT paths are legacy operational implementations. Historical E3/E5 capacity results were produced on older data contracts. E1 preserved the architecture/training contract while refreshing the rolling-origin data window; its completed result is research evidence, not a new architecture or production promotion.

## Invariants

- Architecture, feature, label, matrix, and seed identities must be recorded together.
- A partial seed is not silently counted in an ensemble.
- Graph-disabled means no explicit graph features or edges.
- Research success does not imply trading or production promotion.

## Known Limitations

The corrected two-head diagnostic found near-redundant ensemble outputs in 2026 (correlation about `0.9983`). This is diagnostic evidence, not a license to alter the frozen replication contract.

## Do Not Use / Do Not Mix

- Do not call seed 43 complete or resume it directly without the orphan-recovery rules.
- Do not compare models across different feature/label contracts as a pure architecture test.
- Do not treat E1 research acceptance as strict correctness, portfolio, or production acceptance.

## Important Paths

- `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-replication-v1/incumbent-contract.json`
- `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v2/`
- Legacy model code: `V6/models/`

## Related Domains

[Features](Features.md), [Training](Training.md), [Labels](Labels.md), [Research](Research.md), [Production](Production.md)

## Update Triggers

Update after a frozen contract revision, successful formal experiment, ensemble-membership change, or promotion decision.

**2026-09-22 source integration:** Corrected E5/E1 reusable research modules and selected frozen contract copies are now in development main; see `research/v7/source-adoption-register-20260922.json`. Run results, bundles, postrun diagnostics and operational/deployment authority remain separate. No model training or production promotion occurred in Phase 3B-2.
