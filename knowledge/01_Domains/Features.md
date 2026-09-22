# Features

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: corrected feature contract, ordering, scaling, graph/industry policy
- Update triggers: feature manifest or scaling-policy change

## Purpose

Describe the current corrected feature boundary and distinguish it from legacy V6 and experimental feature sets.

## Current Boundary

The corrected V7 baseline is `E5-PIT-Clean-v1` with exactly 48 ordered features. The feature-order SHA-256 is `7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696`. Group dimensions are `[15, 20, 1, 12]`.

## Authority

The feature manifest and corrected policy at `fix/v7-historical-pit-reconstruction@e3bce91` are authoritative. Model contracts consume that exact order; prose and old notebooks do not redefine it.

## Inputs

Canonical market, macro, flow, and eligible company data whose PIT status satisfies the data policy.

## Outputs

An ordered 48-column feature matrix plus matrix/contract identities used by training and resume validation.

## Current Implementation

- Explicit graph features: disabled.
- Current-industry backfill: disabled.
- Industry neutralization: disabled.
- Macro scaling: causal expanding z-score.
- Cross-sectional features: per-session scaling.
- Financial fields without adequate publication-time coverage were removed.
- `Market_Cap_Log` remains admitted under the corrected policy.

The dropped financial set includes `PER`, `PBR`, `Revenue_MoM`, `Revenue_YoY`, `EPS`, `EPS_Surprise`, `Gross_Margin`, `ROE`, `Book_Value`, `Dividend_Yield_Fwd`, and `Free_Cash_Flow`.

## Evidence Classes

- Manifest order/hash: binding machine contract.
- Matrix identity: completed materialization evidence.
- Feature analysis report: diagnostic only unless incorporated into a manifest.

## Current vs Legacy / Research

Legacy counts of 56 (V6.1), 59 (V6.2 tracked runtime), and 66 (isolated baselines) describe different contracts. They must not be compared as though column membership and preprocessing were unchanged.

## Invariants

- Exact names, order, transforms, and missing-value behavior travel together.
- Reuse a persisted matrix only when data and feature identities match.
- No graph or industry feature may silently re-enter through preprocessing.
- Downstream admission must fail closed on contract mismatch.

## Known Limitations

The manifest records `value_table_status=NOT_EMITTED_DOWNSTREAM_ADMISSION_BLOCKED`; downstream use still depends on the appropriate admission gate and completed artifact identity.

## Do Not Use / Do Not Mix

- Do not substitute a legacy 56/59/66-column matrix.
- Do not infer current membership from similarly named columns.
- Do not rebuild an unchanged matrix merely because a Colab runtime restarted.

## Important Paths

- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/policy.json`
- Replication contract: `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-replication-v1/incumbent-contract.json`

## Related Domains

[Data](Data.md), [Models](Models.md), [Training](Training.md), [Research](Research.md)

## Update Triggers

Update whenever feature membership/order, transforms, graph policy, industry treatment, or matrix identity rules change.

**2026-09-22 source identity:** The frozen 48-feature manifest is copied byte-identically into development main for research-source tests; the original branch/commit and blob are recorded in `research/v7/source-adoption-register-20260922.json`. Feature order and semantics are unchanged.
