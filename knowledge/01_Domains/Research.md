# Research

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: experiment lifecycle, evidence classes, current research decisions
- Update triggers: experiment completion, decision report, baseline, or evidence-policy change

## Purpose

Provide an index for research state without turning reports, diagnostics, or historical artifacts into machine contracts.

## Current Boundary

Corrected V7 research proceeds under explicit dataset, feature, label, model, split, seed, and acceptance contracts. Strict Phase 0 is `STOP`; historical-simulation readiness is separately `PASS`. The latest diagnostic decision is `STOP_FOR_TRADING_CONTINUE_RESEARCH`.

## Authority

Machine contracts and completed artifacts on `fix/v7-historical-pit-reconstruction@e3bce91` and `feature/v7-e1-rolling-origin-refresh@044eef7` outrank prose reports. History explains decisions but does not override current contracts.

## Inputs

Versioned data/feature/label/model contracts, frozen hypotheses, seeds, splits, acceptance criteria, and compute environment.

## Outputs

Immutable run artifacts, diagnostics, comparison tables, decisions, and explicit next-experiment contracts.

## Current Implementation

The corrected E5 replication formally supports seeds 17 and 29. Seed 43 is incomplete and excluded from the formal ensemble. Diagnostics found signal degradation and near-redundant heads, leading to E1: a rolling-origin refresh under unchanged architecture/training. E1 is now admitted as a completed two-seed research experiment with `PASS_REFRESH_HYPOTHESIS`; it remains historical-simulation proxy evidence under strict Phase 0 `STOP`. Post-run month analysis finds July-sensitive 10d absolute IC, while final-support epoch degradation and ex-ante regime dependence remain inconclusive.

## Evidence Classes

- Completed contract-matched run: experiment evidence.
- Diagnostic: explanatory evidence, not promotion.
- Partial checkpoint: forensic/convergence evidence.
- Spec/package without run: planned capability.
- Legacy A/B/C: historical evidence under their original contracts only.

## Current vs Legacy / Research

Legacy A/B/C artifacts remain unchanged and are not tradable evidence. Corrected replication, E1, and Phase A answer different questions and must not be merged into a single readiness claim.

## Invariants

- Freeze the question and acceptance criteria before execution.
- Record all exclusions and partial seeds.
- Compare only identity-compatible artifacts or label the comparison limitation.
- No research result bypasses correctness gates.
- Package success is not run success.

## Known Limitations

E1 result artifacts are versioned, but post-run epoch replay lacks the frozen full matrix and safe checkpoint deserialization; causal regime decomposition lacks ex-ante state variables. Runtime telemetry for the earlier corrected replication is incomplete.

## Do Not Use / Do Not Mix

- Do not continue Phase 1–5 while strict prerequisites say `STOP`.
- Do not promote on simulated labels without stating the evidence class.
- Do not rewrite legacy artifacts after learning corrected semantics.

## Important Paths

- Correctness baseline: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/`
- Diagnostic v2: `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-diagnostic-v2/`
- E1 contract: `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json`
- E1 result: `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json`
- E1 post-run diagnostic: `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json`

## Related Domains

[Data](Data.md), [Features](Features.md), [Models](Models.md), [Training](Training.md), [Labels](Labels.md), [Production](Production.md)

## Update Triggers

Update after a completed/rejected experiment, baseline promotion, acceptance-policy change, or new authoritative diagnostic decision.
