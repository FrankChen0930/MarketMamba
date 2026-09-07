# V7 MAS Research Program Contract

## Objective

Determine which literature-derived hypotheses materially improve MarketMamba over the frozen current baseline under realistic Taiwan-equity cross-sectional forecasting constraints, using the smallest dependency-aware sequence of experiments that preserves causal attribution.

## Current Environment Facts

- WSL canonical repo exists and is the repository for V7 preparation.
- Data stays on `D:` through `MARKETMAMBA_DATA_ROOT`.
- Representative parquet/data checks passed during migration.
- Lightweight tests passed during migration.
- GPU model runtime is not fully ready.
- `torch-scatter`, `torch-sparse`, CUDA-specific `mamba_ssm`, and `causal_conv1d` still need later resolution.
- This runtime dependency state must not be mistaken for a model or research failure.

## Non-Negotiable Rules

- Freeze baseline before architecture changes.
- One causal question at a time.
- Cheapest discriminating experiment first.
- Negative results are first-class results.
- No random time-series train/test split.
- Record predictive, economic, robustness, and systems/compute metrics.
- Papers are priors, not local evidence.
- Human retains final architecture authority.

## Required Metrics

Record these where applicable:

- IC
- RankIC
- ICIR
- RankICIR
- Top-K / Bottom-K
- existing portfolio metrics
- transaction-cost-adjusted metrics
- turnover
- drawdown if available
- parameter count
- training time
- peak VRAM
- inference latency/throughput
- seed/regime stability

Experiment verdict must be exactly one of:

- `SUPPORTED`
- `INCONCLUSIVE`
- `REJECTED`
- `BLOCKED_BY_COST`

## MAS Role Boundaries

### Advisor

May audit, prioritize, design experiment families, and select the next high-information experiment.

May not declare final V7 architecture.

### Builder

May implement approved scoped experiments.

Must preserve provenance, baseline comparability, and data integrity.

### Verifier

Independently checks leakage, experiment isolation, metric correctness, baseline fairness, claimed gain, and compute cost.

### Archivist

Preserves evidence, rejected/inconclusive results, rationale, and references.

### Human

Approves baseline freeze, compute escalation, major scope changes, architecture promotion, and final V7 freeze.

## First MAS Activation

The first MAS activation tomorrow must be planning/audit only:

Audit the WSL MarketMamba repository, GPU/runtime readiness, current V6.x baseline, literature corpus, and hypothesis backlog. Produce a minimal dependency-aware experiment plan and identify the exact first experiment. Do not begin expensive training until Human approves it.

Expected first MAS output:

- baseline/project audit
- GPU/runtime dependency assessment
- hypothesis dependency graph
- proposed first 3-5 experiments
- compute/cost/risk estimate
- missing evidence/questions
- Human approval request
