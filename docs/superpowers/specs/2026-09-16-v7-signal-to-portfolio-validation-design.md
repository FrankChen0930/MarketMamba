# V7 Signal-to-Portfolio Validation Design

## Status and capability map

Approved by the user on 2026-09-16 through the supplied validation brief.

| Module id | Responsibility | Depends on |
|---|---|---|
| `correctness-audit` | Decide whether historical evidence is admissible | - |
| `signal-anatomy` | Measure horizon information and tail behavior | correctness PASS |
| `portfolio-response` | Evaluate frozen head/rebalance/Top-N/cost grids | correctness PASS |
| `failure-slicing` | Falsify hidden regime/exposure dependencies | signal, portfolio |
| `decision-report` | Separate facts, conclusions, hypotheses, training needs | all modules |

Build order: correctness -> (signal, portfolio) -> slicing -> report. Phase 0 is a hard gate.

## Objective and frozen baseline

Determine whether the existing E5 three-seed ensemble signal is correct, tradeable, robust after costs, and ready for forward simulation or a narrowly targeted GPU experiment.

Freeze the E5 5d/10d dual-head model, 1/1/1 graph depths, d_model 64, d_state 32, FP32, sequence length 60, 59 features, current data/targets/preprocessing, and equal three-seed ensemble. Existing predictions only. The 2024-2026 period is research-used, not a pristine final holdout.

## Correctness-audit contract

A versioned JSON document must contain these domains:

1. `label_execution_alignment`
2. `feature_point_in_time`
3. `historical_universe`
4. `preprocessing_leakage`

Every check has a stable id, `PASS|FAIL|UNKNOWN` status, `INFO|MINOR|MAJOR|CRITICAL` severity, finding, direct source evidence, affected scope, rebuild requirement, and minimum fix.

Decision rules:

- `STOP` for any `FAIL` with MAJOR/CRITICAL severity.
- `STOP` for any `UNKNOWN` with MAJOR/CRITICAL severity.
- Reject missing or empty required domains.
- Only a clean `PASS` may assign `V7 Baseline v1` and authorize Phase 1.
- Exit 0 = PASS, 1 = invalid evidence/runtime error, 2 = completed audit with STOP.

The runner writes deterministic JSON and Markdown atomically.

## Evidence boundaries

The audit reads current source, metadata, and immutable deliveries. It does not modify data, predictions, A/B/C deliveries, model weights, targets, feature semantics, V6.1 production, or schedules. Unknown provenance is not evidence of safety.

## Conditional later modules

Only after PASS:

- Signal anatomy: 4-way IC, paired non-IID comparisons, Top-N overlap/displacement, residual IC, disagreement cohorts, quantile monotonicity, persistence.
- Portfolio response: frozen 2x3 head/rebalance surface, Top-N 10/20/50/100, costs 1.0x/1.5x/2.0x, execution diagnostics.
- Failure slicing: year/quarter/regime/sector/cap/liquidity falsification.
- Decision report: `reports/v7_signal_to_portfolio_validation.md`.

## Commands and structure

```bash
python -m unittest V6.experimental.v7_correctness_audit_test
python V6/experimental/v7_correctness_audit.py \
  --input research/v7/v7_phase0_correctness_evidence.json \
  --json-output reports/v7_phase0_correctness_audit.json \
  --markdown-output reports/v7_phase0_correctness_audit.md
```

```text
V6/experimental/v7_correctness_audit.py
V6/experimental/v7_correctness_audit_test.py
research/v7/v7_phase0_correctness_evidence.json
reports/v7_phase0_correctness_audit.{json,md}
tasks/v7-signal-to-portfolio-validation-{plan,todo}.md
```

Pure functions validate and decide; filesystem I/O stays at the CLI boundary. Use standard-library unittest with hand-written fixtures. Test clean pass, blocker stop, critical unknown stop, domain completeness, deterministic reports, atomic outputs, and exit codes.

## Boundaries and success criteria

Always fail closed, preserve evidence, cite file/lines, write atomically, and test before commits. Ask before changing frozen data/model/target definitions or running GPU work. Never continue after STOP, weaken known limitations, mutate deliveries, touch weights, push, or deploy.

Success means malformed/incomplete evidence cannot PASS; blockers name severity/scope/rebuild/minimum fix; timing is explicit from feature cutoff through label interval; and Phase 1-5 do not run after STOP.

Initial source inspection indicates two blockers the executable audit must preserve unless stronger evidence disproves them:

1. Labels use signal-date close as denominator while the close-derived signal is executable only next session.
2. Industry neutralization uses a frozen accumulated classification snapshot, not historical PIT membership.
