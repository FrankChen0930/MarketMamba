# V7 Signal-to-Portfolio Validation Implementation Plan

Work inline on `feature/v7-signal-validation`. Treat the dirty main worktree and deliveries as read-only evidence.

## Task 1 - Design save point

Save the approved capability map, hard-gate contract, plan, and checklist. Verify with `git diff --check`, then commit documentation.

## Task 2 - Audit model with TDD

Files: `V6/experimental/v7_correctness_audit.py`, `V6/experimental/v7_correctness_audit_test.py`.

1. RED clean document -> PASS; GREEN strict validation and decision object.
2. RED major failure / critical unknown -> STOP; GREEN blocking rules.
3. RED missing or empty domain -> invalid; GREEN completeness checks.
4. Refactor and run `python -m unittest V6.experimental.v7_correctness_audit_test`.

## Task 3 - Reports and CLI with TDD

1. RED Markdown must include timing chain, decision, remediation, and citations.
2. GREEN deterministic JSON/Markdown renderers.
3. RED atomic dual-output and exit-code behavior.
4. GREEN CLI and atomic replace.
5. Run audit and existing portfolio unit tests.

## Task 4 - Current Phase 0 evidence

Files: `research/v7/v7_phase0_correctness_evidence.json`, `reports/v7_phase0_correctness_audit.{json,md}`.

Record source-backed findings for all four domains, including exact source paths/lines and delivery identity. Run the gate twice and compare outputs. Expected current exit is 2 (STOP). Commit implementation, evidence, and results.

## Task 5 - Honor the gate

On STOP, do not implement/run Phase 1-5. Report severity, scope, rebuild need, and minimum repair. On PASS only, label V7 Baseline v1 and write the next module plan.
