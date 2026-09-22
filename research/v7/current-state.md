# MarketMamba V7 — Current State

_Last updated: 2026-09-10_

## 0. Purpose

This file is the canonical short-form handoff for the current MarketMamba V7 research / implementation state.

It does **not** replace:

- `research/v7/research-program.md`
- `research/v7/experiment-hypotheses.md`
- `research/v7/literature-synthesis.md`
- `research/literature/paper-index.md`
- `research/literature/notes/`

Those files remain the canonical research corpus.

This file answers only:

- Where is V7 now?
- What has been decided?
- What is still unverified?
- What authority currently exists?
- What is the next Human boundary?

---

## 1. Current V7 Hypothesis

Current selected hypothesis:

**H1 — current Mamba vs Mamba-2**

H1 is an isolated backbone comparison.

Mamba-2 is a **hypothesis**, not an assumed winner and not yet the V7 backbone.

The purpose of H1 is to determine whether replacing the current temporal Mamba implementation with a Mamba-2 equivalent improves the quality–compute trade-off under a controlled comparison.

No architecture promotion has been authorized.

---

## 2. Frozen Baseline Identities

H1 uses two distinct baseline identities and they must not be conflated.

### Predictive / score baseline

`v2_kg_nomacro`

### Economic / portfolio baseline

`v2_kg_nomacro_f20`

Repository evidence previously grounded the current baseline as:

- model: `ShortModelV6`
- temporal window: `60`
- feature dimension: `59`
- fundamentals_v2: enabled
- availability flags: off
- neutralization: `none`
- KG: `knowledge_graph_v2.npz`
- Group D: zeroed
- heads: `[5d, 10d]`
- online primary score head: `5d`
- checkpoint: `v6_short_GD_no_macro_gatv2.pt`
- portfolio: `N=50`, `k=1.5`, rebalance every 20 trading days
- economic reference after backfill: approximately `bt_ann=0.373`

Historical arms marked equivalent or incomparable must remain separated from the canonical H1 baseline.

---

## 3. External Model Contract

The isolated H1 implementation must preserve the externally observable baseline contract:

```text
(x, edge_index, edge_attr, padding_mask)
-> (N, 2)
```

Only the temporal Mamba stack may change.

The following are comparison controls and must remain unchanged unless a new Human-approved experiment explicitly changes them:

- embedding
- feature pipeline
- GATv2
- fusion
- dropout
- output heads
- labels
- loss
- knowledge graph
- Group D zeroing behavior
- portfolio state machine
- production inference baseline

---

## 4. Current Candidate File Boundary

The current isolated H1 implementation boundary is exactly:

```text
V6/experimental/v7_h1_config.py
V6/experimental/v7_h1_mamba2_adapter.py
V6/experimental/v7_h1_probe.py
V6/experimental/v7_h1_test.py
```

These four files currently exist as **unverified candidate Builder output**.

They must not be treated as accepted implementation or trusted experiment evidence merely because they exist or because synthetic tests pass.

No production MarketMamba file has been approved for H1 mutation.

---

## 5. Current Candidate Implementation Status

Known state of the four candidate files:

- isolated config/scaffolding exists
- fake-kernel / synthetic adapter scaffolding exists
- guarded probe scaffolding exists
- unit tests exist
- synthetic tests previously passed
- no trusted real Mamba-2 execution has occurred
- no trusted CUDA/GPU validation has occurred
- no formal training has occurred
- no Verifier PASS exists

Known substantive gaps identified during the durable lifecycle:

1. Frozen baseline identities must be explicitly persisted:
   - `v2_kg_nomacro`
   - `v2_kg_nomacro_f20`

2. The implementation must preserve the full external model contract:
   - `(x, edge_index, edge_attr, padding_mask) -> (N, 2)`

3. The adapter must represent the real isolated Mamba-2 temporal-stack boundary rather than a standalone fake-kernel harness.

4. Fake/stub kernel behavior must be:
   - explicit
   - test/synthetic-only
   - never an automatic production/runtime fallback
   - never represented as real Mamba-2 evidence

5. Provenance must explicitly capture, when available:
   - git SHA
   - timestamp
   - Python version
   - PyTorch version/status
   - `mamba_ssm` version/status
   - `causal_conv1d` version/status
   - CUDA availability/version
   - GPU identity
   - exact command
   - runtime result
   - failure reason/metadata
   - elapsed time / latency
   - peak VRAM or an explicit unavailable status

6. An unclosed-file `ResourceWarning` in the test path must be fixed and verified with warnings promoted to errors.

---

## 6. Runtime / Dependency State

The latest READ_ONLY rebootstrap observation reported:

- MarketMamba git HEAD:
  `6cde79c6f9e9e192ba9a4f57c8deaa83e7817e69`
- Python:
  `3.12.3`
- environment:
  WSL2
- `torch`:
  unavailable in the observation environment
- `mamba_ssm`:
  unavailable
- `causal_conv1d`:
  unavailable
- `torch_geometric`:
  unavailable
- GPU / NVML:
  blocked or unavailable in that observation context

Therefore local synthetic success is **engineering scaffolding evidence only**.

It is not evidence that Mamba-2 works correctly on the intended CUDA environment.

Previous planning judged Colab A100 as the stronger candidate for real H1 runtime validation, while WSL requires dependency / wheel readiness first.

No dependency installation is currently authorized by this state file.

---

## 7. Experiment Policy

H1 must remain an isolated single-variable comparison.

### Allowed conceptual difference

```text
current temporal Mamba backbone
vs
Mamba-2 temporal backbone
```

### Not part of H1

Do not bundle H1 with:

- Bi-Mamba
- GATv2 architecture changes
- regime / Δ conditioning
- NLL
- conformal prediction
- feature redesign
- portfolio-policy redesign
- other V7 hypotheses

Those belong to later experiments.

### One-seed policy

A 1-seed run may be used later as an **engineering probe** to verify:

- implementation correctness
- interface / tensor compatibility
- training stability
- runtime
- memory
- evaluation pipeline
- artifact provenance

A 1-seed result must **not** be used as Mamba-2 promotion evidence.

Formal research comparison should require multi-seed evidence and a separate Human approval.

---

## 8. Evaluation Principles

The frozen V6.x evaluation protocol remains part of the comparison contract.

Relevant quality/economic evidence includes:

- IC
- RankIC
- ICIR
- RankICIR
- Top-K / portfolio metrics
- transaction-cost assumptions
- rebalance timing
- annualization
- segment/regime robustness
- runtime
- peak VRAM / memory feasibility

Mamba-2 must be evaluated on a **quality + compute Pareto**, not assumed to be faster or more accurate.

Previously proposed thresholds such as:

- mean IC improvement `+0.009`
- Newey-West `t >= 2`
- net annualized return `+6pp`

must not become arbitrary promotion gates unless their empirical source is explicitly justified from frozen baseline variance / historical dispersion or another pre-registered evidence basis.

Valid H1 research outcomes include:

- `SUPPORTED`
- `INCONCLUSIVE`
- `REJECTED`
- `BLOCKED_BY_COST`

Negative results are first-class evidence.

---

## 9. Durable Lifecycle Recovery Status

The previous durable lifecycle was stored under:

```text
/tmp/mas-h1-advisor-plan-20260909-r7.sqlite3
```

A Windows / WSL reboot cleared that temporary store.

The old durable authority chain is therefore **not resumable** and must not be reconstructed by manually fabricating checkpoints or approvals.

Lost authority included:

- pending durable Plan selections
- durable Human approvals
- remediation TaskContract issuance
- execution/rework approvals
- SCOPED_WRITE execution authority
- Builder/Verifier trusted lineage
- active durable lifecycle head

Historical IDs and reports remain historical evidence only.

A new persistent durable chain has been bootstrapped.

### New persistent durable store

```text
/home/frank/projects/multi-agent-system/state/market-mamba-h1-rebootstrap-20260910-r1.sqlite
```

### New source task

```text
h1-rebootstrap-20260910-r1
```

### Current checkpoint

```text
checkpoint_kind: ADVISOR_PLAN_AVAILABLE
checkpoint_seq: 1
pending Plan: plan-h1-rebootstrap-20260910-r1
```

Current authority state:

```text
Human approval recorded: NO
SCOPED_WRITE authorized: NO
Builder attempted: NO
```

This is intentionally a safe Human-review boundary.

---

## 10. Rebootstrap Advisor Assessment

The new READ_ONLY Advisor rebootstrap successfully reconstructed the important H1 requirements from repository evidence.

It explicitly recognized:

- predictive baseline:
  `v2_kg_nomacro`
- economic baseline:
  `v2_kg_nomacro_f20`
- external contract:
  `(x, edge_index, edge_attr, padding_mask) -> (N, 2)`
- four exact candidate write paths
- fake-kernel test-only semantics
- missing real Mamba-2/runtime evidence
- provenance requirements
- confirmed `ResourceWarning`
- no production mutation
- no training
- no promotion

The Advisor recommended Human review of:

```text
plan-h1-rebootstrap-20260910-r1
```

No new write authority exists until the Human explicitly approves the new durable Plan chain.

---

## 11. Human Authority Boundaries

### Not currently authorized

- production baseline mutation
- new MarketMamba write paths
- dependency installation
- CUDA repair
- Colab/A100 execution
- real GPU smoke
- formal 1-seed training
- multi-seed training
- compute escalation
- Mamba-2 promotion
- V7 architecture promotion

### Human retains final authority over

- material scope expansion
- experiment-design changes
- compute/resource escalation
- formal experiment execution
- architecture promotion
- interpretation of H1 research evidence
- final V7 architecture

---

## 12. Next Human Boundary

The immediate next action is **Human review of the new pending durable Advisor Plan**:

```text
plan-h1-rebootstrap-20260910-r1
```

Do not infer or reuse approval from the lost `/tmp` durable chain.

After Human review, the canonical lifecycle should create fresh authority step-by-step:

```text
ADVISOR_PLAN_AVAILABLE
-> Human pending-Plan decision
-> issued remediation task
-> execution Plan
-> Human execution-Plan decision
-> Builder / Verifier
```

Every approval must be newly persisted in the new durable store.

---

## 13. Cross-Conversation Usage Rule

For future ChatGPT / Codex / MAS sessions:

1. Treat this file as the **current-state pointer**, not the full research truth.

2. Use these files for deeper V7 reasoning:
   - `research/v7/research-program.md`
   - `research/v7/experiment-hypotheses.md`
   - `research/v7/literature-synthesis.md`
   - `research/literature/paper-index.md`
   - `research/literature/notes/`

3. Do not redo literature discovery unless a new evidence gap is explicitly identified.

4. Do not treat historical MAS reports as executable authority.

5. Use the persistent durable store for current lifecycle authority.

6. MarketMamba owns research/domain truth; MAS owns orchestration/authority.

---

## 14. Current Summary

```text
V7 research corpus
READY

Selected experiment
H1 current Mamba vs Mamba-2

H1 research status
UNTESTED

Isolated candidate implementation
EXISTS / UNVERIFIED

Real Mamba-2 runtime validation
NOT DONE

Formal training
NOT AUTHORIZED

Mamba-2 promotion
NOT AUTHORIZED

Persistent durable lifecycle
REBOOTSTRAPPED

Current durable state
ADVISOR_PLAN_AVAILABLE

Next action
HUMAN REVIEW
```
