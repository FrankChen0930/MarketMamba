# Training

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: training contract, checkpoints, resume behavior, Colab packaging
- Update triggers: trainer/checkpoint contract, runtime package, or formal run change

## Purpose

Define reproducible training and recovery behavior for corrected E5 and its unchanged-contract E1 refresh.

## Current Boundary

The frozen replication training contract uses 20 epochs, patience 5, minimum 5 epochs, learning rate `1e-4`, weight decay `0.01`, warmup ratio `0.15`, minimum learning-rate factor `1e-4`, gradient clipping 1, fp32, AdamW, and linear-warmup/cosine decay. Selection records IC5 and IC10 at the same best-IC5 checkpoint.

## Authority

The incumbent/experiment contracts and checkpoint loader on `feature/v7-e1-rolling-origin-refresh@044eef7` are authoritative. The byte-identical exact lock is also preserved at `environments/v7-colab-2026.04/requirements.lock.txt` in development main; its 2026-09-12 local verification is not formal E1 GPU-run evidence. Colab notebooks are launch UX, not the training contract.

## Inputs

- Persisted identity-matched matrix and split contract.
- Model/feature/label contracts.
- Explicit seed and device/precision choice.

## Outputs

- Durable generation checkpoints (`latest`, `previous`, `best`).
- State identity, epoch/batch/step/phase/terminal status.
- Per-seed metrics, logs, manifests, and completion markers.

## Current Implementation

Resume must validate contract and matrix identity. Orphan recovery may recover a valid latest generation but must not redefine `best`. Training artifacts must be written durably outside an ephemeral Colab runtime. An unchanged verified matrix is reused across runtimes.

Corrected replication formal evidence currently covers seeds 17 and 29. Seed 43 has a surviving loadable non-terminal state (epoch 15, step 39915 in later forensics); it is not a completed seed. E1 seeds 17 and 29 are complete; both selected epoch 1 and stopped at epoch 6 by early stopping. These are training-dynamics observations, not an accepted new recipe. Epoch 2–4 checkpoints were not retained, and no final-support per-epoch replay has been established.

## Evidence Classes

- Terminal checkpoint plus completion artifact: completed seed.
- Loadable non-terminal checkpoint: resumable or forensic state, subject to identity checks.
- Log alone: supporting evidence, not completion.
- External Drive contents: `UNKNOWN` until imported and validated.

## Current vs Legacy / Research

Older notebooks that assume one uninterrupted runtime or reconstruct matrices every session are historical references. The current requirement is durable checkpoints, idempotent stage reuse, identity validation, and explicit recovery.

## Invariants

- Save enough state to continue optimizer/scheduler/model progress safely.
- Validate checkpoint format, contract, matrix, and stage before resume.
- Never count a non-terminal seed as complete.
- Preserve logs, metrics, hashes, environment metadata, and reason for termination.
- Release GPU only after durable artifacts are flushed and verified.

## Known Limitations

2026-10-21 recovery revalidated best/previous/latest file hashes for both E1
seeds. Pointer epochs are 1/5/6; embedded identities remain unvalidated because
only static ZIP/pickle-opcode inspection was performed. Existing unrestricted
loader was not executed. Frozen matrix is not found locally; no replay/training.

Runtime forensics are inconclusive: matrix wall time, epoch timing, GPU utilization, and Drive I/O were not fully captured. Observed completed-seed runtimes were about 6.94 and 7.82 hours on an A100 80GB fp32, but these are run observations, not forecasts.

## Do Not Use / Do Not Mix

- Do not rebuild a matrix solely after a runtime disconnect.
- Do not use a mismatched checkpoint to “continue.”
- Do not promote partial seed 43; E1 completion now rests on versioned result artifacts, not packaging.
- Do not infer final-support epoch degradation from logged validation curves alone.

## Important Paths

- `feature/v7-e1-rolling-origin-refresh@044eef7:V6/experimental/v7_corrected_e5_train.py`
- `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json`
- Known working Colab reference: `deliveries/V7-Confirmation-20260914/V7-三種子確認與續跑/`

## Related Domains

[Data](Data.md), [Features](Features.md), [Models](Models.md), [Labels](Labels.md), [Operations](Operations.md)

## Update Triggers

Update after checkpoint-format changes, new formal run evidence, Colab bundle changes, or added runtime telemetry.

**2026-09-22 source integration:** Corrected E5/E1 reusable research modules and selected frozen contract copies are now in development main; see `research/v7/source-adoption-register-20260922.json`. Run results, bundles, postrun diagnostics and operational/deployment authority remain separate. No model training or production promotion occurred in Phase 3B-2.

## CPU recovery engineering — 2026-10-21

`development/v7-autonomous-charter` (UNMERGED / NOT_DEPLOYED) validates local
checkpoint entry metadata before loading. Malformed/unreadable latest entries
fall back to valid previous; save/orphan recovery retain that validated generation,
not a corrupt pointer. SIGKILL before pointer, restart, identity/seed mismatch and
second latest loss fixtures pass; best is not promoted by recovery.
[Window evidence](../../research/v7/autonomous-window-01-evidence.json): two repair
rounds, final19 tests, independent19 PASS. Existing trusted-local unrestricted
Torch loader remains unchanged: no historical payload safety/replay certification,
no arbitrary pointer corruption or hostile-filesystem guarantee. E1 stays BLOCKED.

## Charter integration — 2026-10-21

Window01 source/governance is now LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED
at exactf7af348 (supersedes its branch-local status only). Pre/post CPU regression
and authority checks are in [integration evidence](../../research/v7/charter-integration-20261021.json).
A separate Human-authorized cross-domain CPU window may proceed in isolation;
research contracts, formal data, production and StrictSTOP boundaries are unchanged.

## Window02 metadata guard — branch-local

`development/v7-cross-domain-window-02` (UNMERGED/NOT_DEPLOYED) pins the existing
E1 frozen contract and rejects matrix metadata drift, including self-rehashed
changes. This does not verify array bytes or certify training readiness. Existing
recovery evaluator byte/provenance work remains branch-qualified. See
[window02 evidence](../../research/v7/cross-domain-window02-evidence.json) and
[unique backlog](../../research/v7/research-program.md#cross-domain-window-02).

## Window03 fairness and readiness preparation — 2026-10-21

Completed source-level E5/E1 split/selection/final-evaluation review and PROPOSED
comparison designs, without changing existing result classifications or policy.
Existing validators/configs match historical E1 contract/config identities;7 copied
result hashes and2 seed best-pointer metadata chains checked. No matrix/weight
payload validation, training or Colab launch. NOT_RUN_READY; original matrix BLOCKED.
[Report](../../reports/v7_data_evaluation_window03.md),
[unique completed queue](../../research/v7/research-program.md#data-evaluation-window-03).
Main9956446 unchanged; branch `development/v7-data-evaluation-preparation` is
UNMERGED/NOT_DEPLOYED. New bounded execution and formal research decisions need Human.

## Window04 reusable metadata audit

`V6.experimental.v7_e1_readiness` joins existing frozen contract/config and pinned
completed-result metadata checks. Output VERIFIED_METADATA_ONLY never authorizes
execution; matrix payload NOT_VERIFIED, checkpoint payload NOT_LOADED. It does not
re-evaluate predictions or replace recovery evaluator provenance gates. See
[window evidence](../../research/v7/fair-evaluation-preparation-evidence.json).

## Window04 CPU logging reliability

Branch-local W04B-LOG changes `v7_e1_colab_runtime.run_logged` so background log
write/decode failures reach the caller with the original cause and log path,
stop a running child through the existing cleanup, and fail even if the child
already exited. Existing success/nonzero/heartbeat behavior remains covered.
Six CPU fixture tests and independent review verify this scope; no live Colab,
Drive storage durability or research readiness certification. UNMERGED / NOT_DEPLOYED.
See [bounded evidence](../../research/v7/window04-cpu-continuation-evidence.json).
