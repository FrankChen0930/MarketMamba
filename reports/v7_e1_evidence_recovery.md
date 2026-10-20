# E1 Frozen Evidence Recovery — 2026-10-21

Status: **STOP_REPLAY_MATRIX_IDENTITY_UNAVAILABLE**.
Source base `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e`; isolated
`recovery/e1-frozen-evidence`, `.worktrees/e1-frozen-evidence`.
Machine inventory: `research/v7/e1-rolling-origin-refresh-v1/evidence-recovery-20261021.json`.
This reuses the existing E1 evidence namespace, not a new experiment or memory system.

## Recovered / verified evidence

- Fixed E1 contract `044eef7` and completed result `eec63c5` remain accessible.
  A failed lookup in the older research checkout was a path/context mismatch,
  not a missing Git artifact; durable `commit:path` recovered the source.
- Seeds 17/29 results, pointers, run summary, evaluation and report (seven files)
  match the completed-result manifest hashes. Both prediction hashes match.
- Six checkpoint files match pointer SHA-256 and byte size, and historical
  inventory hashes: best epoch 1, previous epoch 5, latest terminal epoch 6 for
  both seeds. Directory inventory contains exactly these three `.pt` generations
  per seed. Epoch 2–4 retention cannot be recovered from these local directories.
- Seven postrun diagnostic artifacts match fixed `ef020e5` byte-for-byte.
  No diagnostic statistic was recomputed or promoted; conclusions remain
  EPOCH_DEGRADATION_INCONCLUSIVE / REGIME_DEPENDENCE_INCONCLUSIVE.

Expected contract `fea9948f...58d29152`, config `acdfe7fb...97a2d4`,
matrix **`a590b280f0700dcf36fde9ac51e767d735cc2737b613cbb3b49ef3f472947f62`**.
The contract retains 48 features, d_model64/d_state32, sequence60, no graph;
v2 historical proxy labels P0, train through 2025-11-18 and evaluation from
2026-01-02. These are verified contract/result claims, not proof that uninspected
checkpoint payloads or absent arrays satisfy them.

## Matrix discovery boundary

Inspected 14 `manifest.json` files under local Data, deliveries and worktrees,
excluding environment/dependency trees and without following arbitrary symlinks.
No top-level logical identity matches the historical E1 matrix.
Two matching directory names are explicit migration fixtures:
`matrix-20260920` and a two-row matrix `6c68fb19...`; both are rejected.
Four top-level E1 ZIP central directories contain no E1 matrix namespace or
`X.npy` entry. This is directory inspection, not full ZIP payload certification.

The expected local and Windows matrix paths do not exist; the historical
`/content/drive/MyDrive/...` runtime is not mounted here. Connector discovery
returned `unsupported call`, so remote Drive contents were not queried.
**Not found locally does not mean permanently lost.**
No production data or historical artifacts were changed or regenerated.

## Safe inspection / replay gate

ZIP structure and `data.pkl` presence were inspected for all six files.
For both best checkpoints, `pickletools.genops` inspected GLOBAL references
without executing any reducers. NumPy reconstruction globals are present.
This is static structure evidence, not a certified safe loader or a semantic
checkpoint validation.

Existing `v7_e1_checkpoint.py` calls `torch.load(..., weights_only=False)`.
It was not invoked or altered. Installed CPU Torch metadata says 2.10.0+cpu;
version alone does not validate a checkpoint. No unsafe pickle load, extra
allowlist, model construction, inference, optimizer step or training occurred.

Replay stops because the frozen matrix cannot be identity-validated. Do not
build a new matrix and call it historical recovery. Next prerequisites:

1. Locate the original manifest and all eight serialized array files; verify
   full file checksums, logical identity, ordered features, source hashes,
   configuration and frozen split using existing contracts.
2. Establish safe restricted checkpoint inspection with identity/model-key/
   tensor-shape validation; fail closed on unsupported objects. Hash matching
   alone does not certify pickle execution safety.
3. Only then consider bounded CPU/no-grad replay on frozen approved support.
   No tuning of hypotheses, label definitions or test-set purpose.

## Outcome and Human decision

No product bug or engineering rework was needed; one discovery path assumption
was corrected by using fixed Git evidence. Read-only identity checks passed.
Progress is a precise verified surviving-artifact inventory and a bounded stop,
not a successful replay claim.

Most useful next decision: can Human supply the original Drive/shared-root
matrix location or preserved archive? If it is irretrievable, explicitly retain
the final-support epoch claim as unresolved and separately decide any future
research design; do not silently regenerate history or start E2 training.
Strict Phase 0 remains **STOP**, and completed E1 research remains
historical-simulation evidence, not production/trading approval.

## Independent review and evidence identity

Independent review found no Required or Critical findings. The reviewer
recomputed all six checkpoint hashes, both prediction hashes and the fixed
Git contract/diagnostic/inventory hashes, and confirmed that byte integrity
is not represented as checkpoint semantic validation or successful replay.

Recovery inventory: `research/v7/e1-rolling-origin-refresh-v1/evidence-recovery-20261021.json`
SHA-256: `3fb97ae61e60751861ecef69d4df1037bf88425d11e5c4eb0a0e7bc853a7a73f`.
