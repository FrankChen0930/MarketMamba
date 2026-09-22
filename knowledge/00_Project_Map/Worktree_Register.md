# Worktree Register

> Updated from `git worktree list --porcelain` and branch verification on 2026-09-22 after the four authorized K checkouts were retired. This is a registry of checkouts and responsibility, not authority to merge or remove them. `UNKNOWN` means creation/ownership was not verified. Update in the same coherent change when a branch or responsibility changes; a fresh runtime/worktree observation needs its own date.

| Checkout under `.worktrees/` | Branch / HEAD | Purpose and owner/task | Base | Current state | Authority carried / dependencies | Created / retired |
|---|---|---|---|---|---|---|
| `project-knowledge-k0-discovery` | `docs/project-knowledge-k0-discovery` / `c802f52` | K0 discovery; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | K0 inventory/report; K1/K2 derived from it. Verify later branches retain references before removal. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `project-knowledge-k1-bootstrap` | `docs/project-knowledge-k1-bootstrap` / `31dd74a` | K1 map/bootstrap; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | Current/Authority map bootstrap and legacy Claude archive; K2 successor. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `project-knowledge-k1-domains` | `docs/project-knowledge-k1-domains` / `2c6c6f3` | domain notes; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | Twelve domain notes; K2 successor. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `project-knowledge-k2-governance` | `integration/project-knowledge-k1` / `18152d0` | K2 governance; owner UNKNOWN | `main@c0babe0` merge base | retired checkout; branch preserved; `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT` | Knowledge checks, authority/history lineage. Adopted in main by governance consolidation, but compare source commits and generated refs before retiring checkout. | UNKNOWN / 2026-09-22 14:58–15:00 +08:00 |
| `v7-corrected-e5-2026-diagnostic` | `feature/v7-e1-rolling-origin-refresh` / `044eef7` | corrected E5/E1 bundle; owner UNKNOWN | `main@7bdf2e4` merge base | clean; research reference | E1 frozen contract, runner, notebooks; bundle and research evidence depend on branch. | UNKNOWN / N/A |
| `v7-e1-colab-recovery` | `fix/v7-e1-colab-recovery` / `0f06e36` | Colab recovery/shared Drive; owner UNKNOWN | `main@7bdf2e4` merge base | clean; recovery reference | Recovery logic and acceptance evidence; validate artifact dependencies before retirement. | UNKNOWN / N/A |
| `v7-e1-postrun-diagnostics` | `research/v7-e1-postrun-diagnostics` / `1d8e23f` | E1 admitted results/postrun diagnostics; owner UNKNOWN | `main@7bdf2e4` merge base | clean; latest research evidence | E1 result manifest, month/epoch diagnostic, K2-derived knowledge. Branch-qualified result authority persists. | UNKNOWN / N/A |
| `v7-historical-pit-reconstruction` | `feature/production-phase-a-shadow-ledger` / `f5b0ff3` | Phase A shadow ledger; owner UNKNOWN | `main@7bdf2e4` merge base | clean; unmerged, not deployed | Ops code/tests, Task 4 review and Task 5 replay pending. Directory name is stale; branch is authoritative identity. | UNKNOWN / N/A |

**Safe-removal preconditions for every row:** record owner approval and intended retirement date; confirm checkout clean; retain branch/commit/history; verify no unique uncommitted or ignored artifact is required; verify all authority/manifest absolute-path references and merge dependencies; inventory rebuildable runtime separately. This register grants no cleanup authorization. New worktrees require a bounded objective and a check that an existing checkout does not already serve it.

## Phase 3B-2 retirement preparation (observed 2026-09-22)

All eight checkouts were clean in `git status --short --branch`; their branch refs and commits remain present. A clean checkout alone never authorizes removal. The readiness labels below are planning labels only.

| Checkout | Readiness | Remaining authority / local dependency | Before any removal |
|---|---|---|---|
| `project-knowledge-k0-discovery` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | K0 history/report retained in Git and main navigation; no ignored payload observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `project-knowledge-k1-bootstrap` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | Bootstrap and legacy Claude history in Git; main governance supersedes active rules; no ignored payload observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `project-knowledge-k1-domains` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | Domain note lineage in Git/main; ignored Python bytecode observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `project-knowledge-k2-governance` | `RETIRE_READY_PENDING_HUMAN_APPROVAL` | K2 tools/rules adopted and evolved in main; ignored Python bytecode observed. | Confirm owner and links, retain branch/ref, approve removal. |
| `v7-corrected-e5-2026-diagnostic` | `KEEP_AUTHORITY` | E5/E1 bundles, frozen result/evidence and branch-only diagnostics remain; ignored bytecode present. | Resolve bundle/result and artifact links before retirement review. |
| `v7-e1-colab-recovery` | `KEEP_AUTHORITY` | Drive rescue/migration and recovery evidence remain branch-qualified; local `.venv` and ignored review records require separate reproducibility/retention assessment. | Preserve evidence and verify environment/artifact identity; obtain owner approval. |
| `v7-e1-postrun-diagnostics` | `KEEP_ACTIVE` | Matrix recovery/replay and diagnostic authority remain open. | Complete task and preserve result evidence first. |
| `v7-historical-pit-reconstruction` | `KEEP_ACTIVE` | Actual branch is `feature/production-phase-a-shadow-ledger`; Task 4 correctness approval and Task 5 acceptance remain open; ignored review records exist. | Complete independent Phase A approval and artifact review first. |

A NUL-safe path comparison against main found branch-tracked paths absent at the same main path: K0=1, K1 bootstrap=1 (historical archive path), K1 domains=3 (archive plus two K1 task notes), K2=5 (archive plus K1/K2 task notes). These remain preserved in their branch commits; retirement requires retaining branch refs/history and confirming no checkout-path consumer. No checkout, branch, ignored file or artifact was removed in Phase 3B-2.

## Phase 3B-3A completed retirement (2026-09-22)

Human authorization in the Phase 3B-3A task covered only the four K checkouts. Pre-removal verification found each clean, its branch ref pointing to the recorded HEAD commit, no untracked user source, no symlink, no active absolute-path consumer in AGENTS/knowledge/docs/scripts/tests/tasks, and no required unique local artifact. `archive/V5.5_repo` was an empty checkout directory for a historical gitlink at `f9f5924`; that pointer remains in each branch tree, but its target object is unavailable in the current repository (`git cat-file -t` failed). No local target contents were present to lose. K1 domains had four and K2 twelve ignored `.pyc` files; normal `git worktree remove` removed them with their checkouts. No force option was used.

| Retired checkout | Retired at (+08:00) | Branch and preserved HEAD | Remaining authority | Local disposition |
|---|---|---|---|---|
| `project-knowledge-k0-discovery` | 2026-09-22 14:58–15:00 | `docs/project-knowledge-k0-discovery@c802f52` | Historical K0 discovery in branch Git history and main project map | Checkout removed normally; no ignored files |
| `project-knowledge-k1-bootstrap` | 2026-09-22 14:58–15:00 | `docs/project-knowledge-k1-bootstrap@31dd74a` | Historical K1 bootstrap/legacy guide in branch Git history | Checkout removed normally; no ignored files |
| `project-knowledge-k1-domains` | 2026-09-22 14:58–15:00 | `docs/project-knowledge-k1-domains@2c6c6f3` | Historical domain bootstrap and two task notes in branch Git history | Checkout removed normally; four reproducible `.pyc` files removed with checkout |
| `project-knowledge-k2-governance` | 2026-09-22 14:58–15:00 | `integration/project-knowledge-k1@18152d0` | Historical K2 governance and four K1/K2 task notes in branch Git history; current governance remains in main | Checkout removed normally; twelve reproducible `.pyc` files removed with checkout |

Each preserving branch ref and commit remained after removal. The `archive/V5.5_repo` **pointer** and branch-only task-note contents remain durable through those commits; the gitlink target content is not locally available and its external provenance remains UNKNOWN. Same-path copies in main were not required. Four research/operations worktrees remain active/preserved and were not touched. Exact observed completion time per individual remove command was not captured; interval above bounds this session and avoids false precision.
