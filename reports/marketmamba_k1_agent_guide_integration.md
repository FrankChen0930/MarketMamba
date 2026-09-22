# MarketMamba K1 Agent Guide Integration

## Human Review Fixes

- Seed43 now separates stale pointer metadata (`best_epoch=8`, pointer-named generations missing) from the surviving orphan checkpoint (epoch 15, step 39,915, non-terminal, contract/matrix matched). The orphan remains convergence evidence only and is neither a formal seed nor promotable/directly resumable.
- Phase A now records Tasks 1–3 as completed/reviewed, Task 4 as implemented with sanitizer follow-up fixes but correctness approval not established, and Task 5 as not started. The branch remains `UNMERGED`, `NOT_DEPLOYED`, and `HUMAN_REVIEW_REQUIRED`.
- Accepted decisions remain unchanged: seven-day runtime-observation TTL, historical-simulation `PASS`, strict Phase 0 `STOP`, verified executable labels 0/0, formal corrected E5 seeds 17/29, and E1 `SPECIFIED / PACKAGED / NOT VERIFIED COMPLETE`.

## Canonical Promotion

After reconciliation and validation, the Markdown and JSON project maps were promoted to `CANONICAL_CURRENT_PROJECT_MAP`. This means accepted current navigation/state authority, subject to same-change semantic updates and explicit supersession; it does not mean immutable final state.

## AGENTS Changes

`AGENTS.md` is now stable policy/navigation rather than a second project-state database. It defines the required bootstrap order, destructive-operation safety, Git/worktree discipline, fail-closed correctness, evidence preservation, research/production separation, runtime observation rules, testing, Colab durability invariants, and the explicit knowledge-impact gate. Volatile branch, scheduler, experiment, blocker, count, and current-state facts moved behind project-map references.

## CLAUDE Shrink and Archive Decision

The former 990-line, 65,932-byte guide mixed unique Claude orchestration assets with extensive stale project state. It is preserved verbatim under `docs/archive/legacy-claude-project-guide.md`, with an explicit `HISTORICAL_ONLY / NON_AUTHORITATIVE` header.

The new `CLAUDE.md` is a thin wrapper linking `AGENTS.md`, `Current_State.md`, and `Authority_Map.md`. It retains only references to optional `.claude/agents` roles, the advisor coordination protocol, and builder/verifier independence. These assets cannot override canonical policy or machine evidence.

## Validation Result

- Project-map validator checks formal seeds 17/29, seed43 stale-pointer/orphan distinction, no loadable epoch-8 claim, strict STOP, simulation PASS, E1 not verified complete, Phase A not deployed, Task 4 approval pending, and Task 5 not started.
- Agent-guide validator checks bootstrap order and links, stable policy markers, Claude references, a 5 KB wrapper threshold, absence of duplicated volatile current state, and the historical archive marker.
- JSON parsing, Markdown links, branch-qualified references, authority uniqueness, diff hygiene, scope, and clean-worktree checks are required before completion.

## Remaining K1 Gaps

- Full `knowledge/01_Domains/` migration has not started.
- Full automated drift/staleness checker is not implemented.
- README/OVERVIEW and Obsidian migration remain out of scope.
- Runtime observations still require periodic external re-observation.
