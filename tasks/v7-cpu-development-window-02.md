# V7 CPU development window 02

Status: IN_PROGRESS, UNMERGED / NOT_DEPLOYED. Human authorization: current
“Market Mamba V7 — Autonomous CPU Development Window 02” request. At most five
meaningful CPU engineering units; no requirement to fill the limit. Reuse existing
`.worktrees/v7-cpu-window-01`, branch `development/v7-cpu-window-01`, starting
`7eb9cd9027a312a06d526666e4ad31d2255ff4f2`. Main stays `83a6a2e`.
No new worktree/system or historical artifact replacement.

## Preflight and execution accounting

- Main and candidate clean initially; 11 checkouts including main observed.
- Goal API initially returned null; create_goal succeeded with this V7 objective,
  thread `01a15a57-babe-79e2-9d27-018f19698e0e`, createdAt `1792517093`.
  No MAS goal/database/config was modified. Current tool binding is observed;
  automatic future continuation has not been exercised or certified.
- Python 3.12.3; portfolio stdlib only; CUDA_VISIBLE_DEVICES empty for tests.
  MARKETMAMBA_DATA_ROOT unset. Main Data/processed_v6 realpath:
  `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`; unused.
  PIT tests use existing recovery venv, not a new environment or training run.
- Goal-reported usage is scoped telemetry, not independently verified aggregate
  accounting across agents; no exact aggregate/hard-cap claim. Test logs retain
  measured durations. Extra Human decisions/interventions: 0.
- Contract: approved `docs/superpowers/specs/2026-09-16-v7-replayable-portfolio-engine-design.md`,
  executable engine normalization and journal tests. Synthetic fixtures only.

## Ranked backlog and checkpoints

| Priority / unit | Planning status | Source, dependency, decision and evidence |
|---|---|---|
| U1 normalized quote-key collision | COMPLETED_VERIFIED | Window 01 reproduced backlog; engine `_normalize_quotes` rejects duplicate normalized ticker. Two-line journal fix preserves same rule before persistence. RED 4 tests / 25 failing subcases; GREEN 47 portfolio/process tests, 1.913s. Initial fix rounds 1, review rework 0. Independent 47 PASS (2.158s); independent old-source 25 failures; no Critical/Required. Governance 23 PASS (0.087s); health WARNING/no FAIL. |
| U2 duplicate JSON member ambiguity | READY, reproduced | Journal hash-chain/canonical replay and error-retention contract. Disposable public journal probe inserts `"price":"999","price":"10"` without recomputing hash; reader accepts two records. Independently reproduced; selected under autonomous bounded correctness authorization, separate RED/fix/commit. |
| E1 identity-matched matrix/replay | BLOCKED | `11ae71d:reports/v7_e1_evidence_recovery.md`; no new original matrix location. Do not recreate frozen data or repeat unchanged search. |
| Strict executable labels | BLOCKED | Phase 0 STOP, verified 5d/10d=0; missing qualified tradability/PIT evidence. |
| Signal anatomy/portfolio validation Phase 1–5 | HUMAN_DECISION_REQUIRED | Existing validation design gates on correctness PASS; synthetic engineering is not research admission. |
| E2 hypothesis/regime series | HUMAN_DECISION_REQUIRED | No frozen authorized research contract. |
| GPU/Colab training | BLOCKED | Outside CPU window, separate authorization/runtime required; cannot cure absent historical identity. |
| Production B–F | HUMAN_DECISION_REQUIRED | Existing roadmap, broad new subsystems/promotion outside this authorization. |

U1 modifies only journal normalization/test plus this plan, evidence and relevant
canonical memory. Coverage: 24 collision combinations, invalid duplicate retry,
legal unique whitespace/empty quote, hash-valid persisted collision refusal,
unchanged bytes/state, reopened valid retry and existing real-process recovery.
Reader already refused persisted normalized collision through canonical-payload
comparison; do not claim that negative reader case was newly failing.

Unit protocol: Plan → RED → minimal fix → focused/regression → independent review
→ at most two total implementation rounds → knowledge checkpoint → local commit.
No research correctness approval, merge/push/deploy, GPU, formal data writes or
Windows checkout access. Continue only with independently justified READY work.

## Integration dependencies and resume

Window 01 commits `585ad65`, `f42aef0`, `7eb9cd9` and E1 evidence ancestor
`11ae71d` are not in development main. Window 02 builds on those exact fixes;
never cherry-pick as if independent of locked pre-write validation/shared reads.
Resume by inspecting branch HEAD/status and this backlog/evidence, then verifying
source/log hashes. Strict Phase 0 STOP and E1 BLOCKED remain unchanged.

U1 implementation/source and raw log hashes are bound in `research/v7/cpu-window-02-evidence.json`; commit identity is its containing commit. No research authority changed.
