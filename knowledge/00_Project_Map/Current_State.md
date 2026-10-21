# MarketMamba Current State

> Status: `CANONICAL_CURRENT_PROJECT_MAP`

## Observation Metadata

| Field | Value |
|---|---|
| `observed_at` | `2026-09-22T13:25:21+08:00` for this runtime reconciliation; individual observations retain their own timestamps |
| `owner` | Human + project agents |
| Repository context | WSL development `main@3b0df0f` at observation; Windows operational `main@8b82a68`; research authority remains branch-qualified |
| State updated | `2026-10-21` engineering/decision update; Scheduler observation remains `2026-09-22T13:21:35+08:00`, not refreshed |
| Source basis | Reviewed K0/contracts plus [new Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) and [operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) |
| Human review | Prior research/Phase A scope accepted `2026-09-20`; the new 2026-09-22 runtime observations are agent-observed and have not received separate human review |
| Runtime freshness | External scheduler/deployment observations expire after 7 days and become `STALE_OBSERVATION`; they are retained until re-observed |
| Repository freshness | Contracts do not expire by time; commit identity or explicit supersession changes them |

Semantic change owners must update this file in the same change. K0 remains discovery evidence. `CANONICAL_CURRENT_PROJECT_MAP` means current accepted navigation/state authority, subject to semantic updates and explicit supersession; it does not mean immutable or permanently final.

## Cross-domain window02 execution checkpoint

Isolated `development/v7-cross-domain-window-02` from main closurec1179fe,
UNMERGED/NOT_DEPLOYED. W02-DATA/PREP/EVAL/MODEL selected under current Human grant;
[unique backlog](../../research/v7/research-program.md#cross-domain-window-02)
owns acceptance and state. No production parser edits or training permitted.
W02-DATA completed7-case offline diagnostic with independent review0 Required;
TWSE/TPEx source fixes and MOPS policy are now explicit Human decision items.
[Evidence](../../research/v7/cross-domain-window02-evidence.json).

## Charter integrated and window 02 authorized — 2026-10-21

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved exact
`f7af348710467ac381b2a84e8d192670ced89932`; WSL main ff-only from
`6d74d7384caa5fa90b1b050ce507bfb8282f19df` to that SHA. This containing
closure commit is docs/evidence only. [Integration evidence](../../research/v7/charter-integration-20261021.json)
records pre/post254 tests each (PIT49, non-training E1 31, portfolio67, ops81,
fetch3+governance23), 6 source/4 authority hashes,75 protected blobs unchanged,
17 clean worktrees and independent0 Critical/Required. No tiny training or GPU probe run.
Window01's UNMERGED wording below is historical and superseded only for integration.

[New Human decision](../02_History/Decisions/20261021-charter-integration-window02.json)
authorizes one next isolated CPU window, maximum3–5 meaningful units. Current
selection/checkpoint remains the [unique program](../../research/v7/research-program.md).
Discover Data, Model Research and Experiment Preparation before optional system
work; do not change V6 production. StrictSTOP/E1BLOCKED/Human research gates persist.
The fixed candidate branch remains unchanged. No push/deploy/Windows/MAS activity.

## Autonomous charter first window — 2026-10-21

**WINDOW_COMPLETE / STOP_AFTER_FIRST_WINDOW / UNMERGED / NOT_DEPLOYED.**
Human's [Charter](../GOVERNANCE.md#autonomous-development-charter--2026-10-21)
is persisted in existing governance, and the
[single backlog/checkpoint](../../research/v7/research-program.md#autonomous-charter-window-01)
records the complete first window. Branch `development/v7-autonomous-charter`
starts at unchanged main `6d74d7384caa5fa90b1b050ce507bfb8282f19df`.
Five units: Charter `be7998d`, E1 split completeness `3f07f80`, research decision
preparation `5847afc`, checkpoint fallback `2c60415`, snapshot pre-publication
validation in the containing closure commit. No further work starts in this window.

[Evidence](../../research/v7/autonomous-window-01-evidence.json): E1 33 + PIT49 +
governance23 =105 PASS; independent reviews0 Critical/Required after checkpoint's
second/final repair; snapshot one round; four research sources verified. Source
and output hashes/JSON/syntax/diff checks pass; knowledge health WARNING, no FAIL.
CPU synthetic engineering and PROPOSED research preparation only; no historical
checkpoint payload/matrix/data reads, external network, new training, GPU or MAS.
Existing E1 unittest includes a one-step synthetic CPU smoke. Loader remains
trusted-local and unrestricted: no historical-pickle safety certification.

This supersedes older no-next-program or in-progress statements solely for this
completed window. H0 remains locally integrated; Strict Phase0 STOP, E1 Frozen
Matrix BLOCKED, H2 and production/research gates unchanged. Another session reads
Charter + this backlog + fixed branch evidence, verifies Git/runtime, and waits
for a new bounded window grant before executing. No merge/push/deploy authorized.
No confirmed unexecuted READY item remains in this window; do not infer global
absence of defects or repeat E1 searches without new evidence.

## Portfolio History H0 completed — 2026-10-21

**LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED.** Human accepted the limited
fixture Product Slice and [approved exact4a66928](../02_History/Decisions/20261021-portfolio-final-h0-approval.json).
WSL main fast-forwarded from `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e` to
`4a66928aa4c9bd8a1ddf99a5b42866e9b22efb74`; subsequent containing commit is
**documentation/evidence only**, not a different product candidate. Pre/post CPU,
API/HTTP, frontend and real Chrome font/selection/error checks passed.
[Integration report](../../reports/v7_portfolio_h0_integration_closure.md),
[evidence](../../research/v7/portfolio-h0-integration-closure.json).
CPU01/02, PORT01–04, UX01–04, final UX and font correction are in local main.
This supersedes historical UNMERGED/H0-pending/UX-review-pending prose below for
these units only. Original reports, fixed branches and all16 worktrees are retained;
audit7ecb920 remains separate historical failed-preflight evidence, not merged.
No Required/Critical remains within this tested slice. Main is a valid source
baseline for future separately authorized CPU work; no new Program starts now.
Strict Phase0 STOP, E1 Frozen Matrix BLOCKED, H2 unapproved, research correctness
unapproved. No push/deploy/Windows operational change/training. STOP after closure.
The existing UI's static UNMERGED badge is historical candidate wording, not a live
Git integration indicator; canonical status above owns integration authority.

## Final H0 browser blocker resolved — 2026-10-21

Human [authorized only the late-font visibility repair](../02_History/Decisions/20261021-portfolio-font-blocker-authorization.json).
New branch `fix/v7-timeline-font-visibility` starts from exact `6c894794` product
candidate; `audit/v7-h0-preflight@7ecb920` is evidence only, NOT an ancestor.
True-font RED reproduced desktop3.5px/mobile2.5px clipping. Existing ResizeObserver
now observes all event rows as well as the fixed-height container; GREEN confirms
font-driven layout correction without page movement, focus loss or data change.
[Report](../../reports/v7_portfolio_font_visibility.md),
[evidence index](../../research/v7/portfolio-font-visibility-evidence.json).
Original Required finding RESOLVED within tested fixture scope; bounded preflight
engineering GO, pending Human approval of the new fixed SHA. Final candidate is the
containing commit, not the audit commit. Main remains `83a6a2e`; no merge occurred.
Human limited UX product acceptance remains in force; earlier pending-UX-review
language below is historical. Old H0 authorization does not extend to this new SHA.
UNMERGED / NOT_DEPLOYED; H2 unapproved, Strict Phase0 STOP, E1 Frozen Matrix BLOCKED.
STOP here: no further UX/research work authorized. Native Goal creation was rejected
because the previous H0 goal remains paused; no automatic continuation claim.

## Project Summary

**Final UX closure (2026-10-21):** Human accepted the dual-mode direction and
[authorized only two finishing changes](../02_History/Decisions/20261021-portfolio-ux-closure-decision.json).
On `development/v7-portfolio-ux`, baseline `1553a59`, timeline selection stays visible
inside its container, with explicit selected text and no page anchoring jumps;
event explanations use observable exact-state differences and conservative Chinese.
Both tasks COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED. Final identity is the
containing closure commit (see `git log -1`); [report](../../reports/v7_portfolio_ux_closure.md)
and [evidence index](../../research/v7/portfolio-ux-closure-evidence.json).
Desktop/mobile general seven events, desktop advanced exact values and mobile advanced
basic smoke verified. No additional mobile advanced visual work. Stop for Human final
review; H0/H2 remain unapproved, Strict Phase 0 STOP, E1 BLOCKED. No further UX READY work.
Earlier UX/PORT evidence below is retained as historical acceptance, not rewritten.

**Completed UX refinement (2026-10-21):** Human approved UX-01–04 default dashboard /
advanced research view on isolated `development/v7-portfolio-ux` from fixed `5a2bc6f`.
[Decision](../02_History/Decisions/20261021-portfolio-ux-human-decision.json),
[skills/design/acceptance](../../reports/v7_portfolio_ux_acceptance.md).
UX-01–04 COMPLETED_VERIFIED within fixture UI scope: desktop1440/mobile390
Chrome155, both modes/all7events/exact values/keyboard PASS. Implementation
`764dfbf`; final evidence in containing UX04 commit. Stop for Human UX Product
Slice Review. Focused lint PASS; full frontend baseline58errors/3warnings unchanged.
[UX evidence](../../research/v7/portfolio-ux-evidence.json).
UNMERGED / NOT_DEPLOYED; existing PORT completion remains preserved.
No H0/H2, accounting, data or production authority change. Strict STOP persists.


**Strategic recovery / integration candidate (2026-10-21):** this Human-authorized
planning window is complete; H1 was subsequently approved for the fixture slice. `main@83a6a2e`
remains unchanged; candidate `integration/v7-cpu-roadmap` starts at fixed
`4c388de`. Five CPU fixes already form continuous ancestry; no cherry-picks,
source changes or main merge. Joint 60 portfolio/health/audit, 47 PIT, 81 ops,
3 fetch-only, 23 governance and 6 historical-output fixtures PASS. Independent
review: GO for local development candidate only, no Critical/Required.
[Integration/objective/status evidence](../../reports/v7_strategic_roadmap_recovery.md)
and [hash/test index](../../research/v7/cpu-integration-candidate-20261021.json).

The **single active cross-domain development backlog/program** is the dated
section of [existing research program](../../research/v7/research-program.md#current-development-program--2026-10-21).
Prior CPU task files remain completed-window evidence. Product intent includes
trustworthy data, relative rankings, weight-based simulation and performance;
current engine is complete only as a deterministic component. PORT-01–04 now have explicit
[Human H1 fixture-only engineering approval](../02_History/Decisions/20261021-portfolio-history-h1.json).
Completed isolated branch `development/v7-portfolio-history` from `bebf19e`:
PORT-01 COMPLETED_VERIFIED (7 focused / 67 combined, independent review closed
one compatibility repair); PORT-02 COMPLETED_VERIFIED (4 new + 4 existing API
tests, independent live startup review); PORT-03 COMPLETED_VERIFIED (8 focused +
8 existing JS cases, build and real desktop/mobile/keyboard acceptance); PORT-04
COMPLETED_VERIFIED (actual HTTP/restart, browser corruption/recovery, 47 PIT,81 ops,
3 fetch-only,23 governance). All four units independently reviewed; stop for Human
Product Slice Review. No additional H1 implementation remains READY.
[Evidence/checkpoint](../../research/v7/portfolio-history-evidence.json).
UNMERGED / NOT_DEPLOYED; H0/H2 absent. Strict Phase 0 STOP and E1 matrix BLOCKED
persist. MAS Core Live unverified; no production/runtime authority changed.
Native Goal is V7-bound; automatic future continuation remains unverified.


**CPU-only window 02 (2026-10-21, WINDOW_COMPLETE):** reused window 01 lineage
from `7eb9cd9` on `development/v7-cpu-window-01`; main stays `83a6a2e`.
[Current backlog/checkpoint](../../tasks/v7-cpu-development-window-02.md) and
[evidence index](../../research/v7/cpu-window-02-evidence.json).
U1 `735ae6f` rejects normalized quote collisions before persistence; U2 (commit
containing this closure) rejects duplicate JSON object keys at every depth.
Both COMPLETED_VERIFIED: final 60 portfolio/process/health/audit, 47 PIT,
23 governance PASS; independent review no Critical/Required, one fix round each.
No remaining confirmed READY item in inspected backlog; stop after two units.
Goal API created a correctly bound V7 objective after observing no goal; future
automatic continuation unverified. No MAS/global settings touched. This supersedes
window 01 next-action status only. UNMERGED / NOT_DEPLOYED, no research approval;
strict Phase 0 STOP and E1 matrix BLOCKED remain unchanged. Extra Human decisions 0.

**CPU-only window 01 (2026-10-21):** Human authorized bounded autonomous
engineering without training or research changes. Isolated
`development/v7-cpu-window-01`, base `11ae71d`; main remains `83a6a2e`.
[Ranked backlog and acceptance](../../tasks/v7-cpu-development-window-01.md)
select three existing portfolio scheduling/journal recovery contract units.
All three complete: early-signal state preservation, pre-write semantic
validation and shared-lock reading. Final bounded suites: 53
portfolio/process/health/audit, 47 PIT, 23 governance PASS; independent review
has no Required/Critical findings. See the
[evidence index](../../research/v7/cpu-window-01-evidence.json).
Historical next READY at window 01 closure: normalized quote collisions;
now fixed by window 02 above. Window 01 stopped after three units.
E1 matrix remains BLOCKED; strict Phase 0 STOP and Phase 1–5 gates unchanged.
This branch is UNMERGED / NOT_DEPLOYED; no live MAS Core capability is assumed.

**E1 recovery checkpoint (2026-10-21, branch-local):**
`recovery/e1-frozen-evidence` verified six checkpoint byte hashes, two prediction
hashes, seven result/pointer/evaluation files and seven fixed diagnostic files.
Historical matrix `a590b280...947f62` was not found in the explicit local search;
same-name fixtures rejected, remote Drive not inspected.
**STOP_REPLAY_MATRIX_IDENTITY_UNAVAILABLE**. No checkpoint deserialization,
matrix generation, inference or training. See
[inventory/report](../../reports/v7_e1_evidence_recovery.md).
Human next step: locate the original matrix/manifest or exact archive path.
Do not repeat completed Phase A work or infer permanent matrix loss.

**Current integration closure (2026-10-21):** Human authorized fixed `89f9496`;
WSL development main fast-forwarded from `50c5b78` to that exact candidate.
Pre/post checks each passed 81 Operations, 47 PIT/Snapshot, 3 fetch-only,
38 Portfolio and 23 governance tests. See
[closure](../../reports/production_phase_a_integration_closure.md).
Phase A is `IN_DEVELOPMENT_MAIN / NOT_DEPLOYED`; old pending-merge language below
is historical. Remote tracking was not fetched; Windows operations are untouched.
E1 bounded evidence recovery is now authorized in isolation, not new training.

**2026-10-21 current Phase A decision:** Human explicitly approved Task 4 limited
engineering correctness at `1e119ef`, after revalidation of its index, 12 raw
SHA-256 records and 36 generation digests. Decision provenance:
[Human record](../02_History/Decisions/20261021-task4-human-approval.json).
This supersedes older pending-Task-4 observations below, not research or
production gates. New isolated `integration/phase-a-task5` starts at `63a040d`
and cherry-picks the 17 Phase A commits `9efed7b..1e119ef` with `-x` provenance.
Integrated Operations files match `1e119ef`; snapshot files match `63a040d`.
Joint regression: 75 ops, 47 PIT, 3 fetch-only, 18 governance PASS.
Task 5 limited offline engineering correctness is now **HUMAN_APPROVED** at
fixed `1c5250e67461456b7ca0b361f8da3d9cb3f22514`, after artifact/digest verification.
The [Task 5 Human decision](../02_History/Decisions/20261021-task5-human-approval.json)
retains the explicit scope and exclusions. Original acceptance/report artifacts
remain unchanged historical pre-decision evidence.
Final Operations 81, governance 19, non-training portfolio regression 38 PASS;
joint PIT 47 and fetch-only 3 PASS. See the
[Task 5 report](../../reports/production_phase_a_shadow_ledger.md) and
[evidence index](../../research/production-system/phase-a/task5-evidence-index.json).
Ten fixed replays, real-process restart/concurrency and measured overhead are
offline engineering evidence, not production certification. Final engineering
review recommends **GO for development candidate integration**; this is an Agent
assessment, not final Phase A Human acceptance or permission to merge.
See [final review and guarded integration procedure](../../reports/production_phase_a_integration_review.md).
The decision-validator change passes 23 governance tests (including rejection
of mutated, missing or corrupt Task 5 approval evidence).
The earlier review had no merge authority; the subsequent explicit integration
decision above authorized only local development main. No push/deployment/training.

**2026-10-20 bounded engineering recovery (branch-local):** base development
`main@50c5b785e8c373913c3301a056497eb955f36681`; candidate
`fix/v7-snapshot-immutability`, `UNMERGED / NOT_DEPLOYED`. One engineering unit
fixes immutable V7 source snapshot publication; it does not promote research
evidence or close any correctness gate. See the checkpoint below.

**Historical window 01 objectives (superseded by strategic checkpoint above):** CPU window 01 complete; next bounded engineering candidate
is journal quote-collision refusal. Separately await original E1 frozen
matrix/manifest location.
Bounded local inventory is complete; replay is stopped until identity and safe
inspection gates can be met. No new research hypothesis or training is authorized.

MarketMamba contains separate production-data, legacy model runtime, V7 correctness, research-training, historical-simulation, portfolio, delivery, and operations-control-plane subsystems. Production acquisition uses mutable provider data; V7 research authority uses versioned PIT/provenance evidence. The current corrected research baseline is `E5-PIT-Clean-v1`, with 48 features, no graph and no industry neutralization. Historical simulation is `PASS`, while strict trading correctness remains `STOP`. The observed production scheduler runs fetch-only; full V6.2 inference/portfolio/publication is `DISABLED`. E1 is a completed research experiment with `PASS_REFRESH_HYPOTHESIS`, not trading evidence. Phase A operations work is `IN_DEVELOPMENT_MAIN` and `NOT_DEPLOYED`; historical pending-integration paragraphs below are retained observations.

The `environments/v7-colab-2026.04/` identity files are preserved in development main. Its exact lock matches the E1 branch copy; the E1 frozen contract and run manifests still own experiment semantics and result provenance. The dated local verification record is not proof of a current Colab GPU runtime.

## Knowledge Layers

Canonical development/research checkout is WSL-native `/home/frank/projects/MarketMamba` in the distro named `Ubuntu-24.04` (`3b0df0f` at observation). Windows Scheduler invokes the separate D: checkout `D:\Desktop\work\ProjectForMe\MarketMamba` (`8b82a68`, clean) through the distro named `Ubuntu`, not the development checkout. Both distros report Ubuntu 24.04, but they are distinct registrations. The batch activates `mamba_env` with `/home/frank/miniconda3/etc/profile.d/conda.sh` and uses Python 3.11.15 at `/home/frank/miniconda3/envs/mamba_env/bin/python`. Windows operational `Data/processed_v6` is the D: physical directory; the development checkout symlink resolves there. The read-only interactive query shell in `Ubuntu` had `MARKETMAMBA_DATA_ROOT` unset; the scheduled process environment was not captured; the Windows checkout V6 config falls back to its own `Data/`. Colab remains separate. No path or runtime was changed.

- **Layer 1 — Current State:** this short file and its JSON projection hold active objectives, blockers, next actions, current runtime and branch/worktree context. [Authority Map](Authority_Map.md) determines which evidence wins; [Worktree Register](Worktree_Register.md) records checkout responsibility.
- **Layer 2 — Project Knowledge:** [Domain Notes](../01_Domains/), [GOVERNANCE.md](../GOVERNANCE.md), and stable architecture/decision contracts describe boundaries and durable rules.
- **Layer 3 — History / Evidence:** [History](../02_History/), research contracts/manifests/results, dated runtime observations, reports, Obsidian and Git history preserve completed or superseded work. Machine contracts, hashes, source and executable tests bind technical claims; historical prose does not override them.
- `AGENTS.md` and `CLAUDE.md` are the agent operating contract outside the three memory layers.

## Knowledge Governance State

| Topic | State | Authority / notes |
|---|---|---|
| Domain layer | `COMPLETE_CURRENT_SCOPE` | Twelve reviewed domain notes; machine contracts still bind technical truth. |
| Governance | `ACTIVE` | [`knowledge/GOVERNANCE.md`](../GOVERNANCE.md); impact, supersession, runtime-freshness and history protocols are defined. |
| Unified health | `AVAILABLE` | `tools/project_knowledge/knowledge_health.py`; PASS/WARNING are non-blocking, structural FAIL is blocking. |
| Runtime observation store | `ACTIVE_WITH_STALE_SEMANTICS` | `knowledge/runtime-observations/`; timestamps change only after an actual observation is recorded. |
| Generated navigation | `DETERMINISTIC` | `generate_indexes.py`; declared generated files are not hand-edited. |

## Production State

| Topic | State | Authority / observation |
|---|---|---|
| Active scheduler path | `ACTIVE`, observed 2026-09-22 | `MarketMamba_DataFetch` (Ready/enabled) → `wscript.exe` → `run_hidden.vbs` → D: `v62_daily.bat --fetch-only` → `wsl -d Ubuntu` → `mamba_env` → D: checkout `V6/run_v62_daily.py`. [Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json). Last run 2026-09-21 22:30 +08:00, Task result 0; log exit 0 at 22:39:52. |
| Fetch-only behavior | `ACTIVE` | [`V6/scripts/v62_daily.bat`](../../V6/scripts/v62_daily.bat) and [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) stop before matrix, inference, portfolio and publication. |
| Operational data freshness | `10/10 MATCH_REFERENCE_DATE` | Ten `DAILY_SOURCES` have Parquet metadata max date 2026-09-21, matching `prices_raw` and last fetch-only log. This is date completeness, not data-quality certification; log recorded a stock-universe delta warning. [Operational observation](../runtime-observations/operational-path-20260922T132521+0800.json). |
| Full V6.2 schedule | `DISABLED` | `MarketMamba_V62` is disabled; last run 2026-08-25 22:15, last result 255. Code remains manually runnable. |
| Result publication | `DISABLED_AS_SCHEDULED`; manual state `UNKNOWN` | Full pipeline contains publication logic, but no active scheduled publication was observed. |
| API deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |
| Frontend deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |

Runtime observations are not permanent facts. The new Scheduler observation expires after `2026-09-29T13:21:35+08:00`; the prior 2026-09-20 observation remains preserved. A successful fetch-only run does not prove the next run, deployment, trading, or complete data validity.

## Data / Correctness State

| Topic | State | Authority |
|---|---|---|
| Reusable PIT/provenance source | `IN_MAIN_RESEARCH_ONLY` | Six fail-closed primitives and their focused tests were adopted from `fix/v7-historical-pit-reconstruction@e3bce91`; source and policy bytes match branch blobs, with one test's extra EOF blank line normalized. Blob map: `research/v7/source-adoption-register-20260922.json`. Frozen experiment evidence/readiness remains branch-qualified. |
| Historical-simulation readiness | `PASS` | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` |
| Strict Phase 0 | `STOP` | Same readiness evidence and corrected Phase 0 audit; it is intentionally separate from historical-simulation readiness. |
| Verified executable labels | `5d=0`, `10d=0` | Corrected E5/E1 machine contracts. |
| Corrected feature contract | `48 FEATURES`, ordered and manifest-pinned | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json` |
| Graph | `DISABLED`, zero edges, fusion branch absent | Completed corrected E5 contract. |
| Industry neutralization | `DISABLED` | Corrected baseline and E1 contracts. |
| Historical labels | `RESEARCH_ONLY`, evidence class `HISTORICAL_SIMULATION_PROXY`, policy `P0` | Never call these `VERIFIED_EXECUTABLE` or tradability evidence. |

## Research State

| Topic | State | Authority / notes |
|---|---|---|
| Corrected baseline | `CURRENT_REFERENCE`: `E5-PIT-Clean-v1` | Corrected baseline manifests and completed incumbent contract. |
| Corrected E5 | `COMPLETED_PARTIAL_ENSEMBLE` | Seeds 17 and 29 completed. Architecture authority is `d_model=64`, `d_state=32`; the old 32/8 draft is superseded. |
| Formal seeds | `17`, `29` | Used by diagnostic conclusions. |
| Seed 43 | `INCOMPLETE_EXCLUDED_FROM_FORMAL_CONCLUSIONS` | Pointer metadata referenced `best_epoch=8`, but pointer-named best/latest/previous generations are missing. The surviving orphan checkpoint is epoch 15, step 39,915, `terminal=false`, contract/matrix matched, and is convergence evidence only. It is not a formal seed and must not be promoted or directly resumed. |
| Extended diagnostic | `STOP_FOR_TRADING_CONTINUE_RESEARCH` | Broad 2026 degradation and near-redundant heads; authority is diagnostic v2. |
| E1 rolling-origin refresh | `COMPLETED_RESEARCH_EXPERIMENT`; `PASS_REFRESH_HYPOTHESIS` | Formal seeds 17/29; `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json`. The specification retains its pre-run wording as history. Historical-simulation proxy only; strict Phase 0 `STOP`. |
| E1 post-run diagnostic | `MONTH_SENSITIVITY_FOUND`; `EPOCH_DEGRADATION_INCONCLUSIVE`; `REGIME_DEPENDENCE_INCONCLUSIVE` | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json`. Logged validation declines after epoch 1; final-support replay and causal regime attribution remain unavailable. |

## Production-System State

| Topic | State | Authority / notes |
|---|---|---|
| Architecture audit | `COMPLETE_CURRENT_REFERENCE` | `docs/production-system-architecture-audit@9efed7b:reports/marketmamba_production_system_audit.md` and its production-system roadmap. |
| Phase A implementation | `IN_DEVELOPMENT_MAIN`, `NOT_DEPLOYED` | Fixed `89f9496` integrated locally; original source and candidate branches retained. |
| Phase A task state | Tasks 1–3 `COMPLETED_REVIEWED`; Task 4 limited approval at `1e119ef`; Task 5 limited offline approval at `1c5250e` | Limited local engineering integration completed; production readiness remains outside scope. |
| Current blocker | `NONE_FOR_LOCAL_PHASE_A_INTEGRATION` | Completed local integration; production and research gates remain unchanged. |
| Production success authority | Legacy fetch result remains authoritative | Shadow-ledger failures must not change fetch-only success semantics. |

## Delivery/UI State

- Tracked FastAPI and React legacy delivery surfaces exist; live deployment and deployed revision remain `UNKNOWN` because public endpoint queries were inaccessible and provider dashboards were not accessed.
- V7 status API/UI source is `COMMITTED_NOT_DEPLOYED` in the WSL development main checkout. The public API projects only the summary schema/state/decision/data ID/time/counts; original health artifacts remain local research evidence. This source commit does not change Windows operational checkout or live deployment authority.
- V7 status design is read-only and fail-closed, but live behavior is `UNKNOWN`.

## Active Blockers

1. `STOP`: strict Phase 0 has no verified executable 5d/10d labels.
2. `NOT_DEPLOYED`: Phase A local integration complete, no production promotion.
3. `UNKNOWN`: E1 final-support epoch dynamics and causal regime dependence require the frozen matrix and ex-ante market-state evidence; completed result is admitted.
4. `UNKNOWN`: live API/frontend deployment and publication state.

## Current Decisions / Next Actions

The first Autonomous Charter window is complete. Use the single backlog linked
above; all five selected units are committed/reviewed. No automatic continuation.
Next Human decisions: separate exact-candidate integration, future bounded CPU
window, original E1 matrix location if available, and research admission/design/
compute. Strict gates remain unchanged; no need to repeat completed H0 or E1 search.

## Active worktree boundary

Observed 2026-10-21: main plus16 linked worktrees (17 total after this window's
creation). Only `development/v7-autonomous-charter` changed; main remains6d74d738.
All other checkouts were clean at inventory and preserved. The new tree retains
the closed window for review; no tree removal authorized. See Worktree Register.
No Windows runtime observation was refreshed.

## Repository Rationalization Inventory

Phase 3A read-only inventory completed on 2026-09-22; report: `D:\Desktop\work\project-audit\market-mamba\repository-rationalization\MANIFEST.md`. It classified the three tracked dirty files, 69 collapsed untracked status entries (73 files), eight worktrees, local branches, large artifacts and duplicate candidates. That phase made no source/data/worktree/branch cleanup, Windows checkout sync, push or deployment. Its evidence does not alter research, scheduler or deployment authority.

Phase 3B-1 source canonicalization completed on 2026-09-22 in WSL development main; `D:\Desktop\work\project-audit\market-mamba\phase3b-source-cleanup\MANIFEST.md` is its external audit artifact. The V7 status UI/API is `COMMITTED_NOT_DEPLOYED`, E1-matching environment identity is in Git, early V7 source has a historical SHA register, and local research/task notes are preserved and explicitly dated. `research/v7/current-state.md` defers to this canonical state file. The development main source working tree was clean at completion. Corrected E5/E1/recovery/postrun and Phase A implementation/result authority remain branch-qualified for Phase 3B-2 selective review. No worktree, branch, large artifact or Windows operational checkout was changed, and nothing was pushed or deployed.

## Authority Links

- [Authority Map](Authority_Map.md)
- [Domain knowledge index](../README.md)
- [K0 discovery report](../../reports/marketmamba_k0_project_knowledge_discovery.md)
- [K0 machine authority map](../../research/project-knowledge/k0/authority-map.json)
- [K0 boundary map](../../research/project-knowledge/k0/boundary-map.json)
- Corrected/PIT authorities are branch-qualified in [Authority Map](Authority_Map.md).

## Historical bounded recovery checkpoint — 2026-10-20

The observation-time pending approvals and suggested next steps in this section
are superseded by the current 2026-10-21 decision summary above. They are not
instructions to repeat completed work.

### Snapshot Reader contract continuation (Unit A)

Base `8309ee366aea4c002f9a79d0d97d214b29578fcf`, same isolated branch.
The v1 reader now rejects traversal, absolute/noncanonical raw paths, symlink
files/directories, non-regular files, duplicate keys, malformed/missing/wrong-type
fields, digest/request/publication identity mismatch and incomplete publication.
Directory-relative no-follow reads are anchored at the caller-selected snapshot
root; root ancestors must remain trusted/stable. Integrity is not authenticity,
HTTP success, PIT eligibility or Human correctness approval.

Two repair rounds: initial reader validation (22 failing assertions before fix);
independent review then reproduced nested JSON float overflow (`1e999`), closed
with finite-number parsing. Valid v1 GET/POST, empty payload, naive/offset/UTC
writer timestamps, HTTP error observations and finite floats stay compatible.
Final focused snapshot tests: 17/17; six PIT suites: 47/47 (zero skips);
governance tests: 15/15. Independent reviewer reran 17/17 and approved the
second-round reader changes with no remaining required finding.
No research data was read or modified; all malformed manifests are offline
fixtures. Strict Phase 0 STOP and Task 4 approval pending remain unchanged.
Reader implementation committed as `0a7928061755855faf9aeb55c1702e26a5a62a34`.
The next authorized unit (Task 4 evidence recovery) completed below, on its
separate operations lineage; no operations implementation was imported here.

### Phase A Task 4 evidence recovery (Unit B)

Isolated `fix/phase-a-task4-evidence@46df7b3402f85877a5fc55600d53b15a6f3a7849`
derives directly from original `f5b0ff3df7645098cc01a9eeb36f5df22c4a3e41`.
The original checkout remains unchanged. Evidence index at that commit:
`research/production-system/phase-a/task4-evidence-recovery.json`.
It records specification/report hashes, exact test names, commands and limits.

Recovered a deterministic defect: one artifact referenced by two stages was
counted once but its completed flag and bytes were summed twice. The fix filters
artifact identities by attempt membership before aggregation. Public LedgerStore
fixtures reproduce the baseline failure (`complete_count` 2 instead of 1);
the candidate also verifies equal-sized distinct artifacts, latest empty retry
isolation and unchanged logical ledger contents.

Two bounded rounds: query fix after genuine RED; review-driven test strengthening
from connection-local change count to complete before/after ledger dump.
Final offline ops 70/70 and mocked fetch-only regression 3/3 PASS. Independent
reviewer ran 70+3 in round 1 and reviewed round 2 without rerunning tests; no
Required/Critical finding. These are engineering evidence, not Human acceptance.

At the initial `46df7b3` checkpoint, Task 4 correctness approval remained
**NOT_ESTABLISHED**. The sequential rebuild
test does not prove concurrent committed-only isolation: uncommitted-writer and
mid-read-commit tests are still missing. Sanitizer regression corpus passes but
requires final Human review. Task 5's process crash/restart/replay, >=10 outcome
comparisons, formal artifacts and overhead benchmark remain **NOT_STARTED**.
Do not proceed directly as though Task 4 were approved. Next bounded proposal:
close the two snapshot-isolation evidence gaps, present Task 4 for Human review,
then separately authorize offline Task 5. No research rules/data changed;
strict Phase 0 **STOP** and all research authority remain unchanged.

### Historical Task 4 isolation checkpoint — prior to 2026-10-21 Human approval

`fix/phase-a-task4-evidence@1e119ef855f8fd319f774162beca0bd729d4b921`
adds tests and evidence only, directly after `46df7b3`. The two isolation gaps
above are now closed for the tested SQLite WAL/synthetic-ledger contract:
real spawned writer/reader, uncommitted commit/rollback/SIGKILL cases, and four
deterministic reader boundaries repeated twice. No sleep-based ordering.
Whole projection identity/counts and digest are checked; an explicit no-BEGIN
negative control detects a mixed version. No production correctness bug found
or production code changed. No production repair rounds were needed.

Final Operations 75/75; mocked fetch-only 3/3; separate snapshot branch PIT
47/47 and governance 15/15 PASS. Independent reviewer ran the five isolation
tests (12 scenarios), no Required/Critical finding. Optional cleanup-return
assertion improvement remains deferred; successful child exits/liveness were
directly checked. No cross-branch integration was performed or claimed.

Human review package: Task 4 branch
`research/production-system/phase-a/task4-evidence-recovery.json`,
with 12 raw records and hashes in `isolation-evidence-20261020/`.
Evidence is sufficient to present the bounded Task 4 engineering correctness
decision; **NOT_APPROVED** until Human decides. Tests do not establish research
PIT/label truth, arbitrary-secret absence, power-loss durability or production
readiness. Task 5 remains **NOT_STARTED** and requires a separate authorization:
fixed >=10 observed-vs-shadow replay cases, crash/restart/concurrency, formal
artifacts/report and overhead evidence. Strict Phase 0 remains **STOP**.

- **Completed:** reproduced silent manifest replacement for the same request and
  retrieval timestamp, including parser-only conflicts; identical retries also
  replaced inode/mtime. Replaced shared temporary-name/overwrite publication with
  unique staged files and atomic no-clobber publication. Identical bytes reuse
  the committed file; conflicts, corrupt raw content and manifest symlinks fail
  closed without replacing the original. Valid raw blobs created before a
  rejected manifest remain preserved and do not imply a completed observation.
- **Evidence:** snapshot suite 10/10; all six adopted PIT primitive suites
  40/40, no skips, CPU/local synthetic fixtures, ResourceWarning promoted to
  error. Real competing subprocesses yield one committed manifest; controlled
  SIGKILL before/after manifest publication yields safe retry and valid hashes.
  This is engineering/fault-injection evidence, not market-data or trading
  correctness. Independent review found no required code changes.
  Governance regression 15/15; syntax/JSON and `git diff --check` passed.
  Knowledge-impact review completed; unified knowledge health exits 0 with
  pre-existing Governance/Authority Drift/Legacy Docs WARNING, no structural
  FAIL. Operations/GOVERNANCE rules reviewed and unchanged: no control-plane,
  authority or lifecycle policy change. Full project/model-training tests were
  not run; verification is bounded to the six PIT primitives and governance.
- **Runtime:** Python 3.12.3; related pandas tests use the existing
  `.worktrees/v7-e1-colab-recovery/.venv/bin/python` with GPU hidden. No installs,
  model training or data writes. `MARKETMAMBA_DATA_ROOT` was unset; canonical
  development `Data/processed_v6` resolves to
  `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`; that data was
  not loaded or mounted into the new worktree. Only temporary fixture data used.
- **Limits:** trusted snapshot directories and hard-link/fsync-capable filesystem
  required; unsupported storage fails, with no overwrite fallback. No certification
  of Windows/Drive semantics, adversarial directory mutation or host power-loss
  durability (new-directory parents are not fsynced). Killed-writer temporary
  files stay non-authoritative and are not auto-cleaned.
- **Remaining gates:** strict Phase 0 STOP, verified labels 5d/10d = 0; Phase A
  Task 4 approval not established / Task 5 not started; E1 matrix/replay and
  causal-regime evidence limitations unchanged. No merge, deployment or approval.
- **Resume:** inspect this branch and commit/tests before using the candidate;
  the main source-adoption register remains historical byte-identity evidence.
  Existing memory-hardening branch `91db7d1` remains unmerged; no new hooks or
  memory system were installed. Next bounded candidate: validate source snapshot
  reader path/schema rejection with offline fixtures, without admitting new data.

Reproduce the bounded regression from the candidate worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/frank/projects/MarketMamba/.worktrees/v7-e1-colab-recovery/.venv/bin/python -B -W error::ResourceWarning -m unittest V6.experimental.v7_source_snapshot_test V6.experimental.v7_temporal_contract_test V6.experimental.v7_executable_labels_test V6.experimental.v7_ohlcv_provenance_test V6.experimental.v7_historical_security_master_test V6.experimental.v7_historical_label_materialize_test -v
```

## Knowledge-impact Rule

Update canonical knowledge in the same change for: semantic changes, authority changes, subsystem-boundary changes, active runtime-state changes, correctness-gate changes, new canonical artifacts, deprecations, or major blocker/next-decision changes.

No knowledge update is normally required for formatting, comments, pure refactors, tests-only changes, or internal renames that do not change public semantics or authority.

## Phase 3B-2 selective source integration (2026-09-22)

Development main now contains the reusable corrected PIT primitives, corrected E5 research core, formal E1 research core and frozen contract copies, plus the E1 telemetry correction and reusable Colab log wrapper. The [source adoption register](../../research/v7/source-adoption-register-20260922.json) records source commits, blobs, destinations and test evidence. This is research source authority in main, **not** result, portfolio, operational, Colab run or deployment promotion. The old 32/8 baseline draft remains branch-qualified; the accepted E5/E1 architecture is 64/32. E1 historical-simulation `PASS_REFRESH_HYPOTHESIS` and strict Phase 0 `STOP` remain unchanged. E1 postrun diagnostics remain active; Phase A Task 4 approval and Task 5 acceptance remain open. Eight worktrees were inspected and retained; no branch/worktree/artifact was removed. Windows operational checkout, Scheduler and deployment were not changed; no push.

## Phase 3B-3A governance checkout retirement (2026-09-22)

The four authorized K0/K1/K2 governance checkouts were removed using normal Git worktree removal after clean-state, ignored-file, branch-history and path-dependency checks. Their branch refs and commits remain as `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT`; the four E1/Phase A worktrees remain present. Details and local bytecode disposition are in [Worktree Register](Worktree_Register.md). No research artifact, branch, Windows operational checkout, deployment or scheduler was changed; no push.

## Phase 3C storage audit (2026-09-22)

Read-only reachability/rebuildability audit completed; [external manifest](/mnt/d/Desktop/work/project-audit/market-mamba/storage-audit/MANIFEST.md). No storage item was deleted. D: scheduled fetch-only inputs/outputs, historical baseline caches, dated ZIPs, E1 formal bundle/results, V7 provenance/label snapshots and active recovery `.venv` remain protected. A clean offline frontend install/build demonstrated that local `node_modules` and `dist` are future low-risk generated cleanup candidates (~187 MB), subject to separate cleanup authorization. Large-cache/archive deletion has no approved exact survivor or rebuild path. `DATA_QUALITY_DIAGNOSTIC` remains separate.

## Repository/governance rationalization closure (2026-09-22)

The **repository/governance rationalization program** is complete: canonical governance and authority navigation are established, runtime boundaries were reconciled, development main source is clean, reusable PIT/E5/E1 research source was selectively integrated, historical source was preserved, the four K governance checkouts were retired with branch history retained, and storage received a retention decision without deletion. The E5/E1 refresh, E1 recovery, E1 postrun, and Phase A worktrees remain intentionally present. Development main is ready for remote publication; the finalization audit records the actual Git push and remote-HEAD verification separately.

This closure does **not** complete V7 research, strict Phase 0 (`STOP`), E1 postrun diagnostics, Phase A acceptance, `DATA_QUALITY_DIAGNOSTIC`, Windows operational promotion, backend/frontend deployment verification or promotion, or large-storage cleanup. The Windows operational checkout and Scheduler remain separate and unchanged. Research result authority and branch-local operations authority remain as stated above and in the Authority Map.
