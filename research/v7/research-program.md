# V7 Research Program and Current Development Backlog

> Current planning owner: this file's dated section below, linked by canonical
> Current State/Authority Map. Planning status does not grant research, execution,
> merge or production approval. Existing machine contracts win all conflicts.
> Historical per-window tasks are checkpoints, not parallel active queues.

## Current development program — 2026-10-21

### Cross-domain window 02

Status: EXECUTING_BOUNDED_WINDOW. StageA main6d74d738→f7af348 ff-only,
then docs closure `c1179fe`; pre/post254 PASS, independent0 Required/Critical.
[Decision](../../knowledge/02_History/Decisions/20261021-charter-integration-window02.json),
[integration evidence](charter-integration-20261021.json). New isolated branch
`development/v7-cross-domain-window-02` from exactc1179fe. UNMERGED / NOT_DEPLOYED.

| ID / priority | Objective / authority | State / resource & Human dependencies | Acceptance / value / risk / duplicate check |
|---|---|---|---|
| W02-DATA / P1 | TWSE/TPEx/MOPS reliability diagnosis; Data provenance boundary + current Human direction | COMPLETED_VERIFIED diagnostic only; offline provider fixtures, CPU; production fixes HUMAN_DECISION_REQUIRED | Pin actual parser AST/source, contrast valid/mismatched dates/schema/missing-market cases; report exact expected vs observed; no network/production import side effects. Makes source risk actionable; no certification or new pipeline. Existing validation scripts require operational data/network; reuse code semantics without invoking them |
| W02-PREP / P1 | E1 frozen contract/matrix metadata integrity before training; existing frozen experiment+Training identity contract | READY after mutation RED; synthetic metadata only, no array rebuild/GPU | Reject hash/semantic/feature-order/file-set drift even if self-rehashed; valid frozen contract/manifest unchanged. Prevent expensive run against wrong identities; no new experiment. All-branch history inspected: no equivalent fix |
| W02-EVAL / P1 | Frozen two-seed decision completeness; SEEDS/ACCEPTANCE contract | READY after RED; synthetic summaries only | Missing/wrong seeds or nonfinite gate inputs cannot PASS; preserve legal decision vocabulary/thresholds. Prevent vacuous apparent research success; no historical recalculation. Existing evaluator lacks seed-set validation |
| W02-MODEL / P2 | E5/E1 architecture/ablation/evaluation fairness audit; research objective + Human prep grant | READY; pinned code/manifests only | Distinguish trained ablation vs occlusion vs proposal; trace selection/evaluation support and test reuse; explicit PROPOSED choices, source hashes, no inferred research approval; adds code-level trace beyond Window01 decision summary |
| W02-DATA-FIX / P1 | TWSE date binding / TPEx header-index drift | HUMAN_DECISION_REQUIRED for production edits | Proven7-case probe; no historical data correction authorized; pin source and fix original parser only after scoped production approval |
| W02-MOPS-POLICY / P2 | Monthly market completeness and publication provenance | HUMAN_DECISION_REQUIRED | Existing quarterly all-market guard is not automatically monthly policy; do not usecache Date as PIT |
| W02-PORT / P3 | Portfolio/Operations reliability | DEFERRED; existing H0/PhaseA complete | No demonstrated higher-value gap than first3 domains; avoid repeating UX/low-impact bug work |
| DATA-01 | Frozen E1 matrix | BLOCKED | No new location; no repeated search or generated substitute |
| RANK-01/METRIC-01/RES-02/TRAIN-01 | Ranking/economic/new research | HUMAN_DECISION_REQUIRED | H2, strict correctness, frozen new design/compute; preparation does not admit execution |

Limits: four selected units maximum for this window;2 evidence-backed repair
rounds each. No model training, including tiny smoke; no GPU probes, formal data
or weights loaded. NETWORK=0. Existing recovery venv Python3.12.3, CUDA hidden,
MARKETMAMBA_DATA_ROOT unset; candidate Data absent. Main data symlink to Windows
remains untouched. Only temporary synthetic fixtures. Commit evidence and update
this single queue per unit. STOP after these units or unresolved authorization.

Resume: verify branch/base/status, read this section + Charter + window02 evidence
once created; continue only unresolved authorized READY units. Other17 checkouts
retain previous purposes; fixed Charter branch staysf7af348. No main merge of new work.

### Autonomous charter window 01

Status: `WINDOW_COMPLETE / STOP_AFTER_FIRST_WINDOW`; base `6d74d738`; isolated
`development/v7-autonomous-charter`, UNMERGED / NOT_DEPLOYED.
[Charter](../../knowledge/GOVERNANCE.md#autonomous-development-charter--2026-10-21)
and [Human grant](../../knowledge/02_History/Decisions/20261021-autonomous-charter.json)
supersede the old no-next-program restriction below for this window only.
The previous ranked backlog and S1–S15 remain historical objective/dependency
sources; the following table owns current execution state. No duplicate queue.

| ID / priority | Objective / authority | State; dependencies/resources | Acceptance / value / risk / duplication |
|---|---|---|---|
| AUTO-GOV / P0 | Reusable bounded autonomy; current Human grant | COMPLETED_VERIFIED; CPU/stdlib, no data/GPU | Charter, one backlog, map/register, independent review, governance checks; avoids repeated supervision; risk: accidental permission expansion; no existing window owns it |
| AUTO-DATA / P1 | E1 split completeness; frozen E1 split + validate_split_rows | COMPLETED_VERIFIED; synthetic dates only, no GPU/new data | Reject unmatched row/split lengths rather than silently zip-truncate; preserve legal splits; protects preparation integrity; RED confirmed before implementation; git history has no prior fix |
| AUTO-RES / P2 | Prepare next decision from admitted diagnostics S7–S10 | COMPLETED_VERIFIED; git-pinned JSON/code only; no matrix/checkpoints loaded | Evidence-linked options, leakage/fairness/dependency checklist; explicit proposal vs approval; reduce Human research decision effort; no new run/results; existing diagnosis retained, not recomputed |
| AUTO-CKPT / P1 | E1 malformed pointer fallback; Training latest/previous/best contract | COMPLETED_VERIFIED; synthetic local CPU only | Reject malformed entries before body access, use valid previous after restart, preserve best and identity, SIGKILL/orphan regression; avoids losing recoverable progress; no historical loader certification; reviewer reproduced TypeError/KeyError, no prior fix in all-branch history |
| AUTO-SNAP / P1 | Immutable snapshot writer/reader consistency; existing v1 schema | COMPLETED_VERIFIED; offline RequestSpec fixtures only | Reject reader-invalid input before any raw/manifest publication; valid HTTP-error roundtrip and concurrency/restart intact; prevents immutable unusable artifacts; status0/600 reproduced, no prior all-branch fix |
| AUTO-PORT / P2 | Portfolio H1 inspector follow-on | DEFERRED_NO_CONFIRMED_GAP; existing fixtures available | Engine/export/API/UI already completed H0; avoid repeated UX/test work; real signals/metrics need H2; no new policy authorized |
| DATA-01 / P1 blocker | Original frozen matrix | BLOCKED; new exact location required | No new lead; do not repeat previous search, regenerate or deserialize |
| RANK-01 / METRIC-01 / RES-02 | End-to-end ranking/economic research | HUMAN_DECISION_REQUIRED; H2 and strict gates | High product/research value; no inferred policy, admission or Phase1–5 execution |
| OPS-02 / P3 | Provider certification B–E | PROPOSED; per-domain policy/architecture decisions | Existing Phase A complete; broad migration not implicitly approved; reliability fixes to existing contracts may be separately decomposed |
| TRAIN-01 | Next discriminating training | HUMAN_DECISION_REQUIRED; hypothesis and GPU grant | CPU preparation does not approve training |

Runtime: Python 3.12.3 via existing recovery venv, CUDA hidden; system python3 for
governance. MARKETMAMBA_DATA_ROOT unset; main Data/processed_v6 symlink targets
`/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`; not read.
New checkout has no Data directory. Main started clean; no other tree modified.
Resume checkpoint: this first window is CLOSED. Verify `development/v7-autonomous-charter`
HEAD, ancestry from6d74d738 and clean status; read Charter and this table, then
[window evidence](autonomous-window-01-evidence.json). Do not restart completed
units or old E1 search. Another execution window needs a new bounded Human grant;
merge/push/deploy remain separately unauthorized. No READY item remains in this
window's confirmed selection; this is not an exhaustive no-bugs claim.

| Completed unit | Local commit / verification / review |
|---|---|
| AUTO-GOV | `be7998d`; governance23 PASS; independent authority review0 Critical/Required |
| AUTO-DATA | `3f07f80`; focused14, final E1 suite33 PASS; independent14 PASS; one repair round |
| AUTO-RES | `5847afc`;4/4 pinned source identities and claims independently checked; zero repairs; proposal NOT_APPROVED |
| AUTO-CKPT | `2c60415`; checkpoint/run/runtime19 PASS, independent19 PASS; two rounds, Required closed |
| AUTO-SNAP | Containing snapshot/window closure commit; snapshot19, final PIT49 PASS, independent19 PASS; one round;6 baseline-byte-compatible cases |

Final shared verification: E1 33 + PIT49 + governance23 =105 tests PASS, no skips.
Syntax/JSON/source and four pinned artifact hashes checked; diff check PASS;
knowledge health WARNING (governance/authority/legacy), zero structural FAIL.
Local CPU only; existing suite contains a one-step synthetic smoke, no new
research/model training. No formal datasets/weights loaded or network requests.

Priorities after this window: P1 DATA-01 only with a new original-matrix location;
P2 RANK-01/METRIC-01 and next-study decisions via the
[research preparation package](../../reports/v7_autonomous_research_preparation.md);
OPS-02 remains PROPOSED pending domain/architecture scope. No unexecuted candidate
has been mislabeled READY. GPU/new training requires hypothesis+compute approval.
Portfolio fixtures already meet H0; no repeated UX work. The two reviewer-found
reliability gaps were evaluated and completed in this window, not left for rediscovery.

Efficiency: no duplicate implementation, bulk data search or padding units. Initial
broad reads were truncated, and a reviewer test-module typo required a corrected
run; these were avoidable token/tool overhead. Same review agent resumed after
pending initialization; no extra parallel memory/MAS system. Available native
usage snapshot is in evidence and is unaudited, not a hard cost limit.

### Historical H0 checkpoint (superseded only as to next-window authorization)

Status: `PORTFOLIO_SLICE_LOCAL_DEVELOPMENT_INTEGRATED / NEXT_PROGRAM_NOT_AUTHORIZED`.

Latest H0 checkpoint: Human accepted the limited fixture product and approved exact
`4a66928aa4c9bd8a1ddf99a5b42866e9b22efb74`; main fast-forwarded from83a6a2e.
CPU01/02 + PORT01–04 + UX01–04 + final closure/font fix are now in local main.
Pre/post CPU and actual Chrome verification passed; no Required/Critical remains.
[Human approval](../../knowledge/02_History/Decisions/20261021-portfolio-final-h0-approval.json),
[closure](../../reports/v7_portfolio_h0_integration_closure.md),
[evidence](portfolio-h0-integration-closure.json).
No further Portfolio UX work or new Program is authorized. STOP; future scoped CPU
work may use this main baseline. H2/StrictSTOP/E1BLOCKED remain unchanged.
The older planning/window statements below retain their observation-time context;
this checkpoint supersedes their H0-pending/UNMERGED status, not research authority.

Historical roadmap recovery source record:
Source baseline: `4c388ded57062194114ea3e221d00c177eb96892`; integration candidate
`integration/v7-cpu-roadmap`, UNMERGED / NOT_DEPLOYED; main `83a6a2e` unchanged.
Authority for this update: Human “Strategic Roadmap Recovery & Autonomous
Development Planning” request, authorizing this inventory/proposal, not future
large implementation. [Evidence report and source IDs S1–S15](../../reports/v7_strategic_roadmap_recovery.md)
and [candidate test/hash index](cpu-integration-candidate-20261021.json).

### Recovered objective and interpretation

Human-confirmed product intent (S1/S2): trustworthy full-market Taiwan-equity
research, daily relative rankings, weight-based simulated portfolio and traceable
performance. No personal account/capital/broker orders. Current machine baseline
(S5–S8) is 48 features, no explicit graph/industry neutralization, E5 64/32 and
formal seeds 17/29. Neither old three-seed/59-feature plans nor old optimal V6.2
portfolio parameters define current V7. Strict Phase 0 STOP and E1
portfolio_admission=false remain binding. Research portfolio metrics, risk limits,
benchmark/slippage and next experiments need separate decisions.

Single queue rule: edit work items here, not copies in task handoffs or MAS DB.
Current State projects active IDs and checkpoint only. Reports are dated evidence.
`implementation_state` describes existing code/evidence, `execution_state` describes
what can be attempted, `approval_state` names permission. READY requires a precise
contract, available inputs/runtime, independent acceptance and current authorization.
A proposal awaiting blanket scope approval is **not READY**. No GPU dependency does
not imply no Human/data dependency. H1 was explicitly approved by Human on 2026-10-21; only PORT-01–04 fixture engineering may execute. H0/H2 remain absent.

### Active UX refinement — Human approved 2026-10-21

[Human product direction](../../knowledge/02_History/Decisions/20261021-portfolio-ux-human-decision.json)
authorizes a concise default dashboard and advanced research mode. Sole active
queue remains this file; old PORT closure is historical completion, not a blocker
for this new explicitly authorized presentation window. Base `5a2bc6f`; isolated
`development/v7-portfolio-ux`; no GPU/data/research gate required for these fixtures.

| ID | State | Dependencies / acceptance | Value / risk |
|---|---|---|---|
| UX-01 | COMPLETED_VERIFIED | Same validated API object; accessible default/general vs advanced; preserve event on toggle | Simple overview with research detail; no parallel model |
| UX-02 | COMPLETED_VERIFIED | Presentation decimal formatting only; exact raw in advanced; zero vs unknown; immutable tests | Readability without accounting change |
| UX-03 | COMPLETED_VERIFIED | Dated ordered Chinese event timeline, chart selection sync, observable state changes only | Explain actual state, never infer causes |
| UX-04 | COMPLETED_VERIFIED | Real Chrome desktop/mobile all7 events/modes, exact values, errors, keyboard; regressions/review/evidence | Verifiable product review, not H0/H2 |

Completed as one coherent UX-01–03 page/helper commit `764dfbf` plus UX-04 acceptance
commit. Real Chrome desktop/mobile, both modes/all7events, exact values and errors
verified. Full lint remains identical baseline failure; scoped new code PASS.
No UX unit remains READY: stop for Human Product Slice Review, no H0/H2 inferred.
Implementation shares the same page/helper to avoid artificial subsystem splits.
Each logical unit has at most two evidence-supported repair rounds. Completion
records in [UX report](../../reports/v7_portfolio_ux_acceptance.md); no extension
beyond UX-04 without new scope. Original PORT evidence will not be rewritten.

### Final UX closure — Human authorized 2026-10-21

[Decision](../../knowledge/02_History/Decisions/20261021-portfolio-ux-closure-decision.json):
dual-mode direction accepted; only the following finishing scope was authorized.
Same branch `development/v7-portfolio-ux`, fixed base `1553a59`; containing closure
commit owns final identity. CPU/fixture only, no GPU or new Human research decision.

| ID | State | Dependencies / acceptance | Value / risk |
|---|---|---|---|
| UX-FINAL-01 | COMPLETED_VERIFIED | Existing UX1–4; all7 selections including initial7 visible inside timeline; chart/summary/state sync; pageY/focus stable; keyboard | Closes observed selection defect; one bounded scroll-anchoring repair |
| UX-FINAL-02 | COMPLETED_VERIFIED | Existing validated API states; deterministic exact-delta Chinese narratives; no unsupported causes or action types; 9 unit tests | Readable explanations without financial-policy change; no repair needed |

[Final report](../../reports/v7_portfolio_ux_closure.md) and
[evidence index](portfolio-ux-closure-evidence.json) retain baseline RED, repaired
browser result, screenshots, exact identity and independent review. No further UX
work is READY or authorized. Stop for Human closure review. H0 local main integration
requires explicit approval of the exact full candidate ancestry; H2 remains separate.
Mobile advanced visual optimization is expressly outside this program; basic access
is sufficient. Existing API/JSON is the preferred future AI research interface.

### Current font blocker closure — Human authorized 2026-10-21

The prior limited Product Slice UX Closure was Human accepted. Old H0 preflight at
6c894794 stopped on late-font clipping; audit7ecb920 remains immutable evidence, not
candidate ancestry. This explicit correction scope supersedes the older no-READY-UX
checkpoint solely for the one confirmed defect.

| ID | State | Acceptance / dependency |
|---|---|---|
| UX-FONT-VISIBILITY | COMPLETED_VERIFIED | True delayed-font RED→GREEN, desktop/mobile last selection fully visible; all7 selections, page/focus/precision/source unchanged; independent review |
| H0-NEW-CANDIDATE | COMPLETED_VERIFIED | Human-approved exact4a66928 ff-only from83a6a2e; pre/post regressions and Chrome PASS; LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED |

[Decision](../../knowledge/02_History/Decisions/20261021-portfolio-font-blocker-authorization.json),
[report](../../reports/v7_portfolio_font_visibility.md),
[evidence](portfolio-font-visibility-evidence.json).
No Required/Critical remains in tested scope. No further UX work READY/authorized.
This gate is now closed by explicit final4a66928 Human approval and local integration above. Stop; no next Program starts.
Strict Phase0 STOP/E1 BLOCKED/H2 unapproved unchanged. Never merge audit branch.

### Ranked backlog

Sizes are bounded engineering estimates, not delivery promises. Priority is value
and dependencies; completed component testing is not recycled to fill weeks.
All current implementation observations are pinned to `4c388de` plus the report's
branch-qualified evidence, and new integration verification is in the candidate
index. `COMPLETED_VERIFIED` means the named offline scope only.

| ID / priority / objective | Source; implementation / execution / approval | Dependencies; GPU/Colab; Human decision | Acceptance and value | Complexity / main risk |
|---|---|---|---|---|
| INT-01 / P0 / five-fix integration candidate | S3, window01/02; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / current request covers candidate only | 83a6a2e→4c388de chain; GPU no; merge decision separate | Five unchanged source/test hashes; 60 portfolio,47 PIT,81 ops,3 fetch,23 governance; old journal compatibility; independent GO. Consolidates trustworthy core. | S; accidental promotion/inherited evidence omitted |
| PLAN-01 / P0 / recover full objective and dependency map | S1–S15; IMPLEMENTED_VERIFIED planning artifact / COMPLETED_VERIFIED / current request | Canonical map/contracts/history; GPU no; no research decision made | Each formal objective traces source; proposal vs approval explicit; one queue, recoverable checkpoint, independent authority review. Prevents random bug work. | M; stale prose mistaken for authority |
| INT-02 / P1 / local main integration | INT-01/report §5; NOT_STARTED / HUMAN_DECISION_REQUIRED / exact candidate approval absent | Clean fixed main+candidate, reviewed entire ancestry; GPU no; Human merge approval | Exact approved SHA ff-only; post identity/tests; no push/deploy. Makes all five fixes available on development main. | S; main movement or partial cherry-pick |
| PORT-01 / P1 / deterministic offline journal history exporter | S1,S3 + H1 contract below; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | Five-fix source, H1 approved schema; GPU no; no research decision, engineering scope approval | One locked prefix; per-event states exactly reproduce engine; source digest; deterministic bytes; invalid input leaves output unchanged. Turns engine into inspectable workflow. | M, 2–3 focused days; read/hash race, accidental repair |
| PORT-02 / P1 / read-only artifact API | S1/S2 product intent, S12 patterns + H1 below; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | PORT-01 schema; GPU no; H1 | Missing/corrupt/unsupported explicit, public field projection, no arbitrary path/query/write; existing routes unchanged. Exposes verified simulation artifact safely. | S–M, 1–2 days; stale fallback or private path leak |
| PORT-03 / P1 / offline portfolio history view | S1/S2 + H1 view contract; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | PORT-01 frozen schema; mocks in parallel with PORT-02, real integration after both; GPU no; H1 | NAV/cash/positions/pending/cost/event timeline, provenance, explicit fixture label; API agreement, browser desktop/mobile, missing/error/zero. Completes useful product slice. | M, 2–3 days; simulation mistaken for live/research result |
| PORT-04 / P1 / workflow acceptance and restart handoff | S3 + H1 fixture program; IMPLEMENTED_VERIFIED / COMPLETED_VERIFIED / H1 approved | PORT-01/02/03; GPU no; H1 | Fixed scenarios include no-trade, partial/blocked fills, action, retry, corrupt/truncated rejection; fresh process replay→export→API matches; build/browser; independent review/checkpoint. Closes end-to-end engineering evidence. | M, 1–2 days; synthetic success mistaken for market validation |
| DATA-01 / P1 / original E1 matrix recovery | S8/S9/S10; PARTIAL / BLOCKED / bounded new-location inspection may be requested | Exact original a590b280…947f62 matrix+manifest location; GPU no; Human supplies location/access as needed | Per-file size/hash, aggregate identity, contract/split match; never regenerate replacement. Unlocks original replay. Previous unchanged search not repeated. | Unknown I/O; unique evidence unavailable |
| RES-01 / P2 / safe retained checkpoint replay | S7/S8/S9/S10; PARTIAL / BLOCKED / exact replay scope/runtime must be approved | DATA-01, safe deserialization validation, retained checkpoint identities; no new training; CPU feasibility UNKNOWN | Validate embedded contract/matrix/seed, paired support, existing BEST rule; preserve output namespace. Recover final-support evidence without altering historical result. | M–L; unsafe loading, missing epochs2–4, hardware feasibility |
| DATA-02 / P1 research gate / strict executable evidence | S5/S6; PARTIAL / BLOCKED / source admission and any acquisition permission separate | Direct dated tradability/open/PIT evidence; GPU no; Human source/policy questions if unresolved | Existing zero-tolerance/no-roll/unknown rules unchanged; qualified evidence inventory, no simulated→verified relabel. Unlocks strict capability only when evidence qualifies. | L/unknown; historical coverage/license/provenance |
| RANK-01 / P2 / corrected prediction-to-ranking adapter | S1/S2/S7/S8, existing stability contract; PARTIAL utilities / HUMAN_DECISION_REQUIRED / H2 missing | Precise corrected two-seed schema/admission; existing predictions/identities; GPU no for artifact display | Reuse validator/ranks where semantics match; freeze mismatch/maturity/missing/tie/support behavior before code; never use historical three-seed output as corrected evidence. Delivers actual V7 ranking path. | M; hidden contract migration and portfolio admission |
| METRIC-01 / P2 / economic and risk report | S1/S2/S4; PARTIAL (NAV,cost,RankIC utilities) / HUMAN_DECISION_REQUIRED / H2 + strict gate | Benchmark/turnover/drawdown-calendar/annualization/slippage/exposure definitions and admissible data; GPU no | Frozen definitions, hand-calculated cases, maturity/support and matched comparison; research run only after its gates. Needed for credible portfolio conclusions. | M–L; unverifiable defaults/selection bias |
| RES-02 / P2 / signal anatomy, portfolio-response, failure-slicing | S4/S6/S8; NOT_STARTED under admitted current scope / BLOCKED / research gate absent | Strict correctness PASS + explicit admission/frozen comparisons, METRIC-01; generally existing artifacts, no new training required | Exact approved Phase1–5 acceptance, equal support/costs, evidence classes; no renamed bypass. Core research validation. | L; false readiness/overfit grid |
| OPS-02 / P3 / provider certification and resumable publication B–E | S11; PARTIAL primitives / PROPOSED / separate production scope required | One domain policy, provider licensing/authority, stage contracts; GPU not for framework, inference later artifact-dependent | Per-domain identity/fallback/certification; kill/restart/pointer/rollback per approved contracts. Complete durable operation incrementally, not broad rebuild. | L; production authority/large subsystem |
| PAPER-01 / later / daily forward simulation | S1/S11 Phase F; NOT_STARTED / BLOCKED / explicit research+paper approval | Qualified inputs, model/strategy/risk freeze, publication and calendar time; existing model inference may need compute | Save signals before maturity; replayable simulation; no stale-as-new; measured forward window. Not broker trading. | L; code completion mistaken for forward evidence |
| TRAIN-01 / later / next discriminating experiment | S7/S9/S13; NOT_STARTED / HUMAN_DECISION_REQUIRED / H2 + compute approval | Resolve diagnostic gaps and predeclare question; GPU/Colab yes for new training | One frozen hypothesis, splits/seeds/selection/cost/stop rule; durable evidence. Human retains promotion. | L/compute; unsupported new direction |
| MAS-01 / separate maintenance / Core Live certification | S12 + historical MAS handoff; PARTIAL / BLOCKED / outside V7 implementation grant | MAS owner exact-version certification, valid occurrence approvals/routing; GPU no | Actual Builder+Verifier lifecycle and restart evidence, not a Codex takeover. Never make this a prerequisite for independent Codex CPU work. | Separate project; authority/state corruption |

**H1 execution checkpoint — 2026-10-21:** PORT-01 completed and independently
reviewed (7 focused / 67 combined tests); PORT-02 completed (8 API tests, independent live review); PORT-03 completed (real browser + unit/build); PORT-04 COMPLETED_VERIFIED
(actual HTTP/restart and browser missing/corrupt/restore; 47 PIT,81 ops,3 fetch,23 governance).
All four authorized units are complete; stop for Human Product Slice Review.
No remaining READY work is authorized by H1. Further CPU opportunities are
H0 integration or separately bounded fixture extensions/source recovery; these
need their existing gates, not new training by default. See [unit evidence](portfolio-history-evidence.json).
Branch `development/v7-portfolio-history` starts at `bebf19ee937a2237641e09701d0564590ed30fc6`;
UNMERGED / NOT_DEPLOYED. No other backlog permission is implied.

### H1 approved engineering contract

[Explicit Human H1 decision](../../knowledge/02_History/Decisions/20261021-portfolio-history-h1.json)
approves this local fixture-only offline history slice. No real predictions,
market data, strategy selection, new economic metrics or background jobs.
This exposes the existing deterministic engine; it is not S4 Phase1–5 research.
Engineering binding: source additionally carries `fixture_id` and
`fixture_manifest_sha256`; operator declaration `v7-portfolio-fixture-v1` binds
exact journal bytes. This declaration is not an authenticity certificate or
admission. API requires an operator-pinned artifact SHA-256 and never accepts
client paths. Output is confined to checkout `.artifacts` or OS temporary paths.

1. **Input:** explicit local synthetic journal generated from existing S3 test
   fixtures; operator-selected source path, never a browser-supplied path. Fixtures
   always labeled `OFFLINE_SYNTHETIC_ENGINEERING`. Existing historical journals
   remain read-only compatibility inputs in tests, not real research admission.
2. **Output (proposed `v7-portfolio-history-v1`):** `schema`, `evidence_class`,
   `strict_phase0: STOP`, `research_admission: false`, `source` containing full
   source commit, engine version, original byte SHA-256, final record_hash and
   event_count; `spec`; ordered `events` with seq/event_id/kind/occurred_at and
   exact post-event `state`; `final_state`. State copies existing `snapshot()`
   keys/Decimal text, including pending null versus populated and true zeros.
   No new return/risk/ranking fields, raw private filesystem paths or wall-clock
   generated_at in deterministic payload. Actual run time belongs in evidence receipt.
3. **Consistency:** obtain bytes/hash/validated records from one shared-lock
   prefix, then reduce exactly that prefix in order, not a second unlocked read.
   Preserve the existing journal API. No journal migration/repair/writes during
   export. Export every event including genesis; final_state equals replay of that
   prefix. No repeated full-prefix replay that makes export quadratic by design.
4. **Artifact handling:** explicit distinct output path; no alias of source
   (resolve/samefile checks), models or production. Validate complete payload
   before atomic local temp/fsync/replace; failure leaves existing output bytes
   unchanged, identical replay creates identical content. No shared default output.
   This certifies local cooperative filesystem scope, not arbitrary host power loss.
5. **API:** additive local-only GET `/api/v7/portfolio-history`, separate router
   `app/backend/routers/v7_portfolio.py`, configured artifact path via a dedicated
   opt-in setting; no writes/auth changes/external network. Missing/disabled returns
   HTTP 200 `not_ready` with no states; malformed, unsupported or inconsistent
   payload returns 503 `error`; valid returns 200 `ready` plus projected public
   contract. `ready` means artifact readable, never tradable. No stale fallback.
   Validate event count/order/final equality and required types/evidence labels;
   digest is integrity metadata, not authenticity of arbitrary market facts.
   Consumer validation checks schema/internal consistency of a trusted local
   exporter artifact; it does not independently prove the accounting of an
   arbitrary edited timeline. Exact accounting is verified at exporter tests.
6. **View:** separate opt-in local route `/v7/portfolio-history` using existing
   app components. Show event time, NAV/cash/quantity/last-price/cost/pending and
   source identity; fixture and research-not-admitted banner always visible.
   No holdings weights until a separate display formula/schema is approved;
   no implied target allocation from pending when unknown. Avoid new policy
   calculations; empty/unknown displayed explicitly, true zero retained.
7. **Tests:** unittest temp fixtures for deterministic complete replay and failures;
   corrupt/truncated/hash mismatch/duplicates fail closed; valid old writer format
   retained; one real-process append/read boundary test proves snapshot coherence.
   API uses existing httpx ASGI tests; UI uses Node contract tests plus production
   build and real browser desktop/mobile/keyboard checks. No new paid dependencies.
8. **Files:** reuse `V6/experimental/v7_portfolio_{contract,engine,journal}.py`;
   proposed exporter/test siblings `v7_portfolio_history.py`, `..._test.py`;
   additive router/test and frontend page/contract/test in existing app trees.
   Each milestone narrows exact paths before edits. A small journal read helper
   extraction may be needed for lock-consistent export, with regression tests;
   no broad refactor. Prefix public state keys with no invented strategy semantics.

Initial program command sketch (see PORT-04 acceptance report for exact runnable commands):
`python3 -B -m unittest V6.experimental.v7_portfolio_history_test app.backend.test_v7_portfolio`;
`python3 -B -m V6.experimental.v7_portfolio_history --journal <fixture> --output <isolated-artifact>`.
Existing portfolio/PIT/governance commands remain in candidate index. Frontend:
`node --test <new-contract-test.mjs>` and `npm --prefix app/frontend run build`;
use existing pinned runtime, inspect browser with available tools before claiming
UI verified. If tools/runtime unavailable, mark that milestone BLOCKED, never waive.
Python style follows existing type hints, Decimal text, `JournalIntegrityError`
and unittest conventions; don't duplicate accounting formulas in API or JS.

### Original H1 sequencing plan (all four units now complete)

| Window estimate | Milestone | Gate / parallelism / stopping point |
|---|---|---|
| Days 1–3 | PORT-01 exporter/schema and hand-fixture timeline | H1 first. Freeze engineering schema as approved v1 before implementation; exact replay/hash/atomicity tests and independent review. |
| Days 4–5 | PORT-02 artifact API; PORT-03 UI contract/mock concurrently | Only after schema stable; distinct owners/files, no shared writer. API and mock UI can proceed in parallel. No external deployment. |
| Days 6–8 | PORT-03 real API view and PORT-04 end-to-end/restart | Needs both outputs; browser/build/fixture checks, immutable evidence and one acceptance report. |
| Days 9–10 if needed | Evidence-backed bounded rework and closure | Max two fix rounds per unit; not extra features. Finish early when accepted, stop if no authorized READY work. |

H0 (optional): approve exact local main integration candidate; this is independent
of H1 and not necessary if development continues on verified candidate ancestry.
H1 (APPROVED 2026-10-21): Human approved PORT-01–04 and the engineering contract
above as a single fixture-only program. This is engineering scope approval, not
research correctness. Routine choices within it need no repeated micro-approval.
H2 (defer): real prediction admission, economic/risk definitions and next research
question. Do not demand these just to start H1. Supplying an original E1 matrix
location is a separate data action; it does not authorize replay/training by itself.

### Codex + MAS sustainable work protocol

- At session start read Current State → Authority Map → this queue → selected
  spec/manifest/tests/domain. Verify repository/worktree/head/dirty state, runtime,
  data realpath and goal objective. Never resume from a title or old MAS task id.
- Codex owns implementation in the selected product worktree, executable tests,
  independent review, coherent local commits and memory. One writer per path.
- Native Goal maintains the thread's bounded outcome/continue/check/complete state;
  it does not replace repository memory, authorize new scope or certify research.
  Verify get_goal each session; if unrelated, stop that continuation using normal
  supported lifecycle controls, never internal DB/global edits. Fresh thread restores
  queue/checkpoint first and creates a matching Goal only under user authorization.
- MAS Advisory can propose ordering and challenge dependency/contract gaps;
  Verifier reviews evidence; Archivist checks identities/records. These are roles
  under existing research-program boundaries, not proof an available MAS service
  has been executed. Core Live remains unproven; use ordinary independent Codex
  review now. Historical occurrence approvals never transfer to new tasks.
- Select highest-value authorized READY item; record IN_PROGRESS, source commit,
  exact allowlist, acceptance, input identities and budget/stop envelope. If one
  item blocks, preserve reason and select independent READY work; no polling for
  Colab or repeated matrix search without new evidence.
- Implement/test/review ≤2 evidence-supported fix rounds per unit. On unresolved
  findings mark BLOCKED with exact reproduction; don't relax gate. Defects are
  handled when they block selected functional acceptance or have high-impact
  evidence, not because unlimited easy tests can be written.
- Checkpoint includes commit and source/input/output hashes, commands/counts,
  reviewer findings, rounds, known limits, elapsed observation, telemetry class,
  Human interventions and next ID. Goal token/time counters are tool observations,
  not independently verified cross-agent aggregate or a hard cap unless enforced.
- Update this queue and canonical projections in the same commit; normal tests →
  impact checker → knowledge update/explicit no-change reason → health → clean.
  Completed units are not rerun except new changes, runtime drift or integration
  need. At no READY work, stop with completed/blocked/decision summary.
- Once approved, H1 can preauthorize CPU fixture coding/tests, local commits,
  independent review and bounded repairs. It cannot preauthorize label/PIT changes,
  hypothesis selection, result promotion, GPU/paid services, production changes,
  main merge/push/deploy or MAS Core authority. Those remain explicit Human gates.

Official Goal design reference, retrieved 2026-10-21:
[OpenAI — Using Goals in Codex](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex).
It describes thread-scoped persistent objectives and explicit stopping conditions;
our live API confirms this thread's new V7 objective, not weeks-long unattended
reliability or cross-thread/global project memory. No scheduler is enabled here.

### H1 Goal (now authorized; stop after PORT-04)

> Complete PORT-01–04 in research/v7/research-program.md: a deterministic,
> fixture-only V7 journal history exporter, additive read-only API and local
> portfolio history view, verified by exact replay/identity/atomicity tests,
> old-format compatibility, browser/build checks and independent review. Preserve
> Strict Phase 0 STOP, no real predictions/market data/research admission, no GPU,
> production changes, merge/push/deploy or MAS Core execution. Select only the next
> dependency-ready approved item; at most two evidence-backed repair rounds each;
> checkpoint/update the single backlog and local commit after every milestone.
> Stop once all four are verified or no authorized READY item remains.

## Historical research-program contract (preserved)

The following is the original umbrella research/role record, retained for
provenance. Its “current environment”, “tomorrow” and unexecuted experiment language
are historical observations, not present runtime facts or permission to activate
MAS/train. The current dated planning section above owns work ordering only;
frozen experiment contracts still own research semantics.

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

### AUTO-DATA completion checkpoint — 2026-10-21

Authority: frozen E1 split and existing per-row validation, Charter engineering.
Before implementation: READY after RED confirmed silent omitted tails. One repair
round (`zip(strict=True)`), no new split/label/PIT rule. RED: 6 tests, four failing
subcases; GREEN focused14, E1 regression29 (includes existing one-step synthetic
CPU smoke, not new research training). Independent reviewer reran14 PASS; zero
Critical/Required. [Evidence](autonomous-window-01-evidence.json) retains raw output.
Source/test identity: containing AUTO-DATA commit. Data note and both maps updated;
frozen contracts/historical artifacts untouched. Remaining: original matrix BLOCKED.

### AUTO-RES completion checkpoint — 2026-10-21

[Decision preparation](../../reports/v7_autonomous_research_preparation.md) completed:
4 pinned sources/3 JSON documents checked against recorded SHA256 and values;
independent review confirms4/4 identity and no admission/approval inflation.
Zero repair rounds. Proposal stays NOT_APPROVED. This is preparation, not new
research analysis execution. Human still owns next hypothesis/design/compute.
Identity: containing research-preparation commit; no external data/GPU needed.

### AUTO-CKPT completion checkpoint — 2026-10-21

Two repair rounds: entry rejection/fallback, then independent-review Required fix
for valid previous retention through save/orphan recovery. Final19 CPU tests PASS,
independent15+4 PASS; Critical/Required0. Real spawned SIGKILL before pointer,
restart/orphan, repeated latest loss, seed/contract/matrix mismatch covered.
No historical checkpoint loaded. Existing unsafe-for-untrusted-pickle loader is
not certified. [Evidence](autonomous-window-01-evidence.json); identity: containing
checkpoint commit. Training note/maps updated. AUTO-SNAP subsequently completed below.

### AUTO-SNAP completion checkpoint — 2026-10-21

Writer validates serialized payload with existing reader rules before raw or
manifest publication. RED19 failures7/errors6; GREEN19, full PIT49; independent19
PASS, no Critical/Required. One repair round; valid six-case payload bytes match
base6d74d738. Frozen data/PIT authority unchanged. All five units complete; STOP.
Identity: containing snapshot/window closure commit, preserved in branch history.

### W02-DATA checkpoint

Completed7-case synthetic provider diagnosis, independent7-case rerun/source/probe
hash PASS,0 Required/Critical,0 repair rounds. Existing fetcher unchanged. Source,
numerical observations and limitations in [window evidence](cross-domain-window02-evidence.json)
and [report](../../reports/v7_cross_domain_window02.md). Identity: containing data
unit commit. Production date/schema corrections and monthly completeness remain
Human decisions; no network/official-data inference. Continue W02-PREP/EVAL/MODEL.
