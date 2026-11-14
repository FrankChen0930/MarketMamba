# Window07 — temporal fold and research isolation foundation

ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED. Human authorizes two bounded CPU
components, not a frozen study configuration. StrictSTOP/E1matrixBLOCKED/H2unapproved/
FairEvaluationPROPOSED/PreNextOpenPROPOSED/InferenceRecoveryDEFERRED persist.
No model fits, genuine RankIC, external data, GPU, Colab, production or candidate merge.

## Unit1: parameterized temporal eligibility

`V6.experimental.v7_research_temporal` consumes a caller-supplied fold, session
calendar, feature event, per-horizon label events and independently retained pins.
Pure results are ELIGIBLE, REJECTED or BLOCKED with reason codes/identities and
research_admitted=false. ELIGIBLE denotes only consistency with explicit engineering
inputs, never source authenticity or historical PIT/admission.

Calendar includes explicit dated open/close instants, source hash/vintage and
qualification reference. Existing normalize_calendar and canonical_hash are reused.
Synthetic qualification is visibly synthetic; QUALIFIED_REFERENCE requires source/
vintage/ref but their authenticity must be validated upstream. Hashes fix contents,
not issuer truth. No weekday inference, stock-row counting, source acquisition or
automatic historical-calendar qualification occurs in library code.

Fold expresses study/fold/feature/label/calendar identities, inclusive training,
validation/refit/forward signal-session ranges, selection/refit cutoffs, model-ready
deadline and origin, required horizons, purge/embargo and unresolved policy state.
All are explicit, no12mo/3mo/3origin/30purge/default horizon. Unknown policies/calendar
or missing clocks block. Train must precede validation; refit may include newly
mature supervised samples beyond validation; forward stays separate. Cutoff <=
model-ready deadline <= origin, and origin must allow the first forward signal.

Caller-selected label_definition NEXT_SESSION_OPEN_HOLD_H_CLOSE is the only
supported engineering convention, matching existing documented next-open/h-session
label timing without computing/modifying target values. Each sample's declared
entry is the next supplied trading session; horizon h exits the h-th holding
session. Record clocks must match calendar open/close, exit<=mature<=available<=
phase cutoff; all required horizons must qualify. Each label is explicitly bound
to feature sample_id/stock_id/signal_session/contract, separately pinned, never
just a value at a matching date. New pinning cannot legitimize a stock mismatch.

Feature observation/publication/source/local receipt/QC clocks are aware/ordered,
QC must PASS, all <=freeze<=decision<entry open and stage cutoff. Unknown values
are not observed0. Each per-stock vintage/source identity and entire normalized
record must match the expected pin. Calendar/fold changes are checked independently.
For derived features, caller must provide the actual dependency-aggregate clock/
identity; no automatic dependency provenance or real source-time certification.

Train supervised label intervals cannot cross validation information start; selection
validation/refit cannot cross forward start. Optional explicit extra session gap
is **purge_sessions + embargo_sessions after label exit before the next role's
opening**, with mandatory non-overlap independent of gap0. This is a precisely
stated engineering candidate interpretation, not a universal embargo definition
or new approved research policy. Other gap/maturity interpretations require review.
Fixture0 gaps are illustrative only. Feature warmup context is distinct from
supervised label overlap; library does not construct context or rebuild a matrix.

Initial12 and final13 U1 tests demonstrate allowed cases, immature5/10availability,
weekend/explicit synthetic holiday, interval overlap, future/missing/QC, changed
vintage/contract pins, unresolved policies, additive gaps, legal newer refit sample,
forward features, invalid deadline/naive clocks and serialization. Independent
review found missing cross-stock label binding; RED counterexample and one bounded
repair added explicit per-horizon owner keys. Final13PASS, independent13PASS,
Critical0/Required0. No existing formal feature/label or experiment file changed.

## Scope of the fixture calendar

The test calendar is an explicit SYNTHETIC schedule, with a declared closure on
2024-01-15 and weekend gaps; it is **not** a statement of Taiwan's actual holidays.
Tests prove that caller-provided session adjacency matters, not real historical
calendar correctness. Real use stays blocked until source-qualified calendar,
availability/vintage and approved research semantics are supplied.

## Unit2: read isolation and fresh identities

`V6.experimental.v7_research_isolation` builds declared selection/refit/forward
manifests and authorizes small data views. It reuses Unit1 guards and canonical
identity; no trainer, executor, global registry, scheduler or storage service.

Selection kind may request only SELECTION_TRAIN or SELECTION_VALIDATION ranges at
its supplied cutoff. Refit has a distinct run_id, pinned parent-selection identity,
same explicitly selected recipe/seed, no resume checkpoint and NEW declarations
for model/optimizer/scheduler/RNG. It can include newer qualified mature samples
within the declared refit range, never forward labels. Forward requires the pinned
refit model identity and chronological declared built/verified/approval/ready times
by model-ready deadline/origin; a Selection checkpoint cannot masquerade as refit.
No metric, epoch count or acceptance threshold is selected by these guards.

These identities/readiness/freshness fields are **declarations for synthetic/mock
flows**, not evidence a real model was reset, trained, artifact-verified or approved
by Human. Actual runtime reset/restore/activation needs separately authorized
acceptance. Manifest/pin consistency is not signature authentication. New run IDs
are distinct within the validated parent/model lineage, not a globally persisted
uniqueness service. Caller retention/uniqueness rules remain a future dependency.

The guarded Forward view never queries/iterates label_store; NoLabelReads fixtures
raise on access and still allow mock scores from features only. Selection/refit
out-of-role samples are denied before loading labels. Each record must match its
request key and expected pins; eligibility/denials include sample/fold/calendar/
run identities and reason codes. A single rejected/blocked sample causes an empty
batch view, with individual verdicts retained; no partial usable values escape.
Returned structures are JSON-serializable and detached copies.

This is a Python library trust boundary: a consumer must use the guarded returned
view. Code retaining raw store references could bypass it; no OS/sandbox/security
or encrypted data-access enforcement is claimed. No broad dataset is read here.

If a former forward result is later used for recipe choice, expose_forward_result
requires another later Fold, observed result availability before that selection
cutoff, expected result/ledger roots and valid source identity. It returns the
usable result together with a new explicit DEVELOPMENT_EXPOSED_NOT_FINAL event/list
and root, leaving the prior list untouched. Same-fold forward cannot select itself.
The event records source/target study/fold/run, decision_ref, actual supplied use
time and prior ledger root; no automatic statistical/Final admission. This is a
small explicit audit function, not a database/registry, and callers must persist
its returned ledger. It cannot detect undocumented off-API research decisions.

Initial9/final11 U2 tests (independent11PASS at final review) cover allowed train/validation, forward-label denial,
NoLabelReads+mock prediction, fresh-refit versus resume, newer maturity, late
readiness/wrong model, changed run/fold/vintage, atomic batch denial, same sample in
another Fold, exposure and unknown/pin roots. Independent review found UNKNOWN
text accepted as source_sha256; RED and bounded repair1 enforce SHA256 shape,
never emitting result/ledger for unknown/invalid sources. Source hashes themselves
still require upstream evidence; shape is not authenticity.

## Expected results and verification

[Case matrix](../research/v7/window07-synthetic-cases.json) records16 executed
synthetic cases with explicit expected/actual outcomes and serializable decisions:
5d/10d immaturity rejected; declared holiday respected; wrong holiday entry rejected;
train/validation label overlap rejected; unknown publication blocked; revised
feature/label pin rejected; wrong-stock labels rejected even with self-consistent
new hash; future feature rejected; newer mature refit allowed; forward labels denied
to selection; forward features usable without any label read; late model readiness
rejected; prior forward sample may become validation in a later Fold when eligible;
later result use records development exposure. Test sources preserve reproduction.

Final67 CPU tests PASS:24new focused(U1=13,U2=11),existing temporal7,E1contract8,
E1matrix5,governance23. No model fit, RankIC experiment, GPU or real historical PIT
claim. Unit1 source commit5d736e9; containing Unit2/closure commit is final checkpoint.
Each unit has one evidence-backed review repair; no other engineering units started.

## Common Evaluator readiness and remaining Human choices

The engineering interfaces are sufficient to start a **separately authorized,
synthetic-only Common Evaluator foundation**: immutable run/fold/calendar/sample
identity, per-horizon owner-bound labels, eligible-support views, denials and an
exposure record. This does not grant Window08, real data or evaluator policy.

Human still must freeze actual calendar/source qualification and per-event clocks,
decision/freeze/maturity/required-head policy, ranges/origins/refit data/length,
purge and embargo meanings/parameters, mandatory/missing/stale/revised handling,
model-ready evidence/approval, tie/minN/NaN/common-support/statistics/metric/selection
budget and acceptance/admission rules. No12/3/3 or30gap is adopted. Actual E1 matrix
remains blocked; no substitute reconstruction/search. Existing E5/E1 results untouched.

STOP after this window. Verify cleanHEAD in `.worktrees/v7-window07-temporal-isolation`,
read Current State/Authority/Charter/unique Backlog and evidence. Pending source and
closure stay on this branch, not main; a new bounded grant is needed for more work.
