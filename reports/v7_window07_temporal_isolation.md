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

Thirteen initial/final U1 tests demonstrate allowed cases, immature5/10availability,
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

## Pending Unit2

Selection/refit/forward read-scope views and explicit exposure records are the
second authorized unit. They must remain small library guards, not an orchestrator,
trainer or registry. Unit2 completion/review will be recorded here and in the unique
Backlog/evidence before this window closes.
