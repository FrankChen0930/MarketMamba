# Window05 F1 — low-cost market-derived candidates

PROPOSED / EXPLORATORY / SYNTHETIC_ONLY / UNMERGED / NOT_DEPLOYED. No empirical
RankIC, returns, feature selection or training result was computed. True-data
validation is BLOCKED by F0 availability/vintage uncertainty and lack of a supplied
identity-matched raw panel + admitted proxy-label support. E5/E1 exposed periods
are research-used, never untouched test data. StrictSTOP/H2unapproved unchanged.

The formulas below are research candidates, not changes to the frozen48 inputs.
Economic intuitions are hypotheses about behavior, not established predictive claims.

| Priority | Candidate / intuition | Formula at session t | Additional information / concern |
|---|---|---|---|
| 1 | Abnormal volume: relative trading participation may contextualize a move | V_t / mean(V_{t-20}..V_{t-1}) - 1 | Proposed first diagnostic; normalizes own trading scale unlike per-session standardized Volume. Still derivable from raw history; volatility/liquidity/event confounding and split/lot changes |
| 2 | Foreign flow intensity: net foreign participation relative to turnover | Foreign_Net_t / V_t | Proposed conditional diagnostic only after exact flow units/publication evidence. Combines existing columns nonlinearly; no proof of independent alpha or need for another model input |
| defer | Market-relative momentum: stock movement relative to broad market | (C_t/C_{t-20}-1) - (B_t/B_{t-20}-1) | Algebraically existing RS_20d when raw histories/conventions match; do not add duplicate without a distinct Human-frozen question |
| defer | Short reversal: prior short move might reverse | -(C_t/C_{t-1}-1) | Exact negative existing Return_1d on matching data; no new information as a scalar transformation. Interaction research would need its own future proposal |

## Calculation contract of the research helper

`V6.experimental.v7_window05_signals.candidates` accepts caller-supplied data, a
unique increasing Asia/Taipei session calendar, stock/session and timezone-aware
decision_at (normalized to Asia/Taipei for session-date checks).
There is no data discovery, fetch, label IO or pipeline registration. It returns
four diagnostics for a single snapshot; research namespace only.

Each accessed value requires its source availability_at <= decision_at, finite
numeric value and necessary positivity. Price/volume and benchmark timestamps
are separate from flow release. Unknown/late timestamps yield NaN. Numeric dates
are not timestamps. Shares/lots are acceptable only when flow and volume units
match. A zero current volume or zero trailing denominator yields NaN; missing
observations are not interpreted as zero trading or carried into missing sessions.
The mean requires all20 exact preceding calendar sessions and excludes V_t. Price
momentum uses exact t-20 endpoint; reversal exact t-1; benchmark must be positive.
Duplicate stock/session, out-of-calendar rows and naive decision cutoffs are errors.
No centered window, shift(-1), future mean, or future target is used.

The caller must supply the historical vintage visible at decision_at; availability
checks cannot certify that timestamp/value or reconstruct prior revisions. The helper
does not infer universe membership/tradability. Calendars are explicit and no current
stock list is backfilled. A future panel study must preserve historical membership,
common support and comparable provenance. It must not select today's survivors.
Price and volume adjustment conventions must agree: share-unit conversion invariance
is tested, but a corporate split changes actual economic shares/history and needs
separate vintage/adjustment evidence. No generic split safety is asserted.

Calculation time is after every required observation's availability and quality
validation, using a Human-approved decision cutoff; no fixed deployment time is
chosen here. Same-date late flow cannot enter a literal t-close snapshot. The
synthetic20:00 example is a test cutoff only, not an adopted research policy.

## CPU validation and costs

Nine tests cover hand-computed AV=1, relative momentum=.2, flow=.25 and one-day
reversal, future-row mutation invariance, exact missing-session behavior, late/
unknown release exclusion, zero denominators, mismatched units, duplicate/naive
cutoff rejection, share-unit rescaling, invalid prices and stock isolation; equivalent cutoff instants in different timezones
produce the same market-session treatment.
Synthetic numeric outputs are formula checks, not simulated financial performance.
Time complexity of the simple research implementation is O(N) panel filtering plus
21 lookups/snapshot; memory O(N) for the caller frame. A repeated whole-panel run
would multiply that cost; no production/vectorized scaling claim or benchmark was
made. This small helper is sufficient for fixtures, not large reconstruction.

Minimum future shortlist: Abnormal Volume; Foreign Flow Intensity conditional on
release/units. Both reuse sources, no paid feed, negligible incremental arithmetic
relative to training, but provider evidence and raw-panel IO remain actual costs.
Future GPU value is unproven: first establish qualified coverage, stability and
matched-support exploratory diagnostics, then obtain feature/evaluation approval
before any model ablation. Do not select based on a single high RankIC.

Turnover Rate is deferred: needs PIT shares outstanding/free-float definition and
vintages, not Market_Cap_Log as an unverified substitute. Industry-relative Strength
is deferred: dated industry membership and approved grouping policy needed; frozen
baseline explicitly prohibits current-industry backfill. Neither is computed here.
