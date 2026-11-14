# Window05 F0 — 48-feature integrity audit

Status: SOURCE_AUDIT / UNMERGED / NOT_DEPLOYED. No feature selection, admission or
historical result reclassification. Strict Phase0 STOP, E1 matrix BLOCKED, H2 not
approved, fair evaluation PROPOSED. Base main2ee4ca83e709af64be4dd63032ba4fb47903d769.

## Machine identity and evidence limits

`research/v7/window05-feature-integrity.json` contains all48 canonical names in
frozen order, groups A15/B20/C1/D12, dependencies, provider/artifact lineage,
observation/publication/availability/retrieval/vintage/missing/scaling fields,
classification and references. Source builder `V6.experimental.v7_window05_audit`
checks exact policy membership and refuses silent extra/missing features.
The official manifest/order hash and contracts are unchanged. `feature_engineer.py`
has the same Git blob edb78115159776502accce7d1e57df36c7239824 in main and historical
e3bce91; this verifies source equivalence, not all runtime inputs/options.

Classification: VERIFIED0, CONDITIONAL21, UNKNOWN27, INVALID0. CONDITIONAL means
source/formula traced with availability/vintage/universe conditions unresolved;
it does not mean observed end-to-end PIT safety. INVALID is reserved for demonstrated
invalid feature payload, which was not available. All48 remain members of the
frozen historical-simulation contract; none gains new strict admission from this
audit. Conditional families: price/derived13, RS3, technical3, capitalization1,
TWII1. The remaining27 lack sufficient publication/vintage evidence. Market_Cap_Log
is explicitly retained by frozen policy; independent row-level verification is missing.

No certified raw/unscaled feature table tied to this lineage was supplied in the
isolated worktree. The E1 matrix is missing and was not searched or regenerated.
The mutable operational `Data/processed_v6` symlink is not a PIT authority and its
payload was not read. Per-feature missingness, year coverage, invalid/zero counts,
outliers and empirical correlations therefore remain UNKNOWN, not zero.
`quality()` supports caller-supplied bounded tables and reports finite/zero/year/
quantile diagnostics, but its synthetic tests are not empirical market statistics.
Historical prepared-cache identity from e3bce91 source-provenance is
043816abffe8eb09ab4ec42204c7d9f0c24e4368d8ef3cb265b3c8179bd0d6bd,
2244 parquet/metadata files and8307285 rows in feature manifest; these are pinned
historical declarations, not new payload verification.

## Prediction timing finding — HIGH priority, impact unquantified

Frozen policy says feature_cutoff=session t close and prediction=t after close.
That is not equivalent to waiting for required after-close releases, quality checks
and then deciding before the next open. Observation date is the period/session the
value describes; publication is the source's release; availability includes receipt
and validation; model decision time is a separately declared cutoff. A date string
proves none of the other three.

`_merge_macro` computes returns and joins by Date without timezone/availability.
A synthetic 2026-07-16 US close feeds a same-date TW row. The NYSE core closing
auction is16:00 Eastern, which on this summer date is2026-07-17 04:00 Asia/Taipei,
after TW t close. The same-date value could be eligible for a later next-morning
snapshot if its actual release/receipt/QC precede an explicitly approved decision
time; it cannot simply satisfy the frozen literal t-close cutoff.
Source: [NYSE official hours](https://www.nyse.com/trade/hours-calendars), inspected
2026-10-23. This example does not assign NYSE closing time to TNX, FX or futures:
each requires its own provider/session/timezone convention.

The existing `_sync_macro_data` uses yfinance daily Close with date-only fields;
`_merge_macro` also renames US_QQQ to SPX if present. Which SPX proxy and date
convention fed the actual E5 cache is UNKNOWN. Business_Signal primary claims
publication-date alignment in a comment; no row-level evidence was verified.
Fallback month-start+two calendar months is a heuristic, not a historical release
ledger. FED_Rate needs effective/release/vintage semantics; policy-rate and effective
funds-rate series are not interchangeable. Fear_Greed is provider-specific, with
historical revisions/timezone unknown. No asserted daily t-close availability for
foreign holdings, margin, institutional flows, weekly TDCC, lending, TAIFEX OI/options.

**Dependency stop:** no empirical performance/factor analysis using these unverified
inputs. Source-level risks may affect completed research, but exact payload incidence
and outcome impact are UNKNOWN. Historical result classifications are preserved.

## Missingness, scaling, comparability and redundancy

- Institutional join miss or absent source ->0: missing publication/provider failure
  becomes numerically indistinguishable from true zero flow. Other families ffill/
  default0; macro defaults include Fear_Greed50, Business_Signal23, FED_Rate4 and
  options ratio1. Final finite masks cannot recover missingness erased upstream.
- Source price features use trailing row windows; calendar-contiguous membership,
  split/adjustment conventions and OHLCV field consistency are prerequisites.
  ATR and price levels depend on units; volume/splits distort OBV. Technical source
  includes KD EWM and OBV trailing60 normalization, not raw cumulative OBV alone.
- Day_Trade_Volume is a ratio clipped[0,1], despite its name. Holdings ratio scales
  by100; verify percent/fraction. Institutional/lending data need shares/lot units.
- Cross-sectional scaling winsorizes1/99 percentiles then per-Date zscores; it
  depends on the full eligible daily universe and population, not just target stock.
  Macro uses prior-date expanding mean/std, min252 dates, clip+-3. Source is causal
  in row order; this does not solve upstream release-time leakage or vintage revision.
  Actual historical runtime option/payload equivalence requires cache metadata.
- Existing Return1/5/20, RS5/20/60, MA20/60 and price levels share inputs; OHLC,
  Foreign_Buy/Sell/Net and macro shared across stocks imply structural redundancy.
  No empirical correlation measured; no feature removed. Short reversal and market
  relative momentum proposals are algebraically redundant with current inputs.

## PROPOSED contract candidate and minimal next engineering

PROPOSED only: separate asof session t from timezone-aware decision_at before entry,
require source published_at/received_at/quality_checked_at <= decision_at, capture
source vintage and upstream maximum availability for every derived feature. Human
must decide whether to preserve literal t-close information or adopt a later
pre-next-open snapshot; the latter is a new research contract, not an E5 correction.
No date, lag, acceptance threshold or source is admitted here.

Minimal future engineering candidates: (1) availability/vintage sidecar with
fail-closed join, (2) retain observed/missing/default flags before fill, (3) verify
macro symbol mapping and adjustment/volume units, (4) bounded raw/unscaled quality
report tied to an approved cache hash and historical universe. This audit's tools
and synthetic counterexamples are preparation; production source is untouched.
