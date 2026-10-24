# Window22 — Continuous Historical Panel Candidate

Status: COMPLETE_STOP_REVIEWED_GO / UNMERGED / NOT_DEPLOYED.
Authority: Human Window22, two bounded units. Base/main `428a5fa5cb7b771db39a053f88c7046501399f7e`.
No Model Fit, RankIC, portfolio return, GPU, dataset admission or W19 operation.

## W21 closure and integration

R1/R2 independently CLOSED (15 tests / 2.814s / Critical0 Required0), repaired helper SHA
`14aa7a4623c45f9400d08faf0147316c28735fd8e179f922f80143e37ed274f7`.
133 prior local files size/hash verified. Main ff-only `6a34520d8f9c1166172af3503cad531e23f5ecec`
→ `428a5fa5cb7b771db39a053f88c7046501399f7e`; no extra main closure commit.
Original W21 review and raw remain historical evidence, not rewritten to manufacture GO.
[Closure](../research/v7/window22-integration.json).

## Observed continuous coverage

10 anonymous price GET, HTTP200; metadata0 / diagnostic0, no retry/redirect/fallback/token.
Each query 2019-01-01..2023-12-31; unused20 CLOSED, cannot be reused.
9,864 rows / 1,762,931 raw response bytes / 1,216 reference dates, 2019-01-02..2023-12-29.
HINDSIGHT_SELECTED_SAMPLE, not an unbiased historical market universe.

| Security | Rows | Observed range | Principal limitation |
|---|---:|---|---|
| 2330 / 8069 / 2303 / 2317 / 3105 | 1,216 each | 2019-01-02..2023-12-29 | All structural VALID_PRICE; availability/actions/tradability unverified |
| 5483 | 1,216 | same | One NO_PUBLISHED_PRICE |
| 2499 | 300 | 2019-01-02..2020-04-06 | 149 reference dates absent before reported delisting; cause UNKNOWN |
| 6515 | 1,005 | 2019-11-19..2023-12-29 | 288 rows pre-TWSE listing; 114 UNVERIFIED OHLC semantics |
| 8272 | 54 | 2023-10-17..2023-12-29 | All pre-current-TPEx listing; 17 UNVERIFIED OHLC semantics |
| 9929 | 1,209 | 2019-01-02..2023-12-29 | Seven missing reference dates and seven zero-price rows |

Documented units: Trading_Volume is shares (may include block trades); prices are source TaiwanStockPrice raw monetary quotes. This is source-documented semantics via archived W20 FinMind Technical documentation, not independent exchange reconciliation. No adjusted/total-return interpretation adopted.

Total QC: 9,725 VALID_PRICE (structural only), 131 UNVERIFIED, 8 NO_PUBLISHED_PRICE.
No fill, no price adjustment mixing, no common stock admission from resource membership.
Exact per-security dates/counts/hashes in [coverage](../research/v7/window22-panel-summary.json).
First two symbols each returned 1,216 dates; subsequent full-date panels agree. No truncation observed
in this sample; this is not proof all stocks or periods have unrestricted coverage.

## Calendar and lifecycle

Official OGDL1.0 CSV resources 18419/25036/11543 each downloaded once after exact catalogue/license check;
1,097 / 894 / 265 rows. Separate three catalogue-document GETs preserved licensing evidence.
TPEx delisted page web read returned403: counted as fourth resource attempt, no retry/alternate.
[Official evidence and separate ledger](../research/v7/window22-official-evidence.json).

Current official CSV reports TWSE listing starts: 2330 1994-09-05, 2303 1985-07-16,
2317 1991-06-18, 6515 **2021-01-20**, 9929 1999-06-28;
TPEx: 8069 2004-03-30, 3105 2011-12-13, 5483 2001-03-02, 8272 **2024-10-23**.
TWSE delisting2499 **2020-11-10** verified in official CSV. Official press-release search extract
states2499 already suspended by2020-09-30; exact suspension start and missing-April cause remain UNKNOWN.
6515 prior emerging context is supported only at W21's documented example date, not every earlier row.
8272 current TPEx start does not qualify its2023 emerging prices as TPEx observations.
No daily suspension/transfer-complete historical universe was recovered. [Lifecycle matrix](../research/v7/window22-lifecycle.json).

2330/8069 date equality and W21 vendor calendar2019–23 all agree on1,216 dates.
Year counts242/245/244/246/239 corroborate official annual-report/search excerpts.
Count agreement does NOT certify every date, either exchange calendar, original vintage or every closure.
2023-08-03 TWSE closure has official press-release search evidence; full source-document bytes not archived.
2019-08-09 TAIFEX closure is not promoted into TWSE authority. Weekend vs actual session remains
separate; full historical date-level calendar is UNKNOWN. No new official calendar API queried.

## Label feasibility and admission brief

Private mask checks W07 geometry: entry next reference session, exit signal-index+h (h=5/10),
not stock-row offsets; no return formula/value generated. Missing/zero entry or exit, panel tail,
delisting boundary, immature timestamps and future features have explicit refusal reasons.
All actual rows remain ineligible: calendar qualification, daily membership, corporate actions,
feature availability and label maturity/publication/provenance evidence are unresolved.
Exit in the past is not sufficient label maturity evidence. Synthetic tests do not verify historical PIT.
W07 normalize_calendar / instant reused; no replacement trainer/evaluator/registry.

**Recommendation PROPOSED:** request one bounded exploratory CPU study only after minimum blockers below
are resolved or explicit limited proxy assumptions are approved. Use this fixed ten-symbol diagnostic sample
2019–2023, preserve all initial roles and failures; no performance-based reselection.
Candidate evaluation slice2021–2023 with earlier2019–2020 training/history is an OPTION, not frozen folds;
6515 pre2021 and8272 pre2024 must retain their actual market roles, not become TWSE/TPEx observations.
L0/Ridge/LightGBM via W07–W11, fixed recipes/no search, propose <=30min CPU total and <=12 fits;
these are proposed operational caps, not measured cost or new compute authorization.
Small/hindsight sample cannot support market-wide model superiority or independent final-test claims.

Minimum blockers (no paid account is required for this sample):

1. **DATA_EVIDENCE_BLOCKED:** date-level calendar and historical membership/suspension/transfer scope;
   missing/zero/action treatment. Current CSV listing dates are partial boundary evidence only.
2. **POLICY_REQUIRED:** distinct retrospective dataset/feature identity, temporal availability/proxy assumptions,
   label formula/entry/exit/maturity and corporate-action rules. No Conservative Lag adopted; Strict PIT unavailable.
3. **POLICY_REQUIRED:** concrete folds/purge/selection/refit/evaluator minimumN/ties/aggregation/exposure rules
   and explicit real CPU study authorization. Existing interfaces are engineering ready; research is not admitted.

Do not start real fitting yet. Prefer a targeted calendar/lifecycle closure followed by one consolidated
Human data+evaluation decision, rather than further broad downloads. Legal free noncommercial use remains
scoped to W20/W21 evidence; no raw redistribution, permanent-retention terms not independently clarified.

## Storage / restore / recovery

Private root `/home/frank/projects/MarketMamba/.artifacts/window22-historical-panel/`:
`finmind/` claims/raw/manifests/records/CLOSED; `official-resources/`; `official-docs/`;
`derived/continuous-panel.csv`, `derived/label-geometry.csv`, `source/`.
Both derived payloads remain outside Git. Source-code copies preserve capture implementation.
Independent process restored10 FinMind responses,3 official CSVs and2 derived files with hashes/schema/count checks.
Restore root `.artifacts/window22-historical-restore/` is same host/storage: **BACKUP_PENDING**.
Loss of the device may lose raw vintage; hashes and Git alone cannot restore it. Neither worktree cleanup nor GC authorized.

Rebuild offline with W11 CPU Python and `research/v7/window22-build-evidence.py` (no network).
[Restore](../research/v7/window22-independent-restore.json), [tests](../research/v7/window22-targeted-tests.txt),
[sole backlog](../research/v7/research-program.md#window22-continuous-historical-panel).
Strict Phase0 STOP / E1 BLOCKED / H2 NOT_APPROVED / W12 SAFETY_HALTED unchanged.
W19 untouched/protected, no runtime/scheduler observation asserted. STOP after local commit; no W22 merge/push.

## Review and final acceptance

One major independent review GO / Critical0 Required0 Optional0; zero rework rounds.
Reviewer independently ran23tests(3.040s), verified78 local artifacts, all10FinMind records and restore copies,
official boundaries and19,728 mask rows all ineligible. Main postintegration15testsPASS.
Focused author tests23PASS; schema/syntax/diff/hash/knowledge gates completed in this closure.
Existing Governance/AuthorityDrift/LegacyDocs WARNING retained; no structuralFAIL.
[Review](../research/v7/window22-review.json), [final evidence](../research/v7/window22-evidence.json).

Recovery: `cd /home/frank/projects/MarketMamba/.worktrees/v7-window22-historical-panel`;
read Current_State → Authority_Map → sole backlog Window22 → this report and evidence.
Verify branch/base/HEAD/clean plus local artifact inventory before authorized continuation.
No current READY research execution; acquire targeted missing evidence and Human decisions first.
