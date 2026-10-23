# Window05 F2 — Taiwan monthly-revenue PIT feasibility

Decision: **NOT_CURRENTLY_FEASIBLE** for restoring broad historical Revenue_MoM/
Revenue_YoY under certified issuer-publication/vintage semantics on the evidence
available in this window. This is a data-feasibility judgment, not a new research
admission policy. `EXCLUDED_PUBLIC_RECONSTRUCTION_EXHAUSTED` remains unchanged.
A prospective immutable-vintage design is PROPOSED, not a deployed collector or
permission to change pipeline/schedule. No revenue feature restored to formal48.

## What 0/455040 actually means

Machine evidence at `e3bce91be3e68cd1142ba2bfb21eeae43db6dec0`:
`research/v7/historical-pit-reconstruction-v1/revenue-publications/coverage.json`
revision v7-revenue-publication-coverage-v2 records2334issuers,2002–2026,
455040archive rows,0verified issuer-publication rows,455040unknown.
This is a bounded historical archive audit, not a universal claim that no issuer
anywhere has a discoverable announcement. A single timestamped press release
would not establish coverage or revision identity for the whole historical panel.

Exhaustion-evidence pins frozen revenue parquet SHA256
180732c5a99f5f793b96084891d198b959170db9f0f74bc3e56bd39ec4bdebbe.
Of455040 frozen rows,5162 had parseable create_time and449878 did not; even parsed
times lacked certified issuer-publication semantics, hence verified remains0.
The later mutable provider archive had455160rows (+120), all create_time nonnull,
SHA256 d2db379cd3f78c48f81465e8be5723b6704fdafbe0a6b2c9ec7d38339619e9dc.
Those different bytes/counts must not be silently substituted. These are preserved
historical audit identities; this window reverified the Git metadata blobs, not
the original parquet bytes or current provider archive. No current coverage recensus.

Historical five-surface bounded search inspected: MOPS issuer-month JSON, legacy
MOPS HTML, document category archive, TWSE/TPEx aggregate/announcement schemas,
and repo/provider archives. The evidence records query/response time vs missing
filing times, output-date monthly tables vs issuer events, and daily/current
announcements vs complete original/correction history. Sources were representative
schemas/category indexes plus the recorded local census, not a full exhaustive
proof about every public web page. No identical issuer/month or catalog searches
were repeated now. The old PUBLIC_RECONSTRUCTION_EXHAUSTED status is retained
within its actual bounded scope, not expanded into universal impossibility.

## Bounded official-document update, accessed2026-10-23

1. [TWSE filing rules, Article3](https://twse-regulation.twse.com.tw/ENG/TW/law/DOC01.aspx?FLCODE=FL007250&FLNO=3)
   distinguishes reporting deadlines with exceptions. A deadline is not the actual
   issuer release, receipt time, first-observed vintage or correction sequence.
   It cannot populate missing historical timestamps. This current amended rule also
   cannot be projected over2002–2026 without historical rule versions.
2. [Government dataset18420](https://data.gov.tw/dataset/18420) describes monthly
   aggregate fields including output date/period/company/revenue/comparisons/notes,
   free access and Open Government Data License v1. It does not list an issuer
   publication or revision sequence field. Catalog metadata lists a CSV resource
   at mopsfin.twse.com.tw; the CSV payload was not fetched. Catalog update frequency
   and metadata-modified time are not issuer publication or delivery-SLA evidence.
   The catalog licensing statement applies to this identified resource, not every
   MOPS website/archive; any future snapshots need retained attribution/terms and
   actual resource-level licensing review. No license blanket or paid subscription.
3. A specific newer official [TPEx notice11500542341](https://www.tpex.org.tw/storage/eb_data/11503/11500542341.html)
   was found by targeted document search. Its official indexed excerpt describes
   2026 special handling for insurers/insurance subsidiaries, including a15th-day
   reporting allowance. Direct open returned403; this is LIMITED_INDEX_ONLY
   evidence, not a fully acquired rule or issuer record. It reinforces the need
   to verify historical exceptions before any proposed fixed lag. No retry/bulk
   search was performed and no lag was adopted.

These document checks add metadata/license/exception cautions, not newly verified
historical issuer timestamps. No representative record can be called PIT verified
without original value/version and matching filing/publication evidence.

## Original filings, corrections, coverage and storage

Current historical values can contain later revisions. Query time, output date,
provider create_time and first_observed_at are distinct. An immutable forward
snapshot establishes only what the collector observed by that time; it does not
retroactively establish the original announcement instant or unseen revisions.
Sample frequency controls missed intermediate versions. No complete revision chain
or issuer-level verified historical coverage was obtained in Window05.

PROPOSED minimum forward sidecar: issuer/security identity and dated eligibility,
revenue period/currency/unit/consolidated scope, original raw response, exact URL
and request, retrieved_at+timezone, source publication if actually provided,
first-observed/valid-from/valid-to, content hash, revision relation, parser/config
identity and QA status. Keep each distinct version immutable; never overwrite a
correction. Dedupe by content identity and retain retrieval observations separately.
Availability uses the later required verified receipt/QA event, never a guessed
publication time. Capture at future Human-approved cadence with explicit coverage
and failure counters. No collector was implemented or started here.

## Research value: distinct candidate questions, all PROPOSED

- Revenue YoY: R_m/R_{m-12}-1; seasonality comparison hypothesis, dependent on both
  as-of vintages and unchanged accounting/unit scope. Missing/zero prior denominator
  is unknown, not zero growth. Available only after all used observations are known.
- Growth acceleration: YoY_m - YoY_{m-1}; requires more periods/revisions and may
  amplify base effects. A difference of growth rates is not a forecast error.
- Revenue surprise: actual revenue minus a forecast frozen before announcement,
  divided by an explicitly approved scale. No historical expectation series,
  timestamp/coverage or approved estimator exists here, so genuine surprise is
  **not computable**. YoY, acceleration and an ex-post fitted expectation must not
  be relabelled surprise.
- Announcement age: elapsed sessions since the timestamp-qualified value became
  available; no certified date means no age. A deadline-based age is an approximation,
  not actual announcement age. Release timing may correlate with issuer coverage.

Priority recommendation is evidence first: a small, separately approved original/
revision/forward-vintage proof before selecting any fundamental feature or funding
model tests. Public aggregate values are low cash cost but presently fail required
historical knowledge-time evidence. Licensed vintage acquisition or a conservative
lag would each require separate Human policy/expense approval and validation;
neither is chosen. New forward collection is a possible future design, not current
recoverability of2002–2026. Research correctness/admission remains Human-owned.
