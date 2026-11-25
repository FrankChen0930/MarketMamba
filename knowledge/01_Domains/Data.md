## Step3 source admission 工程 — 2026-11-26

[Step3 報告](../../reports/step3_engineering_20261126.md)記錄原始 TEJ HTML 文件條文與 raw/PIT SHA 檢查；原文及逐列稽核只存指派 worktree `.artifacts/step3/`。文件可核尚不能建立逐期首次版／幣別／合併可比範圍，准入 STOP。年度選樣對回5,334,161鍵，缺價256,550；停牌、下市、公司行動與官方日曆未知，不以價格聯集升格。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，沒有改正式 Data、P2／P4d 或 production authority；下一步是獨立審查及補 source evidence。

## TEJ 第二步 2b 描述性特徵候選 — 2026-11-25

最終診斷430.75秒exit0；月營收等權h20早期0.0263／後期0.0152，季報0.0347／0.0263；四條延遲曲線未觸描述性驟降警旗，不證明PIT。h20有效期實際到2026-08-21。描述性建議值得規劃固定月營收h20第三步預先登記，未啟動實驗。最新focused26 passed／兩項新mutation killed；較早collection全套1347 passed／11 skipped／5 warnings／898.37秒exit0，修正後未再完整全套，限制見報告。五份正式parquet SHA均相符，私有manifest SHA `ec10a1d6781db6fa4e3f165d8e2301d86d1dce963598f186522fe74bffd65b13`；partial-before-coverage-fix不是完成證據。

[工具](../../tools/tej_diag.py)／[報告](../../reports/tej_diag_20261125.md)在核准範圍使用發布日期的保守代理：D 後第一個 observed session 收盤可用，次日 open 至 h 日 close 的 raw-price labels；缺值／缺 session 不補。月營收固定五欄、季報三欄與完整欄位等權分數只作已曝光描述；逐列在本指派 worktree `.artifacts/tej/diag/`，public 只有彙總。月營收 scope 欄全缺、累計比較值版本未核；IFRS 排除 >365 日、非曆年／未知起迄、H、現況金融／分類未知，整表與 EPS 版本未確認。沒有改正式資料或 admission。

PIT log 市值無合格來源，因此完整四控制 IC STOP；三價量控制不可冒稱規模已控制。raw 價未含股利、observed calendar 非官方、C0+ 日成員不是 lifecycle／tradability 證據，早期 C0+ 分數缺失不補。輸入／輸出 SHA 只證身分，不證 source PIT；DESCRIPTIVE_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，獨立 review 與 Human acceptance 另案。

## TEJ 匯出資料層候選 — 2026-11-25

[loader](../../tools/tej_pit.py)與[報告](../../reports/tej_pit_audit_20261125.md)是未提交 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED 候選。UTF-16 TAB／strip／`.`缺值／字串前置零／千元→元（EPS不縮放）均以合成測試驗證；SHA不符STOP，缺發布日保留NO_RELEASE_DATE，無期限倒填。月營收380,514列、IFRS115,876列和所有逐列示例只在指派worktree `.artifacts/tej/`，不入公開git；manifest／input hashes見彙總JSON。

IFRS名目年月與實際財報迄日不同導致14列負延遲（min−82），用實際迄日後0列；>365日3,298列的補發／修訂原因與IFRS值版本UNKNOWN，整表strict PIT STOP。C0+研究成員date／stock聯集不等於正式歷史universe；現況產業不回填歷史身分。合成focused14 passed（0.94秒）、兩項in-memory mutation killed；P4a source-compat位置回歸1 passed（11.35秒）。隔離後全套因本輪30分鐘時間界線中止（exit130），collection1351項但無終局總計，NOT_COMPLETED／不宣稱全套通過；須續段完整重跑報告所列隔離命令；先前直接全套測試可能讀真實fold的邊界偏差、最終程序隔離與額外availability skips詳報告，不能宣稱全程未讀標籤。唯讀21筆EPS REVISED：16僅符新值／4新舊皆符／1皆不符；帳本SHA前後不變，不是公司重編原件證據／未動正式Data／排程／P4d；不變publication/source admission authority，仍待原件／版本與獨立驗收。

## PIT 帳本複審修補 — 2026-11-25

`job-run-mww7h6b0-bba9f3/builder_pit_fix2@04e724d` 未提交候選，RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。[工具](../../tools/pit_ledger.py)拒絕完整 sqlite_master 集合之外物件（含 sqliteXdrop），舊 v2 在同 SHA 捷徑前完整稽核索引鍵。百萬鍵合成外推超過30秒，採成功續寫時同交易升 schema v3 並封存索引 SHA 稽核標記，後續逐次驗 hash、不重複解碼；v1／缺標記 v3／無效舊鍵 STOP。[報告](../../reports/pit_arrival_audit_20261125.md#複審-findings-修補--2026-11-25)保留合成性能、focused56及最終全套1333 passed／4 skipped／5 warnings／0 failed（exit0）實際結果。symlink 檢查與寫入競態未修，依 Human 單機無惡意者威脅模型接受為限制；無實際帳本或正式 Data 寫入，publication／strict PIT／法律適用仍 UNKNOWN，排程 NO_GO 與獨立驗收待辦不變。

## PIT 帳本審查修補 — 2026-11-25

未提交候選 `job-run-mww6g0bv-340962/builder_pit_fix1@47bf465`，RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。[報告](../../reports/pit_arrival_audit_20261125.md#獨立審查-required-修補--2026-11-25)與[測試](../../tests/test_pit_ledger.py)補標準化缺鍵STOP、schema／四trigger定義hash、namespace／symlink限制、暫存歸屬及真實檔案SIGKILL／RLIMIT_FSIZE恢復。focused43 passed、兩項新mutation killed；只合成工程證據，未讀寫實際帳本／正式Data，publication／strict PIT／法律適用仍UNKNOWN。SQLite MEMORY暫存的全量RAM峰值、Windows接線／斷電與獨立複審仍待，可註冊排程NO_GO；不變更來源准入或production authority。

## PIT 帳本 v2 候選邊界 — 2026-11-25

[Ledger工具](../../tools/pit_ledger.py)與[報告](../../reports/pit_arrival_audit_20261125.md)為RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED未提交候選。四份processed_v6僅唯讀；起始原parquet不可變複本＋每鍵完整SHA索引，不逐列存JSON；後續完整diff前後值／SQLite交易／run chain／base與index SHA損毀STOP，來源SHA不變記NO_CHANGE。真實初始13,760,451列／556,400,149 bytes／614.602秒；NO_CHANGE1.748秒、增11,913 bytes、events仍0，工程focused15 passed／2 mutations killed；指定全套1292 passed／4 skipped／0 failed、exit0（詳報告），不升格PIT。新帳本在主checkout `.artifacts/pit-ledger-v2/`（EXTERNAL_LARGE_DATA），舊worktree 9.8GB v1未動，不移植觀測時刻。

create_time可解析5942/459119，期限代理數字保留在報告；中位+11／大量+42日顯示很可能為供應商建檔／回補，不能作歷史遲交上限。TEJ營收／盈餘發布日待Human確認授權，法律遲報率null；不改publication／PIT／source admission，不讀labels。每日一次漏日內版本及四表共同刷新未知仍在，排程只提案。詳[Current State](../00_Project_Map/Current_State.md)。

# Data

## V6 fetch hardening candidate — 2026-11-18

Branch `fix/macro-sync-fetch-hardening`, UNMERGED / NOT_DEPLOYED:
[fetcher](../../V6/marketmamba/data/fetcher.py) rejects missing, invalid,
wrong-session and conflicting date/title metadata for BWIBBU_ALL, TWT96U and
TWSE/TPEx daily institutional responses, including ROC title dates.
Daily margin/daytrade/generic/market-value catch-up reuses the existing
`_trading_calendar()` from `prices_raw` distinct sessions and the pure
`missing_trading_days` set difference. Following `721eb84`, scan starts at the
later of the first stored date and today minus 30 calendar days (positive
`lookback_days` override), ends at today, and selects at most the latest 10 missing
sessions by default. A missing weekday today is a candidate even if prices failed;
None/empty responses cannot write rows. Existing sessions are not rewritten.
Market value skips candidate dates older than 10 calendar days and warns to use
`V6/scripts/backfill_stale_202607.py` with its current-share/non-PIT limitation.
Missing price rows cannot produce market value; returned dates reflect actual writes.
Missing individual stocks, previously mislabeled values and historical
price/institutional catch-up remain outside this change.
[Mock-network tests and mutation runner](../../V6/scripts/test_fetcher_date_gaps.py)
use temporary parquet and isolated config only: 30 tests pass (23 existing + 7
new, two existing expectations adapted); 13/13 assertion-killed mutants, 0 runtime
errors. New tests against `721eb84` yielded 17 assertion failures, 0 runtime errors.
This is offline engineering evidence, not formal-data repair or source/PIT acceptance.
The 10-calendar-day market-value age guard bounds exposure but does not remove
current-share capital-change bias or establish PIT. No formal data, scheduler,
.bat or model change. UNMERGED / NOT_DEPLOYED.

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE_WITH_UNKNOWN`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: source provenance, raw/canonical boundaries, PIT eligibility
- Update triggers: provider change, snapshot revision, lineage policy change, new canonical dataset

## Purpose

Define which market data may support research, simulation, and production claims. This note is navigation; machine contracts and source snapshots remain authoritative.

## Current Boundary

`Data/processed_v6` is a mutable, availability-oriented provider dataset. It is not a point-in-time (PIT) authority. Corrected V7 research uses versioned exchange-verified OHLCV, provenance-remediation inventories, and immutable official-source snapshots referenced by the corrected-baseline manifests.

## Authority

1. Machine-readable manifests, hashes, lineage inventories, and tests.
2. Completed run artifacts tied to those identities.
3. Reusable fail-closed PIT/provenance primitives now in development main, with byte-identical branch provenance in `research/v7/source-adoption-register-20260922.json`; remaining materialization builders and result evidence stay at `fix/v7-historical-pit-reconstruction@e3bce91`.
4. Runtime observations with timestamps and TTLs.
5. Reports and historical notes.

## Inputs

- Official TWSE, TPEx, TAIFEX, TDCC, and MOPS publications.
- Provider feeds such as yfinance and FinMind only where an explicit domain policy admits them.
- Historical lifecycle and exchange-session evidence.

## Outputs

- Raw snapshots with source identity and retrieval metadata.
- Canonical tables with provenance eligibility and revision/cutoff semantics.
- Dataset identities consumed by feature, universe, execution, and label contracts.

## Current Implementation

The corrected pipeline separates raw availability from canonical admissibility. OHLCV source eligibility is evaluated per segment and field, not granted provider-wide. Mixed or unknown provenance fails closed for execution claims. Dataset reuse requires matching identity; a new runtime does not imply rebuilding an unchanged matrix.

## Evidence Classes

- Official immutable snapshot: may establish historical facts within its scope.
- Canonical derived table: usable only with lineage and contract identity.
- Mutable provider cache: input evidence, not standalone PIT proof.
- Synthetic or backfilled value: never official execution evidence.

## Current vs Legacy / Research

Legacy V6 datasets remain useful for reproducing historical work, but cannot be relabeled as corrected PIT evidence. Corrected V7 datasets are separate contracts. Research caches such as `v7_prepared` do not override canonical manifests.

## Invariants

- Preserve raw data separately from canonical outputs.
- Record source, observed/publication time where available, revision policy, and hashes.
- Reject unexplained shrinkage or replacement of established source data.
- Provenance eligibility is evaluated before any proxy rule.
- Current-world classifications must not be backfilled into history without evidence.

## Window21 bounded FinMind proof

Branch research/v7-window21-finmind-proof is UNMERGED/NOT_DEPLOYED; its finite
18-query grant is CLOSED. Actual retrospective price samples support six identifiers,
not market-wide coverage or formal admission. Raw/zero/emerging semantics stay distinct;
vendor TradingDate contains future dates and a known20261023 closure, so it is not
actual-session authority. StockInfo update date is not historical membership.
Canonical bounded evidence: [proof](../../research/v7/window21-evidence.json).
W19 protected collector/runtime is not changed by these data findings.

## Window18 branch-local manual pilot

`development/v7-window18-manual-collector` adds a manual-only CLI and bounded CSV QC,
UNMERGED/NOT_DEPLOYED. W18 exactCSV one-shot budget is consumed; no daily scheduler,
production authority or data/research admission. Raw/manifest recovery reuses v1
snapshot/W09 guards; original QC/source remains immutable, revised diagnostic separate.
Root `.artifacts/window18-manual-pilot` has actual bytes; same-host restore is not
independent backup (BACKUP_PENDING). See [pilot evidence](../../research/v7/window18-evidence.json).

## Known Limitations

Branch-local engineering candidate `fix/v7-snapshot-immutability` (2026-10-20)
adds no-clobber source snapshot publication: same-identity/different-byte
manifests are refused; identical retries retain the original inode/mtime.
Concurrent-writer and SIGKILL fixtures are tested; this is not a data-quality
approval, production rollout or full power-loss durability certification.
Trusted local snapshot directories and hard-link/fsync support are required.
See [recovery checkpoint](../00_Project_Map/Current_State.md#bounded-recovery-checkpoint--2026-10-20).

The same branch's subsequent Unit A completes the v1 reader integrity boundary:
strict fields/types, finite JSON numbers, duplicate-key rejection, canonical
publication identity and directory-relative no-follow raw/manifest reads.
Compatible valid v1 writer output remains readable, including HTTP error
observations (integrity does not mean source success or PIT admission).
Trusted/stable root ancestors remain required. No frozen policy changed.

- Some external publication timestamps and historical revisions remain incomplete.
- Provider availability is broader than evidence eligible for strict PIT use.
- External source behavior can change and requires revalidation.

## Do Not Use / Do Not Mix

- Do not treat `Data/processed_v6` as strict PIT evidence.
- Do not mix official snapshots and fallback OHLCV without segment lineage.
- Do not use synthetic prices or present-day metadata to prove historical executability.
- Do not overwrite legacy A/B/C artifacts or reinterpret them as tradable evidence.

## Important Paths

- Correctness authority: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/`
- Provenance remediation: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/`
- Historical execution evidence: `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/`
- Legacy provider store: `Data/processed_v6/`

## Related Domains

[Features](Features.md), [Universe](Universe.md), [Execution](Execution.md), [Labels](Labels.md), [Training](Training.md)

## Update Triggers

Update after a source-policy revision, new canonical snapshot, lineage mismatch, changed PIT cutoff rule, or provider migration.

## Strategic recovery planning pointer — 2026-10-21

Current objective/implementation gaps are recorded in the
[dated recovery report](../../reports/v7_strategic_roadmap_recovery.md).
The sole active cross-domain backlog is the
[existing research program](../../research/v7/research-program.md#current-development-program--2026-10-21).
This pointer does not replace this domain's machine contracts or promote evidence.
Five CPU fixes have a verified integration candidate from `4c388de`, UNMERGED /
NOT_DEPLOYED. PORT-01–04 H1 fixture engineering is now COMPLETED_VERIFIED;
Human Product Slice Review remains pending. See the
[acceptance report](../../reports/v7_portfolio_history_acceptance.md). Strict Phase 0
STOP, E1 portfolio admission=false and production boundaries remain unchanged.

## E1 preparation integrity — autonomous window 2026-10-21

Branch-local `development/v7-autonomous-charter` rejects unequal date/split
iterables in `validate_split_rows`; every input row needs an assignment. Prior
zip truncation could silently omit a tail. Frozen split/PIT/label/provenance
semantics are unchanged. CPU synthetic evidence, UNMERGED / NOT_DEPLOYED;
no matrix search, generation, formal-data edit or strict readiness promotion.

The same window also validates serialized source snapshot v1 content with the
existing reader rules before publishing raw or manifest bytes. Invalid status,
nonfinite/unserializable parameters and malformed observations fail without any
publication; legal HTTP error observations remain supported. Nineteen snapshot
and49 combined PIT fixtures PASS, independent19 PASS; six valid writer outputs
match main6d74d738 bytes. No schema/PIT policy or formal source data is changed.

## Charter integration — 2026-10-21

Window01 source/governance is now LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED
at exactf7af348 (supersedes its branch-local status only). Pre/post CPU regression
and authority checks are in [integration evidence](../../research/v7/charter-integration-20261021.json).
A separate Human-authorized cross-domain CPU window may proceed in isolation;
research contracts, formal data, production and StrictSTOP boundaries are unchanged.

## Window02 provider diagnostic — 2026-10-21

[Offline7-case diagnostic](../../reports/v7_cross_domain_window02.md) at mainc1179fe
reproduces TWSE wrong-date relabeling and TPEx positional schema drift; MOPS monthly
partial-market return and blank publication time are recorded as completeness/PIT
risks requiring domain decisions. AST function-body fixtures, no production import,
network or data writes. Independent review reproduces all7. This does not certify
external providers or change cache/publication semantics; production source fixes
remain HUMAN_DECISION_REQUIRED in the sole backlog.

## Window03 parser reliability — isolated candidate

Human authorized offline deterministic parser fixes. TWSE response date must match
requested session before Date assignment. TPEx binds exact required names to indices;
missing/ambiguous names reject through None. Existing price fallbacks/units unchanged.
Eight full-module fixture tests pass, independent8 PASS; fake config/transport only.
MOPS quarterly completeness rules verified, monthly partial-result policy unchanged
and still Human-only. No formal data writes, live source checks or production rollout.
[Current queue](../../research/v7/research-program.md#data-evaluation-window-03),
[evidence](../../research/v7/data-evaluation-window03-evidence.json).
Window02's historical provider probe must run at pinnedf34f9f3; its defect oracle
is intentionally historical, not a regression expectation for this repaired parser.

## Window05 monthly revenue feasibility — 2026-10-23

Branch-local F2 reverified the bounded historical0/455040 archive audit (2334issuers,
2002–2026), distinct from later455160mutable provider rows. No newly qualified
historical publication/revision rows acquired. NOT_CURRENTLY_FEASIBLE refers to
broad historical PIT restoration on available evidence, not universal impossibility.
Monthly policy remains EXCLUDED_PUBLIC_RECONSTRUCTION_EXHAUSTED; forward vintages
are PROPOSED/uncollected. [Report](../../reports/v7_window05_revenue_pit.md),
[evidence](../../research/v7/window05-revenue-feasibility.json). UNMERGED/NOT_DEPLOYED.

## Window22 bounded continuous proof

10 hindsight-selected stocks2019–2023,9864rows, current retrospective versions only.
Official listing/delisting boundaries partial; daily tradability/calendar/actions remain UNKNOWN.
No label values/RankIC/fit/admission; zero/missing not filled. Private root and same-host restore BACKUP_PENDING.
See [bounded report](../../reports/v7_window22_historical_panel.md) and sole backlog. W19 protected, W12HALTED unchanged.

## Window23 engineering-only real-data smoke

W22strictmask remains19,728false. Independent engineering mask7,009 samples under explicit W23 scope,
not historical data admission.2019/20andTPExfullcalendar unavailable; only2021–23TWSE subset used.
Raw-price diagnostic labels/actions/publication/vintage limitations explicit. No executable-return claim.
See [report](../../reports/v7_window23_calendar_smoke.md) and existing sole backlog; W19 untouched.

## Window24 limited retrospective raw-price proxy pilot

36frozenhindsightcohort,31new+5oldraw;33eligiblecurrent/newnames,3delistedstartsUNKNOWN retaineddiagnostic.
Threeindependentmasks:engineering46291/limitedexploratory44266/strict0;nocorporateaction/PITqualificationpromoted.
Paired27–33stocks supports only fixedL0/Ridge rawproxy diagnostics; no superiority/trading/admission claim.
See [report](../../reports/v7_window24_expanded_pilot.md)/solebacklog; W19untouched; rawprivate/backuppending.

## Window25 bounded robustness audit

Source2321 rawproxycase crosses6missing sessions; causeUNKNOWN, no newactionthreshold/filter. Historicalpublication/vintage/actions notqualified; originalW24data/masksunchanged.
See [report](../../reports/v7_window25_robustness_audit.md) and solebacklog. 0datarequests/fit;W19protected;FormalAdmissionNOT_APPROVED.

## Window26 bounded label evidence — branch-local / not admitted

2321 的 68% 虧損減資與市場股數單位轉換已有來源證據；三筆既有 10d raw labels 算式正確，但不具同股數未調整報酬語意。
原始停牌公告與其他事件覆蓋仍 UNKNOWN；W24/W25 結果不改寫／不重算。兩路 Label Policy 與 next-study 全為 PROPOSED。
參見 [W26 evidence](../../research/v7/window26-evidence.json) 與 [事件矩陣](../../research/v7/window26-event-matrix.json)。

## Window29 disjoint development / feature-label isolation

68新TWSE候選（W24重疊0）、64有conditional Prediction身分；current survivor/hindsight與4下市起日/普通股UNKNOWN不升格歷史Universe。
最小research feature-only介面先保存Predictions，再做Label Join；future label刪除只改evaluation coverage，18回歸tests與獨立Raw-derivedReplay驗證。
5d正向Mean/Median在新樣本延續、10d不一致；未知actions/asof/vintage與2022/2023曝光保留，不作模型選擇/投資/Strict PIT claim。
[Evidence](../../research/v7/window29-evidence.json)、[Report](../../reports/v7_window29_expanded_development.md)。

## Window33 recent-data preparation

[W33 evidence](../../research/v7/window33-evidence.json) pins64 TWSE historical price captures2024–2026; raw bytes preserved, SHADOW_PREPARATION_ONLY / UNMERGED / NOT_DEPLOYED. The679 observed price sessions are not a qualified calendar; W07 rejects qualification, so model Fit/Offline scores remain BLOCKED. W16/W18 one-session parity does not establish two official consecutive vintages. UNKNOWN events/vintage and current-survivor bias remain; no research admission or W19 access.

## Window35 complete raw missingness

[W35 closure](../../research/v7/window35-closure.json) rebuilds all217 missing rows from12031 raw CSV rows, not capped exception samples. All are outside the current fixedTWSE64, whose5-session Feature-only inputs are64/64 each day. This is scope impact, not confirmation of legitimate no-price/suspension or instrument type. Original raw/vintage/QC retained, no zero-fill/ffill/data admission; [full distribution](../../research/v7/window35-missing-distribution.json) keeps UNKNOWN causes and lexical type hints separate.
