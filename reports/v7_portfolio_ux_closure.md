# Portfolio History Final UX Closure — 2026-10-21

Status: COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED, fixture engineering only.
Engineering recommendation: GO for Human Product Slice UX Closure review.
Human accepted the dual-mode direction and authorized these two finishing tasks;
this report does not grant final Human acceptance, H0 integration or H2 admission.

## Scope and identity

- Candidate: `development/v7-portfolio-ux`, existing `.worktrees/v7-portfolio-ux`.
- Exact baseline: `1553a59dcf7b9479702d586f4d80ad20b5bee9c0`; final identity is the
  commit containing this report (`git log -1`), preserving baseline and old reports.
- Main remains `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e`.
- [Human scope](../knowledge/02_History/Decisions/20261021-portfolio-ux-closure-decision.json),
  [single backlog](../research/v7/research-program.md),
  [complete evidence index](../research/v7/portfolio-ux-closure-evidence.json).
- No Engine, Exporter, accounting, API, fixture, Decimal formatter/comparator or
  model-data change. 131 baseline files (backend, portfolio, original API/evidence,
  pinned skills) are byte-identical; exact hashes are in the index.
- Fixture SHA-256: `d7fd650132e1261a7fb5275cf44c7481b9f86f226253f35f8dd49003953035d6`.
  `research_admission=false`, Synthetic Fixture, Strict Phase 0 STOP remain visible.

## Completed changes

**UX-FINAL-01:** extracted the existing timeline into a small local React component.
A layout effect adjusts only list `scrollTop` from measured bounds; ResizeObserver
handles container size changes. No focus calls or `scrollIntoView`. Initial last
selection and subsequent selections remain visible inside the timeline; `aria-pressed`,
`aria-current=step`, cyan border and “目前選取” distinguish selection. The page itself
is not scrolled to reveal a below-fold timeline, preserving the user's page position.

One evidence-supported repair: mobile chart selection caused browser scroll anchoring
when description height changed (`853→829`). Injecting only page-scoped
`overflow-anchor:none` confirmed `853→853`; the same narrow CSS change then passed
final acceptance without injected styles. Desktop pointer remained `384→384`.

**UX-FINAL-02:** descriptions prioritize precise adjacent-state differences in natural
Traditional Chinese, distinguish retained/removed pending targets, describe signal
kind without claiming execution, and explicitly decline unknown causes or corporate
action types. They use deterministic existing data, no LLM calls or financial policies.
No implementation repair was needed after the failing narrative tests were added.

| Event | Before | After |
|---|---|---|
| Signal 3 | 出現待成交目標。僅比較相鄰事件的完整狀態，不推定原因。 | 與前一筆相比，新增了待成交目標。這筆紀錄標示為訊號事件，並不代表已成交。 |
| Unchanged 4 | NAV、現金、持倉與待成交目標未變，累計成本未變。此處不推定事件原因。 | 與前一筆相比，淨值、現金、持倉及累計成本均未變動，待成交目標仍保留。這筆紀錄標示為市場時段，但現有欄位不足以判定未變動的原因。 |
| Corporate 7 | NAV增加；現金增加；持倉數量改變；持倉最後價格改變。僅比較相鄰事件的完整狀態，不推定原因。 | 與前一筆相比，淨值增加、現金增加、持倉數量改變、持倉最後價格改變。這筆紀錄標示為公司行動，但現有欄位不足以判定具體類型或原因。 |

All seven exact before/after descriptions are preserved in the evidence index.

## Screenshots and browser evidence

- [Desktop initial last event](../research/v7/portfolio-ux-closure-desktop-default.png)
- [Desktop selected event5, pointer moved outside chart](../research/v7/portfolio-ux-closure-desktop-selected-clean.png)
- [Mobile general](../research/v7/portfolio-ux-closure-mobile-general.png)
- [Baseline clipped selection](../research/v7/portfolio-ux-closure-red-desktop.png)

Actual Chrome `155.0.8059.85`, desktop1440×1000 and mobile-emulated390×844;
not a physical-device claim. Final receipt `2026-10-21T06:17:23.978Z` has
`diagnosticNoAnchor=false`, no runtime exceptions. Both initial selections are7,
pageY0, selected item fully inside list; desktop list scroll84/mobile194.
Each viewport's seven selections preserved focus and pageY; NAV/Cash/Costs,
positions/pending, chart-selected dot, summary and timeline match the fixture.
Actual chart/timeline pointer and Enter/Space/Home interactions pass. Full API JSON
matches artifact/source/fixture flags. Desktop advanced7 exact events and mobile
advanced1 basic smoke pass; no additional advanced-mobile visual optimization.

Existing Recharts tooltip may retain a prior hover/keyboard event independently of
the selected dot and state. This is an inherited nonblocking interaction limitation,
not changed in this two-task closure. The [original selected screenshot](../research/v7/portfolio-ux-closure-desktop-selected.png) records it; the final event5 screenshot uses normal timeline pointer selection and pointer movement outside the chart, with no DOM/style injection.

## Tests and independent review

| Verification | Actual result |
|---|---|
| Presentation unit tests | 9 PASS, includes all7 narratives and beyond-rounded-precision comparison |
| Frontend API contract | 8 PASS |
| Existing V7 and summary tests | 4 + 4 PASS |
| Python Portfolio/Journal/History/process/health/audit | 67 PASS |
| Python Portfolio API | 4 PASS, exact artifact and fail-closed errors |
| Governance | 23 PASS |
| Frontend build | PASS; inherited large-chunk warning remains |
| Changed JSX/helper/test/browser lint | PASS |
| Knowledge impact/health and diff-check | See final evidence logs; no structural FAIL |

Independent code reviewer (`gpt-6-astra`, `/root/ux_review`) found no Required/Critical
issues, independently ran9 tests and reviewed the anchoring repair. Separate browser
reviewer (`/root/ux_browser_review`) produced RED, single-variable diagnostic, final
GREEN and visually inspected screenshots. One product repair round, no additional
Human decisions after the initial two-task authorization. Driver screenshot timing
was stabilized; the first helper lint invocation omitted the runtime structuredClone
global, then was rerun with that explicit standard global. No rule was weakened.

Full frontend lint is not claimed clean: inherited58errors/3warnings are documented
in the prior UX acceptance; this task validates changed sources. Missing/corrupt/
unavailable behavior is covered by unchanged API/client contract tests; this closure
did not repeat the prior browser injected-error campaign or training/PIT suites.

## Reproduce and resume

Use the same fixture-only launch procedure in
[original acceptance](v7_portfolio_history_acceptance.md); UI route
`http://127.0.0.1:5178/v7/portfolio-history` when Vite runs on5178. The browser driver
is `app/frontend/src/api/portfolioHistoryClosure.browser.mjs`, arguments:
`<base-url> <fixture-artifact-path> <temporary-output-prefix> green`; dedicated Chrome
CDP9230. It requires real Chrome and existing Node WebSocket, no new dependency.
Do not set the optional diagnostic injection environment variable for acceptance.

From `app/frontend`, run `node src/components/portfolioHistoryPresentation.test.mjs`,
`node src/api/portfolioHistory.test.mjs`, existing `v7.test.mjs` and
`runStatusSummary.test.mjs`; `node node_modules/vite/bin/vite.js build`.
Scoped JSX lint uses the project config. Helper/test `.mjs` lint uses explicit
`--global URL --global structuredClone --rule no-unused-vars:error --rule no-undef:error`
with `--no-config-lookup`, since the existing config targets JSX/JS.
API regression uses fixed FastAPI0.115.12/httpx0.28.1/uvicorn0.34.3 in a temporary
venv; browser server uses the existing identically pinned user venv. WSL Node24.19.0,
Windows browser runner Node24.14.0. MARKETMAMBA_DATA_ROOT is unset; no Data realpath
was consumed. Test services are stopped after evidence collection.

Canonical Current State, Authority Map, UI domain, worktree register, decision history
and the sole research-program backlog are updated in this same coherent commit.
Native Goal belongs to this closure thread; counters are not certified aggregate
usage of all agents and no precise aggregate token/hard-limit claim is made.

## Remaining gates

Stop here; no further UX work is authorized. Recommend Human review of the fixed
closure candidate. H0 requires explicit Human approval naming the exact full candidate
commit and local development main target, including its CPU→PORT→UX ancestry. Before
integration, recheck main/candidate identities, ancestry, clean worktrees and run
appropriate combined integration/knowledge gates; if main changed, prepare a fresh
isolated candidate. Earlier CPU integration GO is not H0 authorization. Existing lint
limitations must remain disclosed. No merge, push or deployment occurred.
H2 real-data/research admission is separately unapproved; Strict Phase 0 remains STOP,
E1 identity-matched Frozen Matrix remains BLOCKED, MAS Core Live remains uncertified.
