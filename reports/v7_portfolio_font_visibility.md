# Final H0 browser blocker resolution — 2026-10-21

COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED. Engineering H0 recommendation: **GO
for new fixed-candidate Human approval**, not permission to merge. Main unchanged.
Human already accepted the limited product direction; this task only repairs the
late-font selected-item clipping. No redesign or financial semantics change.

## Identity and scope

- Base product: `6c89479416ad9fa429db7818cc645f729367eac9`.
- New candidate branch: `fix/v7-timeline-font-visibility`; final fixed identity is
  this containing fix/evidence commit, supplied as a full SHA in the handoff.
- Main: `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e`, unchanged and ancestor.
- Audit evidence: `7ecb92016557af135ffd28c80892aa6be9d727df`, immutable
  `research/v7/portfolio-h0-font-diagnostic.jsonl` and preflight report/index.
  Audit is NOT an ancestor of the new candidate; no audit commit was cherry-picked.
- [Human authorization](../knowledge/02_History/Decisions/20261021-portfolio-font-blocker-authorization.json).
- [Evidence index, raw logs/receipts, hashes and commands](../research/v7/portfolio-font-visibility-evidence.json).

## Root cause and minimal correction

The existing ResizeObserver observed only the fixed-height scroll container.
Loading the existing fonts changes event-row heights and the cumulative position
of the selected last row, but does not change the observed container height.
Measured rows changed from62px to62.5px each; container clientHeight stayed360px
on desktop and250px on mobile. Thus the earlier layout-effect scroll position can become insufficient after fonts
load; changing React selection would correct it, but the initial selection remains7.

Real Chrome font requests were held until the initial loading layout was measured,
then released. Tests await `document.fonts.ready` and two layout frames and assert
geometry directly, without polling visibility until success. Both viewports reproduce
the original failure on the unchanged product before the fix:

| Case | RED after font load | GREEN after font load |
|---|---|---|
| Desktop1440×1000 | selected7 clipped3.5px | fully visible; scrollTop80→84 |
| Mobile390×844 | selected7 clipped2.5px | fully visible; scrollTop190→194 |

Product diff is exactly two added lines: an explanatory comment and
`for (const row of container.children) observer.observe(row);` in the existing
layout effect. Observing all rows covers both the selected row and preceding-row
height changes. Existing scroll bounds, focus behavior, container observation and
observer.disconnect cleanup are unchanged. Scroll changes do not resize these rows.
No font-specific timer, infinite retry, CSS hiding, new dependency or markup change.

The test records row rectangles, selected button, container bounds, font status,
actual held requests, pageY and focus. It rejects missing/failed font loads or no row
size change, so a skipped font race cannot count as PASS. Each viewport held2 real
requests, changed row sizes and ended loaded. PageY stayed0; focus remained unchanged.
The small geometry tolerance follows existing browser subpixel checks; the original
2.5/3.5px failures fail it. Actual content and selected marker were visually reviewed.

- [Desktop after delayed fonts](../research/v7/portfolio-font-green-desktop.png)
- [Mobile after delayed fonts](../research/v7/portfolio-font-green-mobile.png)
- [Desktop RED](../research/v7/portfolio-font-red-desktop.png)
- [Mobile RED](../research/v7/portfolio-font-red-mobile.png)

## Acceptance and bounded H0 preflight

| Verification | Result |
|---|---|
| True delayed-font desktop/mobile regression | RED→GREEN, no failures/exceptions in GREEN |
| Events1–7 after font load | Both viewports visible; page/focus unchanged |
| Existing closure browser | Desktop/mobile7 general events, chart/timeline/summary/state, pointer/keyboard; desktop7 advanced exact, mobile1 smoke PASS |
| Browser missing/corrupt/network | No stale fallback PASS |
| Frontend units | Presentation9 + client8 + V7/status4 + summary4 PASS |
| Frontend build/scoped source and driver lint | PASS |
| Portfolio/Journal/History/process/health/audit | 67 PASS |
| PIT/Snapshot | 47 PASS |
| Operations / mocked fetch-only | 81 / 3 PASS |
| Portfolio/status API / real HTTP corrupt/restart E2E | 8 / 1 PASS |
| Governance | 23 PASS |
| Identity | 609 protected files byte-identical to6c894794 |

Engine/Exporter, backend/API, frontend API, Decimal presentation helper, styles,
prior evidence and Fixture are unchanged. Fixture SHA256 remains
`d7fd650132e1261a7fb5275cf44c7481b9f86f226253f35f8dd49003953035d6`.
No real model signal, trading, research admission or production evidence was generated.

Independent code reviewer `gpt-6-astra /root/ux_review` reviewed the root cause,
observer target/cleanup behavior and meaningful RED. Separate browser reviewer
`/root/ux_browser_review` ran actual Chrome and visually inspected screenshots.
Original Required finding is resolved; no other Required/Critical in tested scope.
Product repair/rework rounds after initial implementation:0. One test-harness exit
handling adjustment avoided Windows Node libuv assertion after throwing; exact checks
and failure list unchanged, and RED still exits nonzero. Driver hashes are recorded.
Additional Human decisions during this authorized correction:0.

Full frontend lint remains58errors/3warnings, message-for-message identical to old
main baseline. Large chunk, inherited tooltip retention and nonblocking Governance/
AuthorityDrift/LegacyDocs warnings remain; no structural knowledge FAIL. Historical
raw CPU logs/vendor whitespace is preserved, not rewritten to make a wider diff green.
Mobile is Chrome emulation390×844, not physical-device validation. No further mobile
advanced visual work. Temporary API/Vite/dedicated Chrome all stopped.

## Reproduce and integrate only after new Human approval

Use fixed fixture/API/Vite/CDP commands in the evidence index. New regression:
`node app/frontend/src/api/portfolioHistoryFont.browser.mjs <base-url> <artifact> <output-prefix> [CDP-url]`.
It requires actual existing fonts to load and exits nonzero on failed evidence;
network unavailability is not PASS. Existing closure driver remains unchanged.
Error-only harness is retained as verification evidence, extracted from the prior
UX driver's same missing/corrupt/network assertions. No product uses that harness.
CPU suite commands and outputs are embedded in the index for exact reproduction.

Native Goal creation was rejected because the old paused H0 goal remains unfinished.
No internal goal database/configuration was changed, no automatic continuation or
precise aggregate token usage claim. Work completed under this explicit bounded
Human instruction with Git/receipts as persistent checkpoint.

After Human explicitly approves the **new full SHA**, use this guarded procedure:

```bash
cd /home/frank/projects/MarketMamba
candidate='<NEW_FULL_SHA_EXPLICITLY_APPROVED_BY_HUMAN>'
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = 83a6a2e6f4a56db318ee07e7e737cc01de3ce48e
test -z "$(git status --porcelain)"
test "$(git -C .worktrees/v7-timeline-font-visibility rev-parse HEAD)" = "$candidate"
test -z "$(git -C .worktrees/v7-timeline-font-visibility status --porcelain)"
git merge-base --is-ancestor HEAD "$candidate"
# Revalidate candidate evidence/hashes and required checks; STOP on any mismatch.
git merge --ff-only "$candidate"
test "$(git rev-parse HEAD)" = "$candidate"
# Then required main-location regressions and a separate integration closure record.
```

These commands were NOT executed this turn. Existing H0 approval names old6c894794
and cannot substitute for approval of this new SHA. Do not merge the audit branch.
If main/candidate moved or checks fail, stop rather than force integration.

Canonical Current State/Authority Map, UI domain, worktree register, decision history
and sole research-program backlog are updated in this same candidate. No further
UX or research work starts. Strict Phase0 STOP, E1 Frozen Matrix BLOCKED and H2
unapproved remain. Stop for Human new-fixed-candidate H0 confirmation.
