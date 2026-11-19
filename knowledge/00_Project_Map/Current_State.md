# MarketMamba Current State

## P3 序列分數／組合工程候選 — 2026-11-20

`job-run-mwnu9iri-e7a54a` fixer，detached `69dc5baaee2d8e5c35d36458335c29fb4f26b619`；UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，依指示不 commit/push/merge/GPU/新增 agents。[P3 修訂1](../../tasks/spec-mamba-portfolio-p3-20261120.md) 明確取代初版 no-conv：官方 Mamba2 temporal 預設卷積、cross d_conv2。[adapter](../../V6/research_lab/rolling_seq_scores.py) 明列完整有效設定與 default baseline，逐項 drift gate、manifest 實際 mixer／Windows-WSL 環境欄位、fsynced 持久 budget.jsonl 含失敗／中止／熱STOP與跨程序保守恢復、STOP人工清除、14h啟動拒絕。[evaluator](../../V6/research_lab/portfolio_eval.py) 明列 CI＋Holm 判定、確認性完整cfg/admission gate及獨立parts目錄；P1/P2 default不變。

真實2021–2026 splits hash／逐列 split_rows／rolling_runner eligibility 比對一致，outer gap21、inner gap20，未放寬 purge。CPU／合成E2E涵蓋神經分數讀取、P1 pinned_reference、readiness與parts/out，官方collect分支由訓練邊界替身證明可達；in-memory七類反證均exit1。指定全套 **525 passed、1 skipped、4 warnings／137.80s**。詳細實際數值、RED→GREEN與解鎖程序見[工程報告](../../reports/v7_p3_portfolio_engineering.md)。SIGKILL無法得知精確終止時刻，恢復保守扣入至重啟時間並鎖STOP；不宣稱精確GPU時數。僅同namespace累計，不得改namespace規避同研究14h上限。

**UNRESOLVED / GPU_SMOKE_NOT_RUN**：模型文字矛盾已由修訂解除；正式GPU環境／forward-backward smoke仍未由本輪驗證。只有source／environment／frozen cfg／mixer相符且人工審核的官方GPU smoke evidence才能讓新manifest RESOLVED；CPU注入永遠CPU_ENGINEERING_ONLY。Next：獨立審查diff／logs，另取得GPU授權後依報告程序產生smoke；不產生正式P3結果或交易ready宣稱。舊builder493測試與CPU冒煙保留為歷史工程證據，不重標成新結果。正式Data、V6/models、production/scheduler與P1/P2研究結論不變。


## P2 low-turnover result — 2026-11-20

Pre-registered P2 ([spec](../../tasks/spec-lowturnover-p2-20261119.md) with amendment 1, [report](../../reports/v7_p2_portfolio_evaluation.md)), source commit `fcfc0e6`, E2 RESEARCH_ONLY / NOT_DEPLOYED. Hold buffer plus 20-session rebalance cut annual turnover from about 15x to about 5x and cost drag from about 13% to about 4.5% for both C0+ and LightGBM (descriptive, six folds). Confirmatory tests on clean folds 2021-2023 remain unconfirmed: H-P2a (net excess vs proportional-cost benchmark) and H-P2b (vs P1 primary) have 95% CIs that cross zero. The P2 primary design was informed by P1 descriptive results on the same folds, so it is not an independent test. Minimum fee (20 TWD per order) costs about 0.6 points a year at 1M TWD capital and vanishes at 10M. Next: an unexposed-forward confirmatory test of the fixed P2 primary cell, adj_close/delisting evidence, and Mamba2/S-Mamba under the same portfolio protocol.

## P2 amendment 1 bounded repair candidate — 2026-11-19

本輪 `job-run-mwnpp2vy-e42792` builder，沿用 detached base `1920fde0abfead53e84a2a9354a695e59a03764c` worker；UNMERGED / NOT_DEPLOYED，Human 禁止 commit/push/merge/新增 agents。依 [P2 登記修訂 1](../../tasks/spec-lowturnover-p2-20261119.md) 修正既有 [engine](../../V6/research_lab/portfolio.py)、[evaluator](../../V6/research_lab/portfolio_eval.py) 與 [P2 tests](../../V6/research_lab/tests/test_p2.py)：P2 主要基準不收 min_fee，只收比例成本，另存也收最低費用的 descriptive sensitivity；策略成本不變。OFAT＋三指定組合去重為 14 格／fold×system（a/c 與 OFAT 重複），六折兩系統 168 格，另 12 個同分數 P1 references，共 180 cells。改 top_k 維持 buffer=2×top_k。凍結加選股數超過 top_k 時該次不新增／不強制賣出可成交持股、保留原股數；terminal liquidation 不變。P2 所有格明確啟用 preserve_frozen_overflow（含buffer=top_k），P1預設關閉，保留原浮點算式。子集 --fold／--system 的 grid_complete 必為 false，不進完整確認性。P2 schema／run_hash 與 P1 不同，舊 P1 parts 無法續跑；位元一致僅指預設路徑逐日策略與基準報酬。

指定全套 **474 passed／1 skipped／4 warnings（90.94s）**；P2 focused 28 passed／6.97s，初始 RED 8 failed／16 passed，補強相等buffer RED 1 failed後GREEN；額外 benchmark／Holm／golden mutations 均被拒絕後還原 GREEN。pre-P2 ace2eef 原始引擎獨立產生 29 日合成 golden；實際 2023 P1 reference 與舊 part 231 日／3,696 bytes 位元一致。2023/C0+ CPU 主要格冒煙 **109.417992s**（含啟動 110.895806s），2 COMPUTED cells／0 ERROR，221 主格報酬日；策略區間淨報酬 25.522813%，比例成本基準 33.948999%，日均超額 -0.000326306791；也收最低費用基準為 13.862428%，僅敏感度。H-P2a/b 都 INCOMPLETE_ENGINEERING_ONLY／未證實（缺 2021/2022、primary-only/subset）。修訂揭示原基準最低費用抬高淨超額，不把敏感度升格。

實測 P2 cell 9.182894s、P1 reference 11.158765s、其餘 90.554148s；完整 180 cells 線性外推單程序 29.45 分鐘，12 個 fold×system job／3 程序理想估計 15.35 分鐘。實測峰值 RSS 2.629967 GiB／程序，加20%裕度3.155960 GiB，三程序加2 GiB系統保留約11.47 GiB；六核不代表12GB可跑六程序。完整網格／多程序未執行，其他格、年份、I/O與累積RSS未知，不作時程上界或承諾。子集各自run_hash，無跨hash確認性彙整器。

GENERATED 本機 evidence `.artifacts/p2/job-run-mwnpp2vy-e42792/`；report SHA-256 `77acb7d28f6337345c0552d1ef08a8cec261e3db2d5aead1d711212679ee6394`，identities/logs/估時見 [工程報告](../../reports/v7_p2_portfolio_engineering.md) 與其 verification-final.json。沿用唯讀 full P1 scores，不重訓；Python3.12.3 pilot09 venv／CPU單執行緒／MARKETMAMBA_DATA_ROOT unset，data realpath `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`；未碰formal Data／V6/models、GPU、acquisition、production、scheduler、wakeups、recurring jobs。起始clean；成果依no-commit指示留核准範圍unstaged dirty，不宣稱clean。Knowledge-impact：基準／網格／frozen gate語義改變，已更新Current State、Authority Map、Portfolio／Research notes及既有 [Worktree Register](Worktree_Register.md)。knowledge_health.py --ci exit0／WARNING，無structural FAIL；syntax／artifact hash／link reference／git diff --check通過。下一步為主控獨立審查diff/artifacts並決定完整執行；工程完成不等於Human acceptance，E2 RESEARCH_ONLY，既有P1與修訂前歷史artifacts不修改、不升格。

## P1 portfolio-level evaluation result — 2026-11-19

Pre-registered P1 ([spec](../../tasks/spec-portfolio-eval-20261119.md), [report](../../reports/v7_p1_portfolio_evaluation.md)), source commit `ace2eef`, E2 RESEARCH_ONLY / NOT_DEPLOYED. Clean folds 2021-2023, primary cell (top 50, 10-session rebalance, 10bp slippage, raw prices): C0+ and LightGBM net excess over the same-universe equal-weight benchmark are both unconfirmed (95% CI crosses 0); C0+ minus LightGBM is also unconfirmed. Net cost drag (about 8-18 points a year at 14-18x annual turnover) is the main go-live risk; slower rebalancing lowered C0+ drag descriptively (needs new pre-registration). No adj_close, unverified delisting coverage, non-PIT market value. Next: pre-register a low-turnover study (20-session rebalance, hold buffer) and compare Mamba2/S-Mamba under the same portfolio protocol. Output 1.35 GB stays under `.artifacts/p1/` (not in Git).

## P1 bounded evaluator follow-up — 2026-11-19

本輪 `job-run-mwmgvttf-837c02`，detached base `d2390b56d633627d942e6f4969771b1382744ba9`，既有 worktree `.worktrees/ruflo-workers/run-mwmgvttf-837c02/fixer`；UNMERGED / NOT_DEPLOYED，Human 明確禁止 commit/push/merge。只修改 [evaluator](../../V6/research_lab/portfolio_eval.py)、既有 [P1 tests](../../V6/research_lab/tests/test_p1.py) 與本節／Authority Map；根目錄 `tests/test_p1.py` 不存在，未建立重複測試樹。重用 frozen [P1 spec](../../tasks/spec-portfolio-eval-20261119.md)，不改 fitting、成本或容量算式。

缺價診斷以 finite positive close 為有效價，零量不等於缺價。依 fold 範圍分列 `pre_fold_terminated`、`in_fold_terminated`（末次有效價在 fold 內，之後連續至少 10 個觀測 session 無價）、`transient_gap`（恢復／不足 10 日／無已知歷史有效價，後者仍 UNKNOWN）。穩定無缺價代碼不進異常清單；只對 in-fold 類別附持有／封鎖次數與價值，舊 `count/codes` alias 也只代表 in-fold。保留 window 前最後有效價 metadata，包括 fold 中已完全無列的舊代碼；這不是補價或正式下市證明。Calendar 仍為 observed union，非合格 exchange calendar。容量欄位與 caveats 明示 `approximation`、NAV=1 名目金額未隨 NAV drift 調整。

`volume_units` 保存 2330、OTC 大型參考股 8069、小型參考股 1236 的實際樣本／年度量級檢查，以及 2,490 股全宇宙 5,378 個觀測日的 raw close×volume 分位數。三參考股有效 rows 為 5,377／5,308／5,370；原始量中位數 37,103,997／7,978,000／45,000。全宇宙每日成交值 p05/p50/p95 為 46,786,720,736.51552／102,778,450,957.32242／622,192,728,217.9954 TWD（shares 假設）；實測 checks 一致支持 shares×1。缺探針、UNKNOWN 或跨參考／年度量級不一致都 fail closed。探索時 6488 的 2014–2016 初期量級使大型股成交值探針出現 lots/UNKNOWN，不能據此斷言單位混雜；最終固定使用較長歷史的 8069 與相同 volume 門檻，未放寬 gate。量級支持不是每列來源、全資料同質性或 PIT 證明，固定參考股也屬 hindsight 診斷選取。

Part 同時驗 run identity、payload SHA-256、fold/system envelope 與每列 fold/system；改檔名搬移其他 fold/system 會拒絕。預設仍保留 ERROR，caveats 說明其可能暫時性。明確 `--retry-errors` 只重算 ERROR parts 中的 ERROR cells，保留成功 cell、原始 part 與原報告，追加 hash-linked `-retry-000001.json` attempts／報告。完整驗證的連續 attempt 鏈擁有最新恢復狀態，缺 base、缺中間 attempt 或 hash 損毀都停止；不升格正式結果。成對描述性差異分 era 列出 paired folds、requested folds 與 missing primary folds/systems，缺任一主格即 `INCOMPLETE_PAIRED_SUPPORT`。

最終指定全套測試 **446 passed／1 skipped／4 warnings（146.57s）**；warnings 均為既有 stale-audit mutation 的 pandas API deprecation。逐項 RED→GREEN：缺價 6 failed→7 passed、容量 1→2、多檔 volume 7→10、part fold/system 2→2、missing paired fold 1→2、ERROR retry 1→1。另 7 個 contract/seed/matrix mismatch、retry chain corruption/missing/orphan、publish 後實際 SIGKILL/restart 測試通過。metadata 隔離新增 RED 1 failed，與缺價／retry 回歸 GREEN 13 passed；歷史 map 僅進診斷，不傳播至 engine pandas 操作。

CPU 2023/seed17/train500 冒煙只訓練一次：每系統 390,606 eval rows、合計 781,212 rank rows，export 111.36s。初次 evaluator 過慢，exit 130 中止，未發布 part／report；隔離診斷 attrs 後，重用原 immutable scores，不重訓，evaluation 145.54s 完成 2 parts／8 COMPUTED cells／0 ERROR。兩系統主格 2023-01-03～2023-12-20，共 231 returns；pre-fold 301／in-fold 16／transient 345，16 個 in-fold 代碼持有／封鎖次數與價值皆 0。清單包含全 raw universe／歷史代碼，不限 scored candidates，不代表正式下市宇宙。2021–2023 paired 只有 2023，明列 2021／2022 缺兩系統，仍 `INCOMPLETE_PAIRED_SUPPORT`，20 bootstrap samples 僅工程 smoke。

Ignored `.artifacts/p1/job-run-mwmgvttf-837c02-2023-s17-r500/portfolio.json` SHA-256 `03b9269d2e9fd0677baf3c1a4bf900b0d14f348ef892f151c6653af01580c8b1`；同 namespace `verification.json` 保存 source/input/output identity、實際 RED/GREEN/full-suite logs 與 interrupted attempt。2 part payload hashes、fold/system、score manifest outputs、輸入未變與當前 implementation hashes 已重新核對。Python 3.12.3 pilot09 venv、CPU（smoke CUDA_VISIBLE_DEVICES 為空）；MARKETMAMBA_DATA_ROOT unset；本 worktree Data 為普通目錄，實際唯讀價格仍從 `/home/frank/projects/MarketMamba/Data/processed_v6` 解析至 `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`，價格 SHA-256 `5df865ee9d57acf4d77de4e716ec3e2dd94dc8afae1869ec8adfdcb0bd10fc14`。無 GPU、資料 acquisition／修改、正式權重、production／scheduler 變動；未 commit/push/merge。

Knowledge-impact 檢查因 canonical map 變更提示 REVIEW_REQUIRED／WARNING；已檢查 GOVERNANCE，artifact 分類／流程不變，毋須修改治理規則。Current State／Authority Map 已更新；knowledge health 為 WARNING、無 structural FAIL（既有 authority repin candidates／legacy drift 保留）。工作樹只留上述 4 個核准檔案的未提交修改，起始無其他 dirty/untracked/staged 狀態。Blockers：本輪工程工作無；Remaining／Next recommended action：獨立檢查此 diff 與 verification evidence，再由 Human 決定 acceptance；完整 registered study 尚未執行。E2 RESEARCH_ONLY，不是 Human acceptance。


## P1 review repairs — 2026-11-19

Bounded builder repair at detached `abf0368d709b83bc916144292da8b5930cc0ff26`, existing `.worktrees/ruflo-workers/run-mwmgdq5i-f34fa7/fixer`; UNMERGED / NOT_DEPLOYED, no commit authorized. [Evaluator](../../V6/research_lab/portfolio_eval.py), [engine diagnostics](../../V6/research_lab/portfolio.py) and [tests](../../V6/research_lab/tests/test_p1.py) now persist immutable fold×system parts with input/implementation/parameter identity and payload hash; resume skips matching parts, refuses mismatch, and preserves cell ERROR type/message including nonfinite results. Final aggregation alone can promote candidate_confirmatory cells after the full clean-fold gate. Paired C0+ minus LightGBM inference separates 2021–2023 and 2024–2026.

Reports identify actual evaluation dates/return counts, early valid-price termination codes and last dates, held/blocked mark counts and normalized values. Termination cause remains UNKNOWN, not verified delisting. Caveats explicitly retain tail truncation, net-to-net benchmark costs and bootstrap blocks crossing fold boundaries. Actual read-only prices_raw unit evidence: 2330 5,377 rows, raw Volume median 37,103,997, latest sample 45,159,352 on 2026-11-18; magnitude and fetcher shares contract support shares (multiplier 1), not row-level source verification. Lot input is converted to shares before both liquidity gates and capacity; UNKNOWN units stop evaluation. Local smoke namespace `.artifacts/p1/review-2023-s17-r500/` reuses the existing P1 artifact contract without overwriting older evidence. Full registered study and Human acceptance remain outstanding; E2 RESEARCH_ONLY engineering evidence only.

Repair verification: focused review GREEN 10 passed; original behavior RED 9 failed, with direct old-evaluator exception/NaN RED 2 failed and stratum-mask/lot-conversion mutations rejected. Requested full suite: 419 passed, 1 skipped, 4 warnings / 70.11s. 2023/seed17/train500 smoke exported 390,606 rows per system (781,212 ranks); after SIGKILL (-9) with one durable part, resume skipped exactly one and computed exactly one new part, preserving original bytes/hash/mtime, then produced 8 cells/0 ERROR. Main cells span 2023-01-03–2023-12-20 / 231 returns; 19 early-termination codes each, zero held/blocked exposure for those codes in this smoke. Report SHA-256 `4930224ad55424f528e3f7ceea2b33d2ba933acd2e5b035ec6c49035d76f4eb7`; local `verification.json` pins resume and report evidence. Python 3.12.3 pilot09 venv, CPU; MARKETMAMBA_DATA_ROOT unset; prices realpath `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/prices_raw.parquet` (read-only), SHA-256 `5df865ee9d57acf4d77de4e716ec3e2dd94dc8afae1869ec8adfdcb0bd10fc14`. No GPU, data acquisition, protected weights, scheduler or production change.


## P1 portfolio evaluation builder candidate — 2026-11-19

Bounded Human-approved builder task at detached base `a0e1f00b70057b7e3fd17a8e4a5fb1f65c3acea1`, worktree `.worktrees/ruflo-workers/run-mwmfoyei-243923/builder`: [registered P1 spec](../../tasks/spec-portfolio-eval-20261119.md), [score exporter](../../V6/research_lab/rolling_scores.py), [portfolio evaluator](../../V6/research_lab/portfolio_eval.py) and [regressions](../../V6/research_lab/tests/test_p1.py) are uncommitted candidates, UNMERGED / NOT_DEPLOYED. Existing `c0plus.run`, rolling split/purge, 60-session eligibility and `portfolio.run_backtest` are reused; `rolling_runner` outputs are unchanged. Optional portfolio universe eligibility changes selection/benchmark membership only, preserving sellability of existing holdings. New thin CLIs are needed because existing IC-only rolling output discards per-row scores and the existing backtest CLI does not orchestrate the frozen grid/strata/capacity study.

Default dry-run, explicit `--run`, fresh ignored `.artifacts/p1/` namespaces only. Export requires existing read-only `stock_idx.npy`, uses unchanged fitting settings and per-date seed rank averaging. Portfolio grid has 432 parameter/basis cells per system/fold; primary and three lagged market-cap strata are separately identified. `--primary-only` is incomplete engineering smoke evidence. Confirmation requires clean folds 2021–2023, all three registered seeds, registered training cap and full grid; other folds are descriptive. No new evidence admission or trading readiness. Input/source hashes and environment accompany artifacts; absent adjustment data makes total_return cells unavailable with explicit primary raw fallback. Market-cap shares remain NON_PIT despite lagging; delisted coverage and exchange-limit flags remain unverified. Full training/full grid and independent Human acceptance remain outstanding. No data, protected models, GPU, scheduler or production changes; no commit/push/merge authorized in this task.

Builder verification (not Human acceptance): requested research suite **372 passed, 1 skipped / 71.24s**; four in-memory causal/accounting mutations each produced exit 1 and one failing regression. Local ignored evidence is `.artifacts/p1/smoke-2023-s17-r2000/verification.json`; its final report is `portfolio-scored-universe.json`. The earlier portfolio diagnostic predates scored-universe benchmark alignment and is retained unchanged. Smoke: 2023 / seed 17 / train 2,000, 390,606 evaluation rows per system, 781,212 rank rows, 91.75s export and 65.23s final evaluation. Main net annualized C0+ 11.15% / LightGBM 10.87%; scored-universe benchmark period return 31.24%; neither incomplete smoke confirms benefit.

Prices: 0 missing predicted keys, 39 missing/nonpositive opens, 15,634 absent code-date grid cells; 31,254 price rows lack prior-session cap. Coverage reasons/delisted support remain unverified. Observed-session terminal semantics end this main smoke on 2023-12-20 (231 returns), while scores extend through 2023-12-29 (239 sessions); no tail extension. Evidence/source hashes retain the evaluation-only alignment change; fitting inputs/exporter are unchanged. Full study remains outstanding.

## Research code import from pilot09 — 2026-11-19

Imported at file level (histories are unrelated) from pilot09 `rd/hybrid` commit `65d0b68da0c8b443dfb0a43b0220ef51614921ac` (also preserved on origin as `archive/pilot09-rd-hybrid`): `V6/research_lab/` (study code, rolling folds, C0+, Mamba2/S-Mamba-style models, repair and audit tools), `V6/experimental/v7_evidence_tier*` and `v7_portfolio_metrics*`, two root tests, and the roadmap/handoff/design notes under `tasks/`. NOT imported: pilot09 `V6/marketmamba` (this repo's fetch-hardening lineage is newer), `.agents`, `.claude*`, `V6/results` and app/tool changes. Verification here: 406 passed, 1 skipped, 8 subtests (pilot09 venv, `PYTHONPATH=V6:V6/research_lab/tests:.`). Large data (12 GB `.local-data`: E5 matrix, rolling folds, backups) stays in pilot09 as EXTERNAL_LARGE_DATA; any use must record path and hash. Prior findings (evidence class E2 RESEARCH_ONLY, simulated labels, survivorship bias): C0+ beats LightGBM on clean folds 2021-2023 (+.0207) but the gain decays (2026 about -.005); Mamba2 and S-Mamba-style models are statistically indistinguishable from LightGBM and show no gain from cross-stock mixing. Status: UNMERGED_TO_PRODUCTION / NOT_DEPLOYED; code presence is not production authority.

## Human policy update: V7 direction, version control, model split — 2026-11-19

Human authorized commit and push of branches and `main` to `origin` (no force-push/history rewrite; deploy, scheduler changes and paid Colab/GPU still need separate approval). V6 is legacy: no dual V6/V7 maintenance; the new system ships as V7. V6 daily fetch remains the data supply until a V7 data pipeline replaces it. Codex is the default executor for routine work; Claude is for planning, independent review and cross-checking ([AGENTS.md](../../AGENTS.md)). Local CPU/RTX 3060 first, Colab only for runs that cannot finish locally. Short-term goal: bring a pure-quant V7 model online, chosen by evidence (Mamba2 + S-Mamba is a hypothesis, not a result). Long-term: fundamentals, news, supply-chain relations. Prior research repo `/home/frank/projects/ruflo-evaluation/pilot09/marketmamba` (`rd/hybrid`, unrelated history) is read-only reference; its results are not current evidence here until imported with identity. Local evidence class: POLICY / DEVELOPMENT_ENGINEERING_ONLY.


## Ruflo development integration — 2026-11-19

Human approved Claude-primary background work, persistent memory and Claude/Codex workers. Branch `development/ruflo-integration`, base `f501a7eb28fab26c86f1bc776ad80a0ff13cfe81`: UNMERGED / NOT_DEPLOYED. [Project contract](../../tools/project_agents/integration-contract.json) and [operator guide](../../docs/operations/ruflo.md) define explicit submitted development jobs, pinned Ruflo/CLI 3.77.1 and official Codex adapter 3.0.8. Reuses official DualModeOrchestrator; project adapter owns process/exit/artifact verification, not a second MAS. One active job, at most3 workers/1writer; no periodic schedule or automatic retry. Memory is non-authoritative, explicitly shared at Git common root `.artifacts/ruflo/memory.db`; formal data/models/research/production boundaries unchanged. Local evidence is DEVELOPMENT_ENGINEERING_ONLY. Bounded acceptance: 21 offline agent tests +23 governance tests; real Claude entry submitted job run-mwmdmets-74c208, Codex fixture worker and independent Claude reviewer completed; 27-byte snapshot/hash checked. [Acceptance manifest](../../reports/ruflo_integration_acceptance.json) records source/outputs/limits. MCP↔CLI durable SQLite readback PASS; AgentDB bridge disabled after observed missing native binding/non-durable write. Local primary checkout uses development/ruflo-primary; main/production not merged/deployed.


## V6 fetch date binding and internal gaps — 2026-11-18

`fix/macro-sync-fetch-hardening`, WSL primary checkout, base `bc88a181a15ad20e41daf29ec68a623d99ccf5d8`:
UNMERGED / NOT_DEPLOYED. [Fetcher](../../V6/marketmamba/data/fetcher.py) now
rejects absent, invalid, mismatched or conflicting response date/title evidence
for BWIBBU_ALL, TWT96U and TWSE/TPEx daily institutional endpoints.
Margin/daytrade/generic/market-value catch-up reuses `_trading_calendar()`
(prices_raw distinct dates). The follow-up to `721eb84` bounds the scan start to
`max(first stored date, today - lookback_days)` (default 30 calendar days;
non-positive lookback rejected), retaining the latest-10 missing-session cap.
A missing weekday today is always a candidate even without today's price calendar;
None/empty responses write no rows. Empty margin/daytrade sources bootstrap only
today; generic/market-value missing-file behavior remains unchanged.
Market value skips candidate gaps older than 10 calendar days with a warning
pointing to `V6/scripts/backfill_stale_202607.py` and its non-PIT share limitation.
It reports only dates actually written; missing prices cannot create market value.
This supersedes `721eb84`'s unbounded historical scan and calendar-only today
selection, not its response-date gates. Whole missing sessions only: incomplete
stock coverage, contaminated existing values and price/institutional historical
catch-up remain outside this patch.

[Offline regressions and in-memory mutation runner](../../V6/scripts/test_fetcher_date_gaps.py):
16 gap/date tests + 8 official parsers + 3 raw append + 3 fetch-only = 30 pass
(original 23 retained, two expectations updated for the requested new semantics).
New regressions against `721eb84`: 17 assertion failures, 0 runtime errors across
16 methods. 13/13 in-memory mutants killed by assertions, 0 runtime errors.
Fetcher SHA-256: `0074a54c31b031accf9d4f3542d9ccebe6ef487ec45d7659ad6d3b7204c0b0ec`.
Evidence class: SYNTHETIC_OFFLINE_ENGINEERING, not actual source validation or PIT
admission. All data writes are temporary and HTTP mocked; no formal data or
.env/credentials accessed. Python 3.12.3 environment:
`/home/frank/.venvs/marketmamba-fetch`, pandas 3.0.2, pyarrow 24.0.0;
MARKETMAMBA_DATA_ROOT unset; formal symlink resolves to
`/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6` (not written).
Current MOPS shares remain non-PIT even within the 10-calendar-day limit; capital
changes can bias recent historical values. No existing historical rows repaired.
Next: review branch; any formal-data remediation/deployment needs separate Human
authorization. No production authority, scheduler, model, push or merge change.

## Run status count patch adoption — 2026-11-13

Human-authorized independent adoption on `feat/run-status-count-validation`,
worktree `.worktrees/run-status-count-validation`, base
`8b0786018d763942d0ea68eb9ce537cb33f8b1fd`; LOCAL_MAIN_INTEGRATED / NOT_DEPLOYED.
Human-authorized fast-forward integrated `e06fb256211f21612a25af4b10f709a756540d32` into local main.
The feature worktree is retained, clean at that commit; retired N/A.
The original MAS delivery patch (2,542 bytes, SHA-256
`d8c25509dab82400f212e803a8c44a92d5b5d854bf926c8b41f0e60f898d20ad`)
changes only `runStatusSummary.mjs` and its test: non-negative integer counts
retain localized formatting; invalid counts display an em dash without coercion.
Four existing Node test files: baseline 25/25; patch regression RED 5 pass/2 fail;
complete patch GREEN 28/28, no skipped tests, using `--test-isolation=none`.
JSON fixture `research/v7/portfolio-history-artifact.json` is readable (7 events,
SHA-256 `d7fd650132e1261a7fb5275cf44c7481b9f86f226253f35f8dd49003953035d6`).
Original MAS goal `19459dd37dab44748c88a69b469eccad` evidence/state
remain untouched; historical Git Delivery remains incomplete. No push/deploy.
Local integration is complete; no further action is authorized by this adoption.

> Status: `CANONICAL_CURRENT_PROJECT_MAP`

## Window37 W34 Read-only Model Integration — VERIFIED CANDIDATE

Main72d35e7/W34 source82cd49c diverged atc57ac3d. Isolated `integration/v7-window37-w34-models` ports only readonly loader and original scores function, plus byte-identical model/recipe/feature/replay evidence. No new Fit/GET, no old training/fetch entrypoints, W35 Recovery unchanged. W34 original161files+restore verified; necessary JSON model/scaler/recipe/serving-schema pins retained. [Integration](../../research/v7/window37-integration.json), [adoption](../../research/v7/window37-model-adoption.json). Independent Gate GO Critical0 Required0；2models tests＋15W35 regression PASS，512原分數exact Replay；Human已授權必要驗證後main ff-only，實際整合identity由W37 closure補記。
W36 remains separate protected branch b2bc041; no Task/runtime/artifact/budget mutation. Its limited tasks are independently active under HumanW36 only, not a Shadow authorization. W34 models OFFLINE_VALIDATED, no Production/StrictPIT admission. BACKUP_PENDING deferred by Human; no independent backup claim. W37 manual Shadow implementation only after integration, no schedule or waiting; current Nov2 night is not Preopen.

## Window35 Five-session Collector Closure / Recovery v0.4 — COMPLETE / STOP

W35 branch `research/v7-window35-recovery-v04`，base main `c57ac3d7822f229ae08849e56581ebb3c969441c`，UNMERGED / NOT_DEPLOYED。W19cf1與main分岔於87695，無合併/Runtime移植；main既有verified_record/QC/snapshot/restore可驗證既有scheduled records。W34僅exactGit82cd固定64ID只讀，無模型載入/修改/整合。
五日5/5runs、10/10sources、原10GET證據確認12031rows/11814basicPASS/217missing/Fatal0。完整Raw TWSE72 vs69exceptions、TPEx145 vs100，補48明細；121個market/security缺1/2/3/4/5天69/23/17/9/3。217全部OUT_OF_W34_TWSE64_UNIVERSE；cause仍UNKNOWN，已確認合法無價/停牌0、SourceFailure0，不因持續性或商品樣式推定。64股每day64有效OHLCV/兩Featureinput，共320，0RankingScores。
Task唯讀觀測2026-11-02T21:27:33+08：Ready/enabled/last19:00:01result0/NextRun=null（不據此稱停用），原End23:59:59觀測時未到，EXPIRING_PROTECTED_EVIDENCE。W19Task/政策/10GETbudget/135原檔hash均保留未改；不延長或重啟。
最小v7_data_recovery.py僅offline分類/決策/immutable claims/outcomes；resource/session整份CSV計數非per-stock，1content與候選2network retries分開，offline physical參考cap4，正式營運Retry參數未凍結。來源停止跨Session保留，wrongSession/sameContent/stillMissing停止，Backfill不冒充Forward。
主要Review原NO_GO/Required1保留；R1 failure-parent retry分叉已單輪修復與[Independent Closure](../../research/v7/window35-review-closure.json) GO Critical0／Required0。15tests PASS；同event冪等、每失敗parent一次、下一retry只能跟新failure。原replay保留，postrepair獨立[Replay](../../research/v7/window35-replay-closure.json) 重現12031/217與Synthetic修復/limit-stop，0GET/Fit/GPU/Taskchanges。
10個W19只讀replay、41offline檔9409116bytes/份restore，135原W19files未變；BACKUP_PENDING，同機非獨立備份。[Evidence](../../research/v7/window35-evidence.json)、[完整Missing](../../research/v7/window35-missing-distribution.json)、[Report](../../reports/v7_window35_collector_recovery.md)。
[Activation Candidate](../../research/v7/window35-activation-candidate.json) 僅PROPOSED：Normal19／Conditional07、Interactive限制/原W19期滿隔離/新限期Task/保存備份監測。真正live Normal/Recovery dispatcher仍未移植接合，不能直接啟用；下一Human最少批准W35整合、有界live adapter/驗收及限期操作設定、calendar/retry/update-evidence與獨立備份位置。
恢復：W35 worktree → Current_State → 唯一Backlog → window35-evidence.json → closure/report。無剩餘本輪READY，STOP；StrictSTOP/FormalAdmissionNOT_APPROVED/W34OFFLINE_PREPARATION_ONLY/W12SAFETY_HALTED。不得自整合W35/新GET/長期Task/Shadow/下一窗口。

## Window33 Recent Data & Initial Shadow Model — DATA PRESERVED / FIT BLOCKED / STOP

W32 fixed `8fa41869a2583f0b5db698db83eed98611f24a7e` 已由main67e2d25本機ff-only整合；Review GO/11Synthetic原檔及restore/13tests/治理核對。W33 branch `research/v7-window33-recent-shadow`，base同固定SHA，worktree `.worktrees/v7-window33-recent-shadow`；UNMERGED / NOT_DEPLOYED。GET前91d7cca固定W29同64檔/日期/guard。
64 FinMind price GET全200、0metadata/retry/token/fallback，7604573rawbytes；最小間隔5.001249s、最高known prior hour63<100，額度CLOSED/剩6不重用。43437rows，2024-01-02～2026-10-22、679OBSERVED sessions（242/243/194）；58zero、19union-relative missing，不補值或推定停牌。
[Data](../../research/v7/window33-data-evidence.json)：43319數值Feature候選，5d42720／10d42173數值Label候選；近期Calendar沒有合格官方/例外休市/時鐘證據，W07回CALENDAR_QUALIFICATION_UNKNOWN。全部新mask fitFalse；未因Vintage未知全面拒絕資料，也不把價格union升成qualified。8422於2025-11-17疑似單位不連續，事件原因UNKNOWN，沿用既有flags不修價。
[Compatibility](../../research/v7/window33-compatibility.json)：W16/W18同Oct22原始CSV；64檔OHLCV精確相等，官方current＋明示FinMind prior的兩Feature hybrid全相等；非連續兩官方Snapshots獨立parity。Feature order/shape2保持。
實際Training N0／Ridge0／LGBM0／Offline Scores NOT_RUN_DEPENDENCY_BLOCKED；沒有新模型、Prediction或RankIC。Prediction Replay N/A，不能冒充完整Shadow準備成功。W30Recipe與W27 safety不變；2024–2026 EXPOSED。
264files58910660bytes/份獨立程序Raw/Derived restore，68old rawpins未變；CPU28.817576＋60reserve<1800，artifact+restore117822087<1GiB。Private `.artifacts/window33-recent-shadow/`／`.artifacts/window33-restore/`，BACKUP_PENDING同機非獨立備份。
8targetedtests PASS；獨立Review GO（資料保存與阻塞收尾）Critical0／Required0，0輪review rework；見window33-review.json，不代表訓練Gate已通過。[Evidence](../../research/v7/window33-evidence.json)、[Report](../../reports/v7_window33_recent_shadow.md)。初期interpreter/import/probe shape失敗trace與CPU計帳保留，無額外GET或輸出覆寫。
下一Human：有界補足近期合格calendar/休市證據，或批准另具身份的觀測Calendar代理語義；不得只改qualification字串。解除Gate後另授權固定4fits及Offline驗收；真正PREOPEN另需模型/source/time/missing/recovery及操作授權。本輪STOP，不自動續期Collector或開下一窗口。
恢復：W33 worktree → Current_State → 唯一Backlog → window33-evidence.json；不得重跑已CLOSED source。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED無任何讀寫/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED，無Scheduler/GPU/Push/Deploy操作。

## Window32 Expanding Closure & Prospective Shadow — PREPARATION COMPLETE / STOP

W31 fixed `67e2d25a710273646d41c0ad861b4a993593ba4d` 已由main b64ac35本機ff-only整合；72原檔/restore、24views/24model identities、maturity及10tests核對。Refit−Train−Validation差額5d300/305/310、10d597/610/620均為TRAIN邊界排除後已成熟的舊labels，非Forward。未重Fit/重算舊RankIC，W31歷史Warning保留。
W32 branch `research/v7-window32-shadow-preparation`，base同固定SHA；worktree `.worktrees/v7-window32-shadow-preparation`，LOCAL_INTEGRATED_FIXED8fa4186 / NOT_DEPLOYED。
[Shadow Contract](../../research/v7/window32-shadow-contract.json) 為PROPOSED；正常目標07–08Asia/Taipei，持久發布嚴格早於合格Entry開盤才PREOPEN，晚到MISSED_PREOPEN/RESEARCH_ONLY。未凍結精確source freeze/wait/missing政策或啟用排程。
Synthetic-only薄入口重用W07時鐘/日曆、W09 strict JSON/lock/safe path及既有atomic-no-clobber；無真實模型載入、Fit或Inference CLI。Feature-only keys先保存，5d/10d成熟後才Join，Backfill不冒充Forward。13targetedtests PASS；11個Synthetic檔16123bytes/份同機restore；BACKUP_PENDING。
[Readiness](../../research/v7/window32-readiness.json)：已核immutable訓練資料僅2021–2023，不能宣稱2026FreshRefit就緒。W31兩Feature schema可重用，Daily CSV尚需units/security/market/feature parity及合格Universe/Calendar/events；V6的56/59維Inference不可直接沿用。
W19僅讀固定Git policy物件作參考，沒有Task/worktree/artifact/runtime存取或續期；五次單日snapshot不能取代多年訓練資料，實際Task狀態未觀測。W06/W12/W19均未整合。
[Evidence](../../research/v7/window32-evidence.json)、[Report](../../reports/v7_window32_shadow_preparation.md)、window32-review.json GO Critical0／Required0，0輪review rework。下一Human：初始資料/事件/Label准入與限額FreshFit、Shadow時間/missing/evaluation契約、模型pins、起訖/環境/儲存備份/監測及操作授權。無剩餘已授權READY；STOP，不產生真實排名或開下一窗口。
恢復：W32 worktree → Current_State → 唯一Backlog → window32-evidence.json；只讀檢視，不啟用任何Collector或Shadow。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED；InferenceRecoveryImplementation仍DEFERRED。

## Window31 First Expanding CPU Development — COMPLETE / STOP

W30 fixed `b64ac353833a9248121f12ebe61e1333decb2dbf` 已由main e83dfe6本機ff-only整合；38pins/restore及6mock-only tests通過。970 daily＝940 defined＋30 None；2910 paired＝2820 defined＋90 None；原模型351/379 split nodes、預測非退化，未重Fit。見 [Integration](../../research/v7/window31-integration.json)。
W31 branch `research/v7-window31-expanding`，base同固定SHA，worktree `.worktrees/v7-window31-expanding`；LOCAL_INTEGRATED_FIXED67e2d25 / NOT_DEPLOYED。首次Fit前 `ddcf5d835b43578ed9fc5567ee677d88e928e203` 固定 [Contract](../../research/v7/window31-contract.json)／[Prepared](../../research/v7/window31-prepared.json)。
三Origin2023-01-03／04-06／07-03，Forward54／59／63sessions。5d TRAIN14113/17559/21321，REFIT29229/32574/36276；10d TRAIN13711/17143/20890，REFIT28743/32046/35715。持有窗跨界/未成熟Label排除，額外purge/embargo0不是E530sessions。每Origin Validation Fit與Fresh Refit不同run/scaler/state；Validation診斷後refit，全部Forward評分延至三Origin完成，無Forward-feedback tuning。
Ridge12fits＋LGBM12fits已達本輪上限；L0±固定。453192完整Feature-only Predictions先保存後Label Join；7264daily/10896paired，含Validation明確None尾端不當0。Forward352個session×horizon全部可算，Daily N59–64，min40未降。
Forward Ridge5d O1/O2/O3 Mean .028606/.035161/.024427；10d −.012394/.008584/−.002463。LGBM5d .023592/.058135/.065496、10d −.021556/.026531/.000766。全部為EXPOSED DEVELOPMENT，不選winner/不宣稱獨立統計或收益。
10純數值/mock tests PASS；獨立程序Replay453192pred/7264daily/10896paired完全吻合，0追加Fit。72files91000410bytes/份同機Restore，310 prior W29/W30pins未變；CPU87.690630＋60reserve<1800，artifact+restore182002029<512MiB；BACKUP_PENDING。
初始導航authority_class缺漏使healthFAIL；prefit shell未set-e，未攔住O1。已修導航，保留原FAIL與修後WARNING證據，數值code/contract未變；不可宣稱所有preflight從未失敗。獨立Review GO Critical0／Required0，0輪review rework；見window31-review.json。無額外API/GPU/W19或V6操作。
[Run](../../research/v7/window31-run.json)、[Results](../../research/v7/window31-results.json)、[Evidence](../../research/v7/window31-evidence.json)、[Report](../../reports/v7_window31_expanding_cpu.md)。W31當輪main保持b64ac35、成果未合併；W32已獲Human授權ff-only整合至67e2d25，無其他未授權整合。
下一Human：正式Dataset/Universe與UNKNOWN事件政策、Decision-time/Label/maturity/purge、selection/refit/evaluation/statistics、Exposure/prospective評估、運算/獨立備份。無剩餘已授權READY；不得重跑已用完額度的O1/O2/O3。
恢復只讀入口：上述worktree → Current_State → 唯一Backlog → window31-evidence.json；W29/W30 raw/models/restore、W11環境依賴保留。StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED。STOP，不開下一窗口。

## Window30 Fixed Real-data LightGBM — COMPLETE / STOP

W29 fixed `e83dfe68e4bc73b59cb6390ec130ba967138e7f1` 已由main1d9a2a7本機ff-only整合；340pins/restore、4tests、ReviewGO核對。
W30 branch `research/v7-window30-real-lightgbm`，base同固定SHA，worktree `.worktrees/v7-window30-real-lightgbm`；LOCAL_INTEGRATED_FIXEDb64ac35 / NOT_DEPLOYED。
任何Fit前60e7ca7固定唯一Recipe/版本/native-lib SHA/feature order/seed/TRAIN view/planned ModelSpec IDs；weights SHA僅Fit後產生。
W11環境LightGBM4.6.0/NumPy2.0.2/SciPy1.16.3/sklearn1.6.1不變；L2/64trees/7leaves/depth3/lr.05/minleaf100/L2=1/maxbin63/seed0/thread1，無eval_set/earlystop/search。
W29 Cohort/Data/兩Features/Labels/Fold/support完全沿用；TRAIN14113/13711，最新exit2021-12-30；identity preprocessing，bin mapper只學TRAIN。
2個fresh LGBM fits各horizon一次，皆64trees，fit約.02777/.02791秒，modeltext52369/54849bytes；Ridge/API/GPU0。
60726完整Feature-only Predictions先immutable保存後Label Join；同W29 support/AVERAGE/min40，970groups940defined N58–64，30tail/maturity保留拒絕。
LGBM Mean/Median：2022/5d .018814/.021627、2022/10d −.012617/−.016793、2023/5d .021182/.026036、2023/10d −.013374/−.013470。
5d正但四組Mean均低於既有Ridge，10d兩年負；沒有改Recipe重Fit、選winner或批准Mamba2。2022/2023 EXPOSED，UNKNOWN actions/asof/vintage與Cohort bias仍保留。
6 mock-only tests PASS；獨立Replay60726pred/970daily/2910paired PASS，0Fit；1主要Review GO Critical0 Required0／0 review rework。
Reviewer不能只由stdout獨立重證shell timeout wrapper；已保存實際tool command/session/start/complete reference，未重跑Fit；OS CPU guard及實測fit保留。
11files34874071bytes/份同機restore、W29原299files hash未變；artifact+restore69749181<256MiB，CPU30.723458＋60reserve<1200；BACKUP_PENDING。
[Recipe](../../research/v7/window30-recipe.json)、[Prepared](../../research/v7/window30-prepared.json)、[Run](../../research/v7/window30-run.json)、
[Results](../../research/v7/window30-results.json)、[Evidence](../../research/v7/window30-evidence.json)、[Report](../../reports/v7_window30_fixed_lightgbm.md)、
[唯一Backlog](../../research/v7/research-program.md#window30-fixed-real-data-lightgbm-baseline)。
下一Human：Expanding origins/model-ready cutoffs/train/validation/forward/maturity/purge/selection/refit/evaluation與data/proxy/universe/events/exposure/CPU/storage上限。
固定CPU E2E基礎已驗，但W29 masks為固定split，各新Origin需重新核對maturity/purge，不能直接宣稱Expanding實驗已批准或認證；無剩餘已授權READY，STOP。
StrictSTOP/FormalAdmissionNOT_APPROVED/W19PROTECTED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED；不mergeW30/push/deploy/nextwindow。

## Window29 Expanded Cross-sectional Development — COMPLETE / STOP

W28 fixed `1d9a2a7a1ee57c455ba58250f63681efbbdd9050` 已由main39e2fed本機ff-only整合；35pins/restore、6tests、ReviewGO核對。
W29 branch `research/v7-window29-expanded-development`，base同固定SHA；worktree `.worktrees/v7-window29-expanded-development`；LOCAL_INTEGRATED_FIXEDe83dfe6 / NOT_DEPLOYED。
GET前746ca04固定68新Cohort（W24重疊0）：60pre2021 current issuers＋4periodlisting＋4delisted start/typeUNKNOWN；64檔進Prediction，仍HINDSIGHT/存活者偏差，非HistoricalUniverseVerified。
68真實FinMind price GET／metadata0／8387840rawbytes，全部HTTP200；單並發、實際起始間隔min5.001297秒，known-local100/hour（外部匿名用量UNKNOWN），unused預算CLOSED。
一次本機mutable counter修復：seq1保留不重送、seq2由Trace證明尚未GET；E1atomic_json只修counter，失敗CPU保守扣600秒未reset；不清除providerSTOP、不擴預算。
最小feature-only介面分離原W24未來Label gate；只讀當前/前一session和當下membership，Predictions先保存後Label Join；未來Label刪除/eligibility變動不能改PredictionUniverse。
48061raw rows／44899feature-only rows；940核心groups N58–64，min40未降；30tail/maturity groups/model N0保留拒絕證據，尾端Predictions仍保存。
原兩Features，L0正／反（反向post-W25 control）、Ridgealpha1兩次freshfit、TRAIN-only scaling，TRAIN N14113/13711。LGBM於GET前略過既有synthetic≤64/toy guard，沒有新Recipe或繞過。
5d Ridge Mean/Median：2022 .037333/.033014、2023 .047138/.050515，正向延續；10d2022 .012073/−.004079、2023 −.008580/−.002417，一致正向未延續。
不能宣稱跨Cohort泛化/優勢/正式統計成立：來源/Universe/support/protocol不同，2022/2023皆EXPOSED；UNKNOWN actions/suspension/asof/vintage保留。
182178predictions，3模型相同可評估Support；18targetedtests，獨立Replay Raw-derived feature/masks＋2910daily/2910paired PASS，0額外Fit。獨立Review GO，Critical0／Required0，1主要輪／0 review rework。
299files／74612661bytes同機restore；總artifact+restore149226490bytes<1GiB，CPU計帳647.740401秒（含failed600）＋60reserve<3600；BACKUP_PENDING。
[Plan](../../research/v7/window29-plan.json)、[Cohort](../../research/v7/window29-cohort.json)、[Coverage](../../research/v7/window29-coverage.json)、
[Run](../../research/v7/window29-run.json)、[Results](../../research/v7/window29-results.json)、[Evidence](../../research/v7/window29-evidence.json)、[Report](../../reports/v7_window29_expanded_development.md)、
[唯一Backlog](../../research/v7/research-program.md#window29-expanded-cross-sectional-ranking-development)。
下一Human：補強ordinary/historical membership與事件未知政策，事先固定新資料/新時間或prospective的5d primary/10dsecondary＋L0±/Ridge研究與精確資料/CPU/storage/backup上限；不因本結果調參或選勝者。
本輪無剩餘已授權READY，STOP；StrictSTOP/FormalAdmissionNOT_APPROVED/W12SAFETY_HALTED/E1BLOCKED/H2NOT_APPROVED/W19完全隔離。

## Window28 Clean Training Refit & Temporal Causality — COMPLETE / STOP

W27 fixed `39e2fedeb550a08d00f4abddc9a6caa7b1d570f8` 已本機 ff-only 整合（main前367005d）；35pins＋原計算source/restore、14tests、ReviewGO核對。
W28 branch `research/v7-window28-clean-refit`，base同固定SHA，worktree `.worktrees/v7-window28-clean-refit`；LOCAL_INTEGRATED_FIXED1d9a2a7 / NOT_DEPLOYED。
Plan/code/prefit先commit `ec0b072` 再執行；原W24兩Features／Ridgealpha1／W27Policy與Mask不變，2021各horizon只fresh fit一次，總2fits，0resume/search。
5d TRAIN 7338→7164、10d 7188→6841；mean/std僅在clean TRAIN重估並逐值驗證。2022/2023完全不參與Fit/Scaling。
同一W27 Eval support、940日期×horizon、N>=20未降：Ridge舊→新 2022/5d .033873→.034682、2022/10d .045471→.044561、
2023/5d .080008→.080407、2023/10d .053166→.054248；2022/2023皆EXPOSED Development，非模型選擇或獨立效能證明。
18個future有效OHLCV數值擾動案例，Features/固定模型Predictions exact-equal、當日positive control有效，Raw不改。
只驗數值因果性：既有feature wrapper也有未來Label資格gate，不宣稱row eligibility為as-of線上接口，亦非歷史publication/vintage認證。
原July1之2321不在W24資料；任何Fit前依原Mask第一共同合格日固定2022-07-18，不改Mask或樣本池。
6targetedtests；獨立Replay1880daily/940paired/29143newRidge scores/18cases PASS，Replay0Fit。獨立Review GO，Critical0／Required0，1輪主要審查／0 review rework。
227 prior private source identities核對；W28 11files／28170524bytes同機restore，BACKUP_PENDING，原W24/W27 models/predictions/metrics均保留。
[Plan](../../research/v7/window28-plan.json)、[Run](../../research/v7/window28-run.json)、[Results](../../research/v7/window28-results.json)、
[Causality](../../research/v7/window28-causality.json)、[Evidence](../../research/v7/window28-evidence.json)、[Report](../../reports/v7_window28_clean_refit.md)、
[唯一Backlog](../../research/v7/research-program.md#window28-clean-training-refit-and-temporal-causality)。
UNKNOWN actions/suspension/asof/vintage、hindsight/universe/已曝光仍保留；不宣稱完整污染已去除或投資收益。
下一Human：限定新資料/universe/事件未知政策及Label用途，凍結Exposure-aware Study/Fold/Recipe/evaluation與資料/CPU/儲存/備份上限；無剩餘已授權READY，STOP。
0 API/GPU/Colab/Bootstrap/LOSO/V6/W19操作。StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED。

## Window27 Pragmatic Label Safety & Frozen-Prediction Sensitivity — COMPLETE / STOP

W26 fixed `367005dfaba0988fca5e294d9b3e281408bbfce8` 已本機 ff-only 整合（main 前27055f7）；38 pins／3 tests／獨立 Closure GO。
W27 branch `research/v7-window27-label-safety`，base367005d，worktree `.worktrees/v7-window27-label-safety`；LOCAL_INTEGRATED_FIXED39e2fed / NOT_DEPLOYED。
Policy `W27_RAW_PRICE_DIAGNOSTIC_DEVELOPMENT_ONLY_v1`／Predicate／Mask 先在 `e77f32d` commit 固定，再重評既有 W24 預測。
原44266→新43148樣本；排除1118（Training521僅診斷，evaluation597），disjoint primary：confirmed event3／missing33／zero1082。
UNKNOWN event/suspension/asof 對所有樣本保留，不升格 CLEAN；新 Mask 不修改原 W24 mask，沒有 refit 或消除原模型 Training 風險。
四組原940日期×horizon仍全可計算；N>=20未降，Ridge均值：2022/5d .037016→.033873、2022/10d .043930→.045471、
2023/5d .080310→.080008、2023/10d .054056→.053166。這是 post-hoc development sensitivity，非正式模型選擇／獨立驗證。
2023/5d排除68筆持有路徑零價，沒有本輪已確認減資跨界；不得用三筆10d事件解釋5d。無 event-only 追加評分。
14 targeted tests PASS；獨立 Replay1880 daily／940 paired，原44266 Mask 由 raw 重建吻合；W24/W25/W26 218 prior byte pins 未變。
一次 import namespace 修正在任何 metric 前；Replay int-key JSON 比較問題以 canonical hash 修復，原計算source留存、calculate AST相同，metrics不覆寫。
獨立 Review GO，Critical0／Required0，1輪主要審查／0輪 review rework；新成果留候選。0 API／fit/refit／new prediction／Bootstrap／LOSO／GPU／V6或W19修改。
Private `.artifacts/window27-label-safety/` 與同機 `.artifacts/window27-restore/`；BACKUP_PENDING，不宣稱獨立備份。
[Policy](../../research/v7/window27-policy.json)、[Mask](../../research/v7/window27-mask-manifest.json)、[Results](../../research/v7/window27-results.json)、
[Report](../../reports/v7_window27_label_safety.md)、[Evidence](../../research/v7/window27-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window27-pragmatic-label-safety-and-frozen-prediction-sensitivity)。
下一 Human：更大研究 Universe/事件覆蓋/未知政策/Label路線、Fold/maturity/selection/refit/evaluation/exposure及精確資料/CPU/保存上限；無剩餘已授權 READY，STOP。
StrictSTOP／FormalAdmissionNOT_APPROVED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED；W19完全隔離。

## Window26 Label Integrity & Next Study — COMPLETE / STOP

W25 固定候選 `27055f7ec34cd4aa6d47f67bd89e5b6616244bc8` 已由 main `90b1327` 本機 ff-only 整合；
4 focused tests、12 private artifacts／21 deliverable pins 與 Review GO 已核對。W26 base 等於該固定 SHA。
W26 branch `research/v7-window26-label-contract`，worktree `.worktrees/v7-window26-label-contract`；LOCAL_INTEGRATED_FIXED367005d / NOT_DEPLOYED。
2321 已確認 68% 虧損減資；2023-02-17 市場參考價單位變更與 2022-12-08 會計減資基準日分離。
三筆既有 10d Label 算式／市場索引吻合，但跨股數單位變更，不可作未調整持有期報酬。
六個缺行情日期是市場交易日，非市場休市；原始官方停牌公告尚未核實，精確停牌區間保留 UNKNOWN。
其他公司行動覆蓋／historical availability／vintage 未認證；不重寫 W24／W25 歷史結果，不重新評分。
兩路 Label Policy、新 Study、L0 正反向對照與所有日期／閾值／資源額度均為 PROPOSED，待 Human 決定。
曝光帳本核對 E5／E1 machine contracts；2022／2023 已曝光，未列期間為 EXPOSURE_UNKNOWN，不自動成為獨立測試。
FinMind 非價格 1/6、行情 0；官方文件 loads/attempts 4/4；全部預算 CLOSED。0 fit/refit/new prediction/RankIC/GPU。
W26 private root `.artifacts/window26-label-integrity/`；15 files／4,785,564 bytes 獨立同機 Restore PASS；
W24 191＋W25 12 files 原 Hash 保持。BACKUP_PENDING，不宣稱正式備份或完整重現。
[Report](../../reports/v7_window26_label_integrity.md)、[Evidence](../../research/v7/window26-evidence.json)、
[事件矩陣](../../research/v7/window26-event-matrix.json)、[Study Candidate](../../research/v7/window26-next-study-candidate.json)、
[Exposure Ledger](../../research/v7/window26-exposure-ledger.json)、[唯一 Backlog](../../research/v7/research-program.md#window26-label-integrity-and-next-study-contract)。
Review／tests 狀態以 window26-evidence.json／window26-review.json 為準；本窗口無剩餘已授權工程，STOP。
下一 Human：Label 路線及未知／停牌／除息／下市政策、資料與 Universe 資格、Development-only 或 prospective、
Fold／Selection／Refit／評估／資源上限；本次未批准其執行。Strict STOP／E1 BLOCKED／H2 NOT APPROVED／W12 SAFETY_HALTED，W19 不操作。

## Window25 Ranking Robustness & Leakage Audit — complete / STOP

W24fixed `90b1327560f717727ad3fd53dfe86cb421c70f5b` 已LOCAL_INTEGRATED；main由87af76b ff-only，
26tests／191sourcefiles+33deliverablepins／GO與rawrestore驗證。W25 `.worktrees/v7-window25-robustness-audit`，
branch `research/v7-window25-robustness-audit`，base90b1327；LOCAL_INTEGRATED_BY_W26_FIXED27055f7/NOT_DEPLOYED。
單一read-onlyanalysis，0GET/fit/refit/newpredictions/GPU；既有W24policy/predictions/masks/data不改。
44266sourceformulas/timing檢查0確認跨期數值錯誤；TRAINhash/mean/std吻合，historicalavailability/maturity/vintage仍UNKNOWN。
四組RidgeMean/Median正，positive約59–62%；LOSO33×4=132cases/31020paired-days不翻負，非排除群組或TRAIN影響。
固定10/20circularblocks敏感度：2022relativecomparison脆弱/部分bounds跨0；2023direction一致但EXPOSED。
公司行動未知；2321約+224%10dproxy跨6缺行情session，來源/停牌/公司行動原因UNKNOWN，無發明過濾門檻或改寫結果。
分類ENGINEERING_VALID_WITH_EXPLORATORY_SIGNAL僅條件rawproxy；2022comparisonSIGNAL_FRAGILE，actions/asofINSUFFICIENT_EVIDENCE；非Humancorrectness/admissiongate。
Root.artifacts/window25-robustness-audit保存LOSO/casetrace/source；12files獨立同機restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window25_robustness_audit.md)、[matrix](../../research/v7/window25-leakage-matrix.json)、
[evidence](../../research/v7/window25-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window25-ranking-signal-robustness-and-leakage-audit)。
10focused/26mainpost/4independent新testsPASS；ReviewGO0CriticalRequired，0rework；16CBBbounds/264LOSOdailycase獨立核對。
下一Human：公司行動/停牌/adjustment與2321有界補證、universe/PIT或proxy政策、matchedbaseline方向與新未曝光資料/正式統計設計。
本輪無剩餘READY；STOP。StrictSTOP/FormalAdmissionNOT_APPROVED/E1BLOCKED/H2NOT_APPROVED/W12HALTED，W19完全不操作。

## Window24 Expanded Retrospective Pilot — complete / STOP

W23fixed `87af76b038fcbbda9e1e4b3a4349e693a3b1ba48` 已LOCAL_INTEGRATED；main由7027f03 ff-only，
26tests／55priorlocalfiles+25deliverables／GO與rawrestore重驗。W24 `.worktrees/v7-window24-expanded-pilot`，
branch `research/v7-window24-expanded-pilot`，base87af76b；LOCAL_INTEGRATED_BY_W25_FIXED90b1327/NOT_DEPLOYED。
下載前固定36檔HINDSIGHT_SELECTED cohort（currentcapital非historicalmarketcap）；31新FinMindGET+5舊raw，
0metadata/retry/redirect/token/fallback，unused14CLOSED。24511rows，QC24160VALID/346zero/5UNKNOWN；不fill或調整價混用。
TWSE2021–2023既有729calendar；52488三分mask，engineering46291／本輪exploratoryrawproxy44266／strict0。
33current/new股票入比較；3下市startUNKNOWN僅保留診斷，不能宣稱已消除存活者偏誤。
固定L0／Ridge各horizon一次（2realRidgefits），Train2021/Val2022/exposed2023；0search/selection/refit。
940core日期×horizon全部N27–33，預先固定min20未降；1880daily/940paired與scores獨立ReplayPASS。
公司行動/publication/vintage/可交易資格仍UNKNOWN，所有Label為獨立rawdiagnosticproxy，非收益或模型優劣證據。
本輪LIMITED_EXPLORATORY授權已行使；FormalResearchAdmission仍NOT_APPROVED／StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED。
Root.artifacts/window24-expanded-pilot/private原始與完整predictions，193files+5舊raw獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window24_expanded_pilot.md)、[evidence](../../research/v7/window24-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window24-expanded-retrospective-ranking-pilot)。
50focused/26mainpost/44independenttestsPASS；ReviewGO0CriticalRequired，0rework；191pins/restore驗證。
下一Human：正式raw/adjusted/actions政策與historicaluniverse/calendar/Label/eval/compute批准；本輪無剩餘READY，STOP。
W19工作樹/task/artifacts/budget/runtime不操作不干擾，不新增runtime觀測；不mergeW06/W12/W19或W24，不push/deploy。

## Window23 Historical Calendar & CPU Smoke — complete / STOP

W22 fixed `7027f03ebdb7b7b3bd90bc2bf5ce97a4c267c470` 已LOCAL_INTEGRATED，main由428a5fa ff-only；
23tests／78localfiles+23deliverablehashes／既有GO重驗。W23 `.worktrees/v7-window23-calendar-smoke`，
branch `research/v7-window23-calendar-smoke`，base7027f03；LOCAL_INTEGRATED_BY_W24_FIXED87af76b/NOT_DEPLOYED。
官方documentloads10已CLOSED（8directGET+2webopen），FinMind0。TWSE2019/20tableempty，2021/22schedule244/246，
2023schedule加官方0803臨時休市239，與observedreference相同；不提升strictPIT／完整TPEx日曆資格。
十檔日期membership僅listinginterval/knownboundary；8272不套用2024TPEx身份，停牌/可交易資格UNKNOWN保留。
W22原19728strictmask不改，Stricteligible0；A195價格幾何失敗／B7009獨立工程診斷／C12524其他阻塞。
Human本輪LIMITED_ENGINEERING_SMOKE_ONLY：五檔TWSE，Train範圍masked到2021、Validation2022、Forward-style2023；
固定L0／兩horizons各一次Ridge，0search/selection/refit；1880dailyrows，1864defined、16minN不足跳過，非模型性能結論。
公式/feature/fold/policy於fit前封存；rawreturn非可執行收益，actions/publication/vintage仍UNKNOWN，不是研究admission。
唯一rework修復JSONmap排序造成replaylist順序不同；原bundle/models/metrics及run-code留存，無重新fit，獨立ReplayPASS。
Root.artifacts/window23-calendar-smoke及同機window23-restore保存資料，56files獨立RestorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window23_calendar_smoke.md)、[evidence](../../research/v7/window23-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window23-historical-calendar-and-cpu-smoke)。
正式study仍PROPOSED/DataAdmissionNOT_APPROVED；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/W12HALTED；W19完全不操作。
58focused/23mainpost/26independenttestsPASS；獨立GO0CriticalRequired，唯一replay修復無refit。
下一Human集中決定正式historicalcalendar/universe/action/label/availability/evaluation/compute；無本輪剩餘工程READY，STOP。

## Window22 Continuous Historical Panel — complete / STOP

W21兩Required獨立複核CLOSED，15testsPASS/Critical0Required0；133localfiles驗Hash。
main由6a34520 ff-only至 `428a5fa5cb7b771db39a053f88c7046501399f7e`；沒有提前main文件commit。
W22 `.worktrees/v7-window22-historical-panel` / `research/v7-window22-historical-panel`，base428a5fa；LOCAL_INTEGRATED_BY_W23_FIXED7027f03/NOT_DEPLOYED。
10FinMindpriceGET、0metadata/diagnostic，均HTTP200，unused20CLOSED；10檔9864rows/1762931rawbytes，2019–2023。
6檔同1216reference日期（不是合格officialcalendar），9725structuralVALID/131UNVERIFIED/8zero，不fill或算return/RankIC。
官方18419/25036/11543CSV各1GET；TPExdelisted403立即停，共4resourceattempts；cataloguedocs另3GET。
6515TWSE上市20210120、8272TPEx上櫃20241023、2499終止上市20201110有officialCSV證據；逐日suspension/membership未知。
實際5d/10d eligible0：calendar/actions/availability/labelmaturity仍UNKNOWN，historicalstudyPROPOSED，不作admission。
Root.artifacts/window22-historical-panel與同機restore保存raw/panel/mask；15產物獨立程序restorePASS，BACKUP_PENDING。
[Report](../../reports/v7_window22_historical_panel.md)、[coverage](../../research/v7/window22-panel-summary.json)、
[唯一Backlog](../../research/v7/research-program.md#window22-continuous-historical-panel)。
23focusedtestsPASS；一次主要獨立ReviewGO/Critical0Required0，0rework；15份restore皆PASS。
下一步只建議精準calendar/lifecycle補證與Human統一data/label/eval/CPU授權；本輪完成後STOP。
W19工作樹/排程/Artifacts/budget不讀寫不觸發不整合；W12SAFETY_HALTED/StrictSTOP/E1BLOCKED/H2NOT_APPROVED不變。

## Window21 FinMind Historical Proof — complete / STOP

W20fixed `6a34520d8f9c1166172af3503cad531e23f5ecec` 已LOCAL_INTEGRATED，main由87695e4 ff-only；
11deliverable／8HTML+manifest／既有ReviewGO驗證。W21 research/v7-window21-finmind-proof，base同上，
`.worktrees/v7-window21-finmind-proof`，COMPLETE/STOP/LOCAL_INTEGRATED_BY_W22/NOT_DEPLOYED；固定428a5fa。
18anonymousHTTP200＝14price+3metadata+1repeat；4名額未用且CLOSED，無Token/retry/redirect/fallback。
6檔116price／48dates／2010-01-04..2025-08-01；100structuralVALID、3NO_PUBLISHED_PRICE（含有量零價）、
2documented-date市場例外、11UNVERIFIED；零ffill／PriceAdj混用／研究准入。2499有2010price及provider下市身份，近/後窗口空不推論漏資料。
Snapshot收到838854bytes，去重後836995rawbytes。root.artifacts/window21-finmind-proof獨立於worktree，
18份初次／修後獨立程序Restore與reviewerRestorePASS；原records/code保留。Backup仍同機BACKUP_PENDING。
VendorCalendar含future及已知20261023休市日期，禁止當actualcalendar；StockInfoupdated不等於active/listing，Universe/lifecycle仍BLOCKED。
來源僅CONDITIONAL_RETROSPECTIVE_CANDIDATE，多年份小窗口≠全市場連續性／StrictPIT／Executable證據。
W21原一次Review 2Required及單輪修復證據保留；W22已針對性獨立複核CLOSED/GO，見Window22integration記錄。
[Proof](../../reports/v7_window21_finmind_proof.md)、[evidence](../../research/v7/window21-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window21-finmind-historical-data-proof)。
下一Human：獨立少特徵retrospective dataset身份、合格calendar/lifecycle、missing/actions/Label/maturity與eval/CPUcompute批准；
本輪無剩餘READY。W19工作樹/排程/artifact/budget完全不操作、不整合；無新runtime觀測。
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/W12SAFETY_HALTED。

## Window20 Historical Source Decision Sprint — complete / STOP

Branch `research/v7-window20-historical-sources`，base/main `87695e4d5dc6e047f5ab39ca3cd8294b971bbc3a`；
`.worktrees/v7-window20-historical-sources`，UNMERGED/NOT_DEPLOYED。僅最多三類來源的兩單元文件調查，沒有行情API或工程實作。
FinMind TaiwanStockPrice為有條件主要候選；富邦HistoricalCandles為帳號/使用權BLOCKED的備用；
W14官方歷史route使用資格未釐清，未套用W15CSV授權。多年份日期皆DOCUMENTED宣稱，不是假定Coverage。
非商業/顯名/禁止raw再散布與匿名300/hr文件已核對；永久保存條件UNKNOWN，0價/興櫃open/調整價/下市生命週期須驗證。
建議下一次無Token資格Pilot至多22requests（18price+3metadata+1重取）、6檔角色×3小窗口，PROPOSED未批准。
不是5d/10d Label／Dataset／研究admission，沒有L0/Ridge/LightGBM訓練或RankIC。
W19固定cf1a84f的限期Task授權獨立保留；本輪不整合/改動/停用/觸發，無新增runtime觀測；以其固定來源及timestamped記錄為準。
[Decision brief](../../reports/v7_window20_source_decision.md)、[matrix](../../research/v7/window20-source-matrix.json)、
[evidence](../../research/v7/window20-evidence.json)、[唯一Backlog](../../research/v7/research-program.md#window20-historical-data-source-decision-sprint)。
一次獨立審查GO／Critical0／Required0，8官方HTML924253bytes／manifest及分類預算驗證PASS；無實作或套件重跑。
下一Human只需決定是否批准有限無帳號非商業來源資格驗證及保存範圍；富邦另需帳號/行情權限，付費並非首輪必要。
完成後STOP，沒有本輪剩餘READY；StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/FormalDataNOT_APPROVED/GPU_SAFETY_HALTED。

## Window18 手動 Forward Collector Pilot — complete / STOP

W17 fixed `96e8c0c4f2f7f78c508048e31f0f163d6c707128` 已 LOCAL_INTEGRATED/NOT_DEPLOYED，
main 由000b0b9 ff-only至固定候選；38treesclean／19deliverablehashes／9CPUtests確認；W06/W12未整合。
W18 `development/v7-window18-manual-collector`，base同上，`.worktrees/v7-window18-manual-collector`，
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED；finalHEAD為含本收尾commit。僅本輪manualpilot批准，daily仍NOT_APPROVED。
TWSE/TPEx exactlicensedCSV各1GET HTTP200，0retry/redirect/fallback；均2026-10-22：1383/1017rows，
137956/141480bytes，raw合計279436。原record/QC/code保留；TPEx初判27INVALID，修後離線27MISSING，TWSE15MISSING；不補零。
本機根目錄`.artifacts/window18-manual-pilot`保存raw+manifest+provenance，獨立程序初次／修後restore均Hash+rowsPASS。
同機restore不是獨立backup：BACKUP_PENDING。無Content-Length證據明確UNKNOWN，不宣稱來源宣告長度驗證。
Capture MANUAL_RESOURCE_CAPTURE_TIME_UNQUALIFIED，publicationUNKNOWN，非VERIFIED_FORWARD_VINTAGE／ResearchDataAdmission。
W17calendar僅PROVISIONAL TWSEboundedmatch；TPEx日曆未知，不以牆上日期判stale，不將商品全列入股票Universe。
新增manualCLI/CSVQC，共重用v1snapshot/W09safe+lock/W07clock/W17calendar；W15adapter仍Synthetic-only，沒有第二套平台。
25targetedtestsPASS（含原W17 9）、integration9PASS；一次review Critical0/Required3→唯一修復輪全部以RED/GREEN＋
獨立程序實際restore自行驗收。沒有第二次獨立GO，不把首次NO-GO改寫。W14/16/17原bytes與W12HALTED不變。
[Brief](../../reports/v7_window18_manual_pilot.md)、[evidence](../../research/v7/window18-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window18-limited-real-forward-collector-pilot)。
下次Human：daily runtime/scheduler、19:00候選與更新/日曆/QC、有限retry、獨立備份、監測通知與backfill/停止撤銷。
本輪GET額度已用完；無剩餘授權執行。StrictSTOP/E1BLOCKED/H2NOT_APPROVED/HistoricalPROPOSED/GPU_SAFETY_HALTED，
FairEval/PreNextOpen PROPOSED／InferenceRecovery DEFERRED。不得自動重跑/排程/訓練或開下一窗口。

## Window17 Forward Vintage 操作準備 — complete / STOP

W16 固定 `000b0b9c40448b11a41d05396dffc6e0f3ea1235` 已 LOCAL_INTEGRATED／NOT_DEPLOYED；
main 由 `27dcb36` ff-only 至固定候選，原有 W16 UNMERGED 記錄僅保留歷史意義。W06／W12 未整合。
W17 `research/v7-window17-forward-readiness`，base 同上，`.worktrees/v7-window17-forward-readiness`；
WINDOW_COMPLETE／STOP／UNMERGED／NOT_DEPLOYED，最終 HEAD 為含本收尾的 local commit。
W16 CSV 279436 bytes＋2 manifests＋request ledger 已獨立程序驗證並保留至根目錄
`.artifacts/window16-durable-evidence/`，原工作副本及 Restore 不動；同機同檔案系統，BACKUP_PENDING。
獨立Review一次GO／Critical0／Required0／0rework。32 次必要定點驗收（整合3＋最終29）、6日離線面板；新9測試僅工程／Synthetic，不是歷史 PIT 准入。
TWSE 官方10/23休市標題／ID／發布時刻及10/26年度補假確認；PDF正文未取得。
TPEx直接公告仍缺，不能以TWSE推定；10/27預定交易、非已實際交易。日曆修訂須有鏈與asof身份。
兩資源每1日更新不等於自動存取配額／SLA；定期服務條件與正式啟用仍 BLOCKED。
新元件限離線calendar／storage狀態guard，重用W07/14/15及Snapshot；沒有網路Collector或Calendar Platform。
[Activation brief](../../reports/v7_window17_activation_brief.md)、[evidence](../../research/v7/window17-evidence.json)、
[唯一Backlog](../../research/v7/research-program.md#window17-forward-vintage-operational-readiness)。
下一Human：先確認兩Resource定期服務條件及各市場有效日曆，再批准runtime／時間頻率／有限重試／
QC缺漏／儲存與獨立備份／監測通知／backfill及停止恢復撤銷。沒有剩餘本輪READY或啟用權限。
StrictSTOP／E1BLOCKED／H2NOT_APPROVED／HistoricalPROPOSED／CollectorNOT_APPROVED／GPU_SAFETY_HALTED；
FairEval／PreNextOpen PROPOSED、InferenceRecovery DEFERRED。不啟動下一窗口。

## Window16 Real Resource Proof — complete / STOP

W15fixed27dcb36b5431b70e8aa831edb6cb9234cb3a40b3 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from4da1bcf;25targetedCPUtestsPASS,source/review/hash/36cleantrees verified.
W14 raw/W15 evidencebytes remain originaltrees;not moved to main. Mainheld27dcb36;
W06/W12/othercandidates notmerged. OlderW15UNMERGED text historical for integration only.
W16 research/v7-window16-resource-proof,base27dcb36,.worktrees/v7-window16-resource-proof,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2units only,no newvalidator/platform. Two exactlicensedCSV GETs,
1each/0retry/redirect/fallback. BothHTTP200 UTF8,only2026-10-22/1session:
TWSE1383rows/1383securities/137956bytes,1368basicQCpass+15missing;
TPEx1017/1017/141480bytes,990basicQCpass+27missing. Nozero-fill orcalendar/universe/
pricecurrency/adjustment/tradabilityadmission. HistoricalTrainingCoverageBLOCKED;
no hiddenhistorysearch. Forwardrawresource CANDIDATE_ONLY, activationNOT_APPROVED.
IndependentPythonrestore2records/279436bytes/hash+rowsPASS;3focusedSnapshot testsPASS.
Raw+restore both localuntrackedcopies in W16tree,not certifiedbackup;tree removalrisk.
[Brief](../../reports/v7_window16_resource_proof.md),[evidence](../../research/v7/window16-evidence.json),
[sole queue](../../research/v7/research-program.md#window16-real-resource-coverage-and-capture-proof).
SingleindependentreviewGO0Critical/Required,0repair;28targetedtests, no fullsuite repeat.
Fallbackpolicy PROPOSED,whitelistempty,no provider switch/collector/fit/RankIC/GPU.
Old120files+4Frozenhashes unchanged. StrictSTOP/E1BLOCKED/H2NOT_APPROVED/
ExploratoryPROPOSED/CollectorNOT_APPROVED/GPU_SAFETY_HALTED. No nextwindow.

## Window15 Licensed Foundations / Dual-track — complete / STOP

W14fixed4da1bcfa81c8ff786de82d29ec461c0851d411d5 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only fromc78098f;pre/post150CPUtestsPASS,independent0Critical/Required.
W14original337221rawbytes/4snapshotfiles remain in W14tree, NOT moved to main.
Mainheld4da1bcf; W06/W12/othercandidates notmerged. OlderW14UNMERGED prose historical.

W15 research/v7-window15-dual-track,base4da1bcf,.worktrees/v7-window15-dual-track,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED. Humanconfirmed dualtrackdirection only.
U1 COMPLETE_REVIEWED:govdatasets11549/11371 exactCSV scopedOGDL1.0+attribution confirmed;
periodicservice/rate/activation conditional; oldhistoricalroutes permissionUNCLEAR.
3officialmetadata/licenseHTML1547650bytes locallyretained/independentverified;0market
payloadrequests. Licensing never implieshistoricalPIT/dataset/researchadmission.
U2 COMPLETE_REVIEWED:independentexploratoryStudy/Dataset/Feature drafts PROPOSED;
7synthetictests. Lag/dates/parameters/universe/calendar/labels/bytes/compute unapproved,
DATA_READY=false; oldsyntheticadapters require separatelyreviewed realdata scope.
U3 COMPLETE_REVIEWED:SYNTHETIC-onlythinarchive adapter reusesSnapshot/W14/W09;
18tests,2boundedrepairs,0remainingCritical/Required. Idempotentraw/newrevision,
currentQC/late vsfirstcapture separated; missingintent/binding/snapshotNOT_RESTORABLE,
no hiddenretryrepair. Initialdryrun+oldsource retained, finalofflinevalidation separate.
175jointCPUtestsPASS; allthree reviews0Critical/Required, U1/U2rework0. No livecollector/
fits/realRankIC/GPU. W11/W12/W14old92files+4Frozenhashes unchanged; no cleanup.
[Brief](../../reports/v7_window15_dual_track_brief.md),
[evidence](../../research/v7/window15-evidence.json),
[sole queue](../../research/v7/research-program.md#window15-licensed-dual-track-foundations).
NextHuman:boundedlicenseddata/coverage/restore qualification, newstudy/eval/admission
policy, separateTrackBoperation/storage/frequency/QC/notice authorization. No READY
execution remains in this completedgrant. FinalHEAD containingclosurecommit.
StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/InferenceRecoveryDEFERRED remain. No nextwindow activation.

## Window14 Evidence-first Data Pilot — complete / STOP

W13fixedc78098f3fdddb74e461bc54d700bde5baac0fcc1 LOCAL_INTEGRATED/NOT_DEPLOYED,
ff-only from8db5b911;pre/post140CPUtestsPASS,independentGO0Critical/Required;
12artifact/24source/88protectedhashes verified,34trees preflightclean. Mainheldc78098f;
W06/W12/othercandidates notmerged. OlderW13 UNMERGED text historical for integration only.
W14 research/v7-window14-data-admission,basec78098f,.worktrees/v7-window14-data-admission,
WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED;2boundedunits,each1reviewrepair.
U1COMPLETE_REVIEWED:2dailyofficialrequests TWSE20220901/TPEx20220902,337221bytes,
4selectedstocks,currentstructuralQC. Both proposedRETROSPECTIVE_ONLY;0asof/0proxy
qualified. Historicalpublication/revision/calendar/universe/labels and rights remain
unresolved. Contentclass != usagepermission != researchadmission;no moredatarequests.
U2COMPLETE_REVIEWED:PROPOSED forwardcontract+readonlyprototype reusesSnapshot/W07/W09;
10synthetictests,150jointCPUtestsPASS;both0Critical/Required. No collector/scheduler/
newplatform. PROTOTYPE_TIMELY neverauthenticates callerclock/pins oradmitshistoricaldata.
[Brief](../../reports/v7_window14_decision_brief.md),
[evidence](../../research/v7/window14-evidence.json),
[sole queue](../../research/v7/research-program.md#window14-evidence-first-admission-pilot).
NextHuman:permitdata route/rights, qualify asof/vintage or distinctlag-proxy/forward
proposal, then calendar/universe/labels/restore+research/eval/CPUcompute. No trueCPUstudy
ready or remainingauthorizedexecution. Raw4snapshotfiles retained locally;hash-only
restore impossible. ProtectedW11 83bytesfiles/W12 5diagnostics and4Frozenhashes unchanged.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED. FinalHEAD containingclosurecommit;no newwindow.

## Window13 Thermal / Admission Preparation — complete / STOP

Branch research/v7-window13-admission-preparation,base/main8db5b9111b98d4965a9f56f10a51c5157284b900,
.worktrees/v7-window13-admission-preparation,WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
Main unchanged; W11 localintegration verified (olderUNMERGED text historical).
W06 fixed96c60564/ProtocolAudit8601c737/Recoverye181f366/W12 fixed16e444f9 only
UNMERGED_REFERENCE_ONLY; no merge or authority promotion.

U1 COMPLETE_REVIEWED,sourcef26dbf5: HUMAN_CONFIRMED_DESIGN_DIRECTION/GPU_RETEST_DEFERRED;
RTX3060 local-first candidate,WindowsHost target87C reference/futurestop87C;
83C warningPROPOSED, not NVIDIA longtraining certification. OldW12 80C/HALTED/
32.106942s unchanged. GPUkernel/F-B/resume/AMP/fullmodelshape NOT_VERIFIED.
U2 COMPLETE_REVIEWED: [Human Decision Brief](../../reports/v7_window13_decision_brief.md),
DT1–4/EV1–6 options PROPOSED only. W05 audit reused0VERIFIED/21CONDITIONAL/27UNKNOWN;
2TPEx oddlot archives176096bytes currentlyverified, no historicalavailability/QC/
calendar/universe/labels/full48panel proven. No realRankIC or dataadmission.
ENGINEERING_READY scopedW07–11 tools; POLICY_REQUIRED/DATA_BLOCKED/RUNTIME_BLOCKED
separate. 140mainCPUtests+13fixedW06prototypePASS; bothreviewCritical0/Required0,
0rework. ProtectedW11 83files+W12 5diagnostics unchanged; no newsource/framework.
[Decision](../02_History/Decisions/20261024-window13-thermal-admission-preparation.json),
[evidence](../../research/v7/window13-evidence.json),
[sole queue](../../research/v7/research-program.md#window13-thermal-and-admission-preparation).
NextHuman: freeze timing/missing/source/evidenceclass, identify qualified bytes or
approve futurevintage collection, then evaluation/compute contract. No READY execution
in this completedgrant; new boundedauthorization required. Do notclearHALTED/retestGPU.
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED unchanged. FinalHEAD containingU2closurecommit.

## Window11 L2 & CPU Resume — complete / STOP

Window10 fixed `f25181637e40138c84dc3722292647d4325882fc` LOCAL_INTEGRATED/NOT_DEPLOYED
ff-only from `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9`;pre/post167testsPASS,
40hashes/independent26 GO,0Critical/Required. Main heldf251816; no main closure commit.
Older Window10 UNMERGED/STOP entries historical, superseded for integration only.

Window11 `development/v7-window11-l2-cpu-resume`,basef251816,
`.worktrees/v7-window11-l2-cpu-resume`, WINDOW_COMPLETE/STOP/UNMERGED/NOT_DEPLOYED.
U1 actualLightGBM4.6.0 fit/refit/forward/common-support/independentReplay PASS;
5officialwheels pinned in dedicated overlay, baseCPU Torch/NumPy/SciPy unchanged.
U2 reuses E1 actual train loop/store/adapter, safe weights_only=True probe codec.
Three new-process exits0/75/0,step4(epoch1batch1) afterwarmup ->12steps/4epochs;
23nontelemetryfields exactlyequal forLATEST/PREVIOUS/BEST. CPU_STEP_BOUNDARY_RESUMABLE
and CPU_MID_EPOCH_RESUMABLE only deterministic daily cursor/0workers/FP32/1CPUthread.
No separateepochinterrupt/GPU/AMP/shuffleworker/generalExactResume/full reproduction.
211jointtestsPASS. U1 environment setupcorrection1,reviewrework0;U2rework2,0remaining
Critical/Required. Source U1 f00bebab33cdeaa75b8401c6a6988dfaa3d3e0e7;U2 e28ac62ed95007e246185c5c2a134d17a5d0e08b.
Final HEAD is containing artifact/closure commit; no further engineering/Window12.

CPUbytes83files/~1.79MB retained `.artifacts/window11-cpu-resume-proof` under this
worktree; exactinventory in CPUevidence. Hash-only Git metadata cannot restore lost
checkpoints. Preserve this tree/basevenv/overlay/cache until separately reviewed.
Current runtime-bound L2 source snapshot uses U2commit and fresh run identity;
initialsourcebundle/report retained unchanged, requires original source to replay.
[CPU proof](../../research/v7/window11-cpu-resume-evidence.json),
[current L2](../../research/v7/window11-l2-runtime-evidence.json),
[evidence](../../research/v7/window11-evidence.json),
[sole Backlog](../../research/v7/research-program.md#window11-l2-and-cpu-resume).
StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. No realresearch/GPU/production/admission/othermerge.
Resume namedtree HEAD/clean/maps/Charter/evidence; new bounded authorization required.

## Window10 Classical Synthetic Foundation — complete / STOP

Window09 fixed `91fa82c2ac5e3b8f22109c55a73fa4e540a916b9` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `bdea30d877fa2c5d02308f343fa5591a35616aec`.
Pre/post141CPUtests PASS;39hashes/independent40tests GO,0Critical/Required;30trees
clean. Main held91fa82c, all new closure/development documents stay isolated.
Older Window09 UNMERGED/STOP entries superseded for integration only.

Window10 branch `development/v7-window10-classical-e2e`, base91fa82c, worktree
`.worktrees/v7-window10-classical-e2e`. UNMERGED/NOT_DEPLOYED; two units only,
<=2rework/unit. Human now permits tiny CPU synthetic fits, not real-data research.
U1 COMPLETE_REVIEWED:13tests/independent13PASS,1rework (full selection run hash and
null containers). L0 explicit toy factor; L1 NumPy train-only scaled Ridge; L2
adapter code present, dependency BLOCKED (lightgbm absent; no install performed).
U2 COMPLETE_REVIEWED:13tests/independent53 including40oldW09testsPASS,1rework
for fit/prediction/config lineage; validation+toyselection and forward pair replay
verified in independent CPU process. Final167joint testsPASS. U1 source 4ec22a4330ed6db7df0b3dafb4106cea820516e2;
U2+closure final HEAD is this containing commit. WINDOW_COMPLETE/STOP.
W09 v1 no-fit profile unchanged; v2 honest tiny-classical-fit profile added. No new
split/evaluator/platform. METADATA/EVALUATION replay proven; TRAINING_RESUMABLE/
FULL_REPRODUCTION_READY unproven. L2 actual fit remains dependency BLOCKED.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No Frozen E5/E1 or production mutation; no other branch
integration, GPU/Colab/search/real research. [Evidence](../../research/v7/window10-evidence.json),
[sole backlog](../../research/v7/research-program.md#window10-classical-synthetic-foundation).
Resume named branch HEAD/clean/maps/Charter/evidence/queue; final candidate is this
containing U2/closure commit. No READY unit remains in this grant; no Window11.

## Window09 Durable Artifacts & Offline Replay — complete / STOP

Window08 fixed `bdea30d877fa2c5d02308f343fa5591a35616aec` LOCAL_INTEGRATED /
NOT_DEPLOYED, ff-only from `1631020bce8223ac4337c7e97473a2a3d9a555f0`.
Pre/post101CPUtests PASS;19hashes/24cases reverified; independent34tests GO,
Critical0/Required0. Main remains bdea30d; no extra main closure commit.
Older Window08 UNMERGED/STOP entries are historical, superseded for this integration only.

Window09 isolated `development/v7-window09-artifact-replay`, base bdea30d, path
`.worktrees/v7-window09-artifact-replay`. UNMERGED / NOT_DEPLOYED. Human authorizes
only two CPU/SYNTHETIC units, <=2evidence-based rework/unit, then STOP.
U1 COMPLETE_REVIEWED:20focusedPASS;2bounded rework rounds corrected traversal,
metadata symlinks/JSON overflow, publication-null/sources containers. Immutable bundle profiles, required
artifacts, staging+diagnostics, fsync/flock/rename, idempotence and verified external
bytes references implemented. U2 COMPLETE_REVIEWED:20testsPASS,1rework for nested
containers and actual maturity recheck; independent final40testsPASS,0Critical/Required.
Final141joint tests;2schemas/3outputs validated. METADATA_VERIFIED and
EVALUATION_REPLAYABLE proven; TRAINING_RESUMABLE/FULL_REPRODUCTION_READY not proven.
WINDOW_COMPLETE/STOP. U1 source 79d6c556d1bb5648a40a02c3b6f0e628c6512b4a; final U1 container repair
and U2 source/evidence are in this containing closure commit.
No tracking platform, retention/GC policy or training/research admission added.

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED. No W06/recovery/audit integration or original matrix search.
[Evidence](../../research/v7/window09-evidence.json),
[backlog](../../research/v7/research-program.md#window09-durable-artifacts-and-offline-replay).
Resume named worktree cleanHEAD/maps/Charter/evidence/sole queue; final HEAD is
containing U2/closure commit. No remaining work in this grant; no Window10 grant.

## Window08 Common Ranking Evaluation — complete / STOP

Window07 fixed `1631020bce8223ac4337c7e97473a2a3d9a555f0` LOCAL_INTEGRATED /
NOT_DEPLOYED by ff-only from `465dca229fc1452aaf1b46572b06886a7d8a2c8f`.
Pre/post67tests PASS;10artifact/frozen hashes and16cases verified; independent24tests,
Critical0/Required0 GO. Main remains exactly1631020; no main closure-doc commit.
Window07 STOP/UNMERGED prose below is historical, superseded only for this integration.

Window08 branch `development/v7-window08-ranking-evaluation`, base1631020, worktree
`.worktrees/v7-window08-ranking-evaluation`. UNMERGED / NOT_DEPLOYED. Human grant
permits only common evaluator and paired comparison, synthetic CPU fixtures, max2
rework/unit. U1 COMPLETE_REVIEWED:826b8ddf798e752584d180181d24cc1db967e019;19tests/independent19PASS,1rework.
U2 COMPLETE_REVIEWED in containing closure commit;15tests,0rework. Both Critical0/
Required0;101joint CPUtests,24explicitcases,2schemas/3outputsPASS. WINDOW_COMPLETE/STOP.
Existing E5 average-tie RankIC, canonical_hash and W07 calendar/fold/role guards reused.
Policy remains named SYNTHETIC_NOT_ADMITTED; minN/weighting explicit, no statistics
or acceptance policy frozen. Metadata-first label gating, raw references and daily
rejection ledger are engineering evidence, not real historical PIT certification.

No W06/recovery/audit branch integrated. Strict Phase0 STOP; E1 Frozen Matrix BLOCKED;
H2 NOT_APPROVED; Fair Evaluation and Pre-next-open PROPOSED; Inference Recovery DEFERRED.
[Evidence](../../research/v7/window08-evidence.json),
[unique backlog](../../research/v7/research-program.md#window08-common-ranking-evaluation).
Final HEAD is this containing U2/closure commit. Resume by HEAD/clean, this entry,
Authority Map, Charter, evidence and backlog; a new bounded grant is required.
No automatic Window09, real-data evaluation, training or main merge of Window08.

## Window07 temporal/isolation foundation — completed / STOP

Human approved two CPU-only synthetic engineering units, no formal study config.
Branch `development/v7-window07-temporal-isolation`, worktree `.worktrees/v7-window07-temporal-isolation`,
base/main `465dca229fc1452aaf1b46572b06886a7d8a2c8f`. UNMERGED / NOT_DEPLOYED.
U1 commit5d736e9:13focused/independent13PASS,1owner-binding repair.
U2 complete:11focused/independent11PASS,
1source-hash repair. Final67related tests/16expectedcasesPASS;0remainingCritical/Required.
WINDOW_COMPLETE / STOP. No further work under this grant. Containing Unit2/closure
commit is final checkpoint. Resume via cleanHEAD here/maps/Charter/uniquequeue;
new bounded authorization required. CommonEvaluator synthetic foundation can be
next scope, not approved policy/data/Window08. [Evidence](../../research/v7/window07-evidence.json),
[decision](../02_History/Decisions/20261023-window07-temporal-isolation.json),
[unique backlog](../../research/v7/research-program.md#window07-temporal-fold--research-isolation).
No fixed validation/forward months/origins/30purge or metrics adopted; unknown policy/
data failclosed. No realPIT certification, no formalFeatureLabel/E5E1/admission change.
Window06/audit/recovery branches only pinned references, not merged or promoted.
StrictSTOP/E1BLOCKED/H2unapproved/FairEvaluationPROPOSED/PreNextOpenPROPOSED/
InferenceRecoveryDEFERRED persist. This grant does not authorize a Window08.

## Window06 — Window05 integrated; decision-time preparation authorized

Fixed `65462ec185783eedb2ba81b66cfa04783f86c6e2` integrated ff-only from main
`2ee4ca83e709af64be4dd63032ba4fb47903d769`; pre/post87 tests PASS,13source/output hashes
verified, independent13tests/0Critical/Required. LOCAL_INTEGRATED / NOT_DEPLOYED.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window06-integration.json).

Same Human grant authorizes isolated `research/v7-window06-decision-time` at
`.worktrees/v7-window06-decision-time` from this closure: VERSIONED PROPOSAL,
<=3source families/~10sessions diagnostic panel and synthetic eligibility tests.
New pre-next-open proposal does not replace frozen E5/E1. No formal cutoff/time,
feature/label/admission or execution policy selected. StrictSTOP/E1BLOCKED/H2unapproved/
FairEvaluationPROPOSED remain. No F2 repeat search/training/production changes.
Read local Current State, Authority Map, Charter and unique Backlog on that branch
for subsequent progress; all pending documents stay isolated. Older STOP entries
are historical for their windows; no indefinite authorization.

## Window05 Feature Integrity and Data Research — complete / STOP

Human authorized F0/F1/F2 only: read-only source audit, synthetic candidate formulas,
and bounded monthly-revenue PIT feasibility. Branch `research/v7-window05-feature-integrity`,
worktree `.worktrees/v7-window05-feature-integrity`, base/main `2ee4ca83e709af64be4dd63032ba4fb47903d769`.
UNMERGED / NOT_DEPLOYED. Source-derived macro Date joins do not prove t-close
availability; missing source values may become defaults. F0 identified material
unquantified risks; no real factor/performance analysis may depend on these
unverified payloads. Historical conclusions are preserved; no result reclassification.

F0 source audit complete:48ordered, VERIFIED0/CONDITIONAL21/UNKNOWN27/INVALID0,
all empirical quality UNKNOWN; F0 commit9fbf1bd,4tests/independent4 PASS.
F1 completed PROPOSED shortlist/formulas,9tests/independent9 PASS,1timezone rework;
source commit1a6a452, real diagnostics BLOCKED. F2 complete NOT_CURRENTLY_FEASIBLE
for broad historical PIT recovery;4historical metadata blobs independently verified,
0newqualified rows,0rework. All independent reviews0Critical/Required;87related tests
PASS. Window complete: STOP / UNMERGED / NOT_DEPLOYED, no automatic continuation.

The containing F2/closure commit on this branch is the final resume checkpoint.
Verify HEAD/clean in the named worktree, then read this section, Authority Map,
`knowledge/GOVERNANCE.md` and the local unique Backlog's Window05 section.
[Backlog](../../research/v7/research-program.md#window05-feature-integrity-and-cost-effective-data-research),
[evidence](../../research/v7/window05-evidence.json). Main stays2ee4ca8; pending
integration documents stay here. No formal manifest/cutoff/label/admission changed.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; Fair Evaluation and all
new feature/timing/forward-vintage candidates PROPOSED. Human timing/data identity/
policy choices precede real-data diagnostics; a new bounded grant is required.
Earlier window execution pointers below are historical; this entry owns latest state.

## Window04 W04B-LOG integration closure — STOP

Human-approved fixed `97fb4c9d669f3aa237ef5545a0ef322a5cc2d6a9` integrated by local ff-only
from `31f7accc475b5b735684d5a06725a237d85d5693`. Engineering source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`;
no new merge commit. This containing main commit adds closure documentation only.
LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED / WINDOW_COMPLETE / STOP.
Pre/post274 tests each PASS; independent runtime6 PASS,0 Critical/Required.
Both source hashes match;23 existing worktrees clean before integration and preserved.
[Closure evidence](../../research/v7/window04-log-integration-closure.json).

Current continuation entry is WSL main: verify HEAD/clean, read this section,
Authority Map, `knowledge/GOVERNANCE.md` and the local unique
`research/v7/research-program.md`. Older branch-local/unmerged pointers below are
historical, superseded only for W04B-LOG local integration. No new work authorized.
No additional high-value READY item was established by the prior bounded discovery;
future concrete CPU defects can be considered under a new bounded grant.
Strict Phase0 STOP; E1 matrix BLOCKED; H2 NOT_APPROVED; research options PROPOSED.

## Window04 Stage B — completed / STOP

Branch `development/v7-window04-cpu-continuation`, base/main `31f7accc475b5b735684d5a06725a237d85d5693`,
worktree `.worktrees/v7-window04-cpu-continuation`. UNMERGED / NOT_DEPLOYED.
WINDOW_COMPLETE / STOP: W04B-LOG source `27b9b41b01a44d3459ab66c6a0ea83395fc2d30a`; final274 tests PASS.
No additional high-value independent READY unit established. A new bounded grant
is required for further development; this grant is exhausted. Resume by verifying
HEAD/clean here, reading this section, Charter and the unique backlog; do not
continue automatically. The containing docs-only closure commit is final checkpoint.
W04B-LOG fixes admitted E1 runtime reader-error propagation: log write/decode
failure reaches caller with original cause/path and stops a still-running child;
a failure observed after child exit also cannot return success. CPU fixtures only,
no Colab/Drive certification or training readiness claim. Focused6 tests PASS,
independent6 PASS,0 Critical/Required,0 review rework rounds (one initial fix).
[Unique backlog](../../research/v7/research-program.md#window04-stage-b--cpu-continuation)
records evaluated alternatives; [evidence](../../research/v7/window04-cpu-continuation-evidence.json).
Strict Phase0 STOP, E1 matrix BLOCKED, H2 NOT_APPROVED and proposals PROPOSED.
This section owns branch-local progress; earlier integration/continuation pointers
below are historical. Stage B source/closure stays here, main remains unchanged.

## Window04 integrated; bounded CPU continuation authorized

Human-approved fixed `98f270dd63f05831b2a42695d8005638fda2bbf5` was ff-only integrated
from main `c733a53f81ff7bd572739b5d75f8baabf1189976`. LOCAL_DEVELOPMENT_INTEGRATED /
NOT_DEPLOYED. Pre/post272 tests each PASS; independent metadata5 PASS,0 Critical/Required.
This containing main commit is integration closure documentation only.
[Evidence](../../research/v7/window04-main-integration.json).

The same Human grant authorizes Stage B, at most four valuable CPU engineering
units with at most two evidenced rework rounds each. Use isolated
`.worktrees/v7-window04-cpu-continuation`, branch `development/v7-window04-cpu-continuation`,
created from this closure; its `research/v7/research-program.md` owns subsequent
selection and completion state. All pending development/closure stays on that branch.
No Stage B merge/push/deploy/training. Strict Phase0 STOP, E1 matrix BLOCKED,
H2 NOT_APPROVED; comparison proposals remain PROPOSED. Earlier awaiting-approval /
no-StageB text below is historical and superseded solely by this explicit grant.

## Window04 Git reconciliation — current candidate

Human authorized isolated version reconciliation only. Branch
`integration/v7-window04-reconciliation`, worktree `.worktrees/v7-window04-reconciliation`,
base/main `c733a53f81ff7bd572739b5d75f8baabf1189976`; second parent fixed
`489a127cf951f93f7bfc873e9333f05113f5b8e9`. Both histories are retained without rewrite.
Original STOP audit `3330427a1d774d30e65216ddab89d668d97c187c` remains separate evidence,
not a third merge parent. This section supersedes older continuation pointers below;
they describe prior checkpoints, not the current working location.

Status: GO_CANDIDATE_ONLY / AWAIT_HUMAN / UNMERGED_IN_MAIN / NOT_DEPLOYED.
Main is unchanged. No Stage B work or research planning is authorized here.
Merge commit `225efa9bb13b67bd5258faba79a47666c8906148` retains exact two parents above.
272 joint tests PASS; independent metadata5 PASS,0 Critical/Required; no unresolved
conflicts. [Integration evidence](../../research/v7/window04-reconciliation-evidence.json). The containing docs-only
closure commit is the final candidate SHA; main and original candidate are ancestors.
Resume by verifying branch/HEAD/clean here, reading this section and the evidence,
then await Human approval of that fixed SHA. Do not start Stage B.
Strict Phase 0 STOP; E1 Frozen Matrix BLOCKED; H2 NOT APPROVED; comparison options
remain PROPOSED. Read the local unique `research/v7/research-program.md` first
for reconciliation checkpoint; original preparation artifacts are unchanged.

## Fair evaluation preparation complete — fixed continuation pointer

Window02/03 source is integrated in local main at approvedeba6b649; this main
commit adds navigation only. New Window04 work remains UNMERGED/NOT_DEPLOYED on
`development/v7-fair-evaluation-decision@489a127cf951f93f7bfc873e9333f05113f5b8e9`, base`f31fcbba4d82d260915d7b640d80abc9c9ba8fc1`.
Worktree `/home/frank/projects/MarketMamba/.worktrees/v7-fair-evaluation-decision`.
WINDOW_COMPLETE / STOP. Readiness tool2a9de32 and decision package489a127 completed;
64 related non-training tests PASS,15 source hashes/cost arithmetic verified,
independent reviews0 Critical/Required. New comparison options remain PROPOSED;
no dates/thresholds/training/H2 approved. Original matrix BLOCKED, Strict Phase0 STOP.

Latest unique backlog is that pinned commit's `research/v7/research-program.md`
Window04 section; report `reports/v7_fair_evaluation_decision_package.md`, evidence
`research/v7/fair-evaluation-preparation-evidence.json`. New audit is not merged
into main. To resume, cd named worktree, verify HEAD/clean, read Current State,
Authority Map and that queue. No full historical rediscovery or repeated matrix
search. New bounded execution needs Human grant; no background continuation.
This pointer supersedes earlier Window04 authorized/READY text below.

## Fair evaluation preparation — completed checkpoint

Isolated `development/v7-fair-evaluation-decision` from integrated main closuref31fcbb.
UNMERGED/NOT_DEPLOYED for new work only. Existing-contract read-only E1 metadata
audit completed,5tests/independent5 PASS,0 Required; always execution_ready=false.
Decision package completed:15 source hashes, exposure ledger, fit/cost formulas and
Colab stop conditions independently verified;0 Critical/Required.64 related tests
PASS. WINDOW_COMPLETE/STOP, all new options PROPOSED. New execution requires
Human bounded grant. Resume at this branch via main pinned closure pointer.
[Unique backlog](../../research/v7/research-program.md#fair-evaluation-preparation-window-04),
[evidence](../../research/v7/fair-evaluation-preparation-evidence.json).

## Window02/03 integrated; fair evaluation preparation authorized

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved fixedeba6b649;
main ff-only9956446→eba6b649, pre/post267 tests each PASS, independent0 Critical/
Required.19 worktrees clean before merge;23 source/2 parser/1 report/7 result hashes
verified, protected contracts/data/models/lock unchanged. Fixed candidates retained.
[Integration evidence](../../research/v7/window0203-integration-20261021.json),
[Human scope](../02_History/Decisions/20261021-engineering-integration-fair-preparation.json).
This supersedes UNMERGED/no-main-merge text below only for admitted Window02/03.
No deployment, policy or research classification change. StrictSTOP/E1BLOCKED/H2
unapproved persist. A separate bounded CPU preparation window is authorized;
its current queue is [the unique program](../../research/v7/research-program.md).

## Data reliability / evaluation preparation window03 — 2026-10-21

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED. Current Human grant superseded the
Window02 STOP solely for this separate window. Main remains9956446; candidate
`development/v7-data-evaluation-preparation` joins main and fixed Window02f34f9f3
in isolation. Five navigation conflicts retain Window02 content, with this new
checkpoint taking precedence; main's previous pointer remains in parent history.
[Decision](../02_History/Decisions/20261021-data-evaluation-window03.json),
[unique backlog](../../research/v7/research-program.md#data-evaluation-window-03),
[evidence](../../research/v7/data-evaluation-window03-evidence.json).
Stage1 isolated integration GO at9bb4247 (259 tests, independent0 Required).
Parser complete: TWSE response date binding and TPEx unique header mapping,
8 tests + independent8 PASS; MOPS source/policy unchanged. No provider certification.
Fairness/readiness completed:12 source identities,7 result hashes,2seed identity
chains; independent0 Required. New designs PROPOSED; NOT_RUN_READY without matrix.
Runtime correction: candidate Data/contains3 inherited tracked historical files,
unchanged and parquet contents unread; no processed_v6 mount. Do not reuse older
Data-absent assumption. Final267 tests PASS, health WARNING/no FAIL. Main stays9956446; next session starts
at `.worktrees/v7-data-evaluation-preparation`, reads this queue/evidence and waits
for a new bounded grant. No main merge authorized. Strict Phase0 STOP/E1 matrix BLOCKED unchanged.

## Observation Metadata

| Field | Value |
|---|---|
| `observed_at` | `2026-09-22T13:25:21+08:00` for this runtime reconciliation; individual observations retain their own timestamps |
| `owner` | Human + project agents |
| Repository context | WSL development `main@3b0df0f` at observation; Windows operational `main@8b82a68`; research authority remains branch-qualified |
| State updated | `2026-10-21` engineering/decision update; Scheduler observation remains `2026-09-22T13:21:35+08:00`, not refreshed |
| Source basis | Reviewed K0/contracts plus [new Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json) and [operational path observation](../runtime-observations/operational-path-20260922T132521+0800.json) |
| Human review | Prior research/Phase A scope accepted `2026-09-20`; the new 2026-09-22 runtime observations are agent-observed and have not received separate human review |
| Runtime freshness | External scheduler/deployment observations expire after 7 days and become `STALE_OBSERVATION`; they are retained until re-observed |
| Repository freshness | Contracts do not expire by time; commit identity or explicit supersession changes them |

Semantic change owners must update this file in the same change. K0 remains discovery evidence. `CANONICAL_CURRENT_PROJECT_MAP` means current accepted navigation/state authority, subject to semantic updates and explicit supersession; it does not mean immutable or permanently final.

## Cross-domain window02 final checkpoint — STOP

Isolated `development/v7-cross-domain-window-02` from main closurec1179fe,
UNMERGED/NOT_DEPLOYED. W02-DATA/PREP/EVAL/MODEL completed; this bounded Human grant is exhausted;
[unique backlog](../../research/v7/research-program.md#cross-domain-window-02)
owns acceptance and state. No production parser edits or training permitted.
W02-DATA completed7-case offline diagnostic with independent review0 Required;
TWSE/TPEx source fixes and MOPS policy are now explicit Human decision items.
W02-PREP completed frozen contract/matrix metadata guards;34 tests and independent14
PASS. Metadata-only, no training readiness/admission; original matrix remains BLOCKED.
W02-EVAL rejects missing seeds/nonfinite gate values;6 tests + independent6 PASS.
Acceptance thresholds unchanged; historical evidence not recomputed.
W02-MODEL completed source-level selection/evaluation reuse audit;11 pinned sources
independently verified. New research choices remain PROPOSED/Human-only.
Final259 tests PASS, four independent reviews0 Critical/Required. No additional
high-value READY item demonstrated; see queue for P1/P2 decisions and recovery.
[Evidence](../../research/v7/cross-domain-window02-evidence.json).

## Charter integrated and window 02 authorized — 2026-10-21

LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED. Human approved exact
`f7af348710467ac381b2a84e8d192670ced89932`; WSL main ff-only from
`6d74d7384caa5fa90b1b050ce507bfb8282f19df` to that SHA. This containing
closure commit is docs/evidence only. [Integration evidence](../../research/v7/charter-integration-20261021.json)
records pre/post254 tests each (PIT49, non-training E1 31, portfolio67, ops81,
fetch3+governance23), 6 source/4 authority hashes,75 protected blobs unchanged,
17 clean worktrees and independent0 Critical/Required. No tiny training or GPU probe run.
Window01's UNMERGED wording below is historical and superseded only for integration.

[New Human decision](../02_History/Decisions/20261021-charter-integration-window02.json)
authorizes one next isolated CPU window, maximum3–5 meaningful units. Current
selection/checkpoint remains the [unique program](../../research/v7/research-program.md).
Discover Data, Model Research and Experiment Preparation before optional system
work; do not change V6 production. StrictSTOP/E1BLOCKED/Human research gates persist.
The fixed candidate branch remains unchanged. No push/deploy/Windows/MAS activity.

## Autonomous charter first window — 2026-10-21

**WINDOW_COMPLETE / STOP_AFTER_FIRST_WINDOW / UNMERGED / NOT_DEPLOYED.**
Human's [Charter](../GOVERNANCE.md#autonomous-development-charter--2026-10-21)
is persisted in existing governance, and the
[single backlog/checkpoint](../../research/v7/research-program.md#autonomous-charter-window-01)
records the complete first window. Branch `development/v7-autonomous-charter`
starts at unchanged main `6d74d7384caa5fa90b1b050ce507bfb8282f19df`.
Five units: Charter `be7998d`, E1 split completeness `3f07f80`, research decision
preparation `5847afc`, checkpoint fallback `2c60415`, snapshot pre-publication
validation in the containing closure commit. No further work starts in this window.

[Evidence](../../research/v7/autonomous-window-01-evidence.json): E1 33 + PIT49 +
governance23 =105 PASS; independent reviews0 Critical/Required after checkpoint's
second/final repair; snapshot one round; four research sources verified. Source
and output hashes/JSON/syntax/diff checks pass; knowledge health WARNING, no FAIL.
CPU synthetic engineering and PROPOSED research preparation only; no historical
checkpoint payload/matrix/data reads, external network, new training, GPU or MAS.
Existing E1 unittest includes a one-step synthetic CPU smoke. Loader remains
trusted-local and unrestricted: no historical-pickle safety certification.

This supersedes older no-next-program or in-progress statements solely for this
completed window. H0 remains locally integrated; Strict Phase0 STOP, E1 Frozen
Matrix BLOCKED, H2 and production/research gates unchanged. Another session reads
Charter + this backlog + fixed branch evidence, verifies Git/runtime, and waits
for a new bounded window grant before executing. No merge/push/deploy authorized.
No confirmed unexecuted READY item remains in this window; do not infer global
absence of defects or repeat E1 searches without new evidence.

## Portfolio History H0 completed — 2026-10-21

**LOCAL_DEVELOPMENT_INTEGRATED / NOT_DEPLOYED.** Human accepted the limited
fixture Product Slice and [approved exact4a66928](../02_History/Decisions/20261021-portfolio-final-h0-approval.json).
WSL main fast-forwarded from `83a6a2e6f4a56db318ee07e7e737cc01de3ce48e` to
`4a66928aa4c9bd8a1ddf99a5b42866e9b22efb74`; subsequent containing commit is
**documentation/evidence only**, not a different product candidate. Pre/post CPU,
API/HTTP, frontend and real Chrome font/selection/error checks passed.
[Integration report](../../reports/v7_portfolio_h0_integration_closure.md),
[evidence](../../research/v7/portfolio-h0-integration-closure.json).
CPU01/02, PORT01–04, UX01–04, final UX and font correction are in local main.
This supersedes historical UNMERGED/H0-pending/UX-review-pending prose below for
these units only. Original reports, fixed branches and all16 worktrees are retained;
audit7ecb920 remains separate historical failed-preflight evidence, not merged.
No Required/Critical remains within this tested slice. Main is a valid source
baseline for future separately authorized CPU work; no new Program starts now.
Strict Phase0 STOP, E1 Frozen Matrix BLOCKED, H2 unapproved, research correctness
unapproved. No push/deploy/Windows operational change/training. STOP after closure.
The existing UI's static UNMERGED badge is historical candidate wording, not a live
Git integration indicator; canonical status above owns integration authority.

## Final H0 browser blocker resolved — 2026-10-21

Human [authorized only the late-font visibility repair](../02_History/Decisions/20261021-portfolio-font-blocker-authorization.json).
New branch `fix/v7-timeline-font-visibility` starts from exact `6c894794` product
candidate; `audit/v7-h0-preflight@7ecb920` is evidence only, NOT an ancestor.
True-font RED reproduced desktop3.5px/mobile2.5px clipping. Existing ResizeObserver
now observes all event rows as well as the fixed-height container; GREEN confirms
font-driven layout correction without page movement, focus loss or data change.
[Report](../../reports/v7_portfolio_font_visibility.md),
[evidence index](../../research/v7/portfolio-font-visibility-evidence.json).
Original Required finding RESOLVED within tested fixture scope; bounded preflight
engineering GO, pending Human approval of the new fixed SHA. Final candidate is the
containing commit, not the audit commit. Main remains `83a6a2e`; no merge occurred.
Human limited UX product acceptance remains in force; earlier pending-UX-review
language below is historical. Old H0 authorization does not extend to this new SHA.
UNMERGED / NOT_DEPLOYED; H2 unapproved, Strict Phase0 STOP, E1 Frozen Matrix BLOCKED.
STOP here: no further UX/research work authorized. Native Goal creation was rejected
because the previous H0 goal remains paused; no automatic continuation claim.

## Project Summary

**Final UX closure (2026-10-21):** Human accepted the dual-mode direction and
[authorized only two finishing changes](../02_History/Decisions/20261021-portfolio-ux-closure-decision.json).
On `development/v7-portfolio-ux`, baseline `1553a59`, timeline selection stays visible
inside its container, with explicit selected text and no page anchoring jumps;
event explanations use observable exact-state differences and conservative Chinese.
Both tasks COMPLETED_VERIFIED / UNMERGED / NOT_DEPLOYED. Final identity is the
containing closure commit (see `git log -1`); [report](../../reports/v7_portfolio_ux_closure.md)
and [evidence index](../../research/v7/portfolio-ux-closure-evidence.json).
Desktop/mobile general seven events, desktop advanced exact values and mobile advanced
basic smoke verified. No additional mobile advanced visual work. Stop for Human final
review; H0/H2 remain unapproved, Strict Phase 0 STOP, E1 BLOCKED. No further UX READY work.
Earlier UX/PORT evidence below is retained as historical acceptance, not rewritten.

**Completed UX refinement (2026-10-21):** Human approved UX-01–04 default dashboard /
advanced research view on isolated `development/v7-portfolio-ux` from fixed `5a2bc6f`.
[Decision](../02_History/Decisions/20261021-portfolio-ux-human-decision.json),
[skills/design/acceptance](../../reports/v7_portfolio_ux_acceptance.md).
UX-01–04 COMPLETED_VERIFIED within fixture UI scope: desktop1440/mobile390
Chrome155, both modes/all7events/exact values/keyboard PASS. Implementation
`764dfbf`; final evidence in containing UX04 commit. Stop for Human UX Product
Slice Review. Focused lint PASS; full frontend baseline58errors/3warnings unchanged.
[UX evidence](../../research/v7/portfolio-ux-evidence.json).
UNMERGED / NOT_DEPLOYED; existing PORT completion remains preserved.
No H0/H2, accounting, data or production authority change. Strict STOP persists.


**Strategic recovery / integration candidate (2026-10-21):** this Human-authorized
planning window is complete; H1 was subsequently approved for the fixture slice. `main@83a6a2e`
remains unchanged; candidate `integration/v7-cpu-roadmap` starts at fixed
`4c388de`. Five CPU fixes already form continuous ancestry; no cherry-picks,
source changes or main merge. Joint 60 portfolio/health/audit, 47 PIT, 81 ops,
3 fetch-only, 23 governance and 6 historical-output fixtures PASS. Independent
review: GO for local development candidate only, no Critical/Required.
[Integration/objective/status evidence](../../reports/v7_strategic_roadmap_recovery.md)
and [hash/test index](../../research/v7/cpu-integration-candidate-20261021.json).

The **single active cross-domain development backlog/program** is the dated
section of [existing research program](../../research/v7/research-program.md#current-development-program--2026-10-21).
Prior CPU task files remain completed-window evidence. Product intent includes
trustworthy data, relative rankings, weight-based simulation and performance;
current engine is complete only as a deterministic component. PORT-01–04 now have explicit
[Human H1 fixture-only engineering approval](../02_History/Decisions/20261021-portfolio-history-h1.json).
Completed isolated branch `development/v7-portfolio-history` from `bebf19e`:
PORT-01 COMPLETED_VERIFIED (7 focused / 67 combined, independent review closed
one compatibility repair); PORT-02 COMPLETED_VERIFIED (4 new + 4 existing API
tests, independent live startup review); PORT-03 COMPLETED_VERIFIED (8 focused +
8 existing JS cases, build and real desktop/mobile/keyboard acceptance); PORT-04
COMPLETED_VERIFIED (actual HTTP/restart, browser corruption/recovery, 47 PIT,81 ops,
3 fetch-only,23 governance). All four units independently reviewed; stop for Human
Product Slice Review. No additional H1 implementation remains READY.
[Evidence/checkpoint](../../research/v7/portfolio-history-evidence.json).
UNMERGED / NOT_DEPLOYED; H0/H2 absent. Strict Phase 0 STOP and E1 matrix BLOCKED
persist. MAS Core Live unverified; no production/runtime authority changed.
Native Goal is V7-bound; automatic future continuation remains unverified.


**CPU-only window 02 (2026-10-21, WINDOW_COMPLETE):** reused window 01 lineage
from `7eb9cd9` on `development/v7-cpu-window-01`; main stays `83a6a2e`.
[Current backlog/checkpoint](../../tasks/v7-cpu-development-window-02.md) and
[evidence index](../../research/v7/cpu-window-02-evidence.json).
U1 `735ae6f` rejects normalized quote collisions before persistence; U2 (commit
containing this closure) rejects duplicate JSON object keys at every depth.
Both COMPLETED_VERIFIED: final 60 portfolio/process/health/audit, 47 PIT,
23 governance PASS; independent review no Critical/Required, one fix round each.
No remaining confirmed READY item in inspected backlog; stop after two units.
Goal API created a correctly bound V7 objective after observing no goal; future
automatic continuation unverified. No MAS/global settings touched. This supersedes
window 01 next-action status only. UNMERGED / NOT_DEPLOYED, no research approval;
strict Phase 0 STOP and E1 matrix BLOCKED remain unchanged. Extra Human decisions 0.

**CPU-only window 01 (2026-10-21):** Human authorized bounded autonomous
engineering without training or research changes. Isolated
`development/v7-cpu-window-01`, base `11ae71d`; main remains `83a6a2e`.
[Ranked backlog and acceptance](../../tasks/v7-cpu-development-window-01.md)
select three existing portfolio scheduling/journal recovery contract units.
All three complete: early-signal state preservation, pre-write semantic
validation and shared-lock reading. Final bounded suites: 53
portfolio/process/health/audit, 47 PIT, 23 governance PASS; independent review
has no Required/Critical findings. See the
[evidence index](../../research/v7/cpu-window-01-evidence.json).
Historical next READY at window 01 closure: normalized quote collisions;
now fixed by window 02 above. Window 01 stopped after three units.
E1 matrix remains BLOCKED; strict Phase 0 STOP and Phase 1–5 gates unchanged.
This branch is UNMERGED / NOT_DEPLOYED; no live MAS Core capability is assumed.

**E1 recovery checkpoint (2026-10-21, branch-local):**
`recovery/e1-frozen-evidence` verified six checkpoint byte hashes, two prediction
hashes, seven result/pointer/evaluation files and seven fixed diagnostic files.
Historical matrix `a590b280...947f62` was not found in the explicit local search;
same-name fixtures rejected, remote Drive not inspected.
**STOP_REPLAY_MATRIX_IDENTITY_UNAVAILABLE**. No checkpoint deserialization,
matrix generation, inference or training. See
[inventory/report](../../reports/v7_e1_evidence_recovery.md).
Human next step: locate the original matrix/manifest or exact archive path.
Do not repeat completed Phase A work or infer permanent matrix loss.

**Current integration closure (2026-10-21):** Human authorized fixed `89f9496`;
WSL development main fast-forwarded from `50c5b78` to that exact candidate.
Pre/post checks each passed 81 Operations, 47 PIT/Snapshot, 3 fetch-only,
38 Portfolio and 23 governance tests. See
[closure](../../reports/production_phase_a_integration_closure.md).
Phase A is `IN_DEVELOPMENT_MAIN / NOT_DEPLOYED`; old pending-merge language below
is historical. Remote tracking was not fetched; Windows operations are untouched.
E1 bounded evidence recovery is now authorized in isolation, not new training.

**2026-10-21 current Phase A decision:** Human explicitly approved Task 4 limited
engineering correctness at `1e119ef`, after revalidation of its index, 12 raw
SHA-256 records and 36 generation digests. Decision provenance:
[Human record](../02_History/Decisions/20261021-task4-human-approval.json).
This supersedes older pending-Task-4 observations below, not research or
production gates. New isolated `integration/phase-a-task5` starts at `63a040d`
and cherry-picks the 17 Phase A commits `9efed7b..1e119ef` with `-x` provenance.
Integrated Operations files match `1e119ef`; snapshot files match `63a040d`.
Joint regression: 75 ops, 47 PIT, 3 fetch-only, 18 governance PASS.
Task 5 limited offline engineering correctness is now **HUMAN_APPROVED** at
fixed `1c5250e67461456b7ca0b361f8da3d9cb3f22514`, after artifact/digest verification.
The [Task 5 Human decision](../02_History/Decisions/20261021-task5-human-approval.json)
retains the explicit scope and exclusions. Original acceptance/report artifacts
remain unchanged historical pre-decision evidence.
Final Operations 81, governance 19, non-training portfolio regression 38 PASS;
joint PIT 47 and fetch-only 3 PASS. See the
[Task 5 report](../../reports/production_phase_a_shadow_ledger.md) and
[evidence index](../../research/production-system/phase-a/task5-evidence-index.json).
Ten fixed replays, real-process restart/concurrency and measured overhead are
offline engineering evidence, not production certification. Final engineering
review recommends **GO for development candidate integration**; this is an Agent
assessment, not final Phase A Human acceptance or permission to merge.
See [final review and guarded integration procedure](../../reports/production_phase_a_integration_review.md).
The decision-validator change passes 23 governance tests (including rejection
of mutated, missing or corrupt Task 5 approval evidence).
The earlier review had no merge authority; the subsequent explicit integration
decision above authorized only local development main. No push/deployment/training.

**2026-10-20 bounded engineering recovery (branch-local):** base development
`main@50c5b785e8c373913c3301a056497eb955f36681`; candidate
`fix/v7-snapshot-immutability`, `UNMERGED / NOT_DEPLOYED`. One engineering unit
fixes immutable V7 source snapshot publication; it does not promote research
evidence or close any correctness gate. See the checkpoint below.

**Historical window 01 objectives (superseded by strategic checkpoint above):** CPU window 01 complete; next bounded engineering candidate
is journal quote-collision refusal. Separately await original E1 frozen
matrix/manifest location.
Bounded local inventory is complete; replay is stopped until identity and safe
inspection gates can be met. No new research hypothesis or training is authorized.

MarketMamba contains separate production-data, legacy model runtime, V7 correctness, research-training, historical-simulation, portfolio, delivery, and operations-control-plane subsystems. Production acquisition uses mutable provider data; V7 research authority uses versioned PIT/provenance evidence. The current corrected research baseline is `E5-PIT-Clean-v1`, with 48 features, no graph and no industry neutralization. Historical simulation is `PASS`, while strict trading correctness remains `STOP`. The observed production scheduler runs fetch-only; full V6.2 inference/portfolio/publication is `DISABLED`. E1 is a completed research experiment with `PASS_REFRESH_HYPOTHESIS`, not trading evidence. Phase A operations work is `IN_DEVELOPMENT_MAIN` and `NOT_DEPLOYED`; historical pending-integration paragraphs below are retained observations.

The `environments/v7-colab-2026.04/` identity files are preserved in development main. Its exact lock matches the E1 branch copy; the E1 frozen contract and run manifests still own experiment semantics and result provenance. The dated local verification record is not proof of a current Colab GPU runtime.

## Knowledge Layers

Canonical development/research checkout is WSL-native `/home/frank/projects/MarketMamba` in the distro named `Ubuntu-24.04` (`3b0df0f` at observation). Windows Scheduler invokes the separate D: checkout `D:\Desktop\work\ProjectForMe\MarketMamba` (`8b82a68`, clean) through the distro named `Ubuntu`, not the development checkout. Both distros report Ubuntu 24.04, but they are distinct registrations. The batch activates `mamba_env` with `/home/frank/miniconda3/etc/profile.d/conda.sh` and uses Python 3.11.15 at `/home/frank/miniconda3/envs/mamba_env/bin/python`. Windows operational `Data/processed_v6` is the D: physical directory; the development checkout symlink resolves there. The read-only interactive query shell in `Ubuntu` had `MARKETMAMBA_DATA_ROOT` unset; the scheduled process environment was not captured; the Windows checkout V6 config falls back to its own `Data/`. Colab remains separate. No path or runtime was changed.

- **Layer 1 — Current State:** this short file and its JSON projection hold active objectives, blockers, next actions, current runtime and branch/worktree context. [Authority Map](Authority_Map.md) determines which evidence wins; [Worktree Register](Worktree_Register.md) records checkout responsibility.
- **Layer 2 — Project Knowledge:** [Domain Notes](../01_Domains/), [GOVERNANCE.md](../GOVERNANCE.md), and stable architecture/decision contracts describe boundaries and durable rules.
- **Layer 3 — History / Evidence:** [History](../02_History/), research contracts/manifests/results, dated runtime observations, reports, Obsidian and Git history preserve completed or superseded work. Machine contracts, hashes, source and executable tests bind technical claims; historical prose does not override them.
- `AGENTS.md` and `CLAUDE.md` are the agent operating contract outside the three memory layers.

## Knowledge Governance State

| Topic | State | Authority / notes |
|---|---|---|
| Domain layer | `COMPLETE_CURRENT_SCOPE` | Twelve reviewed domain notes; machine contracts still bind technical truth. |
| Governance | `ACTIVE` | [`knowledge/GOVERNANCE.md`](../GOVERNANCE.md); impact, supersession, runtime-freshness and history protocols are defined. |
| Unified health | `AVAILABLE` | `tools/project_knowledge/knowledge_health.py`; PASS/WARNING are non-blocking, structural FAIL is blocking. |
| Runtime observation store | `ACTIVE_WITH_STALE_SEMANTICS` | `knowledge/runtime-observations/`; timestamps change only after an actual observation is recorded. |
| Generated navigation | `DETERMINISTIC` | `generate_indexes.py`; declared generated files are not hand-edited. |

## Production State

| Topic | State | Authority / observation |
|---|---|---|
| Active scheduler path | `ACTIVE`, observed 2026-09-22 | `MarketMamba_DataFetch` (Ready/enabled) → `wscript.exe` → `run_hidden.vbs` → D: `v62_daily.bat --fetch-only` → `wsl -d Ubuntu` → `mamba_env` → D: checkout `V6/run_v62_daily.py`. [Scheduler observation](../runtime-observations/scheduler-20260922T132135+0800.json). Last run 2026-09-21 22:30 +08:00, Task result 0; log exit 0 at 22:39:52. |
| Fetch-only behavior | `ACTIVE` | [`V6/scripts/v62_daily.bat`](../../V6/scripts/v62_daily.bat) and [`V6/run_v62_daily.py`](../../V6/run_v62_daily.py) stop before matrix, inference, portfolio and publication. |
| Operational data freshness | `10/10 MATCH_REFERENCE_DATE` | Ten `DAILY_SOURCES` have Parquet metadata max date 2026-09-21, matching `prices_raw` and last fetch-only log. This is date completeness, not data-quality certification; log recorded a stock-universe delta warning. [Operational observation](../runtime-observations/operational-path-20260922T132521+0800.json). |
| Full V6.2 schedule | `DISABLED` | `MarketMamba_V62` is disabled; last run 2026-08-25 22:15, last result 255. Code remains manually runnable. |
| Result publication | `DISABLED_AS_SCHEDULED`; manual state `UNKNOWN` | Full pipeline contains publication logic, but no active scheduled publication was observed. |
| API deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |
| Frontend deployment | `UNKNOWN` | Source exists; live deployment/traffic was not verified. |

Runtime observations are not permanent facts. The new Scheduler observation expires after `2026-09-29T13:21:35+08:00`; the prior 2026-09-20 observation remains preserved. A successful fetch-only run does not prove the next run, deployment, trading, or complete data validity.

## Data / Correctness State

| Topic | State | Authority |
|---|---|---|
| Reusable PIT/provenance source | `IN_MAIN_RESEARCH_ONLY` | Six fail-closed primitives and their focused tests were adopted from `fix/v7-historical-pit-reconstruction@e3bce91`; source and policy bytes match branch blobs, with one test's extra EOF blank line normalized. Blob map: `research/v7/source-adoption-register-20260922.json`. Frozen experiment evidence/readiness remains branch-qualified. |
| Historical-simulation readiness | `PASS` | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json` |
| Strict Phase 0 | `STOP` | Same readiness evidence and corrected Phase 0 audit; it is intentionally separate from historical-simulation readiness. |
| Verified executable labels | `5d=0`, `10d=0` | Corrected E5/E1 machine contracts. |
| Corrected feature contract | `48 FEATURES`, ordered and manifest-pinned | `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json` |
| Graph | `DISABLED`, zero edges, fusion branch absent | Completed corrected E5 contract. |
| Industry neutralization | `DISABLED` | Corrected baseline and E1 contracts. |
| Historical labels | `RESEARCH_ONLY`, evidence class `HISTORICAL_SIMULATION_PROXY`, policy `P0` | Never call these `VERIFIED_EXECUTABLE` or tradability evidence. |

## Research State

| Topic | State | Authority / notes |
|---|---|---|
| Corrected baseline | `CURRENT_REFERENCE`: `E5-PIT-Clean-v1` | Corrected baseline manifests and completed incumbent contract. |
| Corrected E5 | `COMPLETED_PARTIAL_ENSEMBLE` | Seeds 17 and 29 completed. Architecture authority is `d_model=64`, `d_state=32`; the old 32/8 draft is superseded. |
| Formal seeds | `17`, `29` | Used by diagnostic conclusions. |
| Seed 43 | `INCOMPLETE_EXCLUDED_FROM_FORMAL_CONCLUSIONS` | Pointer metadata referenced `best_epoch=8`, but pointer-named best/latest/previous generations are missing. The surviving orphan checkpoint is epoch 15, step 39,915, `terminal=false`, contract/matrix matched, and is convergence evidence only. It is not a formal seed and must not be promoted or directly resumed. |
| Extended diagnostic | `STOP_FOR_TRADING_CONTINUE_RESEARCH` | Broad 2026 degradation and near-redundant heads; authority is diagnostic v2. |
| E1 rolling-origin refresh | `COMPLETED_RESEARCH_EXPERIMENT`; `PASS_REFRESH_HYPOTHESIS` | Formal seeds 17/29; `eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json`. The specification retains its pre-run wording as history. Historical-simulation proxy only; strict Phase 0 `STOP`. |
| E1 post-run diagnostic | `MONTH_SENSITIVITY_FOUND`; `EPOCH_DEGRADATION_INCONCLUSIVE`; `REGIME_DEPENDENCE_INCONCLUSIVE` | `ef020e5:research/v7/e1-postrun-diagnostic-v1/diagnostic-summary.json`. Logged validation declines after epoch 1; final-support replay and causal regime attribution remain unavailable. |

## Production-System State

| Topic | State | Authority / notes |
|---|---|---|
| Architecture audit | `COMPLETE_CURRENT_REFERENCE` | `docs/production-system-architecture-audit@9efed7b:reports/marketmamba_production_system_audit.md` and its production-system roadmap. |
| Phase A implementation | `IN_DEVELOPMENT_MAIN`, `NOT_DEPLOYED` | Fixed `89f9496` integrated locally; original source and candidate branches retained. |
| Phase A task state | Tasks 1–3 `COMPLETED_REVIEWED`; Task 4 limited approval at `1e119ef`; Task 5 limited offline approval at `1c5250e` | Limited local engineering integration completed; production readiness remains outside scope. |
| Current blocker | `NONE_FOR_LOCAL_PHASE_A_INTEGRATION` | Completed local integration; production and research gates remain unchanged. |
| Production success authority | Legacy fetch result remains authoritative | Shadow-ledger failures must not change fetch-only success semantics. |

## Delivery/UI State

- Tracked FastAPI and React legacy delivery surfaces exist; live deployment and deployed revision remain `UNKNOWN` because public endpoint queries were inaccessible and provider dashboards were not accessed.
- V7 status API/UI source is `COMMITTED_NOT_DEPLOYED` in the WSL development main checkout. The public API projects only the summary schema/state/decision/data ID/time/counts; original health artifacts remain local research evidence. This source commit does not change Windows operational checkout or live deployment authority.
- V7 status design is read-only and fail-closed, but live behavior is `UNKNOWN`.

## Active Blockers

1. `STOP`: strict Phase 0 has no verified executable 5d/10d labels.
2. `NOT_DEPLOYED`: Phase A local integration complete, no production promotion.
3. `UNKNOWN`: E1 final-support epoch dynamics and causal regime dependence require the frozen matrix and ex-ante market-state evidence; completed result is admitted.
4. `UNKNOWN`: live API/frontend deployment and publication state.

## Current Decisions / Next Actions

The first Autonomous Charter window is complete. Use the single backlog linked
above; all five selected units are committed/reviewed. No automatic continuation.
Next Human decisions: separate exact-candidate integration, future bounded CPU
window, original E1 matrix location if available, and research admission/design/
compute. Strict gates remain unchanged; no need to repeat completed H0 or E1 search.

## Active worktree boundary

Observed 2026-10-21: main plus16 linked worktrees (17 total after this window's
creation). Only `development/v7-autonomous-charter` changed; main remains6d74d738.
All other checkouts were clean at inventory and preserved. The new tree retains
the closed window for review; no tree removal authorized. See Worktree Register.
No Windows runtime observation was refreshed.

## Repository Rationalization Inventory

Phase 3A read-only inventory completed on 2026-09-22; report: `D:\Desktop\work\project-audit\market-mamba\repository-rationalization\MANIFEST.md`. It classified the three tracked dirty files, 69 collapsed untracked status entries (73 files), eight worktrees, local branches, large artifacts and duplicate candidates. That phase made no source/data/worktree/branch cleanup, Windows checkout sync, push or deployment. Its evidence does not alter research, scheduler or deployment authority.

Phase 3B-1 source canonicalization completed on 2026-09-22 in WSL development main; `D:\Desktop\work\project-audit\market-mamba\phase3b-source-cleanup\MANIFEST.md` is its external audit artifact. The V7 status UI/API is `COMMITTED_NOT_DEPLOYED`, E1-matching environment identity is in Git, early V7 source has a historical SHA register, and local research/task notes are preserved and explicitly dated. `research/v7/current-state.md` defers to this canonical state file. The development main source working tree was clean at completion. Corrected E5/E1/recovery/postrun and Phase A implementation/result authority remain branch-qualified for Phase 3B-2 selective review. No worktree, branch, large artifact or Windows operational checkout was changed, and nothing was pushed or deployed.

## Authority Links

- [Authority Map](Authority_Map.md)
- [Domain knowledge index](../README.md)
- [K0 discovery report](../../reports/marketmamba_k0_project_knowledge_discovery.md)
- [K0 machine authority map](../../research/project-knowledge/k0/authority-map.json)
- [K0 boundary map](../../research/project-knowledge/k0/boundary-map.json)
- Corrected/PIT authorities are branch-qualified in [Authority Map](Authority_Map.md).

## Historical bounded recovery checkpoint — 2026-10-20

The observation-time pending approvals and suggested next steps in this section
are superseded by the current 2026-10-21 decision summary above. They are not
instructions to repeat completed work.

### Snapshot Reader contract continuation (Unit A)

Base `8309ee366aea4c002f9a79d0d97d214b29578fcf`, same isolated branch.
The v1 reader now rejects traversal, absolute/noncanonical raw paths, symlink
files/directories, non-regular files, duplicate keys, malformed/missing/wrong-type
fields, digest/request/publication identity mismatch and incomplete publication.
Directory-relative no-follow reads are anchored at the caller-selected snapshot
root; root ancestors must remain trusted/stable. Integrity is not authenticity,
HTTP success, PIT eligibility or Human correctness approval.

Two repair rounds: initial reader validation (22 failing assertions before fix);
independent review then reproduced nested JSON float overflow (`1e999`), closed
with finite-number parsing. Valid v1 GET/POST, empty payload, naive/offset/UTC
writer timestamps, HTTP error observations and finite floats stay compatible.
Final focused snapshot tests: 17/17; six PIT suites: 47/47 (zero skips);
governance tests: 15/15. Independent reviewer reran 17/17 and approved the
second-round reader changes with no remaining required finding.
No research data was read or modified; all malformed manifests are offline
fixtures. Strict Phase 0 STOP and Task 4 approval pending remain unchanged.
Reader implementation committed as `0a7928061755855faf9aeb55c1702e26a5a62a34`.
The next authorized unit (Task 4 evidence recovery) completed below, on its
separate operations lineage; no operations implementation was imported here.

### Phase A Task 4 evidence recovery (Unit B)

Isolated `fix/phase-a-task4-evidence@46df7b3402f85877a5fc55600d53b15a6f3a7849`
derives directly from original `f5b0ff3df7645098cc01a9eeb36f5df22c4a3e41`.
The original checkout remains unchanged. Evidence index at that commit:
`research/production-system/phase-a/task4-evidence-recovery.json`.
It records specification/report hashes, exact test names, commands and limits.

Recovered a deterministic defect: one artifact referenced by two stages was
counted once but its completed flag and bytes were summed twice. The fix filters
artifact identities by attempt membership before aggregation. Public LedgerStore
fixtures reproduce the baseline failure (`complete_count` 2 instead of 1);
the candidate also verifies equal-sized distinct artifacts, latest empty retry
isolation and unchanged logical ledger contents.

Two bounded rounds: query fix after genuine RED; review-driven test strengthening
from connection-local change count to complete before/after ledger dump.
Final offline ops 70/70 and mocked fetch-only regression 3/3 PASS. Independent
reviewer ran 70+3 in round 1 and reviewed round 2 without rerunning tests; no
Required/Critical finding. These are engineering evidence, not Human acceptance.

At the initial `46df7b3` checkpoint, Task 4 correctness approval remained
**NOT_ESTABLISHED**. The sequential rebuild
test does not prove concurrent committed-only isolation: uncommitted-writer and
mid-read-commit tests are still missing. Sanitizer regression corpus passes but
requires final Human review. Task 5's process crash/restart/replay, >=10 outcome
comparisons, formal artifacts and overhead benchmark remain **NOT_STARTED**.
Do not proceed directly as though Task 4 were approved. Next bounded proposal:
close the two snapshot-isolation evidence gaps, present Task 4 for Human review,
then separately authorize offline Task 5. No research rules/data changed;
strict Phase 0 **STOP** and all research authority remain unchanged.

### Historical Task 4 isolation checkpoint — prior to 2026-10-21 Human approval

`fix/phase-a-task4-evidence@1e119ef855f8fd319f774162beca0bd729d4b921`
adds tests and evidence only, directly after `46df7b3`. The two isolation gaps
above are now closed for the tested SQLite WAL/synthetic-ledger contract:
real spawned writer/reader, uncommitted commit/rollback/SIGKILL cases, and four
deterministic reader boundaries repeated twice. No sleep-based ordering.
Whole projection identity/counts and digest are checked; an explicit no-BEGIN
negative control detects a mixed version. No production correctness bug found
or production code changed. No production repair rounds were needed.

Final Operations 75/75; mocked fetch-only 3/3; separate snapshot branch PIT
47/47 and governance 15/15 PASS. Independent reviewer ran the five isolation
tests (12 scenarios), no Required/Critical finding. Optional cleanup-return
assertion improvement remains deferred; successful child exits/liveness were
directly checked. No cross-branch integration was performed or claimed.

Human review package: Task 4 branch
`research/production-system/phase-a/task4-evidence-recovery.json`,
with 12 raw records and hashes in `isolation-evidence-20261020/`.
Evidence is sufficient to present the bounded Task 4 engineering correctness
decision; **NOT_APPROVED** until Human decides. Tests do not establish research
PIT/label truth, arbitrary-secret absence, power-loss durability or production
readiness. Task 5 remains **NOT_STARTED** and requires a separate authorization:
fixed >=10 observed-vs-shadow replay cases, crash/restart/concurrency, formal
artifacts/report and overhead evidence. Strict Phase 0 remains **STOP**.

- **Completed:** reproduced silent manifest replacement for the same request and
  retrieval timestamp, including parser-only conflicts; identical retries also
  replaced inode/mtime. Replaced shared temporary-name/overwrite publication with
  unique staged files and atomic no-clobber publication. Identical bytes reuse
  the committed file; conflicts, corrupt raw content and manifest symlinks fail
  closed without replacing the original. Valid raw blobs created before a
  rejected manifest remain preserved and do not imply a completed observation.
- **Evidence:** snapshot suite 10/10; all six adopted PIT primitive suites
  40/40, no skips, CPU/local synthetic fixtures, ResourceWarning promoted to
  error. Real competing subprocesses yield one committed manifest; controlled
  SIGKILL before/after manifest publication yields safe retry and valid hashes.
  This is engineering/fault-injection evidence, not market-data or trading
  correctness. Independent review found no required code changes.
  Governance regression 15/15; syntax/JSON and `git diff --check` passed.
  Knowledge-impact review completed; unified knowledge health exits 0 with
  pre-existing Governance/Authority Drift/Legacy Docs WARNING, no structural
  FAIL. Operations/GOVERNANCE rules reviewed and unchanged: no control-plane,
  authority or lifecycle policy change. Full project/model-training tests were
  not run; verification is bounded to the six PIT primitives and governance.
- **Runtime:** Python 3.12.3; related pandas tests use the existing
  `.worktrees/v7-e1-colab-recovery/.venv/bin/python` with GPU hidden. No installs,
  model training or data writes. `MARKETMAMBA_DATA_ROOT` was unset; canonical
  development `Data/processed_v6` resolves to
  `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6`; that data was
  not loaded or mounted into the new worktree. Only temporary fixture data used.
- **Limits:** trusted snapshot directories and hard-link/fsync-capable filesystem
  required; unsupported storage fails, with no overwrite fallback. No certification
  of Windows/Drive semantics, adversarial directory mutation or host power-loss
  durability (new-directory parents are not fsynced). Killed-writer temporary
  files stay non-authoritative and are not auto-cleaned.
- **Remaining gates:** strict Phase 0 STOP, verified labels 5d/10d = 0; Phase A
  Task 4 approval not established / Task 5 not started; E1 matrix/replay and
  causal-regime evidence limitations unchanged. No merge, deployment or approval.
- **Resume:** inspect this branch and commit/tests before using the candidate;
  the main source-adoption register remains historical byte-identity evidence.
  Existing memory-hardening branch `91db7d1` remains unmerged; no new hooks or
  memory system were installed. Next bounded candidate: validate source snapshot
  reader path/schema rejection with offline fixtures, without admitting new data.

Reproduce the bounded regression from the candidate worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home/frank/projects/MarketMamba/.worktrees/v7-e1-colab-recovery/.venv/bin/python -B -W error::ResourceWarning -m unittest V6.experimental.v7_source_snapshot_test V6.experimental.v7_temporal_contract_test V6.experimental.v7_executable_labels_test V6.experimental.v7_ohlcv_provenance_test V6.experimental.v7_historical_security_master_test V6.experimental.v7_historical_label_materialize_test -v
```

## Knowledge-impact Rule

Update canonical knowledge in the same change for: semantic changes, authority changes, subsystem-boundary changes, active runtime-state changes, correctness-gate changes, new canonical artifacts, deprecations, or major blocker/next-decision changes.

No knowledge update is normally required for formatting, comments, pure refactors, tests-only changes, or internal renames that do not change public semantics or authority.

## Phase 3B-2 selective source integration (2026-09-22)

Development main now contains the reusable corrected PIT primitives, corrected E5 research core, formal E1 research core and frozen contract copies, plus the E1 telemetry correction and reusable Colab log wrapper. The [source adoption register](../../research/v7/source-adoption-register-20260922.json) records source commits, blobs, destinations and test evidence. This is research source authority in main, **not** result, portfolio, operational, Colab run or deployment promotion. The old 32/8 baseline draft remains branch-qualified; the accepted E5/E1 architecture is 64/32. E1 historical-simulation `PASS_REFRESH_HYPOTHESIS` and strict Phase 0 `STOP` remain unchanged. E1 postrun diagnostics remain active; Phase A Task 4 approval and Task 5 acceptance remain open. Eight worktrees were inspected and retained; no branch/worktree/artifact was removed. Windows operational checkout, Scheduler and deployment were not changed; no push.

## Phase 3B-3A governance checkout retirement (2026-09-22)

The four authorized K0/K1/K2 governance checkouts were removed using normal Git worktree removal after clean-state, ignored-file, branch-history and path-dependency checks. Their branch refs and commits remain as `HISTORICAL_LINEAGE / NO_ACTIVE_CHECKOUT`; the four E1/Phase A worktrees remain present. Details and local bytecode disposition are in [Worktree Register](Worktree_Register.md). No research artifact, branch, Windows operational checkout, deployment or scheduler was changed; no push.

## Phase 3C storage audit (2026-09-22)

Read-only reachability/rebuildability audit completed; [external manifest](/mnt/d/Desktop/work/project-audit/market-mamba/storage-audit/MANIFEST.md). No storage item was deleted. D: scheduled fetch-only inputs/outputs, historical baseline caches, dated ZIPs, E1 formal bundle/results, V7 provenance/label snapshots and active recovery `.venv` remain protected. A clean offline frontend install/build demonstrated that local `node_modules` and `dist` are future low-risk generated cleanup candidates (~187 MB), subject to separate cleanup authorization. Large-cache/archive deletion has no approved exact survivor or rebuild path. `DATA_QUALITY_DIAGNOSTIC` remains separate.

## Repository/governance rationalization closure (2026-09-22)

The **repository/governance rationalization program** is complete: canonical governance and authority navigation are established, runtime boundaries were reconciled, development main source is clean, reusable PIT/E5/E1 research source was selectively integrated, historical source was preserved, the four K governance checkouts were retired with branch history retained, and storage received a retention decision without deletion. The E5/E1 refresh, E1 recovery, E1 postrun, and Phase A worktrees remain intentionally present. Development main is ready for remote publication; the finalization audit records the actual Git push and remote-HEAD verification separately.

This closure does **not** complete V7 research, strict Phase 0 (`STOP`), E1 postrun diagnostics, Phase A acceptance, `DATA_QUALITY_DIAGNOSTIC`, Windows operational promotion, backend/frontend deployment verification or promotion, or large-storage cleanup. The Windows operational checkout and Scheduler remain separate and unchanged. Research result authority and branch-local operations authority remain as stated above and in the Authority Map.
