## Step3c 來源待 Human 決定／完整性修補 — 2026-11-26

[Step3c報告](../../reports/step3_engineering_20261126.md)取代下方Step3b目前PASS主張。歷史版本gate為HUMAN_DECISION_REQUIRED，strict前瞻仍STOP；無Human接受、fits或IC／績效。跨來源291,847配對中291,360差<0.1%，446差>1%，只證明數值／單位一致而非首次版本或publication PIT。adapter一致列標NTD_BY_CROSS_SOURCE_AGREEMENT，其餘UNIT_UNVERIFIED／NaN；版本0只是匯出重播序，不是首次版認證，合併可比性仍是待Human接受的假設。完整72-job schema、原始來源manifest SHA、job陣列身分、prediction必備與凍結labels／keys／session驗證皆屬UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED工程候選；實際驗證及限制見報告，不改正式Data或既有48欄。下一步為獨立重審及Human明確選擇，不能自動啟動研究。

# Labels

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: executable label semantics, verified versus simulated labels, readiness gates
- Update triggers: label contract, execution evidence, or readiness decision change

## Purpose

Separate strict verified executable labels from historically simulated labels and legacy return targets.

## Current Boundary

For corrected V7, the signal is observed after close on day `t`; entry is the next calendar session open only when executable; exit is the close after the configured 5- or 10-session holding horizon. There is no rolling to another entry or exit. Unknown execution evidence fails closed.

## Authority

The corrected policy, label manifest, execution-proxy policy, and remediation readiness at `fix/v7-historical-pit-reconstruction@e3bce91` define the semantics and current counts.

## Inputs

Eligible universe membership, exchange calendar, provenance-eligible OHLCV, and daily tradability/executable-open evidence.

## Outputs

- Strict corrected verified executable labels.
- Separately named historical-simulation labels.
- Rejection/reason metadata and maturity boundaries.

## Current Implementation

Strict Phase 0 is `STOP`; corrected verified executable labels remain 5d=`0`, 10d=`0` because direct daily tradability/executable-open evidence is unresolved. Historical-simulation readiness is separately `PASS`, with 5d=`6,889,229` and 10d=`6,790,554` labels in v1. Critical false executable count is zero under the frozen P0/P1/P2 proxy policy.

E1 v2 extends the historical-simulation window and reports 5d=`7,023,920`, 10d=`6,923,464`; these remain simulated labels under that experiment contract.

## Evidence Classes

- Verified executable: direct admissible evidence under the strict contract.
- Historical simulated: proxy-qualified research label, never renamed “verified.”
- Legacy label: old contract preserved for reproducibility only.

## Current vs Legacy / Research

Legacy counts (5d=`7,944,471`, 10d=`7,822,665`) belong to a different contract. Historical-simulation PASS permits research under explicit simulated semantics; it does not clear strict Phase 0 or trading readiness.

## Invariants

- Membership is not tradability.
- Unknown execution evidence remains unknown and fails closed.
- No entry/exit rolling and no synthetic proof of executability.
- Strict and simulated readiness are reported independently.
- Acceptance remains zero-tolerance for critical false executables.

## Known Limitations

Direct historical daily tradability and executable-open evidence is not complete, so strict labels remain empty.

## Do Not Use / Do Not Mix

- Do not relabel simulated targets as corrected verified targets.
- Do not use legacy A/B/C results as tradable evidence.
- Do not lower the acceptance threshold or add case-specific exceptions.

## Important Paths

- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/label-manifest.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/historical-execution-proxy-v1/execution-proxy-policy.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/ohlcv-provenance-remediation-v1/readiness.json`

## Related Domains

[Universe](Universe.md), [Execution](Execution.md), [Data](Data.md), [Training](Training.md), [Research](Research.md)

## Update Triggers

Update when direct evidence becomes available, the proxy policy is versioned, label counts change, or a new readiness decision is recorded.

## Window26 bounded label evidence — branch-local / not admitted

2321 的 68% 虧損減資與市場股數單位轉換已有來源證據；三筆既有 10d raw labels 算式正確，但不具同股數未調整報酬語意。
原始停牌公告與其他事件覆蓋仍 UNKNOWN；W24/W25 結果不改寫／不重算。兩路 Label Policy 與 next-study 全為 PROPOSED。
參見 [W26 evidence](../../research/v7/window26-evidence.json) 與 [事件矩陣](../../research/v7/window26-event-matrix.json)。

## Window27 development-only safety sensitivity

Human 限定批准在固定規則下對 W24 原預測重評，Policy/Mask 先於 metrics commit。
原 Strict/W24 masks 不動；新規則為 Development-only，完整事件／停牌／asof UNKNOWN 不變 CLEAN。
Training Mask 是診斷，不代表 frozen fitted model 已去除污染。
[Policy](../../research/v7/window27-policy.json)、[Report](../../reports/v7_window27_label_safety.md)。

## Window29 disjoint development / feature-label isolation

68新TWSE候選（W24重疊0）、64有conditional Prediction身分；current survivor/hindsight與4下市起日/普通股UNKNOWN不升格歷史Universe。
最小research feature-only介面先保存Predictions，再做Label Join；future label刪除只改evaluation coverage，18回歸tests與獨立Raw-derivedReplay驗證。
5d正向Mean/Median在新樣本延續、10d不一致；未知actions/asof/vintage與2022/2023曝光保留，不作模型選擇/投資/Strict PIT claim。
[Evidence](../../research/v7/window29-evidence.json)、[Report](../../reports/v7_window29_expanded_development.md)。
