## Step3b 協調者裁定與歷史bundle候選 — 2026-11-26

[報告Step3b](../../reports/step3_engineering_20261126.md)記錄Claude A–D裁定（待獨立審查）：pin TEJ原始欄位文件，限2013/01起首次單月與發布日，NTD千元轉元，跨2012比較／非NTD為NaN且禁用去年／累計匯出欄。逐列h20、五欄、48欄原矩陣、日曆／價格／成員、185/190共用列、抽樣及版本／程式身分只在指派worktree `.artifacts/step3/` 凍結；runner bundle不符STOP。這是歷史完整價格研究股票之CPU工程准入候選；strict前瞻仍STOP，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，不改既有48欄或正式Data，不執行或查看IC／績效。下方Step3a source STOP及未materialize為當時歷史觀測，本輪實跑結果以報告為準。

## Step3 五欄工程候選 — 2026-11-26

[Step3 工程報告](../../reports/step3_engineering_20261126.md)與[合成測試](../../tests/test_step3.py)涵蓋首次單月值重建 YTD、發布日後首 session 收盤、同 session 穩定重播、舊月晚到、版本碰撞 STOP、缺月及跨年起點。`tools/step3_runner.py` 重用原 `c0plus.build_features`，185／190 欄共同部分逐鍵逐值核對；缺值原因與去重來源鏈另存 metadata，不加研究特徵。候選 UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；沒有正式營收特徵 materialization 或 source admission，版本／幣別／合併口徑 gate 仍 STOP，不改既有48欄。

# Features

## Metadata

- Status: `CANONICAL_DOMAIN_NOTE`
- Owner: MarketMamba maintainers
- Last reviewed: 2026-09-20
- Authority topics: corrected feature contract, ordering, scaling, graph/industry policy
- Update triggers: feature manifest or scaling-policy change

## Purpose

Describe the current corrected feature boundary and distinguish it from legacy V6 and experimental feature sets.

## Current Boundary

The corrected V7 baseline is `E5-PIT-Clean-v1` with exactly 48 ordered features. The feature-order SHA-256 is `7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696`. Group dimensions are `[15, 20, 1, 12]`.

## Authority

The feature manifest and corrected policy at `fix/v7-historical-pit-reconstruction@e3bce91` are authoritative. Model contracts consume that exact order; prose and old notebooks do not redefine it.

## Inputs

Canonical market, macro, flow, and eligible company data whose PIT status satisfies the data policy.

## Outputs

An ordered 48-column feature matrix plus matrix/contract identities used by training and resume validation.

## Current Implementation

- Explicit graph features: disabled.
- Current-industry backfill: disabled.
- Industry neutralization: disabled.
- Macro scaling: causal expanding z-score.
- Cross-sectional features: per-session scaling.
- Financial fields without adequate publication-time coverage were removed.
- `Market_Cap_Log` remains admitted under the corrected policy.

The dropped financial set includes `PER`, `PBR`, `Revenue_MoM`, `Revenue_YoY`, `EPS`, `EPS_Surprise`, `Gross_Margin`, `ROE`, `Book_Value`, `Dividend_Yield_Fwd`, and `Free_Cash_Flow`.

## Evidence Classes

- Manifest order/hash: binding machine contract.
- Matrix identity: completed materialization evidence.
- Feature analysis report: diagnostic only unless incorporated into a manifest.

## Current vs Legacy / Research

Legacy counts of 56 (V6.1), 59 (V6.2 tracked runtime), and 66 (isolated baselines) describe different contracts. They must not be compared as though column membership and preprocessing were unchanged.

## Invariants

- Exact names, order, transforms, and missing-value behavior travel together.
- Reuse a persisted matrix only when data and feature identities match.
- No graph or industry feature may silently re-enter through preprocessing.
- Downstream admission must fail closed on contract mismatch.

## Known Limitations

The manifest records `value_table_status=NOT_EMITTED_DOWNSTREAM_ADMISSION_BLOCKED`; downstream use still depends on the appropriate admission gate and completed artifact identity.

## Do Not Use / Do Not Mix

- Do not substitute a legacy 56/59/66-column matrix.
- Do not infer current membership from similarly named columns.
- Do not rebuild an unchanged matrix merely because a Colab runtime restarted.

## Important Paths

- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/feature-manifest.json`
- `fix/v7-historical-pit-reconstruction@e3bce91:research/v7/corrected-baseline-v1/policy.json`
- Replication contract: `feature/v7-e1-rolling-origin-refresh@044eef7:research/v7/corrected-e5-replication-v1/incumbent-contract.json`

## Related Domains

[Data](Data.md), [Models](Models.md), [Training](Training.md), [Research](Research.md)

## Update Triggers

Update whenever feature membership/order, transforms, graph policy, industry treatment, or matrix identity rules change.

**2026-09-22 source identity:** The frozen 48-feature manifest is copied byte-identically into development main for research-source tests; the original branch/commit and blob are recorded in `research/v7/source-adoption-register-20260922.json`. Feature order and semantics are unchanged.

## Window05 source integrity audit — 2026-10-23

Branch-local source audit finds unresolved release-time/vintage/missing-default
risks; exact48membership/scaling contract stays frozen. VERIFIED0/CONDITIONAL21/
UNKNOWN27/INVALID0 refers to end-to-end audit certainty, not new formal feature
admission or historical result reclassification. Empirical payload quality UNKNOWN.
Unverified date-only macro availability blocks dependent real-data factor analysis.
[Report](../../reports/v7_window05_feature_integrity.md),
[matrix](../../research/v7/window05-feature-integrity.json). UNMERGED/NOT_DEPLOYED.

Window05 F1 shortlist is PROPOSED: Abnormal Volume and conditional Foreign Flow
Intensity. Research-only timestamp-gated formulas and9synthetic tests provide no
RankIC/admission evidence. Existing RS20 and reversal of Return1d are redundant
under equal raw-data conventions. Real diagnostics remain BLOCKED by F0 evidence.
[Report](../../reports/v7_window05_market_candidates.md).
