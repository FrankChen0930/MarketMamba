## WP3 同版本 mask 重建補證候選 — 2026-11-26

[報告](../../reports/forward_wp3_mask_20261126.md)：同一份prepared cache／universe tree重呼原`build_arrays`在補0前取finite mask；roadmap七欄lag配方重現之X與P1實際9778392d…完整file bytes一致，重建mask含2,453,171 false cells且F-layout SHA精確符合舊manifest004771dd…。mask不是舊檔複用、全1或zero-filled X推回；原轉換腳本仍缺，故只為provenance-limited候選，不賦予上游預設值／raw observation／strict PIT真實性。

原C0+ dense／lagged／rolling helpers與相同market expressions完整axis逐欄寫F memmap，建立7,423,927×185 immutable工程reference；3種合成chunk全bytes／NaN相同、256真實列全185欄重查及全部snapshot hashes通過。fold_2026 splits與原W60 eligibility保留，既有P1 scores原樣opaque封存，未載入y5／y10陣列或用成熟labels篩features。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED、forward_admitted=false；未驗真實每日append／score鍵與model語義，不變更48欄、任何歷史artifact／來源／production authority。獨立review與Human acceptance待辦；舊WP3缺mask描述保留為當時歷史觀測。

## P2 WP3 immutable feature-only候選 — 2026-11-26

[工具](../../tools/forward_wp3.py)／[測試](../../tests/test_forward_wp3.py)只消費已causal prepared的frozen48欄，重用原C0+完整prefix建185欄；固定calendar／roster reduction axis，不用尾60日重設cumsum。追加前完整重驗父／delta identity，舊keys／X／mask／dtype／splits／eligibility／185欄與opaque既有scores bytes必須一致；labels另版，不讀未來labels或用label成熟篩日列。partial／重複／錯identity拒絕，未實作daily raw→prepared來源或WP4 pointer／recovery。CPU_ENGINEERING_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，152項合成／相關測試與2 source mutations屬作者自查。

真實完整reference **STOP**：prepared缺原始masks.npy，zero-filled X無法反推；X實測identity符合外層P4 dataset inventory，內層舊manifest另列不同版本。一次唯讀hash／RSS evidence見[報告](../../reports/forward_wp3_20261126.md)；真實185欄與daily延伸未驗，不能推定PIT／來源准入或放寬舊bytes gate。沒有修改正式48欄schema、歷史artifact、Data、P4d或第三步bundle。

## 同業代理工作包3 — 2026-11-26

[資料層報告](../../reports/peer_proxies_data_20261126.md)／[實作](../../tools/peer_proxies.py)是未提交UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED候選；無正式特徵或資料准入。TEJ子產業依粒度及未列變更紀錄比例選主分類，兩套每日分類皆留unknown。首次與現況一致且無變更才重建穩定分類；raw產業／市場變更值方向未知，完整歷史成員及known_at仍STOP，不把現況分類倒填歷史。上市入口與下市端點採WP1保守子集，下市當日排除；價格缺失不移除已識別成員分母。

1a以t當期同業排除本股，聚合t−1／t−2完整session raw收盤價差，至少3家；市場等權另存。1b僅上一完整曆月，首次單月值各自配去年同月，兩期發布日之後首session收盤才可用，目標可用即回缺，至少3家；跨2013/01口徑NaN。逐列、原因及分母只存私有 `.artifacts/peer/`，不修改48欄、正式Data或其他bundle；日曆／公司行動／歷史版本限制見報告。

## Step3c 來源待 Human 決定／完整性修補 — 2026-11-26

[Step3c報告](../../reports/step3_engineering_20261126.md)取代下方Step3b目前PASS主張。歷史版本gate為HUMAN_DECISION_REQUIRED，strict前瞻仍STOP；無Human接受、fits或IC／績效。跨來源291,847配對中291,360差<0.1%，446差>1%，只證明數值／單位一致而非首次版本或publication PIT。adapter一致列標NTD_BY_CROSS_SOURCE_AGREEMENT，其餘UNIT_UNVERIFIED／NaN；版本0只是匯出重播序，不是首次版認證，合併可比性仍是待Human接受的假設。完整72-job schema、原始來源manifest SHA、job陣列身分、prediction必備與凍結labels／keys／session驗證皆屬UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED工程候選；實際驗證及限制見報告，不改正式Data或既有48欄。下一步為獨立重審及Human明確選擇，不能自動啟動研究。

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
