# V7 Phase 0 Correctness Audit

Gate decision: STOP

## Timing chain

session t close -> session t after close (17:00 Asia/Taipei contract example) -> session t+1 open (09:00 Asia/Taipei contract example) -> session t close in current Alpha_5d/Alpha_10d implementation -> session t+5 close or session t+10 close on the independent trading calendar

## Findings

### label_execution_alignment

- [FAIL/CRITICAL] `label_execution_alignment.close_denominator_precedes_execution`: Current labels are Close[t+h] / Close[t] - 1, but a prediction that consumes session-t close information is not executable at Close[t]. The portfolio contract rejects execution at the signal timestamp and first fills at the next session. Both 5d and 10d IC therefore include the untradeable Close[t]-to-next-execution interval.
  - Affected scope: Every current Alpha_5d and Alpha_10d training/evaluation label and every IC derived from them; 2005-2026 prepared history and 2024-2026 A/B/C evaluations.
  - Requires rebuild: yes
  - Minimum fix: Freeze an execution convention (for example next-session open), redefine both horizons from the executable entry price to an explicit end price, regenerate labels, retrain the incumbent seeds, and regenerate predictions before signal/portfolio validation.
  - Evidence: V6/experimental/v7_integrated_data_quality.py:177-193 - calendar_labels uses close[indices+horizon] / close[indices] - 1 for both horizons.
  - Evidence: V6/experimental/v7_portfolio_engine_test.py:135-175 - A 17:00 signal cannot execute at the same timestamp; the first valid fill example is next session at 09:00.

### feature_point_in_time

- [PASS/INFO] `feature_point_in_time.after_close_price_volume`: Daily price/volume at session t is compatible with an after-close prediction and next-session execution, provided no same-close execution is claimed.
  - Affected scope: Daily OHLCV-derived features.
  - Requires rebuild: no
  - Minimum fix: Retain the after-close prediction and next-session execution convention.
  - Evidence: V6/experimental/v7_portfolio_engine_test.py:135-175 - The execution contract separates a 17:00 signal from the next 09:00 market session.
- [UNKNOWN/MAJOR] `feature_point_in_time.revenue_assumed_release_dates`: Revenue rows with create_time use next-session availability, but missing create_time falls back to month-end plus 11 calendar days. The pipeline records assumed rows but does not prove the actual historical publication timestamp for them; the smoke metadata shows 1,156 assumed-release rows.
  - Affected scope: Revenue_MoM and Revenue_YoY rows without historical create_time; full-history affected-row count is not present in the returned A/B/C deliveries.
  - Requires rebuild: yes
  - Minimum fix: Obtain vintage publication timestamps or conservatively exclude/lag all assumed rows, quantify the full-history scope, then rebuild features, labels, training, and predictions.
  - Evidence: V6/experimental/v7_integrated_point_in_time.py:6-34 - Missing release timestamps use the fixed fallback and increment revenue_assumed_release_rows.
  - Evidence: deliveries/history/before-handoff-20260913/v7-colab/MarketMamba/delivery/smoke-data/feature_metadata.json:point_in_time_alignment - The recorded diagnostic metadata reports revenue_assumed_release_rows=1156.
- [UNKNOWN/MAJOR] `feature_point_in_time.financial_statement_fixed_lag`: Quarterly statements use quarter-end plus 45 days and year-end plus 90 days rather than issuer-specific historical filing timestamps. This is a conservative-looking approximation but is not vintage certification, especially around non-business-day deadlines and late filings.
  - Affected scope: EPS, EPS_Surprise, Gross_Margin, ROE, Book_Value, and Free_Cash_Flow across the prepared history.
  - Requires rebuild: yes
  - Minimum fix: Bind each filing to a verified historical publication timestamp (usable no earlier than the next executable session), then rebuild affected features and dependent model artifacts.
  - Evidence: V6/marketmamba/data/feature_engineer.py:790-909 - Financial statement available_from is derived only from period end plus a fixed 45/90-day lag.
  - Evidence: V6/marketmamba/data/feature_engineer.py:1220-1319 - Cash-flow available_from is quarter-end plus a fixed 45-day lag.
- [FAIL/MAJOR] `feature_point_in_time.industry_snapshot`: Industry-neutralized features use one frozen accumulated stock_info snapshot for all historical dates. The implementation and delivered metadata explicitly state that the classification is not historical point-in-time.
  - Affected scope: All historical rows whose 42 non-excluded features are industry demeaned; practical membership-change count is not yet quantified.
  - Requires rebuild: yes
  - Minimum fix: Provide effective-dated industry membership or remove historical industry neutralization, then rebuild features, retrain, and regenerate predictions.
  - Evidence: V6/experimental/v7_integrated_industry.py:26-68 - The classification policy is emitted as frozen accumulated snapshot, not historical PIT, and affects 42 neutralized columns.
  - Evidence: deliveries/stability-c-v1/summary.json:classification_limit - The C delivery repeats the non-PIT frozen-classification limitation.
- [UNKNOWN/MAJOR] `feature_point_in_time.graph_vintage`: One knowledge_graph_v2 snapshot is converted and used across the full history. No effective dates or historical graph vintages are present in the training contract, so future-known relationships cannot currently be ruled out.
  - Affected scope: All graph messages for all training and evaluation dates.
  - Requires rebuild: yes
  - Minimum fix: Audit edge sources and effective dates; either build date-valid graph snapshots/edge masks or exclude unverifiable relationships, then retrain and regenerate predictions.
  - Evidence: V6/experimental/v7_integrated_train.py:731-740 - Preparation fingerprints a single knowledge_graph_v2.npz source.
  - Evidence: research/v7/integrated-candidate-plan.md:19-19 - The plan converts the single graph artifact to CSR and does not define effective-dated edges.

### historical_universe

- [FAIL/MAJOR] `historical_universe.inferred_observation_intervals`: The declared universe is a graph-supported candidate inferred from first/last archived observations, not verified listing/delisting history. It explicitly cannot detect boundary omissions, excludes new listings outside 2,244 frozen graph-supported IDs, and is not unbiased-universe certification.
  - Affected scope: Historical cross-sections from 2005 through 2026, including IPO entry, delisting tails, suspensions, and stocks outside the frozen graph.
  - Requires rebuild: yes
  - Minimum fix: Build an effective-dated TWSE/TPEX ordinary-stock membership table with listing, delisting, suspension, and eligibility rules independent of price coverage and graph availability; re-prepare the universe and all dependent artifacts.
  - Evidence: deliveries/history/before-handoff-20260913/v7-colab/MarketMamba/delivery/full-history-calendar-candidate-summary.json:1-8 - The summary declares 5,263 sessions, 2,244 stocks, and the limitation that first/last observation intervals are not verified listing history.
  - Evidence: deliveries/history/before-handoff-20260913/v7-colab/MarketMamba/delivery/inputs/calendar-through-20260911.json:universe_provenance/universe_limitations - Provenance explicitly excludes unbiased-universe certification and notes frozen-graph omissions.
- [PASS/INFO] `historical_universe.valid_rows_not_silently_dropped`: Within the declared candidate, the preparation contract keeps the union of declared IDs and does not use current market membership to discard numerically valid historical rows.
  - Affected scope: Rows inside the supplied candidate universe only; this does not cure candidate construction bias.
  - Requires rebuild: no
  - Minimum fix: Preserve this admission behavior when replacing the candidate membership source.
  - Evidence: V6/experimental/v7_integrated_data_quality.py:53-69 - The universe contract requires explicit expected IDs for every chosen session.
  - Evidence: V6/experimental/v7_integrated_data_quality.py:140-173 - Quality assessment retains valid mixed-source observations and reports coverage against the declared denominator.

### preprocessing_leakage

- [PASS/INFO] `preprocessing_leakage.daily_cross_section_statistics`: Winsorization and z-scoring for stock-varying features use only the same-date cross-section. Under the after-close/next-session convention this does not use future sessions.
  - Affected scope: Cross-sectional stock-varying features.
  - Requires rebuild: no
  - Minimum fix: Keep calculations date-local and bind them to the corrected historical universe.
  - Evidence: V6/marketmamba/data/feature_engineer.py:1775-1852 - Winsor quantiles and mean/std are grouped by Date; no global future-period fit is used.
- [PASS/INFO] `preprocessing_leakage.macro_expanding_statistics`: Macro time-series normalization uses expanding mean/std shifted by one session.
  - Affected scope: Macro time-series normalized features; Group D is currently masked in the E5 contract.
  - Requires rebuild: no
  - Minimum fix: Retain the shifted expanding implementation if these features become active.
  - Evidence: V6/marketmamba/data/feature_engineer.py:1752-1772 - Both expanding statistics call shift(1), excluding the current and future observations from fitted moments.
- [PASS/INFO] `preprocessing_leakage.imputation_after_scaling`: Remaining missing values are filled with zero only after per-date/expanding normalization, so imputation does not fit global future statistics.
  - Affected scope: All 59 feature columns.
  - Requires rebuild: no
  - Minimum fix: Retain the post-normalization imputation order and continue reporting missingness.
  - Evidence: V6/marketmamba/data/feature_engineer.py:1830-1874 - The final fillna(0) happens after cross-sectional and macro normalization.
- [FAIL/MINOR] `preprocessing_leakage.research_period_reuse`: The 2024-2026 evaluation period has already influenced research decisions. This is selection contamination rather than a row-level preprocessing leak, so it does not independently trigger the correctness hard gate but it prevents final-holdout claims.
  - Affected scope: Any claim treating 2024-2026 as a pristine final holdout.
  - Requires rebuild: no
  - Minimum fix: Use paired/non-IID inference for diagnostics and reserve a future forward period for final confirmation.
  - Evidence: research/v7/current-state.md:1-120 - The project record summarizes repeated capacity, confirmation, stability, and ensemble decisions using this period.

## Blocking findings

- `label_execution_alignment.close_denominator_precedes_execution` (CRITICAL): Freeze an execution convention (for example next-session open), redefine both horizons from the executable entry price to an explicit end price, regenerate labels, retrain the incumbent seeds, and regenerate predictions before signal/portfolio validation.
- `feature_point_in_time.revenue_assumed_release_dates` (MAJOR): Obtain vintage publication timestamps or conservatively exclude/lag all assumed rows, quantify the full-history scope, then rebuild features, labels, training, and predictions.
- `feature_point_in_time.financial_statement_fixed_lag` (MAJOR): Bind each filing to a verified historical publication timestamp (usable no earlier than the next executable session), then rebuild affected features and dependent model artifacts.
- `feature_point_in_time.industry_snapshot` (MAJOR): Provide effective-dated industry membership or remove historical industry neutralization, then rebuild features, retrain, and regenerate predictions.
- `feature_point_in_time.graph_vintage` (MAJOR): Audit edge sources and effective dates; either build date-valid graph snapshots/edge masks or exclude unverifiable relationships, then retrain and regenerate predictions.
- `historical_universe.inferred_observation_intervals` (MAJOR): Build an effective-dated TWSE/TPEX ordinary-stock membership table with listing, delisting, suspension, and eligibility rules independent of price coverage and graph availability; re-prepare the universe and all dependent artifacts.
