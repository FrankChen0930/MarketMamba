---
name: ml-leakage-audit
description: MarketMamba 金融 ML 洩漏與評估稽核；builder／verifier 修改資料工程、特徵、標籤、訓練切分、評估或回測時載入，逐項核對專案事故、共同支援與證據邊界。
---

# 金融 ML 洩漏與評估稽核

本 skill 不覆蓋 AGENTS.md 與 Human 最新指示。先依 project map 找本次 frozen contract；清單是檢查入口，不是新研究授權，也不替代 machine evidence。
沿用 advisor 的協調與 ruflo-project 的證據交接；本清單不派 agent、不定研究門檻。下列「案例」區分已發事故與已知限制，不能把限制寫成已修復。

## 證據入口

- [P3 工程報告](../../../reports/v7_p3_portfolio_engineering.md)：§偵察與重現設定、§修訂 1 bounded repair、§修訂 2 解鎖程序。
- [P4a 工程報告](../../../reports/v7_p4a_colab_engineering.md)：§P4a Q3 事故修補與 pre-flight rehearsal、§d82d020 複審三項 REQUIRED 修補、§完整固定佇列 builder。
- [P3 評估](../../../reports/v7_p3_portfolio_evaluation.md)／[P2 評估](../../../reports/v7_p2_portfolio_evaluation.md)：曝光與資料限制。
- [量化複習表](../../../quant_rigor_checklist.md)提供問題分類；其中 CPCV／DSR 等建議不自動取代本次 frozen walk-forward／Holm 契約。

下列程式路徑相對 repository root；`test_*.py` 均在 `V6/research_lab/tests/`。先讀對應函式；執行測試時用既有核准 CPU Python，設定 `PYTHONPATH=V6 CUDA_VISIBLE_DEVICES=''`，例如 `python -m pytest -q V6/research_lab/tests/test_p3.py::test_p3_training_purge_maturity`；禁止因此啟動資料取得或正式訓練。

## 逐項審查（每項記錄實際輸出或 UNKNOWN／STOP）

1. **Look-ahead 與可知時間**：逐欄追 source、publication／available time、lag 與決策時點；收盤特徵不得假設收盤前可下單，歷史修訂不得回填成當時已知。
   - 怎麼檢查：核對 [Data](../../../knowledge/01_Domains/Data.md)／[Labels](../../../knowledge/01_Domains/Labels.md) authority；讀 `V6/research_lab/rolling_folds.py:make_fold_splits` 與 `V6/research_lab/tests/test_pit_snapshots.py`，抽邊界列比 available_at ≤ decision_at；缺來源證據即 UNKNOWN。
   - 出事案例／限制：P3 工程「完整24次擬合估時與限制」仍是模擬標籤／觀測日曆；時間切分通過不能證明 source PIT。

2. **5／10 日標籤成熟與 purge**：兩個 head 的未來區間都不得跨入驗證／評估；不能只驗內層 finite 子集，須驗完整 outer split1。
   - 怎麼檢查：讀 `seq_runner.py:SeqData.split_rows`、`rolling_seq_scores.py:checked_splits`；P3 outer train session `< eval_first - 20`，內層最後126 eligible sessions 作 validation，fit／val 均 finite y5、y10。對照本次契約，不任意套用其他實驗數字；跑 `test_p3.py::test_p3_training_purge_maturity`，檢視 `test_p3_real_fold_purge` 的真折前置資料需求。
   - 出事案例／限制：P3 修訂1特別驗 outer split，不以被 fit 篩掉的列掩蓋 purge；stride／交易 session 不可誤當日曆天。

3. **尾端 NaN 是未知標籤，不是零報酬**：eval 列保留未成熟 y5／y10；列身分核對 NaN 等價、有限／NaN 不一致及 ±inf 拒絕；IC 才在共同成熟支援上計算。
   - 怎麼檢查：讀 `p4_eval.py:common_rows` 與 `p4_colab.py:permutation_fit`；跑 `test_p4_rehearsal.py::test_common_nan_identity_and_infinity`、`test_permutation_nan_uses_only_mature_support`、`test_portfolio_outcomes_ignore_identity_labels`，驗 input／used／excluded_nan 與排除日期。
   - 出事案例：P4a Q3 nonfinite label 事故，原 `02_formal@4e56e9d` 到 W120 才被尾端 label_y5 NaN 擋下；36 fits 完成是 Human 報告，非本清單的現場驗證。

4. **Rank-gauss／標準化擬合範圍**：區分每 session 截面轉換與跨時間學得的 scaler；後者僅 fit 合格 train rows，不能用 validation／test 選邊界。
   - 怎麼檢查：讀 `seq_runner.py:ensure_rankgauss`、`macro_columns`、`SeqData` 的 normalize 分支；跑 `test_seq_runner.py::test_rankgauss_is_point_in_time_standardised_and_zeroes_macro`、`test_diagnose_validation_fits_only_on_purged_training_rows`、`test_rankgauss_cache_is_invalidated_when_x_changes`；核 cache input／implementation hashes。
   - 出事案例／限制：P3 偵察記錄 macro 欄位是 seed0 抽60 sessions 辨識日內常數，並非人工 frozen schema；既有 `test_regression_raw_scale_features_were_destroyed_by_clipping_but_rankgauss_learns` 保留 raw clipping 退化案例，不能只查是否有 normalize。

5. **跨股票資訊仍須同一可知時點**：同日截面可互看已知 features，不可混日期、看別股未來 label，或用未來 universe 決定當日成員。
   - 怎麼檢查：讀 `seq_models.py` crossstock 路徑與 `seq_runner.py` sampler；跑 `test_p3.py::test_p3_rank_average_pit`；檢視 `test_p4_full.py::test_permutation_summary_order_sensitive` 的 Q5 診斷，核同日群組與固定排列規則，不用 Q5 自行選模型。
   - 出事案例／限制：P3 工程的 rank mutation 移除 date 分組會失敗；P4a Q5 在共同成熟支援不足10名／常數時必須 STOP，不能把無法計算寫成0。

6. **Universe、存活者偏誤與下市**：以當時上市／下市生命週期與來源覆蓋核對，缺股不能直接視為不在 universe。
   - 怎麼檢查：讀 `V6/research_lab/tests/test_pit_universe.py`、[Universe domain](../../../knowledge/01_Domains/Universe.md) 的 source authority；逐期比較預期／實際鍵集合、下市原因與缺價；測試存在不證明正式輸入有 lifecycle evidence。
   - 出事案例／限制：P2 評估「限制」與 P3 工程均明列下市覆蓋／原因未驗證及 survivorship 未知，不能宣稱偏誤已消除。

7. **Raw vs 調整價、市值非 PIT**：分清價差收益與含息總報酬；現在股數×歷史價不能升格為 PIT 市值，也不能靜默換價源改善結果。
   - 怎麼檢查：讀 `V6/research_lab/portfolio_eval.py` 的價格模式與 `portfolio_engine.py`，用 `rg -n 'adj_close|total_return|market_cap|prices_raw' V6/research_lab` 追欄位來源，核 corporate actions／股數 effective date；缺 adjusted／PIT source 即列限制並對嚴格可交易宣稱 STOP。
   - 出事案例／限制：P2／P3 評估使用 raw、無股利調整、現在股數快照非 PIT；quant checklist A2／A6 是待核問題，不能推定本專案已做到。

8. **共同列交集 common_rows**：比較須同 fold／date／stock 與相同 label identity；A／B 要所有比較系統與 C0+ 完整相同列，不能先裁列再說完整；C 才採全比較臂共同交集並報 input／common／excluded 數與比例。
   - 怎麼檢查：讀 `p4_eval.py:common_rows` 及 family 分支；跑 `test_p4_full.py::test_common_rows_reports_and_label_fail_closed`、`test_main_family_rejects_missing_c0_row`；也核對缺 seed、重複鍵與空交集拒絕。
   - 出事案例：P4a d82d020 複審指出完整性與交集語義不同，修補後 A／B 缺列直接拒絕，C 必須揭露排除比例。

9. **種子排名必須在交集上重算**：先裁 raw seed scores 到共同支援，再逐 fold／date percentile rank、平均種子；C0+／W60 同支援，不使用先全域排名後裁切的 ensemble。
   - 怎麼檢查：跑 `test_p4_full.py::test_family_c_ranks_seeds_after_intersection` 與 `test_p4_rehearsal.py::test_family_c_tail_nan_and_distinct_listing_support`；核 tie-break 與打亂輸入後結果一致。
   - 出事案例：P4a d82d020 修補的合成例 common_rows=5、input_rows=10、excluded_fraction=.5；改變排名母體確實會改比較。

10. **預先登記、凍結、曝光與多重比較**：結果前凍結 hypotheses、support、配方、seeds、selection、family／Holm 與門檻；看過結果後不修改原規則，另案登記並承認曝光。統計顯著不會恢復未曝光身分。
    - 怎麼檢查：比 `tasks/p4a-colab-contract-20261121.json`、`tasks/spec-mamba-portfolio-p3-20261120.md` 與結果時間／source identity；跑 `test_p3.py::test_p3_holm_significant_ci_is_not_sufficient`、`test_p4_full.py::test_eval_families_holm_and_shared_support`。P3 Holm2、P4 A/B/C Holm3/2/4 是各自契約，不可挑較有利 family。
    - 出事案例／限制：P3 評估與 P4a 完整佇列報告已承認歷史曝光、replication 及 C0+ 調參不對稱；措辭用「探索性／描述性、此配方未證實優勢」，不能稱獨立確認或「Mamba 無效」。確認性須另有凍結規則與真正未曝光／前瞻資料，不沿用舊提議日期追認 T0。

11. **評估者與實作者不同 session**：builder 自查不是獨立驗證；不把辯護／預期答案交給 verifier，只交原需求、diff、machine artifacts 與驗收標準。
    - 怎麼檢查：核 producer／reviewer job/session identity、原始 evidence hash 與 GO／NO_GO；讀 `test_p3.py::test_revision2_review_separate_command` 與 [verifier](../../agents/verifier.md)。無授權派 agent 時回報獨立審查待辦，不自行假造 reviewed_by。
    - 出事案例／限制：P3 修訂2新增獨立 review artifact；P4a 報告多輪仍 NO_GO／NOT_ACCEPTED，作者測試通過不解除。

12. **合成資料包含真實壞形狀**：尾端5／10日 NaN、新上市歷史不足、缺 session／不同長窗支援、極端 raw 值、±inf／常數與空成熟支援都要測；正常分布小矩陣不足以證明真實路徑安全。
    - 怎麼檢查：讀／跑 `test_p4_rehearsal.py::test_real_seq_support_tail_nan_and_listing`、`test_seq_runner.py::test_windows_gaps_and_alignment` 與上述 raw clipping regression；新增變更的壞形狀反例應先失敗，再驗實作。極端值／非有限值分別驗轉換與 fail-closed。
    - 出事案例：P4a Q3 是首次長窗真資料才撞 NaN；CPU 排練替身不能證明 CUDA kernel、全量推論、Drive 或正式資料都通過。

## 回報

逐項附 contract／source／input identity、檢查命令、實際數值、evidence class、未驗證範圍。任何 UNKNOWN／STOP 不用0、補列或改門檻繞過；完成不等於 Human acceptance。
