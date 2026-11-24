# 預先登記草案 P4b：神經分數與 C0+ 的 rank-average 混合（描述性）

**DRAFT — 待 Human 凍結；尚未登記生效，非執行授權。E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 草案日期 2026-11-24。**已看過 P3／P4a Q1／Q2 結果，本案是在已曝光資料上的描述性、形成假設實驗；即使通過所有門檻，也不是確認性。** 本輪不產生任何混合分數、績效、相關或檢定結果；凍結前不得出現本案結果。既有結果只作輸入身分與設計背景，不填入新的結果表。

## 建立前盤點與身分

已查 AGENTS.md、CLAUDE.md、Current State／Authority Map、[P2](spec-lowturnover-p2-20261119.md)、[P3](spec-mamba-portfolio-p3-20261120.md)、[P4a](spec-p4a-colab-20261121.md) 規格、[P3 報告](../reports/v7_p3_portfolio_evaluation.md)、[P2 報告](../reports/v7_p2_portfolio_evaluation.md)、`p4_eval.py` 的 `averaged_scores/common_rows/portfolio_rows/load_run`、`portfolio_eval.py:p2_primary_config/p3_pair`、`significance.py` 及既有 `test_p4_full.py`／`test_p4_rehearsal.py`／`test_portfolio.py` 等測試。

沿用 `tasks/` 與既有研究 evaluator，不另建框架、目錄或 worktree。P4a 的 family A/B/C 已凍結，不包含本次 B1/B2/B3；P3 不完整且不是 P4a fit。不能直接擴寫其 family、重標結果或把 P4b 當 P4a confirmation。因此新增文件身分 `p4b-blend-20261124`；後續若獲准實作，僅增加薄 adapter／組裝與驗收，重用 shared 函式，不修改 P3 adapter、P2 engine 或正在進行的 Colab 流程。

本工作樹 `job-run-mwuc1teo-b3dba8/spec_writer_fwd_p4b`，detached HEAD/base `b7e4f50d4f061cce32f35f466de7ab992ffc35ed`，初始 clean。WSL，`MARKETMAMBA_DATA_ROOT` unset，Data realpath 是本 worktree 普通目錄，沒有正式資料／本地 artifacts；只唯讀主 checkout artifacts 與 Windows 本機上傳副本。盤點 Python 3.12.3、CPU metadata 讀取，沒有 GPU／訓練／資料取得／雲端 Drive 操作。Human 表示 `03_formal_resume` 正在執行，本任務不讀寫其 Drive 內容、不改 source／bundle／notebook／budget。

## 輸入與准入（完成 fit ≠ 完整 run ≠ Human acceptance）

| 輸入 | 固定身分與所需證據 |
|---|---|
| P4a Q1/Q2 W60 | mamba2／crossstock／gru × seeds 17、29 × folds 2021–2026，共 **36 fits**，W=60、lr=3e-4、BEST validation IC5；依 [P4a machine contract](p4a-colab-contract-20261121.json) 的完整 recipe，不納入 Q3 長窗／Q4 學習率／Q5 permutation |
| 完成狀態 | Human 已報告 36/36 完成；本機有 `p4/eval-q1q2-partial.json`（SHA `a0b3884e048305cfc66f0bb05d250773d6596634fa12d1dccfbd19092dc44b6e`）證明已有分析曝光。**本輪未觀測雲端、未逐一驗證 36 complete／checkpoint／scores，不宣稱正式完整 run 已驗收** |
| C0+ | P1 `full-scores-ace2eef`，source `ace2eef0c580d3484a4c6299de16b1c3d5420898`；LightGBM 4.7.0、48 base＋時間上下文／market 共 185 features、y5、max_train_rows=500000、seeds 17／29／43；使用各 seed 原分數重算共同支援排名，不重新擬合；不可錯用 `lightgbm_day_t` |
| C0+ manifests | 原 manifest SHA `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`，rank_average.parquet SHA `9ea91f369ea8105dd5652cb18bd270c0a81e259a9ce1587fc8aa45822a51ea21`；各 seed 檔須由原 manifest 核對 bytes／SHA，不能只核平均檔 |
| Matrix | 本機上傳副本 `MarketMamba-drive-upload/data/matrix_v1/DATASET_MANIFEST.json`，2183 bytes，SHA `bc6401095260e7d4153060e3106fff8dbae88cfb6df3f349b808b7dd762c7d9f`；其 dates.npy 7,423,927 列，2005-01-03～2026-09-11，日期 SHA `b0632562242f107855770742762f15fa1f82b6d8e012dce8e4cfd1d68c641016`。本輪只驗 manifest／日期，不宣稱全部矩陣完整重驗 |
| 價格與輸出 | 後續凍結單一唯讀 prices_raw／market_value snapshot 的 size/hash、volume 單位、source/provenance；不能接上每次執行都變的 cache。指定 source commit＋dirty bytes／環境 lock／36 fit lineage／P1／matrix／spec hashes 構成新 manifest；輸出另用 `.artifacts/p4b/<run_identity>/`（只提需求，本輪不建立），不得覆寫 P4a／P3 |

現有 `p4_eval.load_run` 明確要求 **64 fits 全部完成**，不能拿它直接載入僅 Q1/Q2 並宣稱完整 P4a。未來 P4b 需要一個獨立的 **Q1/Q2 36-fit subset loader**，對預先指定 queue subset 與每個 fit 呼叫同源 `resume_fit` 等驗證（manifest/binding、recipe、來源相容 lineage、BEST／history、checkpoint／scores hash、keys／labels），且不修改現有 64-fit gate。缺 fit、來源不符或不能重建 lineage 即 STOP；不補訓、不採用 rehearsal／speed／P3 scores、不要求 Q3–Q5 完成才分析此固定 subset。只在獲准且有可讀的完整本機匯出後實作，勿打擾執行中的 Drive。

## 凍結參數與排名規則

以下是待 Human 凍結的完整設計，**不搜尋權重**。所有 rank 都是在同一 fold／同一 signal date 的共同股票截面內，以分數升冪的平均同名次百分位：`r_i=(平均名次_i)/N`，等價 pandas `rank(method='average',pct=True)`；高分較佳。code 一律字串，同分的持股選擇用 code 升冪，不用 code 打破統計 rank ties。

先逐 seed 驗證相同 keys 與 labels，不能在 seed 之間取交集掩蓋缺檔。以 C0+ 與三神經臂的列集合做**同一個全局交集 S**，三個 B 與 C0+ 都使用 S，不各自找最有利支援。先限制原 seed scores 到 S，再各自做日 rank：`C=mean(r_C17,r_C29,r_C43)`；`N_a=mean(r_a17,r_a29)`，a∈{mamba2,crossstock,gru}。以此定義 `R_C=rank(C)`、`R_a=rank(N_a)`；`R_all=rank((R_mamba2+R_crossstock+R_gru)/3)`。這明確指定先種子 rank 平均、再臂 rank、再混合，不能平均不可比的 raw logits。

| 系統 | 唯一凍結公式 | 用途 |
|---|---|---|
| C0+ | R_C | 三個 B 的唯一單獨比較對象；同一 S 上重跑組合，不拿不同宇宙的舊績效作基準 |
| B1 | .5·R_C＋.5·R_mamba2 | 樹＋單一 Mamba2 臂 |
| B2 | .5·R_C＋.5·R_all | 樹＋三神經臂平均；神經內部各 1/3，不以績效／波動重加權 |
| B3（對照） | .5·R_C＋.5·R_gru | 檢查混合現象是否也出現在 GRU；不因對照角色免做校正 |

`R_all` 的外層 rank 是明確設計選擇，與直接把三個 raw score 平均不同；若 Human 想採無外層 rank，只能在首次產生 B 分數前改 DRAFT 並凍結單一版本，不可兩版都跑後挑選。混合後不再校準權重或分數，也不依 fold 選架構／seed。

共同列語義沿 `p4_eval.common_rows`：keys=`fold,date,code`；拒絕 empty／duplicates、infinite labels、共同鍵 label 不一致；**NaN 與 NaN 視為同一未成熟 label，不以 NaN 丟棄組合列**。有限／NaN 不一致拒絕。所有 model scores 必須 finite；缺 score 不能補 0。IC 額外以所有臂同一 finite label mask 計算，報 used／excluded counts；NaN 處理不得改 portfolio 支援。

`p4_eval.averaged_scores` 可重用 seed 檢查與 seed rank 平均；交集必須發生在原 seed rank 前（沿 Family C 的限制後重排語義），不可先在大宇宙 rank 再切片冒充相同算法。原 C0+ cached rank_average 在縮小支援後不足以重建 seed ranks，所以需要 P1 原 seed 檔。全局 S 不同於 p4_eval 每個 family 的支援，不能宣稱與舊 P4a 結果位元相同；支援相同時應重現 shared 路徑。

| 項目 | 凍結提案 |
|---|---|
| 組合設定 | `p2_primary_config()`：top50、20-session 再平衡、buffer100、等權、min_dollar_volume=2e7、commission=.001425×.6、min_fee=20 TWD、sell_tax=.003、slippage=10bp 單邊、initial_capital=1e6、raw、benchmark_proportional_only=True、preserve_frozen_overflow=True |
| 時序／對齊 | 收盤 signal、次 session open；`portfolio_rows` 同 price slice／universe_eligible 建置；同 fold 起點、相位、終點，無歷史跨折持倉繼承；`p3_pair` 以共同 return dates 配對，逐折列出雙方排除日 |
| 支援門檻 | 各 seed 列集合需完全相同；跨臂取 S 並報各系統 input/common/excluded rows、比例與日期。每 fold／date S 至少保留各輸入 95%，且可排名股票≥50；任一不足則完整報告但 `NOT_ELIGIBLE_FOR_FORWARD_NOMINATION`，不刪該折或另縮家族 |
| Bootstrap | circular block=20、1000 draws、seed=17；`paired_ic_test` 用逐日淨報酬差 vs 0、95% percentile CI、centred **雙邊** p_boot；沿 shared 函式，不另造統計引擎 |
| Multiplicity | 正式登記的歷史比較 family={B1,B2,B3}、m=3，三個雙邊 p 一次 Holm；只對 2021–2023 合併表做這一組校正。逐折／2024–2026／六折合併是描述性，CI／p 標示未作額外多重校正，不當獨立顯著證據 |
| Fold pools | 每折 2021…2026 全報；2021–2023 historical registered pool、2024–2026 descriptive pool、全部六折池分列，2026 部分年度。不能把前者稱本案確認性折；只保留其在舊 P3/P4a 契約的名稱作來源欄 |

其餘 Config 固定 weighting=equal、weight_cap=1、sessions_per_year=252、bootstrap_samples=1000、bootstrap_block=20、seed=17、benchmark_code=null；凍結時記錄 Config 全欄 JSON 與實作 hashes，不能由可變 defaults 補值。

合併時先 `p3_pair` 逐折配對，再依 fold/date 串接，沿 p4_eval 的 pooled circular bootstrap 慣例；block 可能跨年度銜接、跨折模型不同，是已知限制，不另外試多個重抽法選有利版本。報告 actual n、CI／p 的支援與每折長度，不以「6 折平均年化」冒充配對日檢定。

## 假設表、判定與前瞻候選規則

| ID | 比較與方向性假設 | 統計／家族 | 可用措辭 |
|---|---|---|---|
| H-B1 | μ(net B1 − net C0+) >0 | 雙邊 p、Holm m=3 | 歷史描述性混合假設 |
| H-B2 | μ(net B2 − net C0+) >0 | 同上 | 歷史描述性混合假設 |
| H-B3 | μ(net B3 − net C0+) >0 | 同上 | 歷史描述性 GRU 混合對照假設 |

僅在來源／支援 gates 通過、2021–2023 合併差值 **95% CI 下界>0 且 p_Holm<.05** 時，標記「已曝光歷史比較通過登記門檻（非獨立確認）」；否則「未證實」。上界<0 且 Holm<.05 可寫「此歷史協定下落後」，不是架構普遍無效。缺失／驗證失敗寫 STOP／INCOMPLETE，不能補 p=0 或把缺結果當零效果。全部三個假設一次完整報告，不因第一個結果好就停止。

**值得另案前瞻檢驗的預先決策規則**（不是確認性結論）：某 B 必須同時滿足：

1. 上述 2021–2023 合併 CI／Holm 門檻及來源／支援 gates；
2. 2021、2022、2023 **至少兩折**各自日均淨差 CI 下界>0，且三折 mean 皆>0（逐折未校正，僅穩健性描述）；
3. 2024–2026 pool mean≥0，且此三折至少兩折 mean>0；2026 即使較短仍完整呈現，不換權重或排除；
4. 下述多樣性診斷完整，無未知被填成 0、同列錯位或洩漏。

若無 B 達標，STOP 放大混合，保留所有失敗；若多 B 達標，全列為候選，不以最大收益挑唯一 winner。Human 可依預先固定的優先序 **B1→B2→B3** 選第一個達標者另開新前瞻規格（簡單性／原問題優先，非最佳歷史績效），或另登記包含多臂及 alpha 校正的新未曝光研究。不得把它加入已開始的 P2 前瞻主要 family。上述嚴格門檻可能無候選，不在看結果後放寬。

## 多樣性與風險診斷（不增加主要假設）

- 每 fold/date 在 S 上報 C0+ 與各 N_a／R_all 的截面 Spearman 秩相關：每日值、mean／median／分位數、有效 N；常數分數、N 不足或無法算為 UNKNOWN，不能以 0 代表完美多樣性。相關低不證明收益有益。
- 報每個 B／各神經單臂與 C0+ 的 signal top50 overlap=`|A∩C|/50`（eligible不足50則門檻失敗）；另報 buffer／不可成交後實際持倉集合 Jaccard、交集／較小集合數，及權重重疊 `Σ min(wA,wC)`（現金另列），逐再平衡日與全持有日摘要。空持股分母為 UNKNOWN。相同 code tie-break。
- 報 gross/net、交易成本／最低費比例、換手、持股／現金、回撤、每 fold paired days／排除日／source coverage；共同樣本上的單獨神經表現僅描述性。raw、股利與下市限制不因混合消失。

## 實作範圍與驗收（僅列需求，本輪不寫程式）

後續另案只做 36-fit subset admission、共同 support／rank blend adapter、重用組合與統計輸出、完整 manifest／報告。不得重訓、調神經參數、換 checkpoint／window／lr，或更改 P2／P3／P4a frozen engine/default 行為。`common_rows`／`averaged_scores`／`portfolio_rows`／`p3_pair`／`paired_ic_test`／`holm_adjust` 與既有 `resume_fit` 必須同源匯入，必要邊界新增 thin adapter，不複製實作。

驗收前先建立可失敗的合成／mutation cases：

1. 手算 ties／N 百分位、兩 seed 平均／三 seed 平均／三臂等權與 B 0.5/0.5，逐日／逐折隔離；常數分數與 code tie-break、未來 score／label 擾動不改較早 blend。
2. 缺 seed／missing entire fold／重複 keys／nonfinite score／±inf label／有限 label 不符皆拒絕；尾端 NaN 對 NaN 成功、NaN 對有限失敗；portfolio 保留 NaN，IC 僅共同成熟列。
3. 不同支援時先交集後 seed rank 的手算反例，cached rank 再切片必失敗；輸出 input/excluded/common counts，95% gate 與少於 50 gate 可反證，不靜默 dropna。
4. B 退化到同 C0+ 分數的合成案例必須給相同持股／daily returns；共用 engine 的 P2 設定／比例成本 benchmark／最低費／折扣／滑價／凍結持股 regression；不以新 pipeline 改舊結果。
5. 配對僅共同 return dates、重複日拒絕、排除日可追溯；三個 p 的已知 Holm 手算、seed=17／block20／draws1000 可重現，fold pool／2026 partial／缺結果不縮家族。
6. subset loader 只接受指定 36 queue jobs；缺 fit、錯 seed/window/lr、錯 BEST／history/hash／matrix／source lineage、speed／rehearsal／P3 artifact 皆拒絕；原 64-fit loader gate 不變，不改正在跑的 artifact。
7. mutation 必須殺死：權重偷偷改 .7/.3、raw logits 平均、交集後未重排、NaN dropna、更換 C0 baseline 支援、忽略 Holm／減少 m、事後換 fold、少一 fit 仍通過、用 LAST、把 gate 失敗寫成功。每例記錄 expected assertion 與實際失敗，不把環境錯誤當 mutation killed。
8. source/schema syntax、focused regression、原 P4 evaluator 合成回歸、artifact hash/reference links、`git diff --check`、knowledge impact／health 全通過。驗收證據明標 CPU_ENGINEERING_ONLY；未驗實際 inputs 時不得寫研究成功。

## 不允許的事、證據等級與已知限制

不允許搜尋權重、按折／seed 選 winner、刪掉不利年份、混入 Q3/Q4、重標 P3 incomplete、把歷史「confirmatory_years」欄位當新的獨立確認、以低相關直接宣稱增益、將 raw 當 total return、把低滑價假設當成交證據、修改成本或支援門檻迎合單例。

已看 P3／P4a，具有**事後選擇與研究者自由度**；此處登記只能約束尚未執行的混合規則，不能清除既有曝光。C0+ 曾調整且三種子，神經臂只兩種子、固定小配方未調參，**不對稱偏向 C0+**，不能由 B 結果歸因 Mamba 架構優劣。三個 B 分享相同 C0+／labels／dates，高相關、Holm 不等於獨立樣本；六折、部分 2026、block 跨折與有限 1000 draws 均限制精度。

價格 raw 無股利調整、模擬 labels、非 PIT 市值／下市覆蓋未驗、觀測日曆／漲跌停／實際流動性／整零股未完整建模。縮小共同宇宙可能改變投資問題；95% 保留不代表排除沒有偏差。36 fits 的 Human 完成報告不取代逐 fit machine admission。

證據上限 **E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED**；容許結論：「在已曝光歷史資料與固定 P2 成本協定下，B? 呈現／未呈現描述性增益，符合／不符合另案前瞻候選規則。」禁止「確認混合有效」、「可交易」、「Mamba 普遍優／劣」。唯一獨立確認路徑仍是未曝光資料；本案通過不改 [P2 前瞻草案](spec-forward-p2-primary-20261124.md) 的單一主要假設或 T0。

若 Human 看過本案結果後要求調 B、rank、seed、門檻、成本或窗口，原 P4b 完整保留並報告；**只能另開 P4b2、重新登記並揭露已曝光**。歷史 P4b2 仍描述性；想確認須另用全新未曝光期間。不覆寫舊結果、不把新版本稱原預先登記成功。

## Open Questions／需要 Human 決定

1. 是否凍結 B1/B2/B3 的三個 0.5/0.5 公式、seed-rank→臂-rank→混合的順序，以及 B2 的外層 rank？預設如表，不搜尋替代權重。
2. 是否同意全局 S／95% 每日保留率／至少50檔，以及同源 NaN／支援處理？P1 原 seed 分數與 36-fit 本機匯出是否可供之後准入？目前不存取雲端。
3. 是否同意 Holm m=3 只在 2021–2023 pool 作登記門檻，逐折、2024–2026、六折池均描述性，原「確認性折」只作舊契約來源標籤？
4. 是否凍結兩折 CI 正、三折方向一致、近期方向與完整 gates 的候選規則，以及多候選 B1→B2→B3 的簡單性優先序？沒有達標者就停止。
5. 認可 C0+ 調參／種子數不對稱、raw／下市等限制與 E2 上限？是否另案做預算對等研究，不能把它混入本家族。
6. 待 03_formal_resume 安全完成／有既有本機匯出後，另案授權 CPU 實作 subset loader、測試與描述性評估；凍結紀錄需批准者、時間、文件／所有輸入與程式 SHA、環境及 output namespace。現在只交 DRAFT，不執行。
