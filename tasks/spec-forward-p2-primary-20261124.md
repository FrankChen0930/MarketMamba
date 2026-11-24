# 預先登記草案：P2 主要格前瞻確認性檢驗

**DRAFT — 待 Human 凍結；尚未登記生效，非執行授權。E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 草案日期 2026-11-24。本研究尚無分數、績效或檢定結果；凍結前不得產生本研究結果。下文數字僅為既有證據盤點與設計估算。凍結後僅追加具時間與理由的修訂，不回寫原文；接觸結果後的語義修改須另開研究並使用新的未曝光期間。

## 建立前盤點與身分

依序查閱 AGENTS.md、CLAUDE.md、Current State、Authority Map、[P2 規格](spec-lowturnover-p2-20261119.md)、[P3 規格](spec-mamba-portfolio-p3-20261120.md)、[P4a 規格](spec-p4a-colab-20261121.md)、P1/P2/P3 報告、machine contract、portfolio_eval／p4_eval／rolling_scores／c0plus／matrix_baselines／rolling_folds、矩陣建置程式及 Portfolio／Features／Research／Data domain notes。既有 `tasks/` 可承載文件，`V6/research_lab/portfolio_eval.py:p2_primary_config` 可承載組合語義；不另建框架、目錄或 worktree。但 P2 是已曝光歷史資料、P3 是 21/24 的探索性結果、P4a 是已曝光 replication，都不能直接改名承接前瞻確認性。故只在既有 tasks 新增此研究身分 `forward-p2-primary-20261124`，不修改歷史規格或 artifact。

盤點工作樹：`job-run-mwuc1teo-b3dba8/spec_writer_fwd_p4b`，detached HEAD/base `b7e4f50d4f061cce32f35f466de7ab992ffc35ed`；初始 clean；既有 worktree 已服務此任務。WSL-native，`MARKETMAMBA_DATA_ROOT` unset；本 worktree 的 `Data/` 是普通目錄，沒有 `.artifacts`。只讀主 checkout `/home/frank/projects/MarketMamba/.artifacts`、本機 Windows 價格與既有上傳副本，沒有把不同 checkout 推定等價。盤點 Python `/home/frank/.venvs/marketmamba-gpu/bin/python`（Python 3.12.3），僅 JSON／NumPy／Parquet CPU 讀取，沒有載入 GPU 或訓練。本文件不授權資料取得、排程、Colab、Drive 操作或實作。

## 未曝光定義、曝光盤點與 T0

「未曝光」必須同時滿足：該交易日的模型分數、標籤、策略／基準報酬、持有期損益、IC、績效摘要及任何可供選擇規則的分析，都未被任何研究參與者用於設計／選擇；原始價格存在與績效已曝光分開記錄。不能只看訊號日、矩陣末日或檔案修改時間；持倉延伸的收益日也算曝光。每日特徵與分數只能在事先凍結規則下、以當時可得資料產生並封存，不能事後補打分冒充前瞻。人工持有相關市場知識仍是無法完全排除的限制。

本機唯讀盤點時段 `2026-11-24T14:50:41+08:00～2026-11-24T14:57:22+08:00`，`observed_at=2026-11-24T14:57:22+08:00`；以下只證明所列本機 artifacts，不證明雲端或未提供輸出的全域完整性：

| 證據／凍結 | 已涵蓋的最後日與身分 | 含義 |
|---|---|---|
| P1，規格 2026-11-19 | `p1/full-scores-ace2eef` 與 `full-scores-d2390b5` 各 19 個 Parquet（18 seed 檔＋平均檔）皆最晚 2026-09-11；正式 manifest SHA `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8` | 分數日期不是評估收益終點；d2390b5 無完成 manifest，不升格正式結果 |
| P1 評估 | `p1/eval-full-ace2eef.json`（1,354,871,936 bytes），SHA `275a2595b6ee9dab9a989a67e297c568e9aa27fd07c3f2124dfa557ca8f42135`，逐日收益到 2026-11-18 | 已曝光持有期 |
| P2，規格 2026-11-19 | `p2/eval-p2-fcfc0e6.json`，SHA `df71de0e9440fcf759e79345cd555e6ed6e73e229898bdf4216beb375179818f`，逐日收益到 **2026-11-19** | 晚於矩陣兩個月，不能從 9 月開始前瞻 |
| P3，規格 2026-11-20 | `p3/scores-c323e79` 21 個 seed Parquet 最晚 2026-09-11；`explore-21of24/eval-explore.json` 收益到 2025-12-30，SHA `c607522c4861e0c0966661623de83f19548fbe5885a39a28a644d714eff34860` | 2026 單一完成 seed 也算分數曝光；探索性不升格 |
| P4a，2026-11-22 FROZEN | 本機 `p4/eval-q1q2-partial.json` 收益到 2026-09-03，SHA `a0b3884e048305cfc66f0bb05d250773d6596634fa12d1dccfbd19092dc44b6e` | 已有 Q1/Q2 分析；不在此重報績效，不把 partial 當完整正式 run |
| P4a 矩陣 | 本機上傳副本 `/mnt/d/Desktop/work/ProjectForMe/MarketMamba-drive-upload/data/matrix_v1/DATASET_MANIFEST.json`，2183 bytes，SHA `bc6401095260e7d4153060e3106fff8dbae88cfb6df3f349b808b7dd762c7d9f`；其 `shared/dates.npy` 實讀 7,423,927 列，2005-01-03～2026-09-11，SHA `b0632562242f107855770742762f15fa1f82b6d8e012dce8e4cfd1d68c641016` | manifest 是 size/hash 清單，本身沒有日期欄；日期取自其綁定陣列，不由檔名推測。X identity 與 p4_bundle.DATASET_X 相同，未重驗全部大檔 |
| 價格可用性 | `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/prices_raw.parquet`，8,877,728 列、231,463,821 bytes，SHA `d2bc2d99a0394bfce91cbda28dc207b0d62e0b8078a48f994b53e798768728e7`，2005-01-03～**2026-11-23**；WSL 主 checkout 同列數／日期 | 價格可用末日不等於模型曝光日，也不是 PIT／下市完整性證明 |

另對上述本機 P1/P2/P3/P4 全部 JSON 掃描逐筆 `date` 欄值：各自最大值依序為 2026-11-18／2026-11-19／2025-12-30／2026-09-03；Parquet 的日期最大值另行讀取，不把 JSON 統計當分數全集。

上述評估／分數聯集的**已證實最後曝光交易日 E=2026-11-19**。候選 A 為其後第一個交易日 **2026-11-20**（觀測價格日期中存在），與 P3 報告當時口頭建議一致；但該建議不是事前凍結紀錄，現在不能追認 11/20 起的資料為已凍結前瞻。候選 B 更保守，排除現有價格末日 11/23，並要求在 Human 凍結與資料管線驗收之後才開始。

**Human 已決定（2026-11-24）採嚴格預設 B：T0 = 合格交易所日曆中，嚴格晚於 max(E、價格可用末日、Human 凍結日期、前置驗收完成日期) 的第一個交易日。** 所有日期按 Asia/Taipei，取整日排除；若 11/24 完成所有條件，候選為 11/25，尚須正式日曆確認，不在 DRAFT 寫死。凍結前重盤點曝光帳，若有更晚分析立即右移 E；未知曝光來源即 STOP。Human 可選 A 的日期作描述性延伸，但除非有當日以前的可驗證凍結與訊號封存證據，A 不得成為確認性樣本。Human 凍結須記錄 UTC／台北時間、批准者、文件 SHA、曝光清單與 hashes、最終 T0、日曆 identity；本輪不填造批准。

## 假設表與被比較系統

| ID／角色 | 母體與比較 | H0／H1 | 預先指定用途 |
|---|---|---|---|
| H-F1（唯一主要） | T0 起固定 252 個收益 session，C0+ 加半年重訓流程（policy）的 P2 主要格淨日報酬 − 同一可交易宇宙、比例成本等權基準淨日報酬 | H0: μ≤0；H1: μ>0，方向性科學假設 | 唯一可進獨立前瞻確認的比較；統計 p 採雙邊保守判準，見下節 |
| D-F1 | 同一 policy 的 P1 主要格、成本分解、IC5／IC10、風險、逐月表與切換前／後分段 | 無新增正式假設 | 描述性，不分 alpha、不選 winner |
| D-F2（預設不納入） | 神經模型／P4b 混合 | 無 | 不需 GPU 的 P2 前瞻不依賴此臂；若 Human 納入須在 T0 前凍結權重／特徵／推論與新假設身分，只可描述性，不能事後加入正式 family |

C0+ 是 **P1 加上時間上下文的 LightGBM**，不是 `lightgbm_day_t`。配方版本 `ace2eef0c580d3484a4c6299de16b1c3d5420898` 的 `rolling_scores.collect → c0plus.run → matrix_baselines.Context` 與原 manifest；LightGBM **4.7.0**、NumPy 2.5.4／pandas 3.1.1／pyarrow 26.0.1（舊 P1 環境記錄，未在本輪重建）。三種子 17／29／43，各自每日截面 percentile rank（average ties）再等權平均。

特徵為 frozen manifest 的有序 48 欄（order SHA `7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696`），另加 c0plus.CORE 16 欄各 lag {1,2,5,10,20} 與 trailing mean {5,20,60}（最低半窗觀測），及 MARKET {Return_5d,Return_20d,Volatility_20d} 當日 mean/std/positive-fraction 共 9 欄，合計 **185 欄**。不加入產業、額外財務或新特徵；dense session×stock 缺日保留 NaN，不壓縮時間。用原 X 的 causal per-session／expanding-macro 處理，不套神經 rank-gauss；不可把 V6 56/59 欄 builder 當同一矩陣。

LightGBM 固定：n_estimators=200、num_leaves=31、learning_rate=.05、min_child_samples=200、subsample=.8、subsample_freq=1、colsample_bytree=.8、reg_lambda=10、n_jobs=4、verbose=-1，random_state 分別取三種子；target=y5，完整訓練日截面中心化 percentile rank 後才依 seed 抽樣最多 500,000 列；無 early stopping／搜尋。推論共同資格沿用 60 session 歷史資格，不依未來 label 是否成熟決定是否打分。

## 重訓政策（Human 已選半年更新，細節仍待凍結）

H-F1 檢驗的是 **C0+＋半年重訓 policy** 的整段收益，不是任一模型的固定效果，也不是重訓相對不重訓的因果效果。Human 於 2026-11-24 排除固定模型為主要方案；以下時鐘、延遲與事故數字是 DRAFT 提案，尚未獲得細節凍結或執行授權。

### 日曆與初始模型

提議採 **T0 錨定、每六個曆月**，而非另選一月／七月：令 A_j = T0 加 6j 個曆月（月底不存在則截到該月末日），R_j 為合格交易所日曆中 A_j 當日或其後第一個交易日。R_0=T0；納入其訊號仍可能影響固定收益窗的所有 R_j。初始三模型必須在 T0 前完成訓練、歷史重現、封存與驗收，供 T0 收盤訊號使用；「T0 當時的訓練」指以 T0 為 origin 的 as-of 配方，不是沿用 fold_2026 的舊模型，也不允許等看到 T0 收益才訓練。

實際 T0 尚缺 Human 凍結／前置驗收日，**不可聲稱已推導出正式日期表**。凍結時必須由同一 calendar identity 展開 R_j、purge 截止、完成期限、signal／entry／252 收益起訖及成熟日，寫入 machine contract。以下僅條件式範例：若嚴格公式與正式日曆最終給出 T0=2026-11-25：

| 版本 | 曆月錨點 A_j | 排定 origin R_j | 訓練訊號最晚 session | 切換／期限 |
|---|---|---|---|---|
| M0 | 2026-11-25 | 2026-11-25（仍須日曆確認） | index(R_0)−21 | T0 前驗收；T0 訊號開始使用 |
| M1 | 2027-05-25 | calendar.next_on_or_after(2027-05-25) | index(R_1)−21 | R_1 收盤訊號起，最遲 R_1+2 sessions |
| M2 | 2027-11-25 | calendar.next_on_or_after(2027-11-25) | index(R_2)−21 | 同規則；只有影響預定收益窗才納入 |

若 T0 改變，整表依公式重算，不挑有利月份。**T0 太靠近一月／七月不另觸發一次重訓**：沒有第二套固定半年日曆，首個更新永遠是 T0+6 個曆月；不得手動插入／略過 origin。此法避免初始模型剛建好幾日就重訓。替代的一月與七月首交易日方案雖易提醒，需額外最短間距／跳過規則，故本草案不選；年度更新也不符合已選半年頻率。此選擇仍須 Human 凍結。

### 訓練視窗、資料與封存

`rolling_folds.make_fold_splits` 的實際預設為 `purge_sessions=20, label_horizons=(5,10)`；條件 **train_session < origin_session−20**，最晚為 origin−21，而非 origin−20 或扣 20 曆日；並驗 `train_last + (max(h)+1) < eval_first`。程式使用觀測日期聯集，前瞻必須另驗與合格 calendar 完全對齊，不能假定等價。原年度 CLI 不能直接產生半年／任意日 origin；需另案薄 adapter 與逐邊界測試，不能宣稱現有 CLI 已支援。

每次均從 **2013-01-02 expanding** 至上述 cutoff；沿 P1 base 非 warmup 資格、`split==1` 且 finite y5，完整合格 train 日截面 rank target，再依 17／29／43 各抽至多 500,000 列。5／10 日的 exit 與 publication 都須早於 origin 且在固定資料快照中已可得；訓練 target 仍僅 y5，不改為雙 head，也不藉未成熟 y10 任意縮小 P1 fit 母體。若無法證明安全成熟，該次驗收失敗。特徵 48→185、超參數、環境與抽樣方法完全沿上節 P1；不調參、不 early-stop、不挑 seed。前瞻期間已成熟的 labels 可依政策進入下次 train，但不向研究者顯示策略績效。

資料快照最晚 as-of 為 R_j 前一交易日收盤後的事前固定 publication cutoff（提議台北 18:00）；train signal cutoff 固定 origin−21。遲到或更正資料不得在重試時偷偷換入；缺快照即依失敗政策處理。每版封存 source commit／dirty bytes、環境 lock、calendar、全量合格／抽樣 keys、features／labels／vintage、模型三份 bytes/size/SHA-256、spec/config 與輸出 hashes。先 local serialize/fsync/hash，再於獨立研究 namespace durable copy/reverify，驗 load→predict round-trip，最後 atomic publication pointer；保留上一版，不寫 V6/models，不覆寫歷史 artifact。

**每次先封存，通過歷史已曝光資料重現檢查才切換**：驗收只用在 T0 前即列入曝光帳的固定 reference。先驗同 source 在原 P1 fold_2026 支援（實讀為 2013-01-02～2025-12-02）的三種子 scores/ranks 與原 artifact 逐位元相等；再以當次 as-of 快照獨立重建／重擬合當版，核 fit keys／模型預測在固定歷史 reference 上逐位元相等、模型 load 後預測相等。不同訓練窗的新舊模型預測本來可以不同，不能要求新模型等於舊模型，也不能用「歷史績效更好」作驗收。模型 SHA 用於封存完整性；若 serialization metadata 不 deterministic，須另記兩份 SHA 並驗參數／樹內容／預測一致，不能冒稱模型 bytes 一致。不能以 T0 後 IC／回報選驗收樣本或模型。

### 切換、延遲與事故

提議新版須在排定 R_j 的台北 18:00 前完成全部無績效驗收，才用於該日收盤資料的訊號；最早執行是次 session open。遲到則沿用上一版，於 R_j+1、R_j+2 session 的同一 18:00 cutoff 依固定順序重試，使用相同 origin／快照／seeds，不重新抽樣；**最遲 R_j+2 的 18:00**。通過後在當日訊號一次切換完整三種子，不混搭；deadline 後即 STOP / INVALID_FOR_CONFIRMATION，不無限延遲、不跳過本次等下一半年。初始 M0 無可沿用模型，T0 前未驗收即不啟動，按嚴格公式重算 T0；已啟動後不得換 T0。延遲不移動後續 R_j。

P2 真正組合入口為 `research_lab/portfolio.py:backtest/_simulate`，不是另一個 `experimental/v7_portfolio_engine.py`。新分數僅影響 T0 起每 20 sessions 的既定再平衡訊號，次 open 執行；若模型切換日不是再平衡日，先封存新版分數，等原相位到期才影響交易。**持倉、現金、NAV、成本、buffer100、frozen holdings 與 pending 訊號不重置**；不因重訓額外交易，不重播已封存的舊訊號。需單一連續組合狀態，禁止按模型版本分段 backtest 後串接收益。基準同步維持原再平衡時鐘。

單純重訓失敗但每日 inputs／舊模型分數仍完整：期限內用上一版、完整記錄失敗與實際切換時間，收益日全部保留；這是事前定義 policy 的結果，不能事後稱為偏離而刪除。資料事故若連舊版也無法產生合格訊號，或估值／成交缺證據，立即 STOP；不可補 0、回填成交、刪缺日或向後延長湊 252 日。僅可用可證明當時已封存的 inputs 技術重播，不能補新知；無法重建完整當時決策則整批確認性失效，原資料保留描述性。完整事故帳須不含收益地封存，期末一次揭露。

重訓不得由績效觸發、取消或改期，亦不得由看過本前瞻績效的人決定；排定日曆與固定機器 gate 決定，執行者只接觸必要 train labels／工程狀態，不可接觸策略收益、效應或排名。獨立封盲保管者管收益輸出與權限；持有訓練 labels 不代表允許看前瞻績效，訓練程序禁止輸出 eval IC／逐月策略報表。既有 `c0plus.run` 會做額外模型／eval 診斷且不持久化 C0+，不能直接無隔離重用 CLI；需保留同配方的 fit/predict/persist 邊界並驗數值等價。人員意外接觸結果仍依開盲事故政策處理，不能換操作者後繼續宣稱盲態。

### 替代方案／描述性對照

固定 P1 fold_2026 三模型可分離更新機制、降低工程與研究自由度，但老化／regime 風險較大，且不符合 Human 已選的半年 policy，**不作主假設**。如要列描述性固定模型影子臂，必須 T0 前另凍結是否收集、初始模型身分／同支援／相位，盲存至同一期末；預設不新增此臂，不事後補打分當前瞻。年度 expanding、績效觸發重訓、選最佳更新頻率均不選。

## 凍結參數表

| 項目 | 凍結值／提案（T0 嚴格與半年政策已決定；細節待 Human） |
|---|---|
| 模型更新 | C0+ 三種子、T0 as-of 初始版；T0+6j 曆月往後首交易日，20-session strict purge、2013-01-02 expanding、同 P1 配方；無績效決策 |
| 切換與失敗 | 全三 seed 封存＋歷史重現 gate，台北18:00；最多延遲2 sessions，期間沿上版，超時 STOP；不改快照／下次日曆／持倉／相位 |
| 組合 | `p2_primary_config()`：top_k=50、rebalance_sessions=20、hold_buffer_rank=100、等權；首個 T0 訊號後次日 open 首建倉，再平衡相位從 T0 算，不沿用歷史持倉 |
| 排序／凍結持股 | score 降冪、code 升冪決勝；缺分數不補未來值；preserve_frozen_overflow=True，凍結超額不強平、不新增，沿既有 engine 行為並記錄 |
| 成本 | commission=.001425×discount .6（比例 .000855）、每筆 min_fee=20 TWD、sell_tax=.003、單邊 slippage=10bp、initial_capital=1e6 TWD；不建模整股／零股 |
| 流動性／價格 | min_dollar_volume=2e7 TWD，沿 P2 決策當時可得成交值／價格語義；raw close/open；訊號收盤、次 session open，無 same-close fill |
| 基準 | 同一事前合格宇宙等權、同再平衡時鐘、比例成本；benchmark_proportional_only=True，不收最低費；另列基準最低費敏感度但不作判定 |
| 時間與樣本 | 固定 T0 後 **252 個連續排定交易日**的淨收益（從首個次日成交收益日計）；不得略過壞日湊滿 252。126 日無法可靠觀察更新後支援，故不選；正式日曆須驗至少一次更新及更新後收益支援 |
| 統計 | 配對日差，circular block bootstrap block=20、1000 draws、seed=17、95% 雙邊 percentile CI；同源 `significance.paired_ic_test` 的 centred two-sided `p_boot`，family=1，α=.05 |
| 標籤／purge | next-open／第 h 個持有 session close，h=5/10；保守封存成熟截止 signal 後 h+1 sessions，實際 exit 與 publication 時間仍必驗；train/eval purge=20 sessions |

其餘 Config 固定 weighting=equal、weight_cap=1、sessions_per_year=252、bootstrap_samples=1000、bootstrap_block=20、seed=17、benchmark_code=null；不准對分數加權持倉。

全參數須在後續 machine contract 顯式列出（含 engine Config 全欄與 source hashes、calendar、source timestamps／model IDs）；不允許默默繼承可變 defaults。固定版本重現與目前 engine 相容性未驗，不宣稱本 DRAFT 可直接執行。

## 特徵延伸、重訓 Blockers 與前置驗收

閱讀 `V6/scripts/build_feature_matrix.py` 可見：存在快取即讀舊檔並返回，`--force` 才 merge_all_data → build_features → clean_and_scale → 整檔 to_parquet；沒有 append API，而且會讀 `.env`、寫正式 V6 cache，**本任務未執行，也不適合作為前瞻程序直接使用**。`V6/experimental/v7_corrected_e5_matrix.py` 消費 immutable prepared features／universe／labels，全量 build_arrays／write_matrix；`rolling_folds.py` 只產生 split views；`p4_bundle.prepare_dataset` 只拷貝／驗證已有矩陣與 rank-gauss。三者均不證明可增量產生 9/11 後的 48 欄。

因此「現有矩陣建置程式支援增量追加」是 **尚未確認、目前 source 未見此功能**。需要另案選擇 (1) 從已封存 as-of inputs 重建至新截止日的獨立矩陣；或 (2) 新增可重播的 feature-only 增量程序。預設先驗證 (1) 為 CPU reference，再驗 (2)；不能改正式 Data 或覆寫 matrix_v1，不能靠已成熟 label 決定當天列集合。前置 Blockers：

1. 固定 48 欄與 185 欄 reference、input vintage／publication、模型 serialization／load／predict identity、正式日曆／lifecycle、每日封存及結果封盲 owner 均待驗收。
2. 延伸前後 **舊區間逐位元一致**：固定原始 inputs、日期／code 排序、dtype／缺值 mask／每個舊 row 的 X 48 欄與 C0+ 185 欄 bytes、keys、eligibility、舊 split，與模型 score／rank／逐日報酬逐筆相等並記 SHA。不得只比列數或容許誤差。若舊區間因 mutable cache 回補而改變，STOP，另建 revision identity，不覆寫舊矩陣。
3. 標籤獨立 version：原本已成熟 labels 必須 bytes 一致；舊尾端未成熟 NaN 日後成熟的新增值必須保留原版與成熟版／publication ledger，不能悄悄修改原檔，也不能要求未成熟 labels 永遠不變。feature-only append 不讀未來 labels。
4. 未來 OHLCV／label／universe mutation 不得改歷史 features／既有 predictions；追加切片與全量 reference 等價、重跑／中斷／重複日拒絕、source identity mismatch／缺時戳／未成熟 train label 全部 fail closed。每一個壞行為須有可失敗合成測試，再授權實作。
5. 每日訊號在下一開盤前以資料 cutoff／as-of、keys／來源 hash、三模型 ID、分數 hash、availability 記錄落盤；僅 append，不覆寫。過時／遲到的訊號不能回填成成功交易。驗收只用合成／歷史已曝光資料，不先跑 T0 後績效。

### 任意 as-of 重現、資源與驗收腳本需求

以下全部是 **尚未實作／未驗收的 Blockers**，完成前不得凍結啟動：

6. 建立「可由截至任一日的資料重現 C0+ 訓練」流程：固定 as-of D、publication ledger 與 immutable inputs，按 origin／purge 產生 train keys、全截面 target、seed 抽樣 keys、185 欄及三模型；重跑、獨立新程序與未來資料擾動不能改舊版本。歷史已曝光日期至少涵蓋年界、半年界、20-session 邊界、缺 session、月底／休市與 label 未成熟；不能以當前 cache 冒充任一日 vintage。舊 P1 原 fold 重現與本 policy 新 origin adapter 各自驗收。
7. 模型持久化／SHA、load/predict、損壞／缺 pointer、partial／缺 seed／錯 origin／錯 matrix／contract 拒絕、kill/restart／orphan 不自動升格，以及版本切換持倉／相位／費用連續性都需可失敗驗收。每次重訓共用同一已凍結 acceptance script（source SHA、非績效 pass/fail、實際數值與 report hash）；首次獨立驗收通過不代表每次自動通過。記錄排定／實際 origin、訓練截止／publication、耗時／RAM、模型 hashes、歷史 reference 結果、pointer／切換日及 delay sessions。此處沒有已存在的腳本名稱或通過證明。
8. **252 日端點的 engine 相容性**：`portfolio.backtest` 目前以 20-session indices 的最後一個排定 entry 作 terminal liquidation，未保證精確252收益日；直接串接模型折也會重置狀態。需另案凍結並合成驗收單一連續研究 adapter 的首日NAV／首日成本、精確252收益日、末日估值／清倉成本及共同基準口徑，不改既有 P2 歷史 engine 語義；不准悄悄縮成240或延到260。未完成維持 STOP。
9. 本機 CPU 耗時估計來源：本輪唯讀 P1 manifest `full-scores-ace2eef/manifest.json`（32,500 bytes，SHA `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`），18 fold×seed jobs／全程 2572.822s；2026 seeds 17/29/43 分別 140.807／131.823／132.032s，合計 **404.663s≈6.74分鐘**。這些 jobs 含 day_t＋C0+ 擬合、feature建置與歷史推論，不是單獨 C0+ 計時；用其作同規模一次三種子重現的粗估 reference，獨立重做兩遍約13.49分鐘，另加原 fold 重現約6.74分鐘。每版最低三次 C0+ fits；現有 collect 實際六 fits。新 cutoff 多資料、首次矩陣建置／hash／落盤不含在估計，不能承諾一小時內完成。未來需在本機 CPU（n_jobs=4、三種子循序、不用GPU）量測並凍結時間／資源預算，確認2-session期限可行；失敗不得改期限救結果。
10. 訓練抽樣 `500000×185×4` 約 **370 MB（353 MiB）** 只是單一 float32 feature payload，下限而非 peak RAM；48欄 full matrix、dense session×stock、LightGBM bins／pandas／推論 buffers、序列 eligibility 與重建 inputs 會額外占用。峰值 RAM、磁碟與全矩陣延伸耗時目前 **UNKNOWN**，需同機量測；不以P2純回測估時替代訓練估時。期限是policy失效界線，不是本輪訓練授權。
11. Human 提醒需求：T0 凍結時交一份靜態 R_j／資料快照日／purge截止／最晚驗收日清單，提議各 origin 前5個交易日由指定 owner 人工提醒既有 Human 一人，當日核對無績效 checklist；失敗／逾期人工回報。owner、管道與責任確認仍 Blocker；只列需求，不建立通知、日曆事件、wakeup、recurring job，不部署或改 scheduler。

## 成熟時程、首次檢驗與檢定力

signal d 收盤後決策，d+1 open 入場，y5／y10 最早於第 5／10 個持有 session close 可計；凍結的安全延遲為 d+6／d+11 sessions 且來源已發布，缺停牌／exit evidence 時維持未知，不能補成 0。20-session purge 是訓練邊界，不是把前瞻前 20 日任意刪除。組合收益不以 label NaN 篩選；IC 只在共同成熟標籤支援上計算並報尾端排除。末日清倉／收益 cutoff 須在 machine contract 用 engine 語義明定並合成驗收，不事後延長持有挑終點。

第一次且唯一正式開盲：252 個排定收益 sessions 完整，且最後 signal 的 10d label 安全延遲與 source publication 已完成後；至少再等 11 sessions，需更久則據實等待。Human 凍結時依 calendar 寫實際 signal／return 起訖及 maturation 日；DRAFT 不預測曆日完成日。

以 [P3 報告](../reports/v7_p3_portfolio_evaluation.md) 的 703 配對日估計：Mamba2 差值 CI [−.0414%, .0084%]，半寬 .0249 個百分點／日，SE≈.0249/1.96=.01270 個百分點；crossstock 半寬 .03205，SE≈.01635。假設方差與自相關相同，半寬隨 √(703/n) 縮放，126 日約 .0588／.0757、252 日約 .0416／.0535 個百分點。若欲偵測 +.02 個百分點／日、雙邊 α=.05、80% power，n≈703×[(1.96+.842)×SE/.02]^2，約 **2,227／3,690 日**。這是舊模型差值的粗略設計估計，非 C0+ 對等權的保證，不能把日數當獨立樣本或宣稱已有 80% power。P2 H-P2a 的 703 日 CI 半寬 .0554 更寬；故六個月尤其可能無法證實。預設 252 日是可行觀察窗，不是足夠檢定力承諾。T0+6月更新通常在約126 sessions，因此約留半年的更新後支援；實際假期、延遲及20-session再平衡會縮短新版持倉暴露。252日足以設計觀察至少一次半年更新，**不足以宣稱驗證長期重訓效果或80% power**；正式calendar展開須列每版訊號／實際影響收益日數，至少有一次成功更新及更新後排定再平衡，否則確認性gate失敗，不能改長度補救。若窗內第二次更新，照已登記日曆執行，不因結果挑是否納入。重訓引入非平穩與跨版本依賴，舊SE縮放只是設計參考，block20/1000也非對所有regime有效的保證；Human 可在凍結前選更長固定窗，不能看結果後續收至顯著。

## 判定規則、多重比較與停止

H-F1 的科學方向為單邊正向，但**統計預設雙邊 p**（沿現有 shared 函式，較保守）：僅當完整資料 gate 通過、95% CI 下界 >0 **且 p_boot<.05**，才記「在本預先登記期間及 raw 模擬假設下，P2 主要格相對指定基準的前瞻淨超額獲得確認性支持」。其餘記「未證實」；gate 不過記 `STOP / INVALID_FOR_CONFIRMATION`，不以未證實混淆無效資料。負向 CI 不得用方向性正向假設宣稱勝出。若改採單邊 p，須凍結前另寫 centred bootstrap 尾端公式與測試，不可將既有雙邊 p 隨手除二。

主端點只有整段252日的 policy−等權配對日均淨差及其CI／p。期末可描述性呈現前126／後126日與依實際切換劃分的各版本期間，兩種分段不必相同；持倉影響可能晚於分數切換，不將差異歸因重訓。分段不另作確認性檢定、不作額外正式假設、不依分段選winner或結束日。

重訓增加的自由度／forking paths 包括：日曆錨點、最短間距、purge off-by-one／publication、expanding起點／截面／抽樣、特徵／參數／seed、資料revision、驗收reference、retry／deadline、切換時間／再平衡、事故缺日及分析分段。上列凍結參數表與重訓章逐一封閉；後續machine contract須完整顯式序列化，凍結後不得按績效或便利性改選。任何未封閉項目都阻擋T0，不能留作執行者現場裁量。

正式 family 只有 1，Holm m=1 等於原 p；描述性項目不參與正式通過，不因結果好而增加假設。禁止中途查看累積報酬、IC、效應 CI／p、逐月績效或以模型排名決定延長；只允許不含收益／label 的 schema、hash、延遲與來源 coverage 檢查，由指定封盲保管者留 audit trail。若要中期效力分析，必須 T0 前新增時點與 alpha-spending／CI 校正；本草案**不允許**。

技術／資料 STOP 可隨時發生，但不是提早宣布勝利。安全停止後不刪除已收樣本、不換 T0、不重置成本／NAV／再平衡相位。若性能被意外開盲、source gate 無法復原、需變更資料語義或主要假設，該批只能描述性；重開前瞻需新規格與全新未曝光期間。

## 成本、raw 限制與資料完整性

成本延續 P2（常見假設，非券商報價），研究成交容量不等於實際可成交。無股利調整的 raw 價格可能把除權息／減資當負報酬，策略與等權暴露不同，不保證偏差抵銷。主要 endpoint 不在途中改 total_return；可封存公司行動與股利供期末描述性敏感度，未驗證時寫 UNKNOWN。若 Human 堅持 total_return 為主要，須在 T0 前另凍結完整來源／as-of／處理契約，不能把 raw 歷史與調整後前瞻混接後宣稱同一 P2。

提議資料 gate（T0 前 Human 凍結數字與合格源清單，不能結果後放寬）：

| Gate | 固定提議與事故處理 |
|---|---|
| 日曆／抓取 | 合格交易所日曆每日清單，交易日抓取漏日=0；必要源 publication／as-of／source hash 完整率 100%；當日 cutoff 以前未知即 STOP，不用觀測價格聯集替代正式日曆 |
| 宇宙／覆蓋 | 每日預先 eligible universe denominator 保留上市／下市／停牌；非停牌必要 OHLCV／features coverage=100%，預測 keys coverage=100%；不把資料缺失股票靜默移出分母。合法不合格／停牌須有事前規則及證據，不算抓取成功的藉口 |
| 價格與標籤 | conflicting duplicates、±inf、錯日期／不合理 OHLC、來源不合格=0；NaN label 只允許尚未成熟且不得控制推論列；持股 missing mark／下市 liquidation 未知不得當零損失或虛構成交 |
| 下市／事件 | T0 前驗證 lifecycle、終止交易日與實際可用退場／估值政策；保留消失股票。缺下市結果證據則前瞻主檢驗 STOP，不能移除虧損列；無漲跌停／可交易證據仍禁止交易就緒主張 |
| 資料事故 | 在開盲前記錄日／code／源／失敗 gate／發現時間；保留全部 revisions 與原預測。as-of 可證明的技術修復只准重播原當時 inputs，不能事後新增可用資訊或回補錯過成交。無法重建當時決策則整批確認性失效；不刪壞日、不以延長換好樣本 |

100% gate 是保守提案，不代表本機 cache 已達標；若合格 PIT universe、公司行動、publication 或下市證據缺失，保持 Blocker，不能借研究 cache 的存在放行。資料 acquisition／scheduler 修補不在本次範圍。

## 不允許的事、證據等級與已知限制

不允許追認 11/20 起為凍結前瞻、用矩陣日期掩蓋收益曝光、選較佳種子／重訓政策／日期／成本、追加權重搜尋、中途偷看、刪除失敗日、以 fill=0 隱藏未知、改正式 Data／V6/models、改 Colab 03_formal_resume 或 Drive。不能把本文件、工程測試或 bootstrap 門檻當成 production／投資授權。

證據上限 **E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED**；即使 H-F1 通過，也只是受限於 raw price／成本／宇宙／執行 proxy 的獨立前瞻研究支持，非 VERIFIED_EXECUTABLE／verified trading、非總報酬保證。限制：小樣本／自相關／regime shift、三種子未涵蓋模型選擇不確定性、舊研究曝光與人工市場知識、mutable non-PIT cache、市值非 PIT、下市與公司行動未全驗、無逐筆成交／漲跌停、整零股與容量未模擬。異常好結果必先稽核 leakage、PIT／universe／label timing／source provenance／selection bias／evaluation window；不能先升格。

## 已決定（Human 2026-11-24）

1. T0 採嚴格預設：晚於 max(已知曝光末日2026-11-19及後續更新、價格可用末日、Human凍結日、前置驗收完成日) 的第一個合格交易日；不追認11/20為確認性。
2. 主研究對象是 **C0+＋半年重訓 policy**；固定模型降為替代／描述性對照。正式假設僅H-F1，Holm m=1；預設固定252交易日，不能看結果續收。
3. 此次只修文件、同步knowledge，維持DRAFT待Human凍結，不訓練、不動Colab／Drive、排程、正式Data／V6/models，不commit／push／merge。已讀P4b：其歷史固定C0+原分數、三混合／Holm3、36-fit subset與64-fit gate皆不受本policy修改，不替換其C0+輸入。

## Open Questions／需要 Human 凍結與前置驗收

1. 確認T0曝光帳、合格calendar及真實凍結／驗收日，展開實際日曆表；接受T0錨定每6曆月（月底截斷／休市順延）與18:00 cutoff、最多延遲2 sessions嗎？本草案不採一月／七月雙日曆，細節尚未決定。
2. 接受252日只能觀察少數更新、power未保證，以及單一整段端點／雙邊保守p嗎？若需更長固定窗，只能在凍結前、無本研究結果時決定。
3. 是否同意神經臂與固定模型影子臂預設不收集，P1／分段只描述性、raw為主要、基準只收比例成本？
4. 指定封盲保管者、隔離的重訓執行owner、手動提醒管道與權限／開盲audit；接受事故不可重建或延遲超時即確認性失效嗎？
5. 確認100% coverage／0漏日、lifecycle／下市／publication gate；缺證據維持STOP。任意as-of重現、舊區間bytes、模型persist／每次驗收script與252端點adapter另案批准並驗收；CPU時間僅粗估、峰值資源UNKNOWN，尚不能啟動。

## 本次修訂身分與限制

2026-11-24，`job-run-mwufq5k5-71212c/spec_writer_retrain`，既有worktree detached HEAD/base `176282179b2428bcc7f9ab1f1b6bd93b0408e164`；沿用已提交草案來源 `8eff9fe`（本輪HEAD另包含後續文件）。初始clean；只修文件、未commit。`/usr/bin/python3` 3.12.3／WSL CPU，只讀source與P1 manifest metadata；`MARKETMAMBA_DATA_ROOT` unset，Data realpath為本worktree的 `Data` 普通目錄，沒有更動正式資料。上方原盤點observed_at與曝光artifacts保留為原輪證據，本輪沒有重新觀測價格／雲端或更新其freshness。E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；完成不等於Human acceptance，獨立審查與工程驗收仍待辦。
