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

**預設 B：T0 = 合格交易所日曆中，嚴格晚於 max(E、價格可用末日、Human 凍結日期、前置驗收完成日期) 的第一個交易日。** 所有日期按 Asia/Taipei，取整日排除；若 11/24 完成所有條件，候選為 11/25，尚須正式日曆確認，不在 DRAFT 寫死。凍結前重盤點曝光帳，若有更晚分析立即右移 E；未知曝光來源即 STOP。Human 可選 A 的日期作描述性延伸，但除非有當日以前的可驗證凍結與訊號封存證據，A 不得成為確認性樣本。Human 凍結須記錄 UTC／台北時間、批准者、文件 SHA、曝光清單與 hashes、最終 T0、日曆 identity；本輪不填造批准。

## 假設表與被比較系統

| ID／角色 | 母體與比較 | H0／H1 | 預先指定用途 |
|---|---|---|---|
| H-F1（唯一主要） | T0 起固定期間，C0+ 的 P2 主要格淨日報酬 − 同一可交易宇宙、比例成本等權基準淨日報酬 | H0: μ≤0；H1: μ>0，方向性科學假設 | 唯一可進獨立前瞻確認的比較；統計 p 採雙邊保守判準，見下節 |
| D-F1 | 同一 C0+ 的 P1 主要格、成本分解、IC5／IC10、風險與逐月表 | 無新增正式假設 | 描述性，不分 alpha、不選 winner |
| D-F2（預設不納入） | 神經模型／P4b 混合 | 無 | 不需 GPU 的 P2 前瞻不依賴此臂；若 Human 納入須在 T0 前凍結權重／特徵／推論與新假設身分，只可描述性，不能事後加入正式 family |

C0+ 是 **P1 加上時間上下文的 LightGBM**，不是 `lightgbm_day_t`。配方版本 `ace2eef0c580d3484a4c6299de16b1c3d5420898` 的 `rolling_scores.collect → c0plus.run → matrix_baselines.Context` 與原 manifest；LightGBM **4.7.0**、NumPy 2.5.4／pandas 3.1.1／pyarrow 26.0.1（舊 P1 環境記錄，未在本輪重建）。三種子 17／29／43，各自每日截面 percentile rank（average ties）再等權平均。

特徵為 frozen manifest 的有序 48 欄（order SHA `7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696`），另加 c0plus.CORE 16 欄各 lag {1,2,5,10,20} 與 trailing mean {5,20,60}（最低半窗觀測），及 MARKET {Return_5d,Return_20d,Volatility_20d} 當日 mean/std/positive-fraction 共 9 欄，合計 **185 欄**。不加入產業、額外財務或新特徵；dense session×stock 缺日保留 NaN，不壓縮時間。用原 X 的 causal per-session／expanding-macro 處理，不套神經 rank-gauss；不可把 V6 56/59 欄 builder 當同一矩陣。

LightGBM 固定：n_estimators=200、num_leaves=31、learning_rate=.05、min_child_samples=200、subsample=.8、subsample_freq=1、colsample_bytree=.8、reg_lambda=10、n_jobs=4、verbose=-1，random_state 分別取三種子；target=y5，完整訓練日截面中心化 percentile rank 後才依 seed 抽樣最多 500,000 列；無 early stopping／搜尋。推論共同資格沿用 60 session 歷史資格，不依未來 label 是否成熟決定是否打分。

模型政策二選一：

| 政策 | 明確訓練視窗／更新規則 | 利弊與選擇 |
|---|---|---|
| **固定模型（預設）** | 以 P1 fold_2026 相同 train 支援 **2013-01-02～2025-12-02**、`split==1` 且 finite y5，沿用原三種子抽樣；T0 前一次 CPU 重現配方並保存三份模型與 SHA，之後全期不重訓 | 分離研究選擇與前瞻成效、較容易稽核；模型老化／新 regime 風險高。P1 只輸出分數，沒有已證明可用的序列化模型；重現及序列化是 Blocker，不是假定已有模型 |
| 年度 expanding（替代） | 每年第一個 session 開盤前更新，起點 2013-01-02，train signal session < 該年度第一日 index−20，標籤 exit／publication 已到；沿原每 seed 500k 抽樣，不調參 | 可跟隨 regime、貼近 rolling_folds；新增重訓隨機性與前瞻期間訓練使用，需凍結年度起點、環境與模型 publication 證據。若改選須 T0 前替換固定政策，不可兩者擇優 |

fold manifest 是 base split 契約，不能直接讀其中 train prose 當 fold_2026 的實際訓練窗；上列日期是實讀 `dates.npy` × `splits.npy`，不是 manifest 的 2023 train_end。本次只盤點，沒有重擬合。

## 凍結參數表

| 項目 | 提議凍結值（全部待 Human） |
|---|---|
| 組合 | `p2_primary_config()`：top_k=50、rebalance_sessions=20、hold_buffer_rank=100、等權；首個 T0 訊號後次日 open 首建倉，再平衡相位從 T0 算，不沿用歷史持倉 |
| 排序／凍結持股 | score 降冪、code 升冪決勝；缺分數不補未來值；preserve_frozen_overflow=True，凍結超額不強平、不新增，沿既有 engine 行為並記錄 |
| 成本 | commission=.001425×discount .6（比例 .000855）、每筆 min_fee=20 TWD、sell_tax=.003、單邊 slippage=10bp、initial_capital=1e6 TWD；不建模整股／零股 |
| 流動性／價格 | min_dollar_volume=2e7 TWD，沿 P2 決策當時可得成交值／價格語義；raw close/open；訊號收盤、次 session open，無 same-close fill |
| 基準 | 同一事前合格宇宙等權、同再平衡時鐘、比例成本；benchmark_proportional_only=True，不收最低費；另列基準最低費敏感度但不作判定 |
| 時間與樣本 | 固定 T0 後 **252 個連續排定交易日**的淨收益（從首個次日成交收益日計）；不得略過壞日湊滿 252。最低 6 個月／126 日僅是替代方案，預設不選 |
| 統計 | 配對日差，circular block bootstrap block=20、1000 draws、seed=17、95% 雙邊 percentile CI；同源 `significance.paired_ic_test` 的 centred two-sided `p_boot`，family=1，α=.05 |
| 標籤／purge | next-open／第 h 個持有 session close，h=5/10；保守封存成熟截止 signal 後 h+1 sessions，實際 exit 與 publication 時間仍必驗；train/eval purge=20 sessions |

其餘 Config 固定 weighting=equal、weight_cap=1、sessions_per_year=252、bootstrap_samples=1000、bootstrap_block=20、seed=17、benchmark_code=null；不准對分數加權持倉。

全參數須在後續 machine contract 顯式列出（含 engine Config 全欄與 source hashes、calendar、source timestamps／model IDs）；不允許默默繼承可變 defaults。固定版本重現與目前 engine 相容性未驗，不宣稱本 DRAFT 可直接執行。

## 特徵延伸 Blockers 與前置驗收

閱讀 `V6/scripts/build_feature_matrix.py` 可見：存在快取即讀舊檔並返回，`--force` 才 merge_all_data → build_features → clean_and_scale → 整檔 to_parquet；沒有 append API，而且會讀 `.env`、寫正式 V6 cache，**本任務未執行，也不適合作為前瞻程序直接使用**。`V6/experimental/v7_corrected_e5_matrix.py` 消費 immutable prepared features／universe／labels，全量 build_arrays／write_matrix；`rolling_folds.py` 只產生 split views；`p4_bundle.prepare_dataset` 只拷貝／驗證已有矩陣與 rank-gauss。三者均不證明可增量產生 9/11 後的 48 欄。

因此「現有矩陣建置程式支援增量追加」是 **尚未確認、目前 source 未見此功能**。需要另案選擇 (1) 從已封存 as-of inputs 重建至新截止日的獨立矩陣；或 (2) 新增可重播的 feature-only 增量程序。預設先驗證 (1) 為 CPU reference，再驗 (2)；不能改正式 Data 或覆寫 matrix_v1，不能靠已成熟 label 決定當天列集合。前置 Blockers：

1. 固定 48 欄與 185 欄 reference、input vintage／publication、模型 serialization／load／predict identity、正式日曆／lifecycle、每日封存及結果封盲 owner 均待驗收。
2. 延伸前後 **舊區間逐位元一致**：固定原始 inputs、日期／code 排序、dtype／缺值 mask／每個舊 row 的 X 48 欄與 C0+ 185 欄 bytes、keys、eligibility、舊 split，與模型 score／rank／逐日報酬逐筆相等並記 SHA。不得只比列數或容許誤差。若舊區間因 mutable cache 回補而改變，STOP，另建 revision identity，不覆寫舊矩陣。
3. 標籤獨立 version：原本已成熟 labels 必須 bytes 一致；舊尾端未成熟 NaN 日後成熟的新增值必須保留原版與成熟版／publication ledger，不能悄悄修改原檔，也不能要求未成熟 labels 永遠不變。feature-only append 不讀未來 labels。
4. 未來 OHLCV／label／universe mutation 不得改歷史 features／既有 predictions；追加切片與全量 reference 等價、重跑／中斷／重複日拒絕、source identity mismatch／缺時戳／未成熟 train label 全部 fail closed。每一個壞行為須有可失敗合成測試，再授權實作。
5. 每日訊號在下一開盤前以資料 cutoff／as-of、keys／來源 hash、三模型 ID、分數 hash、availability 記錄落盤；僅 append，不覆寫。過時／遲到的訊號不能回填成成功交易。驗收只用合成／歷史已曝光資料，不先跑 T0 後績效。

## 成熟時程、首次檢驗與檢定力

signal d 收盤後決策，d+1 open 入場，y5／y10 最早於第 5／10 個持有 session close 可計；凍結的安全延遲為 d+6／d+11 sessions 且來源已發布，缺停牌／exit evidence 時維持未知，不能補成 0。20-session purge 是訓練邊界，不是把前瞻前 20 日任意刪除。組合收益不以 label NaN 篩選；IC 只在共同成熟標籤支援上計算並報尾端排除。末日清倉／收益 cutoff 須在 machine contract 用 engine 語義明定並合成驗收，不事後延長持有挑終點。

第一次且唯一正式開盲：252 個排定收益 sessions 完整，且最後 signal 的 10d label 安全延遲與 source publication 已完成後；至少再等 11 sessions，需更久則據實等待。Human 凍結時依 calendar 寫實際 signal／return 起訖及 maturation 日；DRAFT 不預測曆日完成日。

以 [P3 報告](../reports/v7_p3_portfolio_evaluation.md) 的 703 配對日估計：Mamba2 差值 CI [−.0414%, .0084%]，半寬 .0249 個百分點／日，SE≈.0249/1.96=.01270 個百分點；crossstock 半寬 .03205，SE≈.01635。假設方差與自相關相同，半寬隨 √(703/n) 縮放，126 日約 .0588／.0757、252 日約 .0416／.0535 個百分點。若欲偵測 +.02 個百分點／日、雙邊 α=.05、80% power，n≈703×[(1.96+.842)×SE/.02]^2，約 **2,227／3,690 日**。這是舊模型差值的粗略設計估計，非 C0+ 對等權的保證，不能把日數當獨立樣本或宣稱已有 80% power。P2 H-P2a 的 703 日 CI 半寬 .0554 更寬；故六個月尤其可能無法證實。預設 252 日是可行觀察窗，不是足夠檢定力承諾；Human 可在凍結前選更長固定窗，不能看結果後續收至顯著。

## 判定規則、多重比較與停止

H-F1 的科學方向為單邊正向，但**統計預設雙邊 p**（沿現有 shared 函式，較保守）：僅當完整資料 gate 通過、95% CI 下界 >0 **且 p_boot<.05**，才記「在本預先登記期間及 raw 模擬假設下，P2 主要格相對指定基準的前瞻淨超額獲得確認性支持」。其餘記「未證實」；gate 不過記 `STOP / INVALID_FOR_CONFIRMATION`，不以未證實混淆無效資料。負向 CI 不得用方向性正向假設宣稱勝出。若改採單邊 p，須凍結前另寫 centred bootstrap 尾端公式與測試，不可將既有雙邊 p 隨手除二。

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

## Open Questions／需要 Human 決定

1. 選預設較嚴格 B 的 T0 公式，還是將 11/20 另列描述性？請確認曝光帳完整性與正式 calendar；現在沒有能追認 A 確認性的證據。
2. 固定 fold_2026 三模型（預設）或年度 expanding？接受模型老化與 CPU 重現／保存模型前置驗收嗎？
3. 固定 252 sessions（預設）或在無結果下選更長窗？126 日替代僅低 power，不能承諾顯著；是否接受唯一正式 endpoint 與雙邊保守 p？
4. 是否同意不納入神經描述性臂（預設）、P1 只描述性、raw 為主要、基準只收比例成本？
5. 誰擔任封盲保管者、如何封存 daily inputs／scores 並阻止績效偷看？T0 前須凍結權限／開盲審計。
6. 確認 100% coverage／0 漏日、lifecycle／下市／publication gate 與事故失效政策；目前缺證據時接受 STOP，而非放寬門檻？
7. 另案批准資料管線／模型持久化／machine contract 的實作與驗收，完成後才凍結日期並啟動；本文件完成不等於 Human acceptance。
