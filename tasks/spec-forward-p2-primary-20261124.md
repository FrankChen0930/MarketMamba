# 預先登記草案：P2 主要格前瞻確認性檢驗

**DRAFT — 待 Human 凍結；尚未登記生效，非執行授權。E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。** 草案日期 2026-11-24。本研究尚無分數、績效或檢定結果；相應階段凍結前不得產生該階段結果。下文數字僅為既有證據盤點與設計估算。凍結後僅追加具時間與理由的修訂，不回寫原文；接觸結果後的語義修改須另開研究並使用新的未曝光期間。

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
| H-F1（唯一主要） | T1 起固定 252 個收益 session，C0+ 加半年重訓流程（policy）的 P2 主要格淨日報酬 − 同一可交易宇宙、比例成本等權基準淨日報酬 | H0: μ≤0；H1: μ>0，方向性科學假設 | 唯一可進獨立前瞻確認的比較；統計 p 採雙邊保守判準，見下節 |
| D-F1 | 同一 policy 的 P1 主要格、成本分解、IC5／IC10、風險、逐月表與切換前／後分段 | 無新增正式假設 | 描述性，不分 alpha、不選 winner |
| D-F2（預設不納入） | 神經模型／P4b 混合 | 無 | 不需 GPU 的 P2 前瞻不依賴此臂；若 Human 納入須在 T1 前凍結權重／特徵／推論與新假設身分，只可描述性，不能事後加入正式 family |

C0+ 是 **P1 加上時間上下文的 LightGBM**，不是 `lightgbm_day_t`。配方版本 `ace2eef0c580d3484a4c6299de16b1c3d5420898` 的 `rolling_scores.collect → c0plus.run → matrix_baselines.Context` 與原 manifest；LightGBM **4.7.0**、NumPy 2.5.4／pandas 3.1.1／pyarrow 26.0.1（舊 P1 環境記錄，未在本輪重建）。三種子 17／29／43，各自每日截面 percentile rank（average ties）再等權平均。

特徵為 frozen manifest 的有序 48 欄（order SHA `7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696`），另加 c0plus.CORE 16 欄各 lag {1,2,5,10,20} 與 trailing mean {5,20,60}（最低半窗觀測），及 MARKET {Return_5d,Return_20d,Volatility_20d} 當日 mean/std/positive-fraction 共 9 欄，合計 **185 欄**。不加入產業、額外財務或新特徵；dense session×stock 缺日保留 NaN，不壓縮時間。用原 X 的 causal per-session／expanding-macro 處理，不套神經 rank-gauss；不可把 V6 56/59 欄 builder 當同一矩陣。

LightGBM 固定：n_estimators=200、num_leaves=31、learning_rate=.05、min_child_samples=200、subsample=.8、subsample_freq=1、colsample_bytree=.8、reg_lambda=10、n_jobs=4、verbose=-1，random_state 分別取三種子；target=y5，完整訓練日截面中心化 percentile rank 後才依 seed 抽樣最多 500,000 列；無 early stopping／搜尋。推論共同資格沿用 60 session 歷史資格，不依未來 label 是否成熟決定是否打分。

## 兩階段啟動與重訓政策（Human 2026-11-24 新決定；DRAFT）

H-F1檢驗C0+加固定重訓policy的整段淨收益，不是單版模型效果，也不是重訓的因果效果。本節取代前輪 Q1／Q3 錨點及將11月試運行當確認性起點的提案。**T0只指階段0試運行起點；T1指階段1正式T0**。本文件其餘啟動前置要求，依附錄區分T0最低安全要求及T1完整要求；正式252收益日、不得重置起點及確認性失效規則一律以T1為界。

### 階段0試運行、階段1正式啟動與曝光帳

- **階段0**：保留上節嚴格T0公式，約2026-11-25之後；前置未完成就順延，不能追認。以T0當時凍結配方／as-of驗收封存的C0+三種子走每日訊號、封盲、健康檢查、hash chain與事故演練；無確認性檢定、無績效通過門檻，不併入正式252日。不得用試運行績效改特徵、模型、N、成本、日期或任何規則。試運行只驗流程，不代表production部署、實際交易或PIT gate已通過。
- **階段1**：2027年報錨點後排定重訓日為T1，重新以固定配方與T1 as-of擬合、驗收，不沿用試運行模型冒充重訓；T1收盤第一個正式訊號，次session open首建倉／第一收益日。正式組合現金、NAV、成本及20-session相位重新初始化，試運行持倉不轉入；之後更新不得重置。H-F1固定252收益sessions、Holm m=1、雙邊保守p，僅用T1之後的正式收益。
- **預設封盲決定（本DRAFT具體選擇，待全文簽署）**：試運行績效持續封盲，直到階段1唯一正式端點開盲時才可一併描述性揭曉。這比「T1規格與日曆已簽署即可揭曉」更保守，避免中途得知績效影響事故裁量；不新增假設或alpha。選項B為T1規格／日曆簽署且T1已啟動後描述性揭曉階段0，風險是Human兼任owner，仍可能影響正式期間裁量；選項C永不揭曉最能隔離選擇但降低流程研究資訊。預設選A，不在T1開始後自動解盲，也不以是否好看決定是否揭曉。
- **曝光帳**分記資料產生／機器封存與人員可見時間：價格存在、私有程序使用成熟train labels、封存未讀結果不自動等於人員績效曝光；任何人看到可還原策略績效的分數＋價格／持倉／NAV、IC或摘要即記曝光，不只記直接開報表。階段0整段永不納入確認性樣本，即使未揭曉也不回收。T1「未曝光」指T1後訊號／持有收益與分析未先被用來設計選擇，並非要求訓練資料從未存在。階段0成熟labels可依固定policy進train，不能藉此手動檢視績效。
- T0前簽署試運行邊界與禁止績效調規；T1公式、候選日、全年日曆、延後條件及正式全文，必須在**任何階段0績效被看到之前**簽署。預設直到正式端點才揭曉，更不允許用階段0結果改規則。若提前曝光，立即記範圍／時間與可能影響，禁止按結果挑T1；未能證明正式規則／未來樣本未受影響即STOP，另案新預先登記，不靠換owner恢復。
- **試運行失敗／延後**：只依淨化工程gates（缺日=0、完整率100%、hash無衝突、必要來源及演練通過）判斷；T1前至少連續20個排定sessions完整日鏈，並完成WP1～WP6獨立驗收。20日是流程觀察提案，非績效樣本或放寬既有gate。年報重訓日以前無法達成，T1不啟動；取嚴格晚於完整簽署／驗收日且不早於年報R_A的首session為候選，使用該origin的獨立快照重新重訓，原排定／延後原因全部留帳。若超過當年Q2排定origin仍未啟動，該啟動計畫STOP、另案凍結下一年度方案；不得臨場跨錨點選日期。正式開始後不再重設T1、不延長252日。

### 年報／Q2法定期限、例外與資料緩衝

本案「年報」是年度財務報告，非股東會年報。只採[期限研究](research-financial-deadlines-20261118.md)的2026-11-18已核實範圍，不冒充本輪重新查法。**整體更新時鐘選一般12/31公司年度3/31、Q2 8/14**，含當期官方合法順延；每年僅兩錨點，不以Q1／Q3再觸發。這是一般公司時鐘，不是假定全市場同日公告或相隔恰六個月。

年度：實收資本額達100億元上市櫃年度末後75日，非閏年通常3/16；閏年須按期末計算。保險及公發證券／期貨商等75日規則依研究S2～S7分類；FY2025上市金融業2026-03-16為S9年度公告證據，不能倒推所有金融公司每年一律75日（低資本銀行／票券通則有未驗部分）。Q2：一般8/14，銀行／票券／金控／保險／證券／期貨半年報及第一上市櫃有2個月8/31例外。非曆年、境內外／IPO身分、核准延期、假日順延與歷史分類，按公司×期別留帳；未知不猜。

令D_A、D_H分別為當期核實的上述一般年度／Q2期限；**R_A=嚴格晚於D_A第10交易日（N_A=10），R_H=嚴格晚於D_H第8交易日（N_H=8）**，期限當日不算第1日。N仍是DRAFT提案，Human已決定的是兩個錨點，不把作者選10／8寫成已核准。

| N選項 | 年報取捨 | Q2取捨與選擇 |
|---|---|---|
| 1～2 | 更新快，但幾乎無來源核對／重訓容錯；年報較長期限與分散提交不保證已入管線 | 同樣過緊，不選 |
| 8／8 | 統一規則、沿用前輪工程折衷；年報年度版本與例外核對時間較少 | Q2暫採8，仍待實測資源 |
| **10／8（提案）** | 年報多2 sessions作版本／來源與CPU驗收餘裕；代價是更晚模型，不是已驗證最佳值 | 保留Q2第8日，不按績效調整 |
| 10／10或更久 | 統一較保守；仍不能解決延遲申報或缺vintage | 更慢且未證明需要；不以等待代替PIT證據 |

年報申報窗較長、提交分散，既可能較早入管線也可能最後集中；**不能單憑期限較長推導必須／不必多緩衝**。10／8只是非績效工程提案，須在揭曉前以來源延遲與CPU測量凍結，沒有實測改善證據。

獨立資料gate保留：每筆必要輸入的完整公開／入管線較晚時點映射到session A（18:00後或非交易日歸下一session），要求A+2 sessions不晚於R_j前一session的18:00快照截止，並驗source／vintage／修訂。特殊公司較早75日期限不提前整體R_A；8/31、延報、核准延期或非曆年不使整體R_H漂移。若當次有必要財務輸入尚不合格，只依凍結規則沿用合格舊vintage；要求新期資料或必要source缺失即gate失敗，不臨時刪公司／回填／換快照。**目前48／185欄未納入財報，故不憑空要求全市場8/31半年報收齊**；若未來要求新半年報，選8/31等最晚適用期限另案改contract，不能在本案偷偷延後。

### 預期日曆與正式窗內更新推導

本輪沒有合格2027／2028交易所calendar或當期順延公告；以下為**週一至五、無假日／無順延的算術示例**，非可執行日期。所有正式日期須WP1按calendar identity重算並在績效曝光前簽署。

| 階段／版 | 錨點／示例 | 預期與待核 |
|---|---|---|
| 階段0 M0 | 嚴格T0若11/24所有前置過關，候選2026-11-25 | 目前未驗收，實際只能之後；不要求最近Q3門檻，Q3已取消為錨點 |
| 階段1 F0 | 2027-03-31後第10工作日＝2027-04-14 | 真實T1約4月中下旬或工程延後；T1前完成as-of重訓驗收 |
| 正式F1 | 2027-08-14後第8工作日＝2027-08-25 | 真實約8月下旬；距T1約4個月，應在252日內 |
| 正式F2（條件式） | 2028-03-31後第10工作日＝2028-04-14 | 距示例T1有262個工作日；實際是否進窗須計交易所假日與實際T1 |
| 252日端點 | 示例T1=4/14，首收益4/15，第252收益＝2028-03-31 | 無假日下界示例；若該區間少10～20個交易日，端點約延至4月中下旬，不能定作3/31 |
| 正式唯一開盲 | 端點後最末signal安全成熟至少11 sessions且publication齊備 | 無假日保守示例下界約2028-04-17；真實通常更晚，絕不與試運行混算 |

不能沿舊稿宣稱「252日必兩次」。一般calendar每年休市可能使下一年年報重訓落在窗內；示例F2距T1有262工作日，若**實際R_A(2028)相對實際T1的session距離≤251**，其下一日收益才落在252日內，即排入第二次更新（再驗20-session相位有無實際持倉暴露）；否則僅Q2一次。示例錨點不動時，至少11個被排除工作日便滿足這個條件；但2027／2028第10交易日自身也會順延，必須兩端一起重算。若T1因工程延後，F2更可能入窗。規劃按Q2＋次年年報**兩次更新的工程容量**準備，正式次數UNKNOWN，不挑日曆使結果有利。列全部R_j>T1且next-session open≤第252收益日的更新、origin−21、快照／18:00／R+2、每版訊號及再平衡影響收益日數；至少一次成功更新且有更新後既定再平衡，否則STOP，不延長窗口。

### 為什麼C0+的48欄不含季報／營收（文件事實與未驗證部分）

本輪只讀 [build_feature_matrix.py](../V6/scripts/build_feature_matrix.py) 及其實際呼叫的 [feature_engineer.py](../V6/marketmamba/data/feature_engineer.py)：`_merge_revenue` 以月末+11曆日作 `available_from` 再as-of lookup；`_merge_financial_statements` 預設 `fundamentals_v2=False`，一律期末+45曆日，連年度也如此。只有顯式開 `fundamentals_v2=True` 才將12月期末改+90曆日；此script沒有傳該旗標。`merger.py` 的generic helper預設monthly=11／quarterly=45，不能用檔頭「10日」或註解代替呼叫實況。90日不是通用3/31（閏年不同），45日也不是實際公開／入管線或修訂版本證明。**有固定 publication-lag proxy，沒有由這條路徑證明逐公司首次公告、遲報及revision vintage的嚴格PIT處理**；年度預設+45尤其有提前使用風險。

更關鍵的是，本案P1/C0+沿用的 [48欄manifest](../research/v7/corrected-baseline-v1/feature-manifest.json)（order SHA `7d75d4a74c7802e70e9ca88c13f7c2169a4b4fb974847d907c774e82a9da9696`）已在 `dropped_features` 移除Revenue_MoM／YoY、EPS／Surprise、Gross_Margin、ROE等11個財務欄；[Features domain](../knowledge/01_Domains/Features.md)同樣記錄publication證據不足而排除。`c0plus.CORE/MARKET`新增的137欄不含財報／營收；corrected matrix builder只讀prepared features，不重算這些lag。因此不能說「P1目前會在季報後自動吃進新季財報」。V6 59欄builder不是P1 48→185欄的同一contract，matrix舊artifact的生成vintage亦未在本輪重驗。季報後日曆是Human指定的更新時鐘，不能據此聲稱新財報帶來改善。

**文件記載的排除原因**：[Features](../knowledge/01_Domains/Features.md)明載 financial fields without adequate publication-time coverage were removed；manifest列11欄：`PER`、`PBR`、`Revenue_MoM`、`Revenue_YoY`、`EPS`、`EPS_Surprise`、`Gross_Margin`、`ROE`、`Book_Value`、`Dividend_Yield_Fwd`、`Free_Cash_Flow`。固定lag路徑沒有逐公司實際公告時間與修訂歷史，不能證明嚴格PIT。**未驗證部分**：本輪未重建當初排除決策全程，亦未驗證當時是否量測洩漏幅度／各欄影響，不能寫成已證實洩漏多少或財報沒有預測力。

年報／Q2錨定只是更新時鐘，並不代表納入新財報。未來加回基本面須另案取得逐公司×期別×欄位×版本實際公開時間（例如MOPS公告時間，須核實是否完整報表首次公開）、原始及修訂歷史、進管線時間與不可變source hashes，再做公告前不可見、遲報、盤後公告、更正版本及future-mutation的PIT驗收；缺證據fail closed。不以MOPS頁面表頭存在推定完整資料已取得。本規格不取得資料、不擴欄。

即使未來另案納入財報，重訓日與特徵可用日仍是兩個gate：新財報不得回填進公告前歷史train rows；origin−21可能使最近季報尚未出現在當次train尾端，但每日推論可於其當時合格公開／入管線／lag門檻全過後使用。等到報後訓練不會消除歷史修訂洩漏。新增財報欄會改48／185欄語義，必須另案新contract與預先登記，本案不做。

### 訓練視窗、資料與封存

`rolling_folds.make_fold_splits` 的實際預設為 `purge_sessions=20, label_horizons=(5,10)`；條件 **train_session < origin_session−20**，最晚為 origin−21，而非 origin−20 或扣 20 曆日；並驗 `train_last + (max(h)+1) < eval_first`。程式使用觀測日期聯集，前瞻必須另驗與合格 calendar 完全對齊，不能假定等價。原年度 CLI 不能直接產生半年／任意日 origin；需另案薄 adapter 與逐邊界測試，不能宣稱現有 CLI 已支援。

每次均從 **2013-01-02 expanding** 至上述 cutoff；沿 P1 base 非 warmup 資格、`split==1` 且 finite y5，完整合格 train 日截面 rank target，再依 17／29／43 各抽至多 500,000 列。5／10 日的 exit 與 publication 都須早於 origin 且在固定資料快照中已可得；訓練 target 仍僅 y5，不改為雙 head，也不藉未成熟 y10 任意縮小 P1 fit 母體。若無法證明安全成熟，該次驗收失敗。特徵 48→185、超參數、環境與抽樣方法完全沿上節 P1；不調參、不 early-stop、不挑 seed。前瞻期間已成熟的 labels 可依政策進入下次 train，但不向研究者顯示策略績效。

資料快照最晚 as-of 為 R_j 前一交易日收盤後的事前固定 publication cutoff（台北 18:00）；train signal cutoff 固定 origin−21。遲到或更正資料不得在重試時偷偷換入；缺快照即依失敗政策處理。每版封存 source commit／dirty bytes、環境 lock、calendar、全量合格／抽樣 keys、features／labels／vintage、模型三份 bytes/size/SHA-256、spec/config 與輸出 hashes。先 local serialize/fsync/hash，再於獨立研究 namespace durable copy/reverify，驗 load→predict round-trip，最後 atomic publication pointer；保留上一版，不寫 V6/models，不覆寫歷史 artifact。

**每次先封存，通過歷史已曝光資料重現檢查才切換**：驗收只用在T0前即列入曝光帳的固定歷史reference（不含階段0資料）。先驗同 source 在原 P1 fold_2026 支援（實讀為 2013-01-02～2025-12-02）的三種子 scores/ranks 與原 artifact 逐位元相等；再以當次 as-of 快照獨立重建／重擬合當版，核 fit keys／模型預測在固定歷史 reference 上逐位元相等、模型 load 後預測相等。不同訓練窗的新舊模型預測本來可以不同，不能要求新模型等於舊模型，也不能用「歷史績效更好」作驗收。模型 SHA 用於封存完整性；若 serialization metadata 不 deterministic，須另記兩份 SHA 並驗參數／樹內容／預測一致，不能冒稱模型 bytes 一致。不能以 T1 後 IC／回報選驗收樣本或模型。

### 切換、延遲與事故

新版須在排定 R_j 的台北 18:00 前完成全部無績效驗收，才用於該日收盤資料的訊號；最早執行是次 session open。遲到則沿用上一版，於 R_j+1、R_j+2 session 的同一 18:00 cutoff 依固定順序重試，使用相同 origin／快照／seeds，不重新抽樣；**最遲 R_j+2 的 18:00**。通過後在當日訊號一次切換完整三種子，不混搭；deadline 後即 STOP / INVALID_FOR_CONFIRMATION，不無限延遲、不跳過本次等下一半年。階段0初始M0未驗收即依嚴格公式順延T0；階段1 F0未驗收即不啟動，依上節非績效延後程序重算T1；已啟動後不得換 T1。延遲不移動後續 R_j。

P2 真正組合入口為 `research_lab/portfolio.py:backtest/_simulate`，不是另一個 `experimental/v7_portfolio_engine.py`。新分數僅影響 T1 起每 20 sessions 的既定再平衡訊號，次 open 執行；若模型切換日不是再平衡日，先封存新版分數，等原相位到期才影響交易。**持倉、現金、NAV、成本、buffer100、frozen holdings 與 pending 訊號不重置**；不因重訓額外交易，不重播已封存的舊訊號。需單一連續組合狀態，禁止按模型版本分段 backtest 後串接收益。基準同步維持原再平衡時鐘。

單純重訓失敗但每日 inputs／舊模型分數仍完整：期限內用上一版、完整記錄失敗與實際切換時間，收益日全部保留；這是事前定義 policy 的結果，不能事後稱為偏離而刪除。資料事故若連舊版也無法產生合格訊號，或估值／成交缺證據，立即 STOP；不可補 0、回填成交、刪缺日或向後延長湊 252 日。僅可用可證明當時已封存的 inputs 技術重播，不能補新知；無法重建完整當時決策則整批確認性失效，原資料保留描述性。完整事故帳須不含收益地封存，期末一次揭露。

重訓不得由績效觸發、取消或改期，亦不得由看過本前瞻績效的人決定；排定日曆與固定機器 gate 決定，執行者只接觸必要 train labels／工程狀態，不可接觸策略收益、效應或排名。Human兼任封盲保管者與重訓執行owner，依下節隔離收益輸出與權限；持有訓練 labels 不代表允許看前瞻績效，訓練程序禁止輸出 eval IC／逐月策略報表。既有 `c0plus.run` 會做額外模型／eval 診斷且不持久化 C0+，不能直接無隔離重用 CLI；需保留同配方的 fit/predict/persist 邊界並驗數值等價。人員意外接觸結果仍依開盲事故政策處理，不能換操作者後繼續宣稱盲態。

### 替代方案／描述性對照

固定 P1 fold_2026 三模型可分離更新機制、降低工程與研究自由度，但老化／regime 風險較大，且不符合 Human 已選的半年 policy，**不作主假設**。如要列描述性固定模型影子臂，必須 T1 前另凍結是否收集、初始模型身分／同支援／相位，盲存至同一期末；預設不新增此臂，不事後補打分當前瞻。年度 expanding、績效觸發重訓、選最佳更新頻率均不選。

## 凍結參數表

| 項目 | 凍結值／提案（兩階段與錨點已決定；具體值待Human凍結） |
|---|---|
| 模型更新 | C0+ 三種子、T1 as-of 初始版；年報一般3/31後第10交易日／Q2一般8/14後第8交易日（實際公開／入管線另驗2-session緩衝），20-session strict purge、2013-01-02 expanding、同 P1 配方；無績效決策 |
| 切換與失敗 | 全三 seed 封存＋歷史重現 gate，台北18:00；最多延遲2 sessions，期間沿上版，超時 STOP；不改快照／下次日曆／持倉／相位 |
| 組合 | `p2_primary_config()`：top_k=50、rebalance_sessions=20、hold_buffer_rank=100、等權；首個 T1 訊號後次日 open 首建倉，再平衡相位從 T1 算，不沿用歷史持倉 |
| 排序／凍結持股 | score 降冪、code 升冪決勝；缺分數不補未來值；preserve_frozen_overflow=True，凍結超額不強平、不新增，沿既有 engine 行為並記錄 |
| 成本 | commission=.001425×discount .6（比例 .000855）、每筆 min_fee=20 TWD、sell_tax=.003、單邊 slippage=10bp、initial_capital=1e6 TWD；不建模整股／零股 |
| 流動性／價格 | min_dollar_volume=2e7 TWD，沿 P2 決策當時可得成交值／價格語義；raw close/open；訊號收盤、次 session open，無 same-close fill |
| 基準 | 同一事前合格宇宙等權、同再平衡時鐘、比例成本；benchmark_proportional_only=True，不收最低費；另列基準最低費敏感度但不作判定 |
| 時間與樣本 | 固定 T1 後 **252 個連續排定交易日**的淨收益（從首個次日成交收益日計）；不得略過壞日湊滿 252。126 日無法可靠觀察更新後支援，故不選；正式日曆須驗至少一次更新及更新後收益支援 |
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
5. 每日訊號在下一開盤前以資料 cutoff／as-of、keys／來源 hash、三模型 ID、分數 hash、availability 記錄落盤；僅 append，不覆寫。過時／遲到的訊號不能回填成成功交易。驗收只用合成／歷史已曝光資料，不先跑 T1 後績效。

### 任意 as-of 重現、資源與驗收腳本需求

以下全部是 **尚未實作／未驗收的 Blockers**；完整項目完成前不得正式啟動T1，試運行T0亦須先通過附錄最低安全項目：

6. 建立「可由截至任一日的資料重現 C0+ 訓練」流程：固定 as-of D、publication ledger 與 immutable inputs，按 origin／purge 產生 train keys、全截面 target、seed 抽樣 keys、185 欄及三模型；重跑、獨立新程序與未來資料擾動不能改舊版本。歷史已曝光日期至少涵蓋年界、半年界、20-session 邊界、缺 session、月底／休市與 label 未成熟；不能以當前 cache 冒充任一日 vintage。舊 P1 原 fold 重現與本 policy 新 origin adapter 各自驗收。
7. 模型持久化／SHA、load/predict、損壞／缺 pointer、partial／缺 seed／錯 origin／錯 matrix／contract 拒絕、kill/restart／orphan 不自動升格，以及版本切換持倉／相位／費用連續性都需可失敗驗收。每次重訓共用同一已凍結 acceptance script（source SHA、非績效 pass/fail、實際數值與 report hash）；首次獨立驗收通過不代表每次自動通過。記錄排定／實際 origin、訓練截止／publication、耗時／RAM、模型 hashes、歷史 reference 結果、pointer／切換日及 delay sessions。此處沒有已存在的腳本名稱或通過證明。
8. **252 日端點的 engine 相容性**：`portfolio.backtest` 目前以 20-session indices 的最後一個排定 entry 作 terminal liquidation，未保證精確252收益日；直接串接模型折也會重置狀態。需另案凍結並合成驗收單一連續研究 adapter 的首日NAV／首日成本、精確252收益日、末日估值／清倉成本及共同基準口徑，不改既有 P2 歷史 engine 語義；不准悄悄縮成240或延到260。未完成維持 STOP。
9. 本機 CPU 耗時估計來源：本輪唯讀 P1 manifest `full-scores-ace2eef/manifest.json`（32,500 bytes，SHA `dc3b9b3ad9c2c27a5366440c651f5f6baa7daaec9854368a0e944874890be9d8`），18 fold×seed jobs／全程 2572.822s；2026 seeds 17/29/43 分別 140.807／131.823／132.032s，合計 **404.663s≈6.74分鐘**。這些 jobs 含 day_t＋C0+ 擬合、feature建置與歷史推論，不是單獨 C0+ 計時；用其作同規模一次三種子重現的粗估 reference，獨立重做兩遍約13.49分鐘，另加原 fold 重現約6.74分鐘。每版最低三次 C0+ fits；現有 collect 實際六 fits。新 cutoff 多資料、首次矩陣建置／hash／落盤不含在估計，不能承諾一小時內完成。未來需在本機 CPU（n_jobs=4、三種子循序、不用GPU）量測並凍結時間／資源預算，確認2-session期限可行；失敗不得改期限救結果。
10. 訓練抽樣 `500000×185×4` 約 **370 MB（353 MiB）** 只是單一 float32 feature payload，下限而非 peak RAM；48欄 full matrix、dense session×stock、LightGBM bins／pandas／推論 buffers、序列 eligibility 與重建 inputs 會額外占用。峰值 RAM、磁碟與全矩陣延伸耗時目前 **UNKNOWN**，需同機量測；不以P2純回測估時替代訓練估時。期限是policy失效界線，不是本輪訓練授權。
11. Human 提醒需求：T1 凍結時交一份靜態 R_j／資料快照日／purge截止／最晚驗收日清單，提議各 origin 前5個交易日由指定 owner 人工提醒既有 Human 一人，當日核對無績效 checklist；失敗／逾期人工回報。owner已指定Human；實際人工提醒管道及無績效演練仍待凍結／驗收；只列需求，不建立通知、日曆事件、wakeup、recurring job，不部署或改 scheduler。

## 成熟時程、首次檢驗與檢定力

signal d 收盤後決策，d+1 open 入場，y5／y10 最早於第 5／10 個持有 session close 可計；凍結的安全延遲為 d+6／d+11 sessions 且來源已發布，缺停牌／exit evidence 時維持未知，不能補成 0。20-session purge 是訓練邊界，不是把前瞻前 20 日任意刪除。組合收益不以 label NaN 篩選；IC 只在共同成熟標籤支援上計算並報尾端排除。末日清倉／收益 cutoff 須在 machine contract 用 engine 語義明定並合成驗收，不事後延長持有挑終點。

第一次且唯一正式開盲（T1正式窗，不含階段0）：252 個排定收益 sessions 完整，且最後 signal 的 10d label 安全延遲與 source publication 已完成後；至少再等 11 sessions，需更久則據實等待。Human 凍結時依 calendar 寫實際 signal／return 起訖及 maturation 日；曆日僅以上節示例估計，正式calendar凍結後填入。

以 [P3 報告](../reports/v7_p3_portfolio_evaluation.md) 的 703 配對日估計：Mamba2 差值 CI [−.0414%, .0084%]，半寬 .0249 個百分點／日，SE≈.0249/1.96=.01270 個百分點；crossstock 半寬 .03205，SE≈.01635。假設方差與自相關相同，半寬隨 √(703/n) 縮放，126 日約 .0588／.0757、252 日約 .0416／.0535 個百分點。若欲偵測 +.02 個百分點／日、雙邊 α=.05、80% power，n≈703×[(1.96+.842)×SE/.02]^2，約 **2,227／3,690 日**。這是舊模型差值的粗略設計估計，非 C0+ 對等權的保證，不能把日數當獨立樣本或宣稱已有 80% power。P2 H-P2a 的 703 日 CI 半寬 .0554 更寬；故六個月尤其可能無法證實。預設 252 日是可行觀察窗，不是足夠檢定力承諾。Q2報後更新通常距4月T1約四個月，因此約留八個月的更新後支援；實際假期、延遲及20-session再平衡會縮短新版持倉暴露。252日足以設計觀察至少一次半年更新，**不足以宣稱驗證長期重訓效果或80% power**；正式calendar展開須列每版訊號／實際影響收益日數，至少有一次成功更新及更新後排定再平衡，否則確認性gate失敗，不能改長度補救。若窗內第二次更新，照已登記日曆執行，不因結果挑是否納入。重訓引入非平穩與跨版本依賴，舊SE縮放只是設計參考，block20/1000也非對所有regime有效的保證；Human已維持252日；之後要改，只能在無本研究結果時、凍結前，或另開新預先登記，不能看結果後續收至顯著。

## 判定規則、多重比較與停止

H-F1 的科學方向為單邊正向，但**統計預設雙邊 p**（沿現有 shared 函式，較保守）：僅當完整資料 gate 通過、95% CI 下界 >0 **且 p_boot<.05**，才記「在本預先登記期間及 raw 模擬假設下，P2 主要格相對指定基準的前瞻淨超額獲得確認性支持」。其餘記「未證實」；gate 不過記 `STOP / INVALID_FOR_CONFIRMATION`，不以未證實混淆無效資料。負向 CI 不得用方向性正向假設宣稱勝出。Human已維持252日單一端點與雙邊保守p；之後要改，只能在無本研究結果時、凍結前，或另開新預先登記。若改採單邊 p，須凍結前另寫 centred bootstrap 尾端公式與測試，不可將既有雙邊 p 隨手除二。

主端點只有整段252日的 policy−等權配對日均淨差及其CI／p。期末可描述性呈現前126／後126日與依實際切換劃分的各版本期間，兩種分段不必相同；持倉影響可能晚於分數切換，不將差異歸因重訓。分段不另作確認性檢定、不作額外正式假設、不依分段選winner或結束日。

重訓增加的自由度／forking paths 包括：日曆錨點、最短間距、purge off-by-one／publication、expanding起點／截面／抽樣、特徵／參數／seed、資料revision、驗收reference、retry／deadline、切換時間／再平衡、事故缺日及分析分段。上列凍結參數表與重訓章逐一封閉；後續machine contract須完整顯式序列化，凍結後不得按績效或便利性改選。任何未封閉項目都阻擋T1，不能留作執行者現場裁量。

正式 family 只有 1，Holm m=1 等於原 p；描述性項目不參與正式通過，不因結果好而增加假設。禁止中途查看累積報酬、IC、效應 CI／p、逐月績效或以模型排名決定延長；只允許不含收益／label 的 schema、hash、延遲與來源 coverage 檢查，由指定封盲保管者留 audit trail。若要中期效力分析，必須 T1 前新增時點與 alpha-spending／CI 校正；本草案**不允許**。

技術／資料 STOP 可隨時發生，但不是提早宣布勝利。安全停止後不刪除已收樣本、不換 T1、不重置成本／NAV／再平衡相位。若性能被意外開盲、source gate 無法復原、需變更資料語義或主要假設，該批只能描述性；重開前瞻需新規格與全新未曝光期間。

## 成本、raw 限制與資料完整性

成本延續 P2（常見假設，非券商報價），研究成交容量不等於實際可成交。無股利調整的 raw 價格可能把除權息／減資當負報酬，策略與等權暴露不同，不保證偏差抵銷。主要 endpoint 不在途中改 total_return；可封存公司行動與股利供期末描述性敏感度，未驗證時寫 UNKNOWN。若 Human 堅持 total_return 為主要，須在 T1 前另凍結完整來源／as-of／處理契約，不能把 raw 歷史與調整後前瞻混接後宣稱同一 P2。

提議資料 gate（T1 前 Human 凍結數字與合格源清單，不能結果後放寬）：

| Gate | 固定提議與事故處理 |
|---|---|
| 日曆／抓取 | 合格交易所日曆每日清單，交易日抓取漏日=0；必要源 publication／as-of／source hash 完整率 100%；當日 cutoff 以前未知即 STOP，不用觀測價格聯集替代正式日曆 |
| 宇宙／覆蓋 | 每日預先 eligible universe denominator 保留上市／下市／停牌；非停牌必要 OHLCV／features coverage=100%，預測 keys coverage=100%；不把資料缺失股票靜默移出分母。合法不合格／停牌須有事前規則及證據，不算抓取成功的藉口 |
| 價格與標籤 | conflicting duplicates、±inf、錯日期／不合理 OHLC、來源不合格=0；NaN label 只允許尚未成熟且不得控制推論列；持股 missing mark／下市 liquidation 未知不得當零損失或虛構成交 |
| 下市／事件 | T1 前驗證 lifecycle、終止交易日與實際可用退場／估值政策；保留消失股票。缺下市結果證據則前瞻主檢驗 STOP，不能移除虧損列；無漲跌停／可交易證據仍禁止交易就緒主張 |
| 資料事故 | 在開盲前記錄日／code／源／失敗 gate／發現時間；保留全部 revisions 與原預測。as-of 可證明的技術修復只准重播原當時 inputs，不能事後新增可用資訊或回補錯過成交。無法重建當時決策則整批確認性失效；不刪壞日、不以延長換好樣本 |

100% gate 是保守提案，不代表本機 cache 已達標；若合格 PIT universe、公司行動、publication 或下市證據缺失，保持 Blocker，不能借研究 cache 的存在放行。資料 acquisition／scheduler 修補不在本次範圍。

## 不允許的事、證據等級與已知限制

不允許追認 11/20 起為凍結前瞻、用矩陣日期掩蓋收益曝光、選較佳種子／重訓政策／日期／成本、追加權重搜尋、中途偷看、刪除失敗日、以 fill=0 隱藏未知、改正式 Data／V6/models、改 Colab 03_formal_resume 或 Drive。不能把本文件、工程測試或 bootstrap 門檻當成 production／投資授權。

證據上限 **E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED**；即使 H-F1 通過，也只是受限於 raw price／成本／宇宙／執行 proxy 的獨立前瞻研究支持，非 VERIFIED_EXECUTABLE／verified trading、非總報酬保證。限制：小樣本／自相關／regime shift、三種子未涵蓋模型選擇不確定性、舊研究曝光與人工市場知識、mutable non-PIT cache、市值非 PIT、下市與公司行動未全驗、無逐筆成交／漲跌停、整零股與容量未模擬。異常好結果必先稽核 leakage、PIT／universe／label timing／source provenance／selection bias／evaluation window；不能先升格。

## 封盲協議（Human保管與執行；啟動前須另案驗收）

1. **儲存邊界**：封盲區指定 Windows `D:\Desktop\work\ProjectForMe\MarketMamba-blind\`，WSL `/mnt/d/Desktop/work/ProjectForMe/MarketMamba-blind/`。每日訊號／持倉／分數、逐日NAV／成本／成交、訓練labels／抽樣keys、重訓後模型／checkpoints、可還原訊號的指標，以及任何含收益、IC或績效欄位的檔案／logs／錯誤dump全部留在封盲區；新模型也不能複製到可見區供worker推論偷看。必要inputs／狀態如可合併還原績效亦只供Human隔離程序存取，不在工單貼原值。
2. **可見區白名單**：只放 append-only SHA-256 hash chain 摘要（前一head、當次payload digest及新head）、行數、覆蓋率、錯誤計數及缺日清單。摘要不含收益、IC、分數、股號、持倉內容、模型bytes、排名、NAV或可還原績效的資訊；錯誤以固定代碼計數，不貼列內容／堆疊locals。摘要hash綁定完整私有manifest；公開hash不是私有payload的替代備份。Human保留私有原始檔、vintage、鏈與獨立head副本，以供開盲核對。
3. **誰能讀寫**：Human是唯一封盲保管者及重訓執行owner；Claude與Codex worker皆不得讀寫、列目錄、Glob／Grep或間接要求工具輸出此區。每次Codex worker prompt須明載：**「禁止讀寫 Windows D:\Desktop\work\ProjectForMe\MarketMamba-blind\ 及 WSL /mnt/d/Desktop/work/ProjectForMe/MarketMamba-blind/；禁止透過子程序、symlink、複本或logs取得其內容；僅可看已淨化hash／計數／缺日摘要。」** 本輪未接觸該路徑。
4. **保護邊界**：Human報告已在Claude本機 `.claude/settings.local.json` 加Read／Edit／Write／Glob／Grep／Bash拒絕（不進Git）；本輪不查驗本機設定，不宣稱Codex受此設定約束。Claude拒絕只約束Claude工具；Codex是另一程序，WSL使用者權限相同，故這是防疏忽偷看的紀律保護，**不是防竄改保護，也不是OS存取隔離**。同權限程序仍能重寫檔案與hash chain。Human可另選不同OS使用者／ACL，或未掛載於worker環境的儲存位置；未實施前不能聲稱技術隔離完成。若Human接觸策略績效，一樣適用偷看處置，兼任owner不是豁免。
5. **每日健康檢查**：由Human的隔離程序在私有區檢查schema／來源／模型／覆蓋與當日封存、chain連續性，僅輸出上述hash／計數／缺日摘要；缺日、hash mismatch、錯誤計數>0、coverage未過凍結gate均報STOP或依原重訓2-session retry規則處理。不得以收益正常判成功，不把缺日填零；除每日摘要外，完整診斷僅留私有區。worker只可用已曝光／合成fixture診斷，不能進封盲區排錯。
6. **重訓隔離**：重訓腳本由Human手動，或Human另行授權的排程執行；本文件不建立／啟用排程。fit／predict／persist在Human程序內，僅依固定日曆及工程gates，禁止stdout／notebook／logs輸出前瞻績效、eval IC或排名；使用同一截止快照，18:00切換、最多延遲2 sessions、超時STOP。Human只看淨化工程狀態，成熟train labels可由腳本使用，不手動檢視策略結果；腳本由worker以合成／已曝光資料開發及驗收，再交Human執行。
7. **一次性開盲**：階段0預設同此次描述性揭曉，不提前釋出；固定252收益sessions完整且最末signal至少11-session安全成熟與必要publication全過後，Human手動觸發。先以盲態完成完整性／事故帳／chain與外部head核對並封存判定，再記UTC及台北開盲時間、批准者、規格／來源／資料／模型hashes。Human一次釋出固定端點的結果包供Claude取得；Claude只在此時取得結果。封存原始包，任何後續分析標描述性；不得先試算再決定是否開盲、延長或換規則。
8. **偷看／洩漏**：階段0事故亦依兩階段章節記曝光及停止，未來確認性不得自動放行；任何人或worker意外看到可推導策略結果的資訊，立即停止擴散，記接觸者、時間、資訊類型、範圍／hash與設計可能受影響處（事故紀錄不再次散播績效）。該批確認性身分降為描述性／INVALID_FOR_CONFIRMATION，不換操作者、刪檔、重設T1或補日恢復；新確認性研究須新預先登記及全新未曝光期間。失敗與超時保留完整紀錄，不能為繼續流程放寬gate。

## 已決定（Human 2026-11-24；全文仍DRAFT待凍結）

1. 兩階段：嚴格T0起先試運行驗流程，年報後重訓T1才正式啟動；階段0績效封盲且永不用來調規。
2. 半年更新改年報＋Q2錨點，取代Q1／Q3。一般3/31與8/14為整體時鐘，例外及遲報另驗可得性；N_A=10／N_H=8、實際資料2-session緩衝是本DRAFT提案，尚非Human已凍結值。
3. T1起252日單一端點、唯一H-F1／Holm m=1、雙邊保守p不變。神經／固定影子臂預設不收集，P1／分段只描述性，raw主要、基準比例成本。
4. Human兼任封盲保管與重訓owner；worker禁止讀封盲區，僅淨化摘要，紀律保護不冒稱OS隔離。試運行預設與正式端點同時描述性揭曉，是作者較保守提案；替代選項及風險見上。
5. 工程另案批准／獨立驗收；本輪只文件與knowledge，不啟動。P4b固定歷史C0+、三混合Holm3、36-fit subset／64-fit gate不變，不套用本案新模型或日曆。

## Open Questions／凍結與啟動前仍須完成

1. T0前重盤曝光／完成最低安全日鏈；T1前凍結完整正式規格、N_A=10／N_H=8或替代值、合格2027／2028calendar、順延／例外、實際T1／252日末日／成熟日與各版有效暴露。未知日期不能宣稱正式啟動。
2. 確認試運行預設延後到正式端點揭曉（替代為簽署且T1啟動後揭曉、或永不揭曉）；不重問已決定兩階段／錨點／統計。任何揭曉以前必完成正式規則／日曆簽署，試運行永不作調規依據。
3. 100% coverage／0漏日、必要source、lifecycle／下市／publication、舊vintage與流程20-session連續觀察的提案尚待證據驗收。T0最低安全gate也不得放寬；缺證據就延期。
4. Human人工提醒管道、head保管、隔離／淨化、事故與手動開盲演練待驗；是否OS隔離仍待Human決定，不宣稱已部署。WP1～WP6完整工程與CPU／RAM／disk實測均待另案，不以文件完成代替接受。

## 附錄：前置工程工作包（只規格，不實作／不授權執行）

依賴順序如下；估計是本機CPU牆鐘執行時間，**不含開發工時**，未實測包不承諾上限。P1三seed 404.663s只作含兩模型與推論的舊reference；峰值RAM／磁碟均待量測，不啟動GPU。

| 包 | 依賴與交付 | 驗收條件 | CPU時間（估計） |
|---|---|---|---|
| WP1 日期／來源與as-of契約 | 無；凍結calendar、期限分類／例外、publication／pipeline vintage與keys | 年界／閏年／休市／特殊結算／逾期／更正版／缺時戳fixture；正確展開N_A=10／N_H=8、緩衝、origin−21及252端點，未知fail closed | 合成1～5分鐘；來源盤點與全量hash UNKNOWN |
| WP2 任意日期重現 | WP1；同P1配方的origin／purge／fit-predict薄adapter | 原P1三seed scores/ranks逐位元重現；年／半年／purge邊界及future-mutation不改歷史，fit keys／抽樣keys可重播 | 同規模舊fold約6.74分鐘；新origin兩次約13.49分鐘，均粗估 |
| WP3 矩陣延伸 | WP1、WP2；先全量immutable reference，再驗feature-only追加 | 48／185欄舊keys／X／mask／dtype／scores bytes一致；新增labels另版，不讀未來labels；重複／中斷／錯identity拒絕 | 合成1～10分鐘；首次全矩陣重建、IO與hash UNKNOWN |
| WP4 模型存檔與隔離執行 | WP2、WP3；三seed持久化、版本pointer、Human私有runner／淨化摘要 | serialize/fsync/hash→durable reverify→atomic pointer；load/predict round-trip、kill/restart、corrupt/missing pointer、orphan不升格、seed／matrix／contract mismatch拒絕；模型與敏感logs不外洩 | 合成1～10分鐘；每次fit約6.74分鐘reference，落盤／hash UNKNOWN |
| WP5 每次重訓驗收腳本 | WP1～WP4；同一frozen非績效acceptance script及靜態Human清單 | 已曝光reference獨立重做相等、遲到R+1/R+2相同快照、18:00／超時STOP、不中斷chain；封盲壞輸出／缺日／偷看／手動開盲fixture，公開區白名單通過 | 兩次重現＋原fold約20.23分鐘reference；全量矩陣驗收額外UNKNOWN |
| WP6 252日端點adapter | WP3～WP5；連續單一組合與固定基準 | 恰252收益sessions，首建／末估值清倉成本一致；模型切換不重設NAV／持倉／相位／buffer／pending；無額外重平衡、缺日STOP、端點開盲gate；固定P2歷史路徑不變 | 合成1～10分鐘；完整歷史reference回放UNKNOWN |

### WP1～WP6分階段截止（新增，不降低完整驗收）

**T0前不是零工程啟動**：下列最低安全版本皆須完成／獨立驗收，否則11/25只能是候選。T1完整deadline以排定R_A為準；若延期，依上節非績效程序留帳，不把deadline失敗當已完成。

| 包 | 必須在T0前完成 | 可到T1前完成／建議最遲 |
|---|---|---|
| WP1 | T0合格calendar、來源／as-of／曝光帳與當日fail-closed gates | T1前20 sessions凍結2027／2028錨點、例外、T1／252／成熟全表；正式啟動前覆核 |
| WP2 | 初始C0+配方、origin−21與原P1重現、當期模型可重播 | T1前20 sessions完成任意年度／Q2origin邊界與完整獨立重現 |
| WP3 | 每日48／185欄immutable延伸、歷史bytes不變、labels獨立、future-mutation gate | T1前20 sessions完成全量長期reference／資源量測；當日最低版本不可先欠帳 |
| WP4 | 三seed存檔／load-predict、pointer恢復、私有執行與淨化輸出、hash chain | T1前20 sessions完成跨版切換／完整kill-restart演練；Q2前5 sessions再核固定清單 |
| WP5 | 初始非績效驗收與每日健康／缺日／事故封盲演練（合成／已曝光資料） | T1前20 sessions完成每次重訓R+1/R+2／超時與正式開盲合成演練，source SHA凍結；T1前取得連續20-session流程證據 |
| WP6 | 試運行獨立連續組合／基準、成本與相位，不與正式NAV混接 | T1前5 sessions完成精確252端點、末日成本、正式重置及切換連續性獨立驗收；T1以前全部簽署 |

建議T1約2027年4月中下旬：20-session前約3月中下旬、5-session前約4月上中旬，均非已核實交易日。Q2約8月下旬前5 sessions與次年年報前5 sessions人工核對資料快照、origin−21、模型／驗收deadline與owner，不新增提醒服務。T0前最低版不能以「試運行不是檢定」豁免PIT／安全；T1前20日未過而無法累積20日穩定流程，就按非績效延後程序處理。

WP依賴表按各階段相應交付範圍適用；T0最低日鏈不代表完整WP已結案。各包只准在另案核准後用合成／已曝光資料驗收，驗收報告記source／input／contract／環境／實際秒數與資源／hash；獨立審查及Human接受後才可作前置完成證據。無法在既定期限／證據下完成就STOP，不縮測試、不改日曆救結果。

## 修訂身分與限制（前輪紀錄保留）

前輪紀錄：2026-11-24，`job-run-mwufq5k5-71212c/spec_writer_retrain`，既有worktree detached HEAD/base `176282179b2428bcc7f9ab1f1b6bd93b0408e164`；沿用已提交草案來源 `8eff9fe`（本輪HEAD另包含後續文件）。初始clean；只修文件、未commit。`/usr/bin/python3` 3.12.3／WSL CPU，只讀source與P1 manifest metadata；`MARKETMAMBA_DATA_ROOT` unset，Data realpath為本worktree的 `Data` 普通目錄，沒有更動正式資料。上方原盤點observed_at與曝光artifacts保留為原輪證據，本輪沒有重新觀測價格／雲端或更新其freshness。E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；完成不等於Human acceptance，獨立審查與工程驗收仍待辦。


2026-11-24前輪歷史修訂（Q1／Q3提案已由本輪取代）：`job-run-mwunxhyg-4cd29f/spec_writer_blind`，既有detached HEAD/base `191283bddbcd8107a467bf65fa8cbf7ab0cfb19b`，初始clean；僅4份文件patch，UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED。`/usr/bin/python3` 3.12.3／WSL CPU、MARKETMAMBA_DATA_ROOT unset，Data realpath為本worktree普通目錄，未讀正式Data或封盲資料夾，未重新觀測外部runtime／曝光artifacts。本輪覆蓋前輪T0+6曆月提案及owner待指定狀態；舊盤點時點僅保留原證據，不更新freshness。重大選擇：Q1／Q3後第8交易日優先於T0+6月或每季；維持252日雙邊而非延長／中期分析；Human紀律封盲而非宣稱OS隔離；財報欄維持排除而非臨時加回。完成不等於Human接受或規格凍結。

Knowledge-impact：設計語義／next decision／blocker細分=YES，已同步Current State／Research並檢查Authority Map；machine authority、subsystem boundary、active runtime、correctness threshold、新增canonical artifact、能力deprecation=NO。已檢查GOVERNANCE，本輪不改治理規則、authority-map.json或supersession歷史。


2026-11-24兩階段修訂：`job-run-mwuoasa7-93afef/spec_writer_phases`，既有detached HEAD/base `0db633bcedc97484bff6c316d7cc9d19334137d2`，初始clean。只修改本規格與Current State／Authority Map／Research；不commit、不派agent，不讀blind或正式Data。WSL `/usr/bin/python3` 3.12.3 CPU；MARKETMAMBA_DATA_ROOT unset，Data realpath為本worktree普通目錄。未重觀測既有曝光artifacts／價格或雲端，不更新其observed_at。本輪選擇／替代及限制見兩階段、N比較、WP截止；維持DRAFT / E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED，完成不是Human acceptance。

Knowledge-impact（兩階段修訂）：設計語義／active next decision／工程blocker細分=YES，已同改Current State／Research及Authority Map邊界說明；machine authority／subsystem／runtime／既有correctness thresholds／canonical artifact新增或supersession／deprecation=NO。新N與流程20日是待凍結提案，非修改已凍結gate。驗證紀錄見Current State本輪段落；未commit依本任務明示禁止，保留4份未stage文件供獨立審查。
