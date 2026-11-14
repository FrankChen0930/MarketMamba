# Window30 — Fixed Real-data LightGBM Baseline

狀態：COMPLETE / STOP；獨立Replay PASS／Review GO（Critical0／Required0），UNMERGED / NOT_DEPLOYED。
僅EXPOSED DEVELOPMENT DIAGNOSTICS，不是正式模型選擇、Independent Test、Strict PIT、Executable Return或研究准入。

## 整合、資料與唯一 Recipe

W29 fixed `e83dfe68e4bc73b59cb6390ec130ba967138e7f1` 已由main `1d9a2a7…` 本機ff-only整合。
340 deliverable/private pins及restore、4targeted tests、Review GO與clean state核對。未整合W06/W12/W19。
W30 branch `research/v7-window30-real-lightgbm`，base同固定SHA。
任何Fit前，Recipe、environment/native-lib SHA、Feature order、Seed、Train view及planned ModelSpec IDs已在 `60e7ca7` Local Commit。
ModelSpec在Fit前已知，實際learned weights/hash在Fit後產生，沒有偽造預知權重身份。

沿用W29 Cohort/Data/Feature/Label/Fold與feature-only universe，沒有新增API、資料、清理規則或Ridge Fit。
環境：既有W11 CPU venv，Python3.12.3／LightGBM4.6.0／NumPy2.0.2／SciPy1.16.3／scikit-learn1.6.1；無安裝升級。
[完整Recipe](../research/v7/window30-recipe.json)、[Prepared identities](../research/v7/window30-prepared.json)。

唯一Config：L2 regression、64rounds、7leaves/depth3、learning_rate=.05、min_child_samples=100、L2=1、max_bin63、seed0/thread1。
其餘sampling、seeds、deterministic、CPU、bin sample budget皆在JSON固定；選擇依容量/成本約束，非依W29 RankIC反向最佳化。
不做preprocessing；LightGBM內部bin mapper只由TRAIN Dataset建立。不傳eval_set/callbacks/init_model，沒有Validation Early Stopping或Search。
原Feature order為 raw_close_momentum1、raw_intraday_range；TRAIN僅2021，最新Label Exit2021-12-30，maturity/split依原Mask。

| Horizon | Train N | Model SHA-256 | Actual trees | 模型文字bytes | Fit wall seconds |
|---|---:|---|---:|---:|---:|
| 5d primary | 14,113 | c08586a7351fb20dd2b2fbc3c451e5d2c68c26c5dcb01bf6d53fe1df349dcc6a | 64 | 52,369 | .027772 |
| 10d secondary | 13,711 | baf665c12e2c063279f3218d73d2fa0ad827a30728f70b039787bde8bc4f963b | 64 | 54,849 | .027907 |

共2個真實Fit，Ridge0。5d成功、相同frozen inputs/recipe與資源符合後才執行10d；兩者皆在評估指標產生前完成，未做feedback tuning。
每個Fit子程序有真實timeout300s/kill-after5s與OS CPU cap；counter保留one-fit claim，拒絕重複Fit。

## Feature-only Predictions 與共同支持集合

Scoring不接收Label Table，直接對W29的相同Feature-only rows產生 **60,726** 筆完整Predictions；尾端Label未成熟亦保留。
每列包含Stock/Session/Horizon/實際Score/Model ID及SHA/Feature/Data/Recipe/Source Evidence ID。
先immutable發布Predictions，才依原W29 Mask進行事後Label Join；未來Label不改寫Forecast Keys。
Old L0±/Ridge Predictions/Metrics唯讀引用，Support以key及label ID/metadata/value核對，不依股票列順序對齊。

| Period/Horizon | 有效日/N | L0正Mean | L0反Mean | RidgeMean | LightGBM Mean / Median | LGBM−Ridge paired mean |
|---|---|---:|---:|---:|---:|---:|
| 2022/5d | 241 / 59–63 | −.010740 | .010740 | .037333 | .018814 / .021627 | −.018520 |
| 2022/10d | 236 / 58–63 | .013257 | −.013257 | .012073 | −.012617 / −.016793 | −.024690 |
| 2023/5d | 234 / 60–64 | −.025797 | .025797 | .047138 | .021182 / .026036 | −.025955 |
| 2023/10d | 229 / 59–64 | −.008323 | .008323 | −.008580 | −.013374 / −.013470 | −.004794 |

LGBM−L0正paired means依序 .029554／−.025874／.046980／−.005051；LGBM−L0反 .008073／.000640／−.004615／−.021697。
Average-tie Spearman、min40/EQUAL_SESSION不變。每head/period包含所有請求日期；970group中940有效，30tail/maturity N0保留拒絕原因，不當0 RankIC。
加入LightGBM未改任何Common Support；每日N、ties及2910筆paired results保存在private artifacts。

沿用W29的disjoint primary label exclusions：2022/5d411、2022/10d792、2023/5d512、2023/10d966。
合計2681 unscorable head/sample keys：Entry missing49／Entry invalid-zero230／Holding missing33／Holding invalid-zero481／maturity-tail1888，event0。
這不是重新清理或修改Mask；所有UNKNOWN action/suspension/asof/vintage仍UNKNOWN，不冒充CLEAN。

## 效能、Replay與保存

Fit計時為estimator.fit呼叫本身；Inference計時只涵蓋Booster.predict，不包含程序啟動/IO/模型解析。
5d兩期間Inference約 .011202/.012393秒；10d .010804/.014426秒，總約 .048825秒。完整CPU process overhead另記resource ledger。
獨立程序使用新保存的LightGBM model text重新載入、推論、Daily/paired重算；**0再次Fit**。
60,726 Predictions／970 Daily／2910 Paired Replay PASS，original W29 labels/models/predictions/metrics保持不變。
只載入本次已知來源的LightGBM text，不讀未知PyTorch/pickle；環境與native library身份不合即STOP，不擴環境。

Private root `/home/frank/projects/MarketMamba/.artifacts/window30-real-lightgbm/`；restore `/home/frank/projects/MarketMamba/.artifacts/window30-restore/`。
11 immutable files／34,874,071 bytes/份同機還原PASS；W29原299檔hash不變，BACKUP_PENDING，非獨立備份或FULL_REPRODUCTION_READY。
總artifact+restore **69,749,181 bytes <256MiB**；CPU研究程序合計 **30.723458秒＋60秒verification reserve <1200秒**。
6targeted零實際Fit tests PASS；先前測試fixture漏entry已修正，不是Recipe調整。
1主要獨立Review GO（Critical0／Required0），0 review rework；Model/Train identities、support與保存算術及old/new bytes驗證通過。
Reviewer無法從stdout獨立重證timeout啟動wrapper；已保存實際工具命令/開始與完成reference，沒有重跑Fit。
[Review](../research/v7/window30-review.json)、[Execution evidence](../research/v7/window30-execution-evidence.json)。
Source/nav code依賴保留Git checkout，W29 private bytes及W11 venv須保留；不宣稱單靠hash可重建資料。
[Run](../research/v7/window30-run.json)、[Results](../research/v7/window30-results.json)、[Replay](../research/v7/window30-replay.json)、[Resources](../research/v7/window30-resource-evidence.json)。

## 研究解讀與下一次最小授權

本固定Recipe的5d Mean/Median兩年皆正，但四組Mean均低於既有Ridge；10d兩年皆負。結果完整保留，沒有改參數再Fit。
這只說明固定LightGBM能完成現有真實資料CPU Pipeline及本次分數，不構成模型勝者、可交易收益或Mamba-2訓練批准。

已有可重用的固定CPU baseline E2E基礎，值得準備Expanding Window比較；實際執行仍需Human事先凍結：
Origins/model-ready cutoffs、train/validation/forward範圍、maturity/purge、selection/refit budget、paired/statistical policy，
以及繼續有限retrospective proxy或新prospective資料、universe/event未知政策、曝光帳本與CPU/保存上限。
W29現有Mask針對固定split；新Origins須重新核對model-ready deadline/maturity/purge，不能直接視為Expanding已認證。
本輪不直接開多Origins或改Recipe，也不把已曝光2022/2023改稱Independent Test。
StrictSTOP／FormalAdmissionNOT_APPROVED／W19PROTECTED／W12SAFETY_HALTED／E1BLOCKED／H2NOT_APPROVED。
恢復：Current_State → Authority_Map → 唯一Backlog Window30 → Recipe/Prepared/Run/Results/本報告。
完成後STOP，不整合W30、不Push/Deploy、不啟動下一窗口。
