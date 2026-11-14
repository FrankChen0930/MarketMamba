# Window28 — Clean Training Refit & Temporal Causality

狀態：COMPLETE / STOP；Independent Replay PASS、Review GO（Critical 0／Required 0）；UNMERGED / NOT_DEPLOYED。
僅 **EXPOSED DEVELOPMENT DIAGNOSTICS**，不是正式模型選擇、Strict PIT／Executable Performance 或新 Research Admission。

## Integration、固定 Recipe 與來源

W27 fixed `39e2fedeb550a08d00f4abddc9a6caa7b1d570f8` 已由 main `367005d…` 本機 ff-only 整合。
35 deliverable/private pins、原計算source及restore、14 tests、Review GO 與 clean state 核對；未整合 W06／W12／W19。
W28 branch `research/v7-window28-clean-refit`，base 同固定 SHA；計畫與實作先提交 `ec0b072`，再 Fit。
[Plan](../research/v7/window28-plan.json)、[Prefit](../research/v7/window28-prefit.json)、[Run Manifest](../research/v7/window28-run.json)。

沿用 W24 原 Dataset／Feature／Label identities、W27_RAW_PRICE_DIAGNOSTIC_DEVELOPMENT_ONLY_v1 原凍結 Mask；
Feature 順序仍為 raw_close_momentum1、raw_intraday_range。Ridge alpha=1、fit-intercept、zero-variance UNIT_SCALE 均不變。
每個 Horizon 只 Fresh Fit 一次，exclusive claim 防止重複 Fit；不是 Resume，未讀舊 Ridge 權重作初始化。
新 Run：`w28-fresh-clean-Ridge-5-train2021`、`w28-fresh-clean-Ridge-10-train2021`。

| Horizon | 原 Training N | 新 Training N | Model SHA-256 | Scaling SHA-256 |
|---|---:|---:|---|---|
| 5d | 7,338 | 7,164 | a006337adfe95308c89323680bb06ef1906806c905ca432727d8bb3c1b8e06f6 | 32f5c5dc88c1f4e3f2f78e4230020ac35c952d95a58ab196b3ea323e2ae9bb5a |
| 10d | 7,188 | 6,841 | c0b18106eb6230b88e7ff87564401c2db2da8015e20e7ab09cea3319ad32de03 | 4998018cf0daefe0736b1e12b06082bfa425f22716b1e7de5d07676a22f4faa0 |

TRAIN 僅 2021，所有保留 Label 的 exit 嚴格早於 2022；source sample hash／Mask／Fold identity 皆核對。
Mean／std 由清理後 Training 重新估計，逐值核對；Validation／2023 不參與 Scaling、Fit 或參數選擇。
仍有未完整確認的公司行動／停牌／歷史 availability/vintage；這是套用已知安全規則的 clean-trained，**不是所有污染已清除**。

## 相同 W27 Evaluation Support：舊 Ridge 對 Clean-trained Ridge

兩者使用完全相同 security/session/horizon、Label identity/value；不是各自不同 Support 的均值比較。
W27 舊 Ridge 每日結果重新核對保存值；L0 原方向／分數完整保留，不另反轉或選模。

| Period / Horizon | Days | Daily N | W27 Ridge | W28 Clean Ridge | Mean paired difference |
|---|---:|---|---:|---:|---:|
| 2022 / 5d | 241 | 27–33 | 0.0338734123 | 0.0346821957 | +0.0008087835 |
| 2022 / 10d | 236 | 27–33 | 0.0454711909 | 0.0445606151 | −0.0009105759 |
| 2023 / 5d | 234 | 29–33 | 0.0800076447 | 0.0804071187 | +0.0003994740 |
| 2023 / 10d | 229 | 29–33 | 0.0531657385 | 0.0542475528 | +0.0010818144 |

[Results](../research/v7/window28-results.json) 保存完整 Daily N 分布／Support identity；private artifacts 保存新完整 Predictions、逐日 RankIC、配對差異、ties／rejection欄位。
940 日期×Horizon 都有定義，N20 未降低。2022／2023 均已曝光；未執行 Bootstrap／LOSO、正式不確定區間或 significance／勝者宣告。
四組仍有正向探索性平均值，可作未來事前固定新資料研究的動機；本輪小幅變化本身不是泛化／收益證據，也不據此選 Alpha／Feature／模型。

## Future-data Poisoning：18 個有界案例

固定 2303／2321／2330，2021-07-01／2022-07-18／2023-03-01，5d／10d，共18案例。
原擬2022-07-01的2321不在W24合格資料；在任何Fit與W28 Metric前，按原Mask選第一個全部股票／Horizon共同合格日July18。
[Prefit case record](../research/v7/window28-prefit-case-repair.json) 保存此調整；不改 Mask／股票池或資料期間。

只對記憶體副本 date > signal 的 OHLC 施加有限正倍率，並改 Volume／turnover；past prefix hash 必須一致，真實 Raw hash 不變。
重用 W24 所依賴的 engineering_sample feature code，對原始／Poisoned資料重新取得 Features；兩者在同一已固定模型下預測。
全部18案例 Features／Predictions exact-equal；另修改當日OHLC的 positive control 必須改變 Features，排除測試未觸及實作的假成功。
[Case evidence](../research/v7/window28-causality.json) 保存改動列數、Features、Predictions、past prefix及poisoned future hashes。

此證據只覆蓋所測 **有限、有效 OHLCV 數值擾動**與兩個既有 Feature 的數值因果性。
engineering_sample 同時包含未來 Entry/Exit 的 Label 資格檢查；本測試不宣稱其 row eligibility 是可直接部署的 as-of feature-only interface，
也沒有測試未來資料刪除／非finite導致的資格變動。不宣稱歷史來源無修訂、完整 PIT 或正式線上推論已驗收。

## Replay、保存、Review

獨立程序 Replay 只載入新持久模型 JSON，不再次 Fit；重新核對 Training view／mean／std，重算新 Predictions、daily／paired、18 poison案例。
首次兩次 Fit 的排程有真實 process timeout 300s／kill-after 5s，threads=1，CUDA_VISIBLE_DEVICES 空，Python3.12.3／NumPy2.0.2，沿用W11 CPU venv。
原 W24／W25／W26／W27 source bytes 受 hash pin 保護；W09原Bundle驗證、既有atomic writer、W25 paired helper／W08 primitive重用，不建立新Framework。

Private root `/home/frank/projects/MarketMamba/.artifacts/window28-clean-refit/`；同機restore `/home/frank/projects/MarketMamba/.artifacts/window28-restore/`。
保留原始資料與所有舊模型／Metrics；BACKUP_PENDING，不宣稱正式獨立備份或 FULL_REPRODUCTION_READY。
6 targeted tests PASS；獨立Replay1880 daily／940 paired／29143新Ridge scores／18 cases PASS，Replay0 Fit。
獨立Review GO：Critical0／Required0，1主要輪／0 review rework；Reviewer核對模型normal-equation residual，max約1.78e−15／8.88e−16，沒有額外solve/fit。
11 files／28,170,524 bytes同機還原，227 prior source identities核對；
[Evidence](../research/v7/window28-evidence.json)、[Review](../research/v7/window28-review.json)、[Replay](../research/v7/window28-replay.json)。

## 下一次真正有比較價值的研究所需批准

最少需 Human 事前批准：限定新資料／日期／Universe及公司行動未知政策；Label與decision-time用途；
Exposure Ledger檢查後的development-only或prospectively collected evaluation、固定Recipe/Fold/selection/refit/paired/statistics，
以及精確資料請求／CPU／儲存與備份上限。不能把某個未核對年份自動稱 Independent Test。
本輪沒有新API／GPU／Colab／LightGBM／Mamba／Search／V6或W19操作；W12 SAFETY_HALTED、StrictSTOP、E1BLOCKED、H2NOT_APPROVED維持。
恢復：Current_State → Authority_Map → 唯一Research Backlog Window28 → Plan／Run／Results／Causality／此報告。
完成後 STOP，不整合 W28、不 Push／Deploy、不開始下一窗口。
