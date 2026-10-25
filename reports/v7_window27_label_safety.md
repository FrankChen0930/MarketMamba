# Window27 — Pragmatic Label Safety / Frozen-Prediction Sensitivity

狀態：COMPLETE / STOP；獨立 Review GO（Critical 0／Required 0），UNMERGED / NOT_DEPLOYED。
僅為 **POST_HOC_DEVELOPMENT_SENSITIVITY**，不是正式 Data／Research Admission 或模型優劣驗證。

## Git 與事先固定順序

W26 fixed `367005dfaba0988fca5e294d9b3e281408bbfce8` 已由 main `27055f7…` 本機 ff-only 整合；
38 deliverable/raw pins、3 focused tests、獨立 Required Closure GO 與 clean state 通過。
W27 branch `research/v7-window27-label-safety`，base 同固定候選。未整合 W06／W12／W19。
Policy／Predicate／Mask Manifest 在第一次新 Metric 計算前提交至 `e77f32d`，之後未改規則或 Mask。

## Development-only Policy

Policy ID：`W27_RAW_PRICE_DIAGNOSTIC_DEVELOPMENT_ONLY_v1`。
Semantic SHA-256：`2894968af5f5fd3e7e0fcd1627a823919a9bd44973aa681f1fe209d934d76d1f`。
[Policy](../research/v7/window27-policy.json)、[Mask Manifest](../research/v7/window27-mask-manifest.json)。

- 僅縮小 W24 原 exploratory support，不新增樣本、不改 Label／Feature／Prediction。
- 已確認股數／價格單位事件跨越 feature previous→signal 或 entry→exit 時排除；本輪唯一已確認事件為 2321 的 2023-02-17 減資市場參考日，不使用會計基準日。
- Feature 前一市場 Session／signal OHLC、Entry Open／Exit Close、每個持有期市場 Session OHLC 分別檢查。無資料、零、非 finite、OHLC 不一致拒絕；不補零、ffill 或跳過個股缺漏。
- 原 signal／next-open／5d/10d exit 市場索引與 split maturity 幾何必須一致；來源／Label identity 不一致即停止。實際歷史 publication、原始 vintage 未知保留，Development-only 不要求重建原始 Vintage。
- 完整公司行動／停牌／as-of 風險對所有樣本仍為 UNKNOWN，通過此 Mask 不等於 CLEAN／Strict PIT／Executable。
- Primary reason 優先順序事先固定，重疊理由另存全部 flags，不重複加總排除。

## Eligibility：原 44,266 筆 → 43,148 筆

以下是 **disjoint primary exclusion counts**；端點／Feature 必要價格的新排除皆 0，因原 W24 Gate 已拒絕該類樣本。
本表的 Zero 已另以 Raw 值核對；其他非零 OHLC-invalid 的新增排除為 0。

| Partition / Horizon | 原 eligible | 新 eligible | Confirmed event | Missing path | Zero path | UNKNOWN risk |
|---|---:|---:|---:|---:|---:|---:|
| Train / 5d | 7,338 | 7,164 | 0 | 0 | 174 | 7,338 |
| Train / 10d | 7,188 | 6,841 | 0 | 3 | 344 | 7,188 |
| 2022 / 5d | 7,539 | 7,405 | 0 | 7 | 127 | 7,539 |
| 2022 / 10d | 7,358 | 7,124 | 0 | 19 | 215 | 7,358 |
| 2023 / 5d | 7,512 | 7,444 | 0 | 0 | 68 | 7,512 |
| 2023 / 10d | 7,331 | 7,170 | 3 | 4 | 154 | 7,331 |

UNKNOWN 是每筆的風險狀態，不是另一種排除，不能與 primary totals 相加。
2022/10d 有 6 筆同時 missing/zero；2023/10d 的 3 筆 confirmed event 同時 missing，僅計一次。
Training 的 521 筆新排除僅為診斷；**沒有 refit，過去訓練樣本／公司行動風險的影響未被消除**。

## 四組原凍結 Predictions：新舊 Mean Daily RankIC

W25 paired-support helper 以 security=None 重用其相同 Support 檢查，不移除任何單一股票、不執行 LOSO；
底層仍為 W08/E5 average-tie Spearman。原 L0 momentum 正向不反轉，Ridge scores 不重算。
原 940 個日期×horizon 全數保留且兩模型均可計算，沒有 N<20 或 skipped days；新舊均值的日期集合一致。

| Period / Horizon | Days | Daily N 舊→新 | L0 舊→新 | Ridge 舊→新 | 新 paired Ridge−L0 |
|---|---:|---|---|---|---:|
| 2022 / 5d | 241 | 28–33 → 27–33 | -0.013561 → -0.012307 | 0.037016 → 0.033873 | 0.046180 |
| 2022 / 10d | 236 | 27–33 → 27–33 | -0.005890 → -0.001800 | 0.043930 → 0.045471 | 0.047271 |
| 2023 / 5d | 234 | 29–33 → 29–33 | -0.049949 → -0.050493 | 0.080310 → 0.080008 | 0.130500 |
| 2023 / 10d | 229 | 29–33 → 29–33 | -0.045992 → -0.047951 | 0.054056 → 0.053166 | 0.101117 |

依序排除 134／234／68／161 個共同樣本 key，改變 109／152／62／118 個日期的橫斷面。
完整 Daily N、ties、逐日 old/new、排除 security 與重疊原因保存在 private artifacts。
[四組結果](../research/v7/window27-results.json)、[Diagnostic Manifest](../research/v7/window27-diagnostic-manifest.json)。

**2023／5d**：沒有跨本輪已確認減資事件的合格原始 5d Label；其變動來自 68 筆零價格持有路徑。
Ridge 均值減少約 0.00030248，仍為正；這只支持「在此固定規則下數值變化小」，不證明無其他污染或泛化能力。
**2023／10d**：同時移除減資跨界與 missing/zero，不能把整體變化全歸因於三筆事件；未做另一次 event-only 評分。

## Replay、保存與限制

W09 用於驗證原 Bundle；W27 不是其已批准的 training profile，因此重用既有 atomic byte writer 保存獨立 diagnostic 完成清單，沒有改 V6 Publisher 的 authority/schema。
私有根目錄 `/home/frank/projects/MarketMamba/.artifacts/window27-label-safety/`，不依賴 W27 worktree 存活。
原 Predictions／Metrics／Labels／Masks／W24/W25/W26 bytes 均核對不變；所有新資料留私有根目錄，不提交原始 FinMind 資料。
同機 restore 為 `/home/frank/projects/MarketMamba/.artifacts/window27-restore/`，**BACKUP_PENDING**，不宣稱獨立備份。

首次 Replay 直接比較 Python dict 時，整數 N-key 經 JSON 變成字串而拒絕；以最小 round-trip 案例重現後改用既有 canonical hash。
原計算 source bytes 另保存／還原，AST 測試證明 calculate 函式未改；既有 metrics／mask 不覆寫，重新執行獨立 Replay 核對。
另有一次在任何新 Metric 前即發現的 import namespace 修正。兩者均保留證據，不冒稱一開始就 PASS。
10 Mask tests＋4 serialization/paired-support tests PASS；獨立程序 Replay 1,880 daily／940 paired PASS，44,266 Mask 全部重建吻合。
一次獨立 Review GO：Critical 0／Required 0，0 review rework；Reviewer 核對凍結提交、218 prior pins、8個 diagnostic artifacts／原計算driver 還原、support subset／彙總算術。
詳見 [Evidence](../research/v7/window27-evidence.json)、[Review](../research/v7/window27-review.json)、[Replay](../research/v7/window27-replay.json)。

尚未解決：不完整事件／停牌／歷史 availability/vintage、Raw 非 Total/Executable Return、hindsight cohort／current capital proxy／存活者偏差、
2022/2023 已曝光、原 fitted model 的 Training 污染風險。沒有 Bootstrap／LOSO／正式顯著性／最佳模型結論。

## 下一次 Human 決策與恢復

更大規模研究仍需事先批准：限定 Universe／日期／事件覆蓋與未知政策、Raw diagnostic 或公司行動感知 Label 路線、
Fold／maturity／Selection／Refit／配對評估與 Exposure Ledger、精確資料請求／CPU／儲存上限。
W26 的資源額度只是 PROPOSED，本次不自動啟用；本輪 0 API、0 fit/refit/new prediction、0 GPU/Colab。
恢復：Current_State → Authority_Map → 唯一 research-program 的 Window27 → Policy／Mask／Diagnostic Manifest／此報告。
Strict Phase0 STOP；Formal Admission NOT APPROVED；W19 PROTECTED；W12 SAFETY_HALTED；E1 BLOCKED；H2 NOT APPROVED。
完成後 STOP，不整合 W27、不 push/deploy、不啟動下一窗口。
