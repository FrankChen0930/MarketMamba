# Window33 — Recent Data Preserved / Model Preparation BLOCKED / STOP

W32 fixed `8fa41869a2583f0b5db698db83eed98611f24a7e` 已由main `67e2d25a710273646d41c0ad861b4a993593ba4d` 本機ff-only整合。Review GO/0Required、11個Synthetic原檔/restore、13tests與治理驗收通過。沒有整合W06/W12/W19。

W33 branch `research/v7-window33-recent-shadow`，base同固定SHA，UNMERGED/NOT_DEPLOYED。首次GET前 `91d7cca` 固定64檔（W29已有Feature的同一股票池）、資料日期2024-01-01至2026-10-22、Query/License/限額與guard。Known immutable W21/W22/W24/W29 cache request metadata沒有完整期間superset；沒有掃W19或mutable Data來補。

## 真實資料

64 price GET／0metadata／0token/retry/redirect/fallback，皆HTTP200。請求間隔最小5.001249秒；共享known-local hour最高prior63，低於100保守閘，其他匿名用戶量UNKNOWN。原始7604573bytes，完整保存Raw Responses、RequestSpec、實際retrieval、manifest/SHA、QC。網路額度已CLOSED；剩6不得自行追加請求。

| 範圍 | 證券 | Raw rows | Observed sessions |
|---|---:|---:|---:|
| 2024-01-02～2026-10-22 | 64 | 43437 | 679 |
| 2024 | 固定同一Cohort | 見private逐股報告 | 242 |
| 2025 | 固定同一Cohort | 見private逐股報告 | 243 |
| 2026至10-22 | 固定同一Cohort | 見private逐股報告 | 194 |

58筆zero OHLC，不補值；相對所有股票共同觀測日期union有19筆缺漏：3041六日、8422七日、9924六日，原因維持UNKNOWN，不把缺漏稱為停牌或把union外日期直接稱休市。Daily數值Feature候選61–64檔，總43319；受missing/zero/既有疑似單位事件規則拒絕的reason flags可能重疊，不相加冒充獨立筆數。

8422在2025-11-17跨越缺漏後出現疑似單位不連續，沿用W29固定閾值標記；不因大跳動推定減資/分割。W26既有事件矩陣無該事件來源，W16/W18只可核2026-10-22當日價格一致，不能證明2025事件原因。保留UNKNOWN，沒有修正價格或新增資料源。此事件已由既有Feature/Label path predicates標記，不依模型分數決定排除。

原始資料是現在取得的歷史版本，非當時Original Vintage、Total Return或可執行報酬。新期間公司行動未知本身沒有造成全面拒絕資料；資料仍保留。2024–2026不是Untouched Independent Test。

## 官方CSV與兩Features

只讀W16/W18獨立保存的TWSE CSV；兩者同SHA、同2026-10-22 Session。64檔五欄open/high/low/close/volume全部精確相同，volume保持shares、不做隱式倍數換算；不把CSV全商品當普通股Universe。

重用W29同一feature_at：`close_t/close_prev-1`、`(high_t-low_t)/close_t`，固定此順序、shape2。官方當日row＋明示FinMind前一觀測Session row的混合來源測試，64檔Features全部exact相同；Missing不補0，Zero不ffill。這證明當日schema/units/formula相容，**不是兩個連續官方Snapshot的獨立Feature parity**。不由漲跌價差反推前日收盤或假設除權息參考價相同。

## 為何沒有Fresh Fit及Offline Scores

近期價格union只有OBSERVED_MARKET_SESSION資格，不能證明共同資料缺漏與休市的差別，也沒有補足實際Session時鐘/臨時休市。W31 qualified reference精確來自W23/W29已保存2021–2023日曆，不能直接延伸到2024–2026。W07原validator對新索引回傳 `CALENDAR_QUALIFICATION_UNKNOWN`；不能改字串來通過。

按觀測Session索引，5d有42720、10d有42173筆數值/價格候選；只屬候選maturity。全部86638個新mask保留 `eligible_for_fit=False` 及calendar原因。原始Vintage/事件UNKNOWN另行標記，沒有冒充CLEAN。

因此實際Training N=0、Ridge fits=0、LGBM fits=0、新Model IDs=無、真實Offline Scores=未執行。未把W31舊模型重新命名，也未產生假的Replay成功。Prediction Replay N/A；Raw/Derived獨立還原已完成。原W30固定Recipe及train-only scaler/feature-only Universe原則未變，但本輪不建立未通過資料Gate的訓練/推論入口。

## 保存、測試、下一步

Private Root `/home/frank/projects/MarketMamba/.artifacts/window33-recent-shadow/`，Restore `.artifacts/window33-restore/`；264files、58910660bytes/份，43437 Raw rows獨立程序還原/hash/schema核對，68個既有old raw pins未變。117822087bytes含restore，低於1GiB。CPU28.817576秒（含失敗diagnostic stage）＋60驗證reserve<1800；0 Fit。BACKUP_PENDING，同機副本不是獨立備份，不可清除worktrees或舊資料。

8 targeted tests PASS。早期system Python缺numpy、audit import path、calendar probe shape錯誤均保存trace，改正後驗收；未重送GET或覆寫輸出，失敗CPU保留計帳。W19 Task/worktree/artifact/runtime與V6 Scheduler完全未讀未動。獨立Review結果見window33-review.json。

下一步最小阻塞：有界補足2024–2026的合格市場Calendar/休市證據；若Human願意使用另一種明示代理Calendar，必須另行批准獨立身份與限制，不能升格現有union。解除後才在新授權窗口按本輪固定Recipe做最多4次Fresh Fit及Offline分數驗收。真正PREOPEN仍另需模型/來源/Feature parity、精確freeze/missing/recovery與有界操作授權；不啟動每日Prediction或Collector續期。

獨立主要Review：GO_DATA_PRESERVATION_AND_BLOCKED_CLOSURE，Critical0／Required0，0輪review rework。獨立確認8tests、64snapshot/panel、86638 masks fitFalse、264原檔/restore hashes。此GO不是模型訓練或Shadow啟用批准。
