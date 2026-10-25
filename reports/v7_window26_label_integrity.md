# Window26 — Label Integrity & Next Study Candidate

狀態：COMPLETE / STOP / UNMERGED / NOT_DEPLOYED。僅為本輪有界補證與研究準備，不是研究准入。

## 整合與範圍

W25 固定候選 `27055f7ec34cd4aa6d47f67bd89e5b6616244bc8` 已由 main `90b1327560f717727ad3fd53dfe86cb421c70f5b` 本機 ff-only 整合。
核對 Review GO、4 focused tests、12 private artifacts、21 deliverable pins 與 raw restore；無額外 main 文件 commit。
W26 branch `research/v7-window26-label-contract`，base 同固定候選，worktree `.worktrees/v7-window26-label-contract`。
未整合 W06／W12／W19。W26 不修改 V6、Frozen Contracts、Production 或任何 W19 資產／runtime。

## 2321：已確認原因與未知界線

唯一新 FinMind 非價格查詢 TaiwanStockCapitalReductionReferencePrice，2321／2021–2023，回傳一筆事件：
2023-02-17、Making up losses、前收盤 6.38、減資後參考 19.93、開盤參考 19.95。
既存 Raw Price：2023-02-08 close 6.38；2023-02-17 實際 open 19.60、close 19.20。
**參考價不是實際開盤價。**

發行公司 2023 年個體財報第 38 頁（zero-based 37）確認 68% 彌補虧損減資、剩餘股數 32%。
2022-11-16 股東會、2022-12-06 生效、2022-12-08 會計減資基準日，不能當作市場報價單位切換日。
財報於 2024-03-04 通過，現存 PDF 位於 2025 網址，**不是當年公告時間或原始 Vintage 證明**。
[公司財報](https://www.tecom.com.tw/wp-content/uploads/2025/01/2023%E5%B9%B4%E5%BA%A6-FS-AllReport%E5%90%AB%E9%99%84%E8%A8%BB%E9%99%84%E8%A1%A8-CHT-IFRS.pdf)、
[FinMind 欄位／Free 單股資料說明](https://finmind.github.io/tutor/TaiwanMarket/Fundamental/)。

| Signal | Entry／Raw open | 10d Exit／Raw close | 原存 Label |
|---|---|---|---:|
| 2023-02-03 | 02-06／5.95 | 02-17／19.20 | +222.6891% |
| 2023-02-06 | 02-07／5.95 | 02-20／19.30 | +224.3697% |
| 2023-02-07 | 02-08／6.10 | 02-21／18.70 | +206.5574% |

三筆均符合原 `exit_close / entry_open - 1` 算式與共同市場 Session 的 t+1／t+10 索引；
但跨越股數／報價單位轉換，**不能視為相同股數的未調整持有期報酬**。
不計算替代 Label、不重算 RankIC、不改寫 W24 Predictions／Labels／Masks／歷史分數。

缺行情六日為 2023-02-09、10、13、14、15、16；既有市場日曆及同 cohort 的 2330 等價格證實均為市場交易日。
可確認個股資料缺口，不是市場休市。二手新聞承載公司停牌／換股訊息，但原始官方停牌公告未核實；
**確切停牌區間的第一手證據保留 UNKNOWN**，不可僅憑缺資料判定停牌或供應商錯誤。
`ExrightReferencePrice = -1` 保留為不明 sentinel，不能解讀成現金流。

## 既有 Label 的廣義限制

[事件矩陣](../research/v7/window26-event-matrix.json) 與 private case details 保留事件、來源及市場索引。
在既有 labels 上辨認到下列 path flags；這是資料診斷，不是新資格門檻或新評分：

| Partition | Path zero | Path missing |
|---|---:|---:|
| Training | 518 | 3 |
| Validation | 348 | 26 |
| 2023 exposed diagnostic | 222 | 7 |

另有 3 筆跨已確認減資事件，及 11 段在已報告上市界線內的個股缺行情區間。
這些是樣本／horizon flags，可能重疊，不能相加作唯一事件數。其他區間原因、完整公司行動覆蓋仍未知。
沿用 W25 44,266 筆時間／算式檢查，未重跑 W25 Bootstrap、LOSO 或 Pipeline Audit。
本輪補證只確認此股單位風險，**不能據此量化調整後整體訊號是否消失或保留**。

## 兩條 Label Policy 候選

[Versioned Label Policy Candidates](../research/v7/window26-label-policy-candidates.json) 均為 PROPOSED／HUMAN_DECISION_REQUIRED：

- **RAW_PRICE_DIAGNOSTIC**：保留回溯 Raw proxy 身份；事先定義股數單位事件、feature lookback／持有期間缺值與零價的排除或 UNKNOWN；除息可標記 price-return 或保守排除，由 Human 決定。不以觀察到的 RankIC 設計異常報酬閾值。
- **CORPORATE_ACTION_AWARE_RESEARCH**：先取得有來源／時間身份的事件、股數及現金流映射；區分 Price Return、Total Return 與 Executable Return。Adjusted Close 不能當 Entry Open；調整開盤須有一致因子與來源，或採實際 Raw Open 配合經驗證股數／現金流帳本。

不完整事件覆蓋不是 CLEAN；沒有新資料／研究批准前，不變更任何正式 Label Formula。

## 下一 Study：事前固定候選，不直接開跑

[Study Candidate](../research/v7/window26-next-study-candidate.json) 具獨立 Study／Dataset／Feature／Label／Universe／Event／Calendar／Policy 身份：

- 現有 36 檔 2021–2023 資料可作 **DEVELOPMENT_ONLY** 候選；33 檔 conditional current/new、3 檔下市起日未知，非歷史全市場。不得因結果刪選股票。
- 日期級 Universe／公司行動／停牌／missing／delisting 規則、Feature Freeze、next-open／5d／10d／maturity／overlap exclusion，均須 Human 凍結；不套用 E5 的 30-session purge。
- Expanding、約 12-month validation／quarter forward／3 origins 為候選；2023 多 origins 仍為 exposed development，非三個獨立 final tests。Refit 必須 fresh-state 新 Run，不冒充 Resume。
- L0 正／反 momentum 對照、Ridge alpha 1／seed 0、可選 LightGBM 單一固定 recipe 皆待批准。L0 新方向是 **看過 W25 後的設計提案**，不是原 W24 預註冊比較。
- Common Support、Daily N 20、average ties、equal daily aggregation 為候選；不是正式統計門檻。Selection／refit budget、serial dependence／uncertainty／multiplicity／acceptance 尚未凍結。
- Mamba-2 延後，GPU SAFETY_HALTED。W06 只保留固定 SHA 的 PROPOSED 參考，不合併、不自動採用。

[Exposure Ledger](../research/v7/window26-exposure-ledger.json) 以 machine contracts 核對：
E5 train 2013-01-02–2023-11-17、research/selection evaluation 2024-01-02–2026-09-11；
E1 train 2013-01-02–2025-11-18、evaluation 2026-01-02–09-04（5d）／08-28（10d）。
W21 原始資料檢視、W22 panel、W23／W24 的 2022／2023 診斷、W25 稽核與 W26 補證均保留曝光角色。
**尚無具資格且確認未曝光的歷史資料集**；未列年份為 EXPOSURE_UNKNOWN，不是 untouched。
另一候選為事前固定契約後的 prospective evaluation，需獨立資料／預測／研究授權，不借用 W19 操作權限。

## Human 下一次最少決定

1. 選 Label 路線與報酬語意，凍結 unknown／缺價／停牌／除息／下市政策。
2. 批准限定資料、日期級 Universe、calendar、bias／availability 限制及保存／備份。
3. 選 Development-only 或合格 prospective 路線，檢查 Exposure Ledger；凍結 recipe／fold／selection／refit／evaluation。
4. 另行批准精確資料請求與 CPU 上限；本次未授權執行候選研究。

候選成本上限不是本輪授權或實測成本：price reuse 0 新 GET；event 上限構想 36×4=144（逐來源資格／schema／必要性須先確認，**不保證覆蓋完整**）；official docs ≤4、paid 0；
Ridge ≤12 fits，可選 LightGBM 另 ≤12；CPU 60 分鐘／process 10 分鐘、增量 storage 1 GiB，空間不足即停，不 GC。

## Evidence、驗證與恢復

實際 FinMind 非價格 **1/6**、Price **0**；官方文件 loads/attempts **4/4**（2 archive GET、2 web loads，其中一次不可取得）。
未使用的 5 次 API 額度已 CLOSED，無 retry／redirect／token／paid／fallback。另三類事件資料未查，沒有以預算餘額擴張調查。
FinMind Free 單股文件與既有非商業／顯名／不再散布要求為限定來源依據；公司公開財報私有補證不等於 OGDL 或再散布許可。
完整帳目見 [Network Ledger](../research/v7/window26-network-ledger.json)。

Private root `/home/frank/projects/MarketMamba/.artifacts/window26-label-integrity/` 保存真實 event response／官方文件／case details。
15 files、4,785,564 bytes 獨立程序 Restore＋Hash PASS；W24 191＋W25 12 files 原 bytes reverified。
同機 restore 不等於獨立備份，BACKUP_PENDING。未取得的年報沒有原始 bytes，不宣稱可還原。
3 direct targeted tests PASS；獨立主要審查發現一項 Required（Exposure Ledger 過期 source pin），一輪修復後取得獨立 Closure GO：Critical 0／Required 0。
Reviewer 驗證 PDF bytes，但未能獨立抽取第 38 頁文字；第一手文字由主 agent 的有界官方 PDF 讀取取得，此限制保留。
JSON／Python syntax、diff check 與 knowledge health 通過（既有 Governance／Authority Drift／Legacy Docs WARNING，無 FAIL）。詳細 Review／governance／pins 見 [Evidence](../research/v7/window26-evidence.json)。

恢復：Current_State → Authority_Map → 唯一 research-program 的 Window26 → 此報告／事件矩陣／Study Candidate／Exposure Ledger。
本輪結束 STOP，W26 不自行整合／push／deploy。Strict Phase0 STOP、Formal Admission NOT APPROVED、E1 BLOCKED、H2 NOT APPROVED、W12 SAFETY_HALTED；W19 不操作。
