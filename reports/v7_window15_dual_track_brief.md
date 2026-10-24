# Window15 — Licensed Foundations / Dual-track Brief

WINDOW_COMPLETE / STOP / UNMERGED / NOT_DEPLOYED。只有準備與synthetic工程，沒有研究准入。
W14固定4da1bcfa從c78098f安全ff-only整合；前150tests12.337s、後150tests14.109s PASS，
StageA獨立0Critical/Required。**337,221 raw bytes及4snapshot檔仍只在原W14 worktree**，
沒有隨Git搬到main，未移動/覆寫/公開再散布。W06/W12仍未合併，GPU SAFETY_HALTED。

## C. Source / License Matrix（精確resource，非provider全域授權）

| 資料集／提供來源 | 官方列出的resource | 分類與適用範圍 |
|---|---|---|
| [11549](https://data.gov.tw/dataset/11549)，上市個股日成交，TWSE／登錄機關證期局 | `https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data` | LICENSE_CONFIRMED_FOR_SCOPED_USE：該開放資料的下載保存／研究／衍生／展示再散布須OGDL1.0顯名；不含任意其他route |
| [11371](https://data.gov.tw/dataset/11371)，TPEx收盤行情／登錄機關證期局 | `https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&se=EW&o=data` | 同上。此dataset也包含權證/ETF/ETN，不能直接當普通股historical universe |
| W14 MI_INDEX/date與TPEx otc/date | 兩個不同歷史查詢route，精確params保留於matrix | PERMISSION_UNCLEAR；沒有把兩個catalogue CSV的授權外推至此，也不重新下載 |
| 合格calendar／歷史security lifecycle | 尚未指定合格來源 | PERMISSION_UNCLEAR／DATA_BLOCKED；current收盤股票代號不能代替完整歷史universe |

依[OGDL1.0](https://data.gov.tw/license)明示的利用與顯名條件整理；自動定期服務的頻率／
quota／redirect／SLA未確認，列LICENSE_CONDITIONAL，**本專案也未批准Collector或公開發布**。
每日更新不是歷史範圍、原始vintage或實際發布時刻證明。TWSE一般條款中的開放資料例外
與特定dataset授權分開處理；TPEx一般條款頁403未繞過，不藉此否定或擴大dataset-specific授權。

[完整machine matrix](../research/v7/window15-license-matrix.json)含各用途、版本、顯名、
rate限制、historical coverage與未知項。[授權證據](../research/v7/window15-license-evidence.json)
保留3份官方metadata/license HTML及manifest，1,547,650bytes，本地獨立核驗通過；
新的retrieval沒有冒充較早HTML。原始證據留本tree `.artifacts/window15-license-evidence`。
本輪只有官方metadata/license網頁：5次web open（含1次403）、5次metadata/license HTTP讀取，
**行情payload請求0、未明endpoint請求0、付費0**。Hashes不等於可獨立還原bytes，保留工作樹。

## D. Track A：值得申請的最小歷史探索研究候選

獨立Study `v7-exploratory-ohlcv-study-v0.1`、Dataset `v7-exploratory-ohlcv-dataset-v0.1`、
Feature Manifest `v7-exploratory-ohlcv-features-v0.1`，均PROPOSED，實際data content hash/bytes
與研究執行授權仍空。它不是E5/E1重現、VERIFIED_ASOF、EXECUTABLE或IndependentFinal。

[Study](../research/v7/window15-exploratory-study.json)／[Dataset](../research/v7/window15-exploratory-dataset.json)／
[Feature draft](../research/v7/window15-exploratory-feature-manifest.json)：最小OHLCV family加
return_1、prior-L volume ratio、prior-L volatility候選；L、價格調整、zero/missing/scaling未選。
Momentum/reversal只是待比較替代，不自動增加更多features。按合格session且只用截止前依賴，
**因果公式不能修復raw資料的historical vintage或availability缺口**，本輪未計算任何真實因子。

研究方向保留Expanding、5/10d、約12月validation、一季forward、3origins，全部未凍結。
Conservative Lag仍null。A/B/C/D資料證據等級與使用／研究准入分開；Conditional Proxy
也不能沒有Human批准就訓練。新identity gate固定回PREPARATION_ONLY／DATA_READY=false／
execution=false／admitted=false，不是可由Agent翻status開啟真實資料的閘門。

串接計畫只有一條：合格Dataset→W07 Fold/eligibility/isolation→W10/11 L0/Ridge/L2→
W08 CommonEvaluator→W09 Artifacts→OfflineReplay。既有adapters/publisher仍限制synthetic
profile，**不能直接接真資料或改status繞過**；另批必要的最小scope extension即可，不建新框架。
Label entry/exit與maturity、calendar/universe、cutoff、selection/refit/evaluation/final exposure、
真實bytes/restore及Human研究授權均未完成。新少特徵結果不稱48-feature E5重現或純架構優勢。

## E–F. Track B：Synthetic Dry-run與重用

薄adapter `v7_vintage_dryrun` 僅SYNTHETIC_ONLY／TrackB：重用Snapshot no-clobber raw/manifest、
W14 clock/pin validator、W09 lock/safe-path/strictJSON/artifact verifier。W09 study bundle要求
predictions/metrics，故不製造不存在的研究artifact來包裝raw capture；沒有registry或collector。

8個持久dry-run events示範首次保存、同內容冪等、revision中斷/重試、QC失敗、late重複、
missing gap與backfill；3個不同vintage保留原bytes，14591bytes為該次synthetic輸出，非真CSV容量。
ORIGINAL_FORWARD_CAPTURE/LATE_CAPTURE/HISTORICAL_BACKFILL/REVISION_CAPTURE/MISSING_CAPTURE
分開，UNKNOWN clocks/freshness另標UNQUALIFIED_CAPTURE。相同raw不重複建立vintage，但當次
receipt/QC/late/mode另留attempt，不能用首次成功掩蓋本次失敗。publication缺失不拿retrieval填補。
實際archive_created_at來自執行時，synthetic observation timestamps不是歷史捕獲事實。

[原dry-run](../research/v7/window15-synthetic-dryrun.json)保留當時source hash與事件；該次source
舊bytes也保留，不假裝由修復後source執行。[新增restore validation](../research/v7/window15-restore-validation.json)
用最終verifier重驗既存紀錄。原始snapshot、intent、binding缺失/損壞即NOT_RESTORABLE，
retry不會補寫缺失的身份證據來冒充成功。Caller clocks/QC/授權pin仍只是宣告，不是source認證。
目前是source-wide snapshot工程；per-security QC sidecar、真實時計/事前intent audit、calendar/
field semantics與live collector仍需後續批准/驗收，沒有因此證成Full Reproduction或Historical PIT。

## G–J. 證據、啟用條件與下一次最少批准

U1獨立GO0Critical/Required、0rework；U2 7tests／獨立7PASS，0rework；U3最終18tests／
獨立18PASS，2輪boundedrework（重複擷取仍須QC/late檢查；還原須有intent/binding bytes）。
另1次fixture修正：新revision必須有新的receipt身份，未放寬不可覆寫snapshot規則。
最終175joint CPU tests PASS14.625s；無fits、RankIC、GPU或任何collector執行。

| 分界 | 現狀 |
|---|---|
| DATA_READY | 真實Study仍false；兩個resource有scoped license證據，不代表已取得或准入historical dataset |
| POLICY_REQUIRED | TrackA evidence-class/feature/label/cutoff/fold/selection-refit/eval/compute；TrackB activation/clock/QC/storage/缺漏操作 |
| LICENSE_UNCLEAR | 兩個W14歷史route、未選calendar/lifecycle來源；periodic服務條件未定，不外推第三方API |
| BLOCKED | 新研究缺actual bytes/restore、historical eligibility/calendar/universe/labels與Human admission；GPU獨立SAFETY_HALTED |

[Future Activation Brief](../research/v7/window15-forward-activation.json)只列候選來源與合法方式；
正式保存路徑/環境/啟用日未選。真實CSV未下載，容量UNKNOWN：應在另批bounded量測後，以
raw+manifest+record+attempt+revision大小×捕獲次數×備份份數計算；不能用本輪synthetic或W14
不同API大小當真實預估。缺席保留MISSING、後補仍標Backfill。沿既有Operations/W04B-LOG
傳遞失敗、持久attempt及本地非零退出；通知渠道/收件人與寄送尚未選或啟用。

下一次可先申請「合法resource的小型資料資格/coverage與保存驗證」，**還不能直接申請已就緒
的真實CPU fit**：先補實際歷史來源/版本/rights與bytes、calendar/universe/labels，再凍結
新研究/評估/資源契約。TrackB可另申請限定來源/頻率/保存/備份/缺漏/監測的operation窗口，
但此輪不啟動。StrictSTOP/E1BLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED保持。恢復見CurrentState及唯一Research Backlog Window15。
