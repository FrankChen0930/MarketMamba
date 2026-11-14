# Window14 — Evidence-first Admission Pilot

**PROPOSED CLASSIFICATION / NOT_ADMITTED / UNMERGED / NOT_DEPLOYED。**
本頁只回答下一研究是否有足夠證據；不是新Roadmap、准入或訓練批准。

## A–B. Evidence Matrix：最接近研究的是什麼？有Historical PIT Panel嗎？

| 本輪樣本 | 新取得的證據 | 還缺什麼 | 分類候選／目前用途 |
|---|---|---|---|
| TWSE MI_INDEX，2022-09-01，選2317/2330 | 官方HTTPS回應200／日期相符；208,587原始bytes；現在retrieval與SHA；兩筆OHLC正值、high/low與volume基本QC通過 | 當時publication/availability/QC、原始版本/修訂鏈；volume/session口徑、adjustments、calendar/universe/lifecycle、labels、使用權 | RETROSPECTIVE_ONLY；只支持目前source/schema/QC/保存診斷，非PIT或可交易證據 |
| TPEx otc typeEW，2022-09-02，選6488/8069 | 官方HTTPS回應200／日期相符；128,634原始bytes；現在retrieval與SHA；兩筆同類基本QC通過 | 同上；表名「不含定價」不能自行推出所有整股/零股/成交量口徑均已合格 | RETROSPECTIVE_ONLY；不與TWSE不同日期拼成common-support研究樣本 |

原始回應含當日較多證券，只檢視指定四筆；非全市場coverage或統計樣本。沒有補抓日期、
5d/10d labels、交易日曆、48-feature panel、RankIC或模型fit。既有W13零股證據沒有重新下載。
[Panel](../research/v7/window14-historical-panel.json) 保留精確request/session、response date、
header、retrieval、selected rows/QC與raw/manifest SHA；raw留在本worktree
`.artifacts/window14-historical-pilot`。有實際bytes可重驗，**Git hashes本身不能還原**；
若遺失且無可信可還原來源，必須NOT_RESTORABLE，重新下載不能保證同一vintage。

結論：**官方日行情price/volume比48個混合來源features更接近小型研究起點，但仍未合格。**
指定pilot內沒有可證成historical-asof的panel；不是宣稱所有可能付費/機構檔案皆不可能。
2026-10-24取得2022資料，server Date也是現在；Last-Modified/ETag缺失。
它證明「現在可取得這個版本」，不證明2022時收到的原始值。四檔事後選取亦非historical
universe，不能忽略下市/停牌或把所選仍存在股票當無survivorship bias的樣本。

## 分級候選：驗收條件與Human權限

[機器可讀定義](../research/v7/window14-evidence-classes.json)，目前單位為兩個market-date回應。
分類與使用許可分開；**任何等級都不自動Research Admitted**。

| 等級候選 | 必要證據／驗收重點 | 目前符合／未來用途與授權 |
|---|---|---|
| VERIFIED_HISTORICAL_ASOF | 原始版本bytes與當時public/receipt等可用性證據在已批准freeze前；revision/QC、calendar/universe/field語意合格；研究另需rights、label成熟/provenance | 0；Human另批資料與研究契約後才可asof ranking研究，不解除Strict Phase0 |
| CONDITIONAL_RESEARCH_PROXY | immutable原始來源、calendar/universe/units、允許用途；Human先批獨立lag/proxy規則、revision假設、missing/support、敏感度與曝光限制，labels合格 | 0；本輪日行情是準備候選，不因可設定lag即自動符合；未來只可明示非PIT的獨立proxy研究 |
| RETROSPECTIVE_ONLY | source/request/session與現在取得的bytes/hash可核，當前QC範圍及歷史缺口明列 | 2；目前限資料工程診斷；模型研究要先補齊proxy或asof資格並另批 |
| UNKNOWN / REJECTED | 必要身份/證據缺失維持UNKNOWN；日期/hash/內容錯誤或required QC失敗則REJECTED，保留原因 | 本輪新回應0；不能默默zero-fill或進模型 |

## C. Conservative Lag Proxy：最低條件與侷限

可研究，**未批准，也未選任何lag數值**。最低條件：
1. 固定可用資料段、精確來源/版本/units/交易範圍與restorebytes；先確認使用權。
2. 依具身份calendar建立lagged knowledge timestamp，以發布/延遲上界證據或明示待批假設
   決定lag；不以股票列數、日期同名或fetcher「約15:00」註解代替證據。
3. Human在看結果前凍結lag/missing/stale/revision、universe、調整價格/量、label maturity、
   selection/refit/forward/final和有限敏感度；失敗/排除sample保留在coverage分母。
4. **時間位移不能消除未知的事後修訂**。若僅有目前修訂版，須明示retrospective/revision
   assumption與偏誤，不能叫Verified PIT、untouched final或原E5重現。缺少可辯護的revision
   限制時，lag本身不足以讓資料進入CONDITIONAL_RESEARCH_PROXY。

官方[每日收盤商品說明](https://eshop.twse.com.tw/zh/product/detail/cfec9a1470e448ec91bfde006db361e8)
列14:00/15:30/17:30產製且不同檔案包含的交易段不同；這是**不同付費商品的現行說明**，
不是2022年MI_INDEX發布ledger，也不保證購買後一定取得原始歷史vintage。先確認供應商
是否保留每次產製版本/發布時間/修訂鏈，再由Human決定是否採購；本輪未購買。

## D. 少特徵新基線是否值得？

**PROPOSED：** 可優先資格一個獨立price/volume family研究候選，降低跨provider的時間與
missingness依賴；只是資料資格方向，沒有選定正式feature list、formula、lag或target。
仍要lookback warmup、corporate-action/volume units、historical universe/labels、完整日期
與immutable dataset。不能從本輪四筆資料直接算rolling features或開始fit。
若另批該候選，需新feature/recipe/dataset身份，明確小於或不同於48-feature E5；不是
原E5重現、不是經典模型架構優勢證明，也不改凍結E5/E1 manifest與label。

## E. Forward Vintage Collection最小補強

沿用v1 Snapshot RequestSpec/不可覆寫raw+manifest/verifier，W07 aware clocks，W09
immutable artifacts。新增最小[契約候選](../research/v7/window14-forward-contract.json)與
`v7_forward_vintage.inspect_record` 唯讀原型；不建立collector/platform/registry。

需先有不可覆寫intent（source/session/request/mode、計畫與候選freeze/deadline），再保留
實際host-clock request/receipt、原始bytes/hash、publication evidence、QC status/evidence、
revision/prior-vintage、missing/late/error。Recovery與Backfill另身份；逾freeze的receipt/QC
不能補回舊snapshot或聲稱當時準時Forward。修訂建立新vintage，不覆寫舊pin。

原型只檢查caller pins/宣告clocks與現在snapshot完整性，**不認證clock真實性、source/QC
證據內容或歷史calendar**。自行改寫所有timestamp再重hash不能產生可信證據；正式collector
仍需trusted clock/事前intent持久化與獨立日志、rights、calendar/Universe與QC的實作驗收。
Missing保持missing，0須真實存在；synthetic資料與具名freeze不是批准政策。

## F–G. 下一步與最小阻塞集合

| 阻塞 | 工程能解決 | 還需Human／外部證據 |
|---|---|---|
| 許可來源/使用範圍 | 保存source/terms身份、允許來源與失敗ledger | 先確認特定API/open-data授權或合約允許internal research/storage/自動取得；不能把公開可查當授權 |
| 歷史asof/修訂 | immutable archive、欄位QC、calendar/universe/label sidecars、restore checks | 缺失的原始vintage無法由工程創造；可能需具版本證據的授權archive，或等待另批Forward資料累積 |
| First CPU study | W07–W11 guards/evaluator/bundle/classical adapters可重用 | 選asof或獨立proxy、feature/dataset身份及證據範圍；freeze/lag/missing、fold/label/purge、selection-refit/statistics/final及小型CPU compute另批 |

**目前不能進入有限真實CPU Baseline Research。** 最小順序是：
1. Human確認允許的資料路徑/用途，並選歷史proxy候選資格或前瞻蒐集準備。
2. 指定可還原bytes、calendar/universe/labels及可辯護的availability/revision證據；若無則明列
   DATA_EVIDENCE_BLOCKED，不以加長lag或合成data解鎖。
3. 資料資格完成後凍結最小研究/評估與CPU資源契約。無須先解決GPU，但GPU仍SAFETY_HALTED。

### 使用條款證據與邊界

[TWSE條款](https://www.twse.com.tw/zh/terms/use.html)限制未經允許的自動下載與重用，並有
政府開放資料例外；本輪尚未確認所查API適用何種授權，後續自動蒐集需先釐清。
[TPEx商店條款](https://eshop.tpex.org.tw/zh/useTerms/index)也有下載/重用限制，但商店條款
不能直接當成公開otc API的精確授權；該endpoint用途權限仍UNKNOWN。TPEx產品頁403未繞過。
[官方文件紀錄](../research/v7/window14-official-documents.json)保留來源與適用限制，非法律許可。
本輪2個資料請求完成後已停止取數；沒有訂閱、對外散布、排程或正式pipeline變更。

StrictSTOP/E1matrixBLOCKED/H2NOT_APPROVED/FairEvalPROPOSED/PreNextOpenPROPOSED/
GPU_SAFETY_HALTED/RecoveryDEFERRED不變。Source/test/review/commit與最小恢復見
[本輪證據](../research/v7/window14-evidence.json)及唯一research-program Window14；本輪後STOP。

驗證：W13整合前後各140CPUtests PASS；W14最終150CPUtests PASS（19.246s），含10個
Forward synthetic tests。兩單元各1輪有證據修復：分類/使用權分離、malformed manifest
容器fail-closed；複審均Critical0/Required0。無真實RankIC/fit/GPU；所有新成果留隔離分支。
