# 第三步已曝光歷史洩漏稽核（2026-11-28）

T5 洗牌 sanity 通過；T2 現存來源的子建構15/15通過，但原 shared X 的完整 frozen 重播未證實。T1 查出來源時點／往回還原水準風險，T4 查出存活者偏差。36次無尺度對照中，早期 IC 下降0.00704（MINOR），後期反而增加0.00274（NOT_MATERIAL），兩期皆未達預定 material 門檻；因此不能把約0.15的 IC 主要歸因於這15個價格量綱欄。T6 已修缺價出場，主情境年化淨報酬14.36%／21.65%，但仍不是可交易或前瞻成功證據。第三步仍是有來源與選樣限制的已曝光歷史；不改凍結規格、決定檔或前瞻家族，strict PIT／execution 的缺證據仍須補齊。

證據等級：`EXPOSED_HISTORICAL_AUDIT_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED`。Human 明確核准本案新增擬合；沒有 GPU、網路、資料取得、排程、production、commit／push／merge。完成不等於 Human acceptance。

## 身分與邊界

- Source HEAD：`b447ca53ac5d63088b07c48129f27a1c7ee88be3`；指派 detached worktree `builder_leak_audit`。
- 原始 bundle：主 checkout `/home/frank/projects/MarketMamba/.artifacts/step3/bundle-72`；shared：`/home/frank/projects/p4-bundles/prepared/shared`。原始凍結檔案不修改。
- 本案結果：主 checkout `.artifacts/leak-audit/run-20261128-builder/`；逐列／TEJ 授權內容、模型與預測只存私有路徑，公開報告僅彙總。
- runtime：CPU Python `/home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python`，`CUDA_VISIBLE_DEVICES=''`；未讀研究 repo 的歷史結果。一般執行 `MARKETMAMBA_DATA_ROOT` 未設定，所有資料使用主 checkout 絕對路徑；擾動 replay 的 helper 初始化改用 `/tmp/marketmamba-leak-audit-runtime`，避免初始化正式 Data。Data realpath 為 `/home/frank/projects/MarketMamba/Data`，部分 processed 檔連到 Windows `/mnt/d/Desktop/work/ProjectForMe/MarketMamba/Data/processed_v6/`，不推定兩個 checkout authority 相同。
- 新 namespace 必要性：已有 step3／tradability artifact 對應原凍結研究與舊診斷，不能覆寫為本次新擬合／缺價政策。依 Human 指定使用 `leak-audit`，保留舊證據。

## T1：48 欄來源與時間

真正來源鏈為 `V6/experimental/v7_integrated_train.py::checkpointed_features` → `gap_safe_features` → `V6/marketmamba/data/feature_engineer.py::build_features` → `align_point_in_time_features` → 每股 `.prepare-cache/*.parquet` → `V6/experimental/v7_corrected_e5_matrix.py::build_arrays`。最後一步只取 frozen 48 欄、與歷史 universe join、轉 float32 及缺值補零；不是重新計算來源。

**manifest 與實際值有落差**：shared manifest 寫 causal per-session／macro expanding scaling；但 `build_arrays` 消費未縮放的 `.prepare-cache`，`X.npy` 保留實際水準。例如第一列 Close 約 17.998056、Volume 8,056,429、ATR 約 0.264033。不能因 manifest 的文字就認定價格水準已被消除，也沒有在本案靜默修正舊 manifest。

以下 `OK` 只表示固定快照中的計算沒有向未來讀取；並非逐期 source vintage／strict PIT 認證。`LEAK` 的美國欄以 **t 日台股收盤** 為資訊界線；若實際流程是翌日美股收盤後、台股開盤前才產生分數，需另有收到時間與計算紀錄才能解除該時鐘問題。不能把兩個決策時點混用。

| 欄 | 來源／運算 | 對齊與 t 日可得性 | 判斷 |
|---|---|---|---|
| Open | FinMind TaiwanStockPriceAdj 開盤 | t；當日可觀察但水準被未來除權息回改 | LEAK |
| High | 同來源最高 | t 收盤後完整；往回還原 | LEAK |
| Low | 同來源最低 | t 收盤後完整；往回還原 | LEAK |
| Close | 同來源收盤 | t 收盤；往回還原 | LEAK |
| Volume | 同價格表成交量 | t 收盤後；不是 Close×Volume | OK |
| Return_1d | 還原 Close pct_change(1) | 截至 t，連續有效片段；未來共同乘數理論抵消 | OK |
| Return_5d | 同來源 pct_change(5) | 同上 | OK |
| Return_20d | 同來源 pct_change(20) | 同上 | OK |
| MA_20 | 還原 Close rolling20，min10 | 含 t；保留價格量綱 | LEAK |
| MA_60 | 還原 Close rolling60，min30 | 含 t；保留價格量綱 | LEAK |
| RSI_14 | Close 的歷史漲跌平滑 | 截至 t；無未來索引，來源修訂仍未知 | OK |
| ATR_14 | 還原 OHLC true range、ewm14，min7 | 含 t；保留價格量綱 | LEAK |
| RS_5d | Return_5d−TWII 5 日報酬 | 同日台股指數；截至 t | OK |
| RS_20d | Return_20d−TWII 20 日報酬 | 同上 | OK |
| RS_60d | Close 60 日報酬−TWII 60 日報酬 | 同上 | OK |
| Foreign_Buy | institutional_raw 外資買量 | Date=t 直接 join；盤後，缺實際發布／收到時間 | LAG_RISK |
| Foreign_Sell | 同來源外資賣量 | 同上 | LAG_RISK |
| Foreign_Net | 同來源外資買賣差 | 同上 | LAG_RISK |
| Investment_Trust_Net | 同來源投信買賣差 | 同上 | LAG_RISK |
| Dealer_Net | 同來源自營買賣差 | 同上 | LAG_RISK |
| Margin_Purchase | margin_raw 融資買進 | Date=t；可能晚於盤後推論，缺 receipt | LAG_RISK |
| Margin_Repay | 同來源融資償還 | 同上 | LAG_RISK |
| Short_Sale | 同來源融券賣出 | 同上 | LAG_RISK |
| Short_Cover | 同來源融券回補 | 同上 | LAG_RISK |
| Margin_Balance | 同來源融資餘額 | 同上 | LAG_RISK |
| Short_Balance | 同來源融券餘額 | 同上 | LAG_RISK |
| Day_Trade_Volume | daytrade_raw 比率欄，clip[0,1] | Date=t；實際為比率而非原始股數，盤後時點未知 | LAG_RISK |
| KD_K | OHLC 9 期 RSV、ewm(com2) | 截至 t；含數值 epsilon | OK |
| KD_D | K 再 ewm(com2) | 截至 t | OK |
| OBV | sign(Return_1d)×Volume 累加，再 rolling60 z-score | 截至 t；不乘價格 | OK |
| Volatility_20d | Close log-return rolling20 std | 截至 t；無向未來索引 | OK |
| Holdings_Large_Pct | holdings_raw：1−Retail_Hold_Ratio/100 | Week 直接改 Date，同日 join 後逐股 ffill；沒有公布日 lag | LAG_RISK |
| Holdings_Large_Change | 上欄 diff(5) | 同一錯位可能往後傳遞；不是依發布事件計算 | LAG_RISK |
| Securities_Balance | securities_raw 的 Securities_Lending | **每日**直接 join／ffill；三日核對為借券成交量加總，非餘額 | LAG_RISK |
| Foreign_Holding_Pct | foreign_shareholding_raw 比率欄 | Date=t join／ffill；發布與 vintage 未證實 | LAG_RISK |
| Market_Cap_Log | market_value_raw 的 log1p(market_value) | PIT override 僅 backward asof，最多5 sessions；股數／vintage 未認證 | LAG_RISK |
| TWII_Return | macro_raw.TWII_Close pct_change | 台股同日收盤可觀察；補日／來源修訂有限制 | OK |
| SPX_Return | macro_raw.SPX pct_change | 同曆日美國收盤晚於台股 t 收盤；本來源不是 US_QQQ 分支 | LEAK |
| VIX | macro_raw.VIX | 同曆日美國值未 shift；t 收盤不可知 | LEAK |
| TNX | macro_raw.TNX | 同上 | LEAK |
| Gold_Return | macro_raw.Gold pct_change | 同曆日國際收盤／結算；缺時區與收到時間 | LAG_RISK |
| Oil_Return | macro_raw.Oil pct_change | 同上 | LAG_RISK |
| USD_TWD | macro_raw 匯率欄 | 同曆日，缺具體定盤與收到時間 | LAG_RISK |
| Futures_OI_Foreign | futures_institutional_raw 外資多空 OI 差 | Date=t，盤後發布時點未驗 | LAG_RISK |
| Options_PC_Ratio | options_institutional_raw put/call volume | Date=t，盤後發布時點未驗 | LAG_RISK |
| Fear_Greed | separate fear_greed（本 macro 無 CNN_FearGreed） | 同曆日 join／ffill；美國更新時點未驗 | LAG_RISK |
| Business_Signal | separate business_indicator.monitoring | 本 macro 無 TW_Biz_Signal；period 月初＋2個月 backward asof，非真正 publication vintage | LAG_RISK |
| FED_Rate | separate fed_rate fallback | 本 macro 無 FED_Rate；原 helper 已警示來源只有8筆2004日期，歷史值不可信 | LAG_RISK |

具體日期核對（2330；原始 cache／processed／prepared 身分與逐列內容私存 `t1_examples.json`）：

| signal | 集保來源 Week／retail % → prepared large | 原始每日借券 volume 加總 → prepared | Business period→生效日／分數 | prepared SPX_Return／VIX／TNX |
|---|---|---|---|---|
| 2024-01-05 | 01-05／0.52 → 0.9948，沒有 publication lag | 4 筆合計1426 →1426 | 2023-11-01→2024-01-01／20 | 0.001825686／13.35／4.042 |
| 2024-04-30 | 04-26／0.48 → 0.9952，ffill | 2 筆合計326 →326 | 2024-02-01→2024-04-01／29 | −0.015730514／15.65／4.686 |
| 2024-07-04 | 06-28／0.50 → 0.9950，ffill | 8 筆合計7732 →7732 | 2024-05-01→2024-07-01／36 | 0／20／4（來源缺值的預設，不是真實同日觀測） |

原始檔無 publication_at／received_at，不能由檔名的 Friday 或 Date=t 推論當晚／翌日開盤以前一定可得。集保的發布相對 Week 時序仍未獲離線證據，故列 LAG_RISK；借券並非題目推測的每週來源，而且誤把成交流量稱為餘額是另一個語義風險。外資、投信、融資融券同日資料都不應當成收盤前可知；是否能供翌日開盤使用，需盤後到達證據。

## T2：未來擾動

20 檔 × 2023–2024，2022 暖身；截點 2023-06-30、2023-12-29、2024-06-28；seeds 0／1／2／3／4。在固定原始快照中，擾動截點以後的所有 numeric payload（價格、量、籌碼、macro 等），保留日期、stock_id 與支援。重跑原 `gap_safe_features`／`build_features`；另補驗原 PIT override。逐一比對 ≤t 的 float32 48 欄 bytes，包含 NaN 的位元，不用 allclose。

含 Market_Cap_Log PIT override 的最終重播 **15/15 相同、0個差異 cell**；每組 prefix 分別為2,260／4,780／7,119個公司×日，48欄全部 bitwise 不變（精確列數以 t2.json 為準）。狀態為 `PASS_SUBBUILDER / ORIGINAL_SHARED_NOT_REPRODUCED`。這不是 full shared X 重建：沒有重建完整歷史 universe／最終 join，有限暖身不等於2005起的原始數值；20份 raw source 有14份整檔 hash 與舊 prepared identity 不同，helper hash 也不同。不能因這次現存來源的因果運算通過，就認定當初 shared 的完整來源重播已通過。舊營收 PeriodIndex API 不相容目前 pandas，而 Revenue_MoM／YoY 已排除於48欄，故 PIT override 僅重播保留的市值欄，不修改 V6。

更關鍵的是，擾動已儲存的 t 以後價格，不會重新計算公司行動對 ≤t 還原價的影響；T2 通過不能反駁 T1/T3 的來源洩漏。各 raw source、helper 與 frozen identity 的比較保存於 `t2.json`，不升格 strict PIT。尚無可逐檔驗證的完整原始版本，原 shared X 端到端擾動驗證仍未完成，明確列 Remaining。

## T3：價格水準與還原因子

欄位身分先釐清：`tools/step3_runner.py::build_arms` 的原 A=185、原 B=190。185＝48＋16個 CORE×（5 lags＋3 rolling means）＋3個 MARKET×3聚合；B 再加5個月營收欄。題目所稱「A的190欄」不是原 A 的 schema。本案將**原 B190 明稱 audit A190**，只做36個 matched scalefree fits，不偷換成185欄，不多開另一家族。原 A185 僅保留參考。

T3a：`F = frozen snapshot adjusted Close / (原始 Trading_money / Trading_Volume)`。這是**近似還原因子**：分母是 VWAP，並非實際未還原收盤，所以同時含 close/VWAP 的日內項，不能將其關聯全部歸因未來公司行動。分母缺失／非正即 UNKNOWN，不以還原價×量補成交金額。每日截面至少20家，平均 Spearman；讀取5,237份日檔，7,749,511個有正成交均價的日股。成交額來源截至2026-04-24，與 eval 共同支援的有效日數為1,468／1,285；不將來源尾端未知補0。三個2330實例的 frozen snapshot Close 與 prepared Close 完全相同；daily raw cache 的 adjusted close 略有不同，不混用快照。

| F 的另一側 | 2015–2020 平均ρ | 2021–2026 平均ρ |
|---|---:|---:|
| h20標籤 | −0.082443 | −0.076272 |
| Open | −0.045453 | −0.233380 |
| High | −0.043058 | −0.230408 |
| Low | −0.047495 | −0.235220 |
| Close | −0.045088 | −0.231593 |
| MA_20 | −0.042644 | −0.231068 |
| MA_60 | −0.036960 | −0.227124 |
| ATR_14 | 0.125493 | −0.030306 |

T3b：15個有價格量綱的欄（Open／High／Low／Close／MA_20／MA_60／ATR_14，以及 ATR 的5個 lag、3個 mean）全部除以 signal 同日還原 Close；ATR 衍生欄也除以**當日** Close，不重抽樣、不重新學 scaler。新增 `log(Trading_money/Trading_Volume)`，190→191；合法股價水準仍有來源缺值，不補假值。其餘175欄逐值保留。Volume／籌碼為量，OBV 是量的方向累積後 rolling z-score，沒有價格乘數；Day_Trade_Volume 是比率。Market_Cap_Log 由另一份 market_value_raw 提供，未以還原 Close 重算，保留合法 size 資訊，但其 PIT／股數來源限制仍在。

原 12折×seeds17／29／43、500,000個原抽樣列、完整原 target、200棵／31葉／learning_rate .05／min_child200／subsample .8／colsample .8／lambda10／4 CPU threads。原 rank-target、purge、W60、eval keys／labels 全部不改。每 job 保存 transformed matrix digest、sample／target／keys digest、參數、model／prediction hash。新增一欄會依原 colsample 參數改變實際隨機欄抽樣，這是預定轉換的組成，不聲稱僅有單一訊息變量。

判定事先固定：任一期間下降≥0.02 → `PRICE_LEVEL_LEAK_MATERIAL`；0.005≤下降<0.02 → `MINOR`；下降<0.005 → `NOT_MATERIAL`。每日配對（原−scalefree）循環 block20／1000 replicates／seed17／95% CI，保持無效日的位置。不以 CI 或漂亮結果事後改門檻。實測結果如下，不因結果改門檻。

| 期間 | 原A185參考IC | audit A190（原B）IC | A-scalefree191 IC | 下降（原−無尺度） | 下降95% CI | 有效日 | 固定判定 |
|---|---:|---:|---:|---:|---|---:|---|
|2015–2020|0.157147|0.162916|0.155880|0.007036|[0.003290, 0.011156]|1468|MINOR|
|2021–2026|0.147028|0.151981|0.154725|−0.002744|[−0.006564, 0.000885]|1367|NOT_MATERIAL|

36/36 fits 完成；依事先門檻，**未觸發 PRICE_LEVEL_LEAK_MATERIAL，整體最多為 MINOR**。早期 CI 下界仍小於0.005，不因此更改以點估計分類的規則。後期無尺度臂較高，不以此聲稱新模型勝出或 sources 已無洩漏。這個對照只隔離指定15欄的價格量綱；來源修訂、Market_Cap_Log 的 PIT、存活者與其他 clock 風險仍在，不能把所有差異都解讀為還原因子的純因果效果。

runtime 為 Python3.12.3／NumPy2.5.4／pandas3.1.1／LightGBM4.7.0；原 fit manifest 未記 package versions，不能證明與原 fit 套件版本完全相同。參數、抽樣、target、鍵與 frozen 矩陣相同；原模型在目前 runtime 的推論重現另列於 validation。新 bundle 採原190欄矩陣的 immutable references＋轉換 recipe／digest，不持久化36份巨型 transformed matrix；均價、模型、預測與摘要保存於指定私有 namespace。

## T4：股票池／存活者偏差

原樣本沿用 shared 的公司×session 鍵，再依完整價格及 W60 限定，沒有在第三步新建歷史全股票池。`build_arrays` 的 universe inner join 不能補回上游已消失的公司。TEJ 比較依 TSE／OTC 最早上市日至下市日前的研究區間，取當年有交集者；2026 截至矩陣最後日2026-09-11。逐年表與確切遺漏集合私存 `t4.json`／`t4_omitted.parquet`。

| 年 | shared池家數 | TEJ在市對照 | 對照遺漏 | 當年下市 | 下市在池內／遺漏 |
|---|---:|---:|---:|---:|---:|
| 2015 |1439|1612|173|9|0／9|
| 2016 |1494|1661|167|22|0／22|
| 2017 |1532|1680|148|16|0／16|
| 2018 |1592|1726|134|19|0／19|
| 2019 |1630|1742|112|12|0／12|
| 2020 |1660|1760|100|17|0／17|
| 2021 |1689|1772|83|15|0／15|
| 2022 |1740|1804|65|16|0／16|
| 2023 |1784|1821|46|8|0／8|
| 2024 |1851|1870|37|10|0／10|
| 2025 |1928|1932|25|8|0／8|
| 2026至09-11 |1952|1962|34|7|0／7|

**SURVIVOR_BIAS**：12年對照的159個當年下市事件群全部未進當年研究池。TEJ 現況 basic snapshot 的生效日並非 publication PIT，但遺漏下市群已足以否決「無存活者偏差」。`Data/v7_exchange_verified/manifest.json` 的較早價格准入階段明寫 current ordinary-stock universe；原 `selected_universe` 是宣告 universe 的跨日聯集。**尚未找出能驗證最早完整選股名單與特定晚期快照日期的生成紀錄**；`research/v7/corrected-baseline-v1/universe-manifest.json` 在本 checkout 不存在，不能由舊文件補推 authority。現有 yearly parquet 與 shared 的2015股票數同為1439，不能反推完整歷史股票都進了上游。

此外2022–2026依序有1／9／18／21／24家 pool 公司不在該年 TEJ 在市區間，顯示兩來源邊界不全一致，不用相減抵銷遺漏。TEJ對照是該 basic 檔的 TSE／OTC研究範圍，非已認證的歷史普通股完整母體。暫停交易沒有合格事件歷史，列 UNKNOWN，不能補0或推定都在池內。

## T5：評估 sanity

固定 B score，各日只在成熟共同支援內打亂當日標籤10次（固定 RNG seed17），維持日與股票數，不跨日洗牌。主參照取前一個完整 session 的 Return_20d（嚴格截至 t−1），不跳過缺 session；另完整保留截至 t 收盤的版本及兩者負號反轉。使用相同每日 Spearman 與至少20家規則。

| trial | 2015–2020 平均IC | 2021–2026 平均IC |
|---|---:|---:|
|0|0.0008007|0.0000954|
|1|−0.0001026|−0.0007762|
|2|0.0001986|−0.0005862|
|3|0.0002326|0.0004840|
|4|0.0009106|−0.0008527|
|5|−0.0003379|0.0008904|
|6|0.0003947|−0.0004625|
|7|0.0005364|0.0007810|
|8|0.0008979|0.0000945|
|9|0.0002133|−0.0003968|

十次皆接近0（絕對平均IC皆<0.001，僅描述，沒有新增驗收門檻）。嚴格 t 以前20日動能 IC 為 −0.022038／−0.025386，反轉為 +0.022038／+0.025386；截至 t 收盤的版本為 −0.023381／−0.028804，反轉為 +0.023381／+0.028804。兩版本有效日皆1,468／1,367。原模型約0.15的量級遠大於這個單因子參照。T5 sanity 通過不會消除 label／source／universe 洩漏。

## T6：缺價退出投組修正

`tools/tradability_diagnostic.py` 新增 `exit_missing_marks`，`simulate_portfolios` 預設採 last_close。持有期第一個缺失／非正收盤後永久變現金，後續就算價格重現也不重進；主情境用最後可得收盤，首個持有日就缺價時用 signal 日收盤，保守情境從缺價日記該股−100%。不再因選中股票或同宇宙等權基準的未來缺價而排除整個週期。保留未知 entry open／來源不足的 fail-closed；退出手續費／交易稅／滑價只記一次，下一週期 turnover 排除已退出持股，不假裝還持有。最後可得價退出是診斷假設，不是有撮合證據的真實可執行退出。

重報 B、前20 sessions 日均實際成交額≥1,000萬元、原≥10元／翌日量與漲停代理 gate、前10%做多、20個固定 offset、slippage 0.1%。只要求進場來源足夠，不再因持有期跨 FinMind 成交額來源尾端而排除；持有期仍需落在 frozen price calendar。兩期間以各 offset 支援片段的年化淨報酬／同宇宙等權超額取平均，不能說是整段真實可交易收益。結果如下。

沿用的 t+1 全日量>0 條件也是事後可交易代理，不是開盤前已觀測的撮合證據；本案不將它升格 live admission gate。

| 退出政策 | 期間 | B前10%年化淨報酬 | 相對同宇宙等權年化超額 | offset超額範圍 | 平均有效sessions／offset |
|---|---|---:|---:|---:|---:|
| 最後收盤變現（主） |2015–2020|14.3574%|9.6538%|4.9768%～13.6307%|1456.5|
| 最後收盤變現（主） |2021–2026|21.6543%|10.5901%|7.8047%～16.2317%|1294.5|
| 缺價股−100% |2015–2020|9.7332%|10.0712%|4.2465%～13.9451%|1456.5|
| 缺價股−100% |2021–2026|18.9102%|11.2980%|7.3790%～17.2497%|1294.5|

兩政策均20個 offset。−100%敏感度使 B 的**絕對淨報酬下降**，但基準也受缺價損失，故相對超額略升；不能將相對數字上升稱為風險變小。這些仍受原研究池存活者偏差、原分數來源問題、進場代理與來源尾端支援限制，不能當成補救後正式回測。

支援覆蓋2015-01-06至2026-05-22；兩政策皆排除78個來源不足日、20個超出日曆 horizon 日、1個缺進場價日。舊 JSON key `missing_selected_price` 為相容讀取保留，但新程式只計**進場**缺價，不計持有期缺價；後者不再丟棄週期。此輸出不再與舊完整價格條件片段混稱同一政策。全 CLI 的產出說明與 manifest 規則亦同步標明新退出政策；舊 report／artifact 不改。

## 驗證、限制與交接

新增測試目前22 passed，涵蓋擾動15格、非因果 builder 的負例、無尺度欄位／非法分母／其他欄保留、NaN→JSON null、缺價後不重入、−100%、週期不排除與退出成本不重複。依 Human 指示**不跑完整測試**，亦未啟動舊測試全集。初次 T2 因 parquet 字串日期篩選型別不合失敗，改為股票 predicate 後 normalize 日期；初次 facts 因來源 NaN 無法序列化 JSON 失敗，新增 failing test 後將未知記 null，沒有補0；原 logs 保留。

最終 `validation.json`：22個新增測試、36個 COMPLETE fits、48個 transformed matrices（12 eval＋36 fit）全部重新計算並逐digest吻合；每個模型與預測逐檔hash重驗，sample／target／keys身分符合原B190。原A/B兩期間四格IC重現最大誤差0；原2021-B-seed17模型以目前 runtime 重做推論，最大預測誤差0。36 fits 各34.13–49.68秒。四個Python檔syntax通過、`git diff --check`通過、新檔trailing whitespace為0；正式Data／V6/models／tasks規格與決定檔diff為空。

人工 knowledge-impact 判定：研究投組缺價 semantics／相應 gate、研究結論與 blocker 有改變，因此在同一change更新 Current State、Authority Map、Features／Data／Universe導航；formal authority、前瞻家族、production boundary／runtime均不變，沒有supersession／deprecation。機械 impact 對10個路徑為WARNING（knowledge更新提示）；`knowledge_health.py --ci --json` 為WARNING、無FAIL：Project Map／Agent Guides／Domains／Runtime Freshness／Generated Index為PASS，Governance／Authority Drift／Legacy Docs為WARNING。不自動repin authority或修不相關legacy文件。

所有新能力為 branch-local 未提交候選，沒有前瞻成功或 Human acceptance。可重建較大檔為 `actual_average*.npy`、`original.npz`、`t2_baseline.npy`；來源仍可驗證且無唯一依賴後才可刪。保留 manifest、hash、結果、模型、預測、每日 paired IC 與 logs。本輪不刪任何大檔，原33GB bundle不屬本案清理範圍。`source_at_closure.json` 保存未提交交付碼與文件，原因是Human禁止commit而需可審查的source identity；這是closure版本，不冒充每個階段載入的版本，實際matrix數值以48個重建digest證明一致。最終 `manifest.json` 列逐檔size／SHA、base commit、來源、環境與evidence class。

重播入口（從本指派worktree執行；所有真資料／產物用主checkout絕對路徑）：

```bash
PYTHONPATH=V6:. CUDA_VISIBLE_DEVICES='' /home/frank/projects/ruflo-evaluation/pilot09/.venv/bin/python -m tools.leak_audit_run --out /home/frank/projects/MarketMamba/.artifacts/leak-audit/run-20261128-builder --phase validation 2>&1 | head -c 3000
```

完整有界phase為 facts（來源／F／原輸出）→fits→evaluation→universe→sanity→perturbation→portfolios→validation→finalize；perturbation另設 `MARKETMAMBA_DATA_ROOT=/tmp/marketmamba-leak-audit-runtime`。不得據此重開研究家族或忽略來源gate；實際執行紀錄與失敗trial logs保留於同namespace。沒有新排程或自動喚醒。

| 固定交接欄位 | 內容 |
|---|---|
| Completed | T1來源稽核、T3a／T3b36 fits、T4、T5、T6；T2現存來源子建構15組完成，原shared端到端未完成 |
| Changed | `tools/leak_audit.py`、`tools/leak_audit_run.py`、`tools/tradability_diagnostic.py`、`tests/test_leak_audit.py`、本報告；knowledge的Current_State／Authority_Map／Features／Data／Universe共5檔 |
| Decisions | 原A185／B190身分保留；本案audit A190明指原B；T3依原門檻MINOR／NOT_MATERIAL，未達material；不改凍結規格、決定或前瞻家族 |
| Evidence | 私有`run-20261128-builder/manifest.json`、`validation.json`、T1–T6摘要／daily arrays／36模型與預測；全部是已曝光歷史稽核 |
| Remaining | 原shared完整frozen來源與helper重播；精確publication／receipt、最早完整選股快照與暫停歷史；剩餘高IC原因未完全定位；協調者完整測試與獨立review |
| Blockers | 14/20來源與helper hash不符舊prepared；部分上游manifest缺失；publication／vintage／execution未知。不能宣稱strict PIT或真實可交易績效 |
| Next recommended action | 協調者獨立核對本案資料身分、門檻與缺價政策；由Human決定來源與研究池修復，禁止依此自動改前瞻 |
| Project memory updated | YES，Current State與3個domain notes；不是Human acceptance |
| Authority map updated | YES，僅導航與evidence boundary；machine authority未變 |
| Git status | detached HEAD b447ca5；預期6個tracked修改＋4個untracked新檔；開工時clean，保留未提交差異 |
| Commit | N/A，Human本task明確禁止 |
| Push | N/A，禁止；亦未merge／deploy |
| Worktree | 指派`/home/frank/projects/MarketMamba/.worktrees/ruflo-workers/run-mx0aophh-742f49/builder_leak_audit`；owner builder、memory namespace job-run-mx0aophh-742f49；retired N/A。移除前須協調者保存未提交10檔與私有artifact依賴 |
| Runtime/environment | WSL CPU、Python3.12.3／NumPy2.5.4／pandas3.1.1／LightGBM4.7.0；無GPU／網路／資料取得；擬合前/mnt/d可用141GiB，後續140GiB，逐fit檢查≥60GiB；未新增／修改排程，無遺留本案計算程序 |
