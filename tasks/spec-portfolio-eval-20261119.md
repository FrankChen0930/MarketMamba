# 預先登記 P1：C0+ / LightGBM 組合層（扣成本）評估

登記日期：2026-11-19（任何組合結果出現之前）。本節登記後只追加、不修改。證據等級上限 E2，RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED。

## 目的
先前結論只看排序 IC。上線決策需回答：扣除成本與流動性限制後，訊號能不能產生淨報酬？集中在小型股的 IC 是否可交易？容量多大？

## 輸入與重用（不新建框架）
- 預測：沿用 `V6/research_lab/rolling_runner.py` 與 `c0plus.py`、滾動折目錄 `/home/frank/projects/ruflo-evaluation/pilot09/marketmamba/.local-data/matrices/rolling/fold_20{21..26}`（唯讀）。需新增「保存每折評估列分數」的薄擴充，不改既有 IC 輸出語意。
- 價格：`/home/frank/projects/MarketMamba/Data/processed_v6` 的 prices_raw（唯讀；不得寫入、備份、修復該目錄；不得讀 `.env`）。
- 回測：沿用 `V6/research_lab/portfolio.py` 的 `run_backtest`／`Config`（訊號於收盤、次一觀測日開盤成交）。
- 系統：LightGBM 基準對 C0+，標籤 y5，種子 17／29／43（各折分數以種子間 rank 平均），`max_train_rows=500000`。折：2021–2023 為乾淨折（確認性），2024–2026 為描述性。

## 固定的主要假設（確認性，登記後不得更換）
主要格：`top_k=50`、`rebalance_sessions=10`、等權、`min_dollar_volume=2e7`（以 close×volume 的新台幣成交值，前一日可得資訊）、`commission=.001425`、`commission_discount=1.0`、`sell_tax=.003`、`slippage_bps=10`、`price_basis` 優先 `total_return`（無 adj_close 則 `raw` 並明載）。
H-P1：乾淨折 2021–2023 合併，C0+ 主要格的淨超額報酬（對同一可交易宇宙的等權基準）逐日序列，以區塊 bootstrap（區塊 20）的 95% CI 下界 > 0 → 「淨有益」；否則「未證實」。同樣報告 LightGBM。C0+ − LightGBM 的淨報酬差為次要、描述。
無任何其他格可升格成確認性結論。

## 描述性敏感度網格（全部報告，禁止只挑好的）
top_k ∈ {20, 50, 100}；rebalance ∈ {5, 10, 20}；min_dollar_volume ∈ {0, 1e7, 2e7, 5e7}；slippage_bps ∈ {0, 10, 25}；commission_discount ∈ {1.0, 0.6}。價格基準 raw 對 total_return 各一次。市值分層（小／中／大，每日截面三分位，使用前一日可得市值）分別報告淨報酬、換手、IC。
輸出：年化淨報酬、Sharpe、最大回撤、年換手、成本拖累（毛減淨）、持有檔數、被封鎖／無法成交持股計數、對基準超額；容量代理＝假設資產 1e6／1e7／1e8 新台幣時，單筆成交值占該股前一日成交值的 median 與 95 分位。

## 必須誠實標示
標籤與價格樣本含存活者偏差風險（prices_raw 是否含下市股未證實，需在報告列出實際下市／缺價處理）；市值為現在股數快照（非 PIT）；漲跌停與零成交的無法成交處理沿用 `portfolio.py` 語意，需列出受影響比例；不同折、不同協定的數字不得與先前 IC 直接比較；2024–2026 為描述性。成本為研究假設，非券商實際費率。

## 工程要求
- 新程式放 `V6/research_lab/`；回歸測試必須在舊（壞）行為下失敗（mutation 控制），含：成本被移除、訊號用到當日而非前一日、封鎖持股被忽略、分層用到未來資訊。
- 輸出含輸入檔 SHA-256、source commit、環境、種子、折、參數；寫入 `.artifacts/p1/`（不覆寫、不進 Git）。
- 不抓資料、不用 GPU、不連網、不改 `Data/`。
- 完整訓練（約 36 次擬合）超過單 worker 上限，由主控以背景程序執行；worker 只做程式、測試與小規模冒煙（1 折、1 種子、小 `max_train_rows`）。
