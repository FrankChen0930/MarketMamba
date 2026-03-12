# 🐍 MarketMamba: 深度生成式量化交易預測引擎

![MarketMamba Banner](Parallel_Universes_Plot.png) *(可替換為你的宇宙觀測圖路徑)*

**MarketMamba** 是一個結合了 **狀態空間模型 (Mamba)**、**圖神經網路 (GAT)** 與 **擴散模型 (Diffusion / DDPM)** 的先進量化交易預測系統。

本專案旨在解決傳統金融時間序列預測中「雜訊過高」與「無法量化不確定性」的痛點，透過多重平行宇宙的預測軌跡，尋找市場中的高勝率投資組合。

### 📊 MarketMamba V4.0 全知全能資料矩陣 (The Ultimate Mamba Matrix)

本專案將發揮 Mamba 模型處理高維度矩陣的絕對優勢，將台股微觀籌碼、高頻特徵與國際實體宏觀完美融合。
**戰略 (Hit and Run)：** 歷史 5 年資料庫依賴一個月 Sponsor VIP 權限進行「全市場巨量打包」；實盤階段退回免費方案，採每日增量更新，達成 0 元維運。

| 數據板塊 (Domain) | 核心特徵欄位 (High-Dimensional Features) | 資料表來源 (FinMind API / Others) | Mamba 大腦之戰略作用 (Alpha Value) |
| :--- | :--- | :--- | :--- |
| **1. 國際宏觀與熱錢** | `US_Money_Supply`, `G8_Rate`<br>`Gold/Oil`, `USD_TWD`<br>`US_SOX`, `US_QQQ`, `US_VIX` | `yfinance`<br>FinMind (總體經濟/國際市場) | 建立全球熱錢與通膨的「上帝視角」，從根本預判外資匯出入動能。 |
| **2. 台股量價與防漏** | `Open`, `High`, `Low`, `Close`, `Volume`<br>`US_Market_Closed`, `TW_Typhoon_Day` | `yfinance`<br>本地端 `pd.merge_asof` 計算 | 捕捉價格動能，並嚴格標記休市日以根絕未來數據洩漏 (Lookahead Bias)。 |
| **3. 除權息失真防護** | `Ex_Dividend_Drop` (除息蒸發點數) | FinMind (除權除息結果表) | 告知模型當日價格落差為「配息」而非「暴跌」，防止 Mamba 誤判停損。 |
| **4. 法人與國家隊** | 三大法人買賣超, `Gov_Bank_Buy` | FinMind (三大法人 / 八大行庫) | 追蹤市場大資金流向，以及國家隊暴跌時的護盤底線。 |
| **5. 散戶信用與情緒** | `Margin/Short` (融資券), `Day_Trading` | FinMind (融資融券 / 當沖統計) | 衡量散戶瘋狂程度，避開當沖過熱標的，捕捉融資斷頭潮。 |
| **6. 大戶與暗黑做空** | 大戶/散戶持股比例, `Securities_Lending` | FinMind (股權分散 / 借券明細) | 抓出千張大戶默默吃貨的飆股，提早發覺外資暗黑做空力道。 |
| **7. 主力分點照妖鏡** | `Broker_Branch_Vol` | FinMind (台股分點資料) | 台灣特有：精準抓出「地緣券商吃貨 (公司派)」與「隔日沖大戶倒貨」。 |
| **8. 盤中微觀流動性** | `Bid_Ask_Spread`, `Order_Imbalance` | FinMind (每 5 秒委託成交統計) | 將高頻特徵降維，讓模型具備「流動性嗅覺」，避開買了賣不掉的殭屍股。 |
| **9. 基本面與現金流** | 月營收 YoY/MoM, 現金流, 綜合損益 | FinMind (月營收 / 財務報表) | 破解假帳地雷，篩選具備實質獲利與強大現金流的護城河公司。 |
| **10. 價值與防禦底線** | `PE` (本益比), `PB` (淨值比), `DY` | FinMind (個股 PER) | 建立價值投資防禦底線，避免模型在多頭末期追高泡沫股。 |
| **11. 期權極致微觀** | `Futures_Broker_OI` | FinMind (期貨各券商每日交易) | 放大鏡檢視特定外資在期交所的空單佈局，作為系統崩盤的終極預警。 |

> **⚠️ 終極打包策略：** 以上所有 FinMind 來源皆為免費公開資訊。本專案利用 Sponsor 權限的單日特權，將 5 年歷史資料在 Colab 中一次性高速壓縮為 `.parquet` 本地特徵庫。
> **⚠️ 開發者備註：** `FinMind` 免費版限制為 **300 次 / 小時**。

---

## 🏗️ MarketMamba V4.0 資料工程與時序對齊規範 (Data Engineering Protocol)

為了處理龐大的 7 年期 (2019-01-01 至今) 高維度混合頻率資料，本專案制定以下嚴格的資料庫建置規範，以確保資料的純淨度並徹底根絕未來函數 (Lookahead Bias)。

### 🗂️ 第一層：資料夾分層架構 (Google Drive 儲存規範)
* **`MarketMamba_DB/Raw/` (原始資料層)**
  * `/Daily_Macro/`, `/Daily_Market/`, `/Weekly_Holdings/`, `/Monthly_Revenue/`, `/Quarterly_Financials/`
* **`MarketMamba_DB/Processed_Features/` (最終特徵層)**
  * 存放以單一股票代號命名的 `.parquet` 檔案 (例如 `2330_features.parquet`)。此層資料已完成所有混頻對齊與缺失值填補，可直接輸入 Mamba 模型。

### ⏱️ 第二層：混頻時序對齊守則 (Mixed-Frequency Alignment Rules)
所有對齊皆以「台股每日交易日曆 (Trading Calendar)」為絕對左表 (Left Anchor)。
1. **日頻資料 (Daily)：** 相同日期直接 `merge`。美股日期 `+1 天` (轉為台股可見日) 後使用 `merge_asof(direction='backward')` 對齊。
2. **週頻資料 (Weekly)：** 使用 `merge_asof(direction='backward')` 進行時間戳記對齊，確保觀測值為上週五數據。
3. **月頻/季頻基本面 (Monthly/Quarterly)：** 絕對禁止使用「所屬年月」對齊，必須以 **「公告日 (Announcement Date)」** 作為時間戳記，並使用向後填充 (`ffill`)。

***

# 🐍 MarketMamba 演進史：從踩坑到穩健的量化架構

## 🔴 V1 版本：貪婪的賭徒 (The Greedy Gambler)
* **設計理念：** 結合 Mamba (時間)、GAT (空間) 與 DDPM (不確定性)，打造深度生成式量化引擎。
* **致命失敗：** 1. **OOM 記憶體核彈：** 1930 檔股票全連接圖產生近 370 萬條邊，導致 GPU 頻繁 `CUDA Out of Memory`。
  2. **盲目的貪婪：** 模型只追求「預期獲利最高」，忽略了擴散模型的「波動率 ($\sigma$)」，導致投資組合在黑天鵝事件中崩盤（測試期模擬報酬 -8.95%）。

## 🟢 V2 版本：甦醒的巨獸 (The Awakened Behemoth)
* **工程升級：** 導入 Lazy Loading 動態切片技術化解 RAM 危機；實施 GNN 稀疏化限制每檔股票僅連線 Top 10，將訓練速度暴降至 3 分鐘/Epoch。
* **邏輯進化：** Mamba 感受野擴張至 120 天，並導入總經指標。最關鍵的是，改採 **「夏普分數 ($\mu / \sigma$)」** 選股法，成功將模擬報酬逆轉為正 (+1.87%)。

## 🔵 V3.1 終極對齊版：平行軌跡與凱利資金盤 (The Ultimate Aligned Version)
* **架構升級：** 擴散模型升級為 3D 張量，生成未來 30 天連續軌跡。
* **縮放對齊：** 導入 `SCALE_FACTOR = 10.0` 解決擴散模型處理微小金融數值的尺度幻覺陷阱。
* **資金盤決策：** 結合修改版凱利公式 ($f = \mu / \sigma^2 \times 0.5$) 進行資產配置，達成「預測 + 下單佔比」的全自動化決策。

***

# 🚀 MarketMamba V4.0 演進史：從完美藍圖到實戰洗禮 (The Pure Quant Edition)

## 💡 第一階段：完美的開發藍圖與資料庫基石 (The Ideal Blueprint)
在 V4.0 開發初期，我們建構了頂級的本地 Parquet 資料庫，並制定了極具野心的架構藍圖：
* **資料集規模：** 總筆數 `3,711,525` 筆，涵蓋 `2,782` 檔標的，時間橫跨 2019 至 2026 年。
* **多模態特徵 (56維度)：** 包含宏觀國際、微觀籌碼、基本面估值與超額報酬 (Alpha_1d)。
* **安檢報告 (QA)：** 缺失值與無限值皆為 `0`，並設立 60 日 IPO 隔離牆。
* **構想中的架構：** 深度 Mamba 擴容 (`d_model=256~512`) 搭配全局注意力機制 (Global Attention)，並使用高達 2.5 倍的方向感知損失函數 (Directional-Aware Loss) 逼迫模型提高勝率。

## 🔴 第二階段：實戰危機與巨獸搶救大作戰 (The Reality & Surgery)
然而，「過早的最佳化是萬惡之源」。當藍圖投入 A100 進行全市場實盤訓練時，我們遭遇了量化領域最殘酷的兩大災難：

### 💥 危機一：全連接注意力的坍塌 (Attention Collapse)
* **災難現象：** 全市場 1900 檔股票的預測軌跡一模一樣（停滯在 `-6.65%`）。
* **大腦斷層掃描：** Mamba 獨立思考層極度健康 (變異數 `0.2135`)，但 Attention 層完全死水 (變異數 `0.0000`)。
* **微創手術 (The Surgery)：** 因強行全連接導致模型被雜訊淹沒。我們利用殘差連接 (Residual Connection) 直接 **切除** 壞死的 Attention 層，凍結練了 24 小時的 Mamba 權重，僅重新訓練全新的 Output Head，成功找回個股的獨立特徵。

### 💥 危機二：安全常數陷阱 (The Safe Constant Trap)
* **災難現象：** 切除 Attention 後，預測軌跡變成了一條完美的水平直線，死鎖在微小的 `0.0075`。
* **解除封印 (Brave Mode)：** 源自於 `方向感知損失函數` 嚴厲的做錯懲罰，引發了模型的恐懼而選擇盲猜安全常數。我們在最終微調階段拔除方向懲罰，回歸純粹的 `MSE Loss`，並使用「差異化學習率」讓模型重新勇敢學習市場的高低起伏。

## 🏆 第三階段：V4.0 最終上線型態 (The Production Baseline)
經過兩次史詩級的除錯與搶救，目前的 **MarketMamba V4.0 (V4_GodMode_Production)** 暫時擱置了全局 Attention，將 Mamba 對於「120 天長時間序列」的特徵萃取能力推展到極致。這個 **「純血 Mamba 完全體」** 徹底擺脫了坍塌危機，成為本專案強悍且穩定的實盤基準線，目前已成功對接 Streamlit 前端與 GitHub 全自動推播管線。

***

# 🚀 接下來的 Immediate Plan：V4.1 除權息還原股價升級

在 V4.0 穩定運行的同時，我們發現了資料源底層的一個致命隱患，這將是我們下一個優先解決的升級項目。

## ⚠️ 發現的問題 (The Ex-Dividend Trap)
台股具有「高殖利率」特性。當遇到大型股票除息時，原始股價 (`Close`) 會在一夜之間出現巨大的「假性跳水」。
1. **目標變數污染：** 模型會將該次除息誤判為 `Future_5d_Return = -20%` 的暴跌，導致學習到錯誤的特徵映射。
2. **技術指標失真：** 均線、布林通道與乖離率會因為假性跳水而嚴重扭曲。

## 🎯 V4.1 解決方案 (The Adjusted Price Protocol)
全面將 V4.0 的基石資料從「原始股價」替換為「還原權息股價 (Adjusted Price)」。
1. **API 替換：** 將 FinMind 呼叫的 dataset 參數由 `TaiwanStockPrice` 更改為 `TaiwanStockAdjustedPrice`。
2. **資料庫重建：** 在 Google Drive 建立全新的 `Raw/Daily_Adjusted_Price` 資料夾。
3. **特徵大融合與重算：** 確保所有依賴價格的衍生特徵 (`Return_1d`, `MA_5`, `Future_5d_Return`) 皆使用 `AdjustedClose` 進行計算。
*(預期結果：消滅除權息導致的報酬率異常，大幅提升模型在每年 6~8 月除權息旺季的預測穩定度。)*

---
*Built with passion and tons of coffee. ☕*
