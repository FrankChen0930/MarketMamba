# 🐍 MarketMamba: 深度生成式量化交易預測引擎

![MarketMamba Banner](Parallel_Universes_Plot.png) *(可替換為你的宇宙觀測圖路徑)*

**MarketMamba** 是一個結合了 **狀態空間模型 (Mamba)**、**圖神經網路 (GAT)** 與 **擴散模型 (Diffusion / DDPM)** 的先進量化交易預測系統。

本專案旨在解決傳統金融時間序列預測中「雜訊過高」與「無法量化不確定性」的痛點，透過多重平行宇宙的預測軌跡，尋找市場中的高勝率投資組合。

### 📊 MarketMamba V4.0 全維度多模態資料庫 (The Omniscient Mamba Matrix)

### 📊 MarketMamba V4.0 全知全能資料矩陣 (The Ultimate Mamba Matrix)

本專案將發揮 Mamba 模型處理高維度矩陣的絕對優勢，將台股微觀籌碼、高頻特徵與國際實體宏觀完美融合。
**戰略 (Hit and Run)：** 歷史 5 年資料庫依賴一個月 Sponsor VIP 權限進行「全市場巨量打包」；實盤階段退回免費方案，採每日增量更新，達成 0 元維運。

| 數據板塊 (Domain) | 核心特徵欄位 (High-Dimensional Features) | 資料表來源 (FinMind API / Others) | Mamba 大腦之戰略作用 (Alpha Value) |
| :--- | :--- | :--- | :--- |
| **1. 國際宏觀與熱錢** | `US_Money_Supply` (美印鈔量), `G8_Rate` (央行利率)<br>`Gold/Oil` (金油價格), `USD_TWD` (台幣匯率)<br>`US_SOX`, `US_QQQ`, `US_VIX` | `yfinance`<br>FinMind (總體經濟/國際市場) | 建立全球熱錢與通膨的「上帝視角」，從根本預判外資匯出入動能。 |
| **2. 台股量價與防漏** | `Open`, `High`, `Low`, `Close`, `Volume`<br>`US_Market_Closed`, `TW_Typhoon_Day` | `yfinance`<br>本地端 `pd.merge_asof` 計算 | 捕捉價格動能，並嚴格標記休市日以根絕未來數據洩漏 (Lookahead Bias)。 |
| **3. 除權息失真防護** | `Ex_Dividend_Drop` (除息蒸發點數標記) | FinMind (除權除息結果表) | 告知模型當日價格落差為「配息」而非「暴跌」，防止 Mamba 誤判停損。 |
| **4. 法人與國家隊** | 三大法人買賣超, `Gov_Bank_Buy` (八大行庫) | FinMind (三大法人 / 八大行庫) | 追蹤市場大資金流向，以及國家隊暴跌時的護盤底線。 |
| **5. 散戶信用與情緒** | `Margin/Short` (融資券), `Day_Trading` (當沖佔比) | FinMind (融資融券 / 當沖統計) | 衡量散戶瘋狂程度，避開當沖過熱標的，捕捉融資斷頭潮。 |
| **6. 大戶與暗黑做空** | 大戶/散戶持股比例, `Securities_Lending` (借券) | FinMind (股權分散 / 借券明細) | 抓出千張大戶默默吃貨的飆股，提早發覺外資暗黑做空力道。 |
| **7. 主力分點照妖鏡** | `Broker_Branch_Vol` (各分點券商買賣明細) | FinMind (台股分點資料) | 台灣特有：精準抓出「地緣券商吃貨 (公司派)」與「隔日沖大戶倒貨」。 |
| **8. 盤中微觀流動性** | `Bid_Ask_Spread` (日均價差), `Order_Imbalance` (委託不平衡) | FinMind (每 5 秒委託成交統計) | 將高頻特徵降維，讓模型具備「流動性嗅覺」，避開買了賣不掉的殭屍股。 |
| **9. 基本面與現金流** | 月營收 YoY/MoM, 現金流量表, 綜合損益表 | FinMind (月營收 / 財務報表) | 破解假帳地雷，篩選具備實質獲利與強大現金流的護城河公司。 |
| **10. 價值與防禦底線** | `PE` (本益比), `PB` (淨值比), `DY` (殖利率) | FinMind (個股 PER) | 建立價值投資防禦底線，避免模型在多頭末期追高泡沫股。 |
| **11. 期權極致微觀** | `Futures_Broker_OI` (期貨各特定券商未平倉) | FinMind (期貨各券商每日交易) | 放大鏡檢視特定外資在期交所的空單佈局，作為系統崩盤的終極預警。 |
> **⚠️ 終極打包策略：**
> 以上所有 FinMind 來源，皆為政府/證交所免費公開資訊。本專案僅利用 Sponsor 權限的「單日全市場打包」特權，將 5 年歷史資料在 Colab 中一次性高速壓縮為 `.parquet` 本地特徵庫。

> **⚠️ 開發者備註 (Rate Limit 避坑指南)：**
> * `yfinance` 抓取美股與台股量價，基本無嚴格頻率限制。
> * `FinMind` 免費版限制為 **600 次 / 小時**。實盤每日更新全台股 1700 檔時，務必加上 `time.sleep(6.5)` 避免 IP 被封鎖。

## 🏗️ MarketMamba V4.0 資料工程與時序對齊規範 (Data Engineering Protocol)

為了處理龐大的 7 年期 (2019-01-01 至今) 高維度混合頻率資料，本專案制定以下嚴格的資料庫建置規範，以確保資料的純淨度並徹底根絕未來函數 (Lookahead Bias)。

### 🗂️ 第一層：資料夾分層架構 (Google Drive 儲存規範)
資料庫分為「原始資料湖 (Raw Data Lake)」與「最終特徵倉儲 (Processed Feature Warehouse)」兩大層級：

* **`MarketMamba_DB/Raw/` (原始資料層 - 依更新頻率分類)**
    * `/Daily_Macro/`: 美股大盤、VIX、匯率等每日宏觀數據。
    * `/Daily_Market/`: 每日全市場籌碼打包 (三大法人、融資券、當沖、借券等)。
    * `/Weekly_Holdings/`: 每週五更新之集保大戶/散戶股權分散表。
    * `/Monthly_Revenue/`: 每月 10 日前公布之月營收資料。
    * `/Quarterly_Financials/`: 每季公布之財報 (損益表、資產負債表等)。
* **`MarketMamba_DB/Processed_Features/` (最終特徵層)**
    * 存放以單一股票代號命名的 `.parquet` 檔案 (例如 `2330_features.parquet`)。
    * 此層資料已完成所有混頻對齊與缺失值填補，可直接輸入 Mamba 模型訓練。

### ⏱️ 第二層：混頻時序對齊守則 (Mixed-Frequency Alignment Rules)
所有對齊皆以「台股每日交易日曆 (Trading Calendar)」為絕對左表 (Left Anchor)。

1.  **日頻資料 (Daily)**
    * **對齊方式：** 相同日期直接 `merge`。
    * **跨時區 (美股)：** 美股收盤時間晚於台股，必須將美股日期 `+1 天` (轉為台股可見日)，再使用 `pd.merge_asof(direction='backward')` 向後尋找最近的值，並標記 `US_Market_Closed`。
2.  **週頻資料 (Weekly - 例：集保股權)**
    * **對齊方式：** 由於資料在每週五盤後公布，下週一至週四的觀測值必須等於「上週五」的數值。使用 `pd.merge_asof(direction='backward')` 進行時間戳記對齊。
3.  **月頻/季頻基本面 (Monthly/Quarterly - 例：營收、財報)**
    * **防漏準則 (Lookahead Bias Prevention)：** 絕對禁止使用「資料所屬年月」進行對齊。
    * **對齊方式：** 必須以 **「公告日 / 發布日 (Announcement Date)」** 作為真實時間戳記。在兩次公告日之間的空白交易日，採用 **向後填充 (Forward Fill, `ffill`)** 延續上一次的已知數據。

# 🐍 MarketMamba 演進史：從踩坑到穩健的量化架構

## 🔴 V1 版本：貪婪的賭徒 (The Greedy Gambler)

### 💡 核心設計理念 (The Vision)
MarketMamba V1 是本專案的初代原型，目標是打造一個能同時理解「時間序列特徵」、「股票空間連動性」以及「未來不確定性」的深度生成式量化引擎。
我們融合了三大前沿神經網路架構：
1. **時間維度 (Time-Series)**：使用 **Mamba (Selective State Space Model)** 取代傳統的 LSTM / Transformer，以線性複雜度捕捉股票過去的走勢記憶。
2. **空間維度 (Spatial Relational)**：使用 **GAT (Graph Attention Network)** 建立台股全市場的連動關聯網。
3. **不確定性 (Uncertainty)**：導入 **DDPM (Denoising Diffusion Probabilistic Models)**，不預測單一絕對價格，而是生成 30 個「平行宇宙」的機率分佈軌跡。

### 🛠️ 實作細節 (Implementation Details)
* **資料特徵 (Features)**：僅使用了 7 項最基礎的微觀技術指標，包含 OHLCV (開高低收量)、以及 RSI (14天) 與 MACD。
* **時間窗格 (Window)**：模型觀察過去 20 天 (`seq_len=20`) 的歷史，預測未來 5 天 (`pred_days=5`) 的預期報酬率：
  $y_{t+5} = \frac{Close_{t+5} - Close_t}{Close_t}$
* **大腦規格 (Architecture)**：
  * Mamba 引擎：`d_model=64`, `d_state=16`, `d_conv=4`。
  * GNN 引擎：4 個注意力頭 (`num_heads=4`)。
* **交易策略 (Strategy)**：採用最純粹的「絕對利潤導向」。系統會在 30 個平行宇宙中，算出每一檔股票的預期報酬均值 (Predicted Mean)，並直接買入預期利潤最高的 Top 10 股票。

### 💥 致命的失敗與教訓 (The Downfall & Lessons Learned)
儘管架構聽起來非常先進，但 V1 版本在實戰回測中遭遇了慘烈的失敗（模擬 5 天報酬率達 **-8.95%**），並在工程上暴露了嚴重的瓶頸：

1. **OOM 記憶體核彈 (The Dense Graph Disaster)**：
   在建構 GNN 關聯網時，V1 採取了天真的「全連接圖 (Dense Graph)」設計。1930 多檔台股彼此互連，產生了將近 370 萬條邊 (Edges)。這導致 DataLoader 在切片時瞬間撐爆 RAM，而在 GPU 訓練時也頻繁觸發 `CUDA Out of Memory`。
2. **盲目的貪婪 (Ignorance of Risk)**：
   這是 V1 在交易邏輯上最致命的缺陷。模型只看「預期獲利最高」，卻完全忽略了擴散模型提供的「波動率 (標準差 $\sigma$)」資訊。這導致系統瘋狂選入大起大落的「妖股」，一旦遭遇突發利空（黑天鵝），投資組合就會瞬間崩盤。
3. **宏觀視野狹隘 (Macro-blindness)**：
   僅看過去 20 天的個股技術面，完全沒有大盤 (Taiex) 與匯率 (USD/TWD) 的總經視角，如同井底之蛙，無法判斷資金水位的整體流向。

> **結論**：V1 證明了把頂級技術全部縫合在一起，並不等於一個好的交易策略。它促使我們必須在工程上進行 GNN 稀疏化，並在金融邏輯上導入風險控管，這也成為了後續 V2 誕生的最強驅動力。


## 🟢 V2 版本：甦醒的巨獸 (The Awakened Behemoth)

### 💡 核心設計理念 (The Vision)
如果說 V1 是一個在市場中橫衝直撞的賭徒，那麼 V2 就是一個學會了「風險控管」與「資源分配」的成熟操盤手。
我們在 V2 中進行了史詩級的架構重構（MLOps），徹底解決了記憶體溢出的問題，並將擴散模型（Diffusion）生成的「不確定性（Uncertainty）」真正應用於實戰選股中，實現了從絕對利潤導向到風險調整後報酬（Risk-Adjusted Return）的典範轉移。

### 🛠️ 史詩級工程升級 (Engineering & MLOps Upgrades)
為了馴服這隻龐大的量化巨獸，我們在資料工程上做出了三大突破：
1. **Lazy Loading 動態切片技術**：
   拋棄了 V1 將所有歷史窗格預先切好並塞入 RAM 的天真作法。V2 改為儲存一個極度輕量化（約 500MB）的「基礎時空矩陣」，並透過客製化的 `PyTorch Dataset` 在每個 Batch 訓練時進行動態實時切片。成功將 64GB 的記憶體核彈化解於無形。
2. **GNN 稀疏化 (Graph Sparsification)**：
   導入 Pearson Correlation 矩陣，強制限制每檔股票只能與全市場關聯度最高的 10 檔股票 (`TOP_K=10`) 建立邊緣連線。將無效連線斬斷後，不僅防堵了 OOM，更讓 A100 的訓練速度從 12 分鐘/Epoch 暴降至 **3 分鐘/Epoch**。
3. **智慧早停系統 (Early Stopping)**：
   實作了具備耐心值 (`patience=5`) 的 Early Stopping 機制，讓擴散模型在捕捉到完美常態分佈雜訊（Loss 逼近 1.0）時自動儲存最佳權重並安全下莊，避免過擬合浪費算力。

### 🧠 金融邏輯進化 (Quant Strategy Evolution)
1. **120 天宏觀大局觀**：
   特徵維度擴充至 10 維 (`input_dim=10`)，除了更進階的技術指標（如 MA20_Bias, 10日波動率），更注入了「大盤報酬率 (TAIEX)」與「美元/台幣匯率 (USD/TWD)」。同時，將 Mamba 的時間感受野從 20 天大幅擴張至 **120 天（半年線級別）**。
2. **夏普分數選股法 (Sharpe Ratio Selection)**：
   這是 V2 逆轉勝的絕對關鍵！我們不再盲目追求預期報酬最高，而是利用擴散模型展開 30 個平行宇宙後，計算出預期報酬均值 ($\mu$) 與標準差風險 ($\sigma$)。系統僅買入 $\mu > 0$ 且 **「夏普分數 ($\mu / \sigma$)」最高** 的前 10 大防禦型飆股。

### 🏆 實戰逆轉勝 (The Turnaround)
透過大局觀的注入與嚴格的風險控管，V2 在相同的測試環境下交出了令人驚豔的成績單：
* 資訊係數 (Rank IC) 從 V1 的 0.0275 狂飆至 **+0.0928**。
* 5 天模擬報酬率從 V1 的慘賠 -8.95% 逆轉為 **實質獲利 +1.87%**。
* 成功驗證了「用擴散模型捕捉黑天鵝風險，用夏普分數建構防禦陣地」的量化哲學。

## 🔵 V3.1 終極對齊版：平行軌跡與凱利資金盤 (The Ultimate Aligned Version)

### 💡 核心設計理念 (The Vision)
如果 V2 是一個懂風險的操盤手，那麼 V3.1 就是一個具備「上帝視角」的量化基金大腦。
我們不再滿足於「預測未來的單一時間點（如第 5 天）」，而是大膽要求模型畫出「未來 30 天的連續軌跡」。同時，為了解決深度生成模型（Generative AI）在處理金融微小數值時的「尺度幻覺」，我們導入了對齊機制，並結合華爾街經典的「凱利公式（Kelly Criterion）」，完成了從預測到下單的最後一哩路。

### 🛠️ 史詩級演算法與架構升級 (Algorithmic & Architectural Upgrades)
1. **Mamba 大腦擴容與 30 天多步軌跡生成 (Multi-Step Trajectory Diffusion)**：
   將 Mamba 的隱藏層維度擴張至 `d_model=128`。擴散模型 (DDPM) 的預測目標從 2D 矩陣升級為 3D 張量 `(30個宇宙, 1937檔股票, 30天)`。這讓我們能觀測到強大的「Alpha 衰減表 (Alpha Decay)」，進而發現模型的黃金打擊區落在「未來第 15 天」(Rank IC 飆升至 +0.0208)。
2. **解決擴散縮放陷阱 (Scale Factor Alignment)**：
   **痛點：** 擴散模型使用標準差為 1.0 的 $\mathcal{N}(0, 1)$ 雜訊，但台股真實的日報酬率僅在 0.01~0.10 之間，微弱訊號被雜訊碾碎，導致模型輸出高達 100% 預期報酬的「荒謬幻覺」。
   **解法：** 在 DataLoader 階段導入 `SCALE_FACTOR = 10.0` 強行放大真實訊號以匹配雜訊，並在推論 (Inference) 階段除以 10 還原。成功讓預期報酬率與 MAE 誤差回歸真實世界的物理常理。
3. **半凱利公式資金盤 (Half-Kelly Portfolio Optimization)**：
   利用 30 個平行宇宙的預測結果，計算出精準的均值 ($\mu$) 與變異數 ($\sigma^2$)。套用修改版的凱利公式 $f = \mu / \sigma^2 \times 0.5$ 來決定每檔股票的資金佔比，並加上 `20%` 的硬性防爆上限。讓模型不再只是「報明牌」，而是直接給出「該押多少資金」的專業決策。
4. **前後端算力解耦 (Decoupled MLOps Architecture)**：
   為 Streamlit 網站部署做準備，實作了「批次離線推論 (Offline Batch Inference)」。每天收盤後由 A100 GPU 進行一次性推論，將 15 天資金配置表與 30 天預測軌跡打包成極輕量的 CSV 快取 (`V3_1_Kelly_Allocation_15D.csv`)。讓前端免除掛載龐大 PyTorch 模型的崩潰風險，達成 24 小時免費伺服器秒讀取的架構。

### 🏆 終極實戰成果 (The Final Output)
* **尋找最佳持有期：** 透過 30 天衰減表證明，針對台股日線資料，15 天是本模型的最佳持倉週期。
* **風險調整後的高效配置：** 成功篩選出夏普分數極高（如 0.76）的防禦型飆股，並透過防護鎖將單一標的風險嚴格控制在總資金的 20% 以內。
* **系統準備就緒：** 已具備直接對接前端互動式網頁 (Streamlit) 的資料庫後盾，隨時可上線成為 24 小時自動化量化顧問。

# 🚀 MarketMamba V4.0 開發藍圖 (The Pure Quant Edition)

## 1. 多模態資料鍛造與時序防漏 (Data & Engineering)
* **籌碼與宏觀融合：** 引入 `FinMind` (三大法人買賣超) 與 `yfinance` (美股 SOX/QQQ、VIX、美債)。
* **5 年期 Parquet 本地資料庫：** 建立最近 5 年的特徵庫，捨棄過舊的市場雜訊，並支援每日增量更新。
* **絕對時間對齊 (As-Of Join)：** 以台股開盤日為絕對錨點，向後精準對齊跨國時區的資料。
* **節日狀態標記 (Holiday Flags)：** 新增 `US_Market_Closed` 等二元狀態標記，徹底根絕未來數據洩漏 (Lookahead Bias)。

## 2. 大腦擴容與動態圖學習 (Model Architecture)
* **深度 Mamba 擴容：** 拓寬與加深網路層數 (如 `d_model=256`, 3層堆疊)，榨乾 A100 算力以消化高維度籌碼切片。
* **動態注意力圖 (Dynamic GAT)：** 捨棄靜態 Pearson 矩陣，讓模型每日自動計算股票間的 Attention Score，捕捉極短線資金板塊輪動。

## 3. 獲利導向的雙重優化 (Training Objectives)
* **方向感知損失函數 (Directional-Aware Loss)：** 在 MSE 基礎上加入不對稱懲罰項，針對「看錯方向」給予極高權重懲罰，強迫提升實質勝率。
* **可微夏普值優化 (Differentiable Sharpe Ratio)：** 將「投資組合整體風險報酬比」寫入 Loss 函數，直接優化資金分配效益。

## 4. 終極實盤代理人 (Execution Agent)
* **強化學習交易員 (RLHF for Trading)：** 訓練輕量級 PPO Agent，學習在擴散模型給出的機率雲中執行最佳買進、賣出與空手策略。



## 📊 Dataset Health & Feature Dictionary (V4.1 Pro Max)
> 本資料集已通過嚴格的防未來函數與極端值測試，並包含 56 維高階動能、籌碼與 Alpha 特徵，具備實盤深度學習之頂尖標準。

### 📐 Dataset Dimension (資料集規模)
- **Total Records (總筆數):** `3,711,525` rows
- **Total Features (總維度):** `56` columns
- **Time Range (時間範圍):** `2019-05-08` to `2026-02-26`
- **Unique Tickers (涵蓋標的):** `2,782` tickers (已啟動 60 日 IPO 隔離牆)

### 🧬 Feature Categories (特徵字典 56 維度)
**🎯 Target Label (預測目標):**
`Future_5d_Return`

**🌍 Macro & Global (總經與國際大盤):**
`TWII_Close, USD_TWD, US_SOX, US_QQQ, US_VIX, US_TNX, Gold, Oil, US_Market_Closed`

**📈 Price & Volume (基礎價量):**
`Open, High, Low, Close, Volume`

**📐 Technical & Alpha (技術動能與超額報酬):**
`Return_1d, MA_5, MA_20, Bias_5, Vol_MA_5, Vol_Ratio, MA_60, Bias_60, BB_Upper, BB_Lower, BB_Width, MACD, MACD_Signal, MACD_Hist, RSI_14, Alpha_1d`

**🕵️‍♂️ Institutional Chips (三大法人與微觀籌碼):**
`Foreign_Buy, Trust_Buy, Dealer_Buy, Margin_Balance, Short_Balance, Day_Trading_Ratio, Securities_Lending, Foreign_Sum_5d, Trust_Sum_5d, Foreign_Sum_10d, Foreign_Sum_20d, Trust_Sum_10d, Trust_Sum_20d`

**🏢 Fundamentals & Valuation (基本面、估值與成長性):**
`PER, PBR, DY, Monthly_Revenue, Whale_Hold_Ratio, Retail_Hold_Ratio, EPS, Gross_Margin, Rev_MoM, Rev_YoY`

### 🏥 Data Quality Assurance (安檢報告)
- **Missing Values (NaN):** `0` ✅ (Target: 0)
- **Infinite Values (Inf):** `0` ✅ (Target: 0)
- **Lookahead Bias (未來函數):** `Strictly Mitigated` ✅ (防未來函數對齊與未來價格存活驗證)
- **Mathematical Anomalies (數學異常):** `Resolved` ✅ (RSI 死魚股化簡防護、無限回溯阻斷)

### 📊 Target Distribution (`Future_5d_Return`)
- **Mean (平均報酬):** `0.20%`
- **Std Dev (標準差):** `7.63%`

## 🔴 V4.0 實戰危機與搶救：巨獸微創手術與恐懼封印 (The Surgery & Brave Mode)

### 💡 實戰部署的殘酷現實
在 V4.0 藍圖設計完成並投入 A100 進行全市場 1900 檔股票的實盤訓練時，我們遭遇了深度學習量化領域最經典的兩大災難。這迫使我們在架構上進行了緊急的「微創手術」，從而誕生了目前實盤運作的 **純血 Mamba 完全體 (V4_GodMode_Production)**。

### 💥 危機一：全連接注意力的坍塌 (Attention Collapse)
* **災難現象：** 模型推論出的未來 30 天軌跡，全市場 1900 檔股票的預測值竟然一模一樣（停滯在 `-6.65%`）。
* **大腦斷層掃描 (診斷)：** 透過抽檢模型各層的輸出變異數，我們發現 Mamba 層依然健康 (變異數 `0.2135`)，但 Attention 層完全死水 (變異數 `0.0000`)。
* **根本原因：** 在沒有導入 GNN 產業遮罩的情況下，強行讓 1900 檔股票進行 `MultiheadAttention`。模型為了逃避 Loss 懲罰，選擇了最偷懶的最佳化路徑：「將所有股票特徵平均化成一灘泥巴」。
* **微創手術 (The Surgery)：** 我們利用殘差連接 (Residual Connection) 保留梯度的特性，直接 **「切除」** 壞死的 Attention 層，凍結練了 24 小時的 Mamba 黃金權重，僅用 1 分鐘的時間重新訓練全新的 Output Head，成功找回個股的獨立特徵。

### 💥 危機二：安全常數陷阱 (The Safe Constant Trap)
* **災難現象：** 切除 Attention 後，預測軌跡變成了一條完美的水平直線，每天的預測值死鎖在微小的 `0.0075`。
* **根本原因：** 源自於藍圖中的 `方向感知損失函數 (Directional-Aware Loss)`。高達 2.5 倍的做錯方向懲罰，在充滿雜訊的真實股市中引發了模型的「恐懼」。模型為了不被扣分，放棄了預測波動，選擇盲猜一個極小的正數 (大盤長期漂移值) 當作安全牌。
* **解除封印 (Brave Mode)：** 我們在最終微調階段拔除了方向懲罰，回歸最純粹的 `MSE Loss` (均方誤差)，並使用「差異化學習率」讓模型重新勇敢學習市場的高低起伏。

### 🏆 V4.0 最終型態：純血 Mamba 時序大腦
經過兩次搶救，目前的 V4.0 暫時擱置了不成熟的全局 Attention，轉而將 Mamba 對於「120 天長時間序列」的特徵萃取能力推展到極致。這為我們接下來將 `torch-geometric` 真正融入 V5.0 的圖神經網路計畫，打下了最堅實的地基。
  
# 🚀 MarketMamba V4.1 Upgrade Plan: The Adjusted Price Protocol (除權息還原股價升級計畫)
## ⚠️ 發現的問題 (The Ex-Dividend Trap)
在 V4.0 架構中，模型依賴原始收盤價 (`Close`) 計算技術指標與未來報酬率 (`Future_5d_Return`)。然而，台股具有「高殖利率」特性。當遇到大型股票（如長榮、聯發科等）進行高額現金股息除息時，原始股價會在一夜之間出現巨大的「假性跳水」。

**對模型的影響：**
1. **目標變數污染 (Label Corruption)：** 投資人實際領到現金並未虧損，但模型會將該次除息判斷為 `Future_5d_Return = -20%` 的暴跌，導致模型學習到錯誤的特徵映射。
2. **技術指標失真 (Indicator Distortion)：** 均線 (`MA_20`)、布林通道與乖離率會因為假性跳水而嚴重扭曲，發出錯誤的超賣訊號。

## 🎯 V4.1 解決方案
全面將 V4.0 的基石資料從「原始股價」替換為「還原權息股價 (Adjusted Price)」。

### 🛠️ 實作步驟 (Implementation Steps)

#### 1. API 替換 (Data Extraction Phase)
修改資料採集模組 `Cell 4.5`：
- 將 FinMind 呼叫的 dataset 參數由 `TaiwanStockPrice` 更改為 `TaiwanStockAdjustedPrice`。
- 確保抓取的欄位對齊：`AdjustedOpen`, `AdjustedHigh`, `AdjustedLow`, `AdjustedClose`。

#### 2. 資料庫重建 (Base Table Rebuild)
- 在 Google Drive 建立全新的 `Raw/Daily_Adjusted_Price` 資料夾。
- 執行重新打包，覆蓋 2019 至今的全市場還原日 K 線。

#### 3. 特徵大融合與重算 (Pipeline Modification)
- 修改 `Cell 5` 的讀取路徑，將 Base Table 指向 `Daily_Adjusted_Price`。
- 確保所有依賴價格的衍生特徵 (`Return_1d`, `MA_5`, `Future_5d_Return`) 皆使用 `AdjustedClose` 進行計算。
- **預期結果：** 消滅除權息導致的報酬率異常，提升模型在除權息旺季 (每年 6~8 月) 的預測穩定度與夏普值 (Sharpe Ratio)。

## 📅 預計開發時程
本計畫將於 V4.0 模型初版訓練完成、並建立 Baseline 基準測試後，作為提升 Alpha 的優先升級項目啟動。
---
*Built with passion and tons of coffee. ☕*
