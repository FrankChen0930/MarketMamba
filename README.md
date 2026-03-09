# 🐍 MarketMamba: 深度生成式量化交易預測引擎

![MarketMamba Banner](Parallel_Universes_Plot.png) *(可替換為你的宇宙觀測圖路徑)*

**MarketMamba** 是一個結合了 **狀態空間模型 (Mamba)**、**圖神經網路 (GAT)** 與 **擴散模型 (Diffusion / DDPM)** 的先進量化交易預測系統。

本專案旨在解決傳統金融時間序列預測中「雜訊過高」與「無法量化不確定性」的痛點，透過多重平行宇宙的預測軌跡，尋找市場中的高勝率投資組合。

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

---
*Built with passion and tons of coffee. ☕*
