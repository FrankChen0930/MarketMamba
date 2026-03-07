# 🐍 MarketMamba: 深度生成式量化交易預測引擎

![MarketMamba Banner](Parallel_Universes_Plot.png) *(可替換為你的宇宙觀測圖路徑)*

**MarketMamba** 是一個結合了 **狀態空間模型 (Mamba)**、**圖神經網路 (GAT)** 與 **擴散模型 (Diffusion / DDPM)** 的先進量化交易預測系統。

本專案旨在解決傳統金融時間序列預測中「雜訊過高」與「無法量化不確定性」的痛點，透過多重平行宇宙的預測軌跡，尋找市場中的高勝率投資組合。

## 🚀 架構演進史 (Architecture Evolution)

### 🔴 V1 版本：貪婪的賭徒 (The Greedy Gambler)
* **架構限制**：僅使用 5 項基本特徵 (開高低收量)，觀察窗格僅 20 天。
* **致命缺陷**：
  * **OOM 記憶體爆炸**：使用傳統的密集圖 (Dense Graph) 連結全市場 1937 檔股票，導致 A100 GPU 訓練極度緩慢且容易崩潰。
  * **高風險選股**：僅以「預期最高報酬 (MSE)」作為選股依據，導致模型買入大量波動極大的妖股。
* **實戰回測**：Rank IC 僅 0.0275，5 天模擬報酬率 **-8.95%** (遭遇黑天鵝事件崩盤)。

### 🟢 V2 版本：甦醒的巨獸 (The Awakened Behemoth)
為了解決 V1 的瓶頸，V2 進行了史詩級的架構與 MLOps 重構：
1. **120 天大局觀與宏觀特徵**：引入大盤報酬率與美元/台幣匯率，並將 Mamba 的時間感受野擴展至 120 天 (半年線級別)。
2. **Lazy Loading 動態切片技術**：棄用會撐爆 64GB RAM 的靜態滑動視窗，改寫自訂 `PyTorch Dataset` 進行動態切片，成功將記憶體負載降至 500MB 以下。
3. **GNN 稀疏化 (Sparsity)**：導入 Pearson Correlation 並限制 `TOP_K=10` 連線，將 370 萬條無效連線斬斷至 1.9 萬條，訓練速度從 12 分鐘/Epoch 暴降至 **3 分鐘/Epoch**。
4. **夏普選股法 (Risk-Adjusted Selection)**：利用擴散模型 30 個平行宇宙的輸出，計算 $\mu$ (預期報酬) 與 $\sigma$ (標準差風險)，優先挑選 $\mu/\sigma$ 最高的高性價比股票。
* **實戰回測**：Rank IC 飆升至 **0.0928**，5 天模擬報酬率逆轉為 **+1.87%**。

## 🧠 核心技術棧 (Tech Stack)
* **Time-Series Engine**: Mamba-SSM (解決 Transformer 在長序列的 $O(N^2)$ 複雜度瓶頸)
* **Spatial Relational Engine**: PyTorch Geometric (GATConv)
* **Generative Uncertainty**: Denoising Diffusion Probabilistic Models (DDPM)
* **MLOps**: Early Stopping, Checkpointing, Lazy Loading (CPU/GPU 算力解耦)

## 🔮 未來展望 (V3 Roadmap)
- [ ] **多步軌跡預測 (Multi-step Trajectory)**：從單一目標價預測，升級為生成未來 20 個交易日的完整機率雲漏斗圖。
- [ ] **凱利公式資金白盒化 (Kelly Criterion)**：取代等權重買法，透過模型輸出的風險與報酬，給出精確的資金分配比例建議。
- [ ] **微觀高頻特徵降維打擊**：導入 1 分鐘線/5 分鐘線的盤中波動率與尾盤爆量指標，構建雙塔架構 (Two-Tower Architecture)。

---
*Developed as a research project for quantitative trading architecture optimization.*
