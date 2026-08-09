# MarketMamba — 深度學習驅動的台股量化投資系統

> **Mamba SSM + GATv2 知識圖譜 · 每日全市場 Alpha 推論 · 系統性多模型驗證框架**
>
> 個人獨立開發，自 2026 年 3 月起持續開發、每日收盤後自動執行至今。

**Live Demo** → [marketmamba.vercel.app](https://marketmamba.vercel.app)

---

## 這是什麼

MarketMamba 是一套台股量化投資自動化系統。每個交易日收盤後，系統會自動抓取全市場約 2,900 檔股票的價格、法人籌碼、財報與總體經濟資料，建構特徵矩陣，透過 **Mamba SSM + GATv2** 模型推論出每支股票的 Alpha 分數，再交由組合建構層產出可執行的持股名單——全程無需人工介入，結果每天自動 push 到雲端 dashboard。

這不是一次性的課堂作業或比賽 repo。系統從 2026 年初開始每天實際運行，期間經歷過完整的資料層重建、8 種模型配置的系統性比較、以及多次推翻自己先前結論的除錯過程。這份 README 想呈現的除了模型本身，更是我如何把一個「看起來能用」的系統，一步步逼問到「真的站得住腳」。

## 動機：為什麼是 Mamba

這個架構選擇不是憑感覺，而是從大學專題研究開始的。我和同組夥伴在專題〈Mamba 與 Transformer 在文字分類上之比較〉中，針對兩個文字分類任務（IMDb 情緒分析、新聞分類），在模型參數量相差不到 15% 的前提下，系統性比較了 Mamba 與 Transformer 的表現與資源消耗。結果：

- Mamba 的 accuracy 略遜 Transformer，差距最大 3.4%、最小僅 0.1%
- 但 GPU 記憶體用量最多可**省下 73.3%**
- 訓練時間最多可**縮短 63.5%**
- 推理時間最多可**減少 56.6%**
- 機制上，Self-Attention 的計算複雜度是 O(N²·d)，Mamba 的 Selective SSM 是 O(N·d)——序列越長，這個差距會被放大

這個結論直接影響了 MarketMamba 的架構選擇：每日推論要吃進**每支股票 252 個交易日（約一整年）的價格序列**，正是專題中驗證過「Mamba 優勢會被放大」的長序列場景；而且整套系統跑在個人的 RTX 3060 上，不是雲端叢集，GPU 記憶體與推理速度的餘裕直接決定了系統能不能在本機跑得動、跑得快。專題給的是「Mamba 在這種條件下值得一試」的證據，MarketMamba 則是把這個判斷放到真實市場資料上驗證。

## 目標

不是做一個回測數字漂亮的 demo，而是做一個**每天真的在跑、結果會被每天的市場走勢檢驗**的系統。這意味著：

- 訊號要能自動產出，不能依賴人工每天盯盤判斷
- 回測結論要能重現、要能被自己的下一次實驗推翻
- 資料管線的正確性跟模型架構同等重要——錯的資料上疊再好的模型也沒有意義
- 「有效」不是回測數字漂亮就能宣稱，要看 live tracking 撐不撐得住

## 現在在哪個階段

系統目前拆成兩條並行的線，dashboard 首頁直接呈現這個分野：

| | 🌐 廣度模型（作品展示主軸） | 🎯 高信念模型（個人實盤主軸） |
|---|---|---|
| 標的範圍 | 全市場 ~2,888 檔 | 精選 10–20 檔 |
| 核心假設 | 統計顯著性來自覆蓋面（√Breadth） | 準確性來自對少數標的的深度研究 |
| 核心引擎 | Mamba SSM + GATv2（DL 是決策主軸） | 量化篩選 + LLM 研究增幅 + 人工判斷（DL/LLM 是工具層） |
| 資金定位 | Paper／驗證用，不代表個人實際部位 | 個人實盤操作主要依據 |
| 可回測性 | 可嚴謹 Walk-Forward（12 年結構化數據） | 部分可回測，主體只能前瞻追蹤 |

**廣度模型是這份作品集想呈現的重點**：11 份模型分數（8 種 Mamba 配置 + Ridge / GBDT / GRU）× 19 個投資組合設定（不同再平衡頻率）同時並行，每天自動記錄實際報酬，用來誠實回答「哪個模型設定真的比較好」。2026 年 8 月中開始正式累積 live track record——樣本數還太少，現在還不到能對外宣稱「有效」的時候，這點我選擇老實講清楚，而不是只放一張漂亮的回測圖。

## 重點亮點

比起模型本身，我更想呈現的是把一個量化系統做到「經得起檢驗」所需要的工程紀律：

**1. 系統性多模型比較框架，不用單一指標下結論**
11 個模型在同一套訓練／驗證隔離（purge）、同一份資料面板下比較。用 decile spread Sharpe（run-to-run 標準差僅 0.019）取代單純的 Top50 年化報酬（run-to-run 標準差達 2.68 個百分點）作為模型優劣的主要量尺，因為前者對隨機性穩定約 40 倍。比較表附上完整的方法論紀律——哪些差距落在雜訊底線內、哪些模型因訓練條件不同而不可跨輪比較。

**2. 抓出並修復資料管線的靜默失效**
在清理備份檔案前先做「正式檔案的鍵集合不能比備份少」的交叉驗證，結果抓到寫入邏輯有「整天覆寫」的設計缺陷：只要某天抓到的股票數比已存的少，差額會被靜默刪除、不留任何錯誤訊息。這個 bug 讓核心價格資料表悄悄少了 15% 的資料列、影響近 500 支可交易股票。修好並回補資料後重新驗證全部研究結論，證實沒有任何一項被推翻。

**3. 找到「線上執行」與「回測」之間 0.18 個百分點的分歧，並定位到根因**
新增前瞻績效追蹤工具後，第一天就發現實際執行結果與回測結果對不上。逐日比對定位到單一交易日的單一持股不同——兩者的分數完全相等（float32 並列），但線上排序用的是不穩定排序（quicksort），回測用的是穩定排序，並列時挑出不同的股票。統一排序邏輯後兩者完全一致（誤差 0.000pp）。

**4. 誠實記錄「沒有效果」的實驗，不做事後選擇性引用**
多次消融實驗（特徵維度砍半、移除總經因子等）得出「無顯著差異」或「這個方向沒用」的結論時，一樣完整記錄判準與數據，不因結果不好看就悄悄略過或換個角度重新詮釋。

**5. 完整的每日自動化管線 + 可觀測性**
從資料擷取、特徵工程、多模型推論、組合建構、LLM 市場報告到自動部署，全程排程執行、進度可視化、異常告警（Telegram），失敗時保持視窗開啟以利除錯，而不是靜默失敗。

---

## 模型架構 — MarketMambaV6

~4M 參數，Google Colab A100 訓練，本機 RTX 3060（WSL2）推論。

```
輸入：(N_stocks, SEQ_LEN=252, INPUT_DIM=56~59)
  ↓
FactorGroupedEmbedding
  Group A 價格動能      (12 dims) → sub_dim 54
  Group B 法人籌碼      (20 dims) → sub_dim 94
  Group C 基本面        (12 dims) → sub_dim 54
  Group D 總體環境      (12 dims) → sub_dim 54
  各組投影後 Concat → d_model=256
  ↓
MultiScaleMambaEncoder（3 分支並行，自適應融合）
  Short 分支：最近  20 步 × 2 層 Mamba
  Mid   分支：最近  60 步 × 3 層 Mamba
  Long  分支：完整 252 步 × 3 層 Mamba  ← 套用 zero-padding mask
  融合：scale_gate = Softmax(Linear(d_model×3 → 3))
  ↓
GATv2（知識圖譜引導的橫斷面交互）
  ~640K 條邊，CSR 稀疏矩陣
  ↓
Gating Fusion
  gate = Sigmoid(Linear(d_model×2 → d_model))
  ↓
MultiHorizonHead
  3 個獨立 Linear → pred_5d / pred_20d / pred_60d

輸出：[Alpha_5d, Alpha_20d, Alpha_60d]
```

### 關鍵設計

**Multi-Scale Mamba**：三個時間尺度捕捉不同市場動態——短期（20 步≈1 個月）抓動能、中期（60 步≈1 季）抓趨勢、長期（252 步≈1 年）抓結構性型態，用自適應的 `scale_gate` 動態融合三個分支。

**GATv2 知識圖譜**：邊的來源包含 TWSE 產業分類、集團關係、供應鏈關聯（TPEX 資料）、每日滾動 Pearson 相關性。每次推論重建 CSR 稀疏矩陣，達成 O(1) 的批次子圖抽取。

**不確定性估計**：MC-Dropout（N=30 次採樣）估算每股不確定性，用於信心分層與資金分配的輸入。

**特徵維度**：主線推論用 56 維（不含相對強度特徵），部分並行模型線使用 59 維（多納入 RS_5d/RS_20d/RS_60d 相對強度特徵，總體環境維度以 mask 方式歸零——實驗證實總體環境因子在橫斷面排序上是每日常數、反而會誘發過擬合）。

---

## 每日自動化管線

```
收盤後  Windows Task Scheduler → WSL2 Ubuntu
  [1]  資料擷取     FinMind API + yfinance + 交易所直連 HTTP（14 個資料源）
  [2]  特徵工程     建構因子矩陣 + 資料新鮮度檢查
  [3]  多模型推論   8 種 Mamba 配置 + Ridge / GBDT / GRU → 11 份 Alpha 分數
  [4]  組合建構     分數 × 再平衡頻率 → 19 個投資組合的狀態機
  [5]  前瞻績效彙總 逐日記錄實際報酬，與回測數字交叉驗證
  [6]  LLM 市場報告 Claude API → 每日市場摘要
  [7]  Push         git push → GitHub → 雲端快取自動刷新
```

進度透過本機視窗即時顯示；成功自動關閉，失敗保持開啟並置頂以利除錯；異常另外用 Telegram 告警。

---

## Web Dashboard

**前端**（Vite + React）→ Vercel：[marketmamba.vercel.app](https://marketmamba.vercel.app)
**後端**（FastAPI）→ Render

資料流：GitHub 推論結果 → 後端啟動時拉取並快取（1 小時 TTL）→ REST API → 前端呈現

| 分頁 | 說明 |
|------|------|
| 廣度模型 / 模型預測結果 | 全市場 Alpha 排名、多模型並列 |
| 廣度模型 / 持股組合 | 19 個組合設定的目前持股與再平衡狀態 |
| 廣度模型 / 回測結果 | Walk-Forward 驗證、decile spread 分析 |
| 高信念模型 / 每日訊號 | 進出場評分與判斷依據 |
| 高信念模型 / 持倉追蹤 | 對接券商 API 的真實持倉 |
| 量化分析 | 技術面、籌碼面、市場廣度、型態辨識 |
| AI 市場報告 | Claude 生成的每日市場分析 |

---

## Tech Stack

| 層 | 技術 |
|---|------|
| 序列模型 | Mamba SSM（`mamba-ssm`） |
| 圖神經網路 | GATv2Conv（PyTorch Geometric） |
| 對照模型 | Ridge / GBDT（LightGBM）/ GRU |
| 資料來源 | FinMind API + yfinance + 交易所直連 HTTP |
| LLM 報告 | Anthropic Claude API |
| 前端 | Vite + React |
| 後端 | FastAPI + Uvicorn |
| 訓練環境 | Google Colab A100 |
| 推論環境 | RTX 3060（本機 WSL2 Ubuntu） |
| 部署 | Vercel（前端）/ Render（後端） |

---

## Repository 結構

```
MarketMamba/
├── V6/
│   ├── marketmamba/
│   │   ├── config.py                  ← 全域超參數與路徑
│   │   ├── data/
│   │   │   ├── fetcher.py             ← 資料爬蟲（含指數退避重試）
│   │   │   └── feature_engineer.py   ← 因子特徵工程
│   │   ├── models/                    ← ⚠️ 訓練好的權重，不可修改
│   │   │   ├── architecture.py        ← MarketMambaV6 模型定義
│   │   │   └── trainer.py             ← 訓練迴圈
│   │   ├── signals/                   ← 進出場條件模組
│   │   ├── knowledge/
│   │   │   └── graph_builder.py      ← 知識圖譜建構
│   │   ├── llm/
│   │   │   └── report_generator.py   ← Claude API 每日報告
│   │   └── backtest/                  ← 回測 / 組合建構引擎
│   ├── run_daily_inference.py         ← 每日推論主入口
│   ├── experimental/                  ← 消融實驗、資料除錯腳本
│   ├── results/                       ← 每日輸出（git push 到 GitHub）
│   └── models/                        ← ⚠️ 訓練好的 checkpoint
│
├── app/
│   ├── backend/                       ← FastAPI（Render）
│   └── frontend/                      ← Vite + React（Vercel）
│
└── archive/                           ← 舊版本（僅供參考）
```

---

## 本地執行

```bash
# Clone
git clone https://github.com/FrankChen0930/MarketMamba.git
cd MarketMamba

# 每日推論（WSL2）
wsl -d Ubuntu -- bash -lc "
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_env &&
  cd /mnt/d/Desktop/work/ProjectForMe/MarketMamba &&
  python V6/run_daily_inference.py"

# 前端開發伺服器
cd app/frontend && npm install && npm run dev   # → localhost:5173

# 後端開發伺服器
cd app/backend && pip install -r requirements.txt && uvicorn main:app --reload
```

**環境變數**（`V6/.env`）：
```
FINMIND_TOKEN=...
ANTHROPIC_API_KEY=...
```

---

## License

MIT License © 2024–2026 FrankChen
