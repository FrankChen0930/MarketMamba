# 🐍 MarketMamba V6 升級規劃與實作藍圖

> 狀態: **規劃中 (Planning)**
> 最後更新: 2026-04-20
> 版本目標: **MarketMamba V6.0**
> 撰寫方式: 由 2026-04-20 的 AI 規劃會議整理，包含現況分析、升級方向與決策紀錄

---

## 📊 V5.5 現況完整健檢

> 此段為 2026-04-20 完整代碼審查後的總結，供後續 Agent 快速掌握現況。

### 已完成模組盤點

| 模組 | 檔案 | 品質評估 |
|------|------|---------|
| 資料擷取 | `fetcher.py` (394行) | 優：帶重試、增量補齊、五大資料源 |
| 跨頻融合 | `merger.py` | 良：季報/月營收時間對齊有保護 |
| 特徵工程 | `feature_engineer.py` (165行) | 良：基礎技術指標、Rev_YoY 防未來函數 |
| 模型架構 | `architecture.py` (195行) | 優：Mamba×6層+GATv2，Gating Fusion |
| 推論引擎 | `inference.py` (282行) | 優：自動偵測版本、checkpoint 反向工程 |
| 訓練模組 | `trainer.py` (431行) | 優：FastAlphaDataset、AMP FP16、滑窗步幅 |
| 知識圖譜 | `graph_builder.py` (99行) | ⚠️ 初版：只有同產業=0.5靜態規則 |
| 情緒引擎 | `sentiment/` (6個模組) | 良：雙語FinBERT、Auto-labeler |
| 型態偵測 | `detectors.py` (284行) | 優：6大型態、動態頸線、等幅測距 |
| 投組管理 | `portfolio_manager.py` (239行) | ⚠️ 待強化：缺流動性濾網、無滑價折抵 |
| 前端展示 | `app.py` (449行) | 良：5頁面 Streamlit，Plotly互動圖 |

### 現行模型規格

```
輸入維度：84D（46 量價籌碼 + 38 情緒特徵）
時序框架：Mamba SSM × 6層，seq_len=180，d_model=256，d_state=32
圖推理：  GATv2Conv (4 head)，KG-Enhanced Edge（cosine + 產業分類）
預測目標：未來30天 Alpha 軌跡（相對 TWII 超額累積報酬）
訓練策略：MSELoss + AdamW + CosineAnnealingLR + EarlyStop(7) + AMP FP16
```

### 從代碼直接識別的現存問題

| 嚴重度 | 問題 | 說明 |
|--------|------|------|
| 🔴 邏輯錯誤 | **Cross-Section 污染** | `FastAlphaDataset` 滑窗取樣讓 batch 內可能混入不同時間步的股票，GAT 跨時間步計算 cosine similarity 在邏輯上是錯誤的 |
| 🔴 實盤脫節 | **流動性濾網不存在** | `compute_kelly_sharpe()` 完全沒有日均量或滑價，df_kelly 裡甚至沒有 Avg_Volume 欄位 |
| 🟡 Alpha 缺口 | **KG 過於粗糙** | `_build_knowledge_similarity()` 只有二元規則，且是 O(N²) Python loop |
| 🟡 特徵缺口 | **RSI、KD、ATR 完全沒有** | 台股最重要的三個即時動能指標在 46 維特徵中缺席 |

---

## 🎯 V6 核心目標

根據實務經驗與 V5.5 的優劣勢分析，V6 版本的核心專注於：
1. **市場結構深層聯動性 (Deep Graph)** — 引入真實供應鏈關係取代純產業分類
2. **Cross-Section 正確性** — 修正 DataLoader 讓圖推理邏輯自洽
3. **實盤交易摩擦力擬真 (Execution Reality)** — 流動性與滑價防護

模型骨幹維持 `Mamba SSM + GATv2`，汲取 V4 崩潰教訓，不使用任何不連續懲罰 Loss。

---

## 🛠️ 模組 1：知識圖譜 (KG) 多維度關係擴充

目前 V5.5 的圖論邊界主要建立在「同產業給予 `0.5`」的靜態權重。V6 走向「多重建構」並以聚合權重取代單一條件。

### 資料來源：TPEX 產業鏈平台（已確認可爬）

**網站**：https://ic.tpex.org.tw/

**爬蟲分析結果（2026-04-20 確認）**：
- 資料全部嵌在 HTML，**不需要 AJAX/登入/Token**，可直接用 `requests + BeautifulSoup` 爬取
- URL 格式：`https://ic.tpex.org.tw/introduce.php?ic={ic_code}`
- 公司資訊直接從連結抓取：`<a href="/company_basic.php?stk_code=2330">台積電</a>`
- 涵蓋 **30+ 個產業鏈**（半導體、生技、通信、被動元件、印刷電路板等）

**已確認的產業代號（部分）**：

| ic_code | 產業 | ic_code | 產業 |
|---------|------|---------|------|
| D000 | 半導體 | I000 | 通信網路 |
| C100 | 製藥 | K000 | 連接器 |
| C200 | 醫療器材 | F000 | 電腦週邊 |
| 5300 | 人工智慧 | J000 | 被動元件 |
| 5400 | 雲端運算 | L000 | 印刷電路板 |
| 4100 | 太空衛星科技 | G000 | 平面顯示器 |
| 6000 | 自動化 | N000 | 石化及塑橡膠 |

**爬蟲腳本規劃位置**：`marketmamba/knowledge/tpex_crawler.py`

```python
# 爬蟲核心邏輯示意
def crawl_tpex_supply_chain() -> pd.DataFrame:
    """
    爬取所有產業鏈頁面，產出：
    - ic_code       : 產業代號
    - industry_name : 產業中文名
    - stk_code      : 股票代號
    - company_name  : 公司名稱
    """
    records = []
    for ic_code, industry_name in IC_CODES.items():
        url = f"https://ic.tpex.org.tw/introduce.php?ic={ic_code}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = soup.find_all('a', href=lambda h: h and 'company_basic.php' in h)
        for link in links:
            stk_code = link['href'].split('stk_code=')[-1]
            records.append({'ic_code': ic_code, 'industry_name': industry_name,
                            'stk_code': stk_code, 'company_name': link.text.strip()})
        time.sleep(1.5)  # 禮貌性等待
    return pd.DataFrame(records).drop_duplicates(subset=['ic_code', 'stk_code'])
```

**注意**：TPEX 頁面只提供「哪個產業有哪些公司」，上游/中游/下游的分區是 CSS 視覺佈局，需要 DOM 解析。半導體（D000）頁面有最完整的分區結構，可優先處理。

### 具體圖邊建構方向

改寫 `marketmamba/knowledge/graph_builder.py`：

**1. 供應鏈從屬邊（TPEX 資料）**
- 同一 TPEX 產業鏈的公司，建立從屬邊，權重 **0.6**
- 一間公司可能出現在多個產業鏈（如台達電在半導體+電機機械），自然形成跨產業連結

**2. 統計相關性鄰接矩陣（自動計算）**
- 用歷史半年/一年收益率建立 Pearson 相關矩陣
- 相關係數 > 0.7 者連線（能抓出集團股/主力同步操作）

**3. 集團從屬關係（手動建表）**
- 鴻海家族、台積電相關、聯發科生態等台股集團，賦予高關聯權重 **0.8**

**4. 原有產業分類**
- 維持現有 TWSE 產業分類 = 0.5，作為基底

**最終目標**：以上多種因素疊加為一個 **靜態緩存矩陣 (Cached Pre-computed Adjacency Matrix)**，讓推論與訓練時 O(1) 讀取，完全不影響運算時間。

存放位置：`config.PROCESSED_DIR/knowledge_graph_cache.npz`

---

## 🛠️ 模組 2：Cross-Section DataLoader 修正（已確認採用正確做法）

> **決策（2026-04-20）**：採用最正確的修法，不妥協。

### 問題根源

`FastAlphaDataset.__getitem__` 每次取一個隨機窗口，batch 內不同股票可能屬於不同時間步。GAT 在這種情況下計算的 cosine similarity 是**跨時間污染的**，圖訊息傳遞在邏輯上無效。

### 正確修法：TemporalCrossSectionDataset

每個 `__getitem__` 回傳的是**同一個交易日的全市場截面**：

```python
class TemporalCrossSectionDataset(Dataset):
    """
    每個 sample = 某交易日 t 的 (N_stocks, seq_len, input_dim)
    保證 GAT 的 cross-sectional graph computing 邏輯正確
    """
    def __init__(self, dates: list, date_to_tensors: dict, seq_len, pred_days):
        self.dates = dates
        self.date_to_tensors = date_to_tensors

    def __getitem__(self, idx):
        date = self.dates[idx]
        X, Y = self.date_to_tensors[date]  # (N, seq_len, D), (N, pred_days)
        return X, Y
```

**注意事項**：
- DataLoader 的 `batch_size` 概念改變：每個 sample 已包含全市場 ~2000 支股票，所以 DataLoader `batch_size=1`，相當於每次訓練一個交易日截面
- GPU 顯存需求比滑窗方式高，需要在 A100 上評估實際佔用量
- 若 2000 × seq_len × d_model 超過顯存，可考慮 **每個截面隨機抽樣 N_sample 支股票** 進行訓練（但要保留完整截面作為推論基礎）

---

## 🛠️ 模組 3：Loss 函數求穩策略（V4 防禦機制）

> **背景**：V4 曾嘗試機率分佈 + 方向懲罰，導致模型「全部輸出 0 裝死 (Target Collapse)」。

### 原則
- **完全廢棄** Heaviside step 懲罰，以防梯度爆炸或神經元壞死
- 用**獎勵代替懲罰**

### 建議升級（保守策略）

```
Hybrid Loss = α × MSELoss + β × ListNet Ranking Loss
```

- **MSELoss**：保持 V5.5 的穩定基底
- **ListNet Ranking Loss**：不要求精確預測 Alpha 數值，只要求選股排名正確
  - 這是現代量化因子訓練的主流選擇
  - 比 Pairwise Loss 更穩健
  - 不使用 Heaviside/階躍函數，完全連續可微

**中心思想**：神經網路只管找出相對優秀的 Alpha (Signal)，對於下行風險跟震盪，交由後端資金控管 (Portfolio Manager) 處置。

---

## 🛠️ 模組 4：實盤交易流動性防護網

解決模型挑選出「流動性枯竭但極具理論 Alpha」的小型垃圾股問題。

### 具體實作

於 `inference.py::compute_kelly_sharpe()` 加入：

**1. 絕對流動性濾網 (Hard Filter)**
```python
# 近 5 日平均成交金額低於 1000 萬台幣 → 直接剔除
zombie_vol_mask = df['Avg_Volume_5D'] < 1e7
sharpe[zombie_vol_mask] = -999.0
```

**2. 滑價與摩擦力折抵 (Slippage Penalty)**
```
Net Alpha = Model Expected Return - (Tax + Fee + Liquidity Slippage)
```
- 滑價依市值反比計算：大型股 ~0.2%，小型股 ~0.8%
- 只有 Net Alpha 仍具備爆發力的股票才進入資金分配

**注意**：`Avg_Volume_5D` 欄位目前在 df_kelly 中不存在，需要在特徵工程或推論時補齊傳入。

---

## 🆕 模組 5：特徵工程深化（新增）

目前 46 維量價籌碼特徵缺少台股最重要的技術指標。

### 建議新增至 `feature_engineer.py`

| 指標 | 說明 | 為什麼重要 |
|------|------|----------|
| RSI (14日) | 相對強弱指數 | 台股教科書指標，散戶在 RSI<30 有強烈買入慣性 |
| KD 隨機指標 (9/3/3) | K值、D值 | 台股量化因子中的高頻 Alpha 生產來源 |
| ATR (14日) | 真實波動幅度 | 比 Volatility_20d 更精確的波動衡量，可取代 Kelly 裡的 raw_vol |
| OBV | 累積量價關係 | 對中長線資金動向有更好描述 |
| EPS 驚喜度 | 與上季 EPS 比較的成長率 | 重要的基本面衝量因子 |

**影響**：input_dim 從 84 → 約 90（彈性）。需要重訓，但不需要改架構。

---

## 🆕 模組 6：自動化回測引擎（新增）

> **背景**：目前只有「百萬實盤機器人」的即時帳本，但**沒有在歷史資料上驗證模型選股品質的機制**。

### 必要性

- 無法知道模型在不同市場環境（牛市/熊市/盤整）的勝率
- 無法量化「情緒特徵」到底帶來了多少 Alpha
- 無法比較 V5.5 vs V6 模型的真實效益

### 規劃模組位置

`marketmamba/backtest/` 新增目錄

```python
class BacktestEngine:
    def run(df_kelly_history, df_price_history) -> BacktestResult
    # 輸出：累計報酬、最大回撤、夏普比率、勝率、年化Alpha
    # 比較：Top10等權重策略 vs 凱利加權策略 vs TWII指數
```

### GitHub 儲存策略

**問題**：每天覆蓋 df_kelly.csv，歷史資料會消失。

**解法**：Rolling Window 保留機制
- 每天推送時，把結果同時存到 `results/YYYY-MM-DD/df_kelly.csv`
- 保留最近 90 天
- 自動刪除超過 90 天的資料夾

**容量估算**：
- 每日約 1 MB 資料 × 90 天 = ~90 MB，遠低於 GitHub 1 GB 建議上限
- 即使保留一年（260天），也只有 ~260 MB，仍在安全範圍

---

## 🆕 模組 7：MC-Dropout 不確定性估計（新增）

### 實作方式

```python
# 推論時保持 Dropout 開啟，執行 N 次前向推播
def run_inference_with_uncertainty(model, test_x, n_samples=30):
    model.train()  # 保持 Dropout 活化
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(test_x).cpu().numpy())
    
    pred_mean = np.mean(preds, axis=0)      # 最終預測
    pred_std  = np.std(preds, axis=0)       # 模型置信度
    return pred_mean, pred_std
```

### 效益

- 在前端標示「高信心 / 低信心」選股
- 低置信度的標的提醒用戶謹慎對待
- 不需要修改模型架構，只需修改 `inference.py`

---

## 🆕 模組 8：部署策略 — 本機推論 + 雲端訓練分離 (2026-04-23 更新)

> **核心決策**：**每日推論 → 本機 RTX 3060 + WSL2**，**重新訓練 → Colab A100**

### 推論環境：本機 RTX 3060 Laptop (✅ 可行，推薦)

**為什麼本機比 Colab 更適合做推論：**
- Colab 每次啟動都要重新安裝 Mamba kernel (~5分鐘)，本機 WSL2 安裝一次永久有效
- 推論只需要跑 forward pass，RTX 3060 (6GB VRAM) **完全夠用**
- 不消耗 Colab GPU quota，省真金白銀
- 可以設 Windows 排程任務（Task Scheduler）每天 17:30 自動觸發

**速度估算（2000 支股票 × seq_len=180 × d_model=256）：**

| 環境 | 推論時間（估算） | 每日成本 |
|------|--------------|--------|
| Colab T4 | ~3-8 分鐘（含啟動） | GPU quota |
| Colab A100 | ~1-3 分鐘（含啟動） | GPU quota |
| **本機 RTX 3060** | **~5-12 分鐘（含資料載入）** | **電費 < $0.01** |

> 推論不需要高端 GPU，3060 完全勝任。速度差距不顯著，但成本差距巨大。

**WSL2 安裝 Mamba 環境（一次性設定）：**
```bash
# 1. 在 WSL2 (Ubuntu) 中安裝 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# 2. 安裝 MarketMamba 環境
conda create -n mamba_env python=3.12
conda activate mamba_env
pip install -r requirements.txt
pip install causal_conv1d-*.whl mamba_ssm-*.whl  # 用已有的 wheel

# 3. 一次編譯，永久使用，不需每天重裝
```

**Windows Task Scheduler 自動化：**
```batch
:: daily_inference.bat
@echo off
wsl -d Ubuntu -e bash -c "cd /mnt/d/Desktop/work/MarketMamba && conda run -n mamba_env python run_daily_inference.py"
```
> 設定為每天 17:30 觸發，等資料更新後自動執行，無需手動操作。

### 訓練環境：Colab A100 (維持現況)

重新訓練（V6 更新特徵、重訓模型）時才需要 Colab：
- **頻率**：每次大版本升級時，不需要每天
- **資料搬移**：訓練時把 Parquet 資料壓縮上傳 Google Drive 一次，訓練完把 checkpoint 下載回本機
- **具體做法**：
  ```bash
  # 本機：打包資料上傳
  cd Data && tar czf data_snapshot.tar.gz processed/
  rclone copy data_snapshot.tar.gz gdrive:MarketMamba/
  
  # Colab：掛 Drive 解壓後直接訓練
  # 訓練完：checkpoint 下載回本機 models/ 目錄
  ```

### 已確認不可行的方案

| 方案 | 結論 | 原因 |
|------|------|------|
| GitHub Actions | ❌ | 只有 CPU，無法跑 Mamba CUDA kernel |
| Kaggle 排程 | ❌ | CUDA kernel 無法跨 session 保留（已嘗試確認）|
| Render/Railway CPU | ❌ | Mamba 必須有 CUDA，純 CPU 環境不行 |

---

## 🆕 模組 9：LLM 整合層 — 取代 FinBERT，外包消息面分析 (2026-04-23 新增)

> **架構決策**：MarketMamba 回歸**純量化角色**，消息面分析交由 LLM（Claude/GPT）負責。

### 為什麼要移除 FinBERT？

| 面向 | FinBERT 整合進模型 | LLM 外包分析 |
|------|-----------------|------------|
| 新聞理解品質 | 有限（只有 POS/NEG/NEU） | 遠優於 FinBERT（可理解上下文） |
| 模型複雜度 | input_dim 84→46（移除後更乾淨） | 不影響模型 |
| 更換升級 | 需要重訓模型 | 換 API 即可，不需重訓 |
| 訓練穩定性 | 情緒特徵雜訊多，影響 MSELoss | 完全隔離 |

**結論**：V6 的 input_dim 從 84D 降回 **46D（純量價籌碼）**，情緒不再進模型，改由 LLM 在推論後的「報告層」處理。

### 新架構：推論後整合

```
每日推論 Pipeline:

[步驟 1] MarketMamba 推論（本機 RTX 3060）
  → 輸出：df_kelly.csv（Alpha、Kelly、Sharpe、技術指標）
  → 輸出：df_traj.csv（30天Alpha軌跡）

[步驟 2] LLM 報告生成（本機 CPU，API call，~10秒）
  → 輸入：① df_kelly Top 10 結果
           ② 今日市場數據（TWII漲跌、VIX、美股）
           ③ 恐懼貪婪指數（API）
  → 輸出：market_summary.json（市場背景文字 + 情緒傾向）

[步驟 3] 合併推送 GitHub
  → df_kelly.csv + df_traj.csv + market_summary.json
```

### 實作位置

新增模組：`marketmamba/llm/report_generator.py`

```python
import anthropic

def generate_market_report(df_kelly_top10: dict, market_data: dict) -> dict:
    """
    在 MarketMamba 推論完後呼叫，生成當日市場背景報告。
    不需要 Agent，只需要一次 LLM API 呼叫。
    """
    client = anthropic.Anthropic()  # 讀 ANTHROPIC_API_KEY 環境變數

    prompt = f"""
    今天是 {market_data['date']}，台股大盤收 {market_data['twii_change']:+.2f}%。
    以下是今日量化模型的 Top 選股結果：
    {df_kelly_top10}

    請生成一份200字以內的繁體中文市場背景摘要，包含：
    1. 今日市場整體氛圍（一句話）
    2. 需要特別注意的宏觀風險（川普/Fed/地緣等，如有）
    3. 配合量化信號的操作建議方向

    格式：條列式，emoji 標記重要程度。
    """

    response = client.messages.create(
        model="claude-haiku-4-5",  # 成本最低，每次 < $0.001
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "date": market_data['date'],
        "summary": response.content[0].text,
    }
```

### 費用估算

| LLM 選擇 | 每次呼叫成本 | 每月（30天） |
|---------|------------|------------|
| Claude Haiku | ~$0.001 | **~$0.03/月** |
| Claude Sonnet | ~$0.005 | ~$0.15/月 |
| GPT-4o mini | ~$0.002 | ~$0.06/月 |

> 費用極低，遠比維護 FinBERT 訓練集划算。

---

## 🆕 模組 10：資料時效優化 — 解決 FinMind 延遲瓶頸 (2026-04-23 新增)

> **核心問題**：FinMind 資料全部到位需等到 18:30+，導致推論被迫延後到晚上。

### FinMind 各資料集實際更新時間（已知）

| 資料集 | FinMind 更新時間 | 問題 |
|--------|----------------|------|
| 個股價量 (TaiwanStockPrice) | ~17:00-18:00 | 最慢，必須等 |
| 三大法人 (TaiwanStockInstitutionalInvestors) | ~17:30-18:30 | 比價量更晚 |
| 融資融券 (TaiwanStockMarginPurchaseShortSale) | ~18:00-19:00 | 最晚 |
| 月營收 (TaiwanStockMonthRevenue) | 每月10-15日 | 不影響每日 |

> 現況：最快也要等到 **18:30 以後**才能確定所有資料都到位，才能安全執行推論。

### 解決方案：Hybrid 分層資料源策略

**核心思路**：把需要每天更新的資料 按更新速度 分層，快速來源先走，慢速來源後補。

```
資料分層策略：

[層 1] 立即可用（收盤後 15 分鐘內）
  來源：Yahoo Finance (yfinance)
  資料：個股 OHLCV 價量 → 取代 FinMind TaiwanStockPrice
  限制：只有價量，沒有籌碼

[層 2] 官方直連（TWSE/TPEX 網站）約 16:30-17:00
  來源：證交所公開資料平台 (data.twse.com.tw)
  資料：三大法人買賣超（比 FinMind 快 30-60 分鐘）
  限制：需要自己寫爬蟲，格式稍微複雜

[層 3] FinMind（保留，作為備援）約 17:30-19:00
  資料：融資融券、借券、八大行庫
  策略：若 FinMind 尚未更新，使用前一天的值 Forward Fill
```

### 具體改寫 fetcher.py 策略

**個股價量：優先 yfinance（快），FinMind 降為備援**

```python
def fetch_daily_prices_fast(trading_days: list[str]) -> None:
    """
    V6 新版：優先用 yfinance 批次抓全市場個股收盤價
    速度：收盤後 15 分鐘內即可取得
    """
    # 批次抓取，比逐股 FinMind 快 10 倍以上
    all_tickers = [f"{stk}.TW" for stk in tw_universe] + \
                  [f"{stk}.TWO" for stk in otc_universe]
    df_batch = yf.download(
        all_tickers, start=target_date, end=target_date,
        auto_adjust=True, progress=False, group_by='ticker'
    )
    # 如果 yfinance 某股缺值，才 fallback 到 FinMind
    missing = [t for t in all_tickers if df_batch[t].empty]
    if missing:
        _fetch_missing_from_finmind(missing, target_date)
```

**三大法人：直連 TWSE 公開資料（比 FinMind 快 30-60 分鐘）**

```python
def fetch_institutional_from_twse(date_str: str) -> pd.DataFrame:
    """
    直接爬 TWSE 三大法人公開資料
    URL: https://www.twse.com.tw/rwd/zh/fund/T86?date={YYYYMMDD}&response=json
    更新時間：約 16:30-17:00（比 FinMind 快）
    """
    url = f"https://www.twse.com.tw/rwd/zh/fund/T86"
    params = {"date": date_str.replace('-', ''), "response": "json"}
    res = requests.get(url, params=params,
                       headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    data = res.json()
    if data.get('stat') != 'OK':
        return None  # fallback 到 FinMind
    # 解析 data['data'] 原始表格...
```

**融資融券：接受 Forward Fill（昨日值 ≈ 今日值，誤差極小）**

```python
# 融資融券每天變動極小，用昨日值 Forward Fill 引入的誤差
# 遠小於「等到 19:00 才推論」帶來的時效延遲代價
def fetch_margin_with_fallback(date_str: str) -> pd.DataFrame:
    df = _fetch_finmind(session, "TaiwanStockMarginPurchaseShortSale", date_str)
    if df is None:
        # FinMind 尚未更新，用昨日快取 Forward Fill
        df = load_yesterday_margin_cache()
        logger.warning(f"融資融券使用 Forward Fill: {date_str}")
    return df
```

### 優化後的時間軸

```
舊流程（FinMind 全部）：
  13:30 收盤 → 等待 FinMind → 18:30+ 才能推論 → 19:00 完成

新流程（Hybrid）：
  13:30 收盤
  13:45 yfinance 抓價量完成（速度快）
  16:30 TWSE 三大法人直連（官方最快）
  17:00 ✅ 開始推論（比舊流程早 1.5 小時）
  17:15 推論完成，LLM 報告生成
  17:30 推送 GitHub，儀表板更新

  融資融券（FinMind 較慢）→ 使用前一天值 Forward Fill，接受此近似
```

> **判斷**：融資融券的「昨日值」和「今日值」在大多數日子差異極小（隔天不會突然大變），Forward Fill 引入的誤差遠小於「等到 19:00 才推論」帶來的時效延遲。

### 實作位置

改寫 `marketmamba/data/fetcher.py`：
- `fetch_daily_prices()` → 新增 yfinance 優先路徑
- 新增 `fetch_institutional_from_twse()` 直連 TWSE
- 舊的 FinMind 路徑保留為 fallback

---

## 📅 V6 升級優先排序（最終版，2026-04-23 更新）

| 序號 | 任務 | 需要重訓 | 預估工作量 | 理由 |
|------|------|---------|---------|------|
| 🥇 1 | **Cross-Section DataLoader 修正 (模組2)** | 是（重訓） | 2天 | 邏輯錯誤，圖推理結果不可信 |
| 🥈 2 | **流動性濾網 + 滑價折抵 (模組4)** | 否 | 1天 | 立竿見影，Quick Win |
| 🥉 3 | **Hybrid 資料源 + 時效優化 (模組10)** | 否 | 2天 | 推論提前 1.5hr，日常體驗大升級 |
| 4 | **本機 WSL2 推論環境建置 (模組8)** | 否 | 半天 | 省 Colab quota，自動化排程 |
| 5 | **LLM 報告生成層 (模組9)** | 否（移除 FinBERT） | 1天 | 消息面品質大升，input_dim 降回 46D |
| 6 | **TPEX 爬蟲 + KG 擴充 (模組1)** | 是（重訓） | 3-5天 | 模型真正的 Alpha 升級來源 |
| 7 | **RSI/KD/ATR 特徵 (模組5)** | 是（重訓） | 1天 | 台股最重要的技術指標缺口 |
| 8 | **回測引擎 MVP (模組6)** | 否 | 3天 | 沒有它無法量化V6的提升幅度 |
| 9 | **Ranking Loss 融合 (模組3)** | 是（重訓） | 1天 | 比MSELoss更針對選股排名的訓練信號 |
| 10 | **MC-Dropout 不確定性 (模組7)** | 否 | 1天 | 前端體驗升級，提高用戶信任 |

---

## 📅 Roadmap（2026-04-23 更新版）

- [ ] **Phase 0: 邏輯修復（立刻，不能跳過）**
  - 實作 **模組 2**：TemporalCrossSectionDataset 取代 FastAlphaDataset

- [ ] **Phase 1: 基礎設施升級（Quick Win，不需重訓）**
  - 實作 **模組 10**：Hybrid 資料源，yfinance 取代 FinMind 價量，直連 TWSE 三大法人
  - 實作 **模組 8**：WSL2 推論環境建置 + Windows Task Scheduler 自動化
  - 實作 **模組 9**：LLM 報告生成層（移除 FinBERT，新增 report_generator.py）
  - 實作 **模組 4**：加入流動性濾網與滑價折抵
  - 實作 **模組 6**：回測引擎 + Rolling History 存檔機制
  - *(完成後：推論從 19:00 提前至 17:30，本機自動化，無需每天開 Colab)*

- [ ] **Phase 2: 特徵與圖結構升級（Heavy Lift，需重訓）**
  - 執行 **模組 1**：TPEX 爬蟲 → 供應鏈 Parquet → 改寫 graph_builder.py
  - 執行 **模組 5**：新增 RSI/KD/ATR/OBV，input_dim = 46D（移除情緒特徵後重新計算）
  - 執行 **模組 3**：ListNet Ranking Loss 融合
  - 透過 `V5_5_Training.py` 重新訓練 V6 權重（Colab A100）
  - 訓練完：checkpoint 下載至本機，後續推論全在本機執行

- [ ] **Phase 3: 體驗與生態完善**
  - 實作 **模組 7**：MC-Dropout 置信度標示
  - 前端新增「每日報告」頁面（六維度風格）
  - 前端新增「策略回測報告」頁面

---

## 🔑 重要決策與技術備忘

### 已做的決策

1. **Cross-Section 修法**：採用「每 batch = 同一交易日全市場截面」的正確做法，不妥協
2. **Loss 策略**：維持 MSELoss 為基礎，可選擇性加入 ListNet，絕不用 Heaviside 懲罰
3. **KG 資料來源**：使用 TPEX 產業鏈平台（已確認可爬，HTML 靜態，不需 API key）
4. **Kaggle 排程**：已確認不可行（Mamba CUDA kernel 無法跨 session 保留）
5. **GitHub 回測歷史**：採用 90 天滾動保留，不需要 Git LFS
6. **情緒分析架構**：FinBERT 從模型輸入移除，改由 LLM API（Claude/GPT）在推論後生成報告。input_dim 從 84D 回到 46D（2026-04-23）
7. **每日推論部署**：本機 RTX 3060 + WSL2，不再依賴 Colab 做推論；訓練仍用 Colab A100（2026-04-23）
8. **資料源策略**：個股價量改用 yfinance（快），三大法人改直連 TWSE；FinMind 降為 fallback（2026-04-23）

### 技術陷阱備忘

| 陷阱 | 說明 | 應對方式 |
|------|------|---------|
| Mamba 環境依賴 | Mamba SSM 必須在 Linux + CUDA 環境即時編譯，wheel 不跨環境 | **WSL2 + CUDA** 一次安裝，本機永久有效 |
| V4 Target Collapse | 用 Heaviside 方向懲罰導致模型輸出全 0 | 永遠只用平滑連續的 Loss，懲罰換成獎勵 |
| Cross-Section 污染 | 滑窗 batch 讓 GAT 跨時間步計算 | 用 TemporalCrossSectionDataset 強制同日截面 |
| KG Python O(N²) loop | `_build_knowledge_similarity` 是 Python 雙迴圈 | 改用 torch 向量化運算，或預算好存 npz |
| GitHub 容量 | 單檔 100MB 限制，repo 建議 < 1GB | CSV 用滾動保留，模型權重禁止 commit |
| yfinance 缺股問題 | 部分興櫃/ETF 在 yfinance 可能抓不到 | 抓不到的股票自動 fallback 到 FinMind；建立股票黑名單 |
| TWSE API 格式變更 | 政府網站 API 可能無預警改版 | 爬蟲加版本偵測，失敗時 fallback 到 FinMind |
