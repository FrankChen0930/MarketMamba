# MarketMamba — 系統全覽

> 台股 AI 選股系統：Mamba SSM + KG-Enhanced GATv2 · 每日自動化推論 · 全棧 Web Dashboard  
> 最後更新：2026-06-13

---

## 一句話定位

MarketMamba 是個人量化投資自動化系統，每日收盤後（17:00）對台股全市場（~2,515 支）進行深度學習推論，輸出 Alpha 訊號排名，並透過 Web Dashboard 呈現選股結果、持倉追蹤與 LLM 市場報告。

---

## 部署資訊

| 服務 | 網址 | 觸發方式 |
|------|------|---------|
| 後端（Render） | `https://marketmamba-api.onrender.com` | push to `main` 自動部署 |
| 前端（Vercel） | `https://marketmamba.vercel.app` | push to `main` 自動部署 |

後端啟動時從 GitHub raw URL 抓取 `V6/results/` 下的 CSV/JSON 快取至記憶體（1 小時 TTL）。

---

## 模型架構：MarketMambaV6

### 輸入資料（56 維特徵 × 252 天序列）

| 群組 | 維度 | 內容 |
|------|------|------|
| **Group A — 價格動能** | 12 | 開高低收、成交量、5/10/20/60 日均線、1d/5d 報酬率、RSI(14)、ATR |
| **Group B — 法人籌碼** | 20 | 外資/投信/自營商淨買超（絕對值+佔比）、融資餘額、融券餘額、KD(9,3)、OBV、成交量比等 |
| **Group C — 基本面** | 12 | 月營收 YoY、EPS、本益比、股價淨值比、ROE、市值、殖利率、自由現金流等 |
| **Group D — 宏觀環境** | 12 | TAIEX / SPX / 黃金 報酬率、VIX、美元指數、台幣匯率、PMI、聯邦基準利率等 |

> V6.2 訓練版（`INPUT_DIM=59`）額外啟用 RS_5d/RS_20d/RS_60d 三個相對強度特徵。
>
> **標準化（D1 修復，2026-06-12）**：股票級特徵採 per-date cross-sectional z-score；Group D 宏觀特徵自 V6.2 起改 expanding time-series z-score（shift(1) 無 look-ahead、min 252 天、clip ±3σ）。舊版 cross-sectional 會把同日同值的宏觀特徵歸零，V6.1 的 macro 分支實際上沒有資訊輸入。

訓練資料時間範圍：**2005–今**，約 2,515 支台股。

### 模型結構

```
輸入：(N_stocks, SEQ_LEN=252, INPUT_DIM=56)
  ↓ FactorGroupedEmbedding（各群組獨立投影 → 比例分配子維度 → Concat → d_model=256）
  ↓ MultiScaleMambaEncoder（三分支並行）
      Short branch: 最後 20 步 × 2 層 Mamba
      Mid   branch: 最後 60 步 × 3 層 Mamba
      Long  branch: 全 252 步   × 3 層 Mamba（套用 zero-padding mask）
      自適應融合：scale_gate = Softmax(Linear(d_model×3 → 3))
  ↓ GATv2（知識圖譜引導截面互動）
      ~640K 條邊，CSR 稀疏矩陣加速
  ↓ Gating Fusion（gate = Sigmoid(Linear(d_model×2 → d_model))）
  ↓ MultiHorizonHead（3 個獨立 Linear → pred_5d / pred_20d / pred_60d）
輸出：[Exp_Alpha_5d, Exp_Alpha_20d, Exp_Alpha_60d]
```

**參數量**：~4M（訓練中持續更新）  
**訓練環境**：Google Colab A100  
**推論環境**：本機 RTX 3060（WSL2）  
**Checkpoint**：`V6/models/v6_best.pt`

### 知識圖譜（KG）邊類型

| 邊類型 | 權重 | 說明 |
|--------|------|------|
| TWSE 產業分類 | 0.5 | 同行業公司關聯 |
| 集團從屬（鴻海、台積電等） | 0.8 | 集團內部關聯 |
| TPEX 供應鏈 | 0.6 | 上下游廠商 |
| 60 日滾動相關係數 | 動態 | Pearson 相關 > 閾值 |

### 不確定性估算

使用 **MC-Dropout**（N=30 次採樣）估算每股不確定度（`Uncertainty`）。2026-06-12 起以推論日期為亂數種子（`torch.manual_seed(YYYYMMDD)`），同日重跑結果可重現。

### 推論方式（P0 修復，2026-06-12）

兩段式推論：Mamba encoder 以 128 股分批跑（VRAM 瓶頸在 252 步序列），GAT 一次吃**完整 cross-section 圖**。舊版每批只取批內 KG 邊、跨批邊全部丟失，GAT 在推論時幾乎退化成 identity；修復後驗證 Uncertainty 整體 -34%、Top50 換血 20 支、新進者呈產業群聚（GAT 沿 KG 邊傳播訊號的預期行為）。

### 關鍵衍生指標

- **Signal_Quality**（舊名 Sharpe_Score）= `Net_Alpha_20d / (Uncertainty + 1e-6)`，截斷至 [-10, +10]
- **Confidence**：當日 Uncertainty 的 Q30/Q70 分位數分級（高/中/低信心；2026-06-12 起，原固定 bins 已棄用）
- **Alpha 截斷**：±2.0，防止離群值
- **流動性過濾**：`alpha_20d = -999` 表示流動性不足，前端自動排除

---

## 每日推論流程

```
每日 17:00（Windows Task Scheduler 觸發）
  └─ daily_inference.bat
       └─ WSL2 → run_daily_inference.py

  [Step 1 / 17:00] 資料更新
      yfinance 下載最新股價（前復權）
      TWSE 直連下載法人買賣超、融資融券
      FinMind API 補充基本面、財報資料
      資料新鮮度檢查（不足 5 日則警告）

  [Step 2 / 17:05] 特徵工程
      build_features() → 56 維特徵矩陣（V6.2 部署後 59 維）
      update_correlation_edges() → 更新 KG 相關性邊
      clean_and_scale() → 標準化（含剔除統計輸出）

  [Step 3 / 17:10] 模型推論（RTX 3060，CUDA）
      兩段式推論：Mamba 分批 → GAT 完整 cross-section 圖
      MarketMambaV6 → df_kelly.csv（全市場 Alpha 排名）
      MC-Dropout × 30（日期 seed，可重現）→ Uncertainty 欄位
      df_traj.csv（多期預測軌跡）

  [Step 4 / 17:15] LLM 報告
      build_market_data() → 抓取大盤即時行情
      generate_market_report() → Claude API
      → market_summary.json

  [Step 5 / 17:18] 歸檔
      每日結果複製到 V6/results/{YYYY-MM-DD}/（保留 90 天）
      history_index.json 追加今日 Top 50（保留 60 個交易日）

  [Step 6 / 17:20] 訊號掃描
      scanner.py 讀取 df_kelly.csv + history_index.json
      → action_signals.json（BUY / EXIT / WATCH_LIST）

  [Step 7 / 17:25] 推送
      git add V6/results/ && git commit && git push
      POST /api/signals/cache/refresh → Render 重新拉取快取

tkinter 視窗即時顯示進度（WSLg）
成功 → 3 秒後自動關閉；失敗 → 視窗保持置頂
```

**推論輸出路徑**（GitHub `V6/results/`）：

| 檔案 | 內容 |
|------|------|
| `df_kelly.csv` | 全市場 Alpha 排名（~2,500 支） |
| `df_traj.csv` | 多期預測軌跡 |
| `action_signals.json` | 入場 / 退場 / 觀察清單訊號 |
| `pattern_signals.json` | 型態掃描結果（多方含 failure_stop，空方獨立列表） |
| `history_index.json` | 每日 Top 50 歷史追蹤（60 個交易日） |
| `market_summary.json` | Claude LLM 市場分析報告 |
| `sim_state.json` | 模擬機器人持倉狀態（日更，含 trailing stop、進場理由） |

---

## 訊號系統（V6.2）

### 資料流

```
scanner.py → action_signals.json（Scanner 100 分）
                ↓
pattern_scanner.py → pattern_signals.json（型態 100 分）
                ↓
signal_conditions.py → compute_entry_score()
                ↓
最終進場分數（最高 150 分）= Scanner + 型態加分 + 雙確認加分
```

### 進場評分（最高 150 分）

**Scanner 4 條件（最高 100 分）**：

| # | 條件 | 分數 |
|---|------|------|
| 1 | 排名穩定（Top10 ≥2天 or Top50 ≥3天） | 30 分 |
| 2 | 高信心度（Uncertainty < 當日 Q30） | 25 分 |
| 3 | 機構連續淨買 ≥2 天 | 25 分 |
| 4 | 相對低點（RSI<40 or 價格<MA20） | 20 分 |

**型態加分（最高 +40 分）**：型態分數 60–74 → +20；75–89 → +30；≥90 → +40

**雙確認加分（+10 分）**：型態分數 ≥60 且 Alpha rank ≤200

**入場門檻**：正常市場（TWII > MA60）≥70 分；保守模式 ≥90 分

### 四層退場（`signal_conditions.py`）

| 層 | 觸發條件 | 動作 |
|----|---------|------|
| **第一層** | Trailing Stop / 型態失敗線跌破 / 外資連賣 3 天 / M頭或假突破確認 / 持有 >30 天 | 立即全出 |
| **第二層** | 排名連 2 天出 Top50 / Uncertainty 超進場 2 倍 / RS_20d 負值 3 天 / 排名穩定性消失 | 立即全出 |
| **第三層** | RSI>75 且動能下滑 / 報酬 ≥+20% / Alpha_20d 連降 3 天 | 減半倉 |
| **第四層** | SQ 排名落後市場後 50% 且有新訊號且滿倉 | 換倉 |

### Trailing Stop 四檔（止損線只往上調）

| 峰值報酬 | 止損線 |
|---------|-------|
| < +5% | 進場價 −5% |
| ≥ +5% | 進場價 +2% |
| ≥ +10% | 進場價 +6% |
| ≥ +15% | 進場價 +10% |

### 型態掃描器（`quant/pattern_scanner.py`）

**多方型態（5 種，用於進場加分）**：W底、彈簧型W底、頭肩底、收斂三角底部、上飄旗形

**空方型態（2 種，用於退場觸發）**：M頭（跌破頸線確認）、假突破向下

每個多方訊號含 `failure_stop`（型態失敗退場價），直接接入四層退場第一層。

**評分**：型態強度 40 + 成交量 30 + 位置（波段跌幅）20 + RSI 10 + 漂亮加分 + Alpha 加成

### 部位管理

- 最多同時持有 **15 檔**，單股上限 **10%**，保留至少 **5% 現金**緩衝
- 交易成本：買進 0.15%；賣出 0.15% + 0.30% 證交稅

---

## Web Dashboard 各頁面說明

### 1. Dashboard — 今日選股訊號（`/`）

**用途**：每日推論後，展示全市場 Alpha 排名前 50 支股票，同時呈現大盤總覽。

**資訊面板**：
- 大盤統計卡（4張）：加權指數 TAIEX（點位 + 漲跌幅）、漲/跌家數、VIX 恐慌指數、USD/TWD 匯率 + SPX 漲跌幅
- Alpha 排名訊號表：可切換 **5日 / 20日 / 60日 Alpha** 排序，顯示前 50 名
  - 欄位：排名、股票代號 + 名稱、產業、Alpha 強度（視覺化色條）、Sharpe（= alpha_20d / uncertainty）、建倉比重（Kelly %）、訊號（多/空/觀望）+ 信心等級
  - 流動性過濾：`alpha_20d = -999` 的股票自動排除
  - 點擊任一股票開啟 StockModal 查看 5d/20d/60d 詳情
- 全球市場側欄：S&P 500、黃金、VIX、USD/TWD、100 JPY/TWD 即時行情
- 產業強弱熱力圖：依模型 Alpha 加權計算各產業平均強度

**信心等級判斷**：
- 高信心：Uncertainty < 0.03
- 中信心：0.03 ≤ Uncertainty < 0.06
- 低信心：Uncertainty ≥ 0.06

---

### 2. 交易訊號掃描（`/scanner`）

**用途**：呈現 Signal Scanner 的輸出結果，包含買入推薦、觀察清單與退場警告。

**資訊面板**：
- 狀態卡（4張）：當前大盤環境（正常/保守）、入場門檻（每日動態）、買入推薦家數、退場警告家數
- 買入推薦卡片（每股一張）：
  - 股票名稱 + 代號、信心等級、評分（/100）、符合條件數（X/4）
  - 4 個入場條件的個別結果（✅/❌）：排名穩定、高信心、相對低點、機構買賣超
  - 底部指標：Sharpe、不確定度（±%）、Kelly 建倉比重
  - 異常 Alpha 警示（alpha_20d > 100% 時顯示 ⚠️ 異常Alpha 標籤）
- 觀察清單表：評分未達門檻但有潛力的股票，顯示各條件達成狀態
- 入場/退場規則說明彈窗（點擊「入場門檻」卡片開啟）：
  - 完整 4 條件說明 + 分數
  - 大盤環境判斷（正常 vs 保守）
  - 退場條件清單
  - Trailing Stop 四檔機制
  - Sharpe Score 部位管理對照表（>3 高：15-20%；1-3 中：5-10%；<1：不買）

---

### 3. 量化分析儀表板（`/quant`）

**用途**：多維度市場分析，包含技術面、籌碼面、市場廣度、模型 Alpha 與型態學五個子分頁。

**頂部 KPI 卡（即時，所有分頁共用）**：
- 加權指數 RSI(14)（超買 >70 / 超賣 <30 / 中性）
- 外資近 5 日淨買（億台幣，正/負色）
- 漲跌比（當日上漲:下跌 家數，A/D Ratio）
- 融資餘額（億台幣，資券比）

#### 分頁 1：技術指標

- 大盤技術指標表（^TWII）：MA5/10/20/60、RSI、MACD、KD、Bollinger Band、ATR，每項附帶狀態判斷
- 技術面評分雷達圖（6 維：RSI 動能、MACD 趨勢、KD 隨機、Bollinger、MA 排列、ATR 波動）
- 風險與槓桿指標：10 日實現波動率（年化）、台股 Beta（vs SPX，近 60 日）、MA20、MA60、融資餘額、融券餘額、資券比（>12 偏高，留意軋空）

#### 分頁 2：籌碼面

- 三大法人近 5 日淨買超表（外資 / 投信 / 自營商，億台幣）
- 外資每日淨買走勢（近 5 日迷你圖）
- 產業資金輪動：各產業外資近 5 日淨買，橫向色條比較
- 融資融券概況：融資餘額、融券餘額、資券比（>10 偏高）

#### 分頁 3：市場廣度

- 漲跌家數柱狀圖（近 5 日，上漲/下跌雙色）
- A/D Ratio 趨勢折線圖（>1 = 多方市場廣度）
- 今日上漲/下跌家數 KPI 卡

#### 分頁 4：模型 Alpha

- 產業平均 Alpha 橫向柱狀圖（依模型 20d Alpha 加權，取前 8 大產業）
- 信心分佈（高/中/低信心各幾檔、佔比）
- Alpha 分佈統計：多空比（Alpha>0 : Alpha<0）、平均 Alpha、最強 Alpha、可投資股數

#### 分頁 5：傳統型態學

- KPI 卡：掃描股票數、型態匹配數（Score ≥ 60）、雙重確認數、掃描耗時
- 型態匹配股票表：股票名稱/代號/產業、型態類型、時間框架、分數（色條）、關鍵價/目標/停損、風報比、Alpha 排名、雙確認標籤
- **雙重確認**：型態分數 ≥ 60 **且** Alpha 排名 ≤ 200
- 多方型態（5 種）：**W底**、**彈簧型W底**（破底翻）、**頭肩底**、**收斂三角底部**、**上飄旗形**（中繼）
- 空方型態（2 種，退場警示）：**M頭**（跌破頸線確認）、**假突破向下**
- 型態評分（100 分）：型態強度 40 + 成交量 30 + 位置（波段跌幅）20 + RSI 10；漂亮加分（各型態獨立）；Alpha 加成：Top 200 +10 / Top 300 +5
- 每個多方訊號含 `failure_stop`（型態失敗退場絕對價位）
- **雙重確認**：型態分數 ≥60 且 Alpha 排名 ≤200

---

### 4. 持倉追蹤（`/portfolio`）

**用途**：追蹤永豐證券實際帳戶持倉，並與 Scanner 退場訊號交叉比對。

**資訊面板**：
- 統計卡（4張）：持倉股數、未實現損益（NT$）、模型建議一致比例（BUY/HOLD 家數）、資料更新時間
- 累積損益曲線（折線圖，14 天）
- 持倉比例甜甜圈圖
- 持倉明細表：
  - 欄位：股票名稱/代號、持有量、平均成本、目前價格、損益（NT$）、報酬率（色條 + %）、模型建議（BUY/HOLD/SELL 徽章）
  - 若持倉股觸發 Scanner 退場訊號，顯示 🔴 退場 標籤
- 退場警告面板（有持倉股觸發時顯示）：列出觸發股票、退場原因、目前排名
- 資料來源：永豐證券 Shioaji API 同步，每日 17:00 更新

---

### 5. AI 市場日報（`/market`）

**用途**：展示 Claude LLM 每日自動生成的市場分析報告，以及推論時的大盤快照。

**資訊面板**：
- 大盤 KPI 卡（4 張）：TAIEX 點位 + 漲跌幅、VIX（>30 高度恐慌 / >20 波動偏高 / ≤20 市場平穩）、S&P 500 今日漲跌幅、黃金今日漲跌幅
- Claude AI 市場分析報告（左側主版面）：
  - Markdown 格式解析，分章節呈現
  - 報告由每日推論結束後自動呼叫 Claude API 生成
  - 包含：大盤趨勢分析、Top 10 選股邏輯說明、風險提示等
- 推論時市場快照（右側）：推論當下抓取的台股漲跌、S&P 500、VIX、黃金、USD/TWD 數值
- 今日 AI 精選 Top 10 表（報告底部）：
  - 欄位：排名、股票名稱/代號、20d Alpha、信號強度（Signal_Quality）、建議比重、信心
- 風險提示面板：說明 AI 報告非投資建議

---

### 6. 模型狀態（`/model`）

**用途**：呈現模型訓練狀態（真實資料）、線上 IC 表現與模型架構摘要。2026-06-12 全面改版：移除所有寫死/合成資料，改讀 `training_status.json`（Colab 訓練逐 epoch 記錄）+ `ic_analysis.json`（每日推論線上 IC）。

**資料流**：Colab 訓練 → 每 epoch 寫 JSON 到 Google Drive → 訓練完成手動複製到 `V6/results/` → git push → Render（30 分 TTL，`POST /api/performance/cache/refresh` 強制刷新）。

**資訊面板**：
- 標頭 badge：動態訓練狀態（⚙️ 訓練中 ep X/Y / ✓ 訓練完成 / 早停）
- KPI 卡（4 張）：訓練 Best Val IC（目標 ≥ 0.05）、線上 IC 5d（mean + ICIR + 天數）、線上 IC 20d、最後推論日期
- 訓練學習曲線：Train Loss、Val Loss、Val IC 三條折線，標記 IC=0.05 目標線
- Scale Gate 面板：Short/Mid/Long 三分支權重的 epoch 曲線（觀察是否偏 Long）
- 線上 IC 時序柱狀圖（5d，正藍負紅，標記 0.05 參考線）
- 模型架構摘要：由 training_status.json 的 config 快照動態帶出（維度、參數量、訓練/驗證範圍、GPU、padding mask、macro_norm）
- 無資料時顯示空狀態提示（不再顯示合成數字）；Walk-Forward 面板已移除，待 WF 例行化後以真實 fold 結果加回

---

### 7. 投資模擬機器人（`/sim`）

**用途**：兩套量化策略的歷史回測與即時狀態追蹤，均使用歸檔的 df_kelly.csv 計算，含真實交易成本。

分為兩個子分頁：

#### Alpha 機器人（Signal_Quality 排名驅動）

**進場條件**：
- Signal_Quality 排名進入全市場 **Top-20**
- 以信號日收盤價建倉，最多同時持有 **15 檔**
- 單次建倉至少佔淨值 **2%**

**退場條件（任一觸發）**：
- 排名跌出 **Top-35**（訊號弱化）
- Signal_Quality < 0.5（淨 Alpha 轉負）
- 持有超過 **30 個交易日**（時間停損）

**倉位管理**：
- Kelly 加權比例分配，單股上限 **12%**，保留至少 **5% 現金**緩衝

**交易成本**：
- 買進：0.15%；賣出：0.15% + 0.30% 證交稅 = 0.45%（round-trip 約 0.6%）

**展示內容**：
- KPI 卡：累積報酬（含超額報酬 vs 大盤）、最大回撤 + 夏普 + 勝率、目前持倉狀態、交易筆數 + 平均持有天數
- 資產曲線 vs TWII（基準值 = 100，折線圖）
- 目前持倉表：代號、入場日、入場價、現價、市值、損益、持有天數、SQ 排名
- 觀察清單：排名第 21–35、候選入場股票
- IC 分析面板：IC 均值、ICIR、t 統計量、IC > 0 天數比例、滾動 IC 折線圖、逐日 IC 柱狀圖
- 交易紀錄表（最新 15 筆，可展開更多）

#### Scanner 機器人（4 條件加權訊號驅動）

使用與 `/scanner` 頁面相同的 Signal Scanner（4 條件加權評分），策略稍有不同：

**進場**：評分 ≥ 55 分（正常市場）/ ≥ 70 分（保守模式），最多持 **10 檔**，單股上限 **15%**

**退場**：Scanner EXIT 訊號（排名掉出 Top 50 連 2 天 / 外資連續 3 天淨賣）或 Trailing Stop 觸及

**展示內容**：
- KPI 卡：累積報酬 vs 大盤超額報酬、最大回撤 + 夏普 + 勝率、目前持倉、大盤環境
- 資產曲線 vs TWII
- Scanner 持倉明細（每股展示 4 條件狀態 + 當前止損線 + 峰值報酬）
- 觀察清單（差 1 個條件未達入場門檻的候選股）
- Scanner 策略說明（可摺疊）

---

## 目錄結構速覽

```
MarketMamba/
├── V6/                          ← 當前主力量化引擎
│   ├── marketmamba/             ← 核心 Python 套件
│   │   ├── config.py            ← 全域超參數 & 路徑
│   │   ├── data/
│   │   │   ├── fetcher.py       ← FinMind + yfinance 爬蟲
│   │   │   └── feature_engineer.py ← 特徵工程（V6.1=56 維 / V6.2=59 維，macro_norm 參數）
│   │   ├── training_status.py   ← 訓練狀態 JSON 記錄（模型狀態頁面資料源）
│   │   ├── models/              ← Mamba + GATv2（⚠️ 禁止修改）
│   │   ├── signals/
│   │   │   ├── scanner.py           ← 交易訊號掃描器（加權評分 v1.2）
│   │   │   └── signal_conditions.py ← 共用進退場條件（140分制、四層退場）
│   │   ├── quant/
│   │   │   └── pattern_scanner.py   ← 型態辨識（5多方+2空方，含 failure_stop）
│   │   ├── knowledge/
│   │   │   └── graph_builder.py     ← KG 建構（產業/集團/供應鏈/相關性邊）
│   │   ├── llm/
│   │   │   └── report_generator.py  ← Claude API 每日市場報告
│   │   ├── backtest/
│   │   │   ├── engine.py            ← 回測引擎
│   │   │   └── sim_engine_v3.py     ← 有狀態日更模擬機器人（sim_state.json 持久化）
│   │   └── robot/portfolio_manager.py ← 持倉管理
│   ├── run_daily_inference.py   ← 每日推論主入口（WSL2 執行）
│   ├── scripts/
│   │   └── daily_inference.bat  ← Windows Task Scheduler 觸發點（17:00）
│   ├── results/                 ← 每日推論輸出（git push 到 GitHub）
│   └── models/                  ← ⚠️ 模型 checkpoint，禁止修改
│
├── app/
│   ├── backend/                 ← FastAPI（部署到 Render）
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routers/
│   │       ├── signals.py       ← 訊號 API + 快取（asyncio.Lock）
│   │       └── market.py        ← 大盤指標 API
│   └── frontend/                ← Vite + React（部署到 Vercel）
│       └── src/pages/           ← 7 個頁面
│
└── archive/                     ← 舊版本（V3–V5.5），只讀
```

---

## 技術棧

| 層次 | 技術 |
|------|------|
| 時序模型 | Mamba SSM（`mamba-ssm 2.3.0`） |
| 圖神經網路 | GATv2Conv（`PyTorch Geometric`） |
| 資料源 | FinMind API + yfinance + TWSE 直連 |
| LLM 報告 | Anthropic Claude API |
| 前端 | Vite + React + Recharts |
| 後端 | FastAPI + Uvicorn |
| 持倉 API | 永豐證券 shioaji |
| 訓練平台 | Google Colab A100 |
| 本機推論 | RTX 3060（WSL2 Ubuntu） |
| 前端部署 | Vercel |
| 後端部署 | Render（免費方案，15 分鐘無流量會 spin down） |

---

*Last updated: 2026-06-05*
