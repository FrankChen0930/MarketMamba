# MarketMamba V5.5 — 終極量化決策系統

> Mamba SSM + KG-Enhanced GAT + 雙軌 FinBERT 情緒引擎

自動化台股量化投資系統，結合深度學習時序預測、知識圖譜強化的圖注意力網路、
以及雙語消息面情緒分析，產出 30 天 Alpha 軌跡預測與凱利資金配置建議。

---

## 🏗️ 專案架構

```
MarketMamba/
├── marketmamba/                  # 核心 Python 套件
│   ├── config.py                # 全域設定 (路徑/Token/超參數)
│   ├── data/
│   │   ├── fetcher.py           # FinMind + yfinance 資料擷取
│   │   ├── merger.py            # 跨頻率時序融合 + IFRS 修復
│   │   ├── feature_engineer.py  # 特徵煉金 (技術指標/VIP/營收YoY)
│   │   └── cleaner.py          # 終極清洗 (NaN/Inf/IPO隔離)
│   ├── models/
│   │   ├── architecture.py      # MarketMambaV55 (84維) / V5 (46維)
│   │   ├── inference.py         # 推論 + 夏普/凱利 + 自動版本偵測
│   │   └── trainer.py          # 訓練迴圈 + Early Stopping
│   ├── pattern/
│   │   ├── detectors.py         # 六大型態偵測函數
│   │   └── scanner.py          # 全市場多時間框架掃描
│   ├── sentiment/               # 🆕 雙軌情緒引擎
│   │   ├── crawler_en.py        # Google News RSS 英文爬蟲
│   │   ├── crawler_cn.py        # Google News RSS 中文爬蟲
│   │   ├── finbert_en.py        # ProsusAI/finbert 英文情緒
│   │   ├── finbert_cn.py        # chinese-finbert 中文情緒
│   │   ├── auto_labeler.py      # 股價反應自動標籤產生器
│   │   └── integrator.py       # 情緒特徵整合 + 指數衰減
│   ├── knowledge/               # 🆕 知識圖譜
│   │   ├── sector_mapping.py    # TWSE 24 大產業分類
│   │   └── graph_builder.py    # KG + cosine 混合邊建構
│   ├── robot/
│   │   └── portfolio_manager.py # 調倉 + 帳本管理
│   └── deploy/
│       └── publisher.py         # GitHub 推送
├── notebooks/
│   ├── V5_5_Pipeline.py         # 📋 日常推論管線 (Colab 用)
│   └── V5_5_Training.py        # 📋 模型訓練管線 (Colab 用)
├── app.py                       # Streamlit 前端
├── requirements.txt
└── README.md
```

---

## 🚀 快速開始

### Colab 日常推論 (每日收盤後)

1. 打開 Google Colab，新增一個 Notebook
2. 將 Runtime 類型改為 **GPU (T4)**
3. 打開 `notebooks/V5_5_Pipeline.py`
4. 依序將每個 `# %% Cell` 區塊複製貼入獨立的 Cell
5. 從第一個 Cell 開始依序執行

```python
# Cell 1: 環境建置 (克隆 Repo + 安裝依賴 + 掛載 Drive)
# Cell 2: 資料同步 (FinMind + yfinance)
# Cell 3: 跨頻融合 + 特徵工程
# Cell 4: 消息面情緒 (可選，首次可跳過)
# Cell 5: AI 推論 (Mamba + GAT)
# Cell 6: 型態掃描
# Cell 7: 機器人調倉
# Cell 8: 推送 GitHub → 觸發 Streamlit 更新
```

### Colab 模型訓練 (需要重訓時)

打開 `notebooks/V5_5_Training.py`，同樣的方式操作：

```python
# Cell 1: 環境建置
# Cell 2: 資料同步 + 特徵工程
# Cell 3: 加入情緒特徵 (可選，启用後 input_dim 46→84)
# Cell 4: 開始訓練 (tqdm 進度條即時顯示)
# Cell 5: Loss 曲線視覺化
# Cell 6: 用新模型跑推論驗證
# Cell 7: 推送預測結果
```

### 本機開發

```bash
# 克隆
git clone https://github.com/FrankChen0930/MarketMamba.git
cd MarketMamba

# 安裝依賴
pip install -r requirements.txt

# 啟動前端
streamlit run app.py
```

---

## 🧠 V5.5 vs V5.0 差異

| 維度 | V5.0 | V5.5 |
|------|------|------|
| 輸入維度 | 46 (量價籌碼) | **84** (+38 情緒特徵) |
| 圖建構 | 純 cosine similarity KNN | **KG-Enhanced** (cosine + 產業分類) |
| 消息面 | ❌ 無 | ✅ 雙軌 FinBERT (EN+CN) |
| 情緒衰減 | — | 指數衰減 (半衰期 3 天) |
| 季報對齊 | +90 天粗估 | **IFRS 法定截止日** |
| 錯誤處理 | `except: pass` | `logger.warning()` + 分類策略 |
| 帳本日期 | 混亂 (有的 YYYY-MM-DD HH:MM) | 統一 **YYYY-MM-DD** |

### 向下相容

推論引擎會自動偵測特徵矩陣中是否含有情緒特徵：
- **有** → V5.5 模式 (84 維輸入，KG-GAT)
- **無** → 自動退回 V5.0 模式 (46 維輸入，cosine KNN)

現有的 `V5_DynamicGAT_Production.pth` 權重可直接使用，無需重訓。

---

## 📰 消息面情緒引擎

### 雙軌 FinBERT

| 模型 | 用途 | 來源 |
|------|------|------|
| `ProsusAI/finbert` | 英文國際新聞 | Reuters, CNBC, Bloomberg |
| `hw2942/chinese-finbert-for-sentiment-analysis` | 中文台股新聞 | 鉅亨網, UDN, 工商時報 |

### 情緒特徵清單 (38 維)

| 類型 | 欄位 | 說明 |
|------|------|------|
| 標量 (6) | `Sent_Stock_CN/EN` | 個股情緒 [-1, +1] |
| | `Sent_Market_TW/US` | 大盤情緒 |
| | `Sent_Geopolitical` | 地緣政治風險 |
| | `News_Volume_Stock` | 新聞數量 (log) |
| Embedding (32) | `Sent_Embed_EN_0~15` | 英文 FinBERT [CLS] 投影 |
| | `Sent_Embed_CN_0~15` | 中文 FinBERT [CLS] 投影 |

### Auto-Labeling (自動標籤)

用新聞發布後 5 天的累積 Alpha 來自動標註情緒：
- Alpha > +1% → Positive
- Alpha < -1% → Negative
- 其他 → Neutral

累積 ≥ 1000 條標註後，可微調 FinBERT 提升精度。

---

## 🕸️ 知識圖譜 GAT

混合邊建構公式：

```
edge_score = 0.7 × cosine_similarity + 0.3 × sector_similarity
```

- `cosine_similarity`：Mamba 最後時間步輸出特徵的相似度
- `sector_similarity`：TWSE 產業分類 (同產業=0.5, 不同=0.0)

覆蓋 24 大產業分類，含半導體、金融、航運等主要板塊。

---

## 📐 型態學雷達

六大經典結構 × 四時間框架：

| 型態 | 訊號 | 特色 |
|------|------|------|
| 🟡 標準 W底 | 做多 | 動態傾斜頸線 |
| 🟢 破底翻 | 強多 | 假跌破 (Spring) |
| 🔴 M頭 | 偏空 | 容許 15% 假突破誘多 |
| 🟣 頭肩底 | 做多 | 傾斜頸線 + 等幅測距 |
| 🔵 收斂三角 | 方向未定 | 1/2~3/4 時間密碼 |
| 💀 上飄旗跌破 | 做空 | 波段跌幅測距 |

---

## ⚙️ 技術棧

- **時序模型**：Mamba SSM (mamba_ssm 2.3.0)
- **圖網路**：GATv2Conv (PyTorch Geometric)
- **情緒分析**：HuggingFace Transformers (FinBERT)
- **資料源**：FinMind API + yfinance
- **前端**：Streamlit
- **部署**：GitHub → Streamlit Cloud

---

## 📝 License

MIT License © 2024-2026 FrankChen
