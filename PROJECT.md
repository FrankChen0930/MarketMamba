# MarketMamba V6 — 專案簡介

> 台股截面因子模型，使用 Mamba SSM + GATv2 圖神經網路預測個股超額報酬

---

## 專案定位

MarketMamba 是一個 **每日自動選股系統**，核心功能：
1. 每日推論 → 輸出全市場 2888 支股票的 Alpha 得分（截面排名）
2. 歷史回測 → 驗證選股策略的 IC、ICIR、Sharpe
3. 模型訓練 → 在 Google Colab A100 上訓練，checkpoint 下載到本機

整合系統需要的介面主要是：**每日 Alpha 訊號輸出** 和 **模型績效歷史**。

---

## 目錄結構

```
MarketMamba/
  V6/
    marketmamba/            ← 主要 Python 套件
      config.py             ← 所有超參數和路徑設定
      data/
        fetcher.py          ← FinMind API 爬蟲（下載原始資料）
        feature_engine.py   ← 特徵工程（46 個因子）
        merger.py           ← 合併所有資料來源
      models/
        architecture.py     ← MarketMambaV6 模型架構
        trainer.py          ← 訓練迴圈（train_model 函數）
      knowledge/
        graph_builder.py    ← KG 建構（產業/集團/供應鏈邊）
    notebooks/
      V6_Training.py        ← Colab 訓練腳本（Cell 1~8）
    models/
      v6_final.pt           ← 最終訓練好的 checkpoint（.gitignore）
    Data/
      processed_v6/         ← 特徵矩陣、KG cache（.gitignore）
  app/
    frontend/               ← Vite + React 前端（已建好骨架）
    backend/                ← FastAPI 後端（待建）
  .env                      ← API Keys（.gitignore，永遠不 commit）
  .gitignore
```

---

## 資料流

```
FinMind API（原始股價/財報）
    ↓ data/fetcher.py
Raw Parquet（Data/raw_v6/）
    ↓ data/feature_engine.py
Feature Matrix（9,101,749 rows × 51 cols，2005~今）
    ↓ data/merger.py + graph_builder.py
訓練用 DataFrame + Knowledge Graph CSR Matrix
    ↓ models/trainer.py
v6_final.pt（訓練好的模型）
    ↓ 推論腳本（待建）
Alpha 訊號 CSV（每日，2888 支股票 × Alpha 得分）
    ↓ FastAPI /api/signals
前端 Dashboard
```

---

## 模型架構

```python
MarketMambaV6(
  INPUT: (N_stocks, 252 days, 46 features)   # N = 當日有效股票數

  # Stage 1: 時序建模（每支股票獨立）
  Embedding: Linear(46 → 256)
  MambaBlocks × 8 (d_model=256, d_state=32)  # 捕捉時序模式

  # Stage 2: 截面互動（股票之間）
  GATv2: (N, 256) → (N, 256)                # 利用 KG 邊做跨股票注意力

  # Stage 3: 預測頭
  Output: Linear(256 → 3)                    # [Alpha_5d, Alpha_20d, Alpha_60d]
)

訓練目標: 截面 z-score 後的未來報酬排名
損失函數: MSE（各期）+ ListNet（排名損失）
```

---

## 知識圖譜（Knowledge Graph）

4 種邊類型，存為 scipy CSR sparse matrix：

| 邊類型 | 權重 | 說明 |
|--------|------|------|
| TWSE 產業 | 0.5 | 同 TWSE 行業分類 |
| 集團從屬 | 0.8 | 鴻海族群、台積電生態圈等 |
| TPEX 供應鏈 | 0.6 | 上下游廠商關係 |
| 60日滾動相關 | 動態 | Pearson 相關 > threshold |

- Cache 路徑：`Data/processed_v6/knowledge_graph_cache.npz`
- 共 ~43,000 條邊（2890 個節點）

---

## 訓練參數（V6 Final）

| 參數 | 值 |
|------|----|
| 訓練資料 | 2005-01-03 ~ 2023-12-31（約 4450 天）|
| 驗證資料 | 2024-01-01 ~ 今（約 350 天）|
| N_SAMPLE_TRAIN | None（全部股票）|
| Epochs | 最多 100，IC-based early stop（patience=15）|
| Optimizer | AdamW, lr=1e-4 |
| Scheduler | OneCycleLR |
| AMP | 啟用（A100 FP16）|

---

## 關鍵配置（config.py）

```python
# 路徑
PROCESSED_DIR   = Path("Data/processed_v6")
MODELS_DIR      = Path("models")
KG_CACHE_PATH   = PROCESSED_DIR / "knowledge_graph_cache.npz"

# 模型
INPUT_DIM       = 46       # 特徵數量
D_MODEL         = 256      # Mamba 隱藏維度
D_STATE         = 32       # SSM state size
N_MAMBA_LAYERS  = 8        # Mamba 層數
PRED_HORIZONS   = [5, 20, 60]  # 預測天期（交易日）

# 訓練
LR              = 1e-4
EARLY_STOP      = 15       # IC-based patience
N_SAMPLE_TRAIN  = None     # None = 全部股票
```

---

## 整合系統需要的介面

當 PersonalOS 整合 app 需要 MarketMamba 的資料時，透過：

### 1. 每日 Alpha 訊號

```python
# 推論腳本輸出（待建）
# 路徑：Data/signals/YYYY-MM-DD.csv
# 欄位：stock_id, name, alpha_5d, alpha_20d, alpha_60d, rank, signal
```

### 2. FastAPI 端點（app/backend/，已規劃）

```
GET  /api/signals          → 今日選股排名
GET  /api/signals/{date}   → 指定日期的訊號
GET  /api/performance      → IC 趨勢、Walk-Forward 結果
GET  /api/portfolio        → 永豐持倉（shioaji）
POST /api/run-inference    → 觸發今日推論
```

### 3. 模型載入方式

```python
import torch
from marketmamba.models.architecture import MarketMambaV6
from marketmamba.models.trainer import TrainingHistory

torch.serialization.add_safe_globals([TrainingHistory])
ckpt = torch.load("V6/models/v6_final.pt", map_location="cpu", weights_only=False)
model = MarketMambaV6()
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## 環境與依賴

```bash
# Python 環境
pip install torch mamba-ssm torch-geometric scipy pandas numpy
pip install finmind shioaji fastapi uvicorn python-dotenv

# 前端
cd app/frontend
npm install  # react, recharts, react-router-dom, lucide-react, axios
npm run dev  # → localhost:5173
```

---

## API Keys（存在 .env，永不 commit）

```env
SINOPAC_API_KEY=...      # 永豐證券（行情/帳務）
SINOPAC_SECRET_KEY=...
ANTHROPIC_API_KEY=...    # Claude（新聞整合，待申請）
FINMIND_TOKEN=...        # FinMind 資料 API
```

---

## 訓練狀態（截至 2026-04-27）

| 項目 | 狀態 |
|------|------|
| Quick Fold（N=500）| ✅ 完成，IC=+0.0744 @ep15 |
| Final Training（N=全部）| 🔄 進行中（Colab A100）|
| Walk-Forward（36 folds）| ⏳ 待執行 |
| 推論腳本 | ⏳ 待建 |
| FastAPI backend | ⏳ 待建 |
| 前端骨架 | ✅ 完成（localhost:5173）|

---

## 未來計畫

1. **Final Training 完成** → 下載 `v6_final.pt`，建推論腳本
2. **Walk-Forward** → 評估模型跨時期穩健性
3. **FastAPI backend** → 接推論 + 永豐 shioaji
4. **Claude 新聞整合** → LLM 消息面分析
5. **PersonalOS 整合** → 移入 monorepo，統一啟動
