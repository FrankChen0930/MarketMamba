# MarketMamba V6 — API Reference

> 給 **PersonalOS** 整合使用的快速參考。

---

## 服務 URL

| 環境 | Frontend | Backend |
|------|----------|---------|
| **本機開發** | `http://localhost:5173` | `http://localhost:8000` |
| **雲端（部署後更新）** | `https://marketmamba.vercel.app` | `https://marketmamba-api.onrender.com` |

---

## 啟動（本機開發）

```bash
# Terminal 1 — Backend
cd app/backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd app/frontend
npm run dev  # → localhost:5173

# Swagger UI（測試端點）
http://localhost:8000/docs
```

---

## 資料流向（雲端）

```
PersonalOS 開機
  → 檢查今日是否已推論
  → 執行 run_daily_inference.py（RTX 3060）
  → git push V6/results/df_kelly.csv
  → POST /api/signals/cache/refresh  ← 通知 backend 重新載入

Render Backend（每小時自動 cache refresh）
  → 讀 GitHub raw URL 取最新 df_kelly.csv
  → 提供 /api/signals 給前端
```

---

## Base URL

```
# PersonalOS 裡用這個（雲端）
const MM_API = 'https://marketmamba-api.onrender.com/api'

# 本機測試
const MM_API = 'http://localhost:8000/api'
```

---

## Endpoints

### GET `/api/health`
服務存活確認。

```json
{ "status": "ok", "service": "MarketMamba V6", "version": "6.0.0" }
```

---

### GET `/api/signals`
今日選股訊號，依 alpha_20d 排序。

**Query params:**
| 參數 | 類型 | 說明 |
|------|------|------|
| `top` | int | 只回傳前 N 名（e.g. `?top=5`） |
| `signal` | str | 過濾訊號：`BUY` / `HOLD` / `SELL` |

**Response:**
```json
{
  "date": "2026-04-27",
  "model_version": "V6",
  "total_stocks": 2888,
  "signals": [
    {
      "rank": 1,
      "stock_id": "2330",
      "name": "台積電",
      "sector": "半導體",
      "alpha_5d": 0.142,
      "alpha_20d": 0.187,
      "alpha_60d": 0.231,
      "uncertainty": 0.018,
      "vol_ratio": 1.32,
      "signal": "BUY",
      "suggested_weight": 0.15,
      "confidence": "高信心"
    }
  ]
}
```

**PersonalOS 最簡用法（取今日 Top 5 買入訊號）:**
```javascript
const res = await fetch('http://localhost:8000/api/signals?top=5&signal=BUY')
const { signals } = await res.json()
```

---

### GET `/api/signals/{date}`
指定日期訊號（格式 `YYYY-MM-DD`）。Response 格式同上。

---

### POST `/api/signals/run-inference`
觸發今日推論。訓練完成前回傳 `not_ready`。

```json
{ "status": "not_ready", "message": "Training in progress..." }
```

---

### GET `/api/performance`
完整量化績效資料。

```json
{
  "ic_history": [
    { "epoch": 1, "train_loss": 2.72, "val_loss": 2.49, "val_ic": 0.013 }
  ],
  "wf_folds": [
    { "fold": "F01", "period": "2010-2012", "ic": 0.041, "icir": 0.52, "sharpe": 1.21, "ret": 0.148 }
  ],
  "cumret": [
    { "month": "2024-01", "model": 100.0, "benchmark": 100.0 }
  ],
  "best_val_ic": 0.0744,
  "best_epoch": 15,
  "training_status": "training"
}
```

---

### GET `/api/performance/ic`
只回傳 IC 歷史陣列（比 `/api/performance` 輕量）。

---

### GET `/api/market`
大盤狀態 + 系統狀態。

```json
{
  "taiex": { "value": 22381.4, "change": 87.3, "change_pct": 0.39 },
  "advancing": 563,
  "declining": 412,
  "model_ic": 0.0744,
  "last_run": "2026-04-27 15:30",
  "run_status": "not_ready",
  "training_epoch": 5,
  "training_status": "training"
}
```

`run_status` 的可能值：
| 值 | 說明 |
|----|------|
| `completed` | 推論已完成，訊號有效 |
| `running` | 推論執行中 |
| `not_ready` | 模型訓練中，使用 mock 資料 |

---

### GET `/api/market/ticker`
Ticker 跑馬燈資料。

```json
{
  "items": [
    { "id": "TAIEX", "name": "加權", "price": "22,381", "change": "+87.3", "pct": "+0.39%", "up": true }
  ]
}
```

---

### GET `/api/portfolio`
持倉列表（mock，之後接永豐 shioaji）。

```json
{
  "total_pnl": 86500.0,
  "total_value": 2412000.0,
  "data_source": "mock",
  "last_updated": "2026-04-27 15:30",
  "positions": [
    {
      "stock_id": "2330",
      "name": "台積電",
      "qty": 1000,
      "avg_price": 830.0,
      "current_price": 915.0,
      "pnl": 85000.0,
      "pnl_pct": 10.24,
      "model_signal": "BUY"
    }
  ]
}
```

---

## PersonalOS 整合範例（JavaScript）

```javascript
const MM_BASE = 'http://localhost:8000/api'

// 確認服務在線
async function isOnline() {
  try {
    const r = await fetch(`${MM_BASE}/health`, { signal: AbortSignal.timeout(2000) })
    return r.ok
  } catch { return false }
}

// 取今日 Top 5 訊號
async function getTopSignals(n = 5) {
  const r = await fetch(`${MM_BASE}/signals?top=${n}&signal=BUY`)
  return (await r.json()).signals
}

// 取市場狀態（判斷是否用 mock）
async function getMarketStatus() {
  const r = await fetch(`${MM_BASE}/market`)
  return r.json()
}
```

---

## 資料狀態說明

| 端點 | 現在 | Model 好了之後 |
|------|------|----------------|
| `/api/signals` | mock 資料 | `V6/results/df_kelly.csv` |
| `/api/performance` | mock 資料 | `V6/results/ic_history.json` |
| `/api/market` | mock 資料 | TWSE API |
| `/api/portfolio` | mock 資料 | 永豐 shioaji |

**PersonalOS 不需要改任何程式碼**，只有 backend 內部的資料來源會換。
