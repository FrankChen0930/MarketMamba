# MarketMamba — 對話交接文件

> 把這份文件丟給新的 Antigravity 對話，可以直接接續進度，不需要重新解釋背景。

---

## 當前狀態（截至 2026-04-27）

### 訓練進度

| 項目 | 狀態 | 備註 |
|------|------|------|
| Quick Fold（N=500）| ✅ 完成 | IC=+0.0744 @epoch 15，確認架構有效 |
| **Final Training（N=全部）** | 🔄 **進行中** | Colab A100，epoch 4 時 IC=+0.0621，看起來正常 |
| Walk-Forward（36 folds）| ⏳ 待執行 | Final Training 完成後進行 |
| 推論腳本 | ⏳ 待建 | Final Training 完成後建 |

### 前端進度

| 項目 | 狀態 | 路徑 |
|------|------|------|
| Vite + React 骨架 | ✅ 完成 | `app/frontend/` |
| Dashboard 頁（今日選股）| ✅ 完成（mock 資料）| `src/pages/Dashboard.jsx` |
| 量化分析頁 | ✅ 完成（mock 資料）| `src/pages/QuantAnalysis.jsx` |
| AI 消息面頁 | ✅ 完成（mock 資料）| `src/pages/MarketView.jsx` |
| 持倉追蹤頁 | ✅ 完成（mock 資料）| `src/pages/Portfolio.jsx` |
| FastAPI backend | ⏳ 待建 | `app/backend/`（尚未建立）|

---

## 已確認的設計決策

### 技術架構
- **Frontend**: Vite + React → 部署到 **Vercel** (`app/frontend/`)
- **Backend**: FastAPI → 部署到 **Render 免費方案** (`app/backend/`)
- **每日推論**: **本機 RTX 3060**（PersonalOS 開機自動觸發）→ git push 結果到 GitHub
- **資料流**: GitHub raw URL → Render Backend cache（每小時更新）→ 前端
- **LLM**: Claude API（尚未申請，最後一步）
- **持倉 API**: 永豐證券 shioaji（已申請，Key 存在 `.env`）
- **PersonalOS**: 之後包裝成桌面 app（Electron 或 Tauri）

### 雲端部署設定（部署後填入實際 URL）
```
Frontend: https://marketmamba.vercel.app
Backend:  https://marketmamba-api.onrender.com
```

### PersonalOS 推論整合流程
```
開機 → 檢查今日有沒有推論
  → 沒有 → run_daily_inference.py（RTX 3060）
  → git add V6/results/df_kelly.csv && git commit && git push
  → POST https://marketmamba-api.onrender.com/api/signals/cache/refresh
  → Backend 重新載入最新結果
```

### 介面設計
- 深色 Terminal 風格（參考六維度 V4）
- 量化訊號和 LLM 消息面**分開頁籤**，不互相污染
- 個人版：4 頁籤（含持倉）；爸爸版：3 頁籤（不含持倉）
  - 切換方式：`VITE_USER_MODE=public` 環境變數

### 訓練設定（Final Training）
- 訓練資料：2005-01-03 → 2023-12-31
- 驗證資料：2024-01-01 → 今（真正樣本外）
- N_SAMPLE_TRAIN：None（全部 2888 支股票）
- Early Stop：IC-based，patience=15
- Checkpoint：`v6_final.pt`（不 commit，在 .gitignore）

---

## 已解決的技術問題（不要再改）

1. **N_SAMPLE_TRAIN override 不生效** → 改成在 train_model 裡從 live config 讀取（`import marketmamba.config as _live_cfg`）
2. **PyTorch 2.6 UnpicklingError** → `torch.load(..., weights_only=False)` + `add_safe_globals([TrainingHistory])`
3. **ListNet sum → mean** → val loss 不再系統性高於 train loss
4. **Val < Train IC 低** → 改成 IC-based early stop + ic_mode=True
5. **KG WARNING（Insufficient rolling data）** → 非致命，靜態邊仍有效

---

## 檔案位置（新 workspace）

```
D:\Desktop\work\ProjectForMe\MarketMamba\   ← workspace 根目錄（git repo）
  V6/
    marketmamba/
      models/trainer.py     ← 訓練核心（已修改）
      config.py             ← 超參數
    notebooks/
      V6_Training.py        ← Colab 訓練腳本（Cell 1~8）
    models/
      v6_final.pt           ← 訓練完成後在這（.gitignore）
  app/
    frontend/               ← 前端骨架（localhost:5173）
      src/
        pages/              ← 4 個頁面已建好
        components/AppLayout.jsx
        mockData.js         ← 暫用 mock 資料
  .env                      ← API Keys（.gitignore）
  PROJECT.md                ← 技術架構總覽
  HANDOFF.md                ← 本文件
```

---

## .env 內容（Key 值本人自行填入）

```env
SINOPAC_API_KEY=...         # 永豐證券（行情/帳務，已申請）
SINOPAC_SECRET_KEY=...
ANTHROPIC_API_KEY=...       # Claude（待申請）
FINMIND_TOKEN=...           # FinMind 資料 API
```

---

## 下一步（依優先順序）

### 等 Final Training 完成後（今晚 → 明早）

1. **在 Colab 確認結果**
   - `Best Val IC` 是多少？（目標 > 0.05）
   - Best Epoch 是幾？
   - 下載 `v6_final.pt` 到 `V6/models/`

2. **建推論腳本**（新對話，新 workspace）
   - 讀取 `v6_final.pt`
   - 對當日特徵矩陣做推論
   - 輸出 Alpha 訊號 CSV

3. **建 FastAPI backend**
   - 接推論輸出
   - 接永豐 shioaji 持倉
   - 提供前端 API 端點

4. **前端接上真實資料**（目前是 mock）

5. **Claude 新聞整合**（最後）

---

## 整合系統規劃（PersonalOS，未來）

- 所有個人工具整合進同一個 monorepo
- 資料夾：`D:\Desktop\work\ProjectForMe\`
  - `MarketMamba\`（已移入）
  - `LifeOS\`（Notion 管理，之後重做）
  - `PersonalOS\`（整合 App，之後建）
- 開機自動檢查推論 → pystray + Windows 通知
- 整合前端可包裝成 Electron exe

---

## GitHub

```
https://github.com/FrankChen0930/MarketMamba.git
Branch: main
```
