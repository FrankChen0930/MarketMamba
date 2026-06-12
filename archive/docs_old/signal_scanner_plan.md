# MarketMamba V6.1 — Trading Signal Scanner 設計文件

> 訓練完成後實作。基於 2026/05/07 討論定案。

## 背景

3416 融程電實測：model 推薦 #1 → 買入 → 兩天後 +7% 賣出。
驗證了 alpha 訊號有效，但缺乏系統化的進出場判斷機制。

---

## 入場條件（滿足 2/4 觸發推薦）

| # | 條件 | 判斷邏輯 | 資料來源 |
|---|------|---------|---------|
| 1 | **排名穩定性** | Top 10 連續 ≥2 天 **或** Top 50 連續 ≥3 天 | `history_index.json` |
| 2 | **高信心** | `Uncertainty < 0.02`（MC-Dropout） | `df_kelly.csv` |
| 3 | **相對低點** | RSI < 40 **或** 當前價 < 20日均線 | `prices_raw.parquet` |
| 4 | **機構淨買入** | 外資/投信 連續 2 天淨買入 | `institutional_raw.parquet` |

> [!IMPORTANT]
> Alpha_5d + Alpha_20d 用於入場決策。Alpha_60d **不參與**入場判斷。
> 長期看好的真正信號是「持續留在排名內」，而非單次 60d 預測值。

### 大盤環境過濾

| 環境 | 入場門檻 |
|------|---------|
| TWII > 60日均線（正常市場） | 滿足 2/4 即推薦 |
| TWII < 60日均線（保守模式） | 需滿足 3/4 才推薦 |

---

## 退場條件（任一觸發）

| 條件 | 觸發動作 |
|------|---------|
| Alpha 排名連續 2 天掉出 Top 50 | 🔴 **退場** |
| 觸及 trailing stop（見下表） | 🔴 **退場** |
| 外資連續 3 天淨賣出 | ⚠️ **減碼觀察** |
| 大盤跌破 60MA | ⚠️ **全倉減碼至 50%** |

### Trailing Stop 機制

| 持倉報酬 | 止損線位置 |
|----------|-----------|
| < +5% | 固定 -5%（成本價） |
| ≥ +5% | 成本 +2%（鎖利） |
| ≥ +10% | 成本 +6% |
| ≥ +15% | 成本 +10% |

---

## 部位管理

| Sharpe Score | 建議配置 |
|-------------|---------|
| > 3（高） | 15-20% 資金 |
| 1-3（中） | 5-10% 資金 |
| < 1（低） | 不買 |

> [!WARNING]
> 同一產業（sector）不超過 30% 總部位，避免集中風險。

---

## 系統架構

```
每日推論完成後 (run_daily_inference.py)
    ↓
Signal Scanner（新模組：signal_scanner.py）
    ├── 比對 history_index.json → 排名穩定性 ✓/✗
    ├── 讀取 df_kelly.csv → 信心度 + Sharpe ✓/✗
    ├── 讀取 prices → RSI / 均線位置 ✓/✗
    ├── 讀取 institutional → 機構資金方向 ✓/✗
    ├── 讀取 TWII → 大盤環境判斷
    ├── 比對持倉 → 退場警告
    ↓
輸出：action_signals.json
    ├── BUY:  [{ticker, reason, conditions_met, alpha, confidence, suggested_weight}]
    ├── SELL: [{ticker, reason, current_return, rank_trend, days_held}]
    └── HOLD: [{ticker, days_held, current_rank, return, trailing_stop_level}]
    ↓
LLM 分析（只針對 BUY / SELL 清單做詳細報告）
    ↓
Dashboard 新頁面：Trading Signals
    ├── 今日推薦買入（含 LLM 分析摘要）
    ├── 持倉追蹤（排名趨勢圖 + trailing stop 可視化）
    └── 退場警告
```

---

## 實作步驟

- [ ] **Step 1**: 新增 `marketmamba/signals/scanner.py` — 核心 Signal Scanner 邏輯
- [ ] **Step 2**: 擴充 `history_index.json` — 從 Top 10 擴展到 Top 50 追蹤
- [ ] **Step 3**: 整合到 `run_daily_inference.py` — 推論完自動跑 scanner
- [ ] **Step 4**: 新增 `action_signals.json` 輸出格式
- [ ] **Step 5**: LLM report 整合 — 針對 BUY/SELL 信號做詳細分析
- [ ] **Step 6**: 前端 Trading Signals 頁面
- [ ] **Step 7**: 持倉追蹤 — 手動輸入持倉，自動追蹤 alpha 排名變化

---

## 範例：3416 融程電回測

```
5/2 (五): 3416 排名 #1, 信心=高, RSI=38, 外資買入
           → 4/4 條件滿足 → 🔥 強烈買入
5/5 (一): 3416 排名 #7, 持倉 +3%
           → HOLD（排名穩定，未觸及任何退場條件）
5/6 (二): 3416 排名 #15, 持倉 +7%, 排名下滑中
           → ⚠️ trailing stop 上移至 +2%
5/7 (三): 3416 排名 #35, 持倉 +5%
           → HOLD（仍在 Top 50 內）
5/8 (四): 3416 掉出 Top 50
           → 🔴 退場信號（連續下滑 + 掉出 Top 50）
```
