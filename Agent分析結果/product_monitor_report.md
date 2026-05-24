# 🐍 MarketMamba — 產品監控分析報告
> 分析角色：產品監控者  
> 分析日期：2026-05-24  
> 版本：V6 (Frontend Vite+React / Backend FastAPI / 推論引擎 Mamba+GATv2)

---

## 總覽評分

| 面向 | 評分 | 等級 |
|------|------|------|
| 使用者體驗（UX） | 7.5 / 10 | 🟡 良好，有明顯改善空間 |
| 系統穩定性 | 6.0 / 10 | 🟠 中等，存在關鍵單點失敗風險 |
| 資料完整性 | 5.5 / 10 | 🟠 中等，部分頁面仍用靜態數據 |
| 部署可靠性 | 5.0 / 10 | 🔴 偏低，免費方案限制明顯 |

---

## 一、使用者體驗（UX）分析

### 1.1 優點 ✅

#### 視覺設計
- 暗色主題設計統一，顏色系統（`--positive` 綠 / `--negative` 紅 / `--accent-blue`）使用一致
- Skeleton loading 元件（`SkeletonCard`、`SkeletonTable`）提供良好的載入過渡體驗
- 股票詳情可點擊查看 `StockModal` 彈窗，資訊結構完整（Alpha 5d/20d/60d + 不確定度）
- TopBar 的 Ticker Bar 滾動動畫增加即時感，增強 Dashboard 氛圍
- 頁面導航清晰，圖示輔助標籤讓功能定位直觀

#### 功能設計
- `TradingSignals` 頁面（交易訊號掃描）設計最成熟：入場條件 Modal 詳解規則，觀察清單區分 4 個維度，用 ✅/❌/➖ 直觀呈現條件達成狀況
- 持倉頁面（Portfolio）的模型訊號 vs 持倉的交叉比對邏輯（退場警告自動匹配）非常實用
- `MetricTooltip` 元件提供指標解釋，降低量化術語的理解門檻

---

### 1.2 問題清單 ❌

#### 🔴 嚴重問題

**[UX-01] 量化分析頁面（QuantAnalysis）數據已過時且硬編碼**
- `TAIEX_TECH`、`INSTITUTIONAL`、`BREADTH_DATA`、`SECTOR_FLOW` 等技術指標為靜態常數（對應到 2026-04 市場數據）
- 用戶打開頁面看到的「RSI = 54.3」、「外資近5日 +63.3億」是固定假數據，與實際市場脫節
- **風險**：若用戶基於此數據做出交易決策，後果嚴重
- **建議**：接入 `market.py` 的 yfinance 實時數據；或至少在頁面加上「數據截止 YYYY-MM-DD」警示 Banner

**[UX-02] P&L 曲線為假資料（Portfolio 頁面）**
- `buildPnlHistory()` 函式使用 `Math.sin()` 產生模擬曲線（見 Portfolio.jsx #43-51）
- 即使實際持倉存在，損益曲線仍是「假的正弦波形」
- **風險**：用戶誤以為這是真實損益記錄
- **建議**：從 Supabase 讀取歷史損益記錄，若無則顯示「歷史損益資料尚不足以繪圖」空狀態

---

#### 🟠 中等問題

**[UX-03] S&P 500 / 黃金在 Dashboard 全局市場欄位顯示「—」**
- Dashboard.jsx #242-256：`MacroRow` 的 `value` 欄位硬寫 `"—"`，S&P 500 和黃金價格永遠不顯示實際數值，只顯示漲跌幅
- `market.py` 的 `_build_market()` 確實有從 yfinance 取 `spx_pct`、`gold_pct` 但**沒有取 price**，這造成前端拿不到數值
- **建議**：後端補充 `spx_price`、`gold_price` 欄位

~~**[UX-04] 多個頁面在後端 Render 冷啟動時體驗差**~~ ✅ **已透過 UptimeRobot 定期 ping 解決，服務保持常駐，此問題不存在。**

**[UX-05] 交易訊號掃描在 Scanner 資料為空時 UX 模糊**
- 當 `action_signals.json` 尚未生成（模型尚未完成推論），API 回傳 `buy_signals: [], exit_signals: [], date: null, market_regime: "UNKNOWN"`
- 前端顯示「目前沒有股票達到入場條件」——讓用戶無法分辨是真的「市場無訊號」還是「系統尚未就緒」
- **建議**：當 `date === null` 或 `market_regime === "UNKNOWN"` 時，顯示專屬的「系統尚未執行每日推論」空狀態

**[UX-06] Ticker Bar 顯示的股票價格永遠為「—」**
- `market.py #156-164`：Ticker 只抓 `pct`（漲跌幅），不抓 `price`，所以前端 `TickerItem.price` 永遠傳 `"—"`
- 用戶看到跑馬燈顯示「台積電 — +0.8%」——缺少股價讓體驗大打折扣
- **建議**：在 `_safe_yf_price()` 後補充 `price` 字段填入 TickerItem

**[UX-07] 模型狀態頁（ModelStatus）缺少自動更新機制**
- `ModelStatus.jsx` 若存在（已在路由中），推論進度或訓練狀態需手動重整，缺少 polling 或 WebSocket 支援

---

#### 🟡 輕微問題

**[UX-08] 行動裝置體驗不完整**
- `dashboard` 的主要佈局使用 `gridTemplateColumns: '1fr 330px'`，在小螢幕（< 768px）不會自動折疊
- `WatchListTable` 有 `overflowX: auto` 但主容器無限寬，觸控滑動體驗差
- **建議**：增加 `@media (max-width: 768px)` 讓 Grid 切換為單欄

**[UX-09] 重新整理按鈕無 Loading 狀態**
- 點擊「🔄 重新整理」後，按鈕沒有 Disabled 或 Loading 狀態，用戶可能連按多次觸發重複請求

**[UX-10] QuantAnalysis「傳統型態學」tab 的固定文字令人困惑**
- 「今日匹配股數: 0 — ⚠️ 市場波動過大」是靜態文字，用戶不知道「波動過大」是系統判斷還是寫死的
- 建議若掃描引擎未整合，直接標示為「功能開發中」

---

## 二、系統穩定性分析

### 2.1 架構風險地圖

```
本機 RTX 3060 (run_daily_inference.py)
    │
    ├── ❌ 單點失敗：若本機當機/停電，所有用戶看到昨日舊數據
    ▼
V6/results/df_kelly.csv + action_signals.json
    │
    ▼ git push → GitHub raw URL
    │
    ▼ Render Backend (免費方案，冷啟動延遲)
    │
    ▼ Vercel Frontend
```

### 2.2 後端穩定性問題

**[SYS-01] 🔴 全域可變快取（race condition 風險）**
- `signals.py` 使用模組級別全域變數 `_cache`、`_cache_time`、`_scanner_cache` 等
- FastAPI 使用 async 並發處理請求，多個請求同時到達在快取失效瞬間，可能觸發多次 GitHub fetch（thundering herd）
- **建議**：使用 `asyncio.Lock()` 保護快取更新，或改用 Redis/Memcached

```python
# 現況（有 race condition）
async def _get_signals():
    global _cache, _cache_time
    if _cache and ...:
        return _cache
    result = await _load_from_github()  # 多個請求可能同時執行到這行
    _cache = result
```

**[SYS-02] 🔴 `signals.py` 中變數未定義即使用**
- Line 151-155：`get_scanner_signals()` 引用了 `_scanner_cache`、`_scanner_cache_time`，但這兩個全域變數的定義在 **Line 281-282**（檔案底部）
- Python 在函式執行時才解析全域變數，但這樣的結構易導致維護混亂，NameError 在特定情況下可能出現
- **建議**：將所有全域快取變數集中宣告在文件頂部

**[SYS-03] 🟠 Render 冷啟動 + GitHub fetch 超時鏈**
- `market.py` 的 `_build_market()` 會連續呼叫 5 次 `_safe_yf_pct()` + 2 次 `_safe_yf_price()`（每次 8 秒 timeout）
- 在 Render 冷啟動後，第一個 API 請求可能需要等待 **5~7 個 yfinance 請求串行執行**（最多 56 秒）
- **建議**：改為並行 `asyncio.gather()`；或對 market 數據設更短的 timeout（3 秒）並接受降級

**[SYS-04] 🟠 `ticker` 端點為每個 top-7 股票呼叫 yfinance**
- `market.py #156`：`pct = _safe_yf_pct(f"{ticker}.TW")` 在迴圈中被呼叫 7 次（同步阻塞）
- 每次請求需下載 `.TW` 股票的近兩日資料，7 個合計最多 **56 秒**等待
- 用戶打開任何頁面，AppLayout 都會呼叫此 ticker endpoint
- **建議**：ticker 端點應使用異步並行；或使用 FinMind API 批次抓取台股即時報價

**[SYS-05] 🟠 Portfolio 依賴 Supabase，無本地 fallback**
- `portfolio.py`：若 SUPABASE_URL 未設定，回傳空持倉（無 mock fallback）
- 與其他路由（signals 有 mock fallback）行為不一致
- **建議**：統一 fallback 策略，至少回傳 mock 持倉以確保展示效果

**[SYS-06] 🟡 `fin_news.py` 無快取設計**
- Analyze 端點每次呼叫都執行 5 次 Tavily 搜尋 + 1 次 Claude API
- 若用戶頻繁點擊「生成報告」按鈕，API 費用快速累積
- **建議**：加入 TTL = 4 小時的快取（同一天的新聞分析不需重複生成）

---

### 2.3 每日推論管線風險

**[SYS-07] 🔴 推論失敗無告警機制**
- `run_daily_inference.py` 執行失敗時，只記錄 log 到 stdout
- 若 Task Scheduler 執行失敗（網路斷線、FinMind API 限速、VRAM OOM），隔日 Dashboard 依然顯示昨日舊數據
- 用戶**無從得知**數據是否已更新（Topbar 只顯示日期，沒有「距今 X 小時」提示）
- **建議**：加入 Line Notify / Telegram Bot 推播推論結果；在 Dashboard 顯示「數據更新於 N 小時前」

**[SYS-08] 🟠 git push 失敗為 non-fatal 但影響嚴重**
- `_push_to_github()` 失敗時回傳 `False` 並繼續（non-fatal）
- 但結果未推上 GitHub = Render backend 取不到新數據 = 前端展示昨日數據
- **建議**：git push 失敗應記錄到可監控的位置（或更積極地重試 3 次）

**[SYS-09] 🟠 WSL2 環境依賴（跨系統脆弱性）**
- 推論腳本在 WSL2 Ubuntu 內執行，需設置 `GIT_DISCOVERY_ACROSS_FILESYSTEM=1`
- Windows 重開機後，Task Scheduler 若在 WSL2 啟動前觸發，命令直接失敗
- **建議**：在 Task Scheduler 中加入 `timeout 30` 的 WSL2 預熱步驟，或改用 Windows 原生 Python

**[SYS-10] 🟡 LLM Report 失敗不影響主流程，但 API 費用無上限保護**
- `generate_market_report()` 失敗時只 warning，合理設計
- 但若 ANTHROPIC_API_KEY 有效且 Claude API 正常，每次推論都消耗 token（max_tokens=3000）
- **建議**：加入每日調用次數上限檢查；或改用更便宜的 claude-haiku 作為 daily report 模型

---

### 2.4 資料管線風險

**[SYS-11] 🟠 FinMind API 無速率限制保護**
- `data/fetcher.py` 尚未看到明確的 rate limit handler
- FinMind 免費方案有請求上限，批次更新 2888 支股票時可能被封鎖
- **建議**：加入 exponential backoff retry 和每分鐘請求計數器

**[SYS-12] 🟡 Knowledge Graph 為靜態構建，無增量更新**
- KG 在模型訓練時建構，每日推論重用固定的 `edge_index`
- 若台股有公司下市/新上市，KG 不會自動更新
- **建議**：每月排程重建 KG；或在推論時動態加載最新的股票列表

---

## 三、改善優先級矩陣

| 優先級 | 問題 | 影響 | 工作量 |
|--------|------|------|--------|
| P0 | UX-01 靜態假數據（量化分析頁） | 用戶信任度崩潰 | 中 |
| P0 | UX-02 假P&L曲線 | 數據誤導 | 小 |
| P0 | SYS-07 推論失敗無告警 | 靜默故障 | 小 |
| P1 | SYS-01 快取 race condition | 服務穩定性 | 中 |
| P1 | SYS-03/04 yfinance 串行阻塞 | 冷啟動體驗 | 中 |
| ~~P1~~ | ~~UX-04 Render 冷啟動 UX~~ | ~~首訪體驗~~ | ✅ 已解決（UptimeRobot） |
| P1 | UX-05 Scanner 空狀態模糊 | 功能可信度 | 小 |
| P2 | UX-03 全局市場欄位缺少價格 | 數據完整性 | 小 |
| P2 | UX-06 Ticker 價格顯示「—」 | 資訊完整性 | 小 |
| P2 | SYS-02 全域變數定義位置 | 代碼維護性 | 小 |
| P3 | UX-08 行動裝置支援 | 用戶覆蓋率 | 中 |
| P3 | UX-09 按鈕無 Loading 狀態 | 微體驗 | 小 |
| P3 | SYS-06 FinNews 無快取 | API 費用 | 小 |

---

## 四、亮點功能評估

| 功能 | 評估 |
|------|------|
| MC-Dropout 不確定度量化 | ⭐⭐⭐⭐⭐ 業界少見，顯著提升信號可信度 |
| Trailing Stop 分層機制 | ⭐⭐⭐⭐ 完整且合理的風控設計 |
| 知識圖譜（GATv2）整合 | ⭐⭐⭐⭐⭐ 捕捉供應鏈效應的核心競爭力 |
| 交易訊號掃描（4維條件） | ⭐⭐⭐⭐ 最成熟的用戶體驗，設計清晰 |
| Skeleton Loading | ⭐⭐⭐⭐ 載入體驗良好，有專業感 |
| GitHub → Render 自動更新 | ⭐⭐⭐⭐ 架構巧妙；UptimeRobot 保持常駐，消除冷啟動問題 |
| 財金新聞教學分析（AI） | ⭐⭐⭐ 差異化功能，需補上快取 |

---

## 五、核心建議（執行路線圖）

### 立即修復（本週內）
1. **UX-01** QuantAnalysis 靜態數據加上「數據更新於 YYYY-MM-DD」紅色警示 Banner
2. **UX-02** Portfolio 損益曲線改為「資料不足」空狀態，移除 `Math.sin()` 假曲線
3. **UX-05** Scanner 空狀態分辨「市場無訊號」vs「系統尚未就緒」
4. **SYS-07** 推論完成後發送 Line Notify 告警（成功/失敗各一條）

### 短期改善（兩週內）
5. **SYS-04** Ticker API 改為並行 yfinance 請求（`asyncio.gather`）
6. **UX-06** 後端補充 Ticker 股價欄位
7. **UX-03** 後端補充 `spx_price`、`gold_price` 欄位

### 中期架構改善（一個月內）
9. **SYS-01** 使用 `asyncio.Lock()` 保護全域快取
10. **UX-08** 前端加入 768px 斷點響應式佈局
11. **SYS-06** FinNews analyze 端點加入 4 小時 TTL 快取
12. **SYS-11** FinMind fetcher 加入 retry + rate limit 保護

---

*本報告由 Antigravity 代理人分析生成 · 2026-05-24*
