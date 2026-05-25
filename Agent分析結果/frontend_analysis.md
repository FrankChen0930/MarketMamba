# MarketMamba 前端程式碼深度分析報告

> 分析範圍：`app/frontend/src/` | 技術棧：Vite + React 19 + Recharts + Axios

---

## 一、總體架構評估

### 架構一覽

```
src/
├── App.jsx               ← 路由根（BrowserRouter + 7 個路由）
├── index.css             ← 全局 Design Token & 共用元件樣式
├── layout.css            ← 佈局系統（Grid / Panel / Topbar）
├── mockData.js           ← 開發用假資料（5 筆）
├── api/                  ← API 封裝層（6 個模組）
│   ├── client.js         ← Axios 實例（base URL / 錯誤攔截）
│   ├── signals.js        ← Alpha 訊號、推論觸發
│   ├── market.js         ← 大盤行情
│   ├── performance.js    ← 模型績效
│   ├── portfolio.js      ← 持倉
│   └── reports.js        ← Claude AI 日報
├── hooks/
│   └── useApi.js         ← 通用資料 fetch hook
├── components/
│   ├── AppLayout.jsx     ← 全域佈局（Topbar / TickerBar / Outlet）
│   ├── MetricTooltip.jsx ← 量化指標說明 Popup（25+ 指標）
│   ├── SectorHeatmap.jsx ← 產業強弱熱力圖
│   ├── SkeletonLoader.jsx← 骨架屏 + ApiError 元件
│   └── StockModal.jsx    ← 股票詳情 Modal
└── pages/
    ├── Dashboard.jsx     ← 今日選股（Alpha 排名 + 大盤快照）
    ├── TradingSignals.jsx← 交易訊號掃描（V6.1 Scanner）
    ├── QuantAnalysis.jsx ← 量化分析（5 個子頁籤）
    ├── MarketView.jsx    ← AI 市場日報（Claude 報告）
    ├── Portfolio.jsx     ← 持倉追蹤（Shioaji 同步）
    ├── ModelStatus.jsx   ← 模型狀態（Walk-Forward 圖表）
    └── InvestmentSim.jsx ← 投資模擬機器人（雙 Bot 對比）
```

### 整體評分

| 維度 | 評分 | 說明 |
|------|------|------|
| 功能完整性 | ⭐⭐⭐⭐⭐ | 7 個頁面覆蓋量化投資全流程 |
| 設計美觀度 | ⭐⭐⭐⭐⭐ | 深色 Terminal 風格、設計系統完善 |
| 程式碼結構 | ⭐⭐⭐⭐ | 分層清晰，但部分頁面過於龐大 |
| 效能優化 | ⭐⭐⭐ | 有 useMemo，但缺乏 lazy loading 和 code splitting |
| 可維護性 | ⭐⭐⭐ | 大量 inline style，缺乏型別系統 |
| 響應式設計 | ⭐⭐⭐⭐ | 有完整的 breakpoint，但 Grid 寬度硬編碼較多 |

---

## 二、優點分析

### ✅ 設計系統完整且一致

`index.css` 定義了完整的 Design Token：
- **色彩**：深色底色（`#0d1117`）搭配高飽和 accent（青藍 `#00d4ff`、霓虹綠 `#00ff88`）
- **字體**：Inter + JetBrains Mono，完美適合金融數字顯示
- **元件**：`.stat-card`、`.panel`、`.badge`、`.data-table` 全部統一定義
- **動畫**：`fadeInUp`、`skeleton-shine`、`pulse-glow`、`ticker` 等精細動畫

### ✅ 通用 Hook 設計良好

```js
// useApi.js - 簡潔的通用 fetch hook
const { data, loading, error, refetch } = useApi(() => fetchSignals());
```
- 支援 `loading` / `error` / `refetch` 三態
- 使用 `useRef` 固定 `fetchFn` 避免 stale closure
- 所有頁面統一使用，維護成本低

### ✅ API 層架構清晰

- `client.js` 統一 Axios 設定（base URL、timeout、錯誤 log）
- 各模組按業務拆分（signals / market / portfolio / reports）
- 本地開發透過 Vite proxy，生產環境透過環境變數，零改動切換

### ✅ 骨架屏（Skeleton）體驗完善

`SkeletonLoader.jsx` 提供 `SkeletonCard`、`SkeletonTable`、`SkeletonBlock`，
所有頁面都有 loading 狀態，不會出現空白屏。

### ✅ MetricTooltip 教育意義高

25+ 個量化指標（IC、ICIR、RSI、MACD、VIX 等）附上：
- 指標說明（`desc`）
- 判斷門檻（`threshold`）
讓不懂量化的使用者也能理解每個數字的意義。

### ✅ 響應式設計完整

`layout.css` 有 3 個 breakpoint：
- Desktop：`grid-4`（4 欄）、主側欄兩欄佈局
- Tablet（≤1024px）：`grid-4` 折為 2 欄，側欄消失
- Mobile（≤640px）：單欄、Nav 只顯示 icon

### ✅ InvestmentSim 雙 Bot 對比頁面創意極佳

純量化 Bot A vs 量化+LLM Bot B 的對比模擬，
是整個前端最有創意、最具教育性的功能，概念非常獨特。

---

## 三、問題識別

### 🔴 嚴重問題

#### 1. `mockData.js` 實際上沒有被任何頁面使用

**現況**：`mockData.js` 定義了 `MOCK_SIGNALS`、`MOCK_PORTFOLIO` 等假資料，
但所有頁面都直接呼叫 API（`useApi(fetchSignals)` 等），
API 失敗時只顯示 `ApiError`，**沒有使用 mock 資料作為 fallback**。

```js
// ❌ 現況：API 失敗 → 顯示錯誤訊息
if (sigError) return <ApiError message={sigError} onRetry={refetchSigs} />;

// ✅ 更好的做法：API 失敗 → 降級到 mock 資料
const signals = signalData?.signals ?? MOCK_SIGNALS;
```

**影響**：後端尚未上線時，整個前端等同空白，開發體驗很差。

---

#### 2. `QuantAnalysis.jsx` 資料完全是 hardcode（靜態假資料）

**現況**：技術指標、籌碼面、市場廣度等數據全部是 hardcode 常數：

```js
// ❌ 這些數據是靜態的，永遠不會更新！
const TAIEX_TECH = [
  { label: 'RSI(14)', value: '54.3', status: '中性', ... },  // 2026-04 的數字
  { label: 'MACD', value: '+12.4', status: '偏多', ... },
];

const INSTITUTIONAL = [
  { name: '外資', buy: 352.4, sell: 289.1, net: 63.3 },  // 靜態數字
];
```

**影響**：使用者看到的永遠是 2026-04 的數字，毫無實用價值。

---

#### 3. `Portfolio.jsx` 的 P&L 曲線是假造數據

```js
// ❌ 用 sin 函數生成假的損益曲線
function buildPnlHistory(positions) {
  return Array.from({ length: 14 }, (_, i) => ({
    day: `D-${13 - i}`,
    pnl: Math.round(totalPnl * (0.6 + (i / 13) * 0.4) + Math.sin(i * 0.8) * 5000),
  }));
}
```
這個曲線對使用者毫無意義，甚至有誤導性。

---

#### ~~4. `MarketView.jsx` 有 XSS 安全疑慮~~ ✅ 已完成 (S4)

> ✅ **已完成（2026-05-25）**：已移除 `dangerouslySetInnerHTML`，改用 React 元素渲染 `**bold**` 語法（`.split(/\*\*([^*]+)\*\*/g)` + 條件式 `<strong>`），徹底消除 XSS 風險。

```js
// ⚠️ dangerouslySetInnerHTML 接受 LLM 生成的內容，有 XSS 風險
const parsed = text.replace(/\*\*([^*]+)\*\*/g,
  (_, m) => `<strong style="color:var(--text-primary)">${m}</strong>`);
return <div dangerouslySetInnerHTML={{ __html: parsed }} />;
```
LLM 產生的文字如果包含惡意 HTML，可能被注入執行。

---

### 🟡 中等問題

#### 5. 大量 inline style，難以維護

幾乎每個元件都有大量 `style={{ ... }}` inline style，例如：
```jsx
<div style={{ display: 'flex', justifyContent: 'space-between',
  alignItems: 'center', padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
```
應該抽成 CSS class 或至少集中管理。

#### 6. 缺乏路由層級的 Code Splitting（Lazy Loading）

```jsx
// ❌ 現況：7 個頁面全部同時 import，初始 bundle 大
import Dashboard from './pages/Dashboard';
import QuantAnalysis from './pages/QuantAnalysis';
// ...

// ✅ 建議：lazy import 讓首頁更快載入
const Dashboard = lazy(() => import('./pages/Dashboard'));
const QuantAnalysis = lazy(() => import('./pages/QuantAnalysis'));
```

#### 7. `useApi` 缺乏快取機制

每次切換頁面，`useApi` 都會重新打 API，
```js
useEffect(() => { execute(); }, [execute]);
```
同樣的資料（如大盤行情）在多個頁面重複請求，既浪費也慢。

#### 8. `TradingSignals.jsx` 中 Signal Card 的 hover 效果用 `onMouseEnter/Leave` 操作 DOM

```jsx
// ❌ 直接操作 DOM style，React 反模式
onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(0,255,136,0.5)'; }}
onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(0,255,136,0.25)'; }}
```
應改用 CSS `:hover` 或 `useState` 管理。

#### 9. `MetricTooltip` 的 Popup 在表格靠右欄位時會被截斷

`position: absolute` + `transform: translateX(-50%)` 的 Popup 在靠近邊緣時會超出視窗。

#### 10. InvestmentSim 的隨機模擬沒有真正的「種子」復現機制

```js
// ⚠️ seed 狀態只是讓 useMemo 重新計算，
// 但 Math.random() 每次都不同，無法真正復現同一次模擬
const [seed, setSeed] = useState(42);
const equityA = useMemo(() => simulateEquity(portfolioA, simDays), [portfolioA, simDays, seed]);
```

---

### 🟢 小問題

#### 11. `MOCK_TICKER` 資料重複

```js
// mockData.js 第 51-53 行：TAIEX 和台積電重複了
{ id: "TAIEX", name: "加權", ... },  // 第二次出現
{ id: "2330", name: "台積電", ... }, // 第二次出現
```
Ticker 動畫是把陣列 doubled 之後播放，但原本就有重複資料。

#### 12. `AppLayout.jsx` 的 `TickerBar` 沒有暫停動畫的機制

使用者 hover 時 Ticker 不會暫停，難以閱讀特定項目。

#### 13. `StockModal.jsx` 是空檔案（0 bytes）

```json
{"name":"StockModal.jsx"}  // sizeBytes 缺失，可能為空
```
Dashboard 和 TradingSignals 都有 `import StockModal`，
但實際的 Modal 可能未實作，點擊股票後什麼都不顯示。

---

## 四、改進建議（依優先順序）

### P0 — 影響可用性，立即修復

#### 建議 A：讓 Mock 資料成為真正的開發 Fallback

在所有 API 呼叫加入 mock fallback：

```js
// api/signals.js 增加 mock fallback
export const fetchSignals = async (params = {}) => {
  try {
    return await client.get('/signals', { params }).then(r => r.data);
  } catch {
    // 後端不可用時使用 mock 資料
    return { signals: MOCK_SIGNALS, date: new Date().toISOString().slice(0, 10) };
  }
};
```

或在 `useApi` 加入 `fallback` 選項：

```js
export function useApi(fetchFn, { fallback = null, deps = [] } = {}) {
  // ...
  } catch (err) {
    if (fallback !== null) setData(fallback); // 使用 fallback 資料
    else setError(err.message);
  }
}
```

#### ~~建議 B：修復 MarketView 的 XSS 漏洞~~ ✅ 已完成 (S4)

> ✅ **已完成（2026-05-25）**：採用了「完全避免 innerHTML」方案，使用 React 元素渲染，實作與下方第二種方案相同。

```js
// 使用 DOMPurify 清理 LLM 內容，或改用 React 渲染代替 dangerouslySetInnerHTML
import DOMPurify from 'dompurify';
const sanitized = DOMPurify.sanitize(parsed);
return <div dangerouslySetInnerHTML={{ __html: sanitized }} />;

// 或者完全避免 innerHTML，改用 React span 渲染
function renderBold(text) {
  return text.split(/\*\*([^*]+)\*\*/g).map((part, i) =>
    i % 2 === 1 ? <strong key={i} style={{ color: 'var(--text-primary)' }}>{part}</strong> : part
  );
}
```

---

### P1 — 影響資料品質，下一版本修復

#### 建議 C：QuantAnalysis 接上真實 API

將 `TAIEX_TECH`、`INSTITUTIONAL`、`SECTOR_FLOW`、`BREADTH_DATA` 的靜態資料
移到 FastAPI 後端端點，前端改用 `useApi` 拉取：

```
GET /api/market/technicals   → TAIEX 技術指標
GET /api/market/institutional → 三大法人
GET /api/market/breadth      → 市場廣度
GET /api/market/sector-flow  → 產業資金輪動
```

#### 建議 D：移除 Portfolio 的假造 P&L 曲線

```jsx
// 方案 1：直接移除假曲線，顯示「尚無歷史損益資料」
// 方案 2：後端累積每日損益快照，提供真實曲線
// 方案 3：至少用 0-line 平坦線代替 sin 波動
```

---

### P2 — 提升效能和開發體驗

#### 建議 E：加入 Lazy Loading（Code Splitting）

```jsx
// App.jsx
import { lazy, Suspense } from 'react';
const Dashboard = lazy(() => import('./pages/Dashboard'));
const QuantAnalysis = lazy(() => import('./pages/QuantAnalysis'));
// ...

<Suspense fallback={<div className="page-content"><SkeletonBlock height={400} /></div>}>
  <Outlet />
</Suspense>
```

預估可將首屏 bundle 從 ~300KB 降至 ~80KB。

#### 建議 F：加入簡單的全域資料快取

```js
// 在 App 層用 React Context 或 zustand 快取常用資料
// 市場行情 + Alpha 訊號（每次只需要拉一次，多頁共用）
const MarketDataContext = createContext(null);

// 或引入 SWR / React Query（更完整的快取策略）
// npm install swr
import useSWR from 'swr';
const { data } = useSWR('/api/market', fetcher, { refreshInterval: 60000 });
```

---

### P3 — 程式碼品質提升

#### 建議 G：Signal Card 改用 CSS hover

```css
/* index.css 新增 */
.signal-card {
  border: 1px solid rgba(0, 255, 136, 0.25);
  background: rgba(0, 255, 136, 0.03);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.signal-card:hover {
  border-color: rgba(0, 255, 136, 0.5);
  box-shadow: var(--shadow-glow-green);
}
```

#### 建議 H：MetricTooltip 改用 Popper.js 或 Floating UI

```bash
npm install @floating-ui/react
```
自動計算最佳彈出方向，避免被邊緣截斷。

#### 建議 I：加入 TypeScript（長期）

目前沒有型別定義，API response 的形狀全靠 `?.` 鏈式存取猜測。
加入 TypeScript 介面定義可大幅降低維護成本：

```ts
interface SignalItem {
  stock_id: string;
  name: string;
  alpha_5d: number;
  alpha_20d: number;
  alpha_60d: number;
  sector: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: '高信心' | '中信心' | '低信心';
  uncertainty: number;
  suggested_weight?: number;
  rank: number;
}
```

#### 建議 J：Ticker 加入 hover 暫停

```css
.ticker-track {
  animation: ticker 40s linear infinite;
}
.ticker-bar:hover .ticker-track {
  animation-play-state: paused;  /* 滑鼠懸停暫停 */
}
```

---

## 五、完成度評估（對標 HANDOFF.md）

| 項目 | HANDOFF 狀態 | 實際狀態 | 補充 |
|------|-------------|---------|------|
| Dashboard | ✅ 完成（mock）| ✅ 完成，已接 API | 骨架屏、StockModal 待確認 |
| QuantAnalysis | ✅ 完成（mock）| ⚠️ 圖表接 API，但指標是 hardcode | 需接真實技術指標 API |
| AI 消息面 | ✅ 完成（mock）| ✅ 接 Claude 報告 API | 有 XSS 疑慮需修復 |
| 持倉追蹤 | ✅ 完成（mock）| ✅ 接 Shioaji API，但曲線是假的 | P&L 曲線需改善 |
| 交易訊號 | **新增** | ✅ TradingSignals 完整實作 | V6.1 Scanner 完整 |
| 投資模擬 | **新增** | ✅ InvestmentSim 雙 Bot 完整 | 創意亮點 |
| 模型狀態 | **新增** | ✅ ModelStatus Walk-Forward 圖表 | 資料接 API |
| FastAPI backend | ⏳ 待建 | ❌ 未建 | 所有 API 目前返回錯誤 |

---

## 六、下一步行動建議

```
優先順序：
1. [立即] 修復 StockModal.jsx（確認是否為空檔案，補全實作）
2. [立即] 加入 DOMPurify 修復 MarketView XSS
3. [立即] 在 useApi 加入 mock fallback，讓前端在後端未上線時可展示
4. [本週] 建立 FastAPI 後端框架（至少 /api/signals 和 /api/market 端點）
5. [本週] QuantAnalysis 中靜態數據改用真實 API
6. [下週] 加入 Lazy Loading + SWR/React Query
7. [長期] TypeScript 遷移
```

---

*分析時間：2026-05-24 | 前端版本：Vite 8 + React 19 + Recharts 3*
