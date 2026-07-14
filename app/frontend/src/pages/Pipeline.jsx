import React, { useState } from 'react';

// ─────────────────────────────────────────────────────────────
// Pipeline 逐階段說明頁（方向一 Step 9，2026-07-15）
// 內容底稿：docs/breadth-pipeline-page-draft-2026-07-12.md（§1~§7 + §5.5）
// 全部為靜態研究成果數字（非每日更新資料），數字核實日 2026-07-15。
// ─────────────────────────────────────────────────────────────

const TH  = { fontSize: 12, color: 'var(--text-muted)', padding: '8px 12px', borderBottom: '1px solid var(--border)', textAlign: 'left', whiteSpace: 'nowrap' };
const TD  = { fontSize: 13, padding: '8px 12px', borderBottom: '1px solid rgba(48,54,61,0.4)', textAlign: 'left', lineHeight: 1.6 };
const MONO = { fontFamily: 'var(--font-mono)' };

function Panel({ icon, title, sub, children }) {
  return (
    <div className="panel">
      <div className="panel-header" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
        <div style={{ fontSize: 15, fontWeight: 700 }}>{icon} {title}</div>
        {sub && <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{sub}</div>}
      </div>
      <div className="panel-body">{children}</div>
    </div>
  );
}

// 財金理由 callout —— 設計原則：每步附一句財金理由
function Why({ children }) {
  return (
    <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 12, padding: '10px 12px', background: 'rgba(88,166,255,0.06)', borderLeft: '3px solid var(--accent-blue)', borderRadius: '0 6px 6px 0', lineHeight: 1.7 }}>
      <b>財金理由：</b>{children}
    </div>
  );
}

function ScrollTable({ children }) {
  return <div style={{ overflowX: 'auto' }}><table className="data-table" style={{ width: '100%', minWidth: 560 }}>{children}</table></div>;
}

// ── §7 誠實限制卡片 ──────────────────────────────────────────
const STATUS = {
  fixed:    { label: '✅ 已修復',       color: 'var(--positive)' },
  research: { label: '🔬 研究中',       color: 'var(--accent-blue)' },
  open:     { label: '⚠️ 未修復已揭露', color: 'var(--accent-amber)' },
};

const LIMITATIONS = [
  { status: 'fixed', title: 'GAT 曾在推論時失效近 6 週（M1）',
    problem: '為省 VRAM 分批推論，每批只取批內知識圖譜邊，跨批邊全部被丟掉——GAT 推論時近乎退化成 identity，與訓練（全圖）嚴重不一致。',
    impact: '修復當日 Top50 名單重疊只有 30/50、不確定性整體 −34%，證明先前排名確實被此 bug 扭曲。',
    fix: '2026-06-12 改為兩段式推論（Mamba 分批、GAT 一次吃完整 cross-section 圖）。' },
  { status: 'open', title: 'Macro 特徵組曾整組歸零（D1，部分修復）',
    problem: '總經特徵同日全市場同值，橫斷面 z-score 除以 std=0 → 整組 12 維恆為 0，模型 macro 分支長期無資訊輸入。',
    impact: '雙模型已改 time-series 標準化；V6.1（線上 20d 主力）尚未套用，其 macro 分支至今實質無效。',
    fix: '重訓部署 V6.2 時同步切換 macro_norm="ts"（部署 checklist 已列）。' },
  { status: 'open', title: '存活者偏差（D3，未修復）',
    problem: 'Universe 取自當前上市清單，已下市股不在訓練資料——模型沒看過「走向下市的股票長什麼樣」。',
    impact: '對地雷股辨識力存疑；反轉類訊號的絕對 IC 水位被墊高（量化幅度未知）。',
    fix: '完整修復需歷史成分股名單（多半不可得），現階段靠流動性門檻降低實害並誠實揭露。' },
  { status: 'open', title: '每日有效 cross-section ~1,950 / 2,515',
    problem: '202 天歷史門檻 + 特徵 NaN 剔除。',
    impact: '新上市一年內的股票模型不覆蓋。',
    fix: '屬設計取捨（序列模型需要足夠歷史），不修。' },
  { status: 'fixed', title: 'Scale gate 塌縮：多尺度設計對單一 horizon 增益不成立',
    problem: '趨勢模型 gate 在 ep3 即 100% 押 Long 分支；5d 目標下 Short 分支近 0（診斷顯示是「被 gate 餓死」而非無資訊）。',
    impact: '「多尺度自適應融合」的實際結論是它教會我們何時不需要多尺度。',
    fix: '短線模型已因此簡化為單尺度（60 步、1.66M 參數）。' },
  { status: 'research', title: 'Uncertainty 校準是進行中的研究，不是已解決的功能',
    problem: 'MC-Dropout U 與後續 |超額報酬| 相關約 +0.20~0.25；「低 U = 更準」在 20d 大致成立（U 最有信心箱 IC 0.120 vs 最沒把握兩箱 0.083–0.089），在 5d post-P0 樣本上暫時反向。',
    impact: '5d 反向樣本僅 14 天（實質獨立觀察 ~3 個），不能定論。',
    fix: '7 月底樣本 ≥20 天後重跑校準分析再判（conviction-c-analysis）。' },
  { status: 'open', title: '總經 raw 資料自 2026-04-24 未更新',
    problem: '每日更新不含 macro，保守模式 regime 閘門（TWII vs MA60）實質未啟用。',
    impact: '對 V6.1 推論無影響（macro 分支本來就被 D1 歸零——見上方的諷刺現實）。',
    fix: '待把 macro 加進每日更新，scanner 端 fallback + 新鮮度檢查已備好、資料一更新自動生效。' },
  { status: 'open', title: '線上 IC 樣本仍淺、Walk-Forward 未例行化',
    problem: '線上 20d IC 僅個位數天數；WF 需 36 次重訓，尚未例行化執行。',
    impact: '頁面所有績效數字都附樣本天數，樣本不足處明講、不放合成數字。',
    fix: 'WF 例行化列 backlog（U7）；線上 IC 隨每日歸檔自動累積。' },
];

function LimitCard({ item }) {
  const [open, setOpen] = useState(false);
  const s = STATUS[item.status];
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 8, marginBottom: 8, overflow: 'hidden' }}>
      <button onClick={() => setOpen(!open)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, padding: '10px 14px', background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary, #e6edf3)' }}>{item.title}</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
          <span style={{ fontSize: 11, color: s.color, whiteSpace: 'nowrap' }}>{s.label}</span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{open ? '▲' : '▼'}</span>
        </span>
      </button>
      {open && (
        <div style={{ padding: '2px 14px 12px', fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
          <div><b>問題：</b>{item.problem}</div>
          <div><b>影響：</b>{item.impact}</div>
          <div><b>處置：</b>{item.fix}</div>
        </div>
      )}
    </div>
  );
}

// ── 主頁面 ───────────────────────────────────────────────────
export default function Pipeline() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* 頁首 */}
      <div className="panel">
        <div className="panel-body" style={{ lineHeight: 1.8 }}>
          <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>🔬 Pipeline 逐階段說明</div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            逐階段回答「有什麼／為什麼選它」：具體、放真實數字、每步附財金理由，不寫行銷文案。
            所有數字取自 repo 內實際程式與輸出（核實日 2026-07-15），包含<b>失敗與未解決的問題</b>（見最下方「誠實限制」）。
          </div>
        </div>
      </div>

      {/* §1 資料層 */}
      <Panel icon="🗄️" title="1. 資料層：三層 fallback 與覆蓋率">
        <ScrollTable>
          <thead><tr><th style={TH}>層</th><th style={TH}>來源</th><th style={TH}>提供什麼</th><th style={TH}>資料可用時間</th></tr></thead>
          <tbody>
            <tr><td style={TD}>1（最快）</td><td style={TD}>yfinance</td><td style={TD}>OHLCV 價量</td><td style={TD}>收盤後 ~15 分鐘</td></tr>
            <tr><td style={TD}>2</td><td style={TD}>TWSE / TPEX 官方 API</td><td style={TD}>三大法人買賣超（T86 / dailyTrade）</td><td style={TD}>~16:30–17:00</td></tr>
            <tr><td style={TD}>3（fallback）</td><td style={TD}>FinMind</td><td style={TD}>融資融券、千張大戶、月營收、財報</td><td style={TD}>18:00–19:00</td></tr>
          </tbody>
        </ScrollTable>
        <ul style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.9, margin: '12px 0 0', paddingLeft: 20 }}>
          <li>Universe：2005 至今、上市+上櫃 4 位數字代碼（過濾 ETF/權證/特別股），市值 ≥ 5 億、5 日均成交值 ≥ 1,000 萬</li>
          <li>訓練 universe ~2,515 支 × ~5,200 個交易日（prices_raw 約 874 萬列）；每日推論實際 cross-section ~1,950 支</li>
        </ul>
        <Why>台股的資訊優勢很大一塊在籌碼面（三大法人、大戶持股），這些只有台灣本地源有。三層設計是為了「17:00 推論時每一類資料都已就位」，單一來源斷供時當日推論仍能跑。</Why>
      </Panel>

      {/* §2 Point-in-time */}
      <Panel icon="⏱️" title="2. Point-in-time：對齊「市場何時知道」而非「數字屬於何時」">
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.9 }}>
          財報與月營收若按「所屬期間」對齊，模型會偷看未來——3 月營收 4 月 10 日才公告，回測時 3 月 31 日就用它等於作弊。實作：
          <ul style={{ margin: '8px 0 0', paddingLeft: 20 }}>
            <li>月營收 <span style={MONO}>available_from = 當月月底 + 11 天</span>；季報 <span style={MONO}>季底 + 45 天</span></li>
            <li>全部用 as-of join：每個交易日只取「彼時已公告」的最新一筆</li>
            <li>macro 時序標準化用 <span style={MONO}>shift(1)</span>（只用昨天以前的分布）；forward return 只存在於 label 欄</li>
          </ul>
        </div>
        <Why>這是回測可信度的地板。任何 IC 數字，若 label 或特徵有 look-ahead，整個實驗作廢。</Why>
      </Panel>

      {/* §3 特徵工程 */}
      <Panel icon="🧩" title="3. 特徵工程：59 維、4 因子組">
        <ScrollTable>
          <thead><tr><th style={TH}>組</th><th style={TH}>維度</th><th style={TH}>內容</th><th style={TH}>為什麼是這組</th></tr></thead>
          <tbody>
            <tr><td style={TD}>A 價格動能</td><td style={{ ...TD, ...MONO }}>15</td><td style={TD}>OHLCV、1/5/20 日報酬、MA、RSI、ATR、RS 相對強度</td><td style={TD}>動能與反轉是橫斷面選股最被驗證的因子族</td></tr>
            <tr><td style={TD}>B 機構籌碼</td><td style={{ ...TD, ...MONO }}>20</td><td style={TD}>外資/投信/自營、融資融券、借券、千張大戶、外資持股比</td><td style={TD}>台股散戶佔比高，法人足跡資訊含量特別高——主場優勢，維度給最多</td></tr>
            <tr><td style={TD}>C 基本面</td><td style={{ ...TD, ...MONO }}>12</td><td style={TD}>PER、PBR、月營收 MoM/YoY、EPS、毛利率、ROE、FCF</td><td style={TD}>估值與品質提供 20d/60d horizon 的慢變量</td></tr>
            <tr><td style={TD}>D 總經環境</td><td style={{ ...TD, ...MONO }}>12</td><td style={TD}>TWII/SPX、VIX、美債殖利率、金油匯、期貨未平倉、景氣燈號</td><td style={TD}>控制 regime：同一根 K 線在多頭與空頭的意義不同</td></tr>
          </tbody>
        </ScrollTable>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.9, marginTop: 12 }}>
          標準化兩套規則（教訓換來的設計，見誠實限制 D1）：股票級特徵（A/B/C）用 per-date 橫斷面 z-score（模型要學「今天誰相對強」）；
          總經特徵（D）用 expanding time-series z-score（macro 同日全市場同值，橫斷面 z-score 會把它整組歸零）。
          <br />Label：<span style={MONO}>Alpha_Nd = 個股前瞻 N 日報酬 − TWII 同期</span>，訓練時再轉每日橫斷面百分位 rank（理由見 §5）。
        </div>
      </Panel>

      {/* §4 輸入規格 */}
      <Panel icon="📐" title="4. 模型輸入規格">
        <ul style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.9, margin: 0, paddingLeft: 20 }}>
          <li>輸入張量 <span style={MONO}>(N ≈ 1,950, SEQ_LEN = 252, DIM = 59)</span>——每支股票一整年的日頻特徵序列</li>
          <li>納入門檻：至少 202 天真實歷史；不足 252 天者前段 zero-padding + mask</li>
          <li>知識圖譜（GATv2 用）：<b>642,451 條邊</b>——集團從屬 0.8、產業鏈 0.6、TWSE 產業分類 0.5、60 日滾動相關 ≥ 0.7（動態）</li>
        </ul>
        <Why>252 天覆蓋一個完整年度循環（財報四季 + 除權息）；KG 讓「台積電漲、供應鏈跟漲」這種結構性傳導不必靠模型自己從價格猜。</Why>
      </Panel>

      {/* §5 架構 */}
      <Panel icon="🧠" title="5. 模型架構：為什麼是 Mamba + GATv2 + rank 目標">
        <pre style={{ ...MONO, fontSize: 12, lineHeight: 1.7, background: 'rgba(110,118,129,0.08)', padding: '12px 14px', borderRadius: 8, overflowX: 'auto', margin: 0, color: 'var(--text-secondary)' }}>
{`FactorGroupedEmbedding（4 組按維度比例投影 → d_model 256）
  → MultiScaleMambaEncoder（短 20 / 中 60 / 長 252 三分支，gate 自適應融合）
  → GATv2（KG 邊引導的橫斷面訊息傳遞）
  → Gating Fusion → 3 個獨立線性頭 → Alpha_5d / 20d / 60d
約 4M 參數；MC-Dropout 30 次採樣輸出每股 Uncertainty`}
        </pre>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.9, marginTop: 12 }}>
          <b>為什麼 Mamba 而非 Transformer</b>：252 步序列下自注意力是 O(L²)、SSM 是 O(L)；「每天對 ~2,000 支股票各跑 252 步序列」的推論負載（本機 RTX 3060）下，線性複雜度是可行性差異而非品味差異。
          與簡單序列模型的實證差距由 baseline 對照回答（見 §5.5——答案誠實地說：沒有差距）。
          <br /><b>為什麼 GATv2</b>：橫斷面關係（產業、集團、供應鏈）是台股 alpha 的結構性來源。實證：GAT 修復前後最終排名 Top50 重疊只有 30/50、Uncertainty −34%——圖的貢獻可度量，不是裝飾。
          <br /><b>為什麼 rank 目標（本頁最重要的論述）</b>：評估指標是 Spearman IC（排名相關），舊版訓練目標卻是 z-score 報酬 MSE——優化與評估錯位。台股 ±10% 漲跌幅下報酬厚尾，MSE 會被暴漲跌股主導。消融證據：
        </div>
        <ScrollTable>
          <thead><tr><th style={TH}>階段</th><th style={TH}>改動</th><th style={TH}>5d IC</th></tr></thead>
          <tbody>
            <tr><td style={TD}>Phase 0</td><td style={TD}>z-score 目標、無 ranking loss</td><td style={{ ...TD, ...MONO }}>0.0434</td></tr>
            <tr><td style={TD}>Phase 1</td><td style={TD}>+ listnet ranking loss</td><td style={{ ...TD, ...MONO }}>0.0487（+12%）</td></tr>
            <tr><td style={TD}>Phase 2</td><td style={TD}>目標改每日橫斷面 rank + listnet</td><td style={{ ...TD, ...MONO }}><b>0.0951（近乎翻倍）</b></td></tr>
          </tbody>
        </ScrollTable>
        <div style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.8, marginTop: 10 }}>
          把「優化什麼」對齊「評估什麼」的貢獻，比任何架構調整都大。多尺度的誠實註記：診斷顯示三個時間分支對 5d 預測冗餘，短線模型已簡化為單尺度——「多尺度自適應融合」的實際結論是<b>它教會我們何時不需要多尺度</b>。
        </div>
      </Panel>

      {/* §5.5 Baseline 對照表 */}
      <Panel icon="⚖️" title="5.5 Baseline 對照：用可解釋性換到多少效益？" sub="同一協定：同 universe / rank label / 切分 / 成本；test 2024-01 ~ 2026-06（580 交易日）">
        <ScrollTable>
          <thead><tr>
            <th style={TH}>5d</th><th style={TH}>Ridge</th><th style={TH}>GBDT</th><th style={TH}>GRU</th><th style={TH}>Mamba v6_short</th>
          </tr></thead>
          <tbody>
            <tr><td style={TD}>輸入</td><td style={TD}>300 維扁平</td><td style={TD}>300 維扁平</td><td style={TD}>(60,59) 序列</td><td style={TD}>(252,59) 序列 + KG</td></tr>
            <tr><td style={TD}>參數量</td><td style={{ ...TD, ...MONO }}>301</td><td style={{ ...TD, ...MONO }}>~10⁴ 葉</td><td style={{ ...TD, ...MONO }}>~49K</td><td style={{ ...TD, ...MONO }}>1.66M</td></tr>
            <tr><td style={TD}>IC 全市場 / 高流動組</td>
              <td style={{ ...TD, ...MONO }}>+0.1015 / +0.0705</td>
              <td style={{ ...TD, ...MONO }}>+0.1098 / +0.0802</td>
              <td style={{ ...TD, ...MONO }}><b>+0.1113 / +0.0867</b></td>
              <td style={{ ...TD, ...MONO }}>+0.0870 / —</td></tr>
            <tr><td style={TD}>組合層年化（Top50 等權）</td>
              <td style={{ ...TD, ...MONO }} className="text-positive">+18.7%</td>
              <td style={{ ...TD, ...MONO }} className="text-positive">+10.8%</td>
              <td style={{ ...TD, ...MONO }} className="text-positive"><b>+22.8%</b></td>
              <td style={TD}>未跑</td></tr>
            <tr><td style={TD}>成本×2 年化</td>
              <td style={{ ...TD, ...MONO }} className="text-negative">−5.4%</td>
              <td style={{ ...TD, ...MONO }} className="text-negative">−13.6%</td>
              <td style={{ ...TD, ...MONO }} className="text-negative">−2.6%</td>
              <td style={TD}>—</td></tr>
            <tr><td style={TD}>可解釋性</td><td style={TD}><b>係數可讀</b></td><td style={TD}>SHAP</td><td style={TD}>無</td><td style={TD}>無</td></tr>
          </tbody>
        </ScrollTable>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.9, marginTop: 12 }}>
          <b>答案：在 5d 這條線上，可解釋性的代價是負的。</b>三條獨立證據收斂：
          ① 模型形式只在 ±0.01 內移動，特徵工程（+0.015）與 rank label（近翻倍）才是報酬率最高的投入；
          ② 序列 vs 扁平幾乎無差（GRU vs GBDT 僅 +0.0015）；
          ③ ~49K 參數 GRU 用同特徵、同 window 勝 1.66M 參數 Mamba +0.024——架構紅利不成立。
          同時保留 GBDT 教訓：<b>IC 排名 ≠ 落袋排名</b>（GBDT IC 較高、組合層反輸 Ridge 7.9pp），訊號層與組合層必須並排看。
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.8, marginTop: 10 }}>
          引用紀律：headline IC 被小型股墊高，一律配分層數字；存活者偏差未量化（四階同資料、相對比較公平）；成本×2 全階轉負是高換手策略固有屬性。
          這不推翻 DL 線的其他價值（60d、Uncertainty 校準、KG 事件傳導未對照），但證明本研究流程能誠實回答「複雜度有沒有換到東西」。
        </div>
      </Panel>

      {/* §6 驗證結果 */}
      <Panel icon="📊" title="6. 驗證結果（照實陳列，含樣本量警語）">
        <ScrollTable>
          <thead><tr><th style={TH}>模型</th><th style={TH}>目標</th><th style={TH}>最佳 val IC</th><th style={TH}>對照（舊 z-score 目標）</th></tr></thead>
          <tbody>
            <tr><td style={TD}>v6_short（單尺度、rank）</td><td style={{ ...TD, ...MONO }}>5d</td><td style={{ ...TD, ...MONO }}><b>0.0951</b></td><td style={{ ...TD, ...MONO }}>0.049</td></tr>
            <tr><td style={TD}>v6_trend（多尺度、rank）</td><td style={{ ...TD, ...MONO }}>20d</td><td style={{ ...TD, ...MONO }}><b>0.0961</b></td><td style={{ ...TD, ...MONO }}>0.051</td></tr>
          </tbody>
        </ScrollTable>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.9, marginTop: 12 }}>
          單一切分：train ≤2023-12-31、val 2024-01 ~ 2026-06（~580 交易日）。線上實測（推論輸出 vs 真實後續報酬）樣本仍淺：
          V6.1 的 5d IC 0.071（24 天）；雙模型 5d IC +0.129、Top50 實現超額 +1.35%/期（<b>僅 4 天，樣本不足勿下結論</b>，隨每日歸檔自動累積，見「模擬機器人」頁的雙模型驗證分頁）。
          Walk-Forward 框架已實作但需 36 次重訓、尚未例行化——此處不放合成數字。扣成本績效見 §5.5 組合層。
        </div>
      </Panel>

      {/* §7 誠實限制 */}
      <Panel icon="🪞" title="7. 誠實限制與弱點" sub="全部有據可查，含未修復項">
        <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.7 }}>
          一個只報喜的系統不可信。以下是這套系統實際踩過的坑與尚未解決的問題（點開看細節）：
        </div>
        {LIMITATIONS.map((item, i) => <LimitCard key={i} item={item} />)}
      </Panel>

    </div>
  );
}
