import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { fetchScannerSignals } from '../api/signals';
import { fetchMarket } from '../api/market';
import { fetchIcAnalysis } from '../api/sim';

// ── Feature cards config ──────────────────────────────────────────────────────
const FEATURES = [
  {
    to: '/scanner', icon: '🎯', label: '交易訊號',
    desc: '今日 AI 掃描結果：買入推薦、退場警告、入場評分明細',
    accent: 'rgba(0,255,136,0.25)',
  },
  {
    to: '/', icon: '📈', label: '每日排名',
    desc: '全市場 Alpha 排行榜，5d / 20d / 60d 三個預測視角',
    accent: 'rgba(0,212,255,0.25)',
  },
  {
    to: '/quant', icon: '🔬', label: '量化分析',
    desc: '技術型態辨識、產業熱力圖、多因子篩選',
    accent: 'rgba(168,85,247,0.25)',
  },
  {
    to: '/market', icon: '🤖', label: 'AI 日報',
    desc: 'Claude LLM 生成的每日市場解讀與宏觀摘要',
    accent: 'rgba(245,158,11,0.25)',
  },
  {
    to: '/sim', icon: '🎮', label: '模擬機器人',
    desc: '回測結果、Alpha 機器人持倉、IC 因子分析',
    accent: 'rgba(99,102,241,0.25)',
  },
  {
    to: '/portfolio', icon: '💼', label: '持倉追蹤',
    desc: '個人投資組合管理、損益追蹤、退場提醒',
    accent: 'rgba(236,72,153,0.25)',
  },
];

// ── System spec items ─────────────────────────────────────────────────────────
const SPECS = [
  { label: '模型架構', value: 'Mamba SSM + GATv2 知識圖譜' },
  { label: '模型參數', value: '~4M' },
  { label: '訓練硬體', value: 'Google Colab A100' },
  { label: '推論硬體', value: 'RTX 3060（本機 WSL2）' },
  { label: '預測目標', value: 'Alpha_5d / Alpha_20d / Alpha_60d' },
  { label: '不確定性', value: 'MC-Dropout N=30' },
  { label: '訓練資料', value: '2005 年至今，~5,000 個交易日' },
  { label: '特徵維度', value: '56 因子（4 大因子組）' },
];

const PIPELINE = [
  { step: '01', title: '資料擷取', desc: 'FinMind + yfinance 全市場 2,515 支股票日資料，含機構法人、融資融券、財報、總經指標' },
  { step: '02', title: '特徵工程', desc: '56 維因子：價格動能、機構資金流、基本面、總體環境，FactorGroupedEmbedding 分組投影' },
  { step: '03', title: 'Mamba 推論', desc: 'MultiScaleMambaEncoder（Short 2層 / Mid 3層 / Long 3層）並行，自適應 Scale Gate 融合' },
  { step: '04', title: 'GAT 圖增強', desc: 'GATv2 搭配 640K 條邊的知識圖譜，捕捉產業鏈與供應鏈關聯' },
  { step: '05', title: '訊號掃描', desc: '4 條件加權評分（排名穩定 30 + 信心 25 + 機構 25 + 相對低點 20）+ 型態加分（最高 +50）' },
  { step: '06', title: 'GitHub Push', desc: '結果自動 push 到 GitHub，Render 後端從 raw URL 載入，Vercel 前端即時呈現' },
];

// ── Component ─────────────────────────────────────────────────────────────────
export default function Home() {
  const navigate = useNavigate();
  const { data: scanner } = useApi(fetchScannerSignals);
  const { data: market  } = useApi(fetchMarket);
  const { data: ic      } = useApi(fetchIcAnalysis);

  const buyCount   = scanner?.buy_signals?.length ?? '—';
  const lastDate   = scanner?.date ?? market?.last_run ?? '—';
  const ic5d       = ic?.ic_5d?.mean != null  ? (ic.ic_5d.mean).toFixed(3)  : '—';
  const icir5d     = ic?.ic_5d?.icir != null  ? (ic.ic_5d.icir).toFixed(2)  : '—';
  const ic20d      = ic?.ic_20d?.mean != null ? (ic.ic_20d.mean).toFixed(3) : '—';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 48, paddingBottom: 48 }}>

      {/* ── Hero ── */}
      <div style={{
        borderRadius: 'var(--radius)',
        border: '1px solid var(--border)',
        background: 'linear-gradient(135deg, rgba(0,212,255,0.04) 0%, rgba(168,85,247,0.04) 50%, rgba(0,255,136,0.04) 100%)',
        padding: '40px 36px',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* decorative glow */}
        <div style={{ position: 'absolute', top: -60, right: -60, width: 240, height: 240, borderRadius: '50%', background: 'radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%)', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', bottom: -40, left: '30%', width: 180, height: 180, borderRadius: '50%', background: 'radial-gradient(circle, rgba(0,255,136,0.06) 0%, transparent 70%)', pointerEvents: 'none' }} />

        <div style={{ position: 'relative' }}>
          <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', color: 'var(--accent-blue)', textTransform: 'uppercase', marginBottom: 8 }}>
            Personal Quantitative Investment System
          </div>
          <h1 style={{ margin: '0 0 8px', fontSize: 36, fontWeight: 800, background: 'linear-gradient(135deg, #00d4ff 0%, #a855f7 50%, #00ff88 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text', lineHeight: 1.2 }}>
            MarketMamba V6
          </h1>
          <p style={{ margin: '0 0 28px', fontSize: 15, color: 'var(--text-secondary)', maxWidth: 560, lineHeight: 1.7 }}>
            以深度學習驅動的台股量化投資自動化系統。每日收盤後對全市場 2,515 支股票執行 Mamba+GATv2 推論，輸出 Alpha 訊號排名與進退場建議。
          </p>

          {/* live stat chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {[
              { label: '推論日期', value: lastDate, color: 'var(--accent-blue)' },
              { label: '今日買入推薦', value: `${buyCount} 支`, color: 'var(--positive)' },
              { label: 'IC_5d', value: ic5d, color: ic5d !== '—' && parseFloat(ic5d) > 0 ? 'var(--positive)' : 'var(--text-muted)' },
              { label: 'ICIR_5d', value: icir5d, color: 'var(--accent-blue)' },
              { label: 'IC_20d', value: ic20d, color: ic20d !== '—' && parseFloat(ic20d) > 0 ? 'var(--positive)' : 'var(--text-muted)' },
            ].map(s => (
              <div key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, padding: '5px 12px', borderRadius: 99, background: 'var(--bg-panel-2)', border: '1px solid var(--border)' }}>
                <span style={{ color: 'var(--text-muted)' }}>{s.label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: s.color }}>{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── 使用指南 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16 }}>使用指南</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
          {[
            {
              icon: '🎯', title: '直接看結論',
              desc: '想知道今天可以買哪幾支，不需要自己判斷市場。',
              tags: ['適合初學者', '有明確進場建議'],
              action: '去交易訊號', route: '/scanner',
              accent: 'rgba(0,255,136,0.3)', bg: 'rgba(0,255,136,0.04)',
            },
            {
              icon: '📊', title: '縮小選股範圍',
              desc: '有自己的判斷，只是不想從 2,500 支慢慢挑，需要初篩名單。',
              tags: ['有基本市場概念', '搭配自己的判斷使用'],
              action: '看每日排名', route: '/',
              accent: 'rgba(0,212,255,0.3)', bg: 'rgba(0,212,255,0.04)',
            },
            {
              icon: '🔬', title: '深入研究',
              desc: '想了解 AI 的推論邏輯、市場報告或型態技術分析。',
              tags: ['技術分析愛好者', '想理解模型行為'],
              action: '探索量化分析', route: '/quant',
              accent: 'rgba(168,85,247,0.3)', bg: 'rgba(168,85,247,0.04)',
            },
          ].map(card => (
            <div key={card.title}
              onClick={() => navigate(card.route)}
              style={{ borderRadius: 'var(--radius)', border: `1px solid ${card.accent}`, background: card.bg, padding: '18px 20px', cursor: 'pointer', transition: 'all 0.18s', display: 'flex', flexDirection: 'column', gap: 8 }}
              onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = `0 8px 24px ${card.accent}`; }}
              onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = ''; }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 22 }}>{card.icon}</span>
                <span style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>{card.title}</span>
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: 0, lineHeight: 1.65 }}>{card.desc}</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {card.tags.map(t => <span key={t} style={{ fontSize: 10, color: 'var(--text-muted)', background: 'var(--bg-panel-2)', padding: '2px 8px', borderRadius: 99 }}>{t}</span>)}
              </div>
              <div style={{ marginTop: 'auto', paddingTop: 10, fontSize: 12, fontWeight: 600, color: 'var(--accent-blue)' }}>{card.action} →</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── 功能導覽 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16 }}>功能導覽</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
          {FEATURES.map(f => (
            <div key={f.to}
              onClick={() => navigate(f.to)}
              style={{ borderRadius: 'var(--radius-sm)', border: `1px solid ${f.accent}`, background: 'var(--bg-panel)', padding: '14px 16px', cursor: 'pointer', transition: 'all 0.15s' }}
              onMouseEnter={e => { e.currentTarget.style.background = `color-mix(in srgb, ${f.accent} 8%, var(--bg-panel))`; e.currentTarget.style.transform = 'translateY(-1px)'; }}
              onMouseLeave={e => { e.currentTarget.style.background = 'var(--bg-panel)'; e.currentTarget.style.transform = ''; }}
            >
              <div style={{ fontSize: 18, marginBottom: 6 }}>{f.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>{f.label}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── 系統介紹 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 16 }}>系統介紹</div>

        {/* Pipeline */}
        <div className="panel" style={{ marginBottom: 16 }}>
          <div className="panel-header"><div className="panel-title">📦 推論流程</div></div>
          <div className="panel-body">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
              {PIPELINE.map(p => (
                <div key={p.step} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <span style={{ flexShrink: 0, width: 28, height: 28, borderRadius: 6, background: 'rgba(0,212,255,0.12)', border: '1px solid rgba(0,212,255,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>{p.step}</span>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 2 }}>{p.title}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>{p.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Model Architecture */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="panel">
            <div className="panel-header"><div className="panel-title">🧠 模型架構</div></div>
            <div className="panel-body">
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)', lineHeight: 2, background: 'var(--bg-panel-2)', borderRadius: 'var(--radius-sm)', padding: '12px 14px' }}>
                {[
                  'Input  (N × 252 × 56)',
                  '  ↓ FactorGroupedEmbedding → d=256',
                  '  ↓ MultiScaleMambaEncoder',
                  '     ├─ Short branch  (2L, last 20)',
                  '     ├─ Mid   branch  (3L, last 60)',
                  '     └─ Long  branch  (3L, full 252)',
                  '  ↓ GATv2  (~640K edges)',
                  '  ↓ Gating Fusion',
                  '  ↓ MultiHorizonHead',
                  'Output [α5d, α20d, α60d]',
                ].map((line, i) => (
                  <div key={i} style={{ color: line.startsWith('Input') || line.startsWith('Output') ? 'var(--accent-blue)' : line.includes('branch') ? '#a78bfa' : 'var(--text-secondary)' }}>{line}</div>
                ))}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header"><div className="panel-title">⚙️ 技術規格</div></div>
            <div className="panel-body">
              {SPECS.map(s => (
                <div key={s.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
                  <span style={{ color: 'var(--text-muted)' }}>{s.label}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{s.value}</span>
                </div>
              ))}
              <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {['Python', 'PyTorch', 'Mamba SSM', 'FastAPI', 'React', 'Vercel', 'Render', 'WSL2'].map(t => (
                  <span key={t} style={{ fontSize: 10, padding: '3px 8px', borderRadius: 99, background: 'rgba(0,212,255,0.1)', color: 'var(--accent-blue)', fontWeight: 600 }}>{t}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
