import React from 'react';
import { useNavigate } from 'react-router-dom';

const ENGINE_LAYERS = [
  {
    step: '篩選層', tag: '量化 · 低成本', color: 'var(--accent-blue)',
    desc: '從全市場用規則式篩選縮小到 30–50 檔候選：估值分位數、法人買賣超持續性、事件行事曆（財報 / 法說會 / 除權息）。DL Alpha 在此層只做輔助訊號或尾部風險標記，不參與排序。',
  },
  {
    step: '研究層', tag: 'LLM · 成本控制', color: 'var(--accent-amber)',
    desc: '單一結構化 prompt 依序輸出估值判讀、催化劑、籌碼與情緒交叉檢查、風險檢查四段論述。MVP 先手動觸發（使用者看到候選池變化才按「重新評估」），自動判斷觸發是後續加強項目。',
  },
  {
    step: '整合層', tag: '人工 + PM 角色', color: 'var(--positive)',
    desc: '把研究層輸出收斂成一張 thesis 卡片：核心押注邏輯、信念分級（S/A/B）、部位大小建議、證偽條件——先寫下什麼情況代表這個 thesis 錯了，逆風時才有依據。',
  },
];

const SUBPAGES = [
  { to: '/conviction/signals',     icon: '📋', label: '每日訊號',     desc: '信念分級（S/A/B）與觸發事件標記，沿用進場評分 / Trailing Stop 機制' },
  { to: '/conviction/predictions', icon: '🗂️', label: '模型預測結果', desc: 'Thesis 卡片牆：每檔一張卡片，含催化劑、估值判讀、風險檢查、證偽條件' },
  { to: '/conviction/backtest',    icon: '📉', label: '回測結果',     desc: '可嚴謹回測的事件驅動子引擎，以及無法回測、只能前瞻追蹤的部分' },
  { to: '/conviction/versions',    icon: '🕰️', label: '版本紀錄',     desc: '篩選規則變更 + Agent pipeline 演進（prompt 版本、模型分層策略）' },
  { to: '/conviction/portfolio',   icon: '💼', label: '持倉追蹤',     desc: '對應真實永豐證券帳戶的個人實盤部位' },
];

export default function ConvictionHome() {
  const navigate = useNavigate();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>

      {/* ── 定位敘事 ── */}
      <div className="panel">
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.85, margin: 0 }}>
            這條線解決的問題：資金有限，無法分散到全市場，必須在少數標的上做對決定。
            <b style={{ color: 'var(--text-primary)' }}>DL Alpha 不是排序主力，只在篩選層當輔助訊號／風險檢查</b>——核心判斷來自量化篩選、
            LLM 研究整理、以及人工最終定奪。
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.85, margin: 0 }}>
            誠實聲明：這條線不假裝有金融背景撐起的深度基本面分析，能做到的是把研究流程結構化、可審核、可累積經驗——
            這本身就是 AI 工程能力的展示，不是財經專業的展示。與廣度模型資料源部分重疊，但方法論刻意不同。
          </p>
        </div>
      </div>

      {/* ── 三層引擎 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 14 }}>
          三層引擎結構
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
          {ENGINE_LAYERS.map((l, i) => (
            <div key={l.step} className="panel" style={{ position: 'relative' }}>
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    width: 24, height: 24, borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, fontFamily: 'var(--font-mono)', color: l.color,
                    background: `color-mix(in srgb, ${l.color} 15%, transparent)`, border: `1px solid ${l.color}`,
                  }}>{i + 1}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{l.step}</span>
                  <span style={{ fontSize: 10, color: l.color, marginLeft: 'auto' }}>{l.tag}</span>
                </div>
                <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.75, margin: 0 }}>{l.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── 子頁導覽 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 14 }}>
          子頁導覽
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 10 }}>
          {SUBPAGES.map(s => (
            <div key={s.to} onClick={() => navigate(s.to)}
              style={{ borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', background: 'var(--bg-panel)', padding: '14px 16px', cursor: 'pointer', transition: 'all 0.15s' }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent-blue)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.transform = ''; }}
            >
              <div style={{ fontSize: 18, marginBottom: 6 }}>{s.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>{s.label}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
