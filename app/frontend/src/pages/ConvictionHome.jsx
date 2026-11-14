import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, ChartLine, History, Wallet, ArrowRight, Info } from 'lucide-react';

const ICON = { size: 18, strokeWidth: 1.75 };

const ENGINE_LAYERS = [
  {
    step: '先篩掉大部分', tag: '規則式，成本低', color: 'var(--accent-blue)',
    desc: '用幾條寫得死的條件把 2,500 檔縮到 30 至 50 檔：估值落在什麼位置、法人是不是持續買、近期有沒有財報或法說會。這一層只做粗篩，不排名次。',
  },
  {
    step: '再讓 AI 讀一遍', tag: 'LLM，需控制花費', color: 'var(--accent-amber)',
    desc: '對留下來的每一檔，讓語言模型依序回答四件事：現在的價格算貴還便宜、接下來有什麼事情可能推動股價、法人和市場情緒有沒有互相矛盾、最大的風險是什麼。',
  },
  {
    step: '最後自己拍板', tag: '人工判斷', color: 'var(--positive)',
    desc: '把上一層的內容收成一張卡片：為什麼看好、有多少把握、打算投多少錢，還有最重要的一項——先寫下什麼情況出現就代表自己看錯了。',
  },
];

const SUBPAGES = [
  { to: '/conviction/predictions', icon: <FileText {...ICON} />,  label: '個股研究',
    desc: '每檔一張卡片：看好的理由、可能的催化劑、風險，以及認錯的條件。' },
  { to: '/conviction/backtest',    icon: <ChartLine {...ICON} />, label: '回測結果',
    desc: '哪些部分可以拿歷史資料驗證，哪些只能上線之後往前追蹤。' },
  { to: '/conviction/versions',    icon: <History {...ICON} />,   label: '版本紀錄',
    desc: '篩選條件與 AI 提問流程改過哪些地方。' },
  { to: '/conviction/portfolio',   icon: <Wallet {...ICON} />,    label: '持倉追蹤',
    desc: '個人帳戶的實際部位。這頁主要在 PersonalOS 桌面版使用。' },
];

export default function ConvictionHome() {
  const navigate = useNavigate();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>

      {/* ── 定位 ── */}
      <div className="panel">
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.85, margin: 0 }}>
            這條線要解決的是很實際的問題：<b style={{ color: 'var(--text-primary)' }}>錢就這麼多，分不到 50 檔，
            那就只能在少數幾檔上做對決定。</b>
            所以做法整個反過來——不靠模型排名次，改成用規則先篩掉大部分，
            再讓 AI 把留下來的每一檔研究一輪，最後自己判斷。
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.85, margin: 0 }}>
            為什麼不沿用廣度模型那一套？因為深度學習要靠大量重複出現的模式才學得起來。
            少數幾檔股票的特殊情況，樣本太少，硬拿去訓練只會學到雜訊。
            兩條線用的資料有一部分重疊，但方法刻意做得不一樣。
          </p>
        </div>
      </div>

      {/* ── 目前狀態 ── */}
      <div style={{
        display: 'flex', gap: 'var(--space-3)', alignItems: 'flex-start',
        padding: 'var(--space-4)',
        background: 'rgba(139,148,158,0.06)',
        borderLeft: '3px solid var(--ver-planned)',
        borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
      }}>
        <Info size={16} strokeWidth={2} color="var(--ver-planned)" style={{ flexShrink: 0, marginTop: 2 }} aria-hidden="true" />
        <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
          <b style={{ color: 'var(--text-primary)' }}>這條線目前只有設計，還沒有資料。</b>
          {' '}底下的子頁會告訴你那一頁打算放什麼，但還沒接上真實內容。
          先做完的是<a href="/breadth" style={{ color: 'var(--accent-blue)' }}>廣度模型</a>那一條。
        </p>
      </div>

      {/* ── 三層流程 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          打算怎麼做：三個階段
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 'var(--space-3)' }}>
          {ENGINE_LAYERS.map((l, i) => (
            <div key={l.step} className="panel">
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
                  <span className="mono" style={{
                    width: 24, height: 24, borderRadius: 6,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, color: l.color,
                    background: `color-mix(in srgb, ${l.color} 15%, transparent)`,
                    border: `1px solid ${l.color}`,
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
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          子頁導覽
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 'var(--space-3)' }}>
          {SUBPAGES.map(s => (
            <button
              key={s.to}
              onClick={() => navigate(s.to)}
              style={{
                textAlign: 'left', font: 'inherit', cursor: 'pointer',
                borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)',
                background: 'var(--bg-panel)', padding: 'var(--space-4)',
                transition: 'border-color 0.15s, transform 0.15s',
                display: 'flex', flexDirection: 'column', gap: 'var(--space-2)',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--positive)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.transform = ''; }}
            >
              <span style={{ color: 'var(--positive)' }} aria-hidden="true">{s.icon}</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{s.label}</span>
              <span style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.65 }}>{s.desc}</span>
              <span style={{ marginTop: 'auto', paddingTop: 'var(--space-2)', fontSize: 12, fontWeight: 600, color: 'var(--positive)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                前往 <ArrowRight size={13} strokeWidth={2} aria-hidden="true" />
              </span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
