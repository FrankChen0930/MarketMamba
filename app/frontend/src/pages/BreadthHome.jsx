import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChartLine, Package, History, ArrowRight } from 'lucide-react';

const ICON = { size: 18, strokeWidth: 1.75 };

const SUBPAGES = [
  {
    to: '/breadth/predictions', icon: <ChartLine {...ICON} />, label: '每日評分',
    desc: '全市場今天的分數排行，以及各產業的強弱分布。',
  },
  {
    to: '/breadth/portfolio', icon: <Package {...ICON} />, label: '持股組合（20 天換一次）',
    desc: '目前這一輪持有哪 50 檔，加上上線之後累積到現在的實際報酬。',
  },
  {
    to: '/breadth/versions', icon: <History {...ICON} />, label: '版本紀錄',
    desc: '模型改過哪些地方、每次改動量到多少差異。',
  },
];

// 三步驟：資料 → 分數 → 買賣。刻意不講模型內部細節，那在「研究紀錄」有完整版
const STEPS = [
  {
    n: '1', title: '看過去一年的資料',
    desc: '每一檔股票取最近 252 個交易日（大約一年）的股價、成交量、法人買賣、融資融券、財報等 59 項數字。',
  },
  {
    n: '2', title: '模型給一個分數',
    desc: '分數代表「模型認為這檔接下來會比大盤好多少」。它同時會參考同產業、同集團、有供應鏈往來的其他股票，不是每檔各看各的。',
  },
  {
    n: '3', title: '買前 50 名，20 天換一次',
    desc: '分數最高的 50 檔平均分配資金。中間 19 天不動，第 20 天才重新排名換股——換得少，手續費和買賣價差才吃不掉報酬。',
  },
];

export default function BreadthHome() {
  const navigate = useNavigate();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>

      {/* ── 這套模型在做什麼 ── */}
      <div className="panel">
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.85, margin: 0 }}>
            這套的想法是：<b style={{ color: 'var(--text-primary)' }}>不去賭少數幾檔會不會漲，而是同時押 50 檔，
            靠著「平均而言排得比較準」慢慢累積優勢。</b>
            單看任何一檔都可能看走眼，但檔數夠多的時候，一點點的排序能力就會顯現出來。
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.85, margin: 0 }}>
            這也是這個專案裡唯一能認真驗證的一條線——因為有 2,500 檔 × 12 年的資料，
            可以真的把模型放回過去跑一遍，看它當年會怎麼選。
            對照組是<b>「無腦買下全市場、每檔一樣多」</b>，贏不過它就沒有意義。
          </p>
        </div>
      </div>

      {/* ── 三個步驟 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          每天做的三件事
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 'var(--space-3)' }}>
          {STEPS.map(s => (
            <div key={s.n} className="panel">
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                  <span className="mono" style={{
                    width: 24, height: 24, borderRadius: 6,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, color: 'var(--accent-blue)',
                    background: 'rgba(0,212,255,0.12)', border: '1px solid rgba(0,212,255,0.3)',
                  }}>{s.n}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{s.title}</span>
                </div>
                <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.75, margin: 0 }}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── 誠實聲明 ── */}
      <div style={{
        padding: 'var(--space-4)',
        background: 'rgba(0,212,255,0.04)',
        borderLeft: '3px solid var(--accent-blue)',
        borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
      }}>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.85 }}>
          <b style={{ color: 'var(--text-primary)' }}>這裡的持股不是我的真實部位。</b>
          {' '}我的資金分不到 50 檔，這條線是拿來驗證方法和展示做法的。
          而且回測跑得再好，也只是「照過去的資料重來一次」——真正的答案要靠上線之後一天一天累積，
          那份紀錄就放在「持股組合」頁裡。
        </p>
      </div>

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
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent-blue)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.transform = ''; }}
            >
              <span style={{ color: 'var(--accent-blue)' }} aria-hidden="true">{s.icon}</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{s.label}</span>
              <span style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.65 }}>{s.desc}</span>
              <span style={{ marginTop: 'auto', paddingTop: 'var(--space-2)', fontSize: 12, fontWeight: 600, color: 'var(--accent-blue)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                前往 <ArrowRight size={13} strokeWidth={2} aria-hidden="true" />
              </span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
