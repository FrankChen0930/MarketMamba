import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Globe, Target, History, ChartLine, Bot, Microscope, ArrowRight } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { fetchV62Portfolio, fetchV62Performance } from '../api/v62';
import VersionBadge from '../components/VersionBadge';
import { versionOf } from '../versions';

const ICON = { size: 20, strokeWidth: 1.75 };

// ── 三句話講完這個網站在幹嘛 ─────────────────────────────────────────────
const WHAT = [
  {
    title: '每天收盤後自己跑一輪',
    desc: '下載當天的股價、成交量、法人買賣、融資融券、財報，整理成模型看得懂的格式，然後替全台股 2,500 檔各打一個分數。整個過程沒有人介入。',
  },
  {
    title: '分數代表「相對強弱」',
    desc: '不是預測股價會漲到多少，而是「這一檔接下來會比大盤好還是差」。所以大盤大跌的時候，分數高的股票一樣會跌，只是跌得比較少。',
  },
  {
    title: '結論會被記錄下來對帳',
    desc: '每天的持股都會存檔。過一段時間就能回頭比對：當初挑的到底準不準。這比任何回測數字都誠實，因為它沒有事後諸葛的空間。',
  },
];

// ── 三條線。狀態直接標出來，不要讓人點進去才發現是空的 ─────────────────
const LINES = [
  {
    to: '/breadth', icon: <Globe {...ICON} />, state: 'live',
    label: '廣度量化模型', color: 'var(--accent-blue)',
    one: '同時買 50 檔，每 20 個交易日換一次。',
    desc: '目前每天實際在跑的就是這一套。它不押單一檔，靠的是「平均排得比較準」慢慢累積優勢。',
  },
  {
    to: '/conviction', icon: <Target {...ICON} />, state: 'planned',
    label: '高信念量化模型', color: 'var(--positive)',
    one: '只挑 10 到 20 檔，但每一檔都說得出理由。',
    desc: '規則先篩、AI 研究、自己拍板的三段流程。目前只有設計，還沒接上真實資料。',
  },
  {
    to: '/legacy', icon: <History {...ICON} />, state: 'legacy',
    label: '前一版', color: 'var(--ver-legacy)',
    one: '2026 年 8 月之前每天在跑的舊版本。',
    desc: '用四個條件加權挑股票。已經停止更新，留著是為了跟現在這版對照。',
  },
];

// ── 依照「你想拿它做什麼」分流，不是依照功能分類 ─────────────────────
const START_HERE = [
  {
    icon: <Globe {...ICON} />, title: '我只想看今天的結果',
    desc: '直接看模型目前持有哪 50 檔，還有多久會換股。',
    action: '看持股組合', route: '/breadth/portfolio',
    accent: 'rgba(0,212,255,0.3)', bg: 'rgba(0,212,255,0.04)',
  },
  {
    icon: <ChartLine {...ICON} />, title: '我有自己的想法，只想縮小範圍',
    desc: '看全市場今天的分數排行，當成初篩名單再自己判斷。',
    action: '看每日評分', route: '/breadth/predictions',
    accent: 'rgba(168,85,247,0.3)', bg: 'rgba(168,85,247,0.04)',
  },
  {
    icon: <Bot {...ICON} />, title: '我想知道今天市場發生什麼事',
    desc: '每天一篇由 AI 整理的市場摘要，講當天的重點與變化。',
    action: '看 AI 日報', route: '/market',
    accent: 'rgba(255,165,0,0.3)', bg: 'rgba(255,165,0,0.04)',
  },
  {
    icon: <Microscope {...ICON} />, title: '我想看你怎麼做出來的',
    desc: '做過的實驗、失敗的嘗試，以及每個設計決定背後的理由。',
    action: '看研究紀錄', route: '/research',
    accent: 'rgba(0,255,136,0.3)', bg: 'rgba(0,255,136,0.04)',
  },
];

export default function Home() {
  const navigate = useNavigate();
  const { data: pf }   = useApi(fetchV62Portfolio);
  const { data: perf } = useApi(fetchV62Performance);

  const dash = '—';
  const lastDate  = pf?.date ?? dash;
  const holdCount = pf?.holdings?.length != null ? `${pf.holdings.length} 檔` : dash;
  const nextReb   = pf?.days_to_next != null ? `${pf.days_to_next} 個交易日` : dash;

  // 上線後的累積報酬。刻意用「累積」而不是「年化」——
  // 樣本還很小的時候，年化會外推出荒謬的數字，看起來卻很像結論。
  // 年化與誤差棒留在「持股組合 → 前瞻績效」那一頁，那裡有完整的樣本量說明。
  const primary = perf?.models
    ? Object.values(perf.models).find(m => m.tier === 'primary')
    : null;
  const cumText = primary?.cum_return != null
    ? `${primary.cum_return >= 0 ? '+' : ''}${(primary.cum_return * 100).toFixed(1)}%`
    : dash;
  const cumColor = primary?.cum_return == null
    ? 'var(--text-muted)'
    : primary.cum_return >= 0 ? 'var(--positive)' : 'var(--negative)';

  const CHIPS = [
    { label: '資料日期',   value: lastDate,  color: 'var(--accent-blue)' },
    { label: '目前持有',   value: holdCount, color: 'var(--text-primary)' },
    { label: '距下次換股', value: nextReb,   color: 'var(--accent-amber)' },
    {
      label: perf?.n_days ? `上線後累積（${perf.n_days} 天）` : '上線後累積',
      value: cumText, color: cumColor,
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-7)', paddingBottom: 'var(--space-7)' }}>

      {/* ── Hero ── */}
      <div style={{
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)',
        background: 'linear-gradient(135deg, rgba(0,212,255,0.05) 0%, rgba(168,85,247,0.04) 55%, rgba(0,255,136,0.04) 100%)',
        padding: 'var(--space-6) var(--space-6)',
        position: 'relative', overflow: 'hidden',
      }}>
        <div aria-hidden="true" style={{ position: 'absolute', top: -60, right: -60, width: 240, height: 240, borderRadius: '50%', background: 'radial-gradient(circle, rgba(0,212,255,0.09) 0%, transparent 70%)', pointerEvents: 'none' }} />

        <div style={{ position: 'relative' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
            <VersionBadge state="live" />
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>每個交易日晚上自動更新</span>
          </div>

          <h1 style={{
            margin: '0 0 var(--space-3)', fontSize: 34, fontWeight: 800, lineHeight: 1.2,
            background: 'linear-gradient(135deg, #00d4ff 0%, #a855f7 55%, #00ff88 100%)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
          }}>
            MarketMamba
          </h1>

          <p style={{ margin: '0 0 var(--space-5)', fontSize: 15, color: 'var(--text-secondary)', maxWidth: 620, lineHeight: 1.8 }}>
            一套自己寫的台股選股系統。每天收盤後把全市場 2,500 檔股票跑過一遍，
            挑出模型認為接下來會相對強勢的 50 檔，每 20 個交易日換一次。
            <b style={{ color: 'var(--text-primary)' }}>這裡的持股不是我的真實部位</b>，
            是拿來驗證方法、也拿來展示做法的。
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
            {CHIPS.map(s => (
              <div key={s.label} style={{
                display: 'flex', alignItems: 'center', gap: 6, fontSize: 12,
                padding: '5px 12px', borderRadius: 99,
                background: 'var(--bg-panel-2)', border: '1px solid var(--border)',
              }}>
                <span style={{ color: 'var(--text-muted)' }}>{s.label}</span>
                <span className="mono" style={{ fontWeight: 700, color: s.color }}>{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── 這個網站在做什麼 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          它每天在做什麼
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 'var(--space-3)' }}>
          {WHAT.map((w, i) => (
            <div key={w.title} className="panel">
              <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                  <span className="mono" style={{
                    width: 24, height: 24, borderRadius: 6,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 11, fontWeight: 700, color: 'var(--accent-blue)',
                    background: 'rgba(0,212,255,0.12)', border: '1px solid rgba(0,212,255,0.3)',
                  }}>{i + 1}</span>
                  <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{w.title}</span>
                </div>
                <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.75, margin: 0 }}>{w.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── 三條線 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-2)' }}>
          三條線，狀態各不相同
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.8, margin: '0 0 var(--space-4)', maxWidth: 700 }}>
          同一批資料，我用兩種完全不同的方法在處理，加上一個已經退下來的舊版本。
          會分成兩種方法是因為：買 50 檔靠的是統計上的平均優勢，買 5 檔靠的是對那 5 檔的理解——
          這兩件事用同一套工具做，兩邊都會做不好。
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-3)' }}>
          {LINES.map(l => (
            <button
              key={l.to}
              onClick={() => navigate(l.to)}
              style={{
                textAlign: 'left', font: 'inherit', cursor: 'pointer',
                borderRadius: 'var(--radius-md)',
                border: `1px solid color-mix(in srgb, ${l.color} 35%, transparent)`,
                background: `color-mix(in srgb, ${l.color} 4%, var(--bg-panel))`,
                padding: 'var(--space-5)',
                display: 'flex', flexDirection: 'column', gap: 'var(--space-2)',
                transition: 'transform 0.18s, box-shadow 0.18s',
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = `0 8px 24px color-mix(in srgb, ${l.color} 20%, transparent)`; }}
              onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = ''; }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
                <span style={{ color: l.color }} aria-hidden="true">{l.icon}</span>
                <span style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{l.label}</span>
                <VersionBadge state={l.state} style={{ marginLeft: 'auto' }} />
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{l.one}</div>
              <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.7, margin: 0 }}>{l.desc}</p>
              <span style={{ marginTop: 'auto', paddingTop: 'var(--space-2)', fontSize: 12, fontWeight: 600, color: l.color, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                前往 <ArrowRight size={13} strokeWidth={2} aria-hidden="true" />
              </span>
            </button>
          ))}
        </div>
      </section>

      {/* ── 從哪裡開始看 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          從哪裡開始看
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 'var(--space-3)' }}>
          {START_HERE.map(card => (
            <button
              key={card.title}
              onClick={() => navigate(card.route)}
              style={{
                textAlign: 'left', font: 'inherit', cursor: 'pointer',
                borderRadius: 'var(--radius-md)', border: `1px solid ${card.accent}`,
                background: card.bg, padding: 'var(--space-4)',
                display: 'flex', flexDirection: 'column', gap: 'var(--space-2)',
                transition: 'transform 0.18s, box-shadow 0.18s',
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = `0 8px 24px ${card.accent}`; }}
              onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = ''; }}
            >
              <span style={{ color: 'var(--text-secondary)' }} aria-hidden="true">{card.icon}</span>
              <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{card.title}</span>
              <span style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.7 }}>{card.desc}</span>
              <span style={{ marginTop: 'auto', paddingTop: 'var(--space-2)', fontSize: 12, fontWeight: 600, color: 'var(--accent-blue)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                {card.action} <ArrowRight size={13} strokeWidth={2} aria-hidden="true" />
              </span>
            </button>
          ))}
        </div>
      </section>

      {/* ── 這不是投資建議 ── */}
      <div style={{
        padding: 'var(--space-4)',
        background: 'rgba(255,165,0,0.05)',
        borderLeft: '3px solid var(--ver-legacy)',
        borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
      }}>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.85 }}>
          <b style={{ color: 'var(--text-primary)' }}>這是個人專案，不是投資建議。</b>
          {' '}模型在歷史資料上表現好，不代表接下來也會好——市場的規則會變，
          而模型只學過去發生過的事。這裡的每個數字都附了它的前提和限制，
          看數字之前請先看那些說明。{versionOf('live')} 的實戰紀錄從 2026 年 8 月開始累積，
          目前樣本還太小，不足以下任何結論。
        </p>
      </div>
    </div>
  );
}
