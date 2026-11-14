import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ClipboardList, Gauge, GitCompare, ArrowRight } from 'lucide-react';

const ICON = { size: 18, strokeWidth: 1.75 };

const SUBPAGES = [
  {
    to: '/legacy/signals', icon: <ClipboardList {...ICON} />, label: '每日訊號',
    desc: '每天收盤後挑出幾檔值得買的股票，並且說明是哪幾個條件成立才挑它。',
  },
  {
    to: '/legacy/sim', icon: <Gauge {...ICON} />, label: '模擬操作',
    desc: '假設完全照這套規則買賣，帳面上會變成什麼樣子。沒有真的下單。',
  },
  {
    to: '/legacy/dual', icon: <GitCompare {...ICON} />, label: '雙模型比較',
    desc: '同一天，短線版和趨勢版各自挑出來的股票差多少。用來看兩種目標的取捨。',
  },
];

// 「這一版怎麼決定要不要買」的四個條件，照原本的配分列出來
const RULES = [
  { name: '這幾天一直排在前面', weight: 30, plain: '連續兩天進前 10 名，或連續三天進前 50 名' },
  { name: '模型自己有把握',     weight: 25, plain: '同一檔股票重算三十次，答案都差不多' },
  { name: '法人連續買',         weight: 25, plain: '外資或投信連續兩天買超' },
  { name: '價格還沒漲上去',     weight: 20, plain: 'RSI 低於 40，或股價還在 20 日均線下方' },
];

export default function LegacyHome() {
  const navigate = useNavigate();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>

      {/* ── 這一版在做什麼 ── */}
      <div className="panel">
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.85, margin: 0 }}>
            V6.1 每天收盤後把全台股大約 2,500 檔股票跑過一遍，替每一檔打一個分數，
            再用四個條件加權挑出「今天可以考慮買」的名單，同時盯著手上的股票該不該賣。
            這是這個專案第一個真的每天在跑、而且看得到結果的版本。
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.85, margin: 0 }}>
            那四個條件是我自己憑經驗訂的，配分也是手調的。它跑得起來、看起來也合理，
            但從來沒有跟其他做法認真比過——所以無法回答「這樣挑到底比隨便挑好多少」。
            V6.2 換掉的就是這一塊。
          </p>
        </div>
      </div>

      {/* ── 進場規則 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          它怎麼決定要不要買
        </div>
        <div className="panel">
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8, margin: 0 }}>
              四個條件各有配分，加起來滿分 100。滿 70 分才會出現在買進名單上
              （大盤走弱時門檻提高到 90 分）。
            </p>
            {RULES.map(r => (
              <div key={r.name} style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                <span className="mono" style={{
                  flexShrink: 0, width: 44, textAlign: 'right',
                  fontSize: 13, fontWeight: 700, color: 'var(--accent-blue)',
                }}>{r.weight} 分</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{r.name}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>{r.plain}</div>
                </div>
                {/* 配分長條：純視覺輔助，數字本身左邊已經寫出來了 */}
                <div aria-hidden="true" style={{ flexShrink: 0, width: 90, height: 4, borderRadius: 2, background: 'var(--bg-panel-2)' }}>
                  <div style={{ width: `${r.weight}%`, height: '100%', borderRadius: 2, background: 'var(--accent-blue)' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── 為什麼換掉 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          為什麼換成 V6.2
        </div>
        <div className="panel">
          <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.85, margin: 0 }}>
              V6.1 的問題不在模型，在挑股票那一層。四個條件是拍腦袋訂的，
              而且它幾乎每天都在換股——手續費和買賣價差會把賺到的吃掉一大塊。
              我後來實際算過：在這種換股頻率下，光交易成本一年就要 20% 上下。
            </p>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.85, margin: 0 }}>
              V6.2 把「挑股票」改成一組講得清楚、也回測得動的規則：
              直接取模型分數前 50 名等權買進，每 20 個交易日才換一次股。
              條件變少了，但每一條都經過比較，知道拿掉它會差多少。
            </p>
            <p style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.8, margin: 0 }}>
              兩版的模型骨架其實是同一個（Mamba SSM 加上一層知識圖譜），
              差別在餵給它的特徵欄位數量，以及拿到分數之後怎麼用。
            </p>
          </div>
        </div>
      </section>

      {/* ── 子頁導覽 ── */}
      <section>
        <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.08em', marginBottom: 'var(--space-4)' }}>
          這一區還看得到什麼
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
              onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--ver-legacy)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.transform = ''; }}
            >
              <span style={{ color: 'var(--ver-legacy)' }} aria-hidden="true">{s.icon}</span>
              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{s.label}</span>
              <span style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.65 }}>{s.desc}</span>
              <span style={{ marginTop: 'auto', paddingTop: 'var(--space-2)', fontSize: 12, fontWeight: 600, color: 'var(--ver-legacy)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                前往 <ArrowRight size={13} strokeWidth={2} aria-hidden="true" />
              </span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
