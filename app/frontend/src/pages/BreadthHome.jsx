import React from 'react';
import { useNavigate } from 'react-router-dom';

const SUBPAGES = [
  { to: '/breadth/predictions', icon: '📈', label: '模型預測結果', desc: '全市場 Alpha 排名、產業強弱熱力圖、知識圖譜視覺化' },
  { to: '/breadth/backtest',    icon: '📊', label: '回測結果',     desc: 'IC/ICIR 分佈、因子歸因、換手率與容量分析、累積超額報酬' },
  { to: '/breadth/versions',    icon: '🕰️', label: '版本紀錄',     desc: 'Model Card：V6.0 → V6.3 的訓練狀態與演進' },
  { to: '/breadth/compare',     icon: '🔀', label: '雙模型比較',   desc: '短線 v6_short vs 趨勢 v6_trend，以及未來版本 A/B' },
];

export default function BreadthHome() {
  const navigate = useNavigate();
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>

      <div className="panel">
        <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.85, margin: 0 }}>
            這是機構等級的量化做法：<b style={{ color: 'var(--text-primary)' }}>DL 在這裡是決策核心</b>，
            因為訓練訊號來自 ~2,888 檔 × 12 年的海量橫斷面樣本，撐得起深度網路。
            架構亮點為 Mamba SSM + GATv2 知識圖譜——多尺度序列編碼搭配供應鏈/產業知識圖譜增強。
          </p>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.85, margin: 0 }}>
            此模型用於驗證方法論與市場實測，<b>非個人資金實際操作對象</b>（資金規模不足以分散到全市場）。
            真實部位由 <span style={{ color: 'var(--accent-blue)' }}>高信念模型</span> 負責。
          </p>
        </div>
      </div>

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
