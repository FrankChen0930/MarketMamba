import React, { useEffect } from 'react';

const CONFIDENCE_COLOR = {
  '高信心': 'var(--positive)',
  '中信心': 'var(--accent-amber)',
  '低信心': 'var(--text-muted)',
};

function AlphaHorizonBar({ label, value, max = 0.3 }) {
  const pct = Math.min(Math.abs(value) / max * 100, 100);
  const color = value >= 0 ? 'var(--positive)' : 'var(--negative)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
      <span style={{ width: 28, fontSize: 11, color: 'var(--text-muted)', flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 8, background: 'var(--bg-hover)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          width: `${pct}%`, height: '100%', background: color,
          borderRadius: 4, boxShadow: `0 0 6px ${color}`,
          transition: 'width 0.6s ease',
        }} />
      </div>
      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color, minWidth: 60, textAlign: 'right' }}>
        {value >= 0 ? '+' : ''}{(value * 100).toFixed(2)}%
      </span>
    </div>
  );
}

export default function StockModal({ stock, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  if (!stock) return null;

  const confColor = CONFIDENCE_COLOR[stock.confidence] || 'var(--text-muted)';

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.65)',
        backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="panel"
        style={{
          width: 420, maxWidth: '90vw',
          border: '1px solid var(--border-bright)',
          boxShadow: 'var(--shadow-glow-blue)',
          animation: 'fadeInUp 0.2s ease forwards',
        }}
      >
        {/* Header */}
        <div className="panel-header" style={{ justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>
              {stock.name}
              <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8, fontFamily: 'var(--font-mono)' }}>
                {stock.stock_id}
              </span>
            </div>
            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              <span className="badge badge-neutral" style={{ fontSize: 10 }}>{stock.sector}</span>
              <span className={`badge ${stock.signal === 'BUY' ? 'badge-positive' : stock.signal === 'SELL' ? 'badge-negative' : 'badge-neutral'}`}
                style={{ fontSize: 10 }}>
                {stock.signal}
              </span>
              <span style={{ fontSize: 11, color: confColor, fontFamily: 'var(--font-mono)' }}>
                ● {stock.confidence}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="btn btn-ghost"
            style={{ padding: '4px 10px', fontSize: 16, lineHeight: 1 }}
          >✕</button>
        </div>

        {/* Alpha Bars */}
        <div className="panel-body">
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
            預測 Alpha（截面 z-score 超額報酬）
          </div>
          <AlphaHorizonBar label="5d"  value={stock.alpha_5d} />
          <AlphaHorizonBar label="20d" value={stock.alpha_20d} />
          <AlphaHorizonBar label="60d" value={stock.alpha_60d} />

          <div className="divider" />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 4 }}>
            {[
              { label: '不確定度', value: `±${(stock.uncertainty * 100).toFixed(1)}%`, color: 'var(--accent-amber)' },
              { label: '量比', value: `${stock.vol_ratio?.toFixed(2) ?? '—'}x`, color: stock.vol_ratio > 1 ? 'var(--positive)' : 'var(--text-secondary)' },
              { label: 'Kelly 倉位', value: stock.suggested_weight != null ? `${(stock.suggested_weight * 100).toFixed(1)}%` : '—', color: 'var(--accent-blue)' },
            ].map((m) => (
              <div key={m.label} style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>
                  {m.label}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700, color: m.color }}>
                  {m.value}
                </div>
              </div>
            ))}
          </div>

          <div className="divider" />
          <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
            Rank #{stock.rank} · 數據來源：MarketMamba V6 推論
          </div>
        </div>
      </div>
    </div>
  );
}
