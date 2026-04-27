import React from 'react';

const SECTOR_ORDER = [
  { key: '半導體',     label: '半導體',   baseScore: 72 },
  { key: '電子製造',   label: '電子製造', baseScore: 61 },
  { key: '電子零組件', label: '電子零組', baseScore: 55 },
  { key: '金融',       label: '金融',     baseScore: 44 },
  { key: '化工',       label: '化工',     baseScore: 38 },
  { key: '航運',       label: '航運',     baseScore: 28 },
  { key: '鋼鐵',       label: '鋼鐵',     baseScore: 21 },
  { key: '其他',       label: '其他',     baseScore: 33 },
];

function scoreToColor(score) {
  if (score >= 65) return { bg: 'rgba(0,255,136,0.18)', border: 'rgba(0,255,136,0.4)', text: 'var(--positive)' };
  if (score >= 45) return { bg: 'rgba(255,165,0,0.12)', border: 'rgba(255,165,0,0.3)', text: 'var(--accent-amber)' };
  return { bg: 'rgba(255,71,87,0.12)', border: 'rgba(255,71,87,0.3)', text: 'var(--negative)' };
}

/**
 * SectorHeatmap — grid of coloured tiles for each industry sector.
 * `signals` is optional: if provided, scores are derived from signal counts.
 */
export default function SectorHeatmap({ signals }) {
  // Derive scores from signals if available, else use baseline
  const sectors = SECTOR_ORDER.map((s) => {
    let score = s.baseScore;
    if (signals?.length) {
      const inSector = signals.filter((sig) => sig.sector === s.key);
      if (inSector.length > 0) {
        const avgAlpha = inSector.reduce((a, b) => a + b.alpha_20d, 0) / inSector.length;
        score = Math.min(100, Math.max(0, Math.round(50 + avgAlpha * 300)));
      }
    }
    return { ...s, score };
  });

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
      {sectors.map((s) => {
        const { bg, border, text } = scoreToColor(s.score);
        return (
          <div
            key={s.key}
            style={{
              background: bg,
              border: `1px solid ${border}`,
              borderRadius: 8,
              padding: '10px 8px',
              textAlign: 'center',
              transition: 'transform 0.15s, box-shadow 0.15s',
              cursor: 'default',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.04)';
              e.currentTarget.style.boxShadow = `0 0 12px ${border}`;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4 }}>{s.label}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 16, fontWeight: 700, color: text }}>
              {s.score}
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 2 }}>
              {s.score >= 65 ? '強多' : s.score >= 45 ? '中性' : '偏空'}
            </div>
          </div>
        );
      })}
    </div>
  );
}
