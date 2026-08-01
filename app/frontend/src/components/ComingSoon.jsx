import React from 'react';

// 待串接後端資料的子頁共用版位
export default function ComingSoon({ icon = '🚧', title, desc, bullets, note }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">{icon} {title}</div>
        <span className="badge badge-neutral">尚未串接後端</span>
      </div>
      <div className="panel-body">
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.75, margin: '0 0 12px' }}>{desc}</p>
        {bullets?.length > 0 && (
          <ul style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.9, paddingLeft: 20, margin: 0 }}>
            {bullets.map((b, i) => <li key={i}>{b}</li>)}
          </ul>
        )}
        {note && (
          <div style={{
            marginTop: 14, fontSize: 12, color: 'var(--text-secondary)',
            padding: '10px 12px', background: 'rgba(88,166,255,0.06)',
            borderLeft: '3px solid var(--accent-blue)', borderRadius: '0 6px 6px 0', lineHeight: 1.7,
          }}>
            {note}
          </div>
        )}
      </div>
    </div>
  );
}
