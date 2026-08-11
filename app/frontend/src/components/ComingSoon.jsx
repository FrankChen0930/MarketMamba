import React from 'react';
import { Hammer } from 'lucide-react';

/**
 * 還沒接上真實資料的子頁共用版位。
 *
 * icon 收 React 元素（lucide 圖示）。不給就用預設的槌子。
 * badge 文字刻意寫「還沒接上資料」而不是「尚未串接後端」——
 * 後者是內部用語，看的人不需要知道前後端怎麼分工。
 */
export default function ComingSoon({ icon, title, desc, bullets, note }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <span style={{ color: 'var(--ver-planned)', display: 'inline-flex' }} aria-hidden="true">
            {icon || <Hammer size={16} strokeWidth={1.75} />}
          </span>
          {title}
        </div>
        <span className="badge badge-planned">還沒接上資料</span>
      </div>
      <div className="panel-body">
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8, margin: '0 0 var(--space-3)' }}>{desc}</p>
        {bullets?.length > 0 && (
          <ul style={{ fontSize: 12.5, color: 'var(--text-muted)', lineHeight: 1.9, paddingLeft: 20, margin: 0 }}>
            {bullets.map((b, i) => <li key={i}>{b}</li>)}
          </ul>
        )}
        {note && (
          <div style={{
            marginTop: 'var(--space-4)', fontSize: 12, color: 'var(--text-secondary)',
            padding: 'var(--space-3)', background: 'rgba(0,212,255,0.05)',
            borderLeft: '3px solid var(--accent-blue)',
            borderRadius: '0 var(--radius-sm) var(--radius-sm) 0', lineHeight: 1.8,
          }}>
            {note}
          </div>
        )}
      </div>
    </div>
  );
}
