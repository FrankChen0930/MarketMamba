import React from 'react';

/**
 * SkeletonBlock — animated placeholder for loading states
 * Usage: <SkeletonBlock height={20} width="60%" />
 */
export function SkeletonBlock({ height = 16, width = '100%', style = {} }) {
  return (
    <div
      className="skeleton-block"
      style={{ height, width, borderRadius: 6, ...style }}
    />
  );
}

/**
 * SkeletonTable — placeholder for a data table
 */
export function SkeletonTable({ rows = 8, cols = 5 }) {
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          {Array.from({ length: cols }).map((_, i) => (
            <th key={i} style={{ padding: '8px 12px', borderBottom: '1px solid var(--border)' }}>
              <SkeletonBlock height={10} width={`${50 + i * 10}%`} />
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: rows }).map((_, r) => (
          <tr key={r}>
            {Array.from({ length: cols }).map((_, c) => (
              <td key={c} style={{ padding: '9px 12px', borderBottom: '1px solid rgba(48,54,61,0.4)' }}>
                <SkeletonBlock height={12} width={`${40 + ((r + c) * 13) % 50}%`} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/**
 * SkeletonCard — placeholder for a stat card
 */
export function SkeletonCard() {
  return (
    <div className="stat-card">
      <SkeletonBlock height={10} width="55%" style={{ marginBottom: 10 }} />
      <SkeletonBlock height={24} width="70%" style={{ marginBottom: 8 }} />
      <SkeletonBlock height={10} width="40%" />
    </div>
  );
}

/**
 * ApiError — shown when an API call fails
 */
export function ApiError({ message, onRetry }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 12, padding: 40, color: 'var(--text-muted)',
    }}>
      <span style={{ fontSize: 32 }}>⚠️</span>
      <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
        無法連線到後端
      </div>
      <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent-red)' }}>
        {message}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
        請確認 FastAPI 已啟動：<br />
        <code style={{ color: 'var(--accent-blue)' }}>uvicorn main:app --reload --port 8000</code>
      </div>
      {onRetry && (
        <button className="btn btn-primary" onClick={onRetry} style={{ marginTop: 4 }}>
          重試
        </button>
      )}
    </div>
  );
}
