import { presentRunStatus, RUN_STATE_META } from './runStatusSummary.mjs';

export default function RunStatusSummary({
  state,
  title,
  updatedAt,
  message,
  counts,
}) {
  const view = presentRunStatus({ state, title, updatedAt, message, counts });
  const meta = RUN_STATE_META[view.state];

  return (
    <section
      role="status"
      aria-live="polite"
      aria-atomic="true"
      style={{
        border: `1px solid color-mix(in srgb, ${meta.color} 35%, var(--border))`,
        borderRadius: 'var(--radius-md)',
        background: meta.background,
        padding: 'var(--space-5)',
        minWidth: 0,
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-4)',
      }}
    >
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        gap: 'var(--space-3)',
        flexWrap: 'wrap',
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 17, color: 'var(--text-primary)' }}>
            {view.title}
          </h2>
          <p className="mono" style={{ margin: 'var(--space-1) 0 0', fontSize: 11, color: 'var(--text-muted)', overflowWrap: 'anywhere' }}>
            更新：{view.updatedAt}
          </p>
        </div>
        <span className="badge" style={{
          color: meta.color,
          border: `1px solid color-mix(in srgb, ${meta.color} 35%, transparent)`,
          background: 'var(--bg-panel)',
        }}>
          {meta.label}
        </span>
      </div>

      {view.message && (
        <p style={{ margin: 0, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          {view.message}
        </p>
      )}

      {view.counts.length > 0 && (
        <dl style={{
          margin: 0,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))',
          gap: 'var(--space-3)',
        }}>
          {view.counts.map((item) => (
            <div key={item.key} className="panel" style={{ padding: 'var(--space-3)', minWidth: 0 }}>
              <dt style={{ color: 'var(--text-muted)', fontSize: 11 }}>{item.label}</dt>
              <dd className="mono" style={{ margin: 'var(--space-1) 0 0', fontSize: 18, fontWeight: 700 }}>
                {item.value}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
