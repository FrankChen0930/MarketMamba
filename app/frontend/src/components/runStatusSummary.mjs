export const RUN_STATES = Object.freeze([
  'healthy',
  'degraded',
  'blocked',
  'not_ready',
  'error',
]);

export const RUN_STATE_META = Object.freeze({
  healthy: { label: '健康', color: 'var(--positive)', background: 'rgba(0, 255, 136, 0.08)' },
  degraded: { label: '降級', color: 'var(--accent-amber)', background: 'rgba(255, 165, 0, 0.08)' },
  blocked: { label: '已阻擋', color: 'var(--negative)', background: 'rgba(255, 71, 87, 0.08)' },
  not_ready: { label: '尚未就緒', color: 'var(--text-secondary)', background: 'rgba(139, 148, 158, 0.08)' },
  error: { label: '錯誤', color: 'var(--negative)', background: 'rgba(255, 71, 87, 0.08)' },
});

const COUNT_LABELS = Object.freeze({
  entries: '品質事件',
  affectedDates: '影響日期',
  affectedStocks: '影響股票',
});

export function normalizeRunState(state) {
  return RUN_STATES.includes(state) ? state : 'error';
}

export function formatRunCount(value) {
  return typeof value === 'number' && Number.isFinite(value)
    ? new Intl.NumberFormat('zh-TW').format(value)
    : '—';
}

export function presentRunStatus({
  state,
  title,
  updatedAt,
  message,
  counts,
} = {}) {
  const normalizedState = normalizeRunState(state);
  const sourceCounts = counts && typeof counts === 'object' && !Array.isArray(counts)
    ? counts
    : {};
  return {
    state: normalizedState,
    title: typeof title === 'string' && title ? title : '執行狀態',
    updatedAt: typeof updatedAt === 'string' && updatedAt ? updatedAt : '—',
    message: typeof message === 'string' && message ? message : null,
    counts: Object.entries(sourceCounts).map(([key, value]) => ({
      key,
      label: COUNT_LABELS[key] ?? key,
      value: formatRunCount(value),
    })),
  };
}
