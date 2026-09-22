const STATES = new Set(['healthy', 'degraded', 'blocked', 'not_ready', 'error']);

const numericOrNull = (value) => (
  typeof value === 'number' && Number.isFinite(value) ? value : null
);

const stringOrNull = (value) => (
  typeof value === 'string' && value.length > 0 ? value : null
);

export function normalizeV7Status(payload) {
  const source = payload && typeof payload === 'object' ? payload : {};
  if (source.schema !== 'v7-health-summary-v1') {
    return {
      state: 'error', publishAllowed: false, dataId: null, generatedAt: null,
      message: 'V7 狀態資料格式無效。',
      counts: { entries: null, affectedDates: null, affectedStocks: null },
    };
  }
  const counts = source.counts && typeof source.counts === 'object' ? source.counts : {};
  const state = STATES.has(source.state) ? source.state : 'error';
  const publishAllowed = state === 'healthy' || state === 'degraded'
    ? source.publish_allowed === true
    : false;
  return {
    state,
    publishAllowed,
    dataId: stringOrNull(source.data_id),
    generatedAt: stringOrNull(source.generated_at),
    message: stringOrNull(source.message),
    counts: {
      entries: numericOrNull(counts.entries),
      affectedDates: numericOrNull(counts.affected_dates),
      affectedStocks: numericOrNull(counts.affected_stocks),
    },
  };
}
