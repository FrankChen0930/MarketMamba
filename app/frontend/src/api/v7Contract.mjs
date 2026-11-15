const STATES = new Set(['healthy', 'degraded', 'blocked', 'not_ready', 'error']);

const numericOrNull = (value) => (
  typeof value === 'number' && Number.isFinite(value) ? value : null
);

const stringOrNull = (value) => (
  typeof value === 'string' && value.length > 0 ? value : null
);

const TIERS = new Set(['VERIFIED', 'RESEARCH_ONLY', 'BLOCKED']);

// Fail closed: anything unrecognised is RESEARCH_ONLY; VERIFIED needs the
// backend to say so AND the health publication to be healthy and allowed.
function normalizeEvidenceTier(raw, state, publishAllowed) {
  const source = raw && typeof raw === 'object' ? raw : {};
  let tier = TIERS.has(source.tier) ? source.tier : 'RESEARCH_ONLY';
  if (tier === 'VERIFIED' && !(state === 'healthy' && publishAllowed)) tier = 'RESEARCH_ONLY';
  if (state === 'blocked') tier = 'BLOCKED';
  const reasons = Array.isArray(source.reasons)
    ? source.reasons.filter((item) => typeof item === 'string' && item.length > 0)
    : [];
  return { tier, reasons };
}

export function normalizeV7Status(payload) {
  const source = payload && typeof payload === 'object' ? payload : {};
  if (source.schema !== 'v7-health-summary-v1') {
    return {
      state: 'error', publishAllowed: false, dataId: null, generatedAt: null,
      message: 'V7 狀態資料格式無效。',
      evidenceTier: { tier: 'RESEARCH_ONLY', reasons: [] },
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
    evidenceTier: normalizeEvidenceTier(source.evidence_tier, state, publishAllowed),
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
