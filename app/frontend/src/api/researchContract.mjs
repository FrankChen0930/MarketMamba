const numeric = (value) => typeof value === 'number' && Number.isFinite(value);
const metric = (value) => value === null || numeric(value);
const hash = (value) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const research = (row) => row?.evidence_tier?.tier === 'RESEARCH_ONLY';
const baseModels = ['gru', 'lightgbm', 'lowvol', 'momentum20', 'reversal5', 'ridge', 'selective'];

export function normalizeResearch(payload) {
  const empty = { state: 'not_ready', evidence_tier: { tier: 'RESEARCH_ONLY' },
    caveats: ['僅供研究，未經交易驗證。摘要無效或尚未發布，沒有沿用舊資料。'], models: [], portfolios: [] };
  if (!payload || payload.schema !== 'research-summary-v1' || payload.state !== 'ready' ||
      !research(payload) || !hash(payload.sha256) ||
      payload.status !== 'UNMERGED / NOT_DEPLOYED' ||
      typeof payload.evaluation_policy !== 'string' || !payload.evaluation_policy ||
      Object.keys(payload.identity ?? {}).sort().join(',') !== 'dataset_sha256,evaluation_sha256,manifest_sha256,source_commit' ||
      typeof payload.identity?.source_commit !== 'string' || !/^[a-f0-9]{40}$/.test(payload.identity.source_commit) ||
      !['dataset_sha256', 'manifest_sha256', 'evaluation_sha256'].every(k => hash(payload.identity?.[k])) ||
      !Array.isArray(payload.caveats) || payload.caveats.length < 7 ||
      !payload.caveats.every(v => typeof v === 'string' && v.length > 0) ||
      !Array.isArray(payload.models) || !Array.isArray(payload.portfolios)) return empty;
  const validModels = payload.models.every(row => research(row) && typeof row.model === 'string' &&
    [5, 20].includes(row.horizon) && ['ic_mean','icir','nw_t'].every(k => metric(row[k])) &&
    ['days','sessions'].every(k => Number.isInteger(row[k]) && row[k] >= 0) &&
    Array.isArray(row.ic_ci95) && row.ic_ci95.length === 2 && row.ic_ci95.every(metric));
  const validPortfolios = payload.portfolios.length === 7 && payload.portfolios.every(row => research(row) &&
    baseModels.includes(row.model) && row.horizon === 20 && row.config?.top_k === 20 &&
    row.config?.rebalance_sessions === 20 && typeof row.start === 'string' && typeof row.end === 'string' &&
    ['total_return','annualized_return','benchmark_return','excess_return','sharpe_like',
      'max_drawdown','mean_turnover','cost_drag','information_ratio'].every(k => metric(row[k])) &&
    row.years && typeof row.years === 'object' && Object.keys(row.years).length > 0 &&
    Object.entries(row.years).every(([year, values]) => /^\d{4}$/.test(year) && values &&
      metric(values.total_return) && metric(values.benchmark_return)));
  const complete = baseModels.every(model => [5,20].every(h => payload.models.some(r => r.model === model && r.horizon === h))) &&
    new Set(payload.portfolios.map(r => r.model)).size === 7 &&
    new Set(payload.models.map(r => `${r.model}/${r.horizon}`)).size === payload.models.length;
  return validModels && validPortfolios && complete ? payload : empty;
}

export function formatMetric(value, percent = false) {
  if (!numeric(value)) return '—';
  return percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(3);
}

export function formatCI(values) {
  return Array.isArray(values) && values.length === 2 && values.every(numeric)
    ? `[${values.map(v => formatMetric(v)).join(', ')}]` : '—';
}
