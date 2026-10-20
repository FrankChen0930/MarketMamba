// Display validation only. The trusted exporter owns accounting and artifact verification.
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = value => typeof value === 'string' && value.length > 0;
const decimal = value => typeof value === 'string' && /^-?\d+(?:\.\d+)?$/.test(value);
const integer = value => Number.isSafeInteger(value) && value >= 0;
const canonical = value => JSON.stringify(value, function (_, item) {
  return object(item) ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item;
});
function validState(state, history) {
  if (!object(state) || state.engine_version !== history.source.engine_version || canonical(state.spec) !== canonical(history.spec)) return false;
  if (!['cash', 'net_value', 'total_cost'].every(key => decimal(state[key])) || !integer(state.session_count)) return false;
  if (!(state.last_rebalance_session === null || integer(state.last_rebalance_session)) || !(state.last_session_id === null || text(state.last_session_id))) return false;
  if (!object(state.positions) || !Object.entries(state.positions).every(([ticker, position]) => text(ticker) && object(position) && decimal(position.quantity) && decimal(position.last_price))) return false;
  const pending = state.pending;
  return pending === null || (object(pending) && text(pending.signal_id) && text(pending.as_of) && integer(pending.created_session) && Array.isArray(pending.target_tickers) && pending.target_tickers.every(text));
}
export function normalizePortfolioHistory(input) {
  const error = { state: 'error', message: '歷史資料損壞、身份不符或目前無法使用；未沿用舊資料。' };
  if (!object(input)) return error;
  if (input.state === 'not_ready') return { state: 'not_ready', message: '尚無可用的模擬歷史資料。' };
  const h = input.history;
  if (input.state !== 'ready' || !object(h) || h.schema !== 'v7-portfolio-history-v1' || h.evidence_class !== 'OFFLINE_SYNTHETIC_ENGINEERING' || h.strict_phase0 !== 'STOP' || h.research_admission !== false) return error;
  if (!object(h.source) || !object(h.spec) || !/^[a-f0-9]{40}$/.test(h.source.source_commit) || !['journal_sha256', 'record_hash', 'fixture_manifest_sha256'].every(key => /^[a-f0-9]{64}$/.test(h.source[key])) || !text(h.source.engine_version) || !text(h.source.fixture_id)) return error;
  if (!Array.isArray(h.events) || h.events.length === 0 || h.source.event_count !== h.events.length) return error;
  const ids = new Set();
  for (const [index, event] of h.events.entries()) {
    if (!object(event) || event.seq !== index + 1 || !text(event.event_id) || ids.has(event.event_id) || !text(event.kind) || !text(event.occurred_at) || !validState(event.state, h)) return error;
    ids.add(event.event_id);
  }
  if (!validState(h.final_state, h) || canonical(h.final_state) !== canonical(h.events.at(-1).state)) return error;
  return { state: 'ready', history: h };
}

// One request generation owns displayed data; refresh clears stale state immediately.
export function createHistoryLoader(fetchHistory, publish) {
  let generation = 0;
  let disposed = false;
  return {
    async load() {
      const current = ++generation;
      publish({ state: 'loading' });
      try {
        const result = normalizePortfolioHistory(await fetchHistory());
        if (!disposed && current === generation) publish(result);
      } catch {
        if (!disposed && current === generation) publish({ state: 'error', message: '目前無法連線到歷史服務；未沿用舊資料。' });
      }
    },
    dispose() { disposed = true; generation += 1; },
  };
}
