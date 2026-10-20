import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePortfolioHistory, createHistoryLoader } from './portfolioHistoryContract.mjs';

function fixture() {
  const spec = { engine_version: 'v7-portfolio-v1' };
  const state = { engine_version: spec.engine_version, spec, cash: '0', net_value: '1.234567890123456789', total_cost: '0', positions: { AAA: { quantity: '1', last_price: '1.234567890123456789' } }, pending: null, session_count: 0, last_rebalance_session: null, last_session_id: null };
  return { state: 'ready', history: { schema: 'v7-portfolio-history-v1', evidence_class: 'OFFLINE_SYNTHETIC_ENGINEERING', strict_phase0: 'STOP', research_admission: false, source: { source_commit: 'a'.repeat(40), fixture_id: 'portfolio-history-offline-fixture-v1', fixture_manifest_sha256: 'd'.repeat(64), journal_sha256: 'b'.repeat(64), record_hash: 'c'.repeat(64), engine_version: spec.engine_version, event_count: 1 }, spec, events: [{ seq: 1, event_id: 'genesis', kind: 'GENESIS', occurred_at: '2026-01-01T00:00:00+00:00', state }], final_state: structuredClone(state) } };
}
test('preserves exact decimal strings, true zeros and null pending', () => {
  const input = fixture();
  assert.deepEqual(normalizePortfolioHistory(input), input);
});
test('rejects non-fixture and admitted or non-STOP histories', () => {
  for (const [key, value] of [['evidence_class', 'RESEARCH'], ['research_admission', true], ['strict_phase0', 'PASS'], ['schema', 'unknown']]) {
    const input = fixture(); input.history[key] = value;
    assert.equal(normalizePortfolioHistory(input).state, 'error');
  }
});
test('rejects invalid identity, event count/order, and final state mismatch', () => {
  const mutations = [h => { h.source.journal_sha256 = 'bad'; }, h => { h.source.event_count = 2; }, h => { h.events[0].seq = 2; }, h => { h.final_state.cash = '2'; }, h => { h.source.source_commit = null; }, h => { h.source.fixture_id = ''; }, h => { h.source.fixture_manifest_sha256 = 'invalid'; }];
  for (const mutate of mutations) { const input = fixture(); mutate(input.history); assert.equal(normalizePortfolioHistory(input).state, 'error'); }
});
test('rejects malformed decimals, positions and pending', () => {
  for (const mutate of [s => { s.cash = 0; }, s => { s.total_cost = 'NaN'; }, s => { s.positions.AAA.last_price = null; }, s => { s.pending = {}; }]) {
    const input = fixture(); mutate(input.history.events[0].state); input.history.final_state = structuredClone(input.history.events[0].state);
    assert.equal(normalizePortfolioHistory(input).state, 'error');
  }
});
test('missing/error responses contain no stale history, malformed envelope fails closed', () => {
  assert.deepEqual(normalizePortfolioHistory({ state: 'not_ready', history: fixture().history }), { state: 'not_ready', message: '尚無可用的模擬歷史資料。' });
  for (const input of [null, {}, { state: 'ready' }, { state: 'error', history: fixture().history }]) assert.equal(normalizePortfolioHistory(input).state, 'error');
});
test('reload immediately clears old data and ignores out-of-order completion', async () => {
  const pending = [], updates = [];
  const loader = createHistoryLoader(() => new Promise(resolve => pending.push(resolve)), value => updates.push(value));
  const first = loader.load(), second = loader.load();
  assert.deepEqual(updates, [{ state: 'loading' }, { state: 'loading' }]);
  pending[1]({ state: 'not_ready' }); await second;
  pending[0](fixture()); await first;
  assert.equal(updates.at(-1).state, 'not_ready');
  assert.equal(updates.length, 3);
});
test('network failure and disposal never retain a ready result', async () => {
  const updates = [];
  const loader = createHistoryLoader(() => Promise.reject(new Error('network')), value => updates.push(value));
  await loader.load(); assert.equal(updates.at(-1).state, 'error'); assert.equal(updates.at(-1).history, undefined);
  let resolve; const disposed = createHistoryLoader(() => new Promise(r => { resolve = r; }), value => updates.push(value));
  const request = disposed.load(); disposed.dispose(); resolve(fixture()); await request;
  assert.equal(updates.at(-1).state, 'loading');
});

test('API transport uses only local origin and refuses non-200 data', async () => {
  const { getPortfolioHistory } = await import('./portfolioHistory.js');
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async (url, options) => {
      assert.equal(url, '/api/v7/portfolio-history');
      assert.equal(options.cache, 'no-store');
      assert.ok(options.signal instanceof AbortSignal);
      return { ok: false, json: async () => fixture() };
    };
    assert.deepEqual(await getPortfolioHistory(), { state: 'error' });
    globalThis.fetch = async () => ({ ok: true, json: async () => fixture() });
    assert.deepEqual(await getPortfolioHistory(), fixture());
  } finally { globalThis.fetch = original; }
});
