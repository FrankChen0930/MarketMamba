import test from 'node:test';
import assert from 'node:assert/strict';

import { normalizeV7Status } from './v7Contract.mjs';

test('normalizes a valid health publication without inventing values', () => {
  const result = normalizeV7Status({
    schema: 'v7-health-summary-v1',
    state: 'degraded',
    publish_allowed: true,
    data_id: 'prepared-smoke',
    generated_at: '2026-09-16T00:00:00Z',
    message: 'quality warnings exist',
    counts: { entries: 12, affected_dates: 3 },
  });
  assert.equal(result.state, 'degraded');
  assert.equal(result.publishAllowed, true);
  assert.deepEqual(result.counts, {
    entries: 12,
    affectedDates: 3,
    affectedStocks: null,
  });
});

test('unknown state fails closed to error and missing fields remain null', () => {
  const result = normalizeV7Status({ schema: 'v7-health-summary-v1', state: 'surprise' });
  assert.equal(result.state, 'error');
  assert.equal(result.publishAllowed, false);
  assert.equal(result.dataId, null);
  assert.deepEqual(result.counts, {
    entries: null,
    affectedDates: null,
    affectedStocks: null,
  });
});

test('zero is preserved as a real numeric value', () => {
  const result = normalizeV7Status({
    schema: 'v7-health-summary-v1',
    state: 'healthy',
    publish_allowed: true,
    counts: { entries: 0, affected_dates: 0, affected_stocks: 0 },
  });
  assert.deepEqual(result.counts, {
    entries: 0,
    affectedDates: 0,
    affectedStocks: 0,
  });
});

test('wrong schema and blocked publication fail closed', () => {
  assert.equal(normalizeV7Status({ state: 'healthy', publish_allowed: true }).publishAllowed, false);
  assert.equal(normalizeV7Status({
    schema: 'v7-health-summary-v1', state: 'blocked', publish_allowed: true,
  }).publishAllowed, false);
});
