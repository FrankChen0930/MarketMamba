import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  formatRunCount,
  normalizeRunState,
  presentRunStatus,
} from './runStatusSummary.mjs';

test('only known states pass through and unknown fails closed', () => {
  assert.equal(normalizeRunState('healthy'), 'healthy');
  assert.equal(normalizeRunState('not_ready'), 'not_ready');
  assert.equal(normalizeRunState('surprise'), 'error');
  assert.equal(normalizeRunState(undefined), 'error');
});

test('missing numbers render an em dash while a real zero is preserved', () => {
  assert.equal(formatRunCount(null), '—');
  assert.equal(formatRunCount(undefined), '—');
  assert.equal(formatRunCount(Number.NaN), '—');
  assert.equal(formatRunCount(0), '0');
  assert.equal(formatRunCount(12), '12');
});

test('invalid counts render an em dash without coercing non-numeric values', () => {
  for (const value of [-1, -0.5, 0.5, 12.25, Number.NaN, Infinity, -Infinity, null, undefined, '12', '', true, false, {}, [], 12n]) {
    assert.equal(formatRunCount(value), '—', `unexpected count: ${String(value)}`);
  }
});

test('non-negative integer counts preserve zero and localized grouping', () => {
  assert.equal(formatRunCount(0), '0');
  assert.equal(formatRunCount(12), '12');
  assert.equal(formatRunCount(1234), '1,234');
});

test('presentation normalizes invalid counts and preserves known labels and grouping', () => {
  assert.deepEqual(presentRunStatus({
    counts: {
      entries: 1234,
      affectedDates: 0,
      affectedStocks: 12,
      negative: -1,
      fractional: 1.5,
      nan: Number.NaN,
      infinite: Infinity,
      negativeInfinite: -Infinity,
    },
  }).counts, [
    { key: 'entries', label: '品質事件', value: '1,234' },
    { key: 'affectedDates', label: '影響日期', value: '0' },
    { key: 'affectedStocks', label: '影響股票', value: '12' },
    { key: 'negative', label: 'negative', value: '—' },
    { key: 'fractional', label: 'fractional', value: '—' },
    { key: 'nan', label: 'nan', value: '—' },
    { key: 'infinite', label: 'infinite', value: '—' },
    { key: 'negativeInfinite', label: 'negativeInfinite', value: '—' },
  ]);
});

test('presentation keeps the five-prop contract and normalizes counts', () => {
  const result = presentRunStatus({
    state: 'degraded',
    title: 'Import run',
    updatedAt: null,
    message: 'Two rows quarantined',
    counts: { processed: 0, rejected: null },
  });
  assert.equal(result.state, 'degraded');
  assert.equal(result.title, 'Import run');
  assert.equal(result.updatedAt, '—');
  assert.equal(result.message, 'Two rows quarantined');
  assert.deepEqual(result.counts, [
    { key: 'processed', label: 'processed', value: '0' },
    { key: 'rejected', label: 'rejected', value: '—' },
  ]);
});

test('component exposes an accessible status live region', async () => {
  const source = await readFile(new URL('./RunStatusSummary.jsx', import.meta.url), 'utf8');
  assert.match(source, /role="status"/);
  assert.match(source, /aria-live="polite"/);
});
