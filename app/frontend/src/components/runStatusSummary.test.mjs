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
