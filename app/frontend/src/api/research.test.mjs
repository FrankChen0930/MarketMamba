import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { normalizeResearch, formatMetric, formatCI } from './researchContract.mjs';

const fixture = JSON.parse(fs.readFileSync(new URL('../../../../V6/research_lab/experiments/summary.json', import.meta.url)));
test('real summary preserves research tier, seed rows, windows and annual comparison', () => {
  const result = normalizeResearch(fixture);
  assert.equal(result.state, 'ready');
  assert.equal(result.models.length, 26);
  assert.equal(result.portfolios.length, 7);
  assert.equal(result.evidence_tier.tier, 'RESEARCH_ONLY');
  assert.ok(result.caveats.length >= 7);
  assert.ok(result.portfolios[0].years['2026']);
});
test('malformed, partial, upgraded or not-ready data hides all numbers', () => {
  for (const payload of [null, {}, {...fixture, state:'not_ready'}, {...fixture, schema:'other'},
    {...fixture, evidence_tier:{tier:'VERIFIED'}}, {...fixture, caveats:[]},
    {...fixture, status:null}, {...fixture, evaluation_policy:{}},
    {...fixture, identity:{...fixture.identity, source_commit:{}}},
    {...fixture, identity:{...fixture.identity, unexpected:{}}},
    {...fixture, models:[]}, {...fixture, portfolios:[{}]}]) {
    const result = normalizeResearch(payload);
    assert.equal(result.state, 'not_ready');
    assert.deepEqual(result.models, []);
    assert.deepEqual(result.portfolios, []);
    assert.equal(result.evidence_tier.tier, 'RESEARCH_ONLY');
  }
});
test('invalid numeric values do not become zero or a number', () => {
  for (const value of [null, undefined, '', '0.1', false, NaN, Infinity]) assert.equal(formatMetric(value), '—');
  assert.equal(formatMetric(0), '0.000');
  assert.equal(formatMetric(.125, true), '12.50%');
  assert.equal(formatMetric(-.02, true), '-2.00%');
  assert.equal(formatCI([-.02,.03]), '[-0.020, 0.030]');
  assert.equal(formatCI(null), '—');
  assert.equal(formatCI([null,.03]), '—');
});
