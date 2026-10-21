import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { formatDecimal, compareDecimal, eventName, eventTime, describeEvent, selectedEvent } from './portfolioHistoryPresentation.mjs';
const history = JSON.parse(fs.readFileSync(new URL('../../../../research/v7/portfolio-history-artifact.json', import.meta.url)));
test('default event is latest; explicit event survives independent mode changes', () => {
  assert.equal(selectedEvent(history, null).seq, 7);
  for (const event of history.events) assert.equal(selectedEvent(history, event.seq), event);
  assert.equal(selectedEvent(history, 999).seq, 7);
});
test('display rounds decimal text without float loss or currency/percent semantics', () => {
  assert.equal(formatDecimal('1'), '1.0000');
  assert.equal(formatDecimal('0'), '0.0000');
  assert.equal(formatDecimal('0.000749999'), '0.0007');
  assert.equal(formatDecimal('0.00075'), '0.0008');
  assert.equal(formatDecimal('-0.00075'), '-0.0008');
  assert.equal(formatDecimal('999999999999999999999.99995'), '1,000,000,000,000,000,000,000.0000');
  assert.equal(formatDecimal('0.00000000001'), '≈ 0.0000');
  assert.equal(formatDecimal('0.00012345678', 6), '0.000123');
});
test('unknown is distinct from zero', () => {
  for (const value of [null, undefined, '', 'NaN', 'Infinity', 0, '1e-4']) assert.equal(formatDecimal(value), '資料不可用');
  assert.equal(compareDecimal(null, '0'), null);
});
test('comparison uses complete decimal text including beyond IEEE precision', () => {
  assert.equal(compareDecimal('1.00000000000000000001', '1'), 1);
  assert.equal(compareDecimal('-0.0000000000001', '0'), -1);
  assert.equal(compareDecimal('1.00', '1'), 0);
});
test('names and source timezone are conservative, unknown kind stays unknown', () => {
  assert.equal(eventName('GENESIS'), '初始狀態');
  assert.equal(eventName('MARKET_SESSION'), '市場時段');
  assert.equal(eventName('UNSEEN'), '未識別事件（UNSEEN）');
  assert.equal(eventTime('2026-09-16T08:00:00+08:00'), '2026-09-16 08:00:00 +08:00');
});
test('explanation describes observed changes without claiming causes or fills', () => {
  assert.match(describeEvent(history.events[0], null), /初始/);
  assert.match(describeEvent(history.events[1], history.events[0]), /未變/);
  assert.match(describeEvent(history.events[2], history.events[1]), /出現待成交目標/);
  assert.match(describeEvent(history.events[3], history.events[2]), /未變/);
  assert.match(describeEvent(history.events[4], history.events[3]), /現金減少/);
  assert.match(describeEvent(history.events[5], history.events[4]), /待成交目標清空/);
  assert.match(describeEvent(history.events[6], history.events[5]), /持倉數量/);
  for (let i=0; i<history.events.length;i++) assert.doesNotMatch(describeEvent(history.events[i], history.events[i-1]), /因為|獲利|報酬率|買入成功|停牌|股利|拆股/);
});
test('all presentation passes preserve original artifact bytes and identity', () => {
  const before = JSON.stringify(history);
  for (const event of history.events) {
    for (const value of [event.state.net_value,event.state.cash,event.state.total_cost]) formatDecimal(value);
    describeEvent(event,history.events[event.seq-2]); eventTime(event.occurred_at);
  }
  assert.equal(JSON.stringify(history), before);
});
