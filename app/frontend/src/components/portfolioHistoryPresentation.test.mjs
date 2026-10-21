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
  assert.match(describeEvent(history.events[2], history.events[1]), /新增了待成交目標/);
  assert.match(describeEvent(history.events[3], history.events[2]), /未變/);
  assert.match(describeEvent(history.events[4], history.events[3]), /現金減少/);
  assert.match(describeEvent(history.events[5], history.events[4]), /不再列有待成交目標/);
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

test('seven fixture narratives use deterministic observed changes and explicit limits', () => {
  const expected = [
    '這是組合的初始紀錄，沒有前一筆事件可供比較。',
    '與前一筆相比，淨值、現金、持倉及累計成本均未變動，也沒有待成交目標。這筆紀錄標示為市場時段，但現有欄位不足以判定未變動的原因。',
    '與前一筆相比，新增了待成交目標。這筆紀錄標示為訊號事件，並不代表已成交。',
    '與前一筆相比，淨值、現金、持倉及累計成本均未變動，待成交目標仍保留。這筆紀錄標示為市場時段，但現有欄位不足以判定未變動的原因。',
    '與前一筆相比，淨值減少、現金減少、累計成本增加、持倉數量改變。待成交目標仍保留。這些是已記錄的狀態變化，現有欄位不足以判定原因。',
    '與前一筆相比，淨值減少、現金減少、累計成本增加、持倉數量改變、不再列有待成交目標。這些是已記錄的狀態變化，現有欄位不足以判定原因。',
    '與前一筆相比，淨值增加、現金增加、持倉數量改變、持倉最後價格改變。這筆紀錄標示為公司行動，但現有欄位不足以判定具體類型或原因。',
  ];
  assert.deepEqual(history.events.map((e,i) => describeEvent(e,history.events[i-1])), expected);
});
test('narratives use precise states rather than event ID or a rounded value', () => {
  const before=structuredClone(history.events[3]);
  const after=structuredClone(before);
  after.event_id='dividend-buy-filled';
  after.state.cash='0.99999999999999999999999';
  assert.equal(formatDecimal(before.state.cash),formatDecimal(after.state.cash));
  assert.match(describeEvent(after,before),/現金減少/);
  assert.doesNotMatch(describeEvent(after,before),/股利|買入成功|已成交|dividend/);
  after.kind='CORPORATE_ACTION';
  assert.match(describeEvent(after,before),/不足以判定具體類型或原因/);
});
