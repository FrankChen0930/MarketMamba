// Presentation only: no mutation, floating-point accounting, currency or return policy.
const DECIMAL = /^(-?)(\d+)(?:\.(\d+))?$/;
function parts(value) {
  if (typeof value !== 'string') return null;
  const match = DECIMAL.exec(value);
  return match ? { negative: match[1] === '-', whole: match[2], fraction: match[3] ?? '' } : null;
}
export function formatDecimal(value, digits = 4) {
  const parsed = parts(value);
  if (!parsed || !Number.isInteger(digits) || digits < 0 || digits > 8) return '資料不可用';
  const { negative, whole, fraction } = parsed;
  let scaled = BigInt(whole + fraction.padEnd(digits, '0').slice(0, digits));
  if ((fraction[digits] ?? '0') >= '5') scaled += 1n;
  const padded = scaled.toString().padStart(digits + 1, '0');
  const integer = (digits ? padded.slice(0, -digits) : padded).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const decimal = digits ? `.${padded.slice(-digits)}` : '';
  const tiny = scaled === 0n && /[1-9]/.test(whole + fraction);
  return `${tiny ? '≈ ' : ''}${negative && scaled !== 0n ? '-' : ''}${integer}${decimal}`;
}
export function compareDecimal(a, b) {
  const left = parts(a), right = parts(b);
  if (!left || !right) return null;
  const scale = Math.max(left.fraction.length, right.fraction.length);
  const value = p => BigInt(p.whole + p.fraction.padEnd(scale, '0')) * (p.negative ? -1n : 1n);
  const x = value(left), y = value(right);
  return x === y ? 0 : x > y ? 1 : -1;
}
const EVENT_NAMES = { GENESIS: '初始狀態', SIGNAL: '訊號事件', MARKET_SESSION: '市場時段', CORPORATE_ACTION: '公司行動' };
export const eventName = kind => EVENT_NAMES[kind] ?? `未識別事件（${kind}）`;
export const eventTime = value => typeof value === 'string' ? value.replace('T', ' ').replace(/(Z|[+-]\d\d:\d\d)$/, ' $1') : '時間不可用';
export const selectedEvent = (history, seq) => history.events.find(event => event.seq === seq) ?? history.events.at(-1);
export function describeEvent(event, previous) {
  if (!previous) return '記錄組合的初始狀態；尚無前一事件可比較。';
  const current = event.state, before = previous.state, changes = [];
  for (const [field, name] of [['net_value', 'NAV'], ['cash', '現金'], ['total_cost', '累計成本']]) {
    const direction = compareDecimal(current[field], before[field]);
    if (direction === 1) changes.push(`${name}增加`);
    else if (direction === -1) changes.push(`${name}減少`);
    else if (direction === null) changes.push(`${name}無法比較`);
  }
  const tickers = new Set([...Object.keys(current.positions), ...Object.keys(before.positions)]);
  if ([...tickers].some(ticker => compareDecimal(current.positions[ticker]?.quantity ?? '0', before.positions[ticker]?.quantity ?? '0') !== 0)) changes.push('持倉數量改變');
  if ([...tickers].some(ticker => current.positions[ticker] && before.positions[ticker] && compareDecimal(current.positions[ticker].last_price, before.positions[ticker].last_price) !== 0)) changes.push('持倉最後價格改變');
  if (before.pending === null && current.pending !== null) changes.push('出現待成交目標');
  else if (before.pending !== null && current.pending === null) changes.push('待成交目標清空');
  else if (JSON.stringify(before.pending) !== JSON.stringify(current.pending)) changes.push('待成交目標欄位改變');
  return changes.length ? `${changes.join('；')}。僅比較相鄰事件的完整狀態，不推定原因。` : 'NAV、現金、持倉與待成交目標未變，累計成本未變。此處不推定事件原因。';
}
