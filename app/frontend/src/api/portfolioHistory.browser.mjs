// Manual local-browser acceptance (Node 24+, existing Chrome CDP on 127.0.0.1:9228).
// Usage: node portfolioHistory.browser.mjs http://127.0.0.1:5178 <history.json|-> <output-prefix> [missing|corrupt|any]
// Requires the fixture API on the Vite proxy target. Use a dedicated temporary Chrome
// profile. This script reads the supplied artifact, writes evidence/screenshots only,
// and never mutates source artifacts. "-" instead of artifact checks unavailable UI.
// Error-state CDP interceptions are explicitly distinguished from real API evidence.
import fs from 'node:fs';
import assert from 'node:assert/strict';
const [base = 'http://127.0.0.1:5178', artifactPath, outputPrefix = '/tmp/port03-browser', expectedUnavailable = 'any'] = process.argv.slice(2);
const target = await (await fetch('http://127.0.0.1:9228/json/new?about:blank', { method: 'PUT' })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, { once: true }));
let interception = null;
let id = 0; const pending = new Map(), errors = [];
ws.addEventListener('message', event => { const message = JSON.parse(event.data); if (message.id) { const item = pending.get(message.id); pending.delete(message.id); if (message.error) item.reject(new Error(JSON.stringify(message.error))); else item.resolve(message.result); } else if (message.method === 'Fetch.requestPaused' && interception) { const params = message.params; if (interception === 'network') send('Fetch.failRequest', {requestId:params.requestId,errorReason:'ConnectionRefused'}); else send('Fetch.fulfillRequest', {requestId:params.requestId,responseCode:interception === 'missing' ? 200 : 503,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify(interception === 'missing' ? {state:'not_ready'} : {state:'error'})).toString('base64')}); } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails); });
function send(method, params = {}) { return new Promise((resolve, reject) => { const request = ++id; pending.set(request, { resolve, reject }); ws.send(JSON.stringify({ id: request, method, params })); }); }
async function evaluate(expression) { const response = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }); if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails)); return response.result.value; }
async function waitFor(expression) { for (let n = 0; n < 100; n++) { if (await evaluate(expression)) return; await new Promise(r => setTimeout(r, 100)); } throw new Error(`Timeout: ${expression}`); }
await send('Page.enable'); await send('Runtime.enable');
const artifact = artifactPath && artifactPath !== '-' ? JSON.parse(fs.readFileSync(artifactPath, 'utf8')) : null;
const checks = [];
try {
  for (const [name, width, height, mobile] of [['desktop', 1440, 1000, false], ['mobile', 390, 844, true]]) {
    await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile });
    await send('Page.navigate', { url: `${base}/v7/portfolio-history` });
    await waitFor(`document.querySelector('[data-testid="history-status"]') && !document.querySelector('[data-testid="history-status"]').textContent.includes('正在讀取')`);
    const text = await evaluate('document.body.innerText');
    assert.match(text, /Fixture 模擬資料/); assert.match(text, /Research admission：false/); assert.match(text, /Strict Phase 0：STOP/);
    const overflow = await evaluate('document.documentElement.scrollWidth > window.innerWidth'); assert.equal(overflow, false, `${name} page overflows`);
    if (artifact) {
      assert.match(text, new RegExp(artifact.source.journal_sha256)); assert.match(text, new RegExp(artifact.source.fixture_manifest_sha256));
      const rows = await evaluate(`Array.from(document.querySelectorAll('[data-event-seq]')).map(row => ({seq:Number(row.dataset.eventSeq), net_value:row.querySelector('[data-field="net_value"]').textContent,cash:row.querySelector('[data-field="cash"]').textContent,total_cost:row.querySelector('[data-field="total_cost"]').textContent}))`);
      assert.deepEqual(rows, artifact.events.map(e => ({seq:e.seq,net_value:e.state.net_value,cash:e.state.cash,total_cost:e.state.total_cost})));
      for (const event of artifact.events) {
        await evaluate(`(()=>{const s=document.querySelector('select');s.value=${event.seq};s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
        await waitFor(`document.querySelector('[data-testid="state-nav"]').textContent===${JSON.stringify(event.state.net_value)}`);
        const state = await evaluate(`({nav:document.querySelector('[data-testid="state-nav"]').textContent,cash:document.querySelector('[data-testid="state-cash"]').textContent,cost:document.querySelector('[data-testid="state-cost"]').textContent})`);
        assert.deepEqual(state, {nav:event.state.net_value,cash:event.state.cash,cost:event.state.total_cost});
        const positions = await evaluate(`Object.fromEntries(Array.from(document.querySelectorAll('[data-position-ticker]')).map(row => [row.dataset.positionTicker, {quantity:row.querySelector('[data-field="quantity"]').textContent,last_price:row.querySelector('[data-field="last_price"]').textContent}]))`);
        assert.deepEqual(positions, event.state.positions);
        const pendingText = await evaluate(`document.querySelector('[data-testid="pending-state"]').innerText`);
        if (event.state.pending === null) assert.match(pendingText, /pending = null/);
        else for (const value of [event.state.pending.signal_id, event.state.pending.as_of, ...event.state.pending.target_tickers]) assert.ok(pendingText.includes(value));
      }
    } else {
      assert.match(text, expectedUnavailable === 'missing' ? /尚無可用的模擬歷史資料/ : expectedUnavailable === 'corrupt' ? /歷史資料損壞/ : /尚無可用|無法連線|目前無法使用/);
      assert.equal(await evaluate(`document.querySelector('[data-testid="state-nav"]') === null && document.querySelectorAll('[data-event-seq]').length === 0`), true);
    }
    await evaluate(`document.querySelector('.portfolio-history button').focus()`);
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
    let focus = await evaluate('document.activeElement.tagName');
    const focusOrder = [focus];
    if (artifact) {
      for (let n=0;n<5 && focus !== 'SELECT';n++) {
        await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
        await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Tab', code: 'Tab', windowsVirtualKeyCode: 9 });
        focus = await evaluate('document.activeElement.tagName'); focusOrder.push(focus);
      }
      assert.equal(focus, 'SELECT');
      await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Home', code: 'Home', windowsVirtualKeyCode: 36 });
      await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Home', code: 'Home', windowsVirtualKeyCode: 36 });
      await waitFor(`document.querySelector('select').value === '1'`);
      assert.equal(await evaluate(`document.querySelector('[data-testid="state-nav"]').textContent`), artifact.events[0].state.net_value);
    }
    const screenshot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    fs.writeFileSync(`${outputPrefix}-${name}.png`, Buffer.from(screenshot.data, 'base64'));
    checks.push({ viewport: name, width, height, overflow, keyboard_focus_tag: focus, keyboard_focus_order: focusOrder, events_verified: artifact?.events.length ?? 0, status: await evaluate(`document.querySelector('[data-testid="history-status"]').textContent`) });
  }
  if (artifact) {
    await send('Fetch.enable', {patterns:[{urlPattern:'*/api/v7/portfolio-history',requestStage:'Request'}]});
    for (const scenario of ['missing','corrupt','network']) {
      interception = scenario;
      await evaluate(`document.querySelector('.portfolio-history button').click()`);
      await waitFor(`document.querySelector('[data-testid="history-status"]') && !document.querySelector('[data-testid="history-status"]').textContent.includes('正在讀取')`);
      assert.equal(await evaluate(`document.querySelector('[data-testid="state-nav"]')===null && document.querySelectorAll('[data-event-seq]').length===0`), true);
      const status = await evaluate(`document.querySelector('[data-testid="history-status"]').textContent`);
      assert.match(status, scenario === 'missing' ? /尚無可用/ : /無法|損壞/);
      checks.push({scenario,method:'CDP request interception after actual API validation',stale_history_visible:false,status});
    }
    await send('Fetch.disable');
  }
  assert.deepEqual(errors, []);
  const result = { observed_at: new Date().toISOString(), browser: await (await fetch('http://127.0.0.1:9228/json/version')).json(), checks, runtime_exceptions: errors, unavailable_expectation: artifact ? null : expectedUnavailable };
  fs.writeFileSync(`${outputPrefix}.json`, JSON.stringify(result,null,2)+'\n'); console.log(JSON.stringify(result,null,2));
} finally { await send('Page.close'); ws.close(); }
