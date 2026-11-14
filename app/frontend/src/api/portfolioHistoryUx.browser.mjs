// Fixture-only UX acceptance against actual Chrome CDP. Error cases explicitly use CDP injection.
// Windows Node 24+: node <script> <base-url> <artifact-path> <output-prefix> [cdp-url]
// Uses existing Chrome; writes only screenshots/evidence and never changes fixture data.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { formatDecimal, eventTime } from '../components/portfolioHistoryPresentation.mjs';
const [base, artifactPath, output, cdp='http://127.0.0.1:9229'] = process.argv.slice(2);
const artifact = JSON.parse(fs.readFileSync(artifactPath,'utf8'));
assert.equal(formatDecimal('1'),'1.0000');
assert.equal(formatDecimal('0.999999'),'1.0000');
assert.equal(formatDecimal('1234.56785'),'1,234.5679');
assert.equal(formatDecimal('0.0000001'),'≈ 0.0000');
const original = JSON.stringify(artifact);
const target = await (await fetch(`${cdp}/json/new?about:blank`,{method:'PUT'})).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
let id=0, interception=null, requests=0; const pending=new Map(), errors=[], checks=[];
ws.addEventListener('message',event=>{
 const m=JSON.parse(event.data);
 if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}
 else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);
 else if(m.method==='Network.requestWillBeSent'&&m.params.request.url.endsWith('/api/v7/portfolio-history'))requests++;
 else if(m.method==='Fetch.requestPaused'){
  const requestId=m.params.requestId;
  if(interception==='network')send('Fetch.failRequest',{requestId,errorReason:'ConnectionRefused'});
  else send('Fetch.fulfillRequest',{requestId,responseCode:interception==='missing'?200:503,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify(interception==='missing'?{state:'not_ready'}:{state:'error'})).toString('base64')});
 }
});
function send(method,params={}){return new Promise((resolve,reject)=>{const request=++id;pending.set(request,{resolve,reject});ws.send(JSON.stringify({id:request,method,params}));});}
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function wait(expression){for(let n=0;n<100;n++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw Error(`Timeout: ${expression}`);}
const q=s=>`document.querySelector(${JSON.stringify(s)})`;
async function click(selector){await evaluate(`${q(selector)}.click()`);}
async function key(selector,key,code,number){await evaluate(`${q(selector)}.focus()`);await send('Input.dispatchKeyEvent',{type:'keyDown',key,code,windowsVirtualKeyCode:number,...(key==='Enter'?{text:'\r'}:key===' '?{text:' '}: {})});await send('Input.dispatchKeyEvent',{type:'keyUp',key,code,windowsVirtualKeyCode:number});}
async function selected(seq){await wait(`document.querySelector('select').value === '${seq}'`);}
async function verifyState(event,advanced){
 await selected(event.seq);
 const observed=await evaluate(`({nav:${q('[data-testid="state-nav"]')}.textContent,cash:${q('[data-testid="state-cash"]')}.textContent,cost:${q('[data-testid="state-cost"]')}.textContent,positions:Object.fromEntries(Array.from(document.querySelectorAll('[data-position-ticker]')).map(r=>[r.dataset.positionTicker,{quantity:r.querySelector('[data-field="quantity"]').textContent,last_price:r.querySelector('[data-field="last_price"]').textContent}])),pending:${q('[data-testid="pending-state"]')}.innerText,summary:${q('[data-testid="pending-summary"]')}.innerText})`);
 const fmt=(v,d=4)=>advanced?v:formatDecimal(v,d);
 assert.equal(observed.nav,fmt(event.state.net_value));assert.equal(observed.cash,fmt(event.state.cash));assert.equal(observed.cost,fmt(event.state.total_cost));
 assert.deepEqual(observed.positions,Object.fromEntries(Object.entries(event.state.positions).map(([t,p])=>[t,{quantity:fmt(p.quantity,6),last_price:fmt(p.last_price)}])));
 if(event.state.pending===null){assert.equal(observed.summary,'無待成交');assert.match(observed.pending,/pending = null/);}else{assert.equal(observed.summary,'有待成交目標');for(const v of [event.state.pending.signal_id,eventTime(event.state.pending.as_of),...event.state.pending.target_tickers])assert.ok(observed.pending.includes(v));}
 if(advanced)assert.deepEqual(JSON.parse(await evaluate(`${q('[data-testid="raw-event"]')}.value`)),event);
}
try{
 await send('Page.enable');await send('Runtime.enable');await send('Network.enable');
 for(const [name,width,height,mobile]of[['desktop',1440,1000,false],['mobile',390,844,true]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile});await send('Page.navigate',{url:`${base}/v7/portfolio-history`});
  await wait(`!!document.querySelector('[data-testid="state-nav"]')`);
  assert.equal(await evaluate(`${q('.portfolio-history')}.dataset.mode`),'general');
  const api=await evaluate(`fetch('/api/v7/portfolio-history').then(r=>r.json())`);assert.deepEqual(api.history,artifact);
  const body=await evaluate('document.body.innerText');for(const marker of ['Synthetic Fixture','research_admission=false','Strict Phase 0：STOP',artifact.source.fixture_id])assert.ok(body.includes(marker));
  assert.equal(await evaluate('document.documentElement.scrollWidth > innerWidth'),false);
  for(const event of artifact.events){
   await evaluate(`(()=>{const s=document.querySelector('select');s.value='${event.seq}';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await verifyState(event,false);
   await click(`[data-timeline-seq="${event.seq === 1 ? 7 : 1}"]`);
   await click(`[data-timeline-seq="${event.seq}"]`);await verifyState(event,false);
   await click(`[data-timeline-seq="${event.seq === 1 ? 7 : 1}"]`);
   // CDP pointer click on the actual SVG dot, not a DOM click handler call.
   await evaluate(`${q(`[data-chart-seq="${event.seq}"]`)}.scrollIntoView({block:'center',behavior:"instant"})`);
   const point=await evaluate(`(()=>{const r=${q(`[data-chart-seq="${event.seq}"]`)}.getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2}})()`);
   await send('Input.dispatchMouseEvent',{type:'mousePressed',...point,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',...point,button:'left',clickCount:1});await verifyState(event,false);
  }
  await evaluate('scrollTo({top:0,behavior:"instant"})');await new Promise(r=>setTimeout(r,150));
  fs.writeFileSync(`${output}-${name}-general.png`,Buffer.from((await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  const before=requests;
  await key('.ph-mode button:nth-of-type(2)','Enter','Enter',13);await wait(`${q('.portfolio-history')}.dataset.mode==='advanced'`);await selected(7);
  assert.equal(requests,before,'mode switch must not fetch');
  for(const event of artifact.events){await click(`[data-timeline-seq="${event.seq}"]`);await verifyState(event,true);}
  const rows=await evaluate(`Array.from(document.querySelectorAll('[data-event-seq]')).map(r=>({seq:Number(r.dataset.eventSeq),nav:r.querySelector('[data-field="net_value"]').textContent,cash:r.querySelector('[data-field="cash"]').textContent,cost:r.querySelector('[data-field="total_cost"]').textContent}))`);
  assert.deepEqual(rows,artifact.events.map(e=>({seq:e.seq,nav:e.state.net_value,cash:e.state.cash,cost:e.state.total_cost})));
  const advancedText=await evaluate('document.body.innerText');for(const value of Object.values(artifact.source))assert.ok(advancedText.includes(String(value)));assert.ok(advancedText.includes(artifact.schema));
  await evaluate('scrollTo({top:0,behavior:"instant"})');fs.writeFileSync(`${output}-${name}-advanced.png`,Buffer.from((await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  assert.equal(await evaluate('document.documentElement.scrollWidth > innerWidth'),false);
  await key('.ph-mode button:nth-of-type(1)',' ','Space',32);await wait(`${q('.portfolio-history')}.dataset.mode==='general'`);await selected(7);assert.equal(requests,before);
  await key('[data-timeline-seq="2"]','Enter','Enter',13);await verifyState(artifact.events[1],false);
  await key('[data-chart-seq="3"]',' ','Space',32);await verifyState(artifact.events[2],false);
  await key('[data-chart-seq="4"]','Enter','Enter',13);await verifyState(artifact.events[3],false);
  await key('select','Home','Home',36);await verifyState(artifact.events[0],false);
  checks.push({viewport:name,width,height,events_verified:7,advanced_exact_rows:7,raw_events_exact:7,chart_pointer_clicks:7,mode_preserves_selection:true,mode_additional_requests:0,keyboard:['mode Enter/Space','timeline Enter','chart Enter/Space','select Home'],horizontal_overflow:false});
 }
 await send('Fetch.enable',{patterns:[{urlPattern:'*/api/v7/portfolio-history',requestStage:'Request'}]});
 for(const scenario of ['missing','corrupt','network']){
  interception=scenario;await click('.page-header button');await wait(`!${q('[data-testid="history-status"]')}.textContent.includes('正在讀取')`);
  const status=await evaluate(`${q('[data-testid="history-status"]')}.textContent`);assert.match(status,scenario==='missing'?/尚無可用/:scenario==='corrupt'?/損壞/:/無法連線/);
  assert.equal(await evaluate(`!document.querySelector('[data-testid="state-nav"]')&&!document.querySelector('[data-timeline-seq]')`),true);
  checks.push({scenario,method:'CDP injected response/network failure; not real artifact corruption',status,stale_visible:false});
 }
 await send('Fetch.disable');assert.deepEqual(errors,[]);assert.equal(JSON.stringify(artifact),original);
 const result={observed_at:new Date().toISOString(),browser:await(await fetch(`${cdp}/json/version`)).json(),checks,runtime_exceptions:errors,fixture_identity:artifact.source,source_artifact_unchanged:true};
 fs.writeFileSync(`${output}.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
}finally{await send('Page.close');ws.close();}
