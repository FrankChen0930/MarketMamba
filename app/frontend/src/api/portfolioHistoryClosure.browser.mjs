// Actual Chrome closure acceptance; no product helper is used as a visibility oracle.
// Usage: node script <base> <artifact> <output-prefix> <red|green>
import fs from 'node:fs';
import assert from 'node:assert/strict';
const [base,artifactPath,output,mode='green']=process.argv.slice(2);
const cdp='http://127.0.0.1:9230';
const diagnosticNoAnchor=process.argv.includes('--diagnostic-no-anchor');
// Independent fixed-fixture expected display values computed with Python Decimal ROUND_HALF_UP.
const expected=[{"nav": "1.0000", "cash": "1.0000", "cost": "0.0000"}, {"nav": "1.0000", "cash": "1.0000", "cost": "0.0000"}, {"nav": "1.0000", "cash": "1.0000", "cost": "0.0000"}, {"nav": "1.0000", "cash": "1.0000", "cost": "0.0000"}, {"nav": "0.9993", "cash": "0.5000", "cost": "0.0007"}, {"nav": "0.9985", "cash": "0.0007", "cost": "0.0015"}, {"nav": "1.0085", "cash": "0.0107", "cost": "0.0015"}];
const expectedPositions=[{}, {}, {}, {}, {"A": {"quantity": "0.049925", "last_price": "10.0000"}}, {"A": {"quantity": "0.099775", "last_price": "10.0000"}}, {"A": {"quantity": "0.199551", "last_price": "5.0000"}}];
const artifact=JSON.parse(fs.readFileSync(artifactPath,'utf8'));
const target = await (await fetch(`${cdp}/json/new?about:blank`,{method:'PUT'})).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
let id=0; const pending=new Map(), errors=[], checks=[];
ws.addEventListener('message',event=>{
 const m=JSON.parse(event.data);
 if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);}
 else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);

});
function send(method,params={}){return new Promise((resolve,reject)=>{const request=++id;pending.set(request,{resolve,reject});ws.send(JSON.stringify({id:request,method,params}));});}
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function wait(expression){for(let n=0;n<100;n++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw Error(`Timeout: ${expression}`);}
const q=s=>`document.querySelector(${JSON.stringify(s)})`;
async function click(selector){await evaluate(`${q(selector)}.click()`);}
async function key(selector,key,code,number){await evaluate(`${q(selector)}.focus()`);await send('Input.dispatchKeyEvent',{type:'keyDown',key,code,windowsVirtualKeyCode:number,...(key==='Enter'?{text:'\r'}:key===' '?{text:' '}: {})});await send('Input.dispatchKeyEvent',{type:'keyUp',key,code,windowsVirtualKeyCode:number});}
async function selected(seq){await wait(`document.querySelector('select').value === '${seq}'`);}
async function geometry(){return evaluate(`(()=>{const c=document.querySelector('.ph-timeline'),b=c.querySelector('[aria-pressed="true"]');const cr=c.getBoundingClientRect(),br=b.getBoundingClientRect();return {seq:Number(b.dataset.timelineSeq),ariaCurrent:b.getAttribute("aria-current"),selectedLabel:b.textContent.includes("目前選取"),containerTop:cr.top,containerBottom:cr.bottom,buttonTop:br.top,buttonBottom:br.bottom,visible:br.top>=cr.top-1&&br.bottom<=cr.bottom+1,scrollTop:c.scrollTop,windowY:scrollY,focus:JSON.stringify({tag:document.activeElement.tagName,seq:document.activeElement.dataset.timelineSeq??document.activeElement.dataset.chartSeq??null}),pressedStyle:getComputedStyle(b).backgroundColor}})()`);}
async function verify(seq){await selected(seq);await wait(`document.querySelector('[data-timeline-seq="${seq}"]').getBoundingClientRect().bottom <= document.querySelector('.ph-timeline').getBoundingClientRect().bottom+1 && document.querySelector('[data-timeline-seq="${seq}"]').getBoundingClientRect().top >= document.querySelector('.ph-timeline').getBoundingClientRect().top-1`);return geometry();}
try{
 await send('Page.enable');await send('Runtime.enable');
 for(const[name,width,height,mobile]of[['desktop',1440,1000,false],['mobile',390,844,true]]){
  await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile});await send('Page.navigate',{url:`${base}/v7/portfolio-history`});await wait(`!!document.querySelector('[data-testid="state-nav"]')`);await new Promise(r=>setTimeout(r,200));
  if(diagnosticNoAnchor)await evaluate(`(()=>{const style=document.createElement('style');style.textContent='.portfolio-history {overflow-anchor:none}';document.head.appendChild(style)})()`);
  const initial=await geometry();
  const api=await evaluate(`fetch('/api/v7/portfolio-history').then(r=>r.json())`);assert.deepEqual(api.history,artifact);
  const text=await evaluate(`document.body.innerText`);for(const marker of ['Synthetic Fixture','research_admission=false','Strict Phase 0：STOP',artifact.source.fixture_id])assert.ok(text.includes(marker));assert.equal(initial.seq,7);assert.equal(initial.windowY,0);
  fs.writeFileSync(`${output}-${name}-default.png`,Buffer.from((await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  checks.push({viewport:name,initial});
  if(mode==='red'){fs.writeFileSync(`${output}.json`,JSON.stringify({checks,expected:'selected7 visible',actual:initial.visible},null,2));assert.equal(initial.visible,true,'initial selected event 7 must be fully visible inside timeline');}
  assert.equal(initial.visible,true);assert.equal(initial.ariaCurrent,"step");assert.equal(initial.selectedLabel,true);
  const actions=[];
  for(const event of artifact.events){
   await evaluate(`document.querySelector('select').focus({preventScroll:true})`);const before=await geometry();
   await evaluate(`(()=>{const s=document.querySelector('select');s.value='${event.seq}';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);const after=await verify(event.seq);
   assert.equal(await evaluate(`document.querySelector('[data-testid=\"state-nav\"]').textContent`),expected[event.seq-1].nav);
   assert.equal(await evaluate(`document.querySelector('[data-testid=\"state-cash\"]').textContent`),expected[event.seq-1].cash);
   assert.equal(await evaluate(`document.querySelector('[data-testid=\"state-cost\"]').textContent`),expected[event.seq-1].cost);
   assert.equal(after.windowY,before.windowY,'programmatic selection cannot scroll page');assert.equal(after.focus,before.focus,'selection must preserve focus');
   assert.notEqual(after.pressedStyle,await evaluate(`getComputedStyle(document.querySelector('[data-timeline-seq="${event.seq===1?2:1}"]')).backgroundColor`));
   const positions=await evaluate(`Object.fromEntries(Array.from(document.querySelectorAll('[data-position-ticker]')).map(r=>[r.dataset.positionTicker,{quantity:r.querySelector('[data-field="quantity"]').textContent,last_price:r.querySelector('[data-field="last_price"]').textContent}]))`);assert.deepEqual(positions,expectedPositions[event.seq-1]);
   const pending=await evaluate(`document.querySelector('[data-testid="pending-state"]').innerText`);if(event.state.pending===null)assert.match(pending,/pending = null/);else for(const v of [event.state.pending.signal_id,...event.state.pending.target_tickers])assert.ok(pending.includes(v));
   actions.push({seq:event.seq,beforeY:before.windowY,afterY:after.windowY,visible:after.visible,timelineScroll:after.scrollTop,focusPreserved:true});
  }
  // Real chart pointer then keyboard activation. Setup scrolling is explicit user-test positioning.
  await evaluate(`document.querySelector('[data-chart-seq="3"]').scrollIntoView({block:'center',behavior:'instant'})`);
  const before=await geometry();const pt=await evaluate(`(()=>{const r=document.querySelector('[data-chart-seq="3"]').getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2}})()`);
  await send('Input.dispatchMouseEvent',{type:'mousePressed',...pt,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',...pt,button:'left',clickCount:1});const clicked=await verify(3);assert.equal(clicked.windowY,before.windowY);
  await key('[data-chart-seq="4"]','Enter','Enter',13);await verify(4);
  await key('[data-timeline-seq="5"]',' ','Space',32);await verify(5);
  await key('select','Home','Home',36);await verify(1);
  await evaluate(`document.querySelector('[data-timeline-seq=\"2\"]').scrollIntoView({block:'center',behavior:'instant'})`);
  const tp=await evaluate(`(()=>{const r=document.querySelector('[data-timeline-seq=\"2\"]').getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2}})()`);
  await send('Input.dispatchMouseEvent',{type:'mousePressed',...tp,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',...tp,button:'left',clickCount:1});await verify(2);
  await evaluate(`scrollTo({top:0,behavior:'instant'});document.activeElement.blur()`);await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:5,y:5});await new Promise(r=>setTimeout(r,250));fs.writeFileSync(`${output}-${name}-selected.png`,Buffer.from((await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  await click('.ph-mode button:nth-of-type(2)');await wait(`${q('.portfolio-history')}.dataset.mode==='advanced'`);
  const events=name==='desktop'?artifact.events:[artifact.events[4]];
  for(const event of events){await evaluate(`(()=>{const s=document.querySelector('select');s.value='${event.seq}';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await verify(event.seq);assert.deepEqual(JSON.parse(await evaluate(`${q('[data-testid="raw-event"]')}.value`)),event);for(const[field,test]of[['net_value','nav'],['cash','cash'],['total_cost','cost']])assert.equal(await evaluate(`${q(`[data-testid="state-${test}"]`)}.textContent`),event.state[field]);}
  await click('.ph-mode button:nth-of-type(1)');await selected(events.at(-1).seq);
  checks.at(-1).actions=actions;checks.at(-1).pointer={beforeY:before.windowY,afterY:clicked.windowY,seq:3};checks.at(-1).advancedExactEvents=events.length;checks.at(-1).keyboard=['chart Enter','timeline Space','select Home'];
 }
 assert.deepEqual(errors,[]);fs.writeFileSync(`${output}.json`,JSON.stringify({observed_at:new Date().toISOString(),diagnosticNoAnchor,checks,browser:await(await fetch(`${cdp}/json/version`)).json(),runtime_exceptions:errors},null,2));console.log(JSON.stringify(checks,null,2));
}finally{await send('Page.close');ws.close();}
