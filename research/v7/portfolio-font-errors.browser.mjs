// Fixture-only UX acceptance against actual Chrome CDP. Error cases explicitly use CDP injection.
// Windows Node 24+: node <script> <base-url> <artifact-path> <output-prefix> [cdp-url]
// Uses existing Chrome; writes only screenshots/evidence and never changes fixture data.
import fs from 'node:fs';
import assert from 'node:assert/strict';
const [base, artifactPath, output, cdp='http://127.0.0.1:9229'] = process.argv.slice(2);
const artifact = JSON.parse(fs.readFileSync(artifactPath,'utf8'));
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
try{
 await send('Page.enable');await send('Runtime.enable');await send('Page.navigate',{url:`${base}/v7/portfolio-history`});await wait(`!!document.querySelector('[data-testid="state-nav"]')`);
 const api=await evaluate(`fetch('/api/v7/portfolio-history').then(r=>r.json())`);assert.deepEqual(api.history,artifact);
 await send('Fetch.enable',{patterns:[{urlPattern:'*/api/v7/portfolio-history',requestStage:'Request'}]});
 for(const scenario of ['missing','corrupt','network']){interception=scenario;await click('.page-header button');await wait(`!${q('[data-testid="history-status"]')}.textContent.includes('正在讀取')`);const status=await evaluate(`${q('[data-testid="history-status"]')}.textContent`);assert.match(status,scenario==='missing'?/尚無可用/:scenario==='corrupt'?/損壞/:/無法連線/);assert.equal(await evaluate(`!document.querySelector('[data-testid="state-nav"]')&&!document.querySelector('[data-timeline-seq]')`),true);checks.push({scenario,status,method:'CDP injected response/network failure after real fixture ready',stale:false});}
 assert.deepEqual(errors,[]);fs.writeFileSync(output+'.json',JSON.stringify({checks,errors,observed_at:new Date().toISOString()},null,2));console.log(JSON.stringify(checks));
}finally{await send('Page.close');ws.close();}
