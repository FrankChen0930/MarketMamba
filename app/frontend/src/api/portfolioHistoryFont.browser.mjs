// Real Chrome late-font regression. Holds actual Google font requests, then releases
// and awaits document.fonts.ready + two animation frames (not a visibility retry).
// Usage: Windows Node script <base> <fixture-artifact> <output-prefix> [CDP URL]
// Missing/failed font loads or unchanged row geometry fail; never a synthetic PASS.
import fs from 'node:fs';
import assert from 'node:assert/strict';
const [base,artifactPath,output,cdp='http://127.0.0.1:9230']=process.argv.slice(2);
const artifact=JSON.parse(fs.readFileSync(artifactPath,'utf8'));
const target=await(await fetch(`${cdp}/json/new?about:blank`,{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let id=0,holding=true;const pending=new Map(),held=[],fontRequests=[],checks=[],failures=[],exceptions=[];
ws.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else if(m.method==='Fetch.requestPaused'){fontRequests.push(m.params.request.url);if(holding)held.push(m.params.requestId);else send('Fetch.continueRequest',{requestId:m.params.requestId});}else if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params.exceptionDetails);});
function send(method,params={}){return new Promise((resolve,reject)=>{pending.set(++id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function ev(expression){const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function wait(expression){for(let n=0;n<100;n++){if(await ev(expression))return;await new Promise(r=>setTimeout(r,100));}throw Error(`observable condition timeout: ${expression}`);}
async function frame(){await ev(`new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);}
async function geometry(){return ev(`(()=>{const c=document.querySelector('.ph-timeline'),b=c.querySelector('[aria-pressed="true"]'),cr=c.getBoundingClientRect(),br=b.getBoundingClientRect(),a=document.activeElement;return{seq:Number(b.dataset.timelineSeq),container:{top:cr.top,bottom:cr.bottom,clientTop:c.clientTop,clientHeight:c.clientHeight},button:{top:br.top,bottom:br.bottom,height:br.height},visible:br.top>=cr.top+c.clientTop-1&&br.bottom<=cr.top+c.clientTop+c.clientHeight+1,rows:[...c.children].map(r=>({height:r.getBoundingClientRect().height,top:r.getBoundingClientRect().top})),scrollTop:c.scrollTop,pageY:scrollY,focus:{tag:a.tagName,id:a.id,seq:a.dataset.timelineSeq??a.dataset.chartSeq??null},fontsStatus:document.fonts.status,fontFaces:[...document.fonts].map(f=>({family:f.family,status:f.status}))}})()`);}
function check(ok,message){if(!ok)failures.push(message);}
try{
 await send('Page.enable');await send('Runtime.enable');await send('Network.enable');await send('Network.setCacheDisabled',{cacheDisabled:true});await send('Fetch.enable',{patterns:[{urlPattern:'*fonts.gstatic.com*',requestStage:'Request'}]});
 for(const[name,width,height,mobile]of[['desktop',1440,1000,false],['mobile',390,844,true]]){
  holding=true;held.length=0;fontRequests.length=0;await send('Network.clearBrowserCache');await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile});await send('Page.navigate',{url:`${base}/v7/portfolio-history`});
  await wait(`!!document.querySelector('[data-testid="state-nav"]')&&document.fonts.status==='loading'`);await frame();
  check(held.length>0,`${name}: no actual held font requests`);
  const before=await geometry();check(before.visible,`${name}: initial selection invisible`);check(before.seq===7&&before.pageY===0,`${name}: initial selection/page incorrect`);
  const count=held.length;holding=false;await Promise.all(held.splice(0).map(requestId=>send('Fetch.continueRequest',{requestId})));
  await ev(`Promise.race([document.fonts.ready.then(()=>true),new Promise((_,reject)=>setTimeout(()=>reject(Error('font ready timeout')),10000))])`);await frame();
  const after=await geometry();check(after.fontsStatus==='loaded'&&after.fontFaces.some(f=>f.status==='loaded'),`${name}: fonts did not load`);check(after.fontFaces.every(f=>f.status!=='error'),`${name}: failed font face`);check(before.rows.some((r,i)=>r.height!==after.rows[i].height),`${name}: no row height change, regression not exercised`);
  check(after.visible,`${name}: selected 7 clipped AFTER actual fonts.ready/layout`);check(after.pageY===before.pageY,`${name}: late font page jump`);check(JSON.stringify(after.focus)===JSON.stringify(before.focus),`${name}: late font changed focus`);
  const record={viewport:name,width,height,heldRequests:count,fontURLs:[...fontRequests],before,after,events:[]};checks.push(record);
  fs.writeFileSync(`${output}-${name}.png`,Buffer.from((await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  // Once font layout is observed, select all fixture events without visibility retries.
  const api=await ev(`fetch('/api/v7/portfolio-history').then(r=>r.json())`);assert.deepEqual(api.history,artifact);
  if(!mobile){await ev(`document.querySelector('.ph-mode button:nth-of-type(2)').click()`);await frame();}
  for(const event of artifact.events){await ev(`document.querySelector('select').focus({preventScroll:true})`);const pre=await geometry();await ev(`(()=>{const s=document.querySelector('select');s.value='${event.seq}';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await frame();const post=await geometry();check(post.visible,`${name}: event ${event.seq} invisible`);check(pre.pageY===post.pageY&&JSON.stringify(pre.focus)===JSON.stringify(post.focus),`${name}: event ${event.seq} moved page/focus`);if(!mobile)assert.deepEqual(JSON.parse(await ev(`document.querySelector('[data-testid="raw-event"]').value`)),event);record.events.push({seq:event.seq,visible:post.visible,pageY:post.pageY,focus:post.focus});}
  if(mobile){await ev(`document.querySelector('.ph-mode button:nth-of-type(2)').click()`);await frame();assert.deepEqual(JSON.parse(await ev(`document.querySelector('[data-testid="raw-event"]').value`)),artifact.events.at(-1));}
 }
 check(exceptions.length===0,'runtime exceptions');const result={observed_at:new Date().toISOString(),browser:await(await fetch(`${cdp}/json/version`)).json(),checks,failures,exceptions,pass:failures.length===0};fs.writeFileSync(`${output}.json`,JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));if(failures.length){console.error('late-font acceptance FAILED');process.exitCode=1;}
}finally{holding=false;await Promise.all(held.splice(0).map(requestId=>send('Fetch.continueRequest',{requestId})));await send('Page.close');ws.close();}
