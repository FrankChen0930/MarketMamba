import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {runProcess, processIdentity, isSameProcess, stopProcessGroup} from './process.mjs';
import {validateJob, validateOutput, atomicJson, recoverRecord} from './runtime.mjs';
const temp=()=>fs.mkdtempSync(path.join(os.tmpdir(),'mm-ruflo-'));
const job=()=>({taskContext:'offline fixture',workers:[{id:'coder',platform:'codex',role:'builder',prompt:'fixture',readOnly:false},{id:'reviewer',platform:'claude',role:'verifier',prompt:'review',readOnly:true,dependsOn:['coder']}]});
test('reject invalid platforms, cycles, duplicate ids, invalid bounds and arbitrary workspace',()=>{
 for(const change of [j=>j.workers[0].platform='shell',j=>j.workers[0].dependsOn=['reviewer'],j=>j.workers[1].id='coder',j=>j.timeoutSeconds=0,j=>j.maxConcurrent=9,j=>j.workers[0].worktreePath='/tmp',j=>j.workers[0].readOnly='false']) {
 const j=job();change(j);assert.throws(()=>validateJob(j)); }
 assert.equal(validateJob(job()).workers.length,2);
});
test('requires explicit non-empty prompts and dependencies',()=>{
 for(const change of [j=>j.workers[0].prompt='',j=>j.workers[1].dependsOn=['absent'],j=>j.workers=[],j=>j.workers[0].id='../escape']){const j=job();change(j);assert.throws(()=>validateJob(j));}
});
test('exit zero without a completed model turn or final response fails closed',()=>{
 assert.throws(()=>validateOutput('codex','{"type":"thread.started","thread_id":"x"}\n','done'));
 assert.throws(()=>validateOutput('claude','{"type":"result","subtype":"success","is_error":true,"result":"done"}'));
 assert.throws(()=>validateOutput('codex','{"type":"thread.started","thread_id":"x"}\n{"type":"turn.completed"}\n',''));
 assert.equal(validateOutput('codex','{"type":"thread.started","thread_id":"x"}\n{"type":"turn.completed","usage":{"input_tokens":1}}\n','done').threadId,'x');
 assert.equal(validateOutput('claude','{"type":"result","subtype":"success","is_error":false,"result":"done"}').text,'done');
});
test('real process exit and nonzero status captured without shell interpolation',async()=>{
 const d=temp();const r=await runProcess(process.execPath,['-e','process.stdout.write(process.argv[1]);process.exit(7)','$(echo unsafe)'],{cwd:d,log:path.join(d,'run'),timeoutMs:3000,maxOutputBytes:1000});
 assert.equal(r.exitCode,7);assert.equal(fs.readFileSync(path.join(d,'run.out'),'utf8'),'$(echo unsafe)');
});
test('output overflow terminates instead of accepting a truncated success',async()=>{
 const d=temp();const r=await runProcess(process.execPath,['-e','setInterval(()=>process.stdout.write("x".repeat(2048)),1)'],{cwd:d,log:path.join(d,'run'),timeoutMs:3000,maxOutputBytes:1024});
 assert.equal(r.reason,'output_limit');assert.ok(fs.statSync(path.join(d,'run.out')).size<=1024);
});
test('timeout kills stubborn leader and descendants',async()=>{
 const d=temp();const r=await runProcess(process.execPath,['-e','process.on("SIGTERM",()=>{});require("child_process").spawn(process.execPath,["-e","setInterval(()=>{},1000)"],{stdio:"ignore"});setInterval(()=>{},1000)'],{cwd:d,log:path.join(d,'run'),timeoutMs:180,maxOutputBytes:1024});
 assert.equal(r.reason,'timeout');assert.equal(isSameProcess(r.identity),false);
});
test('abort cancels child and identity mismatch cannot signal another process',async()=>{
 const c=new AbortController();const d=temp();let identity;
 const p=runProcess(process.execPath,['-e','setInterval(()=>{},1000)'],{cwd:d,log:path.join(d,'run'),timeoutMs:3000,maxOutputBytes:1024,signal:c.signal,onSpawn:i=>{identity=i;setTimeout(()=>c.abort(),30);}});
 const r=await p;assert.equal(r.reason,'cancelled');assert.equal(isSameProcess(identity),false);
 assert.equal(stopProcessGroup({...processIdentity(process.pid),start:'wrong'}),false);
});
test('missing executable records failure without hanging',async()=>{
 const d=temp();await assert.rejects(runProcess('/no/such/ruflo-worker',[],{cwd:d,log:path.join(d,'run'),timeoutMs:1000,maxOutputBytes:1024}));
});
test('durable state roundtrip and corruption fails instead of re-running',()=>{
 const d=temp(), f=path.join(d,'state.json');atomicJson(f,{status:'running',pid:processIdentity(process.pid),children:[]});
 assert.equal(recoverRecord(JSON.parse(fs.readFileSync(f))).status,'running');
 assert.throws(()=>recoverRecord({status:'running',pid:{pid:0}}));
 const r=recoverRecord({status:'running',pid:{pid:999999,start:'missing',boot:'missing'},children:[]});assert.equal(r.status,'interrupted');
 assert.equal(recoverRecord({status:'completed',success:true}).status,'completed');
});
test('fresh process reads identical durable state bytes',()=>{
 const d=temp(),f=path.join(d,'x.json');atomicJson(f,{sourceCommit:'abc',status:'interrupted'});
 const r=spawnSync(process.execPath,['-e','process.stdout.write(require("fs").readFileSync(process.argv[1]))',f],{encoding:'utf8'});assert.equal(r.status,0);assert.deepEqual(JSON.parse(r.stdout),{sourceCommit:'abc',status:'interrupted'});
});

const installed='/home/frank/projects/ruflo-evaluation/tools/ruflo-codex/node_modules/@claude-flow/codex/dist/dual-mode/orchestrator.js';
const {DualModeOrchestrator}=await import(installed);
function fixtureOrchestrator(){
 const o=new DualModeOrchestrator({projectPath:temp(),maxConcurrent:2,maxWriters:1,timeout:1000,dependencyFailure:'cancel'});
 o.initializeSharedMemory=async()=>{};o.collectSharedMemory=async()=>({});return o;
}
test('official orchestrator never dispatches failed downstream work',async()=>{
 const o=fixtureOrchestrator(),seen=[];o.executeHeadless=async w=>{seen.push(w.id);throw new Error('fixture failure');};
 const r=await o.runCollaboration(job().workers,'fixture');assert.equal(r.success,false);assert.deepEqual(seen,['coder']);assert.equal(r.workers.find(w=>w.id==='reviewer').status,'skipped');
});
test('official orchestrator enforces one writer and parallel read bounds',async()=>{
 const o=fixtureOrchestrator();let active=0,writers=0,peak=0,peakWriters=0;
 o.executeHeadless=async w=>{active++;if(!w.readOnly)writers++;peak=Math.max(peak,active);peakWriters=Math.max(peakWriters,writers);await new Promise(r=>setTimeout(r,20));active--;if(!w.readOnly)writers--;return 'done';};
 const workers=[false,false,true,true].map((readOnly,i)=>({id:`w${i}`,readOnly,platform:'codex',role:'fixture',prompt:'fixture'}));
 const r=await o.runCollaboration(workers,'fixture');assert.equal(r.success,true);assert.equal(peak,2);assert.equal(peakWriters,1);
});
test('fixed base captures committed outputs even when worker status is clean',()=>{
 const d=temp();const run=args=>{const r=spawnSync('git',args,{cwd:d,encoding:'utf8'});assert.ifError(r.error);assert.equal(r.status,0,r.stderr);return r.stdout.trim();};
 run(['init','-q']);run(['config','user.name','fixture']);run(['config','user.email','fixture@example.invalid']);fs.writeFileSync(path.join(d,'a.txt'),'before\n');run(['add','a.txt']);run(['commit','-qm','base']);const base=run(['rev-parse','HEAD']);
 fs.writeFileSync(path.join(d,'a.txt'),'after\n');run(['add','a.txt']);run(['commit','-qm','worker']);assert.equal(run(['status','--porcelain']),'');assert.match(run(['diff','--binary',base]),/after/);assert.equal(run(['diff','--name-only',base]),'a.txt');
});
import {filterRequest} from './mcp.mjs';
import {gitRaw} from './runtime.mjs';
test('MCP rejects unlisted direct tools/call and pins explicit SQLite dbPath',()=>{
 assert.throws(()=>filterRequest({jsonrpc:'2.0',id:1,method:'tools/call',params:{name:'agent_execute',arguments:{}}},'/project/memory.db'));
 const r=filterRequest({jsonrpc:'2.0',id:2,method:'tools/call',params:{name:'memory_store',arguments:{dbPath:'/tmp/other.db',key:'x',value:'v'}}},'/project/memory.db');assert.equal(r.params.arguments.dbPath,'/project/memory.db');
});
test('raw Git patch retains newline and applies tracked/deleted changes with spaced names',()=>{
 const d=temp();const run=args=>gitRaw(d,args);run(['init','-q']);run(['config','user.name','fixture']);run(['config','user.email','fixture@example.invalid']);
 fs.writeFileSync(path.join(d,' spaced.txt'),'before\n');fs.writeFileSync(path.join(d,'deleted.txt'),'gone\n');run(['add',' spaced.txt','deleted.txt']);run(['commit','-qm','base']);const base=run(['rev-parse','HEAD']).trim();
 fs.writeFileSync(path.join(d,' spaced.txt'),'after\n');fs.unlinkSync(path.join(d,'deleted.txt'));const patch=run(['diff','--binary',base]);assert.ok(patch.endsWith('\n'));assert.equal(run(['diff','--name-only','-z',base]).split('\0')[0],' spaced.txt');
 run(['restore',' spaced.txt','deleted.txt']);const f=path.join(temp(),'worker.patch');fs.writeFileSync(f,patch);run(['apply',f]);assert.equal(fs.readFileSync(path.join(d,' spaced.txt'),'utf8'),'after\n');assert.equal(fs.existsSync(path.join(d,'deleted.txt')),false);
});
import {captureOutputs,validateSubmission,sha} from './runtime.mjs';
test('actual artifact collector retains committed, new, spaced and deleted outputs',()=>{
 const d=temp(),out=temp();const run=args=>gitRaw(d,args);run(['init','-q']);run(['config','user.name','fixture']);run(['config','user.email','fixture@example.invalid']);
 fs.writeFileSync(path.join(d,' spaced.txt'),'before\n');fs.writeFileSync(path.join(d,'deleted.txt'),'gone\n');run(['add','.']);run(['commit','-qm','base']);const base=run(['rev-parse','HEAD']).trim();
 fs.writeFileSync(path.join(d,' spaced.txt'),'after\n');fs.unlinkSync(path.join(d,'deleted.txt'));run(['add','-u']);run(['commit','-qm','worker']);fs.writeFileSync(path.join(d,'new.txt'),'new\n');
 const r=captureOutputs(d,base,out,'worker');assert.equal(r.artifacts.length,3);assert.equal(r.artifacts.find(x=>x.path==='deleted.txt').deleted,true);assert.equal(fs.readFileSync(r.artifacts.find(x=>x.path==='new.txt').snapshot,'utf8'),'new\n');assert.match(fs.readFileSync(path.join(out,'worker.patch'),'utf8'),/after/);assert.throws(()=>captureOutputs(d,base,out,'reviewer',true));
});
test('submission missing/corrupt/hash mismatch fails closed',()=>{
 const d=temp(),file=path.join(d,'job.json');fs.writeFileSync(file,'{}');const valid={sourceCommit:'a'.repeat(40),jobSha256:sha(file)};validateSubmission(valid,file);
 for(const s of [null,{}, {...valid,sourceCommit:'HEAD'},{...valid,jobSha256:'bad'}])assert.throws(()=>validateSubmission(s,file));fs.writeFileSync(file,'{"changed":true}');assert.throws(()=>validateSubmission(valid,file));
});
test('SIGKILL supervisor then recovery stops a recorded orphan process group',async()=>{
 const d=temp(),state=path.join(d,'state.json'),processModule=new URL('./process.mjs',import.meta.url).href;
 const script=`import fs from 'node:fs';import {runProcess,processIdentity} from ${JSON.stringify(processModule)};await runProcess(process.execPath,['-e','setInterval(()=>{},1000)'],{cwd:${JSON.stringify(d)},log:${JSON.stringify(path.join(d,'child'))},timeoutMs:10000,maxOutputBytes:1024,onSpawn:c=>fs.writeFileSync(${JSON.stringify(state)},JSON.stringify({status:'running',pid:processIdentity(process.pid),children:[c]}))});`;
 const {spawn}=await import('node:child_process');const supervisor=spawn(process.execPath,['--input-type=module','-e',script],{stdio:'ignore'});
 for(let n=0;n<200&&!fs.existsSync(state);n++)await new Promise(r=>setTimeout(r,10));assert.ok(fs.existsSync(state));const record=JSON.parse(fs.readFileSync(state));assert.ok(isSameProcess(record.children[0]));
 const exited=new Promise(r=>supervisor.once('close',r));supervisor.kill('SIGKILL');await exited;assert.equal(recoverRecord(record).status,'interrupted');await new Promise(r=>setTimeout(r,30));assert.equal(isSameProcess(record.children[0]),false);
});
test('dead leader with live unknown descendants blocks recovery instead of releasing job',async()=>{
 const d=temp(),pidFile=path.join(d,'descendant.json');const {spawn}=await import('node:child_process');
 const script=`const c=require('child_process').spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});require('fs').writeFileSync(${JSON.stringify(pidFile)},JSON.stringify({pid:c.pid}));setTimeout(()=>process.exit(0),100);`;
 const leader=spawn(process.execPath,['-e',script],{detached:true,stdio:'ignore'});const identity=processIdentity(leader.pid);
 await new Promise(r=>leader.once('close',r));const child=JSON.parse(fs.readFileSync(pidFile));assert.ok(isSameProcess(processIdentity(child.pid)));
 try{assert.throws(()=>recoverRecord({status:'running',pid:{pid:999999,start:'gone',boot:'gone'},children:[identity]}),/UNKNOWN orphan group/);}finally{process.kill(-leader.pid,'SIGKILL');}
});
import {verifyCompletedArtifacts} from './runtime.mjs';
test('completed result binds worker manifest, raw logs, patch and snapshot integrity',()=>{
 const d=temp(),prefix=path.join(d,'worker');for(const suffix of ['.out','.err','.patch','.snapshot'])fs.writeFileSync(prefix+suffix,suffix);
 const manifest={stdoutSha256:sha(prefix+'.out'),stderrSha256:sha(prefix+'.err'),patchSha256:sha(prefix+'.patch'),artifacts:[{snapshot:prefix+'.snapshot',sha256:sha(prefix+'.snapshot'),size:fs.statSync(prefix+'.snapshot').size}]};
 atomicJson(prefix+'.result.json',manifest);atomicJson(path.join(d,'result.json'),{workers:[{status:'completed'}],evidence:{worker:{path:prefix+'.result.json',sha256:sha(prefix+'.result.json')}}});const record={resultSha256:sha(path.join(d,'result.json'))};verifyCompletedArtifacts(record,d);
 fs.writeFileSync(prefix+'.snapshot','corrupt');assert.throws(()=>verifyCompletedArtifacts(record,d),/snapshot/);fs.writeFileSync(prefix+'.snapshot','.snapshot');fs.unlinkSync(prefix+'.patch');assert.throws(()=>verifyCompletedArtifacts(record,d));
});
