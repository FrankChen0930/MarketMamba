#!/usr/bin/env node
// Thin project adapter over the pinned official DualModeOrchestrator.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawn,spawnSync} from 'node:child_process';
import {runProcess,processIdentity,isSameProcess,stopProcessGroup} from './process.mjs';
export const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const CONTRACT=JSON.parse(fs.readFileSync(path.join(ROOT,'tools/project_agents/integration-contract.json')));
const SAFE=/^[a-z0-9][a-z0-9_-]{0,47}$/;
export const sha=f=>crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
export function gitRaw(cwd,args){const r=spawnSync('git',args,{cwd,encoding:'utf8'});if(r.error||r.status!==0)throw new Error(r.error?.message??r.stderr.trim());return r.stdout;}
export function git(cwd,args){return gitRaw(cwd,args).trim();}
export function commonRoot(){return path.dirname(path.resolve(ROOT,git(ROOT,['rev-parse','--git-common-dir'])));}
export function stateRoot(){return path.join(commonRoot(),'.artifacts/ruflo');}
export function atomicJson(file,value){
 fs.mkdirSync(path.dirname(file),{recursive:true,mode:0o700});
 const tmp=`${file}.${process.pid}.${crypto.randomUUID()}.tmp`;
 const fd=fs.openSync(tmp,'wx',0o600);
 try{fs.writeFileSync(fd,JSON.stringify(value,null,2)+'\n');fs.fsyncSync(fd);}finally{fs.closeSync(fd);}
 fs.renameSync(tmp,file);const dir=fs.openSync(path.dirname(file),'r');try{fs.fsyncSync(dir);}finally{fs.closeSync(dir);}
}
export function validateJob(input){
 if(!input||typeof input!=='object'||!Array.isArray(input.workers)||!input.workers.length||input.workers.length>8||typeof input.taskContext!=='string'||!input.taskContext.trim())throw new Error('Expected taskContext and 1..8 workers');
 const timeoutSeconds=input.timeoutSeconds??900,maxConcurrent=input.maxConcurrent??3;
 if(!Number.isInteger(timeoutSeconds)||timeoutSeconds<1||timeoutSeconds>7200||!Number.isInteger(maxConcurrent)||maxConcurrent<1||maxConcurrent>3)throw new Error('Limits: timeoutSeconds 1..7200; maxConcurrent 1..3');
 const ids=new Set();
 const workers=input.workers.map(w=>{
  if(!w||!SAFE.test(w.id)||ids.has(w.id)||!['claude','codex'].includes(w.platform)||typeof w.prompt!=='string'||!w.prompt.trim()||typeof w.readOnly!=='boolean'||typeof w.role!=='string'||!w.role.trim())throw new Error('Invalid worker');
  if(w.worktreePath!==undefined||w.capabilityEnvelope!==undefined)throw new Error('Worker cannot supply workspace/capability override');
  if(w.dependsOn!==undefined&&(!Array.isArray(w.dependsOn)||!w.dependsOn.every(x=>typeof x==='string')))throw new Error('Invalid dependencies');
  if(w.model!==undefined&&(typeof w.model!=='string'||!w.model.trim()||w.model.length>100))throw new Error('Invalid model');
  ids.add(w.id);return {id:w.id,platform:w.platform,role:w.role,prompt:w.prompt,readOnly:w.readOnly,dependsOn:w.dependsOn??[],...(w.model?{model:w.model}:{})};
 });
 const done=new Set();while(done.size<workers.length){const ready=workers.filter(w=>!done.has(w.id)&&w.dependsOn.every(x=>done.has(x)));if(!ready.length)throw new Error('Missing dependency or cycle');ready.forEach(w=>done.add(w.id));}
 return {taskContext:input.taskContext,workers,timeoutSeconds,maxConcurrent};
}
export function validateSubmission(s,file){if(!s||!/^([a-f0-9]{40}|[a-f0-9]{64})$/.test(s.sourceCommit??'')||s.jobSha256!==sha(file))throw new Error('Submission identity mismatch');}
export function validateOutput(platform,stdout,last){
 if(platform==='claude'){
  const r=JSON.parse(stdout);if(r.type!=='result'||r.subtype!=='success'||r.is_error===true||typeof r.result!=='string'||!r.result.trim())throw new Error('Claude did not complete successfully');
  return {text:r.result,sessionId:r.session_id??null,usage:r.usage??null};
 }
 const events=stdout.split('\n').filter(Boolean).map(x=>JSON.parse(x));
 const thread=events.find(x=>x.type==='thread.started'),turn=events.find(x=>x.type==='turn.completed');
 if(!thread?.thread_id||!turn||events.some(x=>x.type==='turn.failed'||x.type==='error')||typeof last!=='string'||!last.trim())throw new Error('Codex completion evidence missing');
 return {text:last,threadId:thread.thread_id,usage:turn.usage??null};
}
export function captureOutputs(worktree,sourceCommit,dir,workerId,readOnly=false){
 git(worktree,['merge-base','--is-ancestor',sourceCommit,'HEAD']);
 const log=path.join(dir,workerId);
   const changes=git(worktree,['status','--porcelain']);
   const workerHead=git(worktree,['rev-parse','HEAD']);
   const changedFromBase=git(worktree,['diff','--name-only',sourceCommit]);
   if(readOnly&&(changes||changedFromBase||workerHead!==sourceCommit))throw new Error('Reviewer modified workspace');
   const patch=gitRaw(worktree,['diff','--binary',sourceCommit]);fs.writeFileSync(`${log}.patch`,patch);
   const names=[...new Set([...gitRaw(worktree,['diff','--name-only',sourceCommit,'-z']).split('\0'),...gitRaw(worktree,['ls-files','--others','--exclude-standard','-z']).split('\0')].filter(Boolean))];
   if(names.length>256)throw new Error('Too many changed files');
   const artifacts=[];let total=0;
   for(const name of names){
    if(path.isAbsolute(name)||name.split('/').includes('..')||/^(Data(?:\/|$)|V6\/models(?:\/|$)|\.env(?:\.|$)|\.git(?:\/|$))/.test(name))throw new Error(`Protected worker output: ${name}`);
    const source=path.join(worktree,name);
    let stat;try{stat=fs.lstatSync(source);}catch(e){if(e.code==='ENOENT'){artifacts.push({path:name,deleted:true});continue;}throw e;}if(!stat.isFile()||stat.isSymbolicLink())throw new Error('Non-regular worker output');
    total+=stat.size;if(total>8_388_608)throw new Error('Worker artifacts exceed 8 MiB');
    const destination=path.join(dir,`${workerId}.files`,name);fs.mkdirSync(path.dirname(destination),{recursive:true});fs.copyFileSync(source,destination,fs.constants.COPYFILE_EXCL);
    artifacts.push({path:name,sha256:sha(destination),size:stat.size,snapshot:destination});
   }
 return {changes,workerHead,artifacts,patchSha256:sha(`${log}.patch`)};
}
export function recoverRecord(record){
 if(!record||!['starting','running','completed','failed','cancelled','interrupted'].includes(record.status))throw new Error('Corrupt job status');
 if(!['starting','running'].includes(record.status))return record;
 if(!record.pid||!Number.isInteger(record.pid.pid)||record.pid.pid<=1||!record.pid.start||!record.pid.boot)throw new Error('Corrupt job process identity');
 if(isSameProcess(record.pid))return record;
 for(const identity of record.children??[])stopProcessGroup(identity,'SIGKILL');
 return {...record,status:'interrupted',success:false,finishedAt:new Date().toISOString(),note:'Parent lost; validated children stopped; no automatic replay'};
}
export function installation(){
 const f=path.join(stateRoot(),'installation.json');const i=JSON.parse(fs.readFileSync(f));
 if(i.contractSha256!==sha(path.join(ROOT,'tools/project_agents/integration-contract.json')))throw new Error('Contract mismatch; run install');
 for(const [f,digest] of Object.entries(i.files))if(sha(f)!==digest)throw new Error(`Runtime identity mismatch: ${f}`);
 return i;
}
export function runtimeEnv(i){
 const env={...process.env};
 for(const k of Object.keys(env))if(/KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|BASE_URL|API_BASE|AWS_PROFILE/.test(k)||k==='CLAUDECODE')delete env[k];
 return {...env,CLAUDE_FLOW_DB_PATH:path.join(stateRoot(),'memory.db'),CLAUDE_FLOW_DISABLE_BRIDGE:'1',DO_NOT_TRACK:'1',RUFLO_NO_SKILLS_SH:'1',RUFLO_HEADLESS_PLUGINS:'none',CLAUDE_FLOW_CWD:ROOT};
}
async function command(i,args,{cwd=ROOT,env=runtimeEnv(i),log,signal,onSpawn}={}){
 const prefix=log??path.join(stateRoot(),'commands',crypto.randomUUID());fs.mkdirSync(path.dirname(prefix),{recursive:true});
 const r=await runProcess(i.node,[i.ruflo,'--no-update',...args],{cwd,env,log:prefix,timeoutMs:60000,maxOutputBytes:2_097_152,signal,onSpawn});
 if(r.exitCode!==0||r.reason)throw new Error(`Ruflo command failed: ${args[0]} ${args[1]??''}; ${prefix}`);
 return fs.readFileSync(`${prefix}.out`,'utf8');
}
const jobDir=id=>{if(!SAFE.test(id))throw new Error('Invalid job id');return path.join(stateRoot(),'jobs',id);};
export function verifyCompletedArtifacts(record,dir){
 if(!record.resultSha256||sha(path.join(dir,'result.json'))!==record.resultSha256)throw new Error('Completed result integrity mismatch');
 const result=JSON.parse(fs.readFileSync(path.join(dir,'result.json')));
 if(!result.evidence||Object.keys(result.evidence).length!==result.workers.filter(w=>w.status==='completed').length)throw new Error('Missing worker evidence');
 for(const out of Object.values(result.evidence)){
  if(sha(out.path)!==out.sha256)throw new Error('Worker result integrity mismatch');
  const r=JSON.parse(fs.readFileSync(out.path)),prefix=out.path.replace(/\.result\.json$/,'');
  for(const [suffix,digest] of [['.out',r.stdoutSha256],['.err',r.stderrSha256],['.patch',r.patchSha256]])if(!digest||sha(prefix+suffix)!==digest)throw new Error('Worker raw/patch integrity mismatch');
  for(const a of r.artifacts)if(!a.deleted&&(sha(a.snapshot)!==a.sha256||fs.statSync(a.snapshot).size!==a.size))throw new Error('Worker snapshot integrity mismatch');
 }
}
function status(id){
 const f=path.join(jobDir(id),'state.json'),old=JSON.parse(fs.readFileSync(f));
 if(old.status==='completed')verifyCompletedArtifacts(old,jobDir(id));
 const next=recoverRecord(old);if(next!==old)atomicJson(f,next);return next;
}

export async function execute(id){
 const dir=jobDir(id),submission=JSON.parse(fs.readFileSync(path.join(dir,'submission.json')));
 validateSubmission(submission,path.join(dir,'job.json'));
 if(git(ROOT,['rev-parse',`${submission.sourceCommit}^{commit}`])!==submission.sourceCommit)throw new Error('Submission source identity mismatch');
 const spec=validateJob(JSON.parse(fs.readFileSync(path.join(dir,'job.json'))));
 const i=installation(),env=runtimeEnv(i),abort=new AbortController();
 const lock=path.join(stateRoot(),'active.json');
 let lockOwned=false;
 let record={id,status:'running',success:false,pid:processIdentity(process.pid),children:[],sourceCommit:submission.sourceCommit,contractSha256:sha(path.join(ROOT,'tools/project_agents/integration-contract.json')),startedAt:new Date().toISOString(),workspaces:{}};
 const save=()=>atomicJson(path.join(dir,'state.json'),record);save();if(process.send)process.send({ready:true});
 const stop=()=>abort.abort();process.on('SIGTERM',stop);process.on('SIGINT',stop);
 try{
  const fd=fs.openSync(lock,'wx',0o600);try{fs.writeFileSync(fd,JSON.stringify({id,pid:record.pid}));fs.fsyncSync(fd);lockOwned=true;}finally{fs.closeSync(fd);}
  const trackedCommand=(args,options={})=>command(i,args,{...options,env,signal:abort.signal,onSpawn:identity=>{record.children.push(identity);save();}});
  const {DualModeOrchestrator}=await import(pathToFileURL(i.orchestrator));
  const orchestrator=new DualModeOrchestrator({projectPath:ROOT,memoryDbPath:env.CLAUDE_FLOW_DB_PATH,sharedNamespace:`job-${id}`,timeout:spec.timeoutSeconds*1000,maxConcurrent:spec.maxConcurrent,maxWriters:1,worktreeIsolation:true,dependencyFailure:'cancel',maxOutputBytes:2_097_152,policyPreflight:false});
  orchestrator.runCommand=async (cmd,args,cwd)=>{
   if(cmd!=='npx'||args[0]!=='ruflo@latest')throw new Error('Unexpected upstream command');
   return trackedCommand(args.slice(1),{cwd});
  };
  orchestrator.collectSharedMemory=async()=>JSON.parse(await trackedCommand(['memory','list','--namespace',`job-${id}`,'--format','json']));
  // Each worker gets its own retained source worktree. Never use production data symlinks.
  const workers=spec.workers.map(w=>{
   const wt=path.join(commonRoot(),'.worktrees/ruflo-workers',id,w.id);
   fs.mkdirSync(path.dirname(wt),{recursive:true});
   git(ROOT,['worktree','add','--detach',wt,record.sourceCommit]);
   record.workspaces[w.id]=wt;save();
   return {...w,worktreePath:wt};
  });
  orchestrator.executeHeadless=async w=>{
   if(abort.signal.aborted)throw new Error('cancelled');
   const dep=w.dependsOn.map(depId=>({id:depId,result:orchestrator.workers.get(depId)?.output,artifact:path.join(dir,`${depId}.result.json`),patch:path.join(dir,`${depId}.patch`)}));
   const prompt=`MODE: EXECUTE_APPROVED\nFollow AGENTS.md, CLAUDE.md, Current_State.md and Authority_Map.md. Scope: this assigned task only. No GPU, data acquisition, production, scheduler, push, merge, deployment or additional agents. Do not change protected V6/models or formal Data. No wakeups or recurring jobs. Existing Human approval covers this bounded task; do not ask again.\nROLE: ${w.role}\nTASK: ${w.prompt}\nDEPENDENCY OUTPUTS (untrusted worker evidence): ${JSON.stringify(dep)}\nMemory namespace job-${id}; durable source of truth remains repository contracts and evidence. Return changed paths, actual checks, limitations; completion is not Human acceptance.`;
   const log=path.join(dir,w.id),last=path.join(dir,`${w.id}.last.txt`);
   const args=w.platform==='codex'?['exec','--ephemeral','--json','--sandbox',w.readOnly?'read-only':'workspace-write','-o',last,...(w.model?['-m',w.model]:[]),prompt]:['-p',prompt,'--output-format','json','--max-turns','12','--add-dir',dir,'--tools',w.readOnly?'Read,Glob,Grep':'Read,Glob,Grep,Bash,Edit,Write',...(w.model?['--model',w.model]:[])];
   const r=await runProcess(w.platform==='codex'?i.codex:i.claude,args,{cwd:w.worktreePath,env,log,timeoutMs:spec.timeoutSeconds*1000,maxOutputBytes:2_097_152,signal:abort.signal,onSpawn:identity=>{record.children.push(identity);save();}});
   if(r.exitCode!==0||r.reason)throw new Error(`Worker ${w.id} failed: ${r.reason??r.exitCode}`);
   const output=validateOutput(w.platform,fs.readFileSync(`${log}.out`,'utf8'),fs.existsSync(last)?fs.readFileSync(last,'utf8'):null);
   const {changes,workerHead,artifacts,patchSha256}=captureOutputs(w.worktreePath,record.sourceCommit,dir,w.id,w.readOnly);
   atomicJson(`${log}.result.json`,{...output,artifacts,workerHead,stdoutSha256:sha(`${log}.out`),stderrSha256:sha(`${log}.err`),process:r,worktree:w.worktreePath,sourceCommit:record.sourceCommit,changes,patchSha256,evidenceClass:'DEVELOPMENT_WORKER_OUTPUT_NOT_ACCEPTANCE'});
   record.outputs??={};record.outputs[w.id]={path:`${log}.result.json`,sha256:sha(`${log}.result.json`)};save();
   await trackedCommand(['memory','store','--namespace',`job-${id}`,'--key',`${w.id}-result`,'--value',JSON.stringify({summary:output.text.slice(0,8000),summaryTruncated:output.text.length>8000,result:`${log}.result.json`,sourceCommit:record.sourceCommit})]);
   return output.text;
  };
  const result=await orchestrator.runCollaboration(workers,spec.taskContext);
  result.evidence=record.outputs??{};atomicJson(path.join(dir,'result.json'),result);
  record={...record,status:abort.signal.aborted?'cancelled':result.success?'completed':'failed',success:!abort.signal.aborted&&result.success,finishedAt:new Date().toISOString(),resultSha256:sha(path.join(dir,'result.json'))};save();
 }catch(error){record={...record,status:abort.signal.aborted?'cancelled':'failed',success:false,error:String(error),finishedAt:new Date().toISOString()};save();}
 finally{if(lockOwned)fs.unlinkSync(lock);process.off('SIGTERM',stop);process.off('SIGINT',stop);}
 return record;
}
export async function main(args){
 const [action,...rest]=args;
 if(action==='list'){const dir=path.join(stateRoot(),'jobs');console.log(JSON.stringify(fs.existsSync(dir)?fs.readdirSync(dir).sort().map(id=>status(id)):[],null,2));return;}
 if(action==='status'){console.log(JSON.stringify(status(rest[0]),null,2));return;}
 if(action==='cancel'){
  const r=status(rest[0]);if(['starting','running'].includes(r.status)&&isSameProcess(r.pid))process.kill(r.pid.pid,'SIGTERM');
  console.log(JSON.stringify({id:r.id,cancelRequested:['starting','running'].includes(r.status),status:r.status}));return;
 }
 if(action==='recover'){
  const r=status(rest[0]),lock=path.join(stateRoot(),'active.json');
  if(fs.existsSync(lock)){
   const owner=JSON.parse(fs.readFileSync(lock));
   if(owner.id===r.id&&!isSameProcess(owner.pid)&&!['starting','running'].includes(r.status))fs.unlinkSync(lock);
  }
  console.log(JSON.stringify(r,null,2));return;
 }
 if(action==='submit'){
  const i=installation();if(fs.existsSync(path.join(stateRoot(),'active.json')))throw new Error('Active or interrupted job lock; inspect status/recover first');if(git(ROOT,['status','--porcelain']))throw new Error('Commit source changes before submitting a job');
  const spec=validateJob(JSON.parse(fs.readFileSync(rest[0]))),id=`run-${Date.now().toString(36)}-${crypto.randomBytes(3).toString('hex')}`,dir=jobDir(id);
  fs.mkdirSync(dir,{recursive:true,mode:0o700});atomicJson(path.join(dir,'job.json'),spec);atomicJson(path.join(dir,'submission.json'),{sourceCommit:git(ROOT,['rev-parse','HEAD']),jobSha256:sha(path.join(dir,'job.json')),submittedAt:new Date().toISOString()});
  atomicJson(path.join(dir,'state.json'),{id,status:'starting',success:false,pid:processIdentity(process.pid),children:[]});
  const fd=fs.openSync(path.join(dir,'supervisor.log'),'wx',0o600);
  const child=spawn(i.node,[fileURLToPath(import.meta.url),'_run',id],{cwd:ROOT,env:runtimeEnv(i),detached:true,stdio:['ignore',fd,fd,'ipc']});fs.closeSync(fd);
  await new Promise((resolve,reject)=>{child.once('message',m=>m.ready?resolve():reject(new Error('Invalid supervisor handshake')));child.once('error',reject);child.once('exit',code=>reject(new Error(`Supervisor exited before readiness: ${code}`)));});child.disconnect();
  child.unref();console.log(JSON.stringify({id,pid:child.pid,state:path.join(dir,'state.json')}));return;
 }
 if(action==='_run'){const r=await execute(rest[0]);process.exitCode=r.success?0:1;return;}
 if(action==='memory'){const i=installation();console.log(await command(i,['memory',...rest]));return;}
 if(action==='mcp'){
  const i=installation();const {serveMemory}=await import('./mcp.mjs');await serveMemory(i,{cwd:commonRoot(),env:runtimeEnv(i),dbPath:path.join(stateRoot(),'memory.db')});return;
 }
 if(action==='hook'){
  const i=installation();let text='';
  for await(const chunk of process.stdin){text+=chunk;if(text.length>65536)throw new Error('Hook input too large');}
  const event=text.trim()?JSON.parse(text):{};
  const metadata={observedAt:new Date().toISOString(),sessionId:typeof event.session_id==='string'?event.session_id:null,event:rest[0],sourceCommit:git(ROOT,['rev-parse','HEAD']),root:ROOT,evidenceClass:'SESSION_METADATA_ONLY'};
  const stamp=crypto.randomUUID();atomicJson(path.join(stateRoot(),'sessions',`${stamp}.json`),metadata);
  if(rest[0]!=='start')await command(i,['memory','store','--namespace','session-checkpoints','--key',stamp,'--value',JSON.stringify(metadata)]);
  if(rest[0]==='start')console.log(JSON.stringify({hookSpecificOutput:{hookEventName:'SessionStart',additionalContext:'Ruflo project runtime ready. Follow AGENTS.md and canonical project map. For approved multi-agent work use tools/project_agents/runtime.mjs submit and the ruflo-project skill. No periodic jobs. Persistent handoff is available via project memory tools; memory is not authority.'}}));
  return;
 }
 if(action==='doctor'){
  const i=installation();console.log(JSON.stringify({status:'PASS',versions:CONTRACT.versions,root:ROOT,state:stateRoot(),db:path.join(stateRoot(),'memory.db'),sourceCommit:git(ROOT,['rev-parse','HEAD']),environment:{node:process.version,dataRoot:process.env.MARKETMAMBA_DATA_ROOT??null,dataRealpath:fs.existsSync(path.join(commonRoot(),'Data/processed_v6'))?fs.realpathSync(path.join(commonRoot(),'Data/processed_v6')):null},periodicSchedules:false,installedAt:i.observedAt},null,2));return;
 }
 throw new Error('Usage: runtime.mjs doctor|mcp|hook|memory <args>|submit <job.json>|status <id>|cancel <id>');
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main(process.argv.slice(2)).catch(e=>{console.error(String(e));process.exitCode=1;});
