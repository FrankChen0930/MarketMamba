// Linux process identity and bounded groups; no shell and no PID-only signalling.
import fs from 'node:fs';
import {spawn} from 'node:child_process';
export function processIdentity(pid) {
 try {
  const raw=fs.readFileSync(`/proc/${pid}/stat`,'utf8');
  const fields=raw.slice(raw.lastIndexOf(')')+2).split(' ');
  return {pid:Number(pid),start:fields[19],group:Number(fields[2]),state:fields[0],boot:fs.readFileSync('/proc/sys/kernel/random/boot_id','utf8').trim()};
 } catch {return null;}
}
export function isSameProcess(identity) {
 if(!identity || !Number.isInteger(identity.pid) || identity.pid<=1 || !identity.start || !identity.boot)return false;
 const current=processIdentity(identity.pid);
 return !!current && current.state!=='Z' && current.start===identity.start && current.boot===identity.boot;
}
export function stopProcessGroup(identity,signal='SIGTERM') {
 if(!isSameProcess(identity)){
  // A dead leader is not proof its detached group has gone. Never release
  // recovery's lock while unidentifiable descendants may still write.
  if(identity?.group===identity?.pid&&identity.pid>1){
   for(const entry of fs.readdirSync('/proc'))if(/^\d+$/.test(entry)){
    const current=processIdentity(Number(entry));
    if(current&&current.state!=='Z'&&current.group===identity.group)throw new Error('UNKNOWN orphan group: leader identity unavailable; retain job lock and inspect processes');
   }
  }
  return false;
 }
 if(identity.group!==identity.pid)throw new Error('Refusing to signal non-owned process group');
 try {process.kill(-identity.pid,signal);return true;}catch(e){if(e.code==='ESRCH')return false;throw e;}
}
export async function runProcess(command,args,{cwd,env=process.env,log,timeoutMs,maxOutputBytes,signal,onSpawn}) {
 if(signal?.aborted)throw new Error('cancelled before spawn');
 if(!Number.isFinite(timeoutMs)||timeoutMs<=0||!Number.isSafeInteger(maxOutputBytes)||maxOutputBytes<=0)throw new Error('Invalid process limits');
 const out=fs.openSync(`${log}.out`,'wx',0o600),err=fs.openSync(`${log}.err`,'wx',0o600);
 const child=spawn(command,args,{cwd,env,stdio:['ignore','pipe','pipe'],detached:true});
 let bytes=0,reason=null,killTimer,timer,identity;
 const stop=why=>{
  if(reason)return;reason=why;
  // This live ChildProcess owns the freshly created detached group.
  // Recovery's PID-reuse/UNKNOWN rule is for a different supervisor lifetime.
  try{process.kill(-child.pid,'SIGTERM');}catch(e){if(e.code!=='ESRCH')reason='cleanup_failed';}
  // Even if the leader exits, kill its original group before returning.
  killTimer=setTimeout(()=>{try{process.kill(-child.pid,'SIGKILL');}catch{}},200);
 };
 const consume=fd=>data=>{
  const remaining=Math.max(0,maxOutputBytes-bytes);if(remaining)fs.writeSync(fd,data.subarray(0,remaining));
  bytes+=data.length;if(bytes>maxOutputBytes)stop('output_limit');
 };
 child.stdout.on('data',consume(out));child.stderr.on('data',consume(err));
 const abort=()=>stop('cancelled');signal?.addEventListener('abort',abort,{once:true});
 try {
  const result=await new Promise((resolve,reject)=>{
   child.once('error',reject);
   child.once('spawn',()=>{
    identity=processIdentity(child.pid);
    if(!identity){stop('identity_unavailable');return;}
    try{onSpawn?.(identity);}catch(e){stop('state_write_failure');reject(e);}
    timer=setTimeout(()=>stop('timeout'),timeoutMs);
    if(signal?.aborted)abort();
   });
   child.once('close',(exitCode,exitSignal)=>resolve({exitCode,exitSignal}));
  });
  return {...result,reason,identity,bytes};
 } finally {
  clearTimeout(timer);clearTimeout(killTimer);signal?.removeEventListener('abort',abort);
  if(child.pid){try{process.kill(-child.pid,'SIGKILL');}catch{}}
  fs.fsyncSync(out);fs.fsyncSync(err);fs.closeSync(out);fs.closeSync(err);
 }
}
