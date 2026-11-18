// Real stdio MCP <-> CLI cross-process/store identity check; no model calls.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import readline from 'node:readline';
import {spawn} from 'node:child_process';
import {ROOT,installation,runtimeEnv,stateRoot,atomicJson} from './runtime.mjs';
import {runProcess} from './process.mjs';
const i=installation(),env=runtimeEnv(i),stamp=Date.now().toString(36),namespace='integration-validation',cliKey=`cli-${stamp}`,mcpKey=`mcp-${stamp}`,marker=`memory-roundtrip-${stamp}`;
const runtime=path.join(ROOT,'tools/project_agents/runtime.mjs');
const logDir=path.join(stateRoot(),'verification',stamp);fs.mkdirSync(logDir,{recursive:true});
const cli=async(args,name)=>{
 const r=await runProcess(i.node,[runtime,'memory',...args],{cwd:ROOT,env,log:path.join(logDir,name),timeoutMs:60000,maxOutputBytes:1048576});assert.equal(r.exitCode,0);assert.equal(r.reason,null);return fs.readFileSync(path.join(logDir,`${name}.out`),'utf8');
};
await cli(['store','--namespace',namespace,'--key',cliKey,'--value',marker],'cli-store');
const child=spawn(i.node,[runtime,'mcp'],{cwd:ROOT,env,stdio:['pipe','pipe',fs.openSync(path.join(logDir,'mcp.err'),'w')]});
const pending=new Map();let counter=0;
const lines=readline.createInterface({input:child.stdout});lines.on('line',line=>{try{const r=JSON.parse(line);const p=pending.get(r.id);if(p){pending.delete(r.id);p.resolve(r);}}catch{}});
const request=(method,params)=>new Promise((resolve,reject)=>{const id=++counter;const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`MCP timeout ${method}`));},60000);pending.set(id,{resolve:r=>{clearTimeout(timer);resolve(r);}});child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');});
try{
 const init=await request('initialize',{protocolVersion:'2025-03-26',capabilities:{},clientInfo:{name:'marketmamba-verification',version:'1'}});assert.ok(init.result);
 child.stdin.write(JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})+'\n');
 const tools=await request('tools/list',{});assert.ok(tools.result.tools.length>0);assert.ok(tools.result.tools.every(t=>t.name.startsWith('memory_')));
 const blocked=await request('tools/call',{name:'agent_execute',arguments:{agentId:'fixture',prompt:'never execute'}});assert.ok(blocked.error);
 const retrieved=await request('tools/call',{name:'memory_retrieve',arguments:{namespace,key:cliKey}});atomicJson(path.join(logDir,'mcp-retrieve.json'),retrieved);assert.ok(JSON.stringify(retrieved.result).includes(marker),JSON.stringify(retrieved));
 const stored=await request('tools/call',{name:'memory_store',arguments:{namespace,key:mcpKey,value:marker,provenance_type:'tool_result'}});atomicJson(path.join(logDir,'mcp-store.json'),stored);assert.ok(stored.result&&!stored.result.isError,JSON.stringify(stored));
 const back=await cli(['retrieve','--namespace',namespace,'--key',mcpKey],'cli-retrieve');assert.ok(back.includes(marker),back);
 const report={status:'PASS',observedAt:new Date().toISOString(),namespace,marker,root:ROOT,db:path.join(stateRoot(),'memory.db'),checks:['stdio initialize','memory tool discovery','blocked direct agent_execute','CLI store -> MCP retrieve','MCP store -> CLI retrieve'],evidenceClass:'LOCAL_COMPONENT_ENGINEERING'};atomicJson(path.join(logDir,'report.json'),report);console.log(JSON.stringify(report,null,2));
}finally{child.stdin.end();child.kill('SIGTERM');lines.close();}
