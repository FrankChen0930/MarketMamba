// Enforce the project's memory-only MCP boundary, including direct tools/call.
import readline from 'node:readline';
import {spawn} from 'node:child_process';
export const MEMORY_TOOLS=new Set(['memory_store','memory_retrieve','memory_search','memory_list','memory_stats']);
export function filterRequest(message,dbPath){
 if(!message||message.jsonrpc!=='2.0'||typeof message.method!=='string')throw new Error('Invalid MCP request');
 if(message.method==='tools/call'){
  if(!MEMORY_TOOLS.has(message.params?.name))throw new Error('Tool not permitted by project memory boundary');
  return {...message,params:{...message.params,arguments:{...message.params.arguments,dbPath}}};
 }
 if(!['initialize','notifications/initialized','notifications/cancelled','ping','tools/list'].includes(message.method))throw new Error('MCP method not permitted');
 return message;
}
export async function serveMemory(i,{cwd,env,dbPath}){
 const child=spawn(i.node,[i.ruflo,'mcp','start','--tools',[...MEMORY_TOOLS].join(',')],{cwd,env,stdio:['pipe','pipe','inherit']});
 const input=readline.createInterface({input:process.stdin});
 input.on('line',line=>{
  let message;try{
   if(Buffer.byteLength(line)>1_048_576)throw new Error('MCP request too large');
   message=JSON.parse(line);child.stdin.write(JSON.stringify(filterRequest(message,dbPath))+'\n');
  }catch(error){if(message?.id!==undefined)process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:message.id,error:{code:-32602,message:String(error)}})+'\n');}
 });
 input.once('close',()=>child.stdin.end());child.stdout.pipe(process.stdout);
 const stop=()=>child.kill('SIGTERM');process.on('SIGTERM',stop);process.on('SIGINT',stop);
 try{await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',code=>{process.exitCode=code??1;resolve();});});}
 finally{input.close();process.off('SIGTERM',stop);process.off('SIGINT',stop);}
}
