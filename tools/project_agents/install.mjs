#!/usr/bin/env node
// No download/global mutation: record and verify the already installed runtime.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {ROOT,sha,atomicJson,stateRoot,git,main} from './runtime.mjs';
const base=process.argv[2]??'/home/frank/projects/ruflo-evaluation/tools';
const ruflo=path.join(base,'ruflo/node_modules/ruflo/bin/ruflo.js');
const cli=path.join(base,'ruflo/node_modules/@claude-flow/cli');
const adapter=path.join(base,'ruflo-codex/node_modules/@claude-flow/codex');
const contract=JSON.parse(fs.readFileSync(path.join(ROOT,'tools/project_agents/integration-contract.json')));
const packages={ruflo:[path.join(base,'ruflo/node_modules/ruflo/package.json'),contract.versions.ruflo],cli:[path.join(cli,'package.json'),contract.versions.cli],codexAdapter:[path.join(adapter,'package.json'),contract.versions.codexAdapter]};
for(const [name,[file,version]] of Object.entries(packages))if(JSON.parse(fs.readFileSync(file)).version!==version)throw new Error(`STOP ${name} version mismatch`);
const orchestrator=path.join(adapter,'dist/dual-mode/orchestrator.js');
const node=fs.realpathSync(process.execPath),claude=fs.realpathSync('/home/frank/.local/bin/claude'),codex=fs.realpathSync('/home/frank/.local/bin/codex');
const files=Object.fromEntries([node,claude,codex,ruflo,orchestrator,path.join(cli,'bin/cli.js'),...Object.values(packages).map(([f])=>f)].map(f=>[f,sha(f)]));
const record={schemaVersion:1,observedAt:new Date().toISOString(),sourceCommit:git(ROOT,['rev-parse','HEAD']),node,ruflo,cli,adapter,orchestrator,claude,codex,contractSha256:sha(path.join(ROOT,'tools/project_agents/integration-contract.json')),files,environment:{node:process.version,dataRoot:process.env.MARKETMAMBA_DATA_ROOT??null},evidenceClass:'LOCAL_RUNTIME_IDENTITY_NOT_DEPLOYMENT'};
atomicJson(path.join(stateRoot(),'installation.json'),record);
await main(['memory','init','--backend','sqlite']);
await main(['doctor']);
