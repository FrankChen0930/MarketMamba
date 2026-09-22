"""Crash-safe E1 checkpoints adapted from the verified confirmation runner."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,os,shutil,tempfile,time,uuid
from pathlib import Path
from typing import Any,Mapping
import torch

FORMAT="marketmamba-v7-e1-checkpoint-v1"
POINTER_SCHEMA="marketmamba-v7-e1-checkpoint-pointer-v1"
REQUIRED={"format","contract_sha256","matrix_sha256","seed","epoch","batch","step",
          "phase","terminal","model","optimizer","scheduler","rng"}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(4*1024**2),b""): h.update(block)
 return h.hexdigest()

def atomic_json(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
 try:
  with tmp.open("w",encoding="utf-8") as f:
   json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
   f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path); _fsync_dir(path.parent)
 finally: tmp.unlink(missing_ok=True)

def _fsync_dir(path):
 try:
  fd=os.open(path,os.O_RDONLY)
  try: os.fsync(fd)
  finally: os.close(fd)
 except OSError: pass

def _identity(state):
 return (state.get("contract_sha256"),state.get("matrix_sha256"),state.get("seed"))

def validate_state(state,identity):
 missing=REQUIRED-set(state)
 if missing: raise ValueError("incomplete E1 checkpoint: "+", ".join(sorted(missing)))
 if state.get("format")!=FORMAT: raise ValueError("unsupported E1 checkpoint format")
 if _identity(state)!=identity: raise ValueError("E1 checkpoint identity mismatch")
 if state["phase"] not in {"train","validate","finished"}: raise ValueError("invalid phase")
 if min(int(state[k]) for k in ("epoch","batch","step"))<0: raise ValueError("negative progress")
 if bool(state["terminal"])!=(state["phase"]=="finished"): raise ValueError("terminal/phase mismatch")

class DurableCheckpointStore:
 """Publish pointer only after body is fsynced, hashed, copied and reloaded."""
 def __init__(self,drive_root,staging_root,contract_sha256,matrix_sha256,seed,
              copy_hook=None):
  self.root=Path(drive_root); self.root.mkdir(parents=True,exist_ok=True)
  self.staging=Path(staging_root); self.staging.mkdir(parents=True,exist_ok=True)
  self.pointer=self.root/"checkpoint.json"
  self.identity=(str(contract_sha256),str(matrix_sha256),int(seed))
  self.last_io={}
  self.copy_hook=copy_hook

 def manifest(self):
  if not self.pointer.exists(): return {}
  try:
   value=json.loads(self.pointer.read_text(encoding="utf-8"))
  except (OSError,json.JSONDecodeError): return {}
  return value if isinstance(value,dict) and value.get("schema_version")==POINTER_SCHEMA else {}

 def _entry_state(self,entry):
  if not entry or set(entry)<{"file","sha256","bytes"}: return None
  path=self.root/entry["file"]
  if not path.is_file() or path.stat().st_size!=entry["bytes"] or sha256(path)!=entry["sha256"]:
   return None
  try: state=torch.load(path,map_location="cpu",weights_only=False)
  except Exception: return None
  try: validate_state(state,self.identity)
  except ValueError: return None
  return state

 def load(self,role="latest",allow_previous=True):
  manifest=self.manifest(); roles=[role]
  if role=="latest" and allow_previous: roles.append("previous")
  for candidate in roles:
   state=self._entry_state(manifest.get(candidate))
   if state is not None: return state
  return None

 def save(self,state,best=False,interrupt_at=None):
  state=dict(state); validate_state(state,self.identity)
  local_started=time.perf_counter()
  old=self.manifest(); name="state-"+uuid.uuid4().hex+".pt"
  local=self.staging/(name+".local"); partial=self.root/(name+".partial"); dest=self.root/name
  try:
   torch.save(state,local)
   with local.open("rb") as f: os.fsync(f.fileno())
   local_sha=sha256(local)
   local_seconds=time.perf_counter()-local_started
   drive_started=time.perf_counter()
   if interrupt_at=="after_local_body": raise RuntimeError("injected after local body")
   with local.open("rb") as src,partial.open("wb") as out:
    shutil.copyfileobj(src,out,4*1024**2); out.flush(); os.fsync(out.fileno())
   if self.copy_hook: self.copy_hook(partial)
   if interrupt_at=="before_drive_flush": raise RuntimeError("injected before Drive publish")
   if sha256(partial)!=local_sha: raise OSError("Drive checkpoint copy verification failed")
   os.replace(partial,dest); _fsync_dir(self.root)
   reloaded=torch.load(dest,map_location="cpu",weights_only=False)
   validate_state(reloaded,self.identity)
   self.last_io={"local_seconds":local_seconds,
                 "drive_seconds":time.perf_counter()-drive_started}
   entry={"file":name,"sha256":local_sha,"bytes":dest.stat().st_size,
          "saved_at":datetime.now(timezone.utc).isoformat(),
          "epoch":state["epoch"],"batch":state["batch"],"step":state["step"],
          "phase":state["phase"],"terminal":state["terminal"]}
   updated={"schema_version":POINTER_SCHEMA,"latest":entry,
            "previous":old.get("latest"),"best":entry if best else old.get("best")}
   if interrupt_at=="before_pointer": raise RuntimeError("injected before pointer")
   atomic_json(self.pointer,updated)
   self._prune(updated)
   return entry
  finally:
   local.unlink(missing_ok=True); partial.unlink(missing_ok=True)

 def _prune(self,manifest):
  keep={e["file"] for key in ("latest","previous","best")
        if (e:=manifest.get(key))}
  for path in self.root.glob("state-*.pt"):
   if path.name not in keep: path.unlink()

 def scan_orphans(self):
  referenced={e["file"] for e in self.manifest().values() if isinstance(e,dict) and "file" in e}
  found=[]
  for path in self.root.glob("state-*.pt"):
   if path.name in referenced: continue
   entry={"file":path.name,"sha256":sha256(path),"bytes":path.stat().st_size}
   state=self._entry_state(entry)
   if state is not None:
    found.append({"status":"RECOVERABLE_ORPHAN_CANDIDATE",**entry,
     "epoch":state["epoch"],"batch":state["batch"],"step":state["step"],
     "phase":state["phase"],"terminal":state["terminal"]})
  return sorted(found,key=lambda x:(x["step"],x["file"]),reverse=True)

 def recover_latest_orphan(self):
  candidates=self.scan_orphans()
  if not candidates: return None
  candidate=candidates[0]; old=self.manifest()
  updated={"schema_version":POINTER_SCHEMA,"latest":candidate,
           "previous":old.get("latest") or old.get("previous"),
           "best":old.get("best"),
           "recovery":{"mode":"validated_orphan","candidate":candidate["file"],
                       "recovered_at":datetime.now(timezone.utc).isoformat(),
                       "best_promoted":False}}
  atomic_json(self.pointer,updated)
  return candidate

class LifecycleCheckpointAdapter:
 """Expose incumbent lifecycle calls over the E1 durable store."""
 def __init__(self,store,lifecycle_config_sha256=None):
  self.store=store; self.pointer=store.pointer
  self.lifecycle_config_sha256=lifecycle_config_sha256 or store.identity[0]
  self._wall=time.perf_counter(); self._cpu=time.process_time()
  self._telemetry=self._empty_telemetry()
  self._last_step=0; self._last_validation=0
 def _empty_telemetry(self):
  return {"train_seconds":0.,"validation_seconds":0.,"checkpoint_seconds":0.,
   "local_checkpoint_seconds":0.,"drive_checkpoint_seconds":0.,
   "data_wait_proxy_seconds":0.,"cpu_seconds":0.,"updates":0,
   "validation_batches":0}
 def save(self,state,best=False):
  before=time.perf_counter(); cpu_now=time.process_time()
  elapsed=max(0.,before-self._wall)
  phase=state.get("phase")
  if phase=="validation": self._telemetry["validation_seconds"]+=elapsed
  else: self._telemetry["train_seconds"]+=elapsed
  self._telemetry["data_wait_proxy_seconds"]+=elapsed
  self._telemetry["cpu_seconds"]+=max(0.,cpu_now-self._cpu)
  step=int(state.get("step",0)); val=int(state.get("validation_index",0))
  self._telemetry["updates"]+=max(0,step-self._last_step)
  self._telemetry["validation_batches"]+=max(0,val-self._last_validation)
  state["e1_adapter_telemetry"]=dict(self._telemetry)
  phase="finished" if state.get("terminal") else (
   "validate" if state.get("phase")=="validation" else "train")
  wrapped=checkpoint_state(self.store.identity[0],self.store.identity[1],
   self.store.identity[2],epoch=int(state["epoch"]),batch=int(state["batch"]),
   step=int(state["step"]),phase=phase,terminal=bool(state.get("terminal")),
   model=state.get("model_state",{}),optimizer=state.get("optimizer_state",{}),
   scheduler=state.get("scheduler_state",{}),rng=state.get("rng_state",{}),
   payload=state)
  entry=self.store.save(wrapped,best=best)
  checkpoint_elapsed=time.perf_counter()-before
  self._telemetry["checkpoint_seconds"]+=checkpoint_elapsed
  self._telemetry["local_checkpoint_seconds"]+=self.store.last_io.get("local_seconds",0.)
  self._telemetry["drive_checkpoint_seconds"]+=self.store.last_io.get("drive_seconds",0.)
  self._wall=time.perf_counter(); self._cpu=time.process_time()
  self._last_step=step; self._last_validation=val
  return entry
 def load(self,contract_sha256,matrix_sha256,best=False):
  if contract_sha256!=self.lifecycle_config_sha256 or matrix_sha256!=self.store.identity[1]:
   raise ValueError("E1 checkpoint identity mismatch")
  wrapped=self.store.load("best" if best else "latest")
  if wrapped is None: return None
  payload=wrapped["payload"]
  saved=dict(payload.get("e1_adapter_telemetry",self._empty_telemetry()))
  if not any(float(value) for value in self._telemetry.values()):
   self._telemetry=saved
  self._last_step=int(payload.get("step",0)); self._last_validation=int(payload.get("validation_index",0))
  return payload
 def snapshot_epoch(self):
  value=dict(self._telemetry)
  value["total_seconds"]=value["train_seconds"]+value["validation_seconds"]+value["checkpoint_seconds"]
  value["updates_per_second"]=value["updates"]/max(value["train_seconds"],1e-12)
  self._telemetry=self._empty_telemetry()
  # Step is global across epochs.  Resetting it made the next epoch report all
  # prior updates again; validation_index, in contrast, restarts every epoch.
  self._last_validation=0
  self._wall=time.perf_counter(); self._cpu=time.process_time()
  return value

 def manifest(self): return self.store.manifest()
 def recover(self):
  candidate=self.store.recover_latest_orphan()
  return None if candidate is None else self.load(self.lifecycle_config_sha256,self.store.identity[1])

def checkpoint_state(contract_sha256,matrix_sha256,seed,**values):
 state={"format":FORMAT,"contract_sha256":str(contract_sha256),
  "matrix_sha256":str(matrix_sha256),"seed":int(seed),"epoch":0,"batch":0,
  "step":0,"phase":"train","terminal":False,"model":{},"optimizer":{},
  "scheduler":{},"rng":{},**values}
 validate_state(state,(str(contract_sha256),str(matrix_sha256),int(seed)))
 return state
