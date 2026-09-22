"""Formal E1 two-seed runner; CUDA is mandatory outside explicit smoke mode."""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from V6.experimental.v7_corrected_e5_replication import CorrectedE5Config
from V6.experimental.v7_corrected_e5_train import train_seed
from V6.experimental.v7_e1_checkpoint import DurableCheckpointStore,LifecycleCheckpointAdapter,atomic_json
from V6.experimental.v7_e1_contract import SEEDS,build_contract,validate_contract
from V6.experimental.v7_e1_matrix import e1_config,validate_manifest
def gpu_sample():
 try:
  row=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total",
   "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=2,check=True).stdout.splitlines()[0]
  util,used,total=map(float,row.split(","))
  return {"gpu_utilization_percent":util,"gpu_memory_used_mib":used,"gpu_memory_total_mib":total}
 except (OSError,ValueError,subprocess.SubprocessError,IndexError): return {}
def append_event(path,event,**values):
 row={"time":datetime.now(timezone.utc).isoformat(),"event":event,**values}
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("a",encoding="utf-8") as f:
  f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
 print(json.dumps(row,ensure_ascii=False),flush=True)
def run_seed(matrix,output,contract,base_config,seed,device,smoke_steps=None,model_factory=None):
 if seed not in SEEDS: raise ValueError("E1 formal seeds are 17 and 29 only")
 if device.type!="cuda" and smoke_steps is None: raise ValueError("formal E1 requires CUDA")
 config=e1_config(base_config); matrix_doc=json.loads((matrix/"manifest.json").read_text())
 validate_manifest(matrix_doc,contract,config)
 seed_root=output/f"seed-{seed}"; events=seed_root/"telemetry.jsonl"
 local=(output/".staging"/str(seed)) if str(output).startswith("/tmp") else Path("/content")/f"v7-e1-seed-{seed}-staging"
 durable=DurableCheckpointStore(seed_root/"checkpoints",local,contract["contract_sha256"],
                                matrix_doc["logical_identity"],seed)
 adapter=LifecycleCheckpointAdapter(durable,config.sha256)
 state=adapter.load(config.sha256,matrix_doc["logical_identity"])
 if state is None: state=adapter.recover()
 append_event(events,"resume_scan",seed=seed,status="RESUME" if state else "FRESH")
 start=time.perf_counter(); append_event(events,"seed_started",seed=seed,**gpu_sample())
 kwargs={}
 if model_factory is not None: kwargs["model_factory"]=model_factory
 def epoch_telemetry(epoch,values):
  append_event(events,"epoch_telemetry",seed=seed,epoch=epoch,**values,
               **gpu_sample())
 result=train_seed(matrix,seed_root,config,seed=seed,device=device,smoke_steps=smoke_steps,
  result_path=seed_root/"result.json",predictions_path=None if smoke_steps else seed_root/"predictions.parquet",
  checkpoint_store=adapter,
  matrix_manifest_validator=lambda doc,cfg:validate_manifest(doc,contract,cfg),
  telemetry_hook=epoch_telemetry,telemetry_provider=adapter.snapshot_epoch,
  **kwargs)
 append_event(events,"seed_finished",seed=seed,status=result["status"],
  total_seconds=time.perf_counter()-start,steps=result.get("step"),**gpu_sample())
 return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--matrix",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
 p.add_argument("--contract",type=Path,required=True); p.add_argument("--incumbent-contract",type=Path,required=True)
 p.add_argument("--decision",type=Path,required=True); p.add_argument("--feature-manifest",type=Path,required=True)
 p.add_argument("--seed",type=int,choices=SEEDS); p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
 p.add_argument("--smoke-steps",type=int)
 a=p.parse_args(argv); contract=json.loads(a.contract.read_text()); validate_contract(contract)
 expected=build_contract(a.incumbent_contract,a.decision,a.feature_manifest)
 if contract["contract_sha256"]!=expected["contract_sha256"]: raise ValueError("E1 contract file drift")
 base=CorrectedE5Config.from_contract(a.incumbent_contract,a.feature_manifest)
 a.output.mkdir(parents=True,exist_ok=True); summary={}
 for seed in ([a.seed] if a.seed else list(SEEDS)):
  summary[str(seed)]=run_seed(a.matrix,a.output,contract,base,seed,torch.device(a.device),a.smoke_steps)
 atomic_json(a.output/"run-summary.json",{"schema_version":"v7-e1-run-summary-v1",
  "contract_sha256":contract["contract_sha256"],"seeds":summary,
  "strict_phase0":"STOP","historical_simulation_readiness":"PASS"})
 return 0
if __name__=="__main__": raise SystemExit(main())
