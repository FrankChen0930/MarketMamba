"""Run the ordered V7 capacity ladder; rerun the same command to resume."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import torch
from v7_experiment_storage import atomic_json, digest, fingerprint, verify_matrix
from v7_experiment_data import PackedIndex, FastGraph, FastDataset
from v7_experiment_model import ExperimentConfig
from v7_experiment_train import run_experiment
from v7_integrated_train import (PreparedDataIndex,KnowledgeGraphCSR,official_ssd_callable,
    validate_protocol,validate_feature_metadata,validate_frozen_dates,_runtime_versions)

STAGES=("E0-small","E1-temporal3","E2-bidirectional2","E3-width64","E4-width128","E5-state32")

def settings_from(args):
    return dict(epochs=args.epochs,seed=args.seed,learning_rate=args.learning_rate,weight_decay=.01,
                warmup_fraction=.15,patience=args.patience,min_epochs=5,min_delta=0.,
                grad_clip=1.,precision=args.precision,prefetch=True,
                checkpoint_interval=100,checkpoint_seconds=180,progress_interval=100)

def resolve_spec(stage, completed):
    def best(names):
        viable=[completed[n] for n in names if n in completed and completed[n]["status"]=="complete"]
        if not viable:
            raise ValueError("no completed prerequisite experiment")
        return max(viable,key=lambda row:row["best_rank_ic_5d"])["config"].copy()
    cfg=asdict(ExperimentConfig())
    if stage==1:cfg["temporal_layers"]=3
    elif stage==2:
        cfg=best(STAGES[:2]);cfg.update(forward_layers=2,reverse_layers=2)
    elif stage in (3,4):
        cfg=best(STAGES[:3]);cfg["d_model"]=64 if stage==3 else 128
    elif stage==5:
        cfg=best(STAGES[:5]);cfg["d_state"]=32
    # JSON tuple/list round trips must have identical configuration semantics.
    return {"id":STAGES[stage],"config":json.loads(json.dumps(cfg))}

def normalize_spec(spec):
    out={"id":spec["id"],"config":dict(spec["config"])}
    for key in ("group_dims","horizons"):
        out["config"][key]=tuple(out["config"][key])
    return out

def stage_matrix(source, destination, saved):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():
        return source
    destination.mkdir(parents=True,exist_ok=True)
    for name,sha in saved["files"].items():
        dest=destination/name
        if dest.is_file() and digest(dest)==sha:
            continue
        tmp=dest.with_suffix(dest.suffix+".copying")
        shutil.copyfile(source/name,tmp)
        if digest(tmp)!=sha:raise OSError("local matrix copy failed: "+name)
        os.replace(tmp,dest)
    atomic_json(destination/".prepare-complete.json",saved)
    return destination

@contextlib.contextmanager
def process_lock(output):
    # Process lock is released by kernel even after SIGKILL. Same-runtime protection.
    path=Path("/tmp")/("v7-suite-"+fingerprint(str(Path(output).resolve()))+".lock")
    with path.open("a+") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("這個 runtime 已有同一組實驗執行中")
        yield

def run_suite(args):
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    def emit(event,**values):
        row={"event":event,"time":datetime.now(timezone.utc).isoformat(),**values}
        text=json.dumps(row,ensure_ascii=False,allow_nan=False)
        print(text,flush=True)
        with (output/"events.jsonl").open("a",encoding="utf-8") as f:f.write(text+"\n")
    if args.status:
        p=output/"suite.json"
        print(p.read_text() if p.exists() else "尚未開始");return
    with process_lock(output):
        emit("matrix_check_started")
        saved=verify_matrix(args.data,allow_diagnostic=args.diagnostic)
        emit("matrix_reused",rows=saved["result"]["rows"],source=str(args.data),rebuild=False)
        root=stage_matrix(args.data,args.local_data,saved) if args.local_data else args.data
        metadata=json.loads((root/"feature_metadata.json").read_text())
        split_doc=json.loads((root/"splits.json").read_text())
        splits=split_doc["splits"]
        validate_protocol(metadata);validate_feature_metadata(metadata,ExperimentConfig())
        if split_doc["trading_calendar"]!=metadata["trading_calendar"]:
            raise ValueError("split calendar mismatch")
        validate_frozen_dates(splits,split_doc["trading_calendar"])
        if args.train_dates or args.validation_dates:
            if not args.diagnostic:raise ValueError("date limits require --diagnostic")
            splits={key:values[:args.train_dates if key=="train" else args.validation_dates]
                    if (args.train_dates if key=="train" else args.validation_dates) else values
                    for key,values in splits.items()}
        settings=settings_from(args)
        device=torch.device(args.device)
        if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        if settings["precision"]=="bf16" and (device.type!="cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 requires a compatible CUDA GPU")
        if settings["precision"]=="fp16" and device.type!="cuda":raise ValueError("FP16 requires CUDA")
        here=Path(__file__).parent
        sources={p.name:digest(p) for p in sorted(here.glob("v7_*.py")) if not p.name.endswith("_test.py")}
        identity=fingerprint({"matrix":saved["files"],"splits":splits,"settings":settings,"sources":sources,
                              "runtime":_runtime_versions(),"stages":STAGES,"diagnostic":args.diagnostic})
        state_path=output/"suite.json"
        state=json.loads(state_path.read_text()) if state_path.exists() else {"identity":identity,"experiments":{},"results":{}}
        if state["identity"]!=identity:
            raise ValueError("實驗設定／資料／程式／環境已變更，請使用新 output，不可混入原實驗")
        atomic_json(state_path,state)
        emit("index_loading",device=str(device),precision=settings["precision"])
        original=PreparedDataIndex.from_parquet(root/"features_59.parquet",metadata)
        index=PackedIndex(original);del original;gc.collect()
        graph=FastGraph(KnowledgeGraphCSR(root/"knowledge_graph_v2_csr.npz"))
        train=FastDataset(index,graph,splits["train"])
        validation=FastDataset(index,graph,splits["validation"])
        stop=[False]
        def request_stop(signum,frame):
            stop[0]=True
        old_handlers={sig:signal.signal(sig,request_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            for stage,name in enumerate(STAGES):
                if stop[0]:break
                if name not in state["experiments"]:
                    state["experiments"][name]=resolve_spec(stage,state["results"])
                    atomic_json(state_path,state)
                spec=normalize_spec(state["experiments"][name])
                experiment_output=output/name
                cached=experiment_output/"result.json"
                expected=fingerprint({"suite":identity,"spec":spec,"settings":settings})
                if cached.exists():
                    result=json.loads(cached.read_text())
                    if result.get("identity")!=expected:raise ValueError("cached experiment identity mismatch")
                    if result.get("status")=="complete":
                        # Ensure a completed result still has its best checkpoint.
                        from v7_experiment_storage import CheckpointStore
                        if CheckpointStore(experiment_output/"checkpoints").load(expected,best=True) is None:
                            raise OSError("completed experiment lacks best checkpoint")
                        state["results"][name]=result
                        atomic_json(state_path,state)
                        emit("experiment_skipped_completed",experiment=name)
                        continue
                if state["results"].get(name,{}).get("status")=="failed":
                    emit("experiment_skipped_failed",experiment=name)
                    continue
                state["active"]=name;state["status"]="running";atomic_json(state_path,state)
                try:
                    result=run_experiment(spec,settings,train,validation,experiment_output,identity,
                        device=device,ssd_fn=official_ssd_callable(),emit=emit,
                        stop_requested=lambda:stop[0],invocation_steps=args.pause_after_steps)
                except (torch.cuda.OutOfMemoryError,FloatingPointError) as exc:
                    # An architecture-specific failure is recorded, never counted as a winner.
                    result={"status":"failed","experiment":name,"reason":str(exc),"config":spec["config"]}
                    emit("experiment_failed",**result)
                    gc.collect()
                    if device.type=="cuda":torch.cuda.empty_cache()
                gc.collect()
                if device.type=="cuda":torch.cuda.empty_cache()
                state["results"][name]=result
                if result["status"]=="paused":
                    state["status"]="paused";atomic_json(state_path,state);return
                atomic_json(state_path,state)
            state["status"]="paused" if stop[0] else "complete"
            winners=[r for r in state["results"].values() if r["status"]=="complete"]
            state["winner"]=max(winners,key=lambda r:r["best_rank_ic_5d"])["experiment"] if winners else None
            atomic_json(state_path,state)
            rows=[{"experiment":name,"status":r["status"],"config":r.get("config"),
                   "best_rank_ic_5d":r.get("best_rank_ic_5d"),"epochs":r.get("epochs"),
                   "reason":r.get("reason")} for name,r in state["results"].items()]
            atomic_json(output/"summary.json",{"status":state["status"],"winner":state["winner"],"experiments":rows,
                "interpretation":"Validation selects capacity; this is not an untouched test or live-profit claim."})
            emit("suite_"+state["status"],winner=state["winner"])
        finally:
            for sig,handler in old_handlers.items():signal.signal(sig,handler)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--local-data",type=Path)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--precision",choices=("fp32","bf16","fp16"),default="fp32")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--patience",type=int,default=5)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--learning-rate",type=float,default=1e-4)
    p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--train-dates",type=int)
    p.add_argument("--validation-dates",type=int)
    p.add_argument("--pause-after-steps",type=int)
    p.add_argument("--status",action="store_true")
    return p
def main():
    args=parser().parse_args()
    if args.epochs<1 or args.patience<1 or args.learning_rate<=0:
        raise ValueError("invalid training settings")
    for n in ("train_dates","validation_dates","pause_after_steps"):
        if getattr(args,n) is not None and getattr(args,n)<1:raise ValueError(n+" must be positive")
    run_suite(args)
if __name__=="__main__":main()
