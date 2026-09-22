"""Run four paired-seed confirmations and one width64 temporal-depth experiment."""
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

from v7_experiment_suite import normalize_spec, stage_matrix, process_lock, settings_from
from v7_confirmation_design import STAGES, SEEDS, read_reference, resolve_spec, comparison, validate_reference, REFERENCE
from v7_confirmation_benchmark import benchmark

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
        reference=read_reference()
        settings=settings_from(args)
        if not args.diagnostic and settings!=reference["settings"]:
            raise ValueError("正式確認實驗須沿用 seed17 的 FP32 訓練設定；勿修改 epoch、patience 或 learning rate")
        if args.precision!="fp32":raise ValueError("此批正式訓練固定 FP32；BF16 只用於短測")
        proof={"verified":False,"diagnostic":True} if args.diagnostic else validate_reference(
            reference,saved["files"],splits,_runtime_versions(),Path(__file__).parent)
        atomic_json(output/"reference-verification.json",proof)
        device=torch.device(args.device)
        if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        if settings["precision"]=="bf16" and (device.type!="cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 requires a compatible CUDA GPU")
        if settings["precision"]=="fp16" and device.type!="cuda":raise ValueError("FP16 requires CUDA")
        here=Path(__file__).parent
        sources={p.name:digest(p) for p in sorted(here.glob("v7_*.py")) if not p.name.endswith("_test.py")}
        identity=fingerprint({"matrix":saved["files"],"splits":splits,"settings":settings,"sources":sources,
                              "runtime":_runtime_versions(),"stages":STAGES,"seeds":SEEDS,"reference":digest(REFERENCE),"diagnostic":args.diagnostic})
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
            bench_identity=fingerprint({"suite":identity,"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
                "capability":torch.cuda.get_device_capability(device) if device.type=="cuda" else None,
                "warmup":args.benchmark_warmup,"steps":args.benchmark_steps,"repeats":args.benchmark_repeats})
            bench=benchmark(train,reference["results"]["E5-state32"]["config"],settings,
                output/"benchmark.json",bench_identity,device,official_ssd_callable(),emit,
                stop_requested=lambda:stop[0],warmup=args.benchmark_warmup,
                steps=args.benchmark_steps,repeats=args.benchmark_repeats)
            if bench["status"]=="failed":
                state["status"]="failed";atomic_json(state_path,state)
                raise RuntimeError("FP32 基準短測失敗，請先查看 benchmark.json；尚未開始正式訓練")
            if bench["status"]=="paused":
                state["status"]="paused";atomic_json(state_path,state);return
            for stage,name in enumerate(STAGES):
                if stop[0]:break
                settings={**settings,"seed":SEEDS[stage]}
                if stage==4 and not args.diagnostic and not state.get("selection"):
                    state["selection"]=comparison(state["results"],reference)
                    atomic_json(state_path,state)
                    emit("paired_comparison_complete",**state["selection"])
                if name not in state["experiments"]:
                    state["experiments"][name]=resolve_spec(stage,state["results"],reference,args.diagnostic)
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
                state["active"]=name;state["status"]="running";atomic_json(state_path,state)
                try:
                    result=run_experiment(spec,settings,train,validation,experiment_output,identity,
                        device=device,ssd_fn=official_ssd_callable(),emit=emit,
                        stop_requested=lambda:stop[0],invocation_steps=args.pause_after_steps)
                except (torch.cuda.OutOfMemoryError,FloatingPointError) as exc:
                    # An architecture-specific failure is recorded, never counted as a winner.
                    result={"status":"failed","experiment":name,"reason":str(exc),"config":spec["config"]}
                    emit("experiment_failed",**result)
                    state["results"][name]=result;state["status"]="failed";atomic_json(state_path,state)
                    raise
                gc.collect()
                if device.type=="cuda":torch.cuda.empty_cache()
                state["results"][name]=result
                if result["status"]=="paused":
                    state["status"]="paused";atomic_json(state_path,state);return
                atomic_json(state_path,state)
            state["status"]="paused" if stop[0] else "complete"
            paired=None
            if all(n in state["results"] and state["results"][n]["status"]=="complete" for n in STAGES[:4]) and not args.diagnostic:
                paired=comparison(state["results"],reference)
            state["selection"]=paired
            atomic_json(state_path,state)
            rows=[{"experiment":name,"seed":SEEDS[STAGES.index(name)],"status":v["status"],
                   "config":v.get("config"),"best_rank_ic_5d":v.get("best_rank_ic_5d"),"epochs":v.get("epochs")}
                  for name,v in state["results"].items()]
            depth=None
            if state["status"]=="complete":
                from v7_confirmation_design import metrics
                deep=state["results"][STAGES[4]]
                baseline=reference["results"]["E5-state32" if deep["config"]["d_state"]==32 else "E3-width64"]
                depth={"diagnostic_only":args.diagnostic,"d_state":deep["config"]["d_state"],
                       "depth_metrics":metrics(deep),
                       "seed17_shallow_metrics":None if args.diagnostic else metrics(baseline),
                       "delta_ic5":None if args.diagnostic else metrics(deep)["rank_ic_5d"]-metrics(baseline)["rank_ic_5d"],
                       "delta_ic10":None if args.diagnostic else metrics(deep)["rank_ic_10d"]-metrics(baseline)["rank_ic_10d"],
                       "note":"Depth comparison is one seed only; 2/2/2 is still untested."}
            atomic_json(output/"summary.json",{"status":state["status"],"diagnostic":args.diagnostic,
                "paired_comparison":paired,"depth_comparison":depth,"experiments":rows,
                "benchmark":bench,"next":"Stop for review. Do not automatically run 2/2/2 or more depth seeds."})
            emit("suite_"+state["status"],selected_d_state=paired["selected_d_state"] if paired else None)
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
    p.add_argument("--benchmark-warmup",type=int,default=10)
    p.add_argument("--benchmark-steps",type=int,default=30)
    p.add_argument("--benchmark-repeats",type=int,default=3)
    return p
def main():
    args=parser().parse_args()
    if args.epochs<1 or args.patience<1 or args.learning_rate<=0:
        raise ValueError("invalid training settings")
    for n in ("train_dates","validation_dates","pause_after_steps"):
        if getattr(args,n) is not None and getattr(args,n)<1:raise ValueError(n+" must be positive")
    if min(args.benchmark_warmup,args.benchmark_steps,args.benchmark_repeats)<1:
        raise ValueError("benchmark counts must be positive")
    if not args.diagnostic and (args.benchmark_warmup,args.benchmark_steps,args.benchmark_repeats)!=(10,30,3):
        raise ValueError("正式短測固定 warmup10 / measured30 / repeats3")
    run_suite(args)
if __name__=="__main__":main()
