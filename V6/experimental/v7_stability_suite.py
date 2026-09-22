"""Colab entrypoint for A/B prediction diagnostics and C rolling-window training."""
from __future__ import annotations
import argparse, gc, json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import torch
from v7_experiment_data import FastDataset, FastGraph, PackedIndex
from v7_experiment_model import ExperimentConfig
from v7_experiment_storage import CheckpointStore, atomic_json, digest, fingerprint
from v7_experiment_suite import process_lock
from v7_experiment_train import run_experiment
from v7_integrated_train import KnowledgeGraphCSR, PreparedDataIndex, _runtime_versions, official_ssd_callable
from v7_stability_diagnostics import build_diagnostics
from v7_stability_export import export_checkpoint_predictions
from v7_stability_windows import build_window_splits
REVISION = "v7-stability-suite-v1"

def window_training_settings():
    return {"epochs":20,"seed":17,"learning_rate":1e-4,"weight_decay":.01,
        "warmup_fraction":.15,"patience":5,"min_epochs":5,"min_delta":0.,
        "grad_clip":1.,"precision":"fp32","prefetch":True,"checkpoint_interval":100,
        "checkpoint_seconds":180,"progress_interval":100}

def make_window_experiment_spec(window):
    if window.get("evaluation_used_for_selection"):
        raise ValueError("evaluation split cannot select a checkpoint")
    if not window.get("train") or not window.get("selection") or not window.get("evaluation"):
        raise ValueError("window requires non-empty train, selection and evaluation splits")
    config=ExperimentConfig(d_model=64,d_state=32,temporal_layers=1,forward_layers=1,
        reverse_layers=1,train_cutoff=max(window["train"]),validation_end=max(window["selection"]))
    config.validate()
    return {"id":f'{window["id"]}-E5-seed17',"config":asdict(config)}

def run_window_sequence(windows,output,identity,run_one,verify_completed=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/"suite.json"
    state=json.loads(path.read_text()) if path.exists() else {
        "revision":REVISION,"identity":identity,"status":"running","results":{}}
    if state.get("identity")!=identity:raise ValueError("suite identity changed; use a new output directory")
    if state.get("status")=="complete":
        if verify_completed:
            for window in windows:
                verify_completed(window,state["results"][window["id"]])
        return {"status":"skipped","windows":len(state["results"]),"identity":identity}
    for window in windows:
        key=window["id"]
        if state["results"].get(key,{}).get("status")=="complete":
            if verify_completed:verify_completed(window,state["results"][key])
            continue
        state.update(active=key,status="running");atomic_json(path,state)
        result=run_one(window);state["results"][key]=result
        if result.get("status")=="paused":
            state["status"]="paused";atomic_json(path,state)
            return {"status":"paused","active":key,"identity":identity}
        if result.get("status")!="complete":
            state["status"]="failed";atomic_json(path,state);raise RuntimeError(f"{key} did not complete: {result}")
        atomic_json(path,state)
    state.update(status="complete",active=None);atomic_json(path,state)
    return {"status":"complete","windows":len(state["results"]),"identity":identity}

def _load_matrix(data,diagnostic):
    from v7_experiment_storage import verify_matrix
    saved=verify_matrix(data,allow_diagnostic=diagnostic);root=Path(data)
    metadata=json.loads((root/"feature_metadata.json").read_text())
    split_doc=json.loads((root/"splits.json").read_text())
    if split_doc["trading_calendar"]!=metadata["trading_calendar"]:raise ValueError("split calendar mismatch")
    index=PackedIndex(PreparedDataIndex.from_parquet(root/"features_59.parquet",metadata))
    graph=FastGraph(KnowledgeGraphCSR(root/"knowledge_graph_v2_csr.npz"))
    data_id=fingerprint({"files":saved["files"],"protocol":metadata["protocol_fingerprint"]})
    return saved,metadata,split_doc,index,graph,data_id

def _source(directory,model_id,seed):
    directory=Path(directory);path=directory/"result.json"
    if not path.is_file():raise ValueError(f"missing completed result: {path}")
    result=json.loads(path.read_text())
    if result.get("status")!="complete" or not result.get("identity"):
        raise ValueError(f"incomplete model source: {directory}")
    return {"directory":directory,"model_id":model_id,"seed":seed,"identity":result["identity"]}

def _sources(capacity,confirmation):
    return (_source(Path(capacity)/"E5-state32","E5-seed17",17),
        _source(Path(confirmation)/"E5-seed29","E5-seed29",29),
        _source(Path(confirmation)/"E5-seed43","E5-seed43",43))

def _read_rows(directory):
    directory=Path(directory);manifest=json.loads((directory/"manifest.json").read_text())
    if manifest.get("status")!="complete":raise ValueError(f"prediction export is not complete: {directory}")
    rows=[]
    for item in manifest["chunks"]:
        path=directory/item["file"]
        if digest(path)!=item["sha256"]:raise OSError("prediction chunk digest mismatch: "+str(path))
        frame=pd.read_parquet(path).where(lambda value:pd.notna(value),None)
        rows.extend(frame.to_dict("records"))
    return rows

def _regimes(path):
    frame=pd.read_parquet(path,columns=["Date","stock_id","Close"])
    frame["Date"]=frame["Date"].astype(str).str[:10];frame=frame.sort_values(["stock_id","Date"])
    frame["return_1d"]=frame.groupby("stock_id",sort=False)["Close"].pct_change()
    proxy=frame.groupby("Date",sort=True)["return_1d"].median().rolling(20,min_periods=5).sum()
    return {day:"up" if value>0 else "down" if value<0 else "flat" for day,value in proxy.dropna().items()}

def run_ab(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with process_lock(output):
        _,_,splits,index,graph,data_id=_load_matrix(args.data,args.diagnostic)
        dates=splits["splits"]["validation"]
        if args.date_limit:
            if not args.diagnostic:raise ValueError("--date-limit is diagnostic-only")
            dates=dates[:args.date_limit]
        dataset=FastDataset(index,graph,dates);device=torch.device(args.device)
        if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        sources=_sources(args.capacity_results,args.confirmation_results);statuses={}
        for source in sources:
            target=output/"predictions"/source["model_id"]
            statuses[source["model_id"]]=export_checkpoint_predictions(
                checkpoint_dir=source["directory"]/"checkpoints",
                expected_checkpoint_identity=source["identity"],dataset=dataset,output=target,
                model_id=source["model_id"],data_id=data_id,seed=source["seed"],device=device,
                ssd_fn=official_ssd_callable(),max_dates=args.max_export_dates)
            if statuses[source["model_id"]]["status"]=="paused":
                atomic_json(output/"summary.json",{"status":"paused","phase":"export","models":statuses});return
            gc.collect()
            if device.type=="cuda":torch.cuda.empty_cache()
        rows=[]
        for source in sources:rows.extend(_read_rows(output/"predictions"/source["model_id"]))
        atomic_json(output/"diagnostics.json",build_diagnostics(rows,regimes=_regimes(Path(args.data)/"market_prices_raw.parquet")))
        atomic_json(output/"summary.json",{"status":"complete","mode":"A/B diagnostics only",
            "models":statuses,"data_id":data_id,"dates":len(dates),"rows":len(rows),
            "next":"Review diagnostics before starting C; this completion does not approve retraining."})

def run_c(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with process_lock(output):
        saved,metadata,_,index,graph,data_id=_load_matrix(args.data,args.diagnostic)
        if args.window and not args.diagnostic:
            raise ValueError("--window is diagnostic-only; formal C always runs all three windows")
        windows=build_window_splits(metadata["trading_calendar"],purge_sessions=30,
            window_ids=tuple(args.window) if args.window else None)
        if args.date_limit:
            if not args.diagnostic:raise ValueError("--date-limit is diagnostic-only")
            for window in windows:
                for name in ("train","selection","evaluation"):window[name]=window[name][:args.date_limit]
        settings=window_training_settings()
        if args.diagnostic:settings={**settings,"epochs":args.diagnostic_epochs,"min_epochs":1,
            "patience":1,"prefetch":False,"progress_interval":1,"checkpoint_interval":1}
        sources={p.name:digest(p) for p in Path(__file__).parent.glob("v7_*.py") if not p.name.endswith("_test.py")}
        identity=fingerprint({"revision":REVISION,"matrix":saved["files"],"windows":windows,
            "settings":settings,"sources":sources,"runtime":_runtime_versions(),"diagnostic":args.diagnostic})
        device=torch.device(args.device)
        if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        def run_one(window):
            spec=make_window_experiment_spec(window);target=output/window["id"]
            result=run_experiment(spec,settings,FastDataset(index,graph,window["train"]),
                FastDataset(index,graph,window["selection"]),target,identity,device=device,
                ssd_fn=official_ssd_callable(),emit=lambda event,**values:print(json.dumps(
                    {"event":event,"window":window["id"],**values},ensure_ascii=False,allow_nan=False),flush=True),
                invocation_steps=args.pause_after_steps)
            if result["status"]=="paused":return result
            exported=export_checkpoint_predictions(checkpoint_dir=target/"checkpoints",
                expected_checkpoint_identity=result["identity"],dataset=FastDataset(index,graph,window["evaluation"]),
                output=target/"evaluation-predictions",model_id=spec["id"],data_id=data_id,
                seed=17,device=device,ssd_fn=official_ssd_callable(),max_dates=args.max_export_dates)
            if exported["status"]=="paused":
                return {"status":"paused","phase":"evaluation_export","training":result}
            return {"status":"complete","experiment":spec["id"],"best_rank_ic_5d":result["best_rank_ic_5d"],
                "epochs":result["epochs"],"stop_reason":result["stop_reason"],"evaluation_export":exported}
        def verify_one(window,result):
            target=output/window["id"]
            saved=json.loads((target/"result.json").read_text())
            CheckpointStore(target/"checkpoints").load(saved["identity"],best=True)
            export_dir=target/"evaluation-predictions"
            manifest=json.loads((export_dir/"manifest.json").read_text())
            if manifest.get("status")!="complete" or manifest.get("identity")!=result["evaluation_export"]["identity"]:
                raise OSError("completed evaluation export identity mismatch")
            for item in manifest["chunks"]:
                if digest(export_dir/item["file"])!=item["sha256"]:
                    raise OSError("completed evaluation export chunk mismatch")
        result=run_window_sequence(windows,output,identity,run_one,verify_completed=verify_one)
        atomic_json(output/"summary.json",{**result,"mode":"C three-window retrospective stability",
            "data_id":data_id,"classification_limit":metadata["industry_neutralization"]["classification_policy"],
            "interpretation":"Selection years choose checkpoints. Evaluation years are retrospective and were not used for early stopping."})

def parser():
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--data",type=Path,required=True);common.add_argument("--output",type=Path,required=True)
    common.add_argument("--device",default="cuda");common.add_argument("--diagnostic",action="store_true")
    common.add_argument("--date-limit",type=int);common.add_argument("--max-export-dates",type=int)
    root=argparse.ArgumentParser(description=__doc__);sub=root.add_subparsers(dest="mode",required=True)
    ab=sub.add_parser("ab",parents=[common]);ab.add_argument("--capacity-results",type=Path,required=True)
    ab.add_argument("--confirmation-results",type=Path,required=True);ab.set_defaults(func=run_ab)
    c=sub.add_parser("c",parents=[common]);c.add_argument("--diagnostic-epochs",type=int,default=1)
    c.add_argument("--pause-after-steps",type=int);c.add_argument("--window",action="append",choices=("W2024","W2025","W2026"))
    c.set_defaults(func=run_c);return root

def main(argv=None):
    args=parser().parse_args(argv);args.func(args)
if __name__=="__main__":main()
