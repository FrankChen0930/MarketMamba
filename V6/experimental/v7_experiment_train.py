"""Resumable train/validation state machine for isolated V7 capacity experiments."""
from __future__ import annotations
import math
import subprocess
from dataclasses import asdict
import time
from pathlib import Path
import torch
from v7_integrated_train import (adamw_parameter_groups, capture_rng_state, restore_rng_state,
    set_reproducible_seed, short_loss, rank_ic_by_horizon, aggregate_validation_metrics)
from v7_experiment_data import batches, to_device
from v7_experiment_model import ExperimentConfig, ExperimentModel
from v7_experiment_storage import CheckpointStore, atomic_json, fingerprint

def gpu_utilization():
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=2,check=True)
        return float(result.stdout.splitlines()[0])
    except (OSError,ValueError,subprocess.SubprocessError):
        return None

def lr_factor(step, total, warmup_fraction):
    if total<=1:
        return 1.
    warmup=max(1,int(total*warmup_fraction))
    if step<warmup:
        return .04+.96*step/max(1,warmup-1)
    progress=min(1.,(step-warmup)/max(1,total-1-warmup))
    return .0001+.9999*.5*(1+math.cos(math.pi*progress))

def fresh_state(identity):
    return dict(identity=identity,epoch=0,batch=0,step=0,phase="train",val_index=0,
        val_rows=[],history=[],best_metric=None,bad_epochs=0,patience_reference=None,
        loss_sum=0.,zero_loss_sum=0.,updates=0,skipped=0,train_seconds=0.,
        validation_seconds=0.,head_counts=[0,0])

def update_selection(state, score, min_delta):
    best=state["best_metric"] is None or score>state["best_metric"]
    if best:
        state["best_metric"]=score
    reference=state["patience_reference"]
    if reference is None or score>reference+min_delta:
        state["patience_reference"]=score
        state["bad_epochs"]=0
    else:
        state["bad_epochs"]+=1
    return best

def run_experiment(spec, settings, train, validation, output, suite_identity, *, device,
                   ssd_fn, emit, stop_requested=lambda:False, invocation_steps=None,
                   model_factory=ExperimentModel):
    cfg=ExperimentConfig(**spec["config"])
    cfg.validate()
    identity=fingerprint({"suite":suite_identity,"spec":spec,"settings":settings})
    store=CheckpointStore(Path(output)/"checkpoints")
    set_reproducible_seed(settings["seed"])
    model=model_factory(cfg,ssd_fn=ssd_fn).to(device)
    optimizer=torch.optim.AdamW(adamw_parameter_groups(model,settings["weight_decay"]),
                               lr=settings["learning_rate"])
    total=len(train)*settings["epochs"]
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step:lr_factor(step,total,settings["warmup_fraction"]))
    use_amp=settings["precision"]!="fp32"
    dtype=torch.bfloat16 if settings["precision"]=="bf16" else torch.float16
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and settings["precision"]=="fp16")
    state=store.load(identity)
    if state:
        model.load_state_dict(state.pop("model"))
        optimizer.load_state_dict(state.pop("optimizer"))
        scheduler.load_state_dict(state.pop("scheduler"))
        scaler.load_state_dict(state.pop("scaler"))
        restore_rng_state(state.pop("rng"))
        emit("resumed",experiment=spec["id"],epoch=state["epoch"]+1,
             phase=state["phase"],next_batch=state["batch"],next_validation=state["val_index"],step=state["step"])
    else:
        state=fresh_state(identity)
        state["model_config"]=asdict(cfg)
        state["training_settings"]=dict(settings)
    initial_step=state["step"]
    last_save=time.monotonic()
    def save(best=False):
        nonlocal last_save
        snapshot={**state,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                  "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                  "rng":capture_rng_state(device.type=="cuda")}
        start=time.perf_counter()
        store.save(snapshot,best=best)
        atomic_json(Path(output)/"history.json",state["history"])
        last_save=time.monotonic()
        emit("checkpoint_saved",experiment=spec["id"],step=state["step"],phase=state["phase"],
             seconds=round(time.perf_counter()-start,3),best=best)
    def pause_due():
        return stop_requested() or (invocation_steps is not None and state["step"]-initial_step>=invocation_steps)
    emit("experiment_start",experiment=spec["id"],config=spec["config"],
         parameters=sum(p.numel() for p in model.parameters()),precision=settings["precision"],
         max_epochs=settings["epochs"],train_dates=len(train),validation_dates=len(validation))
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def gpu_sync():
        if device.type=="cuda":
            torch.cuda.synchronize(device)
    while state["phase"]!="finished":
        if pause_due():
            save()
            return {"status":"paused","step":state["step"],"identity":identity}
        if state["phase"]=="train":
            model.train()
            iterator=iter(batches(train,state["batch"],prefetch=settings["prefetch"]))
            try:
                while state["batch"]<len(train):
                    start=time.perf_counter()
                    batch,sample=next(iterator)
                    loaded=time.perf_counter()
                    profile=(state["step"]%settings["progress_interval"]==0)
                    lr_used=optimizer.param_groups[0]["lr"]
                    sample=to_device(sample,device)
                    if profile:gpu_sync()
                    transferred=time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
                        predictions=model.forward_prepared(sample)
                    if profile:gpu_sync()
                    forwarded=time.perf_counter()
                    # Preserve the previous objective, compute sensitive reductions in FP32.
                    try:
                        loss=short_loss(predictions.float(),sample["labels"])
                    except ValueError as exc:
                        if "nonfinite predictions" in str(exc):
                            raise FloatingPointError(str(exc)) from exc
                        raise
                    state["batch"]=batch+1
                    if loss is None:
                        state["skipped"]+=1
                    else:
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite loss")
                        with torch.no_grad():
                            zero=short_loss(torch.zeros_like(predictions,dtype=torch.float32),sample["labels"])
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        try:
                            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings["grad_clip"],error_if_nonfinite=True)
                        except RuntimeError as exc:
                            if "non-finite" in str(exc):
                                raise FloatingPointError("nonfinite gradients") from exc
                            raise
                        if profile:gpu_sync()
                        backwarded=time.perf_counter()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        if profile:gpu_sync()
                        optimized=time.perf_counter()
                        value=float(loss.detach())
                        zero_value=float(zero)
                        counts=torch.isfinite(sample["labels"]).sum(0).tolist()
                        state["step"]+=1
                        state["updates"]+=1
                        state["loss_sum"]+=value
                        state["zero_loss_sum"]+=zero_value
                        state["head_counts"]=[a+b for a,b in zip(state["head_counts"],counts)]
                        if profile:
                            emit("training_progress",experiment=spec["id"],epoch=state["epoch"]+1,
                                 batch=state["batch"],batches=len(train),step=state["step"],
                                 loss=value,zero_prediction_loss=zero_value,
                                 relative_to_zero=value/max(zero_value,1e-12),lr=lr_used,
                                 gradient_norm=float(norm),stocks=len(sample["stock_ids"]),edges=sample["edge_index"].shape[1],
                                 timing_seconds={"data_wait":loaded-start,"transfer":transferred-loaded,
                                   "forward":forwarded-transferred,"loss_backward":backwarded-forwarded,
                                   "optimizer":optimized-backwarded},
                                 gpu_peak_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=="cuda" else 0.,
                                 gpu_utilization_percent=gpu_utilization() if device.type=="cuda" else None)
                    state["train_seconds"]+=time.perf_counter()-start
                    if state["batch"]==len(train):
                        state["phase"]="validate"
                        save()
                        break
                    if state["batch"]%settings["checkpoint_interval"]==0 or time.monotonic()-last_save>=settings["checkpoint_seconds"]:
                        save()
                    if pause_due():
                        save()
                        return {"status":"paused","step":state["step"],"identity":identity}
            finally:
                iterator.close()
        if state["phase"]=="validate":
            model.eval()
            iterator=iter(batches(validation,state["val_index"],prefetch=settings["prefetch"]))
            try:
                with torch.no_grad():
                    for i,sample in iterator:
                        start=time.perf_counter()
                        sample=to_device(sample,device)
                        with torch.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
                            pred=model.forward_prepared(sample)
                        pred=pred.float().cpu()
                        if not torch.isfinite(pred).all():
                            raise FloatingPointError("nonfinite validation predictions")
                        metrics=rank_ic_by_horizon(pred,sample["labels"].cpu())
                        # JSON-safe saved rows; None is restored to NaN for aggregation.
                        state["val_rows"].append({k:v if math.isfinite(v) else None for k,v in metrics.items()})
                        state["val_index"]=i+1
                        state["validation_seconds"]+=time.perf_counter()-start
                        if (i+1)%settings["progress_interval"]==0 or i+1==len(validation):
                            emit("validation_progress",experiment=spec["id"],epoch=state["epoch"]+1,
                                 dates=i+1,total=len(validation))
                        if (i+1)%settings["checkpoint_interval"]==0 or time.monotonic()-last_save>=settings["checkpoint_seconds"]:
                            save()
                        if stop_requested():
                            save()
                            return {"status":"paused","step":state["step"],"identity":identity}
            finally:
                iterator.close()
            if state["updates"]==0:
                raise ValueError("whole epoch has no targets")
            metrics=aggregate_validation_metrics([{k:float("nan") if v is None else v for k,v in row.items()}
                                                  for row in state["val_rows"]])
            score=metrics["rank_ic_5d"]
            if not math.isfinite(score):
                raise FloatingPointError("5d validation Rank IC is unavailable; selection cannot silently change heads")
            improved=update_selection(state,score,settings["min_delta"])
            mean_loss=state["loss_sum"]/state["updates"]
            mean_zero=state["zero_loss_sum"]/state["updates"]
            record={"epoch":state["epoch"]+1,"step":state["step"],"train_loss":mean_loss,
                    "zero_prediction_loss":mean_zero,"relative_to_zero":mean_loss/max(mean_zero,1e-12),
                    "validation":{k:v if math.isfinite(v) else None for k,v in metrics.items()},
                    "best_rank_ic_5d":state["best_metric"],"bad_epochs":state["bad_epochs"],
                    "lr":optimizer.param_groups[0]["lr"],"train_seconds":state["train_seconds"],
                    "validation_seconds":state["validation_seconds"],"head_counts":state["head_counts"],
                    "skipped":state["skipped"],"steps_per_second":state["updates"]/max(state["train_seconds"],1e-9)}
            state["history"].append(record)
            state["epoch"]+=1
            early=(state["epoch"]>=max(settings["min_epochs"],math.ceil(settings["epochs"]*settings["warmup_fraction"])+1)
                   and state["bad_epochs"]>=settings["patience"])
            state["phase"]="finished" if early or state["epoch"]>=settings["epochs"] else "train"
            state["stop_reason"]="early_stopping" if early else "epoch_limit" if state["phase"]=="finished" else None
            state.update(batch=0,val_index=0,val_rows=[],loss_sum=0.,zero_loss_sum=0.,updates=0,
                         skipped=0,train_seconds=0.,validation_seconds=0.,head_counts=[0,0])
            save(best=improved)
            emit("epoch_complete",experiment=spec["id"],**record)
    result={"status":"complete","experiment":spec["id"],"config":spec["config"],"identity":identity,
            "best_rank_ic_5d":state["best_metric"],"epochs":state["epoch"],"step":state["step"],
            "stop_reason":state["stop_reason"],"history":state["history"],
            "checkpoint_manifest":str(store.pointer)}
    atomic_json(Path(output)/"result.json",result)
    return result
