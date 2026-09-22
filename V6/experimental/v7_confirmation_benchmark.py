"""Bounded, disposable fixed-sample FP32/BF16 training throughput probe."""
import gc
import statistics
import time
import torch
from v7_experiment_model import ExperimentConfig, ExperimentModel
from v7_experiment_data import to_device
from v7_experiment_storage import atomic_json
from v7_integrated_train import set_reproducible_seed, adamw_parameter_groups, short_loss

def benchmark(train, config, settings, output, identity, device, ssd_fn, emit,
              stop_requested=lambda:False, warmup=10, steps=30, repeats=3):
    import json
    config=dict(config)
    for key in ("group_dims","horizons"):config[key]=tuple(config[key])
    if output.exists():
        saved=json.loads(output.read_text())
        if saved.get("identity")==identity and saved.get("status")=="complete":
            emit("benchmark_reused");return saved
    if device.type!="cuda":
        result={"identity":identity,"status":"complete","skipped":"CUDA benchmark only","trials":[]}
        atomic_json(output,result);return result
    indices=sorted(set((0,len(train)//2,len(train)-1)))
    samples=[train[i] for i in indices]
    trials=[]
    for repeat in range(repeats):
        for precision in (("fp32","bf16") if repeat%2==0 else ("bf16","fp32")):
            if stop_requested():return {"status":"paused"}
            model=optimizer=sample=pred=loss=None
            row={"repeat":repeat+1,"precision":precision}
            try:
                if precision=="bf16" and not torch.cuda.is_bf16_supported():
                    row.update(status="unsupported");trials.append(row);continue
                set_reproducible_seed(17)
                model=ExperimentModel(ExperimentConfig(**config),ssd_fn=ssd_fn).to(device).train()
                optimizer=torch.optim.AdamW(adamw_parameter_groups(model,settings["weight_decay"]),
                                            lr=settings["learning_rate"])
                losses=[]
                for step in range(warmup+steps):
                    if stop_requested():return {"status":"paused"}
                    if step==warmup:
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        start=time.perf_counter()
                    sample=to_device(samples[step%len(samples)],device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=precision=="bf16"):
                        pred=model.forward_prepared(sample)
                    if not torch.isfinite(pred).all():raise FloatingPointError("nonfinite predictions")
                    loss=short_loss(pred.float(),sample["labels"])
                    if loss is None or not torch.isfinite(loss):raise FloatingPointError("invalid loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),settings["grad_clip"],error_if_nonfinite=True)
                    optimizer.step()
                    if step>=warmup:losses.append(float(loss.detach()))
                torch.cuda.synchronize(device)
                seconds=time.perf_counter()-start
                row.update(status="complete",seconds=seconds,steps=steps,steps_per_second=steps/seconds,
                           peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                           mean_loss=statistics.mean(losses),finite_outputs_loss_gradients=True)
            except (RuntimeError,ValueError,FloatingPointError) as exc:
                row.update(status="failed",reason=str(exc))
            finally:
                del model,optimizer,sample,pred,loss
                gc.collect();torch.cuda.empty_cache()
            trials.append(row);emit("benchmark_trial",**row)
            atomic_json(output,{"identity":identity,"status":"running","trials":trials})
    summary={}
    for precision in ("fp32","bf16"):
        good=[r for r in trials if r["precision"]==precision and r["status"]=="complete"]
        summary[precision]={"successful_repeats":len(good),
            "median_steps_per_second":statistics.median(r["steps_per_second"] for r in good) if good else None}
    a,b=summary["fp32"],summary["bf16"]
    speedup=b["median_steps_per_second"]/a["median_steps_per_second"] if a["successful_repeats"]==repeats and b["successful_repeats"]==repeats else None
    result={"identity":identity,"status":"complete","trials":trials,"summary":summary,"bf16_speedup":speedup,
            "sample_indices":indices,"warmup_steps":warmup,"measured_steps":steps,
            "scope":"Fixed CPU-cached dates; includes H2D, forward, loss, backward, optimizer. Excludes data assembly and checkpoint I/O; not predictive-quality evidence.",
            "full_training_precision":"fp32"}
    if a["successful_repeats"]!=repeats:result["status"]="failed"
    atomic_json(output,result);emit("benchmark_complete",status=result["status"],summary=summary,bf16_speedup=speedup)
    return result
