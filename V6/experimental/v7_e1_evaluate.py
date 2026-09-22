"""Exact-support E1 signal evaluation and frozen acceptance decision."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from V6.experimental.v7_e1_contract import ACCEPTANCE

KEYS=["Date","stock_id"]
def _rank_ic(group,pred,label):
 x=group[pred]; y=group[label]; mask=x.notna()&y.notna()
 return float(x[mask].corr(y[mask],method="spearman")) if mask.sum()>=3 else np.nan
def _spread(group,pred,label):
 clean=group[[pred,label]].dropna()
 if len(clean)<10: return np.nan
 rank=clean[pred].rank(method="first")
 bucket=pd.qcut(rank,10,labels=False,duplicates="drop")
 means=clean.groupby(bucket,observed=True)[label].mean()
 return float(means.iloc[-1]-means.iloc[0]) if len(means)==10 else np.nan
def _monotonicity(group,pred,label):
 clean=group[[pred,label]].dropna()
 if len(clean)<10:return np.nan
 bucket=pd.qcut(clean[pred].rank(method="first"),10,labels=False,duplicates="drop")
 means=clean.groupby(bucket,observed=True)[label].mean()
 return float(pd.Series(range(len(means))).corr(means.reset_index(drop=True),method="spearman"))
def daily_metrics(frame,prefix="prediction"):
 rows=[]
 for day,g in frame.groupby("Date",sort=True):
  row={"Date":str(day)[:10]}
  for h in (5,10):
   pred=f"{prefix}_{h}d"; label=f"Alpha_{h}d"
   row[f"rank_ic_{h}d"]=_rank_ic(g,pred,label)
   row[f"d10_minus_d1_{h}d"]=_spread(g,pred,label)
   row[f"monotonicity_{h}d"]=_monotonicity(g,pred,label)
  rows.append(row)
 return pd.DataFrame(rows)
def summarize(frame,prefix="prediction"):
 daily=daily_metrics(frame,prefix)
 out={"dates":len(daily),"rows":len(frame),"daily":daily.to_dict("records")}
 for h in (5,10):
  ic=daily[f"rank_ic_{h}d"]; spread=daily[f"d10_minus_d1_{h}d"]
  out[f"mean_rank_ic_{h}d"]=float(ic.mean())
  out[f"positive_days_{h}d"]=int((ic>0).sum())
  out[f"positive_day_fraction_{h}d"]=float((ic>0).mean())
  out[f"mean_d10_minus_d1_{h}d"]=float(spread.mean())
  out[f"mean_monotonicity_{h}d"]=float(daily[f"monotonicity_{h}d"].mean())
  monthly=daily.assign(month=daily.Date.str[:7]).groupby("month")[f"rank_ic_{h}d"].mean()
  out[f"monthly_rank_ic_{h}d"]={str(k):float(v) for k,v in monthly.items()}
 return out
def rank_average_ensemble(seed_frames):
 merged=None
 for seed,frame in sorted(seed_frames.items()):
  cols=KEYS+["Alpha_5d","Alpha_10d","prediction_5d","prediction_10d"]
  x=frame[cols].copy()
  if x.duplicated(KEYS).any(): raise ValueError(f"duplicate prediction key seed {seed}")
  for h in (5,10):
   x[f"rank_{seed}_{h}d"]=x.groupby("Date")[f"prediction_{h}d"].rank(pct=True)
  keep=KEYS+[f"rank_{seed}_5d",f"rank_{seed}_10d"]
  if merged is None: merged=x[cols+keep[2:]]
  else: merged=merged.merge(x[keep],on=KEYS,how="inner",validate="one_to_one")
 rank_cols={h:[f"rank_{s}_{h}d" for s in sorted(seed_frames)] for h in (5,10)}
 for h in (5,10): merged[f"prediction_{h}d"]=merged[rank_cols[h]].mean(axis=1)
 return merged[KEYS+["Alpha_5d","Alpha_10d","prediction_5d","prediction_10d"]]
def paired_daily_delta(new,old):
 required=KEYS+["prediction_5d","prediction_10d","Alpha_5d","Alpha_10d"]
 n=new[required].rename(columns={f"prediction_{h}d":f"new_{h}d" for h in (5,10)})
 o=old[KEYS+["prediction_5d","prediction_10d"]].rename(
  columns={f"prediction_{h}d":f"old_{h}d" for h in (5,10)})
 paired=n.merge(o,on=KEYS,how="inner",validate="one_to_one")
 rows=[]
 for day,g in paired.groupby("Date",sort=True):
  row={"Date":str(day)[:10],"rows":len(g)}
  for h in (5,10):
   label=f"Alpha_{h}d"
   row[f"new_ic_{h}d"]=_rank_ic(g,f"new_{h}d",label)
   row[f"old_ic_{h}d"]=_rank_ic(g,f"old_{h}d",label)
   row[f"delta_ic_{h}d"]=row[f"new_ic_{h}d"]-row[f"old_ic_{h}d"]
  rows.append(row)
 return pd.DataFrame(rows),{"new_rows":len(new),"old_rows":len(old),"paired_rows":len(paired)}
def block_bootstrap_lower(values,block=20,repeats=2000,seed=20260919):
 values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
 if not len(values): return np.nan
 rng=np.random.default_rng(seed); n=len(values); means=[]
 for _ in range(repeats):
  sampled=[]
  while len(sampled)<n:
   start=int(rng.integers(0,n)); sampled.extend(values[(start+np.arange(block))%n])
  means.append(float(np.mean(sampled[:n])))
 return float(np.quantile(means,.025))
def decide(seed_summaries,ensemble_summary,paired_daily,bootstrap_repeats=2000):
 lower5=block_bootstrap_lower(paired_daily["delta_ic_5d"],repeats=bootstrap_repeats)
 lower10=block_bootstrap_lower(paired_daily["delta_ic_10d"],repeats=bootstrap_repeats)
 checks={
  "10d_mean_ic":ensemble_summary["mean_rank_ic_10d"]>=ACCEPTANCE["10d_mean_ic_min"],
  "10d_paired_improvement":lower10>ACCEPTANCE["10d_paired_delta_bootstrap_lower_95_gt"],
  "5d_mean_ic":ensemble_summary["mean_rank_ic_5d"]>=ACCEPTANCE["5d_mean_ic_min"],
  "5d_noninferiority":lower5>ACCEPTANCE["5d_noninferiority_lower_95_gt"],
  "positive_spreads":all(ensemble_summary[f"mean_d10_minus_d1_{h}d"]>0 for h in (5,10)),
  "seed_nonnegative":all(s[f"mean_rank_ic_{h}d"]>=0 for s in seed_summaries.values() for h in (5,10))}
 if all(checks.values()): outcome="PASS_REFRESH_HYPOTHESIS"
 elif not checks["10d_mean_ic"] and not checks["5d_mean_ic"]: outcome="FAIL_REFRESH_HYPOTHESIS"
 else: outcome="PARTIAL_SUPPORT"
 return {"outcome":outcome,"checks":checks,
         "paired_bootstrap_lower_95":{"5d":lower5,"10d":lower10},
         "thresholds":dict(ACCEPTANCE)}

def _labels_from_matrix(root):
 root=Path(root); dates=np.load(root/"dates.npy",mmap_mode="r")
 stocks=np.load(root/"stock_ids.npy",mmap_mode="r")
 splits=np.load(root/"splits.npy",mmap_mode="r"); y5=np.load(root/"y5.npy",mmap_mode="r")
 y10=np.load(root/"y10.npy",mmap_mode="r"); mask=splits==2
 return pd.DataFrame({"Date":dates[mask].astype(str),"stock_id":stocks[mask].astype(str),
                      "Alpha_5d":y5[mask],"Alpha_10d":y10[mask]})
def _with_labels(path,labels):
 frame=pd.read_parquet(path); frame["Date"]=frame.Date.astype(str).str[:10]
 frame["stock_id"]=frame.stock_id.astype(str)
 return frame.merge(labels,on=KEYS,how="inner",validate="one_to_one")
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--matrix",type=Path,required=True)
 p.add_argument("--seed17",type=Path,required=True); p.add_argument("--seed29",type=Path,required=True)
 p.add_argument("--old-seed17",type=Path,required=True); p.add_argument("--old-seed29",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True); a=p.parse_args(argv)
 labels=_labels_from_matrix(a.matrix)
 seeds={17:_with_labels(a.seed17,labels),29:_with_labels(a.seed29,labels)}
 old={17:_with_labels(a.old_seed17,labels),29:_with_labels(a.old_seed29,labels)}
 seed_summary={seed:summarize(frame) for seed,frame in seeds.items()}
 ensemble=rank_average_ensemble(seeds); old_ensemble=rank_average_ensemble(old)
 ensemble_summary=summarize(ensemble); old_summary=summarize(old_ensemble)
 paired,support=paired_daily_delta(ensemble,old_ensemble)
 decision=decide(seed_summary,ensemble_summary,paired)
 result={"schema_version":"v7-e1-evaluation-v1","support":support,
  "seed_metrics":seed_summary,"ensemble_metrics":ensemble_summary,
  "incumbent_metrics_on_paired_labels":old_summary,
  "paired_daily":paired.to_dict("records"),"decision":decision,
  "strict_phase0":"STOP","historical_simulation_readiness":"PASS",
  "evidence_class":"HISTORICAL_SIMULATION_PROXY"}
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/"evaluation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
 lines=["# V7 E1 Rolling-Origin Refresh Report","",
  f"- Outcome: **{decision['outcome']}**",
  f"- Exact paired rows: {support['paired_rows']:,}",
  f"- E1 ensemble IC5: {ensemble_summary['mean_rank_ic_5d']:.6f}",
  f"- E1 ensemble IC10: {ensemble_summary['mean_rank_ic_10d']:.6f}",
  f"- Paired delta lower 95% (5d): {decision['paired_bootstrap_lower_95']['5d']:.6f}",
  f"- Paired delta lower 95% (10d): {decision['paired_bootstrap_lower_95']['10d']:.6f}",
  "","Strict Phase 0 remains STOP. This is historical-simulation evidence, not tradability evidence."]
 (a.output/"v7_e1_rolling_origin_refresh_report.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"stage":"EVALUATION","outcome":decision["outcome"],"paired_rows":support["paired_rows"]}))
 return 0
if __name__=="__main__": raise SystemExit(main())
