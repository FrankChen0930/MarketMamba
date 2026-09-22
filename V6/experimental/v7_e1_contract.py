"""Fail-closed contract for E1 rolling-origin unchanged-contract refresh."""
from __future__ import annotations
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from V6.experimental.v7_corrected_e5_replication import canonical_hash

SCHEMA="v7-e1-rolling-origin-refresh-contract-v1"
MATRIX_NAMESPACE="Data/v7_e1_rolling_origin_refresh_v1"
OUTPUT_NAMESPACE="MarketMamba_V7_E1_Rolling_Origin_Refresh"
SEEDS=(17,29)
LABEL_COUNTS={"5d":7023920,"10d":6923464}
SPLIT={"train_start":"2013-01-02","train_end":"2025-11-18",
 "purge_start":"2025-11-19","purge_end":"2025-12-31","purge_sessions":30,
 "evaluation_start":"2026-01-02","evaluation_end_5d":"2026-09-04",
 "evaluation_end_10d":"2026-08-28"}
ACCEPTANCE={"10d_mean_ic_min":.03,"10d_paired_delta_bootstrap_lower_95_gt":0.,
 "5d_mean_ic_min":.04,"5d_noninferiority_lower_95_gt":-.01,
 "d10_minus_d1_positive_both_horizons":True,
 "both_seeds_non_negative_both_horizons":True}
ARCH={"d_model":64,"d_state":32,"expand":2,"head_dim":8,"n_groups":1,
 "sequence_length":60,"temporal_layers":1,"cross_stock_forward_layers":1,
 "cross_stock_reverse_layers":1,"dropout":0.,"horizons":[5,10],
 "graph":{"enabled":False,"edge_count":0,"fusion_branch":"ABSENT"}}
TRAIN={"epochs":20,"patience":5,"minimum_epochs":5,"learning_rate":.0001,
 "weight_decay":.01,"warmup_fraction":.15,"minimum_lr_factor":.0001,
 "gradient_clip":1.,"precision":"fp32","optimizer":"AdamW",
 "schedule":"linear warmup then cosine","selection_metric":"mean_daily_rank_ic_5d",
 "secondary_metric":"mean_daily_rank_ic_10d_from_same_best_ic5_checkpoint"}

def _load(path):
 value=json.loads(Path(path).read_text(encoding="utf-8"))
 if not isinstance(value,dict): raise ValueError("expected JSON object")
 return value

def build_contract(incumbent_path,decision_path,feature_manifest_path):
 incumbent,decision,features=map(_load,(incumbent_path,decision_path,feature_manifest_path))
 if decision.get("selected")!="E1 rolling-origin unchanged-contract refresh": raise ValueError("E1 not selected")
 if decision.get("gpu_experiment_executed") is not False: raise ValueError("execution state drift")
 if decision.get("split")!=SPLIT: raise ValueError("split drift")
 if tuple(decision["frozen_contract"].get("seeds",()))!=SEEDS: raise ValueError("seed drift")
 if features.get("feature_count")!=48: raise ValueError("48 features required")
 f=incumbent["fields"]
 c={"schema_version":SCHEMA,"name":"E1 Rolling-Origin Unchanged-Contract Refresh",
  "baseline":incumbent["baseline"],"hypothesis":decision["hypothesis"],
  "frozen":{"features":deepcopy(f["features"]["value"]),
   "architecture":deepcopy(f["architecture"]["value"]),
   "training":deepcopy(f["training"]["value"]),
   "labels":{**deepcopy(f["labels"]["value"]),"valid_5d":LABEL_COUNTS["5d"],
             "valid_10d":LABEL_COUNTS["10d"],"source_version":"v2"},
   "preprocessing":deepcopy(incumbent["matrix"]["preprocessing"]),
   "environment":f["environment"]["value"]},
  "seeds":list(SEEDS),"split":deepcopy(SPLIT),"acceptance":deepcopy(ACCEPTANCE),
  "correctness_semantics":{"strict_phase0":"STOP","historical_simulation_readiness":"PASS",
   "corrected_verified_executable_labels":{"5d":0,"10d":0},
   "evidence_class":"HISTORICAL_SIMULATION_PROXY","proxy_policy":"P0"},
  "matrix":{"namespace":MATRIX_NAMESPACE,"reuse_incumbent_matrix":False,
            "reuse_stable_inputs":True,"ordering":["Date","stock_id"]},
  "output_namespace":OUTPUT_NAMESPACE}
 validate_contract(c); c["contract_sha256"]=canonical_hash(c); return c

def validate_contract(c):
 f=c.get("frozen",{}); labels=f.get("labels",{}); semantics=c.get("correctness_semantics",{})
 checks={"schema":c.get("schema_version")==SCHEMA,
  "features":f.get("features")=={"input_dim":48,"group_dims":[15,20,1,12],"order_from":"feature-manifest.json"},
  "architecture":f.get("architecture")==ARCH,"training":f.get("training")==TRAIN,
  "seeds":tuple(c.get("seeds",()))==SEEDS,"split":c.get("split")==SPLIT,
  "labels":labels.get("evidence_class")=="HISTORICAL_SIMULATION_PROXY" and
           labels.get("proxy_policy")=="P0" and
           (labels.get("valid_5d"),labels.get("valid_10d"))==(LABEL_COUNTS["5d"],LABEL_COUNTS["10d"]),
  "preprocessing":f.get("preprocessing")=={"industry_neutralization":False,
   "current_industry_backfill":False,"cross_sectional_scaling":"causal per-session",
   "macro_scaling":"causal expanding"},
  "semantics":semantics.get("strict_phase0")=="STOP" and
   semantics.get("historical_simulation_readiness")=="PASS" and
   semantics.get("corrected_verified_executable_labels")=={"5d":0,"10d":0},
  "matrix":c.get("matrix",{}).get("namespace")==MATRIX_NAMESPACE and
   c.get("matrix",{}).get("reuse_incumbent_matrix") is False,
  "acceptance":c.get("acceptance")==ACCEPTANCE}
 errors=[k for k,v in checks.items() if not v]
 if errors: raise ValueError("invalid E1 contract: "+", ".join(errors))

def contract_diff(c):
 validate_contract(c)
 return {"schema_version":"v7-e1-contract-diff-v1",
  "substantive_changes":[{"field":"split.train_end","from":"2023-11-17",
   "to":SPLIT["train_end"],"derived":{"purge":[SPLIT["purge_start"],SPLIT["purge_end"]],
   "evaluation_start":SPLIT["evaluation_start"],"maturity":{
   "5d":SPLIT["evaluation_end_5d"],"10d":SPLIT["evaluation_end_10d"]}}}],
  "execution_scope":{"from_seeds":[17,29,43],"to_seeds":list(SEEDS),
                     "reason":"pre-registered E1 scope; not model search"},
  "unchanged":["48-feature contract","architecture","heads","loss","optimizer",
   "schedule","selection","FP32","P0 labels","provenance gates",
   "no industry neutralization","no graph"]}

def assign_split(value):
 d=date.fromisoformat(str(value)[:10])
 if d<date.fromisoformat(SPLIT["train_start"]): return "warmup"
 if d<=date.fromisoformat(SPLIT["train_end"]): return "train"
 if d<=date.fromisoformat(SPLIT["purge_end"]): return "purge"
 if d<=date.fromisoformat(SPLIT["evaluation_end_5d"]): return "research_evaluation"
 return "out_of_contract"

def validate_split_rows(dates,splits):
 pairs=list(zip((str(x)[:10] for x in dates),map(str,splits)))
 if not pairs: raise ValueError("E1 matrix has no rows")
 for day,seen in pairs:
  expected=assign_split(day)
  if seen!=expected: raise ValueError(f"E1 split mismatch at {day}: {seen} != {expected}")
 train=[d for d,s in pairs if s=="train"]; evaluation=[d for d,s in pairs if s=="research_evaluation"]
 if train and max(train)>SPLIT["train_end"]: raise ValueError("training leakage")
 if evaluation and min(evaluation)<SPLIT["evaluation_start"]: raise ValueError("evaluation leakage")
