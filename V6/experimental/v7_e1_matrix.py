"""E1 matrix identity and isolated materialization helpers."""
import argparse
from dataclasses import replace
import json,os
from pathlib import Path
import numpy as np
from V6.experimental.v7_corrected_e5_matrix import (
 MATRIX_ARRAYS,_normalized_frames,load_full_inputs)
from V6.experimental.v7_corrected_e5_replication import (
 CorrectedE5Config,REQUIRED_SOURCE_HASHES,canonical_hash,file_sha256)
from V6.experimental.v7_e1_contract import MATRIX_NAMESPACE,SPLIT,assign_split,validate_contract,validate_split_rows
SCHEMA="v7-e1-matrix-v1"
def validate_e1_upstream(source,coverage):
 from V6.experimental.v7_corrected_e5_replication import ALLOWED_SOURCE_CLASSES
 admitted={k for k,v in source.get("classes",{}).items() if v.get("eligible") is True}
 if admitted!=ALLOWED_SOURCE_CLASSES: raise ValueError("E1 source eligibility drift")
 expected={"evidence_class":"HISTORICAL_SIMULATION_PROXY","selected_proxy_policy":"P0",
           "strict_phase0_decision":"STOP","materialization_status":"MATERIALIZED"}
 for key,value in expected.items():
  if coverage.get(key)!=value: raise ValueError("E1 v2 label evidence drift: "+key)
 if coverage.get("historical_simulated_valid_labels")!={"5d":7023920,"10d":6923464}:
  raise ValueError("E1 v2 label counts drift")
 if coverage.get("corrected_verified_executable_labels")!={"5d":0,"10d":0}:
  raise ValueError("verified executable labels must remain zero")

def e1_config(base):
 return replace(base,train_end=SPLIT["train_end"],evaluation_start=SPLIT["evaluation_start"],
                evaluation_end=SPLIT["evaluation_end_5d"])
def build_e1_arrays(features,universe,labels,base_config):
 config=e1_config(base_config); frame=_normalized_frames(features,universe,labels,config)
 values=frame.loc[:,config.feature_order].to_numpy(dtype=np.float32,copy=True)
 masks=np.isfinite(values); values[~masks]=0.
 dates=frame["Date"].to_numpy(dtype="datetime64[D]")
 names=np.array([assign_split(str(x)) for x in dates],dtype="U24")
 validate_split_rows(dates.astype(str),names)
 encoded=np.array([{"warmup":0,"train":1,"research_evaluation":2,
                    "out_of_contract":3,"purge":4}[x] for x in names],dtype=np.uint8)
 _,counts=np.unique(dates,return_counts=True)
 return {"X":values,"y5":frame["Alpha_5d"].to_numpy(dtype=np.float32),
  "y10":frame["Alpha_10d"].to_numpy(dtype=np.float32),
  "stock_ids":frame["stock_id"].astype(str).to_numpy(dtype="U32"),"dates":dates,
  "end_indices":np.cumsum(counts,dtype=np.int64),"splits":encoded,
  "split_names":names,"masks":masks}
def make_manifest(contract,config,arrays,files,source_hashes,build_mode="full_memory"):
 validate_contract(contract)
 if set(files)!=set(MATRIX_ARRAYS): raise ValueError("E1 matrix file set drift")
 if set(source_hashes)!=REQUIRED_SOURCE_HASHES: raise ValueError("E1 source hash set drift")
 value={"schema_version":SCHEMA,"namespace":MATRIX_NAMESPACE,
  "contract_sha256":contract["contract_sha256"],"config_sha256":config.sha256,
  "rows":len(arrays["dates"]),"files":dict(files),"source_hashes":dict(source_hashes),
  "build_mode":build_mode,"feature_order":list(config.feature_order),
  "split_contract":dict(SPLIT),"ordering":["Date","stock_id"],
  "preprocessing":{"industry_neutralization":False,"current_industry_backfill":False,
   "cross_sectional_scaling":"causal per-session","macro_scaling":"causal expanding"},
  "graph":{"enabled":False,"edge_count":0,"fusion_branch":"ABSENT"},
  "label_contract":{"evidence_class":"HISTORICAL_SIMULATION_PROXY","proxy_policy":"P0",
                    "source_version":"v2","horizons":[5,10]}}
 value["logical_identity"]=canonical_hash(value); validate_manifest(value,contract,config)
 return value
def validate_manifest(value,contract,config):
 checks={"schema":value.get("schema_version")==SCHEMA,
  "namespace":value.get("namespace")==MATRIX_NAMESPACE,
  "contract":value.get("contract_sha256")==contract["contract_sha256"],
  "config":value.get("config_sha256")==config.sha256,
  "split":value.get("split_contract")==SPLIT,
  "ordering":value.get("ordering")==["Date","stock_id"],
  "graph":value.get("graph",{}).get("enabled") is False,
  "industry":value.get("preprocessing",{}).get("industry_neutralization") is False,
  "identity":value.get("logical_identity")==canonical_hash(
   {k:v for k,v in value.items() if k!="logical_identity"})}
 errors=[k for k,v in checks.items() if not v]
 if errors: raise ValueError("invalid E1 matrix manifest: "+", ".join(errors))
def _atomic_npy(path,value):
 tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("wb") as f: np.save(f,value,allow_pickle=False); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def write_matrix(output,contract,base_config,arrays,source_hashes):
 output=Path(output)
 if output.name!="v7_e1_rolling_origin_refresh_v1":
  raise ValueError("E1 matrix must use isolated namespace")
 output.mkdir(parents=True,exist_ok=True)
 for filename,key in MATRIX_ARRAYS.items(): _atomic_npy(output/filename,np.asarray(arrays[key]))
 files={name:file_sha256(output/name) for name in MATRIX_ARRAYS}
 doc=make_manifest(contract,e1_config(base_config),arrays,files,source_hashes)
 tmp=output/"manifest.json.tmp"
 with tmp.open("w",encoding="utf-8") as f:
  json.dump(doc,f,ensure_ascii=False,sort_keys=True,indent=2); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,output/"manifest.json"); return doc
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--feature-cache",type=Path,required=True)
 p.add_argument("--universe",type=Path,required=True)
 p.add_argument("--labels",type=Path,required=True)
 p.add_argument("--source-reliability",type=Path,required=True)
 p.add_argument("--label-coverage",type=Path,required=True)
 p.add_argument("--incumbent-contract",type=Path,required=True)
 p.add_argument("--e1-contract",type=Path,required=True)
 p.add_argument("--feature-manifest",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args(argv)
 if a.output.name!="v7_e1_rolling_origin_refresh_v1":
  raise ValueError("output must be Data/v7_e1_rolling_origin_refresh_v1")
 contract=json.loads(a.e1_contract.read_text()); validate_contract(contract)
 base=CorrectedE5Config.from_contract(a.incumbent_contract,a.feature_manifest)
 source=json.loads(a.source_reliability.read_text())
 coverage=json.loads(a.label_coverage.read_text())
 validate_e1_upstream(source,coverage)
 features,universe,labels,source_hashes=load_full_inputs(a,e1_config(base))
 arrays=build_e1_arrays(features,universe,labels,base)
 doc=write_matrix(a.output,contract,base,arrays,source_hashes)
 print(json.dumps({"stage":"MATRIX_BUILD","status":"COMPLETE","rows":doc["rows"],
  "logical_identity":doc["logical_identity"]},ensure_ascii=False),flush=True)
 return 0

if __name__=="__main__":
 raise SystemExit(main())
