"""Confirmation design: paired seeds first, then width-64 temporal depth."""
import json
import math
import statistics
from pathlib import Path
from v7_experiment_storage import fingerprint, digest
from v7_experiment_suite import STAGES as ORIGINAL_STAGES

STAGES=("E3-seed29","E5-seed29","E3-seed43","E5-seed43","T3-width64-seed17")
SEEDS=(29,29,43,43,17)
REFERENCE=Path(__file__).with_name("v7_confirmation_reference.json")

def read_reference():
    return json.loads(REFERENCE.read_text())

def metrics(result):
    if result.get("status")!="complete":
        raise ValueError("paired comparison requires all completed results")
    row=max(result["history"],key=lambda h:h["validation"]["rank_ic_5d"])
    v=row["validation"]
    if not all(math.isfinite(v[k]) for k in ("rank_ic_5d","rank_ic_10d")):
        raise ValueError("paired comparison requires finite validation scores")
    if v["rank_ic_5d"]!=result["best_rank_ic_5d"]:
        raise ValueError("best checkpoint metric disagrees with history")
    return v

def comparison(completed, reference):
    rows={}
    for family,old in (("E3","E3-width64"),("E5","E5-state32")):
        results=[reference["results"][old],completed[family+"-seed29"],completed[family+"-seed43"]]
        values=[metrics(r) for r in results]
        # Counts must agree within each horizon; never mix incomplete validation.
        for key in ("rank_ic_5d_dates","rank_ic_10d_dates"):
            if len({v[key] for v in values})!=1:
                raise ValueError("validation date coverage differs between seeds")
        rows[family]={"seeds":[17,29,43],"values":values}
        for key in ("rank_ic_5d","rank_ic_10d"):
            rows[family][key]={"mean":statistics.mean(v[key] for v in values),
                              "sample_std":statistics.stdev(v[key] for v in values)}
    for a,b in zip(rows["E3"]["values"],rows["E5"]["values"]):
        for key in ("rank_ic_5d_dates","rank_ic_10d_dates"):
            if a[key]!=b[key]:raise ValueError("paired validation coverage mismatch")
    diffs=[b["rank_ic_5d"]-a["rank_ic_5d"] for a,b in zip(rows["E3"]["values"],rows["E5"]["values"])]
    # Descriptive budget rule, not a significance test.
    chosen=32 if statistics.mean(diffs)>0 and sum(d>0 for d in diffs)>=2 else 8
    return {"models":rows,"paired_ic5_differences":diffs,"selected_d_state":chosen,
            "rule":"Choose state32 only if mean paired IC5 improvement > 0 and at least 2/3 seeds improve; otherwise state8.",
            "interpretation":"Three seeds describe sensitivity, not statistical significance or an untouched test."}

def resolve_spec(stage, completed, reference, diagnostic=False):
    family="E3" if stage in (0,2) else "E5"
    cfg=dict(reference["results"]["E3-width64" if family=="E3" else "E5-state32"]["config"])
    if stage==4:
        if diagnostic:
            # Smoke data must never be pooled with historical full-matrix metrics.
            chosen=32 if completed["E5-seed43"]["best_rank_ic_5d"]>completed["E3-seed43"]["best_rank_ic_5d"] else 8
        else:chosen=comparison(completed,reference)["selected_d_state"]
        cfg.update(temporal_layers=3,forward_layers=1,reverse_layers=1,d_model=64,d_state=chosen)
    return {"id":STAGES[stage],"config":cfg}

def validate_reference(reference, matrix, splits, runtime, here):
    # Training/model/data code must remain identical to the seed17 experiment.
    for name,sha in reference["original_sources"].items():
        if digest(Path(here)/name)!=sha:
            raise ValueError("seed17 reference source differs: "+name)
    payload={"matrix":matrix,"splits":splits,"settings":reference["settings"],
             "sources":reference["original_sources"],"runtime":runtime,
             "stages":ORIGINAL_STAGES,"diagnostic":False}
    # The old suite saved only a hash. Recover its Python patch version by exact
    # hash matching; this does not relax any matrix, split or package checks.
    candidates=[runtime]+[{**runtime,"python":f"3.12.{i}"} for i in range(31)]
    for version in candidates:
        payload["runtime"]=version
        if fingerprint(payload)==reference["suite_identity"]:
            return {"verified":True,"original_runtime":version,"original_suite":reference["suite_identity"]}
    raise ValueError("資料／切分／套件與 seed17 原始實驗不一致，不能合併比較；請保留原矩陣與固定環境。")
