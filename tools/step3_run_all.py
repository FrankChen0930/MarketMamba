"""Run the frozen Step3 72-job historical queue from the admitted bundle (fits only, no evaluation).

Mirrors step3_runner.measure for every job in fixed order; resumable via fit_job/resume.
Requires an explicit Human source-gate decision record.
"""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np
from tools.step3 import require_admission,job_id,SEEDS,YEARS
from tools.step3_runner import fit_job



def run_all(bundle_dir,decision,out_name='fits'):
    directory=Path(bundle_dir).absolute()
    gate=json.loads((directory/'admission/admission.json').read_text())
    bundle=require_admission(gate,decision)
    if len(bundle['jobs'])!=72:raise ValueError('STOP queue size')
    done=0
    for year in YEARS:
        for seed in SEEDS:
            for arm in ('A','B'):
                job=bundle['jobs'][job_id(year,seed,arm)]
                load=lambda role:np.load(bundle['files'][job[role+'_file']]['path'],mmap_mode='r',allow_pickle=False)
                t=time.monotonic()
                fit_job(gate,directory/out_name,year,seed,arm,load('keys'),load('sample'),load('target'),
                        load('fit'),load('eval'),gate['bundle']['sha256'],human_accepted_source_gate=decision)
                done+=1;print(f'{done}/72 {job_id(year,seed,arm)} {time.monotonic()-t:.1f}s',flush=True)
    return done


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir',type=Path,required=True);p.add_argument('--human-accepted-source-gate',type=Path,required=True)
    a=p.parse_args();print('jobs',run_all(a.bundle_dir,a.human_accepted_source_gate))
