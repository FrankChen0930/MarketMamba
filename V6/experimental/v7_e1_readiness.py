"""Read-only audit of the admitted E1 metadata chain; never authorizes a run.

Uses fixed Git result blobs, existing candidate contract/config validators and
hashes only. Does not evaluate predictions, read matrix/weights or start a process
other than local git. This is not a generic artifact admission mechanism.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from V6.experimental.v7_corrected_e5_replication import CorrectedE5Config
from V6.experimental.v7_e1_contract import ACCEPTANCE, SEEDS, validate_contract
from V6.experimental.v7_e1_matrix import e1_config

RESULT_COMMIT = 'eec63c551270f271e26a5aca831fe5b3b38e3f42'
RESULT_ROOT = 'research/v7/e1-rolling-origin-refresh-v1/results/'
MANIFEST_SHA256 = 'f4d63c74af623991814d301230f065818850829d69a176309bd7769c7b76a1a6'


def read_result_blob(repo: Path, name: str) -> bytes:
    return subprocess.check_output(['git', 'show', RESULT_COMMIT+':'+RESULT_ROOT+name],
                                   cwd=repo, stderr=subprocess.PIPE)


def candidate_config(repo: Path):
    return e1_config(CorrectedE5Config.from_contract(
        repo/'research/v7/corrected-e5-replication-v1/incumbent-contract.json',
        repo/'research/v7/corrected-baseline-v1/feature-manifest.json'))


def audit(repo: Path, *, read_blob=None, config=None) -> dict:
    repo=Path(repo)
    reader=read_blob if read_blob is not None else lambda name:read_result_blob(repo,name)
    def read(name):
        try: return reader(name)
        except (OSError,subprocess.CalledProcessError) as exc:
            raise ValueError('missing/unreadable result artifact: '+name) from exc
    raw=read('result-manifest.json')
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA256:
        raise ValueError('result manifest is not the admitted fixed identity')
    manifest=json.loads(raw)
    documents={}
    for name,expected in manifest['copied_files'].items():
        raw=read(name)
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('result artifact hash mismatch: '+name)
        if name.endswith('.json'): documents[name]=json.loads(raw)
    contract=json.loads((repo/'research/v7/e1-rolling-origin-refresh-v1/experiment-contract.json').read_text())
    validate_contract(contract)
    cfg=candidate_config(repo) if config is None else config
    run=documents['run-summary.json']; evaluation=documents['evaluation.json']
    seed_keys={str(seed) for seed in SEEDS}
    checks=[manifest['contract_identity']==contract['contract_sha256'],
            manifest['formal_seeds']==list(SEEDS),
            run['contract_sha256']==contract['contract_sha256'],
            set(run['seeds'])==seed_keys, set(evaluation['seed_metrics'])==seed_keys,
            evaluation['decision']['thresholds']==ACCEPTANCE,
            evaluation['decision']['outcome']==manifest['outcome'],
            manifest['strict_phase0']==run['strict_phase0']==evaluation['strict_phase0']=='STOP',
            manifest['portfolio_admission'] is False]
    for seed in SEEDS:
        result=documents[f'seed-{seed}-result.json']
        pointer=documents[f'seed-{seed}-checkpoint.json']
        checks.extend([result['seed']==seed, result['status']=='complete',
                       run['seeds'][str(seed)]['seed']==seed,
                       run['seeds'][str(seed)]['status']=='complete',
                       result['config_sha256']==cfg.sha256,
                       result['matrix_identity']==manifest['matrix_identity'],
                       result['checkpoint_identity']==pointer['best']['sha256'],
                       result['best_epoch']==pointer['best']['epoch'],
                       pointer['latest']['terminal'] is True])
    if not all(checks): raise ValueError('candidate/result metadata identity mismatch')
    return {'status':'VERIFIED_METADATA_ONLY','result_commit':RESULT_COMMIT,
            'result_manifest_sha256':MANIFEST_SHA256,
            'copied_artifacts_verified':len(manifest['copied_files']),
            'contract_sha256':contract['contract_sha256'],'config_sha256':cfg.sha256,
            'declared_matrix_identity':manifest['matrix_identity'],
            'split':contract['split'],'formal_seeds':list(SEEDS),'features':len(cfg.feature_order),
            'execution_ready':False,'matrix_payload':'NOT_VERIFIED',
            'checkpoint_payload':'NOT_LOADED','strict_phase0':'STOP',
            'h2_admission':'NOT_APPROVED','research_evaluation_performed':False,
            'limitations':['Original matrix required for payload verification',
                           'No checkpoint deserialization or live environment validation',
                           'New comparison/training requires separate Human approval']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args(argv)
    try: report=audit(args.repo)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        parser.exit(1,'E1 metadata audit refused: '+str(exc)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0


if __name__=='__main__': raise SystemExit(main())
