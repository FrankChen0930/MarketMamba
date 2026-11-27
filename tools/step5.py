"""Frozen Step5 R-A engineering contract. No fitting or evaluation entry point.

RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED.
Matrices use shared baseline/block components; labels remain opaque to admission.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from tools.step3 import sha256, verify_file, SEEDS, YEARS, PARAMS, FEATURES as MONTHLY

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'tasks/spec-step5-revenue-refinement-h20-20261127.md'
SPEC_SHA = '977ec654dd8018d9524259141e8b9d0b43d476b220ec289ef5fdfa960012d78c'
DECISION = ROOT / 'tasks/decision-step5-freeze-20261127.md'
DECISION_SHA = 'bb0e66fcad32e65a52874ab5458b41c29ff01ae6cb0db605b959db23d78224c2'
STEP3_BUNDLE_SHA = '97a9d3e8f43bfde984455f2439b0ba9bb22561beb3a65088dfbde27a7ea14613'
from tools.step4 import BASE48, BASELINE_FEATURES
EXTRA_FEATURES = ('revenue_age_sessions', 'revenue_yoy_acceleration',
                  'revenue_yoy_signed_streak', 'revenue_seasonal_mom')
ARMS = ('A', 'B')
CONTRASTS = ('B-A',)
FEATURE_ORDER = dict(A=BASELINE_FEATURES, B=BASELINE_FEATURES + EXTRA_FEATURES)
HOLM_M = 1
OUTPUT_ROOT = Path('/home/frank/projects/MarketMamba/.artifacts/step5')
SAMPLE_CAP = 500_000
MIN_FINITE = 2  # Spec: observable AND variable, not a percentage coverage gate.
LIMIT_BYTES = 40_000_000_000
THRESHOLDS = dict(mean_increment=.005, ci_lower_strict=0., alpha=.05,
                  min_daily_stocks=20, forward_valid_days=227, forward_sessions=252,
                  quarterly_announcement_quarters=4, quarterly_company_quarters=1000)
BOOTSTRAP = dict(block_sessions=20, replicates=1000, seed=17, ci_percentiles=[2.5, 97.5],
                 p='(1+count(abs(d_star-mean_d)>=abs(mean_d)))/1001', holm_m=1)


def frozen_decision(path=None):
    p = DECISION if path is None else Path(path)
    return verify_file(p, DECISION_SHA)


def contract():
    return dict(spec_sha256=SPEC_SHA, decision_sha256=DECISION_SHA, design='R-A',
                feature_order={k: list(v) for k, v in FEATURE_ORDER.items()},
                contrasts=list(CONTRASTS), years=list(YEARS), seeds=list(SEEDS),
                params=PARAMS, sample_cap=SAMPLE_CAP,
                support='each new column >=2 finite and min<max, each fit/eval partition',
                label='close(t+20)/open(t+1)-1', purge='t+21<origin',
                bootstrap=deepcopy(BOOTSTRAP), thresholds=deepcopy(THRESHOLDS),
                reuse='Step3 B matrices/samples/targets only; NO model or score reuse')


def new_output(path):
    p = Path(path).absolute()
    root = OUTPUT_ROOT
    if not p.is_relative_to(root) or p == root or p.resolve() != p:
        raise ValueError('STOP private namespace/symlink')
    if p.exists():
        raise ValueError('STOP output exists (no overwrite)')
    p.mkdir(parents=True)
    return p


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def file_path(directory, info):
    root = Path(directory).resolve()
    p = root / info['path']
    if Path(info['path']).is_absolute() or p.resolve() != p or not p.is_relative_to(root):
        raise ValueError('STOP unsafe bundle member')
    return p


def load(directory, bundle, ref):
    return np.load(file_path(directory, bundle['files'][ref]), mmap_mode='r', allow_pickle=False)


def matrix(directory, bundle, job, phase, rows=slice(None)):
    """Materialize one logical arm on demand; never access labels or fit a model."""
    if phase not in ('fit', 'eval') or job['arm'] not in ARMS:
        raise ValueError('STOP matrix role')
    base = load(directory, bundle, job[phase + '_base'])[rows]
    if base.ndim != 2 or base.shape[1] != 190 or base.dtype != np.float32:
        raise ValueError('STOP baseline matrix schema')
    if job['arm'] == 'A':
        return base
    extra = load(directory, bundle, job[phase + '_extra'])[rows]
    if extra.shape != (len(base), 4) or extra.dtype != np.float32:
        raise ValueError('STOP extra matrix schema')
    return np.column_stack([base, extra])



def job_id(year, seed, arm):
    return f'fold{year}-{arm}-seed{seed}'
