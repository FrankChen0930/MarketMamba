"""Frozen Step4 Q-A engineering contract. No fitting or evaluation entry point.

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
SPEC = ROOT / 'tasks/spec-step4-peer-quarterly-h20-20261127.md'
SPEC_SHA = '79a0c24bc57369d03edc709933b44c21a95e81f0c532127f72af93156c8e7a5d'
DECISION = ROOT / 'tasks/decision-step4-freeze-20261127.md'
DECISION_SHA = '18909bacde1cbe4096fb52b66317b6d56d25a51ee4ffc930a6f8fce04f8e7744'
STEP3_BUNDLE_SHA = '97a9d3e8f43bfde984455f2439b0ba9bb22561beb3a65088dfbde27a7ea14613'
PEER_MANIFEST_SHA = '08b6ecb48c5a99cc35c761a0a3a5f606ca6a2c40e7a55ae6000ad5663e76f248'
PEER_REVIEW_SHA = 'd45ca09ddaaa5607900e8e0186b76ba4698da05decc43b8d7486eefebaeffe37'
BASE48 = tuple('Open High Low Close Volume Return_1d Return_5d Return_20d MA_20 MA_60 RSI_14 ATR_14 RS_5d RS_20d RS_60d Foreign_Buy Foreign_Sell Foreign_Net Investment_Trust_Net Dealer_Net Margin_Purchase Margin_Repay Short_Sale Short_Cover Margin_Balance Short_Balance Day_Trade_Volume KD_K KD_D OBV Volatility_20d Holdings_Large_Pct Holdings_Large_Change Securities_Balance Foreign_Holding_Pct Market_Cap_Log TWII_Return SPX_Return VIX TNX Gold_Return Oil_Return USD_TWD Futures_OI_Foreign Options_PC_Ratio Fear_Greed Business_Signal FED_Rate'.split())
CORE = tuple('Return_1d Return_5d Return_20d RS_5d RS_20d RS_60d RSI_14 ATR_14 Volatility_20d Volume Foreign_Net Investment_Trust_Net Dealer_Net Margin_Balance Short_Balance Day_Trade_Volume'.split())
BASELINE_FEATURES = (BASE48 + tuple(n for c in CORE for n in
    [*(f'{c}_lag{k}' for k in (1, 2, 5, 10, 20)), *(f'{c}_mean{k}' for k in (5, 20, 60))])
    + tuple(f'mkt_{c}_{k}' for c in ('Return_5d', 'Return_20d', 'Volatility_20d')
            for k in ('mean', 'std', 'pos')) + tuple(MONTHLY))
FEATURES_1A = ('peer_return', 'return_peer_count', 'market_return')
FEATURES_1B = ('revenue_spillover', 'revenue_peer_count')
EXTRA_FEATURES = FEATURES_1A + FEATURES_1B
ARMS = ('A', 'B1a', 'B1b')
CONTRASTS = ('B1a-A', 'B1b-A')
FEATURE_ORDER = dict(A=BASELINE_FEATURES, B1a=BASELINE_FEATURES + FEATURES_1A,
                     B1b=BASELINE_FEATURES + FEATURES_1B)
HOLM_M = 2
MIN_PEERS = 3
SAMPLE_CAP = 500_000
MIN_FINITE = 2  # Spec: observable AND variable, not a percentage coverage gate.
LIMIT_BYTES = 60_000_000_000
THRESHOLDS = dict(mean_increment=.005, ci_lower_strict=0., alpha=.05,
                  min_daily_stocks=20, forward_valid_days=227, forward_sessions=252,
                  quarterly_announcement_quarters=4, quarterly_company_quarters=1000)
BOOTSTRAP = dict(block_sessions=20, replicates=1000, seed=17, ci_percentiles=[2.5, 97.5],
                 p='(1+count(abs(d_star-mean_d)>=abs(mean_d)))/1001', holm_m=2)


def frozen_decision(path=None):
    p = DECISION if path is None else Path(path)
    return verify_file(p, DECISION_SHA)


def contract():
    return dict(spec_sha256=SPEC_SHA, decision_sha256=DECISION_SHA, quarterly='Q-A',
                feature_order={k: list(v) for k, v in FEATURE_ORDER.items()},
                contrasts=list(CONTRASTS), years=list(YEARS), seeds=list(SEEDS),
                params=PARAMS, sample_cap=SAMPLE_CAP, min_peers=MIN_PEERS,
                support='each new column >=2 finite and min<max, each fit/eval partition',
                label='close(t+20)/open(t+1)-1', purge='t+21<origin',
                bootstrap=deepcopy(BOOTSTRAP), thresholds=deepcopy(THRESHOLDS),
                reuse='Step3 B matrices/samples/targets only; NO model or score reuse')


def new_output(path):
    p = Path(path).absolute()
    root = ROOT / '.artifacts/step4'
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
    if extra.shape != (len(base), 5) or extra.dtype != np.float32:
        raise ValueError('STOP extra matrix schema')
    cols = slice(0, 3) if job['arm'] == 'B1a' else slice(3, 5)
    return np.column_stack([base, extra[:, cols]])


def normalize_peer(frame):
    """Convert v3 audit counts to model UNKNOWN→NaN, without filtering rows."""
    required = {'date', 'stock_id', 'industry', 'eligible', 'target_revenue_available', *EXTRA_FEATURES}
    if not required <= set(frame):
        raise ValueError('STOP peer schema')
    if frame.duplicated(['date', 'stock_id']).any():
        raise ValueError('STOP duplicate peer key')
    if frame[['date', 'stock_id']].isna().any().any():
        raise ValueError('STOP missing peer key')
    for col in ('eligible', 'target_revenue_available'):
        if frame[col].isna().any() or frame[col].dtype.kind != 'b':
            raise ValueError('STOP unknown boolean evidence')
    x = frame[list(EXTRA_FEATURES)].to_numpy(np.float32, copy=True)
    if np.isinf(x).any():
        raise ValueError('STOP infinite peer value')
    known = frame.eligible.to_numpy() & frame.industry.notna().to_numpy()
    for j in (1, 4):
        counts = x[:, j]
        if np.any(known & (~np.isfinite(counts) | (counts < 0) | (counts != np.floor(counts)))):
            raise ValueError('STOP invalid known peer count')
        x[~known, j] = np.nan
    if np.any(np.isfinite(x[:, 0]) & (~known | (x[:, 1] < MIN_PEERS))):
        raise ValueError('STOP minimum peers/unknown classification')
    available = frame.target_revenue_available.to_numpy()
    if np.any(np.isfinite(x[:, 3]) & (~known | available | (x[:, 4] < MIN_PEERS))):
        raise ValueError('STOP minimum peers/target announcement')
    return x


def job_id(year, seed, arm):
    return f'fold{year}-{arm}-seed{seed}'
