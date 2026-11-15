"""Offline CLI: PYTHONPATH=V6 python -m research_lab.backtest --help."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from io import BytesIO
from pathlib import Path
import platform
import subprocess
import time

import V6.experimental.v7_portfolio_metrics as reused_metrics
import V6.experimental.v7_evidence_tier as reused_tier
import numpy as np
import pandas as pd

from research_lab.portfolio import Config, run_backtest, validate_predictions, canonical_sha256


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def baseline_predictions(prices, kind, config):
    """Momentum causal; random seeded; oracle deliberately future-contaminated diagnostic."""
    p = prices.sort_values(['code', 'date']).copy()
    if kind == 'random':
        # Keys are independent of future listings, calendar extent and row order.
        out = p[['code', 'date']].copy()
        out['score'] = [int.from_bytes(hashlib.sha256(
            f'{config.seed}|{pd.Timestamp(d).isoformat()}|{code}'.encode()).digest()[:8], 'big') / 2**64
            for code, d in zip(out.code, out.date)]
        return out.assign(fold=0, model=f'random_keyed_s{config.seed}', horizon=config.rebalance_sessions)[
            ['code', 'date', 'fold', 'model', 'horizon', 'score']].reset_index(drop=True)
    days = pd.DatetimeIndex(sorted(p.date.unique()))
    close = p.pivot(index='date', columns='code', values='close').reindex(days)
    op = p.pivot(index='date', columns='code', values='open').reindex(days)
    if kind == 'momentum20':
        score = close/close.shift(20)-1
    elif kind == 'oracle':
        score = op.shift(-(config.rebalance_sessions+1))/op.shift(-1)-1
    else:
        raise ValueError('unknown baseline')
    out = score.rename_axis(index='date', columns='code').stack().rename('score').reset_index()
    out = out[np.isfinite(out.score)].copy()
    return out.assign(fold=0, model=kind, horizon=config.rebalance_sessions)[['code', 'date', 'fold', 'model', 'horizon', 'score']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', required=True, type=Path)
    parser.add_argument('--predictions', nargs='+', type=Path)
    parser.add_argument('--baseline', choices=['momentum20', 'random', 'oracle'])
    parser.add_argument('--prediction-out', type=Path)
    parser.add_argument('--baseline-model', help='explicit namespaced baseline identity')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--start')
    parser.add_argument('--end')
    for name, default in asdict(Config()).items():
        parser.add_argument('--'+name.replace('_', '-'), type=str if default is None else type(default), default=default)
    args = parser.parse_args()
    config = Config(**{k: getattr(args, k) for k in asdict(Config())})
    if bool(args.predictions) == bool(args.baseline):
        parser.error('choose predictions or baseline')
    if args.out.exists() or (args.prediction_out and args.prediction_out.exists()):
        parser.error('refusing to overwrite existing artifacts')
    if args.baseline and not args.prediction_out:
        parser.error('baseline requires --prediction-out')
    implementation_paths = [Path(__file__), Path(__file__).with_name('portfolio.py'),
                            Path(reused_metrics.__file__), Path(reused_tier.__file__)]
    implementation_hashes = {p.name: file_hash(p) for p in implementation_paths}
    wall, cpu = time.perf_counter(), time.process_time()
    def read_pinned(path):
        content = path.read_bytes()
        return pd.read_parquet(BytesIO(content)), dict(path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest())

    prices, price_identity = read_pinned(args.prices)
    prediction_identities = []
    if args.baseline:
        pred = baseline_predictions(prices, args.baseline, config)
        if args.baseline_model:
            pred['model'] = args.baseline_model
    else:
        frames = []
        for path in args.predictions:
            frame, identity = read_pinned(path)
            frames.append(frame)
            prediction_identities.append(identity)
        pred = validate_predictions(pd.concat(frames, ignore_index=True))
    if args.start:
        pred = pred[pred.date >= pd.Timestamp(args.start)]
    if args.end:
        pred = pred[pred.date <= pd.Timestamp(args.end)]
    pred = validate_predictions(pred)
    result = run_backtest(prices, pred, config)
    if args.prediction_out:
        args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
        with args.prediction_out.open('xb') as stream:
            pred.to_parquet(stream, index=False)
        prediction_identities.append(dict(path=str(args.prediction_out.resolve()), sha256=file_hash(args.prediction_out)))
    inputs = dict(prices=price_identity, predictions=prediction_identities)
    manifest_path = args.prices.with_suffix('.manifest.json')
    if manifest_path.exists():
        inputs['dataset_manifest_sha256'] = file_hash(manifest_path)
    if implementation_hashes != {p.name: file_hash(p) for p in implementation_paths}:
        raise RuntimeError('implementation changed during run; result not published')
    result['provenance'] = dict(inputs=inputs,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_status='UNMERGED / NOT_DEPLOYED / uncommitted',
        implementation_hashes=implementation_hashes,
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, device='CPU'),
        prediction_rows=len(pred), prediction_folds=sorted(int(x) for x in pred.fold.unique()),
        seed=config.seed, wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    # Timing provenance is excluded from deterministic result hash.
    result['result_sha256'] = result.pop('canonical_sha256')
    result['canonical_sha256'] = canonical_sha256(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(output=str(args.out), result_sha256=result['result_sha256'], statistics=result['statistics'],
                          bootstrap_excess_ci95=result['bootstrap_excess_ci95'], compute=result['provenance']['cpu_seconds'])))


if __name__ == '__main__':
    main()
