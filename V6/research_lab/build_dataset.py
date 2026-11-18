"""CLI: PYTHONPATH=V6 python -m research_lab.build_dataset --prices ... --out ..."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow
from . import CONTRACT_VERSION
from .data import load_prices, coverage_report, dataset_manifest, file_sha256
from .features import compute_features, FEATURE_COLUMNS
from .labels import compute_labels, HORIZONS


def build_dataset(source: str | Path, out: str | Path) -> dict:
    source, out = Path(source).resolve(), Path(out).resolve()
    manifest_path = out.with_suffix('.manifest.json')
    if out.exists() or manifest_path.exists():
        raise FileExistsError('output or manifest exists; use a separate output namespace')
    prices = load_prices(source)
    features, labels = compute_features(prices), compute_labels(prices)
    dataset = features.merge(labels,on=['code','date'],validate='one_to_one').sort_values(['code','date']).reset_index(drop=True)
    dataset.attrs = {}
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    code_hashes = {p.name:file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    manifest = dict(schema=CONTRACT_VERSION, artifact_class='GENERATED', evidence_class='RESEARCH_ONLY',
                    status=['UNMERGED','NOT_DEPLOYED'],source_commit=commit,
                    implementation_sha256=hashlib.sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest(),
                    implementation_files=code_hashes,
                    environment=dict(python=platform.python_version(),executable=sys.executable,
                                     pandas=pd.__version__,numpy=np.__version__,pyarrow=pyarrow.__version__,
                                     MARKETMAMBA_DATA_ROOT=os.getenv('MARKETMAMBA_DATA_ROOT')),
                    input=dataset_manifest(prices,source),input_file_sha256=file_sha256(source),
                    source_sidecars={p.name:dict(path=str(p),sha256=file_sha256(p))
                                     for p in [source.with_name('meta.json'),source.with_name('universe.csv')] if p.exists()},
                    dataset=dataset_manifest(dataset,out),quality=prices.attrs['quality'],
                    coverage=coverage_report(prices).to_dict('records'),feature_columns=FEATURE_COLUMNS,
                    complete_feature_rows=int(dataset[FEATURE_COLUMNS].notna().all(axis=1).sum()),
                    label_counts={str(h):int(dataset[f'fwd_return_{h}d'].notna().sum()) for h in HORIZONS},
                    contract=dict(signal='after close t',entry='open t+1',exit='close t+h',horizons=list(HORIZONS),
                                  calendar='observed date union proxy; not exchange verified',
                                  price_basis='raw OHLC; adj_close retained but unused',
                                  missing='no fills; full observed-union window required',
                                  outlier='abs raw close return > 0.35 flags UNKNOWN event',
                                  rank_ties='average percentile; singleton unknown',
                                  splits='not materialized; expanding helper requires purge >= horizon plus pre-test embargo'),
                    model_identity=None,limitations=['survivor/current-turnover-selected universe; includes ETFs',
                    'Yahoo adjusted prices and historical revisions lack as-of vintages',
                    'no fundamentals/publication timestamps, official calendar or tradability evidence',
                    'raw returns can contain unrecognized corporate actions below flag threshold',
                    'no costs/slippage/limit rules; not verified trading evidence'])
    out.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation: never overwrite another run's artifacts. Manifest last.
    with out.open('xb') as stream:
        dataset.to_parquet(stream,index=False)
    manifest['output_file_sha256'] = file_sha256(out)
    with manifest_path.open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=Path('.local-data/public_tw/prices_daily.parquet'))
    parser.add_argument('--out', type=Path, default=Path('.local-data/research/dataset.parquet'))
    args = parser.parse_args()
    manifest = build_dataset(args.prices,args.out)
    print(json.dumps({k:manifest[k] for k in ['dataset','quality','complete_feature_rows','label_counts']},ensure_ascii=False))

if __name__ == '__main__':
    main()
