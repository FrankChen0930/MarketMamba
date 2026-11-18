"""Read-only daily availability diagnostics (UNMERGED / NOT_DEPLOYED).

Recon: V6/scripts/data_quality_check.py builds the full feature matrix;
check_data.py inventories files; data_inventory.py reports historical coverage;
healthcheck_backfill_202607.py binds July staging and prints backfill checks.
None supplies daily JSON/status/exit contracts, own-source rolling medians,
bounded known exceptions or PER/PBR/DY carry detection. A separate additive
reader avoids changing these historical tools or importing config (loads .env).

--lookback-days means N observed price trading sessions, ending at prices max
Date, NOT wall-clock freshness or an official exchange calendar. Weekends and
price days below 50% of the period weekday median are excluded. Coverage uses
up to 60 most recent normal observed sessions per source, ending at that anchor.
All checks are scoped to the resulting calendar interval (including extras).
Missing days/invalid inputs/duplicate keys/unexplained nontrading rows FAIL;
low row counts, >90% unchanged triplets, and any macro NaN WARN. No paired
finite triplets WARN with null ratio. Macro keys are Date only. This report
cannot certify PIT, tradability, publication timing or source correctness.

Only --output is written; no repair, config imports, network or .env reads.
Default: MARKETMAMBA_DATA_DIR, else MARKETMAMBA_DATA_ROOT/processed_v6,
else repository Data/processed_v6, always resolved. Override known exceptions
using --known-anomalies JSON_FILE (a list replacing, not extending defaults).
Example entry: {"dataset":"prices_raw", "check":"nontrading_rows",
"date":"2026-09-20", "rows":2}. Optional rows bounds the exception exactly.
The output includes source/input SHA-256 identities; exceptions outside the
window remain configuration only and are never reported as observed KNOWN.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATASETS = ('prices_raw', 'institutional_raw', 'daytrade_raw',
            'market_value_raw', 'per_raw', 'securities_raw', 'macro_raw')
KNOWN_ANOMALIES = [
    {'dataset': 'prices_raw', 'check': 'nontrading_rows', 'date': '2026-09-20', 'rows': 2},
    {'dataset': 'market_value_raw', 'check': 'missing_days', 'date': '2026-09-04'},
    {'dataset': 'market_value_raw', 'check': 'nontrading_rows', 'date': '2026-09-20', 'rows': 2},
    *[{'dataset': 'per_raw', 'check': 'nontrading_rows', 'date': d}
      for d in ('2026-09-25', '2026-09-28', '2026-10-09', '2026-10-23', '2026-10-26')],
]
RANK = {'OK': 0, 'KNOWN': 1, 'WARN': 2, 'FAIL': 3}


def _status(rows):
    return max((r['status'] for r in rows), key=RANK.get, default='OK')


def _validate_known(known):
    if not isinstance(known, list):
        raise ValueError('known anomalies must be a list')
    for row in known:
        if (not isinstance(row, dict) or row.get('dataset') not in DATASETS
                or row.get('check') not in ('missing_days', 'nontrading_rows')):
            raise ValueError('invalid known anomaly dataset/check')
        if pd.Timestamp(row.get('date')).strftime('%Y-%m-%d') != row.get('date'):
            raise ValueError('known anomaly date must be YYYY-MM-DD')
        if 'rows' in row and (type(row['rows']) is not int or row['rows'] < 0):
            raise ValueError('known anomaly rows must be a nonnegative integer')


def _known(known, dataset, kind, date, rows):
    return any(r['dataset'] == dataset and r['check'] == kind and r['date'] == date
               and ('rows' not in r or r['rows'] == rows) for r in known)


def _price_days(prices, count):
    counts = prices.groupby('Date').size().sort_index()
    weekdays = counts[counts.index.day_of_week < 5]
    sample = weekdays.tail(count)
    median = float(sample.median()) if len(sample) else 0.
    valid = weekdays[weekdays >= median * .5].index if median else pd.DatetimeIndex([])
    return valid, median


def analyze(frames, lookback_days=45, known_anomalies=None):
    """Pure diagnostics over caller-owned frames; never mutates inputs."""
    if type(lookback_days) is not int or lookback_days < 1:
        raise ValueError('lookback_days must be positive')
    known = KNOWN_ANOMALIES if known_anomalies is None else known_anomalies
    _validate_known(known)
    checks, data = [], {}

    def add(dataset, kind, status, **values):
        checks.append(dict(dataset=dataset, check=kind, status=status, **values))

    for name in DATASETS:
        try:
            f = frames[name].copy()
            required = {'Date'} | (set() if name == 'macro_raw' else {'stock_id'})
            if name == 'per_raw':
                required |= {'PER', 'PBR', 'DY'}
            if not required.issubset(f.columns):
                raise ValueError('missing columns: ' + ', '.join(sorted(required - set(f.columns))))
            f['Date'] = pd.to_datetime(f.Date, errors='raise').dt.normalize()
            if f.Date.isna().any() or ('stock_id' in required and f.stock_id.isna().any()):
                raise ValueError('null keys')
            if f.empty:
                raise ValueError('empty data')
            data[name] = f
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            add(name, 'input', 'FAIL', error=str(exc), threshold='required nonempty schema and valid keys')
    if 'prices_raw' not in data:
        return dict(status='FAIL', exit_code=1, checks=checks, trading_days=[])
    valid, price_median = _price_days(data['prices_raw'], lookback_days)
    trading = valid[-lookback_days:]
    if trading.empty:
        add('prices_raw', 'calendar', 'FAIL', count=0, threshold='at least one normal weekday')
        return dict(status='FAIL', exit_code=1, checks=checks, trading_days=[])
    start, end = trading.min(), data['prices_raw'].Date.max()
    calendar = set(valid)
    for name, full in data.items():
        f = full[full.Date.between(start, end)]
        counts = f.groupby('Date').size()
        missing = [str(d.date()) for d in trading if counts.get(d, 0) == 0]
        known_missing = [d for d in missing if _known(known, name, 'missing_days', d, 0)]
        add(name, 'missing_days', 'FAIL' if len(missing) > len(known_missing) else 'KNOWN' if missing else 'OK',
            count=len(missing), dates=missing, known_dates=known_missing, threshold=0)
        history = full[full.Date.isin(calendar) & (full.Date <= end)].groupby('Date').size().sort_index().tail(60)
        raw_median = float(history.median()) if len(history) else None
        normal = history[history >= raw_median * .5] if raw_median else history
        median = float(normal.median()) if len(normal) else None
        for day in trading:
            rows = int(counts.get(day, 0))
            ratio = rows / median if median else None
            add(name, 'row_ratio', 'WARN' if ratio is None or ratio < .5 else 'OK',
                date=str(day.date()), rows=rows, baseline_median=median, baseline_days=len(normal),
                ratio=ratio, threshold=.5)
        keys = ['Date'] if name == 'macro_raw' else ['Date', 'stock_id']
        dup = f.duplicated(keys, keep=False)
        excess = int(f.duplicated(keys).sum())
        add(name, 'duplicates', 'FAIL' if excess else 'OK', keys=len(f.loc[dup, keys].drop_duplicates()),
            excess_rows=excess, member_rows=int(dup.sum()), threshold=0, key_columns=keys)
        extras = counts[~counts.index.isin(calendar)]
        if extras.empty:
            add(name, 'nontrading_rows', 'OK', rows=0, threshold=0)
        for day, rows in extras.items():
            date = str(day.date())
            status = 'KNOWN' if _known(known, name, 'nontrading_rows', date, int(rows)) else 'FAIL'
            add(name, 'nontrading_rows', status, date=date, rows=int(rows), threshold=0,
                weekend=day.day_of_week >= 5)
    if 'per_raw' in data:
        f = data['per_raw']
        cols = ['PER', 'PBR', 'DY']
        # Never choose an arbitrary duplicate as evidence of identical values.
        f = f[~f.duplicated(['Date', 'stock_id'], keep=False)]
        for previous, day in zip(valid[:-1], valid[1:]):
            if day not in set(trading):
                continue
            left = f[f.Date == previous].set_index('stock_id')[cols]
            right = f[f.Date == day].set_index('stock_id')[cols]
            pair = left.merge(right, how='inner', left_index=True, right_index=True,
                              suffixes=('_prev', '_now'))
            numeric = pair.apply(pd.to_numeric, errors='coerce')
            pair = numeric[np.isfinite(numeric).all(axis=1)]
            n = len(pair)
            same = int((pair[[c + '_prev' for c in cols]].to_numpy() ==
                        pair[[c + '_now' for c in cols]].to_numpy()).all(axis=1).sum())
            ratio = same / n if n else None
            add('per_raw', 'carry_forward', 'WARN' if ratio is None or ratio > .9 else 'OK',
                date=str(day.date()), previous_date=str(previous.date()), matched_stocks=n,
                identical_stocks=same, ratio=ratio, threshold=.9)
    if 'macro_raw' in data:
        macro = data['macro_raw']
        columns = [c for c in macro if c != 'Date']
        if not columns:
            add('macro_raw', 'input', 'FAIL', error='no macro value columns', threshold='at least one value column')
        for day in trading[-5:]:
            rows = macro[macro.Date == day]
            for col in columns:
                count = int(rows[col].isna().sum())
                ratio = count / len(rows) if len(rows) else None
                add('macro_raw', 'nan_ratio', 'FAIL' if ratio is None else 'WARN' if ratio > 0 else 'OK',
                    date=str(day.date()) + ':' + col, trading_date=str(day.date()), column=col,
                    rows=len(rows), nan_rows=count, ratio=ratio, threshold=0.)
    status = _status(checks)
    return dict(status=status, exit_code=int(status == 'FAIL'), checks=checks,
                trading_days=[str(d.date()) for d in trading], start=str(start.date()), end=str(end.date()),
                price_calendar_median=price_median, price_calendar_threshold=.5,
                known_anomalies=known, lookback_sessions=lookback_days)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load(data_dir, lookback):
    """Project only required columns, keeping a bounded session interval in RAM."""
    frames, identities, errors = {}, [], []
    cutoff = None
    try:
        prices = pd.read_parquet(data_dir / 'prices_raw.parquet', columns=['Date'])
        prices['Date'] = pd.to_datetime(prices.Date).dt.normalize()
        days, _ = _price_days(prices, lookback)
        if len(days):
            cutoff = days[-min(len(days), max(lookback, 60) + 1)]
    except (OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    for name in DATASETS:
        path = (data_dir / (name + '.parquet')).resolve()
        try:
            before = path.stat()
            parquet = pq.ParquetFile(path)
            columns = (parquet.schema_arrow.names if name == 'macro_raw' else
                       ['Date', 'stock_id', 'PER', 'PBR', 'DY'] if name == 'per_raw' else ['Date', 'stock_id'])
            pieces = []
            for batch in parquet.iter_batches(batch_size=131072, columns=columns):
                f = batch.to_pandas()
                f['Date'] = pd.to_datetime(f.Date).dt.normalize()
                if cutoff is not None:
                    f = f[f.Date.isna() | (f.Date >= cutoff)]
                pieces.append(f)
            frames[name] = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
            digest = _sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError('input changed during observation')
            identities.append(dict(dataset=name, path=str(path), bytes=before.st_size,
                                   mtime_ns=before.st_mtime_ns, sha256=digest))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            frames.pop(name, None)
            errors.append(name + ': ' + str(exc))
    return frames, identities, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = os.environ.get('MARKETMAMBA_DATA_DIR') or str(
        Path(os.environ.get('MARKETMAMBA_DATA_ROOT', Path(__file__).resolve().parents[2] / 'Data')) / 'processed_v6')
    parser.add_argument('--data-dir', default=default)
    parser.add_argument('--lookback-days', type=int, default=45)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--known-anomalies', type=Path)
    args = parser.parse_args(argv)
    root, output = Path(args.data_dir).resolve(), args.output.resolve()
    # Protect both the selected dataset and any containing Data boundary,
    # including resolved symlink aliases and symlinked individual source files.
    protected = [root, Path(__file__).resolve().parents[2] / 'Data']
    protected += [p for p in root.parents if p.name.lower() == 'data']
    protected += [(root / (n + '.parquet')).resolve().parent for n in DATASETS]
    if (any(output.is_relative_to(p.resolve()) for p in protected)
            or any(output == (root / (n + '.parquet')).resolve() for n in DATASETS)
            or args.known_anomalies and output == args.known_anomalies.resolve()
            or output.exists() and any(output.samefile(p) for p in root.glob('*.parquet'))):
        parser.error('--output must be outside all data locations and input files')
    if args.lookback_days < 1:
        parser.error('--lookback-days must be positive')
    try:
        known = json.loads(args.known_anomalies.read_text()) if args.known_anomalies else KNOWN_ANOMALIES
        _validate_known(known)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    frames, identities, errors = _load(root, args.lookback_days)
    report = analyze(frames, args.lookback_days, known)
    if errors:
        report['checks'].append(dict(dataset='inputs', check='read', status='FAIL', errors=errors,
                                     threshold='all inputs readable and stable'))
        report.update(status='FAIL', exit_code=1)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'UNKNOWN'
    report.update(schema_version=1, observed_at=datetime.now(timezone.utc).isoformat(),
                  evidence_class='READ_ONLY_AVAILABILITY_DIAGNOSTIC', deployment='UNMERGED / NOT_DEPLOYED',
                  data_dir=str(root), inputs=identities, source_commit=commit,
                  source_sha256=_sha(Path(__file__)), python=os.sys.version,
                  environment={'pandas': pd.__version__, 'numpy': np.__version__})
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], exit_code=report['exit_code'], output=str(output),
                          counts={s: sum(r['status'] == s for r in report['checks']) for s in RANK})))
    return report['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
