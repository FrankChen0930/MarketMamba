"""Bounded seven-session derived-data repair. UNMERGED / NOT_DEPLOYED.

Dry-run by default; public sources only, no config/.env imports. Reuses the
existing repair_prices calendar convention, but never its price-specific writer.
Institutional legacy dealer buy/sell columns remain NaN, as in the direct
production fetcher. Market value reproduces Close * CURRENT MOPS shares, NOT PIT.
Local public_fill artifacts are GENERATED evidence, never production authority.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
import pyarrow as pa
import pyarrow.parquet as pq

from .repair_prices import trading_calendar, UPDATE_WINDOW

DAYS = ('2026-09-24', '2026-10-01', '2026-10-02', '2026-10-05',
        '2026-10-07', '2026-10-12', '2026-10-13')
KEY = ['Date', 'stock_id']
INST_VALUES = ['Foreign_Buy', 'Foreign_Sell', 'Foreign_Net', 'Investment_Trust_Buy',
               'Investment_Trust_Sell', 'Investment_Trust_Net', 'Dealer_Net']
COLUMNS = {
    'institutional': KEY + INST_VALUES + ['buy_Dealer', 'sell_Dealer'],
    'daytrade': KEY + ['Day_Trade_Volume'],
    'market_value': KEY + ['market_value'],
}
ENDPOINTS = {
    'institutional': [('TWSE', 'https://www.twse.com.tw/rwd/zh/fund/T86'),
                      ('TPEX', 'https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade')],
    'daytrade': [('TWSE', 'https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U'),
                 ('TPEX', 'https://www.tpex.org.tw/www/zh-tw/intraday/stat')],
    'market_value': [('L', 'https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv'),
                     ('O', 'https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv')],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def archive(root, content, label, metadata):
    """Immutable content-addressed payload; separate manifest per observation."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    path = root / f'{label}_{digest}.raw'
    if not path.exists():
        with path.open('xb') as f:
            f.write(content)
    if sha256(path) != digest:
        raise ValueError('archive hash mismatch')
    meta = dict(metadata, sha256=digest, size=len(content), evidence_class='GENERATED',
                observed_at=pd.Timestamp.now(tz='UTC').isoformat())
    # Run-local namespace makes this immutable even when the same payload recurs.
    manifest = path.with_suffix('.manifest.json')
    if not manifest.exists():
        with manifest.open('x') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def check_date(body, day):
    got = str(body.get('date', '')).replace('/', '').replace('-', '')
    if got != day.replace('-', ''):
        raise ValueError(f'response date {got!r} != requested date {day}')
    if str(body.get('stat', 'OK')).lower() not in ('ok', 'success'):
        raise ValueError(f'source status {body.get("stat")}')


def parse_institutional(body, day, market):
    check_date(body, day)
    table = body if market == 'TWSE' else body.get('tables', [{}])[0]
    width = 19 if market == 'TWSE' else 24
    indices = [2, 3, 4, 8, 9, 10, 11] if market == 'TWSE' else [2, 3, 4, 11, 12, 13, 22]
    fields = table.get('fields', [])
    expected = (['證券代號', '證券名稱', '外陸資買進股數(不含外資自營商)',
                 '外陸資賣出股數(不含外資自營商)', '外陸資買賣超股數(不含外資自營商)',
                 '外資自營商買進股數', '外資自營商賣出股數', '外資自營商買賣超股數',
                 '投信買進股數', '投信賣出股數', '投信買賣超股數', '自營商買賣超股數',
                 '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
                 '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)', '三大法人買賣超股數']
                if market == 'TWSE' else ['代號', '名稱'] +
                ['買進股數', '賣出股數', '買賣超股數'] * 7 + ['三大法人買賣超股數合計'])
    if fields != expected:
        raise ValueError('institutional schema drift')
    records = []
    for row in table.get('data', []):
        sid = str(row[0]).strip() if row else ''
        if not sid.isdigit() or len(sid) != 4:
            continue
        if len(row) < width:
            raise ValueError('institutional short row')
        def number(value):
            s = str(value).replace(',', '')
            if market == 'TWSE':
                s = s.replace('--', '0')
            return int(s or '0')
        values = [number(row[i]) for i in indices]
        records.append(dict(zip(INST_VALUES, values), Date=day, stock_id=sid,
                            buy_Dealer=np.nan, sell_Dealer=np.nan))
    if not records:
        raise ValueError('institutional no numeric rows')
    return pd.DataFrame(records)


def parse_daytrade(body, day):
    check_date(body, day)
    records = []
    for table in body.get('tables', []):
        fields = table.get('fields', [])
        if len(fields) < 4 or not any('代號' in str(f) for f in fields):
            continue
        if '股數' not in str(fields[3]):
            raise ValueError('daytrade shares field moved')
        for row in table.get('data', []):
            if len(row) < 4:
                continue
            sid = str(row[0]).strip()
            if not sid.isdigit() or len(sid) != 4:
                continue
            try:
                shares = float(str(row[3]).replace(',', '').strip())
            except (ValueError, TypeError):
                continue
            if np.isfinite(shares) and shares >= 0:
                records.append(dict(Date=day, stock_id=sid, DayTrade_Shares=shares))
    if not records:
        raise ValueError('daytrade no numeric rows / schema drift')
    return pd.DataFrame(records)


def normalize(frame):
    f = frame.copy()
    f['Date'] = pd.to_datetime(f.Date).dt.strftime('%Y-%m-%d')
    f['stock_id'] = f.stock_id.astype(str)
    if f.duplicated(KEY).any():
        raise ValueError('duplicate keys')
    return f


def derive(dataset, raw, prices):
    p = normalize(prices)
    if dataset == 'institutional':
        return normalize(raw)
    if dataset == 'daytrade':
        out = normalize(raw).merge(p[KEY+['Volume']], on=KEY, how='inner', validate='one_to_one')
        out = out[np.isfinite(out.Volume) & (out.Volume > 0)].copy()
        out['Day_Trade_Volume'] = out.DayTrade_Shares / out.Volume
        out = out[np.isfinite(out.Day_Trade_Volume) & out.Day_Trade_Volume.between(0, 1)]
    else:
        if raw.stock_id.duplicated().any():
            raise ValueError('duplicate MOPS security')
        out = p.merge(raw[['stock_id', 'shares_outstanding']], on='stock_id', how='inner', validate='many_to_one')
        close = pd.to_numeric(out.Close, errors='coerce')
        shares = pd.to_numeric(out.shares_outstanding, errors='coerce')
        out['market_value'] = close * shares
        out = out[np.isfinite(close) & (close > 0) & np.isfinite(shares) & (shares > 0)]
        out = out[np.isfinite(out.market_value) & (out.market_value > 0)]
    return out[COLUMNS[dataset]]


def typed(fill, before, dataset):
    if set(before.columns) != set(COLUMNS[dataset]) or set(fill.columns) != set(before.columns):
        raise ValueError('schema columns mismatch')
    f = normalize(fill)[list(before.columns)]
    for col, dtype in before.dtypes.items():
        if col == 'Date' and pd.api.types.is_datetime64_any_dtype(dtype):
            f[col] = pd.to_datetime(f[col])
        f[col] = f[col].astype(dtype)
    return f


def daytrade_baseline(counts):
    """Latest 60 normal target days; filter against the full-period median first."""
    normal = counts[counts >= .5 * counts.median()].sort_index().tail(60)
    return float(normal.median()) if not normal.empty else float('nan')


def target_daily_counts(path):
    """Scan only Date in bounded batches, retaining one count per target day."""
    counts = pd.Series(dtype='int64')
    for batch in pq.ParquetFile(path).iter_batches(columns=['Date']):
        dates = pd.to_datetime(batch.column(0).to_pandas()).dt.strftime('%Y-%m-%d')
        counts = counts.add(dates.value_counts(), fill_value=0)
    return counts


def plan(before, prices, fill, calendar, dataset, *, daily_counts=None):
    b, p = normalize(before), normalize(prices)
    f = normalize(typed(fill, before, dataset))
    cal = {pd.Timestamp(d).strftime('%Y-%m-%d') for d in calendar if pd.Timestamp(d).day_of_week < 5}
    report, additions = {}, []
    target_expected = (daytrade_baseline(b.groupby('Date').size() if daily_counts is None else daily_counts)
                       if dataset == 'daytrade' else None)
    # Only the explicitly requested incident sessions, never a historical sweep.
    for day in DAYS:
        bd, pd_, fd = (x[x.Date == day] for x in (b, p, f))
        expected = target_expected if dataset == 'daytrade' else len(pd_)
        fd = fd[fd.stock_id.isin(pd_.stock_id)]
        overlap = bd.merge(fd, on=KEY, suffixes=('_old', '_source'))
        conflict = pd.Series(False, index=overlap.index)
        for c in before.columns:
            if c in KEY:
                continue
            a, z = overlap[c+'_old'], overlap[c+'_source']
            conflict |= ~(a.eq(z) | (a.isna() & z.isna()))
        new = fd[~fd.stock_id.isin(bd.stock_id)]
        status = 'READY'
        if day not in cal:
            status = 'SKIP_NONTRADING'
        elif len(pd_) == 0 or not np.isfinite(expected) or expected <= 0 or len(fd) < .9 * expected:
            status = 'REFUSED_COVERAGE'
        # Reject a broken denominator day as well as incomplete exchange data.
        elif len(pd_) < .9 * p.groupby('Date').size().tail(60).median():
            status = 'REFUSED_PRICE_COVERAGE'
        report[day] = dict(existing=len(bd), expected=expected, source_rows=len(fd),
                           to_add=len(new) if status == 'READY' else 0,
                           conflicts=int(conflict.sum()), status=status)
        if status == 'READY':
            additions.append(new)
    out = pd.concat(additions, ignore_index=True) if additions else f.iloc[:0]
    return typed(out, before, dataset), report


def apply(target, prices, fill, calendar, dataset, backup_dir, now=None, clock=None):
    """Writer for separately authorized use; this task invokes it ONLY in tests."""
    override = now
    clock = clock or (lambda: pd.Timestamp.now(tz='Asia/Taipei') if override is None else pd.Timestamp(override))
    now = pd.Timestamp(clock())
    if now.tzinfo is None:
        raise ValueError('timezone required')
    now = now.tz_convert('Asia/Taipei')
    minute = now.hour*60 + now.minute
    if UPDATE_WINDOW[0] <= minute <= UPDATE_WINDOW[1]:
        raise RuntimeError('nightly window: 22:20–22:50 Taipei')
    target = Path(target)
    identity = sha256(target)
    before = pd.read_parquet(target)
    add, report = plan(before, prices, fill, calendar, dataset)
    if any(r['status'].startswith('REFUSED') for r in report.values()):
        raise RuntimeError('REFUSED plan; no writes')
    if add.empty:
        return {'added': 0}
    out = pd.concat([before, add], ignore_index=True)
    if normalize(out).duplicated(KEY).any():
        raise ValueError('collision')
    backup_dir = Path(backup_dir); backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f'{target.stem}.{identity}.before_repair.parquet'
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(target, backup)
    if sha256(backup) != identity:
        raise ValueError('backup identity mismatch')
    fd, name = tempfile.mkstemp(dir=target.parent, suffix='.tmp'); os.close(fd)
    tmp = Path(name)
    try:
        out.to_parquet(tmp, index=False)
        with tmp.open('rb') as handle:
            os.fsync(handle.fileno())
        checked = pd.read_parquet(tmp)
        pd.testing.assert_frame_equal(checked, out)
        pd.testing.assert_frame_equal(checked.iloc[:len(before)].reset_index(drop=True), before.reset_index(drop=True))
        if sha256(target) != identity:
            raise RuntimeError('target changed concurrently')
        # Recheck window after serialization, immediately before replacement.
        actual = pd.Timestamp(clock()).tz_convert('Asia/Taipei')
        if UPDATE_WINDOW[0] <= actual.hour*60+actual.minute <= UPDATE_WINDOW[1]:
            raise RuntimeError('nightly window before replace')
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return dict(added=len(add), backup=str(backup), before_sha256=identity, after_sha256=sha256(target))


def fetch(session, dataset, days, root):
    frames, sources = [], []
    for day in ([None] if dataset == 'market_value' else days):
        for market, url in ENDPOINTS[dataset]:
            if dataset == 'market_value':
                params = {}
            elif market == 'TWSE':
                params = dict(date=day.replace('-', ''), response='json',
                              selectType='ALLBUT0999' if dataset == 'institutional' else 'All')
            else:
                params = dict(date=day.replace('-', '/'), response='json', type='Daily')
                if dataset == 'institutional':
                    params['sect'] = 'EW'
            response = session.get(url, params=params, headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
            path = archive(root, response.content, f'{dataset}_{market}_{day or "current"}',
                           dict(url=response.url, http_status=response.status_code,
                                replay_from=getattr(response, 'replay_from', None)))
            sources.append(dict(url=response.url, raw=str(path), sha256=sha256(path),
                                replay_from=getattr(response, 'replay_from', None)))
            response.raise_for_status()
            if dataset == 'market_value':
                frame = pd.read_csv(io.BytesIO(path.read_bytes()), encoding='utf-8')
                column = next((c for c in frame if '已發行普通股數' in c), None)
                if column is None or '公司代號' not in frame:
                    raise ValueError('MOPS shares schema unavailable')
                frame = frame[['公司代號', column]].rename(columns={'公司代號':'stock_id', column:'shares_outstanding'})
                frame.stock_id = frame.stock_id.astype(str).str.strip()
                frame = frame[frame.stock_id.str.match(r'^\d{4}$')].copy()
                frame.shares_outstanding = pd.to_numeric(frame.shares_outstanding, errors='coerce')
                frame = frame.dropna(subset=['shares_outstanding'])
            else:
                body = json.loads(path.read_bytes())
                frame = (parse_institutional(body, day, market) if dataset == 'institutional'
                         else parse_daytrade(body, day))
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS[dataset])
    keys = ['stock_id'] if dataset == 'market_value' else KEY
    # Production direct merge gives TWSE precedence for exchange overlap.
    return out.drop_duplicates(keys, keep='first'), sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=COLUMNS)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--prices', required=True, type=Path)
    parser.add_argument('--backup-dir', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if args.apply and not args.backup_dir:
        parser.error('--backup-dir required with --apply')
    root = Path('.local-data/public_fill') / f'derived_{args.dataset}_{uuid4().hex}'
    root.mkdir(parents=True)
    inputs = {str(p.resolve()): sha256(p) for p in [args.target, args.prices]}
    report = dict(dataset=args.dataset, evidence_class='GENERATED_DRY_RUN', inputs=inputs,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                  source_sha256=sha256(__file__), environment=platform.python_version(),
                  data_root_env=os.environ.get('MARKETMAMBA_DATA_ROOT'), apply=bool(args.apply),
                  sources=[], limitations=['market_value uses CURRENT MOPS shares, NOT PIT'] if args.dataset == 'market_value'
                  else ['institutional buy_Dealer/sell_Dealer remain legacy NaN'] if args.dataset == 'institutional' else [])
    macro = args.prices.parent/'macro_raw.parquet'
    try:
        date_type = pq.read_schema(args.target).field('Date').type
        dates = [pd.Timestamp(d) for d in DAYS] if pa.types.is_timestamp(date_type) else list(DAYS)
        before = pd.read_parquet(args.target, filters=[('Date', 'in', dates)])
        daily_counts = target_daily_counts(args.target) if args.dataset == 'daytrade' else None
        price_type = pq.read_schema(args.prices).field('Date').type
        start, end = '2026-07-01', max(DAYS)
        if pa.types.is_timestamp(price_type):
            start, end = pd.Timestamp(start), pd.Timestamp(end)
        prices = pd.read_parquet(args.prices, columns=KEY+['Close', 'Volume'],
                                 filters=[('Date', '>=', start), ('Date', '<=', end)])
        calendar = trading_calendar(macro)
        report['calendar'] = dict(path=str(macro.resolve()), sha256=sha256(macro), basis='existing macro TWII_Close convention; not strict PIT calendar')
        cal = {d.strftime('%Y-%m-%d') for d in calendar if d.day_of_week < 5}
        days = [d for d in DAYS if d in cal]
        with requests.Session() as session:
            # No .env, authentication, cookies, netrc, or proxy credentials.
            session.trust_env = False
            raw, report['sources'] = fetch(session, args.dataset, days, root)
        selected = normalize(prices); selected = selected[selected.Date.isin(days)]
        fill = derive(args.dataset, raw, selected)
        buffer = io.BytesIO(); fill.to_parquet(buffer, index=False)
        archive(root, buffer.getvalue(), 'parsed', dict(dataset=args.dataset, format='parquet'))
        add, report['days'] = plan(before, prices, fill, calendar, args.dataset, daily_counts=daily_counts)
        report['schema'] = {c: str(t) for c,t in before.dtypes.items()}
        report['schema_match'] = add.dtypes.to_dict() == before.dtypes.to_dict()
        report['total_to_add'] = len(add)
        report['status'] = 'STOP' if any(r['status'].startswith('REFUSED') for r in report['days'].values()) else 'DRY_RUN'
        if args.apply:
            report['apply_result'] = apply(args.target, prices, fill, calendar, args.dataset, args.backup_dir)
            report.update(status='APPLIED', evidence_class='GENERATED_REPAIR')
    except (ValueError, OSError, RuntimeError, requests.RequestException) as exc:
        report.update(status='STOP', error=f'{type(exc).__name__}: {exc}')
    report['inputs_unchanged'] = all(sha256(p) == h for p,h in inputs.items())
    if not report['inputs_unchanged'] and not args.apply:
        report.update(status='STOP', error='input changed during dry-run')
    (root/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Report: {root}/report.json')
    return 2 if report['status'] == 'STOP' else 0


if __name__ == '__main__':
    raise SystemExit(main())
