"""Bounded TWSE per_raw repair candidate. UNMERGED / NOT_DEPLOYED.

Dry-run defaults to nine fixed dates; prices_raw Date union is an observed-session
proxy, not an official calendar. No TPEx, production imports, .env, or processed
outputs. DY preserves the fetcher's dividend_yield percent units (no /100).
Apply is explicit and requires separate consent flags for existing-row changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from .repair_prices import UPDATE_WINDOW

DAYS = ('2026-10-09', '2026-10-12', '2026-10-13', '2026-10-14',
        '2026-10-23', '2026-10-26', '2026-10-27', '2026-09-25', '2026-09-28')
KEY = ['Date', 'stock_id']
VALUES = ['PER', 'PBR', 'DY']
COLS = KEY + VALUES
URL = 'https://www.twse.com.tw/exchangeReport/BWIBBU_d'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def validate(frame):
    if list(frame) != COLS or frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError('unexpected schema, null or duplicate keys')
    if not pd.api.types.is_datetime64_any_dtype(frame.Date):
        raise ValueError('Date must preserve formal timestamp dtype')
    if not frame.Date.eq(frame.Date.dt.normalize()).all():
        raise ValueError('non-daily Date')
    if not frame.stock_id.map(lambda s: isinstance(s, str) and bool(re.fullmatch(r'[0-9]{4}', s))).all():
        raise ValueError('unexpected security id')
    if not all(pd.api.types.is_float_dtype(frame[c]) for c in VALUES):
        raise ValueError('unexpected numeric dtype')
    if np.isinf(frame[VALUES].to_numpy(dtype=float)).any():
        raise ValueError('infinite values')


def digest(frame):
    f = frame.sort_values(KEY).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(f, index=False).values.tobytes()).hexdigest()


def parse_source(body, day, template):
    """Bind BWIBBU_d by unique field names; never use BWIBBU_ALL positions."""
    d = pd.Timestamp(day)
    if body.get('date') != d.strftime('%Y%m%d'):
        raise ValueError('response date mismatch/missing')
    title = re.match(r'^\s*(\d+)年(\d+)月(\d+)日(?:\s|$)', str(body.get('title', '')))
    if not title or tuple(map(int, title.groups())) != (d.year-1911, d.month, d.day):
        raise ValueError('response title date mismatch/missing')
    if body.get('stat') != 'OK': raise ValueError('source status not OK')
    fields = body.get('fields', [])
    names = ['證券代號', '本益比', '股價淨值比', '殖利率(%)']
    if any(fields.count(n) != 1 for n in names): raise ValueError('source schema drift')
    indices = [fields.index(n) for n in names]
    def num(value):
        try: x = float(str(value).replace(',', '').strip())
        except (ValueError, TypeError): return np.nan
        if not np.isfinite(x): raise ValueError('nonfinite source number')
        return x
    records = []
    for row in body.get('data', []):
        if len(row) != len(fields): raise ValueError('source row width mismatch')
        sid = str(row[indices[0]]).strip()
        if not re.fullmatch(r'[0-9]{4}', sid): continue
        records.append([d, sid] + [num(row[i]) for i in indices[1:]])
    out = pd.DataFrame(records, columns=COLS).astype(template.dtypes.to_dict())
    validate(out)
    return out


def equal(a, b):
    return (a[VALUES].eq(b[VALUES]) | (a[VALUES].isna() & b[VALUES].isna())).all(axis=1)


def compare(old, new):
    a, b = old.set_index('stock_id'), new.set_index('stock_id')
    keys = a.index.intersection(b.index).sort_values()
    same = equal(a.loc[keys], b.loc[keys])
    return keys, same, a, b


def plan(before, calendar, sources):
    validate(before)
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    if len(cal) == 0: raise ValueError('empty calendar')
    # Last 60 observed sessions, exclude all suspect dates; no history scan or fetch.
    recent = cal[cal <= pd.Timestamp(max(DAYS))][-60:]
    counts = before[before.Date.isin(recent) & ~before.Date.isin(pd.to_datetime(DAYS))].groupby('Date').size()
    typical = float(counts.median()) if len(counts) else None
    rep = dict(typical_rows=typical, baseline_counts={str(k.date()):int(v) for k,v in counts.items()},
               coverage_threshold=.9, calendar='prices_raw Date union; not official calendar',
               days={}, before_rows=len(before), before_digest=digest(before))
    overwrite, add, remove = [], [], []
    for day in DAYS:
        d = pd.Timestamp(day); old = before[before.Date == d]
        r = dict(existing=len(old)); rep['days'][day] = r
        if d < cal.min() or d > cal.max():
            r.update(status='REFUSED', reason='outside observed calendar range'); continue
        if d not in cal:
            neighbors = {}
            for name, candidates in [('previous', cal[cal < d]), ('next', cal[cal > d])]:
                nd = candidates[-1] if name == 'previous' else candidates[0]
                keys, same, _, _ = compare(old, before[before.Date == nd])
                neighbors[name] = dict(date=str(nd.date()), overlap=len(keys), equal_rows=int(same.sum()),
                                       equal_ratio=float(same.mean()) if len(keys) else None,
                                       overlap_ratio=len(keys)/len(old) if len(old) else None)
            r.update(status='NONTRADING', removable=len(old), neighbors=neighbors)
            remove.append(old); continue
        try:
            source = sources.get(day)
            if isinstance(source, Exception): raise ValueError(str(source))
            if source is None: raise ValueError('source unavailable')
            new = parse_source(source, day, before)
            r['source_rows'] = len(new)
            if typical is None or typical <= 0 or len(new) < .9 * typical:
                raise ValueError('exchange coverage below 90% or baseline unknown')
            keys, same, a, b = compare(old, new)
            changed = keys[~same.to_numpy()]
            ow = b.loc[changed].reset_index()[COLS]
            ad = new[~new.stock_id.isin(old.stock_id)]
            delta = b.loc[changed, VALUES] - a.loc[changed, VALUES]
            field_changes = {}
            for c in VALUES:
                valid = delta[c].dropna()
                field_changes[c] = dict(paired_numeric=len(valid), mean_delta=float(valid.mean()) if len(valid) else None,
                                        mean_absolute_delta=float(valid.abs().mean()) if len(valid) else None,
                                        null_transitions=int((a.loc[changed,c].isna() != b.loc[changed,c].isna()).sum()))
            ranked = pd.DataFrame({'stock_id':changed, 'magnitude':delta.abs().max(axis=1).to_numpy()})
            ranked = ranked.sort_values(['magnitude','stock_id'], ascending=[False,True], na_position='last').head(10)
            samples = [dict(stock_id=s, before=a.loc[s,VALUES].to_dict(), after=b.loc[s,VALUES].to_dict()) for s in ranked.stock_id]
            r.update(status='READY', to_overwrite=len(ow), to_add=len(ad), unchanged=int(same.sum()),
                     existing_not_in_source=len(old)-len(keys),
                     change_ratio=len(ow)/len(keys) if len(keys) else None,
                     field_changes=field_changes, field_changes_scope='changed overlapping rows; numeric pairs only',
                     largest_changes=samples)
            overwrite.append(ow); add.append(ad)
        except (ValueError, TypeError, KeyError) as exc:
            r.update(status='REFUSED', reason=str(exc))
    def combined(parts):
        return pd.concat(parts, ignore_index=True).astype(before.dtypes.to_dict()) if parts else before.iloc[:0].copy()
    return dict(report=rep, overwrite=combined(overwrite), add=combined(add), remove=combined(remove))


def safe_output(path):
    p = Path(path).resolve()
    if 'processed_v6' in p.parts or '.env' in p.parts: raise ValueError('protected output path')
    return p


def json_safe(value):
    if isinstance(value, dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, np.generic): return value.item()
    return value


def write_json(path, obj):
    with safe_output(path).open('x') as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def check_window(now=None):
    t = pd.Timestamp.now(tz='Asia/Taipei') if now is None else pd.Timestamp(now)
    if t.tzinfo is None: raise ValueError('timezone required')
    t = t.tz_convert('Asia/Taipei')
    if UPDATE_WINDOW[0] <= t.hour*60+t.minute <= UPDATE_WINDOW[1]:
        raise ValueError('nightly update window 22:20–22:50 Taipei')


def apply(target, candidate, backup_dir, *, allow_overwrite=False, allow_remove=False, now=None):
    """Apply independently opted-in changes; additions accompany overwrites.

    Addition-only plans retain their flag-free behavior. With remove alone, both
    overwrites and additions are skipped. Caller must serialize other writers.
    """
    check_window(now)
    target = Path(target).resolve()
    if target.name == '.env': raise ValueError('protected target')
    before_hash = sha256(target); before = pd.read_parquet(target); validate(before)
    if digest(before) != candidate['report']['before_digest']: raise ValueError('stale repair plan')
    ow, ad, rm = (candidate[k] for k in ('overwrite','add','remove'))
    if (len(ow) or len(rm)) and not (allow_overwrite or allow_remove):
        raise ValueError('--allow-overwrite or --allow-remove required')
    for f in (ow, ad, rm):
        validate(f)
        if not f.Date.isin(pd.to_datetime(DAYS)).all(): raise ValueError('out-of-scope mutation')
    old_keys = pd.MultiIndex.from_frame(before[KEY])
    ow_keys, ad_keys, rm_keys = [pd.MultiIndex.from_frame(f[KEY]) for f in (ow,ad,rm)]
    if not ow_keys.isin(old_keys).all() or not rm_keys.isin(old_keys).all() or ad_keys.isin(old_keys).any() or ow_keys.isin(rm_keys).any():
        raise ValueError('invalid change keys')
    removable_left_untouched = {
        str(day.date()): int(count) for day, count in rm.groupby('Date').size().items()
    } if not allow_remove else {}
    # Validate the complete plan above, then select only authorized operations.
    if not allow_overwrite:
        ow = ow.iloc[:0]
        if allow_remove: ad = ad.iloc[:0]
    if not allow_remove: rm = rm.iloc[:0]
    ow_keys, ad_keys, rm_keys = [pd.MultiIndex.from_frame(f[KEY]) for f in (ow,ad,rm)]
    touched = old_keys.isin(ow_keys.union(rm_keys))
    untouched = before.loc[~touched]
    out = pd.concat([untouched,ow,ad], ignore_index=True).astype(before.dtypes.to_dict())
    out = out.sort_values(KEY).reset_index(drop=True)
    backup = safe_output(backup_dir); backup.mkdir(parents=True, exist_ok=False)
    bk = backup/'before.parquet'; shutil.copy2(target,bk)
    if sha256(bk) != before_hash: raise ValueError('backup verification failed')
    fd, tmpname = tempfile.mkstemp(prefix='.repair_per-',suffix='.tmp',dir=target.parent); os.close(fd)
    tmp = Path(tmpname)
    try:
        out.to_parquet(tmp,index=False)
        with tmp.open('rb') as f: os.fsync(f.fileno())
        chk = pd.read_parquet(tmp); validate(chk)
        if len(chk) != len(before)+len(ad)-len(rm): raise ValueError('row count mismatch')
        if list(chk) != list(before) or not chk.dtypes.equals(before.dtypes) or not pq.read_schema(tmp).equals(pq.read_schema(target), check_metadata=False):
            raise ValueError('schema changed')
        chk_keys = pd.MultiIndex.from_frame(chk[KEY])
        if digest(chk[~chk_keys.isin(ow_keys.union(ad_keys))]) != digest(untouched): raise ValueError('untouched rows changed')
        if digest(chk) != digest(out): raise ValueError('temporary content mismatch')
        result = dict(candidate['report'], before_sha256=before_hash, after_sha256=sha256(tmp),
                      rows_after=len(chk), overwritten=len(ow), added=len(ad), removed=len(rm),
                      untouched_sha256=digest(untouched),
                      removable_left_untouched=removable_left_untouched)
        write_json(backup/'diff.json',result)
        check_window(now)
        if sha256(target) != before_hash: raise ValueError('target changed during repair')
        os.replace(tmp,target)
        return result
    finally:
        tmp.unlink(missing_ok=True)


def fetch_source(day, archive):
    response = requests.get(URL, params={'date':day.replace('-',''),'selectType':'ALL','response':'json'}, timeout=30)
    response.raise_for_status()
    path = archive/f'{day}.body'; path.write_bytes(response.content)
    write_json(archive/f'{day}.meta.json',dict(url=response.url, observed_at=pd.Timestamp.now(tz='UTC').isoformat(),
                                              sha256=sha256(path), http_status=response.status_code))
    return response.json()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--prices',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True, help='new run namespace; never overwritten')
    parser.add_argument('--source-dir',type=Path,help='optional cached twse_per_DATE.body, missing dates fetched')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--backup-dir',type=Path)
    parser.add_argument('--allow-overwrite',action='store_true')
    parser.add_argument('--allow-remove',action='store_true')
    args = parser.parse_args(argv)
    if args.apply and not args.backup_dir: parser.error('--backup-dir required with --apply')
    outdir = safe_output(args.output_dir); outdir.mkdir(parents=True,exist_ok=False)
    inputs = {name:dict(path=str(p.resolve()),sha256=sha256(p)) for name,p in [('per',args.target),('prices',args.prices)]}
    before = pd.read_parquet(args.target)
    validate(before)
    calendar = pd.to_datetime(pd.read_parquet(args.prices,columns=['Date']).Date).dropna().dt.normalize().unique()
    sources, identities = {}, {}
    for day in DAYS:
        if pd.Timestamp(day) not in calendar: continue
        try:
            cached = args.source_dir/f'twse_per_{day}.body' if args.source_dir else None
            if cached and cached.exists():
                raw = cached.read_bytes(); sources[day] = json.loads(raw)
                identities[day] = dict(path=str(cached.resolve()),sha256=hashlib.sha256(raw).hexdigest(), mode='cached')
            else:
                sources[day] = fetch_source(day,outdir)
                raw_path = outdir/f'{day}.body'
                identities[day] = dict(path=str(raw_path.resolve()), sha256=sha256(raw_path), mode='fetched')
        except (requests.RequestException, ValueError, OSError) as exc: sources[day] = exc
    candidate = plan(before,calendar,sources)
    for name,p in [('per',args.target),('prices',args.prices)]:
        if sha256(p) != inputs[name]['sha256']: raise ValueError('input changed during dry run')
    candidate['report'].update(inputs=inputs,sources=identities,classification='GENERATED',
        evidence_class='RESEARCH_ONLY_RETROSPECTIVE_DIAGNOSTIC',status='UNMERGED / NOT_DEPLOYED',
        observed_at=pd.Timestamp.now(tz='UTC').isoformat(), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha256(__file__),runtime=dict(executable=sys.executable,python=sys.version,pandas=pd.__version__,platform=platform.platform(),MARKETMAMBA_DATA_ROOT=os.getenv('MARKETMAMBA_DATA_ROOT')),
        mode='apply' if args.apply else 'dry_run')
    for name in ('overwrite','add','remove'):
        candidate[name].to_parquet(outdir/f'{name}.parquet',index=False)
    write_json(outdir/'report.json',candidate['report'])
    if args.apply:
        result = apply(args.target,candidate,args.backup_dir,allow_overwrite=args.allow_overwrite,allow_remove=args.allow_remove)
        write_json(outdir/'apply.json',result)
    print(json.dumps(json_safe(candidate['report']['days']),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
