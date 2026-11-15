"""Month-end snapshot pilot: PIT universe vs labels universe vs delisting events. RESEARCH_ONLY.
python tools/research_env/pit_snapshots_run.py --years 2012 2019 --raw .local-data/pit/raw --labels DIR --out .local-data/pit/snapshot_pilot.json
"""
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'V6'))
import pandas as pd
from research_lab.pit_snapshots import (month_end_snapshot, stock_only, explain_disappearances, events_missing_from_snapshots)
from research_lab.pit_universe import build_events, load_finmind, load_tpex, load_twse

ap = argparse.ArgumentParser()
ap.add_argument('--years', nargs='+', type=int, required=True); ap.add_argument('--raw', type=Path, required=True)
ap.add_argument('--labels', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    raise FileExistsError(a.out)
cache = a.raw.parent/'snapshots'
tw, (tp, _), fm = load_twse(a.raw/'twse_delisted.json'), load_tpex(sorted(a.raw.glob('tpex_delisted_*.json'))), load_finmind(a.raw/'finmind_delisting.json')
ev = build_events(tw, tp, fm)
lab = pd.concat([pd.read_parquet(f, columns=['stock_id', 'signal_date']) for y in a.years for f in glob.glob(str(a.labels/f'*-{y}.parquet'))])
lab['signal_date'] = pd.to_datetime(lab.signal_date)
out = {}
for y in a.years:
    parts = [month_end_snapshot(m, y, mo, cache) for mo in range(1, 13) for m in ('TWSE', 'TPEX')]
    sn = stock_only(pd.concat(parts, ignore_index=True))
    per = []
    for d, g in sn.groupby('date'):
        u = set(lab[lab.signal_date == d].stock_id)
        q = set(g.code)
        per.append(dict(date=str(d.date()), quoted=len(q), in_labels=len(q & u), missing_from_labels=len(q-u), labels_only=len(u-q),
                        missing_fraction=round(len(q-u)/len(q), 4) if q else None))
    ex = explain_disappearances(sn, ev)
    miss = events_missing_from_snapshots(sn, ev)
    out[str(y)] = dict(snapshots=sn.date.nunique(), per_snapshot=per, disappearances=len(ex), status=ex.status.value_counts().to_dict(),
                       unexplained=[dict(code=r.code, last=str(r.last_snapshot.date()), nxt=str(r.next_snapshot.date())) for r in ex[ex.status == 'UNEXPLAINED'].itertuples()][:60],
                       events_in_span_not_seen_disappearing=len(miss), event_reason=ex.reason_class.value_counts().to_dict())
a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
for y, r in out.items():
    mf = [p['missing_fraction'] for p in r['per_snapshot'] if p['missing_fraction'] is not None]
    print(y, 'snapshots', r['snapshots'], 'disappearances', r['disappearances'], r['status'], 'events not seen', r['events_in_span_not_seen_disappearing'],
          'missing_from_labels mean %.3f' % (sum(mf)/len(mf)) if mf else '')
