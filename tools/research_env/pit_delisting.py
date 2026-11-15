"""Fetch public delisting lists and report coverage vs a labels universe. RESEARCH_ONLY.

  python tools/research_env/pit_delisting.py fetch  --out .local-data/pit/raw
  python tools/research_env/pit_delisting.py report --raw .local-data/pit/raw --labels DIR_OF_PARQUET --out .local-data/pit/report.json
Fetch is anonymous public GET (TWSE openapi, TPEx year pages, one FinMind dataset call); no overwrite, no secrets.
"""
import argparse, glob, json, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'V6'))
import pandas as pd
from research_lab.pit_universe import (build_events, coverage_table, load_finmind, load_tpex, load_twse, matrix_status)

URLS = {'twse_delisted.json': 'https://openapi.twse.com.tw/v1/company/suspendListingCsvAndHtml',
        'finmind_delisting.json': 'https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDelisting'}
TPEX = 'https://www.tpex.org.tw/www/zh-tw/company/deListed?response=json&date={y}'


def get(url, dest):
    dest = Path(dest)
    if dest.exists():
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research-client'}), timeout=60) as r:
        dest.write_bytes(r.read())


def fetch(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        get(url, out/name)
    for y in range(1995, pd.Timestamp.today().year+1):
        get(TPEX.format(y=y), out/f'tpex_delisted_{y}.json'); time.sleep(.5)


def report(raw, labels, out):
    raw = Path(raw)
    tw, (tp, declared), fm = load_twse(raw/'twse_delisted.json'), load_tpex(sorted(raw.glob('tpex_delisted_*.json'))), load_finmind(raw/'finmind_delisting.json')
    ev = build_events(tw, tp, fm)
    l = pd.concat([pd.read_parquet(f, columns=['stock_id', 'signal_date']) for f in sorted(glob.glob(str(Path(labels)/'*.parquet')))])
    rng = l.groupby('stock_id').signal_date.agg(first='min', last='max')
    st = matrix_status(ev, rng)
    w = st[st.stock_like & (st.date >= '2005-01-01')]
    cov = coverage_table(st)
    result = dict(events_all=len(ev), events_stock_like_since_2005=len(w), status=w.matrix_status.value_counts().to_dict(),
                  reason_class=w.reason_class.value_counts().to_dict(), market=w.market.value_counts().to_dict(),
                  tpex_declared=sum(a for a, _ in declared.values()), tpex_retrieved=sum(b for _, b in declared.values()),
                  universe_codes=int(len(rng)), universe_last_before_2026=int((rng['last'] < '2026-01-01').sum()),
                  by_year=json.loads(cov.reset_index().to_json(orient='records')))
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result, st


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch'); f.add_argument('--out', required=True)
    r = sub.add_parser('report'); r.add_argument('--raw', required=True); r.add_argument('--labels', required=True); r.add_argument('--out', required=True)
    a = p.parse_args()
    if a.cmd == 'fetch':
        fetch(a.out)
    else:
        res, _ = report(a.raw, a.labels, a.out)
        print({k: res[k] for k in ('events_stock_like_since_2005', 'status', 'tpex_declared', 'tpex_retrieved', 'universe_codes', 'universe_last_before_2026')})
