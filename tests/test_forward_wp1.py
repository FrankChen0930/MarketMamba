"""Synthetic contract tests: no licensed rows or future performance."""
import json
from datetime import date, timedelta
import pytest
from tools import forward_wp1 as w


def calendar():
    # Explicit synthetic sessions include leap day, year boundary and typhoon closure.
    days = ['2023-12-29','2024-01-02','2024-02-27','2024-02-29','2024-03-01',
            '2024-07-23','2024-07-26','2024-12-31']
    days += [(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(300)]
    return w.Calendar(days, '2023-12-29', '2025-10-27', {'fixture':True})


def test_calendar_edges_and_purge():
    c=calendar()
    assert c.after('2023-12-31')=='2024-01-02'
    assert c.shift('2024-02-27',1)=='2024-02-29'
    assert c.after('2024-07-24')=='2024-07-26'
    assert c.start_after('2024-02-27','2024-02-29')=='2024-03-01'
    assert c.origin_minus21('2025-02-01')=='2025-01-11'
    assert c.h20_maturity('2025-01-01')=='2025-01-21'
    assert c.p2_endpoint('2025-01-01')=='2025-09-10'
    assert c.step3_endpoint('2025-01-01')=='2025-09-09'
    for call in [lambda:c.after('2026-01-01'),lambda:c.shift('2023-12-29',-1),
                 lambda:c.shift('2024-07-24',1),lambda:c.h20_maturity('2025-10-27')]:
        with pytest.raises(w.Stop):call()


def test_date_source_and_cutoff_missing_evidence():
    c=calendar()
    assert w.available_at(c,'2024-02-27','2024-02-29T12:00:00+08:00')=='2024-02-29T13:30:00+08:00'
    assert w.source_usable(c,'2024-02-29','2024-02-27','2024-02-29T18:00:00+08:00')
    assert not w.source_usable(c,'2024-02-29','2024-02-27','2024-02-29T18:00:01+08:00')
    for pub,obs in [(None,'2024-02-29T12:00:00+08:00'),('2024-02-27',None),('2024-02-27','2024-02-29T12:00:00')]:
        with pytest.raises(w.Stop):w.available_at(c,pub,obs)


def test_lifecycle_unknown_never_passes_strict():
    row={'stock_id':'SYNTH','首次掛牌日期':'20200102','首次掛牌市場':'OTC',
         '上市別':'TSE','下市日期':'20240229','TSE產業_代碼':'M1',
         '首次掛牌TSE產業':'M1 example','前一次TSE產業變更':'M2 example',
         '前一次TSE產業變更日':'20220101','全額交割起日(一)':'20240101'}
    book=w.Lifecycle([row])
    assert book.asof('SYNTH','2019-12-31')['in_market'] is False
    assert book.asof('SYNTH','2024-02-29')['in_market'] is None # terminal convention unknown
    assert book.asof('SYNTH','2024-03-01')['in_market'] is False
    r=book.asof('SYNTH','2024-02-27')
    assert r['in_market'] is None and r['industry'] is None
    assert r['market'] is None and r['full_delivery'] is True
    assert r['strict_pit_eligible'] is False
    with pytest.raises(w.Stop):book.asof('MISSING','2024-01-01')
    with pytest.raises(w.Stop):book.asof('SYNTH','2024-01-01',strict=True)


def test_ledger_run_selection_timestamps_and_ties():
    runs=[(1,'a',{'observed_at':'2024-02-29T17:00:00+08:00'}),
          (2,'b',{'observed_at':'2024-02-29T18:00:00+08:00'}),
          (3,'c',{'observed_at':'2024-02-29T18:00:01+08:00'})]
    assert w.select_run(runs,w.cutoff('2024-02-29'))=='b'
    assert w.select_run(runs,'2024-02-29T10:00:00+00:00')=='b'
    with pytest.raises(w.Stop):w.select_run(runs,'2024-02-28T18:00:00+08:00')
    with pytest.raises(w.Stop):w.select_run(runs,'2024-02-29T18:00:00')


def test_w34_identity_mutation(tmp_path):
    payload={'schema':'w34-development-twse-calendar-v1','sessions':[], 'research_admitted':False}
    p=tmp_path/'calendar.json';p.write_text(json.dumps(payload))
    with pytest.raises(w.Stop):w.load_w34(p,'0'*64)


def test_ledger_revisions_disappearance_and_integrity(tmp_path, monkeypatch):
    import pandas as pd
    from tools import pit_ledger as p
    from datetime import datetime
    root=tmp_path/'source';root.mkdir();ledger=tmp_path/'ledger'
    monkeypatch.setattr(p,'ALLOWED_LEDGER_ROOTS',(ledger,))
    def write(value, code='SYNTH'):
        for name in p.KEYS:
            row=({'stock_id':code,'revenue_year':2024,'revenue_month':1,'revenue':value}
                 if name=='revenue_raw' else
                 {'stock_id':code,'Date':pd.Timestamp('2024-01-31'),'type':'Revenue','value':value})
            pd.DataFrame([row]).to_parquet(root/(name+'.parquet'),index=False)
    write(10);a=p.snapshot(root,ledger,datetime.fromisoformat('2024-02-27T17:00:00+08:00'))
    write(20);p.snapshot(root,ledger,datetime.fromisoformat('2024-02-29T18:00:01+08:00'))
    key={'stock_id':'SYNTH','revenue_year':2024,'revenue_month':1}
    r=w.ledger_latest(ledger,w.cutoff('2024-02-29'),'revenue_raw',key)
    assert r['run_id']==a['run_id'] and r['value']['revenue']==10
    assert w.ledger_latest(ledger,w.cutoff('2024-03-01'),'revenue_raw',key)['value']['revenue']==20
    write(30,code='OTHER');p.snapshot(root,ledger,datetime.fromisoformat('2024-03-02T17:00:00+08:00'))
    assert w.ledger_latest(ledger,w.cutoff('2024-03-02'),'revenue_raw',key)['value'] is None
    with pytest.raises(w.Stop):w.ledger_latest(ledger,w.cutoff('2024-01-01'),'revenue_raw',key)
    (ledger/'ledger.sqlite3').write_bytes(b'corrupt')
    with pytest.raises(w.Stop):w.ledger_latest(ledger,w.cutoff('2024-03-02'),'revenue_raw',key)


def test_t1_and_buffer_and_synthetic_mutations():
    c=calendar()
    assert c.t1('2025-01-10','2025-01-08','2025-01-09')=='2025-01-10'
    assert c.t1('2025-01-10','2025-01-10','2025-01-11')=='2025-01-12'
    assert c.after('2025-01-01',10)=='2025-01-11'
    assert c.after('2025-01-01',8)=='2025-01-09'
    assert c.shift('2025-01-01',2)=='2025-01-03'
    with pytest.raises(w.Stop):c.start_after('2025-01-01',None)
    with pytest.raises(w.Stop):c.require_forward_admission()
    # Oracle comparisons detect common off-by-one mutants, no source mutation on disk.
    assert c.shift('2025-01-01',251)!=c.p2_endpoint('2025-01-01')
    assert c.shift('2025-01-01',252)!=c.step3_endpoint('2025-01-01')


def test_pipeline_buffer_18_boundary():
    c=calendar()
    assert w.buffered_ready_at(c,'2025-01-01','2025-01-02T18:00:00+08:00')=='2025-01-04T18:00:00+08:00'
    assert w.buffered_ready_at(c,'2025-01-01','2025-01-02T18:00:01+08:00')=='2025-01-05T18:00:00+08:00'
    with pytest.raises(w.Stop):w.buffered_ready_at(c,'2025-10-26','2025-10-27T18:00:01+08:00')


def test_normalized_lifecycle_dates_and_future_stop():
    row={'stock_id':'SYNTH','首次掛牌日期':'20200102','首次掛牌市場':'OTC',
         'OTC上市日':'20200102','上市別':'OTC','下市日期':'20240229',
         '全額交割起日(一)':'20240101','全額交割迄日(一)':'20240201'}
    book=w.Lifecycle([row])
    d=book.describe('SYNTH')
    assert d['listing_dates']['OTC上市日']=='2020-01-02'
    assert d['full_delivery_intervals'][0]['start']=='2024-01-01'
    assert book.asof('SYNTH','2024-01-15')['in_market'] is True
    with pytest.raises(w.Stop):book.asof('SYNTH','2027-01-01')
    row['全額交割迄日(一)']='20230101'
    with pytest.raises(w.Stop):book.describe('SYNTH')
