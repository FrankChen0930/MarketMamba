"""Synthetic-only TEJ ingestion contracts; no licensed rows in fixtures."""
import hashlib
import zipfile

import pandas as pd
import pytest

from tools.tej_pit import load_export, verify_zip, MONTHLY_COLUMNS, IFRS_COLUMNS


def export(tmp_path, columns, rows):
    path = tmp_path / 'synthetic.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('synthetic.tsv', ('\t'.join(columns)+'\n'+'\n'.join('\t'.join(r) for r in rows)).encode('utf-16'))
    sums = tmp_path / 'SHA256SUMS'
    sums.write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  synthetic.zip\n')
    return path, sums


def test_monthly_units_leading_zero_missing_and_no_backfill(tmp_path):
    row = [' 0050 合成 ', ' 202601 ', '.', ' 1,234 ', '.', '2', '3', '.', 'Y', '4']
    p, s = export(tmp_path, [' '+c+' ' for c in MONTHLY_COLUMNS], [row])
    d = load_export(p, s, 'monthly')
    assert d.stock_id.iloc[0] == '0050'
    assert d.revenue.iloc[0] == 1234000
    assert pd.isna(d.available_date.iloc[0])
    assert d.release_status.iloc[0] == 'NO_RELEASE_DATE'
    assert pd.isna(d['去年單月營收(千元)'].iloc[0])
    assert d.input_sha256.iloc[0] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_monthly_release_uses_publication_not_deadline(tmp_path):
    p,s = export(tmp_path, MONTHLY_COLUMNS, [['0012 合成','202601','20260203','2','.','.','.','.','N','2']])
    d = load_export(p,s,'monthly')
    assert d.available_date.iloc[0] == pd.Timestamp('2026-02-03')


def test_ifrs_negative_lag_and_version_unknown(tmp_path):
    row = ['0001 合成','202603','2026/02/01','1','H','6','2026/01','2026/06','Y','3','TWD','M1']+['2']*11
    p,s = export(tmp_path, IFRS_COLUMNS, [row])
    d = load_export(p,s,'ifrs')
    assert d.revenue.iloc[0] == 2000
    assert d.eps.iloc[0] == 2
    assert d.period_end.iloc[0] == pd.Timestamp('2026-06-30')
    assert d.release_status.iloc[0] == 'BEFORE_PERIOD_END'
    assert not d.strict_pit_eligible.iloc[0]
    assert d.version_semantics.iloc[0] == 'UNKNOWN'


def test_sha_mutation_stops(tmp_path):
    p,s = export(tmp_path, MONTHLY_COLUMNS, [])
    p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='STOP.*SHA'):
        verify_zip(p,s)


@pytest.mark.parametrize('bad', ['20261301','garbage'])
def test_invalid_release_stops(tmp_path,bad):
    p,s = export(tmp_path, MONTHLY_COLUMNS, [['0012 合成','202601',bad,'2','.','.','.','.','N','2']])
    with pytest.raises(ValueError, match='STOP'):
        load_export(p,s,'monthly')


def test_schema_mutation_stops(tmp_path):
    p,s = export(tmp_path, ['wrong']+MONTHLY_COLUMNS[1:], [])
    with pytest.raises(ValueError, match='STOP.*schema'):
        load_export(p,s,'monthly')


def test_nominal_vs_actual_end_and_calendar_cumulative_periods(tmp_path):
    from tools.tej_pit import timing_audit
    mpath, ms = export(tmp_path, MONTHLY_COLUMNS, [['0012 合成','202601','20260203','2','.','.','.','.','N','2']])
    m = load_export(mpath,ms,'monthly')
    row = ['0001 合成','202609','2026/09/10','3','Q','9','2026/01','2026/09','Y','3','NTD','M1']+['2']*11
    p,s = export(tmp_path,IFRS_COLUMNS,[row])
    f=load_export(p,s,'ifrs')
    audit,_=timing_audit(m,f)
    assert audit['ifrs']['non_calendar_period_rows'] == 0
    assert audit['ifrs']['anomalies']['before_nominal_end']['rows'] == 1


def test_reconciliation_zero_missing_and_duplicate_reference():
    from tools.tej_pit import reconciliation
    t=pd.DataFrame({'stock_id':['01','02','03','04'],'period':pd.to_datetime(['2026-01-01']*4),'revenue':[0.,10.,None,5.]})
    r=pd.DataFrame({'stock_id':['01','02','03','04','04'],'period':pd.to_datetime(['2026-01-01']*5),'raw_value':[0.,0.,1.,5.,6.]})
    a,b = reconciliation(t,r,'revenue',[])
    assert a['raw_duplicate_key_rows_excluded']==2
    assert a['matched_keys']==3
    assert a['paired_finite_values']==2
    assert a['raw_zero_tej_nonzero']==1
    assert a['over_1pct']==1


@pytest.mark.parametrize('mutation', ['units','backfill'])
def test_in_memory_mutations_are_killed(tmp_path,mutation):
    from pathlib import Path
    import tools.tej_pit as module
    source=Path(module.__file__).read_text()
    if mutation=='units':
        source=source.replace("d['revenue'] = d['單月營收(千元)']*1000", "d['revenue'] = d['單月營收(千元)']*1")
    else:
        source=source.replace("d['release_status'] = np.where", "d['available_date'] = d.available_date.fillna(d.period + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9))\n    d['release_status'] = np.where",1)
    namespace={'__name__':'tej_mutant','__file__':module.__file__}
    exec(compile(source,'<tej-mutant>','exec'),namespace)
    p,s=export(tmp_path,MONTHLY_COLUMNS,[['0050 合成','202601','.','1,234','.','2','3','.','Y','4']])
    d=namespace['load_export'](p,s,'monthly')
    with pytest.raises(AssertionError):
        assert d.revenue.iloc[0]==1234000
        assert pd.isna(d.available_date.iloc[0])


def test_universe_reads_only_membership_and_dates(tmp_path):
    import numpy as np
    from tools.tej_pit import universe_coverage
    np.save(tmp_path/'stock_ids.npy',np.array(['0012','0012','0013']))
    np.save(tmp_path/'dates.npy',np.array(['2026-01-02','2026-01-03','2026-01-02'],dtype='datetime64[D]'))
    # Any attempt to load a matrix or label would fail.
    (tmp_path/'X.npy').write_bytes(b'DO NOT READ')
    (tmp_path/'y5.npy').write_bytes(b'DO NOT READ')
    m=pd.DataFrame({'stock_id':['0012','0013','0050'],'period':pd.to_datetime(['2026-01-01']*3),'available_date':pd.to_datetime(['2026-02-01',None,'2026-02-01'])})
    a=universe_coverage(m,tmp_path)
    assert a['rows'][0]['research_matrix_members']==2
    assert a['rows'][0]['released_intersection']==1
    assert a['rows'][0]['matrix_members_missing_release']==1


def test_missing_hash_stops_before_any_outputs(tmp_path,monkeypatch):
    import tools.tej_pit as module
    monkeypatch.setattr(module,'__file__',str(tmp_path/'tools/tej_pit.py'))
    raw=tmp_path/'raw'
    raw.mkdir()
    (raw/'SHA256SUMS').write_text('')
    (raw/'monthly_revenue_2013_2026_export20261125.zip').write_bytes(b'synthetic')
    out=tmp_path/'.artifacts/tej/pit'
    with pytest.raises(ValueError,match='STOP SHA'):
        module.run(raw,tmp_path,out,tmp_path/'report.json')
    assert not out.exists()
    assert not (tmp_path/'report.json').exists()


def test_missing_ifrs_period_end_is_not_usable(tmp_path):
    row=['0001 合成','202603','2026/05/10','1','Q','3','.','.','Y','3','NTD','M1']+['2']*11
    p,s=export(tmp_path,IFRS_COLUMNS,[row])
    d=load_export(p,s,'ifrs')
    assert d.release_status.iloc[0]=='NO_PERIOD_END'
    assert not d.publication_date_usable.iloc[0]
