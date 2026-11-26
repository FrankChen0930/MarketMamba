"""Synthetic, licensed-data-free version reconciliation contracts."""
import hashlib
import zipfile

import pandas as pd
import pytest

from tools import tej_quarterly_versions as q


def export(tmp_path, rows):
    p = tmp_path / 'synthetic.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('data.tsv', ('\t'.join(q.SELF_COLUMNS)+'\n'+'\n'.join('\t'.join(r) for r in rows)).encode('utf-16'))
    s = tmp_path / 'SHA256SUMS.20261126'
    s.write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
    return p, s


def row(scope='Y', date='20260501'):
    return ['0050 合成','202603',scope,date,'1,234','.','100','20','80','70','60','2','3']


def test_loader_keeps_scope_and_unknown_basis_units(tmp_path):
    p,s = export(tmp_path,[row(),row('N','20260502')])
    d=q.load_self(p,s)
    assert list(d.stock_id)==['0050','0050']
    assert d.revenue.iloc[0]==1234000
    assert d.eps.iloc[0]==2
    assert pd.isna(d['營業成本'].iloc[0])
    assert set(d.basis_status)=={'UNKNOWN_SINGLE_OR_YTD'}
    assert not d.strict_pit_eligible.any()
    a=q.duplicate_audit(d)
    assert a['company_period_duplicate_rows']==2
    assert a['same_scope_duplicate_rows']==0
    assert a['groups_with_different_dates']==1


def test_same_scope_collision_not_silently_selected(tmp_path):
    p,s=export(tmp_path,[row(),row(date='20260502')])
    with pytest.raises(ValueError,match='scope key'):
        q.load_self(p,s)


@pytest.mark.parametrize('date',['20261301','bad'])
def test_bad_dates_stop(tmp_path,date):
    p,s=export(tmp_path,[row(date=date)])
    with pytest.raises(ValueError,match='STOP'):
        q.load_self(p,s)


def test_missing_date_and_pre_end_fail_closed(tmp_path):
    p,s=export(tmp_path,[row(date='.')])
    assert pd.isna(q.load_self(p,s).available_date.iloc[0])
    p,s=export(tmp_path,[row(date='20260201')])
    assert not q.load_self(p,s).publication_date_usable.iloc[0]


def test_difference_denominators_missing_zero_sign_precision():
    a=pd.Series([0,1,0,-2,10,10,None,1])
    b=pd.Series([0,0,1,2,10.004,10.02,2,float('inf')])
    assert list(q.difference_class(a,b,'eps'))==['EXACT','REFERENCE_ZERO','GT_1PCT','SIGN','WITHIN_PRECISION','LE_1PCT','MISSING_OR_NONFINITE','MISSING_OR_NONFINITE']


def test_revision_name_only_is_not_numeric_change():
    assert q.revision_class({'value':2,'origin_name':'A'},{'value':2,'origin_name':'B'})=='NAME_ONLY'
    assert q.revision_class({'value':2},{'value':3})=='NUMERIC_CHANGE'


def test_unknown_first_value_never_retroactively_admitted():
    d=pd.DataFrame({'stock_id':['01']*3,'period':pd.to_datetime(['2026-03-01']*3),'observed_at':pd.to_datetime(['2026-05-01','2026-05-02','2026-05-03'],utc=True),'value':[1,2,3]})
    assert q.observed_asof(d,pd.Timestamp('2026-04-30',tz='UTC')).empty
    assert q.observed_asof(d,pd.Timestamp('2026-05-02',tz='UTC')).value.tolist()==[2]


def mutated_loader(old,new):
    import inspect
    source=inspect.getsource(q.load_self)
    assert old in source
    namespace=dict(q.__dict__)
    exec(compile(source.replace(old,new),'<in-memory-loader-mutation>','exec'),namespace)
    return namespace['load_self']


def test_mutation_publication_backfill_killed(tmp_path):
    p,s=export(tmp_path,[row(date='.')])
    mutant=mutated_loader("dates(d['發佈日期'], '%Y%m%d')", "dates(d['發佈日期'], '%Y%m%d').fillna(d['period'])")
    with pytest.raises(AssertionError):
        assert pd.isna(mutant(p,s).available_date.iloc[0])


def test_mutation_eps_scaling_killed(tmp_path):
    p,s=export(tmp_path,[row()])
    mutant=mutated_loader("d['eps'] = d['每股盈餘(元)']", "d['eps'] = d['每股盈餘(元)']*1000")
    with pytest.raises(AssertionError):
        assert mutant(p,s).eps.iloc[0]==2


def test_private_output_guard(tmp_path):
    with pytest.raises(ValueError,match='quarterly'):
        q.check_output(tmp_path)


def test_reconciliation_retains_both_missing_sides_and_rejects_duplicates():
    a=pd.DataFrame({'stock_id':['01','02'],'period':pd.to_datetime(['2026-03-01']*2),'eps':[1.,2.]})
    b=pd.DataFrame({'stock_id':['02','03'],'period':pd.to_datetime(['2026-03-01']*2),'eps':[2.,3.]})
    out=q.reconcile(a,b,'test',['stock_id','period'],['eps'])
    assert out.key_coverage.value_counts().to_dict()=={'left_only':1,'both':1,'right_only':1}
    with pytest.raises(ValueError,match='duplicate'):
        q.reconcile(pd.concat([a,a]),b,'test',['stock_id','period'],['eps'])


def test_cumulative_diagnostic_requires_full_prefix_and_never_sums_eps():
    d=pd.DataFrame({'stock_id':['01']*3+['02']*2,
                    'period':pd.to_datetime(['2026-03-01','2026-06-01','2026-09-01','2026-03-01','2026-09-01']),
                    '合併(Y/N)':['Y']*5,'幣別':['NTD']*5,'structure_candidate':[True]*5,
                    'revenue':[1.,2.,3.,1.,3.],'gross_profit':[1.,None,3.,1.,3.],
                    'operating_income':[1.,2.,3.,1.,3.],'eps':[1.,2.,3.,1.,3.]})
    out=q.cumulative_diagnostic(d)
    assert out.revenue.iloc[:3].tolist()==[1.,3.,6.]
    assert pd.isna(out.revenue.iloc[-1])
    assert pd.isna(out.gross_profit.iloc[2])
    assert 'eps' not in out


def test_float_roundoff_is_not_material_revision():
    assert q.revision_class({'value':0.1+0.2},{'value':0.3})=='FLOAT_REPRESENTATION_ONLY'
    assert q.revision_class({'value':0.3},{'value':0.30001})=='NUMERIC_CHANGE'


def test_asof_does_not_collapse_different_metrics():
    d=pd.DataFrame({'stock_id':['01','01'],'period':pd.to_datetime(['2026-03-01']*2),
                    'metric':['eps','revenue'],'value':[1,2],
                    'observed_at':pd.to_datetime(['2026-05-01']*2,utc=True)})
    assert len(q.observed_asof(d,pd.Timestamp('2026-05-02',tz='UTC')))==2


def test_real_ledger_adapter_replays_revision_and_first_seen_readonly(tmp_path,monkeypatch):
    from tools import pit_ledger as p
    source=tmp_path/'source';source.mkdir()
    ledger=tmp_path/'ledger'
    monkeypatch.setattr(p,'ALLOWED_LEDGER_ROOTS',(ledger,))
    for name in p.KEYS:
        data={'stock_id':['01'],'revenue_year':[2026],'revenue_month':[3],'revenue':[1.]} if name=='revenue_raw' else {
            'stock_id':['01'],'Date':[pd.Timestamp('2026-03-31')],'type':['EPS'],'value':[1.],'origin_name':['基本每股盈餘']}
        pd.DataFrame(data).to_parquet(source/(name+'.parquet'),index=False)
    p.snapshot(source,ledger)
    data={'stock_id':['01','02'],'Date':[pd.Timestamp('2026-03-31')]*2,'type':['EPS']*2,'value':[2.,3.],'origin_name':['基本每股盈餘']*2}
    pd.DataFrame(data).to_parquet(source/'financials_raw.parquet',index=False)
    p.snapshot(source,ledger)
    before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in ledger.rglob('*') if x.is_file()}
    formal=pd.DataFrame({'stock_id':['01','02'],'period':pd.to_datetime(['2026-03-01']*2),'eps':[2.,3.]})
    summary,events,versions=q.ledger_evidence(ledger,formal)
    assert summary['replayed_event_transitions']==2
    assert summary['target_first_seen_events']==1
    assert len(versions)==3
    assert events.formal_match.tolist()==['NEW_ONLY','NEW_ONLY']
    assert {str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in ledger.rglob('*') if x.is_file()}==before
