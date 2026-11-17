"""Offline safety regression; all writer tests use pytest temporary files only."""
import json

import numpy as np
import pandas as pd
import pytest

from V6.research_lab import repair_per as rp

DAY = '2026-10-12'
CAL = ['2026-09-24', '2026-09-29', '2026-10-08', DAY, '2026-10-13', '2026-10-14', '2026-10-27', '2026-10-28']


def body(day=DAY, n=10):
    d = pd.Timestamp(day)
    return dict(stat='OK', date=d.strftime('%Y%m%d'),
                title=f'{d.year-1911}年{d.month:02d}月{d.day:02d}日 個股日本益比',
                fields=['證券代號', '殖利率(%)', '本益比', '股價淨值比'],
                data=[[str(1100+i), '5.25', '12.5', '1.2'] for i in range(n)])


def frame(day=DAY, n=10):
    return pd.DataFrame(dict(Date=pd.to_datetime([day]*n), stock_id=pd.Series([str(1100+i) for i in range(n)], dtype='str'),
                             PER=np.full(n, 10.), PBR=np.full(n, 1.), DY=np.full(n, 5.)))


def production():
    return pd.concat([frame(d) for d in CAL] + [frame('2026-10-09')], ignore_index=True)


@pytest.mark.parametrize('mutation', ['missing', 'wrong', 'title', 'no_title', 'status'])
def test_refuse_wrong_or_missing_source_identity(mutation):
    b = body()
    if mutation == 'missing': b.pop('date')
    if mutation == 'wrong': b['date'] = '20261117'
    if mutation == 'title': b['title'] = '115年11月17日 個股日本益比'
    if mutation == 'no_title': b.pop('title')
    if mutation == 'status': b['stat'] = 'error'
    with pytest.raises(ValueError): rp.parse_source(b, DAY, frame())


def test_mapping_nulls_and_exact_schema():
    b = body(); b['data'][0][2] = '-'
    out = rp.parse_source(b, DAY, frame())
    assert list(out) == list(frame()) and out.dtypes.equals(frame().dtypes)
    assert out.DY.iloc[0] == 5.25 and out.PBR.iloc[0] == 1.2 and pd.isna(out.PER.iloc[0])
    b['data'].append(b['data'][0])
    with pytest.raises(ValueError): rp.parse_source(b, DAY, frame())


def test_plan_changes_additions_null_equality_and_scope():
    p = production(); p = p[~((p.Date == DAY) & (p.stock_id == '1109'))]
    b = body(); b['data'][0][1:] = ['5', '10', '1']
    plan = rp.plan(p, CAL, {DAY: b})
    r = plan['report']['days'][DAY]
    assert (r['to_overwrite'], r['to_add'], r['unchanged']) == (8, 1, 1)
    assert r['change_ratio'] == 8/9 and r['field_changes']['PER']['mean_delta'] == 2.5
    assert len(r['largest_changes']) == 8
    assert set(plan['report']['days']) == set(rp.DAYS)
    assert plan['report']['days']['2026-10-09']['removable'] == 10
    assert plan['report']['days']['2026-10-09']['neighbors']['previous']['equal_ratio'] == 1


def test_low_coverage_and_unknown_calendar_refused():
    p = production()
    result = rp.plan(p, CAL, {DAY: body(n=8)})
    assert result['report']['days'][DAY]['status'] == 'REFUSED'
    assert result['overwrite'].empty and result['add'].empty
    assert result['report']['typical_rows'] == 10
    result = rp.plan(p, CAL[:-2], {})
    assert result['report']['days']['2026-10-27']['status'] == 'REFUSED'


@pytest.mark.parametrize('overwrite,remove', [(False,False),(False,True),(True,False)])
def test_apply_requires_independent_flags(tmp_path, overwrite, remove):
    t = tmp_path/'per.parquet'; production().to_parquet(t, index=False)
    before = t.read_bytes(); plan = rp.plan(production(), CAL, {d:body(d) for d in rp.DAYS if d in CAL})
    with pytest.raises(ValueError):
        rp.apply(t, plan, tmp_path/'backup', allow_overwrite=overwrite, allow_remove=remove, now='2026-11-18T10:00+08:00')
    assert t.read_bytes() == before and not (tmp_path/'backup').exists()


@pytest.mark.parametrize('now', ['2026-11-18T22:20+08:00','2026-11-18T22:50:59+08:00','2026-11-18T14:35Z'])
def test_apply_blocks_taipei_window(tmp_path, now):
    t = tmp_path/'per.parquet'; production().to_parquet(t, index=False)
    with pytest.raises(ValueError):
        rp.apply(t, rp.plan(production(), CAL, {}), tmp_path/'bk', now=now)
    assert not (tmp_path/'bk').exists()


def test_apply_roundtrip_backup_untouched_and_report(tmp_path):
    p = production(); t = tmp_path/'per.parquet'; p.to_parquet(t,index=False); original = t.read_bytes()
    plan = rp.plan(p, CAL, {d:body(d) for d in rp.DAYS if d in CAL})
    result = rp.apply(t, plan, tmp_path/'bk', allow_overwrite=True, allow_remove=True, now='2026-11-18T10:00+08:00')
    out = pd.read_parquet(t)
    assert len(out) == len(p)-10 and not out.duplicated(rp.KEY).any()
    assert out.dtypes.equals(p.dtypes) and list(out) == list(p)
    pd.testing.assert_frame_equal(out[out.Date == '2026-10-08'].reset_index(drop=True), p[p.Date == '2026-10-08'].reset_index(drop=True))
    assert (tmp_path/'bk'/'before.parquet').read_bytes() == original
    assert json.loads((tmp_path/'bk'/'diff.json').read_text())['after_sha256'] == rp.sha256(t)
    assert result['rows_after'] == len(out)


def test_apply_rejects_stale_plan_and_corrupt_temp(tmp_path, monkeypatch):
    p = production(); t = tmp_path/'per.parquet'; p.to_parquet(t,index=False)
    plan = rp.plan(p, CAL, {DAY:body()})
    p.loc[0,'PER'] = 99.; p.to_parquet(t,index=False)
    with pytest.raises(ValueError): rp.apply(t, plan, tmp_path/'bk', allow_overwrite=True, allow_remove=True)
    p = production(); p.to_parquet(t,index=False); original = t.read_bytes()
    original_write = pd.DataFrame.to_parquet
    def corrupt(self, path, **kwargs):
        f = self.copy(); f.loc[0,'PER'] = 999.; return original_write(f,path,**kwargs)
    monkeypatch.setattr(pd.DataFrame,'to_parquet',corrupt)
    with pytest.raises(ValueError):
        rp.apply(t, plan, tmp_path/'bk2', allow_overwrite=True, allow_remove=True, now='2026-11-18T10:00+08:00')
    assert t.read_bytes() == original


def test_dry_run_never_calls_writer_or_fetches_nontrading(tmp_path, monkeypatch):
    t=tmp_path/'per.parquet'; production().to_parquet(t,index=False)
    prices=tmp_path/'prices.parquet'; pd.DataFrame({'Date':pd.to_datetime(CAL)}).to_parquet(prices,index=False)
    original=t.read_bytes(); seen=[]
    def fetch(day, archive): seen.append(day); return body(day)
    monkeypatch.setattr(rp,'fetch_source',fetch)
    monkeypatch.setattr(rp,'apply',lambda *a,**kw: pytest.fail('dry run called writer'))
    rp.main(['--target',str(t),'--prices',str(prices),'--output-dir',str(tmp_path/'report')])
    assert seen == [d for d in rp.DAYS if d in CAL] and t.read_bytes() == original


def test_null_equality_and_transition_are_distinct():
    p = production(); p.loc[(p.Date == DAY) & p.stock_id.isin(['1100','1101']), 'PER'] = np.nan
    b = body(); b['data'][0][2] = '-'
    r = rp.plan(p, CAL, {DAY:b})['report']['days'][DAY]
    assert r['to_overwrite'] == 10  # all DY values differ, including the paired-null row
    assert r['field_changes']['PER']['null_transitions'] == 1
    assert r['field_changes']['PER']['paired_numeric'] == 8
    p.loc[p.Date == DAY, ['DY','PBR']] = [5.25,1.2]
    p.loc[(p.Date == DAY) & ~p.stock_id.isin(['1100','1101']), 'PER'] = 12.5
    r = rp.plan(p, CAL, {DAY:b})['report']['days'][DAY]
    assert r['to_overwrite'] == 1 and r['unchanged'] == 9


def test_ninety_percent_boundary_passes_and_schema_drift_refuses():
    result = rp.plan(production(),CAL,{DAY:body(n=9)})
    assert result['report']['days'][DAY]['status'] == 'READY'
    b=body(); b['fields'][1] = '未知殖利率'
    assert rp.plan(production(),CAL,{DAY:b})['report']['days'][DAY]['status'] == 'REFUSED'


def test_apply_addition_only_does_not_require_existing_row_flags(tmp_path):
    p=production(); p=p[p.Date != '2026-10-09'].reset_index(drop=True)
    p.loc[p.Date == DAY, ['PER','PBR','DY']] = [12.5,1.2,5.25]
    p=p[~((p.Date == DAY) & (p.stock_id == '1109'))].reset_index(drop=True)
    t=tmp_path/'per.parquet'; p.to_parquet(t,index=False)
    result=rp.apply(t,rp.plan(p,CAL,{DAY:body()}),tmp_path/'bk',now='2026-11-18T10:00+08:00')
    assert result['added'] == 1 and result['overwritten'] == result['removed'] == 0
    assert len(pd.read_parquet(t)) == len(p)+1
