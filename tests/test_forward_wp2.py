"""Synthetic engineering fixtures; no production data or external fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tools import forward_wp2 as w
from tools.forward_wp1 import Calendar, Stop
from V6.research_lab import c0plus, rolling_folds


def fixture(root):
    root.mkdir()
    days = pd.bdate_range('2023-10-02', '2024-09-02').to_numpy().astype('datetime64[D]')
    days = days[days != np.datetime64('2024-01-01')]
    dates = np.repeat(days, 16)
    names = list(dict.fromkeys(c0plus.CORE + c0plus.MARKET + ['Market_Cap_Log']))
    rng = np.random.default_rng(31)
    a = dict(X=rng.normal(size=(len(dates),len(names))).astype('float32'),
             dates=dates, stock_ids=np.tile(np.array([f'S{i:02}' for i in range(16)]),len(days)),
             stock_idx=np.tile(np.arange(16,dtype='int32'),len(days)),
             end_indices=np.arange(16,len(dates)+1,16),
             splits=np.where(dates < days[3],0,1).astype('uint8'),
             y5=rng.normal(size=len(dates)).astype('float32'),
             y10=np.full(len(dates),np.nan,dtype='float32'))
    for k,v in a.items(): np.save(root/f'{k}.npy',v)
    (root/'manifest.json').write_text(json.dumps(dict(rows=len(dates),feature_order=names)))
    return a, Calendar([str(d) for d in days],str(days[0]),str(days[-1]),{'fixture':True})


@pytest.mark.parametrize('origin',['2024-01-02','2024-04-16','2024-08-26'])
def test_origin_inclusive_minus21_and_annual_equivalence(tmp_path,origin):
    a,c=fixture(tmp_path/'matrix')
    s=w.origin_splits(a['dates'],a['splits'],c,origin,'2024-09-02')
    cutoff=np.datetime64(c.origin_minus21(origin))
    assert np.max(a['dates'][s==1])==cutoff
    assert np.all(s[(a['dates']==cutoff)&(a['splits']!=0)]==1)
    assert np.all(s[(a['dates']>cutoff)&(a['dates']<np.datetime64(origin))&(a['splits']!=0)]==4)
    assert np.all(s[a['dates']==np.datetime64(origin)]==2)
    if origin=='2024-01-02':
        assert np.array_equal(s,rolling_folds.make_fold_splits(a['dates'],a['splits'],2024))


def test_calendar_missing_origin_and_history_fail_closed(tmp_path):
    a,c=fixture(tmp_path/'matrix')
    for origin in ['2024-01-01','2023-10-03','2025-01-02']:
        with pytest.raises(Stop):w.origin_splits(a['dates'],a['splits'],c,origin,'2024-09-02')
    with pytest.raises(Stop):w.origin_splits(a['dates'][16:],a['splits'][16:],c,'2024-01-02','2024-09-02')


@pytest.mark.parametrize('mutation',['future_labels','future_features'])
def test_future_mutation_preserves_fit_sample_model_and_fixed_probe(tmp_path,mutation):
    source=tmp_path/'matrix'; a,c=fixture(source)
    view=w.prepare_view(source,tmp_path/'view',c,'2024-04-16','2024-09-02')
    first=w.fit_predict(view,17,max_train_rows=700)
    future=a['dates']>np.datetime64('2024-04-16')
    if mutation=='future_labels':
        a['y5'][future]=np.nan; np.save(source/'y5.npy',a['y5'])
    else:
        a['X'][future]=1e15; np.save(source/'X.npy',a['X'])
    second=w.fit_predict(view,17,max_train_rows=700)
    for key in ['train_idx','fit_idx','target']:
        assert first[key].tobytes()==second[key].tobytes()
    assert first['model_text']==second['model_text']
    probe=first['dates']==np.datetime64('2024-04-16')
    assert first['scores'][probe].tobytes()==second['scores'][probe].tobytes()
    assert first['fit_probe_scores'].tobytes()==second['fit_probe_scores'].tobytes()


def test_key_replay_and_no_y10_filter_or_eval_label_filter(tmp_path):
    source=tmp_path/'matrix';a,c=fixture(source)
    view=w.prepare_view(source,tmp_path/'view',c,'2024-04-16','2024-09-02')
    a['y5'][a['dates']>=np.datetime64('2024-04-16')]=np.nan
    np.save(source/'y5.npy',a['y5'])
    r=w.fit_predict(view,29,max_train_rows=700)
    assert len(r['fit_idx'])==700
    assert len(r['eval_idx'])==np.count_nonzero(a['dates']>=np.datetime64('2024-04-16'))
    keys=w.save_replay(view,r,tmp_path/'replay',29)
    assert w.replay_keys(view,keys,29,700)
    p=Path(keys['sample']['path']); values=np.load(p);values[0]+=1;np.save(p,values)
    with pytest.raises(Stop):w.replay_keys(view,keys,29,700)


def test_output_namespace_and_overwrite(tmp_path):
    with pytest.raises(Stop):w.output_path(Path('/tmp/wp2-outside'))
    with pytest.raises(FileExistsError):w.output_path(w.ROOT)


def test_key_comparison_normalizes_datetime_storage_unit_only():
    a=pd.DataFrame(dict(date=pd.to_datetime(['2024-01-02']).as_unit('ns'),code=['SYNTH']))
    b=a.copy();b['date']=b.date.dt.as_unit('ms')
    assert w.same_keys(a,b,['date','code'])
    b['date']=pd.to_datetime(['2024-01-03'])
    assert not w.same_keys(a,b,['date','code'])
