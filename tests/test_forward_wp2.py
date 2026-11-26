"""Synthetic engineering fixtures; no production data or external fits."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tools import forward_wp2 as w
from tools.forward_wp1 import Calendar, Stop
from V6.research_lab import c0plus, rolling_folds


@pytest.fixture(autouse=True)
def _checkout_root_under_tmp(tmp_path,monkeypatch,request):
    # Outputs must live below the checkout; tests treat tmp_path as the checkout so they
    # do not depend on where pytest's basetemp happens to be.
    if request.node.name.startswith('test_output_namespace'):return
    monkeypatch.setattr(w,'ROOT',tmp_path)


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


def array_hash(values):
    """Hash actual values, dtype and shape; never a caller-reported digest."""
    a = np.asarray(values)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def observed_fit_predict(view, seed, monkeypatch):
    """Observe real LightGBM boundaries while retaining real fit/predict code."""
    from lightgbm import LGBMRegressor
    fit, predict = LGBMRegressor.fit, LGBMRegressor.predict
    used = dict(fit=[], predict=[])

    def fit_spy(model, x, y, *args, **kwargs):
        used['fit'].append((array_hash(x), array_hash(y)))
        return fit(model, x, y, *args, **kwargs)

    def predict_spy(model, x, *args, **kwargs):
        # Copy at the boundary so subsequent caller mutation cannot mask a leak.
        used['predict'].append(np.array(x, copy=True))
        return predict(model, x, *args, **kwargs)

    with monkeypatch.context() as spy:
        spy.setattr(LGBMRegressor, 'fit', fit_spy)
        spy.setattr(LGBMRegressor, 'predict', predict_spy)
        result = w.fit_predict(view, seed, max_train_rows=700)
    assert len(used['fit']) == 1 and len(used['predict']) == 2
    assert used['predict'][0].shape[0] == len(result['eval_idx'])
    assert used['fit'][0][1] == array_hash(result['target'])
    return result, used


def assert_future_invariance(tmp_path, monkeypatch, mutation, offset, seed=17):
    source = tmp_path/'matrix'
    a, c = fixture(source)
    origin = '2024-04-16'
    t = np.datetime64(c.shift(origin, offset))
    view = w.prepare_view(source, tmp_path/'view', c, origin, c.shift(origin, 13))
    if mutation == 'future_feature_missingness':
        future=np.flatnonzero(a['dates'] > t)
        a['X'][future[::3],0]=np.nan
        np.save(source/'X.npy',a['X'])
    first, first_used = observed_fit_predict(view, seed, monkeypatch)
    if mutation in ('future_features','future_feature_missingness'):
        mask = a['dates'] > t
        if mutation == 'future_features':
            a['X'][mask] = 1e15
        else:
            changed=a['X'][mask].copy()
            changed[:,::2]=np.where(np.isfinite(changed[:,::2]),np.nan,17.0)
            a['X'][mask]=changed
        np.save(source/'X.npy', a['X'])
    else:
        # A row's y5 realizes at signal+5, not at the signal date. Changing
        # only label rows dated >t cannot expose scores += y5[eval_idx].
        for label in ('y5', 'y10'):
            horizon = int(label[1:])
            boundary = t if mutation in ('future_labels','future_label_missingness') else np.datetime64(c.shift(str(t), -horizon))
            mask = a['dates'] > boundary
            a[label][mask] = (np.where(np.isfinite(a[label][mask]),np.nan,17.0)
                              if mutation == 'future_label_missingness' else np.nan)
            np.save(source/(label+'.npy'), a[label])
    assert mask.any()
    second, second_used = observed_fit_predict(view, seed, monkeypatch)
    for key in ('train_idx', 'fit_idx', 'target', 'feature_order'):
        assert array_hash(first[key]) == array_hash(second[key]), key
    assert first_used['fit'] == second_used['fit'], 'fit_features/target hashes'
    assert first['model_text'] == second['model_text']
    for key in ('train_idx', 'fit_idx'):
        for column in ('dates', 'stock_ids'):
            assert array_hash(a[column][first[key]]) == array_hash(a[column][second[key]]), key
    before = first['dates'] <= t
    after = second['dates'] <= t
    assert before.sum() == (offset+1)*16  # every historical decision row, not just t
    assert array_hash(first['eval_idx'][before]) == array_hash(second['eval_idx'][after]), 'row keys'
    for column in ('dates', 'stock_ids'):
        assert array_hash(a[column][first['eval_idx'][before]]) == array_hash(a[column][second['eval_idx'][after]]), column
    assert array_hash(first_used['predict'][0][before]) == array_hash(second_used['predict'][0][after]), 'predict_features hash'
    assert array_hash(first['scores'][before]) == array_hash(second['scores'][after]), 'historical scores'
    assert array_hash(first['fit_probe_scores']) == array_hash(second['fit_probe_scores'])
    evidence = dict(
        decision=str(t), seed=seed, mutation=mutation, historical_rows=int(before.sum()),
        fit_features_sha256=first_used['fit'][0][0], target_sha256=first_used['fit'][0][1],
        predict_features_sha256=array_hash(first_used['predict'][0][before]),
        scores_sha256=array_hash(first['scores'][before]),
        row_keys_sha256=array_hash(first['eval_idx'][before]),
        date_keys_sha256=array_hash(a['dates'][first['eval_idx'][before]]),
        stock_keys_sha256=array_hash(a['stock_ids'][first['eval_idx'][before]]))
    (tmp_path/'invariance.json').write_text(json.dumps(evidence, sort_keys=True)+'\n')


@pytest.mark.parametrize('seed', [17, 29, 43])
@pytest.mark.parametrize('offset', range(14))
@pytest.mark.parametrize('mutation', ['future_labels', 'immature_labels', 'future_features',
                                     'future_feature_missingness','future_label_missingness'])
def test_future_mutation_preserves_all_history_and_actual_model_inputs(tmp_path, monkeypatch, mutation, offset, seed):
    assert_future_invariance(tmp_path, monkeypatch, mutation, offset, seed)


def leaking_fit_predict(kind):
    """Source mutants keep SeqData/Context/C0+/LightGBM real; disk stays intact."""
    source = inspect.getsource(w.fit_predict)
    old = 'scores=model.predict(xe)'
    if kind == 'eval_label':
        new = "scores=model.predict(xe) + np.nan_to_num(data.y['y5'][-16:].mean()) * 1e-12"
    elif kind == 'origin9':
        new = "scores=model.predict(xe)\n    scores[144:160] += np.float64(data.X[-16:,0].mean())"
    elif kind == 'missingness':
        new = 'scores=model.predict(xe) + np.isfinite(data.X[-16:,0]).mean()'
    elif kind == 'sample_order':
        old = 'xf,names=c0plus.build_features(data,ctx.fit_idx,data.feature_names)'
        new = "if data.X[-16:,0].mean() > 1e10: ctx.fit_idx = ctx.fit_idx[::-1]\n    xf,names=c0plus.build_features(data,ctx.fit_idx,data.feature_names)"
    elif kind == 'target':
        old = 'xf,names=c0plus.build_features(data,ctx.fit_idx,data.feature_names)'
        new = "ctx.y_fit = ctx.y_fit.astype('float64') + np.nan_to_num(data.y['y5'][-16:].mean()) * 1e-12\n    xf,names=c0plus.build_features(data,ctx.fit_idx,data.feature_names)"
    else:
        new = 'xe = xe + np.float32(data.X[-16:, 0].mean())\n    scores=model.predict(xe)'
    assert source.count(old) == 1
    namespace = dict(w.__dict__)
    exec(compile(source.replace(old, new), '<forward-wp2-'+kind+'>', 'exec'), namespace)
    return namespace['fit_predict']


@pytest.mark.parametrize('kind,mutation,reason', [
    ('eval_label', 'future_labels', 'historical scores'),
    ('origin9', 'future_features', 'historical scores'),
    ('sample_order', 'future_features', 'fit_idx'),
    ('target', 'future_labels', 'target'),
    ('future_feature', 'future_features', 'predict_features hash'),
    ('missingness', 'future_feature_missingness', 'historical scores'),
])
@pytest.mark.parametrize('offset', [9])
def test_future_mutation_kills_leaking_mutants(tmp_path, monkeypatch, kind, mutation, reason, offset):
    mutant = leaking_fit_predict(kind)
    monkeypatch.setattr(w, 'fit_predict', mutant)
    with pytest.raises(AssertionError, match=reason):
        assert_future_invariance(tmp_path, monkeypatch, mutation, offset)


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
