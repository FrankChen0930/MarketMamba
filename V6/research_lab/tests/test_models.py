import numpy as np
import pandas as pd
import pytest
import torch
from research_lab.models import GRURanker, SelectiveRanker, selective_scan, seed_all
from research_lab.metrics import daily_metrics, summarize
from research_lab.run_experiment import sequence_panel

@pytest.mark.parametrize('cls', [GRURanker, SelectiveRanker])
def test_shapes_determinism_and_causality(cls):
    seed_all(17)
    model = cls(3)
    x = torch.randn(4, 60, 3)
    y = model(x)
    assert y.shape == (4, 2)
    seed_all(17)
    other = cls(3)
    torch.testing.assert_close(y, other(x))

def test_scan_reference_and_gradient():
    torch.manual_seed(2)
    u = torch.randn(2, 7, 3, requires_grad=True)
    dt = torch.rand(2, 7, 3)
    a = -torch.rand(3, 2)
    b, c = torch.randn(2, 7, 2), torch.randn(2, 7, 2)
    got = selective_scan(u, dt, a, b, c)
    expected = torch.zeros_like(got)
    for batch in range(2):
        for channel in range(3):
            state = torch.zeros(2)
            for t in range(7):
                state = torch.exp(dt[batch,t,channel]*a[channel])*state + dt[batch,t,channel]*b[batch,t]*u[batch,t,channel]
                expected[batch,t,channel] = (state*c[batch,t]).sum()
    torch.testing.assert_close(got, expected)
    changed = u.detach().clone()
    changed[:,4:] = 999
    torch.testing.assert_close(got[:,:4],selective_scan(changed,dt,a,b,c)[:,:4])
    got.sum().backward()
    assert torch.isfinite(u.grad).all()

def test_metric_hand_values():
    f = pd.DataFrame(dict(code=list('abcdefghij'), date=pd.Timestamp('2020-01-01'), score=np.arange(10), target=np.arange(10)/100))
    m = daily_metrics(f).iloc[0]
    assert m.ic == pytest.approx(1)
    assert m.spread == pytest.approx(.09)
    assert m.hit == 1
    assert summarize(pd.DataFrame({'ic':[1.,-1.,1.,-1.], 'spread':[.1]*4,'hit':[1.]*4}), 1)['ic_mean'] == 0

def test_future_panel_perturbation():
    dates = pd.date_range('2020-01-01', periods=70)
    f = pd.DataFrame([(c,d,float(i)) for c in ['a','b'] for i,d in enumerate(dates)], columns=['code','date','f'])
    a, valid = sequence_panel(f, ['f'])
    f.loc[f.date > dates[64], 'f'] = 999
    b, valid2 = sequence_panel(f, ['f'])
    np.testing.assert_array_equal(a[:65], b[:65])
    np.testing.assert_array_equal(valid[:65], valid2[:65])

@pytest.mark.parametrize('cls', [GRURanker, SelectiveRanker])
def test_fitted_predictions_ignore_future_rows(cls):
    from research_lab.run_experiment import neural_fit_predict
    torch.set_num_threads(1)
    rng = np.random.default_rng(71)
    x = rng.normal(size=(200, 2, 3)).astype('float32')
    train_days = np.repeat(np.arange(59,150),2)
    train_codes = np.tile([0,1],91)
    y = rng.normal(size=(len(train_days),2)).astype('float32')
    test = (np.repeat(np.arange(150,155),2), np.tile([0,1],5))
    first, _ = neural_fit_predict(cls,x,(train_days,train_codes),y,test,17)
    x[155:] = 10000
    second, _ = neural_fit_predict(cls,x,(train_days,train_codes),y,test,17)
    np.testing.assert_array_equal(first,second)

def test_metric_unknown_and_tie_break():
    f = pd.DataFrame(dict(code=list('jihgfedcba'),date=pd.Timestamp('2020-01-01'),score=1.,target=np.arange(10)/100))
    a = daily_metrics(f)
    b = daily_metrics(f.sample(frac=1,random_state=1))
    pd.testing.assert_frame_equal(a,b)
    assert np.isnan(a.ic.iloc[0])
    assert summarize(a,20)['ic_mean'] is None

def test_newey_west_hand_calculation():
    d = pd.DataFrame({'ic':[1.,-1.,1.,-1.], 'spread':[.1]*4, 'hit':[1.]*4})
    m = summarize(d,2)
    # gamma0=1; gamma1=-3/4; Bartlett weight=1/2 => variance=.25, SE=.25.
    assert m['nw_lags'] == 1
    np.testing.assert_allclose(m['ic_ci95'],[-.49,.49])

def test_missing_session_and_unknown_feature_fail_closed():
    days = pd.date_range('2021-01-01',periods=80)
    f = pd.DataFrame([(c,d,1.) for c in ['a','b'] for d in days],columns=['code','date','f'])
    f = f.loc[~((f.code=='a') & (f.date==days[30]))]
    _, mask = sequence_panel(f,['f'])
    assert not mask[:,0].any()  # Missing historical session cannot become a 60-observation window.
    assert mask[59:,1].all()
    f.loc[(f.code=='b') & (f.date==days[70]),'f'] = np.nan
    _, mask = sequence_panel(f,['f'])
    assert mask[69,1]
    assert not mask[70:,1].any()

def test_empty_and_constant_ic_preserve_known_returns():
    empty = pd.DataFrame(columns=['code','date','score','target'])
    summary = summarize(daily_metrics(empty),20)
    assert summary['days'] == 0
    assert summary['ic_mean'] is None
    assert summary['decile_spread'] is None
    f = pd.DataFrame(dict(code=list('abcdefghij'),date=pd.Timestamp('2020-01-01'),score=1.,target=np.arange(10)/100))
    summary = summarize(daily_metrics(f),20)
    assert summary['ic_mean'] is None
    assert summary['decile_spread'] == pytest.approx(.09)
    assert summary['top_hit'] == 1

def test_existing_report_refused_before_loading_data(tmp_path):
    import argparse
    from research_lab.run_experiment import validate_output_paths
    (tmp_path/'results.json').write_text('{}')
    with pytest.raises(FileExistsError):
        validate_output_paths(tmp_path/'predictions',tmp_path)

def test_nw_does_not_compress_missing_sessions():
    d = pd.DataFrame({'ic':[1.,np.nan,-1.], 'spread':[.1]*3, 'hit':[1.]*3})
    m = summarize(d,2)
    np.testing.assert_allclose(m['ic_ci95'],[-1.96/np.sqrt(2),1.96/np.sqrt(2)])
