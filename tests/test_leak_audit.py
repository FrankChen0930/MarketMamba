"""Bounded synthetic audit checks; no formal data or fits."""
import numpy as np
import pandas as pd
import pytest
from tools.leak_audit import scalefree, feature_names, price_columns, perturb_future, bitwise_prefix
from tools.tradability_diagnostic import exit_missing_marks


def test_json_unknown_is_null_and_never_zero(tmp_path):
    import json
    from tools.leak_audit_run import write
    path=tmp_path/'unknown.json'
    write(path, {'raw': [{'value': np.nan}], 'metric': float('inf')})
    assert json.loads(path.read_text())=={'raw':[{'value':None}],'metric':None}


def test_reference_uses_strict_prior_calendar_session_and_never_fills_gaps():
    from tools.leak_audit_run import prior_session_values
    panel=np.array([[1.],[np.nan],[9.]])
    got=prior_session_values(panel,np.array([[0,0],[1,0],[2,0]]))
    assert np.isnan(got[0]) and got[1]==1. and np.isnan(got[2])


def test_scalefree_price_units_and_lagged_atr_preserve_other_columns():
    names = feature_names()
    x = np.full((2, 190), 8., np.float32)
    x[:, names.index('Close')] = [2., 4.]
    out = scalefree(x, names, np.array([10., np.nan]))
    assert out.shape == (2, 191)
    np.testing.assert_array_equal(out[:, names.index('Close')], [1., 1.])
    np.testing.assert_array_equal(out[:, names.index('ATR_14_lag20')], [4., 2.])
    np.testing.assert_array_equal(out[:, names.index('Volume')], x[:, names.index('Volume')])
    np.testing.assert_array_equal(out[:, names.index('monthly_yoy')], [8., 8.])
    assert out[0, -1] == np.float32(np.log(10.))
    assert np.isnan(out[1, -1])
    np.testing.assert_array_equal(x[:, 0], [8., 8.])


def test_scalefree_invalid_denominator_fails_closed():
    with pytest.raises(ValueError, match='Close'):
        scalefree(np.zeros((1, 190), np.float32), feature_names(), np.array([10.]))


def test_per_stock_backward_scale_multiplier_cancels_in_all_price_dimensions():
    names=feature_names(); columns=price_columns(names)
    assert len(names)==190 and len(columns)==15
    rng=np.random.default_rng(17)
    x=rng.uniform(1,20,size=(2,190)).astype(np.float32)
    changed=x.copy(); changed[:,columns]*=np.array([2.,4.],np.float32)[:,None]
    avg=np.array([15.,20.])
    # Powers of two avoid introducing rounding differences in this property.
    np.testing.assert_array_equal(scalefree(x,names,avg),scalefree(changed,names,avg))


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('cut', [3, 6, 9])
def test_future_perturbation_bitwise_and_detects_noncausal_builder(seed, cut):
    f = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=12),
                      'Close': np.arange(12.) + 1, 'stock_id': ['a'] * 12})
    other = perturb_future(f, f.Date.iloc[cut], seed)
    assert bitwise_prefix(f[['Close']].to_numpy(), other[['Close']].to_numpy(), np.arange(12) <= cut)
    a = f.Close.rolling(3).mean().to_numpy()
    b = other.Close.rolling(3).mean().to_numpy()
    assert bitwise_prefix(a, b, np.arange(12) <= cut)
    a = f.Close.to_numpy() / f.Close.mean()
    b = other.Close.to_numpy() / other.Close.mean()
    assert not bitwise_prefix(a, b, np.arange(12) <= cut)


def test_missing_exit_liquidates_at_last_close_and_never_reenters():
    entry = np.array([10., 20.])
    marks = np.array([[11., 21.], [np.nan, 22.], [99., 23.]])
    out, exits = exit_missing_marks(entry, marks, 'last_close')
    np.testing.assert_array_equal(out[:, 0], [11., 11., 11.])
    np.testing.assert_array_equal(out[:, 1], [21., 22., 23.])
    assert exits.tolist() == [0, -1]
    loss, exits = exit_missing_marks(entry, marks, 'loss100')
    np.testing.assert_array_equal(loss[:, 0], [11., 0., 0.])
    # If first close absent, last available close is signal-day close, not entry open.
    out, exits = exit_missing_marks(np.array([10.]), np.array([[np.nan], [11.]]),
                                    'last_close', previous_close=np.array([9.]))
    np.testing.assert_array_equal(out[:, 0], [9., 9.])
    assert exits.tolist() == [0]
    with pytest.raises(ValueError):
        exit_missing_marks(np.array([np.nan]), marks[:, :1], 'last_close')
    with pytest.raises(ValueError, match='preceding close'):
        exit_missing_marks(np.array([10.]), np.array([[np.nan]]), 'last_close')


def test_missing_stock_does_not_remove_cycle_and_exit_cost_is_once():
    from tools.tradability_diagnostic import simulate_portfolios
    days=np.arange('2020-01-01','2020-01-25',dtype='datetime64[D]')
    keys=np.array([(0,j) for j in range(20)])
    prices={c:np.full((len(days),20),10.) for c in ('Open','Close')}
    prices['Close'][5:,19]=np.nan
    scores={a:np.arange(20.) for a in 'AB'}
    paths,buy,sell,fail=simulate_portfolios(keys,scores,np.ones(20,bool),prices,days,days[-1],kinds=('top10','equal'))
    assert fail['missing_selected_price']==0
    r=paths['B','top10',.001][0,1:21]
    assert np.isfinite(r).all()
    # Initial buy=1; forced exit 0.5; terminal exit surviving 0.5.
    assert np.nansum(buy['B','top10'])==1.
    assert np.nansum(sell['B','top10'])==1.
    np.testing.assert_allclose(np.prod(1+r),1-.002425-.005425)
    loss,_,_,_=simulate_portfolios(keys,scores,np.ones(20,bool),prices,days,days[-1],kinds=('top10','equal'),missing_policy='loss100')
    np.testing.assert_allclose(np.prod(1+loss['B','top10',.001][0,1:21]),.5-.002425-.005425)
