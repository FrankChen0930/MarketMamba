"""Synthetic, label-free Step6 causal and fail-closed checks."""
import numpy as np
import pandas as pd
import pytest

from tools import step3
from tools.step6_features import (
    FEATURES, select_version, eligible_edges, customer_features, revenue_snapshot,
)


def edge(a, b, kind='supplier', evidence='primary', **kwargs):
    return dict(src=a, dst=b, type=kind, evidence_level=evidence, **kwargs)


def test_visibility_two_clocks_and_historical_description():
    versions = [dict(commit='a', committed_at='2026-10-01T00:00Z', order=0),
                dict(commit='b', committed_at='2026-11-29T00:00Z', order=1)]
    obs = [dict(head='b', ancestors=['a', 'b'], observed_at='2026-11-28T00:00Z',
                validated_at='2026-11-28T01:00Z', valid=True)]
    assert select_version(versions, obs, '2026-10-02T00:00Z')['visibility'] == 'upstream_commit_only'
    assert select_version(versions, obs, '2026-11-28T00:30Z', strict=True) is None
    assert select_version(versions, obs, '2026-11-28T02:00Z', strict=True)['commit'] == 'a'
    assert select_version(versions, obs, '2026-11-29T00:00Z', strict=True)['commit'] == 'b'
    assert select_version(versions, obs, '2026-12-06T00:00Z', strict=True) is None
    assert select_version(versions, [dict(obs[0], valid=False)], '2026-11-29T00:00Z', strict=True) is None


def test_direction_group_components_e_pm_and_unknown_transaction():
    rows = [edge('tw.S', 'tw.A'), edge('tw.S', 'tw.B', evidence='media'),
            edge('tw.S', 'tw.C', evidence='other'), edge('tw.S', 'us.D'),
            edge('tw.P', 'tw.S', 'subsidiary'), edge('tw.P', 'tw.G', 'subsidiary'),
            edge('tw.G', 'tw.H', 'subsidiary'), edge('tw.S', 'tw.H'),
            edge('tw.S', 'tw.S'), edge('tw.S', 'tw.A', weight=999),
            edge('tw.T', 'tw.Z'), edge('tw.T', 'tw.Z', 'transaction', detail='子公司交易')]
    r = eligible_edges(rows, {'S', 'A', 'B', 'C', 'P', 'G', 'H', 'T', 'Z'})
    assert r['customers']['S'] == ['A', 'B']
    assert 'A' not in r['customers']  # src is supplier, dst is customer
    assert 'T' in r['unknown_suppliers']
    assert r['excluded']['SAME_GROUP'] == 1
    assert r['excluded']['EVIDENCE_LEVEL'] == 1
    assert r['excluded']['TRANSACTION_ENTITY_UNKNOWN'] == 1


def test_common_denominator_count_unknown_and_zero():
    r = eligible_edges([edge('tw.S', 'tw.A'), edge('tw.S', 'tw.B'), edge('tw.S', 'tw.C')], {'S','A','B','C','Q'})
    rev = {'A': dict(yoy=.2, surprise=.3), 'B': dict(yoy=.4, surprise=.5),
           'C': dict(yoy=100, surprise=np.nan)}
    x = customer_features('S', r, rev)
    assert [x[n] for n in FEATURES] == pytest.approx([.3, .4, 2])
    assert x['max_customer_share'] == .5
    rev.pop('B')
    x = customer_features('S', r, rev)
    assert np.isnan(x[FEATURES[0]]) and x[FEATURES[2]] == 1
    assert customer_features('Q', r, rev)[FEATURES[2]] == 0
    assert np.isnan(customer_features('UNMAPPED', r, rev)[FEATURES[2]])
    assert np.isnan(customer_features('S', None, rev)[FEATURES[2]])
    assert np.isnan(customer_features('S', r, rev, scope_known=False)[FEATURES[2]])
    r['unknown_suppliers'].add('S')
    assert all(np.isnan(customer_features('S', r, rev)[n]) for n in FEATURES)


def revenue_fixture():
    periods = pd.date_range('2013-01-01', '2016-05-01', freq='MS')
    f = pd.DataFrame(dict(stock_id='A', period=periods,
        available_date=periods+pd.offsets.MonthEnd(0)+pd.Timedelta(days=10),
        revenue=np.arange(len(periods))+100., scope='C', currency='NTD', unit='yuan',
        comparability_id='known', version=0, source_row=np.arange(len(periods))))
    return f, pd.bdate_range('2013-01-01','2016-08-01')


def test_reuses_step3_surprise_and_future_invariance():
    f, days = revenue_fixture()
    cutoff = pd.Timestamp('2016-03-14')
    expected = step3.monthly_events(f, days)
    row = expected[expected.session <= days.get_loc(cutoff)].iloc[-1]
    snap = revenue_snapshot(f, days, cutoff)
    assert snap['A']['surprise'] == row.monthly_surprise
    assert snap['A']['yoy'] == row.monthly_yoy
    changed = f.copy()
    changed.loc[changed.available_date > cutoff, 'revenue'] = 1e12
    assert snap == revenue_snapshot(changed, days, cutoff)
    # Release day is excluded; late old months must not replace latest month.
    release = pd.Timestamp('2016-03-10')
    assert revenue_snapshot(f, days, release)['A']['period'] == '2016-01'
    old = f.period == pd.Timestamp('2013-01-01')
    f.loc[old, 'available_date'] = pd.Timestamp('2016-03-11')
    assert revenue_snapshot(f, days, cutoff)['A']['period'] == '2016-02'


def test_reverse_conflict_and_unknown_mapping_fail_closed():
    r = eligible_edges([edge('tw.S','tw.A'), edge('tw.A','tw.S'), edge('tw.S','tw.Z')], {'S','A'})
    assert r['unknown_suppliers'] == {'S', 'A'}
    assert r['excluded']['DIRECTION_CONFLICT'] == 2


def test_no_observations_does_not_invent_visibility():
    assert select_version([dict(commit='a', committed_at='2026-01-01T00:00Z', order=0)], [],
                          '2026-02-01T00:00Z') is None


def test_reobservation_preserves_first_visibility_and_version_causality():
    versions = [dict(commit='a', committed_at='2026-11-01T00:00Z', order=0),
                dict(commit='b', committed_at='2026-11-28T00:00Z', order=1)]
    first = dict(head='a', ancestors=['a'], observed_at='2026-11-10T00:00Z',
                 validated_at='2026-11-10T01:00Z', valid=True)
    repeat = dict(first, observed_at='2026-11-16T00:00Z', validated_at='2026-11-16T01:00Z')
    x = select_version(versions, [first, repeat], '2026-11-18T00:00Z', strict=True)
    assert x['commit'] == 'a' and x['visible_at'] == '2026-11-10T01:00:00+00:00'
    future = dict(head='b', ancestors=['a','b'], observed_at='2026-11-29T00:00Z',
                  validated_at='2026-11-29T01:00Z', valid=True)
    assert x == select_version(versions, [first, repeat, future], '2026-11-18T00:00Z', strict=True)


def test_actual_receipt_and_validation_can_only_delay_visibility():
    f, days = revenue_fixture()
    f['received_at'] = pd.to_datetime(f.available_date).dt.tz_localize('Asia/Taipei')
    f['validated_at'] = f.received_at
    last = f.period == pd.Timestamp('2016-02-01')
    f.loc[last, 'received_at'] = pd.Timestamp('2016-03-14T12:00+08:00')
    f.loc[last, 'validated_at'] = pd.Timestamp('2016-03-14T13:00+08:00')
    assert revenue_snapshot(f, days, '2016-03-14T12:30+08:00')['A']['period'] == '2016-01'
    assert revenue_snapshot(f, days, '2016-03-14T13:30+08:00')['A']['period'] == '2016-02'
    with pytest.raises(ValueError, match='incomplete actual'):
        revenue_snapshot(f.drop(columns='validated_at'), days, '2016-03-14')


def test_transaction_verified_resolution_and_missing_group_scope():
    rows = [edge('tw.S','tw.A'), edge('tw.S','tw.A','transaction', detail='子公司交易',
                                     entity_resolution='verified_same_group')]
    r = eligible_edges(rows, {'S','A'})
    assert r['excluded']['SAME_GROUP'] == 1 and not r['unknown_suppliers']
    rows = [edge('tw.S','tw.A'), dict(type='subsidiary', src=None, dst='tw.P')]
    r = eligible_edges(rows, {'S','A'})
    assert 'S' in r['unknown_suppliers']
