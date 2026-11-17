from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_lab import fundamentals as f


def fixture():
    dates = pd.bdate_range('2024-01-01', '2026-11-20')
    ids = ['A', 'B', 'C', 'F', 'N']
    t, s = np.indices((len(dates), len(ids)))
    keep = (s != 4) | (t >= len(dates)-100)
    data = SimpleNamespace(date_idx=t[keep], stock_idx=s[keep])
    rows = []
    for q in pd.period_range('2023Q1', '2026Q3', freq='Q'):
        for i, sid in enumerate(ids):
            for name, value in dict(Revenue=100, GrossProfit=10+i*10,
                                    OperatingIncome=5+i, EPS=(q.ordinal-200)*(i+1)).items():
                rows.append((sid, q.end_time.normalize(), name, float(value)))
    fin = pd.DataFrame(rows, columns=['stock_id', 'Date', 'type', 'value'])
    info = pd.DataFrame({'stock_id': ids, 'industry_category': ['水泥', '', '', '金融保險業', '電子']})
    return fin, info, dates, ids, data


def output(fin=None, shift=0):
    base, info, dates, ids, data = fixture()
    fn = f.make_extra(base if fin is None else fin, info, dates, ids, shift)
    a, names = fn(data, np.arange(len(data.date_idx)))
    assert names == ['GM', 'OM', 'EPS_YoY'] and a.dtype == np.float32
    return a, dates[data.date_idx], data.stock_idx, fn, data


def test_deadlines_and_weekend():
    assert list(f.availability_dates(pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'])).strftime('%Y-%m-%d')) == ['2024-05-15', '2024-08-14', '2024-11-14', '2025-03-31']
    assert list(f.latest_period(pd.to_datetime(['2025-03-31', '2025-04-01', '2022-05-13', '2022-05-16'])).astype(str)) == ['2024Q3', '2024Q4', '2021Q4', '2022Q1']


def test_boundary_no_fallback_shift_and_no_leakage():
    fin, *_ = fixture()
    missing = fin[~((fin.stock_id == 'A') & (fin.Date == '2026-09-30'))]
    a, d, s, _, _ = output(missing)
    assert np.isfinite(a[(d == '2026-11-13') & (s == 0)]).all()
    assert np.isnan(a[(d >= '2026-11-16') & (s == 0)]).all()
    # A missing entire next quarter makes the strict deadline boundary observable.
    missing = fin[fin.Date != '2025-03-31']
    a, d, s, _, _ = output(missing)
    assert np.isfinite(a[d == '2025-05-15']).any()
    assert np.isnan(a[d == '2025-05-16']).all()
    shifted = output(missing, 30)[0]
    assert not np.array_equal(a, shifted, equal_nan=True)
    altered = fin.copy()
    altered.loc[(altered.stock_id == 'A') & (altered.Date == '2025-03-31'), 'value'] = [1, 1e10, -1e10, 1e10]
    original, d, *_ = output()
    extreme = output(altered)[0]
    np.testing.assert_equal(original[d <= '2025-05-15'], extreme[d <= '2025-05-15'])
    assert not np.array_equal(original[d == '2025-05-16'], extreme[d == '2025-05-16'], equal_nan=True)


def test_exclusions_rank_determinism_and_subset():
    a, d, s, fn, data = output()
    assert np.isnan(a[s == 3]).all() and np.isnan(a[s == 4]).all()
    first = np.unique(d[s == 0])[:252]
    assert np.isnan(a[np.isin(d, first)]).all()
    sample = a[d == '2025-05-16'][:3]
    np.testing.assert_allclose(sample, np.tile(np.array([1/3, 2/3, 1])[:, None]-.5, (1, 3)), atol=1e-7)
    np.testing.assert_equal(a, fn(data, np.arange(len(s)))[0])
    pick = np.arange(0, len(s), 7)
    np.testing.assert_equal(a[pick], fn(data, pick)[0])


@pytest.mark.parametrize('value', [0, -1, np.nan])
def test_invalid_revenue(value):
    fin, *_ = fixture()
    fin.loc[(fin.stock_id == 'A') & (fin.Date == '2025-03-31') & (fin.type == 'Revenue'), 'value'] = value
    a, d, s, *_ = output(fin)
    assert np.isnan(a[(d == '2025-05-16') & (s == 0)]).all()


def test_eps_requires_exact_prior_year_quarter():
    fin, *_ = fixture()
    fin = fin[~((fin.stock_id == 'A') & (fin.Date == '2024-03-31') & (fin.type == 'EPS'))]
    a, d, s, *_ = output(fin)
    row = a[(d == '2025-05-16') & (s == 0)][0]
    assert np.isfinite(row[:2]).all() and np.isnan(row[2])


@pytest.mark.parametrize('industry', ['金融保險', '金融業'])
def test_financial_industry_variants(industry):
    fin, info, dates, ids, data = fixture()
    info.loc[info.stock_id == 'A', 'industry_category'] = industry
    a, _ = f.make_extra(fin, info, dates, ids)(data, np.arange(len(data.date_idx)))
    assert np.isnan(a[data.stock_idx == 0]).all()


def test_first_year_boundary_ties_and_eps_difference():
    fin, info, dates, ids, data = fixture()
    fn = f.make_extra(fin, info, dates, ids)
    a, _ = fn(data, np.arange(len(data.date_idx)))
    assert np.isnan(a[(data.date_idx == 251) & (data.stock_idx == 0)]).all()
    assert np.isfinite(a[(data.date_idx == 252) & (data.stock_idx == 0)]).all()
    # Difference (A:2, B:5) and ratio (A:2, B:1.05) imply opposite ranks.
    for sid, old, new in [('A', 1., 3.), ('B', 100., 105.), ('C', 5., 5.)]:
        fin.loc[(fin.stock_id == sid) & (fin.Date == '2024-03-31') & (fin.type == 'EPS'), 'value'] = old
        fin.loc[(fin.stock_id == sid) & (fin.Date == '2025-03-31') & (fin.type == 'EPS'), 'value'] = new
    fin.loc[(fin.Date == '2025-03-31') & fin.stock_id.isin(['A', 'B']) & (fin.type == 'GrossProfit'), 'value'] = 10.
    a, d, s, *_ = output(fin)
    rows = a[d == '2025-05-16'][:3]
    np.testing.assert_allclose(rows[:, 0], [0., 0., .5])
    np.testing.assert_allclose(rows[:, 2], [1/6, .5, -1/6], atol=1e-7)
