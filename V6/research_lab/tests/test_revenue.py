import numpy as np, pandas as pd
from research_lab import revenue as rv


def rev_df():
    rows = []
    for sid, base in (('1101', 100.), ('2330', 1000.)):
        for k in range(30):                                  # 2023-01 .. 2025-06
            p = pd.Period('2023-01', 'M')+k
            rows.append((sid, p.year, p.month, base*(1.01**k)*(2. if k >= 12 else 1.)))
    r = pd.DataFrame(rows, columns=['stock_id', 'revenue_year', 'revenue_month', 'revenue'])
    r.loc[(r.stock_id == '1101') & (r.revenue_year == 2024) & (r.revenue_month == 5), 'revenue'] = 0.   # non-positive -> missing
    return r


def test_availability_rule_day_10_vs_11():
    d = pd.DatetimeIndex(['2025-04-10', '2025-04-11', '2025-04-30', '2025-05-01', '2025-05-10', '2025-05-11'])
    assert [str(p) for p in rv.latest_period(d)] == ['2025-02', '2025-03', '2025-03', '2025-03', '2025-03', '2025-04']


def test_growth_values_and_missing_propagation():
    periods, panel, ids = rv.monthly_panel(rev_df())
    g = rv.growth_panels(panel)
    i, j = list(periods).index(pd.Period('2024-06', 'M')), ids.index('2330')
    assert abs(g['rev_yoy'][i, j]-(2*1.01**12-1)) < 1e-9      # 2024-06 (k=17, doubled) vs 2023-06 (k=5, not doubled)
    assert abs(g['rev_mom'][i, j]-.01) < 1e-9
    a = list(periods).index(pd.Period('2024-05', 'M')); b = ids.index('1101')
    assert np.isnan(g['rev_yoy'][a, b]) and np.isnan(g['rev_mom'][a+1, b]) and np.isnan(g['rev_yoy3'][a+1, b])
    assert not np.isnan(g['rev_yoy3'][a+3, b])               # three months after the gap the 3-month window is complete again


def test_extra_uses_only_released_month(tmp_path):
    class D:                                                  # minimal stand-in for SeqData
        date_idx = np.array([0, 1, 1]); stock_idx = np.array([0, 0, 1])
    dates = pd.DatetimeIndex(['2025-04-10', '2025-04-11'])
    f = rv.make_extra(rev_df(), dates, ['1101', '2330'])
    out, names = f(D, np.arange(3))
    assert names == rv.NAMES
    p = rev_df(); v = lambda sid, per: float(p[(p.stock_id == sid) & (p.revenue_year == per.year) & (p.revenue_month == per.month)].revenue.iloc[0])
    feb, mar = pd.Period('2025-02', 'M'), pd.Period('2025-03', 'M')
    assert abs(out[0, 1]-(v('1101', feb)/v('1101', feb-1)-1)) < 1e-6                         # Apr-10 sees February
    assert abs(out[1, 1]-(v('1101', mar)/v('1101', mar-1)-1)) < 1e-6                         # Apr-11 sees March
    assert abs(out[2, 0]-(v('2330', mar)/v('2330', mar-12)-1)) < 1e-6
    f2 = rv.make_extra(rev_df(), pd.DatetimeIndex(['2025-09-15', '2025-09-16']), ['1101', '2330'])   # after the data ends -> NaN, not stale
    assert np.isnan(f2(D, np.arange(3))[0]).all()


def test_year_boundary_january_sees_prior_december_only_after_the_10th():
    d = pd.DatetimeIndex(['2025-01-10', '2025-01-11', '2025-12-31'])
    assert [str(p) for p in rv.latest_period(d)] == ['2024-11', '2024-12', '2025-11']
