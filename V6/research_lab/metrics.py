"""Date-equal ranking diagnostics; overlapping returns are not portfolio P&L."""
import numpy as np
import pandas as pd


def daily_metrics(frame, calendar=None):
    records = []
    for date, original in frame.groupby('date', sort=True):
        g = original.loc[np.isfinite(original.score) & np.isfinite(original.target)].sort_values(['score','code'], kind='stable')
        row = dict(date=date, n=len(g), predicted=len(original), ic=np.nan, spread=np.nan, hit=np.nan)
        if len(g) >= 10:
            k = max(1, len(g)//10)
            row.update(ic=g.score.rank().corr(g.target.rank()),
                       spread=g.target.iloc[-k:].mean()-g.target.iloc[:k].mean(),
                       hit=(g.target.iloc[-k:]>0).mean())
        records.append(row)
    result = pd.DataFrame(records, columns=['date','n','predicted','ic','spread','hit'])
    if calendar is not None:
        result = result.set_index('date').reindex(pd.DatetimeIndex(calendar,name='date')).reset_index()
        result[['n','predicted']] = result[['n','predicted']].fillna(0)
    return result


def summarize(daily, horizon):
    x = daily.ic.to_numpy(float)
    finite = np.isfinite(x)
    n = int(finite.sum())
    known_mean = lambda col: float(daily[col].mean()) if daily[col].notna().any() else None
    result = dict(days=n,sessions=len(daily),ic_mean=None,icir=None,nw_t=None,ic_ci95=None,
                  decile_spread=known_mean('spread'),top_hit=known_mean('hit'),
                  stock_dates=int(daily.n.sum()) if 'n' in daily else None,
                  predicted_stock_dates=int(daily.predicted.sum()) if 'predicted' in daily else None)
    if n == 0:
        return result
    mean = x[finite].mean()
    result['ic_mean'] = float(mean)
    if n < 2:
        return result
    std = x[finite].std(ddof=1)
    # Retain observed-session lags: missing IC contributes no covariance pair,
    # rather than moving distant sessions next to one another. Denominator is
    # valid IC count; SE estimates variance of the finite-observation mean.
    z = np.where(finite,x-mean,0.)
    lag = min(horizon-1,len(x)-1)
    variance_sum = np.dot(z,z)
    for k in range(1,lag+1):
        variance_sum += 2*(1-k/(lag+1))*np.dot(z[k:],z[:-k])
    se = np.sqrt(max(variance_sum,0))/n
    result.update(icir=float(mean/std) if std>0 else None,nw_lags=lag,
                  nw_t=float(mean/se) if se>0 else None,
                  ic_ci95=[float(mean-1.96*se),float(mean+1.96*se)] if se>0 else None)
    return result
