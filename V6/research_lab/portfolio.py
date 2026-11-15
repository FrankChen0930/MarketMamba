"""Offline ranking portfolios. RESEARCH_ONLY; fractional shares and proxy fills.

Signals at close, trades at next observed session open. Terminal schedule entry
liquidates tradable holdings; blocked holdings remain marked and explicitly counted.
No claim of executable fills, PIT prices, or label-equivalent close exits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

from V6.experimental.v7_portfolio_metrics import SessionRecord, compute_performance, canonical_sha256
from V6.experimental.v7_evidence_tier import classify_evidence_tier


@dataclass(frozen=True)
class Config:
    top_k: int = 20
    rebalance_sessions: int = 5
    weighting: str = 'equal'
    weight_cap: float = 1.
    min_dollar_volume: float = 0.
    commission: float = .001425
    commission_discount: float = 1.
    sell_tax: float = .003
    slippage_bps: float = 0.
    sessions_per_year: int = 252
    bootstrap_samples: int = 1000
    bootstrap_block: int = 20
    seed: int = 17
    benchmark_code: str | None = None

    def __post_init__(self):
        for name in ('top_k', 'rebalance_sessions', 'sessions_per_year', 'bootstrap_samples', 'bootstrap_block'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'invalid {name}')
        for name in ('min_dollar_volume', 'commission', 'commission_discount', 'sell_tax', 'slippage_bps', 'weight_cap'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'invalid {name}')
        if not 0 < self.weight_cap <= 1 or not 0 <= self.commission_discount <= 1:
            raise ValueError('invalid cap/discount')
        if self.commission*self.commission_discount + self.sell_tax + self.slippage_bps/10000 >= 1:
            raise ValueError('cost rate must be below one')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('invalid seed')
        if self.weighting not in ('equal', 'score'):
            raise ValueError('invalid weighting')


def transaction_cost(buy, sell, config):
    fee = config.commission * config.commission_discount
    slip = config.slippage_bps / 10000
    return buy*(fee+slip) + sell*(fee+config.sell_tax+slip)


def validate_predictions(frame):
    columns = ['code', 'date', 'fold', 'model', 'horizon', 'score']
    if set(columns) - set(frame.columns) or frame.empty:
        raise ValueError('empty or incomplete prediction contract')
    d = frame[columns].copy()
    if d.isna().any().any() or not d.code.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('null prediction or non-string code')
    d['date'] = pd.to_datetime(d.date, errors='raise')
    if d.date.dt.tz is not None or not d.date.eq(d.date.dt.normalize()).all():
        raise ValueError('prediction dates must be daily and naive')
    for name in ('fold', 'horizon'):
        if not pd.api.types.is_integer_dtype(d[name]) or (d[name] < (1 if name == 'horizon' else 0)).any():
            raise ValueError(f'invalid {name}')
    if (not pd.api.types.is_numeric_dtype(d.score) or pd.api.types.is_bool_dtype(d.score)
            or not d.model.map(lambda x: isinstance(x, str) and bool(x.strip())).all()):
        raise ValueError('score must be numeric; model must be a nonempty string')
    d['score'] = d.score.astype(float)
    if not np.isfinite(d.score.to_numpy(dtype=float)).all() or d.duplicated(['code', 'date']).any():
        raise ValueError('nonfinite scores or overlapping predictions')
    if d.model.nunique() != 1 or d.horizon.nunique() != 1:
        raise ValueError('run each model/horizon separately')
    return d.sort_values(['date', 'code']).reset_index(drop=True)


def _weights(scores, config, benchmark=False):
    if not len(scores):
        return scores
    if benchmark or config.weighting == 'equal':
        w = np.ones(len(scores))
    else:
        # Translation-invariant nonnegative score weights; all ties equal.
        w = scores.to_numpy(float) - scores.min()
        w += max(float(w.max()), 1.) * 1e-12
    cap = 1. if benchmark else config.weight_cap
    result = np.zeros(len(w))
    active = np.ones(len(w), dtype=bool)
    remaining = min(1., len(w)*cap)
    while active.any() and remaining > 1e-14:
        proposed = w[active]/w[active].sum()*remaining
        indices = np.flatnonzero(active)
        capped = proposed > cap
        if not capped.any():
            result[indices] = proposed
            break
        result[indices[capped]] = cap
        active[indices[capped]] = False
        remaining -= cap * capped.sum()
    return pd.Series(result, index=scores.index)


def _simulate(panel, days, signals, predictions, config, benchmark=False, free=False):
    codes = panel['open'].columns
    q = pd.Series(0., index=codes)
    marks = pd.Series(np.nan, index=codes)
    cash, cost = 1., 0.
    rows, trades, records = [], [], []
    blocked = stale = 0
    schedule = {entry: signal for signal, entry in signals}
    start, end = signals[0][0], signals[-1][1]
    groups = {d: x.set_index('code').score for d, x in predictions.groupby('date')}
    for day in days[(days >= start) & (days <= end)]:
        buy = sell = 0.
        filled = blocked_today = 0
        terminal = day == end
        if day in schedule:
            signal = schedule[day]
            op = panel['open'].loc[day]
            prior_ok = panel['valid'].loc[signal]
            can_trade = op.gt(0) & np.isfinite(op) & prior_ok
            marks.loc[can_trade] = op[can_trade]
            held = q.gt(0)
            if (held & marks.isna()).any():
                raise ValueError('held position without mark')
            nav = cash + (q*marks.fillna(0)).sum()
            frozen = held & ~can_trade
            blocked_today = int(frozen.sum())
            if terminal:
                weights = pd.Series(dtype=float)
            else:
                liquid = prior_ok & (panel['close'].loc[signal]*panel['volume'].loc[signal]).ge(config.min_dollar_volume)
                if benchmark:
                    selected = codes[liquid]
                    if config.benchmark_code is not None:
                        selected = selected[selected == config.benchmark_code]
                    scores = pd.Series(1., index=selected)
                else:
                    scores = groups.get(signal, pd.Series(dtype=float))
                    scores = scores.reindex(codes[liquid]).dropna().sort_index().sort_values(ascending=False, kind='stable').iloc[:config.top_k]
                # No replacement by lower-ranked names when next open is missing.
                weights = _weights(scores, config, benchmark).reindex(codes, fill_value=0.)
                weights = weights.where(can_trade, 0.)
            frozen_value = (q[frozen]*marks[frozen]).sum()
            old = q*marks.fillna(0.)

            old_array = old.to_numpy()
            frozen_array = frozen.to_numpy()
            trade_array = can_trade.to_numpy()
            weight_array = weights.reindex(codes, fill_value=0.).to_numpy()

            def orders(net):
                target = np.where(frozen_array, old_array, 0.)
                target += max(0., net-frozen_value)*weight_array
                delta = np.where(trade_array, target-old_array, 0.)
                return target, delta, np.maximum(delta, 0).sum(), -np.minimum(delta, 0).sum()

            lo, hi = frozen_value, nav
            for _ in range(60):
                mid = (lo+hi)/2
                _, _, b, s = orders(mid)
                fees = 0. if free else transaction_cost(b, s, config)
                if mid+fees > nav:
                    hi = mid
                else:
                    lo = mid
            # Stay on the feasible side and reserve a few ULPs of cash. This
            # prevents roundoff from financing fractional buys with negative cash.
            principal = max(frozen_value, lo - 32*np.spacing(float(nav)))
            target, delta, buy, sell = orders(principal)
            target = pd.Series(target, index=codes)
            delta = pd.Series(delta, index=codes)
            fees = 0. if free else transaction_cost(buy, sell, config)
            cost += fees
            q.loc[can_trade] = target[can_trade]/op[can_trade]
            cash = float(nav - (q*marks.fillna(0.)).sum() - fees)
            for code in codes[delta.abs() > 1e-13]:
                q[code] = target[code]/op[code]
                trades.append(dict(date=day.strftime('%Y-%m-%d'), signal=signal.strftime('%Y-%m-%d'), code=code,
                                   notional=float(delta[code]), price=float(op[code])))
                filled += 1
            blocked += blocked_today
        # Daily close marks; terminal is OPEN, with no terminal close lookahead.
        if not terminal:
            close = panel['close'].loc[day]
            good = panel['valid'].loc[day]
            marks.loc[good] = close[good]
            stale += int((q.gt(0) & ~good).sum())
        nav = float(cash + (q*marks.fillna(0)).sum())
        if not np.isfinite(nav) or not np.isfinite(cash) or nav <= 0 or cash < -1e-9:
            raise ValueError(f'insolvent simulation: date={day}, nav={nav}, cash={cash}')
        stamp = day.tz_localize('Asia/Taipei') + pd.Timedelta(hours=9 if terminal else 13, minutes=0 if terminal else 30)
        records.append(SessionRecord(stamp.to_pydatetime(), Decimal(str(nav)), Decimal(str(cost)),
                                     Decimal(str(buy)), Decimal(str(sell)),
                                     (('FILLED', filled), ('PARTIAL', 0), ('BLOCKED', blocked_today))))
        rows.append(dict(date=day.strftime('%Y-%m-%d'), nav=nav, cash=float(cash), cumulative_cost=float(cost)))
    return records, rows, trades, dict(blocked_orders=blocked, stale_position_sessions=stale,
                                      unresolved_positions=int(q.gt(1e-12).sum()))


def _statistics(values, benchmark, config):
    r = values[1:]/values[:-1]-1
    b = benchmark[1:]/benchmark[:-1]-1
    vol = float(np.std(r, ddof=1)*np.sqrt(config.sessions_per_year)) if len(r)>1 else None
    active = r-b
    tracking = float(np.std(active, ddof=1)*np.sqrt(config.sessions_per_year)) if len(r)>1 else None
    return dict(total_return=float(values[-1]/values[0]-1), benchmark_return=float(benchmark[-1]/benchmark[0]-1),
                annualized_return=float((values[-1]/values[0])**(config.sessions_per_year/len(r))-1),
                annualized_volatility=vol, sharpe_like=float(np.mean(r)*config.sessions_per_year/vol) if vol else None,
                hit_rate=float(np.mean(r>0)), tracking_error=tracking,
                information_ratio=float(np.mean(active)*config.sessions_per_year/tracking) if tracking else None,
                mean_daily_excess=float(active.mean()), sessions=len(r))


def run_backtest(prices, predictions, config=Config()):
    pred = validate_predictions(predictions)
    p = prices.copy()
    required = {'code', 'date', 'open', 'close', 'volume'}
    if required-set(p) or p.empty or not p.code.map(lambda x: isinstance(x, str)).all():
        raise ValueError('invalid prices')
    p['date'] = pd.to_datetime(p.date)
    if p.duplicated(['code', 'date']).any() or p.date.isna().any():
        raise ValueError('duplicate/missing price keys')
    if p.date.dt.tz is not None or not p.date.eq(p.date.dt.normalize()).all():
        raise ValueError('invalid price dates')
    if set(pred.code)-set(p.code) or set(pred.date)-set(p.date):
        raise ValueError('predictions outside price keys/calendar')
    p['valid'] = np.isfinite(p.close) & p.close.gt(0) & np.isfinite(p.volume) & p.volume.gt(0)
    for flag in ('outlier_flag', 'split_or_outlier_flag'):
        if flag in p:
            p['valid'] &= ~p[flag].fillna(True)
    days = pd.DatetimeIndex(sorted(p.date.unique()))
    panel = {name: p.pivot(index='date', columns='code', values=name).reindex(days) for name in ('open', 'close', 'volume', 'valid')}
    panel['valid'] = panel['valid'].fillna(False).astype(bool)
    first = days.get_loc(pred.date.min())
    last = days.get_loc(pred.date.max())
    indices = list(range(first, min(last+1, len(days)-1), config.rebalance_sessions))
    if len(indices)<2:
        raise ValueError('need at least two schedule entries including terminal exit')
    signals = [(days[i], days[i+1]) for i in indices]
    if config.benchmark_code is not None and config.benchmark_code not in p.code.unique():
        raise ValueError('unknown benchmark code')
    rec, daily, trades, diagnostics = _simulate(panel, days, signals, pred, config)
    bench, bdaily, _, bdiag = _simulate(panel, days, signals, pred, config, benchmark=True)
    _, gdaily, _, _ = _simulate(panel, days, signals, pred, config, free=True)
    metrics = compute_performance(rec, {x.occurred_at: x.net_value for x in bench}, evidence_class='RESEARCH_ONLY')
    values = np.array([x['nav'] for x in daily])
    bv = np.array([x['nav'] for x in bdaily])
    r, b = values[1:]/values[:-1]-1, bv[1:]/bv[:-1]-1
    rng = np.random.default_rng(config.seed)
    # Paired circular moving blocks preserve within-block serial dependence.
    estimates = []
    block = min(config.bootstrap_block, len(r))
    for _ in range(config.bootstrap_samples):
        starts = rng.integers(0, len(r), size=(len(r)+block-1)//block)
        idx = ((starts[:, None]+np.arange(block)) % len(r)).ravel()[:len(r)]
        estimates.append(float(np.prod(1+r[idx])-np.prod(1+b[idx])))
    years = {}
    for year in sorted({x['date'][:4] for x in daily[1:]}):
        idx = [i for i in range(1, len(daily)) if daily[i]['date'].startswith(year)]
        take = [idx[0]-1]+idx
        years[year] = _statistics(values[take], bv[take], config)
    result = dict(version='research-portfolio-v1', evidence_class='RESEARCH_ONLY',
                  evidence_tier=classify_evidence_tier(evidence_class='RESEARCH_ONLY', baseline_compared=True),
                  config=asdict(config), config_sha256=canonical_sha256(asdict(config)),
                  model=str(pred.model.iloc[0]), horizon=int(pred.horizon.iloc[0]),
                  start=daily[0]['date'], end=daily[-1]['date'],
                  schedule=[dict(signal=s.strftime('%Y-%m-%d'), entry=e.strftime('%Y-%m-%d')) for s,e in signals],
                  metrics=metrics, statistics=_statistics(values, bv, config), years=years,
                  bootstrap_excess_ci95=[float(x) for x in np.quantile(estimates, [.025, .975])],
                  gross_minus_net_return=float(gdaily[-1]['nav']-daily[-1]['nav']),
                  daily=daily, benchmark_daily=bdaily, trades=trades,
                  benchmark_diagnostics=bdiag, **diagnostics)
    result['canonical_sha256'] = canonical_sha256(result)
    return result
