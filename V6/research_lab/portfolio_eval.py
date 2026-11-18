"""P1 offline portfolio evaluation. E2 RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Full frozen descriptive grid by default; --primary-only is explicitly incomplete
engineering smoke evidence. No data repairs, acquisition or production access.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from io import BytesIO
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .portfolio import Config, run_backtest
from .rolling_scores import identity, output_path, provenance, write_json

DATA_ROOT = Path('/home/frank/projects/MarketMamba/Data/processed_v6')


def primary_config(**kwargs):
    return Config(**dict(top_k=50, rebalance_sessions=10, min_dollar_volume=2e7,
                          commission=.001425, commission_discount=1., sell_tax=.003,
                          slippage_bps=10., price_basis='raw', **kwargs))


def configurations():
    for k, reb, dv, slip, discount, basis in itertools.product(
            (20,50,100), (5,10,20), (0.,1e7,2e7,5e7), (0.,10.,25.), (1.,.6), ('raw','total_return')):
        yield replace(primary_config(), top_k=k, rebalance_sessions=reb,
                      min_dollar_volume=dv, slippage_bps=slip,
                      commission_discount=discount, price_basis=basis)


def lagged_strata(caps, calendar):
    """Previous global observed session, no forward-fill over missing stock rows.

    At signal close t, use cap at t-1 (conservative one-session lag); snapshot
    shares are still NON_PIT. Code sorting gives deterministic equal-cap breaks.
    """
    c = caps.copy()
    c['date'] = pd.to_datetime(c.date)
    if c.duplicated(['date','code']).any():
        raise ValueError('duplicate market value keys')
    panel = c.pivot(index='date', columns='code', values='market_value').reindex(calendar).shift(1)
    panel = panel.where(np.isfinite(panel) & panel.gt(0))
    ranks = panel.sort_index(axis=1).rank(axis=1, method='first', pct=True)
    out = panel.rename_axis(index='date', columns='code').stack(future_stack=True).to_frame('lagged_market_value')
    rank = ranks.rename_axis(index='date', columns='code').stack(future_stack=True)
    out['stratum'] = pd.cut(rank, [0,1/3,2/3,1], labels=['small','mid','large']).astype(object)
    return out


def capacity(result, prices):
    dollar = prices.assign(dollar=prices.close*prices.volume).set_index(['date','code']).dollar
    signals = {x['entry']: pd.Timestamp(x['signal']) for x in result['schedule']}
    ratios, unknown = [], 0
    for t in result['trades']:
        dv = dollar.get((signals[t['date']], t['code']), np.nan)
        if np.isfinite(dv) and dv > 0:
            ratios.append(abs(t['notional'])/dv)
        else:
            unknown += 1
    return {str(asset): dict(median=float(np.median(ratios)*asset) if ratios else None,
                             p95=float(np.quantile(ratios,.95)*asset) if ratios else None,
                             trades=len(ratios), unknown=unknown)
            for asset in (1_000_000,10_000_000,100_000_000)}


def evaluate_cell(prices, predictions, config):
    result = run_backtest(prices, predictions, config)
    result['capacity'] = capacity(result, prices)
    q, counts = {}, []
    trades = {}
    for t in result['trades']:
        trades.setdefault(t['date'], []).append(t)
    for day in result['daily']:
        for t in trades.get(day['date'], []):
            q[t['code']] = q.get(t['code'],0.) + t['notional']/t['price']
        counts.append(sum(v > 1e-10 for v in q.values()))
    result['holding_count'] = dict(mean=float(np.mean(counts)), minimum=min(counts), maximum=max(counts), terminal=counts[-1])
    result['annual_turnover'] = float(result['metrics']['mean_turnover'])*config.sessions_per_year
    result['blocked_fraction_of_order_attempts'] = result['blocked_orders']/max(1,len(result['trades'])+result['blocked_orders'])
    result['stale_fraction_of_held_sessions'] = result['stale_position_sessions']/sum(counts) if sum(counts) else None
    # Keep original engine hash explicitly scoped to its unextended payload.
    result['engine_sha256'] = result.pop('canonical_sha256')
    return result


def summary(result):
    return {k:v for k,v in result.items() if k not in ('daily','benchmark_daily','trades','metrics','years')} | {
        'max_drawdown': result['metrics']['max_drawdown'],
        'daily_returns': [dict(date=x['date'], net=float(x['nav']/a['nav']-1),
                              benchmark=float(y['nav']/b['nav']-1))
                          for a,x,b,y in zip(result['daily'],result['daily'][1:],result['benchmark_daily'],result['benchmark_daily'][1:])]}


def pooled_test(rows, n_boot=1000, seed=17):
    """Bootstrap concatenated within-fold daily excess, never bridge fold NAVs."""
    values = np.asarray([x['net']-x['benchmark'] for r in rows for x in r['daily_returns']])
    if not len(values):
        return dict(status='UNAVAILABLE')
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        starts = rng.integers(0,len(values),size=(len(values)+19)//20)
        idx = ((starts[:,None]+np.arange(20)) % len(values)).ravel()[:len(values)]
        estimates.append(float(values[idx].mean()))
    lo,hi = np.quantile(estimates,[.025,.975])
    return dict(mean_daily_excess=float(values.mean()), ci95=[float(lo),float(hi)],
                block=20, samples=n_boot, seed=seed, days=len(values),
                decision='淨有益' if lo > 0 else '未證實')


def read_pinned(path):
    content = Path(path).read_bytes()
    frame = pd.read_parquet(BytesIO(content))
    return frame, dict(path=str(Path(path).resolve()), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def load_inputs(scores_dir, data_root):
    mf = scores_dir/'manifest.json'
    payload = mf.read_bytes()
    manifest = json.loads(payload)
    scores, sid = read_pinned(scores_dir/'rank_average.parquet')
    if sid != manifest['outputs']['rank_average.parquet']:
        raise ValueError('score manifest identity mismatch')
    prices, pid = read_pinned(data_root/'prices_raw.parquet')
    caps, cid = read_pinned(data_root/'market_value_raw.parquet')
    prices = prices.rename(columns=dict(stock_id='code',Date='date',Open='open',Close='close',Volume='volume',Adj_Close='adj_close'))
    caps = caps.rename(columns=dict(stock_id='code',Date='date'))
    for frame in (scores,prices,caps):
        frame['date'] = pd.to_datetime(frame.date)
    # Retain the preceding session and next available session for lagging/exit.
    dates = sorted(prices.date.unique())
    before = [d for d in dates if d < scores.date.min()]
    after = [d for d in dates if d > scores.date.max()]
    lo = before[-1] if before else scores.date.min()
    hi = after[0] if after else scores.date.max()
    prices = prices[prices.date.between(lo,hi)].copy()
    caps = caps[caps.date.between(lo,hi)].copy()
    if prices.duplicated(['date','code']).any():
        raise ValueError('duplicate price keys; no silent repair')
    # Missing predictions must never be silently dropped to improve results.
    keys = pd.MultiIndex.from_frame(prices[['date','code']])
    missing = ~pd.MultiIndex.from_frame(scores[['date','code']]).isin(keys)
    diagnostics = dict(missing_prediction_price_keys=int(missing.sum()),
        missing_prediction_price_examples=scores.loc[missing,['code','date']].head(10).astype(str).to_dict('records'),
        nonpositive_or_missing_close=int((~np.isfinite(prices.close) | prices.close.le(0)).sum()),
        zero_or_missing_volume=int((~np.isfinite(prices.volume) | prices.volume.le(0)).sum()),
        missing_open=int((~np.isfinite(prices.open) | prices.open.le(0)).sum()),
        observed_price_rows=len(prices), observed_codes=int(prices.code.nunique()),
        delisted_coverage='UNVERIFIED; no delisting universe metadata; missing held marks remain stale, never fabricate liquidation',
        price_grid_missing_rows=int(prices.date.nunique()*prices.code.nunique()-len(prices)))
    inputs = dict(scores=sid, score_manifest=dict(path=str(mf.resolve()),sha256=hashlib.sha256(payload).hexdigest()), prices=pid, market_value=cid)
    return scores, prices, caps, manifest, inputs, diagnostics


def evaluate(scores, prices, caps, manifest, primary_only=False, bootstrap_samples=1000):
    strata = lagged_strata(caps, sorted(prices.date.unique()))
    merged = prices.merge(strata.reset_index()[['date','code','stratum']], on=['date','code'],how='left')
    has_adj = ('adj_close' in prices and (np.isfinite(prices.adj_close/prices.close) & (prices.adj_close/prices.close).gt(0)).all())
    primary = replace(primary_config(), price_basis='total_return' if has_adj else 'raw', bootstrap_samples=bootstrap_samples)
    results, primaries = [], {}
    cells = [primary] if primary_only else [replace(c,bootstrap_samples=bootstrap_samples) for c in configurations()]
    for (fold,system), group in scores.groupby(['fold','system']):
        pred = group.assign(model=system,horizon=5)
        if pred.duplicated(['date','code']).any():
            raise ValueError('overlapping score rows')
        # Match strategy and benchmark to the same scored universe each signal day.
        # Keep all price rows so an exiting/unscored held name can still sell.
        calendar = sorted(prices.date.unique())
        previous = [d for d in calendar if d < pred.date.min()]
        following = [d for d in calendar if d > pred.date.max()]
        lo,hi = previous[-1] if previous else pred.date.min(), following[0] if following else pred.date.max()
        p = merged[merged.date.between(lo,hi)].copy()
        scored_keys = pd.MultiIndex.from_frame(pred[['date','code']])
        p['universe_eligible'] = pd.MultiIndex.from_frame(p[['date','code']]).isin(scored_keys)
        for config in cells:
            header = dict(fold=int(fold),system=system,role='confirmatory' if config==primary and fold<=2023 else 'descriptive',stratum='all')
            if config.price_basis=='total_return' and not has_adj:
                results.append(dict(**header,config=asdict(config),status='UNAVAILABLE_NO_ADJ_CLOSE'))
                continue
            result = summary(evaluate_cell(p,pred,config))
            results.append(dict(**header,status='COMPUTED',**result))
            if config==primary:
                primaries[(int(fold),system)] = result
        for stratum in ('small','mid','large'):
            pp = p.assign(universe_eligible=p.universe_eligible & p.stratum.eq(stratum))
            # Keep full predictions/calendar; engine masks candidates, preserving
            # real tradability and avoiding invented frozen stratum exits.
            result = summary(evaluate_cell(pp,pred,primary))
            labels = pred.merge(strata.reset_index()[['date','code','stratum']], on=['date','code'],how='left')
            selected = labels[labels.stratum.eq(stratum)]
            ics = []
            if 'label_y5' in selected:
                for _,g in selected.groupby('date'):
                    ok = np.isfinite(g.label_y5) & np.isfinite(g.score)
                    if ok.sum()>=10:
                        val = g.loc[ok,'score'].rank().corr(g.loc[ok,'label_y5'].rank())
                        if np.isfinite(val):
                            ics.append(float(val))
            results.append(dict(fold=int(fold),system=system,stratum=stratum,status='COMPUTED',role='descriptive',
                ic=dict(mean=float(np.mean(ics)) if ics else None, sessions=len(ics),label='simulation y5'),**result))
        print(json.dumps(dict(fold=int(fold),system=system,primary=primaries[(int(fold),system)]['statistics'])), flush=True)
    clean = {}
    for system in sorted(scores.system.unique()):
        rows = [primaries[(year,system)] for year in (2021,2022,2023) if (year,system) in primaries]
        complete = (len(rows)==3 and sorted(manifest['seeds'])==[17,29,43]
                    and manifest['parameters']['max_train_rows']==500_000 and not primary_only)
        clean[system] = dict(status='CONFIRMATORY' if complete else 'INCOMPLETE_ENGINEERING_ONLY',
                             **pooled_test(rows,bootstrap_samples))
        if not complete:
            clean[system]['decision'] = '未證實（未完成預登記範圍）'
    # Secondary paired descriptive difference at exactly matched dates/folds.
    paired = []
    for fold in sorted(scores.fold.unique()):
        if (fold,'c0plus') in primaries and (fold,'lightgbm_day_t') in primaries:
            a,b = [pd.DataFrame(primaries[(fold,s)]['daily_returns']).set_index('date') for s in ('c0plus','lightgbm_day_t')]
            if not a.index.equals(b.index):
                raise ValueError('unpaired primary return dates')
            paired.extend(dict(fold=int(fold),date=d,net=float(v),benchmark=0.) for d,v in (a.net-b.net).items())
    return dict(primary_config=asdict(primary), grid_complete=not primary_only, results=results,
                clean_fold_hypothesis=clean, paired_system_difference_descriptive=pooled_test([dict(daily_returns=paired)],bootstrap_samples),
                missing_lagged_market_value_rows=int(merged.stratum.isna().sum()), price_basis_fallback=None if has_adj else 'raw: adj_close unavailable/invalid')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores-dir', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--primary-only', action='store_true')
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    p.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    out = output_path(a.out)
    if not a.run:
        print(json.dumps(dict(dry_run=True,output=str(out),primary_only=a.primary_only,grid_cells=1 if a.primary_only else 432)))
        return
    started = time.perf_counter()
    prov = provenance()
    scores,prices,caps,manifest,inputs,diagnostics = load_inputs(a.scores_dir,a.data_root)
    if diagnostics['missing_prediction_price_keys']:
        result = dict(status='STOP_MISSING_PRICE_KEYS', diagnostics=diagnostics)
    else:
        result = dict(status='COMPLETED_RESEARCH_ONLY',diagnostics=diagnostics,
                      **evaluate(scores,prices,caps,manifest,a.primary_only,a.bootstrap_samples))
    result.update(version='p1-portfolio-v1',artifact_class='GENERATED',evidence_class='E2 RESEARCH_ONLY',
        provenance=prov,inputs=inputs,seeds=manifest['seeds'],folds=manifest['years'],
        parameters=dict(primary_only=a.primary_only,bootstrap_samples=a.bootstrap_samples),
        caveats=['survivorship bias: delisted coverage unverified', 'market value uses current shares snapshot, NON_PIT even after lag',
                 'missing positions stay marked; no delisting cash recovery invented',
                 'zero-volume/missing prior close blocks fills; exchange limit-up/down flags unavailable in prices_raw',
                 'observed-session calendar, not verified exchange calendar; historical gaps may bridge sessions',
                 'simulation labels and historical prices; no executable/trading readiness',
                 'costs are research assumptions, not actual broker fees; 2024-2026 descriptive only',
                 'gross minus net includes cost-induced allocation differences; capacity is normalized trade notional / prior raw traded value'],
        wall_seconds=time.perf_counter()-started)
    # Verify unchanged pinned inputs and implementation before publication.
    for name in ('prices','market_value','scores'):
        if identity(inputs[name]['path']) != inputs[name]:
            raise RuntimeError('input changed during run')
    if prov['implementation'] != provenance()['implementation']:
        raise RuntimeError('source changed during run')
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,result)
    print(json.dumps(dict(output=str(out),status=result['status'],sha256=identity(out)['sha256'],wall_seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
