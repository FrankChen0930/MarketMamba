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
import os
import tempfile
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .portfolio import Config, run_backtest
from .rolling_scores import identity, output_path, provenance, write_json

CAVEATS = [
    'Backtest ends at last scheduled entry (terminal liquidation); tail truncated; different rebalance cells cover different periods.',
    'Benchmark also pays transaction costs; excess returns are net-to-net.',
    'Circular block bootstrap on concatenated within-fold returns may cross fold boundaries; no cross-fold NAV return is invented.',
    'survivorship bias: early price termination is observable, delisting cause and full delisted coverage UNVERIFIED',
    'market value uses current shares snapshot, NON_PIT even after lag',
    'missing positions stay marked; no delisting cash recovery invented',
    'zero-volume/missing prior close blocks fills; exchange limit flags unavailable',
    'observed-session calendar, not verified exchange calendar; historical gaps may bridge sessions',
    'simulation labels and historical prices; no executable/trading readiness',
    'costs are research assumptions, not actual broker fees; 2024-2026 descriptive only',
    'gross minus net includes cost-induced allocation differences; capacity uses raw close times volume converted to shares',
]


def volume_evidence(prices):
    sample = prices[prices.code.astype(str).eq('2330') & prices.volume.gt(0)].sort_values('date')
    median = float(sample.volume.median()) if len(sample) else None
    # Broad magnitude check, corroborated by fetcher shares contract. This is
    # engineering unit evidence, not row-level source verification for all stocks.
    unit = 'shares' if median is not None and 1e6 <= median <= 1e9 else 'lots' if median is not None and 1e3 <= median <= 1e5 else 'UNKNOWN'
    return dict(unit=unit, shares_per_volume_unit={'shares':1, 'lots':1000}.get(unit),
                code='2330', rows=len(sample), median_raw_volume=median,
                samples=sample[['date','close','volume']].tail(5).assign(date=lambda x:x.date.astype(str)).to_dict('records'),
                method='2330 median magnitude: shares [1e6,1e9], lots [1e3,1e5]; otherwise UNKNOWN',
                corroboration='V6/marketmamba/data/fetcher.py: TWSE 成交股數; yfinance/TWSE Volume in shares',
                limitation='Magnitude inference, not independent exchange verification or proof of homogeneous units across all rows',
                dollar_volume_unit='TWD = raw Close * raw Volume * shares_per_volume_unit')


def normalize_volume(prices, evidence):
    factor = evidence['shares_per_volume_unit']
    if factor is None:
        raise ValueError('UNKNOWN Volume unit; refuse liquidity/capacity evaluation')
    return prices.assign(volume=prices.volume*factor)


def content_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',',':')).encode()).hexdigest()


def publish_part(path, payload):
    """Serialize before publication; fsync then atomic no-clobber hard link."""
    raw = json.dumps(payload, indent=2, allow_nan=False).encode()+b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.partial-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def checked_cell(prices, predictions, config):
    try:
        result = summary(evaluate_cell(prices, predictions, config))
        # Validate before writing any durable JSON. Nonfinite metrics invalidate
        # the cell rather than silently becoming zero/null and entering inference.
        try:
            json.dumps(result, allow_nan=False)
        except ValueError as error:
            return dict(config=asdict(config), status='ERROR', error=dict(type='NonFiniteResult', message=str(error)))
        return dict(status='COMPUTED', **result)
    except Exception as error:
        return dict(config=asdict(config), status='ERROR', error=dict(type=type(error).__name__, message=str(error)))


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
    valid = prices[np.isfinite(prices.close) & prices.close.gt(0) & np.isfinite(prices.volume) & prices.volume.gt(0)]
    last = valid.groupby('code').date.max()
    end = pd.Timestamp(predictions.date.max())
    codes = []
    for code in sorted(prices.code.unique()):
        day = last.get(code, pd.NaT)
        if pd.isna(day) or day < end:
            codes.append(dict(code=code, last_price_date=None if pd.isna(day) else day.strftime('%Y-%m-%d'),
                **result['position_exposure'].get(code, dict(held_sessions=0, held_value_session_sum=0., blocked_orders=0, blocked_value_sum=0.))))
    result['price_termination'] = dict(fold_end=end.strftime('%Y-%m-%d'), count=len(codes), codes=codes,
        value_unit='initial NAV = 1; held_value_session_sum sums end-of-session marks; blocked_value_sum sums frozen marks at scheduled opens',
        cause='UNKNOWN: termination/missing valid prices is not proof of delisting')
    result['selected_buy_codes'] = sorted({t['code'] for t in result['trades'] if t['notional'] > 0})
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
        'return_days': len(result['daily'])-1,
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
    units = volume_evidence(prices)
    prices = normalize_volume(prices, units)
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
    diagnostics = dict(volume_units=units, missing_prediction_price_keys=int(missing.sum()),
        missing_prediction_price_examples=scores.loc[missing,['code','date']].head(10).astype(str).to_dict('records'),
        nonpositive_or_missing_close=int((~np.isfinite(prices.close) | prices.close.le(0)).sum()),
        zero_or_missing_volume=int((~np.isfinite(prices.volume) | prices.volume.le(0)).sum()),
        missing_open=int((~np.isfinite(prices.open) | prices.open.le(0)).sum()),
        observed_price_rows=len(prices), observed_codes=int(prices.code.nunique()),
        delisted_coverage='UNVERIFIED; no delisting universe metadata; missing held marks remain stale, never fabricate liquidation',
        price_grid_missing_rows=int(prices.date.nunique()*prices.code.nunique()-len(prices)))
    inputs = dict(scores=sid, score_manifest=dict(path=str(mf.resolve()),sha256=hashlib.sha256(payload).hexdigest()), prices=pid, market_value=cid)
    return scores, prices, caps, manifest, inputs, diagnostics


def evaluate(scores, prices, caps, manifest, primary_only=False, bootstrap_samples=1000, part_dir=None, input_identity=None):
    strata = lagged_strata(caps, sorted(prices.date.unique()))
    merged = prices.merge(strata.reset_index()[['date','code','stratum']], on=['date','code'],how='left')
    has_adj = ('adj_close' in prices and (np.isfinite(prices.adj_close/prices.close) & (prices.adj_close/prices.close).gt(0)).all())
    primary = replace(primary_config(), price_basis='total_return' if has_adj else 'raw', bootstrap_samples=bootstrap_samples)
    results, primaries = [], {}
    cells = [primary] if primary_only else [replace(c,bootstrap_samples=bootstrap_samples) for c in configurations()]
    # Content identity covers inputs, manifest, parameters and all implementation
    # files; a stale/corrupt/mismatched part fails closed, never overwritten.
    contract = dict(inputs=input_identity, manifest=manifest, primary_only=primary_only, bootstrap_samples=bootstrap_samples,
                    implementation=provenance()['implementation'],
                    frames=[hashlib.sha256(f.to_parquet(index=False)).hexdigest() for f in (scores,prices,caps)])
    run_hash = content_hash(contract)
    if part_dir is None:
        from .rolling_scores import ARTIFACT_ROOT
        part_dir = ARTIFACT_ROOT / ('evaluate-'+run_hash)
    part_dir = Path(part_dir)
    for (fold,system), group in scores.groupby(['fold','system']):
        part_path = part_dir / f'part-{int(fold)}-{hashlib.sha256(str(system).encode()).hexdigest()[:16]}.json'
        if part_path.exists():
            saved = json.loads(part_path.read_text())
            if saved['run_hash'] != run_hash or saved['payload_sha256'] != content_hash(saved['payload']):
                raise ValueError(f'part identity/hash mismatch: {part_path}')
            payload = saved['payload']
            results.extend(payload['results'])
            if payload['primary'] is not None:
                primaries[(int(fold),system)] = payload['primary']
            print(json.dumps(dict(fold=int(fold),system=system,part=str(part_path),action='SKIP_HASH_MATCH')),flush=True)
            continue
        offset = len(results)
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
            header = dict(fold=int(fold),system=system,role='candidate_confirmatory' if config==primary and fold in (2021,2022,2023) else 'descriptive',stratum='all')
            if config.price_basis=='total_return' and not has_adj:
                results.append(dict(**header,config=asdict(config),status='UNAVAILABLE_NO_ADJ_CLOSE'))
                continue
            result = checked_cell(p,pred,config)
            results.append(dict(**header,**result))
            if config==primary and result['status']=='COMPUTED':
                primaries[(int(fold),system)] = result
        for stratum in ('small','mid','large'):
            pp = p.assign(universe_eligible=p.universe_eligible & p.stratum.eq(stratum))
            # Keep full predictions/calendar; engine masks candidates, preserving
            # real tradability and avoiding invented frozen stratum exits.
            result = checked_cell(pp,pred,primary)
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
            results.append(dict(fold=int(fold),system=system,stratum=stratum,role='descriptive',
                ic=dict(mean=float(np.mean(ics)) if ics else None, sessions=len(ics),label='simulation y5'),**result))
        payload = dict(results=results[offset:], primary=primaries.get((int(fold),system)))
        publish_part(part_path, dict(run_hash=run_hash,payload_sha256=content_hash(payload),payload=payload))
        print(json.dumps(dict(fold=int(fold),system=system,part=str(part_path),action='WROTE',
                             errors=sum(x['status']=='ERROR' for x in payload['results']))), flush=True)
    clean = {}
    for system in sorted(scores.system.unique()):
        rows = [primaries[(year,system)] for year in (2021,2022,2023) if (year,system) in primaries]
        complete = (len(rows)==3 and sorted(manifest['seeds'])==[17,29,43]
                    and manifest['parameters']['max_train_rows']==500_000 and not primary_only
                    and all(x['status'] in ('COMPUTED','UNAVAILABLE_NO_ADJ_CLOSE') for x in results if x['system']==system and x['fold'] in (2021,2022,2023)))
        clean[system] = dict(pooled_test(rows,bootstrap_samples), status='CONFIRMATORY' if complete else 'INCOMPLETE_ENGINEERING_ONLY')
        if not complete:
            clean[system]['decision'] = '未證實（未完成預登記範圍）'
    for row in results:
        if row['role']=='candidate_confirmatory' and clean[row['system']]['status']=='CONFIRMATORY':
            row['role']='confirmatory'
    # Secondary paired descriptive difference at exactly matched dates/folds.
    paired = []
    for fold in sorted(scores.fold.unique()):
        if (fold,'c0plus') in primaries and (fold,'lightgbm_day_t') in primaries:
            a,b = [pd.DataFrame(primaries[(fold,s)]['daily_returns']).set_index('date') for s in ('c0plus','lightgbm_day_t')]
            if not a.index.equals(b.index):
                raise ValueError('unpaired primary return dates')
            paired.extend(dict(fold=int(fold),date=d,net=float(v),benchmark=0.) for d,v in (a.net-b.net).items())
    return dict(primary_config=asdict(primary), grid_complete=not primary_only, results=results,
                clean_fold_hypothesis=clean, paired_system_difference_descriptive={era:pooled_test([dict(daily_returns=[x for x in paired if lo<=x['fold']<=hi])],bootstrap_samples) for era,lo,hi in [('2021_2023',2021,2023),('2024_2026',2024,2026)]},
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
                      **evaluate(scores,prices,caps,manifest,a.primary_only,a.bootstrap_samples,part_dir=out.parent/out.stem,input_identity=inputs))
        if any(x['status']=='ERROR' for x in result['results']):
            result['status']='COMPLETED_WITH_ERRORS_RESEARCH_ONLY'
    result.update(version='p1-portfolio-v1',artifact_class='GENERATED',evidence_class='E2 RESEARCH_ONLY',
        provenance=prov,inputs=inputs,seeds=manifest['seeds'],folds=manifest['years'],
        parameters=dict(primary_only=a.primary_only,bootstrap_samples=a.bootstrap_samples),
        caveats=CAVEATS,
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
