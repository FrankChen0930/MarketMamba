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
from .significance import holm_adjust
from .rolling_scores import identity, output_path, provenance, write_json

CAVEATS = [
    'Backtest ends at last scheduled entry (terminal liquidation); tail truncated; different rebalance cells cover different periods.',
    'Capacity approximation uses NAV=1 nominal notionals, without adjustment for NAV drift.',
    'ERROR cells may reflect transient failures; default resume preserves them. --retry-errors retries only ERROR cells in ERROR parts and retains immutable attempt history.',
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


P2_CAVEATS = [
    'P2 result schema and run_hash differ from P1; old P1 parts cannot resume P2.',
    'Bitwise consistency refers only to default-path daily strategy and benchmark returns.',
    'P2 primary benchmark pays proportional costs only (no min_fee); benchmark with min_fee is descriptive sensitivity, excluded from confirmation.',
    'Frozen holdings causing selected plus frozen count above top_k preserve all incumbent quantities: no additions or forced tradable sales at that rebalance; terminal liquidation unchanged.',
    'Minimum fee=20 TWD is an unverified common assumption; fractional shares, no board/odd-lot modeling.',
    'Capital sizes are descriptive research assumptions, not user assets.',
    'minimum_fee_cost_fraction: full commission on floor-binding orders / total fees+tax+slippage; uplift reported separately.',
    'H-P2b pairs common dates only; different terminal tails explicitly excluded.',
]


def volume_evidence(prices):
    """Measured, conservative magnitude cross-checks, not source certification.

    Fixed reference names are hindsight diagnostic probes, not a PIT universe.
    Ambiguous magnitude or disagreement refuses liquidity/capacity evaluation.
    """
    def infer(value, shares_range, lots_range):
        if value is None:
            return 'UNKNOWN'
        return ('shares' if shares_range[0] <= value <= shares_range[1] else
                'lots' if lots_range[0] <= value <= lots_range[1] else 'UNKNOWN')

    def measurement(frame, field, shares_range, lots_range):
        value = float(frame[field].median()) if len(frame) else None
        segments = []
        for year, group in frame.groupby(frame.date.dt.year):
            median = float(group[field].median())
            segments.append(dict(year=int(year), rows=len(group), median=median,
                                 unit=infer(median, shares_range, lots_range)))
        units = {x['unit'] for x in segments}
        unit = next(iter(units)) if len(units) == 1 else 'UNKNOWN'
        return dict(rows=len(frame), median=value, unit=unit, annual_checks=segments,
                    shares_range=list(shares_range), lots_range=list(lots_range))

    p = prices.assign(code=prices.code.astype(str), date=pd.to_datetime(prices.date))
    valid = p[np.isfinite(p.close) & p.close.gt(0) & np.isfinite(p.volume) & p.volume.gt(0)].copy()
    valid['raw_value'] = valid.close * valid.volume
    checks = []
    references = [('2330', 'TWSE large', 'volume', (1e6,1e9), (1e3,1e5)),
                  ('8069', 'OTC large', 'volume', (1e6,1e9), (1e3,1e5)),
                  ('1236', 'TWSE small/mid', 'raw_value', (1e5,1e9), (1e2,1e4))]
    for code, role, field, shares_range, lots_range in references:
        sample = valid[valid.code.eq(code)].sort_values('date')
        checks.append(dict(code=code, role=role, field=field,
                           **measurement(sample, field, shares_range, lots_range),
                           samples=sample[['date','close','volume']].tail(5).assign(date=lambda x:x.date.astype(str)).to_dict('records')))
    daily = valid.groupby('date', as_index=False).raw_value.sum()
    universe = measurement(daily, 'raw_value', (1e10,5e12), (1e7,5e9))
    universe.update(codes=int(valid.code.nunique()),
                    raw_close_times_volume_quantiles={f'p{int(q*100):02}':float(daily.raw_value.quantile(q)) if len(daily) else None
                                                     for q in (.05,.25,.5,.75,.95)},
                    rationale='Whole-universe daily TWD median below 1e10 is too small for this Taiwan-wide panel in shares; lots imply x1000.')
    units = [x['unit'] for x in checks] + [universe['unit']]
    unit = units[0] if len(set(units)) == 1 else 'UNKNOWN'
    return dict(unit=unit, shares_per_volume_unit={'shares':1,'lots':1000}.get(unit),
                checks=checks, universe_daily=universe,
                corroboration=dict(reference_units={x['code']:x['unit'] for x in checks}, universe_unit=universe['unit'],
                                   consistent=unit != 'UNKNOWN'),
                method='Fixed multi-stock and whole-universe magnitude checks, including annual segments; ambiguity fails closed',
                limitation='Magnitude inference only; missing probes or incomplete universe can stop a valid dataset. Not per-row exchange provenance, homogeneity proof, or PIT admission.',
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



def p2_primary_config():
    return replace(primary_config(), rebalance_sessions=20, hold_buffer_rank=100,
                   commission_discount=.6, min_fee=20., initial_capital=1e6, benchmark_proportional_only=True, preserve_frozen_overflow=True)


def p2_configurations():
    """Amendment 1: one factor at a time plus three registered combinations.

    Buffer is a multiple of top_k; varying top_k preserves the primary 2x
    multiplier. Combinations (a) and (c) duplicate OFAT cells: 16 entries,
    14 unique cells, each evaluated once and reported in deterministic order.
    """
    primary = p2_primary_config()
    candidates = [primary]
    for field, values in (
            ('rebalance_sessions', (10,20,40)),
            ('hold_buffer_rank', (50,75,100)),
            ('top_k', (30,50,100)),
            ('commission_discount', (1.,.6,.28)),
            ('min_fee', (0.,20.)), ('slippage_bps', (10.,25.)),
            ('initial_capital', (1e6,1e7,1e8))):
        for value in values:
            changes = {field:value}
            if field == 'top_k':
                changes['hold_buffer_rank'] = 2*value
            candidates.append(replace(primary, **changes))
    candidates.extend([
        replace(primary, initial_capital=1e7),
        replace(primary, commission_discount=.28, initial_capital=1e7),
        replace(primary, rebalance_sessions=40),
    ])
    yield from dict.fromkeys(candidates)


def p2_hypotheses(primaries, references, manifest, primary_only, samples):
    """Two registered one-sided tests; Holm controls their familywise error.

    P1 and P2 have different terminal dates. Pair only identical within-fold
    return dates, disclose excluded tails, never impute zero or cross-fold NAV.
    """
    rows, paired, support = [], [], []
    for year in (2021,2022,2023):
        key = (year,'c0plus')
        if key not in primaries or key not in references:
            support.append(dict(fold=year, status='MISSING_PRIMARY_OR_REFERENCE'))
            continue
        a, b = primaries[key], references[key]
        rows.append(a)
        aa = {r['date']:r for r in a['daily_returns']}
        bb = {r['date']:r for r in b['daily_returns']}
        common = sorted(aa.keys() & bb.keys())
        paired.append(dict(daily_returns=[dict(date=d,net=aa[d]['net']-bb[d]['net'],benchmark=0.) for d in common]))
        support.append(dict(fold=year, status='PAIRED', days=len(common),
                            excluded_p2_dates=sorted(aa.keys()-bb.keys()), excluded_p1_dates=sorted(bb.keys()-aa.keys())))
    complete = (len(rows)==3 and all(x['days']>0 for x in support if x['status']=='PAIRED')
                and sorted(manifest['seeds'])==[17,29,43]
                and manifest['parameters']['max_train_rows']==500_000 and samples==1000 and not primary_only)
    tests = {}
    for name, data in [('H-P2a',rows),('H-P2b',paired)]:
        result = pooled_test(data,samples)
        values = np.asarray([r['net']-r['benchmark'] for row in data for r in row['daily_returns']])
        if len(values):
            rng = np.random.default_rng(17)
            centered = values-values.mean()
            estimates = []
            for _ in range(samples):
                starts = rng.integers(0,len(values),size=(len(values)+19)//20)
                idx = ((starts[:,None]+np.arange(20)) % len(values)).ravel()[:len(values)]
                estimates.append(float(centered[idx].mean()))
            result['p_one_sided'] = float((1+sum(v>=values.mean() for v in estimates))/(samples+1))
        tests[name] = result
    names = list(tests)
    for name, adjusted in zip(names, holm_adjust([tests[n].get('p_one_sided',1.) for n in names])):
        r = tests[name]
        r['p_holm'] = adjusted
        r['status'] = 'CONFIRMATORY' if complete else 'INCOMPLETE_ENGINEERING_ONLY'
        passed = complete and r.get('ci95',[0])[0]>0 and adjusted<.05
        r['decision'] = ('淨有益' if name=='H-P2a' else '低換手設計有改善') if passed else '未證實'
    return dict(tests=tests, pairing=support, correction='Holm, family=2, alpha=.05',
                p_value_method='one-sided centered circular block bootstrap, plus-one correction')

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
                             trades=len(ratios), unknown=unknown, approximation=True,
                             basis='NAV=1 nominal notionals; no adjustment for NAV drift')
            for asset in (1_000_000,10_000_000,100_000_000)}


def price_termination(prices, predictions, exposure, minimum_sessions=10):
    """Observed close gaps, not verified delistings; zero volume is not no price."""
    history = prices.attrs.get('prior_valid_price_dates', {})
    prices = prices.copy(deep=False)
    prices.attrs = {}  # Do not deepcopy the history map on every group/Series operation.
    start, end = pd.Timestamp(predictions.date.min()), pd.Timestamp(predictions.date.max())
    calendar = pd.DatetimeIndex(sorted(set(prices.date) | set(predictions.date)))
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    groups = {name: dict(count=0, codes=[]) for name in
              ('pre_fold_terminated', 'in_fold_terminated', 'transient_gap')}
    frames = {code:frame for code,frame in prices.groupby('code', sort=True)}
    for code in sorted(set(frames) | set(history)):
        frame = frames.get(code, prices.iloc[:0])
        valid = frame.loc[np.isfinite(frame.close) & frame.close.gt(0) & frame.date.le(end), 'date']
        last = valid.max()
        prior = pd.Timestamp(history[code]) if code in history else pd.NaT
        if pd.notna(prior) and (pd.isna(last) or prior > last):
            last = prior
        missing = calendar.difference(pd.DatetimeIndex(valid))
        if not len(missing):
            continue
        trailing = int((calendar > last).sum()) if pd.notna(last) else len(calendar)
        name = ('pre_fold_terminated' if pd.notna(last) and last < start else
                'in_fold_terminated' if pd.notna(last) and trailing >= minimum_sessions else 'transient_gap')
        row = dict(code=code, last_price_date=None if pd.isna(last) else last.strftime('%Y-%m-%d'),
                   trailing_invalid_sessions=trailing)
        if name == 'in_fold_terminated':
            row.update(exposure.get(code, dict(held_sessions=0, held_value_session_sum=0.,
                                               blocked_orders=0, blocked_value_sum=0.)))
        groups[name]['codes'].append(row)
        groups[name]['count'] += 1
    return dict(fold_start=start.strftime('%Y-%m-%d'), fold_end=end.strftime('%Y-%m-%d'),
                minimum_invalid_sessions=minimum_sessions, **groups,
                **groups['in_fold_terminated'],
                value_unit='initial NAV = 1; held_value_session_sum sums end-of-session marks; blocked_value_sum sums frozen marks at scheduled opens',
                cause='UNKNOWN: missing valid close is not proof of delisting; no prior valid price remains transient_gap/UNKNOWN',
                calendar='union of observed prices and prediction sessions within fold; not verified exchange calendar')


def evaluate_cell(prices, predictions, config):
    execution_prices = prices.copy(deep=False)
    execution_prices.attrs = {}  # History is diagnostic-only, never an execution input.
    result = run_backtest(execution_prices, predictions, config)
    result['capacity'] = capacity(result, execution_prices)
    result['price_termination'] = price_termination(prices, predictions, result['position_exposure'])
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
    prior = prices.loc[prices.date.lt(lo) & np.isfinite(prices.close) & prices.close.gt(0)].groupby('code').date.max()
    prices = prices[prices.date.between(lo,hi)].copy()
    prices.attrs['prior_valid_price_dates'] = {str(code):day.isoformat() for code,day in prior.items()}
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


def evaluate(scores, prices, caps, manifest, primary_only=False, bootstrap_samples=1000, part_dir=None, input_identity=None, retry_errors=False, mode='p1', selection_partial=False):
    if mode not in ('p1','p2'):
        raise ValueError('invalid evaluation mode')
    strata = lagged_strata(caps, sorted(prices.date.unique()))
    merged = prices.merge(strata.reset_index()[['date','code','stratum']], on=['date','code'],how='left')
    input_history = dict(prices.attrs.get('prior_valid_price_dates', {}))
    merged.attrs = {}
    has_adj = ('adj_close' in prices and (np.isfinite(prices.adj_close/prices.close) & (prices.adj_close/prices.close).gt(0)).all())
    primary = replace(primary_config(), price_basis='total_return' if has_adj else 'raw', bootstrap_samples=bootstrap_samples)
    if mode == 'p2':
        primary = replace(p2_primary_config(), bootstrap_samples=bootstrap_samples)
    results, primaries, references = [], {}, {}
    cells = [primary] if primary_only else [replace(c,bootstrap_samples=bootstrap_samples) for c in (p2_configurations() if mode=='p2' else configurations())]
    # Content identity covers inputs, manifest, parameters and all implementation
    # files; a stale/corrupt/mismatched part fails closed, never overwritten.
    contract = dict(inputs=input_identity, manifest=manifest, primary_only=primary_only, bootstrap_samples=bootstrap_samples,
                    implementation=provenance()['implementation'],
                    frames=[hashlib.sha256(f.to_parquet(index=False)).hexdigest() for f in (scores,prices,caps)])
    if mode == 'p2':
        contract.update(mode=mode, source_commit=provenance()['source_commit'],
                        selection_partial=selection_partial,
                        spec_sha256=identity(Path(__file__).parents[2]/'tasks/spec-lowturnover-p2-20261119.md')['sha256'])
    run_hash = content_hash(contract)
    if part_dir is None:
        from .rolling_scores import ARTIFACT_ROOT
        part_dir = (ARTIFACT_ROOT.parent/'p2' if mode=='p2' else ARTIFACT_ROOT) / ('evaluate-'+run_hash)
    part_dir = Path(part_dir)
    for (fold,system), group in scores.groupby(['fold','system']):
        part_path = part_dir / f'part-{int(fold)}-{hashlib.sha256(str(system).encode()).hexdigest()[:16]}.json'
        retry_paths = sorted(part_dir.glob(part_path.stem + '-retry-*.json'))
        if retry_paths and not part_path.exists():
            raise ValueError(f'part identity missing base: {part_path}')
        cached, previous_hash = {}, None
        for attempt, path in enumerate(([part_path] if part_path.exists() else []) + retry_paths):
            if attempt and path.name != f'{part_path.stem}-retry-{attempt:06d}.json':
                raise ValueError(f'part identity retry sequence mismatch: {path}')
            saved = json.loads(path.read_text())
            if ((mode=='p2' and content_hash(saved.get('contract')) != run_hash) or saved['run_hash'] != run_hash or saved['payload_sha256'] != content_hash(saved['payload'])
                    or saved.get('part_key') != dict(fold=int(fold), system=str(system))
                    or saved.get('previous_payload_sha256') != previous_hash
                    or any(row.get('fold') != int(fold) or row.get('system') != system for row in saved['payload']['results'])):
                raise ValueError(f'part identity/hash mismatch: {path}')
            previous_hash = saved['payload_sha256']
        if previous_hash is not None:
            payload = saved['payload']
            if not retry_errors or not any(row['status']=='ERROR' for row in payload['results']):
                results.extend(payload['results'])
                if mode=='p2' and payload.get('p1_reference') is not None:
                    references[(int(fold),system)] = payload['p1_reference']
                if payload['primary'] is not None:
                    primaries[(int(fold),system)] = payload['primary']
                print(json.dumps(dict(fold=int(fold),system=system,part=str(path),action='SKIP_HASH_MATCH')),flush=True)
                continue
            cached = {(row['stratum'], content_hash(row['config'])):row for row in payload['results']}
            part_path = part_path.with_name(f'{part_path.stem}-retry-{len(retry_paths)+1:06d}.json')

        def compute_cell(frame, pred, config, stratum):
            row = cached.get((stratum, content_hash(asdict(config))))
            if row is not None and row['status'] != 'ERROR':
                return {key:value for key,value in row.items() if key not in ('fold','system','role','stratum','ic')}
            return checked_cell(frame, pred, config)

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
        prior = merged.loc[merged.date.lt(lo) & np.isfinite(merged.close) & merged.close.gt(0)].groupby('code').date.max()
        p = merged[merged.date.between(lo,hi)].copy()
        history = dict(input_history)
        history.update({str(code):day.isoformat() for code,day in prior.items()})
        p.attrs['prior_valid_price_dates'] = history
        scored_keys = pd.MultiIndex.from_frame(pred[['date','code']])
        p['universe_eligible'] = pd.MultiIndex.from_frame(p[['date','code']]).isin(scored_keys)
        for config in cells:
            header = dict(fold=int(fold),system=system,role='candidate_confirmatory' if config==primary and fold in (2021,2022,2023) and (mode=='p1' or system=='c0plus') else 'descriptive',stratum='all')
            if config.price_basis=='total_return' and not has_adj:
                results.append(dict(**header,config=asdict(config),status='UNAVAILABLE_NO_ADJ_CLOSE'))
                continue
            result = compute_cell(p,pred,config,'all')
            results.append(dict(**header,**result))
            if config==primary and result['status']=='COMPUTED':
                primaries[(int(fold),system)] = result
        for stratum in (() if mode=='p2' else ('small','mid','large')):
            pp = p.assign(universe_eligible=p.universe_eligible & p.stratum.eq(stratum))
            # Keep full predictions/calendar; engine masks candidates, preserving
            # real tradability and avoiding invented frozen stratum exits.
            result = compute_cell(pp,pred,primary,stratum)
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
        if mode=='p2':
            reference = compute_cell(p,pred,replace(primary_config(),bootstrap_samples=bootstrap_samples),'p1_reference')
            results.append(dict(fold=int(fold),system=system,stratum='p1_reference',role='descriptive',**reference))
            if reference['status']=='COMPUTED':
                references[(int(fold),system)] = reference
        payload = dict(results=results[offset:], primary=primaries.get((int(fold),system)))
        if mode=='p2':
            payload['p1_reference'] = references.get((int(fold),system))
        publish_part(part_path, dict(run_hash=run_hash,part_key=dict(fold=int(fold),system=str(system)),payload_sha256=content_hash(payload),previous_payload_sha256=previous_hash,payload=payload,
                                     **(dict(contract=contract) if mode=='p2' else {})))
        print(json.dumps(dict(fold=int(fold),system=system,part=str(part_path),action='WROTE',
                             errors=sum(x['status']=='ERROR' for x in payload['results']))), flush=True)
    if mode=='p2':
        return dict(primary_config=asdict(primary), grid_complete=not primary_only and not selection_partial and {(int(f),str(y)) for f,y in zip(scores.fold,scores.system)}=={(f,y) for f in range(2021,2027) for y in ('c0plus','lightgbm_day_t')} and {(int(x['fold']),str(x['system'])) for x in results}=={(f,y) for f in range(2021,2027) for y in ('c0plus','lightgbm_day_t')}, results=results,
                    hypotheses=p2_hypotheses(primaries,references,manifest,primary_only or selection_partial,bootstrap_samples),
                    actual_folds=sorted(int(f) for f in scores.fold.unique()), actual_systems=sorted(scores.system.unique()),
                    expected_grid_cells_per_fold_system=len(list(p2_configurations())),
                    expected_full_grid_cells=12*len(list(p2_configurations())), expected_p1_reference_cells=12, price_basis_fallback='raw registered by P2')
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
    paired_report = {}
    for era, lo, hi in [('2021_2023',2021,2023), ('2024_2026',2024,2026)]:
        missing = [dict(fold=year, missing_systems=[system for system in ('c0plus','lightgbm_day_t')
                                                   if (year,system) not in primaries])
                   for year in range(lo,hi+1)]
        missing = [row for row in missing if row['missing_systems']]
        rows = [x for x in paired if lo <= x['fold'] <= hi]
        inference = pooled_test([dict(daily_returns=rows)], bootstrap_samples)
        paired_report[era] = dict(inference,
            status='INCOMPLETE_PAIRED_SUPPORT' if missing else 'COMPLETE_PAIRED_SUPPORT',
            paired_folds=sorted({x['fold'] for x in rows}), missing_primary_folds=missing,
            requested_folds=sorted(int(y) for y in manifest.get('years',scores.fold.unique()) if lo<=y<=hi))
    return dict(primary_config=asdict(primary), grid_complete=not primary_only, results=results,
                clean_fold_hypothesis=clean, paired_system_difference_descriptive=paired_report,
                missing_lagged_market_value_rows=int(merged.stratum.isna().sum()), price_basis_fallback=None if has_adj else 'raw: adj_close unavailable/invalid')


def p3_pair(model, baseline):
    """Pair within-fold common return dates; expose excluded tails."""
    aa={r['date']:r for r in model['daily_returns']}
    bb={r['date']:r for r in baseline['daily_returns']}
    if len(aa)!=len(model['daily_returns']) or len(bb)!=len(baseline['daily_returns']):
        raise ValueError('duplicate paired return dates')
    common=sorted(aa.keys() & bb.keys())
    return dict(daily_returns=[dict(date=d,net=aa[d]['net']-bb[d]['net'],benchmark=0.) for d in common]),dict(
        days=len(common),excluded_model_dates=sorted(aa.keys()-bb.keys()),excluded_c0plus_dates=sorted(bb.keys()-aa.keys()))


def p3_hypotheses(primaries, complete, samples):
    from .significance import paired_ic_test
    tests,support={},[]
    for name,model in [('H-P3a','mamba2'),('H-P3b','crossstock')]:
        pairs=[]
        for year in (2021,2022,2023):
            if (year,model) not in primaries or (year,'c0plus') not in primaries:
                support.append(dict(fold=year,system=model,status='MISSING'))
                continue
            pair,info=p3_pair(primaries[year,model],primaries[year,'c0plus'])
            pairs.append(pair)
            support.append(dict(fold=year,system=model,status='PAIRED',**info))
        values=[r['net'] for row in pairs for r in row['daily_returns']]
        result=pooled_test(pairs,samples)
        if len(values)>=20:
            stat=paired_ic_test(pd.Series(values),pd.Series(np.zeros(len(values))),5,n_boot=samples,block=20)
            result.update(p_boot=stat['p_boot'],ci95=stat['ci95_boot'])
        else: result['p_boot']=1.
        tests[name]=result
    family_complete=complete and samples==1000 and all(x['status']=='PAIRED' and x['days']>=20 for x in support)
    for name,pval in zip(tests,holm_adjust([r['p_boot'] for r in tests.values()])):
        r=tests[name]; r['p_holm']=pval
        r['status']='CONFIRMATORY' if family_complete else 'INCOMPLETE_ENGINEERING_ONLY'
        lo,hi=r.get('ci95',[0.,0.])
        r['decision']=('優於 C0+' if lo>0 else '劣於 C0+') if family_complete and pval<.05 and (lo>0 or hi<0) else '未證實'
    return dict(tests=tests,pairing=support,correction='Holm family=2; two-sided centered block bootstrap; block=20')


def p3_ic_description(scores, samples):
    from .significance import paired_ic_test
    daily={}
    for (fold,system),group in scores.groupby(['fold','system']):
        values={}
        for date,g in group.groupby('date'):
            ok=np.isfinite(g.score)&np.isfinite(g.label_y5)
            if ok.sum()>=10:
                value=g.loc[ok,'score'].rank().corr(g.loc[ok,'label_y5'].rank())
                if np.isfinite(value): values[date]=float(value)
        daily[int(fold),system]=pd.Series(values,dtype=float)
    rows=[]
    for year in sorted(scores.fold.unique()):
        for model in ('mamba2','crossstock'):
            a=daily.get((year,model),pd.Series(dtype=float)); b=daily.get((year,'c0plus'),pd.Series(dtype=float))
            row=dict(fold=int(year),system=model,role='descriptive',model_mean=float(a.mean()) if len(a) else None,
                     c0plus_mean=float(b.mean()) if len(b) else None)
            if len(a.index.intersection(b.index))>=20:
                r=paired_ic_test(a,b,5,n_boot=samples,block=20)
                row['paired']={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items()}
            else: row['status']='INSUFFICIENT_COMMON_SESSIONS'
            rows.append(row)
    return rows


def evaluate_p3(scores, prices, caps, manifest, bootstrap_samples=1000, part_dir=None, input_identity=None):
    """Fixed P2 primary cell, three systems; P1/P2 execution is unchanged."""
    from .rolling_seq_scores import require_same_rows
    systems={'c0plus','mamba2','crossstock'}
    if set(scores.system)!=systems: raise ValueError('P3 requires exactly three systems')
    for year,g in scores.groupby('fold'):
        if set(g.system)!=systems: raise ValueError('P3 incomplete fold system support')
        for model in ('mamba2','crossstock'):
            require_same_rows(g[g.system.eq(model)],g[g.system.eq('c0plus')])
    config=replace(p2_primary_config(),bootstrap_samples=bootstrap_samples)
    contract=dict(mode='p3',manifest=manifest,inputs=input_identity,config=asdict(config),
                  implementation=provenance()['implementation'],
                  spec=identity(Path(__file__).parents[2]/'tasks/spec-mamba-portfolio-p3-20261120.md'),
                  frames=[hashlib.sha256(f.to_parquet(index=False)).hexdigest() for f in (scores,prices,caps)])
    run_hash=content_hash(contract)
    if part_dir is None: part_dir=Path(__file__).parents[2]/'.artifacts/p3'/('portfolio-'+run_hash)
    part_dir=Path(part_dir)
    primaries,rows={},[]
    for (year,system),group in scores.groupby(['fold','system']):
        path=part_dir/f'part-{year}-{system}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['run_hash']!=run_hash or saved['payload_sha256']!=content_hash(saved['payload']):
                raise ValueError('P3 part identity/hash mismatch')
            row=saved['payload']
            if row['fold']!=int(year) or row['system']!=system: raise ValueError('P3 part key mismatch')
        else:
            pred=group.assign(model=system,horizon=5)
            calendar=sorted(prices.date.unique())
            before=[d for d in calendar if d<pred.date.min()]; after=[d for d in calendar if d>pred.date.max()]
            lo=before[-1] if before else pred.date.min(); hi=after[0] if after else pred.date.max()
            p=prices[prices.date.between(lo,hi)].copy()
            p['universe_eligible']=pd.MultiIndex.from_frame(p[['date','code']]).isin(pd.MultiIndex.from_frame(pred[['date','code']]))
            result=summary(evaluate_cell(p,pred,config))
            row=dict(result,fold=int(year),system=system,status='COMPUTED',role='descriptive')
            publish_part(path,dict(run_hash=run_hash,payload_sha256=content_hash(row),payload=row))
        primaries[int(year),system]=row; rows.append(row)
    complete=(set(scores.fold)>={2021,2022,2023} and manifest.get('confirmatory_ready') is True)
    return dict(results=rows,primary_config=asdict(config),grid_complete=set(scores.fold)==set(range(2021,2027)),
                hypotheses=p3_hypotheses(primaries,complete,bootstrap_samples),
                ic_paired_descriptive=p3_ic_description(scores,bootstrap_samples),
                net_excess_descriptive={f'{y}-{m}':pooled_test([r],bootstrap_samples) for (y,m),r in primaries.items()})


def p3_main(a):
    from .rolling_seq_scores import pinned_reference, require_same_rows
    root=(Path(__file__).parents[2]/'.artifacts/p3').resolve(); out=a.out.resolve()
    if out==root or not out.is_relative_to(root): raise ValueError('output must be below .artifacts/p3/')
    if out.exists(): raise FileExistsError(out)
    if a.system or a.retry_errors: raise ValueError('P3 requires all systems; use new namespace after failure')
    if not a.run:
        print(json.dumps(dict(dry_run=True,mode='p3',grid_cells=1,output=str(out)))); return
    scores,prices,caps,manifest,inputs,diagnostics=load_inputs(a.scores_dir,a.data_root)
    reference,p1_manifest,p1_inputs=pinned_reference(a.p1_scores_dir)
    if set(scores.system)!={'mamba2','crossstock'}: raise ValueError('P3 neural score systems mismatch')
    years=sorted(a.fold or scores.fold.unique()); scores=scores[scores.fold.isin(years)]; reference=reference[reference.fold.isin(years)]
    for year in years:
        for model in ('mamba2','crossstock'):
            require_same_rows(scores[(scores.fold==year)&(scores.system==model)],reference[reference.fold==year])
    scores=pd.concat([scores,reference],ignore_index=True)
    prov=provenance()
    combined=dict(neural=manifest,c0plus=p1_manifest,confirmatory_ready=(
        manifest.get('version')=='p3-scores-v1' and manifest.get('parameters',{}).get('smoke') is False
        and sorted(manifest.get('seeds',[]))==[17,29] and manifest.get('contract_status')=='RESOLVED'
        and sorted(p1_manifest.get('seeds',[]))==[17,29,43]
        and p1_manifest.get('parameters',{}).get('max_train_rows')==500_000))
    if diagnostics['missing_prediction_price_keys']: raise ValueError('STOP missing prediction price keys')
    result=evaluate_p3(scores,prices,caps,combined,a.bootstrap_samples,a.out.parent/a.out.stem,dict(**inputs,p1=p1_inputs))
    for value in list(inputs.values())+list(p1_inputs.values()):
        if isinstance(value,dict) and 'path' in value and identity(value['path'])['sha256']!=value['sha256']:
            raise RuntimeError('P3 input changed during run')
    if prov['implementation']!=provenance()['implementation']: raise RuntimeError('P3 source changed during run')
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,dict(version='p3-portfolio-v1',artifact_class='GENERATED',evidence_class='E2 RESEARCH_ONLY',
                        provenance=prov,inputs=dict(**inputs,p1=p1_inputs),caveats=CAVEATS+P2_CAVEATS,
                        diagnostics=diagnostics,**result))
    print(json.dumps(dict(output=str(out),sha256=identity(out)['sha256'],cells=len(result['results']))))



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('p1','p2','p3'), default='p1')
    p.add_argument('--fold', type=int, action='append', help='Explicit partial engineering subset')
    p.add_argument('--system', action='append', help='Explicit partial engineering subset')
    p.add_argument('--scores-dir', type=Path, required=True)
    p.add_argument('--p1-scores-dir', type=Path, default=Path(__file__).parents[2]/'.artifacts/p1/full-scores-ace2eef')
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--primary-only', action='store_true')
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    p.add_argument('--retry-errors', action='store_true', help='Retry ERROR cells in ERROR parts only; preserve original parts and append immutable attempts')
    p.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    if a.mode=='p3':
        return p3_main(a)
    report_path = a.out
    if a.retry_errors and report_path.exists():
        attempt = 1
        while report_path.exists():
            report_path = a.out.with_name(f'{a.out.stem}-retry-{attempt:06d}{a.out.suffix}')
            attempt += 1
    if a.mode=='p2':
        root = Path(__file__).parents[2]/'.artifacts'/'p2'
        out = report_path.resolve()
        if out==root.resolve() or not out.is_relative_to(root.resolve()):
            raise ValueError('output must be below .artifacts/p2/')
        if out.exists() or out.is_symlink():
            raise FileExistsError(f'refusing overwrite: {out}')
    else:
        if a.fold or a.system:
            p.error('--fold/--system require --mode p2')
        out = output_path(report_path)
    if not a.run:
        print(json.dumps(dict(dry_run=True,output=str(out),primary_only=a.primary_only,grid_cells=1 if a.primary_only else (len(list(p2_configurations())) if a.mode=='p2' else 432))))
        return
    started = time.perf_counter()
    prov = provenance()
    scores,prices,caps,manifest,inputs,diagnostics = load_inputs(a.scores_dir,a.data_root)
    if a.mode=='p2':
        if a.fold:
            scores = scores[scores.fold.isin(a.fold)]
        if a.system:
            scores = scores[scores.system.isin(a.system)]
        if scores.empty:
            raise ValueError('empty P2 score selection')
        prov['p2_spec'] = identity(Path(__file__).parents[2]/'tasks/spec-lowturnover-p2-20261119.md')
    if diagnostics['missing_prediction_price_keys']:
        result = dict(status='STOP_MISSING_PRICE_KEYS', diagnostics=diagnostics)
    else:
        result = dict(status='COMPLETED_RESEARCH_ONLY',diagnostics=diagnostics,
                      **evaluate(scores,prices,caps,manifest,a.primary_only,a.bootstrap_samples,part_dir=a.out.parent/a.out.stem,input_identity=inputs,retry_errors=a.retry_errors,mode=a.mode,selection_partial=bool(a.fold or a.system)))
        if any(x['status']=='ERROR' for x in result['results']):
            result['status']='COMPLETED_WITH_ERRORS_RESEARCH_ONLY'
    result.update(version=a.mode+'-portfolio-v1',artifact_class='GENERATED',evidence_class='E2 RESEARCH_ONLY',
        provenance=prov,inputs=inputs,seeds=manifest['seeds'],folds=manifest['years'],
        parameters=dict(mode=a.mode,fold=a.fold,system=a.system,primary_only=a.primary_only,bootstrap_samples=a.bootstrap_samples,retry_errors=a.retry_errors),
        caveats=CAVEATS + (P2_CAVEATS if a.mode=='p2' else []),
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
