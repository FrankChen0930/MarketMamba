"""Exposed-history diagnostics only. No fitting, acquisition or trading authority.

Reuses Step3's fully validated loader and its exact mature-support seed ranks.
Optional round two uses explicitly authorized local FinMind money/spread fields.
Unknown evidence fails closed; adjusted close*volume is never actual turnover.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tools.step3 import SPEC_SHA, sha256, verify_file
from tools.step3_evaluation import _spearman

SOURCE = Path('/home/frank/projects/MarketMamba')
EVIDENCE = 'EXPOSED_HISTORICAL_DIAGNOSTIC_ONLY / RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED'


def composite(keys, labels, predictions):
    """Exactly rank each seed on COMMON MATURE daily support, then average."""
    out = np.full(len(keys), np.nan)
    for rows in day_groups(keys):
        rows = rows[np.isfinite(labels[rows])]
        if len(rows):
            out[rows] = np.mean([rankdata(p[rows], method='average') / len(rows)
                                 for p in predictions], axis=0)
    return out


def day_groups(keys):
    if not len(keys):
        return []
    if np.any(np.diff(keys[:, 0]) < 0):
        raise ValueError('STOP unsorted sessions')
    return np.split(np.arange(len(keys)), np.flatnonzero(np.diff(keys[:, 0])) + 1)


def daily_ic(keys, score, labels, ndays, mask=None):
    out = np.full(ndays, np.nan)
    valid = np.isfinite(score) & np.isfinite(labels)
    if mask is not None:
        valid &= mask
    for rows in day_groups(keys):
        d = keys[rows[0], 0]
        rows = rows[valid[rows]]
        if len(rows) >= 20:
            out[d] = _spearman(score[rows], labels[rows])
    return out


def quintiles(values, stock):
    """1=smallest; deterministic stock-index ties; missing/nonpositive=0."""
    q = np.zeros(len(values), dtype=np.int8)
    rows = np.flatnonzero(np.isfinite(values) & (values > 0))
    rows = rows[np.lexsort((stock[rows], values[rows]))]
    if len(rows):
        q[rows] = np.arange(len(rows)) * 5 // len(rows) + 1
    return q


def tick_size(price):
    return float(np.select([price < 10, price < 50, price < 100,
                            price < 500, price < 1000], [.01, .05, .1, .5, 1.], default=5.))


def top_decile(score, stock):
    """Positions in top ceil(N/10), ties by ascending stock code index."""
    return np.lexsort((stock, -score))[:(len(score) + 9) // 10]


def limit_locked(open_price, previous_close, entry_date):
    """Conservative requested one-tick proxy, NOT verified executable status.

    Caller must supply admissible previous/reference close; never silently infer
    an exchange reference price across a corporate action. Unknown is blocked.
    """
    if not np.isfinite(open_price) or not np.isfinite(previous_close) or min(open_price, previous_close) <= 0:
        return True
    limit = .07 if np.datetime64(entry_date) < np.datetime64('2015-06-01') else .10
    ceiling = previous_close * (1 + limit)
    return bool(open_price >= ceiling - tick_size(ceiling) - 1e-12)


def turnover(before, after):
    """Buy/sell traded notional / NAV from the supplied weight dictionaries.

    Includes cash initiation/termination; half (buy+sell) is one-way turnover.
    Short-cover is a buy and short-initiation a sell (diagnostic only).
    Round two explicitly supplies equal-weight lists per Human instruction.
    """
    delta = [after.get(k, 0.) - before.get(k, 0.) for k in set(before) | set(after)]
    if not all(np.isfinite(v) for v in delta):
        raise ValueError('STOP unknown weights')
    return sum(max(v, 0.) for v in delta), sum(max(-v, 0.) for v in delta)


def trading_cost(buy, sell, slippage=0.):
    if not all(np.isfinite(v) and v >= 0 for v in (buy, sell, slippage)):
        raise ValueError('STOP invalid cost inputs')
    return buy * (.001425 + slippage) + sell * (.001425 + .003 + slippage)


def prior_amount(amount):
    """Strict t-20..t-1 full-calendar mean; never fill a missing session."""
    return pd.DataFrame(amount).shift(1).rolling(20, min_periods=20).mean().to_numpy()


def admission(mean_amount, money, volume, next_open, previous, next_volume, date, threshold):
    arrays = [np.asarray(v, dtype=float) for v in
              (mean_amount, money, volume, next_open, previous, next_volume)]
    mean_amount, money, volume, next_open, previous, next_volume = arrays
    known = np.logical_and.reduce([np.isfinite(v) for v in arrays])
    known &= (volume > 0) & (previous > 0) & (next_open > 0)
    actual = np.divide(money, volume, out=np.full_like(money, np.nan), where=volume > 0)
    limit = .07 if np.datetime64(date) < np.datetime64('2015-06-01') else .10
    ceiling = previous * (1 + limit)
    tick = np.select([ceiling < 10, ceiling < 50, ceiling < 100, ceiling < 500, ceiling < 1000],
                     [.01, .05, .1, .5, 1.], default=5.)
    locked = next_open >= ceiling - tick - 1e-12
    return known & (mean_amount >= threshold) & (actual >= 10) & (next_volume > 0) & ~locked, known


def cycle_path(entry, closes, weights, cost):
    """Buy-and-hold equal entry notionals, 20 close marks, cost at entry.

    Unknown marks invalidate the whole selected cycle, never drop a loser.
    Weights can be +1 long and -1 short (gross 2, net 0 on NAV 1).
    """
    if (not np.isfinite(entry).all() or not np.isfinite(closes).all()
            or (entry <= 0).any() or (closes <= 0).any()):
        return None
    nav = 1 + (closes / entry - 1) @ weights - cost
    if not np.isfinite(nav).all() or (nav <= 0).any():
        return None
    return nav / np.r_[1., nav[:-1]] - 1


def load_finmind(days, codes, source):
    """Local daily files only. Align to frozen calendar, never compress gaps."""
    fields = ['Trading_money', 'Trading_Volume', 'open', 'close', 'spread']
    mats = {f: np.full((len(days), len(codes)), np.nan) for f in fields}
    identities = []
    # 2014 includes more than the twenty prior sessions for first 2015 signal.
    for d in np.flatnonzero(days >= np.datetime64('2014-01-01')):
        path = source / f'{days[d]}_adj_price.parquet'
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if set(frame.date.astype(str)) != {str(days[d])} or frame.stock_id.duplicated().any():
            raise ValueError(f'STOP FinMind date/duplicate key {path.name}')
        frame = frame.set_index('stock_id').reindex(codes)
        for f in fields:
            mats[f][d] = frame[f].to_numpy(dtype=float)
        identities.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path)))
        if len(identities) % 500 == 0:
            print(f'FinMind validated {len(identities)} daily files', flush=True)
    mats['amount20'] = prior_amount(mats['Trading_money'])
    return mats, identities


def price_matrices(price_path, days, codes):
    frame = pd.read_parquet(price_path, columns=['Date', 'stock_id', 'Open', 'Close'])
    frame['Date'] = pd.to_datetime(frame.Date)
    if frame.duplicated(['Date', 'stock_id']).any():
        raise ValueError('STOP duplicate price keys')
    return {f: frame.pivot(index='Date', columns='stock_id', values=f).reindex(
        index=pd.to_datetime(days), columns=codes).to_numpy(dtype=float) for f in ['Open', 'Close']}


def validate_label_marks(keys, labels, prices, signal_days, full_days):
    if not np.array_equal(signal_days, full_days[:len(signal_days)]):
        raise ValueError('STOP independent label calendar prefix mismatch')
    mature = np.flatnonzero(np.isfinite(labels)); d, s = keys[mature].T
    if not len(mature) or (d+20 >= len(full_days)).any():
        raise ValueError('STOP independent label horizon unavailable')
    reconstructed = prices['Close'][d+20,s] / prices['Open'][d+1,s] - 1
    error = float(np.max(np.abs(reconstructed-labels[mature])))
    if not np.isfinite(reconstructed).all() or error > 1e-9:
        raise ValueError('STOP independent price/horizon label reconstruction mismatch')
    return dict(rows=len(mature),max_absolute_error=error,full_calendar_sessions=len(full_days),
                signal_calendar_sessions=len(signal_days),full_calendar_last=str(full_days[-1]))


def source_facts(mats, prices, days, codes):
    """Four fixed high-dividend stocks, 2024 local cash-ex-date records."""
    path = SOURCE / 'Data/v7_aligned_20260911/processed_v6/dividend_raw.parquet'
    div = pd.read_parquet(path)
    chosen = ['2412', '3045', '4904', '2886']
    div = div[div.stock_id.isin(chosen) & div.CashExDividendTradingDate.astype(str).str.startswith('2024')]
    rows = []
    for code in chosen:
        sub = div[div.stock_id == code]
        if len(sub) != 1:
            raise ValueError('STOP ambiguous 2024 dividend record')
        row = sub.iloc[0]; date = str(row.CashExDividendTradingDate)
        d = int(np.flatnonzero(days == np.datetime64(date))[0]); s = int(np.flatnonzero(codes == code)[0])
        cash = float(row.CashEarningsDistribution + row.CashStatutorySurplus)
        prev_actual = mats['Trading_money'][d-1, s] / mats['Trading_Volume'][d-1, s]
        actual = mats['Trading_money'][d, s] / mats['Trading_Volume'][d, s]
        rows.append([code, date, cash, cash / prev_actual,
                     mats['close'][d-1, s], mats['close'][d, s],
                     mats['close'][d, s] / mats['close'][d-1, s] - 1,
                     prices['Close'][d-1, s], prices['Close'][d, s],
                     prices['Close'][d, s] / prices['Close'][d-1, s] - 1,
                     actual / prev_actual - 1])
    return rows, dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def selected_weights(score, stock, kind):
    order = np.lexsort((stock, -score))
    if kind == 'equal':
        return {int(s): 1 / len(stock) for s in stock}
    n = min(30, len(stock)) if kind == 'top30' else (len(stock) + 9) // 10
    weights = {int(stock[i]): 1 / n for i in order[:n]}
    if kind == 'longshort':
        for i in order[-n:]:
            s = int(stock[i]); weights[s] = weights.get(s, 0.) - 1 / n
    return weights


def simulate_portfolios(keys, inference_scores, universe, prices, days, last_source,
                        kinds=('top10', 'top30', 'longshort', 'equal')):
    """Twenty fixed calendar offsets. All variants share complete-cycle support.

    Selection uses inference ranks only; missing future marks invalidate the
    entire cycle across arms/variants, without substituting securities.
    """
    n = len(days); groups = day_groups(keys); first = int(keys[0, 0])
    variants = [(a, k) for a in 'BA' for k in kinds]
    paths = {(a, k, slip): np.full((20, n), np.nan) for a, k in variants for slip in [0., .001, .003, -1.]}
    buys = {(a, k): np.full((20, n), np.nan) for a, k in variants}
    sells = {(a, k): np.full((20, n), np.nan) for a, k in variants}
    failures = dict(insufficient_universe=0, source_horizon=0, missing_selected_price=0)
    plans = [[] for _ in range(20)]
    for rows in groups:
        d = int(keys[rows[0], 0]); offset = (d - first) % 20
        if d + 20 >= n or days[d + 20] > last_source:
            failures['source_horizon'] += 1; continue
        valid = rows[universe[rows]]
        if len(valid) < 20:
            failures['insufficient_universe'] += 1; continue
        weights = {(a, k): selected_weights(inference_scores[a][valid], keys[valid, 1], k) for a, k in variants}
        valid_marks = True
        for w in weights.values():
            stock = np.array(list(w)); entry = prices['Open'][d+1, stock]
            marks = prices['Close'][d+1:d+21, stock]
            if (not np.isfinite(entry).all() or not np.isfinite(marks).all()
                    or (entry <= 0).any() or (marks <= 0).any()):
                valid_marks = False; break
            if cycle_path(entry, marks, np.array(list(w.values())), 0.) is None:
                raise ValueError('STOP selected portfolio insolvent; never drop a losing cycle')
        if not valid_marks:
            failures['missing_selected_price'] += 1; continue
        plans[offset].append((d, weights))
    for offset, plan in enumerate(plans):
        before = {v: {} for v in variants}; previous_d = None
        for j, (d, weights) in enumerate(plan):
            # Unknown/missing cycles break continuity and liquidate the preceding
            # observed segment; restart from cash, never pretend no trade.
            continuous = previous_d is not None and d == previous_d + 20
            next_continuous = j + 1 < len(plan) and plan[j+1][0] == d + 20
            for a, k in variants:
                w = weights[a, k]; stock = np.array(list(w)); vals = np.array(list(w.values()))
                buy, sell = turnover(before[a, k] if continuous else {}, w)
                end_buy, end_sell = (0., 0.) if next_continuous else turnover(w, {})
                buys[a, k][offset, d+1:d+21] = 0.; sells[a, k][offset, d+1:d+21] = 0.
                buys[a, k][offset, d+1] = buy; sells[a, k][offset, d+1] = sell
                buys[a, k][offset, d+20] += end_buy; sells[a, k][offset, d+20] += end_sell
                for slip in [0., .001, .003, -1.]:
                    cost = trading_cost(buy, sell, slip) if slip >= 0 else 0.
                    path = cycle_path(prices['Open'][d+1, stock], prices['Close'][d+1:d+21, stock], vals, cost)
                    if path is None:
                        raise ValueError('STOP net portfolio insolvent; never drop a losing cycle')
                    end_cost = trading_cost(end_buy, end_sell, slip) if slip >= 0 else 0.
                    # Termination cost is deducted from final NAV, not twice.
                    if end_cost:
                        nav = np.cumprod(1 + path); nav[-1] -= end_cost
                        if nav[-1] <= 0:
                            raise ValueError('STOP nonpositive terminal NAV')
                        path = nav / np.r_[1., nav[:-1]] - 1
                    paths[a, k, slip][offset, d+1:d+21] = path
                before[a, k] = w
            previous_d = d
    return paths, buys, sells, failures


def performance(r, benchmark, buy, sell):
    valid = np.isfinite(r) & np.isfinite(benchmark)
    r = r[valid]; benchmark = benchmark[valid]
    if not len(r):
        return None
    nav = np.cumprod(1 + r); bnav = np.cumprod(1 + benchmark)
    ann = nav[-1] ** (252 / len(r)) - 1
    bann = bnav[-1] ** (252 / len(r)) - 1
    vol = float(np.std(r, ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
    dd = float(np.min(nav / np.maximum.accumulate(np.r_[1., nav])[1:] - 1))
    return [float(ann), float(ann-bann), vol, float(np.mean(r)*252/vol) if vol > 0 else None,
            dd, float(np.sum((buy[valid]+sell[valid])/2)*252/len(r)), len(r)]


def run_finmind(inputs, keys, scores, inference_scores, days, codes, periods, baseline, private_out, report):
    source = SOURCE / 'Data/raw_cache/daily'
    mats, identities = load_finmind(days, codes, source)
    if not identities:
        raise ValueError('STOP no FinMind source')
    last_source = np.datetime64(Path(identities[-1]['path']).name[:10])
    print(f'FinMind complete: {len(identities)} files; aligning frozen price marks', flush=True)
    full_days = np.asarray(np.load(inputs['bundle']['files']['calendar']['path']),dtype='datetime64[D]')
    prices = price_matrices(Path(inputs['bundle']['files']['prices']['path']), full_days, codes)
    label_check = validate_label_marks(keys,inputs['labels'],prices,days,full_days)
    print(f'Independent labels: {label_check}',flush=True)
    facts, dividend_identity = source_facts(mats, prices, days, codes)
    year = days.astype('datetime64[Y]').astype(int) + 1970
    masks = {threshold: np.zeros(len(keys), dtype=bool) for threshold in [10_000_000, 50_000_000]}
    known = np.zeros(len(keys), dtype=bool)
    amount = mats['amount20'][keys[:, 0], keys[:, 1]]
    q = np.zeros(len(keys), dtype=np.int8)
    support = np.zeros((len(days), 5), dtype=int)
    for rows in day_groups(keys):
        d = int(keys[rows[0], 0]); s = keys[rows, 1]
        q[rows] = quintiles(amount[rows], s)
        support[d, 0] = len(rows)
        if d + 1 >= len(days):
            support[d, 1] = len(rows); continue
        for threshold in masks:
            ok, known_rows = admission(amount[rows], mats['Trading_money'][d, s], mats['Trading_Volume'][d, s],
                mats['open'][d+1, s], mats['close'][d+1, s] - mats['spread'][d+1, s],
                mats['Trading_Volume'][d+1, s], days[d+1], threshold)
            masks[threshold][rows] = ok
        known[rows] = known_rows
        support[d, 1:] = [int((~known_rows).sum()), int(masks[10_000_000][rows].sum()),
                          int(masks[50_000_000][rows].sum()), int(np.isfinite(amount[rows]).sum())]
    ic = {(a, threshold): daily_ic(keys, scores[a], inputs['labels'], len(days), mask)
          for a in 'BA' for threshold, mask in masks.items()}
    liq = {(a, g): daily_ic(keys, scores[a], inputs['labels'], len(days), q == g)
           for a in 'BA' for g in range(1, 6)}
    ic_rows = []; support_rows = []; liq_rows = []; concentration_rows = []
    for name, lo, hi in periods:
        sel = (year >= lo) & (year <= hi); active = sel & (support[:, 0] > 0)
        count = int(active.sum()); total = int(support[active, 0].sum())
        support_rows.append([name, count, int(np.count_nonzero(active & (support[:, 1] > 0))),
            int(np.count_nonzero(active & (support[:, 1] == support[:, 0]))),
            int(np.count_nonzero(active & (days >= last_source))), int(support[active, 1].sum()),
            float(support[active, 2].sum()/total) if total else None,
            float(support[active, 3].sum()/total) if total else None])
        for a in 'BA':
            ic_rows.append([name, a, _mean(baseline[a][sel])] +
                [_mean(ic[a, th][sel]) for th in masks] +
                [_mean(baseline[a][sel & np.isfinite(ic[a, th])]) for th in masks] +
                [int(np.isfinite(ic[a, th][sel]).sum()) for th in masks])
            liq_rows.append([name, a] + [_mean(liq[a, g][sel]) for g in range(1, 6)])
            conc = []
            for rows in day_groups(keys):
                if not sel[keys[rows[0], 0]]:
                    continue
                chosen = rows[top_decile(inference_scores[a][rows], keys[rows, 1])]
                conc.append(np.bincount(q[chosen], minlength=6)/len(chosen))
            concentration_rows.append([name, a] + np.mean(conc, axis=0)[[1,2,3,4,5,0]].tolist())
    payload = dict(days=days, support=support,
        **{f'ic_{a}_{th}': v for (a, th), v in ic.items()},
        **{f'amount_ic_{a}_{g}': v for (a, g), v in liq.items()})
    np.savez_compressed(private_out / 'finmind_daily.npz', **payload)
    (private_out / 'finmind_sources.json').write_text(json.dumps(identities, ensure_ascii=False, indent=2)+'\n')
    rows = []; offset_rows = []; breakeven_rows = []; failures = {}; priority = {}; priority_summary = []
    print('Tradable IC complete; priority top10/equal 20-offset portfolios first', flush=True)
    for th, mask in masks.items():
        paths, buys, sells, fail = simulate_portfolios(keys, inference_scores, mask, prices, days, last_source,
                                                      kinds=('top10', 'equal'))
        priority[th] = paths, buys, sells, fail
        np.savez_compressed(private_out / f'priority_top10_{th}.npz', days=days,
                            **{f'r_{a}_{k}_{slip}': v for (a,k,slip),v in paths.items()})
        for name, lo, hi in periods:
            sel = (year >= lo) & (year <= hi)
            for a in 'BA':
                for slip in [0., .001, .003]:
                    m = [performance(paths[a,'top10',slip][off,sel], paths[a,'equal',slip][off,sel],
                                     buys[a,'top10'][off,sel], sells[a,'top10'][off,sel]) for off in range(20)]
                    m = [v for v in m if v is not None]
                    priority_summary.append(dict(threshold=th, period=name, arm=a, slippage=slip,
                        offsets=len(m), annual_return=float(np.mean([v[0] for v in m])) if m else None))
        print(f'Priority top10 threshold {th} complete: {fail}', flush=True)
    (private_out/'priority_top10_summary.json').write_text(json.dumps(priority_summary,ensure_ascii=False,indent=2)+'\n')
    print('Priority saved; extending to top30/longshort on identical benchmark-complete support', flush=True)
    for th, mask in masks.items():
        paths, buys, sells, fail = simulate_portfolios(keys, inference_scores, mask, prices, days, last_source,
                                                      kinds=('top30', 'longshort', 'equal'))
        ppaths, pbuys, psells, pfail = priority[th]
        if fail != pfail or not np.array_equal(paths['B','equal',-1.], ppaths['B','equal',-1.],equal_nan=True):
            raise ValueError('STOP priority/extended portfolio support mismatch')
        paths.update(ppaths); buys.update(pbuys); sells.update(psells)
        failures[str(th)] = fail
        np.savez_compressed(private_out / f'portfolios_{th}.npz', days=days,
            **{f'r_{a}_{k}_{slip}': v for (a,k,slip),v in paths.items()},
            **{f'buy_{a}_{k}': v for (a,k),v in buys.items()},
            **{f'sell_{a}_{k}': v for (a,k),v in sells.items()})
        for name, lo, hi in periods:
            sel = (year >= lo) & (year <= hi)
            for a in 'BA':
                for kind in ['top10','top30','longshort','equal']:
                    for slip in [-1.,0.,.001,.003]:
                        metrics = []
                        for off in range(20):
                            m = performance(paths[a,kind,slip][off,sel], paths[a,'equal',slip][off,sel],
                                            buys[a,kind][off,sel], sells[a,kind][off,sel])
                            if m is not None:
                                metrics.append(m)
                                offset_rows.append([th,name,a,kind,slip,off]+m)
                        if metrics:
                            summary = [float(np.mean([m[i] for m in metrics if m[i] is not None]))
                                       if any(m[i] is not None for m in metrics) else None for i in range(7)]
                            rows.append([th,name,a,kind,'gross' if slip < 0 else slip,len(metrics)]+summary+
                                        [min(m[0] for m in metrics),max(m[0] for m in metrics)])
                    # Cycle-entry attribution, arithmetic profit per initial NAV.
                    # Break-even all-in flat fee per traded side; negative stays negative.
                    profit = notional = fixed = 0.; cycles = 0
                    for off in range(20):
                        for d in range(int(keys[0,0])+off, len(days)-20, 20):
                            if not (lo <= year[d] <= hi):
                                continue
                            r = paths[a,kind,-1.][off,d+1:d+21]
                            if not np.isfinite(r).all():
                                continue
                            b = float(buys[a,kind][off,d+1:d+21].sum())
                            s = float(sells[a,kind][off,d+1:d+21].sum())
                            profit += float(np.prod(1+r)-1); notional += b+s
                            fixed += trading_cost(b,s); cycles += 1
                    breakeven_rows.append([th,name,a,kind,cycles,profit/notional if notional else None,
                                          (profit-fixed)/notional if notional else None])
        print(f'Portfolio threshold {th}: {fail}', flush=True)
    headers = ['均額門檻','期間','臂','投組','每邊滑價','偏移數','年化報酬','年化超額','年化波動','Sharpe RF=0','最大回撤','年化單邊換手','平均有效sessions','偏移年化最低','偏移年化最高']
    pd.DataFrame(rows, columns=headers).to_csv(private_out / 'portfolio_summary.csv', index=False)
    pd.DataFrame(offset_rows, columns=['threshold','period','arm','portfolio','slippage','offset','annual_return','excess','vol','sharpe','drawdown','turnover','sessions']).to_csv(private_out/'portfolio_offsets.csv',index=False)
    pd.DataFrame(breakeven_rows, columns=['threshold','period','arm','portfolio','cycles','all_in_per_side','extra_slippage_per_side']).to_csv(private_out/'breakeven.csv',index=False)
    manifest_path = private_out/'manifest.json'; manifest = json.loads(manifest_path.read_text())
    manifest['skipped'] = []
    source_inventory = sorted(source.glob('*_adj_price.parquet'))
    lineage_path = Path(__file__).resolve().parents[1]/'archive/V5_scripts/fetch_v6_data.py'
    manifest['independent_label_reconstruction'] = label_check
    manifest['completion_status'] = 'COMPUTED_DIAGNOSTIC_GRIDS_WITH_PARTIAL_COVERAGE_STOP'
    manifest['finmind'] = dict(root=str(source.resolve()), first=identities[0]['path'], last=identities[-1]['path'],
        inventory=dict(files=len(source_inventory), first=source_inventory[0].name,last=source_inventory[-1].name),
        historical_assembly=dict(path=str(lineage_path),sha256=sha256(lineage_path),authority='HISTORICAL_LINEAGE_ONLY'),
        files=len(identities), identities='finmind_sources.json', dividend_identity=dividend_identity,
        rules=dict(amount='mean t-20..t-1 complete sessions; NTD', price='t Trading_money/Trading_Volume >=10',
                   entry='t+1 open/(close-spread), one theoretical-ceiling tick tolerance; volume>0',
                   turnover='equal-weight list change, not drifted weights', horizon='t+1 open to t+20 close; 20 fixed offsets'),
        portfolio_excluded_signal_days=failures)
    for p in private_out.iterdir():
        if p.is_file() and p.name != 'manifest.json':
            manifest['outputs'][p.name] = sha256(p)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    text = report.read_text().split('## 預指定未完成項與 STOP')[0]
    original_signal_days = len(day_groups(keys))
    coverage_rows = [[th, original_signal_days, fail['source_horizon'], fail['missing_selected_price'],
                      original_signal_days - sum(fail.values()),
                      (original_signal_days - sum(fail.values())) / original_signal_days]
                     for th, fail in failures.items()]
    summary = ('## 第二輪結論與完整性 STOP\n\n'
        '診斷2的來源／宇宙IC已實算；診斷3–4的前10%／前30／多空／等權及全部成本格已算出，但只是完整價格週期的條件片段，**整段歷史投組完整性STOP／PARTIAL**。'
        '選股不使用未來報酬，但報表因未來價格缺漏而排除整個週期，尤其同宇宙等權基準涉及更多股票，造成嚴重事後支援選擇。'
        '不得把下表淨年化當整段歷史可賺報酬，無法量化真實可賺百分比；不把排除期補零，亦不更換股票或放寬門檻。\n\n' +
        _table(['均額門檻','原signal日','horizon跨來源終點日','選定名單／基準缺未來價格日','完整20-session週期','完整週期比例'], coverage_rows)+'\n')
    text = text.replace('## IC 重現 gate', summary+'## IC 重現 gate', 1)
    text += '\n## 來源事實（第二輪實際核對；更正第一輪描述）\n\n'
    text += f'本機來源存量{len(source_inventory)}個日檔，檔名範圍{source_inventory[0].name[:10]}～{source_inventory[-1].name[:10]}；實際realpath `{source.resolve()}`。本輪唯讀其中2014起{len(identities)}檔，以既有完整日曆逐日對齊，供2015前20 sessions暖身；不把全部來源存量說成本輪全讀。來源欄位 Trading_money 為成交金額（元）、Trading_Volume 為股數；未以還原收盤價×量推算。四檔固定選取中華電2412、台灣大3045、遠傳4904、兆豐金2886，使用本機 dividend_raw 的2024現金除息日／配息金額，未用績效挑選。\n\n'
    text += _table(['股票','除息日','現金股利','股利/前日成交均價','FinMind前日收','FinMind當日收','FinMind收盤變動','快照前日收','快照當日收','快照收盤變動','實際成交均價變動'],facts)
    text += '\n兩來源皆為除權息還原價。[既有組裝程式](../archive/V5_scripts/fetch_v6_data.py)第138／161–162／418–437行明確指定TaiwanStockPriceAdj並由adj_price組裝prices_raw，僅作歷史來源鏈線索，不升格為目前runtime authority。四檔兩來源除息前後報酬近同，價格亦含不合實際交易檔位的數字（例如本表2412的114.09及快照109.8417995763），與實際金額／量均價不同。2412的現金股利約為前日均價3.80%，FinMind收盤變動約−1.86%，實際成交均價約−5.38%，符合股利缺口已調整的方向；其餘三檔照列原數字，除息日仍有市場跌幅，不能籠統寫成「沒有下跌」，也不能單憑兩個收盤報酬推定完整調整。成交均價並非未還原收盤，兩者差額不能精確當股利。第三步標籤使用還原open／close，已反映股利調整；規格「未含股利的原始開收盤價」描述不精確，本輪只在此更正事實、不改規格。這仍非現金股利再投資帳本，不能保證完整公司行動或可實現總報酬。\n\n'
    text += f'獨立以凍結快照及完整凍結日曆重建全部{label_check["rows"]}個成熟標籤close(t+20)/open(t+1)−1，最大絕對誤差={label_check["max_absolute_error"]:.12g}（gate≤1e−9）；先驗證inference日曆為完整日曆前綴，標籤出場可在最後inference日之後，不改既有標籤或用重建值選股。\n\n'
    text += '## 流動性五等分IC與集中度\n\n以t之前20個完整sessions成交金額均值排序，Q1最小、Q5最大；缺任一天為未知、不補值。沿用原共同成熟支援合成分數。\n\n'+_table(['期間','臂','Q1','Q2','Q3','Q4','Q5'],liq_rows)
    text += '\n全inference前10%名單的流動性集中度（最後欄不重分配）：\n\n'+_table(['期間','臂','Q1','Q2','Q3','Q4','Q5','未知'],concentration_rows)
    text += '\n## 可交易代理宇宙IC（診斷2）\n\n門檻固定≥1,000萬／≥5,000萬元；t日實際成交均價≥10元。t+1前收=close−spread；open/前收−1達7%（2015-06-01以前）／10%減理論上限價一檔容差即排除，並要求t+1量>0。來源缺列／無20完整session／必要欄未知即UNKNOWN。此為事後進場代理，日總量及還原價spread無法驗證開盤撮合、公司行動參考價、特殊無漲跌幅證券；strict execution仍STOP。\n\n'
    text += _table(['期間','臂','原宇宙全期IC','≥1,000萬IC','≥5,000萬IC','原IC同1,000萬有效日','原IC同5,000萬有效日','1,000萬有效IC日','5,000萬有效IC日'],ic_rows)
    text += '\n保留率分母為所有原inference keys（含未知）；UNKNOWN影響日與全日排除日分列，不能把部分缺股日說成全部排除。來源尾端欄為signal≥最後來源日，因此t+1必無來源。\n\n'
    text += _table(['期間','原signal日','含UNKNOWN日','全日UNKNOWN排除日','來源尾端排除日','UNKNOWN鍵','1,000萬鍵保留率','5,000萬鍵保留率'],support_rows)
    text += '\n## 投組／成本（診斷3–4）\n\n20個固定起始偏移，相鄰signal相隔20個完整calendar sessions；t+1開盤進、t+20收盤出，跨週期close到次open持現金。前10%為ceil(N/10)、前30為min(30,N)、多空為前後10%各±100%（gross 200%）；同宇宙等權基準為100%多頭。各seed全inference日內秩平均，名次相同依股票代碼，絕不按未來標籤選股。等權進場後持固定股數至該期結束，每日還原收盤mark；換手依Human第二輪指定等權名單差計算，忽略權重漂移，連續週期同名單免換手，這是成本代理而非實際全部出清再買帳本。初始建倉、最後清倉及缺口前後清倉／重啟均計費。\n\n'
    text += '每邊手續費0.1425%，賣出稅0.3%，每邊滑價0／0.1／0.3%；gross完全不扣費。成本從週期NAV扣除，起始費在首日、終止費在末日。缺未來價格不剔除個股或替補，整個20-session週期在所有臂／投組共同排除；來源結束後以及horizon跨來源終點的週期排除。每個偏移獨立算年化（252 sessions）、幾何報酬、同成本基準年化差、樣本標準差波動、RF=0日均/波動Sharpe、每日峰值回撤、單邊換手，再對20個已執行偏移中有支援者等權平均（無支援偏移為N/A、不補0；表列有效偏移數）；附偏移最低／最高。逐年按實際mark日期切片，並非把20重疊路徑當獨立樣本或組成新投組。缺口不補0；有缺口時僅連接有效觀測，風險及年化是有效期條件估計，非完整日曆投資經驗。\n\n'
    text += '優先結果：前10%做多（B主、A並列）；數值均為小數報酬率，換手為倍／年。\n\n'
    text += _table(headers,[r for r in rows if r[3]=='top10'])
    text += '\n其餘預指定投組：\n\n'+_table(headers,[r for r in rows if r[3]!='top10'])
    text += '\n損益兩平每邊成本＝所有完整週期毛損益（各期初NAV=1）總和／買賣notional總和；額外滑價再扣固定手續費及稅後除同分母。這是算術週期損益兩平、非複利CAGR根；依signal年歸屬，和按mark日期的年化表不同。負值照列，代表免費也未損益兩平，不截為0。\n\n'
    text += _table(['均額門檻','signal期間','臂','投組','完整週期數（20偏移合計）','全包每邊成本','固定費稅後可容每邊滑價'],breakeven_rows)
    text += '\n排除signal日數（各門檻的20偏移合計，互斥原因，整週期共同支援）：`'+json.dumps(failures,ensure_ascii=False)+'`。\n\n'
    text += '## 限制與解讀\n\n高IC或歷史淨報酬不等於verified trading evidence；兩段歷史皆曝光，僅描述性，不能推算真實可賺百分比。當日全量、t+1開盤與量過濾含事後資訊，未知逐列排除可造成選樣偏差。無委託簿、撮合、容量、買賣價差、借券可得性／費用、融券限制、PIT首次版本或官方公司行動漲停參考價；多空僅理想化診斷。研究成員／存活者偏差、觀測日曆、來源歷史修訂仍保留；已含股利調整不能再另加股利。未讀blind、無新增擬合／網路／GPU／資料取得，正式Data、V6/models、排程、production與凍結規格均未修改。\n\n'
    text += '## Artifact identity／重播\n\n沿用既有tools/tests/reports與私有tradability樹；新run namespace只隔離第二輪輸出，不覆寫第一輪證據或第三步artifact。\n\n'
    text += f'Source HEAD `{manifest["source_commit"]}`；bundle `{manifest["bundle_sha256"]}`；價格 `{manifest["price"]["sha256"]}`。本輪私有artifact：`{private_out}`，FinMind {len(identities)}個來源檔，來源截至`{last_source}`；逐檔size/SHA-256、來源與程式／模型／環境identity、每日彙總、全部20偏移及CSV見manifest.json與finmind_sources.json。不是正式canonical績效或Human acceptance。\n\n'
    text += '重播：`CUDA_VISIBLE_DEVICES=\'\' /home/frank/.venvs/marketmamba-gpu/bin/python -m tools.tradability_diagnostic --finmind --private-out /home/frank/projects/MarketMamba/.artifacts/tradability/<fresh-run>`。本機CPU，Data工作樹realpath及MARKETMAMBA_DATA_ROOT見manifest，實際來源使用顯式絕對路徑。\n'
    report.write_text(text)
    (private_out/'executed_tradability_diagnostic.py').write_bytes(Path(__file__).read_bytes())
    manifest['outputs']['executed_tradability_diagnostic.py'] = sha256(__file__)
    manifest['test_sha256'] = sha256(Path(__file__).resolve().parents[1]/'tests/test_tradability_diagnostic.py')
    manifest['report'] = dict(path=str(report.resolve()),sha256=sha256(report),bytes=report.stat().st_size)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(finmind_files=len(identities), last_source=str(last_source), ic=ic_rows[:4], summary_rows=len(rows), report=str(report)),ensure_ascii=False),flush=True)


def _mean(x):
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def _table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + ''.join(
        '| ' + ' | '.join('N/A' if v is None else f'{v:.10f}' if isinstance(v, float) else str(v) for v in row) + ' |\n' for row in rows)


def run(bundle_dir, private_out, report, finmind=False):
    from tools.run_step3_evaluation import load_inputs

    if shutil.disk_usage('/mnt/d').free < 60 * 1024**3:
        raise ValueError('STOP /mnt/d free <60GiB')
    root = (SOURCE / '.artifacts/tradability').resolve()
    private_out = private_out.resolve()
    if not private_out.is_relative_to(root) or private_out == root or private_out.exists():
        raise ValueError('STOP use fresh run namespace under .artifacts/tradability')
    print('Validating frozen Step3 inputs/72 COMPLETE outputs', flush=True)
    inputs = load_inputs(bundle_dir, SOURCE / '.artifacts/step3/human-source-gate-20261126.json')
    keys = inputs['label_keys']; y = inputs['labels']; days = inputs['calendar']; bundle = inputs['bundle']
    scores = {a: composite(keys, y, [inputs['predictions'][a, s] for s in (17, 29, 43)]) for a in 'AB'}
    baseline = {a: daily_ic(keys, scores[a], y, len(days)) for a in 'AB'}
    year = days.astype('datetime64[Y]').astype(int) + 1970
    periods = [('2015-2020', 2015, 2020), ('2021-2026', 2021, 2026)] + [(str(v), v, v) for v in range(2015, 2027)]
    reference_path = Path(__file__).resolve().parents[1] / 'reports/step3_results_20261126.json'
    reference = json.loads(reference_path.read_text())
    reproduction = []
    for name, lo, hi in periods[:2]:
        mask = (year >= lo) & (year <= hi)
        for a in 'AB':
            actual = _mean(baseline[a][mask]); expected = reference['periods'][name]['mean_' + a.lower()]
            error = abs(actual - expected)
            if error > 1e-9:
                raise ValueError(f'STOP IC mismatch {name} {a}: {actual} vs {expected}')
            reproduction.append([name, a, expected, actual, error])
    print('IC reproduction passed <=1e-9; reading source schemas', flush=True)
    price_path = Path(bundle['files']['prices']['path'])
    price_manifest = json.loads(Path(bundle['files']['price_manifest']['path']).read_text())
    cap_path = price_path.with_name('market_value_raw.parquet')
    cap_identity = price_manifest['files']['market_value_raw.parquet']
    verify_file(cap_path, cap_identity['sha256'], cap_identity['bytes'])
    import pyarrow.parquet as pq
    schemas = {'prices': pq.read_schema(price_path).names, 'market_value': pq.read_schema(cap_path).names}
    # This frozen source has no turnover amount or official reference close.
    # Stop dependent diagnostics rather than synthesize those fields.
    if set(schemas['prices']) != {'Date', 'stock_id', 'Open', 'High', 'Low', 'Close', 'Volume'}:
        raise ValueError('STOP source schema changed; review amount/reference fields explicitly')
    ids = np.load(bundle['files']['membership']['path'], mmap_mode='r')
    stock = np.load(bundle['files']['stock_index']['path'], mmap_mode='r')
    _, first = np.unique(stock, return_index=True)
    codes = np.asarray(ids[first])
    for start in range(0, len(ids), 100000):
        end = min(start + 100000, len(ids))
        if not np.array_equal(codes[stock[start:end]], ids[start:end]):
            raise ValueError('STOP stock code mapping')
    cap = pd.read_parquet(cap_path)
    cap['Date'] = pd.to_datetime(cap['Date'])
    if cap.duplicated(['Date', 'stock_id']).any():
        raise ValueError('STOP duplicate cap keys')
    lookup = cap.set_index(['Date', 'stock_id'])['market_value']
    row_index = pd.MultiIndex.from_arrays([pd.to_datetime(days[keys[:, 0]]), codes[keys[:, 1]]], names=['Date', 'stock_id'])
    values = lookup.reindex(row_index).to_numpy(dtype=float)
    q = np.zeros(len(keys), dtype=np.int8)
    # Stratify on ALL original inference keys, before mature-label filtering.
    for rows in day_groups(keys):
        q[rows] = quintiles(values[rows], keys[rows, 1])
    stratified = {(a, g): daily_ic(keys, scores[a], y, len(days), q == g) for a in 'AB' for g in range(1, 6)}
    # Concentration uses inference-only rankings, without future label selection.
    inference_scores = {a: composite(keys, np.zeros(len(y)), [inputs['predictions'][a, s] for s in (17, 29, 43)]) for a in 'AB'}
    concentration = {a: np.full((len(days), 6), np.nan) for a in 'AB'}
    coverage = np.full((len(days), 2), np.nan)
    for rows in day_groups(keys):
        d = keys[rows[0], 0]
        coverage[d] = [len(rows), np.count_nonzero(q[rows]) / len(rows)]
        for a in 'AB':
            selected = rows[top_decile(inference_scores[a][rows], keys[rows, 1])]
            concentration[a][d] = np.bincount(q[selected], minlength=6) / len(selected)
    ic_rows = []; conc_rows = []; support_rows = []; stratum_support = []
    for name, lo, hi in periods:
        mask = (year >= lo) & (year <= hi)
        support_rows.append([name, int(np.isfinite(baseline['B'][mask]).sum()), _mean(coverage[mask, 0]), _mean(coverage[mask, 1])])
        for a in 'BA':
            ic_rows.append([name, a, _mean(baseline[a][mask])] + [_mean(stratified[a, g][mask]) for g in range(1, 6)])
            stratum_support.append([name, a] + [int(np.isfinite(stratified[a, g][mask]).sum()) for g in range(1, 6)])
            conc_rows.append([name, a] + [_mean(concentration[a][mask, g]) for g in range(1, 6)] + [_mean(concentration[a][mask, 0])])
    private_out.mkdir(parents=True)
    np.savez_compressed(private_out / 'daily_aggregates.npz', days=days, **{'baseline_' + a: baseline[a] for a in 'AB'},
                        **{f'cap_ic_{a}_{g}': v for (a, g), v in stratified.items()},
                        **{'cap_concentration_' + a: concentration[a] for a in 'AB'}, coverage=coverage)
    manifest = dict(evidence_class=EVIDENCE, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, runtime='local CPU; CUDA unused',
                                     MARKETMAMBA_DATA_ROOT=os.environ.get('MARKETMAMBA_DATA_ROOT'), Data_realpath=str(Path('Data').resolve())),
                    bundle_sha256=inputs['gate']['bundle']['sha256'], spec_sha256=SPEC_SHA,
                    human_source_acceptance=inputs['acceptance'],
                    fold_support=inputs['folds'], seeds=[17, 29, 43],
                    model_manifest_sha256={m['job']: m['sha256'] for m in inputs['manifests']},
                    code_sha256=sha256(__file__), reference_sha256=sha256(reference_path), source_schemas=schemas,
                    cap=dict(path=str(cap_path), **cap_identity), price=bundle['files']['prices'],
                    reproduction=reproduction, skipped=['liquidity IC/concentration: no actual amount', 'tradable universe at 10m/50m: no amount/reference close',
                                                       'all 20-offset portfolios and realized costs: no qualified universe'],
                    outputs={'daily_aggregates.npz': sha256(private_out / 'daily_aggregates.npz')})
    (private_out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    text = '# 可交易性歷史診斷（2026-11-28）\n\n' + EVIDENCE + '\n\n已曝光歷史，只作診斷，不作確認、不新增擬合、不改凍結規格或前瞻家族；完成不等於 Human acceptance。\n\n'
    text += '## IC 重現 gate\n\n三 seed（17/29/43）在每日共同成熟標籤支援內各取平均秩／股數後平均，再與標籤作 Spearman；每日至少20股、兩側非恆定。完整原鍵映射到 frozen full calendar，使用第三步原 loader 驗證72個 COMPLETE 輸出、凍結標籤及身分。所有四格誤差≤1e-9，通過後才做分層。\n\n'
    text += _table(['期間', '臂', '原始 IC', '重現 IC', '絕對誤差'], reproduction)
    text += '\n## 市值分層 IC（診斷1）\n\n同價格快照 manifest 的 market_value_raw.parquet，非第三步新增特徵。以 t 日實際列值，不 forward fill；Q1最小、Q5最大，等值以股票代碼順序決定。每日先在原 inference universe 分五等分，再在各層成熟標籤支援計IC；分層沿用原共同成熟支援合成分數，不重新選 seed。各層不足20股記NaN；各期為有效日IC等權平均。市值單位不影響分層，未宣稱官方／strict PIT。\n\n'
    text += _table(['期間', '臂', '原宇宙 IC', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'], ic_rows)
    text += '\n各層有效IC日數（未成熟／不足20股／恆定不記零）：\n\n' + _table(['期間', '臂', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'], stratum_support)
    text += '\n## 前10%市值集中度（診斷5）\n\n每日全體 inference keys 各 seed 秩平均，降序前 ceil(10%×N)，等值以股票代碼排序，不用未來成熟標籤挑選名單。表為每日占比等權平均；缺市值獨立列示，不重分配。\n\n'
    text += _table(['期間', '臂', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', '市值未知'], conc_rows)
    text += '\n## 支援\n\n' + _table(['期間', 'B有效IC日', '每日平均原宇宙股數', '市值可分層比例'], support_rows)
    text += '\n## 預指定未完成項與 STOP\n\n|診斷|狀態／原因|\n|---|---|\n|1 成交金額五等分IC|SKIPPED：第三步價格快照及原始 processed_v6 僅 Date/stock_id/OHLCV，無實際成交金額；未以 close×volume 猜測。|\n|2 可交易宇宙≥1,000萬及≥5,000萬，含IC與保留率|SKIPPED：成交金額與官方前收／公司行動調整後漲停參考價皆缺；不把未知當可交易，不報0%或替代保留率。|\n|3 前10%／前30／多空、全宇宙等權，20偏移，分期逐年績效|NOT_RUN：無合格可交易宇宙。年化報酬／超額／波動／Sharpe／最大回撤／換手均N/A；未實作完整投組引擎。|\n|4 毛／淨報酬、每邊0／0.1%／0.3%滑價、損益兩平成本|NOT_RUN：依賴3；只有可執行且經合成測試的成本／換手純函式，不能當已完成投組結果。|\n|5 流動性集中度|SKIPPED：無成交金額；市值部分如上。|\n\n'
    text += '成本純函式：買進notional×(0.001425+s)，賣出notional×(0.001425+0.003+s)，不打折；換手輸入需先按價格漂移，計實際買／賣notional，單邊換手=(買+賣)/2。漲停純函式依進場日2015-06-01切7%／10%，以理論上限價所在級距一檔作保守容差；未知一律阻擋。這些函式僅合成驗證，不構成來源或成交證據。\n\n'
    text += '## 限制與解讀\n\n市值分層可描述高IC分布，不能量化真實可獲利部分；沒有可交易宇宙與投組結果，不能推算看得到但賺不到的百分比。原標籤 close(t+20)/open(t+1)−1 未含股利，高殖利率股存在偏差；無借券成本／台股放空受限；等權假設與成交量容量未建模。原價公司行動、PIT、研究成員選樣／存活者偏差、觀測日曆、2026未成熟尾端等第三步限制全保留；t+1日總量>0亦不能證明開盤可成交。未讀 blind、未取得新資料、未改正式Data／V6/models／排程／production。\n\n'
    text += '## Artifact identity／重播\n\n既有 tools/reports/tests 結構可承載本案，未建立新頂層樹；新的私有 tradability namespace 用於隔離本診斷，避免覆寫第三步。\n\n'
    text += f"Source HEAD `{manifest['source_commit']}`；bundle `{manifest['bundle_sha256']}`；價格 `{bundle['files']['prices']['sha256']}`；市值 `{cap_identity['sha256']}`。私有彙總／環境／逐檔hash：`{private_out}`（無逐股公開輸出）。\n\n"
    text += '重播：`python -m tools.tradability_diagnostic --private-out /home/frank/projects/MarketMamba/.artifacts/tradability/<fresh-run>`；每次拒絕覆寫已存在namespace，先檢查/mnt/d≥60GiB。\n'
    text += f'\n來源commit `{manifest["source_commit"]}`；bundle SHA-256 `{manifest["bundle_sha256"]}`；spec SHA-256 `{SPEC_SHA}`。價格SHA-256 `{manifest["price"]["sha256"]}`；市值SHA-256 `{cap_identity["sha256"]}`。\n\n'
    text += f'私有輸出 `{private_out}`：manifest.json及daily_aggregates.npz（只含每日彙總、不含逐股）；後者SHA-256 `{manifest["outputs"]["daily_aggregates.npz"]}`。完整輸入／程式／環境身分見manifest。\n'
    report.write_text(text)
    if finmind:
        run_finmind(inputs, keys, scores, inference_scores, days, codes, periods, baseline, private_out, report)
    print(json.dumps(dict(reproduction=reproduction, report=str(report), private_out=str(private_out)), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-dir', type=Path, default=SOURCE / '.artifacts/step3/bundle-72')
    p.add_argument('--private-out', type=Path, required=True)
    p.add_argument('--report', type=Path, default=Path('reports/tradability_diagnostic_20261128.md'))
    p.add_argument('--finmind', action='store_true', help='Use explicitly authorized local FinMind daily source for round two')
    args = p.parse_args()
    run(args.bundle_dir, args.private_out, args.report, args.finmind)


if __name__ == '__main__':
    main()
