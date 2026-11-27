"""Step6 H-0/E-PM pure feature helpers; no fitting or historical matrices.

Mechanical relation filtering does not certify security identity, source PIT,
or group completeness. Callers must keep those admission gates separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd

from tools import step3

FEATURES = ['customer_revenue_yoy_median', 'customer_revenue_surprise_mean',
            'customer_revenue_valid_count']
GROUP_CLUE = re.compile(r'子公司|母公司|關係人|關係企業|集團|subsidiar|affiliate|related.party|parent.company', re.I)


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('STOP missing timezone/time')
    return stamp.tz_convert('UTC')


def select_version(versions, observations, decision_at, *, strict=False):
    """Select latest upstream lineage version visible at t.

    observations contain verified blob/HEAD integrity (valid), and ancestor
    commit IDs. validated_at is additional admission evidence, never inferred
    from observed_at. Before our first observation only descriptive upstream
    visibility is available; it cannot satisfy strict forward admission.
    """
    t = utc(decision_at)
    good = [o for o in observations if o.get('valid') is True and not o.get('error')]
    if not good:
        return None
    first = min(utc(o['observed_at']) for o in good)
    candidates = []
    for version in versions:
        committed = utc(version['committed_at'])
        linked = [o for o in good if version['commit'] in o['ancestors']]
        visible = []
        for o in linked:
            at = max(committed, utc(o['observed_at']))
            if strict:
                if not o.get('validated_at'):
                    continue
                at = max(at, utc(o['validated_at']))
            if at <= t:
                visible.append((at, o))
        healthy = not strict or any(t - max(utc(o['observed_at']), utc(o['validated_at'])) < pd.Timedelta(days=7)
                                    for _, o in visible)
        if visible and healthy:
            at, _ = min(visible, key=lambda v: v[0])
            candidates.append(dict(version, visible_at=at.isoformat(), visibility='observed_validated' if strict else 'observed'))
        elif not strict and committed < first and committed <= t and linked:
            candidates.append(dict(version, visible_at=committed.isoformat(), visibility='upstream_commit_only'))
    return max(candidates, key=lambda v: v['order']) if candidates else None


def eligible_edges(edges, known_codes):
    """E-PM filter, union-find groups, and conservative transaction quarantine.

    known_codes is a supplied mapping scope, not an inferred trading universe.
    Transaction clues without explicit entity resolution remain UNKNOWN.
    Every input edge gets a replayable disposition; none of its prose is copied.
    """
    known = set(known_codes)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    def tw(x):
        return isinstance(x, str) and x.startswith('tw.') and len(x) > 3

    unknown_pairs = set()
    group_pairs = set()
    malformed_group = False
    for e in edges:
        a, b = e.get('src'), e.get('dst')
        if e.get('type') == 'subsidiary':
            if isinstance(a, str) and isinstance(b, str) and a and b:
                union(a, b)
            else:
                malformed_group = True
        if e.get('type') == 'transaction':
            text = ' '.join(str(e.get(k, '')) for k in ('detail', 'details', 'evidence', 'events'))
            if GROUP_CLUE.search(text):
                pair = frozenset((a, b))
                # Only a separately verified resolution may upgrade a clue.
                if e.get('entity_resolution') == 'verified_same_group':
                    group_pairs.add(pair)
                else:
                    unknown_pairs.add(pair)
    directed = {(e.get('src'), e.get('dst')) for e in edges
                if e.get('type') == 'supplier' and e.get('evidence_level') in {'primary', 'media'}}
    customers = defaultdict(set)
    unknown = set()
    dispositions = []
    for index, e in enumerate(edges):
        a, b = e.get('src'), e.get('dst')
        reason = 'OK'
        if e.get('type') != 'supplier':
            reason = 'NOT_SUPPLIER'
        elif not (tw(a) and tw(b)):
            reason = 'NOT_TW_PAIR'
        elif e.get('evidence_level') not in {'primary', 'media'}:
            reason = 'EVIDENCE_LEVEL'
        elif a == b:
            reason = 'SELF_LOOP'
        elif find(a) == find(b) or frozenset((a, b)) in group_pairs:
            reason = 'SAME_GROUP'
        elif (b, a) in directed:
            reason = 'DIRECTION_CONFLICT'
        elif a[3:] not in known or b[3:] not in known:
            reason = 'MAPPING_UNKNOWN'
        elif malformed_group:
            reason = 'GROUP_SCOPE_UNKNOWN'
        elif frozenset((a, b)) in unknown_pairs:
            reason = 'TRANSACTION_ENTITY_UNKNOWN'
        elif b[3:] in customers[a[3:]]:
            reason = 'DUPLICATE'
        if reason in {'DIRECTION_CONFLICT', 'MAPPING_UNKNOWN', 'GROUP_SCOPE_UNKNOWN', 'TRANSACTION_ENTITY_UNKNOWN'} and tw(a):
            unknown.add(a[3:])
        if reason == 'OK':
            customers[a[3:]].add(b[3:])
        dispositions.append(dict(index=index, src=a, dst=b, reason=reason))
    return dict(customers={k: sorted(v) for k, v in sorted(customers.items())},
                known_codes=known, unknown_suppliers=unknown, dispositions=dispositions,
                excluded=dict(Counter(d['reason'] for d in dispositions if d['reason'] != 'OK')))


def revenue_snapshot(frame, days, decision_at):
    """Reuse Step3 first-announcement YoY and surprise, with causal cutoffs.

    Returns the latest revenue month per company, not the latest arrival month.
    Real-source strict admission remains external to this numerical replay.
    """
    days = pd.DatetimeIndex(days)
    t = pd.Timestamp(decision_at)
    if t.tzinfo is not None:
        t = t.tz_convert('Asia/Taipei').tz_localize(None)
    session = int(days.searchsorted(t, side='right') - 1)
    if session < 0:
        return {}
    f = frame.copy()
    f['available_date'] = pd.to_datetime(f.available_date)
    f = f[f.available_date < days[session].normalize()].copy()
    # Forward adapters may supply actual receive/validation times. Both are
    # required together and can only delay the publication-based clock.
    timing = {'received_at', 'validated_at'} & set(f)
    if timing:
        if timing != {'received_at', 'validated_at'}:
            raise ValueError('STOP incomplete actual revenue timing')
        # At a given decision close the actual clock is an additional filter;
        # it must not turn receipt before this close into a next-day delay.
        actual = [max(utc(r.received_at), utc(r.validated_at)).tz_convert('Asia/Taipei').tz_localize(None)
                  for r in f.itertuples()]
        f = f[np.array([stamp <= t for stamp in actual], dtype=bool)]
    if f.empty:
        return {}
    events = step3.monthly_events(f, days)
    reasons = {(r['stock_id'], r['session']): r['reasons'] for r in events.attrs['missing_reasons']}
    latest_period = f.assign(period=pd.to_datetime(f.period)).groupby('stock_id').period.max()
    answer = {}
    for r in events.groupby('stock_id', sort=True).tail(1).itertuples():
        code = str(r.stock_id)
        period = latest_period.loc[r.stock_id]
        answer[code] = dict(yoy=float(r.monthly_yoy), surprise=float(r.monthly_surprise),
                            period=period.strftime('%Y-%m'), session=int(r.session),
                            age_sessions=session-int(r.session),
                            age_months=(days[session].year-period.year)*12+days[session].month-period.month,
                            reasons=reasons.get((code, int(r.session)), {}))
    return answer


def customer_features(supplier, relations, revenue, *, scope_known=True):
    out = dict.fromkeys(FEATURES, np.nan)
    out.update(reason='RELATION_SCOPE_UNKNOWN', missing_reasons={}, valid_customers=[], max_customer_share=None)
    if (not scope_known or relations is None or supplier not in relations['known_codes']
            or supplier in relations['unknown_suppliers']):
        return out
    valid = []
    missing = Counter()
    for customer in relations['customers'].get(supplier, []):
        r = revenue.get(customer)
        if r is None:
            missing['NO_VISIBLE_REVENUE'] += 1
            continue
        if not np.isfinite(r['yoy']):
            missing['YOY_MISSING_OR_INCOMPARABLE_OR_NONPOSITIVE_DENOMINATOR'] += 1
        if not np.isfinite(r['surprise']):
            missing['SURPRISE_FEWER_THAN_3_YOY_OR_INVALID_EXPECTATION'] += 1
        if np.isfinite(r['yoy']) and np.isfinite(r['surprise']):
            valid.append(customer)
    n = len(valid)
    out[FEATURES[2]] = float(n)
    out.update(reason='OK' if n >= 2 else 'FEWER_THAN_2_VALID_CUSTOMERS',
               missing_reasons=dict(missing), valid_customers=valid,
               max_customer_share=1/n if n else None)
    if n >= 2:
        out[FEATURES[0]] = float(np.median([revenue[c]['yoy'] for c in valid]))
        out[FEATURES[1]] = float(np.mean([revenue[c]['surprise'] for c in valid]))
    return out
