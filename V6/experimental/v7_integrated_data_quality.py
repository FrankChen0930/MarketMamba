"""V7 explicit-calendar numeric admission and reusable graded report contract."""
from dataclasses import dataclass, asdict
from datetime import date
import hashlib
import json
import numpy as np
import pandas as pd

PROTOCOL_VERSION = 'v7-calendar-v2'


class QualityBlocked(ValueError):
    """A fatal contract error still exposes the same reusable graded report."""
    def __init__(self, message, reason='AMBIGUOUS_CALENDAR_SCHEMA', dates=()):
        super().__init__(message)
        self.report = {'schema_version':'v7-quality-report-v1','blocking':True,
            'entries':[{'severity':'BLOCK','reason_code':reason,'dates':list(dates),
                        'stocks':[],'denominator':None,'threshold':None}],
            'uncertain_market_history':list(dates), 'summary_zh_TW':f'BLOCK：{message}'}


@dataclass(frozen=True)
class QualityPolicy:
    major_outage_fraction: float = .5
    minimum_usable_days: int = 2
    interior_gap_invalidates_target: bool = True

    def __post_init__(self):
        if not 0 < self.major_outage_fraction <= 1 or self.minimum_usable_days < 1:
            raise ValueError('invalid quality thresholds')


def calendar_contract(document):
    days = document.get('trading_calendar')
    if not isinstance(days, list) or not days or not isinstance(document.get('provenance'), str) or not document.get('provenance', '').strip():
        raise QualityBlocked('BLOCK ambiguous calendar: independent trading_calendar and provenance required')
    try:
        parsed = [date.fromisoformat(d).isoformat() for d in days]
    except (ValueError, TypeError):
        raise QualityBlocked('BLOCK ambiguous calendar dates') from None
    if parsed != sorted(set(parsed)):
        raise QualityBlocked('BLOCK ambiguous calendar order/duplicates')
    return parsed


def validate_stock_ids(values):
    if any(pd.isna(value) or not str(value).strip() or str(value) != str(value).strip()
           for value in values):
        raise QualityBlocked('股票識別鍵為空值或含歧義空白', 'AMBIGUOUS_STOCK_KEY')
    return [str(value) for value in values]


def universe_contract(document, days):
    provenance = document.get('universe_provenance')
    if not isinstance(provenance, str) or not provenance.strip():
        raise QualityBlocked('BLOCK declared universe_provenance required')
    universe = document.get('expected_universe')
    if not isinstance(universe, dict) or set(universe) != set(days):
        raise QualityBlocked('BLOCK expected_universe must cover exactly the chosen calendar')
    normalized = {}
    for day in days:
        ids = universe[day]
        if not isinstance(ids, list) or not ids:
            raise QualityBlocked('BLOCK nonempty expected universe required')
        ids = validate_stock_ids(ids)
        if len(set(ids)) != len(ids):
            raise QualityBlocked('BLOCK duplicate expected universe IDs')
        normalized[day] = sorted(ids)
    return normalized


def fingerprint(contract, policy):
    payload = {'version': PROTOCOL_VERSION, 'contract': contract, 'policy': asdict(policy),
               'window': '60 calendar positions; leading padding only',
               'mask': 'history and observation distinct; learned missing embedding',
               'labels': 'raw Close t+5/t+10; no benchmark', 'rolling': 'contiguous valid price segments'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def protocol_metadata(document, policy=QualityPolicy()):
    days = calendar_contract(document)
    universe = universe_contract(document, days)
    contract = {'trading_calendar': days, 'calendar_provenance': document['provenance'],
                'expected_universe': universe, 'universe_provenance': document['universe_provenance'],
                'selected_universe': sorted(set().union(*map(set, universe.values()))),
                'universe_selection': 'union of declared expected IDs over chosen calendar'}
    return {**contract, 'protocol_version': PROTOCOL_VERSION,
            'calendar_sha256': hashlib.sha256(json.dumps(days).encode()).hexdigest(),
            'quality_policy': asdict(policy), 'protocol_fingerprint': fingerprint(contract, policy)}


def validate_protocol(metadata):
    if not isinstance(metadata.get('quality_policy'), dict):
        raise QualityBlocked('BLOCK calendar protocol quality_policy required')
    rebuilt = protocol_metadata({'trading_calendar': metadata.get('trading_calendar'),
        'provenance': metadata.get('calendar_provenance'),
        'expected_universe': metadata.get('expected_universe'),
        'universe_provenance': metadata.get('universe_provenance')},
        QualityPolicy(**metadata.get('quality_policy', {})))
    if any(metadata.get(key) != value for key, value in rebuilt.items()):
        raise QualityBlocked('incompatible prepared protocol fingerprint; rebuild required')
    return rebuilt['trading_calendar'], QualityPolicy(**rebuilt['quality_policy'])


def coverage_entries(frame, days, universe, policy):
    valid_by_date = {str(day.date()): set(group.stock_id)
                     for day, group in frame[frame.observation_valid.eq(True)].groupby('Date')}
    entries = []
    for day in days:
        expected = set(universe[day])
        missing = expected - valid_by_date.get(day, set())
        if missing:
            blocked = len(missing) / len(expected) >= policy.major_outage_fraction
            entries.append(dict(severity='BLOCK' if blocked else 'WARN',
                reason_code='MAJOR_COVERAGE_OUTAGE' if blocked else 'SPARSE_GAP',
                dates=[day], stocks=sorted(missing), denominator=len(expected),
                threshold=policy.major_outage_fraction))
    return entries


def select_declared_prices(prices, document):
    days = calendar_contract(document)
    universe = universe_contract(document, days)
    ids = set().union(*map(set, universe.values()))
    frame = prices.copy()
    frame['stock_id'] = validate_stock_ids(frame.stock_id)
    selected = frame.stock_id.isin(ids)
    audit = {'selection': 'union of declared expected IDs over chosen calendar',
             'universe_provenance': document['universe_provenance'],
             'selected_universe': sorted(ids), 'excluded_rows': int((~selected).sum()),
             'excluded_stocks': sorted(set(frame.loc[~selected, 'stock_id']))}
    selected_frame = frame.loc[selected].copy()
    expected_sets = {day:set(ids) for day,ids in universe.items()}
    audit['extra_observations_outside_daily_expected_membership'] = {
        str(day.date()): int((~group.stock_id.isin(expected_sets.get(str(day.date()), set()))).sum())
        for day,group in selected_frame.assign(Date=pd.to_datetime(selected_frame.Date)).groupby('Date')}
    return selected_frame, audit


def assess_prices(prices, document, policy=QualityPolicy(), expected_universe=None):
    days = calendar_contract(document)
    required = {'Date', 'stock_id', 'Open', 'High', 'Low', 'Close', 'Volume'}
    if required - set(prices): raise QualityBlocked('BLOCK ambiguous price schema')
    frame = prices.copy()
    frame['Date'] = pd.to_datetime(frame.Date, errors='coerce')
    if frame.Date.isna().any() or frame.duplicated(['Date', 'stock_id']).any():
        raise QualityBlocked('BLOCK ambiguous price date/key schema')
    frame['stock_id'] = validate_stock_ids(frame.stock_id)
    if frame.duplicated(['Date', 'stock_id']).any():
        raise QualityBlocked('BLOCK ambiguous price date/key schema')
    numeric = frame[['Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce')
    frame[numeric.columns] = numeric
    bad = (~np.isfinite(numeric)).any(axis=1) | (numeric[['Open','High','Low','Close']] <= 0).any(axis=1) | (numeric.Volume < 0)
    ohlc = (numeric.Low > numeric[['Open','Close']].min(axis=1)) | (numeric.High < numeric[['Open','Close']].max(axis=1)) | (numeric.Low > numeric.High)
    frame['observation_valid'] = ~(bad | ohlc)
    entries = []
    def entry(severity, reason, dates=(), stocks=(), denominator=None, threshold=None):
        entries.append(dict(severity=severity, reason_code=reason, dates=list(dates), stocks=list(stocks), denominator=denominator, threshold=threshold))
    observed_days = frame.Date.dt.strftime('%Y-%m-%d')
    unexpected = sorted(set(observed_days) - set(days))
    if unexpected: entry('BLOCK','CALENDAR_UNCERTAIN',unexpected)
    for reason, mask in [('NUMERIC_INVALID',bad),('OHLC_CONTRADICTION',ohlc)]:
        if mask.any():
            for day, group in frame.loc[mask].groupby('Date'):
                entry('QUARANTINE', reason, [str(day.date())], group.stock_id.tolist())
    universe = universe_contract(document, days)
    if expected_universe is not None and expected_universe != document['expected_universe']:
        raise QualityBlocked('BLOCK inconsistent expected universe')
    entries.extend(coverage_entries(frame, days, universe, policy))
    entry('INFO','NUMERIC_ADMISSION_SOURCE_AGNOSTIC')
    report = {'schema_version':'v7-quality-report-v1', 'policy':asdict(policy), 'entries':entries,
              'uncertain_market_history':unexpected, 'market_membership_history':'未以來源或現況市場名單推定歷史市場別；有效混合來源觀測保留', 'blocking':any(e['severity']=='BLOCK' for e in entries)}
    report['summary_zh_TW'] = '\n'.join(f"{e['severity']}：{e['reason_code']}；日期 {','.join(e['dates'])}；股票 {len(e['stocks'])}；分母 {e['denominator']}；門檻 {e['threshold']}" for e in entries)
    return frame, report


def calendar_labels(prices, calendar, interior_gap=True):
    result = []
    for stock, rows in prices.groupby('stock_id', sort=False):
        rows = rows.set_index('Date').reindex(pd.to_datetime(calendar))
        close = pd.to_numeric(rows.Close, errors='coerce').to_numpy(float)
        valid = np.isfinite(close) & (close > 0) & rows.observation_valid.eq(True).to_numpy(bool)
        close = np.where(valid, close, np.nan)
        out = pd.DataFrame({'Date':pd.to_datetime(calendar), 'stock_id':stock})
        invalid_prefix = np.r_[0, np.cumsum(~valid)]
        for horizon in (5,10):
            labels = np.full(len(calendar), np.nan)
            n = max(0,len(calendar)-horizon)
            usable = valid[:n] & valid[horizon:]
            if interior_gap: usable &= (invalid_prefix[horizon+1:] - invalid_prefix[:n]) == 0
            indices = np.flatnonzero(usable)
            labels[indices] = close[indices+horizon]/close[indices]-1
            out[f'Alpha_{horizon}d'] = labels
        result.append(out)
    return pd.concat(result,ignore_index=True)


def assess_training(frame, training_dates, policy, report):
    """Append finite per-head coverage and the conservative execution floor."""
    subset = frame[frame.observation_valid & frame.Date.dt.strftime('%Y-%m-%d').isin(training_dates)]
    finite = pd.DataFrame({'Date':subset.Date,
                          'Alpha_5d':np.isfinite(subset.Alpha_5d),
                          'Alpha_10d':np.isfinite(subset.Alpha_10d)})
    counts = finite.groupby('Date')[['Alpha_5d','Alpha_10d']].sum()
    usable = int((counts >= 2).any(axis=1).sum())
    report['training_coverage'] = {'usable_dates':usable,
        'head_target_counts':{name:int(finite[name].sum()) for name in ('Alpha_5d','Alpha_10d')},
        'minimum_targets_per_usable_date':2, 'minimum_usable_days':policy.minimum_usable_days}
    blocked = usable < policy.minimum_usable_days
    report['entries'].append({'severity':'BLOCK' if blocked else 'INFO',
        'reason_code':'INSUFFICIENT_USABLE_TRAINING_DATA' if blocked else 'TRAINING_COVERAGE',
        'dates':list(training_dates), 'stocks':[], 'denominator':len(training_dates),
        'threshold':policy.minimum_usable_days})
    report['blocking'] |= blocked
    report['summary_zh_TW'] += f"\n{'BLOCK' if blocked else 'INFO'}：可更新訓練日 {usable}；最低 {policy.minimum_usable_days}；各 horizon target 計數 {report['training_coverage']['head_target_counts']}"
    return report
