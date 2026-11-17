"""Offline contracts and executable mutation controls for the read-only audit."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from V6.research_lab import stale_audit as audit


def frame(days, size=100):
    return pd.DataFrame([(d, f'{i:04}', float(i)) for d in days for i in range(size)],
                        columns=['Date', 'stock_id', 'value'])


def test_ratio_boundary_and_all_fields():
    df = frame(['2026-10-12', '2026-10-13', '2026-10-14'])
    df.loc[(df.Date == '2026-10-13') & (df.stock_id == '0000'), 'value'] = 999
    df.loc[(df.Date == '2026-10-14') & df.stock_id.isin(['0000', '0001']), 'value'] = 999
    result = audit.audit_frame(df, df.Date.unique())
    assert result['comparisons'][0]['equal_ratio'] == .99
    assert result['comparisons'][0]['stale_candidate'] is True
    assert result['segments'][0]['trading_days'] == 3
    other = df.assign(second=1.)
    other.loc[other.Date == '2026-10-13', 'second'] = 2.
    assert not audit.audit_frame(other, df.Date.unique())['segments']


def test_prices_dates_override_weekdays_and_missing_breaks_run():
    # Friday is absent from prices; Saturday is present. Never use weekday().
    df = frame(['2026-10-08', '2026-10-09', '2026-10-10', '2026-10-13'])
    r = audit.audit_frame(df, ['2026-10-08', '2026-10-10', '2026-10-12', '2026-10-13'])
    assert r['missing_trading_days'] == ['2026-10-12']
    assert r['segments'][0]['end'] == '2026-10-10'
    assert r['segments'][0]['contains_nontrading_rows'] is True
    assert r['segments'][0]['rows'] == 300
    assert r['nontrading_comparisons'][0]['date'] == '2026-10-09'
    assert r['comparisons'][-1]['overlap'] == 0


def test_overlap_is_bilateral_and_exact_boundary():
    df = frame(['2026-10-12', '2026-10-13'])
    df = df[~((df.Date == '2026-10-13') & (df.stock_id.astype(int) >= 80))]
    assert audit.audit_frame(df, df.Date.unique())['segments']
    df = df[~((df.Date == '2026-10-13') & (df.stock_id == '0079'))]
    assert not audit.audit_frame(df, df.Date.unique())['segments']


def test_nulls_reported_and_duplicate_keys_rejected():
    df = frame(['2026-10-12', '2026-10-13'], size=2)
    df.loc[df.stock_id == '0000', 'value'] = float('nan')
    c = audit.audit_frame(df, df.Date.unique())['comparisons'][0]
    assert c['equal_rows'] == 2
    assert c['fully_observed_overlap'] == 1
    with pytest.raises(ValueError, match='duplicate'):
        audit.audit_frame(pd.concat([df, df.iloc[:1]]), df.Date.unique())


def test_nontrading_copies_can_match_later_trading_day():
    df = frame(['2026-10-08', '2026-10-09', '2026-10-12'])
    df.loc[df.Date == '2026-10-08', 'value'] += 1
    r = audit.audit_frame(df, ['2026-10-08', '2026-10-12'])
    assert not r['segments']
    s = r['observed_date_segments'][0]
    assert s['dates'] == ['2026-10-09', '2026-10-12']
    assert s['nontrading_dates'] == ['2026-10-09']
    assert s['rows'] == 200


def test_lending_historical_quota_is_explicitly_a_different_field():
    payload = {'date': '20261012', 'fields': ['代號', '次一營業日可限額'],
               'data': [['1101', '1,234']]}
    parsed = audit.parse_public(payload, 'twse_sbl_balance', '2026-10-12')
    assert parsed.iloc[0].next_session_short_sale_quota == 1234
    assert 'Securities_Lending' not in parsed


def test_public_date_and_schema_fail_closed():
    payload = {'date': '20261012', 'stat': 'OK', 'fields': ['證券代號', '殖利率(%)', '本益比', '股價淨值比'],
               'data': [['1101', '2.5', '10', '1.2']]}
    parsed = audit.parse_public(payload, 'twse_per', '2026-10-12')
    assert parsed.iloc[0].PER == 10
    assert parsed.iloc[0].DY == 2.5
    with pytest.raises(ValueError):
        audit.parse_public(payload, 'twse_per', '2026-10-13')
    payload['fields'][2] = 'unknown'
    with pytest.raises(ValueError):
        audit.parse_public(payload, 'twse_per', '2026-10-12')


def test_repair_counts_distinguish_missing_changed_and_equal():
    old = frame(['2026-10-12'], size=3)
    new = frame(['2026-10-12'], size=4)
    new.loc[0, 'value'] = 20
    assert audit.repair_counts(old, new) == {'insert': 1, 'overwrite': 1, 'unchanged': 2,
                                           'local_only_unresolved': 0}


@pytest.mark.parametrize('before,after', [
    ('equal_ratio >= 0.99', 'equal_ratio >= 1.0'),
    ('sessions = sorted(set(pd.to_datetime(trading_dates).strftime("%Y-%m-%d")))',
     'sessions = sorted(d for d in daily if pd.Timestamp(d).dayofweek < 5)'),
])
def test_mutation_controls(tmp_path, before, after):
    source = Path(audit.__file__).read_text()
    assert source.count(before) == 1
    mutant = tmp_path / 'mutant.py'
    mutant.write_text(source.replace(before, after))
    spec = importlib.util.spec_from_file_location('mutant', mutant)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Execute the same behavioral assertions against the mutant: they MUST fail.
    original = globals()['audit']
    globals()['audit'] = module
    try:
        with pytest.raises(AssertionError):
            if 'equal_ratio' in before:
                test_ratio_boundary_and_all_fields()
            else:
                test_prices_dates_override_weekdays_and_missing_breaks_run()
    finally:
        globals()['audit'] = original


def test_output_cannot_target_data(tmp_path):
    with pytest.raises(ValueError):
        audit.output_path(Path('Data/processed_v6/not_allowed.json'))
