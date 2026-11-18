"""Synthetic contracts: removing any core diagnostic must break these tests."""
import hashlib
import json

import pandas as pd
import pytest

from research_lab import data_health as health


@pytest.fixture
def frames():
    days = pd.bdate_range('2026-07-01', periods=70)
    base = pd.DataFrame([(d, str(i)) for d in days for i in range(10)],
                        columns=['Date', 'stock_id'])
    result = {name: base.copy() for name in health.DATASETS}
    result['per_raw'] = base.assign(PER=base.Date.dt.day_of_year.astype(float), PBR=2., DY=3.)
    result['macro_raw'] = pd.DataFrame({'Date': days, 'VIX': 1., 'Oil': 2.})
    return result


def check(frames, **kwargs):
    return health.analyze(frames, lookback_days=45, known_anomalies=[], **kwargs)


def item(report, dataset, kind, date=None):
    found = [x for x in report['checks'] if x['dataset'] == dataset and x['check'] == kind
             and (date is None or x.get('date') == date)]
    assert len(found) == 1, found
    return found[0]


def test_clean_and_scope(frames):
    report = check(frames)
    assert report['status'] == 'OK'
    assert report['exit_code'] == 0
    assert len(report['trading_days']) == 45
    assert item(report, 'institutional_raw', 'row_ratio', report['trading_days'][0])['baseline_median'] == 10
    json.dumps(report, allow_nan=False)


def test_missing_day_fails_independently(frames):
    day = frames['prices_raw'].Date.max()
    frames['institutional_raw'] = frames['institutional_raw'].query('Date != @day')
    report = check(frames)
    row = item(report, 'institutional_raw', 'missing_days')
    assert row['dates'] == [day.strftime('%Y-%m-%d')]
    assert row['count'] == 1 and row['status'] == 'FAIL'
    assert report['exit_code'] == 1


def test_row_ratio_uses_own_sixty_normal_days_and_strict_boundary(frames):
    day = frames['prices_raw'].Date.max()
    f = frames['daytrade_raw']
    frames['daytrade_raw'] = f[(f.Date != day) | f.stock_id.isin(['0', '1', '2', '3'])]
    report = check(frames)
    row = item(report, 'daytrade_raw', 'row_ratio', str(day.date()))
    assert row['rows'] == 4 and row['baseline_median'] == 10
    assert row['ratio'] == .4 and row['threshold'] == .5 and row['status'] == 'WARN'
    assert report['exit_code'] == 0
    frames['daytrade_raw'] = f[(f.Date != day) | f.stock_id.isin(['0', '1', '2', '3', '4'])]
    assert item(check(frames), 'daytrade_raw', 'row_ratio', str(day.date()))['status'] == 'OK'


def test_duplicate_keys_counts_groups_excess_and_members(frames):
    f = frames['securities_raw']
    frames['securities_raw'] = pd.concat([f, f.iloc[[-1]], f.iloc[[-1]]])
    report = check(frames)
    row = item(report, 'securities_raw', 'duplicates')
    assert (row['keys'], row['excess_rows'], row['member_rows']) == (1, 2, 3)
    assert row['status'] == 'FAIL' and report['exit_code'] == 1


def test_carry_forward_matches_triplet_not_individual_fields(frames):
    f = frames['per_raw']
    day, previous = sorted(f.Date.unique())[-2:][::-1]
    f.loc[f.Date == day, 'PER'] = f.loc[f.Date == previous, 'PER'].to_numpy()
    report = check(frames)
    row = item(report, 'per_raw', 'carry_forward', str(pd.Timestamp(day).date()))
    assert row['matched_stocks'] == 10 and row['identical_stocks'] == 10
    assert row['ratio'] == 1. and row['threshold'] == .9 and row['status'] == 'WARN'
    assert report['exit_code'] == 0
    f.loc[(f.Date == day) & (f.stock_id == '0'), 'DY'] = 4.
    assert item(check(frames), 'per_raw', 'carry_forward', str(pd.Timestamp(day).date()))['status'] == 'OK'


def test_price_sparse_weekday_and_weekend_excluded_known_is_bounded(frames):
    day = frames['prices_raw'].Date.max()
    f = frames['prices_raw']
    frames['prices_raw'] = pd.concat([f[(f.Date != day) | (f.stock_id == '0')],
                                     f.iloc[:2].assign(Date=pd.Timestamp('2026-09-20'))])
    known = [{'dataset': 'prices_raw', 'check': 'nontrading_rows', 'date': '2026-09-20', 'rows': 2}]
    report = health.analyze(frames, 45, known)
    assert str(day.date()) not in report['trading_days']
    assert item(report, 'prices_raw', 'nontrading_rows', '2026-09-20')['status'] == 'KNOWN'
    assert item(report, 'prices_raw', 'nontrading_rows', str(day.date()))['status'] == 'FAIL'
    known[0]['rows'] = 1
    assert item(health.analyze(frames, 45, known), 'prices_raw', 'nontrading_rows', '2026-09-20')['status'] == 'FAIL'


def test_macro_latest_five_nan_and_missing_day(frames):
    f = frames['macro_raw']
    day = f.Date.max()
    f.loc[f.Date == day, 'VIX'] = float('nan')
    report = check(frames)
    rows = [x for x in report['checks'] if x['check'] == 'nan_ratio']
    assert len(rows) == 10
    row = item(report, 'macro_raw', 'nan_ratio', str(day.date()) + ':VIX')
    assert row['ratio'] == 1. and row['nan_rows'] == 1 and row['status'] == 'WARN'
    frames['macro_raw'] = f[f.Date != day]
    row = item(check(frames), 'macro_raw', 'nan_ratio', str(day.date()) + ':VIX')
    assert row['ratio'] is None and row['status'] == 'FAIL'


def test_known_nontrading_per_excluded_and_missing_known(frames):
    f = frames['per_raw']
    frames['per_raw'] = pd.concat([f, f.iloc[-10:].assign(Date=pd.Timestamp('2026-09-20'))])
    day = str(frames['prices_raw'].Date.max().date())
    frames['market_value_raw'] = frames['market_value_raw'][frames['market_value_raw'].Date.astype(str) != day]
    known = [{'dataset': 'per_raw', 'check': 'nontrading_rows', 'date': '2026-09-20'},
             {'dataset': 'market_value_raw', 'check': 'missing_days', 'date': day}]
    report = health.analyze(frames, 45, known)
    assert report['exit_code'] == 0
    assert item(report, 'market_value_raw', 'missing_days')['status'] == 'KNOWN'
    assert not any(x['check'] == 'carry_forward' and x.get('date') == '2026-09-20' for x in report['checks'])


def test_missing_schema_empty_and_no_paired_values_fail_closed(frames):
    frames['per_raw']['PER'] = float('nan')
    assert any(x['check'] == 'carry_forward' and x['status'] == 'WARN' and x['ratio'] is None
               for x in check(frames)['checks'])
    frames['per_raw'] = frames['per_raw'].drop(columns='PBR')
    assert check(frames)['exit_code'] == 1
    frames['prices_raw'] = frames['prices_raw'].iloc[:0]
    assert check(frames)['exit_code'] == 1


def test_cli_read_only_output_guards_env_and_failure_json(frames, tmp_path, monkeypatch):
    data = tmp_path / 'Data' / 'processed_v6'
    data.mkdir(parents=True)
    for name, f in frames.items():
        f.to_parquet(data / (name + '.parquet'), index=False)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    monkeypatch.setenv('MARKETMAMBA_DATA_ROOT', str(data.parent))
    output = tmp_path / 'health.json'
    assert health.main(['--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['data_dir'] == str(data.resolve())
    assert len(report['inputs']) == 7
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    for target in [data / 'bad.json', data.parent / 'bad.json', data / 'prices_raw.parquet']:
        with pytest.raises(SystemExit):
            health.main(['--data-dir', str(data), '--output', str(target)])
    alias = tmp_path / 'alias'
    alias.symlink_to(data, target_is_directory=True)
    with pytest.raises(SystemExit):
        health.main(['--data-dir', str(data), '--output', str(alias / 'bad.json')])
    (data / 'institutional_raw.parquet').unlink()
    assert health.main(['--data-dir', str(data), '--output', str(output)]) == 1
    assert json.loads(output.read_text())['status'] == 'FAIL'


def test_baseline_uses_sixty_days_not_just_report_window(frames):
    days = sorted(frames['prices_raw'].Date.unique())
    frames['daytrade_raw'] = pd.DataFrame(
        [(d, str(i)) for j, d in enumerate(days) for i in range(100 if j < 45 else 10)],
        columns=['Date', 'stock_id'])
    row = item(check(frames), 'daytrade_raw', 'row_ratio', str(pd.Timestamp(days[-1]).date()))
    assert row['baseline_median'] == 100 and row['ratio'] == .1
    assert row['status'] == 'WARN'


def test_cli_override_replaces_known_list_and_rejects_invalid_config(frames, tmp_path):
    data = tmp_path / 'inputs'
    data.mkdir()
    frames['prices_raw'] = pd.concat([frames['prices_raw'],
        frames['prices_raw'].iloc[:2].assign(Date=pd.Timestamp('2026-09-20'))])
    for name, f in frames.items():
        f.to_parquet(data / (name + '.parquet'), index=False)
    output, config = tmp_path / 'report.json', tmp_path / 'known.json'
    args = ['--data-dir', str(data), '--output', str(output)]
    assert health.main(args) == 0
    config.write_text('[]')
    assert health.main(args + ['--known-anomalies', str(config)]) == 1
    row = item(json.loads(output.read_text()), 'prices_raw', 'nontrading_rows', '2026-09-20')
    assert row['rows'] == 2 and row['status'] == 'FAIL'
    config.write_text('[{"dataset":"prices_raw","check":"duplicates","date":"2026-09-20"}]')
    with pytest.raises(SystemExit):
        health.main(args + ['--known-anomalies', str(config)])
