"""Announcement-time research exclusion, never proof of tradability."""
import numpy as np
import pandas as pd
import pytest
from research_lab import disposal


def frame(days, announced='2025-01-08', stock='2330'):
    return pd.DataFrame({'Date': days, 'stock_id': stock,
                         'status': 'disposal', 'announced': announced})


def test_start_before_announcement_and_next_session_boundary():
    dates = np.array(['2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10'])
    result = disposal.disposal_known(frame(dates), dates, ['2330'] * 5)
    assert result.dtype == bool
    assert result.tolist() == [False, False, False, True, True]


@pytest.mark.parametrize('announcement', ['2025-01-11', '2025-01-12', '2025-01-13'])
def test_weekend_or_missing_session_needs_extra_session(announcement):
    dates = ['2025-01-10', '2025-01-14', '2025-01-15']
    assert disposal.disposal_known(frame(dates, announcement), dates, ['2330'] * 3).tolist() == [False, False, True]


def test_exact_daily_membership_duplicates_stock_and_order():
    dates = ['2025-01-10', '2025-01-09', '2025-01-09', '2025-01-08']
    status = frame(['2025-01-09', '2025-01-09'])
    assert disposal.disposal_known(status, dates, ['2330', '2330', '1101', '2330']).tolist() == [False, True, False, False]


def test_attention_missing_announcement_and_calendar_exhaustion():
    dates = ['2025-01-08', '2025-01-09']
    status = pd.concat([frame(dates, None), frame(dates, '2025-01-09'),
                        frame(dates, '2025-01-08').assign(status='attention')])
    assert not disposal.disposal_known(status, dates, ['2330'] * 2).any()


def test_empty_and_no_mutation():
    dates = np.array(['2025-01-08', '2025-01-09'], dtype='datetime64[D]')
    ids = np.array(['2330', '2330'])
    status = frame(dates)
    old = status.copy(deep=True)
    d0, s0 = dates.copy(), ids.copy()
    disposal.disposal_known(status, dates, ids)
    pd.testing.assert_frame_equal(status, old)
    np.testing.assert_array_equal(dates, d0)
    np.testing.assert_array_equal(ids, s0)
    assert not disposal.disposal_known(status.iloc[:0], dates, ids).any()
    assert disposal.disposal_known(status, [], []).shape == (0,)


def test_mismatched_rows_rejected():
    with pytest.raises(ValueError):
        disposal.disposal_known(frame([]), ['2025-01-08'], [])


@pytest.mark.parametrize('has_volume', [False, True])
def test_audit_eval_only_finite_means_duplicates_and_readonly(tmp_path, has_volume):
    import json
    base, folds = tmp_path / 'base', tmp_path / 'folds'
    base.mkdir()
    folds.mkdir()
    dates = np.array([f'{year}-01-{day:02d}' for year in range(2021, 2027)
                      for day in (4, 5, 6)], dtype='datetime64[D]')
    ids = np.array(['2330'] * len(dates))
    y5 = np.tile([100., .1, np.nan], 6)
    for name, values in [('dates', dates), ('stock_ids', ids), ('y5', y5)]:
        np.save(base / f'{name}.npy', values)
    (base / 'manifest.json').write_text(json.dumps({'feature_order': ['Volume'] if has_volume else []}))
    if has_volume:
        np.save(base / 'X.npy', np.tile([1000., 20., 40.], 6)[:, None])
    for year in range(2021, 2027):
        folder = folds / f'fold_{year}'
        folder.mkdir()
        np.save(folder / 'splits.npy', np.where(dates.astype('datetime64[Y]') == np.datetime64(str(year)), 2, 1).astype('uint8'))
    status = pd.concat([frame([f'{year}-01-04', f'{year}-01-05', f'{year}-01-06'], f'{year}-01-04')
                        for year in range(2021, 2027)])
    status = pd.concat([status, status.iloc[[1]]], ignore_index=True)
    path = tmp_path / 'status.parquet'
    status.to_parquet(path)
    files = [p for p in tmp_path.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    report = disposal.universe_audit(path, base, folds)
    assert report['duplicate_stock_date_excess_rows'] == 1
    assert report['duplicate_stock_date_member_rows'] == 2
    for row in report['annual']:
        assert row['eval_rows'] == 3 and row['disposal_known_rows'] == 2
        assert row['disposal_known_ratio'] == pytest.approx(2 / 3)
        assert row['excluded_y5_alpha'] == {'rows': 2, 'finite_rows': 1, 'mean': .1}
        if has_volume:
            assert row['excluded_volume_proxy']['mean'] == 30.
        else:
            assert 'excluded_volume_proxy' not in row
    assert before == {p: p.read_bytes() for p in files}
    json.dumps(report, allow_nan=False)


def test_conflicting_duplicate_announcements_use_only_known_daily_key():
    status = pd.concat([frame(['2025-01-09'], '2025-01-08'),
                        frame(['2025-01-09'], '2025-01-10')])
    assert disposal.disposal_known(status, ['2025-01-08', '2025-01-09'], ['2330'] * 2).tolist() == [False, True]


def test_cli_rejects_output_outside_local_data(tmp_path):
    with pytest.raises(SystemExit):
        disposal.main(['--output', str(tmp_path / 'not_allowed.json')])
