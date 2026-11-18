"""Synthetic, offline contracts for rolling matrix views."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from research_lab import rolling_folds as rf


@pytest.fixture
def sample():
    # Weekends AND a holiday are absent; rows are deliberately not date-sorted.
    sessions = np.busday_offset('2021-11-01', np.arange(90), holidays=['2021-12-24'])
    dates = np.repeat(sessions.astype('datetime64[D]'), 2)[::-1].copy()
    base = np.full(len(dates), 3, dtype=np.uint8)
    base[dates < sessions[5]] = 0
    return dates, base


def test_splits_session_gap_conservation_warmup_and_determinism(sample):
    dates, base = sample
    before = base.copy()
    splits = rf.make_fold_splits(dates, base, 2022)
    sessions, idx = np.unique(dates, return_inverse=True)
    first = idx[splits == 2].min()
    last = idx[splits == 1].max()
    assert last == first - 21
    assert last + 11 < first
    assert np.unique(idx[splits == 4]).size == 20
    assert (dates[splits == 2].astype('datetime64[Y]') == np.datetime64('2022')).all()
    assert not np.intersect1d(dates[splits == 1], dates[splits == 2]).size
    assert sum(np.count_nonzero(splits == code) for code in range(5)) == len(dates)
    assert (splits[base == 0] == 0).all()
    assert splits.dtype == np.uint8
    np.testing.assert_array_equal(base, before)
    np.testing.assert_array_equal(splits, rf.make_fold_splits(dates, base, 2022))
    # A calendar-day cutoff would select a different last training session.
    assert sessions[last] < sessions[first] - np.timedelta64(21, 'D')


def test_partial_year_and_summary(sample):
    dates, base = sample
    splits = rf.make_fold_splits(dates, base, 2022)
    summary = rf.fold_summary(splits, dates)
    assert summary['eval_first'] == '2022-01-03'
    assert summary['eval_last'] < '2022-12-31'
    assert summary['eval_sessions'] == len(np.unique(dates[splits == 2]))
    for name, code in [('warmup', 0), ('train', 1), ('eval', 2), ('outside', 3), ('purge', 4)]:
        assert summary[name + '_rows'] == int((splits == code).sum())
    assert summary['train_first'] == str(dates[splits == 1].min())
    assert summary['train_last'] == str(dates[splits == 1].max())


def test_future_rows_and_old_purge_become_train(sample):
    dates, base = sample
    base[base != 0] = 4
    dates = np.concatenate([dates, np.array(['2023-01-03'], dtype='datetime64[D]')])
    base = np.append(base, np.uint8(2))
    splits = rf.make_fold_splits(dates, base, 2022, purge_sessions=11)
    assert splits[-1] == 3
    assert (splits[dates == np.datetime64('2021-11-15')] == 1).all()


@pytest.mark.parametrize('year', [1999, 2021, 2023])
def test_missing_year_or_training_rejected(sample, year):
    with pytest.raises(ValueError):
        rf.make_fold_splits(*sample, year)


@pytest.mark.parametrize('purge', [-1, 0, 10])
def test_small_purge_rejected(sample, purge):
    with pytest.raises(ValueError, match='purge'):
        rf.make_fold_splits(*sample, 2022, purge_sessions=purge)


def test_minimum_purge_and_custom_horizon(sample):
    dates, base = sample
    for horizons, purge in [((5, 10), 11), ((20,), 21)]:
        splits = rf.make_fold_splits(dates, base, 2022, purge, horizons)
        _, idx = np.unique(dates, return_inverse=True)
        assert idx[splits == 1].max() + max(horizons) + 1 < idx[splits == 2].min()
    with pytest.raises(ValueError):
        rf.make_fold_splits(dates, base, 2022, 20, (20,))


def test_exit_session_equality_is_rejected():
    # Direct boundary test: the configuration guard makes equality unreachable
    # in normal builds, but the independent label-exit assertion must stay strict.
    with pytest.raises(ValueError, match='label'):
        rf._validate_label_gap(10, 21, 10)
    rf._validate_label_gap(10, 22, 10)


@pytest.mark.parametrize('kind', ['nat', 'length', 'unknown', 'float', 'empty'])
def test_invalid_inputs_fail_closed(sample, kind):
    dates, base = (x.copy() for x in sample)
    if kind == 'nat':
        dates[0] = np.datetime64('NaT', 'D')
    elif kind == 'length':
        base = base[:-1]
    elif kind == 'unknown':
        base[0] = 9
    elif kind == 'float':
        base = base.astype(float)
    else:
        dates, base = dates[:0], base[:0]
    with pytest.raises(ValueError):
        rf.make_fold_splits(dates, base, 2022)


@pytest.fixture
def matrix(tmp_path, sample):
    base = tmp_path / 'base'
    base.mkdir()
    dates, splits = sample
    np.save(base / 'dates.npy', dates)
    np.save(base / 'splits.npy', splits)
    np.save(base / 'X.npy', np.zeros((len(dates), 2)))
    (base / 'manifest.json').write_text(json.dumps({'rows': len(dates)}))
    (base / 'extra.tag').write_text('share every file')
    (base / 'linked.tag').symlink_to(base / 'extra.tag')
    return base


def hashes(base):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}


def test_build_links_all_files_and_preserves_base(matrix, tmp_path):
    before = hashes(matrix)
    out = tmp_path / 'fold'
    rf.build_fold_dir(matrix, out, 2022)
    assert hashes(matrix) == before
    assert {p.name for p in out.iterdir()} == set(before)
    for source in matrix.iterdir():
        target = out / source.name
        if source.name != 'splits.npy':
            assert os.path.islink(target)
            assert Path(os.readlink(target)).is_absolute()
            assert os.path.samefile(source, target)
        else:
            assert not os.path.islink(target)
            assert not os.path.samefile(source, target)
    np.testing.assert_array_equal(np.load(out / 'splits.npy'),
                                  rf.make_fold_splits(*sample_arrays(matrix), 2022))
    with pytest.raises(FileExistsError):
        rf.build_fold_dir(matrix, out, 2022)
    assert hashes(matrix) == before


def sample_arrays(base):
    return np.load(base / 'dates.npy'), np.load(base / 'splits.npy')


def test_existing_empty_directory_and_broken_symlink_refused(matrix, tmp_path):
    out = tmp_path / 'existing'
    out.mkdir()
    with pytest.raises(FileExistsError):
        rf.build_fold_dir(matrix, out, 2022)
    out.rmdir()
    out.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        rf.build_fold_dir(matrix, out, 2022)


def test_cli_dry_run_override_and_write(matrix, tmp_path, capsys):
    root = tmp_path / 'output'
    args = ['--base', str(tmp_path / 'absent'), '--base-for', f'2022={matrix}',
            '--out-root', str(root), '--years', '2022']
    rf.main(args)
    dry = json.loads(capsys.readouterr().out)
    assert dry[0]['fold_year'] == 2022
    assert dry[0]['base'] == str(matrix.resolve())
    assert not root.exists()
    rf.main([*args, '--write'])
    assert json.loads(capsys.readouterr().out) == dry
    assert (root / 'fold_2022' / 'splits.npy').is_file()


def test_invalid_build_leaves_no_directory(matrix, tmp_path):
    out = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        rf.build_fold_dir(matrix, out, 1999)
    assert not out.exists()
