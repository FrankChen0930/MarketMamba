"""Annual expanding-window matrix views; RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Only splits.npy is generated. All other files, including manifest.json, are
shared by absolute symlinks. The manifest retains the BASE identity/split
contract, not the derived fold identity; CLI JSON describes the fold instead.
Sessions are the observed union of dates, not a qualified exchange calendar.
This utility does not certify label maturity, PIT, universe or tradability.
"""
from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
import shutil

import numpy as np


def _arrays(dates, splits):
    dates, splits = np.asarray(dates), np.asarray(splits)
    if dates.ndim != 1 or splits.ndim != 1 or len(dates) != len(splits) or not len(dates):
        raise ValueError('dates and splits must be nonempty, equally sized 1D arrays')
    if dates.dtype.kind != 'M' or np.isnat(dates).any():
        raise ValueError('dates must be datetime64 without NaT')
    if splits.dtype.kind not in 'iu' or not np.isin(splits, [0, 1, 2, 3, 4]).all():
        raise ValueError('splits must contain integer codes 0/1/2/3/4')
    return dates.astype('datetime64[D]'), splits


def _validate_label_gap(train_last, eval_first, h_max):
    """The exit at t+h+1 must strictly precede the first evaluation signal."""
    if not (train_last + (h_max + 1) < eval_first):
        raise ValueError('label exit overlaps evaluation: train_last + h_max + 1 must be < eval_first')


def make_fold_splits(dates, base_splits, fold_year, purge_sessions=20, label_horizons=(5, 10)):
    """Return uint8 split codes using sorted unique observed trading sessions.

    Train signals have session < eval_first - purge_sessions and base code != 0.
    Pre-evaluation warmup rows remain 0, even in the purge interval. All rows
    in the requested year are evaluation rows, regardless of their base code;
    later years are outside the fold (3). Neither input is mutated.
    """
    horizons = tuple(label_horizons)
    if not horizons or any(not isinstance(h, Integral) or isinstance(h, bool) or h < 1 for h in horizons):
        raise ValueError('label_horizons must be nonempty positive integers')
    h_max = int(max(horizons))
    if not isinstance(purge_sessions, Integral) or isinstance(purge_sessions, bool) or purge_sessions < h_max + 1:
        raise ValueError('purge_sessions must be an integer >= max(label_horizons) + 1')
    if not isinstance(fold_year, Integral) or isinstance(fold_year, bool) or not 1 <= fold_year <= 9999:
        raise ValueError('fold_year must be an integer in 1..9999')
    dates, base = _arrays(dates, base_splits)
    _, row_session = np.unique(dates, return_inverse=True)
    in_year = dates.astype('datetime64[Y]') == np.datetime64(str(fold_year).zfill(4), 'Y')
    if not in_year.any():
        raise ValueError(f'no evaluation data for year {fold_year}')
    eval_first = int(row_session[in_year].min())
    before = row_session < eval_first
    train = (row_session < eval_first - purge_sessions) & (base != 0)
    if not train.any():
        raise ValueError(f'no training rows before year {fold_year} after purge')
    train_last = int(row_session[train].max())
    _validate_label_gap(train_last, eval_first, h_max)
    out = np.full(len(dates), 3, dtype=np.uint8)
    out[before & (row_session > train_last)] = 4
    out[before & (base == 0)] = 0
    out[train] = 1
    out[in_year] = 2
    return out


def fold_summary(splits, dates):
    """JSON-ready row counts, train/evaluation date bounds and eval sessions."""
    dates, splits = _arrays(dates, splits)
    summary = {f'{name}_rows': int(np.count_nonzero(splits == code))
               for name, code in [('warmup', 0), ('train', 1), ('eval', 2), ('outside', 3), ('purge', 4)]}
    for name, code in [('train', 1), ('eval', 2)]:
        selected = dates[splits == code]
        summary[f'{name}_first'] = str(selected.min()) if len(selected) else None
        summary[f'{name}_last'] = str(selected.max()) if len(selected) else None
    summary['eval_sessions'] = int(len(np.unique(dates[splits == 2])))
    return summary


def _load_fold(base_dir, fold_year, purge_sessions, label_horizons):
    base = Path(base_dir).resolve(strict=True)
    dates = np.load(base / 'dates.npy', mmap_mode='r', allow_pickle=False)
    original = np.load(base / 'splits.npy', mmap_mode='r', allow_pickle=False)
    splits = make_fold_splits(dates, original, fold_year, purge_sessions, label_horizons)
    return base, dates, splits


def build_fold_dir(base_dir, out_dir, fold_year, purge_sessions=20, label_horizons=(5, 10)):
    """Create a fresh matrix view; refuse existing paths, including broken links.

    Validate before creating anything and remove only this call's new directory
    on failure. Base files are opened read-only; large arrays are never copied.
    Returns the fold summary.
    """
    out = Path(out_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'fold output already exists: {out}')
    base, dates, splits = _load_fold(base_dir, fold_year, purge_sessions, label_horizons)
    links = [(p.name, p.resolve(strict=True)) for p in sorted(base.iterdir())
             if p.name != 'splits.npy' and p.is_file()]
    summary = fold_summary(splits, dates)
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name, source in links:
            (out / name).symlink_to(source)
        # Exclusive creation ensures we never follow/overwrite a splits symlink.
        with (out / 'splits.npy').open('xb') as stream:
            np.save(stream, splits, allow_pickle=False)
    except BaseException:
        shutil.rmtree(out)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--years', type=int, nargs='+', required=True)
    parser.add_argument('--base-for', action='append', default=[], metavar='YEAR=PATH')
    parser.add_argument('--purge-sessions', type=int, default=20)
    parser.add_argument('--label-horizons', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--write', action='store_true', help='create directories (default: dry run)')
    args = parser.parse_args(argv)
    overrides = {}
    for value in args.base_for:
        year, sep, path = value.partition('=')
        try:
            number = int(year)
        except ValueError:
            parser.error('--base-for requires YEAR=PATH')
        if not sep or not path or number in overrides:
            parser.error('--base-for requires a unique YEAR=PATH')
        overrides[number] = Path(path)
    if len(set(args.years)) != len(args.years) or set(overrides) - set(args.years):
        parser.error('years must be unique and include every --base-for year')
    results = []
    # Preflight every fold before writing any directory.
    for year in args.years:
        base, dates, splits = _load_fold(overrides.get(year, args.base), year,
                                         args.purge_sessions, args.label_horizons)
        out = args.out_root / f'fold_{year}'
        if args.write and (out.exists() or out.is_symlink()):
            raise FileExistsError(f'fold output already exists: {out}')
        results.append(dict(fold_year=year, base=str(base), out_dir=str(out.absolute()),
                            purge_sessions=args.purge_sessions, label_horizons=args.label_horizons,
                            **fold_summary(splits, dates)))
    if args.write:
        for result in results:
            build_fold_dir(result['base'], result['out_dir'], result['fold_year'],
                           args.purge_sessions, args.label_horizons)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
