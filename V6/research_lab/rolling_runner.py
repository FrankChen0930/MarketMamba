"""Roadmap §12.32 H1 collection only; RESEARCH_ONLY / UNMERGED / NOT_DEPLOYED.

Uses the existing C0+ recipe and 60-session sequence eligibility. No inference
or adoption decision is made here. Fold manifests describe the shared base,
not the derived splits. Historical-simulation labels and survivorship/PIT
limitations remain. Resume checks JSON readability only, not run identity:
use a distinct output directory when changing inputs or configuration.
Only one writer may own an output directory at a time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from .seq_runner import SeqData

LOCAL_DATA_ROOT = Path(__file__).resolve().parents[2] / '.local-data'
SYSTEMS = ('lightgbm_day_t', 'c0plus')


def run_fold_seed(fold_dir, seed, label='y5', max_train_rows=500_000,
                  n_boot=50, run_fn=None):
    """Collect daily IC and elapsed seconds for one fold/seed, without writes.

    normalize=None avoids generating a neural input cache; eligibility still
    uses SeqData's default 60-session window. Existing stock_idx.npy is reused.
    c0plus.run's exploratory diagnostics are discarded, never interpreted here.
    """
    if run_fn is None:
        from .c0plus import run
        run_fn = run
    started = time.perf_counter()
    data = SeqData(fold_dir, normalize=None)
    result = run_fn(fold_dir, label, max_train_rows, seed, n_boot=n_boot,
                    row_mask=data.eligible, log=lambda *args, **kwargs: None)
    return dict(fold=Path(fold_dir).name, seed=int(seed), label=label,
                wall_seconds=time.perf_counter() - started,
                fit_rows=int(result['fit_rows']), eval_rows=int(result['eval_rows']),
                daily_ic={name: {str(date): float(ic) for date, ic in result['daily_ic'][name].items()}
                          for name in SYSTEMS})


def _output_dir(out_dir):
    root = LOCAL_DATA_ROOT.resolve()
    out = Path(out_dir).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError('out_dir must be below repository .local-data/')
    return out


def _completed(path):
    if path.exists() or path.is_symlink():
        with path.open(encoding='utf-8') as stream:
            json.load(stream)  # Corrupt/unreadable results stop the run unchanged.
        return True
    return False


def _atomic_json(path, result):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_all(folds_root, years, seeds, out_dir, label='y5', max_train_rows=500_000,
            n_boot=50, run_fn=None, dry_run=False):
    """Collect years × seeds; readable JSON is resumed without recomputation.

    dry_run lists pending/completed jobs and counts, without loading matrices
    or creating directories. Exceptions propagate to the module CLI.
    """
    out = _output_dir(out_dir)
    years, seeds = tuple(years), tuple(seeds)
    total, skipped = len(years) * len(seeds), 0
    for year in years:
        for seed in seeds:
            path = out / f'fold_{year}_seed_{seed}.json'
            if _completed(path):
                skipped += 1
                print(f'SKIP fold_{year} seed={seed} {path}', flush=True)
                continue
            if dry_run:
                print(f'RUN fold_{year} seed={seed} {path}', flush=True)
                continue
            result = run_fold_seed(Path(folds_root) / f'fold_{year}', seed, label,
                                   max_train_rows, n_boot, run_fn)
            out.mkdir(parents=True, exist_ok=True)
            _atomic_json(path, result)
            print(f'DONE fold_{year} seed={seed} wall_seconds={result["wall_seconds"]:.3f}', flush=True)
    if dry_run:
        print(f'DRY_RUN total={total} pending={total - skipped} skipped={skipped}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds-root', type=Path, required=True)
    parser.add_argument('--years', type=int, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--label', choices=('y5', 'y10'), default='y5')
    parser.add_argument('--max-train-rows', type=int, default=500_000)
    parser.add_argument('--n-boot', type=int, default=50)
    parser.add_argument('--run', action='store_true', help='execute (default: dry run)')
    args = parser.parse_args(argv)
    run_all(args.folds_root, args.years, args.seeds, args.out_dir, args.label,
            args.max_train_rows, args.n_boot, dry_run=not args.run)


if __name__ == '__main__':
    main()
