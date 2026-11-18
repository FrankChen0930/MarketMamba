"""Offline roadmap 12.32 H1 statistics: E2 / RESEARCH_ONLY.

UNMERGED / NOT_DEPLOYED. H1 alone; no Romano–Wolf family/adoption decision.
Per task instruction, clean-fold classification and descriptive noninferiority
are separate. This differs from the roadmap's combined beneficial/adoption gate.
Historical-simulation proxy labels, survivorship bias and UNVERIFIED PIT remain.
Input hashes identify bytes only: rolling_runner does not establish full run,
feature, universe or training provenance. Never infer trading readiness here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

SYSTEMS = ('lightgbm_day_t', 'c0plus')


def _common(series):
    """Chronologically ordered finite support shared by every supplied seed."""
    frame = pd.concat(series, axis=1, join='inner').sort_index()
    frame = frame.loc[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)]
    if frame.empty:
        raise ValueError('no common finite days across seeds/systems')
    return frame


def load_results(out_dir, systems=SYSTEMS):
    """Return {year: {seed: record}}; reject duplicate identities/seed mismatch.

    Every present fold is validated, including unselected folds. Systems may
    omit dates, never entire seeds. JSON null/nonfinite ICs are unavailable days.
    Filenames must match payload identities; date support must belong to fold.
    """
    paths = sorted(Path(out_dir).glob('fold_*_seed_*.json'))
    if not paths:
        raise ValueError('no fold results found')
    results = {}
    for path in paths:
        match = re.fullmatch(r'fold_(\d{4})_seed_(\d+)\.json', path.name)
        if match is None:
            raise ValueError(f'invalid result filename: {path.name}')
        year, seed = map(int, match.groups())
        raw = path.read_bytes()
        record = json.loads(raw)
        if seed in results.get(year, {}):
            raise ValueError(f'duplicate fold/seed: {year}/{seed}')
        if record.get('fold') != f'fold_{year}' or type(record.get('seed')) is not int or record['seed'] != seed:
            raise ValueError(f'filename/payload identity mismatch: {path.name}')
        wall = record.get('wall_seconds')
        if not isinstance(wall, (int, float)) or isinstance(wall, bool) or not math.isfinite(wall) or wall < 0:
            raise ValueError(f'invalid wall_seconds: {path.name}')
        if record.get('label', 'y5') != 'y5':
            raise ValueError('H1 requires y5 labels')
        daily = record.get('daily_ic')
        if not isinstance(daily, dict) or (set(daily) != set(SYSTEMS) if systems == SYSTEMS else not set(systems).issubset(daily)):
            raise ValueError(f'system seed sets mismatch: {path.name}')
        parsed = {}
        for system in systems:
            if not isinstance(daily[system], dict) or not daily[system]:
                raise ValueError(f'missing system seed data: {path.name}/{system}')
            s = pd.Series(daily[system], dtype=float)
            s.index = pd.to_datetime(s.index, errors='raise')
            if s.index.has_duplicates or s.index.tz is not None or not (s.index.year == year).all():
                raise ValueError(f'invalid/duplicate fold dates: {path.name}')
            if (s[np.isfinite(s)].abs() > 1).any():
                raise ValueError('IC must be in [-1, 1]')
            parsed[system] = s.sort_index()
        results.setdefault(year, {})[seed] = dict(
            daily_ic=parsed, wall_seconds=float(wall), path=str(path),
            sha256=hashlib.sha256(raw).hexdigest())
    return results


def system_daily(results, system, year):
    """Average seeds on dates with finite IC in every seed of this fold/system."""
    return _common({s: r['daily_ic'][system] for s, r in results[year].items()}).mean(axis=1)


def paired_diff(results, year, candidate='c0plus', baseline='lightgbm_day_t'):
    """C0+ minus baseline, aligned after each system's seed average."""
    both = _common({s: system_daily(results, s, year) for s in (baseline, candidate)})
    return (both[candidate] - both[baseline]).rename('diff')


def _values(diff):
    values = np.asarray(diff, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('need at least two finite daily differences')
    return values


def _stat(mean, se, n):
    if se == 0:
        return (0., 1.) if mean == 0 else (math.copysign(math.inf, mean), 0.)
    t = mean / se
    return float(t), float(2 * stats.t.sf(abs(t), df=n - 1))


def hac_t(diff, lag=None):
    """Bartlett Newey–West mean SE; two-sided Student t with n−1 df.

    Autocovariances use denominator n, without a finite-sample correction.
    Default bandwidth max(4, floor(cuberoot(n))) is reported even for short
    series; only observable autocovariances contribute.
    """
    d = _values(diff)
    n = len(d)
    lag = max(4, math.floor(np.cbrt(n))) if lag is None else lag
    if not isinstance(lag, (int, np.integer)) or isinstance(lag, bool) or lag < 0:
        raise ValueError('lag must be a nonnegative integer')
    mean = float(d.mean())
    z = d - mean
    total = float(z @ z)
    for k in range(1, min(lag, n-1) + 1):
        total += 2 * (1 - k / (lag + 1)) * float(z[k:] @ z[:-k])
    se = math.sqrt(max(total, 0.)) / n
    if np.ptp(d) == 0:
        se = 0.
    t, p = _stat(mean, se, n)
    return dict(mean=mean, se=se, t=t, p=p, lag=int(lag), n=n)


def _seed_diffs(results, year, candidate='c0plus', baseline='lightgbm_day_t'):
    # Freeze common support before seed resampling: no changing missing-day mask.
    dates = paired_diff(results, year, candidate, baseline).index
    return {seed: (r['daily_ic'][candidate] - r['daily_ic'][baseline]).loc[dates]
            for seed, r in results[year].items()}


def block_bootstrap_ci(diff_by_year, n_boot=2000, block=5, seed=0):
    """Hierarchical percentile CI: folds → seeds → moving day blocks.

    Accept {year: {seed: Series}}. Intersect seed support before resampling.
    Sample the original fold count, then its original seed count, with replacement.
    Within each selected fold use shared day indices for all sampled seeds,
    preserving their paired market shocks. Noncircular moving blocks never wrap,
    including at fold boundaries; block is capped at that fold's length.
    Seed means are computed first, then fold means weighted by sampled day count.
    """
    if not diff_by_year:
        raise ValueError('need at least one fold')
    if any(type(x) is not int or x < 1 for x in (n_boot, block)):
        raise ValueError('n_boot and block must be positive integers')
    arrays = []
    for year in sorted(diff_by_year):
        if not diff_by_year[year]:
            raise ValueError('need seeds in every fold')
        frame = _common(diff_by_year[year])
        if frame.index.has_duplicates:
            raise ValueError('duplicate daily index')
        arrays.append(frame.to_numpy(float).T)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        total, days = 0., 0
        for chosen in rng.integers(0, len(arrays), size=len(arrays)):
            a = arrays[chosen]
            ns, n = a.shape
            sampled_seeds = rng.integers(0, ns, size=ns)
            width = min(block, n)
            starts = rng.integers(0, n - width + 1, size=math.ceil(n / width))
            indices = (starts[:, None] + np.arange(width)).reshape(-1)[:n]
            daily = a[sampled_seeds][:, indices].mean(axis=0)
            total += float(daily.sum())
            days += n
        means[i] = total / days
    lo, hi = np.quantile(means, [.025, .975])
    point = sum(float(a.mean(axis=0).sum()) for a in arrays) / sum(a.shape[1] for a in arrays)
    return dict(point=point, ci_low=float(lo), ci_high=float(hi),
                few_folds=len(arrays) < 3, n_boot=n_boot, block=block, seed=seed)


def classify(ci_low, ci_high, point, sesoi=.005):
    """Ordered rules: 有益 → 無益或劣於 → 等價 → 不確定.

    有益: low>0 and point>=SESOI; 無益或劣於: high<0 or
    (high<SESOI and low<−SESOI); 等價: low>=−SESOI and high<=SESOI.
    The first satisfied rule wins, including overlapping boundary cases.
    """
    if not all(math.isfinite(x) for x in (ci_low, ci_high, point, sesoi)) or ci_low > ci_high or sesoi <= 0:
        raise ValueError('invalid CI/point/SESOI')
    if ci_low > 0 and point >= sesoi:
        return '證實有益'
    if ci_high < 0 or (ci_high < sesoi and ci_low < -sesoi):
        return '證實無益或劣於'
    if ci_low >= -sesoi and ci_high <= sesoi:
        return '等價'
    return '不確定'


def mde(se, power=.8, alpha=.05):
    """Two-sided normal-approximation minimum detectable effect, in IC units."""
    if not math.isfinite(se) or se < 0 or not .5 < power < 1 or not 0 < alpha < 1:
        raise ValueError('invalid SE/power/alpha')
    return float((stats.norm.ppf(1-alpha/2) + stats.norm.ppf(power)) * se)


def noninferiority(ci_low, delta0=-.01, *, point=None, se=None):
    """Descriptive lower bound must strictly exceed delta0; precision cannot pass.

    Optional point and se must be supplied together. If point−1.96*se<=delta0,
    return 最近期無法證實 even when a different supplied CI would pass.
    """
    if not all(math.isfinite(x) for x in (ci_low, delta0)):
        raise ValueError('nonfinite noninferiority bound')
    if (point is None) != (se is None):
        raise ValueError('point and se must be supplied together')
    if point is not None:
        if not math.isfinite(point) or not math.isfinite(se) or se < 0:
            raise ValueError('invalid point/SE')
        if point <= delta0 + 1.96 * se:
            return '最近期無法證實'
    return '通過' if ci_low > delta0 else '最近期無法證實'


def _fixed_hac(results, years, candidate='c0plus', baseline='lightgbm_day_t'):
    rows = {str(y): hac_t(paired_diff(results, y, candidate, baseline)) for y in years}
    n = sum(r['n'] for r in rows.values())
    mean = sum(r['mean'] * r['n'] for r in rows.values()) / n
    se = math.sqrt(sum((r['n'] * r['se'])**2 for r in rows.values())) / n
    t, p = _stat(mean, se, n)
    critical = float(stats.t.ppf(.975, n-1))
    return dict(mean=mean, se=se, t=t, p=p, n=n, df=n-1,
                lag_by_year={y: r['lag'] for y, r in rows.items()},
                ci_low=mean-critical*se, ci_high=mean+critical*se,
                method='day-count weighted fixed effects; independent folds'), rows


def _json_safe(value):
    # Degenerate zero-SE statistics are undefined/infinite; strict JSON uses null.
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def report(out_dir, clean_years=(2021, 2022, 2023),
           descriptive_years=(2024, 2025, 2026), sesoi=.005, delta0=-.01,
           n_boot=2000, seed=0, candidate='c0plus', baseline='lightgbm_day_t',
           candidate_dir=None, baseline_dir=None):
    """Return strict JSON-compatible H1 statistics, with no adoption decision.

    Missing selected folds fail closed. Per-fold t tests are descriptive only.
    Clean classification uses hierarchical bootstrap CI; fixed-effect HAC
    assumes independent folds and combines their n-weighted sampling variances.
    Seed SD is the sample SD of each fold's per-seed mean on common support;
    one seed has unknown SD (null). Descriptive CI is fixed-effect HAC, with
    the additional requested point−1.96*SE precision gate.
    """
    if candidate_dir is None and baseline_dir is None and (baseline, candidate) == SYSTEMS:
        results = load_results(out_dir)
    else:
        candidate_results = load_results(candidate_dir or out_dir, (candidate,))
        results = load_results(baseline_dir or out_dir, (baseline,))
        identities = lambda records: {(y, s) for y, fold in records.items() for s in fold}
        if identities(results) != identities(candidate_results):
            raise ValueError('candidate/baseline fold/seed mismatch')
        for y, fold in results.items():
            for s, record in fold.items():
                other = candidate_results[y][s]
                record['daily_ic'][candidate] = other['daily_ic'][candidate]
                record['candidate_input'] = {k: other[k] for k in ('path', 'sha256', 'wall_seconds')}
    clean_years, descriptive_years = tuple(clean_years), tuple(descriptive_years)
    selected = clean_years + descriptive_years
    if not clean_years or not descriptive_years or len(set(selected)) != len(selected):
        raise ValueError('need nonempty, disjoint, unique clean/descriptive years')
    missing = set(selected) - results.keys()
    if missing:
        raise ValueError(f'missing selected folds: {sorted(missing)}')
    clean, clean_rows = _fixed_hac(results, clean_years, candidate, baseline)
    desc, desc_rows = _fixed_hac(results, descriptive_years, candidate, baseline)
    boot = block_bootstrap_ci({y: _seed_diffs(results, y, candidate, baseline) for y in clean_years},
                              n_boot=n_boot, seed=seed)
    seed_sd = {}
    for year in selected:
        means = [float(s.mean()) for s in _seed_diffs(results, year, candidate, baseline).values()]
        seed_sd[str(year)] = float(np.std(means, ddof=1)) if len(means) > 1 else None
    return _json_safe(dict(
        evidence='E2', status='RESEARCH_ONLY', preregistration='roadmap 12.32 H1',
        implementation_status='UNMERGED / NOT_DEPLOYED',
        classification=classify(boot['ci_low'], boot['ci_high'], clean['mean'], sesoi),
        clean=dict(years=list(clean_years), hac=clean, bootstrap=boot, mde=mde(clean['se'])),
        descriptive=dict(years=list(descriptive_years), hac=desc,
                         noninferiority=noninferiority(desc['ci_low'], delta0,
                                                       point=desc['mean'], se=desc['se'])),
        per_fold={y: dict(r, descriptive_only=True) for y, r in (clean_rows | desc_rows).items()},
        seed_sd=seed_sd, sesoi=sesoi, delta0=delta0,
        inputs=[dict(year=y, seed=s, path=r['path'], sha256=r['sha256'],
                     wall_seconds=r['wall_seconds'],
                     **({'candidate_input': r['candidate_input']} if 'candidate_input' in r else {}))
                for y in sorted(results) for s, r in sorted(results[y].items())],
        limitations=['Clean classification separated from descriptive noninferiority per task; '
                     'roadmap combined adoption gate is not evaluated.',
                     'H1 only; no Romano-Wolf family correction or adoption authorization.',
                     'HISTORICAL_SIMULATION_PROXY; survivorship bias; UNVERIFIED PIT.',
                     'Input byte hashes do not validate upstream run provenance.',
                     'Nonfinite or infinite report statistics are JSON null.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.out_dir), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
