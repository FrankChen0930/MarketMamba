"""Synthetic-only checks for roadmap 12.32 H1; never loads real results."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from research_lab import rolling_analysis as ra


def series(values, year=2021):
    return pd.Series(values, index=pd.bdate_range(f'{year}-01-01', periods=len(values)))


def write_run(root, year, seed, values, base=None):
    values = series(values, year)
    baseline = values * 0 if base is None else series(base, year)
    payload = dict(fold=f'fold_{year}', seed=seed, wall_seconds=1.25,
                   daily_ic={name: {str(d): float(x) for d, x in s.items()}
                             for name, s in [('lightgbm_day_t', baseline), ('c0plus', values)]})
    path = root / f'fold_{year}_seed_{seed}.json'
    path.write_text(json.dumps(payload))
    return path


def populate(root, effect=.02, descriptive=.02):
    rng = np.random.default_rng(81)
    for year in range(2021, 2027):
        common = rng.normal(0, .006, 90)
        for seed in (17, 29, 43):
            write_run(root, year, seed, common + rng.normal(0, .003, 90)
                      + (effect if year < 2024 else descriptive))


def test_common_days_and_pairing(tmp_path):
    a = write_run(tmp_path, 2021, 17, [.1, .2, .3])
    b = write_run(tmp_path, 2021, 29, [.3, .4, .5])
    data = json.loads(b.read_text())
    del data['daily_ic']['c0plus'][next(iter(data['daily_ic']['c0plus']))]
    b.write_text(json.dumps(data))
    results = ra.load_results(tmp_path)
    assert ra.system_daily(results, 'c0plus', 2021).tolist() == pytest.approx([.3, .4])
    assert ra.paired_diff(results, 2021).tolist() == pytest.approx([.3, .4])
    assert len(ra.system_daily(results, 'lightgbm_day_t', 2021)) == 3
    assert a.exists()


def test_system_seed_mismatch_and_duplicates(tmp_path):
    path = write_run(tmp_path, 2021, 17, [.1, .2])
    data = json.loads(path.read_text())
    del data['daily_ic']['c0plus']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='seed'):
        ra.load_results(tmp_path)
    path = write_run(tmp_path, 2021, 17, [.1, .2])
    (tmp_path / 'fold_2021_seed_017.json').write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='duplicate'):
        ra.load_results(tmp_path)


@pytest.mark.parametrize('n', [7, 64, 125, 216])
def test_hac_hand_formula_and_student_t(n):
    x = np.resize([-.03, .01, .02, .04, -.01, .07, .03], n)
    x = x - x.mean() + .001
    r = ra.hac_t(x)
    lag = max(4, math.floor(np.cbrt(n)))
    z = x - x.mean()
    variance_sum = sum(z * z)
    for k in range(1, min(lag, n - 1) + 1):
        variance_sum += 2 * (1 - k / (lag + 1)) * sum(z[k:] * z[:-k])
    se = math.sqrt(variance_sum) / n
    assert r == pytest.approx(dict(mean=x.mean(), se=se, t=x.mean()/se,
                                  p=2 * stats.t.sf(abs(x.mean()/se), n-1), lag=lag, n=n))
    assert r['p'] != pytest.approx(2 * stats.norm.sf(abs(r['t'])), rel=1e-7, abs=0)


def test_hac_validation_zero_variance_and_mde():
    assert ra.hac_t(np.zeros(8))['p'] == 1
    assert ra.hac_t(np.ones(8))['p'] == 0
    assert ra.hac_t(np.arange(8), lag=0)['lag'] == 0
    for x in ([], [1], [1, np.nan], [1, np.inf]):
        with pytest.raises(ValueError):
            ra.hac_t(x)
    assert ra.mde(.003) == pytest.approx((stats.norm.ppf(.975)+stats.norm.ppf(.8))*.003)


@pytest.mark.parametrize('lo,hi,point,want', [
    (.001, .01, .005, '證實有益'), (0, .01, .005, '不確定'),
    (.001, .01, .004999, '不確定'), (-.02, -.0001, -.01, '證實無益或劣於'),
    (-.006, .0049, 0, '證實無益或劣於'), (-.006, .005, 0, '不確定'),
    (-.005, .005, 0, '等價'), (-.005, .004, 0, '等價'),
    (-.006, 0, 0, '證實無益或劣於'), (0, .005, .005, '等價'),
    (.001, .004, .003, '等價'), (-.01, .01, 0, '不確定'),
])
def test_classification_boundaries(lo, hi, point, want):
    assert ra.classify(lo, hi, point) == want


def test_noninferiority_boundary_and_precision():
    assert ra.noninferiority(-.009) == '通過'
    assert ra.noninferiority(-.01) == '最近期無法證實'
    assert ra.noninferiority(-.009, point=0, se=.01) == '最近期無法證實'
    assert ra.noninferiority(-.009, point=.02, se=.001) == '通過'


def test_bootstrap_blocks_stay_inside_fold_and_day_weighting():
    # With a single full-length block, each selected fold must remain constant.
    # Fold-resampling counts give exactly {-1, 1/3, 1}, with unequal lengths.
    data = {2021: {17: series([-1.] * 5)}, 2022: {17: series([1.] * 10, 2022)}}
    r = ra.block_bootstrap_ci(data, n_boot=1000, block=10, seed=7)
    assert r['ci_low'] == -1 and r['ci_high'] == 1
    assert r['few_folds'] is True
    # One fold and block >= n: no day resampling uncertainty at all.
    one = {2021: {17: series([0., 0., 0., 0., 1.])}}
    r = ra.block_bootstrap_ci(one, n_boot=200, block=5, seed=0)
    assert r['ci_low'] == pytest.approx(.2) and r['ci_high'] == pytest.approx(.2)
    assert ra.block_bootstrap_ci(data, 1000, 10, 7) == ra.block_bootstrap_ci(data, 1000, 10, 7)


def test_bootstrap_seed_hierarchy_and_weighted_point():
    data = {2021: {17: series([0.] * 5), 29: series([.1] * 5)},
            2022: {17: series([.2] * 10, 2022), 29: series([.3] * 10, 2022)}}
    r = ra.block_bootstrap_ci(data, n_boot=2000, block=5, seed=5)
    assert r['point'] == pytest.approx((.05 * 5 + .25 * 10) / 15)
    assert r['ci_low'] < .05 and r['ci_high'] > .25
    with pytest.raises(ValueError):
        ra.block_bootstrap_ci({})


def test_white_noise_coverage_and_type_one_rate():
    rng = np.random.default_rng(90)
    rejected = 0
    for trial in range(40):
        data = {y: {s: series(rng.normal(0, .03, 100), y) for s in (17, 29, 43)}
                for y in (2021, 2022, 2023)}
        r = ra.block_bootstrap_ci(data, n_boot=300, seed=trial)
        rejected += not (r['ci_low'] <= 0 <= r['ci_high'])
        if trial == 0:
            assert r['ci_low'] <= 0 <= r['ci_high']
            assert ra.classify(r['ci_low'], r['ci_high'], r['point']) in ('不確定', '等價')
    assert rejected / 40 < .15


@pytest.mark.parametrize('effect,want', [(.02, '證實有益'), (-.02, '證實無益或劣於')])
def test_report_effects_json_and_descriptive_isolation(tmp_path, effect, want):
    populate(tmp_path, effect)
    positive = ra.report(tmp_path, n_boot=300, seed=8)
    assert positive['classification'] == want
    assert positive['evidence'] == 'E2'
    assert positive['status'] == 'RESEARCH_ONLY'
    assert positive['preregistration'] == 'roadmap 12.32 H1'
    assert positive['descriptive']['noninferiority'] == '通過'
    populate(tmp_path, effect, descriptive=-.02)
    negative = ra.report(tmp_path, n_boot=300, seed=8)
    assert negative['clean'] == positive['clean']
    assert negative['classification'] == positive['classification']
    assert negative['descriptive']['noninferiority'] == '最近期無法證實'
    assert all(row['descriptive_only'] for row in negative['per_fold'].values())
    json.dumps(negative, allow_nan=False)


def test_fixed_effect_hac_does_not_join_fold_boundaries(tmp_path):
    populate(tmp_path)
    result = ra.report(tmp_path, n_boot=50)
    folds = [ra.hac_t(ra.paired_diff(ra.load_results(tmp_path), y)) for y in (2021, 2022, 2023)]
    n = sum(r['n'] for r in folds)
    expected = math.sqrt(sum((r['n'] * r['se'])**2 for r in folds)) / n
    assert result['clean']['hac']['se'] == pytest.approx(expected)
    assert result['clean']['hac']['n'] == n
    assert result['seed_sd']['2021'] > 0


def test_missing_years_and_cli(tmp_path):
    with pytest.raises(ValueError):
        ra.load_results(tmp_path)
    populate(tmp_path)
    result = subprocess.run([sys.executable, '-m', 'research_lab.rolling_analysis',
                             '--out-dir', str(tmp_path)], text=True, capture_output=True,
                            env=os.environ.copy(), check=True)
    assert json.loads(result.stdout)['classification'] == '證實有益'
    (tmp_path / 'fold_2021_seed_17.json').unlink()
    (tmp_path / 'fold_2021_seed_29.json').unlink()
    (tmp_path / 'fold_2021_seed_43.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        ra.report(tmp_path, n_boot=20)
