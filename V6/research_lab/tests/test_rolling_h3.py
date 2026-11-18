"""Synthetic H3 checks: no real training or formal analysis."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_lab import rolling_h3 as h
from research_lab import rolling_analysis as a


def record(year, seed, gain=.02):
    baseline = {f'{year}-01-{d:02}': .05+d*.0001 for d in range(1, 21)}
    return dict(fold=f'fold_{year}', seed=seed, label='y5', wall_seconds=1.,
                fit_rows=100, eval_rows=20, excluded_rows=2,
                daily_ic={'lightgbm_day_t': baseline,
                          'c0plus': {d: v+gain for d, v in baseline.items()}})


def test_mask_excludes_known_and_preserves_inputs(tmp_path, monkeypatch):
    fold = tmp_path/'fold_2022'
    fold.mkdir()
    # First-occurrence order differs from alphabetical order.
    data = SimpleNamespace(dates=pd.date_range('2022-01-03', periods=3),
                           date_idx=np.repeat(np.arange(3), 2),
                           stock_idx=np.tile([1, 0], 3), n_stocks=2,
                           eligible=np.array([False, False, True, True, True, False]))
    np.save(fold/'stock_idx.npy', data.stock_idx)
    np.save(fold/'stock_ids.npy', np.tile(['B', 'A'], 3))
    status = pd.DataFrame({'Date': ['2022-01-04', '2022-01-05'],
                           'stock_id': ['B', 'A'], 'announced': ['2022-01-03']*2,
                           'status': ['disposal']*2})
    original = status.copy(deep=True)
    mask = data.eligible.copy()
    before = {p.name: p.read_bytes() for p in fold.iterdir()}
    monkeypatch.setattr(h, 'SeqData', lambda *args, **kw: data)
    def run(directory, label, max_train_rows, seed, **kw):
        assert (directory, label, max_train_rows, seed, kw['n_boot']) == (fold, 'y5', 500_000, 17, 50)
        np.testing.assert_array_equal(kw['row_mask'], [False, False, False, True, True, False])
        return record(2022, 17)
    result = h.run_fold_seed(fold, 17, status, run_fn=run)
    assert result['excluded_rows'] == 1  # known but ineligible A is not counted
    assert set(result['daily_ic']) == set(a.SYSTEMS)
    pd.testing.assert_frame_equal(status, original)
    np.testing.assert_array_equal(data.eligible, mask)
    assert before == {p.name: p.read_bytes() for p in fold.iterdir()}


def test_requires_cache_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'SeqData', lambda *a, **k: pytest.fail('must not load/create caches'))
    with pytest.raises(ValueError, match='stock_idx'):
        h.run_fold_seed(tmp_path, 17, pd.DataFrame(), run_fn=lambda: None)


def test_output_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(h.rr, 'LOCAL_DATA_ROOT', tmp_path)
    for out in (tmp_path, tmp_path.parent/'outside', tmp_path/'rolling_h1', tmp_path/'rolling_h1/sub'):
        with pytest.raises(ValueError):
            h.run_all(tmp_path, out, tmp_path/'missing')
    alias = tmp_path/'alias'
    alias.symlink_to(tmp_path/'rolling_h1', target_is_directory=True)
    with pytest.raises(ValueError):
        h.run_all(tmp_path, alias, tmp_path/'missing')


def test_dry_run_resume_atomic_and_corruption(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(h.rr, 'LOCAL_DATA_ROOT', tmp_path)
    calls = []
    monkeypatch.setattr(h.pd, 'read_parquet', lambda p: pd.DataFrame())
    monkeypatch.setattr(h, 'run_fold_seed', lambda fold, seed, *a: calls.append(seed) or record(2021, seed))
    out = tmp_path/'h3'
    h.run_all(tmp_path, out, tmp_path/'missing')
    assert not calls and not out.exists()
    assert 'DRY_RUN total=30 pending=30 skipped=0' in capsys.readouterr().out
    h.run_all(tmp_path, out, tmp_path/'status', years=[2021], seeds=[17], dry_run=False)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    h.run_all(tmp_path, out, tmp_path/'status', years=[2021], seeds=[17], dry_run=False)
    assert calls == [17] and before == {p.name: p.read_bytes() for p in out.iterdir()}
    (out/'fold_2021_seed_17.json').write_text('{')
    with pytest.raises(ValueError):
        h.run_all(tmp_path, out, tmp_path/'status', dry_run=False)
    assert calls == [17]


def test_cli_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(h, 'run_all', lambda *args, **kw: calls.append((args, kw)))
    h.main(['--folds-root', str(tmp_path), '--out-dir', str(tmp_path/'h3'), '--status-path', 'missing'])
    assert calls[0][1]['dry_run'] is True


@pytest.fixture
def sources(tmp_path):
    paths = [tmp_path/name for name in ('h1', 'h3')]
    for path, gain in zip(paths, (.02, .012)):
        path.mkdir()
        for year in h.YEARS:
            for seed in h.SEEDS:
                (path/f'fold_{year}_seed_{seed}.json').write_text(json.dumps(record(year, seed, gain)))
    return paths


def test_complete_report_absolute_ic_and_read_only(sources):
    h1, h3 = sources
    before = {p: p.read_bytes() for root in sources for p in root.iterdir()}
    actual = h.analyze(h1, h3, n_boot=30)
    expected = a.report(h3, n_boot=30)
    for key in ('clean', 'descriptive', 'classification', 'per_fold', 'seed_sd', 'inputs'):
        assert actual[key] == expected[key]
    assert actual['h3']['h1_point'] == a.report(h1, n_boot=30)['clean']['hac']['mean']
    assert actual['h3']['verdict'] == '穩健'
    for row in actual['absolute_ic_difference'].values():
        assert row['difference'] == pytest.approx(-.008)
        assert row['descriptive_only'] is True and row['n'] == 20
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('source', [0, 1])
def test_missing_pair_rejected(sources, source):
    (sources[source]/'fold_2026_seed_97.json').unlink()
    with pytest.raises(ValueError, match='fold/seed'):
        h.analyze(*sources)


@pytest.mark.parametrize('point,classification,expected', [
    (.01, '證實有益', '穩健'), (.009999999, '證實有益', '對處置宇宙敏感'),
    (.02, '不確定', '對處置宇宙敏感')])
def test_half_boundary(sources, monkeypatch, point, classification, expected):
    def report(path, **kw):
        return dict(clean={'hac': {'mean': .02 if path == sources[0] else point}},
                    classification=classification, inputs=[], limitations=[])
    monkeypatch.setattr(a, 'report', report)
    result = h.analyze(*sources)['h3']
    assert result['half_h1'] == .01
    assert result['at_least_half_h1'] == (point >= .01)
    assert result['verdict'] == expected
