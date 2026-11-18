import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research_lab import rolling_h2 as h
from research_lab import rolling_analysis as a


def record(year, seed, system='c0plus_fund', value=.03):
    return dict(fold=f'fold_{year}', seed=seed, label='y5', wall_seconds=1.,
                daily_ic={system: {f'{year}-01-{d:02}': value+d*.0001 for d in range(1, 21)}})


def test_runner_wiring(monkeypatch, tmp_path):
    dates = pd.bdate_range('2020-01-01', periods=300)
    data = SimpleNamespace(dates=dates, date_idx=np.repeat(np.arange(300), 2),
                           stock_idx=np.tile([0, 1], 300), eligible=np.ones(600, bool), n_stocks=2)
    fold = tmp_path/'fold_2022'
    fold.mkdir()
    np.save(fold/'stock_idx.npy', data.stock_idx)
    np.save(fold/'stock_ids.npy', np.tile(['A', 'B'], 300))
    monkeypatch.setattr(h, 'SeqData', lambda *args, **kwargs: data)
    def make(fin, info, d, ids):
        assert list(ids) == ['A', 'B']
        return lambda data, rows: (np.zeros((len(rows), 3), np.float32), ['GM', 'OM', 'EPS_YoY'])
    monkeypatch.setattr(h, 'make_extra', make)
    def run(directory, label, max_train_rows, seed, **kw):
        assert (directory, label, max_train_rows, seed) == (fold, 'y5', 500_000, 17)
        assert kw['n_boot'] == 50 and kw['row_mask'] is data.eligible
        assert kw['extra'](data, np.arange(2))[0].shape == (2, 3)
        return dict(fit_rows=100, eval_rows=20, daily_ic={'c0plus': {'2022-01-03': .1}, 'other': {}})
    r = h.run_fold_seed(fold, 17, pd.DataFrame(), pd.DataFrame(), run_fn=run)
    assert set(r['daily_ic']) == {'c0plus_fund'} and r['wall_seconds'] >= 0


def test_dry_resume_corruption_and_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(h.rr, 'LOCAL_DATA_ROOT', tmp_path)
    out = tmp_path/'h2'
    calls = []
    monkeypatch.setattr(h, 'run_fold_seed', lambda fold, seed, *args, **kwargs: calls.append((fold, seed)) or record(2022, seed))
    monkeypatch.setattr(h, '_load_financials', lambda p: (None, None))
    h.run_all(tmp_path, out, years=[2022], seeds=[17])
    assert not calls and not out.exists()
    h.run_all(tmp_path, out, years=[2022], seeds=[17], dry_run=False)
    h.run_all(tmp_path, out, years=[2022], seeds=[17], dry_run=False)
    assert len(calls) == 1 and len(list(out.iterdir())) == 1
    path = out/'fold_2022_seed_17.json'
    path.write_text('{}')
    with pytest.raises(ValueError):
        h.run_all(tmp_path, out, years=[2022], seeds=[17])
    path.write_text('{')
    with pytest.raises(ValueError):
        h.run_all(tmp_path, out, years=[2022], seeds=[17])
    with pytest.raises(ValueError):
        h.run_all(tmp_path, tmp_path.parent/'outside')


def test_cli_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(h, 'run_all', lambda *args, **kwargs: calls.append(kwargs))
    h.main(['--out-dir', str(tmp_path)])
    assert calls[-1]['dry_run'] is True
    h.main(['--out-dir', str(tmp_path), '--run'])
    assert calls[-1]['dry_run'] is False


def test_analysis_same_statistics_and_missing_pair(tmp_path):
    h1, h2, combined = [tmp_path/x for x in ('h1', 'h2', 'combined')]
    for p in (h1, h2, combined):
        p.mkdir()
    for y in h.YEARS:
        for seed in h.SEEDS:
            base = record(y, seed, 'c0plus', .01)
            base['daily_ic']['lightgbm_day_t'] = dict(base['daily_ic']['c0plus'])
            cand = record(y, seed, value=.02+seed*.00001)
            merged = dict(base, daily_ic={'lightgbm_day_t': base['daily_ic']['c0plus'], 'c0plus': cand['daily_ic']['c0plus_fund']})
            for p, r in ((h1, base), (h2, cand), (combined, merged)):
                (p/f'fold_{y}_seed_{seed}.json').write_text(json.dumps(r))
    actual = h.analyze(h1, h2, n_boot=30)
    expected = a.report(combined, n_boot=30)
    for key in ('clean', 'descriptive', 'classification', 'per_fold', 'seed_sd'):
        assert actual[key] == expected[key]
    assert len(actual['inputs']) == 30 and 'candidate_input' in actual['inputs'][0]
    (h2/'fold_2026_seed_97.json').unlink()
    with pytest.raises(ValueError, match='fold/seed'):
        h.analyze(h1, h2)
    (h1/'fold_2026_seed_97.json').unlink()
    with pytest.raises(ValueError, match='fold/seed'):
        h.analyze(h1, h2)


@pytest.mark.parametrize('p,noninferiority,expected', [
    (.049, '通過', '證實有益'), (.05, '通過', '不確定'),
    (.001, '最近期無法證實', '不確定'), (None, '通過', '不確定')])
def test_registered_holm_and_descriptive_gates(monkeypatch, p, noninferiority, expected):
    records = {y: {s: {} for s in h.SEEDS} for y in h.YEARS}
    monkeypatch.setattr(a, 'load_results', lambda *args: records)
    monkeypatch.setattr(a, 'report', lambda *args, **kwargs: dict(
        clean={'hac': {'p': p}}, descriptive={'noninferiority': noninferiority},
        classification='證實有益', limitations=['', '']))
    result = h.analyze('h1', 'h2')
    assert result['classification'] == expected
    assert result['h2_holm']['threshold'] == .05
