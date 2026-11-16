"""Synthetic rolling orchestration checks; never load a real matrix."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from research_lab import rolling_runner as rr


@pytest.fixture
def setup(tmp_path, monkeypatch):
    local = tmp_path / '.local-data'
    monkeypatch.setattr(rr, 'LOCAL_DATA_ROOT', local)
    mask = np.array([False, True, True])
    def data(directory, *, normalize):
        assert normalize is None
        return SimpleNamespace(eligible=mask)
    monkeypatch.setattr(rr, 'SeqData', data)
    calls = []
    def fake(directory, label, max_train_rows, seed, *, n_boot, row_mask, log):
        calls.append((Path(directory).name, label, max_train_rows, seed, n_boot))
        assert row_mask is mask
        log('must be silent')
        return dict(fit_rows=12, eval_rows=2, models={'unused': {}}, daily_ic={
            'lightgbm_day_t': {'2022-01-03': .1},
            'c0plus': {'2022-01-03': .2}, 'reversal_ret5': {'2022-01-03': -.1}})
    return local, local / 'results', fake, calls


def test_fold_seed_arguments_and_output(setup, capsys):
    local, _, fake, calls = setup
    result = rr.run_fold_seed(local / 'fold_2022', 29, 'y10', 123, 7, fake)
    assert calls == [('fold_2022', 'y10', 123, 29, 7)]
    assert result['fold'] == 'fold_2022'
    assert result['seed'] == 29 and result['label'] == 'y10'
    assert result['fit_rows'] == 12 and result['eval_rows'] == 2
    assert result['wall_seconds'] >= 0
    assert result['daily_ic'] == {'lightgbm_day_t': {'2022-01-03': .1}, 'c0plus': {'2022-01-03': .2}}
    assert set(result) == {'fold', 'seed', 'label', 'wall_seconds', 'fit_rows', 'eval_rows', 'daily_ic'}
    assert capsys.readouterr().out == ''


def test_resume_skips_completed(setup, capsys):
    local, out, fake, calls = setup
    rr.run_all(local, [2021, 2022], [17, 29], out, run_fn=fake)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    rr.run_all(local, [2021, 2022], [17, 29], out, run_fn=fake)
    assert len(calls) == 4
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()} == before
    assert all(json.loads(p.read_text())['seed'] in [17, 29] for p in out.iterdir())
    assert 'wall_seconds=' in capsys.readouterr().out


def test_corrupt_result_stops_without_overwrite(setup):
    local, out, fake, calls = setup
    out.mkdir(parents=True)
    bad = out / 'fold_2022_seed_17.json'
    bad.write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        rr.run_all(local, [2022], [17, 29], out, run_fn=fake)
    assert bad.read_text() == '{broken' and not calls
    assert list(out.iterdir()) == [bad]


def test_atomic_replace_and_cleanup(setup, monkeypatch):
    local, out, fake, _ = setup
    original = os.replace
    seen = []
    def replace(source, target):
        assert Path(source).parent == out.resolve()
        assert json.loads(Path(source).read_text())['seed'] == 17
        assert not Path(target).exists()
        seen.append(source)
        original(source, target)
    monkeypatch.setattr(rr.os, 'replace', replace)
    rr.run_all(local, [2022], [17], out, run_fn=fake)
    assert len(seen) == 1
    assert [p.name for p in out.iterdir()] == ['fold_2022_seed_17.json']


def test_replace_failure_cleans_temporary(setup, monkeypatch):
    local, out, fake, _ = setup
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(rr.os, 'replace', fail)
    with pytest.raises(OSError, match='replace failed'):
        rr.run_all(local, [2022], [17], out, run_fn=fake)
    assert list(out.iterdir()) == []


def test_failure_propagates_and_process_is_nonzero(setup):
    local, out, _, _ = setup
    def fail(*args, **kwargs):
        raise RuntimeError('job failed')
    with pytest.raises(RuntimeError, match='job failed'):
        rr.run_all(local, [2022], [17], out, run_fn=fail)
    assert not list(out.glob('*.json'))
    code = '''from pathlib import Path
from research_lab import rolling_runner as rr
rr.LOCAL_DATA_ROOT = Path(__import__('sys').argv[1])
def fail(*args, **kwargs):
    raise RuntimeError('injected job failed')
rr.run_fold_seed = fail
rr.main(['--folds-root', 'missing', '--years', '2022', '--seeds', '17', '--out-dir', str(rr.LOCAL_DATA_ROOT/'out'), '--run'])
'''
    proc = subprocess.run([sys.executable, '-c', code, str(local)], capture_output=True, text=True)
    assert proc.returncode != 0 and 'injected job failed' in proc.stderr


def test_output_guard_including_symlink_escape(setup, tmp_path):
    local, _, fake, calls = setup
    local.mkdir()
    escape = local / 'escape'
    escape.symlink_to(tmp_path, target_is_directory=True)
    for out in [tmp_path / 'outside', local, local / '..' / 'other', escape / 'out']:
        with pytest.raises(ValueError, match='local-data'):
            rr.run_all(local, [2022], [17], out, run_fn=fake)
    assert not calls and not (tmp_path / 'out').exists()


def test_cli_dry_run_lists_pending_and_skipped_without_writes(setup, capsys):
    local, out, fake, _ = setup
    rr.run_all(local, [2021], [17], out, run_fn=fake)
    before = {p: p.stat().st_mtime_ns for p in local.rglob('*')}
    rr.main(['--folds-root', str(local), '--years', '2021', '2022', '--seeds', '17', '--out-dir', str(out)])
    text = capsys.readouterr().out
    assert 'SKIP' in text and 'RUN' in text and 'total=2' in text and 'pending=1' in text
    assert {p: p.stat().st_mtime_ns for p in local.rglob('*')} == before
    fresh = local / 'absent'
    rr.main(['--folds-root', 'missing', '--years', '2022', '--seeds', '17', '--out-dir', str(fresh)])
    assert not fresh.exists()


def test_real_seqdata_eligibility_on_synthetic_matrix(tmp_path, monkeypatch):
    from research_lab.seq_runner import SeqData
    monkeypatch.setattr(rr, 'SeqData', SeqData)
    day = np.repeat(np.arange(65), 2)
    arrays = dict(X=np.zeros((130, 1), dtype=np.float32), y5=np.ones(130), y10=np.ones(130),
                  dates=np.datetime64('2022-01-01') + day.astype('timedelta64[D]'),
                  end_indices=np.arange(2, 131, 2), stock_idx=np.tile([0, 1], 65), splits=np.ones(130, dtype=np.uint8))
    for name, values in arrays.items():
        np.save(tmp_path / f'{name}.npy', values)
    (tmp_path / 'manifest.json').write_text(json.dumps({'rows': 130, 'feature_order': ['f'], 'files': {}}))
    def fake(*args, row_mask, **kwargs):
        np.testing.assert_array_equal(row_mask, day >= 59)
        return dict(fit_rows=0, eval_rows=12, daily_ic={'lightgbm_day_t': {}, 'c0plus': {}})
    assert rr.run_fold_seed(tmp_path, 17, run_fn=fake)['eval_rows'] == 12
