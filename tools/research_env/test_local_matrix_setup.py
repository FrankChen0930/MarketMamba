import json, sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'V6')); sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'V6/research_lab/tests'))
from research_lab import colab_runner as cr
from test_colab_runner import build
sys.path.insert(0, str(Path(__file__).parent))
import local_matrix_setup as lms


def make(tmp):
    a, reg_a = build(tmp/'dl5', name='mat')
    for n in ('stock_ids.npy',):
        np.save(a/n, np.array(['1000']*10, dtype='U32'))
    reg_a['files']['stock_ids.npy'] = cr.sha256_file(a/'stock_ids.npy')
    m = json.loads((a/'manifest.json').read_text()); m['files']['stock_ids.npy'] = reg_a['files']['stock_ids.npy']; (a/'manifest.json').write_text(json.dumps(m))
    return a, reg_a


def test_layout_links_and_verification(tmp_path):
    a, reg = make(tmp_path)
    out = tmp_path/'out'
    r = lms.setup(out, a, registry={'E5': reg}, log=lambda *_: None)
    d = r['E5']
    assert (d/'X.npy').is_symlink() and (out/'shared'/'X.npy').is_file() and np.load(d/'X.npy').shape[1] == 48
    assert json.loads((d/'manifest.json').read_text())['logical_identity'] == reg['logical_identity']
    assert lms.setup(out, a, registry={'E5': reg}, log=lambda *_: None)                # idempotent re-run


def test_corrupted_or_missing_file_rejected(tmp_path):
    a, reg = make(tmp_path)
    y = np.load(a/'y5.npy'); y[0] += 1; np.save(a/'y5.npy', y)
    with pytest.raises(cr.IdentityError, match='y5.npy'):
        lms.setup(tmp_path/'o', a, registry={'E5': reg}, log=lambda *_: None)
    a2, reg2 = make(tmp_path/'x') if (tmp_path/'x').mkdir() is None else None
    (a2/'splits.npy').unlink()
    with pytest.raises(FileNotFoundError):
        lms.setup(tmp_path/'o2', a2, registry={'E5': reg2}, log=lambda *_: None)
