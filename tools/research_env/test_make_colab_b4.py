import json, subprocess, sys, zipfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'tools/research_env/make_colab_b4.py'


def test_bundle_notebook_and_import_in_clean_dir(tmp_path):
    out = tmp_path/'o'
    r = subprocess.run([sys.executable, str(SCRIPT), '--out', str(out)], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    nb = json.loads((out/'b4_matrix_baselines.ipynb').read_text())
    assert nb['nbformat'] == 4 and info['sha256'] in ''.join(''.join(c['source']) for c in nb['cells'])
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    compile(code, 'nb', 'exec')
    assert {'E5', 'E1'} <= set(code.split('"')) and 'cr.run_experiment' in code
    ex = tmp_path/'x'
    zipfile.ZipFile(out/'b4_research_lab_bundle.zip').extractall(ex)
    check = subprocess.run([sys.executable, '-I', '-c', f'import sys; sys.path.insert(0, {str(ex)!r}); '
                            'from research_lab import colab_runner as c; print(sorted(c.REGISTRY))'],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    again = subprocess.run([sys.executable, str(SCRIPT), '--out', str(out)], capture_output=True, text=True)
    assert again.returncode != 0  # no overwrite
    out2 = tmp_path/'o2'
    subprocess.run([sys.executable, str(SCRIPT), '--out', str(out2)], check=True, capture_output=True)
    assert (out/'b4_research_lab_bundle.zip.sha256').read_text() == (out2/'b4_research_lab_bundle.zip.sha256').read_text()  # deterministic


def test_probe_notebook_and_bundle_contains_seq_models(tmp_path):
    out = tmp_path/'o'
    info = json.loads(subprocess.run([sys.executable, str(SCRIPT), '--out', str(out)], capture_output=True, text=True, check=True).stdout)
    nb = json.loads((out/'b7_mamba_probe.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    compile(code, 'nb7', 'exec')
    assert info['sha256'] in code and 'gpu_env.setup' in code and 'train' not in code.lower().replace('training', '')
    assert {'research_lab/seq_models.py', 'research_lab/gpu_env.py'} <= set(zipfile.ZipFile(out/'b4_research_lab_bundle.zip').namelist())
