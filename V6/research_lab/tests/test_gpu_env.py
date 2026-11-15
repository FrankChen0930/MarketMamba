import json
from pathlib import Path
import pytest
from research_lab import gpu_env as g

FP = dict(py_tag='cp312', torch='2.8.0', torch_full='2.8.0+cu126', cuda='12.6', cxx11abi=True, system='Linux',
          machine='x86_64', cc='7.5', gpu='Tesla T4')


def fake_wheel(d, name='mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl', data=b'abc'):
    p = d/name
    p.write_bytes(data)
    return p


def test_compatibility_matrix():
    meta = dict(fp=FP, cc_list=['7.5', '8.0'])
    assert g.compatible(meta, FP) == (True, [])
    for k, v in dict(py_tag='cp311', torch='2.9.0', cuda='12.4', cxx11abi=False, cc='9.0').items():
        ok, why = g.compatible(meta, {**FP, k: v})
        assert not ok and k.replace('cc', 'gpu cc') in why[0]
    assert g.compatible(meta, {**FP, 'cc': '8.0'})[0]          # cc listed -> ok


def test_save_find_roundtrip_hash_check_and_no_duplicate(tmp_path):
    store, src = tmp_path/'store', tmp_path/'src'; src.mkdir()
    w = fake_wheel(src)
    dst, how = g.save_wheel(w, store, FP, 'mamba-ssm', '2.3.2.post1', 'built', 2400.)
    assert how == 'saved' and g.save_wheel(w, store, FP, 'mamba-ssm', '2.3.2.post1', 'built')[1] == 'already_saved'
    meta = json.loads((store/(dst.name+'.json')).read_text())
    assert meta['sha256'] == g.sha256_file(dst) and meta['fp']['cuda'] == '12.6' and meta['build_seconds'] == 2400.
    assert list(g.find_wheels(store, FP)) == ['mamba-ssm']
    assert g.find_wheels(store, {**FP, 'torch': '2.9.0'}) == {}      # incompatible env -> nothing
    dst.write_bytes(b'tampered')
    assert g.find_wheels(store, FP) == {}                            # hash mismatch rejected


def test_pip_cache_harvest_matches_tag_and_installed_version(tmp_path, monkeypatch):
    c = tmp_path/'cache/wheels/ab/cd'; c.mkdir(parents=True)
    fake_wheel(c, 'mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl')
    fake_wheel(c, 'mamba_ssm-2.3.2.post1-cp311-cp311-linux_x86_64.whl')
    fake_wheel(c, 'causal_conv1d-9.9-cp312-cp312-linux_x86_64.whl')
    monkeypatch.setattr(g.metadata, 'version', lambda n: {'mamba-ssm': '2.3.2.post1', 'causal-conv1d': '1.5.0'}[n])
    got = g.pip_cache_wheels(FP, cache_dir=str(tmp_path/'cache'))
    assert list(got) == ['mamba-ssm'] and 'cp312' in got['mamba-ssm'][0].name   # causal wheel version differs -> excluded


YES = lambda fp, v=None: {'tag': 't', 'mamba-ssm': {'available': True}, 'causal-conv1d': {'available': True}}
NO = lambda fp, v=None: {'tag': 't', 'mamba-ssm': {'available': False}, 'causal-conv1d': {'available': False}}


class Runner:
    def __init__(self, work, produce=True):
        self.calls, self.work, self.produce = [], work, produce

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[3:])
        if cmd[3] == 'wheel' and self.produce:
            name = cmd[-1].split('==')[0].replace('-', '_')
            fake_wheel(self.work, f'{name}-1.0-cp312-cp312-linux_x86_64.whl', name.encode())
        return type('R', (), dict(returncode=0, stdout='', stderr=''))()


def make_env(work, mamba=False, causal=False, produce=True):
    """Per-module install state; a `pip install <wheel>` flips the module its filename names."""
    st = {'mamba_ssm': mamba, 'causal_conv1d.cpp_functions': causal}
    runner = Runner(work, produce)
    def run(cmd, **kw):
        r = runner(cmd, **kw)
        if cmd[3] == 'install':
            for a in cmd:
                n = Path(a).name
                if n.startswith('mamba_ssm'):
                    st['mamba_ssm'] = True
                if n.startswith('causal_conv1d'):
                    st['causal_conv1d.cpp_functions'] = True
        return r
    return st, (lambda m: st[m]), run, runner


def go(tmp, store='store', work=None, **kw):
    kw.setdefault('prebuilt', YES)
    return g.setup(tmp/store, fp=FP, work=str(work or tmp/'work'), diagnostics=lambda: {}, log=lambda *_: None, **kw)


def test_setup_build_then_save_only_after_verify(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    st, imp, run, _ = make_env(work)
    rep = go(tmp_path, run=run, verify=lambda: {'ok': 1}, importable=imp)
    assert rep['status'] == 'verified' and {x['package'] for x in rep['saved']} == {'mamba-ssm', 'causal-conv1d'}
    assert all((tmp_path/'store'/x['file']).exists() for x in rep['saved'])
    # fresh environment: Drive store hit, no build
    st2, imp2, run2, runner2 = make_env(work)
    rep2 = go(tmp_path, run=run2, verify=lambda: {'ok': 1}, importable=imp2)
    assert not any(c[0] == 'wheel' for c in runner2.calls) and any('install_from_store' in x for x in rep2['steps']) and rep2['status'] == 'verified'


def test_failed_verification_saves_nothing_and_no_build_when_disallowed(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    def bad():
        raise RuntimeError('kernel launch failed')
    st, imp, run, _ = make_env(work)
    rep = go(tmp_path, run=run, verify=bad, importable=imp)
    assert rep['status'] == 'installed_but_verification_failed' and rep['saved'] == [] and not list((tmp_path/'store').glob('*.whl'))
    st3, imp3, run3, runner3 = make_env(work)
    rep3 = go(tmp_path, store='s3', allow_build=False, run=run3, verify=bad, importable=imp3)
    assert rep3['status'] == 'unavailable' and any('build_not_allowed' in x for x in rep3['steps']) and not runner3.calls


def test_already_installed_without_cached_wheel_does_not_claim_saved(tmp_path):
    st, imp, _, _ = make_env(tmp_path, mamba=True, causal=True)
    run = lambda cmd, **kw: type('R', (), dict(returncode=1, stdout='', stderr='no cache'))()
    rep = go(tmp_path, run=run, verify=lambda: {'ok': 1}, importable=imp)
    assert rep['status'] == 'verified' and rep['saved'] == [] and any('nothing saved' in x for x in rep['steps'])


def test_verify_requires_cuda_or_raises():
    pytest.importorskip('torch')
    try:
        import mamba_ssm  # noqa: F401
        pytest.skip('mamba_ssm present')
    except ImportError:
        with pytest.raises(RuntimeError, match='causal_conv1d unusable'):
            g.verify_official_mamba2()


def test_prebuilt_check_matches_tag_python_and_machine():
    assets = [{'name': 'mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl'},
              {'name': 'mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp313-cp313-linux_aarch64.whl'},
              {'name': 'mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp313-cp313-linux_x86_64.whl'}]
    fake = lambda url: {'tag_name': 'v', 'assets': assets}
    fp = {**FP, 'py_tag': 'cp313', 'torch': '2.10.0', 'cuda': '12.8'}
    r = g.prebuilt_check(fp, urlopen=fake)
    assert r['tag'] == 'cu12torch2.10cxx11abiTRUE' and r['mamba-ssm']['available'] and r['mamba-ssm']['asset'].endswith('x86_64.whl')
    assert not g.prebuilt_check({**fp, 'torch': '2.14.1', 'cuda': '13.0'}, urlopen=fake)['mamba-ssm']['available']
    assert not g.prebuilt_check({**fp, 'py_tag': 'cp312'}, urlopen=fake)['mamba-ssm']['available']
    def boom(u):
        raise OSError('offline')
    assert g.prebuilt_check(fp, urlopen=boom)['mamba-ssm']['available'] is None


def test_no_prebuilt_does_not_compile_unless_allowed(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    st, imp, run, runner = make_env(work)
    rep = go(tmp_path, run=run, verify=lambda: {}, importable=imp, prebuilt=NO)
    assert rep['status'] == 'unavailable' and not runner.calls and any('not compiling' in x for x in rep['steps'])
    st2, imp2, run2, runner2 = make_env(work)
    rep2 = go(tmp_path, store='s2', allow_compile=True, run=run2, verify=lambda: {}, importable=imp2, prebuilt=NO)
    assert rep2['status'] == 'verified' and any(c[0] == 'wheel' for c in runner2.calls)


def test_stale_causal_conv1d_is_reinstalled_and_missing_one_is_reported(tmp_path):
    """mamba_ssm usable but causal_conv1d unusable (e.g. built for another torch): rebuild only that package."""
    work = tmp_path/'work'; work.mkdir()
    st, imp, run, runner = make_env(work, mamba=True, causal=False)
    rep = go(tmp_path, run=run, verify=lambda: {'ok': 1}, importable=imp)
    wheels = [c for c in runner.calls if c[0] == 'wheel']
    assert rep['status'] == 'verified' and len(wheels) == 1 and 'causal-conv1d' in wheels[0][-1]
    assert any('--force-reinstall' in c for c in runner.calls if c[0] == 'install')
    assert [x['package'] for x in rep['saved']] == ['causal-conv1d']


def test_verify_names_causal_conv1d_instead_of_nonetype_error():
    pytest.importorskip('torch')
    try:
        import causal_conv1d  # noqa: F401
        pytest.skip('causal_conv1d present')
    except ImportError:
        with pytest.raises(RuntimeError, match='causal_conv1d unusable'):
            g.verify_official_mamba2()


def test_wheel_build_bypasses_pip_cache_and_force_rebuilds_usable_package(tmp_path):
    """Regression: a stale cached wheel (built for another torch) must never be reused: --no-cache-dir always."""
    work = tmp_path/'work'; work.mkdir()
    st, imp, run, runner = make_env(work, mamba=True, causal=True)
    rep = go(tmp_path, run=run, verify=lambda: {'ok': 1}, importable=imp, force=('mamba-ssm', 'causal-conv1d'))
    wheels = [c for c in runner.calls if c[0] == 'wheel']
    assert len(wheels) == 2 and all('--no-cache-dir' in c for c in wheels)
    assert {x['package'] for x in rep['saved']} == {'mamba-ssm', 'causal-conv1d'}


def test_pip_cache_harvest_is_opt_in_and_marked_unverified(tmp_path, monkeypatch):
    st, imp, _, _ = make_env(tmp_path, mamba=True, causal=True)
    c = tmp_path/'cache/wheels/a'; c.mkdir(parents=True)
    fake_wheel(c)
    monkeypatch.setattr(g.metadata, 'version', lambda n: '2.3.2.post1')
    def run(cmd, **kw):
        return type('R', (), dict(returncode=0, stdout=str(tmp_path/'cache'), stderr=''))()
    off = go(tmp_path, run=run, verify=lambda: {}, importable=imp)
    assert off['saved'] == []
    on = go(tmp_path, store='s2', run=run, verify=lambda: {}, importable=imp, harvest_cache=True)
    assert on['saved'] and on['saved'][0]['source'] == 'pip_cache_harvest_unverified_provenance'


def test_purge_only_stale_modules_with_none_bindings():
    import types
    def mod(**kw):
        m = types.ModuleType('x')
        m.__dict__.update(kw)
        return m
    mods = {'mamba_ssm': mod(), 'mamba_ssm.ops.triton.ssd_combined': mod(causal_conv1d_fwd_function=None),
            'mamba_ssm.modules.mamba2': mod(causal_conv1d_fn=None), 'causal_conv1d': mod(), 'torch': mod()}
    stale = g.purge_stale_mamba_modules(mods)
    assert len(stale) == 2 and set(mods) == {'torch'}                  # whole mamba/causal family dropped, others kept
    healthy = {'mamba_ssm': mod(), 'mamba_ssm.ops.triton.ssd_combined': mod(causal_conv1d_fwd_function=lambda *a: 1)}
    assert g.purge_stale_mamba_modules(healthy) == [] and len(healthy) == 2   # healthy modules never re-imported
    assert g.purge_stale_mamba_modules({}) == []


def test_setup_reports_traceback_and_purge(tmp_path):
    st, imp, _, _ = make_env(tmp_path, mamba=True, causal=True)
    def bad():
        raise TypeError("'NoneType' object is not callable")
    rep = go(tmp_path, run=lambda *a, **k: None, verify=bad, importable=imp)
    assert rep['status'] == 'installed_but_verification_failed' and 'TypeError' in rep['verify_traceback'] and rep['purged_stale_modules'] == []
