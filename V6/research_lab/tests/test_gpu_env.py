import json
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


YES = lambda fp, v=None: {'tag': 't', 'mamba-ssm': {'available': True}}
NO = lambda fp, v=None: {'tag': 't', 'mamba-ssm': {'available': False}}


class Runner:
    def __init__(self, work, produce=True):
        self.calls, self.work, self.produce = [], work, produce

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[3:])
        if cmd[3] == 'wheel' and self.produce:
            name = cmd[-1].split('==')[0].replace('-', '_')
            fake_wheel(self.work, f'{name}-1.0-cp312-cp312-linux_x86_64.whl', name.encode())
        return type('R', (), dict(returncode=0, stdout='', stderr=''))()


def state(installed=False):
    s = {'on': installed}
    return s, (lambda pkg: s['on'])


def test_setup_build_then_save_only_after_verify(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    s, imp = state(False)
    run = Runner(work)
    def flip(cmd, **kw):
        r = run(cmd, **kw)
        if cmd[3] == 'install':
            s['on'] = True
        return r
    rep = g.setup(tmp_path/'store', fp=FP, run=flip, work=str(work), verify=lambda: {'ok': 1}, importable=imp, log=lambda *_: None, prebuilt=YES)
    assert rep['status'] == 'verified' and {x['package'] for x in rep['saved']} == {'mamba-ssm', 'causal-conv1d'}
    assert all((tmp_path/'store'/x['file']).exists() for x in rep['saved'])
    # second environment: store hit, no build, nothing re-saved as new
    s['on'] = False
    run2 = Runner(work, produce=False)
    def flip2(cmd, **kw):
        r = run2(cmd, **kw)
        if cmd[3] == 'install':
            s['on'] = True
        return r
    rep2 = g.setup(tmp_path/'store', fp=FP, run=flip2, work=str(work), verify=lambda: {'ok': 1}, importable=imp, log=lambda *_: None, prebuilt=YES)
    assert not any(c[0] == 'wheel' for c in run2.calls) and any('install_from_store' in x for x in rep2['steps'])


def test_failed_verification_saves_nothing_and_no_build_when_disallowed(tmp_path):
    work = tmp_path/'work'; work.mkdir()
    s, imp = state(False)
    run = Runner(work)
    def flip(cmd, **kw):
        r = run(cmd, **kw)
        if cmd[3] == 'install':
            s['on'] = True
        return r
    def bad():
        raise RuntimeError('kernel launch failed')
    rep = g.setup(tmp_path/'store', fp=FP, run=flip, work=str(work), verify=bad, importable=imp, log=lambda *_: None, prebuilt=YES)
    assert rep['status'] == 'installed_but_verification_failed' and rep['saved'] == [] and not list((tmp_path/'store').glob('*.whl'))
    s['on'] = False
    run3 = Runner(work)
    rep3 = g.setup(tmp_path/'store3', allow_build=False, fp=FP, run=run3, verify=bad, importable=imp, log=lambda *_: None, prebuilt=YES)
    assert rep3['status'] == 'unavailable' and 'build_not_allowed' in rep3['steps'] and not run3.calls


def test_already_installed_without_cached_wheel_does_not_claim_saved(tmp_path):
    s, imp = state(True)
    run = lambda cmd, **kw: type('R', (), dict(returncode=1, stdout='', stderr='no cache'))()
    rep = g.setup(tmp_path/'store', fp=FP, run=run, verify=lambda: {'ok': 1}, importable=imp, log=lambda *_: None, prebuilt=YES)
    assert rep['status'] == 'verified' and rep['saved'] == []
    assert any('nothing saved' in x for x in rep['steps'])


def test_verify_requires_cuda_or_raises():
    pytest.importorskip('torch')
    try:
        import mamba_ssm  # noqa: F401
        pytest.skip('mamba_ssm present')
    except ImportError:
        with pytest.raises(ImportError):
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
    s, imp = state(False)
    run = Runner(work)
    rep = g.setup(tmp_path/'s', fp=FP, run=run, work=str(work), verify=lambda: {}, importable=imp, log=lambda *_: None, prebuilt=NO)
    assert rep['status'] == 'unavailable' and not run.calls and any('not compiling' in x for x in rep['steps'])
    run2 = Runner(work)
    def flip(cmd, **kw):
        r = run2(cmd, **kw)
        if cmd[3] == 'install':
            s['on'] = True
        return r
    rep2 = g.setup(tmp_path/'s2', allow_compile=True, fp=FP, run=flip, work=str(work), verify=lambda: {}, importable=imp, log=lambda *_: None, prebuilt=NO)
    assert rep2['status'] == 'verified' and any(c[0] == 'wheel' for c in run2.calls)
