"""Shared Colab GPU environment entry (B7 probe, later B8 training): reuse/save compiled mamba-ssm wheels.

setup(store) order: already installed -> compatible wheel from the Drive store -> (allow_build) pip wheel
(setup.py downloads an official prebuilt wheel when one matches, else compiles) -> real Mamba2 GPU
forward+backward -> only then save wheels (+ sidecar json with fingerprint and sha256). Nothing is reported
as saved unless a wheel file was actually copied and hashed. RESEARCH_ONLY.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

PKGS = {'mamba_ssm': 'mamba-ssm', 'causal_conv1d': 'causal-conv1d'}
DEFAULT_VERSIONS = {'mamba-ssm': '2.3.2.post1', 'causal-conv1d': None}   # None = resolver's latest


def fingerprint():
    import platform
    import torch
    fp = dict(py_tag='cp%d%d' % sys.version_info[:2], torch=torch.__version__.split('+')[0], torch_full=torch.__version__,
              cuda=torch.version.cuda, cxx11abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
              system=platform.system(), machine=platform.machine(), cc=None, gpu=None)
    if torch.cuda.is_available():
        fp['cc'] = '%d.%d' % torch.cuda.get_device_capability(0)
        fp['gpu'] = torch.cuda.get_device_name(0)
    return fp


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def compatible(meta, fp):
    """Wheel sidecar `meta` vs current fingerprint -> (ok, [mismatch reasons])."""
    b, why = meta.get('fp', {}), []
    for k in ('py_tag', 'torch', 'cuda', 'cxx11abi', 'system', 'machine'):
        if b.get(k) != fp.get(k):
            why.append(f'{k}: wheel {b.get(k)} != env {fp.get(k)}')
    if fp.get('cc') not in meta.get('cc_list', [b.get('cc')]):
        why.append(f'gpu cc: wheel {meta.get("cc_list", [b.get("cc")])} lacks {fp.get("cc")}')
    return not why, why


def find_wheels(store, fp):
    """Newest compatible, hash-verified wheel per package from the store."""
    out = {}
    for side in sorted(Path(store).glob('*.whl.json'), key=lambda p: p.stat().st_mtime):
        meta = json.loads(side.read_text())
        whl = side.with_suffix('')
        ok, _ = compatible(meta, fp)
        if ok and whl.exists() and sha256_file(whl) == meta['sha256']:
            out[meta['package']] = (whl, meta)
    return out


def save_wheel(whl, store, fp, package, version, source, seconds=None, cc_list=None):
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    dst = store/Path(whl).name
    digest = sha256_file(whl)
    if dst.exists() and sha256_file(dst) == digest:
        return dst, 'already_saved'
    shutil.copyfile(whl, str(dst)+'.part')
    if sha256_file(str(dst)+'.part') != digest:
        Path(str(dst)+'.part').unlink()
        raise IOError('copy verification failed')
    Path(str(dst)+'.part').replace(dst)
    meta = dict(file=dst.name, sha256=digest, size=dst.stat().st_size, package=package, version=version, fp=fp,
                cc_list=cc_list or [fp.get('cc')], source=source, build_seconds=seconds,
                saved_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    side = Path(str(dst)+'.json'); tmp = Path(str(side)+'.tmp')
    tmp.write_text(json.dumps(meta, indent=2)); tmp.replace(side)
    return dst, 'saved'


def pip_cache_wheels(fp, cache_dir=None, run=subprocess.run):
    """Wheels pip already built/cached whose tag and version match what is installed. May be empty."""
    if cache_dir is None:
        r = run([sys.executable, '-m', 'pip', 'cache', 'dir'], capture_output=True, text=True)
        cache_dir = r.stdout.strip() if r.returncode == 0 else None
    found = {}
    if not cache_dir:
        return found
    for pkg in PKGS:
        try:
            ver = metadata.version(pkg.replace('_', '-'))
        except metadata.PackageNotFoundError:
            continue
        for w in glob.glob(f'{cache_dir}/wheels/**/{pkg}-{ver}-{fp["py_tag"]}-*.whl', recursive=True):
            found[PKGS[pkg]] = (Path(w), ver)
    return found


REPOS = {'mamba-ssm': 'state-spaces/mamba', 'causal-conv1d': 'Dao-AILab/causal-conv1d'}


def prebuilt_check(fp, versions=None, urlopen=None):
    """Does an official prebuilt wheel exist for this exact (CUDA major, torch major.minor, C++ ABI, python)?
    Looks only at GitHub release asset names; never downloads. Returns per-package availability."""
    import urllib.request
    urlopen = urlopen or (lambda u: json.load(urllib.request.urlopen(u, timeout=30)))
    versions = {**DEFAULT_VERSIONS, **(versions or {})}
    tag = 'cu%storch%s.%scxx11abi%s' % (str(fp['cuda']).split('.')[0], *fp['torch'].split('.')[:2], 'TRUE' if fp['cxx11abi'] else 'FALSE')
    out = {'tag': tag, 'py_tag': fp['py_tag']}
    for pkg, repo in REPOS.items():
        try:
            v = versions.get(pkg)
            rel = urlopen(f'https://api.github.com/repos/{repo}/releases/' + (f'tags/v{v}' if v else 'latest'))
            names = [a['name'] for a in rel['assets']]
            hit = [n for n in names if f'+{tag}-{fp["py_tag"]}-' in n and n.endswith(f'{fp["machine"]}.whl')]
            out[pkg] = dict(available=bool(hit), asset=hit[0] if hit else None, release=rel.get('tag_name'))
        except Exception as e:  # noqa: BLE001
            out[pkg] = dict(available=None, error=f'{type(e).__name__}: {e}')
    return out


STALE_CHECKS = (('mamba_ssm.ops.triton.ssd_combined', 'causal_conv1d_fwd_function'),
                ('mamba_ssm.modules.mamba2', 'causal_conv1d_fn'))


def purge_stale_mamba_modules(modules=None):
    """If mamba_ssm was imported while causal_conv1d was broken, its module globals hold None forever, even after
    causal_conv1d is reinstalled. Drop those modules so the next import binds the working functions. Healthy
    modules are left alone (re-importing them would re-register torch custom ops)."""
    modules = sys.modules if modules is None else modules
    stale = [m for m, attr in STALE_CHECKS if m in modules and getattr(modules[m], attr, 'missing') is None]
    if stale:
        for k in [k for k in modules if k.split('.')[0] in ('mamba_ssm', 'causal_conv1d')]:
            del modules[k]
        import importlib
        importlib.invalidate_caches()
    return stale


USABLE = {'mamba-ssm': 'mamba_ssm', 'causal-conv1d': 'causal_conv1d.cpp_functions'}   # modules that must import


def diagnose(importable=None):
    """Import results and installed versions of the pieces Mamba2's fused path needs (for the report)."""
    import importlib
    d = {'imports': {}, 'versions': {}}
    for m in ('mamba_ssm', 'mamba_ssm.ops.triton.ssd_combined', 'causal_conv1d', 'causal_conv1d.cpp_functions'):
        try:
            importlib.import_module(m)
            d['imports'][m] = 'ok'
        except Exception as e:  # noqa: BLE001
            d['imports'][m] = f'{type(e).__name__}: {str(e)[:200]}'
    for p in ('mamba-ssm', 'causal-conv1d', 'torch', 'triton'):
        try:
            d['versions'][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            d['versions'][p] = None
    return d


def verify_official_mamba2(device='cuda', require_causal=True):
    """Real Mamba2 GPU forward+backward (official class, official kernels). Raises on any failure.
    mamba-ssm 2.3.2.post1's default fused path (use_mem_eff_path=True) calls causal_conv1d_fwd_function, which is
    None when causal_conv1d fails to import -> an opaque "'NoneType' object is not callable"; check it first."""
    import torch
    if require_causal:
        try:
            from causal_conv1d.cpp_functions import causal_conv1d_fwd_function  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f'causal_conv1d unusable ({type(e).__name__}: {str(e)[:200]}); '
                               'the official Mamba2 fused path needs it (reinstall a wheel built for this torch)') from e
    from mamba_ssm import Mamba2
    from mamba_ssm.modules.mamba2 import Mamba2 as Official
    from .seq_models import SeqRanker, mamba2_dims
    assert Mamba2 is Official and Mamba2.__module__ == 'mamba_ssm.modules.mamba2'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available: official Mamba2 cannot run')
    mamba2_dims(64, 2, 8)
    res = {}
    for kind in ('mamba2', 'samba'):
        m = SeqRanker(kind, 48, 64, 1, mixer_kw=dict(d_state=16, headdim=8, expand=2)).to(device)
        y = m(torch.randn(16, 60, 48, device=device))
        (y['score5'].mean()+y['score10'].mean()).backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert torch.isfinite(y['score5']).all() and grads and all(torch.isfinite(g).all() for g in grads)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if 'layers' in n)
        res[kind] = dict(params=sum(p.numel() for p in m.parameters()), grad_tensors=len(grads))
    torch.cuda.synchronize()
    return res


def _import_ok(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:  # noqa: BLE001
        return False


def setup(store, allow_build=True, allow_compile=False, versions=None, with_causal_conv1d=True, work='/content/gpu_wheels_build',
          run=subprocess.run, fp=None, verify=verify_official_mamba2, importable=_import_ok, log=print, prebuilt=prebuilt_check,
          diagnostics=diagnose, force=(), harvest_cache=False):
    fp = fp or fingerprint()
    versions = {**DEFAULT_VERSIONS, **(versions or {})}
    want = ['mamba-ssm'] + (['causal-conv1d'] if with_causal_conv1d else [])
    rep = dict(fingerprint=fp, store=str(store), steps=[], saved=[], status=None)
    Path(store).mkdir(parents=True, exist_ok=True)
    built, cache = {}, {}
    pip = lambda *a: run([sys.executable, '-m', 'pip', *a], capture_output=True, text=True)
    ok = lambda pkg: importable(USABLE[pkg])
    step = rep['steps'].append

    def ensure(pkg):
        if pkg not in force and ok(pkg):   # forced packages are not imported first, so the fresh wheel is what gets verified
            return step(f'{pkg}: usable')
        found = find_wheels(store, fp)
        if pkg in found:
            r = pip('install', '-q', '--no-deps', '--no-index', '--force-reinstall', str(found[pkg][0]))
            step(f'{pkg}: install_from_store rc={r.returncode} {found[pkg][0].name}')
            if ok(pkg):
                return
        else:
            step(f'{pkg}: no_compatible_wheel_in_store')
        if not allow_build:
            return step(f'{pkg}: build_not_allowed')
        if 'pb' not in cache:
            cache['pb'] = prebuilt(fp, versions)
            rep['prebuilt'] = cache['pb']
            step('prebuilt_check ' + cache['pb']['tag'] + ': ' + ', '.join(f'{k}={v.get("available")}' for k, v in cache['pb'].items() if isinstance(v, dict)))
        if not cache['pb'].get(pkg, {}).get('available') and not allow_compile:
            return step(f'{pkg}: no_official_prebuilt_for_torch{fp["torch"]}_cu{fp["cuda"]}: not compiling (allow_compile=False). '
                        'Pin torch to a version with an official wheel (roadmap 8.6) or set allow_compile=True (~40 min).')
        Path(work).mkdir(parents=True, exist_ok=True)
        spec = pkg+(f'=={versions[pkg]}' if versions.get(pkg) else '')
        t = time.time()
        # --no-cache-dir: pip's wheel cache is keyed by sdist only, so it would silently reuse a wheel built for another torch
        r = pip('wheel', '-q', '--no-cache-dir', '--no-deps', '--no-build-isolation', '-w', work, spec)
        sec = time.time()-t
        step(f'{pkg}: pip_wheel {spec} rc={r.returncode} {sec:.0f}s')
        if r.returncode:
            return step(r.stderr[-800:])
        ws = sorted(Path(work).glob(f'{pkg.replace("-", "_")}-*.whl'), key=lambda p: p.stat().st_mtime)
        if not ws:
            return step(f'{pkg}: pip wheel returned 0 but produced no wheel file')
        w = ws[-1]
        built[pkg] = (w, w.name.split('-')[1], 'built' if sec > 180 else 'prebuilt_download_or_fast_build', sec)
        r = pip('install', '-q', '--no-deps', '--no-index', '--force-reinstall', str(w))
        step(f'{pkg}: install_built rc={r.returncode}')

    if allow_build:   # an installed package with no compatible wheel in the store is rebuilt once so a wheel can be saved
        have = find_wheels(store, fp)
        auto = [p for p in want if p not in have and p not in force]
        if auto:
            step(f'store has no compatible wheel for {auto}: will re-fetch to save')
        force = tuple(force)+tuple(auto)
    for pkg in want:
        ensure(pkg)
    rep['purged_stale_modules'] = purge_stale_mamba_modules()
    rep['diagnostics'] = diagnostics()
    if not ok('mamba-ssm'):
        rep['status'] = 'unavailable'
    else:
        try:
            rep['verify'] = verify()
            rep['status'] = 'verified'
        except Exception as e:  # noqa: BLE001
            rep['status'] = 'installed_but_verification_failed'
            rep['verify_error'] = f'{type(e).__name__}: {e}'
            rep['verify_traceback'] = __import__('traceback').format_exc()[-2500:]

    if rep['status'] == 'verified':          # save only wheels from a working environment
        if not built and harvest_cache:   # pip's cache does not record which torch a wheel was built for: opt-in only
            for pkg, (w, ver) in pip_cache_wheels(fp, run=run).items():
                built[pkg] = (w, ver, 'pip_cache_harvest_unverified_provenance', None)
            step('pip_cache_harvest found=' + str(sorted(built)))
        elif not built:
            step('no wheel was produced in this run: nothing saved (use force=(...) to rebuild and save)')
        for pkg, (w, ver, source, sec) in built.items():
            dst, how = save_wheel(w, store, fp, pkg, ver, source, sec)
            rep['saved'].append(dict(package=pkg, file=dst.name, how=how, sha256=sha256_file(dst), source=source))
    tag = (fp.get('gpu') or 'cpu').replace(' ', '_')
    out = Path(store)/f'env_report_{tag}_{fp["py_tag"]}_torch{fp["torch"]}_cu{fp["cuda"]}.json'
    out.write_text(json.dumps(rep, indent=2, default=str))
    log(json.dumps({k: rep[k] for k in ('status', 'steps', 'purged_stale_modules', 'saved', 'verify_error', 'verify_traceback') if k in rep}, indent=1, default=str))
    return rep
