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


def verify_official_mamba2(device='cuda'):
    """Real Mamba2 GPU forward+backward (official class, official kernels). Raises on any failure."""
    import torch
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


def setup(store, allow_build=True, versions=None, with_causal_conv1d=True, work='/content/gpu_wheels_build',
          run=subprocess.run, fp=None, verify=verify_official_mamba2, importable=_import_ok, log=print):
    fp = fp or fingerprint()
    versions = {**DEFAULT_VERSIONS, **(versions or {})}
    want = ['mamba-ssm'] + (['causal-conv1d'] if with_causal_conv1d else [])
    rep = dict(fingerprint=fp, store=str(store), steps=[], saved=[], status=None)
    Path(store).mkdir(parents=True, exist_ok=True)
    built = {}
    pip = lambda *a: run([sys.executable, '-m', 'pip', *a], capture_output=True, text=True)

    if importable('mamba_ssm'):
        rep['steps'].append('already_installed')
    else:
        found = find_wheels(store, fp)
        if all(w in found for w in ['mamba-ssm']):
            files = [str(found[w][0]) for w in want if w in found]
            r = pip('install', '-q', '--no-deps', '--no-index', *files)
            rep['steps'].append(f'install_from_store rc={r.returncode} files={[Path(f).name for f in files]}')
        else:
            rep['steps'].append('no_compatible_wheel_in_store')
        if not importable('mamba_ssm') and allow_build:
            Path(work).mkdir(parents=True, exist_ok=True)
            for pkg in want:
                spec = pkg+(f'=={versions[pkg]}' if versions.get(pkg) else '')
                t = time.time()
                r = pip('wheel', '-q', '--no-deps', '--no-build-isolation', '-w', work, spec)
                sec = time.time()-t
                rep['steps'].append(f'pip_wheel {spec} rc={r.returncode} {sec:.0f}s')
                if r.returncode:
                    rep['steps'].append(r.stderr[-800:])
                    if pkg == 'mamba-ssm':
                        break
                    continue
                w = sorted(Path(work).glob(f'{pkg.replace("-", "_")}-*.whl'), key=lambda p: p.stat().st_mtime)[-1]
                built[pkg] = (w, w.name.split('-')[1], 'built' if sec > 180 else 'prebuilt_download_or_fast_build', sec)
            if built:
                pip('install', '-q', '--no-deps', '--no-index', *[str(v[0]) for v in built.values()])
        elif not importable('mamba_ssm'):
            rep['steps'].append('build_not_allowed')

    if not importable('mamba_ssm'):
        rep['status'] = 'unavailable'
    else:
        try:
            rep['verify'] = verify()
            rep['status'] = 'verified'
        except Exception as e:  # noqa: BLE001
            rep['status'] = 'installed_but_verification_failed'
            rep['verify_error'] = f'{type(e).__name__}: {e}'

    if rep['status'] == 'verified':          # save only wheels from a working environment
        if not built:
            for pkg, (w, ver) in pip_cache_wheels(fp, run=run).items():
                built[pkg] = (w, ver, 'pip_cache_harvest', None)
            rep['steps'].append('pip_cache_harvest found=' + str(sorted(built)) if built else 'pip_cache_harvest: no matching cached wheel (nothing saved)')
        for pkg, (w, ver, source, sec) in built.items():
            dst, how = save_wheel(w, store, fp, pkg, ver, source, sec)
            rep['saved'].append(dict(package=pkg, file=dst.name, how=how, sha256=sha256_file(dst), source=source))
    tag = (fp.get('gpu') or 'cpu').replace(' ', '_')
    out = Path(store)/f'env_report_{tag}_{fp["py_tag"]}_torch{fp["torch"]}_cu{fp["cuda"]}.json'
    out.write_text(json.dumps(rep, indent=2, default=str))
    log(json.dumps({k: rep[k] for k in ('status', 'steps', 'saved')}, indent=1, default=str))
    return rep
