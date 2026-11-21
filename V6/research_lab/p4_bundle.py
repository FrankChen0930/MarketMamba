"""P4 self-contained source/data bundle. Standard library bootstrap; never installs implicitly."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_BYTES = 64 * 1024**3
MAX_TOTAL_BYTES = 256 * 1024**3
MAX_MEMBERS = 10000
MAX_MANIFEST_BYTES = 4 * 1024**2


def executable_cache(name):
    return '__pycache__' in PurePosixPath(name).parts or name.endswith(('.pyc', '.pyo'))


def prepare_source(directory):
    """Trusted bootstrap must call before importing any bundled source."""
    source = Path(directory)/'source'
    caches = list(source.rglob('__pycache__'))
    bytecode = list(source.rglob('*.pyc')) + list(source.rglob('*.pyo'))
    # Inventory before deletion; never follow cache symlinks.
    for path in caches:
        if path.is_symlink(): path.unlink()
        elif path.is_dir(): shutil.rmtree(path)
    for path in bytecode:
        if path.exists() or path.is_symlink(): path.unlink()
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True


MATRIX_FILES = ('X.npy','dates.npy','end_indices.npy','splits.npy','stock_ids.npy','stock_idx.npy','y5.npy','y10.npy','manifest.json')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return dict(bytes=Path(path).stat().st_size,sha256=h.hexdigest())


def safe_name(name):
    p=PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or any(x in ('..','.') for x in name.split('/')) or ':' in name:
        raise ValueError(f'unsafe archive path: {name}')
    return p


def write_archive(output, files, metadata):
    """Allowlisted files only; hash both before and during archive publication."""
    files={str(k):Path(v) for k,v in files.items() if not executable_cache(str(k))}
    for name,path in files.items():
        safe_name(name)
        if not path.is_file() or path.is_symlink(): raise ValueError(f'not regular input: {path}')
    manifest=dict(version='p4-bundle-v1',artifact_class='GENERATED',evidence_class='ENGINEERING_ONLY',
                  metadata=metadata,files={n:digest(p) for n,p in sorted(files.items())})
    if 'bundle-manifest.json' in files: raise ValueError('reserved manifest name')
    with Path(output).open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tf:
            for name,path in sorted(files.items()):
                tf.add(path,arcname=name,recursive=False)
                if digest(path)!=manifest['files'][name]: raise ValueError('input changed during packing')
            payload=json.dumps(manifest,sort_keys=True,indent=2).encode()
            info=tarfile.TarInfo('bundle-manifest.json');info.size=len(payload)
            tf.addfile(info,io.BytesIO(payload))
        stream.flush();os.fsync(stream.fileno())
    return manifest


def verify(directory, allow_derived=False):
    root=Path(directory)
    mf=json.loads((root/'bundle-manifest.json').read_text())
    if mf['version']!='p4-bundle-v1': raise ValueError('bundle version')
    expected=set(mf['files'])|{'bundle-manifest.json'}
    paths=list(root.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('bundle links forbidden')
    if any(executable_cache(p.relative_to(root).as_posix()) for p in paths):
        raise ValueError('executable cache forbidden')
    actual={p.relative_to(root).as_posix() for p in paths if p.is_file()}
    if allow_derived:
        derived={'X_rankgauss.npy','X_rankgauss.tag','p4-rankgauss.json'}
        actual={n for n in actual if not (len(PurePosixPath(n).parts)==3 and n.startswith('folds/fold_') and PurePosixPath(n).name in derived)}
    if expected!=actual: raise ValueError('bundle inventory mismatch')
    for name,ident in mf['files'].items():
        safe_name(name); path=root/name
        if path.is_symlink() or not path.is_file() or digest(path)!=ident: raise ValueError(f'bundle hash mismatch: {name}')
    return mf


def extract(archive,destination,expected_sha256=None):
    if expected_sha256 is not None and digest(archive)['sha256']!=expected_sha256:
        raise ValueError('archive hash mismatch')
    destination=Path(destination)
    if destination.exists() or destination.is_symlink(): raise FileExistsError(destination)
    # Validate every header and exact inventory before writing any payload.
    with tarfile.open(archive,'r:gz') as tf:
        members=[]; names=[]; total=0
        for member in tf:
            if len(members)>=MAX_MEMBERS: raise ValueError('archive member limit')
            if member.issparse() or member.type==tarfile.GNUTYPE_SPARSE: raise ValueError('sparse member forbidden')
            if executable_cache(member.name): raise ValueError('executable cache forbidden')
            if member.size<0 or member.size>MAX_FILE_BYTES: raise ValueError('archive single file size limit')
            total+=member.size
            if total>MAX_TOTAL_BYTES: raise ValueError('archive total size limit')
            if member.name=='bundle-manifest.json' and member.size>MAX_MANIFEST_BYTES: raise ValueError('manifest size limit')
            members.append(member)
            safe_name(member.name)
            if not member.isfile(): raise ValueError('only regular files allowed (no links/directories/devices)')
            names.append(member.name)
        if len(names)!=len(set(names)) or 'bundle-manifest.json' not in names: raise ValueError('duplicate/missing manifest')
        mf=json.load(tf.extractfile('bundle-manifest.json'))
        if set(names)!=set(mf['files'])|{'bundle-manifest.json'}: raise ValueError('archive inventory mismatch')
        for member in members:
            if member.name!='bundle-manifest.json' and member.size!=mf['files'][member.name]['bytes']:
                raise ValueError('archive size mismatch')
        declared=sum(v['bytes'] for v in mf['files'].values())
        manifest_size=next(m.size for m in members if m.name=='bundle-manifest.json')
        if total!=declared+manifest_size or declared>MAX_TOTAL_BYTES-manifest_size:
            raise ValueError('declared total size mismatch/limit')
        destination.mkdir(parents=True)
        for member in members:
            target=destination/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(member) as src,target.open('xb') as dst:
                import shutil
                shutil.copyfileobj(src,dst,8*1024*1024)
                dst.flush();os.fsync(dst.fileno())
    verify(destination)
    return mf


def pack(folds_root, years, p1, lock, output):
    if not years or len(years)!=len(set(years)) or not set(years)<=set(range(2021,2027)):
        raise ValueError('unique P4 fold years required')
    files={}
    # Existing research implementation plus experimental dependencies; no production/models/data.
    for subtree in ('V6/research_lab','V6/experimental'):
        for path in (ROOT/subtree).rglob('*.py'):
            if 'tests' not in path.parts: files['source/'+path.relative_to(ROOT).as_posix()]=path
    for name in ('tasks/p4a-colab-contract-20261121.json','tasks/spec-p4a-colab-20261121.md','docs/operations/colab-p4.md'):
        files['source/'+name]=ROOT/name
    for year in years:
        for name in MATRIX_FILES: files[f'folds/fold_{year}/{name}']=Path(folds_root)/f'fold_{year}'/name
    for name in ('rank_average.parquet','manifest.json'): files['reference/'+name]=Path(p1)/name
    contract=json.loads((ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    if digest(lock)['sha256']!=contract['environment_lock']['sha256']:
        raise ValueError('P4 exact lock hash mismatch')
    text=Path(lock).read_text()
    if not all(v in text for v in ('torch==2.10.0+cu128','mamba_ssm-2.3.2.post1','causal_conv1d-1.7.0','#sha256=')):
        raise ValueError('P4 exact lock mismatch')
    files['requirements.lock.txt']=Path(lock)
    # P1 output identities contain original absolute paths; verify bytes, not runtime paths.
    ref=json.loads((Path(p1)/'manifest.json').read_text())['outputs']['rank_average.parquet']
    if any(digest(Path(p1)/'rank_average.parquet')[k]!=ref[k] for k in ('bytes','sha256')): raise ValueError('P1 reference hash mismatch')
    metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_status='UNMERGED / NOT_DEPLOYED; payload hashes include uncommitted source',
                  years=years,environment_lock_source=str(Path(lock).resolve()),
                  MARKETMAMBA_DATA_ROOT=os.environ.get('MARKETMAMBA_DATA_ROOT'),
                  folds_realpath=str(Path(folds_root).resolve()),p1_realpath=str(Path(p1).resolve()),
                  python=sys.version,rankgauss='EXCLUDED; rebuild with ensure_rankgauss and verify tag + input/output hashes')
    return write_archive(output,files,metadata)


def installation_commands(root):
    """Archive V3/V5.5 wheel strategy, exact current lock URLs; no guessed compilation."""
    root=Path(root);verify(root)
    if sys.version_info[:2]!=(3,12): raise RuntimeError('wheel requires Python 3.12; Human must confirm compatible runtime')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required before Colab installation')
    import shlex
    lock=shlex.quote(str(root/'requirements.lock.txt'))
    py=shlex.quote(sys.executable)
    return [f'{py} -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu128 -r {lock}',
            f'{py} -m pip check']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    pk=sub.add_parser('pack');pk.add_argument('--folds-root',type=Path,required=True);pk.add_argument('--years',type=int,nargs='+',default=[2021,2022,2023,2024,2025]);pk.add_argument('--p1',type=Path,required=True);pk.add_argument('--lock',type=Path,default=Path.home()/'.venvs/marketmamba-gpu.requirements.lock.txt');pk.add_argument('--output',type=Path,required=True)
    ex=sub.add_parser('extract');ex.add_argument('archive',type=Path);ex.add_argument('destination',type=Path);ex.add_argument('--sha256',required=True)
    ve=sub.add_parser('verify');ve.add_argument('directory',type=Path)
    ins=sub.add_parser('install-plan');ins.add_argument('directory',type=Path)
    a=p.parse_args(argv)
    if a.cmd=='pack': pack(a.folds_root,a.years,a.p1,a.lock,a.output);print(json.dumps(digest(a.output)))
    elif a.cmd=='extract': extract(a.archive,a.destination,a.sha256)
    elif a.cmd=='verify': print(json.dumps(verify(a.directory),indent=2))
    else: print('\n'.join(installation_commands(a.directory)))

if __name__=='__main__': main()
