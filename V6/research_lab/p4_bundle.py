"""P4 self-contained source/data bundle. Standard library bootstrap; never installs implicitly."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import shutil
import tarfile
import zipfile
import stat
import textwrap

ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_BYTES = 64 * 1024**3
MAX_TOTAL_BYTES = 256 * 1024**3
MAX_MEMBERS = 10000
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_READ_BYTES = 1024**2
MAX_EXTENDED_BYTES = 64 * 1024
MAX_ARCHIVE_OVERHEAD = 32 * 1024**2
MAX_ZIP_COMPRESSION_RATIO = 2000


def allowed_path(name):
    p = safe_name(name)
    if executable_cache(name): raise ValueError('executable cache forbidden')
    if p.name in {'sitecustomize.py', 'usercustomize.py', 'conftest.py', 'setup.py'} or p.suffix in {'.pth', '.so', '.pyd'}:
        raise ValueError('bundle startup/native path forbidden')
    parts = p.parts
    source = (len(parts) >= 4 and parts[:2] == ('source', 'V6')
              and parts[2] in {'research_lab', 'experimental'} and p.suffix == '.py'
              and 'tests' not in parts)
    documents = {'source/tasks/p4a-colab-contract-20261121.json',
                 'source/tasks/spec-p4a-colab-20261121.md', 'source/docs/operations/colab-p4.md',
                 'source/environments/p4a-colab-runtime.json'}
    matrix = (len(parts) == 3 and parts[0] == 'folds' and parts[1] in
              {f'fold_{y}' for y in range(2021, 2027)} and parts[2] in MATRIX_FILES)
    compat = (len(parts)==3 and parts[:2]==('source','tasks') and p.name.startswith('p4a-source-compat-') and p.suffix=='.json')
    if not (source or matrix or compat or name in documents or name in {
            'reference/rank_average.parquet', 'reference/keys.parquet', 'reference/manifest.json', 'DATASET_MANIFEST.json',
            'requirements.lock.txt', 'bundle-manifest.json'}):
        raise ValueError(f'bundle path not allowlisted: {name}')


class _BoundedReader:
    """Bound requests before they reach gzip, and cap each decompression pass."""
    def __init__(self, stream, limit):
        self.stream, self.limit, self.total = stream, limit, 0

    def read(self, size):
        if size < 0 or size > MAX_READ_BYTES: raise ValueError('archive read request limit')
        if self.total + size > self.limit: raise ValueError('archive decompression limit')
        data = self.stream.read(size)
        self.total += len(data)
        return data


def _declared_bytes(manifest):
    sizes = [ident['bytes'] for ident in manifest['files'].values()]
    if any(type(size) is not int or size < 0 or size > MAX_FILE_BYTES for size in sizes):
        raise ValueError('manifest file size limit')
    declared = sum(sizes)
    if declared > MAX_TOTAL_BYTES: raise ValueError('manifest total size limit')
    return declared


def _raw_preflight(archive):
    """Read physical headers without tarfile's implicit PAX/GNU parsing."""
    with gzip.GzipFile(fileobj=archive, mode='rb') as compressed:
        reader = _BoundedReader(compressed, MAX_TOTAL_BYTES + MAX_ARCHIVE_OVERHEAD)
        count = total = extended_run = 0
        manifest_seen = False
        while True:
            block = reader.read(512)
            if block == b'\0' * 512: break
            if len(block) != 512: raise ValueError('truncated archive header')
            try: member = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
            except (tarfile.HeaderError, UnicodeError) as e: raise ValueError('invalid archive header') from e
            count += 1
            if count > 2 * MAX_MEMBERS: raise ValueError('archive physical header limit')
            extended = member.type in {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}
            extended_run = extended_run + 1 if extended else 0
            if extended_run > 1: raise ValueError('nested extended headers forbidden')
            limit = MAX_EXTENDED_BYTES if extended else MAX_FILE_BYTES
            if member.size < 0 or member.size > limit: raise ValueError('archive header size limit')
            if member.type == tarfile.GNUTYPE_SPARSE: raise ValueError('sparse member forbidden')
            is_manifest = not extended and member.name == 'bundle-manifest.json'
            if is_manifest and member.size > MAX_MANIFEST_BYTES: raise ValueError('manifest size limit')
            if not extended:
                total += member.size
                if total > MAX_TOTAL_BYTES: raise ValueError('archive total size limit')
            remaining = (member.size + 511) // 512 * 512
            payload = bytearray()
            # Check the whole declared block before requesting any payload bytes.
            if reader.total + remaining > reader.limit: raise ValueError('archive decompression limit')
            if reader.total + remaining - total > MAX_ARCHIVE_OVERHEAD:
                raise ValueError('archive metadata overhead limit')
            while remaining:
                data = reader.read(min(remaining, MAX_READ_BYTES))
                if not data: raise ValueError('truncated archive payload')
                remaining -= len(data)
                if is_manifest: payload.extend(data)
            if is_manifest:
                if manifest_seen: raise ValueError('duplicate manifest')
                manifest_seen = True
                declared = _declared_bytes(json.loads(payload[:member.size]))
                reader.limit = min(reader.limit, declared + MAX_MANIFEST_BYTES + MAX_ARCHIVE_OVERHEAD)
                if reader.total > reader.limit: raise ValueError('declared archive decompression limit')
        if not manifest_seen: raise ValueError('missing physical manifest')
        return reader.total, reader.limit


@contextmanager
def _tar_stream(archive, limit):
    archive.seek(0)
    with gzip.GzipFile(fileobj=archive, mode='rb') as compressed:
        with tarfile.open(fileobj=_BoundedReader(compressed, limit), mode='r|') as tf:
            yield tf


def _directory_at(parent, name, create=True):
    if create:
        try: os.mkdir(name, dir_fd=parent)
        except FileExistsError: pass
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


@contextmanager
def _destination_fd(destination):
    # Anchor every ancestor at /; never resolve an untrusted symlink in a string path.
    parts = destination.absolute().parts[1:]
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in parts[:-1]:
            child = _directory_at(fd, name)
            os.close(fd); fd = child
        os.mkdir(parts[-1], dir_fd=fd)  # exclusive destination creation
        child = _directory_at(fd, parts[-1], create=False)
        os.close(fd); fd = child
        yield fd
    finally: os.close(fd)


@contextmanager
def _output_at(root_fd, name):
    fd = os.dup(root_fd)
    try:
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            child = _directory_at(fd, part)
            os.close(fd); fd = child
        output = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=fd)
        with os.fdopen(output, 'wb') as stream: yield stream
    finally: os.close(fd)



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
    if not name or '\0' in name or '' in name.split('/') or '\\' in name or p.is_absolute() or any(x in ('..','.') for x in name.split('/')) or ':' in name:
        raise ValueError(f'unsafe archive path: {name}')
    return p


def write_archive(output, files, metadata):
    """Allowlisted files only; hash both before and during archive publication."""
    files={str(k):Path(v) for k,v in files.items() if not executable_cache(str(k))}
    # Only trusted fold inputs may resolve symlinks; archive members remain regular.
    for n,p in list(files.items()):
        if n.startswith('folds/'):
            root=Path(metadata.get('folds_realpath',p.parent)).resolve()
            resolved=p.resolve(strict=True)
            if not resolved.is_relative_to(root): raise ValueError('fold source escapes designated root')
            files[n]=resolved
    for name,path in files.items():
        allowed_path(name)
        if not path.is_file() or path.is_symlink(): raise ValueError(f'not regular input: {path}')
    manifest=dict(version='p4-bundle-v1',artifact_class='GENERATED',evidence_class='ENGINEERING_ONLY',
                  metadata=metadata,files={n:digest(p) for n,p in sorted(files.items())})
    if 'bundle-manifest.json' in files: raise ValueError('reserved manifest name')
    if Path(output).suffix == '.zip':
        return _write_zip(output, files, manifest)
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


def _write_zip(output, files, manifest):
    aliases={}; seen={}
    for name, ident in manifest['files'].items():
        if name.startswith('folds/') and name.endswith('/X.npy'):
            key=(ident['bytes'],ident['sha256'])
            if key in seen: aliases[name]=seen[key]
            else: seen[key]=name
    manifest['aliases']=aliases
    manifest['total_bytes']=_declared_bytes(manifest)
    payload=json.dumps(manifest,sort_keys=True,indent=2).encode()
    if len(payload)>MAX_MANIFEST_BYTES or len(files)+1>MAX_MEMBERS or _declared_bytes(manifest)+len(payload)>MAX_TOTAL_BYTES:
        raise ValueError('bundle manifest/member limit')
    with Path(output).open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
            for name,path in sorted(files.items()):
                if name not in aliases: z.write(path,name)
                if digest(path)!=manifest['files'][name]: raise ValueError('input changed during packing')
            z.writestr('bundle-manifest.json',payload)
        stream.flush();os.fsync(stream.fileno())
    return manifest


def _zip_inventory(z):
    members=z.infolist()
    if len(members)>MAX_MEMBERS: raise ValueError('archive member limit')
    names=[];total=0
    for member in members:
        allowed_path(member.filename)
        mode=stat.S_IFMT(member.external_attr >> 16)
        if member.is_dir() or mode not in (0,stat.S_IFREG):
            raise ValueError('only regular zip files allowed (no links)')
        if member.flag_bits & 1: raise ValueError('encrypted member forbidden')
        if member.file_size<0 or member.file_size>MAX_FILE_BYTES: raise ValueError('archive file size limit')
        total+=member.file_size
        if total>MAX_TOTAL_BYTES: raise ValueError('archive total size limit')
        names.append(member.filename)
    if len(names)!=len(set(names)) or 'bundle-manifest.json' not in names:
        raise ValueError('duplicate/missing manifest')
    info=z.getinfo('bundle-manifest.json')
    if info.file_size>MAX_MANIFEST_BYTES: raise ValueError('manifest size limit')
    mf=json.loads(z.read(info))
    if mf.get('version')!='p4-bundle-v1': raise ValueError('bundle version')
    declared=_declared_bytes(mf)
    if declared+info.file_size>MAX_TOTAL_BYTES or len(mf['files'])+1>MAX_MEMBERS:
        raise ValueError('expanded bundle limit')
    aliases=mf.get('aliases',{})
    for name,ident in mf['files'].items(): allowed_path(name)
    for alias,target in aliases.items():
        if (alias not in mf['files'] or target not in mf['files'] or target in aliases
            or alias==target or not alias.startswith('folds/') or not alias.endswith('/X.npy')
            or not target.startswith('folds/') or not target.endswith('/X.npy')
            or mf['files'][alias]!=mf['files'][target]):
            raise ValueError('invalid dedup alias')
    if set(names)!=(set(mf['files'])-set(aliases))|{'bundle-manifest.json'}:
        raise ValueError('archive inventory mismatch')
    for member in members:
        # Check logical expected size first; 2000x accommodates even zero-filled npy.
        if member.file_size > max(1,member.compress_size)*MAX_ZIP_COMPRESSION_RATIO:
            raise ValueError('archive compression ratio limit')
        if member.filename!='bundle-manifest.json' and member.file_size!=mf['files'][member.filename]['bytes']:
            raise ValueError('archive size mismatch')
    return mf


def _extract_zip(archive, destination):
    with zipfile.ZipFile(archive) as z:
        mf=_zip_inventory(z)  # All names, limits and aliases before destination creation.
        with _destination_fd(destination) as root_fd:
            for name,ident in mf['files'].items():
                target=mf.get('aliases',{}).get(name,name)
                h=hashlib.sha256();size=0
                with z.open(target) as src, _output_at(root_fd,name) as dst:
                    while True:
                        block=src.read(MAX_READ_BYTES)
                        if not block: break
                        size+=len(block)
                        if size>ident['bytes']: raise ValueError('zip payload size limit')
                        h.update(block);dst.write(block)
                    dst.flush();os.fsync(dst.fileno())
                if dict(bytes=size,sha256=h.hexdigest())!=ident: raise ValueError('bundle hash mismatch')
            with _output_at(root_fd,'bundle-manifest.json') as dst:
                dst.write(z.read('bundle-manifest.json'));dst.flush();os.fsync(dst.fileno())
    verify(destination)
    return mf


def verify(directory, allow_derived=False):
    root=Path(directory)
    if root.is_symlink(): raise ValueError('bundle root links forbidden')
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
        allowed_path(name); path=root/name
        if path.is_symlink() or not path.is_file() or digest(path)!=ident: raise ValueError(f'bundle hash mismatch: {name}')
    return mf


def extract(archive,destination,expected_sha256=None):
    archive_sha256 = digest(archive)['sha256']
    print(json.dumps(dict(archive_sha256=archive_sha256)))
    if expected_sha256 is not None and archive_sha256!=expected_sha256:
        raise ValueError('archive hash mismatch')
    destination=Path(destination)
    if destination.exists() or destination.is_symlink(): raise FileExistsError(destination)
    # Hold the same archive inode across all passes. Physical preflight happens before
    # tarfile.open, then streaming inventory validation, then bounded streaming writes.
    with Path(archive).open('rb') as archive_file:
        if zipfile.is_zipfile(archive_file):
            archive_file.seek(0)
            return _extract_zip(archive_file, destination)
        archive_file.seek(0)
        raw_size, budget = _raw_preflight(archive_file)
        with _tar_stream(archive_file, budget) as tf:
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
                allowed_path(member.name)
                if not member.isfile(): raise ValueError('only regular files allowed (no links/directories/devices)')
                names.append(member.name)
                if member.name=='bundle-manifest.json':
                    mf=json.loads(tf.extractfile(member).read(MAX_MANIFEST_BYTES))
            if len(names)!=len(set(names)) or 'bundle-manifest.json' not in names: raise ValueError('duplicate/missing manifest')
            if set(names)!=set(mf['files'])|{'bundle-manifest.json'}: raise ValueError('archive inventory mismatch')
            for member in members:
                if member.name!='bundle-manifest.json' and member.size!=mf['files'][member.name]['bytes']:
                    raise ValueError('archive size mismatch')
            declared=_declared_bytes(mf)
            manifest_size=next(m.size for m in members if m.name=='bundle-manifest.json')
            if total!=declared+manifest_size or declared>MAX_TOTAL_BYTES-manifest_size:
                raise ValueError('declared total size mismatch/limit')
        budget = min(MAX_TOTAL_BYTES + MAX_ARCHIVE_OVERHEAD,
                     declared + MAX_MANIFEST_BYTES + MAX_ARCHIVE_OVERHEAD)
        if raw_size > budget: raise ValueError('declared archive decompression limit')
        with _destination_fd(destination) as root_fd, _tar_stream(archive_file, budget) as tf:
            for member, expected in zip(tf, members, strict=True):
                if (member.name, member.size, member.type) != (expected.name, expected.size, expected.type):
                    raise ValueError('archive changed between passes')
                with tf.extractfile(member) as src, _output_at(root_fd, member.name) as dst:
                    shutil.copyfileobj(src, dst, MAX_READ_BYTES)
                    dst.flush(); os.fsync(dst.fileno())
    verify(destination)
    return mf


def pack(folds_root, years, p1, lock, output, resume_from_commit=None):
    if not years or len(years)!=len(set(years)) or not set(years)<=set(range(2021,2027)):
        raise ValueError('unique P4 fold years required')
    files={}
    # Existing research implementation plus experimental dependencies; no production/models/data.
    for subtree in ('V6/research_lab','V6/experimental'):
        for path in (ROOT/subtree).rglob('*.py'):
            if 'tests' not in path.parts: files['source/'+path.relative_to(ROOT).as_posix()]=path
    for name in ('tasks/p4a-colab-contract-20261121.json','tasks/spec-p4a-colab-20261121.md','docs/operations/colab-p4.md','environments/p4a-colab-runtime.json'):
        files['source/'+name]=ROOT/name
    external=(Path(folds_root)/'shared').is_dir()
    if not external:
        for year in years:
            for name in MATRIX_FILES:files[f'folds/fold_{year}/{name}']=Path(folds_root)/f'fold_{year}'/name
        for name in ('rank_average.parquet','manifest.json'):files['reference/'+name]=Path(p1)/name
    else:
        validate_dataset(folds_root)
        if json.loads((Path(folds_root)/'DATASET_MANIFEST.json').read_text())['files']['shared/X.npy']!=DATASET_X:
            raise ValueError('formal dataset X identity mismatch')
        files['DATASET_MANIFEST.json']=Path(folds_root)/'DATASET_MANIFEST.json'
    contract=json.loads((ROOT/'tasks/p4a-colab-contract-20261121.json').read_text())
    if digest(lock)['sha256']!=contract['environment_lock']['sha256']:
        raise ValueError('P4 exact lock hash mismatch')
    text=Path(lock).read_text()
    if not all(v in text for v in ('torch==2.10.0+cu128','mamba_ssm-2.3.2.post1','causal_conv1d-1.7.0','#sha256=')):
        raise ValueError('P4 exact lock mismatch')
    files['requirements.lock.txt']=Path(lock)
    if resume_from_commit:
        if not external:raise ValueError('source-compatible upgrade requires DATASET_MANIFEST')
        from .p4_source_compat import declaration
        import tempfile
        # Temporary GENERATED declaration lives only for this immutable ZIP build.
        with tempfile.TemporaryDirectory(prefix='p4-source-compat-') as scratch:
            target=Path(scratch)/'declaration.json'
            target.write_text(json.dumps(declaration(resume_from_commit,
                {n:digest(p) for n,p in files.items() if n.startswith('source/')},
                digest(Path(folds_root)/'DATASET_MANIFEST.json')),sort_keys=True,indent=2))
            tag=subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()
            files[f'source/tasks/p4a-source-compat-{tag}.json']=target
            return _pack_reference(files,external,folds_root,years,p1,output,lock)
    return _pack_reference(files,external,folds_root,years,p1,output,lock)


def _pack_reference(files,external,folds_root,years,p1,output,lock):
    # P1 output identities contain original absolute paths; verify bytes, not runtime paths.
    ref=json.loads((Path(p1)/'manifest.json').read_text())['outputs']['rank_average.parquet']
    if any(digest(Path(p1)/'rank_average.parquet')[k]!=ref[k] for k in ('bytes','sha256')): raise ValueError('P1 reference hash mismatch')
    metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_status='UNMERGED / NOT_DEPLOYED; payload hashes include uncommitted source',
                  years=years,environment_lock_source=str(Path(lock).resolve()),
                  MARKETMAMBA_DATA_ROOT=os.environ.get('MARKETMAMBA_DATA_ROOT'),
                  folds_realpath=str(Path(folds_root).resolve()),p1_realpath=str(Path(p1).resolve()),
                  python=sys.version,rankgauss='EXCLUDED; rebuild with ensure_rankgauss and verify tag + input/output hashes')
    if external:
        import tempfile
        import pandas as pd
        metadata['external_dataset']=digest(Path(folds_root)/'DATASET_MANIFEST.json')
        with tempfile.TemporaryDirectory(prefix='p4-reference-keys-') as scratch:
            frame=pd.read_parquet(Path(p1)/'rank_average.parquet')
            keys=frame.loc[frame.system.eq('c0plus'),['fold','date','code','label_y5']]
            if keys.duplicated(['fold','date','code']).any() or keys.empty:raise ValueError('reference key duplicates/empty')
            target=Path(scratch)/'keys.parquet';keys.to_parquet(target,index=False)
            files['reference/keys.parquet']=target
            provenance=Path(scratch)/'manifest.json'
            provenance.write_text(json.dumps(dict(source=ref,outputs={'keys.parquet':digest(target)})))
            files['reference/manifest.json']=provenance
            return write_archive(output,files,metadata)
    return write_archive(output,files,metadata)


SHARED_FILES = tuple(n for n in MATRIX_FILES if n != 'splits.npy')
DATASET_X = dict(bytes=1425394112, sha256='9778392d4c337d1f9cb5768002340a0d1aec13f7689b2f601b77bd2329b55363')


def dataset_names():
    return {f'shared/{n}' for n in SHARED_FILES} | {f'folds/fold_{y}/splits.npy' for y in range(2021,2027)}


def dataset_inventory(root):
    root=Path(root)
    paths=[p for subtree in ('shared','folds') for p in (root/subtree).rglob('*')]
    if root.is_symlink() or any(p.is_symlink() for p in paths): raise ValueError('dataset links forbidden')
    names={p.relative_to(root).as_posix() for p in paths if p.is_file()}
    if names!=dataset_names(): raise ValueError('dataset inventory mismatch')
    return names


def dataset_manifest(root, enforce_x=True):
    """Offline one-time inventory; no links, no formal matrix writes."""
    root=Path(root);names=dataset_inventory(root)
    files={n:digest(root/n) for n in sorted(names)}
    if enforce_x and files['shared/X.npy']!=DATASET_X: raise ValueError('matrix_v1 X identity mismatch')
    mf=dict(version='p4-dataset-v1',artifact_class='GENERATED',files=files,
            evidence_class='HISTORICAL_SIMULATION_PROXY / RESEARCH_ONLY')
    from .rolling_seq_scores import atomic_json
    atomic_json(root/'DATASET_MANIFEST.json',mf)
    return mf


def validate_dataset(root):
    root=Path(root)
    if (root/'DATASET_MANIFEST.json').is_symlink(): raise ValueError('dataset manifest links forbidden')
    mf=json.loads((root/'DATASET_MANIFEST.json').read_text())
    if mf.get('version')!='p4-dataset-v1' or set(mf['files'])!=dataset_inventory(root):
        raise ValueError('dataset inventory/version mismatch')
    for n,ident in mf['files'].items():
        if digest(root/n)!=ident: raise ValueError('dataset hash mismatch: '+n)
    return mf


def prepare_dataset(drive, local):
    """Verify Drive bytes, copy to SSD, build/reuse hash-bound shared rankgauss once."""
    from .seq_runner import ensure_rankgauss, _x_tag
    from .rolling_seq_scores import atomic_json
    drive,local=Path(drive),Path(local)
    if local.resolve()==drive.resolve() or local.resolve().is_relative_to(drive.resolve()):
        raise ValueError('local dataset must be separate from Drive')
    mf=validate_dataset(drive);matrix_identity=digest(drive/'DATASET_MANIFEST.json')
    for n,ident in mf['files'].items():
        dst=local/n;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.is_symlink(): raise ValueError('local dataset original links forbidden')
        if not dst.exists() or digest(dst)!=ident:
            tmp=dst.with_name(dst.name+'.partial');shutil.copy2(drive/n,tmp)
            if digest(tmp)!=ident: raise ValueError('SSD copy hash mismatch')
            os.replace(tmp,dst)
    shared=local/'shared';cache=drive/'rankgauss_cache';cache.mkdir(exist_ok=True)
    cache_contract=dict(X=mf['files']['shared/X.npy'],ends=mf['files']['shared/end_indices.npy'],
                        implementation=digest(Path(__file__).with_name('seq_runner.py')))
    meta=cache/'manifest.json';cached=cache/'X_rankgauss.npy'
    valid=False
    if meta.is_file() and cached.is_file() and not meta.is_symlink() and not cached.is_symlink():
        saved=json.loads(meta.read_text())
        valid=(saved.get('contract')==cache_contract and saved.get('output')==digest(cached)
               and (cache/'X_rankgauss.tag').is_file() and not (cache/'X_rankgauss.tag').is_symlink()
               and (cache/'X_rankgauss.tag').read_text()==saved.get('tag'))
    if valid:
        if not (shared/'X_rankgauss.npy').exists() or digest(shared/'X_rankgauss.npy')!=saved['output']:
            shutil.copy2(cached,shared/'X_rankgauss.npy')
        (shared/'X_rankgauss.tag').write_text(_x_tag(shared))
    else:
        # Tag alone cannot admit an old cache after X or implementation changes.
        (shared/'X_rankgauss.tag').unlink(missing_ok=True)
        ensure_rankgauss(shared)
        tmp=cache/'X_rankgauss.partial.npy';shutil.copy2(shared/'X_rankgauss.npy',tmp)
        if digest(tmp)!=digest(shared/'X_rankgauss.npy'): raise ValueError('Drive rankgauss copy mismatch')
        os.replace(tmp,cached)
        (cache/'X_rankgauss.tag').write_text(_x_tag(shared))
        atomic_json(meta,dict(contract=cache_contract,output=digest(cached),tag=_x_tag(shared)))
    for year in range(2021,2027):
        fold=local/f'folds/fold_{year}'
        for name in (*SHARED_FILES,'X_rankgauss.npy','X_rankgauss.tag'):
            dst=fold/name;target=shared/name
            if dst.is_symlink():
                if dst.resolve()!=target.resolve(): raise ValueError('unexpected local fold link')
            elif dst.exists(): raise ValueError('local fold shared collision')
            else:dst.symlink_to(target.resolve())
    atomic_json(local/'prepared.json',dict(matrix_identity=matrix_identity,rankgauss=digest(cached)))
    return dict(matrix_identity=matrix_identity,rankgauss=digest(cached))

def installation_commands(root, wheel_dir=None):
    """Plan only: Drive mode installs exactly two local wheels without dependencies."""
    root=Path(root);verify(root)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required before Colab installation')
    import shlex
    py=shlex.quote(sys.executable)
    if wheel_dir is not None:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename, canonicalize_name
        directory=Path(wheel_dir)
        compatible=set(sys_tags())
        found={'causal-conv1d':[], 'mamba-ssm':[]}
        incompatible=[]
        for path in sorted(directory.glob('*.whl')):
            name,version,build,tags=parse_wheel_filename(path.name)
            name=canonicalize_name(name)
            if name not in found: continue
            print(json.dumps(dict(wheel=path.name,version=str(version),tags=sorted(map(str,tags)),
                                  identity=digest(path))))
            found[name].append(path)
            if not tags & compatible: incompatible.append(path.name)
        if incompatible:
            raise RuntimeError('wheel Python/ABI/platform mismatch: '+', '.join(incompatible)+
                               '; switch Colab runtime version to match the wheel tags')
        if any(len(paths)!=1 for paths in found.values()):
            raise ValueError('wheel directory requires exactly one causal_conv1d and one mamba_ssm wheel; missing/ambiguous wheel')
        wheels=' '.join(shlex.quote(str(paths[0].resolve())) for paths in found.values())
        return [f'{py} -m pip install --no-deps --no-index --find-links {shlex.quote(str(directory.resolve()))} {wheels}']
    if sys.version_info[:2]!=(3,12): raise RuntimeError('wheel requires Python 3.12; Human must confirm compatible runtime')
    lock=shlex.quote(str(root/'requirements.lock.txt'))
    return [f'{py} -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu128 -r {lock}']


def colab_notebook(run_name='02_formal', drive_root='/content/drive/MyDrive/MarketMamba', expected_archive=None):
    """Generated launch UX; research/environment contracts remain authoritative."""
    if not run_name or run_name in {'.','..'} or '/' in run_name or '\\' in run_name or any(c in run_name for c in '\r\n'):
        raise ValueError('run-name must be a single folder name')
    drive_root=Path(drive_root)
    if not drive_root.is_absolute() or any(c in str(drive_root) for c in '\r\n'):
        raise ValueError('drive-root must be an absolute path')
    cells=[]
    def code(source):
        cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=textwrap.dedent(source).strip()+'\n'))
    def markdown(source):
        cells.append(dict(cell_type='markdown',metadata={},source=source))
    code('''
        # 設定：P4 僅記錄耗時，不設時數／CU 上限；>=85°C 仍停止。
        MODE = 'FORMAL'  # 可選 SPEED_TEST；兩者輸出隔離
        REHEARSE_FIRST = MODE != 'SPEED_TEST'  # FORMAL／resume 預設 True；可明設 False
        if MODE not in ('SPEED_TEST','FORMAL'): raise ValueError('STOP: invalid MODE')
        from pathlib import Path
        RUN_DIR = __RUN_DIR__
        DATA_DIR = __DATA_DIR__
        EXPERIMENT_DIR = str(Path(RUN_DIR).parent)  # 整個 P4a 共用，不隨交付子目錄重置
        WHEEL_DIR = __WHEEL_DIR__
        LEGACY_WHEEL_DIR = '/content/drive/MyDrive/Mamba_Wheels_T4_new'
        EXPECTED_ARCHIVE = __EXPECTED_ARCHIVE__  # deliver 自動綁定 ZIP bytes/SHA
        BUNDLE_NAME = 'p4a-bundle.zip'
        LOCAL_ROOT = __LOCAL_ROOT__
        ALLOW_NEW_RUN = False  # 只有明確要重跑新批次時才設 True；不相容預設 STOP
    ''')
    cells[0]['source']=cells[0]['source'].replace('__RUN_DIR__',repr(str(drive_root/'p4a'/run_name))).replace('__DATA_DIR__',repr(str(drive_root/'data/matrix_v1'))).replace('__WHEEL_DIR__',repr(str(drive_root/'wheels'))).replace('__EXPECTED_ARCHIVE__',repr(expected_archive)).replace('__LOCAL_ROOT__',repr('/content/p4a-'+expected_archive['sha256'][:16] if expected_archive else '/content/p4a'))
    markdown('兩檔交付：將同批 p4a-bundle.zip 與 notebook 上傳 RUN_DIR，再開啟 notebook 從上到下執行。需 GPU runtime。UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED；Colab ZIP／編譯／Drive 行為 UNVERIFIED。只執行自己信任來源的這兩個檔案；內部 hash 不證明來源。')
    code('''
        from pathlib import Path
        import os, sys, json, subprocess, hashlib, zipfile, tempfile, importlib.util, stat
        def find_archive(run_dir, experiment_dir, content_dir, name, expected):
            candidates=[Path(run_dir)/name,Path(experiment_dir)/'inputs'/name,Path(content_dir)/name]
            local=candidates[-1]
            ignored=False
            if local.is_file() and expected:
                with local.open('rb') as stream:
                    actual=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual!=expected['sha256']:
                    print('WARNING: '+str(local)+' SHA 與預期不同，忽略此檔；可能是殘留或上傳不完整。Drive 優先。')
                    ignored=True
            for candidate in candidates:
                if candidate.is_file() and not (candidate==local and ignored): return candidate
            raise FileNotFoundError('STOP: 找不到合格 ZIP，已搜尋：'+', '.join(map(str,candidates)))

        def resolve_wheel_dir(preferred, legacy):
            for directory in (Path(preferred),Path(legacy)):
                if directory.is_dir() and any(directory.glob('*.whl')): return directory
            return Path(preferred)  # 無 wheel 時編譯寫回 MarketMamba/wheels

        def validate_archive_size(archive, expected):
            message='STOP: ZIP 大小 (size) 與交付／bundle manifest 不符，可能上傳不完整；請重新核對 Drive 檔案。'
            if expected and archive.stat().st_size!=expected['bytes']: raise ValueError(message)
            if expected:
                with archive.open('rb') as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=expected['sha256']:
                        raise ValueError('STOP: ZIP SHA mismatch；請核對同批交付檔案')
            try:
                with zipfile.ZipFile(archive) as z:
                    info=z.getinfo('bundle-manifest.json')
                    if info.file_size>4*1024**2: raise ValueError('STOP: manifest size limit')
                    manifest=json.loads(z.read(info))
                    total=sum(ident['bytes'] for ident in manifest['files'].values())
                    if total!=manifest.get('total_bytes',total): raise ValueError(message)
                    aliases=manifest.get('aliases',{})
                    members=[m for m in z.infolist() if m.filename!='bundle-manifest.json']
                    physical=sum(ident['bytes'] for name,ident in manifest['files'].items() if name not in aliases)
                    if sum(m.file_size for m in members)!=physical: raise ValueError(message)
                    if any(m.file_size!=manifest['files'][m.filename]['bytes'] for m in members): raise ValueError(message)
            except (zipfile.BadZipFile,EOFError) as error:
                raise ValueError(message) from error

        from google.colab import drive
        drive.mount('/content/drive')
        archive = find_archive(RUN_DIR,EXPERIMENT_DIR,'/content',BUNDLE_NAME,EXPECTED_ARCHIVE)
        validate_archive_size(archive,EXPECTED_ARCHIVE)
        print('archive',archive,'bytes',archive.stat().st_size,'SHA-256',end=' ')
        with archive.open('rb') as stream:
            print(hashlib.file_digest(stream,'sha256').hexdigest())
        # 僅用標準庫取出固定 bootstrap，先限制 inventory、類型及兩個小檔大小。
        bootstrap_name = 'source/V6/research_lab/p4_bundle.py'
        with zipfile.ZipFile(archive) as z:
            members=z.infolist()
            if len(members)>10000: raise ValueError('STOP: member limit')
            if len({m.filename for m in members})!=len(members): raise ValueError('STOP: duplicate member')
            for name in ('bundle-manifest.json',bootstrap_name):
                member=z.getinfo(name)
                if member.file_size>4*1024**2 or member.is_dir() or stat.S_IFMT(member.external_attr>>16) not in (0,stat.S_IFREG):
                    raise ValueError('STOP: invalid bootstrap/manifest')
            manifest=json.loads(z.read('bundle-manifest.json'))
            payload=z.read(bootstrap_name)
            if dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())!=manifest['files'][bootstrap_name]:
                raise ValueError('STOP: bootstrap hash mismatch')
        sys.dont_write_bytecode=True
        bootstrap_dir=tempfile.TemporaryDirectory(prefix='p4-bootstrap-')
        bootstrap=Path(bootstrap_dir.name)/'p4_bundle.py'
        bootstrap.write_bytes(payload)
        spec=importlib.util.spec_from_file_location('p4_bootstrap',bootstrap)
        p4=importlib.util.module_from_spec(spec);spec.loader.exec_module(p4)
        root=Path(LOCAL_ROOT)
        if root.exists():
            # 重連沿用目錄前比對同一 bundle identity，不覆寫 partial。
            with zipfile.ZipFile(archive) as z:
                if (root/'bundle-manifest.json').read_bytes()!=z.read('bundle-manifest.json'):
                    raise ValueError('STOP: existing bundle identity mismatch')
        else: p4.extract(archive,root)
        p4.prepare_source(root)
        p4.verify(root,allow_derived=True)
        sys.path[:0]=[str(root/'source/V6'),str(root/'source')]
    ''')
    code('''
        # 資料準備：Drive -> 本機 SSD；只在 /content 建立 fold links。
        from research_lab.p4_bundle import prepare_dataset, digest
        manifest=json.loads((root/'bundle-manifest.json').read_text())
        if manifest['metadata'].get('external_dataset'):
            if digest(Path(DATA_DIR)/'DATASET_MANIFEST.json')!=manifest['metadata']['external_dataset']:
                raise ValueError('STOP: dataset manifest differs from bundle identity')
            prepare_dataset(DATA_DIR,'/content/data')
        import torch, importlib.metadata, platform
        if not torch.cuda.is_available(): raise RuntimeError('STOP: CUDA 不可用；請選 GPU runtime')
        contract=json.loads((root/'source/environments/p4a-colab-runtime.json').read_text())
        def installed_version(name):
            try: return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: return None
        actual_packages={n:installed_version(n) for n in contract['packages']}
        print(json.dumps(dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                              gpu=torch.cuda.get_device_name(0),packages=actual_packages),ensure_ascii=False,indent=2))
        subprocess.run(['nvidia-smi'],check=True)
        matches=(platform.python_version()==contract['runtime']['python'] and
                 torch.version.cuda==contract['runtime']['cuda'] and
                 bool(torch._C._GLIBCXX_USE_CXX11_ABI)==contract['runtime']['cxx11abi'] and
                 all(actual_packages[n]==v for n,v in contract['packages'].items()))
        needs_compile=False
        if matches:
            print('已符合環境契約，跳過 wheel 安裝')
        else:
            wheels=resolve_wheel_dir(WHEEL_DIR,LEGACY_WHEEL_DIR)
            both_absent=not list(wheels.glob('mamba_ssm-*.whl')) and not list(wheels.glob('causal_conv1d-*.whl'))
            if both_absent:
                needs_compile=True
                print('兩個 wheel 均未找到，下一格將編譯（需 GPU，可能 30–90 分鐘或更久；不包含在 runner 預算帳本）')
            else:
                for command in p4.installation_commands(root,wheel_dir=wheels):
                    subprocess.run(command,shell=True,check=True)
    ''')
    markdown('編譯備援格：只有兩個 wheel 均不存在才執行；部分缺失、標籤不符或多版本皆 STOP。使用官方 PyPI sdist 並核對其 SHA；mamba setup.py 的所有 -std=c++17 改為 -std=c++20。這是未在本輪 Colab 現場驗證的流程。已裝套件符合契約時整格略過。')
    code('''
        if needs_compile:
            import urllib.request, tarfile
            if not torch.cuda.is_available(): raise RuntimeError('STOP: CUDA required for compilation')
            wheels=Path(WHEEL_DIR);wheels.mkdir(parents=True,exist_ok=True)
            build_env=dict(os.environ,MAMBA_FORCE_BUILD='TRUE',CAUSAL_CONV1D_FORCE_BUILD='TRUE',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='7.5')
            if torch.cuda.get_device_capability(0)!=(7,5): raise RuntimeError('STOP: 此編譯配方僅供 T4 sm75')
            with tempfile.TemporaryDirectory(prefix='p4-build-') as build_dir:
                for package,version in (('causal-conv1d','1.7.0'),('mamba-ssm','2.3.2.post1')):
                    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json') as response:
                        release=json.load(response)
                    artifact=next(a for a in release['urls'] if a['packagetype']=='sdist')
                    sdist=Path(build_dir)/(package+'.tar.gz')
                    urllib.request.urlretrieve(artifact['url'],sdist)
                    if p4.digest(sdist)['sha256']!=artifact['digests']['sha256']: raise ValueError('STOP: sdist hash mismatch')
                    unpack=Path(build_dir)/package;unpack.mkdir()
                    with tarfile.open(sdist) as tf:
                        for member in tf:
                            p4.safe_name(member.name)
                            if not (member.isfile() or member.isdir()): raise ValueError('STOP: sdist links forbidden')
                        tf.extractall(unpack,filter='data')
                    setup=next(unpack.glob('*/setup.py'))
                    if package=='mamba-ssm':
                        text=setup.read_text()
                        if '-std=c++17' not in text: raise ValueError('STOP: sdist recipe drift')
                        setup.write_text(text.replace('-std=c++17','-std=c++20'))
                    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',
                                    '--wheel-dir',str(wheels),str(setup.parent)],env=build_env,check=True)
            for command in p4.installation_commands(root,wheel_dir=wheels):
                subprocess.run(command,shell=True,check=True)
        # 所有環境路徑均測 import + forward/backward；不能只信 metadata。
        from mamba_ssm import Mamba2
        model=Mamba2(d_model=64,d_state=16,headdim=8,expand=2).cuda()
        x=torch.randn(8,60,64,device='cuda',requires_grad=True)
        y=model(x);y.square().mean().backward();torch.cuda.synchronize()
        smoke=dict(output_finite=bool(torch.isfinite(y).all()),input_grad_finite=bool(torch.isfinite(x.grad).all()))
        print('Mamba2 smoke',smoke)
        if not all(smoke.values()): raise RuntimeError('STOP: Mamba2 smoke failed')
        del model,x,y
        torch.cuda.empty_cache()
        # Environment contract is the gate; no pip check.
    ''')
    code('''
        if MODE not in ('SPEED_TEST','FORMAL'): raise ValueError('STOP: MODE must be SPEED_TEST or FORMAL')
        p4.prepare_source(root);p4.verify(root,allow_derived=True)
        training_env=dict(os.environ,PYTHONPATH=f'{root}/source/V6:{root}/source',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        BUDGET_DIR=Path(EXPERIMENT_DIR)/('budget_speedtest' if MODE=='SPEED_TEST' else 'budget')
        PROGRESS_DIR=Path(EXPERIMENT_DIR)/'progress'/('speedtest' if MODE=='SPEED_TEST' else 'formal')
        common=[sys.executable,'-u','-m','research_lab.p4_colab','--bundle',str(root),'--drive-root',str(Path(EXPERIMENT_DIR)/'budget'),
                '--data-dir','/content/data']
        if ALLOW_NEW_RUN: common.append('--allow-new-run')
        # FROZEN + OBSERVED 環境完整核對；DRAFT 仍拒絕。
        def stream_training(command,env):
            # 合併 stderr 並持續排空 PIPE；排練另由 gate 驗 JSON、產物與 identity，返回碼 0 不代表通過。
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,bufsize=1,env=env) as process:
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True)
                    actual=process.wait()
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
                    print('訓練步驟中斷；期望=正常完成；實際=kernel 中斷；下一步=查看 Drive 進度與檢查點，使用原交付重新執行',flush=True)
                    raise
            if actual!=0:
                print(f'訓練步驟失敗；期望=回傳碼 0；實際=回傳碼 {actual}；下一步=查看上方最後錯誤及 Drive 進度，檢查環境／空間／identity；保留帳本與原交付，修復原因後重跑',flush=True)
                raise subprocess.CalledProcessError(actual,command)
        try:
            from research_lab.p4_colab import launch_with_rehearsal, environment_snapshot
            # prepared.json 由前格逐檔驗證／Drive→SSD 準備後產生，不猜資料身分。
            rehearsal_identity=None
            if REHEARSE_FIRST:
                runtime=json.loads((root/'source/environments/p4a-colab-runtime.json').read_text())
                rehearsal_identity=dict(source=p4.digest(root/'bundle-manifest.json'),
                    data=json.loads(Path('/content/data/prepared.json').read_text()),
                    data_realpath=str(Path('/content/data').resolve()),
                    environment=environment_snapshot(runtime['packages']))
            launch_with_rehearsal(common+(['--speed-test'] if MODE=='SPEED_TEST' else []),training_env,
                                  PROGRESS_DIR,rehearsal_identity,stream_training,enabled=REHEARSE_FIRST)
        finally:
            print('結果資料夾：',BUDGET_DIR,'進度檔：',PROGRESS_DIR/'PROGRESS.txt',flush=True)
            print('run manifests',*[str(p) for p in BUDGET_DIR.glob('run-*/manifest.json')],sep='\\n',flush=True)
    ''')
    code('''
        # runner 已直接持久寫入 Drive：speed JSON、run manifest 與預算帳本。
        if MODE=='SPEED_TEST':
            for path in sorted(BUDGET_DIR.glob('run-*/**/speed.json')):
                report=json.loads(path.read_text())
                print(path)
                print(json.dumps({k:report[k] for k in ('arm','epoch_seconds','gpu_model') if k in report},ensure_ascii=False))
                records=report['telemetry']
                print(dict(temperature_c_max=max(r['temperature_c'] for r in records),
                           clock_mhz_min=min(r['clock_mhz'] for r in records),clock_mhz_max=max(r['clock_mhz'] for r in records)))
        print('run manifests',*[str(p) for p in BUDGET_DIR.glob('run-*/manifest.json')],sep='\\n')
    ''')
    markdown('目前進度摘要：Colab 單一 kernel 訓練執行中無法再執行其他格，主要請用手機或 Drive 網頁開啟 EXPERIMENT_DIR/progress/formal/PROGRESS.txt（speed-test 在 progress/speedtest/）。以下儲存格可在完成或中斷後讀取快照；快照不是完成或交易證據。')
    code('''
        # 目前進度摘要：不依賴訓練格已成功執行。
        import json
        from pathlib import Path
        progress_dir=Path(EXPERIMENT_DIR)/'progress'/('speedtest' if MODE=='SPEED_TEST' else 'formal')
        snapshot=progress_dir/'progress.json'
        if snapshot.exists():
            summary=json.loads(snapshot.read_text())
            # SPEED_TEST 僅顯示工程欄位，不轉印未知／歷史績效欄位。
            keys=('observed_at','status','mode','queue','fit_index','total_fits','completed_fits',
                  'arm','year','seed','window','lr','epoch','step','loss','fit_elapsed_seconds',
                  'fit_eta_seconds','queue_eta_seconds','temperature_c','clock_mhz','last_checkpoint_at')
            print('目前進度摘要（請確認更新時間；斷線後可能是舊快照）')
            for key in keys+(() if MODE=='SPEED_TEST' else ('val_ic5',)):
                print(key,summary.get(key) if summary.get(key) is not None else ('預估中' if 'eta' in key else '未知／尚無'))
        else:
            print('尚無進度快照；下一步：確認訓練已啟動、模式與 Drive 掛載，查看',snapshot)
    ''')
    markdown('跨版本接續：斷線後開新 notebook，將新版 zip＋ipynb 上傳 03_formal_resume，從上到下執行；保留 02_formal 原 ZIP 與 budget。來源相容宣告與所有身分驗證後才沿用舊 run；不符 STOP，勿刪帳本。相同版本中斷後重跑同一本 notebook：相同 source／契約／矩陣 identity 才可從完整 epoch 接續；已完成 fit 逐檔驗證後跳過。orphan 保留但不自動升格；85°C 硬體 STOP 需人工查明，不自動清除。所有結果屬研究 evidence，Colab 現場仍 UNVERIFIED。')
    code('''
        from google.colab import files
        download=Path('/content/p4a-results.zip')
        # 評估前打包全部 durable scores／metrics／checkpoints／ledger，保留相對路徑。
        with zipfile.ZipFile(download,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(BUDGET_DIR.rglob('*')):
                if path.is_file() and not path.is_symlink() and path.suffix in {'.json','.parquet','.pt','.jsonl'}:
                    z.write(path,path.relative_to(BUDGET_DIR))
        files.download(str(download))
    ''')
    for i,cell in enumerate(cells): cell['id']=f'p4a-{i:02d}'
    return dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),cells=cells)


def delivery(folds_root, years, p1, lock, output_dir, run_name='02_formal', drive_root='/content/drive/MyDrive/MarketMamba', resume_from_commit=None):
    """Publish the approved two-file interface; never overwrite shared outputs."""
    colab_notebook(run_name,drive_root)  # validate paths before publishing outputs
    output_dir=Path(output_dir)
    archive=output_dir/'p4a-bundle.zip';notebook=output_dir/'p4a-colab.ipynb'
    readme=output_dir/'README.txt'
    if archive.exists() or notebook.exists() or readme.exists(): raise FileExistsError('delivery outputs already exist')
    output_dir.mkdir(parents=True,exist_ok=True)
    mf=pack(folds_root,years,p1,lock,archive,resume_from_commit=resume_from_commit)
    with notebook.open('x') as stream:
        json.dump(colab_notebook(run_name,drive_root,digest(archive)),stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())
    with readme.open('x') as stream:
        stream.write('上傳到 Drive 資料夾：'+str(Path(drive_root)/'p4a'/run_name)+'\n開啟檔案：p4a-colab.ipynb\n')
        stream.flush();os.fsync(stream.fileno())
    return mf


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    pk=sub.add_parser('pack');pk.add_argument('--folds-root',type=Path,required=True);pk.add_argument('--years',type=int,nargs='+',default=[2021,2022,2023,2024,2025,2026]);pk.add_argument('--p1',type=Path,required=True);pk.add_argument('--lock',type=Path,default=Path.home()/'.venvs/marketmamba-gpu.requirements.lock.txt');pk.add_argument('--output',type=Path,required=True)
    dl=sub.add_parser('deliver')
    dl.add_argument('--folds-root',type=Path,required=True);dl.add_argument('--years',type=int,nargs='+',default=[2021,2022,2023,2024,2025,2026])
    dl.add_argument('--p1',type=Path,required=True);dl.add_argument('--lock',type=Path,default=ROOT/'environments/p4a-colab.requirements.lock.txt')
    dl.add_argument('--output-dir',type=Path,required=True)
    dl.add_argument('--run-name',default=None)
    dl.add_argument('--resume-from-commit')
    dl.add_argument('--drive-root',default='/content/drive/MyDrive/MarketMamba')
    ds=sub.add_parser('dataset-manifest');ds.add_argument('directory',type=Path)
    ex=sub.add_parser('extract');ex.add_argument('archive',type=Path);ex.add_argument('destination',type=Path);ex.add_argument('--sha256')
    ve=sub.add_parser('verify');ve.add_argument('directory',type=Path)
    ins=sub.add_parser('install-plan');ins.add_argument('directory',type=Path);ins.add_argument('--wheel-dir',type=Path)
    a=p.parse_args(argv)
    if a.cmd=='dataset-manifest':print(json.dumps(dataset_manifest(a.directory),indent=2))
    elif a.cmd=='pack': pack(a.folds_root,a.years,a.p1,a.lock,a.output);print(json.dumps(digest(a.output)))
    elif a.cmd=='deliver':
        delivery(a.folds_root,a.years,a.p1,a.lock,a.output_dir,a.run_name or ('03_formal_resume' if a.resume_from_commit else '02_formal'),a.drive_root,a.resume_from_commit)
        print(json.dumps({p.name:digest(p) for p in sorted(a.output_dir.iterdir()) if p.name in {'p4a-bundle.zip','p4a-colab.ipynb','README.txt'}}))
    elif a.cmd=='extract': extract(a.archive,a.destination,a.sha256)
    elif a.cmd=='verify': print(json.dumps(verify(a.directory),indent=2))
    else: print('\n'.join(installation_commands(a.directory,a.wheel_dir)))

if __name__=='__main__': main()
