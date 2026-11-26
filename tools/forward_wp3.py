"""WP3 immutable feature-only reference/append. CPU engineering, NOT_DEPLOYED.

Consumes already prepared, causal 48-column slices. Does not fetch, transform raw
OHLCV, load labels, train or evaluate. Full-prefix C0+ rebuild is the deliberately
conservative reference implementation: resetting rolling cumsums at a 60-day tail
can change float32 bytes. Production/as-of admission remains a separate STOP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from tools.forward_wp1 import Stop
from V6.research_lab import c0plus

ROOT = Path(__file__).resolve().parents[1] / '.artifacts/forward'
REPO = Path(__file__).resolve().parents[1]
FIELDS = ('dates', 'stock_ids', 'X', 'masks', 'splits', 'eligibility')
DTYPES = dict(dates='<M8[D]', stock_ids='<U32', X='<f4', masks='|b1', splits='|u1', eligibility='|b1')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def producer_identity():
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                implementation_sha256=file_hash(Path(__file__)),
                environment=dict(python=sys.version, numpy=np.__version__, runtime='WSL_CPU',
                                 data_root=os.environ.get('MARKETMAMBA_DATA_ROOT', 'UNSET'),
                                 data_realpath=str((REPO/'Data').resolve())))


def array_record(a):
    a = np.asarray(a)
    if a.dtype.hasobject:
        raise Stop('object dtype forbidden')
    h = hashlib.sha256()
    for lo in range(0, len(a), 16384):
        h.update(np.ascontiguousarray(a[lo:lo+16384]).tobytes())
    return dict(shape=list(a.shape), dtype=a.dtype.str, bytes_sha256=h.hexdigest())


def batch_id(arrays):
    if set(arrays) != set(FIELDS):
        raise Stop('feature-only fields required; labels forbidden')
    return digest_bytes(canonical({k: array_record(arrays[k]) for k in FIELDS}))


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise Stop('missing/invalid identity')


def output_path(path):
    path = Path(path).absolute()
    # Refuse symlinks in any path component, including an existing output root.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise Stop('symlink output forbidden')
    if path == ROOT.resolve() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise Stop('output must be below checkout .artifacts/forward/')
    if path.exists():
        raise FileExistsError(path)
    return path


def frozen_names():
    return json.loads((REPO/'research/v7/corrected-baseline-v1/feature-manifest.json').read_text())['feature_order']


def make_context(names, sessions, roster, contract):
    require_sha(contract)
    if list(names) != frozen_names():
        raise Stop('frozen 48-column order mismatch')
    days = np.asarray(sessions, dtype='datetime64[D]')
    if (not len(days) or np.isnat(days).any() or np.any(days[1:] <= days[:-1]) or
            list(sessions) != [str(d) for d in days] or not roster or
            list(roster) != sorted(set(roster)) or any(not isinstance(s, str) or not s or len(s)>32 for s in roster)):
        raise Stop('invalid fixed session/stock axis')
    return dict(schema='forward-wp3-context-v1', feature_order=list(names),
                sessions=list(sessions), roster=list(roster), contract=contract,
                c0plus_sha256=file_hash(Path(c0plus.__file__)),
                matrix_builder_sha256=file_hash(REPO/'V6/experimental/v7_corrected_e5_matrix.py'),
                forward_admitted=False)


def validate_context(context):
    expected = make_context(context['feature_order'], context['sessions'], context['roster'], context['contract'])
    if expected != context:
        raise Stop('context/implementation identity mismatch')


def validate_batch(a, context):
    batch_id(a)  # Reject extra fields before touching their values.
    validate_context(context)
    n = len(a['dates'])
    if not n:
        raise Stop('empty feature batch')
    for k in FIELDS:
        shape = (n, 48) if k in ('X', 'masks') else (n,)
        if a[k].shape != shape or a[k].dtype.str != DTYPES[k]:
            raise Stop('shape/dtype mismatch: '+k)
    d, s, x, m = a['dates'], a['stock_ids'], a['X'], a['masks']
    if (np.isnat(d).any() or not np.isfinite(x).all() or
            np.any(x[~m] != 0) or not np.isin(a['splits'], [0, 1, 2, 3, 4]).all() or
            not np.isin(s, context['roster']).all()):
        raise Stop('invalid feature values/mask/roster/split')
    if np.any((d[1:] < d[:-1]) | ((d[1:] == d[:-1]) & (s[1:] <= s[:-1]))):
        raise Stop('duplicate or unsorted keys')
    expected = np.array([v for v in context['sessions'] if str(d[0]) <= v <= str(d[-1])], dtype='datetime64[D]')
    if not np.array_equal(np.unique(d), expected):
        raise Stop('missing or unknown session')


def context_features(a, context):
    """Feature-only proxy for C0+: intentionally never constructs SeqData/y."""
    validate_batch(a, context)
    days = np.unique(a['dates'])
    axis = [d for d in context['sessions'] if d <= str(days[-1])]
    mapping = {s: i for i, s in enumerate(context['roster'])}
    data = SimpleNamespace(X=a['X'], ends=np.zeros(len(axis), dtype='int64'),
                           n_stocks=len(mapping),
                           date_idx=np.searchsorted(np.array(axis, dtype='datetime64[D]'), a['dates']),
                           stock_idx=np.array([mapping[s] for s in a['stock_ids']], dtype='int32'))
    values, names = c0plus.build_features(data, np.arange(len(a['X'])), context['feature_order'])
    if values.shape != (len(a['X']), 185) or len(names) != 185 or np.isinf(values).any():
        raise Stop('invalid C0+ output')
    return values


def save_array(path, array):
    with path.open('xb') as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())


def save_bytes(path, value):
    with path.open('xb') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())


def seal(path, manifest):
    manifest['identity'] = digest_bytes(canonical(manifest))
    save_bytes(path/'manifest.json', canonical(manifest)+b'\n')
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return manifest


def publish(path, arrays, context, scores, source_identity, parent):
    require_sha(source_identity)
    path = output_path(path)
    path.mkdir(parents=True, exist_ok=False)
    for k, a in arrays.items():
        save_array(path/(k+'.npy'), a)
    save_bytes(path/'scores.bin', scores)
    files = {p.name: dict(size=p.stat().st_size, sha256=file_hash(p)) for p in path.iterdir()}
    return seal(path, dict(schema='forward-wp3-snapshot-v1', context=context,
                parent=parent, source_identity=source_identity, files=files,
                producer=producer_identity(), input_batch_identity=batch_id({k: arrays[k] for k in FIELDS}),
                arrays={k: array_record(a) for k, a in arrays.items()},
                evidence_class='CPU_ENGINEERING_ONLY', forward_admitted=False,
                score_scope='opaque pre-existing bytes; no new scores computed'))


def create_reference(out, arrays, context, scores, source_identity):
    output_path(out)
    validate_batch(arrays, context)
    if not isinstance(scores, bytes) or not scores:
        raise Stop('pre-existing score bytes required')
    return publish(out, dict(arrays, C185=context_features(arrays, context)),
                   context, scores, source_identity, None)


def load_snapshot(path):
    path = Path(path)
    try:
        m = json.loads((path/'manifest.json').read_bytes())
        identity = m.pop('identity')
        if digest_bytes(canonical(m)) != identity or m['schema'] != 'forward-wp3-snapshot-v1':
            raise Stop('manifest identity mismatch')
        m['identity'] = identity
        validate_context(m['context'])
        expected = {k+'.npy' for k in (*FIELDS, 'C185')} | {'scores.bin'}
        if set(m['files']) != expected or {p.name for p in path.iterdir()} != expected | {'manifest.json'}:
            raise Stop('incomplete/extra snapshot files')
        for name, record in m['files'].items():
            p = path/name
            if p.is_symlink() or p.stat().st_size != record['size'] or file_hash(p) != record['sha256']:
                raise Stop('snapshot file identity mismatch: '+name)
        arrays = {k: np.load(path/(k+'.npy'), allow_pickle=False, mmap_mode='r') for k in (*FIELDS, 'C185')}
        if {k: array_record(v) for k, v in arrays.items()} != m['arrays']:
            raise Stop('array identity mismatch')
        validate_batch({k: arrays[k] for k in FIELDS}, m['context'])
        if arrays['C185'].shape != (len(arrays['X']), 185) or arrays['C185'].dtype != np.dtype('float32'):
            raise Stop('invalid persisted C185')
        return arrays, m
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Stop('incomplete/corrupt snapshot') from exc


def snapshot_id(path):
    return load_snapshot(path)[1]['identity']


def verify_prefix(old, combined):
    n = len(old['X'])
    for k, a in old.items():
        if array_record(a) != array_record(combined[k][:n]):
            raise Stop('historical bytes changed: '+k)


def append_features(parent, out, delta, expected_parent, context, expected_batch, source_identity):
    output_path(out)
    old, m = load_snapshot(parent)
    if m['identity'] != expected_parent or m['context'] != context:
        raise Stop('parent/context identity mismatch')
    validate_batch(delta, context)
    if batch_id(delta) != expected_batch:
        raise Stop('delta input identity mismatch')
    if delta['dates'][0] <= old['dates'][-1]:
        raise Stop('duplicate/overlapping append date')
    combined = {k: np.concatenate([old[k], delta[k]]) for k in FIELDS}
    validate_batch(combined, context)
    combined['C185'] = context_features(combined, context)
    verify_prefix(old, combined)
    scores = (Path(parent)/'scores.bin').read_bytes()
    return publish(out, combined, context, scores, source_identity, m['identity'])


def write_labels(parent, out, labels, as_of, source_identity):
    """Independent label version; finite heads require exit/publication by cutoff.

    No label join mutates a feature snapshot or controls feature eligibility.
    Source identity is caller-supplied engineering provenance, not PIT admission.
    """
    out = output_path(out)
    require_sha(source_identity)
    arrays, m = load_snapshot(parent)
    n = len(arrays['dates'])
    if set(labels) != {'y5', 'y10', 'available_at', 'exit5', 'exit10'}:
        raise Stop('label schema mismatch')
    cutoff = np.datetime64(as_of, 'D')
    if np.isnat(cutoff):
        raise Stop('unknown label cutoff')
    for k, a in labels.items():
        expected = np.dtype('float32') if k in ('y5', 'y10') else np.dtype('datetime64[D]')
        if a.shape != (n,) or a.dtype != expected:
            raise Stop('label shape/dtype mismatch')
    for h in (5, 10):
        y, exit_ = labels[f'y{h}'], labels[f'exit{h}']
        available = labels['available_at']
        finite = np.isfinite(y)
        sessions = np.array(m['context']['sessions'], dtype='datetime64[D]')
        exit_indices = np.searchsorted(sessions, arrays['dates']) + h
        safe = exit_indices < len(sessions)
        minimum_exit = sessions[np.minimum(exit_indices, len(sessions)-1)]
        if (np.isinf(y).any() or np.any(finite & (np.isnat(exit_) | np.isnat(available) |
                ~safe | (exit_ > cutoff) | (available > cutoff) | (exit_ < minimum_exit) | (available < exit_)))):
            raise Stop('future/unknown/invalid label maturity')
    out.mkdir(parents=True, exist_ok=False)
    for k, a in labels.items():
        save_array(out/(k+'.npy'), a)
    return seal(out, dict(schema='forward-wp3-labels-v1', feature_identity=m['identity'],
                as_of=as_of, source_identity=source_identity, forward_admitted=False,
                producer=producer_identity(),
                files={p.name: file_hash(p) for p in out.iterdir()}))


def audit_existing(source, out):
    """One full read-only 48-column hash inventory, without labels/evaluation.

    Missing original masks cannot be inferred from finite(zero-filled X).
    Produces an immutable INCOMPLETE inventory and returns STOP in that case.
    """
    started = time.monotonic()
    source, out = Path(source).resolve(strict=True), output_path(out)
    manifest = json.loads((source/'manifest.json').read_bytes())
    if manifest['feature_order'] != frozen_names():
        raise Stop('reference feature order mismatch')
    records, missing, conflicts, schema_conflicts = {}, [], [], []
    for name in ('X.npy', 'dates.npy', 'stock_ids.npy', 'end_indices.npy', 'masks.npy'):
        p = source/name
        if not p.is_file():
            missing.append(name)
            continue
        before = p.stat()
        sha = file_hash(p)
        a = np.load(p, mmap_mode='r', allow_pickle=False)
        record = dict(path=str(p.resolve()), size=before.st_size, sha256=sha, **array_record(a))
        if name == 'end_indices.npy':
            n = records.get('X.npy', {}).get('shape', [-1])[0]
            record['session_ends_valid'] = bool(a.ndim == 1 and len(a) and
                a.dtype == np.dtype('int64') and a[-1] == n and a[0] > 0 and (np.diff(a) > 0).all())
        after = p.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise Stop('reference changed while hashing')
        expected = manifest.get('files', {}).get(name)
        record['manifest_expected_sha256'] = expected
        record['manifest_match'] = sha == expected
        if sha != expected:
            conflicts.append(name)
        records[name] = record
        print(f'hashed {name}: {record["shape"]} {record["dtype"]}', flush=True)
        del a
    n = records.get('X.npy', {}).get('shape', [-1])[0]
    for name, r in records.items():
        key = name[:-4]
        if key == 'end_indices':
            valid = r['session_ends_valid']
        else:
            shape = [n, 48] if key in ('X', 'masks') else [n]
            valid = r['shape'] == shape and r['dtype'] == DTYPES[key]
        if not valid:
            schema_conflicts.append(name)
    if n <= 0 or manifest.get('rows', n) != n:
        schema_conflicts.append('rows')
    status = 'COMPLETE' if not missing and not conflicts and not schema_conflicts else 'STOP_INCOMPLETE_REFERENCE'
    result = dict(schema='forward-wp3-existing-reference-v1', status=status,
        source_manifest=dict(path=str(source/'manifest.json'), sha256=file_hash(source/'manifest.json')),
        source_commit=os.environ.get('WP3_SOURCE_COMMIT', 'UNKNOWN'),
        contract_sha256=file_hash(REPO/'tasks/spec-forward-p2-primary-20261124.md'),
        implementation_sha256=file_hash(Path(__file__)),
        environment=dict(python=sys.version, numpy=np.__version__, data_root=os.environ.get('MARKETMAMBA_DATA_ROOT', 'UNSET'),
                         data_realpath=str((REPO/'Data').resolve()), runtime='WSL_CPU'),
        records=records, missing=missing, conflicts=conflicts, schema_conflicts=schema_conflicts,
        elapsed_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evidence_class='READONLY_REFERENCE_INVENTORY_ONLY', forward_admitted=False,
        labels_read=False, evaluations=0, fits=0,
        c185_real='NOT_BUILT: real-data scope limited to 48-column hash/resource inventory',
        scores_real='NOT_READ: no real scoring/evaluation in this task')
    out.mkdir(parents=True, exist_ok=False)
    seal(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = audit_existing(args.source, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'missing', 'conflicts', 'elapsed_seconds', 'peak_rss_kib')}))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())


def resolve_artifact(record, relocation, root=None):
    """Resolve immutable content via a separate portable SHA-to-relative-path map.

    Never edits historical manifests. Relocation paths are confined to the map's
    directory and must match the original content digest, including on reuse.
    """
    try:
        require_sha(record['sha256'])
        relocation = Path(relocation).resolve(strict=True)
        mapping = json.loads(relocation.read_bytes())['artifacts']
        base = Path(root).resolve(strict=True) if root is not None else relocation.parent
        relative = Path(mapping[record['sha256']])
        if relative.is_absolute() or '..' in relative.parts: raise Stop('relocation boundary')
        candidate = (base/relative).resolve(strict=True)
        if not candidate.is_relative_to(base): raise Stop('relocation symlink')
        if file_hash(candidate) != record['sha256']: raise Stop('relocation identity')
        return candidate
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise Stop('unresolved artifact identity') from exc
