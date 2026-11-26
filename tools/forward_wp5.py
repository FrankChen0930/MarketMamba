"""Frozen non-performance WP5 acceptance rehearsal; SYNTHETIC_CPU_ONLY.

UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED. No real input/forward/blind loader.
Switches below are private synthetic authorizations, never production pointers.
WP4 owns all model persistence and sanitized source publication.
"""
from __future__ import annotations

import argparse
import os
import secrets
from datetime import timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tools import forward_wp4 as store
from tools import forward_wp4_summary as summary
from tools import forward_wp2 as wp2
from tools import forward_wp3 as wp3
from tools.forward_wp1 import Calendar, Stop, cutoff, timestamp
from tools.forward_wp2 import SEEDS

REPO = Path(__file__).resolve().parents[1]
CHECKLIST = 'docs/operations/forward-retraining-checklist.md'
# Include transitive executable dependencies, not merely this wrapper's source.
FROZEN_FILES = tuple(sorted(set([
    'tools/forward_wp1.py', 'tools/forward_wp2.py', 'tools/forward_wp3.py',
    'tools/forward_wp4.py', 'tools/forward_wp4_summary.py', 'tools/forward_wp5.py',
    'tests/test_forward_wp5.py', 'tests/test_forward_wp4_summary.py',
    'tests/forward_tmp.py', CHECKLIST,
    'tasks/spec-forward-p2-primary-20261124.md',
] + [str(p.relative_to(REPO)) for p in (REPO/'tools').glob('step3*.py')]
  + [str(p.relative_to(REPO)) for p in (REPO/'V6/research_lab').glob('*.py')]
  + ['tools/pit_ledger.py', 'tools/tej_pit.py',
     'V6/experimental/v7_corrected_e5_matrix.py',
     'research/v7/corrected-baseline-v1/feature-manifest.json'])))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def freeze_script(path):
    """Create once; caller must retain this exact pin externally for each retry.

    This is a candidate SHA freeze, NOT a fabricated Human signature/approval.
    Package/native environment is separately recorded by WP4 before every fit.
    """
    path = store.private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pin = dict(schema='forward-wp5-frozen-source-v1', synthetic=True,
               files={name: store.file_info(REPO/name) for name in FROZEN_FILES})
    store.write_sync(path, store.canonical(pin)); store._sync_dir(path.parent)
    return pin


def verify_script(path, expected_pin):
    path = store.private_path(path)
    try:
        pin = json.loads(path.read_bytes())
        current = dict(schema='forward-wp5-frozen-source-v1', synthetic=True,
                       files={name: store.file_info(REPO/name) for name in FROZEN_FILES})
        if pin != expected_pin or pin != current: raise ValueError('source identity')
        return pin
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise Stop('WP5 frozen script mismatch') from exc


def calendar_identity(calendar):
    return sha256_bytes(store.canonical(dict(sessions=calendar.sessions,
        start=calendar.start, end=calendar.end, identity=calendar.identity)))


def freeze_request(calendar, origin, snapshot):
    if not isinstance(snapshot, bytes) or not snapshot: raise Stop('missing snapshot bytes')
    calendar.require_session(origin)
    return dict(schema='forward-wp5-synthetic-request-v1', synthetic=True,
        origin=origin, train_last=calendar.origin_minus21(origin),
        snapshot_cutoff=cutoff(calendar.shift(origin, -1)),
        snapshot_sha256=sha256_bytes(snapshot), calendar_sha256=calendar_identity(calendar),
        seeds=list(SEEDS), deadline=cutoff(calendar.shift(origin, 2)))


def validate_request(calendar, request):
    try:
        expected = freeze_request(calendar, request['origin'], b'validation-only')
        expected['snapshot_sha256'] = request['snapshot_sha256']
        if request != expected or not summary.HEX.fullmatch(request['snapshot_sha256']):
            raise ValueError('request identity')
    except (ValueError, KeyError, TypeError) as exc:
        raise Stop('WP5 request mismatch') from exc


def human_schedule(calendar, origins, t1):
    """Static private rows only. Calendar qualification is still an external gate."""
    if not origins or list(origins) != sorted(set(origins)):
        raise Stop('missing/unsorted origins')
    rows = []
    for origin in origins:
        calendar.require_session(origin)
        rows.append(dict(origin=origin, owner='Human', manual_check=calendar.shift(origin,-5),
            snapshot_cutoff=cutoff(calendar.shift(origin,-1)),
            train_last=calendar.origin_minus21(origin),
            attempt_cutoffs=[cutoff(calendar.shift(origin,i)) for i in range(3)]))
    return dict(calendar_sha256=calendar_identity(calendar), retraining=rows,
                t1_full_wp5_deadline=calendar.shift(t1,-20),
                t1_wp6_deadline=calendar.shift(t1,-5),
                t0_minimum='Initial acceptance and daily health/blind incident rehearsal required',
                t1_stable_chain_sessions=20, scheduler_created=False)


def synthetic_acceptance():
    """Test-only attestation shape; cannot confer real/forward admission.

    Production integration must verify the actual frozen reference reports;
    this module deliberately exposes only a synthetic rehearsal interface.
    """
    return dict(schema='forward-wp5-synthetic-acceptance-v1', synthetic=True,
                reference_equal=True, seeds=list(SEEDS), roundtrip_equal=True)


NO_EFFECT = frozenset(('SOURCE_OPTIONAL', 'PRIVATE_EXPECTED'))


def verdict(sealed_chain):
    """Pure sole incident authority over verified, ordered OPEN/CLOSE records.

    Integrity adapters validate hashes before constructing this value. No I/O,
    calendar metadata, caller summaries or mutable global registry is consulted.
    """
    states = {}; identities = {}; pending = {}; seen = set()
    def rejected(reason):
        return dict(status='REJECTED', reason=reason, states=states)
    if sealed_chain.get('integrity_error'): return rejected('BLIND_INTEGRITY')
    if sealed_chain.get('already_opened'): return rejected('ALREADY_OPENED')
    try:
        for record in sealed_chain['records']:
            ident = record['attempt_id']
            if record['kind'] == 'OPEN':
                if record.get('identity_error'): return rejected('RETRAIN_IDENTITY')
                if ident in seen: return rejected('BLIND_INTEGRITY')
                seen.add(ident); pending[ident] = record
                origin = record['origin']
                identity = (record['snapshot'], tuple(record['seeds']), record['request'])
                if origin in identities and identities[origin] != identity:
                    return rejected('RETRAIN_IDENTITY')
                identities[origin] = identity
                states.setdefault(origin, ('PENDING', 0))
            elif record['kind'] == 'CLOSE':
                item = pending.pop(ident, None)
                if item is None: return rejected('BLIND_INTEGRITY')
                origin = item['origin']; outcome = record['outcome']
                if outcome in NO_EFFECT: continue
                state, count = states.get(origin, ('PENDING', 0))
                if state == 'SWITCHED': return rejected('UNKNOWN_ERROR')
                if item['k'] != count: return rejected('RETRAIN_IDENTITY')
                clock = record.get('clock', item['clock'])
                if clock == 'EXPIRED': return rejected('DEADLINE')
                if outcome == 'SUCCESS' and clock == 'ONTIME':
                    states[origin] = ('SWITCHED', count)
                elif outcome == 'RETRAIN_TRANSIENT' and count < 2 and item['phase'] == 'FORMAL':
                    states[origin] = ('RETRYING', count + 1)
                else:
                    return rejected(outcome if outcome in summary.SOURCE_CODES + summary.PRIVATE_CODES
                                    else 'UNKNOWN_ERROR')
            else: return rejected('BLIND_INTEGRITY')
    except (KeyError, TypeError, ValueError): return rejected('BLIND_INTEGRITY')
    if pending: return rejected('UNKNOWN_ERROR')
    if any(state != 'SWITCHED' for state, _ in states.values()): return rejected('RETRY_PENDING')
    return dict(status='ACCEPTED', reason=None, states=states)


def _outcome(codes, clock, acceptance, phase):
    if not isinstance(codes,(list,tuple)) or any(c not in summary.SOURCE_CODES+summary.PRIVATE_CODES for c in codes):
        return 'UNKNOWN_ERROR'
    fatal = [c for c in codes if c not in NO_EFFECT and c != 'RETRAIN_TRANSIENT']
    if fatal: return fatal[0]
    if clock == 'EXPIRED': return 'UNKNOWN_ERROR'
    if 'RETRAIN_TRANSIENT' in codes or clock == 'LATE': return 'RETRAIN_TRANSIENT'
    if clock == 'EARLY': return 'PRIVATE_EXPECTED'
    return 'SUCCESS' if acceptance == synthetic_acceptance() else 'RETRAIN_IDENTITY'


def classify_clock(calendar, request, at):
    """Shared fail-closed session/clock classification for probe and journal.

    A civil date without a qualified session cannot be a retraining attempt,
    including at 18:00. The durable caller has already persisted its OPEN.
    """
    now = timestamp(at)
    day = calendar.require_session(now.date().isoformat())
    if now > timestamp(request['deadline']): return 'EXPIRED'
    if now < timestamp(cutoff(day)) or day < request['origin']: return 'EARLY'
    return 'ONTIME' if now == timestamp(cutoff(day)) else 'LATE'


def decide(calendar, request, *, snapshot, at, codes, phase, acceptance=None):
    """Clock adapter; verdict is the sole incident/state transition authority."""
    validate_request(calendar,request)
    if not isinstance(snapshot,bytes) or sha256_bytes(snapshot) != request['snapshot_sha256']:
        raise Stop('RETRAIN_IDENTITY: retry changed snapshot')
    if phase not in ('PRESTART','FORMAL'): raise Stop('unknown phase')
    day=timestamp(at).date().isoformat(); origin=request['origin']
    delay=sum(origin<d<=day for d in calendar.sessions)
    clock=classify_clock(calendar,request,at)
    outcome=_outcome(codes,clock,acceptance,phase)
    # The standalone clock probe has implicit earlier transient attempts. The
    # durable runner separately evaluates the actual sealed history.
    records=[]
    for k in range(min(delay,2)+1):
        opened=dict(kind='OPEN',attempt_id=str(k),origin=origin,k=k,
            snapshot=request['snapshot_sha256'],seeds=request['seeds'],request='clock-probe',
            clock=clock if k==min(delay,2) else 'ONTIME',phase=phase)
        records.extend([opened,dict(kind='CLOSE',attempt_id=str(k),
            outcome=outcome if k==min(delay,2) else 'RETRAIN_TRANSIENT')])
    result=verdict(dict(records=records))
    fatal='STOP' if phase=='PRESTART' else 'INVALID_FOR_CONFIRMATION'
    action=('SWITCH_ALL_THREE' if result['status']=='ACCEPTED' and outcome=='SUCCESS' else
            'WAIT' if clock=='EARLY' and outcome=='PRIVATE_EXPECTED' else
            'RETRY' if result['reason']=='RETRY_PENDING' else fatal)
    return dict(synthetic=True,forward_admitted=False,origin=origin,train_last=request['train_last'],
        snapshot_sha256=request['snapshot_sha256'],deadline=request['deadline'],delay_sessions=delay,
        action=action,switch_session=day if action=='SWITCH_ALL_THREE' else None,
        effective_open=calendar.shift(day,1) if action=='SWITCH_ALL_THREE' else None,
        next_attempt=calendar.shift(day,1) if action=='RETRY' else None)


def compare_stores(reference, candidate):
    """Verify BOTH WP4 pointers, all files and native roundtrip before byte compare.

    Models, sample keys, probes and predictions must all match. Telemetry and
    environment are retained but naturally not expected to have identical bytes.
    """
    context = store.synthetic_context()
    left = store.load_current(reference, context)
    right = store.load_current(candidate, context)
    identities = {}; predictions = samples = 0
    for seed in SEEDS:
        a, b = left[seed][1], right[seed][1]
        identity = {}
        for name in ('model.txt', 'sample.npy', 'probe.npy', 'prediction.npy'):
            aa, bb = (a/name).read_bytes(), (b/name).read_bytes()
            if aa != bb: raise Stop('RETRAIN_IDENTITY: independent reference bytes')
            identity[name] = dict(sha256=sha256_bytes(aa), bytes=len(aa))
        predictions += len(np.load(a/'prediction.npy', allow_pickle=False))
        samples += len(np.load(a/'sample.npy', allow_pickle=False))
        identities[str(seed)] = identity
    return dict(reference_equal=True, seeds=list(SEEDS), identities=identities,
                compared_predictions=predictions, compared_samples=samples)


def c0plus_reference_worker(root, request=None):
    """Independently rebuild synthetic 48/185 matrix, keys, three fits and ranks.

    Reuse WP3 full reference and WP2 origin/target/sample/fit/rank implementations.
    This fixture is intentionally separate from WP4's persistence probe fixture.
    No performance metric is computed, and no exposed P1 result is relabelled.
    """
    root=store.private_path(root); root.mkdir(parents=True,exist_ok=False)
    from datetime import date, timedelta
    import pandas as pd
    start = date.fromisoformat(request['origin'])-timedelta(days=100) if request else date(2023,1,1)
    days=[(start+timedelta(days=i)).isoformat() for i in range(130)]
    cal=Calendar(days,days[0],days[-1],{'synthetic':True})
    rng=np.random.default_rng(218)
    n=130*16; names=wp3.frozen_names()
    arrays=dict(dates=np.repeat(np.array(days,dtype='datetime64[D]'),16),
        stock_ids=np.tile(np.array([f'S{i:02}' for i in range(16)],dtype='<U32'),130),
        X=rng.normal(size=(n,48)).astype('<f4'), masks=np.ones((n,48),dtype='bool'),
        splits=np.ones(n,dtype='uint8'), eligibility=np.repeat(np.arange(130)>=59,16))
    context=wp3.make_context(names,days,[f'S{i:02}' for i in range(16)],store.sha256(store.SPEC))
    wp3.create_reference(root/'matrix-reference',arrays,context,b'SYNTHETIC_OPAQUE_SCORES',
                         sha256_bytes(b'SYNTHETIC_GENERATOR_218'))
    matrix=root/'matrix';matrix.mkdir()
    values=dict(X=arrays['X'],dates=arrays['dates'],stock_ids=arrays['stock_ids'],
        stock_idx=np.tile(np.arange(16,dtype='int32'),130),
        end_indices=np.arange(16,n+1,16),splits=arrays['splits'],
        y5=rng.normal(size=n).astype('float32'),y10=np.full(n,np.nan,dtype='float32'))
    for k,v in values.items(): wp3.save_array(matrix/f'{k}.npy',v)
    store.write_sync(matrix/'manifest.json',store.canonical(dict(rows=n,feature_order=names)))
    view=wp2.prepare_view(matrix,root/'view',cal,days[100],days[-1])
    frames=[]; artifacts=root/'compared';artifacts.mkdir()
    for seed in SEEDS:
        r=wp2.fit_predict(view,seed)
        for key in ('train_idx','fit_idx','eval_idx','target','scores','dates','fit_probe_scores'):
            wp3.save_array(artifacts/f'{seed}-{key}.npy',r[key])
        # Persist actual ordered business keys, not just row counts.
        wp3.save_array(artifacts/f'{seed}-codes.npy',arrays['stock_ids'][r['eval_idx']])
        store.write_sync(artifacts/f'{seed}-model.txt',r['model_text'].encode())
        frame=pd.DataFrame(dict(code=arrays['stock_ids'][r['eval_idx']],
            date=pd.to_datetime(r['dates']),score=r['scores'],fold=2023,seed=seed,system='c0plus'))
        ranks=frame.groupby('date').score.rank(pct=True).to_numpy()
        wp3.save_array(artifacts/f'{seed}-ranks.npy',ranks)
        frames.append(frame)
    avg=wp2.rank_average(pd.concat(frames,ignore_index=True))
    wp3.save_array(artifacts/'rank-average.npy',avg.score.to_numpy())
    store._sync_dir(artifacts)
    store.write_sync(root/'execution.json',store.canonical(dict(pid=os.getpid(),origin=days[100],request=request)))


def compare_c0plus(reference,candidate):
    """Exact full matrix semantics and every seed's ordered keys/scores/ranks."""
    reference,candidate=store.private_path(reference),store.private_path(candidate)
    try:
        a,am=wp3.load_snapshot(reference/'matrix-reference')
        b,bm=wp3.load_snapshot(candidate/'matrix-reference')
        wp3.verify_prefix(a,b)
        expected={f'{seed}-{key}.npy' for seed in SEEDS for key in
            ('train_idx','fit_idx','eval_idx','target','scores','dates','fit_probe_scores','codes','ranks')}
        expected|={f'{seed}-model.txt' for seed in SEEDS}|{'rank-average.npy'}
        identities={}
        for root in (reference,candidate):
            if {p.name for p in (root/'compared').iterdir()} != expected:
                raise Stop('incomplete C0+ reference')
        for name in sorted(expected):
            p,q=reference/'compared'/name,candidate/'compared'/name
            store.private_path(p);store.private_path(q)
            if p.read_bytes()!=q.read_bytes(): raise Stop('C0+ reference bytes mismatch')
            identities[name]=store.file_info(p)
        return dict(c0plus_equal=True,c0plus_features=185,c0plus_files=identities,
                    matrix_identity=am['identity'],candidate_matrix_identity=bm['identity'])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise Stop('incomplete/corrupt C0+ reference') from exc


def _accept_evidence(request, reference, candidate, reference_c0plus, candidate_c0plus):
    paths=[store.private_path(p) for p in (reference,candidate,reference_c0plus,candidate_c0plus)]
    if len(set(p.resolve() for p in paths)) != 4: raise Stop('independent stores required')
    receipts=[json.loads((p/'execution.json').read_bytes()) for p in paths[2:]]
    if (receipts[0]['pid']==receipts[1]['pid'] or any(
        r['request']!=request or r['origin']!=request['origin'] for r in receipts)):
        raise Stop('independent request/origin evidence required')
    executions=[json.loads((p/'execution-receipt.json').read_bytes()) for p in paths]
    if (executions[0]!=executions[2] or executions[1]!=executions[3]
            or executions[0]['nonce']==executions[1]['nonce']
            or executions[0]['pid']==executions[1]['pid']
            or executions[0]['started_ns']==executions[1]['started_ns']):
        raise Stop('independent execution receipts required')
    for receipt in executions:
        if (receipt['request']!=request or not isinstance(receipt['pid'],int)
                or receipt['pid']<=0 or not isinstance(receipt['started_ns'],int)
                or receipt['started_ns']<=0 or len(receipt['nonce'])!=64
                or any(c not in '0123456789abcdef' for c in receipt['nonce'])):
            raise Stop('invalid execution receipt')
    for p,receipt in zip(paths,executions):
        bound=json.loads((p/'output-binding.json').read_bytes())
        actual={str(f.relative_to(p)):store.file_info(f) for f in p.rglob('*')
                if f.is_file() and f.name!='output-binding.json'}
        if bound!=dict(request=request,receipt=receipt,files=actual): raise Stop('output binding mismatch')
    evidence=dict(compare_stores(reference,candidate),
        **compare_c0plus(reference_c0plus,candidate_c0plus))
    return evidence


def journal_records(journal):
    try: return _journal_records(journal)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Stop('corrupt/missing write-ahead journal') from exc


def _journal_records(journal):
    """Validate immutable write-ahead records; returns only state-machine input."""
    plan = json.loads((journal/'request.json').read_bytes())
    opens = journal/'opens'; events = journal/'events'
    records = []; previous = '0'*64
    open_paths = sorted(opens.iterdir())
    close_paths = sorted(events.iterdir())
    if len(close_paths) > len(open_paths): raise Stop('orphan CLOSE')
    for i, path in enumerate(open_paths):
        opened = json.loads(path.read_bytes())
        if path.name != f'{i:03}.json' or opened['previous_event_sha256'] != previous:
            raise Stop('corrupt OPEN chain')
        identity_error = (opened['request'] != sha256_bytes(store.canonical(plan))
            or opened['origin'] != plan['request']['origin']
            or opened['snapshot'] != plan['request']['snapshot_sha256']
            or opened['seeds'] != plan['request']['seeds'])
        records.append(dict(opened, identity_error=True) if identity_error else opened)
        if i >= len(close_paths): continue
        path = close_paths[i]; closed = json.loads(path.read_bytes())
        if (path.name != f'{i:03}.json' or closed['previous_event_sha256'] != previous
                or closed['request_sha256'] != opened['request']
                or closed['open_sha256'] != sha256_bytes(store.canonical(opened))
                or closed['attempt_id'] != opened['attempt_id'] or closed['kind'] != 'CLOSE'):
            raise Stop('corrupt CLOSE chain')
        records.append(closed); previous = sha256_bytes(store.canonical(closed))
    return records


def run_attempt(journal, *, calendar, request, snapshot, at, codes, phase,
                frozen_path, pin, reference=None, candidate=None,
                reference_c0plus=None, candidate_c0plus=None):
    """Write/fsync/register OPEN before validation; CLOSE is append-only.

    Exceptions, including caller-swallowed identity failures, leave an OPEN.
    Every action and admission is derived by verdict, also used by daily sealing
    and the full cross-session opening gate. No production pointer is changed.
    """
    journal = store.private_path(journal)
    with store.locked(journal):
        events = journal/'events'; opens = journal/'opens'; plan = journal/'request.json'
        wanted = dict(request=request, phase=phase, frozen_sha256=sha256_bytes(store.canonical(pin)))
        events.mkdir(exist_ok=True); opens.mkdir(exist_ok=True)
        if not plan.exists(): store.write_sync(plan, store.canonical(wanted))
        i = len(list(opens.iterdir()))
        previous = sha256_bytes((events/f'{i-1:03}.json').read_bytes()) if i and (events/f'{i-1:03}.json').exists() else '0'*64
        day = str(at)[:10]
        k = sum(request.get('origin','') < d <= day for d in calendar.sessions)
        opened = dict(schema='forward-wp5-open-v1', kind='OPEN', attempt_id=secrets.token_hex(32),
            origin=request.get('origin'), k=k, snapshot=sha256_bytes(snapshot) if isinstance(snapshot,bytes) else None,
            seeds=request.get('seeds'), request=sha256_bytes(store.canonical(wanted)),
            phase=phase, clock='UNVALIDATED', at=str(at), previous_event_sha256=previous)
        store.write_sync(opens/f'{i:03}.json', store.canonical(opened)); store._sync_dir(opens)
        summary.register_attempt(journal)
        # Everything that can reject input or compute evidence is after OPEN.
        verify_script(frozen_path,pin); validate_request(calendar,request)
        clock = classify_clock(calendar,request,at)
        if not isinstance(snapshot,bytes) or sha256_bytes(snapshot)!=request['snapshot_sha256']:
            raise Stop('RETRAIN_IDENTITY: retry changed snapshot')
        if json.loads(plan.read_bytes()) != wanted: raise Stop('RETRAIN_IDENTITY: journal request mismatch')
        history = journal_records(journal)
        prior = verdict(dict(records=history[:-1]))
        if prior['reason'] not in (None,'RETRY_PENDING'): raise Stop('terminal/corrupt journal')
        if i and prior['status']=='ACCEPTED': raise Stop('terminal journal; cannot restart')
        retries = prior['states'].get(request['origin'], ('PENDING', 0))[1]
        if k != retries and clock != 'EARLY': raise Stop('retry order: must attempt R, R+1, R+2')
        evidence = None; acceptance = None
        if set(codes) <= NO_EFFECT and clock == 'ONTIME':
            try:
                evidence = _accept_evidence(request,reference,candidate,reference_c0plus,candidate_c0plus)
                acceptance = synthetic_acceptance()
            except (Stop,OSError,ValueError,TypeError,KeyError): codes=['RETRAIN_IDENTITY']
        decision = decide(calendar,request,snapshot=snapshot,at=at,codes=codes,phase=phase,acceptance=acceptance)
        outcome = _outcome(codes, clock, acceptance, phase)
        event = dict(schema='forward-wp5-close-v1', synthetic=True, kind='CLOSE',
            attempt_id=opened['attempt_id'], open_sha256=sha256_bytes(store.canonical(opened)),
            request_sha256=opened['request'], previous_event_sha256=previous,
            at=timestamp(at).isoformat(), codes=list(codes), outcome=outcome, clock=clock,
            decision=decision,evidence=evidence)
        store.write_sync(events/f'{i:03}.json',store.canonical(event)); store._sync_dir(events)
        summary.register_attempt(journal)
        result=verdict(dict(records=journal_records(journal)))
        if result['status']=='REJECTED' and decision['action']=='SWITCH_ALL_THREE':
            raise Stop('state machine rejected switch')
        return event


def independent_reference(out, frozen_path, pin, request=None):
    """Two fresh CPU processes; fixed WP4 synthetic recipe, never real P1 replay."""
    out = store.private_path(out); out.mkdir(parents=True, exist_ok=False)
    script = ('from pathlib import Path; import sys,json; '
              'from tools.forward_wp5 import independent_reference_worker; '
              'independent_reference_worker(Path(sys.argv[1]),json.loads(sys.argv[2]))')
    elapsed = []
    for i in range(2):
        verify_script(frozen_path, pin)
        started = time.monotonic()
        with (out/f'process-{i}.log').open('xb') as log:
            completed = subprocess.run([sys.executable, '-c', script, str(out/f'run-{i}'), json.dumps(request)],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed.append(time.monotonic()-started)
        if completed.returncode != 0:
            raise Stop('independent reference process failed: ' + (out/f'process-{i}.log').read_text()[-1500:])
    for p in [out/f'run-{i}{suffix}' for i in range(2) for suffix in ('','-c0plus')]:
        files={str(f.relative_to(p)):store.file_info(f) for f in p.rglob('*') if f.is_file()}
        receipt=json.loads((p/'execution-receipt.json').read_bytes())
        store.write_sync(p/'output-binding.json',store.canonical(dict(request=request,receipt=receipt,files=files)))
    verify_script(frozen_path, pin)
    result = dict(compare_stores(out/'run-0', out/'run-1'),
                  **compare_c0plus(out/'run-0-c0plus',out/'run-1-c0plus'), processes=2,
                  seconds=elapsed, evidence_class='SYNTHETIC_CPU_ONLY', forward_admitted=False)
    store.write_sync(out/'reference-report.json', store.canonical(result))
    store._sync_dir(out)
    return result


def independent_reference_worker(root, request):
    """One fresh execution binds both outputs to its start identity."""
    receipt=dict(nonce=secrets.token_hex(32),pid=os.getpid(),
        started_ns=time.time_ns(),request=request)
    store.synthetic_run(root)
    c0root=root.parent/(root.name+'-c0plus')
    c0plus_reference_worker(c0root,request)
    for path in (root,c0root):
        store.write_sync(path/'execution-receipt.json',store.canonical(receipt))
        store._sync_dir(path)


def verify_chain(directories, expected_sessions, external_head):
    try:
        return summary.verify_synthetic_sessions(directories, expected_sessions, external_head)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Stop('BLIND_INTEGRITY: synthetic chain') from exc


def manual_unblind(*, calendar, expected_sessions, directories, external_head,
                   state, at, publication_complete, human, manual, frozen_path, pin, output):
    """One-shot synthetic audit record only; does not open or export any results.

    Real endpoint aggregation/portfolio semantics belong to WP6. Same-user file
    exclusivity is not OS isolation, authenticated Human identity or tamper proof.
    """
    output = store.private_path(output)
    verify_script(frozen_path, pin)
    if (state != 'VALID' or human != 'Human' or manual is not True
            or publication_complete is not True or len(expected_sessions) != 252):
        raise Stop('manual unblind prerequisites')
    first = calendar.require_session(expected_sessions[0])
    expected = [calendar.shift(first, i) for i in range(252)]
    if list(expected_sessions) != expected: raise Stop('unblind missing/shifted sessions')
    maturity = cutoff(calendar.shift(expected[-1], 11))
    if timestamp(at) < timestamp(maturity): raise Stop('unblind immature endpoint')
    verify_chain(directories, expected, external_head)
    # Merge immutable prefixes copied to later sessions by attempt identity;
    # conflicting duplicates are integrity errors, never a new retry history.
    records=[]; seen={}
    try:
        for directory in directories:
            chain=summary.private_operation_chain(store.private_path(directory)/'private')
            for record in chain['records']:
                key=(record['kind'],record['attempt_id'])
                if key in seen:
                    if seen[key]!=record: raise Stop('conflicting copied attempt')
                else:
                    seen[key]=record; records.append(record)
    except (Stop,OSError,ValueError,KeyError,TypeError) as exc:
        raise Stop('irreversible incident invalidity') from exc
    identity = sha256_bytes(store.canonical(dict(spec_sha256=store.sha256(store.SPEC),
        sessions=expected,external_head=external_head)))
    result=verdict(dict(records=records,already_opened=(store.ROOT/'unblind-registry'/(identity+'.json')).exists()))
    if result['status']!='ACCEPTED':
        if result['reason']=='ALREADY_OPENED': raise Stop('already unblinded; no second opening')
        raise Stop('irreversible incident invalidity: '+result['reason'])
    registry = store.private_path(store.ROOT/'unblind-registry')
    registry.mkdir(parents=True, exist_ok=True)
    result = dict(schema='forward-wp5-synthetic-unblind-v1', synthetic=True,
        forward_admitted=False, human=human, return_sessions=252, external_head=external_head,
        frozen_sha256=sha256_bytes(store.canonical(pin)),
        at_taipei=timestamp(at).isoformat(), at_utc=timestamp(at).astimezone(timezone.utc).isoformat(),
        maturity=maturity, integrity_checked_before_unblind=True)
    try:
        store.write_sync(registry/(identity+'.json'), store.canonical(result))
        store._sync_dir(registry)
        store.write_sync(output, store.canonical(result)); store._sync_dir(output.parent)
    except FileExistsError as exc: raise Stop('already unblinded; no second opening') from exc
    return result


def synthetic_rehearsal(out):
    """Manual, finite CPU CLI; no real data path, scheduling, waiting or network."""
    out = store.private_path(out); out.mkdir(parents=True, exist_ok=False)
    pin = freeze_script(out/'frozen.json')
    started = time.monotonic()
    sessions=[str(d) for d in np.arange(np.datetime64('2023-01-01'),
        np.datetime64('2023-01-01')+np.timedelta64(130,'D'))]
    calendar=Calendar(sessions,sessions[0],sessions[-1],{'synthetic':True})
    origin=sessions[100]
    chain_days=[calendar.shift(origin,i) for i in range(4)]
    # Source-only projection is frozen BEFORE any private model/reference read.
    sources=[summary.synthetic_source(d) for d in chain_days]
    store.write_sync(out/'sources-before-private.json',store.canonical(sources))
    snapshot=store.canonical(dict(recipe='SYNTHETIC_GENERATOR_218',context=store.synthetic_context()))
    request=freeze_request(calendar,origin,snapshot)
    reference = independent_reference(out/'reference', out/'frozen.json', pin, request)
    store.write_sync(out/'static-human-schedule.json',
        store.canonical(human_schedule(calendar,[origin],sessions[100])))
    # Preserve actual acceptance events as reviewable artifacts, not test counts.
    attempt_args=dict(calendar=calendar,request=request,snapshot=snapshot,phase='FORMAL',
        frozen_path=out/'frozen.json',pin=pin,reference=out/'reference/run-0',
        candidate=out/'reference/run-1',reference_c0plus=out/'reference/run-0-c0plus',
        candidate_c0plus=out/'reference/run-1-c0plus')
    decisions={}
    for delay in range(3):
        decisions[str(delay)]=run_attempt(out/'retry-success',
            at=cutoff(calendar.shift(origin,delay)),
            codes=['RETRAIN_TRANSIENT'] if delay<2 else [],**attempt_args)
        run_attempt(out/'retry-timeout',at=cutoff(calendar.shift(origin,delay)),
                    codes=['RETRAIN_TRANSIENT'],**attempt_args)
    directories=[];head='0'*64;private_head='0'*64
    for i,source in enumerate(sources):
        directory=out/'source-chain'/str(i)
        record=summary.run_synthetic_session(directory,source=source,
            private_operation=lambda _: None,previous_head=head,
            previous_private_commitment=private_head)
        directories.append(directory);head=record['head'];private_head=record['commitment']
    verify_chain(directories,chain_days,head)
    store.write_sync(out/'source-chain-audit.json',store.canonical(dict(
        expected_sessions=chain_days,external_head=head,verified=True,
        continues_after_retraining_stop=True,synthetic=True)))
    # Run behavioral/incident/mutation acceptance under the standard command.
    with (out/'acceptance-tests.log').open('xb') as log:
        proc = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_forward_wp5.py',
            'tests/test_forward_wp4_summary.py', '-q', '-p', 'no:cacheprovider'],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
    verify_script(out/'frozen.json', pin)
    report = dict(schema='forward-wp5-rehearsal-v1', evidence_class='SYNTHETIC_CPU_ONLY',
        forward_admitted=False, human_accepted=False, source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        frozen_sha256=sha256_bytes(store.canonical(pin)), reference=reference,
        delayed_switch=decisions['2']['decision'],source_chain_sessions=len(chain_days),
        tests_exit=proc.returncode, seconds=time.monotonic()-started,
        report_scope='synthetic engineering; real P1 reference and source gates remain STOP',
        tests_log=store.file_info(out/'acceptance-tests.log'))
    store.write_sync(out/'acceptance-report.json', store.canonical(report)); store._sync_dir(out)
    if proc.returncode != 0: raise Stop('WP5 acceptance tests failed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-out', type=Path, required=True)
    args = parser.parse_args()
    # All detailed output is private; CLI emits no scores, reports or status JSON.
    synthetic_rehearsal(args.synthetic_out)
