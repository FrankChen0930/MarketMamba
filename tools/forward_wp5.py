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


def decide(calendar, request, *, snapshot, at, codes, phase, acceptance=None):
    """Pure SYNTHETIC clock decision; no fit, writes, pointer switch or timer.

    Invoke at R, R+1 and R+2 in that order. This is not an operational runner
    or a durable retry queue. Exact 18:00 is the synthetic switch instant;
    completion earlier must be held until then, and late completion never backdates.
    """
    validate_request(calendar, request)
    if not isinstance(snapshot, bytes) or sha256_bytes(snapshot) != request['snapshot_sha256']:
        raise Stop('RETRAIN_IDENTITY: retry changed snapshot')
    if phase not in ('PRESTART', 'FORMAL'): raise Stop('unknown phase')
    now = timestamp(at); deadline = timestamp(request['deadline'])
    origin = request['origin']; day = now.date().isoformat()
    calendar.covered(day)
    delay = sum(origin < d <= day for d in calendar.sessions)
    fatal = 'STOP' if phase == 'PRESTART' else 'INVALID_FOR_CONFIRMATION'
    result = dict(synthetic=True, forward_admitted=False, origin=origin,
        train_last=request['train_last'], snapshot_sha256=request['snapshot_sha256'],
        deadline=request['deadline'], delay_sessions=delay, action=fatal,
        switch_session=None, effective_open=None, next_attempt=None)
    # These checks precede acceptance: passing a report never extends the deadline.
    if now > deadline: return result
    allowed = set(summary.SOURCE_CODES + summary.PRIVATE_CODES)
    if not isinstance(codes, (list, tuple)) or any(c not in allowed for c in codes): return result
    if set(codes) - {'SOURCE_OPTIONAL', 'PRIVATE_EXPECTED', 'RETRAIN_TRANSIENT'}: return result
    if now < timestamp(cutoff(origin)):
        return dict(result, action='WAIT')
    if day not in calendar.sessions:
        return dict(result, action='WAIT', next_attempt=calendar.after(day))
    if now < timestamp(cutoff(day)): return dict(result, action='WAIT')
    late = now > timestamp(cutoff(day))
    if 'RETRAIN_TRANSIENT' in codes or late:
        # Initial M0/F0 cannot borrow an incumbent retry policy to start late.
        # T0/T1 must be deferred and freshly frozen by the separate start policy.
        if phase == 'PRESTART': return result
        if delay < 2:
            return dict(result, action='RETRY', next_attempt=calendar.shift(day, 1))
        return result
    if acceptance != synthetic_acceptance(): return result
    return dict(result, action='SWITCH_ALL_THREE', switch_session=day,
                effective_open=calendar.shift(day, 1))


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


def run_attempt(journal, *, calendar, request, snapshot, at, codes, phase,
                frozen_path, pin, reference=None, candidate=None,
                reference_c0plus=None, candidate_c0plus=None):
    """Private durable synthetic acceptance, same frozen script on EVERY attempt.

    Immutable request + ordered hash-linked events prevent accidental retry
    reset. WP4 owns flock/fsync and model pointer validation; no new model is
    published here. The report is a synthetic switch authorization only.
    Missing/corrupt evidence is fatal, never RETRAIN_TRANSIENT. The public source
    producer is intentionally independent and does not consume this report.
    """
    verify_script(frozen_path,pin)
    validate_request(calendar,request)
    # Always enforce snapshot identity, even for transient failures.
    preliminary=decide(calendar,request,snapshot=snapshot,at=at,codes=codes,phase=phase)
    journal=store.private_path(journal)
    with store.locked(journal):
        plan=journal/'request.json'; events=journal/'events'
        wanted=dict(request=request,phase=phase,frozen_sha256=sha256_bytes(store.canonical(pin)))
        if not plan.exists():
            if events.exists(): raise Stop('missing journal request; no reset')
            store.write_sync(plan,store.canonical(wanted));events.mkdir();store._sync_dir(journal)
        try:
            store.private_path(plan);store.private_path(events)
            if json.loads(plan.read_bytes()) != wanted: raise Stop('journal request/phase/source mismatch')
            previous='0'*64; history=[]
            for i,p in enumerate(sorted(events.iterdir())):
                store.private_path(p)
                event=json.loads(p.read_bytes())
                if (p.name!=f'{i:03}.json' or event['previous_event_sha256']!=previous
                        or event['request_sha256']!=sha256_bytes(store.canonical(wanted))
                        or event['decision']['origin']!=request['origin']
                        or event['decision']['snapshot_sha256']!=request['snapshot_sha256']
                        or event['decision']['delay_sessions']!=i
                        or event['decision']['action'] not in ('RETRY','SWITCH_ALL_THREE','STOP','INVALID_FOR_CONFIRMATION')):
                    raise Stop('corrupt retry journal')
                if history and history[-1]['decision']['action']!='RETRY':
                    raise Stop('event after terminal decision')
                previous=sha256_bytes(store.canonical(event));history.append(event)
            if history and history[-1]['decision']['action']!='RETRY':
                raise Stop('terminal journal; cannot restart or reset')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise Stop('corrupt/missing retry journal') from exc
        delay=preliminary['delay_sessions']
        if preliminary['action']=='WAIT':
            return dict(decision=preliminary,evidence=None,persisted=False)
        if delay!=len(history) and delay<=2: raise Stop('retry order: must attempt R, R+1, R+2')
        evidence=None; accepted=None
        # Never pay for validation after the deadline or when a failure is known.
        if not codes and timestamp(at)<=timestamp(request['deadline']):
            try:
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
                accepted=synthetic_acceptance()
            except (Stop,OSError,ValueError,TypeError,KeyError):
                codes=['RETRAIN_IDENTITY']
        decision=decide(calendar,request,snapshot=snapshot,at=at,codes=codes,
                        phase=phase,acceptance=accepted)
        verify_script(frozen_path,pin)
        event=dict(schema='forward-wp5-synthetic-attempt-v1',synthetic=True,
            request_sha256=sha256_bytes(store.canonical(wanted)),
            previous_event_sha256=previous,at=timestamp(at).isoformat(),
            codes=list(codes),decision=decision,evidence=evidence)
        store.write_sync(events/f'{len(history):03}.json',store.canonical(event))
        store._sync_dir(events);store._sync_dir(journal)
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


def incident_state(previous, incident):
    """Absorbing invalidity: changing owner, deleting a day or NONE cannot reset it."""
    return ('VALID' if previous == 'VALID' and incident == 'NONE'
            else 'INVALID_FOR_CONFIRMATION')


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
    for directory in directories:
        try:
            status=summary.private_operation_status(store.private_path(directory)/'private')
        except (Stop,OSError,ValueError,KeyError,TypeError) as exc:
            raise Stop('irreversible incident invalidity') from exc
        if status not in ('PRIVATE_EXPECTED','SWITCH_ALL_THREE'):
            raise Stop('irreversible incident invalidity')
    # Caller-controlled metadata cannot create a new unexposed research identity.
    identity = sha256_bytes(store.canonical(dict(spec_sha256=store.sha256(store.SPEC),
        sessions=expected,external_head=external_head)))
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
