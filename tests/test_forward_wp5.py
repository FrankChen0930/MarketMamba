"""Only synthetic WP5 fixtures; standard PYTHONPATH, no basetemp override."""
from datetime import date, timedelta
import types
from pathlib import Path

import pytest
from tests.forward_tmp import tmp_path  # noqa: F401
from tools import forward_wp5 as w
from tools import forward_wp4_summary as summary
from tools.forward_wp1 import Calendar, Stop, cutoff


def calendar():
    days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(450)]
    # Deliberately irregular synthetic calendar, never a trading calendar claim.
    days = [d for i, d in enumerate(days) if i % 7 not in (2, 5)]
    return Calendar(days, days[0], days[-1], {'synthetic': True})


def ticket():
    c = calendar()
    return c, w.freeze_request(c, c.sessions[30], b'fixed synthetic snapshot')


@pytest.mark.parametrize('hour', ['17:00:00', '18:00:00', '19:00:00'])
def test_non_session_clock_and_attempt_both_fail_closed(tmp_path, hour):
    c, r = ticket()
    from datetime import timedelta
    day = next((date.fromisoformat(r['origin']) + timedelta(days=i)).isoformat()
               for i in range(1, 7)
               if (date.fromisoformat(r['origin']) + timedelta(days=i)).isoformat() not in c.sessions)
    assert day not in c.sessions
    at = f'{day}T{hour}+08:00'
    args = dict(calendar=c, request=r, snapshot=b'fixed synthetic snapshot',
                at=at, codes=['RETRAIN_TRANSIENT'], phase='FORMAL')
    frozen = tmp_path/'frozen.json'; pin = w.freeze_script(frozen)
    for operation in (lambda: w.classify_clock(c, r, at),
                      lambda: w.decide(**args),
                      lambda: w.run_attempt(tmp_path/'journal', frozen_path=frozen, pin=pin, **args)):
        with pytest.raises(Stop, match='not a qualified session'):
            operation()
    records = w.journal_records(tmp_path/'journal')
    assert len(records) == 1 and records[0]['kind'] == 'OPEN'
    assert w.verdict(dict(records=records))['status'] == 'REJECTED'


def test_session_clock_classification():
    c, r = ticket()
    origin = r['origin']
    assert w.classify_clock(c, r, cutoff(origin)) == 'ONTIME'
    assert w.classify_clock(c, r, cutoff(origin).replace('18:00:00', '17:59:59')) == 'EARLY'
    assert w.classify_clock(c, r, cutoff(origin).replace('18:00:00', '18:00:01')) == 'LATE'
    assert w.classify_clock(c, r, cutoff(c.shift(origin, 3))) == 'EXPIRED'


def reference_verdict(records, opened=False, continuation=False):
    """Independent specification oracle; no product functions or constants."""
    harmless = {'SOURCE_OPTIONAL', 'PRIVATE_EXPECTED'}
    states, pending, identities = {}, {}, {}
    if opened: return False
    for record in records:
        if record['kind'] == 'OPEN':
            if record.get('identity_error'): return False
            key = record['origin']
            if record['attempt_id'] in pending: return False
            identity = (record['snapshot'], tuple(record['seeds']), record['request'])
            if key in identities and identities[key] != identity: return False
            identities[key] = identity
            states.setdefault(key, ('PENDING', 0))
            pending[record['attempt_id']] = record
        elif record['kind'] == 'CLOSE':
            item = pending.pop(record['attempt_id'], None)
            if item is None: return False
            key, outcome = item['origin'], record['outcome']
            state, count = states.get(key, ('PENDING', 0))
            if outcome in harmless: continue
            if state == 'SWITCHED' or item['k'] != count: return False
            clock = record.get('clock',item['clock'])
            if clock == 'EXPIRED': return False
            if outcome == 'SUCCESS' and clock == 'ONTIME':
                states[key] = ('SWITCHED', count)
            elif outcome == 'RETRAIN_TRANSIENT' and count < 2 and item['phase'] == 'FORMAL':
                states[key] = ('RETRYING', count + 1)
            else: return False
        else: return False
    if continuation:
        return (tuple(sorted(states.items())), tuple(sorted(pending)),
                tuple(sorted((key,str(value)) for key,value in identities.items())))
    return not pending and all(value[0] == 'SWITCHED' for value in states.values())


def abstract_attempt(index, outcome, origin='R0', **changes):
    item = dict(kind='OPEN', attempt_id=f'{origin}-{index}', origin=origin,
                k=index, snapshot='snapshot', seeds=[17,29,43], request=origin+'-request',
                clock='ONTIME', phase='FORMAL')
    item.update(changes)
    return [item] if outcome == 'UNCLOSED' else [item, dict(
        kind='CLOSE', attempt_id=item['attempt_id'], outcome=outcome)]


def test_state_machine_exhaustive_independent_reference():
    from itertools import product
    alphabet = ['SOURCE_OPTIONAL','PRIVATE_EXPECTED','RETRAIN_TRANSIENT','SUCCESS',
        'SOURCE_MISSING','SOURCE_COVERAGE','SOURCE_UNIVERSE','SOURCE_INVALID',
        'RETRAIN_IDENTITY','PRIVATE_MARK_EXECUTION','BLIND_INTEGRITY','UNKNOWN_ERROR',
        'BOGUS','UNCLOSED','EXPIRED','ORIGIN','SNAPSHOT','SEEDS']
    # All words of length <=4 per R_j. Pair every local state class across two
    # origins in both interleavings; local exhaustive words already cover order.
    representatives = {}
    for size in range(5):
        for word in product(alphabet, repeat=size):
            records = []
            retry_count = 0
            for i, token in enumerate(word):
                changes = dict(attempt_id=f'R0-{i}')
                outcome = token
                if token in ('EXPIRED','ORIGIN','SNAPSHOT','SEEDS'):
                    outcome = 'SUCCESS'
                    changes.update({'EXPIRED':dict(clock='EXPIRED'), 'ORIGIN':dict(origin='wrong',identity_error=True),
                        'SNAPSHOT':dict(snapshot='wrong',identity_error=True), 'SEEDS':dict(seeds=[1],identity_error=True)}[token])
                records += abstract_attempt(retry_count, outcome, **changes)
                retry_count += token == 'RETRAIN_TRANSIENT'
            expected = reference_verdict(records)
            assert (w.verdict(dict(records=records))['status'] == 'ACCEPTED') == expected, word
            # Equivalent continuation states collapse exponentially many words.
            signature = reference_verdict(records,continuation=True)
            representatives.setdefault(str(signature), records)
    values = list(representatives.values())
    def attempts(records):
        groups=[]
        for item in records:
            if item['kind']=='OPEN': groups.append([item])
            else: groups[-1].append(item)
        return groups
    for left, right in product(values, repeat=2):
        right = [dict(x, **({'origin':'R1','request':'R1-request'} if x['kind']=='OPEN' else {}),
                      attempt_id='other-'+x['attempt_id']) for x in right]
        # All order-preserving cross-session interleavings of both R_j traces.
        def interleave(a,b):
            if not a or not b:
                yield a+b; return
            for tail in interleave(a[1:],b): yield [a[0]]+tail
            for tail in interleave(a,b[1:]): yield [b[0]]+tail
        for groups in interleave(attempts(left),attempts(right)):
            records = [item for group in groups for item in group]
            assert (w.verdict(dict(records=records))['status']=='ACCEPTED') == reference_verdict(records)


def test_r5_swallowed_identity_is_write_ahead_recorded(tmp_path):
    c,r=ticket(); frozen=tmp_path/'frozen.json'; pin=w.freeze_script(frozen)
    def operation(private):
        try:
            w.run_attempt(private/'journal',calendar=c,request=r,snapshot=b'wrong',
                at=cutoff(r['origin']),codes=[],phase='FORMAL',frozen_path=frozen,pin=pin)
        except Stop: pass
    summary.run_synthetic_session(tmp_path/'sealed',private_operation=operation)
    assert summary.private_operation_status(tmp_path/'sealed/private') != 'PRIVATE_EXPECTED'


def test_wait_does_not_consume_same_origin_retry_slot(tmp_path):
    c,r=ticket(); frozen=tmp_path/'frozen.json'; pin=w.freeze_script(frozen)
    args=dict(calendar=c,request=r,snapshot=b'fixed synthetic snapshot',phase='FORMAL',
              frozen_path=frozen,pin=pin)
    first=w.run_attempt(tmp_path/'journal',at=cutoff(r['origin']).replace('18:00:00','17:59:59'),
                        codes=['PRIVATE_EXPECTED'],**args)
    assert first['decision']['action']=='WAIT'
    second=w.run_attempt(tmp_path/'journal',at=cutoff(r['origin']),codes=['RETRAIN_TRANSIENT'],**args)
    assert second['decision']['action']=='RETRY'


def transient(c, request, delay=0):
    return w.decide(c, request, snapshot=b'fixed synthetic snapshot',
                    at=cutoff(c.shift(request['origin'], delay)),
                    codes=['RETRAIN_TRANSIENT'], phase='FORMAL')


@pytest.mark.parametrize('delay', [0, 1, 2])
def test_same_frozen_snapshot_and_origin_for_retries(delay):
    c, r = ticket()
    result = transient(c, r, delay)
    assert result['action'] == ('RETRY' if delay < 2 else 'INVALID_FOR_CONFIRMATION')
    assert result['origin'] == r['origin']
    assert result['snapshot_sha256'] == r['snapshot_sha256']
    assert result['delay_sessions'] == delay
    assert result['next_attempt'] == (c.shift(r['origin'], delay+1) if delay < 2 else None)


@pytest.mark.parametrize('delay', [0, 1, 2])
def test_1800_switch_authorization_requires_acceptance(delay):
    c, r = ticket()
    out = w.decide(c, r, snapshot=b'fixed synthetic snapshot',
                   at=cutoff(c.shift(r['origin'], delay)), codes=[], phase='FORMAL',
                   acceptance=w.synthetic_acceptance())
    assert out['action'] == 'SWITCH_ALL_THREE'
    assert out['switch_session'] == c.shift(r['origin'], delay)
    assert out['effective_open'] == c.shift(r['origin'], delay+1)
    assert out['train_last'] == c.origin_minus21(r['origin'])


@pytest.mark.parametrize('when', ['early', 'late', 'timeout', 'before_origin'])
def test_clock_boundary_does_not_backdate(when):
    c, r = ticket()
    d = c.shift(r['origin'], {'early': 0, 'late': 0, 'timeout': 2, 'before_origin': -1}[when])
    at = cutoff(d)
    if when == 'early': at = at.replace('18:00:00', '17:59:59')
    if when in ('late', 'timeout'): at = at.replace('18:00:00', '18:00:01')
    result = w.decide(c, r, snapshot=b'fixed synthetic snapshot', at=at,
                      codes=[], phase='FORMAL', acceptance=w.synthetic_acceptance())
    assert result['action'] == ('WAIT' if when in ('early', 'before_origin') else
                                'RETRY' if when == 'late' else 'INVALID_FOR_CONFIRMATION')
    assert result['switch_session'] is None


@pytest.mark.parametrize('phase', ['PRESTART', 'FORMAL'])
@pytest.mark.parametrize('code', list(summary.SOURCE_CODES) + list(summary.PRIVATE_CODES) + ['BOGUS'])
def test_error_routing_fails_closed(phase, code):
    c, r = ticket()
    result = w.decide(c, r, snapshot=b'fixed synthetic snapshot', at=cutoff(r['origin']),
                      codes=[code, 'RETRAIN_TRANSIENT'], phase=phase)
    if code in ('RETRAIN_TRANSIENT', 'SOURCE_OPTIONAL', 'PRIVATE_EXPECTED'):
        assert result['action'] == ('RETRY' if phase=='FORMAL' else 'STOP')
    else:
        assert result['action'] == ('STOP' if phase == 'PRESTART' else 'INVALID_FOR_CONFIRMATION')


def test_changed_snapshot_seed_origin_or_unknown_acceptance_stops():
    c, r = ticket()
    with pytest.raises(Stop):
        w.decide(c, r, snapshot=b'late revised input', at=cutoff(r['origin']), codes=[], phase='FORMAL')
    for key, value in [('seeds', [17, 29]), ('train_last', r['origin']), ('snapshot_cutoff', cutoff(r['origin']))]:
        bad = dict(r, **{key: value})
        with pytest.raises(Stop): w.validate_request(c, bad)
    for evidence in [None, {}, dict(w.synthetic_acceptance(), reference_equal=False)]:
        result = w.decide(c, r, snapshot=b'fixed synthetic snapshot', at=cutoff(r['origin']),
                          codes=[], phase='FORMAL', acceptance=evidence)
        assert result['action'] == 'INVALID_FOR_CONFIRMATION'


def test_prestart_failure_does_not_start_late_with_incumbent_retry():
    c,r=ticket()
    for delay in range(3):
        result=w.decide(c,r,snapshot=b'fixed synthetic snapshot',
            at=cutoff(c.shift(r['origin'],delay)),codes=['RETRAIN_TRANSIENT'],phase='PRESTART')
        assert result['action']=='STOP'
        assert result['next_attempt'] is None


def test_freeze_covers_code_tests_checklist_and_spec(tmp_path):
    pin = w.freeze_script(tmp_path/'frozen.json')
    assert w.verify_script(tmp_path/'frozen.json', pin) == pin
    assert {'tools/forward_wp5.py', 'tools/forward_wp1.py', 'tools/forward_wp4.py',
            'tools/forward_wp4_summary.py', 'tests/test_forward_wp5.py',
            'docs/operations/forward-retraining-checklist.md'} <= set(pin['files'])
    with pytest.raises(Stop): w.verify_script(tmp_path/'frozen.json', dict(pin, schema='changed'))
    with pytest.raises(FileExistsError): w.freeze_script(tmp_path/'frozen.json')


def test_two_mutations_killed():
    from _pytest.outcomes import Failed
    source = Path(w.__file__).read_text()
    cases = [('snapshot', "sha256_bytes(snapshot) != request['snapshot_sha256']", 'False'),
             ('deadline', "now > timestamp(request['deadline'])", 'False')]
    c, r = ticket()
    for name, old, new in cases:
        assert source.count(old) == 1
        mutant = types.ModuleType('wp5_mutant'); mutant.__file__ = w.__file__
        exec(compile(source.replace(old, new), w.__file__, 'exec'), mutant.__dict__)
        def oracle(module):
            if name == 'snapshot':
                with pytest.raises(Stop):
                    module.decide(c, r, snapshot=b'changed', at=cutoff(r['origin']), codes=[], phase='FORMAL')
            else:
                out = module.decide(c, r, snapshot=b'fixed synthetic snapshot',
                        at=cutoff(c.shift(r['origin'], 3)),
                    codes=[], phase='FORMAL', acceptance=w.synthetic_acceptance())
                assert out['action'] == 'INVALID_FOR_CONFIRMATION'
        oracle(w)
        with pytest.raises((Failed, AssertionError)): oracle(mutant)


def test_reference_two_independent_processes_and_corruption(tmp_path):
    pin = w.freeze_script(tmp_path/'frozen.json')
    c,r=ticket()
    result = w.independent_reference(tmp_path/'reference', tmp_path/'frozen.json', pin, request=r)
    assert result['reference_equal'] is True
    assert result['processes'] == 2
    assert result['seeds'] == [17, 29, 43]
    assert result['compared_predictions'] == 111
    assert result['compared_samples'] == 2400
    assert result['c0plus_equal'] is True
    assert result['c0plus_features'] == 185
    c,r=ticket()
    for delay in range(3):
        journal=tmp_path/f'acceptance-delay-{delay}'
        for previous in range(delay):
            w.run_attempt(journal,calendar=c,request=r,snapshot=b'fixed synthetic snapshot',
                at=cutoff(c.shift(r['origin'],previous)),codes=['RETRAIN_TRANSIENT'],phase='FORMAL',
                frozen_path=tmp_path/'frozen.json',pin=pin)
        accepted=w.run_attempt(journal,calendar=c,request=r,snapshot=b'fixed synthetic snapshot',
            at=cutoff(c.shift(r['origin'],delay)),codes=[],phase='FORMAL',
            frozen_path=tmp_path/'frozen.json',pin=pin,
            reference=tmp_path/'reference/run-0',candidate=tmp_path/'reference/run-1',
            reference_c0plus=tmp_path/'reference/run-0-c0plus',candidate_c0plus=tmp_path/'reference/run-1-c0plus')
        assert accepted['decision']['action']=='SWITCH_ALL_THREE'
        assert accepted['decision']['origin']==r['origin']
        assert accepted['evidence']['reference_equal'] is True
    import shutil
    clone=tmp_path/'cloned-store'
    shutil.copytree(tmp_path/'reference/run-0',clone)
    cloned=w.run_attempt(tmp_path/'cloned-journal',calendar=c,request=r,
        snapshot=b'fixed synthetic snapshot',at=cutoff(r['origin']),codes=[],phase='FORMAL',
        frozen_path=tmp_path/'frozen.json',pin=pin,reference=tmp_path/'reference/run-0',
        candidate=clone,reference_c0plus=tmp_path/'reference/run-0-c0plus',
        candidate_c0plus=tmp_path/'reference/run-1-c0plus')
    assert cloned['decision']['action']=='INVALID_FOR_CONFIRMATION'
    args=dict(calendar=c,request=r,snapshot=b'fixed synthetic snapshot',
        at=cutoff(r['origin']),codes=[],phase='FORMAL',frozen_path=tmp_path/'frozen.json',pin=pin,
        reference=tmp_path/'reference/run-0',candidate=tmp_path/'reference/run-0',
        reference_c0plus=tmp_path/'reference/run-0-c0plus',candidate_c0plus=tmp_path/'reference/run-0-c0plus')
    assert w.run_attempt(tmp_path/'same-store',**args)['decision']['action']=='INVALID_FOR_CONFIRMATION'
    other=w.freeze_request(c,c.shift(r['origin'],1),b'fixed synthetic snapshot')
    args.update(request=other,at=cutoff(other['origin']),candidate=tmp_path/'reference/run-1',
        candidate_c0plus=tmp_path/'reference/run-1-c0plus')
    assert w.run_attempt(tmp_path/'wrong-origin',**args)['decision']['action']=='INVALID_FOR_CONFIRMATION'
    assert w.compare_stores(tmp_path/'reference/run-0', tmp_path/'reference/run-1')['reference_equal']
    pointer = tmp_path/'reference/run-1/current.json'
    pointer.write_text('{}')
    with pytest.raises(Stop): w.compare_stores(tmp_path/'reference/run-0', tmp_path/'reference/run-1')


def test_retry_journal_is_ordered_immutable_and_timeout_absorbing(tmp_path):
    c,r=ticket(); frozen=tmp_path/'frozen.json';pin=w.freeze_script(frozen)
    args=dict(calendar=c,request=r,snapshot=b'fixed synthetic snapshot',phase='FORMAL',
              frozen_path=frozen,pin=pin,codes=['RETRAIN_TRANSIENT'])
    events=[]
    for delay in range(3):
        events.append(w.run_attempt(tmp_path/'journal',at=cutoff(c.shift(r['origin'],delay)),**args))
    assert [e['decision']['action'] for e in events]==['RETRY','RETRY','INVALID_FOR_CONFIRMATION']
    assert events[1]['previous_event_sha256']==w.sha256_bytes(w.store.canonical(events[0]))
    with pytest.raises(Stop):
        w.run_attempt(tmp_path/'journal',at=cutoff(c.shift(r['origin'],3)),**args)
    with pytest.raises(Stop):
        w.run_attempt(tmp_path/'skip',at=cutoff(c.shift(r['origin'],1)),**args)
    assert len(list((tmp_path/'journal/events').glob('*.json')))==3


def test_corrupt_journal_cannot_reset_or_change_snapshot(tmp_path):
    c,r=ticket(); frozen=tmp_path/'frozen.json';pin=w.freeze_script(frozen)
    args=dict(calendar=c,request=r,snapshot=b'fixed synthetic snapshot',phase='FORMAL',
              frozen_path=frozen,pin=pin,codes=['RETRAIN_TRANSIENT'])
    w.run_attempt(tmp_path/'journal',at=cutoff(r['origin']),**args)
    (tmp_path/'journal/events/000.json').write_text('{}')
    with pytest.raises(Stop):
        w.run_attempt(tmp_path/'journal',at=cutoff(c.shift(r['origin'],1)),**args)


def test_static_schedule_uses_sessions_and_keeps_origin():
    c=calendar(); origin=c.sessions[40]; t1=c.sessions[60]
    schedule=w.human_schedule(c,[origin],t1)
    row=schedule['retraining'][0]
    assert row['manual_check']==c.shift(origin,-5)
    assert row['snapshot_cutoff']==cutoff(c.shift(origin,-1))
    assert row['train_last']==c.shift(origin,-21)
    assert row['attempt_cutoffs']==[cutoff(c.shift(origin,i)) for i in range(3)]
    assert schedule['t1_full_wp5_deadline']==c.shift(t1,-20)


def test_cli_source_projection_is_frozen_before_any_private_reference(tmp_path,monkeypatch):
    sources=[]
    original=summary.synthetic_source
    def source(session):
        value=original(session);sources.append(value);return value
    def stop_before_private(*args):
        assert len(sources)==4
        raise Stop('synthetic pre-fit boundary probe')
    monkeypatch.setattr(summary,'synthetic_source',source)
    monkeypatch.setattr(w,'independent_reference',stop_before_private)
    with pytest.raises(Stop,match='pre-fit boundary probe'):
        w.synthetic_rehearsal(tmp_path/'boundary')


def test_missing_reference_model_or_rank_is_not_accepted(tmp_path):
    # No missing report may become a vacuous equality pass.
    with pytest.raises(Stop): w.compare_c0plus(tmp_path/'absent',tmp_path/'also-absent')


def chain(root, count=3):
    c = calendar(); days = c.sessions[31:31+count]
    dirs=[]; head='0'*64; private_head='0'*64
    for i, d in enumerate(days):
        directory=root/str(i)
        p=summary.run_synthetic_session(directory, source=summary.synthetic_source(d),
            private_operation=lambda _: None, previous_head=head, previous_private_commitment=private_head)
        dirs.append(directory); head=p['head']; private_head=p['commitment']
    return c, days, dirs, head


def test_timeout_keeps_source_chain_and_private_failure_does_not_leak(tmp_path):
    c, r=ticket()
    assert transient(c,r,2)['action']=='INVALID_FOR_CONFIRMATION'
    c, days, dirs, head=chain(tmp_path/'chain')
    assert summary.verify_synthetic_sessions(dirs, days, head)
    for directory in dirs:
        import json
        public=json.loads((directory/'public-summary.json').read_text())
        assert set(public)==summary.SOURCE_KEYS | summary.CHAIN_KEYS
        assert summary.release_bytes(public,summary.release_at(public['session']))
    with pytest.raises(Stop): summary.verify_synthetic_sessions(dirs[1:], days, head)


@pytest.mark.parametrize('incident', ['PEEK', 'BAD_OUTPUT', 'MISSING_DAY', 'UNKNOWN'])
def test_incidents_absorbing_and_not_reset_by_owner(incident):
    records=abstract_attempt(0,incident)
    assert w.verdict(dict(records=records))['status']=='REJECTED'
    for later in ('PRIVATE_EXPECTED','CHANGE_OWNER','SUCCESS'):
        assert w.verdict(dict(records=records+abstract_attempt(1,later)))['status']=='REJECTED'


def test_manual_unblind_requires_full_chain_maturity_human_once(tmp_path):
    c, days, dirs, head=chain(tmp_path/'chain',252)
    pin=w.freeze_script(tmp_path/'frozen.json')
    args=dict(calendar=c, expected_sessions=days, directories=dirs, external_head=head,
              state='VALID', at=cutoff(c.shift(days[-1],11)),
              publication_complete=True, human='Human', manual=True,
              frozen_path=tmp_path/'frozen.json', pin=pin, output=tmp_path/'unblind.json')
    for changes in [dict(manual=False), dict(human='worker'), dict(state='INVALID_FOR_CONFIRMATION'),
                    dict(publication_complete=False), dict(at=cutoff(c.shift(days[-1],10))),
                    dict(directories=dirs[1:]), dict(external_head='1'*64), dict(expected_sessions=days[:-1])]:
        with pytest.raises(Stop): w.manual_unblind(**dict(args,**changes))
    assert not (tmp_path/'unblind.json').exists()
    report=w.manual_unblind(**args)
    assert report['synthetic'] is True
    assert report['return_sessions']==252
    with pytest.raises(Stop): w.manual_unblind(**args)


@pytest.mark.parametrize('kind', ['public', 'private', 'nonce'])
def test_blind_bad_output_rejected(tmp_path,kind):
    c, days, dirs, head=chain(tmp_path/'chain')
    file=dirs[0]/{'public':'public-summary.json', 'private':'private/source-before-fit.json',
                 'nonce':'private/nonce.bin'}[kind]
    file.write_bytes(b'corrupt')
    with pytest.raises(Stop): w.verify_chain(dirs,days,head)


def test_unblind_identity_cannot_change_output(tmp_path):
    c, days, dirs, head=chain(tmp_path/'chain',252)
    pin=w.freeze_script(tmp_path/'frozen.json')
    args=dict(calendar=c, expected_sessions=days, directories=dirs, external_head=head,
        state='VALID', at=cutoff(c.shift(days[-1],11)), publication_complete=True,
        human='Human', manual=True, frozen_path=tmp_path/'frozen.json',pin=pin)
    w.manual_unblind(**args,output=tmp_path/'one.json')
    with pytest.raises(Stop): w.manual_unblind(**args,output=tmp_path/'two.json')
    changed=Calendar(c.sessions,c.start,c.end,dict(c.identity,comment='same research'))
    with pytest.raises(Stop,match='already unblinded'):
        w.manual_unblind(**dict(args,calendar=changed),output=tmp_path/'metadata.json')


@pytest.mark.parametrize('codes,phase,action', [
    (['UNKNOWN_ERROR'],'FORMAL','INVALID_FOR_CONFIRMATION'),
    (['RETRAIN_TRANSIENT'],'FORMAL','RETRY'),
    (['UNKNOWN_ERROR'],'PRESTART','STOP'),
])
@pytest.mark.parametrize('discard_return', [False,True])
def test_returned_journal_incident_cannot_be_relabelled(tmp_path,monkeypatch,codes,phase,action,discard_return):
    import json
    c=calendar(); days=c.sessions[31:283]; dirs=[]; head='0'*64; previous='0'*64
    pin=w.freeze_script(tmp_path/'frozen.json')
    r=w.freeze_request(c,days[0],b'snapshot')
    if discard_return:
        seal=summary.seal
        def misleading_summary(source,manifest,previous_head):
            manifest['incident']='PRIVATE_EXPECTED'
            return seal(source,manifest,previous_head)
        monkeypatch.setattr(summary,'seal',misleading_summary)
    def operation(private):
        event=w.run_attempt(private/'journal',calendar=c,request=r,snapshot=b'snapshot',
            at=cutoff(r['origin']),codes=codes,phase=phase,
            frozen_path=tmp_path/'frozen.json',pin=pin)
        assert event['decision']['action']==action
        return None if discard_return else event
    for i,d in enumerate(days):
        directory=tmp_path/'chain'/str(i)
        record=summary.run_synthetic_session(directory,source=summary.synthetic_source(d),
            private_operation=operation if i==0 else lambda _:None,
            previous_head=head,previous_private_commitment=previous)
        dirs.append(directory);head=record['head'];previous=record['commitment']
    manifest=json.loads((dirs[0]/'private/private-manifest.json').read_bytes())
    assert manifest['incident']==('PRIVATE_EXPECTED' if discard_return else action)
    # The committed summary is deliberately ignored by the unblind decision.
    with pytest.raises(Stop,match='irreversible incident invalidity'):
        w.manual_unblind(calendar=c,expected_sessions=days,directories=dirs,external_head=head,
            state='VALID',at=cutoff(c.shift(days[-1],11)),publication_complete=True,
            human='Human',manual=True,frozen_path=tmp_path/'frozen.json',pin=pin,
            output=tmp_path/'opened.json')


@pytest.mark.parametrize('tamper', ['rename', 'delete', 'unregistered'])
def test_r4_attempt_inventory_cannot_disappear_before_seal(tmp_path, tamper):
    c,r=ticket(); pin=w.freeze_script(tmp_path/'frozen.json')
    def operation(private):
        w.run_attempt(private/'journal',calendar=c,request=r,snapshot=b'fixed synthetic snapshot',
            at=cutoff(r['origin']),codes=['UNKNOWN_ERROR'],phase='FORMAL',
            frozen_path=tmp_path/'frozen.json',pin=pin)
        event=private/'journal/events/000.json'
        if tamper=='rename': event.parent.rename(private/'journal/attempt-records')
        elif tamper=='delete': event.unlink()
        else:
            import shutil
            shutil.copyfile(event,private/'unregistered.json')
        # Deliberately discard the fatal return value.
    with pytest.raises(Stop):
        summary.run_synthetic_session(tmp_path/'sealed',private_operation=operation)


@pytest.mark.parametrize('delay', [1,2])
@pytest.mark.parametrize('scenario', ['same_session','cross_session','SOURCE_OPTIONAL','PRIVATE_EXPECTED'])
def test_r4_recovered_retry_allows_unblind(tmp_path, delay, scenario):
    c=calendar(); days=c.sessions[31:283]
    r=w.freeze_request(c,days[0],b'snapshot'); pin=w.freeze_script(tmp_path/'frozen.json')
    w.independent_reference(tmp_path/'reference',tmp_path/'frozen.json',pin,request=r)
    def operation(private):
        import shutil
        if scenario=='cross_session' and session_index:
            shutil.copytree(dirs[-1]/'private/journal',private/'journal')
        attempts = [session_index] if scenario=='cross_session' else range(delay+1)
        for i in attempts:
            event=w.run_attempt(private/'journal',calendar=c,request=r,snapshot=b'snapshot',
                at=cutoff(c.shift(r['origin'],i)),codes=['RETRAIN_TRANSIENT'] if i<delay else
                    [scenario] if scenario in ('SOURCE_OPTIONAL','PRIVATE_EXPECTED') else [],
                phase='FORMAL',frozen_path=tmp_path/'frozen.json',pin=pin,
                reference=tmp_path/'reference/run-0',candidate=tmp_path/'reference/run-1',
                reference_c0plus=tmp_path/'reference/run-0-c0plus',candidate_c0plus=tmp_path/'reference/run-1-c0plus')
        assert event['decision']['action']==('RETRY' if scenario=='cross_session' and session_index<delay else 'SWITCH_ALL_THREE')
        # Also verify discarded successful returns are derived from the journal.
    dirs=[]; head='0'*64; previous='0'*64
    for i,d in enumerate(days):
        session_index=i
        directory=tmp_path/'chain'/str(i)
        record=summary.run_synthetic_session(directory,source=summary.synthetic_source(d),
            private_operation=operation if i==0 or scenario=='cross_session' and i<=delay else lambda _:None,
            previous_head=head,previous_private_commitment=previous)
        dirs.append(directory);head=record['head'];previous=record['commitment']
    assert w.manual_unblind(calendar=c,expected_sessions=days,directories=dirs,external_head=head,
        state='VALID',at=cutoff(c.shift(days[-1],11)),publication_complete=True,
        human='Human',manual=True,frozen_path=tmp_path/'frozen.json',pin=pin,
        output=tmp_path/'opened.json')['return_sessions']==252
    # P2 L285: transient is allowed only BEFORE producing/switching a version.
    # A new failure after success is terminal misuse, never a fresh retry.
    def failure_after_success(private):
        nonlocal session_index
        session_index=delay if scenario=='cross_session' else 0
        if scenario=='cross_session':
            import shutil
            shutil.copytree(dirs[delay]/'private/journal',private/'journal')
        else:
            operation(private)
        w.run_attempt(private/'journal',calendar=c,request=r,snapshot=b'snapshot',
            at=cutoff(c.shift(r['origin'],delay+1)),codes=['RETRAIN_TRANSIENT'],
            phase='FORMAL',frozen_path=tmp_path/'frozen.json',pin=pin)
    summary.run_synthetic_session(tmp_path/'after-success',private_operation=failure_after_success)
    assert summary.private_operation_status(tmp_path/'after-success/private')=='UNKNOWN_ERROR'


@pytest.mark.parametrize('phase,delays,codes,expected', [
    ('FORMAL',1,['RETRAIN_TRANSIENT'],'RETRY'),
    ('FORMAL',3,['RETRAIN_TRANSIENT'],'INVALID_FOR_CONFIRMATION'),
    ('PRESTART',1,['RETRAIN_TRANSIENT'],'STOP'),
    ('FORMAL',1,['UNKNOWN_ERROR'],'INVALID_FOR_CONFIRMATION'),
    ('FORMAL',1,['BLIND_INTEGRITY'],'INVALID_FOR_CONFIRMATION'),
    ('FORMAL',1,['SOURCE_MISSING'],'INVALID_FOR_CONFIRMATION'),
    ('FORMAL',1,['RETRAIN_IDENTITY'],'INVALID_FOR_CONFIRMATION'),
    ('FORMAL',1,['PRIVATE_MARK_EXECUTION'],'INVALID_FOR_CONFIRMATION'),
])
def test_r4_spec_incident_transitions(tmp_path,phase,delays,codes,expected):
    # P2 L118 / L275-290: fixed snapshot retry, timeout, prestart and fatal table.
    c,r=ticket(); pin=w.freeze_script(tmp_path/'frozen.json')
    def operation(private):
        for i in range(delays):
            w.run_attempt(private/'journal',calendar=c,request=r,snapshot=b'fixed synthetic snapshot',
                at=cutoff(c.shift(r['origin'],i)),codes=codes,phase=phase,
                frozen_path=tmp_path/'frozen.json',pin=pin)
    summary.run_synthetic_session(tmp_path/'sealed',private_operation=operation)
    assert summary.private_operation_status(tmp_path/'sealed/private')==expected


def test_unblind_derives_incident_invalidity(tmp_path):
    c=calendar(); days=c.sessions[31:283]; dirs=[]; head='0'*64; previous='0'*64
    def failure(_): raise RuntimeError('synthetic incident')
    for i,d in enumerate(days):
        directory=tmp_path/'chain'/str(i)
        record=summary.run_synthetic_session(directory,source=summary.synthetic_source(d),
            private_operation=failure if i==0 else lambda _:None,
            previous_head=head,previous_private_commitment=previous)
        dirs.append(directory);head=record['head'];previous=record['commitment']
    assert w.verify_chain(dirs,days,head)
    pin=w.freeze_script(tmp_path/'frozen.json')
    with pytest.raises(Stop):
        w.manual_unblind(calendar=c,expected_sessions=days,directories=dirs,external_head=head,
            state='VALID',at=cutoff(c.shift(days[-1],11)),publication_complete=True,
            human='Human',manual=True,frozen_path=tmp_path/'frozen.json',pin=pin,output=tmp_path/'open.json')


def test_same_store_rejected_before_acceptance(tmp_path):
    pin=w.freeze_script(tmp_path/'pin.json'); c,r=ticket()
    event=w.run_attempt(tmp_path/'journal',calendar=c,request=r,snapshot=b'fixed synthetic snapshot',
        at=cutoff(r['origin']),codes=[],phase='FORMAL',frozen_path=tmp_path/'pin.json',pin=pin,
        reference=tmp_path/'same',candidate=tmp_path/'same',
        reference_c0plus=tmp_path/'same-c0',candidate_c0plus=tmp_path/'same-c0')
    assert event['decision']['action']=='INVALID_FOR_CONFIRMATION'


def test_public_extra_file_stops(tmp_path):
    c,days,dirs,head=chain(tmp_path/'chain')
    (dirs[0]/'scores.csv').write_text('private score')
    with pytest.raises(Stop): w.verify_chain(dirs,days,head)


@pytest.fixture(autouse=True)
def isolated_unblind_registry(request,tmp_path,monkeypatch):
    if 'unblind' in request.node.name:
        monkeypatch.setattr(w.store,'ROOT',tmp_path)
