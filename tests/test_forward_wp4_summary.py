"""Synthetic blind-boundary pairs; no market or blind data."""
import copy
from datetime import timedelta
import hashlib

import pytest
from tools import forward_wp4_summary as s


@pytest.mark.parametrize('missing,unknown', [(False, False), (True, False), (False, True)])
def test_public_market_information_cannot_distinguish_private_holdings(missing, unknown):
    source = s.synthetic_source(missing=missing, unknown_lifecycle=unknown)
    a, nonce_a = s.seal(source, dict(holding='X', exposure='suspended', pnl=-9, incident='PRIVATE_EXPECTED'))
    b, nonce_b = s.seal(source, dict(holding='Y', exposure='delisted', pnl=42, incident='PRIVATE_MARK_EXECUTION'))
    assert {k:v for k,v in a.items() if k not in s.CHAIN_KEYS} == {k:v for k,v in b.items() if k not in s.CHAIN_KEYS}
    assert len(s.canonical(a)) == len(s.canonical(b))
    assert nonce_a != nonce_b and len(nonce_a) == len(nonce_b) == 32
    for record, nonce in [(a, nonce_a), (b, nonce_b)]:
        output = s.release_bytes(record, s.release_at(record['session']))
        assert nonce.hex().encode() not in output
        for forbidden in [b'holding', b'pnl', b'incident', b'PRIVATE_', b'nonce', b'score', b'NAV']:
            assert forbidden not in output
    assert s.release_at(a['session']) == s.release_at(b['session'])


def test_unsalted_dictionary_attack_and_private_nonce():
    dictionary = [dict(holding=v) for v in ['X', 'Y', 'Z']]
    private = dictionary[1]
    unsalted = hashlib.sha256(s.canonical(private)).hexdigest()
    assert [p for p in dictionary if hashlib.sha256(s.canonical(p)).hexdigest() == unsalted] == [private]
    public, nonce = s.seal(s.synthetic_source(), private)
    assert all(hashlib.sha256(s.canonical(p)).hexdigest() != public['commitment'] for p in dictionary)
    assert all(s.commitment(p, bytes(32)) != public['commitment'] for p in dictionary)
    assert s.commitment(private, nonce) == public['commitment']


@pytest.mark.parametrize('place,key', [('root','score'), ('root','nonce'), ('root','error'),
    ('source','MODEL_FAILURE'), ('fields','holdings'), ('counts','PRIVATE_MARK_EXECUTION')])
def test_closed_schema_rejects_private_or_unknown_fields(place, key):
    source = s.synthetic_source()
    if place == 'root': source[key] = 7
    elif place == 'source': source['source_files'][key] = dict(expected=1,received=1,valid=1)
    elif place == 'fields': source['universe_rows'][key] = dict(expected=1,valid=1)
    else: source['source_error_counts'][s.SOURCE][key] = 1
    with pytest.raises(s.Stop): s.seal(source, {})


@pytest.mark.parametrize('value', [-1, True, 1.1, '1', None])
def test_invalid_source_count_rejected(value):
    source = s.synthetic_source(); source['source_files'][s.SOURCE]['expected'] = value
    with pytest.raises(s.Stop): s.seal(source, {})


def test_zero_and_unknown_denominators_do_not_turn_into_zero_risk():
    assert s.coverage_ratio(dict(expected=0, valid=0)) is None
    assert s.coverage_ratio(dict(expected=None, valid=None)) is None
    assert s.coverage_ratio(dict(expected=100, valid=99)) == .99
    unknown = s.synthetic_source(unknown_lifecycle=True)
    assert unknown['source_error_counts'][s.SOURCE]['SOURCE_UNIVERSE'] == 1
    unknown['source_error_counts'][s.SOURCE]['SOURCE_UNIVERSE'] = 0
    with pytest.raises(s.Stop): s.validate_source(unknown)


@pytest.mark.parametrize('code', s.SOURCE_CODES + s.PRIVATE_CODES + ('UNRECOGNIZED',))
@pytest.mark.parametrize('phase', ['PRESTART', 'FORMAL'])
def test_all_error_branches_and_multierror_priority(code, phase):
    expected = ('NO_EFFECT' if code in ('SOURCE_OPTIONAL','PRIVATE_EXPECTED') else
                'RETRY_SAME_SNAPSHOT_R_PLUS_2_LIMIT' if code == 'RETRAIN_TRANSIENT' else
                'STOP' if phase == 'PRESTART' else 'INVALID_FOR_CONFIRMATION')
    assert s.disposition([code],phase) == expected
    fatal = 'STOP' if phase == 'PRESTART' else 'INVALID_FOR_CONFIRMATION'
    assert s.disposition([code, 'RETRAIN_IDENTITY'],phase) == fatal


def test_chain_external_head_missing_session_and_fixed_clock():
    records = []; previous = '0'*64
    sessions = ['2026-11-26','2026-11-27']
    for day in sessions:
        public, _ = s.seal(s.synthetic_source(session=day), {}, previous)
        records.append(public); previous = public['head']
    assert s.verify_chain(records,sessions,previous)
    for at in [s.release_at(sessions[0])-timedelta(seconds=1), s.release_at(sessions[0])+timedelta(seconds=1)]:
        with pytest.raises(s.Stop): s.release_bytes(records[0],at)
    with pytest.raises(s.Stop): s.verify_chain(records[:1],sessions,previous)
    with pytest.raises(s.Stop): s.verify_chain(records,sessions,'f'*64)
    changed = copy.deepcopy(records); changed[1]['source_files'][s.SOURCE]['valid'] = 0
    with pytest.raises(s.Stop): s.verify_chain(changed,sessions,previous)


def test_private_runner_captures_python_native_logs_and_failure_without_public_signal(tmp_path, capfd):
    import os
    def operation(out):
        print('SECRET_MODEL_OR_SCORE')
        os.write(2,b'SECRET_NATIVE_DIAGNOSTIC\n')
        raise RuntimeError('SECRET_EXCEPTION_SCORE')
    bad = s.run_synthetic_session(tmp_path/'bad', private_operation=operation)
    good = s.run_synthetic_session(tmp_path/'good', private_operation=lambda p: None)
    captured = capfd.readouterr()
    assert 'SECRET_' not in captured.out + captured.err
    log = (tmp_path/'bad/private/runner.log').read_text()
    assert 'SECRET_MODEL_OR_SCORE' in log and 'SECRET_NATIVE_DIAGNOSTIC' in log
    assert {k:v for k,v in bad.items() if k not in s.CHAIN_KEYS} == {k:v for k,v in good.items() if k not in s.CHAIN_KEYS}
    assert len(s.canonical(bad)) == len(s.canonical(good))
    import json
    manifest = json.loads((tmp_path/'bad/private/private-manifest.json').read_text())
    nonce = (tmp_path/'bad/private/nonce.bin').read_bytes()
    assert manifest['incident'] == 'UNKNOWN_ERROR'
    assert s.commitment(manifest,nonce) == bad['commitment']
    for path in [tmp_path/'bad/public-summary.json', tmp_path/'good/public-summary.json']:
        assert b'SECRET_' not in path.read_bytes()


def test_synthetic_private_chain_and_nonce_tampering(tmp_path):
    first = tmp_path/'first'; second = tmp_path/'second'
    a = s.run_synthetic_session(first, private_operation=lambda p: None)
    b = s.run_synthetic_session(second, private_operation=lambda p: None,
        source=s.synthetic_source('2026-11-27'),previous_head=a['head'],
        previous_private_commitment=a['commitment'])
    days = ['2026-11-26','2026-11-27']
    assert s.verify_synthetic_sessions([first,second],days,b['head'])
    (second/'private/nonce.bin').write_bytes(bytes(32))
    with pytest.raises(s.Stop): s.verify_synthetic_sessions([first,second],days,b['head'])


def test_private_sensitive_log_tampering_rejected(tmp_path):
    root = tmp_path/'session'
    record = s.run_synthetic_session(root,private_operation=lambda p: None)
    (root/'private/runner.log').write_text('tampered')
    with pytest.raises(s.Stop): s.verify_synthetic_sessions([root],['2026-11-26'],record['head'])


def test_full_universe_denominator_cannot_be_trimmed_for_one_field():
    source = s.synthetic_source()
    source['universe_rows']['base_features'] = dict(expected=99,valid=99)
    with pytest.raises(s.Stop): s.seal(source,{})
