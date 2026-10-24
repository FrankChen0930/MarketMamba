"""Offline proposal guards, not a collector/calendar feed or research admission.

Caller-pinned calendar evidence is a consistency boundary, not authenticated
market history. Explicit daily coverage and revision chains are required; no
weekday heuristic can silently promote an unknown day to an actual session.
"""
from datetime import date, timedelta
from zoneinfo import ZoneInfo
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_research_temporal import instant
from V6.experimental.v7_temporal_contract import normalize_calendar, first_session_after


def calendar_status(calendar, expected_pin, session, asof, *, captured_sessions=()):
    result = dict(status='BLOCKED', reasons=[], missing_session=False,
                  last_actual_session=None, next_scheduled_session=None,
                  next_actual_verified=False, research_admitted=False,
                  collector_activated=False)
    try:
        if canonical_hash(calendar) != expected_pin:
            raise ValueError('CALENDAR_IDENTITY_MISMATCH')
        if calendar['schema'] != 'v7-capture-calendar-proposal-v1' or not calendar['market']:
            raise ValueError('CALENDAR_SCHEMA_OR_MARKET_UNKNOWN')
        start, end = date.fromisoformat(calendar['coverage_start']), date.fromisoformat(calendar['coverage_end'])
        target, now = date.fromisoformat(session), instant(asof)
        if not isinstance(captured_sessions, (list, tuple)):
            raise ValueError('CAPTURE_SESSION_IDENTITIES_REQUIRED')
        if captured_sessions:
            normalize_calendar(captured_sessions)
        if not start <= target <= end:
            raise ValueError('OUTSIDE_EXPLICIT_COVERAGE')
        by_day, seen = {}, set()
        for row in calendar['days']:
            day = date.fromisoformat(row['session'])
            known = instant(row['known_at'])
            if not start <= day <= end or row['state'] not in {'TRADED', 'CLOSED', 'SCHEDULED'}:
                raise ValueError('INVALID_DAY_STATE')
            if not isinstance(row['evidence_ref'], str) or not row['evidence_ref'].strip():
                raise ValueError('MISSING_ANNOUNCEMENT_EVIDENCE')
            rid = row['revision_id']
            if not isinstance(rid, str) or not rid.strip() or (day, rid) in seen:
                raise ValueError('AMBIGUOUS_CALENDAR_REVISION')
            seen.add((day, rid))
            by_day.setdefault(day, []).append((known, row))
        coverage = [start + timedelta(days=i) for i in range((end-start).days+1)]
        if set(by_day) != set(coverage):
            raise ValueError('MISSING_CALENDAR_DAY')
        active = {}
        for day, chain in by_day.items():
            chain.sort(key=lambda item: item[0])
            previous, previous_time = None, None
            for known, row in chain:
                if row.get('supersedes') != previous or (previous_time is not None and known <= previous_time):
                    raise ValueError('UNLINKED_OR_AMBIGUOUS_REVISION')
                previous, previous_time = row['revision_id'], known
                if known <= now:
                    active[day.isoformat()] = row
        if set(active) != {d.isoformat() for d in coverage}:
            raise ValueError('CALENDAR_NOT_KNOWN_ASOF')
        local_day = now.astimezone(ZoneInfo('Asia/Taipei')).date().isoformat()
        actual = [d for d, r in active.items() if r['state'] == 'TRADED' and d <= local_day and d <= session]
        if any(d > local_day and r['state'] == 'TRADED' for d, r in active.items()):
            raise ValueError('FUTURE_ACTUAL_SESSION_CLAIM')
        if actual:
            result['last_actual_session'] = normalize_calendar(sorted(actual))[-1]
        scheduled = sorted(d for d, r in active.items() if r['state'] in {'TRADED', 'SCHEDULED'})
        if scheduled:
            result['next_scheduled_session'] = first_session_after(scheduled, session)
        state = active[session]['state']
        if state == 'CLOSED':
            result['status'] = 'NON_TRADING_DAY'
        elif state == 'SCHEDULED':
            result['status'] = 'SCHEDULED_NOT_YET_VERIFIED'
        else:
            result['missing_session'] = session not in captured_sessions
            result['status'] = 'MISSING_TRADING_SESSION' if result['missing_session'] else 'CAPTURE_PRESENT'
        result['calendar_sha256'] = expected_pin
        result['revision_id'] = active[session]['revision_id']
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        result.update(status='BLOCKED', reasons=[str(exc)], missing_session=False,
                      last_actual_session=None, next_scheduled_session=None)
    return result


def capture_status(*, storage_verified, backup_status, qc_status):
    """Report storage separately; no resource fetch, QC calculation or activation."""
    status = 'FAILED'
    if storage_verified is True and backup_status in {'PENDING', 'VERIFIED'}:
        status = ('QC_REJECTED' if qc_status != 'PASS' else
                  'STORED_BACKUP_PENDING' if backup_status == 'PENDING' else 'STORED_AND_BACKUP_VERIFIED')
    return dict(status=status, storage_verified=storage_verified is True,
                backup_status=backup_status, qc_status=qc_status,
                collector_activated=False, research_admitted=False)
