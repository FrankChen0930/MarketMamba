"""Classified, retrying, raw-archiving HTTP fetch for public market-data endpoints (prototype for the data-stability work).

Why: the production fetchers collapse every failure into `return None` (51 broad `except Exception`, 124 `return None` in
fetcher.py), so "holiday", "not published yet", "transient network error", "HTTP 429/5xx" and "schema changed" are
indistinguishable and cannot be handled differently (retry vs. wait vs. alarm). Here every call returns a Result whose
`status` says which, only TRANSIENT is retried (with exponential backoff + jitter), and the raw payload is archived so a
parser bug can be fixed later without re-downloading.  Public endpoints only; no credentials are read or sent.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

OK, NO_DATA, TRANSIENT, SCHEMA, BLOCKED = 'OK', 'NO_DATA', 'TRANSIENT', 'SCHEMA', 'BLOCKED'


@dataclass
class Result:
    status: str
    payload: object = None
    reason: str = ''
    attempts: int = 1
    raw_path: str | None = None
    meta: dict = field(default_factory=dict)


def classify(resp_status, body, expect_keys=(), date_check=None):
    """-> (status, reason). `body` is parsed JSON or None. NO_DATA = the endpoint answered correctly that there is nothing for
    that date (holiday / not yet published); it is a valid outcome, not an error, and must not be retried in a tight loop."""
    if resp_status in (429, 500, 502, 503, 504):
        return TRANSIENT, f'HTTP {resp_status}'
    if resp_status in (401, 403):
        return BLOCKED, f'HTTP {resp_status}'
    if resp_status != 200:
        return SCHEMA, f'HTTP {resp_status}'
    if body is None:
        return TRANSIENT, 'unparseable body'                       # truncated / HTML error page: retry
    stat = str(body.get('stat', 'OK')) if isinstance(body, dict) else 'OK'
    if stat.strip().upper() not in ('OK', 'SUCCESS'):          # TWSE says 'OK', TPEx says 'ok'
        return (NO_DATA, f'stat={stat}') if ('沒有' in stat or 'No' in stat or '很抱歉' in stat) else (SCHEMA, f'stat={stat}')
    missing = [k for k in expect_keys if not (isinstance(body, dict) and k in body)]
    if missing:
        return SCHEMA, f'missing keys {missing}'
    if date_check is not None and not date_check(body):
        return NO_DATA, 'response date != requested date'          # TWSE answers with the latest session for holidays
    return OK, ''


def fetch_json(session, url, params=None, *, expect_keys=(), date_check=None, retries=4, base_delay=1.5, timeout=30,
               archive_dir=None, key=None, sleep=time.sleep, rng=random.random, headers=None):
    attempts, last = 0, (TRANSIENT, 'not attempted')
    for attempts in range(1, retries+2):
        try:
            resp = session.get(url, params=params, timeout=timeout, headers=headers or {'User-Agent': 'Mozilla/5.0'})
            try:
                body = resp.json()
            except ValueError:
                body = None
            last = classify(resp.status_code, body, expect_keys, date_check)
        except Exception as e:                                       # noqa: BLE001 - network layer only; classified, never swallowed
            body, last = None, (TRANSIENT, f'{type(e).__name__}: {e}')
        if last[0] != TRANSIENT or attempts > retries:
            break
        sleep(base_delay*(2**(attempts-1))*(0.5+rng()))
    raw = None
    if archive_dir is not None and body is not None:
        name = key or hashlib.sha256((url+json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()[:16]
        p = Path(archive_dir)/f'{name}.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(body, ensure_ascii=False)); tmp.replace(p)
        raw = str(p)
    return Result(last[0], body if last[0] == OK else None, last[1], attempts, raw)


def coverage(results):
    """results: {date: Result} -> counts per status and the dates needing action (TRANSIENT/SCHEMA/BLOCKED), for a fill report."""
    out = {}
    for d, r in results.items():
        out.setdefault(r.status, []).append(d)
    return dict(counts={k: len(v) for k, v in out.items()}, needs_action={k: sorted(v) for k, v in out.items() if k in (TRANSIENT, SCHEMA, BLOCKED)})
