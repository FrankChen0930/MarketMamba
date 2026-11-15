import json
from research_lab import fetch_core as fc


class Resp:
    def __init__(self, code, body):
        self.status_code, self._b = code, body

    def json(self):
        if self._b is None:
            raise ValueError('bad json')
        return self._b


class Sess:
    def __init__(self, seq):
        self.seq, self.calls = list(seq), 0

    def get(self, *a, **k):
        self.calls += 1
        x = self.seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


def run(seq, **kw):
    s = Sess(seq); slept = []
    return fc.fetch_json(s, 'http://x', {'d': 1}, sleep=slept.append, rng=lambda: .5, **kw), s, slept


def test_retries_only_transient_then_succeeds_with_backoff():
    r, s, slept = run([Resp(503, None), ConnectionError('reset'), Resp(200, None), Resp(200, {'stat': 'OK', 'data': [1]})], expect_keys=('data',))
    assert r.status == fc.OK and r.attempts == 4 and s.calls == 4 and slept == [1.5, 3.0, 6.0]
    assert r.payload == {'stat': 'OK', 'data': [1]}


def test_holiday_is_no_data_not_error_and_not_retried():
    r, s, _ = run([Resp(200, {'stat': '很抱歉，沒有符合條件的資料!'})])
    assert r.status == fc.NO_DATA and s.calls == 1
    r2, s2, _ = run([Resp(200, {'stat': 'OK', 'date': '20260101'})], date_check=lambda b: b['date'] == '20260102')
    assert r2.status == fc.NO_DATA and 'date' in r2.reason and s2.calls == 1


def test_schema_change_and_blocked_are_not_retried_and_are_reported():
    r, s, _ = run([Resp(200, {'stat': 'OK'})], expect_keys=('tables',))
    assert r.status == fc.SCHEMA and s.calls == 1
    r, s, _ = run([Resp(403, None)])
    assert r.status == fc.BLOCKED and s.calls == 1


def test_gives_up_after_retries_and_archives_raw_payload(tmp_path):
    r, s, _ = run([Resp(500, None)]*10, retries=2)
    assert r.status == fc.TRANSIENT and r.attempts == 3
    r, *_ = run([Resp(200, {'stat': 'OK', 'x': '股'})], archive_dir=tmp_path, key='k')
    assert json.loads((tmp_path/'k.json').read_text(encoding='utf-8')) == {'stat': 'OK', 'x': '股'} and r.raw_path


def test_coverage_report_lists_actionable_dates():
    res = {'a': fc.Result(fc.OK), 'b': fc.Result(fc.NO_DATA), 'c': fc.Result(fc.TRANSIENT), 'd': fc.Result(fc.SCHEMA)}
    c = fc.coverage(res)
    assert c['counts'] == {'OK': 1, 'NO_DATA': 1, 'TRANSIENT': 1, 'SCHEMA': 1} and c['needs_action'] == {'TRANSIENT': ['c'], 'SCHEMA': ['d']}
