import numpy as np, pandas as pd, pytest
from research_lab import macro_sync as ms

TK = {'^VIX': 'VIX', '^TWII': 'TWII_Close'}
ALL6 = {'^VIX': 'VIX', '^GSPC': 'SPX', '^TWII': 'TWII_Close', 'GC=F': 'Gold', 'CL=F': 'Oil', '^TNX': 'TNX', 'TWD=X': 'USD_TWD'}
NIGHT = '2026-07-27 22:30'          # Taipei, Taiwan market closed
MIDDAY = '2026-07-27 11:00'         # Taiwan market still open


class Sleeper:
    def __init__(self): self.calls = []
    def __call__(self, s): self.calls.append(s)


def frame(dates, vals):
    return pd.DataFrame({'Close': vals}, index=pd.DatetimeIndex(dates))


def base(tmp_path):
    p = tmp_path/'macro.parquet'
    pd.DataFrame({'Date': pd.to_datetime(['2026-07-22', '2026-07-23']), 'VIX': [15., 16.], 'TWII_Close': [100., 101.]}).to_parquet(p)
    return p


class Dl:
    """Recording downloader: table[ticker] is a frame, an Exception, or a list consumed one per call."""
    def __init__(self, table): self.t, self.calls = table, []
    def __call__(self, tk, start, end):
        self.calls.append((tk, start, end)); v = self.t[tk]
        if isinstance(v, list): v = v.pop(0) if len(v) > 1 else v[0]
        if isinstance(v, Exception): raise v
        return v
    def n(self, tk): return sum(c[0] == tk for c in self.calls)


def run(p, today, t, tickers=TK, now=NIGHT, **kw):
    d = Dl(t); sl = kw.pop('sleep', Sleeper())
    return ms.sync_macro(p, today, tickers, d, sleep=sl, now=now, **kw), d, sl


def test_stale_cache_is_refreshed_not_skipped(tmp_path):            # regression: old code returned early whenever the file existed
    p = base(tmp_path)
    r, _, _ = run(p, '2026-07-25', {'^VIX': frame(['2026-07-23', '2026-07-24'], [99., 17.]), '^TWII': frame(['2026-07-24'], [102.])})
    out = pd.read_parquet(p).set_index('Date')
    assert out.loc['2026-07-24', 'VIX'] == 17. and out.loc['2026-07-24', 'TWII_Close'] == 102.
    assert out.loc['2026-07-23', 'VIX'] == 16.                    # stored value kept although the provider now says 99 ...
    assert r['VIX']['conflicts']['n'] == 1 and r['VIX']['conflicts']['dates'] == ['2026-07-23']   # ... and the disagreement is reported
    assert r['VIX']['status'] == ms.OK and r['_written'] and r['_stale'] == []
    assert list(pd.read_parquet(p).columns) == ['Date', 'VIX', 'TWII_Close'] and not (tmp_path/'macro.tmp').exists()


def test_download_window_end_is_exclusive_and_start_overlaps(tmp_path):
    p = base(tmp_path)
    _, d, _ = run(p, '2026-07-25', {'^VIX': frame(['2026-07-24'], [17.]), '^TWII': frame(['2026-07-24'], [102.])}, overlap_days=10)
    assert d.calls == [('^VIX', '2026-07-13', '2026-07-26'), ('^TWII', '2026-07-13', '2026-07-26')]
    q = tmp_path/'new.parquet'
    _, d2, _ = run(q, '2026-07-25', {'^VIX': frame(['2026-07-24'], [17.]), '^TWII': frame(['2026-07-24'], [102.])})
    assert d2.calls[0][1] == '2005-01-01'


def test_empty_response_is_no_data_not_retried_not_defaulted(tmp_path):
    p = base(tmp_path); before = pd.read_parquet(p)
    r, d, sl = run(p, '2026-09-01', {'^VIX': pd.DataFrame(), '^TWII': pd.DataFrame()})
    assert r['VIX']['status'] == ms.NO_DATA and r['VIX']['attempts'] == 1 and d.n('^VIX') == 1 and sl.calls == [] and '_written' not in r
    pd.testing.assert_frame_equal(pd.read_parquet(p), before)     # nothing written, no default values invented
    assert set(r['_stale']) == {'VIX', 'TWII_Close'}


def test_transient_error_retried_with_exponential_backoff(tmp_path):
    p = base(tmp_path)
    r, d, sl = run(p, '2026-07-25', {'^VIX': [ConnectionError('429'), ConnectionError('429'), frame(['2026-07-24'], [17.])], '^TWII': frame(['2026-07-24'], [102.])})
    assert r['VIX']['status'] == ms.OK and r['VIX']['attempts'] == 3 and sl.calls == [2., 4.]
    assert pd.read_parquet(p).set_index('Date').loc['2026-07-24', 'VIX'] == 17.     # the retried bar really was stored


def test_persistent_failure_reported_other_columns_still_updated(tmp_path):
    p = base(tmp_path)
    r, d, sl = run(p, '2026-07-25', {'^VIX': TimeoutError('x'), '^TWII': frame(['2026-07-24'], [102.])})
    assert r['VIX']['status'] == ms.FAILED and r['VIX']['attempts'] == 3 and sl.calls == [2., 4.] and r['TWII_Close']['status'] == ms.OK
    assert np.isnan(pd.read_parquet(p).set_index('Date').loc['2026-07-24', 'VIX'])


@pytest.mark.parametrize('bad', [pd.DataFrame({'Price': [1.]}, index=pd.DatetimeIndex(['2026-07-24'])),
                                 pd.DataFrame({'Close': ['abc']}, index=pd.DatetimeIndex(['2026-07-24']))])
def test_schema_change_or_unparseable_payload_is_schema_not_retried_not_written(tmp_path, bad):
    p = base(tmp_path); before = pd.read_parquet(p)
    r, d, _ = run(p, '2026-07-25', {'^VIX': bad, '^TWII': bad})
    assert r['VIX']['status'] == ms.SCHEMA and r['VIX']['attempts'] == 1 and d.n('^VIX') == 1 and '_written' not in r
    pd.testing.assert_frame_equal(pd.read_parquet(p), before)


def test_multiindex_columns_accepted(tmp_path):
    mi = pd.DataFrame({('Close', '^VIX'): [17.]}, index=pd.DatetimeIndex(['2026-07-24']))
    r, _, _ = run(base(tmp_path), '2026-07-25', {'^VIX': mi, '^TWII': pd.DataFrame()})
    assert r['VIX']['status'] == ms.OK and r['VIX']['filled'] == 1


def test_unfinished_bar_dated_today_never_stored_for_any_non_tw_column(tmp_path):
    p = tmp_path/'m.parquet'
    day = lambda v: frame(['2026-07-24', '2026-07-27'], [v, v+1])
    t = {tk: day(10.+i) for i, tk in enumerate(ALL6)}
    r, _, _ = run(p, '2026-07-27', t, tickers=ALL6)
    out = pd.read_parquet(p).set_index('Date')
    for i, col in enumerate(ALL6.values()):
        if col != 'TWII_Close': assert (pd.Timestamp('2026-07-27') not in out.index) or np.isnan(out.loc['2026-07-27', col]), col
        assert out.loc['2026-07-24', col] == 10.+i        # yesterday's finished bar is stored
    assert out.loc['2026-07-27', 'TWII_Close'] == 13.             # closed Taiwan index: today's bar accepted at 22:30


def test_taiwan_intraday_bar_not_stored_before_close(tmp_path):     # reviewer: a manual midday run must not freeze a half-finished TWII bar
    p = base(tmp_path)
    r, _, _ = run(p, '2026-07-27', {'^VIX': frame(['2026-07-24'], [17.]), '^TWII': frame(['2026-07-24', '2026-07-27'], [102., 103.])}, now=MIDDAY)
    out = pd.read_parquet(p).set_index('Date')
    assert out.loc['2026-07-24', 'TWII_Close'] == 102. and (pd.Timestamp('2026-07-27') not in out.index or np.isnan(out.loc['2026-07-27', 'TWII_Close']))


def test_today_defaults_to_taipei_calendar_day(tmp_path):
    p = base(tmp_path)
    d = Dl({'^VIX': frame(['2026-07-24'], [17.]), '^TWII': frame(['2026-07-24'], [102.])})
    ms.sync_macro(p, None, TK, d, sleep=Sleeper(), now='2026-07-25 01:00+08:00')   # 17:00 UTC on the 24th is already the 25th in Taipei
    assert d.calls[0][2] == '2026-07-26'


def test_first_build_all_failed_reports_every_column_stale(tmp_path):   # reviewer: this used to look healthy
    r, _, _ = run(tmp_path/'none.parquet', '2026-07-25', {'^VIX': TimeoutError('x'), '^TWII': pd.DataFrame()})
    assert r['_stale'] == ['TWII_Close', 'VIX'] and not (tmp_path/'none.parquet').exists()


def test_expected_column_missing_from_file_is_stale(tmp_path):
    r, _, _ = run(base(tmp_path), '2026-07-25', {'^VIX': frame(['2026-07-24'], [17.]), '^TWII': frame(['2026-07-24'], [102.]), '^TNX': pd.DataFrame()},
                  tickers={**TK, '^TNX': 'TNX'})
    assert r['_stale'] == ['TNX']


def test_freshness_boundary_lag_7_ok_lag_8_stale():
    df = pd.DataFrame({'Date': pd.to_datetime(['2026-07-18']), 'A': [1.]})
    assert not ms.freshness(df, '2026-07-25', 7)['A']['stale'] and ms.freshness(df, '2026-07-26', 7)['A']['stale']
    assert ms.freshness(pd.DataFrame({'Date': pd.to_datetime(['2026-07-18']), 'A': [np.nan]}), '2026-07-19')['A']['stale']


def test_no_new_data_does_not_rewrite_file(tmp_path):
    p = base(tmp_path); before = p.read_bytes()
    r, _, _ = run(p, '2026-07-24', {'^VIX': frame(['2026-07-22', '2026-07-23'], [15., 16.]), '^TWII': frame(['2026-07-22', '2026-07-23'], [100., 101.])})
    assert '_written' not in r and p.read_bytes() == before
