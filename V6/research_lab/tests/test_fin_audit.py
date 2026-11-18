"""Synthetic, offline evidence for the financial audit (no production writes)."""
import json

import pandas as pd
import pytest

from research_lab import fin_audit as audit


def frame(rows):
    return pd.DataFrame(rows, columns=['Date', 'stock_id', 'type', 'value', 'origin_name'])


def test_all_aliases_from_type_and_origin_and_no_selection():
    names = {
        'revenue': [('Revenue', '收入'), ('x', '營業收入'), ('ServiceFeeRevenue', '手續費')],
        'gross_profit': [('GrossProfit', '毛利'), ('y', '營業毛利')],
        'operating_income': [('OperatingIncome', '營業利益'), ('z', '營業利益')],
        'net_income': [('IncomeAfterTax', ''), ('IncomeAfterTaxes', ''), ('NetIncome', ''),
                       ('n', '本期淨利'), ('n2', '淨利（淨損）歸屬於母公司業主')],
        'eps': [('EPS', ''), ('e', '基本每股盈餘'), ('earnings_per_share', '')],
    }
    d = frame([('2025-03-31', '2330', t, 1, o) for pairs in names.values() for t, o in pairs])
    d = pd.concat([d, frame([('2025-06-30', '2317', 'Revenue', 2, '收入')])], ignore_index=True)
    out = audit.candidate_types(d)
    for category, pairs in names.items():
        assert {x['type'] for x in out[category]} == {t for t, _ in pairs}
    r = next(x for x in out['revenue'] if x['type'] == 'Revenue')
    assert (r['stock_count'], r['period_count']) == (2, 2)
    assert audit.candidate_types(d.sample(frac=1, random_state=4)) == out


def test_type_counts_missing_and_deterministic_ties():
    d = frame([('2025-06-30', '2330', 'B', None, '中文'),
               ('2025-03-31', '2330', 'A', 1, 'English'),
               ('2025-06-30', '2317', 'A', 2, '中文')])
    out = audit.audit_types(d)
    assert [x['type'] for x in out] == ['A', 'B']
    assert out[0] == dict(type='A', row_count=2, stock_count=2, period_count=2,
                          first_period='2025-03-31', last_period='2025-06-30',
                          missing_ratio=0.0, origin_names=['English', '中文'])
    assert out[1]['missing_ratio'] == 1.0
    assert out == audit.audit_types(d.iloc[::-1])


def test_four_digit_filter_and_universe_coverage():
    d = frame([('2025-03-31', s, 'EPS', 1, '')
               for s in ['2330', '2317', '0050', '123', '000116', 'ABCD', '１２３４', None]])
    row = audit.period_coverage(d, ['2330', '2317', '9999', '123'], as_of='2025-05-16')[0]
    assert row['stock_count'] == 3
    assert row['universe_stock_count'] == 2
    assert row['expected_stock_count'] == 3
    assert row['universe_coverage_ratio'] == pytest.approx(2 / 3)
    assert row['overdue_insufficient'] is True
    assert audit.period_coverage(d, [], as_of='2025-05-16')[0]['overdue_insufficient'] is None


def test_deadlines_missing_quarter_and_strictly_after_due_date():
    d = frame([('2025-03-31', '2330', 'EPS', 1, ''),
               ('2025-09-30', '2317', 'EPS', 1, ''),
               ('2025-12-31', '2330', 'EPS', 1, '')])
    out = audit.period_coverage(d, as_of='2026-04-01')
    assert [r['deadline'] for r in out] == ['2025-05-15', '2025-08-14', '2025-11-14', '2026-03-31']
    assert [r['stock_count'] for r in out] == [1, 0, 1, 1]
    assert all(r['overdue_insufficient'] for r in out)
    assert audit.period_coverage(d, as_of='2025-05-15')[0]['overdue_insufficient'] is False
    assert audit.period_coverage(d, as_of='2025-05-16')[0]['overdue_insufficient'] is True


def test_duplicate_and_sign_checks_all_pairs_no_arbitrary_duplicate_selection():
    d = frame([('2025-03-31', '2330', t, v, '') for t, v in [
        ('Revenue', 0), ('GrossProfit', 5), ('EPS', -1), ('IncomeAfterTax', 3), ('NetIncome', -3)]] +
        [('2025-03-31', '2317', 'Revenue', -1, ''),
         ('2025-03-31', '2317', 'Revenue', -2, ''),
         ('2025-03-31', '2317', 'GrossProfit', 9, '')])
    out = audit.duplicate_and_sign_checks(d)
    assert out['duplicate_extra_rows'] == 1
    assert out['duplicate_involved_rows'] == 2
    assert out['duplicate_key_count'] == 1
    assert out['revenue_nonpositive_rows'] == 3
    assert out['revenue_negative_rows'] == 2
    assert out['revenue_zero_rows'] == 1
    assert out['gross_profit_vs_revenue'][0]['anomaly_count'] == 1
    assert out['gross_profit_vs_revenue'][0]['compared_count'] == 1
    pairs = {r['right_type']: r for r in out['eps_vs_net_income']}
    assert pairs['IncomeAfterTax']['mismatch_ratio'] == 1.0
    assert pairs['NetIncome']['mismatch_ratio'] == 0.0
    assert audit.duplicate_and_sign_checks(d.iloc[::-1]) == out


def test_missing_nonfinite_and_zero_signs_are_explicit():
    d = frame([('2025-03-31', s, t, v, '') for s, t, v in [
        ('2330', 'EPS', float('inf')), ('2330', 'NetIncome', 1),
        ('2317', 'EPS', 0), ('2317', 'NetIncome', -1)]])
    pair = audit.duplicate_and_sign_checks(d)['eps_vs_net_income'][0]
    assert (pair['compared_count'], pair['anomaly_count'], pair['mismatch_ratio']) == (1, 1, 1.0)
    assert audit.period_coverage(frame([]), as_of='2025-01-01') == []
    assert audit.audit_types(frame([])) == []


def test_financial_industry_names_distinct_stocks_and_snapshots(tmp_path):
    p = tmp_path / 'info.parquet'
    pd.DataFrame({'industry_category': ['金融保險業', '金融保險業', '銀行業', 'Insurance', '半導體業'],
                  'stock_id': ['2881', '2881', '9999', '8888', '2330']}).to_parquet(p)
    out = audit.financial_sector_flag(p)
    assert out == [dict(industry_name='Insurance', stock_count=1),
                   dict(industry_name='金融保險業', stock_count=1),
                   dict(industry_name='銀行業', stock_count=1)]


def test_out_path_guard_traversal_symlinks_and_existing_file(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    local = root / '.local-data'
    local.mkdir()
    assert audit.validate_output_path(local / 'report.json', root) == local / 'report.json'
    for bad in [root / 'report.json', local / '..' / 'report.json', root / '.local-data-other' / 'r.json']:
        with pytest.raises(ValueError):
            audit.validate_output_path(bad, root)
    (local / 'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        audit.validate_output_path(local / 'escape' / 'report.json', root)
    existing = local / 'old.json'
    existing.write_text('keep')
    with pytest.raises(FileExistsError):
        audit.validate_output_path(existing, root)
    assert existing.read_text() == 'keep'
    local.rename(root / 'old-local')
    local.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        audit.validate_output_path(local / 'x.json', root)


def test_cli_one_strict_json_with_input_identity_and_no_input_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'REPO_ROOT', tmp_path)
    data = tmp_path / 'inputs'
    data.mkdir()
    d = frame([('2025-03-31', '2330', 'Revenue', 10, '營業收入')])
    for name in ['financials_raw', 'balance_sheet_raw', 'cashflow_raw']:
        d.to_parquet(data / (name + '.parquet'))
    pd.DataFrame({'stock_id': ['2881'], 'industry_category': ['金融保險業']}).to_parquet(data / 'stock_info.parquet')
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    out = tmp_path / '.local-data' / 'audit.json'
    assert audit.main(['--data-dir', str(data), '--out', str(out), '--as-of', '2025-05-16']) == 0
    report = json.loads(out.read_text())
    assert (report['evidence'], report['status']) == ('E1', 'RESEARCH_ONLY')
    assert set(report['tables']) == {'financials_raw', 'balance_sheet_raw', 'cashflow_raw'}
    assert len(report['inputs']) == 4
    assert all(len(x['sha256']) == 64 for x in report['inputs'].values())
    assert before == {p.name: p.read_bytes() for p in data.iterdir()}
    out2 = out.with_name('again.json')
    audit.main(['--data-dir', str(data), '--out', str(out2), '--as-of', '2025-05-16'])
    assert out.read_bytes() == out2.read_bytes()
    with pytest.raises(SystemExit):
        audit.main(['--data-dir', str(data / 'missing'), '--out', str(tmp_path / 'bad.json')])


def test_fully_missing_trailing_overdue_quarter_is_flagged():
    d = frame([('2026-06-30', '2330', 'Revenue', 10, '收入')])
    out = audit.period_coverage(d, as_of='2026-11-17')
    assert out[-1]['period_end'] == '2026-09-30'
    assert out[-1]['stock_count'] == 0
    assert out[-1]['overdue_insufficient'] is True


def test_financial_sector_continuing_operations_profit_alias():
    d = frame([('2025-03-31', '2881', 'IncomeFromContinuingOperations', -10,
                '繼續營業單位本期純益（純損）'),
               ('2025-03-31', '2881', 'EPS', 1, '每股盈餘')])
    assert audit.candidate_types(d)['net_income'][0]['type'] == 'IncomeFromContinuingOperations'
    pair = audit.duplicate_and_sign_checks(d)['eps_vs_net_income'][0]
    assert pair['anomaly_count'] == 1
