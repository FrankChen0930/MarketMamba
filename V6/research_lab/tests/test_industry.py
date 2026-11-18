import json
import numpy as np
from research_lab import industry as ind


def test_map_and_codes(tmp_path):
    (tmp_path/'twse.json').write_text(json.dumps([{'公司代號': '1101', '產業別': '01'}, {'公司代號': '2330', '產業別': '24'}]), encoding='utf-8')
    (tmp_path/'tpex.json').write_text(json.dumps([{'SecuritiesCompanyCode': '6488', 'SecuritiesIndustryCode': '24'}]), encoding='utf-8')
    m = ind.load_industry_map(tmp_path)
    assert m == {'1101': 'L01', '2330': 'L24', '6488': 'O24'}                 # same code on two boards stays distinct
    c = ind.industry_codes(np.array(['2330', '1101', '9999', '2330']), m)
    assert c[2] == -1 and c[0] == c[3] and c[0] != c[1]


def test_leave_one_out_mean_hand_case_and_session_isolation():
    v = np.array([[1.], [2.], [3.], [4.], [10.], [20.], [30.], [40.]])           # day 0: 4 stocks, day 1: 4 stocks, one industry
    d = np.array([0]*4+[1]*4); g = np.zeros(8, int)
    out = ind.relative_features(v, d, g, min_peers=3)
    assert out[0, 1] == (2+3+4)/3 and out[0, 0] == 1-(2+3+4)/3                   # mean of the OTHER three, same session only
    assert out[4, 1] == (20+30+40)/3                                              # day 1 never sees day 0


def test_unknown_industry_and_small_groups_are_nan_not_guessed():
    v = np.arange(1., 7.).reshape(-1, 1)
    d = np.zeros(6, int); g = np.array([0, 0, 0, 1, 1, -1])
    out = ind.relative_features(v, d, g, min_peers=3)
    assert np.isnan(out).all()          # group 0 has 2 peers per stock, group 1 has 1, unknown (-1) has none: all below min_peers=3 -> NaN


def test_panel_relative_matches_rowwise_version_and_uses_all_stocks():
    rng = np.random.default_rng(0)
    T, S = 6, 10
    p = rng.normal(size=(T, S)).astype(np.float32); p[2, 3] = np.nan
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, 1])
    rel, mean = ind.panel_relative(p, g, min_peers=3)
    t, s = np.meshgrid(np.arange(T), np.arange(S), indexing='ij')
    row = ind.relative_features(p.reshape(-1, 1), t.reshape(-1), np.tile(g, T), min_peers=3)
    assert np.allclose(mean.reshape(-1), row[:, 1], equal_nan=True) and np.allclose(rel.reshape(-1), row[:, 0], equal_nan=True)
    assert np.isnan(mean[:, 8]).all()
