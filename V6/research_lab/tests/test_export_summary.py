import copy
import json

import pytest

from research_lab.export_summary import MODELS, build_summary, canonical_hash, validate_summary


@pytest.fixture
def sources(tmp_path):
    rank = {'evidence_class': 'RESEARCH_ONLY', 'dataset_sha256': 'a'*64,
            'manifest_sha256': 'b'*64, 'git_head': 'c'*40, 'config': {},
            'evaluation_policy': 'full session grid', 'models': {}}
    directory = tmp_path / 'portfolios'
    directory.mkdir()
    for model in MODELS:
        for h in (5, 20):
            rank['models'][f'{model}_h{h}'] = {'summary': {'ic_mean': -.01, 'icir': -.1,
                'nw_t': -.7, 'ic_ci95': [-.04, .02], 'days': 10, 'sessions': 12},
                'prediction_sha256': 'd'*64}
        portfolio = {'model': model, 'horizon': 20, 'evidence_class': 'RESEARCH_ONLY',
            'evidence_tier': {'tier': 'RESEARCH_ONLY'}, 'config': {'top_k': 20, 'rebalance_sessions': 20},
            'start': '2020-01-01', 'end': '2021-01-01',
            'statistics': {'total_return': .2, 'annualized_return': .2, 'benchmark_return': .3,
                          'sharpe_like': None, 'information_ratio': -.1},
            'metrics': {'max_drawdown': {'fraction': '.1'}, 'mean_turnover': '.02',
                        'cost_drag': '.03', 'benchmark': {'excess_return': '-.1'}},
            'years': {'2020': {'total_return': .2, 'benchmark_return': .3}},
            'provenance': {'inputs': {'prices': {'sha256': 'a'*64},
                'dataset_manifest_sha256': 'b'*64, 'predictions': [{'sha256': 'd'*64}]},
                'source_commit': 'c'*40}}
        portfolio['canonical_sha256'] = canonical_hash(portfolio, 'canonical_sha256')
        (directory / f'{model}_h20_k20_r20.json').write_text(json.dumps(portfolio))
    path = tmp_path / 'rank.json'
    path.write_text(json.dumps(rank))
    return path, directory


def test_deterministic_schema_hash_and_projection(sources):
    first = build_summary(*sources)
    assert first == build_summary(*sources)
    assert len(first['models']) == 14 and len(first['portfolios']) == 7
    assert validate_summary(first) == first
    assert first['sha256'] == canonical_hash(first)
    assert first['evidence_tier']['tier'] == 'RESEARCH_ONLY'
    assert len(first['caveats']) >= 7
    assert first['portfolios'][0]['sharpe_like'] is None
    assert first['portfolios'][0]['max_drawdown'] == .1
    assert 'prediction_path' not in json.dumps(first)


def test_missing_input_fails(sources):
    path, directory = sources
    path.unlink()
    with pytest.raises(FileNotFoundError):
        build_summary(path, directory)


def test_missing_portfolio_fails(sources):
    path, directory = sources
    next(directory.iterdir()).unlink()
    with pytest.raises(FileNotFoundError):
        build_summary(path, directory)


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(schema='unknown'),
    lambda d: d['models'][0].update(ic_mean=99),
    lambda d: d.update(caveats=[]),
    lambda d: d['portfolios'][0].update(evidence_tier={'tier': 'VERIFIED'}),
])
def test_invalid_summary_rejected(sources, mutate):
    doc = copy.deepcopy(build_summary(*sources))
    mutate(doc)
    with pytest.raises(ValueError):
        validate_summary(doc)


def test_input_hash_and_dataset_mismatch_rejected(sources):
    path, directory = sources
    target = next(directory.iterdir())
    doc = json.loads(target.read_text())
    doc['statistics']['total_return'] = 900
    target.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        build_summary(path, directory)
    doc['canonical_sha256'] = canonical_hash(doc, 'canonical_sha256')
    doc['provenance']['inputs']['prices']['sha256'] = 'e'*64
    doc['canonical_sha256'] = canonical_hash(doc, 'canonical_sha256')
    target.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        build_summary(path, directory)


def test_cli_writes_identical_bytes(sources, tmp_path, monkeypatch):
    from research_lab.export_summary import main
    path, directory = sources
    out = tmp_path / 'summary.json'
    monkeypatch.setattr('sys.argv', ['export_summary', '--evaluation', str(path),
                                   '--portfolios', str(directory), '--out', str(out)])
    main()
    first = out.read_bytes()
    main()
    assert out.read_bytes() == first
    assert b'\n' not in first.rstrip(b'\n')
    assert validate_summary(json.loads(first))['sha256'] == canonical_hash(json.loads(first))
