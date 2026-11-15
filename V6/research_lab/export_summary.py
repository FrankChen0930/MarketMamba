"""Offline, deterministic research publication. Never reads data on API requests.

sha256 covers canonical ASCII JSON excluding only the top-level sha256 field.
Input hashes identify exact bytes; they establish integrity, not trading validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from V6.experimental.v7_evidence_tier import classify_evidence_tier

SCHEMA = 'research-summary-v1'
MODELS = ('gru', 'lightgbm', 'lowvol', 'momentum20', 'reversal5', 'ridge', 'selective')
TIER = classify_evidence_tier(evidence_class='RESEARCH_ONLY')
CAVEATS = [
    '150-name universe is survivorship-biased (includes ETFs); no historical membership proof.',
    'Yahoo prices: vintage, corporate actions and executable prices are not verified.',
    'No fundamentals or verified point-in-time fundamentals evidence.',
    'Equal-weight benchmark is inflated by survivorship; it is not a market index.',
    'Momentum IC is negative while portfolio return is high: selection artifact to be investigated.',
    'No multiple-comparison correction; comparisons are exploratory.',
    'SSM is not shown superior; these results are not verified trading evidence.',
    'Rank labels exit at t+h close; portfolios exit at next rebalance open. These are different targets.',
    'Costs use research assumptions; cost drag is cumulative cost / initial NAV, not gross-minus-net return.',
    '2019 and 2026 are partial years; horizons/models can have different valid-label support.',
]
RANK_NUMBERS = ('ic_mean', 'icir', 'nw_t')
PORT_NUMBERS = ('total_return', 'annualized_return', 'benchmark_return', 'excess_return',
                'sharpe_like', 'max_drawdown', 'mean_turnover', 'cost_drag', 'information_ratio')


def canonical_hash(document, key='sha256'):
    body = {k: v for k, v in document.items() if k != key}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _load(path):
    raw = Path(path).read_bytes()
    document = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(document, dict):
        raise ValueError('expected JSON object')
    return document, hashlib.sha256(raw).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid SHA-256 identity')
    return value


def _number(value, *, decimal=False):
    if value is None:
        return None  # Explicitly undefined metrics stay unknown, never zero.
    if decimal and isinstance(value, str):
        value = float(value)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('metric must be finite or null')
    return value


def _tier(row):
    if row.get('evidence_tier', {}).get('tier') != 'RESEARCH_ONLY':
        raise ValueError('research tier required')


def validate_summary(doc):
    """Validate a complete snapshot before exposing any numbers; no stale fallback."""
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA or doc.get('state') != 'ready':
        raise ValueError('unsupported summary')
    if doc.get('sha256') != canonical_hash(doc):
        raise ValueError('summary hash mismatch')
    _tier(doc)
    if doc.get('status') != 'UNMERGED / NOT_DEPLOYED':
        raise ValueError('missing branch-local status')
    if not isinstance(doc.get('evaluation_policy'), str) or not doc['evaluation_policy']:
        raise ValueError('invalid evaluation policy')
    if not isinstance(doc.get('ranking_config'), dict):
        raise ValueError('invalid ranking config')
    if set(doc['identity']) != {'source_commit', 'dataset_sha256', 'manifest_sha256', 'evaluation_sha256'}:
        raise ValueError('unexpected identity fields')
    commit = doc['identity']['source_commit']
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('invalid source commit')
    if doc.get('caveats') != CAVEATS:
        raise ValueError('mandatory caveats missing')
    for key in ('dataset_sha256', 'manifest_sha256', 'evaluation_sha256'):
        _hash(doc['identity'][key])
    models, portfolios = doc['models'], doc['portfolios']
    if not isinstance(models, list) or not isinstance(portfolios, list):
        raise ValueError('invalid comparison arrays')
    keys = set()
    for row in models:
        _tier(row)
        model, horizon = row['model'], row['horizon']
        if not isinstance(model, str) or not re.fullmatch('[a-z0-9_]+', model) or type(horizon) is not int or horizon not in (5, 20):
            raise ValueError('invalid model identity')
        if (model, horizon) in keys:
            raise ValueError('duplicate model')
        keys.add((model, horizon))
        for key in RANK_NUMBERS:
            _number(row[key])
        ci = row['ic_ci95']
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError('invalid CI')
        for value in ci:
            _number(value)
        if all(v is not None for v in ci) and ci[0] > ci[1]:
            raise ValueError('reversed CI')
        for key in ('days', 'sessions'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('invalid support count')
        _hash(row['prediction_sha256'])
    if not {(model, h) for model in MODELS for h in (5, 20)} <= keys:
        raise ValueError('incomplete model comparison')
    if len(portfolios) != len(MODELS) or {p['model'] for p in portfolios} != set(MODELS):
        raise ValueError('incomplete portfolio comparison')
    for row in portfolios:
        _tier(row)
        if row['horizon'] != 20 or row['config']['top_k'] != 20 or row['config']['rebalance_sessions'] != 20:
            raise ValueError('incompatible portfolio config')
        for key in PORT_NUMBERS:
            _number(row[key])
        _hash(row['input_sha256'])
        if not isinstance(row['start'], str) or not isinstance(row['end'], str):
            raise ValueError('missing portfolio window')
        if not isinstance(row['years'], dict) or not row['years']:
            raise ValueError('missing annual comparison')
        for year, values in row['years'].items():
            if not re.fullmatch(r'\d{4}', year):
                raise ValueError('invalid year')
            _number(values['total_return'])
            _number(values['benchmark_return'])
    return doc


def build_summary(evaluation_path, portfolio_dir):
    rank, rank_hash = _load(evaluation_path)
    if rank['evidence_class'] != 'RESEARCH_ONLY':
        raise ValueError('unexpected evaluation evidence')
    models = []
    for name, result in sorted(rank['models'].items()):
        model, h = name.rsplit('_h', 1)
        if int(h) not in (5, 20):
            continue
        stats = result['summary']
        models.append(dict(model=model, horizon=int(h), evidence_tier=TIER,
            **{key: stats[key] for key in (*RANK_NUMBERS, 'ic_ci95', 'days', 'sessions')},
            prediction_sha256=result['prediction_sha256']))
    portfolios = []
    for model in MODELS:
        source, source_hash = _load(Path(portfolio_dir) / f'{model}_h20_k20_r20.json')
        if source['canonical_sha256'] != canonical_hash(source, 'canonical_sha256'):
            raise ValueError('portfolio input hash mismatch')
        _tier(source)
        if source['model'] != model or source['evidence_class'] != 'RESEARCH_ONLY':
            raise ValueError('portfolio identity mismatch')
        inputs = source['provenance']['inputs']
        if inputs['prices']['sha256'] != rank['dataset_sha256'] or inputs['dataset_manifest_sha256'] != rank['manifest_sha256']:
            raise ValueError('dataset/manifest mismatch')
        if [p['sha256'] for p in inputs['predictions']] != [rank['models'][f'{model}_h20']['prediction_sha256']]:
            raise ValueError('prediction identity mismatch')
        stats, metrics = source['statistics'], source['metrics']
        portfolios.append(dict(model=model, horizon=source['horizon'], config=source['config'],
            start=source['start'], end=source['end'], evidence_tier=TIER,
            **{key: _number(stats[key]) for key in ('total_return', 'annualized_return', 'benchmark_return', 'sharpe_like', 'information_ratio')},
            excess_return=_number(metrics['benchmark']['excess_return'], decimal=True),
            max_drawdown=_number(metrics['max_drawdown']['fraction'], decimal=True),
            mean_turnover=_number(metrics['mean_turnover'], decimal=True),
            cost_drag=_number(metrics['cost_drag'], decimal=True),
            years={year: {k: values[k] for k in ('total_return', 'benchmark_return')}
                   for year, values in sorted(source['years'].items())},
            input_sha256=source_hash, source_commit=source['provenance']['source_commit']))
    doc = dict(schema=SCHEMA, state='ready', status='UNMERGED / NOT_DEPLOYED',
        evidence_tier=TIER, caveats=CAVEATS,
        identity=dict(dataset_sha256=rank['dataset_sha256'], manifest_sha256=rank['manifest_sha256'],
                      evaluation_sha256=rank_hash, source_commit=rank['git_head']),
        evaluation_policy=rank['evaluation_policy'], ranking_config=rank['config'],
        models=models, portfolios=portfolios)
    doc['sha256'] = canonical_hash(doc)
    return validate_summary(doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--portfolios', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'experiments/summary.json')
    args = parser.parse_args()
    document = build_summary(args.evaluation, args.portfolios)
    # No clock, filesystem paths or process-specific values enter the publication.
    args.out.write_text(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'sha256': document['sha256'], 'models': len(document['models']),
                      'portfolios': len(document['portfolios'])}))


if __name__ == '__main__':
    main()
