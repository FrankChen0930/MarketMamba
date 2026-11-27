"""Step6 support gates do not infer historical sessions or read targets."""
import numpy as np
from tools.step6_features import FEATURES
from tools.step6_audit import support_gate, summarize_version, load_catalog


def test_support_requires_companies_keys_and_variation():
    stats = [dict(variable=True) for _ in FEATURES]
    assert support_gate(50, 1000, stats, True) == 'PASS'
    assert support_gate(49, 1000, stats, True) == 'STOP'
    assert support_gate(50, 999, stats, True) == 'STOP'
    assert support_gate(50, 1000, [dict(variable=False)]*3, True) == 'STOP'
    assert support_gate(50, 1000, stats, False) == 'STOP'


def test_snapshot_summary_does_not_multiply_by_history():
    edges = [dict(src='tw.S', dst='tw.'+c, type='supplier', evidence_level='primary') for c in ['A','B']]
    revenue = {c: dict(yoy=.1, surprise=.2, period='2026-09', age_sessions=1, age_months=2) for c in ['A','B']}
    summary, rows, _ = summarize_version(edges, {'S','A','B'}, revenue)
    assert summary['effective_suppliers'] == 1
    assert summary['effective_company_session_keys'] == 1
    assert summary['status'] == 'STOP'
    assert summary['historical_folds'] == 'NOT_RUN'
    assert all(not f['variable'] for f in summary['features'])
    assert summary['features'][0]['finite_ratio'] == 1


def test_catalog_validates_observation_blob_and_ancestry(tmp_path):
    import subprocess, json, hashlib
    def git(*args):
        return subprocess.check_output(['git','-C',str(tmp_path),*args],text=True).strip()
    git('init','-q'); git('config','user.email','test@example.invalid'); git('config','user.name','test')
    rel = tmp_path/'v1/tw/relations/edges.json'; rel.parent.mkdir(parents=True); rel.write_text('[]')
    git('add','v1'); git('commit','-qm','one'); head = git('rev-parse','HEAD')
    obs = tmp_path/'observations.jsonl'
    record = dict(source='shioaji-data',head=head,head_committed_at=git('show','-s','--format=%cI',head),
                  observed_at='2026-11-28T00:00Z',files={'v1/tw/relations/edges.json':hashlib.sha256(b'[]').hexdigest()})
    obs.write_text(json.dumps(record)+'\n')
    versions, observations = load_catalog(tmp_path, obs)
    assert observations[0]['valid'] and head in observations[0]['ancestors']
    assert len(versions) == 2 and versions[-1]['latest']
    record['files']['v1/tw/relations/edges.json'] = '0'*64
    obs.write_text(json.dumps(record)+'\n')
    _, observations = load_catalog(tmp_path, obs)
    assert not observations[0]['valid']
