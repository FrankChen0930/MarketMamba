import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.backend.routers import research
from research_lab.export_summary import canonical_hash


@pytest.fixture
def publication(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[2] / 'V6/research_lab/experiments/summary.json'
    path = tmp_path / 'summary.json'
    path.write_bytes(original.read_bytes())
    monkeypatch.setattr(research, 'SUMMARY_PATH', path)
    return path


def get(url):
    async def request():
        app = FastAPI()
        app.include_router(research.router, prefix='/api')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
            return await client.get(url)
    return asyncio.run(request())


@pytest.mark.parametrize('endpoint,key,count', [('summary','models',26), ('models','models',26), ('portfolios','portfolios',7)])
def test_ready(publication, endpoint, key, count):
    response = get('/api/research/' + endpoint)
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    doc = response.json()
    assert len(doc[key]) == count
    assert doc['evidence_tier']['tier'] == 'RESEARCH_ONLY'
    assert len(doc['caveats']) >= 7
    assert all(row['evidence_tier']['tier'] == 'RESEARCH_ONLY' for row in doc[key])


@pytest.mark.parametrize('damage', ['missing', 'malformed', 'hash', 'schema', 'number', 'tier', 'caveats', 'status', 'policy', 'identity', 'extra_identity'])
def test_not_ready_no_stale_fallback(publication, damage):
    assert get('/api/research/summary').status_code == 200
    doc = json.loads(publication.read_text())
    if damage == 'missing':
        publication.unlink()
    elif damage == 'malformed':
        publication.write_text('{')
    else:
        if damage == 'hash': doc['models'][0]['ic_mean'] = 99
        if damage == 'schema': doc['schema'] = 'unknown'
        if damage == 'number': doc['portfolios'][0]['total_return'] = '900'
        if damage == 'tier': doc['evidence_tier']['tier'] = 'VERIFIED'
        if damage == 'caveats': doc['caveats'] = []
        if damage == 'status': del doc['status']
        if damage == 'policy': doc['evaluation_policy'] = {'bad': 'object'}
        if damage == 'identity': doc['identity']['source_commit'] = {'bad': 'object'}
        if damage == 'extra_identity': doc['identity']['unexpected'] = {'bad': 'object'}
        if damage != 'hash': doc['sha256'] = canonical_hash(doc)
        publication.write_text(json.dumps(doc))
    for endpoint in ('summary','models','portfolios'):
        response = get('/api/research/' + endpoint)
        assert response.status_code == 503
        assert response.headers['cache-control'] == 'no-store'
        body = response.json()
        assert body['state'] == 'not_ready'
        assert body['evidence_tier']['tier'] == 'RESEARCH_ONLY' and body['caveats']
        assert 'models' not in body and 'portfolios' not in body


def test_client_paths_rejected(publication):
    assert get('/api/research/summary?path=../../etc/passwd').status_code == 400
    assert get('/api/research/summary?file=/etc/passwd').status_code == 400
    assert get('/api/research/..%2F..%2Fetc%2Fpasswd').status_code == 404
