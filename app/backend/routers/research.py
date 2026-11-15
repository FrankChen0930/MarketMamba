"""Read-only offline research snapshot; never accepts client filesystem paths."""
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from V6.research_lab.export_summary import CAVEATS, SCHEMA, TIER, validate_summary

router = APIRouter(prefix='/research', tags=['research'])
NO_STORE = {'Cache-Control': 'no-store'}
SUMMARY_PATH = Path(__file__).resolve().parents[3] / 'V6/research_lab/experiments/summary.json'


def _read(request, section=None):
    if request.query_params:
        return JSONResponse(status_code=400, headers=NO_STORE, content=dict(schema=SCHEMA, state='not_ready',
            evidence_tier=TIER, caveats=CAVEATS, message='此端點不接受查詢參數或檔案路徑。'))
    try:
        doc = validate_summary(json.loads(SUMMARY_PATH.read_text(encoding='utf-8')))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return JSONResponse(status_code=503, headers=NO_STORE, content=dict(schema=SCHEMA, state='not_ready',
            evidence_tier=TIER, caveats=CAVEATS, message='研究摘要缺少或無效；未沿用舊資料。'))
    if section is not None:
        doc = {key: doc[key] for key in ('schema', 'state', 'status', 'sha256', 'identity',
                                        'evidence_tier', 'caveats', section)}
    return JSONResponse(content=doc, headers=NO_STORE)


@router.get('/summary')
async def summary(request: Request):
    return _read(request)


@router.get('/models')
async def models(request: Request):
    return _read(request, 'models')


@router.get('/portfolios')
async def portfolios(request: Request):
    return _read(request, 'portfolios')
