"""Offline diagnostic of pinned provider functions, not a production certifier.

Execute original AST function bodies only: importing fetcher would create cache
paths and require operational config. HTTP/HTML transport is fixture-supplied.
Run with recovery venv Python from repository root; stdout is evidence JSON.
"""
import ast
import hashlib
import json
import logging
from pathlib import Path
import re
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd

SOURCE = Path(__file__).resolve().parents[2] / 'V6/marketmamba/data/fetcher.py'
NAMES = {'fetch_prices_twse_direct', 'fetch_prices_tpex_direct',
         'fetch_revenue_mops_direct', '_mops_to_number',
         '_mops_norm_col', '_mops_pick_company_tables'}

class Response:
    status_code = 200
    text = 'fixture HTML transport'
    def __init__(self, payload=None): self.payload = payload
    def json(self): return self.payload
    def raise_for_status(self): pass


def run():
    raw = SOURCE.read_bytes()
    module = ast.parse(raw)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    assert {node.name for node in functions} == NAMES
    ns = {'pd': pd, 're': re, 'StringIO': StringIO, 'logger': logging.getLogger('provider-probe'),
          'HEADERS': {}, 'MOPS_HOST': 'https://fixture.invalid',
          'time': SimpleNamespace(sleep=lambda _: None)}
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *functions]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), 'exec'), ns)
    calls = []
    def get_response(payload):
        def get(url, **kwargs):
            calls.append({'url': url, 'params': kwargs.get('params')})
            return Response(payload)
        ns['requests'] = SimpleNamespace(get=get)
    results = []
    fields = ['證券代號', '開盤價', '最高價', '最低價', '收盤價', '成交股數']
    rows = [[str(1100+i), '10', '12', '9', '11', '1000'] for i in range(101)]
    for returned in ('20260716', '20260717'):
        get_response({'stat': 'OK', 'date': returned, 'tables': [{'fields': fields, 'data': rows}]})
        frame = ns['fetch_prices_twse_direct']('2026-07-16')
        results.append({'case': 'TWSE_date_'+returned, 'returned_date': returned,
                        'rows': 0 if frame is None else len(frame),
                        'assigned_dates': [] if frame is None else sorted(frame.Date.unique().tolist())})
    tfields = ['代號','名稱','收盤','漲跌','開盤','最高','最低','成交股數']
    trow = ['1101','fixture','11','1','10','12','9','1000']
    for reordered in (False, True):
        fs, row = tfields[:], trow[:]
        if reordered:
            fs[2],fs[4]=fs[4],fs[2];row[2],row[4]=row[4],row[2]
        get_response({'stat':'ok','date':'20260716','tables':[{'fields':fs,'data':[row]}]})
        frame=ns['fetch_prices_tpex_direct']('2026-07-16')
        results.append({'case':'TPEx_reordered_'+str(reordered),'rows':len(frame),
                        'Open':float(frame.Open.iloc[0]),'Close':float(frame.Close.iloc[0])})
    get_response({'stat':'ok','date':'20260717','tables':[{'fields':tfields,'data':[trow]}]})
    results.append({'case':'TPEx_wrong_date_rejected','rejected':ns['fetch_prices_tpex_direct']('2026-07-16') is None})
    for missing_otc in (False,True):
        def get(url, **kwargs):
            calls.append({'url':url,'params':kwargs.get('params')})
            if missing_otc and '/otc/' in url: raise OSError('synthetic OTC transport failure')
            response=Response();response.text='otc' if '/otc/' in url else 'sii';return response
        ns['requests']=SimpleNamespace(get=get)
        def tables(stream):
            stock='2201' if stream.getvalue()=='otc' else '1101'
            return [pd.DataFrame({'公司代號':[stock], '營業收入|當月營收':['100']})]
        with patch.object(pd,'read_html',side_effect=tables):
            frame=ns['fetch_revenue_mops_direct'](2026,5)
        results.append({'case':'MOPS_missing_otc_'+str(missing_otc),'rows':len(frame),
                        'stocks':frame.stock_id.tolist(),'dates':frame.Date.astype(str).tolist(),
                        'values':frame.revenue.tolist(),'publication_time':frame.create_time.tolist()})
    # Diagnostic oracle guards fixture construction; these are observed defects,
    # not assertions that current provider behavior is acceptable.
    assert results[0]['rows']==results[1]['rows']==101
    assert results[1]['assigned_dates']==['2026-07-16']
    assert (results[2]['Open'],results[2]['Close'])==(10.0,11.0)
    assert (results[3]['Open'],results[3]['Close'])==(11.0,10.0)
    assert results[4]['rejected'] is True
    assert results[5]['rows']==2 and results[6]['rows']==1
    assert results[6]['dates']==['2026-06-01'] and results[6]['publication_time']==['']
    return {'evidence_class':'OFFLINE_DIAGNOSTIC_NOT_CERTIFICATION','source':'V6/marketmamba/data/fetcher.py',
            'source_sha256':hashlib.sha256(raw).hexdigest(),'function_names':sorted(NAMES),
            'network_requests':0,'fixture_transport_calls':len(calls),'results':results,
            'limitations':'AST exact function bodies; transport/HTML parse fixtures; no whole-module or external provider certification, no production data loaded/written'}

if __name__=='__main__': print(json.dumps(run(),ensure_ascii=False,indent=2))
