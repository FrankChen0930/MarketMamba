"""Archived W26 document acquisition; network budget is permanently closed. Do not rerun."""
import sys,json,urllib.request,urllib.error,datetime
from pathlib import Path
sys.path.insert(0,'/home/frank/projects/MarketMamba/.worktrees/v7-window26-label-contract')
from V6.experimental.v7_source_snapshot import RequestSpec,snapshot_response,verify_snapshot
from V6.experimental.v7_manual_collector import NoRedirect
root=Path('/home/frank/projects/MarketMamba/.artifacts/window26-label-integrity/documents');root.mkdir(parents=True,exist_ok=True)
if json.loads(Path(__file__).with_name('window26-network-ledger.json').read_text())['official_documents']['CLOSED']:
 raise SystemExit('W26_DOCUMENT_BUDGET_CLOSED_NO_NETWORK')
name,url=sys.argv[1:];claim=root/(name+'.claim.json');assert not claim.exists();claim.write_text(json.dumps({'url':url,'claimed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'W26boundedprimarydocument,notquotationAPI'}))
op=urllib.request.build_opener(NoRedirect(),urllib.request.ProxyHandler({}))
try:r=op.open(urllib.request.Request(url,headers={'User-Agent':'MarketMamba-W26-Label-Evidence'}),timeout=30)
except urllib.error.HTTPError as e:r=e
with r:b=r.read(8000001);code=r.code;headers={k.lower():v for k,v in r.headers.items()}
at=datetime.datetime.now(datetime.timezone.utc).isoformat();mp=snapshot_response(RequestSpec('GET',url),b,root/'snapshots',retrieved_at=at,parser_version='w26-source-doc-v1',content_type=headers.get('content-type','unknown'),status_code=code);assert verify_snapshot(mp)
r={'url':url,'http_status':code,'retrieved_at':at,'bytes':len(b),'headers':{k:v for k,v in headers.items() if k in ['content-type','content-length','date']},'manifest':str(mp),'raw':str(root/'snapshots'/json.load(open(mp))['response']['path']),'new_document_request':True};(root/(name+'.record.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(name,code,len(b))
