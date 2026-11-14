"""Read-only reason decomposition and same-host byte restore; no model or metric calls."""
import runpy,json,hashlib
from pathlib import Path
from collections import Counter
M=runpy.run_path(str(Path(__file__).with_name('window27-safety.py')));ROOT=M['ROOT'];OUT=M['OUT']
_,_,data,days,raw=M['source_views']();frozen=json.loads((ROOT/'FROZEN_MASK.json').read_text());mask=M['checked'](frozen['mask']);mi={tuple(r['key']):r for r in mask};details={}
for r in data:
 k=(r['signal_session'],r['stock_id'],r['horizon']);q=mi[k];g=details.setdefault(r['split']+'_'+str(r['horizon']),Counter());i=days.index(r['signal_session']);path=[raw[r['stock_id']].get(d) for d in days[i+1:i+r['horizon']+1]]
 zero=any(p is not None and any(p.get(f)==0 for f in ('open','close','max','min')) for p in path)
 other_invalid=any(p is not None and not M['valid_ohlc'](p) and not any(p.get(f)==0 for f in ('open','close','max','min')) for p in path)
 g['path_zero_any']+=zero;g['path_other_invalid_any']+=other_invalid
 if q['primary']=='HOLDING_PATH_INVALID_OR_ZERO':g['primary_path_zero']+=zero;g['primary_path_other_invalid']+=other_invalid
 assert not q['eligible'] or not (zero or other_invalid)
M['immutable'](OUT/'window27-price-reasons.json',details)
# Restore completed private outputs to a separate local directory. Not independent media.
target=ROOT.parent/'window27-restore';files=[]
for p in sorted(ROOT.rglob('*')):
 if not p.is_file():continue
 relative=p.relative_to(ROOT);q=target/relative;q.parent.mkdir(parents=True,exist_ok=True);rawbytes=p.read_bytes();sha=hashlib.sha256(rawbytes).hexdigest()
 if q.exists():assert q.read_bytes()==rawbytes
 else:
  with q.open('xb') as f:f.write(rawbytes)
 assert q.stat().st_size==len(rawbytes) and hashlib.sha256(q.read_bytes()).hexdigest()==sha
 files.append({'path':str(p),'relative':str(relative),'bytes':len(rawbytes),'sha256':sha})
proof={'status':'LOCAL_BYTES_RESTORED','files':files,'file_count':len(files),'bytes':sum(r['bytes'] for r in files),'source':str(ROOT),'restore':str(target),'backup':'BACKUP_PENDING_SAME_HOST_NOT_INDEPENDENT_BACKUP','private_data_not_committed':True,'prior_pins':M['input_pins']()}
M['immutable'](OUT/'window27-local-artifacts.json',proof)
print(json.dumps({'files':len(files),'bytes':proof['bytes'],'status':proof['status'],'prior_pins':proof['prior_pins']}))
