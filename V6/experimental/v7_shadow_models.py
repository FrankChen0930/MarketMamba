"""Read-only W34 model adoption. No fitting, acquisition, or publication entrypoint.

The scores function is preserved from W34@82cd49c; the small loader verifies
its original immutable JSON model/scaler/recipe and serving-schema bindings.
"""
import json,sys,platform
from pathlib import Path
import numpy as np
import lightgbm as lgb
import scipy,sklearn
from V6.experimental.v7_classical_baseline import ridge_predict
from V6.experimental.v7_corrected_e5_replication import canonical_hash as H,file_sha256 as sha
from V6.experimental.v7_research_artifacts import _safe,strict_json
REPO=Path(__file__).resolve().parents[2]
FEATURE_ORDER=['raw_close_momentum1','raw_intraday_range']
def require(ok,reason):
 if not ok:raise ValueError(reason)
def checked(ref):
 path=Path(ref['path']);path=_safe(path if path.is_absolute() else REPO/path)
 require(path.is_file() and sha(path)==ref['sha256'] and path.stat().st_size==ref.get('bytes',path.stat().st_size),'ARTIFACT_MISSING_OR_CHANGED')
 return strict_json(path.read_bytes())
def load_bundle(adoption,pin):
 require(H(adoption)==pin and adoption['source_commit']=='82cd49c25e585d857c05e9835af6c77f6ac9e860','ADOPTION_IDENTITY')
 p=checked(adoption['prepared']);run=checked(adoption['run']);e=checked(adoption['envelope']);env=p['environment']
 require((platform.python_version(),np.__version__,lgb.__version__,scipy.__version__,sklearn.__version__)==(env['python'],env['numpy'],env['lightgbm'],env['scipy'],env['scikit_learn']) and sha(Path(env['native_library']))==env['native_library_sha256'],'MODEL_ENVIRONMENT_CHANGED')
 require(p['feature_order']==e['feature_order']==FEATURE_ORDER and p['feature_formula_id']==e['feature_contract_id'] and p['dataset_id']==e['dataset_id'] and p['label_id']==e['label_id'] and run['models']==e['model_refs'],'SERVING_CONTRACT_MISMATCH')
 require(p['feature_formula_id']==H({'source':sha(REPO/'research/v7/window29-panel.py'),'order':FEATURE_ORDER}),'FEATURE_FORMULA_CHANGED')
 models={}
 for rid,ref in run['models'].items():
  m=checked(ref);require(m['spec']==p['model_specs'][rid] and m['spec']['initialization']=='FRESH_STATE' and m['spec']['resume_checkpoint'] is None and m['dataset_id']==p['dataset_id'] and m['feature_formula_id']==p['feature_formula_id'] and m['feature_order']==FEATURE_ORDER and m['label_id']==p['label_id'],'MODEL_DATA_FEATURE_RECIPE_IDENTITY')
  models[rid]=m
 require(set(models)==set(p['model_specs']) and len(models)==4,'MODEL_SET_INCOMPLETE');packet=checked(e['prediction_ref']);require(packet['model_refs']==run['models'] and packet['feature_formula_id']==p['feature_formula_id'],'ORIGINAL_PREDICTION_IDENTITY')
 return p,models,packet
def scores(p,models,fs):
 x=np.asarray([f['features'] for f in fs]);rows=[]
 for h in [5,10]:
  values={'L0_POS':x[:,0],'L0_NEG':-x[:,0]}
  for fam in ['Ridge','LightGBM']:
   m=models[f'W34-{fam}-H{h}-FRESH_REFIT'];values[fam]=ridge_predict(m['state'],x) if fam=='Ridge' else lgb.Booster(model_str=m['state']['booster_text']).predict(x,num_iteration=m['state']['actual_iterations'],num_threads=1)
  for fam,vs in values.items():
   assert np.isfinite(vs).all()
   for f,v in zip(fs,vs):rows.append({'stock_id':f['stock_id'],'signal_session':f['signal_session'],'horizon':h,'family':fam,'score':float(v),'model_id':f'W34-{fam}-H{h}-FRESH_REFIT' if fam in ['Ridge','LightGBM'] else 'W34-'+fam+'-FIXED_MOMENTUM'})
 return rows
