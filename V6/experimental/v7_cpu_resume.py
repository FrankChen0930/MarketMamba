"""Bounded real CPU resume verification through E1's existing training loop/store.

Not another trainer. Toy linear+dropout factory, map-style deterministic daily cursor,
zero workers, fp32, single CPU thread. No GPU/AMP/shuffled-loader guarantee.
Only this run's hash-verified payloads are read, strictly weights_only=True.
"""
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from contextlib import contextmanager
import argparse,fcntl,hashlib,json,math,os,random,uuid
import numpy as np
import torch
from torch import nn
from V6.experimental.v7_corrected_e5_replication import CorrectedE5Config,canonical_hash
from V6.experimental.v7_corrected_e5_train import train_seed,lr_factor,adamw_parameter_groups
from V6.experimental.v7_e1_checkpoint import DurableCheckpointStore,LifecycleCheckpointAdapter,sha256,atomic_json,validate_state
from V6.experimental.v7_e1_run import append_event
from V6.experimental.v7_corrected_e5_lifecycle import fresh_lifecycle_state,update_selection
from V6.experimental.v7_research_artifacts import _safe,strict_json

ROOT=Path(__file__).resolve().parents[2]
MODEL_SPEC={'kind':'TOY_LINEAR_DROPOUT','features':2,'heads':2,'dropout':.25,'device':'cpu','dtype':'float32'}


def config():
    base=CorrectedE5Config.from_contract(ROOT/'research/v7/corrected-e5-replication-v1/incumbent-contract.json',ROOT/'research/v7/corrected-baseline-v1/feature-manifest.json')
    return replace(base,feature_order=('toy_f0','toy_f1'),group_dims=(2,0,0,0),sequence_length=2,epochs=4,minimum_epochs=3,patience=2,
      learning_rate=.01,weight_decay=.01,warmup_fraction=.25,precision='float32',evidence_class='SYNTHETIC_CPU_ENGINEERING',proxy_policy='NO_FINANCIAL_EVIDENCE',
      train_start='2024-01-02',train_end='2024-01-04',evaluation_start='2024-01-05',evaluation_end='2024-01-08',embargo_trading_days=0)


def make_fixture(root):
    root=_safe(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():return make_binding(root)
    dates=np.repeat(np.array(['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-08']),4)
    n=np.arange(20,dtype=np.float32);x=np.stack((np.sin(n*.4),np.cos(n*.7)),axis=1).astype('float32')
    values={'X':x,'y5':(x[:,0]*.7+x[:,1]*.2).astype('float32'),'y10':(x[:,0]*.3-x[:,1]*.4).astype('float32'),
     'stock_ids':np.tile(np.array(['toy0','toy1','toy2','toy3']),5),'dates':dates,'splits':np.repeat(np.array([1,1,1,2,2]),4),'masks':np.ones_like(x,dtype=bool)}
    for name,array in values.items():np.save(root/(name+'.npy'),array,allow_pickle=False)
    files={name+'.npy':sha256(root/(name+'.npy')) for name in values};identity=canonical_hash(files)
    atomic_json(root/'manifest.json',{'schema':'window11-cpu-fixture-v1','evidence_class':'SYNTHETIC','logical_identity':identity,'files':files,'train_days':['2024-01-02','2024-01-03','2024-01-04'],'config_sha256':config().sha256})
    return make_binding(root)


def make_binding(matrix,run_id='cpu-proof'):
    matrix=_safe(matrix);doc=strict_json((matrix/'manifest.json').read_bytes())
    if doc.get('schema')!='window11-cpu-fixture-v1' or doc.get('evidence_class')!='SYNTHETIC' or doc['config_sha256']!=config().sha256:raise ValueError('SYNTHETIC_DATA_CONFIG_MISMATCH')
    required={name+'.npy' for name in ('X','y5','y10','stock_ids','dates','splits','masks')}
    if set(doc['files'])!=required or canonical_hash(doc['files'])!=doc['logical_identity']:raise ValueError('DATA_MANIFEST_MISMATCH')
    for name,h in doc['files'].items():
        if sha256(_safe(matrix/name))!=h:raise ValueError('DATA_BYTES_MISMATCH')
    return {'schema':'window11-cpu-run-binding-v1','run_id':run_id,'config_sha256':config().sha256,'data_sha256':doc['logical_identity'],
            'model_sha256':canonical_hash(MODEL_SPEC),'seed':17,'sampler_days':doc['train_days'],'sampler_policy':'DETERMINISTIC_DAILY_ORDER_NO_WORKERS',
            'cpu_policy':{'threads':1,'deterministic_algorithms':True,'amp':False,'precision':'float32'}}


class TinyCPUModel(nn.Module):
    def __init__(self,_):
        super().__init__();self.dropout=nn.Dropout(.25);self.linear=nn.Linear(2,2)
    def forward(self,x,stock_ids,observation_mask):return self.linear(self.dropout(x[:,-1,:]))


def _safe_values(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_safe_values(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_safe_values(v) for v in value)
    if isinstance(value,list):return [_safe_values(v) for v in value]
    return value


def _sampler(payload,binding):
    return {'policy':binding['sampler_policy'],'days_sha256':canonical_hash(binding['sampler_days']),'epoch':payload['epoch'],
       'next_batch':payload['batch'] if payload['phase']=='train' else len(binding['sampler_days']),
       'validation_index':payload.get('validation_index',0)}


def _validate_optimizer(payload,cfg,expected_lr):
    # Same factory/groups as the actual core. Meta tensors avoid consuming CPU RNG.
    with torch.device('meta'):model=TinyCPUModel(None)
    optimizer=torch.optim.AdamW(adamw_parameter_groups(model,cfg.weight_decay),lr=cfg.learning_rate)
    expected=optimizer.state_dict();state=payload['optimizer_state'];shapes={}
    if not isinstance(state,Mapping) or set(state)!={'state','param_groups'} or len(state['param_groups'])!=len(expected['param_groups']):raise ValueError('INVALID_OPTIMIZER_TOPOLOGY')
    for group,template,parameters in zip(state['param_groups'],expected['param_groups'],optimizer.param_groups):
        template={**template,'lr':expected_lr,'initial_lr':cfg.learning_rate}
        if group!=template:raise ValueError('INVALID_OPTIMIZER_HYPERPARAMETERS_OR_TOPOLOGY')
        for index,parameter in zip(template['params'],parameters['params']):shapes[index]=(tuple(parameter.shape),parameter.dtype)
    if not isinstance(state['state'],Mapping) or set(state['state'])!=set(shapes):raise ValueError('INCOMPLETE_OPTIMIZER_MOMENTS')
    for index,value in state['state'].items():
        if not isinstance(value,Mapping) or set(value)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('INCOMPLETE_OPTIMIZER_MOMENTS')
        step=value['step']
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or not torch.isfinite(step).all() or float(step.item())!=payload['step']:raise ValueError('OPTIMIZER_STEP_MISMATCH')
        for key in ('exp_avg','exp_avg_sq'):
            moment=value[key];shape,dtype=shapes[index]
            if not isinstance(moment,torch.Tensor) or tuple(moment.shape)!=shape or moment.dtype!=dtype or moment.device.type!='cpu' or not torch.isfinite(moment).all():raise ValueError('INVALID_OPTIMIZER_MOMENTS')
        if (value['exp_avg_sq']<0).any():raise ValueError('INVALID_OPTIMIZER_VARIANCE')


class ProofStore(DurableCheckpointStore):
    """E1 storage protocol, safer probe codec, strict identities, no generation GC."""
    def __init__(self,root,staging,binding,cut_step=None,fault=None):
        super().__init__(root,staging,binding['config_sha256'],binding['data_sha256'],binding['seed'])
        self.binding=binding;self.cut_step=cut_step;self.fault=fault;self.failures=[]
    def _read_payload(self,path):
        try:
            state=torch.load(path,map_location='cpu',weights_only=True)
            validate_state(state,self.identity);p=state['payload']
            if p.get('run_binding')!=self.binding:raise ValueError('RUN_MODEL_DATA_CONFIG_IDENTITY_MISMATCH')
            required={'model_state','optimizer_state','scheduler_state','rng_state','sampler','step_trace','epoch','batch','step','phase','history','best_epoch','bad_epochs','patience_reference'}
            if not required<=set(p):raise ValueError('INCOMPLETE_TRAINING_STATE')
            if p['sampler']!=_sampler(p,self.binding):raise ValueError('SAMPLER_POSITION_MISMATCH')
            if p['scheduler_state'].get('last_epoch')!=p['step'] or p['scheduler_state'].get('_step_count')!=p['step']+1:raise ValueError('INVALID_SCHEDULER_STATE')
            if not set(fresh_lifecycle_state(self.binding['config_sha256'],self.binding['data_sha256']))<=set(p):raise ValueError('INCOMPLETE_SELECTION_STATE')
            n=len(self.binding['sampler_days']);cfg=config();total=n*cfg.epochs
            if not 0<=p['epoch']<=cfg.epochs or not 0<=p['batch']<=n or p['step']!=(p['epoch']*n+p['batch'] if p['phase']=='train' else (p['epoch']+1)*n):raise ValueError('INCONSISTENT_TRAINING_PROGRESS')
            expected_lr=cfg.learning_rate*lr_factor(p['step'],total,cfg.warmup_fraction)
            if p['scheduler_state'].get('_last_lr')!=[expected_lr]*2 or p['scheduler_state'].get('base_lrs')!=[cfg.learning_rate]*2 or any(g['lr']!=expected_lr for g in p['optimizer_state']['param_groups']):raise ValueError('INVALID_SCHEDULER_LR')
            _validate_optimizer(p,cfg,expected_lr)
            rng=p['rng_state']
            random.Random(0).setstate(rng['python']);np.random.RandomState(0).set_state(rng['numpy']);torch.Generator(device='cpu').set_state(rng['torch'])
            if rng.get('cuda') is not None:raise ValueError('CPU_ONLY_RNG_REQUIRED')
            shapes={'linear.weight':(2,2),'linear.bias':(2,)}
            if set(p['model_state'])!=set(shapes) or any(not isinstance(t,torch.Tensor) or tuple(t.shape)!=shapes[k] or t.dtype!=torch.float32 or not torch.isfinite(t).all() for k,t in p['model_state'].items()):raise ValueError('MODEL_STATE_MISMATCH')
            if len(p['step_trace'])!=p['step']:raise ValueError('INCOMPLETE_LOSS_LR_TRACE')
            for index,event in enumerate(p['step_trace']):
                if event['step']!=index+1 or event['epoch']!=index//n or event['batch']!=index%n+1 or not math.isfinite(event['loss']):raise ValueError('INVALID_STEP_TRACE')
                if event['lr_before']!=cfg.learning_rate*lr_factor(index,total,cfg.warmup_fraction) or event['lr_after']!=cfg.learning_rate*lr_factor(index+1,total,cfg.warmup_fraction):raise ValueError('INVALID_LR_TRACE')
            selected=fresh_lifecycle_state(self.binding['config_sha256'],self.binding['data_sha256'])
            if len(p['history'])!=p['epoch']:raise ValueError('INCOMPLETE_SELECTION_STATE')
            for h in p['history']:update_selection(selected,h['rank_ic_5d'],epoch=h['epoch'],min_delta=0.)
            if any(p[k]!=selected[k] for k in ('best_rank_ic_5d','best_epoch','bad_epochs','patience_reference')):raise ValueError('INVALID_SELECTION_STATE')
            if not p['model_state'] or not p['optimizer_state'] or not p['rng_state']:raise ValueError('INCOMPLETE_TRAINING_STATE')
            return state
        except Exception as exc:
            self.failures.append(type(exc).__name__+':'+str(exc));raise
    def save(self,state,best=False,interrupt_at=None):
        state=_safe_values(state);p=state['payload'];p['run_binding']=self.binding;p['sampler']=_sampler(p,self.binding)
        if self.fault=='checkpoint':interrupt_at='before_pointer'
        result=super().save(state,best=best,interrupt_at=interrupt_at)
        if self.cut_step is not None and p['step']==self.cut_step:os._exit(75) # full step + pointer fsync committed
        return result
    def _prune(self,manifest):pass # retain every owned probe generation, never GC historical artifacts


def read_state(root,matrix,run_id='cpu-proof',role='latest'):
    b=make_binding(matrix,run_id);store=ProofStore(Path(root)/'checkpoints',Path(root)/'local',b);state=store.load(role)
    if state is None:raise ValueError('NO_VALID_COMPLETE_STATE:'+','.join(store.failures))
    return state['payload']


def _normalized(value):
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous();return {'dtype':str(t.dtype),'shape':list(t.shape),'sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest()}
    if isinstance(value,Mapping):return {str(k):_normalized(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_normalized(v) for v in value]
    return value


def semantic_state(state):
    fields=('model_state','optimizer_state','scheduler_state','rng_state','sampler','epoch','batch','step','phase','terminal','step_trace','history','best_rank_ic_5d','best_epoch','bad_epochs','patience_reference','run_binding')
    return {k:_normalized(state[k]) for k in fields}


@contextmanager
def _attempt_lock(root):
    with _safe(root/'resume.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('DUPLICATE_ACTIVE_ATTEMPT')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def run_probe(matrix,output,mode='fresh',cut_step=None,run_id='cpu-proof',fault=None):
    output=_safe(output);output.mkdir(parents=True,exist_ok=True);attempt=uuid.uuid4().hex;report=output/'attempts'/attempt/'result.json'
    try:
        if mode not in {'fresh','resume'}:raise ValueError('INVALID_RUN_MODE')
        binding=make_binding(matrix,run_id)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        with _attempt_lock(output):
            pin=output/'run-binding.json'
            if mode=='fresh':
                if pin.exists():raise ValueError('EXISTING_RUN_REQUIRES_RESUME')
                atomic_json(pin,binding)
            elif not pin.is_file() or strict_json(pin.read_bytes())!=binding:raise ValueError('RUN_BINDING_MISMATCH')
            store=ProofStore(output/'checkpoints',output/'local',binding,cut_step,fault)
            adapter=LifecycleCheckpointAdapter(store,config().sha256)
            if mode=='resume':
                state=adapter.load(config().sha256,binding['data_sha256'])
                if state is None:raise ValueError('NO_VALID_COMPLETE_STATE:'+','.join(store.failures))
                if state['terminal']:raise ValueError('ALREADY_COMPLETED')
            def observe(event,state):
                state.setdefault('step_trace',[]).append(event)
                append_event(output/'events.jsonl','optimizer_step',**event)
            result=train_seed(Path(matrix),output,config(),seed=binding['seed'],device=torch.device('cpu'),smoke_steps=100000,
                model_factory=TinyCPUModel,checkpoint_store=adapter,matrix_manifest_validator=lambda doc,cfg:make_binding(matrix),
                step_observer=observe,engineering_probe=True,result_path=output/'engine-result.json')
            latest=read_state(output,matrix,run_id);semantic=semantic_state(latest)
            proof={'schema':'v7-actual-cpu-resume-proof-v1','status':'CPU_TRAINING_COMPLETE','research_admitted':False,
                'run_binding':binding,'step':latest['step'],'epoch':latest['epoch'],'semantic_state':semantic,'semantic_sha256':canonical_hash(semantic),
                'roles':{r:canonical_hash(semantic_state(read_state(output,matrix,run_id,r))) for r in ('latest','previous','best')},
                'trace':latest['step_trace'],'selection':latest['history'],'engine_evidence_class':result['evidence_class'],
                'granularity_claim':'NOT_YET_VERIFIED_UNTIL_CROSS_PROCESS_COMPARISON','gpu_resumable':False,'full_reproduction_ready':False}
            atomic_json(report,proof);return {**proof,'report':str(report)}
    except Exception as exc:
        failure={'schema':'v7-actual-cpu-resume-proof-v1','status':'FAILED','research_admitted':False,'reason':type(exc).__name__+':'+str(exc),'mode':mode}
        atomic_json(report,failure);return {**failure,'report':str(report)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('fresh','resume'),required=True);p.add_argument('--cut-step',type=int);p.add_argument('--run-id',default='cpu-proof');p.add_argument('--fault',choices=('checkpoint',))
    args=p.parse_args();result=run_probe(args.matrix,args.output,args.mode,args.cut_step,args.run_id,args.fault);print(json.dumps(result,allow_nan=False),flush=True)
    return 0 if result['status']=='CPU_TRAINING_COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
