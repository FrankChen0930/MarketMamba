"""P4d stage1 source-only delivery and mandatory, evidence-validated rehearsal."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
from research_lab import p4_bundle as pb
from . import p4d

# Isolated archive policy: leave the legacy P4a module and its source hashes intact.
P4D_SOURCE_NAMES=('p4d.py','p4d_train.py','p4d_bundle.py')
P4D_EXTRA_PATHS={*('source/tools/'+n for n in P4D_SOURCE_NAMES),
    'source/tasks/p4d-stage1-contract-20261125.json',
    'source/tasks/spec-p4d-equal-tuning-20261125.md'}


def archive_api():
    import types
    module=types.ModuleType('p4d_archive')
    module.__file__=pb.__file__
    exec(compile(Path(pb.__file__).read_bytes(),pb.__file__,'exec'),module.__dict__)
    legacy=module.allowed_path
    def allowed(name):
        if name in P4D_EXTRA_PATHS:
            module.safe_name(name);return
        return legacy(name)
    module.allowed_path=allowed
    return module


LEGACY_BUNDLE=pb
pb=archive_api()


REHEARSAL_JOBS=((256,2),(64,4),(32,2))


def guarded_launch(gate,formal):
    evidence=gate()  # Exceptions, missing/invalid publications never reach formal.
    if not isinstance(evidence,dict) or not evidence:raise ValueError('STOP: missing rehearsal evidence')
    return formal(evidence)


def validate_gate(evidence,binding,publication_root=None):
    if not isinstance(evidence,dict) or set(evidence)!={'binding','evidence_class','fits','evaluation','elapsed_seconds'}:
        raise ValueError('STOP: rehearsal schema')
    if evidence['binding']!=binding or evidence['evidence_class']!='REHEARSAL / 不是研究結果':raise ValueError('STOP: rehearsal identity')
    if type(evidence['elapsed_seconds']) not in (int,float) or not 0<evidence['elapsed_seconds']<float('inf'):raise ValueError('STOP: rehearsal elapsed')
    expected={'gru:256:2:anchor','gru:64:4:anchor','gru:32:2:anchor','c0plus:64:2:anchor','gru:32:2:outer','c0plus:64:2:outer'}
    if not isinstance(evidence['fits'],dict) or set(evidence['fits'])!=expected:raise ValueError('STOP: rehearsal coverage')
    from .p4d_train import validate_complete
    for name,value in evidence['fits'].items():
        if set(value)!={'path','complete','binding','seconds','peak_allocated','peak_reserved'}:raise ValueError('STOP: rehearsal fit schema')
        path=Path(value['path'])
        allowed=Path('/tmp') if publication_root is None else Path(publication_root)
        if path.is_symlink() or not path.resolve().is_relative_to(allowed.resolve()) or (path/'complete.json').is_symlink() or value['complete']!=pb.digest(path/'complete.json'):raise ValueError('STOP: rehearsal publication')
        validate_complete(path,value['binding'])
        metrics=json.loads((path/'metrics.json').read_text())
        if metrics.get('evidence_class')!=evidence['evidence_class']:raise ValueError('STOP: rehearsal metrics class')
        arm,width,layers,phase=name.split(':')
        expected_job=dict(arm=arm,width=int(width),layers=int(layers),seed=17,phase=phase,origin='anchor' if phase=='anchor' else 2021)
        if value['binding']['job']!=expected_job or value['binding']['run']!=binding:raise ValueError('STOP: rehearsal request mismatch')
        cfg=value['binding']['cfg']
        if cfg['epochs']!=1 or cfg['steps_per_epoch']!=1:raise ValueError('STOP: rehearsal update count')
        for k in ('seconds','peak_allocated','peak_reserved'):
            v=value[k]
            if type(v) not in (int,float) or not 0<=v<float('inf'):raise ValueError('STOP: rehearsal telemetry')
    evaluation=evidence['evaluation']
    if not isinstance(evaluation,dict) or set(evaluation)!={'path','identity'} or Path(evaluation['path']).is_symlink() or not Path(evaluation['path']).resolve().is_relative_to(allowed.resolve()) or pb.digest(evaluation['path'])!=evaluation['identity']:raise ValueError('STOP: rehearsal evaluation publication')
    report=json.loads(Path(evaluation['path']).read_text())
    if not report.get('daily') or report.get('independent_confirmation') is not False:raise ValueError('STOP: rehearsal evaluation schema')
    import pandas as pd
    gru=pd.read_parquet(Path(evidence['fits']['gru:32:2:outer']['path'])/'scores.parquet')
    c0=pd.read_parquet(Path(evidence['fits']['c0plus:64:2:outer']['path'])/'scores.parquet')
    if report!=p4d.descriptive_pair(gru,c0):raise ValueError('STOP: rehearsal evaluation content mismatch')



def ensure_rehearsal(progress_root,binding,gate):
    """Persist validated rehearsal publications; reconnect never trusts exit0.

    Revalidate all durable bytes on reuse. This is deliberately separate from
    the P4d training ledger and never requires the old /tmp directory to survive.
    """
    from copy import deepcopy
    import shutil
    import uuid
    from research_lab.rolling_seq_scores import atomic_json
    progress_root=Path(progress_root)
    progress_root.mkdir(parents=True,exist_ok=True)
    marker=progress_root/'rehearsal_passed.json'
    publications=progress_root/'rehearsal'
    if marker.exists() or marker.is_symlink():
        try:
            if marker.is_symlink():raise ValueError('marker symlink')
            saved=json.loads(marker.read_text())
            if set(saved)!={'version','binding','evidence','evidence_sha256'} or saved['version']!='p4d-rehearsal-v1':
                raise ValueError('marker schema')
            if saved['binding']!=binding or saved['evidence_sha256']!=p4d.hashed(saved['evidence']):
                raise ValueError('marker identity/digest')
            validate_gate(saved['evidence'],binding,publications)
            print('排練證據已重新驗證，沿用同一身分（不是研究結果）。',flush=True)
            return saved['evidence']
        except (ValueError,TypeError,KeyError,OSError,OverflowError,RecursionError) as error:
            print('排練標記無效；期望=同身分完整證據；實際=',repr(error),'；下一步=重新排練。',flush=True)
            marker.unlink()
    evidence=gate()
    validate_gate(evidence,binding)
    durable=deepcopy(evidence)
    target=publications/uuid.uuid4().hex
    target.mkdir(parents=True,exist_ok=False)
    for name,fit in durable['fits'].items():
        source=Path(fit['path'])
        if any(p.is_symlink() for p in source.rglob('*')):raise ValueError('rehearsal publication symlink')
        destination=target/name.replace(':','-')
        shutil.copytree(source,destination)
        fit['path']=str(destination.resolve())
    destination=target/'evaluation.json'
    shutil.copy2(evidence['evaluation']['path'],destination)
    durable['evaluation']['path']=str(destination.resolve())
    validate_gate(durable,binding,publications)
    atomic_json(marker,dict(version='p4d-rehearsal-v1',binding=binding,
        evidence=durable,evidence_sha256=p4d.hashed(durable)))
    return durable


def runtime(bundle):
    """GRU-only subset of the existing observed Colab runtime; no Mamba import."""
    from research_lab.p4_colab import require_cuda,environment_snapshot
    require_cuda()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    observed=json.loads((Path(bundle)/'source/environments/p4a-colab-runtime.json').read_text())
    required={n:v for n,v in observed['packages'].items() if n not in ('mamba-ssm','causal-conv1d','einops')}
    required['lightgbm']='4.6.0'
    actual=environment_snapshot(required)
    import torch
    mismatch={k:dict(expected=v,actual=actual.get(k)) for k,v in observed['runtime'].items() if actual.get(k)!=v}
    mismatch.update({n:dict(expected=v,actual=actual['packages'].get(n)) for n,v in required.items() if actual['packages'].get(n)!=v})
    if actual['gpu']!='Tesla T4':mismatch['gpu']=actual['gpu']
    if mismatch:raise ValueError('STOP: P4d runtime mismatch; '+json.dumps(mismatch))
    actual['determinism']=dict(torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace_config=__import__('os').getenv('CUBLAS_WORKSPACE_CONFIG'))
    actual['reference_observation']=pb.digest(Path(bundle)/'source/environments/p4a-colab-runtime.json')
    return actual


def run_binding(data_root,bundle):
    from .p4d_train import dataset_identity
    pb.verify(bundle)
    matrix=dataset_identity(data_root)
    mf=json.loads((Path(bundle)/'bundle-manifest.json').read_text())
    expected=mf['metadata'].get('matrix_identity')
    if not mf['metadata'].get('split_manifests'):raise ValueError('STOP: frozen split manifests required')
    if expected is None or expected!=matrix:raise ValueError('STOP: prepared matrix differs from delivery; repack exact data')
    return dict(experiment='p4d_stage1',contract=p4d.hashed(p4d.CONTRACT),queue=p4d.QUEUE_SHA,
        split_manifests=mf['metadata']['split_manifests'],bundle=pb.digest(Path(bundle)/'bundle-manifest.json'),matrix=matrix,environment=runtime(bundle))


def rehearse(data_root,bundle,cache):
    import torch
    from .p4d_train import prepare,execute_job
    p4d.verify_contract();binding=run_binding(data_root,bundle)
    out=Path(tempfile.mkdtemp(prefix='p4d-REHEARSAL-'));started=time.perf_counter()
    fits={};frames={};selected={'gru:17':1,'c0plus:17':1}
    for phase in ('anchor','outer'):
        data,support,_=prepare(Path(data_root)/'folds/fold_2021',cache,anchor=phase=='anchor')
        require_frozen_split(binding,'anchor' if phase=='anchor' else 2021,support)
        specs=[('gru',w,l) for w,l in REHEARSAL_JOBS] if phase=='anchor' else [('gru',32,2)]
        specs.append(('c0plus',64,2))
        for arm,width,layers in specs:
            job=dict(phase=phase,origin='anchor' if phase=='anchor' else 2021,arm=arm,seed=17,width=width,layers=layers)
            name=f'{arm}:{width}:{layers}:{phase}';path=out/name.replace(':','-')
            torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
            mf=execute_job(data,support,job,path,binding,selected,rehearsal=True)
            fits[name]=dict(path=str(path),complete=pb.digest(path/'complete.json'),binding=mf['binding'],
                seconds=time.perf_counter()-t0,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            if phase=='outer':
                import pandas as pd
                frames[arm]=pd.read_parquet(path/'scores.parquet')
            print('排練完成',name,fits[name]['seconds'],'seconds',flush=True)
    report=p4d.descriptive_pair(frames['gru'],frames['c0plus']);target=out/'evaluation.json'
    from research_lab.rolling_seq_scores import atomic_json
    atomic_json(target,report)
    result=dict(binding=binding,evidence_class='REHEARSAL / 不是研究結果',fits=fits,
        evaluation=dict(path=str(target),identity=pb.digest(target)),elapsed_seconds=time.perf_counter()-started)
    validate_gate(result,binding);atomic_json(out/'rehearsal.json',result)
    return result


def delivery(output,*,prepared=None,run_name='p4d_stage1'):
    p4d.verify_contract()
    if not run_name or Path(run_name).name!=run_name or run_name in ('.','..'):raise ValueError('single run-name required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    files={'source/'+p.relative_to(pb.ROOT).as_posix():p for p in pb.source_paths()}
    for name in P4D_SOURCE_NAMES:files['source/tools/'+name]=p4d.ROOT/'tools'/name
    for name in ('tasks/p4d-stage1-contract-20261125.json','tasks/spec-p4d-equal-tuning-20261125.md',
                 'tasks/p4a-colab-contract-20261121.json','tasks/spec-p4a-colab-20261121.md',
                 'environments/p4a-colab-runtime.json','docs/operations/colab-p4.md'):
        files['source/'+name]=pb.ROOT/name
    files['requirements.lock.txt']=pb.ROOT/'environments/p4a-colab.requirements.lock.txt'
    metadata=dict(experiment='p4d_stage1',contract=p4d.hashed(p4d.CONTRACT),queue=p4d.QUEUE_SHA,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pb.ROOT,text=True).strip(),
        source_status='UNMERGED / NOT_DEPLOYED / NOT_ACCEPTED; working bytes pinned',
        matrix_identity=None,split_manifests=None,source_compatibility='new run; legacy p4_source_compat retained, no automatic upgrades')
    if prepared:
        from .p4d_train import dataset_identity
        metadata['matrix_identity']=dataset_identity(prepared)
        frozen=json.loads((p4d.ROOT/'tasks/p4d-stage1-prepared-splits-20261125.json').read_text())
        if frozen['matrix_identity']!=metadata['matrix_identity']:raise ValueError('frozen split input identity mismatch')
        metadata['split_manifests']=frozen['splits']
    archive=output/'p4d-bundle.zip';pb.write_archive(archive,files,metadata)
    # Reuse the audited stdlib-only safe extractor bootstrap, not the P4a runner.
    old=pb.colab_notebook(expected_archive=pb.digest(archive))
    cells=[]
    def code(source):cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=source+'\n'))
    code("from pathlib import Path\nRUN_DIR='/content/drive/MyDrive/MarketMamba/p4d/"+run_name+"'\nEXPERIMENT_DIR=str(Path(RUN_DIR).parent)\nDATA_DIR='/content/drive/MyDrive/MarketMamba/data/matrix_v1'\nBUNDLE_NAME='p4d-bundle.zip'\nLOCAL_ROOT='/content/p4d-"+pb.digest(archive)['sha256'][:16]+"'\nEXPECTED_ARCHIVE="+repr(pb.digest(archive)))
    cells.append(dict(cell_type='markdown',metadata={},source='P4d第一段；僅GRU A/B與baseline，140 fits，無時間上限。先上傳同批zip＋ipynb再從上到下執行。現場UNVERIFIED，獨立審查前不得正式交付。PROGRESS.txt位於MarketMamba/p4d/progress/formal；同身分排練成功標記為progress/formal/rehearsal_passed.json，重連會重驗持久化產物。'))
    bootstrap=old['cells'][2]['source']
    extension="\nlegacy_allowed=p4.allowed_path\nP4D_EXTRA_PATHS="+repr(sorted(P4D_EXTRA_PATHS))+"\ndef p4d_allowed(name):\n    if name in P4D_EXTRA_PATHS: return p4.safe_name(name)\n    return legacy_allowed(name)\np4.allowed_path=p4d_allowed\n"
    bootstrap=bootstrap.replace("spec.loader.exec_module(p4)","spec.loader.exec_module(p4)"+extension)
    code(bootstrap)
    code("""import os, sys, subprocess, importlib.metadata
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONPATH']=str(root/'source/V6')
# 第一段不安裝／啟動Mamba2；缺必要CPU評估套件才補安裝。
try: version=importlib.metadata.version('lightgbm')
except importlib.metadata.PackageNotFoundError: version=None
if version!='4.6.0': subprocess.run([sys.executable,'-m','pip','install','lightgbm==4.6.0'],check=True)
from tools.p4d_bundle import prepare_raw_dataset
# 矩陣由Drive複製到本機，逐檔size/SHA驗證，重用完整副本。
prepare_raw_dataset(DATA_DIR,'/content/p4d-data')
""")
    code("""# 串流與flush由直接同kernel呼叫保留；60秒heartbeat寫PROGRESS.txt。
from tools.p4d_bundle import guarded_launch,rehearse,ensure_rehearsal,run_binding
from tools.p4d_train import run
try:
    result=guarded_launch(
        lambda:ensure_rehearsal(Path(EXPERIMENT_DIR)/'progress/formal',
            run_binding('/content/p4d-data',root),
            lambda:rehearse('/content/p4d-data',root,Path(EXPERIMENT_DIR)/'rankgauss_cache')),
        lambda evidence:run('/content/p4d-data',EXPERIMENT_DIR,root,rehearsal_evidence=evidence))
except BaseException as error:
    print('STOP: P4d排練／執行失敗；實際=',repr(error),'；下一步=保留帳本與checkpoint，檢查錯誤後重新從頭執行。',flush=True)
    raise
""")
    code("""# 成功後下載，Human自行關閉runtime；不新增排程或通知。
import zipfile
from google.colab import files
archive='/content/p4d-stage1-results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as packed:
    for namespace in ('budget_p4d_stage1','progress/formal'):
        for path in sorted((Path(EXPERIMENT_DIR)/namespace).rglob('*')):
            if path.is_file(): packed.write(path,path.relative_to(EXPERIMENT_DIR))
    packed.write(root/'bundle-manifest.json','provenance/bundle-manifest.json')
files.download(archive)
print('完成產物已驗證；不等於Human acceptance。')
""")
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    (output/(run_name+'.ipynb')).write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    (output/'README.txt').write_text('同批zip＋ipynb放到MyDrive/MarketMamba/p4d/'+run_name+'，從上到下執行。\n獨立審查前不正式交付；現場UNVERIFIED。\n')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();delivery(a.output,prepared=a.prepared)




def prepare_raw_dataset(source,destination):
    """Copy verified raw matrix only; no legacy mask detection or source writes."""
    import os,shutil
    source,destination=Path(source),Path(destination)
    if destination.resolve()==source.resolve() or destination.resolve().is_relative_to(source.resolve()):raise ValueError('local destination must be separate')
    mf=pb.validate_dataset(source)
    for name,ident in mf['files'].items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink():raise ValueError('local raw link forbidden')
        if target.exists() and pb.digest(target)==ident:continue
        partial=target.with_suffix(target.suffix+'.partial');shutil.copy2(source/name,partial)
        if pb.digest(partial)!=ident:raise ValueError('raw copy hash mismatch')
        os.replace(partial,target)
    for year in range(2021,2027):
        fold=destination/f'folds/fold_{year}'
        for name in pb.SHARED_FILES:
            target=fold/name;shared=destination/'shared'/name
            if target.is_symlink():
                if target.resolve()!=shared.resolve():raise ValueError('raw shared link mismatch')
            elif target.exists():raise ValueError('raw shared collision')
            else:target.symlink_to(shared.resolve())
    return pb.digest(source/'DATASET_MANIFEST.json')


def require_frozen_split(binding,origin,support):
    if binding['split_manifests'].get(str(origin))!=support['manifest']:raise ValueError('STOP: frozen split/support mismatch')


if __name__=='__main__':main()
