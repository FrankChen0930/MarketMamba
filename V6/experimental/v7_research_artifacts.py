"""Small immutable filesystem bundles. Synthetic engineering, no research admission.

POSIX flock + same-filesystem rename define local publication semantics. Staging
and attempt diagnostics survive failures. No old bundle is replaced or collected.
External refs are reverified every use; a hash without bytes is NOT_RESTORABLE.
Trusted caller controls storage roots; not an OS sandbox against hostile writers.
"""
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import fcntl,json,math,os,re,shutil,uuid
from V6.experimental.v7_corrected_e5_replication import canonical_hash
from V6.experimental.v7_e1_checkpoint import sha256,atomic_json,_fsync_dir
from V6.experimental.v7_research_temporal import instant,_sha,_text

SCHEMA='v7-study-run-bundle-v1'
REQUIRED_KINDS={'RESOLVED_CONFIG','ENVIRONMENT','DATA_SPLIT_ELIGIBILITY','PREDICTIONS','DAILY_METRICS','REJECTION_LEDGER','COMPLETION_EVIDENCE'}
KINDS=REQUIRED_KINDS|{'TRAINING_LOGS','PAIRED_COMPARISON','EXPOSURE_LEDGER','CHECKPOINT_REFERENCE','DATASET_SNAPSHOT','CODE_MANIFEST','FEATURE_MANIFEST','LABEL_MANIFEST','UNIVERSE_MANIFEST','EVALUATION_POLICY','REPLAY_REQUEST'}
IDENTITY_HASHES=('code_manifest_sha256','data_sha256','feature_sha256','label_sha256','universe_sha256','evaluation_policy_sha256','model_sha256','training_contract_sha256')

class BundleError(ValueError):pass

def strict_json(data):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise BundleError('DUPLICATE_JSON_KEY')
            result[k]=v
        return result
    def nonfinite(_):raise BundleError('NONFINITE_JSON')
    def finite_float(value):
        result=float(value)
        if not math.isfinite(result):raise BundleError('NONFINITE_JSON')
        return result
    return json.loads(data,object_pairs_hook=pairs,parse_constant=nonfinite,parse_float=finite_float)


def run_key(identity):return canonical_hash({'study_id':identity['study_id'],'run_id':identity['run_id']})
def _stamp():return datetime.now(timezone.utc).isoformat()

def _safe(path):
    if '..' in Path(path).parts:raise BundleError('PARENT_TRAVERSAL_FORBIDDEN')
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise BundleError('SYMLINK_PATH_FORBIDDEN')
    return path


def _relative(value):
    if not isinstance(value,str) or not value or '\\' in value:raise BundleError('UNSAFE_ARTIFACT_PATH')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p)!=value or value in {'manifest.json','publication.json'}:
        raise BundleError('UNSAFE_ARTIFACT_PATH')
    return value


def validate_manifest(m,expected_sha256):
    if not isinstance(m,Mapping) or canonical_hash(m)!=expected_sha256:raise BundleError('MANIFEST_IDENTITY_MISMATCH')
    profile=(m.get('schema'),m.get('profile'),m.get('training_status'))
    synthetic=profile in {(SCHEMA,'SYNTHETIC_EVALUATION','NOT_RUN_SYNTHETIC'),('v7-study-run-bundle-v2','SYNTHETIC_CLASSICAL_FIT','SYNTHETIC_CPU_FIT_COMPLETED')} and m.get('evidence_class')=='SYNTHETIC'
    engineering=profile==('v7-study-run-bundle-v2','RETROSPECTIVE_ENGINEERING_SMOKE','CPU_ENGINEERING_SMOKE_COMPLETED') and m.get('evidence_class')=='RETROSPECTIVE_ENGINEERING_SMOKE' and m.get('authority_scope')=='HumanWindow23_LIMITED_ENGINEERING_SMOKE'
    exploratory=profile==('v7-study-run-bundle-v2','RETROSPECTIVE_EXPLORATORY_LIMITED','CPU_EXPLORATORY_PILOT_COMPLETED') and m.get('evidence_class')=='RETROSPECTIVE_EXPLORATORY_LIMITED' and m.get('authority_scope')=='HumanWindow24_LIMITED_EXPLORATORY_PILOT'
    if not (synthetic or engineering or exploratory):raise BundleError('BUNDLE_SCHEMA_OR_SCOPE_MISMATCH')
    if m.get('research_admitted') is not False:raise BundleError('AUTHORITY_OR_TRAINING_STATUS_MISMATCH')
    i=m['identity']
    if not isinstance(i,Mapping) or not all(_text(i.get(k)) for k in ('study_id','recipe_id','run_id','fold_id','origin_id')):raise BundleError('RUN_IDENTITY_UNKNOWN')
    instant(i['origin_at'])
    if type(i.get('seed')) is not int or i['seed']<0 or not isinstance(i.get('code_commit'),str) or not re.fullmatch('[0-9a-f]{40}',i['code_commit']):raise BundleError('CODE_OR_SEED_IDENTITY_INVALID')
    if any(not _sha(i.get(k)) for k in IDENTITY_HASHES):raise BundleError('IDENTITY_HASH_UNKNOWN')
    if i.get('prediction_role') not in {'SELECTION','FORWARD'}:raise BundleError('PREDICTION_ROLE_INVALID')
    if i.get('checkpoint_status')=='NOT_GENERATED_SYNTHETIC_NO_TRAINING':
        if m['schema']!=SCHEMA or i.get('checkpoint_sha256') is not None:raise BundleError('CHECKPOINT_STATUS_MISMATCH')
    elif i.get('checkpoint_status')=='NOT_GENERATED_TINY_CLASSICAL_NO_RESUME':
        if m['schema']!='v7-study-run-bundle-v2' or i.get('checkpoint_sha256') is not None:raise BundleError('CHECKPOINT_STATUS_MISMATCH')
    elif i.get('checkpoint_status')=='SYNTHETIC_REFERENCE':
        if not _sha(i.get('checkpoint_sha256')):raise BundleError('CHECKPOINT_IDENTITY_UNKNOWN')
    else:raise BundleError('CHECKPOINT_STATUS_UNKNOWN')
    refs=i.get('source_vintages')
    if not isinstance(refs,list) or not refs or any(not isinstance(x,Mapping) or not _sha(x.get('source_sha256')) or not _text(x.get('vintage_id')) for x in refs):raise BundleError('SOURCE_VINTAGE_UNKNOWN')
    artifacts=m['artifacts'];required=m['required_artifacts']
    if not isinstance(artifacts,list) or not isinstance(required,list) or any(not isinstance(a,Mapping) for a in artifacts):raise BundleError('ARTIFACT_SCHEMA_INVALID')
    names=[];paths=[]
    for a in artifacts:
        name=a['name'];path=a['path'];storage=a['storage']
        if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9_-]+',name) or a['kind'] not in KINDS:raise BundleError('ARTIFACT_KIND_OR_NAME_INVALID')
        if storage=='BUNDLE':_relative(path)
        elif storage=='EXTERNAL_REFERENCE':
            if not isinstance(path,str) or not Path(path).is_absolute():raise BundleError('REFERENCE_PATH_NOT_ABSOLUTE')
        else:raise BundleError('STORAGE_MODE_UNKNOWN')
        if not _sha(a['sha256']) or type(a['bytes']) is not int or a['bytes']<=0 or a['format'] not in {'JSON','BYTES'}:raise BundleError('ARTIFACT_IDENTITY_INVALID')
        if a['format']=='JSON' and not _text(a.get('schema_id')):raise BundleError('ARTIFACT_SCHEMA_UNKNOWN')
        names.append(name);paths.append(path)
    if len(set(names))!=len(names) or len(set(paths))!=len(paths):raise BundleError('DUPLICATE_ARTIFACT_NAME_OR_PATH')
    if any(not isinstance(n,str) for n in required) or len(set(required))!=len(required) or not set(required)<=set(names):raise BundleError('REQUIRED_ARTIFACT_IDENTITY_INVALID')
    required_kinds=REQUIRED_KINDS|({'TRAINING_LOGS'} if m['schema']=='v7-study-run-bundle-v2' else set())
    if not required_kinds<={a['kind'] for a in artifacts if a['name'] in required}:raise BundleError('REQUIRED_PROFILE_ARTIFACTS_MISSING')


def artifact_path(bundle,a,external_roots=()):
    if a['storage']=='BUNDLE':return _safe(Path(bundle)/_relative(a['path']))
    path=_safe(a['path']);roots=[_safe(p) for p in external_roots]
    if not any(path.is_relative_to(root) for root in roots):raise BundleError('REFERENCE_ROOT_NOT_ALLOWED:'+a['name'])
    return path


def verify_artifact(bundle,a,external_roots=()):
    path=artifact_path(bundle,a,external_roots);name=a['name']
    if not path.is_file():raise BundleError(('NOT_RESTORABLE:' if a['storage']=='EXTERNAL_REFERENCE' else 'MISSING_ARTIFACT_BYTES:')+name)
    if path.stat().st_size!=a['bytes'] or sha256(path)!=a['sha256']:raise BundleError('ARTIFACT_HASH_MISMATCH:'+name)
    if a['format']=='JSON':
        value=strict_json(path.read_bytes())
        if not isinstance(value,Mapping) or value.get('schema')!=a['schema_id']:raise BundleError('ARTIFACT_SCHEMA_MISMATCH:'+name)
    return path


def _verify_at(path,expected,external_roots):
    path=_safe(path);m=strict_json(_safe(path/'manifest.json').read_bytes());validate_manifest(m,expected)
    pub=strict_json(_safe(path/'publication.json').read_bytes())
    if not isinstance(pub,Mapping):raise BundleError('PUBLICATION_EVIDENCE_MISMATCH')
    if pub.get('schema')!='v7-bundle-publication-v1' or pub.get('manifest_sha256')!=expected or pub.get('state')!='ARTIFACTS_VERIFIED':raise BundleError('PUBLICATION_EVIDENCE_MISMATCH')
    instant(pub['generated_at'])
    for a in m['artifacts']:verify_artifact(path,a,external_roots)
    return m,pub


def verify_bundle(root,key,expected_manifest_sha256,*,external_roots=()):
    base={'research_admitted':False,'status':'NOT_VERIFIED','reasons':[]}
    try:
        if not _sha(key):raise BundleError('RUN_KEY_INVALID')
        p=_safe(Path(root)/'completed'/key)
        if not p.is_dir():raise BundleError('NO_COMPLETED_BUNDLE')
        m,pub=_verify_at(p,expected_manifest_sha256,external_roots)
        if run_key(m['identity'])!=key:raise BundleError('RUN_KEY_MISMATCH')
        return {**base,'status':'ARTIFACTS_VERIFIED','bundle_path':str(p),'manifest':m,'publication':pub,'training_status':m['training_status'],'manifest_sha256':expected_manifest_sha256}
    except (OSError,KeyError,TypeError,ValueError) as exc:return {**base,'reasons':[str(exc)]}


@contextmanager
def _lock(path):
    fd=os.open(_safe(path),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX);yield
    finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def publish_bundle(root,manifest,expected_manifest_sha256,sources,*,external_roots=(),interrupt_at=None):
    """sources maps artifact names to bytes or caller-owned source Paths.

    Completion is visible only under completed/<study/run hash> after rename.
    Per-run locks serialize competing publications. Retries never repair/overwrite
    corrupt completed bundles; attempt logs retain evidence outside immutable data.
    """
    root=Path(root);attempt=uuid.uuid4().hex;log=None;stage=None
    def result(status,reasons=(),**extra):return {'status':status,'reasons':list(reasons),'research_admitted':False,'attempt_id':attempt,**extra}
    try:
        _safe(root).mkdir(parents=True,exist_ok=True)
        for folder in ('attempts','staging','completed','locks'):_safe(root/folder).mkdir(exist_ok=True)
        _fsync_dir(root);_fsync_dir(root.parent)
        log=root/'attempts'/(attempt+'.json');atomic_json(log,{'state':'INCOMPLETE','started_at':_stamp()})
        validate_manifest(manifest,expected_manifest_sha256)
        if not isinstance(sources,Mapping):raise BundleError('SOURCES_CONTAINER_INVALID')
        key=run_key(manifest['identity'])
        final=root/'completed'/key
        with _lock(root/'locks'/(key+'.lock')):
            if final.exists():
                existing=strict_json(_safe(final/'manifest.json').read_bytes())
                if canonical_hash(existing)!=expected_manifest_sha256:raise BundleError('IMMUTABLE_RUN_CONFLICT')
                verified=verify_bundle(root,key,expected_manifest_sha256,external_roots=external_roots)
                if verified['status']!='ARTIFACTS_VERIFIED':raise BundleError('EXISTING_BUNDLE_INVALID:'+','.join(verified['reasons']))
                atomic_json(log,{'state':'ALREADY_PUBLISHED','manifest_sha256':expected_manifest_sha256,'generated_at':_stamp()})
                return result('ALREADY_PUBLISHED',bundle_path=str(final),manifest_sha256=expected_manifest_sha256)
            stage=root/'staging'/attempt;stage.mkdir()
            atomic_json(stage/'manifest.json',manifest)
            for a in manifest['artifacts']:
                if a['storage']=='BUNDLE':
                    source=sources.get(a['name'])
                    if source is None:raise BundleError('MISSING_ARTIFACT_BYTES:'+a['name'])
                    dest=_safe(stage/a['path']);dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as out:
                        if isinstance(source,bytes):out.write(source)
                        else:
                            with _safe(source).open('rb') as inp:shutil.copyfileobj(inp,out,4*1024*1024)
                        out.flush();os.fsync(out.fileno())
                    _fsync_dir(dest.parent)
                verify_artifact(stage,a,external_roots)
                if interrupt_at=='after_artifact':raise BundleError('INTERRUPTED_AFTER_ARTIFACT')
            atomic_json(stage/'publication.json',{'schema':'v7-bundle-publication-v1','manifest_sha256':expected_manifest_sha256,'generated_at':_stamp(),'state':'ARTIFACTS_VERIFIED','research_admitted':False})
            _verify_at(stage,expected_manifest_sha256,external_roots)
            if interrupt_at=='before_publish':raise BundleError('INTERRUPTED_BEFORE_PUBLISH')
            os.rename(stage,final);_fsync_dir(final.parent);_fsync_dir(root/'staging')
            if interrupt_at=='after_publish':raise BundleError('INTERRUPTED_AFTER_PUBLISH')
            atomic_json(log,{'state':'ARTIFACTS_VERIFIED','generated_at':_stamp(),'manifest_sha256':expected_manifest_sha256})
            return result('ARTIFACTS_VERIFIED',bundle_path=str(final),manifest_sha256=expected_manifest_sha256)
    except (OSError,KeyError,TypeError,ValueError) as exc:
        reasons=[str(exc)]
        if log is not None:
            atomic_json(log,{'state':'INCOMPLETE','generated_at':_stamp(),'reasons':reasons,'staging_path':str(stage) if stage else None})
        return result('INCOMPLETE',reasons)
