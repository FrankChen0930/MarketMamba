"""Auditable, fail-closed source lineage adoption; no training or artifact migration.

The old run stores only a manifest digest. Its original delivery ZIP supplies the
missing manifest bytes, never a guessed reconstruction of historical metadata.
"""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from .p4_bundle import ROOT, digest, MAX_MANIFEST_BYTES

CONTRACT='tasks/p4a-colab-contract-20261121.json'
RUNTIME='environments/p4a-colab-runtime.json'
LOCK='environments/p4a-colab.requirements.lock.txt'
QUEUE_SHA='4dc90ec0ada6c6d34b43c429d359424c502634e502c6856c6774bf138de7ba23'
BUDGET=dict(policy='record_only',units_per_hour=0,thermal_stop_c=85)
# Bounded reviewed surface. Any other changed source requires a new review;
# a category is an explanation, not a numerical-equivalence proof.
REASONS={
    'V6/research_lab/seq_runner.py':(['進度顯示'],'observer progress callbacks only; CPU bitwise regression required'),
    'V6/research_lab/p4_colab.py':(['進度顯示','日誌','speed-test 帳本分離'],'progress, isolated timing ledger and audited lineage selection; frozen training/fit validation unchanged'),
    'V6/research_lab/p4_bundle.py':(['進度顯示','日誌','文件'],'streaming notebook, source-upgrade declaration packaging and instructions'),
    'V6/research_lab/p4_source_compat.py':(['日誌'],'source lineage evidence validation and SOURCE_UPGRADE audit; no numerical operations'),
    'docs/operations/colab-p4.md':(['文件'],'operator instructions'),
}


def identity(payload):
    return dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


def source_name(name):
    return name.startswith('source/') and not name.startswith('source/tasks/p4a-source-compat-')


def current_sources():
    paths=[p for subtree in ('V6/research_lab','V6/experimental') for p in (ROOT/subtree).rglob('*.py') if 'tests' not in p.parts]
    paths += [ROOT/n for n in (CONTRACT,'tasks/spec-p4a-colab-20261121.md','docs/operations/colab-p4.md',RUNTIME)]
    return {'source/'+p.relative_to(ROOT).as_posix():digest(p) for p in sorted(paths)}


def git_bytes(commit,path):
    return subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT)


def neutral_surface(commit):
    """Do not automatically classify changed training/validation as mere logging."""
    import ast
    old=ast.parse(git_bytes(commit,'V6/research_lab/p4_colab.py'))
    new=ast.parse((ROOT/'V6/research_lab/p4_colab.py').read_text())
    protected={'Ledger','fit_queue','resume_fit','complete_fit',
               'validate_fit_outputs','validate_training_state','effective_recipe'}
    def definitions(tree):
        return {node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in protected}
    if definitions(old)!=definitions(new):
        raise ValueError('training/fit/ledger semantics changed; source-compatible declaration forbidden')
    # These observer-bearing functions already differ only in reviewed progress
    # callbacks at this task's base. Refuse further edits without new review.
    baseline=ast.parse(git_bytes('20e6b5408d9f9b13d56290a72a2dc313c35d1b94','V6/research_lab/p4_colab.py'))
    for name in ('train_fit','permutation_fit','Observer'):
        left=next(n for n in baseline.body if getattr(n,'name',None)==name)
        right=next(n for n in new.body if getattr(n,'name',None)==name)
        if ast.dump(left)!=ast.dump(right):raise ValueError('unreviewed observer-bearing function: '+name)
    path='V6/research_lab/seq_runner.py'
    left=ast.parse(git_bytes('20e6b5408d9f9b13d56290a72a2dc313c35d1b94',path))
    right=ast.parse((ROOT/path).read_text())
    if ast.dump(left)!=ast.dump(right):
        raise ValueError('seq_runner differs from the numerically tested progress baseline; declaration forbidden')


def declaration(commit,to_sources,matrix):
    """Generate from git object bytes, never from memory or handwritten hashes."""
    from .p4_colab import fit_queue
    full=subprocess.check_output(['git','rev-parse',commit+'^{commit}'],cwd=ROOT,text=True).strip()
    if full!='4e56e9db65a453a186511368788deb732a10eb1b':
        raise ValueError('uncertified from commit: this upgrade has bitwise evidence only for 4e56e9d; add reviewed cross-version tests first')
    neutral_surface(full)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',full],cwd=ROOT,text=True).splitlines()
    paths=[n for n in paths if (n.startswith(('V6/research_lab/','V6/experimental/')) and n.endswith('.py') and 'tests' not in Path(n).parts)
           or n in (CONTRACT,'tasks/spec-p4a-colab-20261121.md','docs/operations/colab-p4.md',RUNTIME)]
    before={'source/'+n:identity(git_bytes(full,n)) for n in paths}
    changes=[]
    for name in sorted(before.keys()|to_sources.keys()):
        if before.get(name)==to_sources.get(name):continue
        path=name.removeprefix('source/')
        if path not in REASONS:raise ValueError(f'unreviewed source difference: {path}; independent numerical-neutrality review required')
        categories,reason=REASONS[path]
        old=git_bytes(full,path).decode() if name in before else ''
        new=(ROOT/path).read_text() if name in to_sources else ''
        changes.append(dict(path=name,before=before.get(name),after=to_sources.get(name),categories=categories,reason=reason,
                            diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=full+'/'+path,tofile='working/'+path))))
    invariants={}
    for path in (CONTRACT,RUNTIME,LOCK):
        old=identity(git_bytes(full,path));actual=digest(ROOT/path)
        if old!=actual:raise ValueError(f'invariant bytes differ: {path}: expected {old}, actual {actual}')
        invariants[path]=old
    contract=json.loads(git_bytes(full,CONTRACT))
    queue=fit_queue(contract)
    queue_sha=hashlib.sha256(json.dumps(queue,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if queue_sha!=QUEUE_SHA:raise ValueError('frozen queue SHA mismatch')
    return dict(version='p4-source-compat-v1',artifact_class='GENERATED',evidence_class='CPU_ENGINEERING_ONLY / COLAB_UNVERIFIED',
                **{'from':dict(commit=full,sources=before),'to':dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                status='UNMERGED / NOT_DEPLOYED; hashes bind working bytes',sources=to_sources)},
                differences=changes,invariants=invariants,queue_sha256=queue_sha,matrix=matrix,budget=dict(BUDGET),research_budget=contract['budget'])


def same(label,expected,actual):
    if type(expected) is not type(actual) or json.dumps(expected,sort_keys=True,allow_nan=False)!=json.dumps(actual,sort_keys=True,allow_nan=False):
        raise ValueError(f'STOP: {label}; 預期={expected!r}; 實際={actual!r}; 下一步=保留舊帳本／02_formal ZIP，核對交付與宣告；使用原 notebook 接續或重新審查相容性，勿重置結果')


def original_manifest(root,binding):
    """Read only bounded manifest bytes; no extraction/import of historical code."""
    candidates=sorted(root.parent.glob('*/p4a-bundle.zip'))
    for archive in candidates:
        with zipfile.ZipFile(archive) as z:
            infos=[i for i in z.infolist() if i.filename=='bundle-manifest.json']
            if len(infos)!=1 or infos[0].file_size>MAX_MANIFEST_BYTES:continue
            payload=z.read(infos[0])
        if identity(payload)==binding['bundle']:return json.loads(payload)
    same('舊 bundle manifest 缺失（搜尋 '+str(candidates)+'）',binding['bundle'],'not found')


def resolve(bundle,mf,binding,root,budget,allow_new=False):
    try:
        return _resolve(bundle,mf,binding,root,budget,allow_new)
    except (OSError,KeyError,TypeError,ValueError,zipfile.BadZipFile) as error:
        if isinstance(error,ValueError) and str(error).startswith('STOP:'):raise
        raise ValueError(f'STOP: 升級證據無法讀取／schema 不符；預期=完整宣告與原 ZIP；實際={error}; 下一步=保留 budget 與 02_formal，修復交付或使用原 notebook 接續') from error


def _resolve(bundle,mf,binding,root,budget,allow_new=False):
    """Return original binding plus audit event; never silently restart a lineage."""
    from .p4_colab import hashed
    from datetime import datetime,timezone
    root=Path(root);bundle=Path(bundle)
    if allow_new:return binding,None
    directories=sorted(root.glob('run-*'))
    for directory in directories:
        same('run identity present '+str(directory),True,not directory.is_symlink() and (directory/'run_identity.json').is_file())
    runs=[directory/'run_identity.json' for directory in directories]
    if not runs:return binding,None
    current_hash=hashed(binding)
    exact=root/f'run-{current_hash}'/'run_identity.json'
    if exact.exists():
        same('existing run identity',binding,json.loads(exact.read_text()))
        return binding,None
    names=[n for n in mf['files'] if n.startswith('source/tasks/p4a-source-compat-') and n.endswith('.json')]
    same('來源相容宣告檔數量',1,len(names))
    name=names[0];path=bundle/name
    same('宣告檔 hash',mf['files'][name],digest(path))
    d=json.loads(path.read_text())
    same('declaration schema','p4-source-compat-v1',d.get('version'))
    sources={n:i for n,i in mf['files'].items() if source_name(n)}
    # Report paths individually so a mismatching file is directly actionable.
    same('to source path set',sorted(d['to']['sources']),sorted(sources))
    for n,i in sources.items():same('to '+n,d['to']['sources'][n],i)
    differences={n for n in d['from']['sources'].keys()|sources.keys() if d['from']['sources'].get(n)!=sources.get(n)}
    same('difference path set',sorted(differences),sorted(x['path'] for x in d['differences']))
    for item in d['differences']:
        n=item['path'];same('diff before '+n,d['from']['sources'].get(n),item['before']);same('diff after '+n,sources.get(n),item['after'])
        same('difference reason '+n,True,bool(item.get('reason')) and bool(item.get('diff')) and bool(item['categories']) and set(item['categories'])<={'進度顯示','日誌','speed-test 帳本分離','文件'})
    for n,i in d['invariants'].items():
        target='requirements.lock.txt' if n==LOCK else 'source/'+n
        same('invariant '+n,i,mf['files'].get(target))
    same('invariant path set',sorted((CONTRACT,RUNTIME,LOCK)),sorted(d['invariants']))
    same('budget',d['budget'],budget)
    same('research budget',d['research_budget'],binding['contract'].get('budget'))
    same('contract SHA',d['invariants'][CONTRACT]['sha256'],hashlib.sha256((bundle/('source/'+CONTRACT)).read_bytes()).hexdigest())
    same('matrix DATASET_MANIFEST',d['matrix'],binding['matrix'])
    queue_sha=hashlib.sha256(json.dumps(binding['queue'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    same('declaration queue',QUEUE_SHA,d['queue_sha256']);same('runtime queue',QUEUE_SHA,queue_sha)
    same('formal mode',False,binding['speed_test'])
    matches=[]
    for run in runs:
        old=json.loads(run.read_text())
        same('run directory hash '+str(run),run.parent.name,'run-'+hashed(old))
        historical=original_manifest(root,old)
        old_sources={n:i for n,i in historical['files'].items() if source_name(n)}
        same('from source path set',sorted(d['from']['sources']),sorted(old_sources))
        for n,i in d['from']['sources'].items():same('from '+n,i,old_sources[n])
        for k in ('contract','queue','matrix','environment','speed_test'):same('lineage '+k,old[k],binding[k])
        # No reference/data/non-source inputs may drift while adopting a run.
        old_inputs={n:i for n,i in historical['files'].items() if not source_name(n)}
        new_inputs={n:i for n,i in mf['files'].items() if not n.startswith('source/')}
        same('non-source inputs',old_inputs,new_inputs)
        matches.append(old)
    same('unique matching old lineage',1,len(matches))
    event=dict(event='SOURCE_UPGRADE',from_sources_hash=hashed(d['from']['sources']),to_sources_hash=hashed(sources),
               from_sources=d['from']['sources'],to_sources=sources,declaration=digest(path),declaration_path=name,
               observed_at=datetime.now(timezone.utc).isoformat(),environment=binding['environment'],budget=budget,
               from_binding=matches[0],to_binding=binding)
    return matches[0],event


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-commit',required=True)
    parser.add_argument('--dataset-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=declaration(args.from_commit,current_sources(),digest(args.dataset_manifest))
    with args.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,ensure_ascii=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    print(json.dumps(dict(path=str(args.output),identity=digest(args.output),matrix=result['matrix'])))


if __name__=='__main__':main()
