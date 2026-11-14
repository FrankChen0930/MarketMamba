"""Real CPU forward/backward and new-process resume, owned synthetic bytes only."""
from pathlib import Path
import contextlib,fcntl,io,json,subprocess,sys,tempfile,unittest
import torch
from V6.experimental.v7_cpu_resume import make_fixture,run_probe,semantic_state,read_state,make_binding,ProofStore
from V6.experimental.v7_e1_checkpoint import sha256,atomic_json


class CPUResumeTest(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.matrix=self.base/'matrix';make_fixture(self.matrix)
    def tearDown(self):self.temp.cleanup()
    def child(self,root,mode='fresh',cut=None,**extra):
        args=[sys.executable,'-B','-m','V6.experimental.v7_cpu_resume','--matrix',str(self.matrix),'--output',str(root),'--mode',mode]
        if cut is not None:args+=['--cut-step',str(cut)]
        for key,value in extra.items():args+=['--'+key.replace('_','-'),str(value)]
        return subprocess.run(args,capture_output=True,text=True)
    def test_real_mid_epoch_new_process_exact_state(self):
        a=self.base/'a';b=self.base/'b'
        reference=self.child(a);self.assertEqual(reference.returncode,0,reference.stderr)
        interrupted=self.child(b,cut=4);self.assertEqual(interrupted.returncode,75,interrupted.stderr)
        before=read_state(b,self.matrix);self.assertEqual(before['step'],4);self.assertEqual(before['batch'],1)
        self.assertEqual(before['sampler']['policy'],'DETERMINISTIC_DAILY_ORDER_NO_WORKERS')
        resumed=self.child(b,'resume');self.assertEqual(resumed.returncode,0,resumed.stderr)
        x,y=read_state(a,self.matrix),read_state(b,self.matrix)
        self.assertEqual(semantic_state(x),semantic_state(y));self.assertTrue(y['terminal'])
        self.assertEqual([r['loss'] for r in x['step_trace']],[r['loss'] for r in y['step_trace']])
        self.assertEqual([r['lr_after'] for r in x['step_trace']],[r['lr_after'] for r in y['step_trace']])
        self.assertGreater(x['step'],4)
    def test_missing_checkpoint_and_wrong_identity_fail_without_fresh(self):
        missing=self.child(self.base/'missing','resume');self.assertNotEqual(missing.returncode,0)
        root=self.base/'run';self.assertEqual(self.child(root,cut=4).returncode,75)
        wrong=self.child(root,'resume',run_id='wrong-run');self.assertNotEqual(wrong.returncode,0);self.assertIn('RUN_BINDING_MISMATCH',wrong.stdout)
    def test_corrupt_latest_uses_valid_previous(self):
        root=self.base/'run';self.assertEqual(self.child(root,cut=4).returncode,75)
        pointer=json.loads((root/'checkpoints/checkpoint.json').read_text());latest=root/'checkpoints'/pointer['latest']['file'];latest.write_bytes(b'corrupt')
        res=self.child(root,'resume');self.assertEqual(res.returncode,0,res.stderr)
        reference=self.base/'reference';self.assertEqual(self.child(reference).returncode,0)
        self.assertEqual(semantic_state(read_state(root,self.matrix)),semantic_state(read_state(reference,self.matrix)))
        self.assertEqual(json.loads((root/'checkpoints/checkpoint.json').read_text())['best']['phase'],'train')
    def test_duplicate_resume_or_active_process_is_rejected(self):
        root=self.base/'run';self.assertEqual(self.child(root,cut=4).returncode,75)
        with (root/'resume.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=self.child(root,'resume');self.assertNotEqual(result.returncode,0);self.assertIn('DUPLICATE_ACTIVE_ATTEMPT',result.stdout)
        self.assertEqual(self.child(root,'resume').returncode,0)
        again=self.child(root,'resume');self.assertNotEqual(again.returncode,0);self.assertIn('ALREADY_COMPLETED',again.stdout)
    def test_broken_log_and_interrupted_publication_not_success(self):
        root=self.base/'broken';root.mkdir();(root/'events.jsonl').mkdir()
        result=self.child(root);self.assertNotEqual(result.returncode,0);self.assertIn('FAILED',result.stdout)
        root=self.base/'partial';result=self.child(root,fault='checkpoint')
        self.assertNotEqual(result.returncode,0);self.assertFalse((root/'checkpoints/checkpoint.json').exists())
        self.assertNotEqual(self.child(root,'resume').returncode,0)
    def test_missing_sampler_invalid_scheduler_and_incomplete_runtime_rejected(self):
        for mode in ('sampler','scheduler','optimizer'):
            root=self.base/mode;self.assertEqual(self.child(root,cut=4).returncode,75)
            pointer=json.loads((root/'checkpoints/checkpoint.json').read_text());seen=set()
            for role in ('latest','previous','best'):
                entry=pointer.get(role)
                if not entry or entry['file'] in seen:continue
                seen.add(entry['file']);path=root/'checkpoints'/entry['file'];state=torch.load(path,weights_only=True,map_location='cpu')
                if mode=='sampler':state['payload'].pop('sampler',None)
                elif mode=='scheduler':state['payload']['scheduler_state']['last_epoch']=-100
                else:state['payload'].pop('optimizer_state',None)
                torch.save(state,path)
            for entry in pointer.values():
                if isinstance(entry,dict) and 'file' in entry:
                    path=root/'checkpoints'/entry['file'];entry.update(sha256=sha256(path),bytes=path.stat().st_size)
            atomic_json(root/'checkpoints/checkpoint.json',pointer)
            result=self.child(root,'resume');self.assertNotEqual(result.returncode,0);self.assertIn('NO_VALID_COMPLETE_STATE',result.stdout)
    def test_data_or_model_binding_mismatch(self):
        root=self.base/'run';self.assertEqual(self.child(root,cut=4).returncode,75)
        binding=make_binding(self.matrix)
        wrong={**binding,'model_sha256':'0'*64}
        store=ProofStore(root/'checkpoints',root/'local',wrong)
        self.assertIsNone(store.load())
        (self.matrix/'X.npy').write_bytes(b'corrupt')
        self.assertNotEqual(self.child(root,'resume').returncode,0)

    def test_safe_codec_rejects_pickle_and_inconsistent_lr_rng_or_cursor(self):
        import copy
        from unittest.mock import patch
        root=self.base/'run';self.assertEqual(self.child(root,cut=4).returncode,75)
        binding=make_binding(self.matrix);store=ProofStore(root/'checkpoints',root/'local',binding)
        entry=json.loads(store.pointer.read_text())['latest'];original=torch.load(store.root/entry['file'],weights_only=True,map_location='cpu')
        mutations=[lambda p:p['scheduler_state'].update(_last_lr=[999.,999.]),lambda p:p['rng_state'].update(torch=torch.tensor([1],dtype=torch.uint8)),lambda p:p.update(epoch=100)]
        for n,mutate in enumerate(mutations):
            state=copy.deepcopy(original);mutate(state['payload']);path=self.base/('invalid-'+str(n)+'.pt');torch.save(state,path)
            with self.assertRaises(Exception):store._read_payload(path)
        path=self.base/'unsafe.pt';torch.save({'bad':set([object()])},path)
        with patch('torch.load',wraps=torch.load) as loader:
            with self.assertRaises(Exception):store._read_payload(path)
            self.assertTrue(all(c.kwargs.get('weights_only') is True for c in loader.call_args_list))

    def test_optimizer_moments_and_config_cannot_be_missing_or_replaced(self):
        import copy
        root=self.base/'optimizer';self.assertEqual(self.child(root,cut=4).returncode,75)
        store=ProofStore(root/'checkpoints',root/'local',make_binding(self.matrix));entry=json.loads(store.pointer.read_text())['latest']
        original=torch.load(store.root/entry['file'],map_location='cpu',weights_only=True)
        changes=[lambda o:o.update(state={}),lambda o:o['param_groups'][0].update(weight_decay=999),
                 lambda o:next(iter(o['state'].values())).update(exp_avg=torch.zeros(1)),
                 lambda o:next(iter(o['state'].values())).update(step=torch.tensor(1.)),
                 lambda o:o['param_groups'][0].update(params=[999])]
        for n,change in enumerate(changes):
            state=copy.deepcopy(original);change(state['payload']['optimizer_state']);path=self.base/('invalid-optim-'+str(n)+'.pt');torch.save(state,path)
            with self.assertRaises(Exception):store._read_payload(path)
if __name__=='__main__':unittest.main()
