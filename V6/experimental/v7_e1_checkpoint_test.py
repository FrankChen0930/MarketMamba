import json,tempfile,unittest
import multiprocessing
from unittest.mock import patch
from pathlib import Path
from V6.experimental.v7_e1_checkpoint import *

def _unpublished_checkpoint(root, connection):
 import V6.experimental.v7_e1_checkpoint as checkpoint
 store=DurableCheckpointStore(Path(root)/"drive",Path(root)/"child-local","c"*64,"m"*64,17)
 def stop_before_pointer(path,value):
  connection.send("body-durable")
  connection.recv()
 checkpoint.atomic_json=stop_before_pointer
 store.save(checkpoint_state("c"*64,"m"*64,17,step=3))

class E1CheckpointTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); root=Path(self.tmp.name)
  self.drive=root/"drive"; self.local=root/"local"
  self.store=DurableCheckpointStore(self.drive,self.local,"c"*64,"m"*64,17)
 def tearDown(self): self.tmp.cleanup()
 def state(self,step=1,**kw):
  return checkpoint_state("c"*64,"m"*64,17,step=step,**kw)
 def test_latest_previous_best_are_loadable(self):
  a=self.store.save(self.state(1),best=True)
  b=self.store.save(self.state(2))
  self.assertEqual(self.store.load()["step"],2)
  self.assertEqual(self.store.load("best")["step"],1)
  self.assertEqual(self.store.manifest()["previous"]["file"],a["file"])
  self.assertEqual(self.store.manifest()["latest"]["file"],b["file"])
 def test_body_interrupt_never_advances_pointer(self):
  self.store.save(self.state(1),best=True); before=self.store.manifest()
  for point in ("after_local_body","before_drive_flush","before_pointer"):
   with self.subTest(point=point):
    with self.assertRaises(RuntimeError): self.store.save(self.state(2),interrupt_at=point)
    self.assertEqual(self.store.manifest(),before)
    self.assertEqual(self.store.load()["step"],1)
 def test_missing_latest_and_corrupt_best_fall_back_safely(self):
  self.store.save(self.state(1),best=True); self.store.save(self.state(2))
  latest=self.store.manifest()["latest"]; (self.drive/latest["file"]).unlink()
  self.assertEqual(self.store.load()["step"],1)
  best=self.store.manifest()["best"]; (self.drive/best["file"]).write_bytes(b"bad")
  self.assertIsNone(self.store.load("best"))
 def test_valid_orphan_recovery_never_promotes_best(self):
  self.store.save(self.state(1),best=True)
  with self.assertRaises(RuntimeError):
   self.store.save(self.state(3),interrupt_at="before_pointer")
  candidates=self.store.scan_orphans()
  self.assertEqual(candidates[0]["status"],"RECOVERABLE_ORPHAN_CANDIDATE")
  old_best=self.store.manifest()["best"]
  self.store.recover_latest_orphan()
  self.assertEqual(self.store.load()["step"],3)
  self.assertEqual(self.store.manifest()["best"],old_best)
  self.assertFalse(self.store.manifest()["recovery"]["best_promoted"])
 def test_corrupt_pointer_can_recover_valid_body(self):
  self.store.save(self.state(4)); self.store.pointer.write_text("{")
  self.assertIsNone(self.store.load())
  self.assertEqual(self.store.recover_latest_orphan()["step"],4)
  self.assertEqual(self.store.load()["step"],4)
 def test_wrong_seed_matrix_or_contract_is_not_recoverable(self):
  self.store.save(self.state(1))
  for identity in (("x"*64,"m"*64,17),("c"*64,"x"*64,17),("c"*64,"m"*64,29)):
   other=DurableCheckpointStore(self.drive,self.local,*identity)
   self.assertIsNone(other.load()); self.assertEqual(other.scan_orphans(),[])
 def test_terminal_and_phase_must_agree(self):
  with self.assertRaises(ValueError):
   self.state(1,phase="finished",terminal=False)
 def test_malformed_latest_falls_back_without_mutating_pointer(self):
  self.store.save(self.state(1),best=True); self.store.save(self.state(2))
  original=self.store.manifest(); latest=original["latest"]
  cases=[42, "bad", ["file"], {"file":latest["file"],"sha256":latest["sha256"],"extra":1},
         {**latest,"file":42}, {**latest,"bytes":True}, {**latest,"bytes":-1},
         {**latest,"sha256":42}, {**latest,"sha256":"bad"}]
  for entry in cases:
   with self.subTest(entry=entry):
    atomic_json(self.store.pointer,{**original,"latest":entry})
    before=self.store.pointer.read_bytes()
    restarted=DurableCheckpointStore(self.drive,self.local,*self.store.identity)
    self.assertEqual(restarted.load()["step"],1)
    self.assertIsNone(restarted.load(allow_previous=False))
    self.assertEqual(restarted.load("best")["step"],1)
    self.assertEqual(self.store.pointer.read_bytes(),before)
    restarted.save(self.state(3))
    self.assertEqual(restarted.manifest()["previous"],original["previous"])
    (self.drive/restarted.manifest()["latest"]["file"]).unlink()
    self.assertEqual(restarted.load()["step"],1)
    # Preserve the original step2 body for the next malformed-pointer case.
    self.store.save(self.state(2))
    original=self.store.manifest(); latest=original["latest"]
 def test_checkpoint_entry_cannot_escape_root_or_follow_symlink(self):
  self.store.save(self.state(1),best=True); self.store.save(self.state(2))
  original=self.store.manifest(); latest=original["latest"]
  target=self.drive/latest["file"]
  outside=self.drive.parent/"outside.pt"; outside.write_bytes(target.read_bytes())
  (self.drive/"linked.pt").symlink_to(outside)
  for name in (str(outside),"../outside.pt","linked.pt"):
   with self.subTest(name=name):
    atomic_json(self.store.pointer,{**original,"latest":{**latest,"file":name}})
    self.assertEqual(self.store.load()["step"],1)
    self.assertIsNone(self.store.load(allow_previous=False))
 def test_unreadable_latest_falls_back(self):
  self.store.save(self.state(1),best=True); self.store.save(self.state(2))
  latest=self.store.manifest()["latest"]["file"]; real_hash=sha256
  def unreadable(path):
   if Path(path).name==latest: raise PermissionError("fixture unreadable body")
   return real_hash(path)
  with patch("V6.experimental.v7_e1_checkpoint.sha256",side_effect=unreadable):
   self.assertEqual(self.store.load()["step"],1)
 def test_kill_before_pointer_restart_and_corrupt_latest_recovery(self):
  self.store.save(self.state(1),best=True); self.store.save(self.state(2))
  original=self.store.manifest()
  atomic_json(self.store.pointer,{**original,"latest":42})
  context=multiprocessing.get_context("spawn"); parent,child=context.Pipe()
  process=context.Process(target=_unpublished_checkpoint,args=(self.drive.parent,child))
  process.start();child.close()
  try:
   self.assertTrue(parent.poll(15));self.assertEqual(parent.recv(),"body-durable")
   process.kill();process.join(10);self.assertFalse(process.is_alive())
   self.assertEqual(process.exitcode,-9)
   restarted=DurableCheckpointStore(self.drive,self.local,*self.store.identity)
   self.assertEqual(restarted.load()["step"],1)
   self.assertEqual(restarted.recover_latest_orphan()["step"],3)
   self.assertEqual(restarted.load()["step"],3)
   self.assertEqual(restarted.manifest()["best"],original["best"])
   self.assertEqual(restarted.manifest()["previous"],original["previous"])
   (self.drive/restarted.manifest()["latest"]["file"]).unlink()
   self.assertEqual(restarted.load()["step"],1)
  finally:
   if process.is_alive(): process.kill();process.join(10)
   parent.close()
 def test_epoch_snapshot_does_not_recount_prior_updates(self):
  adapter=LifecycleCheckpointAdapter(self.store)
  base={"epoch":0,"batch":1,"step":10,"phase":"train","terminal":False,
        "model_state":{},"optimizer_state":{},"scheduler_state":{},"rng_state":{}}
  adapter.save(dict(base)); self.assertEqual(adapter.snapshot_epoch()["updates"],10)
  base.update(epoch=1,batch=1,step=12)
  adapter.save(dict(base)); self.assertEqual(adapter.snapshot_epoch()["updates"],2)

if __name__=="__main__": unittest.main()
