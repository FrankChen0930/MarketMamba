import json,tempfile,unittest
from pathlib import Path
from V6.experimental.v7_e1_checkpoint import *

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

if __name__=="__main__": unittest.main()
