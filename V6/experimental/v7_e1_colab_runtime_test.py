import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

from V6.experimental.v7_e1_colab_runtime import run_logged


class E1ColabRuntimeTest(unittest.TestCase):
 def test_success_streams_output_and_persists_log(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); stdout=io.StringIO()
   with contextlib.redirect_stdout(stdout):
    path=run_logged(
     [sys.executable,"-u","-c","print(\"epoch=2 loss=0.25\")"],
     root/"seed-29.log",cwd=root,heartbeat_seconds=.01,poll_seconds=.005)
   self.assertEqual(path,root/"seed-29.log")
   self.assertIn("epoch=2 loss=0.25",stdout.getvalue())
   self.assertIn("epoch=2 loss=0.25",path.read_text())

 def test_failure_reports_tail_and_durable_log_path(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); log_path=root/"seed-29.log"; stdout=io.StringIO()
   with contextlib.redirect_stdout(stdout):
    with self.assertRaisesRegex(RuntimeError,"Full log: .*seed-29.log"):
     run_logged([sys.executable,"-u","-c","print(\"fatal detail\");raise SystemExit(7)"],
                log_path,cwd=root,poll_seconds=.005)
   self.assertIn("fatal detail",stdout.getvalue())
   self.assertIn("fatal detail",log_path.read_text())

 def test_evaluation_stage_labels_heartbeat(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); log_path=root/"evaluation.log"; stdout=io.StringIO()
   with contextlib.redirect_stdout(stdout):
    try:
     run_logged(
      [sys.executable,"-u","-c","import time;time.sleep(.04)"],
      log_path,cwd=root,stage="EVALUATION",heartbeat_seconds=.01,poll_seconds=.005)
    except TypeError as error:
     self.fail(str(error))
   self.assertIn("'stage': 'EVALUATION_HEARTBEAT'",stdout.getvalue())
   self.assertIn("'stage': 'EVALUATION_HEARTBEAT'",log_path.read_text())

 def test_evaluation_stage_labels_failure(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); log_path=root/"evaluation.log"
   try:
    with self.assertRaisesRegex(RuntimeError,"Evaluation child exited 7"):
     run_logged(
      [sys.executable,"-u","-c","raise SystemExit(7)"],
      log_path,cwd=root,stage="EVALUATION",poll_seconds=.005)
   except TypeError as error:
    self.fail(str(error))


if __name__=="__main__": unittest.main()
