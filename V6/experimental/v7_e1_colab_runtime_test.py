import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import subprocess
import time

from V6.experimental.v7_e1_colab_runtime import run_logged


class E1ColabRuntimeTest(unittest.TestCase):
 def test_reader_failure_stops_child_and_propagates_cause(self):
  for failure in ("write", "decode"):
   with self.subTest(failure=failure), tempfile.TemporaryDirectory() as d:
    root=Path(d); log=root/"failure.log"; children=[]
    popen=subprocess.Popen
    def launch(*args, **kwargs):
     child=popen(*args, **kwargs); children.append(child); return child
    real_open=Path.open
    class FailingLog:
     def __enter__(self):
      self.stream=real_open(log,"a",encoding="utf-8"); return self
     def __exit__(self,*args): self.stream.close()
     def write(self,text):
      if "evidence" in text: raise OSError("fixture disk unavailable")
      return self.stream.write(text)
     def flush(self): self.stream.flush()
    output="print('evidence',flush=True)" if failure=="write" else "import os;os.write(1,b'\\xff\\n')"
    patch=(mock.patch.object(Path,"open",return_value=FailingLog()) if failure=="write" else contextlib.nullcontext())
    started=time.monotonic()
    with patch, mock.patch("V6.experimental.v7_e1_colab_runtime.subprocess.Popen",side_effect=launch):
     with self.assertRaisesRegex(RuntimeError,"log stream failed.*failure.log") as raised:
      run_logged([sys.executable,"-u","-c",output+";import time;time.sleep(3)"],
                 log,cwd=root,poll_seconds=.005)
    self.assertIsInstance(raised.exception.__cause__,OSError if failure=="write" else UnicodeDecodeError)
    self.assertLess(time.monotonic()-started,2)
    self.assertIsNotNone(children[0].poll())

 def test_reader_failure_after_child_exit_never_returns_success(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with self.assertRaisesRegex(RuntimeError,"log stream failed"):
    run_logged([sys.executable,"-u","-c","import os;os.write(1,b'\\xff\\n')"],
               root/"failure.log",cwd=root,poll_seconds=.05)

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
