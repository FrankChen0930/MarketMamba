"""Small Colab process wrapper with live output and a durable append-only log."""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def run_logged(command,log_path,*,cwd,stage="TRAINING",heartbeat_seconds=30.,poll_seconds=2.):
 """Run a child, mirror every line to stdout, and preserve evidence on Drive."""
 log_path=Path(log_path); log_path.parent.mkdir(parents=True,exist_ok=True)
 tail=deque(maxlen=40); lock=threading.Lock()
 with log_path.open("a",encoding="utf-8",buffering=1) as log:
  log.write("\n=== START "+time.strftime("%Y-%m-%d %H:%M:%S")+" ===\n"); log.flush()
  process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,bufsize=1,start_new_session=True)
  def stream():
   assert process.stdout is not None
   for line in process.stdout:
    with lock:
     log.write(line); log.flush(); tail.append(line.rstrip())
    print(line.rstrip(),flush=True)
  reader=threading.Thread(target=stream,daemon=True); reader.start(); last=time.monotonic()
  try:
   while process.poll() is None:
    if time.monotonic()-last>=heartbeat_seconds:
     message={"stage":stage+"_HEARTBEAT","status":"RUNNING","log":str(log_path)}
     print(message,flush=True)
     with lock:
      log.write(str(message)+"\n"); log.flush()
     last=time.monotonic()
    time.sleep(poll_seconds)
   process.wait()
  except BaseException:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGKILL); process.wait()
   raise
  finally:
   reader.join(timeout=10)
   if process.stdout is not None: process.stdout.close()
 if process.returncode:
  print("\n".join(tail),flush=True)
  raise RuntimeError(stage.title()+" child exited "+str(process.returncode)+". Full log: "+str(log_path))
 return log_path
