"""Durable, content-verified V7 preparation checkpoints (no training state)."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path

REVISION = "v7-stock-checkpoints-v1"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def progress(directory, event, **fields):
    memory = {}
    for path, keys in [("/proc/self/status", ("VmRSS", "VmHWM")),
                       ("/proc/meminfo", ("MemAvailable",))]:
        try:
            for line in Path(path).read_text().splitlines():
                key = line.partition(":")[0]
                if key in keys:
                    memory[key + "_GiB"] = round(int(line.split()[1]) / 1024**2, 3)
        except OSError:
            pass
    entry = dict(time=time.time(), event=event, **memory, **fields)
    text = json.dumps(entry, ensure_ascii=False)
    print(text, flush=True)
    with (Path(directory) / "prepare-progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()

class StockCache:
    def __init__(self, output, identity):
        self.directory = Path(output) / ".prepare-cache"
        self.directory.mkdir(parents=True, exist_ok=True)
        state = self.directory / "identity.json"
        expected = {"revision": REVISION, "identity": identity}
        if state.exists():
            if json.loads(state.read_text()) != expected:
                raise ValueError("Preparation inputs/code changed; use a new output directory. Existing checkpoints preserved.")
        else:
            if any(self.directory.iterdir()):
                raise ValueError("Unidentified preparation cache; use a new output directory.")
            atomic_json(state, expected)

    def paths(self, stock):
        key = hashlib.sha256(str(stock).encode()).hexdigest()
        return self.directory / (key + ".parquet"), self.directory / (key + ".json")

    def valid(self, stock):
        data, marker = self.paths(stock)
        if not data.is_file() or not marker.is_file():
            return None
        try:
            doc = json.loads(marker.read_text())
            if doc["stock_id"] == str(stock) and doc["sha256"] == digest(data):
                return doc
        except (ValueError, KeyError, OSError):
            pass
        return None

    def write(self, stock, frame, alignment):
        data, marker = self.paths(stock)
        tmp = data.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, data)
        doc = {"stock_id": str(stock), "rows": len(frame),
               "sha256": digest(data), "alignment": alignment}
        atomic_json(marker, doc)
        return doc
