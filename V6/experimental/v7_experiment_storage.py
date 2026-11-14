"""Durable bounded experiment checkpoints and prepared-matrix verification."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import torch

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def verify_matrix(directory, *, allow_diagnostic=False):
    root = Path(directory)
    marker = root / ".prepare-complete.json"
    if not marker.is_file():
        raise ValueError("完整矩陣缺少完成標記；請先用原本組裝 Notebook 完成 Cell 4。")
    saved = json.loads(marker.read_text())
    required = {"features_59.parquet", "feature_metadata.json", "splits.json",
                "knowledge_graph_v2_csr.npz", "market_prices_raw.parquet", "data_health.json"}
    if not required <= saved.get("files", {}).keys():
        raise ValueError("矩陣完成標記不完整")
    for name, sha in saved["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != sha:
            raise ValueError("矩陣校驗失敗：" + name)
    metadata = json.loads((root / "feature_metadata.json").read_text())
    if not allow_diagnostic and metadata.get("artifact_kind") != "full-prepared-candidate":
        raise ValueError("正式實驗不能使用診斷子集")
    if metadata.get("parquet_sha256") != saved["files"]["features_59.parquet"]:
        raise ValueError("矩陣與 metadata 指紋不一致")
    return saved

class CheckpointStore:
    """Commit a small pointer only after a new immutable payload is durable.

    Keep latest, previous and best generations. A killed copy cannot replace
    the committed checkpoint; digest failure can recover the previous one.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / "checkpoint.json"

    def manifest(self):
        return json.loads(self.pointer.read_text()) if self.pointer.exists() else {}

    def save(self, state, *, best=False):
        old = self.manifest()
        name = "state-" + uuid.uuid4().hex + ".pt"
        dest = self.root / name
        with tempfile.TemporaryDirectory(prefix="v7-checkpoint-") as tmp:
            local = Path(tmp) / name
            torch.save(state, local)
            sha = digest(local)
            with local.open("rb") as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out, 4 * 1024**2)
                out.flush()
                os.fsync(out.fileno())
            if digest(dest) != sha:
                raise OSError("checkpoint copy verification failed")
        entry = {"file": name, "sha256": sha}
        updated = {"latest": entry, "previous": old.get("latest"),
                   "best": entry if best else old.get("best")}
        atomic_json(self.pointer, updated)
        keep = {v["file"] for v in updated.values() if v}
        for path in self.root.glob("state-*.pt"):
            if path.name not in keep:
                path.unlink()
        return entry

    def load(self, expected_identity, *, best=False):
        manifest = self.manifest()
        entries = [manifest.get("best")] if best else [manifest.get("latest"), manifest.get("previous")]
        for entry in entries:
            if not entry:
                continue
            path = self.root / entry["file"]
            if not path.is_file() or digest(path) != entry["sha256"]:
                continue
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state.get("identity") != expected_identity:
                raise ValueError("實驗、資料、程式或環境已改變，不能接續舊 checkpoint。請使用新的實驗資料夾。")
            return state
        if manifest:
            raise OSError("沒有可驗證的 checkpoint；保留現有檔案供診斷")
        return None
