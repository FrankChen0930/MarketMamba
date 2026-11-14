"""Metadata-only Mamba-2 wheel preflight; never installs or compiles anything."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import sysconfig
import zipfile
from email.parser import Parser
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeMetadata:
    python: str
    python_tag: str
    torch: str
    cuda: str | None
    cxx11_abi: bool | None
    platform: str


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def _normalized_linux_platform(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    if "linux" not in normalized:
        raise PreflightError(f"unsupported wheel platform {value!r}; Linux is required")
    if "x86_64" in normalized or "amd64" in normalized:
        return "linux_x86_64"
    if "aarch64" in normalized or "arm64" in normalized:
        return "linux_aarch64"
    raise PreflightError(f"unsupported Linux wheel platform architecture: {value!r}")


def _cuda_release_tag(value: str | None) -> str:
    if not value:
        return ""
    major = value.split(".", 1)[0]
    if not major.isdigit():
        raise PreflightError(f"invalid CUDA build metadata: {value!r}")
    return major


def collect_runtime_metadata() -> RuntimeMetadata:
    """Read package metadata only; CUDA/ABI must be supplied by the runtime record."""
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreflightError("PyTorch package metadata is unavailable") from exc
    return RuntimeMetadata(
        python=platform.python_version(),
        python_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
        torch=_base_version(torch_version),
        cuda=None,
        cxx11_abi=None,
        platform=sysconfig.get_platform(),
    )


def collect_colab_runtime_metadata() -> RuntimeMetadata:
    """Explicit setup-only metadata capture; never calls CUDA availability APIs."""
    import torch
    return RuntimeMetadata(python=platform.python_version(),
        python_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
        torch=_base_version(torch.__version__), cuda=torch.version.cuda,
        cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI), platform=sysconfig.get_platform())


def load_runtime_metadata(path: str | Path | None) -> RuntimeMetadata | None:
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return RuntimeMetadata(**value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise PreflightError(f"cannot read runtime metadata: {exc}") from exc


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    required = {"filename", "python_tag", "torch", "cuda", "cxx11_abi", "platform", "sha256"}
    missing = sorted(required - candidate.keys())
    if missing:
        raise PreflightError(f"wheel metadata missing fields: {', '.join(missing)}")
    filename = str(candidate["filename"])
    if "/" in filename or "\\" in filename or not filename.endswith(".whl"):
        raise PreflightError("wheel metadata filename must be a local wheel basename")
    digest = str(candidate["sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
        raise PreflightError("wheel metadata sha256 must be a 64-character hexadecimal digest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wheel_requirements(path: Path, runtime: RuntimeMetadata | None = None) -> set[str]:
    try:
        with zipfile.ZipFile(path) as wheel:
            metadata_files = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_files) != 1: raise PreflightError("wheel must contain exactly one METADATA record")
            message = Parser().parsestr(wheel.read(metadata_files[0]).decode("utf-8"))
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise PreflightError(f"cannot inspect real wheel METADATA: {exc}") from exc
    if message.get("Name", "").lower().replace("_", "-") != "mamba-ssm" or message.get("Version") != "2.3.2.post1":
        raise PreflightError("wheel METADATA must identify mamba-ssm==2.3.2.post1 exactly")
    requirements: set[str] = set()
    marker_environment = default_environment()
    if runtime is not None:
        marker_environment.update({
            "python_full_version": runtime.python,
            "python_version": ".".join(runtime.python.split(".")[:2]),
            "sys_platform": "linux",
            "platform_system": "Linux",
            "platform_machine": _normalized_linux_platform(runtime.platform).removeprefix("linux_"),
        })
    for value in message.get_all("Requires-Dist", []):
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise PreflightError(f"invalid wheel Requires-Dist entry {value!r}: {exc}") from exc
        if requirement.marker is None or requirement.marker.evaluate(marker_environment):
            requirements.add(requirement.name.lower().replace("_", "-"))
    missing = {"torch", "einops", "transformers"} - requirements
    if missing: raise PreflightError(f"wheel METADATA omits runtime requirements: {sorted(missing)}")
    return requirements


def preflight(manifest_path: str | Path, runtime: RuntimeMetadata | None = None) -> dict[str, Any]:
    """Select compatible declared metadata or fail before any install/compiler call."""
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read environment manifest: {exc}") from exc
    if manifest.get("mamba_ssm_pin") != "2.3.2.post1":
        raise PreflightError("manifest must pin mamba-ssm==2.3.2.post1")
    candidates = manifest.get("wheel_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PreflightError("compatible wheel metadata is absent; verify exact release wheel availability first")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise PreflightError("wheel metadata entries must be objects")
        _validate_candidate(candidate)
    runtime = runtime or collect_runtime_metadata()
    matches = [candidate for candidate in candidates if (
        candidate["python_tag"] == runtime.python_tag
        and _base_version(str(candidate["torch"])) == runtime.torch
        and candidate["cuda"] == runtime.cuda
        and candidate["cxx11_abi"] == runtime.cxx11_abi
        and _normalized_linux_platform(str(candidate["platform"])) == _normalized_linux_platform(runtime.platform)
    )]
    if len(matches) != 1:
        raise PreflightError(f"expected exactly one compatible wheel metadata entry, found {len(matches)}")
    wheel_path = path.parent / matches[0]["filename"]
    if not wheel_path.is_file():
        raise PreflightError("compatible wheel metadata found, but the declared local wheel is absent")
    if _sha256(wheel_path) != str(matches[0]["sha256"]).lower():
        raise PreflightError("declared local wheel SHA-256 does not match")
    requirements = _wheel_requirements(wheel_path, runtime)
    missing = [name for name in ("mamba_ssm", "selective_scan_cuda", "einops", "transformers", "torch_geometric")
               if importlib.util.find_spec(name) is None]
    return {"status": "compatible-metadata", "runtime": asdict(runtime), "missing_packages": missing,
            "wheel": matches[0], "wheel_path": str(wheel_path.resolve()),
            "wheel_requires_dist": sorted(requirements),
            "dependency_note": "selective_scan_cuda is required by release import; causal-conv1d is optional for no-conv SSD; transformers is required by eager generation imports"}


_OFFICIAL_WHEEL_PATTERN = re.compile(
    r"mamba_ssm-2\.3\.2\.post1\+cu(?P<cu>\d+)torch(?P<torch>\d+\.\d+)"
    r"cxx11abi(?P<abi>TRUE|FALSE)-(?P<py>cp\d+)-[^-]+-(?P<platform>[^.]+)\.whl$"
)


def select_official_release_asset(runtime: RuntimeMetadata,
                                  assets: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select one exact pinned binary for Python/Torch/CUDA-major/ABI/architecture."""
    runtime_platform = _normalized_linux_platform(runtime.platform)
    runtime_cuda = _cuda_release_tag(runtime.cuda)
    matches: list[Mapping[str, Any]] = []
    for asset in assets:
        match = _OFFICIAL_WHEEL_PATTERN.fullmatch(str(asset.get("name", "")))
        if (match and match["py"] == runtime.python_tag and match["cu"] == runtime_cuda
                and runtime.torch.startswith(match["torch"] + ".")
                and (match["abi"] == "TRUE") == runtime.cxx11_abi
                and _normalized_linux_platform(match["platform"]) == runtime_platform):
            matches.append(asset)
    if len(matches) != 1:
        available = [asset.get("name") for asset in assets
                     if str(asset.get("name", "")).endswith(".whl")]
        raise PreflightError("no unique official prebuilt wheel matches runtime platform; "
                             "source build is forbidden; "
                             f"runtime={asdict(runtime)}, release_wheels={available}")
    return matches[0]


def setup_colab(manifest_path: Path, runtime_path: Path, wheel_dir: Path, *, install: bool = False) -> dict[str, Any]:
    """Explicit network/install action for a future Colab cell; official pinned release only."""
    import subprocess
    import urllib.request
    if manifest_path.resolve().parent != wheel_dir.resolve():
        raise PreflightError("writable manifest copy and wheel-dir must be the same directory")
    runtime = collect_colab_runtime_metadata()
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(asdict(runtime), indent=2, sort_keys=True), encoding="utf-8")
    api = "https://api.github.com/repos/state-spaces/mamba/releases/tags/v2.3.2.post1"
    request = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "MarketMamba-V7-preflight"})
    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    asset = select_official_release_asset(runtime, release.get("assets", []))
    wheel_dir.mkdir(parents=True, exist_ok=True); wheel_path = wheel_dir / asset["name"]
    urllib.request.urlretrieve(asset["browser_download_url"], wheel_path)
    downloaded = _sha256(wheel_path); published = asset.get("digest")
    if published and published != f"sha256:{downloaded}":
        raise PreflightError("downloaded wheel does not match published release asset SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["wheel_candidates"] = [{"filename": wheel_path.name, "python_tag": runtime.python_tag,
        "torch": runtime.torch, "cuda": runtime.cuda, "cxx11_abi": runtime.cxx11_abi,
        "platform": _normalized_linux_platform(runtime.platform),
        "sha256": downloaded, "published_digest": published, "official_asset_url": asset["browser_download_url"]}]
    manifest["resolved_runtime_json"] = str(runtime_path)
    manifest["resolved_wheel_directory"] = str(wheel_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    result = preflight(manifest_path, runtime)
    if install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--only-binary=:all:", str(wheel_path)], check=True)
        result["installed"] = True
    else:
        result["installed"] = False
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runtime-metadata", type=Path,
                        help="JSON with python/python_tag/torch/cuda/cxx11_abi/platform; avoids importing torch")
    parser.add_argument("--setup-colab", action="store_true",
                        help="explicitly query/download the pinned official release asset")
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--install", action="store_true",
                        help="after compatibility+digest+METADATA checks, install binary wheels only")
    args = parser.parse_args(argv)
    try:
        if args.setup_colab:
            if args.runtime_metadata is None or args.wheel_dir is None:
                raise PreflightError("setup-colab requires runtime-metadata output path and wheel-dir")
            result = setup_colab(args.manifest, args.runtime_metadata, args.wheel_dir, install=args.install)
        else:
            runtime = load_runtime_metadata(args.runtime_metadata)
            result = preflight(args.manifest, runtime)
        print(json.dumps(result, indent=2, sort_keys=True))
    except PreflightError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
