"""Daily point-in-time observation of external public datasets kept as git clones.

Research-only. For each configured clone under .artifacts/external/, fast-forward
pull and append one observation line (our observed_at, upstream HEAD, upstream
commit time, and file hashes) to observations.jsonl. Upstream git history keeps
every version; the observation log records *when we could first see* a version,
which is what point-in-time use needs. Third-party data never enters git here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / '.artifacts/external'
SOURCES = {
    'shioaji-data': ['v1/tw/relations/edges.json', 'v1/tw/companies/cards.json', 'v1/tw/themes/themes.json'],
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def observe(name: str, repo: Path, files: list[str], pull: bool = True) -> dict:
    before = _git(repo, 'rev-parse', 'HEAD')
    if pull:
        _git(repo, 'pull', '--ff-only', '--quiet')
    head = _git(repo, 'rev-parse', 'HEAD')
    return dict(
        source=name,
        observed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        head=head,
        head_committed_at=_git(repo, 'log', '-1', '--format=%cI'),
        changed=head != before,
        files={f: _sha(repo / f) for f in files},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external', type=Path, default=EXTERNAL)
    p.add_argument('--no-pull', action='store_true')
    args = p.parse_args(argv)
    log = args.external / 'observations.jsonl'
    failed = 0
    for name, files in SOURCES.items():
        repo = args.external / name
        try:
            rec = observe(name, repo, files, pull=not args.no_pull)
        except (subprocess.CalledProcessError, OSError) as exc:
            rec = dict(source=name, observed_at=dt.datetime.now(dt.timezone.utc).isoformat(), error=str(exc)[:500])
            failed += 1
        with log.open('a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(json.dumps({k: rec.get(k) for k in ('source', 'head', 'changed', 'error')}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
