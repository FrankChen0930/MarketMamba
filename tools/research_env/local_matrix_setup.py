"""Lay out locally downloaded E5/E1 matrix files for local runs; verify every file against the registered SHA-256.

Download from Drive (browser) into two folders, then:
  python tools/research_env/local_matrix_setup.py --e5-dir ~/Downloads/E5 --e1-dir ~/Downloads/E1 --out .local-data/matrices
E5 folder needs: X.npy stock_ids.npy dates.npy end_indices.npy y5.npy y10.npy splits.npy manifest.json
E1 folder needs only: y5.npy y10.npy splits.npy manifest.json (X/dates/stock_ids/end_indices are byte-identical to E5's and are linked).
Result: <out>/shared/{X,stock_ids,dates,end_indices}.npy and <out>/E5, <out>/E1 each a complete matrix directory (symlinks to shared).
"""
import argparse, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'V6'))
from research_lab import colab_runner as cr

SHARED = ('X.npy', 'stock_ids.npy', 'dates.npy', 'end_indices.npy')
OWN = ('y5.npy', 'y10.npy', 'splits.npy', 'manifest.json')


def setup(out, e5_dir, e1_dir=None, registry=cr.REGISTRY, move=False, log=print):
    out = Path(out); (out/'shared').mkdir(parents=True, exist_ok=True)
    plan = [('E5', Path(e5_dir))] + ([('E1', Path(e1_dir))] if e1_dir else [])
    for exp, src in plan:
        files = registry[exp]['files']
        manifest = json.loads((src/'manifest.json').read_text())
        cr.check_manifest(manifest, exp, registry)
        dst = out/exp; dst.mkdir(exist_ok=True)
        for name in OWN + (SHARED if exp == 'E5' else ()):
            if not (src/name).exists():
                raise FileNotFoundError(f'{exp}: missing {src/name}')
            if name == 'manifest.json':
                shutil.copyfile(src/name, dst/name); continue
            got = cr.sha256_file(src/name)
            if got != files[name]:
                raise cr.IdentityError(f'{exp}/{name}: sha256 {got[:12]} != expected {files[name][:12]}')
            target = (out/'shared'/name) if name in SHARED else (dst/name)
            if not target.exists():
                (shutil.move if move else shutil.copyfile)(src/name, target)
            log(f'verified {exp}/{name}')
        for name in SHARED:
            link = dst/name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.relpath(out/'shared'/name, dst), link)
    return {exp: out/exp for exp, _ in plan}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--e5-dir', required=True); ap.add_argument('--e1-dir'); ap.add_argument('--out', required=True)
    ap.add_argument('--move', action='store_true', help='move instead of copy (saves disk)')
    a = ap.parse_args()
    print(setup(a.out, a.e5_dir, a.e1_dir, move=a.move))
