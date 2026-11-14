# Window11 CPU overlay (engineering only)

Existing Colab lock is CUDA-specific; new purpose-bounded overlay preserves existing
CPU torch/numpy/scipy instead of reinstalling/upgrading them. Python3.12.3 venv
`.runtimes/window11-cpu` has a .pth to the retained CPUbase at
`/home/frank/projects/MarketMamba/.worktrees/v7-e1-colab-recovery/.venv/lib/python3.12/site-packages`.
Only five packages installed locally, exact requirements.overlay.lock + official
PyPI JSON/file SHA proof in wheel-provenance.json. wheels cached under
`.runtimes/window11-wheels` (GENERATED; not GC). Not a fully self-contained environment.

Recreate only in a new isolated venv: create python3 venv, add .pth of exact retained
CPUbase, then pip install --no-index --find-links <cache> --no-deps --require-hashes
-r requirements.overlay.lock. Trusted retrieval URLs/hash in provenance if cache
unavailable; network/install requires applicable authorization. Do not update global
Python/Torch/CUDA/Windows. Verify all listed versions and CPUdevice before fitting.

scikit-learn1.6.1/joblib1.6.0/threadpoolctl3.6.0/cloudpickle3.1.2 reused existing repo
Colab lock versions. Actual joblib missing-cloudpickle import diagnosed then fixed;
this explicit transitive dependency is now pinned. LightGBM4.6.0 official PyPI release:
https://pypi.org/project/lightgbm/4.6.0/ . Hashes verify downloaded bytes, not package
security audit or GPU certification. Official wheels downloaded ~17MB total.
