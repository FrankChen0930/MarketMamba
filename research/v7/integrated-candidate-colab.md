# Colab runbook: V7 integrated candidate

Every command below is an explicit future Colab action. Local synthetic SSD tests and diagnostic preparation are integration checks, not official Mamba-2, GPU, H1, performance, or architecture-promotion evidence.

## 0. Mount Drive, deliver the approved source, and install base dependencies

Start from a fresh Colab runtime. Mount Drive, then copy a reviewed repository snapshot from Drive into ephemeral storage; do not assume `/content/MarketMamba` already exists. The snapshot must contain the repository's 16 supporting V6 source files as well as these candidate files, but no data, weights, or `.env`.

```python
from google.colab import drive
from pathlib import Path
import shutil, subprocess, sys
drive.mount('/content/drive')
source = Path('/content/drive/MyDrive/MarketMamba-source')
repo = Path('/content/MarketMamba')
assert (source / 'V6/experimental/v7_integrated_train.py').is_file(), source
if repo.exists():
    raise RuntimeError(f'refusing to overwrite existing delivery: {repo}')
shutil.copytree(source, repo)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
    'torch', 'pandas', 'numpy', 'pyarrow', 'scipy', 'packaging', 'requests', 'yfinance',
    'python-dotenv', 'torch-geometric'], check=True)
```

This base setup must finish before `prepare`: `marketmamba.data.__init__` eagerly imports the fetcher and therefore needs `requests` and `yfinance` even for local artifact preparation.

## 1. Prepare real artifacts

Use the raw `Data/processed_v6` directory and a new output directory. Never use `baseline_cache_v2/baseline_base_66d.parquet` as 59D input and never use `knowledge_graph_cache.npz`; preparation calls the V6 59D helpers and converts `knowledge_graph_v2.npz` itself.

Bounded declared-basket diagnostics keep the requested price window while retaining pre-window stock fundamental/revenue/dividend/cashflow and macro context needed by V6 point-in-time joins. Stock predicates still apply to stock-keyed context. Any stock or date restriction is automatically labeled `diagnostic-restricted-not-performance-evidence`, even if `--diagnostic` is omitted.

Use the notebook's explicit `SMOKE_CALENDAR=/content/smoke-calendar.json`: this is the supervisor-supplied real fixed seven-stock basket, with nonblank `universe_provenance`. The notebook reads its declared IDs and first/last calendar dates for diagnostic filters, so the date range and basket agree. Do not invent the seven IDs or use the full-market calendar for a restricted basket.

A full preparation omits diagnostic filters. Supply `--calendar` with independently expected ordered `trading_calendar`, a nonblank `provenance`, and `expected_universe` mapping each calendar date to its expected stock IDs, plus nonblank `universe_provenance`. The selected dataset is the UNION of all declared IDs over the chosen calendar, preserving valid rows even before a date-specific expected membership begins. Other instruments (including declared-out ETFs) are excluded with explicit row/stock counts. Every declared expected ID must have graph coverage; missing graph IDs block rather than shrink the denominator. A full-market universe must be supplied explicitly with an honest provenance; today’s listing snapshot cannot establish historical ordinary-stock membership. Do not derive either input from selected price rows. Diagnostics require a correspondingly bounded independent calendar and expected universe. Source date union is audit information only. Missing independent evidence blocks preparation; no download is performed.

The versioned `v7-quality-report-v1` report contains graded entries with severity, reason code, dates, stocks, denominator and threshold, plus a Traditional Chinese summary and uncertain market history. INFO/WARN continue; invalid numeric/OHLC observations are QUARANTINE without price repair. BLOCK applies to ambiguous schema/calendar, unknown coverage denominator, major outages or insufficient usable training data. The defaults are at least 50% expected stocks missing/invalid on any session (including zero valid observations), and at least two usable training dates with two targets in either horizon. Configure `--major-outage-fraction` and `--minimum-usable-days`; these conservative execution floors do not establish research adequacy. Valid mixed-source rows remain admitted without the earlier exchange-key exclusion.

```bash
python V6/experimental/v7_integrated_train.py prepare --calendar /content/v7-calendar.json --raw-dir /content/drive/MyDrive/MarketMamba/Data/processed_v6 --output-dir /content/drive/MyDrive/MarketMamba_V7/prepared-full --date-to 2026-06-02
python V6/experimental/v7_integrated_train.py data-check --feature-parquet /content/drive/MyDrive/MarketMamba_V7/prepared-full/features_59.parquet --feature-metadata /content/drive/MyDrive/MarketMamba_V7/prepared-full/feature_metadata.json --splits /content/drive/MyDrive/MarketMamba_V7/prepared-full/splits.json --kg /content/drive/MyDrive/MarketMamba_V7/prepared-full/knowledge_graph_v2_csr.npz --market-data /content/drive/MyDrive/MarketMamba_V7/prepared-full/market_prices_raw.parquet
```

Active 47 features must be finite for admitted observations; masked Group D may be absent. Active RS features require numeric TWII coverage. Low-frequency fundamental context remains available for protected PIT/as-of helpers. Rolling price state restarts after missing or quarantined sessions. Labels use raw Close at exact calendar t+5/t+10; missing endpoints and, by default, interior gaps invalidate only that horizon. No benchmark is subtracted. Sparse labels are excluded per head from ranks, loss and IC; no-target batches skip optimizer/scheduler updates and are counted. A whole run with no updates fails. Nonfinite predictions or gradients fail before an optimizer update.

Prepared protocol `v7-calendar-v2` fingerprints the complete calendar, daily expected universes, both provenances, selected union, masks, label/gap semantics and quality settings. Incomplete v1 artifacts must be rebuilt. Prepare and data-check recheck effective feature coverage on all calendar dates, including validation and absent dates. Older prepared data/checkpoints require an explicit rebuild; loading never silently migrates them. Windows retain 60 calendar positions, distinguish leading pre-history from internal observations, and exclude current-missing stocks from graphs. The 59 feature order and model architecture stay frozen.

Colab preparation defaults to the frozen date cutoff 2026-06-02 and requires an explicitly selected independent calendar ending on or before it. The default split is exactly two sets: training from 2013 with the final 30 trading rows before the 2023-12-31 boundary purged, and validation/evaluation from 2024 through 2026-06-02. It is not an untouched test. `--frozen-splits` may supply an explicit optional future test only when it is source-covered and separated by 30 trading rows.

## 2. Resolve the pinned binary runtime

Copy the manifest to a writable runtime directory. The setup helper captures actual Python, PyTorch, CUDA-build, ABI, and platform metadata without `torch.cuda.is_available`; queries only the official `v2.3.2.post1` GitHub release; selects one matching binary wheel; records and checks the published SHA-256 when available; records the downloaded digest; and inspects real wheel METADATA for `torch`, `einops`, and `transformers`. The pinned release requires its `selective_scan_cuda` extension. `causal-conv1d` is guarded and not required by this no-conv path.

```bash
mkdir -p /content/v7-runtime
cp V6/experimental/v7_integrated_environment.json /content/v7-runtime/manifest.json
python V6/experimental/v7_integrated_probe.py --setup-colab --manifest /content/v7-runtime/manifest.json --runtime-metadata /content/v7-runtime/runtime.json --wheel-dir /content/v7-runtime
```

CUDA 12.x runtime metadata is matched to the release's `cu12` tag, while platform architecture is matched exactly (`linux_x86_64` versus `linux_aarch64`). If no exact wheel exists, the resolver fails with the pinned release asset list before training; source compilation is forbidden.

One known compatible optional route is Python 3.12, official PyTorch 2.10.0 CUDA 12.8 binaries, ABI true, and the pinned release's `cp312`/`cu12`/Torch 2.10 wheel. Use the official [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) command in an explicit cell, then restart the kernel and rerun metadata capture; never continue in the old kernel:

```python
import os, signal, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
    'torch==2.10.0', 'torchvision==0.25.0', 'torchaudio==2.10.0',
    '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
os.kill(os.getpid(), signal.SIGKILL)  # expected Colab kernel restart
```

After reconnecting, rerun the mount/source variables and the resolver command so `runtime.json` is recaptured. Only after reviewing the selected exact asset is installation an explicit separate rerun using binary packages only:

```bash
python V6/experimental/v7_integrated_probe.py --setup-colab --install --manifest /content/v7-runtime/manifest.json --runtime-metadata /content/v7-runtime/runtime.json --wheel-dir /content/v7-runtime
```

## 3. Smoke, full train, and resume on persistent Drive

Set `D=/content/drive/MyDrive/MarketMamba_V7/prepared-full`, `R=/content/v7-runtime`, and `C=/content/drive/MyDrive/MarketMamba_V7/checkpoints`. Smoke uses a separate checkpoint and is explicitly capped, so it cannot overwrite a completed full run. Full training has no silent cap. JSON progress is emitted and an exact resume checkpoint (including RNG and next epoch/batch position) is saved every 100 updates by default, plus epoch/end/cap boundaries. Tune the bounded cadence with `--progress-interval` and `--checkpoint-interval`; avoid per-batch Drive writes.

```bash
python V6/experimental/v7_integrated_train.py smoke --device cuda --runtime-metadata $R/runtime.json --manifest $R/manifest.json --feature-parquet $D/features_59.parquet --feature-metadata $D/feature_metadata.json --splits $D/splits.json --kg $D/knowledge_graph_v2_csr.npz --checkpoint $C/v7-smoke.pt --smoke-steps 1
python V6/experimental/v7_integrated_train.py train --device cuda --runtime-metadata $R/runtime.json --manifest $R/manifest.json --feature-parquet $D/features_59.parquet --feature-metadata $D/feature_metadata.json --splits $D/splits.json --kg $D/knowledge_graph_v2_csr.npz --checkpoint $C/v7-full.pt --epochs 20 --progress-interval 100 --checkpoint-interval 100
python V6/experimental/v7_integrated_train.py train --device cuda --runtime-metadata $R/runtime.json --manifest $R/manifest.json --feature-parquet $D/features_59.parquet --feature-metadata $D/feature_metadata.json --splits $D/splits.json --kg $D/knowledge_graph_v2_csr.npz --checkpoint $C/v7-full.pt --resume $C/v7-full.pt --epochs 20
```

## 4. Forecast validation and evaluate

Forecast defaults to `--split validation`; use `--split test` only for a genuinely separate supplied test. Existing frozen baseline weights/scores are absent locally, so provide the future verified `v2_kg_nomacro` score/checkpoint input explicitly and make no comparison claim until then.

```bash
python V6/experimental/v7_integrated_train.py forecast --split validation --device cuda --runtime-metadata $R/runtime.json --manifest $R/manifest.json --feature-parquet $D/features_59.parquet --feature-metadata $D/feature_metadata.json --splits $D/splits.json --kg $D/knowledge_graph_v2_csr.npz --checkpoint $C/v7-full.pt.best --output-dir /content/drive/MyDrive/MarketMamba_V7/evaluation/scores
python V6/experimental/v7_integrated_train.py evaluate --feature-parquet $D/features_59.parquet --feature-metadata $D/feature_metadata.json --splits $D/splits.json --kg $D/knowledge_graph_v2_csr.npz --candidate-scores /content/drive/MyDrive/MarketMamba_V7/evaluation/scores/v7_integrated_scores.parquet --baseline-scores /content/drive/MyDrive/MarketMamba_V7/baselines/v2_kg_nomacro_scores.parquet --market-data $D/market_prices_raw.parquet --output-dir /content/drive/MyDrive/MarketMamba_V7/evaluation/comparison
```

Economic replay keeps absent daily scores as NaN, uses every supplied market trading day in the interval, applies holdings on the next day, ranks lexical stock columns with `method='first'`, and calls the unchanged V6 f20 configuration N=50, buffer=1.5, frequency=20, buy cost .0015, and sell cost .0045. Never target live result/state directories.

## Builder and supervisor verification

The sole Builder verification is `python -B V6/experimental/v7_integrated_test.py` using the installed project runtime, small deterministic CPU fixtures and explicitly injected `synthetic_ssd`. This is not official Mamba evidence. The root supervisor independently repeats CPU checks, runs the above CUDA smoke with `--smoke-steps 1` using official installed Mamba SSD and existing GATv2, and refreshes delivery. No full-data training or GPU execution is part of Builder work.

Next-stage ingestion may implement download/retry/source switching against this reusable report contract. This change implements no such service.
