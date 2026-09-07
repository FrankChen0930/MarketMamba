# V7 Experiment Hypotheses

No component becomes part of V7 because of a single best run or a paper citation. Every promoted component must survive baseline-comparable evaluation, leakage checks, economic validation, robustness review, and compute review.

Status vocabulary:

- `UNTESTED`
- `SUPPORTED`
- `INCONCLUSIVE`
- `REJECTED`
- `BLOCKED_BY_COST`
- `PARKED`

## P0

| ID | Hypothesis | Status |
|---|---|---|
| H0 | Frozen V6.x baseline. | UNTESTED |

## P1 Backbone

| ID | Hypothesis | Status |
|---|---|---|
| H1 | current Mamba vs Mamba-2. | UNTESTED |
| H2 | temporal lookback scaling: approximately 30/60/120/250/longer when justified. | UNTESTED |

## P2 Cross-Sectional

| ID | Hypothesis | Status |
|---|---|---|
| H3 | No cross-stock vs GATv2 vs Bi-Mamba vs GATv2+Bi-Mamba. | UNTESTED |
| H4 | reduced-universe Attention vs Bi-Mamba quality + N-scaling runtime/VRAM comparison. | UNTESTED |
| H5 | Temporal to Cross vs Cross to Temporal vs Parallel to Fusion. | UNTESTED |

## P3 Non-Stationarity

| ID | Hypothesis | Status |
|---|---|---|
| H6 | raw/regime statistics conditioning of Mamba selective dynamics/Delta. | UNTESTED |
| H7 | Market-state scanner/factor gating. | UNTESTED |
| H8 | MMD/CORAL regime-alignment regularization. | UNTESTED |
| H9 | full NSC/attention hybrid only after H6-H8. | PARKED |

## P4 UQ/Loss

| ID | Hypothesis | Status |
|---|---|---|
| H10 | lightweight variance head + Gaussian NLL. | UNTESTED |
| H11 | post-hoc conformal calibration. | UNTESTED |
| H12 | U-shaped rank-position loss only after portfolio policy is defined. | PARKED |

## P5 Secondary

| Hypothesis | Status |
|---|---|
| Conv keep/remove/smaller/dilated depending on input/token semantics. | UNTESTED |
| Gaussian learned adjacency / low-rank graph factorization. | UNTESTED |
| VAST/ATSP. | UNTESTED |
| direct/single-pass if multi-horizon output is relevant. | UNTESTED |
| patching. | UNTESTED |

## P6 Parked

| Hypothesis | Status |
|---|---|
| intraday/multi-frequency. | PARKED |
| synthetic financial pretraining. | PARKED |
| TSFM transfer/frozen backbone. | PARKED |
| mixture-distribution head. | PARKED |
| behavioral/IRL portfolio modeling. | PARKED |
