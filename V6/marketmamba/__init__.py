"""
MarketMamba V6 Package
======================
Pure-quant Mamba + GATv2 stock ranking model for Taiwan equities.

Key changes from V5.5:
  - input_dim: 84D → 46D (FinBERT sentiment removed)
  - FactorGroupedEmbedding: factor-aware input projection
  - MultiScaleMambaEncoder: parallel 20d / 60d / 252d branches
  - MultiHorizonHead: joint 5d / 20d / 60d prediction
  - TemporalCrossSectionDataset: correct cross-section batching
  - Evaluation: IC / ICIR replacing MSE
  - Walk-Forward Validation: Expanding-Window, 36 folds
"""

__version__ = "6.0.0-dev"
