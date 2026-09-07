# V7 Literature Synthesis

This synthesis converts the staged literature review into V7 priors. Literature evidence is not local evidence for MarketMamba; every architecture change remains a hypothesis until tested against the frozen current baseline under Taiwan-equity cross-sectional constraints.

## 1. Backbone

Literature evidence:

- Mamba-2 is the default V7 backbone candidate because its state-space duality, hardware efficiency, and larger practical state capacity justify a direct comparison with the current MarketMamba Mamba baseline.
- This is an engineering prior about scalability and representational capacity, not proof of higher financial forecasting accuracy.

Our hypothesis:

- H1 should compare current Mamba against Mamba-2 under matched data, label, horizon, split, and portfolio evaluation. Mamba-2 must earn promotion through local predictive, economic, robustness, and compute metrics.

## 2. Temporal vs Cross-Sectional

Literature evidence:

- Temporal sequence modeling and cross-sectional stock interaction modeling should be treated as separate problems.
- S-Mamba/Bi-Mamba is the leading implicit cross-stock interaction candidate.
- GATv2 remains useful as a coarse, high-confidence explicit prior when relations are known enough to be trusted.

Our hypothesis:

- Do not assume Temporal to Cross is optimal. Retain Temporal to Cross, Cross to Temporal, and Parallel to Fusion as explicit experiments.
- Use GATv2 for coarse explicit priors and let implicit/data-driven modules handle unknown or unstable relations.

## 3. Attention

Literature evidence:

- Full attention is not forbidden.
- It is not the default for large full-market and long-context modeling because quadratic scaling is unattractive for the target geometry.

Our hypothesis:

- Bi-Mamba versus attention should be framed as an accuracy-compute Pareto question. The question is not whether Mamba is always more accurate; it is whether Bi-Mamba reaches a better quality/runtime/VRAM frontier for MarketMamba's full-market use case.

## 4. Graph

Literature evidence:

- RSR and HATS mainly back the current use of ranking and relational modeling.
- Static graph relations can help when they encode stable, meaningful structure, but poor or noisy static relations may hurt.

Our hypothesis:

- Prefer coarse explicit graph priors. Avoid encoding fine-grained, uncertain, or stale relations as hard graph structure unless local ablation shows value.

## 5. Non-Stationarity

Literature evidence:

- Normalization can stabilize training but may also remove information needed to recognize regime and scale changes.
- AdaRNN-style MMD/CORAL alignment is a plausible candidate for distribution shift handling, with a real risk of over-invariance under concept drift.
- Full NSC-Mamba-style attention hybrids are relevant but lower priority if they sacrifice the efficiency that made Mamba attractive.

Our hypothesis:

- The highest-value Mamba-native hypothesis is to allow raw/regime statistics to condition Mamba selective dynamics, especially the Delta-related pathway.
- Market-state scanner/factor gating should be tested as a separate, interpretable route for regime dependence.
- MMD/CORAL should be treated as a regularizer candidate, not assumed protection against financial non-stationarity.

## 6. Uncertainty

Literature evidence:

- Probabilistic forecasting and conformal calibration provide modular ways to evaluate uncertainty through coverage and sharpness.

Our hypothesis:

- Mean model plus lightweight variance head plus Gaussian NLL is a high-value experimental candidate.
- Post-hoc conformal calibration is modular and gives coverage/sharpness metrics without forcing architecture changes.
- Conformal tail/head classification is lower priority because it adds design complexity and may not map cleanly to the ranking and portfolio objective.

## 7. Ranking / Portfolio

Literature evidence:

- Stock prediction literature supports ranking-aware evaluation and losses, but loss shape must match the downstream action.

Our hypothesis:

- Keep alpha ranking and portfolio construction decoupled.
- StockMamba U-shaped loss should only be tested once the downstream portfolio objective makes its head/tail emphasis appropriate.

## 8. Foundation Models

Literature evidence:

- Mamba4Cast is useful backing for Mamba-2 scalability and efficient zero-shot time-series forecasting.
- TimesFM, Chronos, and Moirai show time-series foundation models can work, but they do not establish suitability for full-market cross-sectional ranking.
- Chronos provides evidence that text-language pretrained weights are not intrinsically required for numerical time-series forecasting.

Our hypothesis:

- Pretraining/frozen-backbone transfer, synthetic data, multi-frequency patching, and mixture distribution heads remain future branches, not V7 requirements.

## Non-Conclusions

- Mamba-2 is not assumed to improve financial accuracy.
- Bi-Mamba is not proven to beat attention in unconstrained accuracy.
- Transformer is not invalid for finance; it is unattractive for our target scaling geometry unless evidence justifies the cost.
- Literature evidence alone cannot promote a feature into V7.
