# 2026-09-22 local early-V7 source preservation register

Status: `HISTORICAL_SOURCE_NOT_CURRENT_CORRECTED_E5_OR_E1`. All paths below are preserved in place in the same Git commit as this register. This is a provenance register, not a runnable current V7 pipeline or result admission. Corrected E5 uses 48 features, no graph/industry neutralization; E1 result authority remains its frozen branch contract and run manifest. Exact source commit is the Git commit containing this register; use `git log --follow` for later moves.

Approximate periods and source relationships are evidence-limited. A delivery name or linked note does not prove each file generated every artifact. No large delivery or data was copied or changed.

| Cohort | Approximate period | Source relation | Meaning / current status |
|---|---|---|---|
| `v7_experiment_*` | 2026-09-14 | `deliveries/V7-Experiments-20260914/README.md; research/v7/training-efficiency-capacity-plan.md` | 早期容量/六組實驗; historical, not corrected E5/E1 authority |
| `v7_confirmation_*` | 2026-09-14–15 | `deliveries/confirmation-v1-fp32; research/v7/confirmation-v1-fp32-results-review.md` | 早期 confirmation; historical, not corrected E5/E1 authority |
| `v7_depth222_*` | 2026-09-14–15 | `deliveries/depth222-v1-fp32; research/v7/p0-inventory-20260915.md` | 早期深度 222 比較; historical, not corrected E5/E1 authority |
| `v7_integrated_*` | 約 2026-09-15 | `research/v7/integrated-candidate-plan.md; research/v7/integrated-candidate-colab.md` | 59-feature graph/industry isolated candidate; historical, not corrected E5/E1 authority |
| `v7_h1_*` | 約 2026-09-10–15 | `research/v7/current-state.md (dated historical); research/v7/experiment-hypotheses.md` | isolated H1 Mamba/Mamba-2 probe; historical, not corrected E5/E1 authority |
| `v7_stability_*` | 約 2026-09-15 | `deliveries/stability-ab-v1; deliveries/stability-c-v1; deliveries/V7-Stability-20260915` | 早期 stability suite; historical, not corrected E5/E1 authority |

## File inventory

SHA-256 is for file identity at preservation time. Nine stability files had redundant final blank lines removed solely to satisfy `git diff --check`; no logic was changed. Files retain their original `V6/experimental/` paths because delivery/runbook references use those paths.

| Path | Bytes | SHA-256 | Cohort |
|---|---:|---|---|
| `V6/experimental/v7_experiment_data.py` | 4671 | `0dc1da2a67976f1a932a548239bbb21a94d646533d5502935d2888b67ca6db94` | `v7_experiment_*` |
| `V6/experimental/v7_experiment_model.py` | 2283 | `7337478f2b15ed0eff5592c3a62afb6f71b57dbe8e592daa59f7949126e0476e` | `v7_experiment_*` |
| `V6/experimental/v7_experiment_storage.py` | 4678 | `25e4535b77293d92ca358c404a5f2658c484ff95a009a153cacd6444097a6676` | `v7_experiment_*` |
| `V6/experimental/v7_experiment_suite.py` | 11684 | `2b14cae8ee386a2970e6d9cb1aecbb24256e37a504504825a751e04427467d20` | `v7_experiment_*` |
| `V6/experimental/v7_experiment_test.py` | 7085 | `fa075127ae21c1eacb791f39f2aa0e45123c95fa5ca72fc44c72704afb9d0624` | `v7_experiment_*` |
| `V6/experimental/v7_experiment_train.py` | 14143 | `ce5b323ff44bf5b1325d9185c3317aa5aa79403b085fda96f8f2dec31c2cdf8e` | `v7_experiment_*` |
| `V6/experimental/v7_confirmation_benchmark.py` | 5000 | `85d127bb44cb1e478031feca543d94536f31ffacb54c52136a4243b9cc858e04` | `v7_confirmation_*` |
| `V6/experimental/v7_confirmation_design.py` | 4407 | `a1fbe3827a14ad69f66a5cc4e294c4ee8a26b1c9cf42e6e1421af55e47536aaa` | `v7_confirmation_*` |
| `V6/experimental/v7_confirmation_reference.json` | 33763 | `570a8af57de2566958ae9ddf1ab614d1317fad8a876c1fec88613cd113182e6a` | `v7_confirmation_*` |
| `V6/experimental/v7_confirmation_suite.py` | 12879 | `33d34fdfaeb3fd23e7b0978b7fe60625eba61f6b3381445eb2724a539dca5638` | `v7_confirmation_*` |
| `V6/experimental/v7_confirmation_test.py` | 5617 | `a952d7375e18fd03d105fa1a5a357afc2053ef037c504e490b811a9a9fe8cc86` | `v7_confirmation_*` |
| `V6/experimental/v7_depth222_suite.py` | 10585 | `73609d9ede0a81c70fe752543849ff74ff0621301a08acdc38d8692c8d0a2213` | `v7_depth222_*` |
| `V6/experimental/v7_depth222_test.py` | 2907 | `029d7035ee7f7cb84c9425452ed12696bba5442b91979de6839a39eb58ab09d7` | `v7_depth222_*` |
| `V6/experimental/v7_integrated_colab.ipynb` | 18279 | `fae8271bb4d8c84dcc09fb98efe6ee4c51f49ab0f22ec4c623d4748db1466b2b` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_config.py` | 5241 | `6e0a0370fd45fe9e9aabf523a62699dc32173e9b90d23bf4154cfb1fdadc6fcd` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_data_quality.py` | 12053 | `3e3249ada163f74e41fa79499586925f55bf3c223754d9d1320b943ac3371086` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_environment.json` | 1987 | `0179678c1e0e93180e85c60bd86ebc6e3b132fa5cad4ab857488ba15d5ae68e0` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_industry.py` | 6670 | `211f82f46cad0476361b53b29c406b71e4ebda77244fe67c77ee4b349eec3352` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_industry_test.py` | 4212 | `1051b168f57f63a6144e111d6e56cb3d3159dfd8b91584ecf3851142948522c2` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_model.py` | 12187 | `a3a5bbad05672af343c00e1ba1775be64437e3130487daa421fa8fd422573f97` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_point_in_time.py` | 5597 | `b3ea9203932eaf8d3cd183d72eb32c03602f5e2025f05b82e36402bd3cd5a204` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_point_in_time_test.py` | 3255 | `dccce4704f188f3cce1eeb76fb047426c0e9f94cca24847fc77d70f6eda52456` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_prepare_cache.py` | 3221 | `c16c3d1ce1a5ebe3c1a80ce592f332acaae28d3c6664e5d0254f33a9d73d133e` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_prepare_cache_test.py` | 1671 | `68ee4e10e650c711ab1808ac927986eb81f3db0e4eac4854365d47111e94247a` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_probe.py` | 13731 | `4577b4f6523408ec2076504103c5e835ce017ffe82386baf00c1de60935a4a32` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_test.py` | 71683 | `0965384344dae6ce9f13010c37d8072269d1218559236c1733d551b180a1a523` | `v7_integrated_*` |
| `V6/experimental/v7_integrated_train.py` | 92448 | `4ce1d8b495c6b28d2a6f3749c10a8537f1d84aefa05ff929e93b31ee0421d132` | `v7_integrated_*` |
| `V6/experimental/v7_h1_config.py` | 4077 | `8d374311d1716f0d63e917c7bd79219a98d3c56dc5ac64811c0fb87b1a9a497c` | `v7_h1_*` |
| `V6/experimental/v7_h1_mamba2_adapter.py` | 8668 | `75f9e152431dce4f57bd79b1ff616deea0ae2bdb8b589aa8060e44c805914d2d` | `v7_h1_*` |
| `V6/experimental/v7_h1_probe.py` | 5612 | `6141daf7be3cbf41a4f2ad08adff8387a062554ea99d9ee9216ad4d9d6d6249b` | `v7_h1_*` |
| `V6/experimental/v7_h1_test.py` | 6265 | `c7792d7111dc73ea55e440dcbd33994f9f1dee9997f567d90b0ec420a641599d` | `v7_h1_*` |
| `V6/experimental/v7_stability_contract.py` | 4684 | `8a7056f5b83e7bb7731d545dac4ee5494b5d455fa0b0303fa2375f99aa3d8e7d` | `v7_stability_*` |
| `V6/experimental/v7_stability_delivery_test.py` | 1030 | `102704c89e2b9b596eb3bfac6cca11595d620656b8a2bd96e8b4c649e8ab0096` | `v7_stability_*` |
| `V6/experimental/v7_stability_diagnostics.py` | 6613 | `8e753599f4368b2f15743cf258a4fccaa374a8cde7183703a01e9f0a9112d4e9` | `v7_stability_*` |
| `V6/experimental/v7_stability_diagnostics_test.py` | 3606 | `760c3bff70de757647049a34d350eb1d7466093356ba87253fa7c84e64f4d731` | `v7_stability_*` |
| `V6/experimental/v7_stability_export.py` | 5659 | `6a2c1f5a711c926a2a0695e81c2a387e3231d020a9c46f8480e7b07a576fa1df` | `v7_stability_*` |
| `V6/experimental/v7_stability_export_test.py` | 4363 | `3df57cea079dcba55243fe40801a70b33807289dfe477549114d834920201d7c` | `v7_stability_*` |
| `V6/experimental/v7_stability_suite.py` | 12849 | `54c3ed7d6844564264244b6c9954a472a067920f9f811a759d6d3adf89ac87b3` | `v7_stability_*` |
| `V6/experimental/v7_stability_suite_test.py` | 2927 | `5fa9c6dff818e7b0dafca36ce1749c157b38503a5b287c4276af1b1010ee9c2b` | `v7_stability_*` |
| `V6/experimental/v7_stability_test.py` | 3560 | `e37a2bc01387ec29a96e1ec5cc667c7d09b4b8d797b354649fb672d7e215d2ca` | `v7_stability_*` |
| `V6/experimental/v7_stability_windows.py` | 2275 | `dd2780891f4974290163f075d5ba7c9369cc1b6fd447a3e0721287e9e291f827` | `v7_stability_*` |

## Limits and safe future actions

- `v7_integrated_test.py` contains old absolute `/home/frank/...` test fallback paths. They are historical local path assumptions, not a current environment contract.
- This source set was syntax compiled but not re-run as a formal experiment; historical delivery results retain their own manifests/evidence class.
- Do not remove or move a file until references, exact bytes, branch lineage and authoritative survivor have been checked and accepted. Preserve this Git commit and linked deliveries.
