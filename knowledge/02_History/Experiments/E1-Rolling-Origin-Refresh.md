# Experiment: E1 Rolling-Origin Refresh

- Date: `2026-09-21` (repository admission; run artifacts retain their own timestamps)
- Evidence class: `HISTORICAL_SIMULATION_PROXY` / `RESEARCH_EVIDENCE`

## Question

Does a rolling-origin refresh recover 2026 signal under the unchanged corrected E5 model/training contract?

## Frozen contract

The E1 experiment contract pins the corrected 48-feature, verified-lifecycle, no-graph/no-industry-neutralization setup; seeds 17 and 29; matrix identity `a590b280f0700dcf36fde9ac51e767d735cc2737b613cbb3b49ef3f472947f62`; and its frozen comparison/acceptance rules.

## Result

The completed result is `PASS_REFRESH_HYPOTHESIS`: E1 mean daily Rank IC 0.06341/0.05202 versus old corrected E5 0.02912/0.00354 on the identical paired support (5d/10d). See the machine evaluation and result manifest, not this note, for exact evidence.

## Conclusion

The refresh hypothesis is supported for this historical-simulation research experiment. It does not establish strict executable correctness, portfolio admission, trading readiness, or a causal mechanism for epoch degradation.

## Limitations

7,471 of 314,780 prediction keys have no matched labels and remain missing. Only 307,309 matched label keys are materialized. Strict Phase 0 is `STOP`; verified executable labels are 0/0. Post-run diagnosis shows month sensitivity; final-support per-epoch replay and ex-ante regime attribution remain unresolved.

## Authority

`eec63c5:research/v7/e1-rolling-origin-refresh-v1/results/result-manifest.json` and same-directory evaluation. This note is navigation/history only.

## Next decision

Inspect post-run epoch/month evidence, recover the identity-matched matrix and causal market-state inputs before proposing E2. Do not start training automatically.
