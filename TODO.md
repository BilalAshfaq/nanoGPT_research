# Deferred Experiment TODOs

These items are intentionally deferred while the Variant 1 exploratory study runs. They do not block inspecting preliminary results from uninterrupted runs.

## Variant 1 study completion

- Select the winning exploratory configuration for global SGDM and AdamW using the predefined fixed-token validation-loss rule.
- Run the two predefined confirmation seeds for each selected configuration so each optimizer has three seeds in total.
- Generate and review the final aggregate report, including individual runs, mean, sample standard deviation, failures, and instability.
- Before accepting results, verify that all compared runs used identical dataset and environment fingerprints and inspect whether any outcome was resumed.

## Reproducibility hardening

- Make checkpoint resume restore CPU and CUDA random-number-generator state so resumed training and evaluation continue the original batch sequence.
- Save and restore GradScaler state for configurations that use mixed-precision scaling.
- Preserve the maximum peak GPU memory across resumed process sessions.
- Add a study-configuration preflight using the real model, dataset, distributed setup, and batch controls, in addition to the existing tiny synthetic preflight.
- Require exactly three successful seed results per selected optimizer before the final report can be considered complete.
- Update the exploratory-manifest authorization test so it reflects the authorized study phase without weakening launch-safety coverage.

## Active-run operating rule

- Keep the cluster working tree unchanged while a sequential manifest job is running; pulling or editing between candidates could make one study contain runs from different code states.
