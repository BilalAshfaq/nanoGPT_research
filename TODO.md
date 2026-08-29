# Deferred Experiment TODOs

These items are intentionally deferred while the Variant 1 exploratory study runs. They do not block inspecting preliminary results from uninterrupted runs.

## Variant 1 study completion

- Confirmation seeds `2027` and `4099` are explicitly deferred for now. The
  seed-`1337` winner may be used to launch the Task 2.5 exploratory study, but
  it does not support a final multi-seed improvement claim.
- Select the winning exploratory configuration for global SGDM and AdamW using the predefined fixed-token validation-loss rule.
- Run the two predefined confirmation seeds for each selected configuration so each optimizer has three seeds in total.
- Generate and review the final aggregate report, including individual runs, mean, sample standard deviation, failures, and instability.
- Before accepting results, verify that all compared runs used identical dataset and environment fingerprints and inspect whether any outcome was resumed.

## Variant 2 study completion

- After selecting the seed-`1337` static winner, defer its confirmation seeds
  `2027` and `4099` alongside the Variant 1 confirmations.
- Until both selected global SGDM and static per-matrix SGDM configurations have
  three successful seeds, label their comparison exploratory and make no final
  claim that static scaling improves global SGDM.

## Reproducibility hardening

- Make checkpoint resume restore CPU and CUDA random-number-generator state so resumed training and evaluation continue the original batch sequence.
- Save and restore GradScaler state for configurations that use mixed-precision scaling.
- Preserve the maximum peak GPU memory across resumed process sessions.
- Add a study-configuration preflight using the real model, dataset, distributed setup, and batch controls, in addition to the existing tiny synthetic preflight.
- Require exactly three successful seed results per selected optimizer before the final report can be considered complete.

## Active-run operating rule

- Keep the cluster working tree unchanged while a sequential manifest job is running; pulling or editing between candidates could make one study contain runs from different code states.
