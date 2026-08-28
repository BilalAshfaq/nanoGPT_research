# Static Per-Matrix SGDM Tasks

## Variant goal

Test whether persistent, non-uniform learning-rate scales across eligible Transformer matrices improve on the best tuned global SGDM baseline while preserving the SGDM update direction.

```text
M_l,t = beta * M_l,t-1 + (1 - beta) * G_l,t
Delta W_l,t = -eta * a_l * M_l,t
```

Each multiplier `a_l` is fixed for the full run. The primary scientific comparison is best tuned global SGDM versus best static per-matrix SGDM under matched conditions.

## Prerequisites and scope constraints

- All Tuned Global SGDM tasks must be approved first.
- Reuse the shared eligible/auxiliary parameter partition, SGDM momentum implementation, auxiliary AdamW path, scheduler, checkpointing, and diagnostics.
- Implement this variant by extending the existing optimizer factory and training-loop-facing interfaces; do not create a parallel training path or modify the mathematics of an existing optimizer.
- Reuse the existing composite optimizer, metadata, experiment-manifest, outcome-tracking, selection, and reporting utilities. Extend the smallest genuinely shared interface only when the static variant requires additional structured data.
- Keep static multiplier resolution and static update behavior inside `static_per_matrix_sgdm/` unless a utility is demonstrably shared by multiple variants.
- Do not duplicate parameter partitioning, auxiliary AdamW construction, LR scheduling, checkpoint orchestration, diagnostics collection, or experiment execution logic.
- Do not add dynamic normalization or Muon calculations in this variant.
- Preserve `global_sgdm` behavior when every multiplier equals `1.0`.
- Add this variant to the common `optimizer_name` selector without changing or removing the `adamw` and `global_sgdm` choices.

## Task 2.1 — Define static multiplier configuration and resolution

### Task goal

Provide an explicit, validated, reproducible way to assign a fixed scale to each eligible matrix.

### Required changes

- Add `static_per_matrix_sgdm` as a selectable optimizer variant.
- Support documented matrix-type defaults for QKV, attention output, MLP input, and MLP output projections.
- Support optional exact-parameter-name overrides so experiments can use one multiplier per eligible matrix.
- Define deterministic precedence: exact-name override, then matrix-type value, then an explicit default.
- Resolve all multipliers once during optimizer construction and save the fully expanded name-to-multiplier mapping in run/checkpoint metadata.
- Require every resolved multiplier to be finite and strictly positive; zero and negative multipliers are invalid because this variant does not test frozen matrix updates.
- Remove the scale ambiguity between the shared base LR and the multiplier mapping by requiring the geometric mean of the fully resolved eligible-matrix multipliers to equal `1.0` within a documented numerical tolerance. Reject, rather than silently renormalize, a mapping that violates this convention.
- Reject unknown names, missing eligible matrices, duplicate matches, and invalid multiplier mappings.
- Ensure structured multiplier configuration is captured even though the current configuration collector primarily handles scalar globals.

### Task passing criteria

- Tests cover matrix-type assignment, exact-name override precedence, and a multi-layer model.
- Every eligible matrix resolves to exactly one finite allowed multiplier.
- The resolved mapping satisfies the geometric-mean-one convention, and an otherwise valid but unnormalized mapping fails clearly before training.
- Invalid or misspelled parameter names fail before training begins with a useful error.
- The expanded mapping is stable across repeated construction and checkpoint resume.
- A configuration with all multipliers `1.0` resolves identically to global SGDM scaling.
- Explicit `optimizer_name = 'adamw'` still routes to the protected original AdamW path after this variant is registered.

## Task 2.2 — Apply fixed per-matrix scaling in the optimizer

### Task goal

Scale each eligible matrix's SGDM step by its resolved constant without changing its momentum or within-matrix direction.

### Required changes

- Reuse the global SGDM momentum buffers and update equation.
- Apply the multiplier only to the momentum-derived matrix update. Preserve the common decoupled matrix weight-decay shrinkage at the scheduled shared base LR:

  ```text
  W_l <- (1 - eta_t * weight_decay) * W_l
  W_l <- W_l - eta_t * a_l * M_l
  ```

  Do not multiply the weight-decay contribution by `a_l`, because that would introduce per-matrix regularization as an additional experimental change.
- Keep multipliers constant and independent of gradients, norms, step number, and model state.
- Expose each matrix's effective learning rate or multiplier to diagnostics.
- Avoid one optimizer object per matrix unless measurements show it is necessary.

### Task passing criteria

- Synthetic tests show `Delta W_l = -eta * a_l * M_l` for several shapes and multipliers.
- Frobenius cosine similarity between each nonzero static update and the corresponding global-SGDM momentum direction is `-1` within tolerance.
- Ratios of update norms for identical momentum tensors match the configured multiplier ratios.
- Momentum evolution is identical to global SGDM when all other settings match.
- With all multipliers `1.0`, parameters and optimizer state match global SGDM step-for-step.
- Tests with non-unit multipliers verify that momentum updates are scaled while decoupled weight-decay shrinkage remains at the shared base rate.

## Task 2.3 — Integrate resume validation and diagnostics

### Task goal

Ensure static mappings cannot silently change across resumes and that experiments expose their effective scales.

### Required changes

- Save both the user-provided multiplier specification and fully resolved mapping.
- On resume, compare the current resolved mapping with checkpoint metadata and fail on mismatch unless an explicitly approved migration behavior exists.
- Log per-matrix multiplier, effective matrix LR, update norm, and relative update norm at selected diagnostic checkpoints.
- Include a deterministic multiplier fingerprint and summary in run metadata. Include the fingerprint in each run name, using `static-{mapping_id}` as the scaling-rule component, so distinct mappings cannot collide.
- Preserve auxiliary AdamW configuration exactly from the global baseline protocol.

### Task passing criteria

- A save/resume test preserves momentum and the complete multiplier mapping.
- A deliberately changed multiplier causes a clear compatibility error before the next optimizer step.
- Diagnostic values agree with direct calculations on a tiny model.
- No dynamic norm or singular-value transformation is performed.

## Task 2.4 — Define the fair comparison and tuning plan

### Task goal

Specify how static heterogeneity will be evaluated against the strongest global SGDM baseline.

### Required changes

- Define the allowed multiplier parameterization and search budget before results are inspected.
- Tune the shared base LR rather than assuming the best global-SGDM LR remains optimal after scaling.
- Fix momentum to the winning Variant 1 global-SGDM momentum for this first static study. Do not spend additional trials retuning momentum unless a separately approved follow-up protocol is defined.
- Permit exactly 12 exploratory static configurations at seed `1337`, matching the Variant 1 global-SGDM exploratory run count. Before any static result is inspected, materialize the exact base-LR values, multiplier mappings, run names, and deterministic tie-breaking order in the approved manifest.
- Use confirmation seeds `2027` and `4099` for the selected static configuration; together with its seed-`1337` exploratory result, these provide three seeds without rerunning a valid exploratory winner.
- Match initialization seed, data order, token budget, batch/accumulation, schedule, clipping, precision, evaluation steps, and auxiliary AdamW settings.
- Reuse the Variant 1 processed-token budget, evaluation checkpoints, and unrounded validation loss at the fixed selection checkpoint as the primary selection metric.
- Require at least three seeds for any final claim that static scaling improves global SGDM.
- Record sensitivity to the base LR and multipliers, not only the winning configuration.

### Task passing criteria

- The plan explicitly compares best tuned global SGDM with best tuned static per-matrix SGDM.
- Search budgets are comparable and documented.
- The exact 12-run exploratory grid, mapping fingerprints, and tie-breaking order are frozen in a manifest before launch.
- Exploratory and confirmatory runs are separated.
- No large sweep starts without separate approval and a successful smoke run.

## Task 2.5 — Execute the approved static per-matrix study

### Task goal

Run the separately approved static per-matrix protocol and produce the artifacts needed for a direct comparison with the completed Variant 1 global-SGDM study.

### Required changes

- Start only after Tasks 2.1 through 2.4 are approved, the Variant 1 winner is available, and the user explicitly authorizes the compute budget.
- Reuse the shared experiment-manifest runner, environment and dataset locks, outcome tracking, fixed-token selector, confirmation-manifest generation, and final-report machinery. Extend those shared utilities minimally for multiplier specifications and fingerprints rather than creating static-specific copies.
- Materialize and save all 12 exploratory configurations before launch, including base LR, fixed momentum, fully resolved multiplier mapping and fingerprint, seed, token limit, and evaluation checkpoints.
- Run one successful smoke configuration before the exploratory manifest.
- Record completed, failed, divergent, and interrupted outcomes without silently replacing unsuccessful configurations.
- Select the winner solely by the predefined unrounded validation-loss rule and deterministic tie-breaking order.
- Run the selected configuration at seeds `2027` and `4099` and combine them with the valid seed-`1337` winner.
- Compare the best static result directly with the already selected best tuned global-SGDM result under the same controls. Do not retune or redefine the Variant 1 winner after static results are observed.
- Preserve the complete configurations, mappings, fingerprints, checkpoints, logs, audits, and run metadata required to reproduce the comparison.

### Task passing criteria

- Every approved exploratory and confirmation entry has a recorded outcome.
- All successful static and referenced global-SGDM runs use matching dataset/environment locks, model and batching controls, processed-token selection checkpoint, evaluation schedule, precision, clipping, auxiliary AdamW settings, and shared seeds where compared.
- The selected static configuration has three successful seed results before a final improvement claim is permitted.
- The final report contains individual-seed results, mean, sample standard deviation, failures or instability, exact base LR, fixed momentum, and complete multiplier mapping and fingerprint.
- The report clearly separates observed measurements from interpretations and makes no unsupported Muon claim.
- No unapproved configurations or additional tuning runs are introduced after results are inspected.

## Variant completion criteria

- Tasks 2.1 through 2.5 are completed and approved sequentially; Task 2.5 additionally requires explicit authorization for its compute budget.
- Static scaling is fixed for the entire run and preserves SGDM direction.
- Unit tests demonstrate exact equivalence to global SGDM when all multipliers are one.
- Checkpoints fully capture and validate the per-matrix mapping.
- The completed report compares best tuned global SGDM with best static per-matrix SGDM and supports, but does not prejudge, hypothesis H2.
