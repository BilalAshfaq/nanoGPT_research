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
- Reject unknown names, missing eligible matrices, duplicate matches, non-finite values, and negative multipliers. Decide explicitly whether zero is permitted.
- Ensure structured multiplier configuration is captured even though the current configuration collector primarily handles scalar globals.

### Task passing criteria

- Tests cover matrix-type assignment, exact-name override precedence, and a multi-layer model.
- Every eligible matrix resolves to exactly one finite allowed multiplier.
- Invalid or misspelled parameter names fail before training begins with a useful error.
- The expanded mapping is stable across repeated construction and checkpoint resume.
- A configuration with all multipliers `1.0` resolves identically to global SGDM scaling.
- Explicit `optimizer_name = 'adamw'` still routes to the protected original AdamW path after this variant is registered.

## Task 2.2 — Apply fixed per-matrix scaling in the optimizer

### Task goal

Scale each eligible matrix's SGDM step by its resolved constant without changing its momentum or within-matrix direction.

### Required changes

- Reuse the global SGDM momentum buffers and update equation.
- Apply the multiplier to the final matrix update, with weight decay handled according to the common documented protocol.
- Keep multipliers constant and independent of gradients, norms, step number, and model state.
- Expose each matrix's effective learning rate or multiplier to diagnostics.
- Avoid one optimizer object per matrix unless measurements show it is necessary.

### Task passing criteria

- Synthetic tests show `Delta W_l = -eta * a_l * M_l` for several shapes and multipliers.
- Frobenius cosine similarity between each nonzero static update and the corresponding global-SGDM momentum direction is `-1` within tolerance.
- Ratios of update norms for identical momentum tensors match the configured multiplier ratios.
- Momentum evolution is identical to global SGDM when all other settings match.
- With all multipliers `1.0`, parameters and optimizer state match global SGDM step-for-step.

## Task 2.3 — Integrate resume validation and diagnostics

### Task goal

Ensure static mappings cannot silently change across resumes and that experiments expose their effective scales.

### Required changes

- Save both the user-provided multiplier specification and fully resolved mapping.
- On resume, compare the current resolved mapping with checkpoint metadata and fail on mismatch unless an explicitly approved migration behavior exists.
- Log per-matrix multiplier, effective matrix LR, update norm, and relative update norm at selected diagnostic checkpoints.
- Include a concise multiplier fingerprint or summary in run metadata and use `static` as the scaling-rule component of run names.
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
- Match initialization seed, data order, token budget, batch/accumulation, schedule, clipping, precision, evaluation steps, and auxiliary AdamW settings.
- Use validation loss at a fixed processed-token budget as the primary selection metric.
- Require at least three seeds for any final claim that static scaling improves global SGDM.
- Record sensitivity to the base LR and multipliers, not only the winning configuration.

### Task passing criteria

- The plan explicitly compares best tuned global SGDM with best tuned static per-matrix SGDM.
- Search budgets are comparable and documented.
- Exploratory and confirmatory runs are separated.
- No large sweep starts without separate approval and a successful smoke run.

## Variant completion criteria

- Tasks 2.1 through 2.4 are completed and approved sequentially.
- Static scaling is fixed for the entire run and preserves SGDM direction.
- Unit tests demonstrate exact equivalence to global SGDM when all multipliers are one.
- Checkpoints fully capture and validate the per-matrix mapping.
- The comparison protocol supports, but does not prejudge, hypothesis H2.
