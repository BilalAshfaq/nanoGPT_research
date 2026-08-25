# Frobenius-Normalized SGDM Tasks

## Variant goal

Test whether dynamic per-matrix magnitude normalization can improve on static per-matrix SGDM while preserving the SGDM momentum direction and avoiding any Muon computation.

```text
r_l = min(d_out_l, d_in_l)
M_hat_l,t = M_l,t / (||M_l,t||_F + epsilon)
Delta W_l,t = -eta * sqrt(r_l) * M_hat_l,t
```

Any additional fixed shape factor must be separately configurable, justified, and documented. The default rule is the equation above.

## Prerequisites and scope constraints

- Tuned Global SGDM and Static Per-Matrix SGDM tasks must be approved first.
- Reuse the shared partition, momentum convention, auxiliary AdamW path, scheduler, checkpointing, and diagnostic framework.
- Do not call a Muon transform, SVD, or Newton–Schulz routine in this variant.
- Normalization must act on the momentum matrix used for the SGDM update, not independently on each raw micro-batch gradient.
- Add this variant to the common `optimizer_name` selector without changing or removing any existing optimizer choice, especially `adamw`.

## Task 3.1 — Implement the normalization rule as a focused primitive

### Task goal

Create a numerically safe, directly testable function that converts an SGDM momentum matrix into the prescribed Frobenius-normalized update direction and target scale.

### Required changes

- Add a focused normalization function accepting a matrix, epsilon, and documented optional fixed scale.
- Compute the norm with sufficient precision for the active training dtype.
- Define behavior for exactly zero and near-zero momentum matrices without NaNs or unintended large updates.
- Compute `r_l` from the actual two-dimensional parameter shape.
- Return or expose the raw norm, target norm, and applied multiplier for diagnostics without retaining unnecessary tensors.
- Validate epsilon and scaling settings at construction time.

### Task passing criteria

- For nonzero matrices across square, tall, and wide shapes, the normalized direction has the expected Frobenius norm.
- Zero input produces a finite zero update.
- Near-zero fp16/bfloat16 inputs do not produce NaN or Inf in supported execution paths.
- Scaling a nonzero input matrix by a positive constant does not change the normalized direction beyond tolerance.
- The implementation contains no Muon, SVD, polar-decomposition, or Newton–Schulz call.

## Task 3.2 — Integrate Frobenius-normalized SGDM

### Task goal

Add a training variant that maintains ordinary SGDM momentum but dynamically normalizes each eligible matrix's candidate update.

### Required changes

- Add `frobenius_normalized_sgdm` as a selectable optimizer variant.
- Reuse the exact global-SGDM momentum update.
- Apply the default `sqrt(min(shape))` target rule after momentum formation.
- Add separately named configuration for this variant's base LR, epsilon, and any approved fixed shape factor.
- Keep auxiliary AdamW and all parameter groups unchanged.
- Ensure the LR scheduler scales this variant's base rate correctly.
- Save all normalization settings and optimizer state in checkpoints.

### Task passing criteria

- For each nonzero synthetic momentum matrix, the actual parameter-step norm matches `eta * sqrt(r_l)` within tolerance when weight decay is disabled.
- Frobenius cosine similarity between the parameter update and momentum is `-1` within tolerance.
- Momentum buffers match global SGDM under identical gradients and momentum settings.
- Eligible and auxiliary parameters remain disjoint and use the intended optimizers.
- A tiny mixed-precision-capable smoke test, where available, completes without numerical failure.
- Explicit `optimizer_name = 'adamw'` continues to pass the protected AdamW regression smoke test.

## Task 3.3 — Add normalization diagnostics and regression tests

### Task goal

Make it possible to verify during training that the intended target norms are actually achieved.

### Required changes

- At selected diagnostic steps, log momentum norm, target update norm, actual update norm, relative update norm, effective multiplier, and epsilon-clamp/zero-momentum events.
- Define effective multiplier as the scalar applied to the unnormalized momentum, including scheduled LR and fixed shape scaling.
- Add a direct comparison diagnostic against ordinary SGDM direction.
- Keep diagnostic norm calculation off the critical path when diagnostics are disabled.
- Add checkpoint/resume coverage for normalization settings and momentum state.

### Task passing criteria

- Logged actual norms agree with parameter deltas measured in a tiny test.
- Target and actual norms agree for nonzero momentum when weight decay is disabled; any weight-decay contribution is reported separately under the common protocol.
- Direction cosine is `-1` relative to momentum for nonzero matrices.
- Resume produces the same next update as uninterrupted execution.
- Diagnostics do not retain full updates or computation graphs between steps.

## Task 3.4 — Define tuning and comparison protocol

### Task goal

Prepare a fair test of dynamic generic normalization against both tuned global and static per-matrix SGDM.

### Required changes

- Define a fresh LR search for the normalized rule; do not reuse an SGDM LR without tuning.
- Fix epsilon unless numerical evidence requires it to be tuned.
- Match model initialization, data order, tokens, batch/accumulation, schedule shape, clipping, precision, evaluation, and auxiliary optimizer settings.
- Compare best tuned global SGDM, best static per-matrix SGDM, and best Frobenius-normalized SGDM.
- Require multi-seed confirmation for final claims and report numerical-instability/divergence rates.

### Task passing criteria

- Search budgets and selection criteria are predefined and comparable.
- Primary selection uses validation loss at fixed processed tokens.
- The plan can distinguish an optimization gain from instability or an unmatched auxiliary setting.
- No large run begins until unit tests and a short smoke run pass.

## Variant completion criteria

- Tasks 3.1 through 3.4 are completed and approved sequentially.
- The optimizer dynamically matches the generic Frobenius target while preserving SGDM direction.
- No Muon computation exists in this variant.
- Numerical edge cases, diagnostics, checkpointing, and resume behavior are tested.
- The comparison protocol supports, but does not prejudge, hypothesis H3.
