# Muon-Scaled SGDM Tasks

## Variant goal

Measure how much of Muon's behavior can be recovered by using Muon only as a per-matrix update-norm oracle while retaining the original SGDM momentum direction.

```text
P_l,t = MuonTransform(M_l,t)
Delta W_oracle_l,t =
    -eta_muon * s_l * ||P_l,t||_F
    / (||M_l,t||_F + epsilon)
    * M_l,t
```

The Muon-transformed direction must be discarded. Only the norm of the actual candidate Muon step may affect the oracle SGDM update.

## Prerequisites and scope constraints

- All earlier SGDM variant tasks must be approved first.
- Reuse the shared parameter partition, SGDM momentum convention, auxiliary AdamW configuration, scheduler, checkpointing, and diagnostics.
- The transform, coefficients, iteration count, input normalization, shape scale `s_l`, momentum convention, and weight-decay semantics must match the selected Muon reference intended for the Full Muon variant.
- This is a scientific ablation and is not expected to be cheaper than Muon.
- Do not apply `P_l,t` as the parameter direction in this variant.
- Add this variant to the common `optimizer_name` selector without changing or removing the protected `adamw` path or any earlier variant.

## Task 4.1 — Specify and test the reference Muon transform

### Task goal

Establish one reference-equivalent Muon candidate-step calculation that can serve as the norm oracle now and the full update in Variant 5.

### Required changes

- Document the chosen reference source/version and any locally reproduced code.
- Implement or integrate the Newton–Schulz/polar-factor approximation as a focused function.
- Record its coefficients, iteration count, input normalization, transpose handling for tall/wide matrices, numerical dtype, and shape scaling.
- Separate the geometry transform `P_l,t` from the final candidate-step scale `eta_muon * s_l`.
- Add numerical guards for zero and near-zero momentum.
- Avoid a new dependency unless separately approved.

### Task passing criteria

- Tests cover square, tall, wide, zero, and near-zero matrices.
- For small full-rank fp32 matrices, the transform is compared with an SVD polar factor using documented tolerances appropriate to the selected iteration count.
- The implementation produces finite results in supported training dtypes.
- Transpose handling returns the original matrix shape.
- The exact reference details are saved in configuration/run metadata.

## Task 4.2 — Implement Muon norm-oracle SGDM

### Task goal

Apply the candidate Muon step norm to the SGDM momentum direction for each eligible matrix.

### Required changes

- Add `muon_scaled_sgdm` as a selectable optimizer variant.
- Form the same momentum matrix used by global SGDM.
- Compute the reference Muon transform and candidate-step norm using the documented Muon LR and shape scale.
- Rescale the SGDM momentum to that norm using a validated epsilon.
- Explicitly discard the Muon direction before constructing the parameter update.
- Keep auxiliary AdamW and parameter grouping identical to prior variants.
- Save all oracle and transform settings in checkpoint metadata.

### Task passing criteria

- With weight decay disabled, `||Delta W_oracle||_F` matches the actual candidate Muon step norm for every tested eligible matrix within tolerance.
- Frobenius cosine similarity between `Delta W_oracle` and `M` is `-1` for nonzero momentum.
- A test demonstrates that `Delta W_oracle` is generally not equal to the Muon-direction update even though their norms match.
- Zero momentum produces a finite zero oracle update.
- The parameter update never uses `P_l,t` as its direction.
- Explicit `optimizer_name = 'adamw'` remains independently runnable and does not construct the Muon transform.

## Task 4.3 — Add paired oracle diagnostics

### Task goal

Provide direct evidence that magnitude is matched while geometry remains SGDM-like.

### Required changes

- At selected steps, log SGDM momentum norm, candidate Muon-transform norm, candidate Muon-step norm, oracle-step norm, relative norm error, and effective oracle multiplier.
- Log Frobenius cosine similarity between oracle update and momentum.
- Log Frobenius cosine similarity between the candidate full-Muon update and ordinary SGDM/oracle update.
- For a small selected subset of matrices/checkpoints, record singular values of momentum, normalized SGDM/oracle update, and candidate Muon update.
- Control frequency and matrix selection to prevent material runtime/storage overhead.

### Task passing criteria

- A unit test verifies all diagnostic formulas against direct tensor calculations.
- Relative norm error is within the predefined tolerance for nonzero test matrices.
- Oracle-to-momentum cosine is `-1`, while Muon-to-SGDM cosine is measured rather than assumed.
- Singular spectra are logged only at explicitly selected checkpoints.
- Diagnostics distinguish geometry differences from norm differences.

## Task 4.4 — Verify checkpointing, smoke behavior, and comparison protocol

### Task goal

Make the oracle variant reproducible and define the comparison needed to isolate Muon's target-magnitude rule.

### Required changes

- Test checkpoint/resume for momentum, auxiliary AdamW state, transform settings, and oracle settings.
- Run a short smoke test covering forward/backward, clipping, oracle calculation, optimizer step, evaluation, and save/resume.
- Define a fresh LR search for the oracle variant with a budget comparable to other families.
- Compare it directly with best Frobenius-normalized SGDM and, after Variant 5 exists, full Muon under matched seeds/data/tokens/settings.
- Include transform overhead, optimizer-step time, and peak memory in the planned measurements.

### Task passing criteria

- Resumed execution produces the same next update as uninterrupted execution within the supported determinism tolerance.
- Smoke testing produces finite losses and update diagnostics.
- The planned comparison holds parameter groups, auxiliary optimizer, momentum convention, shape scale, and weight decay constant.
- Primary selection uses validation loss at a fixed processed-token budget.
- No expensive sweep begins without separate approval.

## Variant completion criteria

- Tasks 4.1 through 4.4 are completed and approved sequentially.
- Oracle SGDM matches candidate Muon update norms while preserving SGDM direction.
- The transform is reference-documented and reusable by Full Muon.
- Tests demonstrate norm equality and direction separation rather than assuming them.
- The comparison protocol supports, but does not prejudge, hypothesis H4.
