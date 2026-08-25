# Full Muon Tasks

## Variant goal

Integrate or verify the full reference Muon update for eligible Transformer matrices, with the same auxiliary AdamW groups and settings used by all SGDM variants. This provides the reference containing both per-matrix magnitude control and within-matrix singular-value transformation.

```text
P_l,t = MuonTransform(M_l,t)
Delta W_l,t = -eta_muon * s_l * P_l,t
```

The central final comparison is full Muon versus Muon-scaled SGDM after confirming that their per-matrix candidate update norms match.

## Prerequisites and scope constraints

- All four earlier variants must be completed and approved.
- Reuse the exact Muon transform, coefficients, iteration count, normalization, transpose handling, shape scaling, and momentum convention established for the norm oracle.
- Reuse the shared eligible/auxiliary partition and identical auxiliary AdamW settings.
- Do not silently substitute a different Muon implementation or change experimental settings to favor either optimizer.
- Preserve AdamW as the repository default.
- Register Full Muon as the final value in the common `optimizer_name` selector. AdamW plus all five experimental variants must remain independently selectable from the same `train.py` entry point.

## Task 5.1 — Audit reference equivalence and shared settings

### Task goal

Confirm that the transform introduced for the norm oracle and the planned full update reproduce the chosen Muon reference exactly enough for scientific comparison.

### Required changes

- Produce a code-level audit of momentum convention, Nesterov use, Newton–Schulz coefficients/iterations, input normalization, shape scaling, weight decay, eligible parameters, auxiliary optimizer, and numerical dtype.
- Compare each setting with the pinned reference source/version.
- Resolve and document any unavoidable deviation before enabling full training.
- Confirm that the oracle candidate step and full-Muon candidate step are produced by the same shared function/settings.
- Add a reference-settings fingerprint to checkpoint/run metadata.

### Task passing criteria

- The audit contains no unknown or implicit optimizer setting.
- Reference-equivalence tests pass on deterministic synthetic matrices.
- The oracle and full-Muon paths report identical candidate Muon norms for identical momentum inputs.
- Any deviation from the reference has explicit justification and approval.
- No duplicate independent Muon-transform implementation exists.

## Task 5.2 — Integrate the full Muon update

### Task goal

Apply the reference Muon-transformed direction to eligible matrices through the existing training entry point.

### Required changes

- Add `full_muon` as a selectable optimizer variant.
- Maintain momentum exactly as defined by the selected reference.
- Apply the shared transform and candidate-step scaling directly as the eligible-matrix update.
- Implement the documented weight-decay behavior in the correct order.
- Keep auxiliary parameters on the same AdamW implementation and settings used in all prior variants.
- Support AMP/GradScaler, clipping, DDP, LR scheduling, checkpoint save/load, and `zero_grad` through the common training-loop interface.

### Task passing criteria

- Synthetic tests match the documented full-Muon equation for multiple shapes and multiple steps.
- Eligible-matrix updates use `P_l,t` as their direction; auxiliary updates do not pass through Muon.
- Candidate full-Muon and oracle-SGDM update norms match within tolerance when configured identically.
- At least one nontrivial test matrix demonstrates different Muon and SGDM directions.
- Optimizer state round-trips through a checkpoint and produces the same next update as uninterrupted execution.

## Task 5.3 — Add final Muon diagnostics and performance measurements

### Task goal

Collect the evidence needed to attribute any remaining performance gap to within-matrix geometry after magnitude matching.

### Required changes

- Log all common per-matrix norms and relative update norms at selected checkpoints.
- Log Muon-to-SGDM Frobenius cosine similarity and paired full-Muon/oracle norm error.
- For selected matrices/checkpoints, log singular values of momentum, oracle SGDM update, and full Muon update.
- Measure mean optimizer-step time, wall-clock time, and peak GPU memory using a consistent procedure.
- Report diagnostics separately from validation outcomes and avoid causal claims in raw logs.

### Task passing criteria

- Paired diagnostics confirm matched candidate norms before any geometry conclusion is allowed.
- Cosine and singular-value diagnostics demonstrate whether directions/geometries differ on actual training steps.
- Timing excludes or separately labels optional heavy diagnostic overhead.
- Diagnostics remain frequency-controlled and do not store full matrices.
- All measurements include matrix names and processed-token checkpoints.

## Task 5.4 — Run final smoke and regression verification

### Task goal

Verify that all six optimizer choices—AdamW plus the five experimental variants—remain runnable from the same entry point before any expensive experiment.

### Required changes

- Run the focused optimizer unit-test suite.
- Run a short smoke configuration for AdamW and each of the five experimental variants.
- Exercise evaluation, clipping, checkpoint save, and resume for full Muon.
- Review the complete diff for unrelated changes and unnecessary runtime overhead.
- Confirm default behavior still selects the original AdamW path.
- Confirm explicit `--optimizer_name=adamw` selects the same path as the omitted/default setting.
- Confirm every accepted optimizer name selects exactly its requested implementation and unknown names fail clearly.

### Task passing criteria

- All focused unit tests pass.
- Every optimizer variant completes at least one finite training step and evaluation cycle.
- Full Muon saves and resumes successfully.
- Default AdamW behavior passes its regression checks.
- All six explicit optimizer selections pass construction and short smoke coverage.
- No large sweep has been launched.

## Task 5.5 — Define and execute the final experimental protocol only after approval

### Task goal

Produce the controlled evidence needed to separate update-magnitude effects from Muon's within-matrix geometry.

### Required changes

- Predefine comparable tuning budgets for global SGDM, static per-matrix SGDM, Frobenius-normalized SGDM, Muon-scaled SGDM, and full Muon.
- Preserve and run AdamW as an additional comparison baseline under the same architecture, initialization, data order, processed-token budget, evaluation checkpoints, and reporting protocol. Its inclusion must not reduce the comparable tuning budgets assigned to the five causal variants unless explicitly approved.
- Match architecture/initialization, data order, tokenizer, sequence length, effective batch, total tokens, precision, clipping, schedule, regularization, evaluation checkpoints, and auxiliary AdamW settings.
- Select configurations by validation loss after a fixed processed-token budget.
- Run at least three seeds for promising final configurations and claimed differences.
- If compute permits, repeat final comparisons across at least two model sizes and two batch sizes.
- Report mean/standard deviation, instability, timing, memory, and sensitivity—not only the best run.

### Task passing criteria

- Full Muon is compared directly with Muon-scaled SGDM only after paired norm matching is verified.
- Best global SGDM is compared with best static per-matrix SGDM before claims about non-uniform rates.
- Frobenius-normalized and oracle variants receive independently appropriate tuning.
- AdamW results are reported separately and identify whether they use the preserved repository defaults or a separately approved tuning budget.
- Observations, interpretations, and unsupported hypotheses are clearly separated.
- A claim for hypothesis H5 requires a consistent full-Muon advantage across seeds with matched update norms and all listed controls.

## Variant completion criteria

- Tasks 5.1 through 5.5 are completed and approved sequentially; Task 5.5 requires explicit authorization for expensive runs.
- Full Muon uses the same reference transform and candidate-step definition as the norm oracle.
- Parameter grouping and auxiliary AdamW settings are identical across all experimental variants.
- AdamW and every experimental variant remain explicitly selectable through `optimizer_name`; omission still defaults to AdamW.
- Tests, smoke runs, checkpointing, diagnostics, and performance measurements pass.
- The final protocol can answer how much Muon's advantage comes from per-matrix magnitude control and whether its within-matrix geometry adds a consistent benefit.
