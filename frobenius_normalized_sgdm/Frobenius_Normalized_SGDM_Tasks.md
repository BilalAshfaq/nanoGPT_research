# Frobenius-Normalized SGDM Tasks

## Variant goal

Test whether dynamic per-matrix magnitude normalization can improve on static
per-matrix SGDM while preserving the SGDM momentum direction and avoiding any
Muon computation.

For eligible matrix `l`, define

```text
r_l = min(d_out_l, d_in_l)
q_l = fixed_shape_factor_l
N_l,t = q_l * sqrt(r_l) * M_l,t / (||M_l,t||_F + epsilon)
Delta W_l,t = -eta_t * N_l,t
```

The first study fixes `q_l = 1.0` for every eligible matrix. Therefore its
default rule is exactly

```text
Delta W_l,t =
    -eta_t * sqrt(r_l) * M_l,t / (||M_l,t||_F + epsilon).
```

The denominator is additive `||M||_F + epsilon`; it is not
`max(||M||_F, epsilon)`. For nonzero momentum, the nominal normalized-matrix
target is `q_l * sqrt(r_l)`, while the exact epsilon-adjusted expected norm is

```text
||N_l,t||_F =
    q_l * sqrt(r_l) * ||M_l,t||_F / (||M_l,t||_F + epsilon).
```

Consequently, `||Delta W_l,t||_F` approaches
`eta_t * q_l * sqrt(r_l)` when `||M_l,t||_F >> epsilon`, but is intentionally
smaller in the epsilon-dominated regime. Tests and diagnostics must not confuse
the nominal target with this exact expected norm.

Any non-unit fixed shape factor is outside the first study and requires a
separately approved protocol. This variant must never compute a Muon transform,
SVD, polar decomposition, or Newton–Schulz iteration.

## Prerequisites and scope constraints

- Tuned Global SGDM and Static Per-Matrix SGDM implementation tasks must be
  approved first.
- Reuse the shared eligible/auxiliary partition, EMA momentum primitive,
  auxiliary AdamW construction, composite optimizer, scheduler, checkpointing,
  metadata, experiment-manifest, result-selection, and diagnostic framework.
- Keep the normalization primitive and optimizer behavior inside
  `frobenius_normalized_sgdm/`; promote only functionality already needed by at
  least two variants to `shared_utils/` as part of an explicitly approved task.
- Do not import another variant's private utilities. In particular, do not
  reuse static-multiplier code to represent dynamic normalization.
- Normalization acts on the post-accumulation, post-unscale, post-clipping SGDM
  momentum used by the optimizer. It must not normalize each raw micro-batch
  gradient independently.
- Preserve the momentum convention
  `M_t = beta * M_(t-1) + (1 - beta) * G_t`, with Nesterov disabled.
- Preserve the common parameter partition and auxiliary AdamW settings exactly.
- Preserve `adamw`, `global_sgdm`, and `static_per_matrix_sgdm` behavior and
  register this variant through the same `optimizer_name` selector and training
  entry point.
- Do not launch a broad tuning run while Tasks 3.1 through 3.4 are incomplete.
- The current repository does not restore RNG state on checkpoint resume. Until
  that separately approved shared reproducibility work is completed and tested,
  a resumed study run is not eligible for a matched data-order comparison. It
  must be reported as such and may not silently replace an uninterrupted result.

## Task 3.1 — Implement the normalization rule as a focused primitive

### Task goal

Create a numerically safe, directly testable, variant-local function that
converts one dense two-dimensional SGDM momentum matrix into the exact
additive-epsilon normalized matrix defined above.

### Required changes

- Add a focused function accepting a momentum matrix, positive finite epsilon,
  and positive finite fixed shape factor. The first-study default shape factor
  is `1.0`.
- Reject non-two-dimensional inputs and invalid epsilon or shape-factor values
  with clear errors.
- Compute the Frobenius norm and all scalar scaling arithmetic in FP32 or higher
  precision even when the input is FP16 or BF16. Return the normalized matrix in
  the input dtype unless a documented supported execution path requires a safer
  dtype.
- Compute `r_l = min(d_out_l, d_in_l)` from the actual matrix shape and compute
  the nominal target as `fixed_shape_factor * sqrt(r_l)`.
- Use the exact additive denominator `raw_norm + epsilon`. Do not silently use a
  clamp, branch to a different normalization equation, or normalize by a raw
  micro-batch gradient.
- Special-case exactly zero momentum as a finite zero normalized matrix. For
  diagnostics, report zero applied multiplier, `zero_momentum = true`, and
  `epsilon_dominated = true`; do not expose a meaningless `target / epsilon`
  multiplier for the exact-zero case.
- For nonzero momentum, expose enough detached scalar information to verify the
  equation without retaining full tensors or computation graphs: raw norm,
  denominator, rank `r_l`, fixed shape factor, nominal target norm, exact
  epsilon-adjusted expected norm, applied normalization multiplier,
  `zero_momentum`, and `epsilon_dominated` where the latter means
  `raw_norm <= epsilon`.
- Avoid a new dependency and avoid any Muon, SVD, polar, or Newton–Schulz code.

### Task passing criteria

- Deterministic FP32/FP64 tests cover square, tall, and wide nonzero matrices and
  match the exact formula, including the epsilon-adjusted expected norm.
- Exactly zero input produces a finite zero output and the documented zero-event
  metadata.
- Nonzero inputs below, at, and above epsilon match the additive-denominator
  formula and never produce NaN or Inf.
- Supported FP16/BF16 execution paths use sufficiently precise norm arithmetic
  and produce finite outputs for near-zero values.
- Positive rescaling of a matrix is approximately scale-invariant only in the
  documented `raw_norm >> epsilon` regime; epsilon-dominated tests verify the
  exact formula rather than asserting false scale invariance.
- The normalized output is collinear with the input for every nonzero test
  matrix, with Frobenius cosine similarity `+1` within dtype-appropriate
  tolerance.
- The implementation and its imports contain no Muon, SVD, polar-decomposition,
  or Newton–Schulz call.

## Task 3.2 — Integrate Frobenius-normalized SGDM

### Task goal

Add a training variant that maintains ordinary SGDM momentum and dynamically
normalizes each eligible matrix after momentum formation, without changing the
shared training loop or auxiliary optimizer.

### Required changes

- Add `frobenius_normalized_sgdm` as a selectable optimizer variant through the
  shared optimizer factory.
- Reuse the exact shared EMA momentum update. Do not create an independent
  momentum implementation.
- Add explicit configuration named `frobenius_learning_rate`,
  `frobenius_epsilon`, and `frobenius_shape_factor`. The first-study default for
  `frobenius_shape_factor` is `1.0`; momentum and matrix weight decay continue
  to use the common experimental settings.
- Apply the normalization primitive to the momentum buffer, then apply the
  scheduled Frobenius base LR. With decoupled weight decay, the exact order is

  ```text
  W_l <- (1 - eta_t * matrix_weight_decay) * W_l
  W_l <- W_l - eta_t * q_l * sqrt(r_l)
                  * M_l,t / (||M_l,t||_F + epsilon)
  ```

  where `eta_t` is the scheduled value derived from
  `frobenius_learning_rate`. The normalization scale must not multiply or
  otherwise alter the weight-decay contribution.
- Keep all eligible matrices in one practical optimizer implementation unless
  measurements demonstrate that another grouping is necessary. Do not create
  one optimizer object per matrix by default.
- Reject sparse gradients and validate every owned eligible parameter is dense
  and two-dimensional.
- Keep the shared eligible/auxiliary partition and auxiliary AdamW construction
  unchanged.
- Scale `frobenius_learning_rate` and the auxiliary base LR with the same
  dimensionless schedule shape while preserving their configured ratio.
- Save optimizer identity, normalization equation/version, epsilon, shape
  factor, Frobenius base LR, momentum, weight-decay semantics, parameter-group
  signature, and complete optimizer state in checkpoint/run metadata.
- Validate all normalization settings and optimizer group compatibility before
  loading checkpoint state.
- Preserve the training-loop-facing `step`, `zero_grad`, `param_groups`,
  `state_dict`, and `load_state_dict` behavior required by AMP, clipping, DDP,
  scheduling, and checkpointing.

### Task passing criteria

- For synthetic nonzero momentum matrices with weight decay disabled, observed
  parameter deltas match
  `-eta_t * q_l * sqrt(r_l) * M/(||M||_F + epsilon)` directly.
- Tests separately verify the nominal target and the exact epsilon-adjusted
  expected step norm; exact equality to `eta_t * q_l * sqrt(r_l)` is required
  only where the declared tolerance makes the epsilon term negligible.
- Frobenius cosine similarity between the normalization-only parameter update
  and nonzero momentum is `-1` within dtype-appropriate tolerance.
- Exactly zero momentum produces a finite zero momentum-derived update. Any
  configured decoupled weight decay remains independently observable.
- With nonzero matrix weight decay, tests verify decay uses only the scheduled
  Frobenius base LR and is not multiplied by the dynamic normalization factor.
- Momentum buffers match Global SGDM under identical gradient sequences,
  momentum settings, and step-skipping behavior.
- Eligible and auxiliary parameters remain complete, disjoint, and owned by the
  intended optimizers.
- Scheduler tests verify both Frobenius and auxiliary effective rates at
  multiple schedule scales.
- State serialization and restoration produce the same next update as an
  uninterrupted execution for identical model, gradients, and RNG-independent
  inputs.
- A tiny supported mixed-precision smoke test completes forward, backward,
  clipping, optimizer step, evaluation, save, and resume without numerical
  failure.
- Omitted or explicit `optimizer_name = 'adamw'` continues to pass the protected
  AdamW regression path, and Variants 1 and 2 remain unchanged.

## Task 3.3 — Add normalization diagnostics and regression tests

### Task goal

Provide direct evidence that the dynamic rule is being applied as specified,
while keeping nominal targets, epsilon-adjusted candidate norms, observed total
parameter deltas, and weight-decay effects mathematically distinct.

### Required changes

- At selected diagnostic steps, record for every eligible matrix: momentum
  norm, denominator, rank, shape factor, nominal normalized-matrix target norm,
  epsilon-adjusted expected normalized-matrix norm, scheduled Frobenius LR,
  expected momentum-derived step norm, observed total parameter-delta norm,
  update-to-weight ratio, normalization multiplier, full effective multiplier,
  `zero_momentum`, and `epsilon_dominated`.
- Define the normalization multiplier for nonzero momentum as
  `q_l * sqrt(r_l) / (||M_l,t||_F + epsilon)` and the full effective multiplier
  as that value times scheduled `eta_t`. For exactly zero momentum, report both
  as zero together with the zero event.
- Report the decoupled weight-decay contribution separately. Do not label the
  norm of the total observed parameter delta as the normalization target when
  weight decay is nonzero.
- Log Frobenius cosine similarity between the momentum-derived update and
  momentum for nonzero matrices. Assert `-1` for that component; assert `-1`
  for the total observed delta only in tests where weight decay is disabled.
- Reuse scalar metadata already required by the optimizer step where practical.
  When diagnostics are disabled, avoid parameter snapshots, extra diagnostic
  norms, cosine calculations, spectral calculations, and retained tensors. The
  norm inherently required by the optimizer is not considered diagnostic
  overhead.
- Add checkpoint/resume coverage for normalization settings, momentum state,
  auxiliary AdamW state, and the next deterministic optimizer update.
- Add selector, scheduler, parameter-partition, protected AdamW, Global SGDM,
  and Static Per-Matrix SGDM regression tests.

### Task passing criteria

- Every logged scalar is verified against direct tensor calculations on
  deterministic square, tall, wide, zero, and epsilon-dominated inputs.
- Logged observed total-update norms agree with before/after parameter deltas.
- With weight decay disabled, expected momentum-derived and observed update
  norms agree with the exact additive-epsilon equation.
- With weight decay enabled, the normalization and decay contributions are
  reported separately and neither is misidentified as the total update.
- Direction cosine is `-1` relative to momentum for the momentum-derived update
  of every nonzero test matrix.
- Resume with identical deterministic inputs produces the same next parameters,
  momentum buffers, and auxiliary AdamW state as uninterrupted execution.
- Diagnostics retain no full update, momentum, or computation graph after the
  selected step and add no extra matrix-norm work when disabled beyond the norm
  required to perform normalization itself.
- Focused regression tests prove all earlier optimizer choices still construct
  and update according to their existing equations.

## Task 3.4 — Freeze the tuning and comparison protocol

### Task goal

Define and materialize a fair, reviewable study design for dynamic generic
normalization before any broad Variant 3 result is inspected.

### Required changes

- Define exactly 12 exploratory Frobenius-normalized configurations at seed
  `1337`, matching the Variant 1 global-SGDM and Variant 2 static exploratory
  budgets. Use a frozen Cartesian product of exactly four Frobenius base LRs and
  the three predeclared momentum values `{0.90, 0.95, 0.99}`.
- Choose and document a fresh four-value LR grid appropriate to normalized
  updates. Do not copy the raw Global SGDM LR grid without justification. If a
  bounded non-budget calibration is required to set the range, define and obtain
  separate approval for it, complete it before freezing the exploratory
  manifest, and do not use its outcomes as study results.
- Fix `frobenius_epsilon` to one documented value and
  `frobenius_shape_factor = 1.0` for every exploratory and confirmation run.
  Neither setting is tuned in the first study.
- Before results are inspected, materialize the exact LR values, momentum
  values, seed, run names, normalization settings, tie-breaking order, token
  budget, evaluation checkpoints, and all overrides in a deterministic candidate
  design and unauthorized manifest.
- Use the same GPT-2 124M/OpenWebText controls as the approved Variant 1 and 2
  studies: initialization and data-order seed, tokenizer, context length,
  effective batch and accumulation, two-GPU world size, precision, compilation,
  clipping, warmup, cosine schedule shape, weight decay, auxiliary AdamW,
  evaluation batches/checkpoints, hardware class, environment lock, and dataset
  fingerprints.
- Reuse selection at step `999`, corresponding to `491,028,480` processed
  tokens, with the existing hard limit of 1,000 attempted updates and
  `491,520,000` tokens. Rank by unrounded step-999 validation loss; break an
  exact tie by lower Frobenius base LR and then lower momentum.
- Reference immutable selected Variant 1 and Variant 2 run IDs, configurations,
  selection records, and mapping fingerprints. Audit any shared training-code
  changes between their recorded Git commits and the Variant 3 commit; rerun a
  comparator or obtain approval for a documented limitation if a change affects
  a matched control or measurement.
- Use confirmation seeds `2027` and `4099` for the selected Variant 3
  configuration and combine them with its valid seed-`1337` exploratory result.
- Require three successful matched seeds for the selected Global SGDM, Static
  Per-Matrix SGDM, and Frobenius-Normalized SGDM configurations before making a
  final H3 claim. Missing Variant 1 or 2 confirmations do not block mechanical
  Variant 3 implementation, but they do block a final three-family claim.
- Define run names that encode optimizer, model, Frobenius LR, momentum, weight
  decay, seed, and a deterministic normalization-rule identifier containing the
  shape rule and epsilon identity.
- Record completed, failed, divergent, interrupted, resumed, and numerically
  unstable outcomes. Do not replace a candidate or change the grid after seeing
  results without a separately approved follow-up protocol.

### Task passing criteria

- The candidate design and manifest contain exactly 12 exploratory
  configurations and are deterministic under repeated materialization.
- Search budget, exact candidates, selection criterion, tie-breaking order,
  confirmation seeds, and maximum processed-token budget are fixed before
  launch authorization.
- The protocol gives normalized SGDM its own LR search and a momentum search
  comparable to Variant 1 rather than assuming an earlier optimizer's winning
  values transfer.
- All matched controls are enumerated and machine-checkable where practical;
  comparator run identities and static mapping fingerprints cannot change
  silently.
- The plan explicitly distinguishes an optimization gain from instability,
  epsilon-dominated behavior, diagnostic overhead, resume/data-order mismatch,
  or an unmatched auxiliary setting.
- No broad run is authorized until focused tests and one representative smoke
  run pass.

## Task 3.5 — Execute the approved normalized study and report comparison

### Task goal

Run the separately approved frozen protocol, select and confirm the Variant 3
configuration, and produce the artifacts required for direct comparison with
the selected Variant 1 and Variant 2 configurations.

### Required changes

- Start only after Tasks 3.1 through 3.4 are completed and approved, the frozen
  manifest is reviewed, and the user explicitly authorizes the smoke and study
  compute budgets.
- Extend the shared manifest, outcome, selection, confirmation, and reporting
  utilities only as needed for the Frobenius-specific scalar settings and
  normalization-rule identity. Do not create a parallel experiment runner.
- Run one successful representative smoke configuration before authorizing the
  exploratory manifest. The smoke must cover forward/backward, accumulation,
  unscale where applicable, clipping, normalization, both optimizer children,
  evaluation, diagnostics, checkpoint save, and deterministic next-update
  resume coverage.
- Launch only the 12 frozen exploratory entries. Record every completed, failed,
  divergent, interrupted, resumed, or unstable outcome without silent
  replacement.
- Under the current resume limitation, exclude a resumed run from matched
  selection unless RNG-state restoration was separately implemented and
  verified before launch. Any approved rerun must retain an audit trail and may
  not be introduced silently.
- Select the seed-`1337` winner solely by the predefined unrounded validation
  loss and deterministic tie-breaking rule at the fixed token checkpoint.
- Materialize a separate, unauthorized confirmation manifest for only seeds
  `2027` and `4099`; launch it only after explicit approval.
- Combine the two successful confirmation runs with the valid exploratory
  winner and report individual losses, arithmetic mean, sample standard
  deviation, failures, instability, clipping, timing, memory, epsilon-dominated
  events, and the exact winning normalization configuration.
- Compare the selected Frobenius configuration directly with the immutable
  selected Global SGDM and Static Per-Matrix SGDM configurations under verified
  matched controls. Report exploratory one-seed comparisons separately if the
  earlier confirmation seeds remain incomplete.
- Preserve manifests, candidate design, complete configurations, checkpoints,
  evaluation logs, diagnostics, optimizer states, parameter audits, environment
  and dataset fingerprints, Git commits, outcome records, selection report,
  confirmation manifest, and final comparison report.
- Separate observed measurements, interpretations, and unsupported hypotheses.
  Do not claim H3 without three successful matched seeds for all three selected
  optimizer configurations.

### Task passing criteria

- The smoke completes successfully before the exploratory launch.
- Every authorized exploratory and confirmation manifest entry has a recorded
  outcome; no unapproved configuration is introduced after results are seen.
- The winner is selected solely by the frozen rule at the fixed processed-token
  checkpoint.
- The selected Variant 3 configuration has three successful seed results before
  any final claim about it is made.
- The comparison verifies model/data/batching/schedule/numerical/auxiliary
  controls, dataset and environment fingerprints, comparator identities, static
  mapping fingerprint, normalization settings, and resume eligibility.
- The final report includes individual seeds, mean, sample standard deviation,
  failures or instability, timing, memory, clipping, and normalization-event
  measurements—not only the best loss.
- Any H3 conclusion is based on best tuned Global SGDM versus best tuned Static
  Per-Matrix SGDM versus best tuned Frobenius-Normalized SGDM and is explicitly
  limited by the available matched multi-seed evidence.

## Variant completion criteria

- Tasks 3.1 through 3.5 are completed and approved sequentially; Task 3.5 also
  requires explicit authorization for each compute-bearing launch.
- The optimizer implements the exact additive-epsilon Frobenius rule, dynamically
  controls each eligible matrix's momentum-derived update magnitude, and
  preserves SGDM direction.
- The first study fixes the shape factor to `1.0`, tunes its own Frobenius base
  LR and momentum within a frozen comparable budget, and contains no Muon
  computation.
- Numerical edge cases, mixed precision, weight-decay separation, diagnostics,
  checkpointing, deterministic next-update restoration, and regressions for all
  earlier optimizer paths are tested.
- All optimizer choices remain selectable through the same training entry point,
  and AdamW remains the omitted/default behavior.
- Study artifacts support a direct matched comparison with Variants 1 and 2;
  final H3 claims additionally require three successful matched seeds for every
  selected comparator.
