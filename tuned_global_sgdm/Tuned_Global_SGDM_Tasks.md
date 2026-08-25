# Tuned Global SGDM Tasks

## Variant goal

Establish a strong, reproducible SGDM baseline for the Transformer hidden-layer matrices while keeping all ineligible parameters on a fixed auxiliary AdamW configuration. Every eligible matrix must use the same SGDM learning rate and momentum settings. This variant is the reference against which all non-uniform SGDM variants are judged.

Conceptual eligible-matrix update:

```text
M_t = beta * M_(t-1) + (1 - beta) * G_t
Delta W_t = -eta * M_t
```

This variant must use the momentum equation shown above. Nesterov momentum is disabled so the implemented update has one unambiguous definition. The placement and mathematical effect of weight decay must be stated explicitly and verified from the code.

## Phase 0 baseline status

- The current repository provides AdamW through `GPT.configure_optimizers`.
- The current repository does not provide an SGDM implementation, a Muon implementation, or a split experimental-optimizer/auxiliary-AdamW path to reproduce.
- These implementation tasks may proceed in the approved sequence because the missing baselines cannot yet be reproduced.
- Do not run or interpret causal comparisons between SGDM and Muon until tuned global SGDM and full Muon have both been implemented, mechanically verified, and tuned with comparable budgets.

## Scope constraints

- Preserve the current AdamW-only behavior as the default.
- Treat AdamW as a protected comparison baseline, not merely a fallback implementation.
- Use one explicit configuration flag, `optimizer_name`, for optimizer selection from the existing `train.py` entry point. The final accepted values must be `adamw`, `global_sgdm`, `static_per_matrix_sgdm`, `frobenius_normalized_sgdm`, `muon_scaled_sgdm`, and `full_muon` as those variants are implemented.
- Omitting the flag must select `adamw`; explicitly passing `--optimizer_name=adamw` must select the same path and produce equivalent behavior under the same seed and configuration.
- Reject unknown optimizer names before model training rather than silently falling back to AdamW or another optimizer.
- Use one common parameter-partition function for this and all later variants.
- Treat the four hidden projection types in every Transformer block as Muon-eligible:
  `attn.c_attn.weight`, `attn.c_proj.weight`, `mlp.c_fc.weight`, and `mlp.c_proj.weight`.
- Exclude token/position embeddings, the tied output head, biases, and LayerNorm parameters from SGDM. They must use auxiliary AdamW.
- Handle the tied `transformer.wte.weight`/`lm_head.weight` parameter exactly once.
- Do not launch a large tuning sweep as part of implementation.

## Task 1.1 — Add and audit the shared parameter partition

### Task goal

Create one deterministic source of truth that separates eligible hidden matrices from auxiliary parameters and makes the partition inspectable by name.

### Required changes

- Add a small parameter-partition helper near optimizer construction or in a focused optimizer module.
- Return named parameters for both the eligible-matrix group and auxiliary group.
- Validate that every trainable parameter is assigned to exactly one correct optimizer group, while frozen parameters are excluded.
- Add an audit output that records parameter name, shape, parameter count, assigned optimizer, and weight-decay treatment.
- Document that the token embedding and output head share one weight tensor, and confirm that this shared parameter is assigned to auxiliary AdamW exactly once.
- Keep the existing AdamW grouping available for the default optimizer path.

### Task passing criteria

- A unit test proves that every trainable parameter appears in exactly one group.
- A unit test proves that all expected attention/MLP matrices are eligible for multiple model depths.
- A unit test proves that embeddings, tied output head, LayerNorm parameters, and biases are auxiliary.
- The audit is deterministic and reports totals equal to the model's trainable parameter count.
- Running the existing default path still constructs the same AdamW decay/no-decay groups.

## Task 1.2 — Implement the global SGDM matrix optimizer path

### Task goal

Implement the exact global SGDM update for eligible matrices and coordinate it with auxiliary AdamW without duplicating the training loop.

### Required changes

- Add the common string selector `optimizer_name`, initially supporting `adamw` and `global_sgdm` and designed to accept later variant names without changing the training loop.
- Keep the AdamW branch routed through the existing `GPT.configure_optimizers` behavior unless an approved compatibility wrapper is required; do not reimplement AdamW mathematics.
- Implement the exact convention `M_t = beta * M_(t-1) + (1 - beta) * G_t`, followed by `Delta W_t = -eta * M_t` when weight decay is disabled.
- Add explicit configuration for matrix learning rate, momentum, and weight decay. Keep Nesterov momentum disabled for this variant.
- Keep auxiliary AdamW settings separately configurable and identical in meaning across later variants.
- Provide a single training-loop-facing optimizer interface supporting `step`, `zero_grad`, `param_groups`, `state_dict`, and `load_state_dict` so AMP and checkpointing continue to work.
- Ensure LR scheduling changes the intended matrix and auxiliary rates without accidentally replacing their relative scales.
- Make matrix and auxiliary weight-decay semantics explicit in saved configuration and audit output.

### Task passing criteria

- On a synthetic matrix with known gradients, the first and subsequent observed parameter changes match the documented SGDM equations within numerical tolerance.
- Tests compare the expected update with the actual before/after parameter difference, rather than validating only an intermediate candidate update.
- Tests cover zero momentum, nonzero momentum, and the documented weight-decay behavior, and confirm that no Nesterov modification is applied.
- Auxiliary parameters change only through AdamW; eligible matrices change only through SGDM.
- `zero_grad(set_to_none=True)` works across both parameter sets.
- Optimizer state can be serialized, restored, and used to produce the same next update as an uninterrupted run.
- AdamW remains the default and its existing construction behavior is unchanged.
- An unknown `optimizer_name` produces a clear configuration error before the first batch is loaded.

## Task 1.3 — Integrate configuration, scheduling, logging, and checkpoints

### Task goal

Make global SGDM runnable and reproducible from the existing `train.py` entry point.

### Required changes

- Expose all new scalar settings through the current configuration mechanism.
- Make the random seed an explicit configuration value rather than relying on a hard-coded training seed.
- Ensure optimizer selection works both in a configuration file and through an explicit command-line override such as `--optimizer_name=adamw` or `--optimizer_name=global_sgdm`.
- Include optimizer identity, momentum convention, parameter-group audit, matrix/auxiliary settings, random seed, Git commit hash, model architecture and parameter count, dataset and tokenizer identifiers, sequence length, batch size and gradient accumulation, precision, processed-token count, optimizer-step count, and hardware information in checkpoint/run metadata.
- Update checkpoint resume logic to validate optimizer type and compatible group structure before loading state.
- Log both effective matrix LR and auxiliary LR.
- Record training and validation loss, processed tokens, total wall-clock time, mean optimizer-step time, peak GPU memory, divergence or numerical-instability status, and gradient-clipping frequency for each run. Mark GPU-only measurements as not applicable during CPU runs rather than inventing values.
- Preserve gradient accumulation, gradient clipping, GradScaler, DDP wrapping, evaluation, and W&B behavior.
- Create a dedicated tiny smoke-test configuration only if existing configuration options cannot provide a simple and repeatable test run.

### Task passing criteria

- A short CPU or available-device smoke run completes forward, backward, clipping, optimizer step, evaluation, and checkpoint save.
- A resume smoke test loads the checkpoint and continues without resetting momentum or auxiliary AdamW state.
- Configuration saved in the checkpoint is sufficient to reconstruct the optimizer choice and settings.
- The saved run configuration is sufficient to identify the seed, model, data, batching, precision, optimizer groups, and processed training budget used by the run.
- Processed tokens are derived from effective batch size and iteration count and are available for comparison.
- A smoke run records training/validation loss, processed tokens, wall-clock time, mean optimizer-step time, clipping frequency, and numerical status; it also records peak GPU memory on GPU or explicitly marks that field as not applicable on CPU.
- Run metadata records the Git commit hash and available hardware information.
- Default AdamW smoke behavior remains functional.
- Omitted `optimizer_name` and explicit `--optimizer_name=adamw` pass an equivalence regression test under the same seed and inputs.
- A checkpoint records the selected optimizer name, and resume rejects an incompatible optimizer selection with a clear error.

## Task 1.4 — Verify and log global SGDM updates

### Task goal

Confirm that eligible matrices receive the intended global SGDM updates and provide optional measurements for comparison with later variants.

### Required changes

- Add optional per-matrix diagnostics that are disabled by default.
- Allow diagnostics to run only at configured training steps.
- At each configured diagnostic step, record scalar measurements for every eligible matrix: weight, gradient, momentum, and actual applied-update Frobenius norms; update-to-weight norm ratio; and the scheduled global SGDM learning rate.
- Record update spectral norms only for explicitly selected matrices and steps because they are more expensive to calculate.
- Do not save complete gradients, momentum matrices, or parameter updates.
- Report weight-decay contributions separately if they are part of the update.

### Task passing criteria

- A deterministic unit test verifies every reported value through direct tensor calculations.
- A deterministic test confirms that each reported applied-update norm equals the observed before/after parameter change.
- A short smoke run produces the required scalar diagnostics for every eligible matrix only at the configured steps, and produces spectral norms only for the configured subset.
- Disabling diagnostics avoids additional matrix-norm calculations.
- All eligible matrices still use the same global SGDM learning rate.
- Diagnostic collection does not change model parameters or optimizer results.

## Task 1.5 — Define the baseline tuning and comparison protocol

### Task goal

Define how global SGDM and AdamW will be evaluated fairly before running any expensive tuning experiments.

### Required changes

- Define the candidate global SGDM learning rates and momentum values, or define the exact method used to generate them.
- State whether weight decay, warmup, and learning-rate schedule settings are fixed or tuned.
- Define the maximum number of tuning runs and processed tokens allowed per run.
- Use validation loss at the same processed-token checkpoint to select configurations.
- Define a matched AdamW comparison using the same model architecture and initialization, dataset, tokenizer, sequence length, data order, batch size and gradient accumulation, processed-token budget, learning-rate schedule and warmup policy, weight decay and other regularization, eligible/auxiliary parameter definitions where applicable, auxiliary AdamW settings, evaluation frequency and checkpoints, numerical precision, gradient clipping, and random seed.
- Define a deterministic preflight check that compares AdamW and global SGDM runs with the same seed and training configuration before optimizer-specific updates, confirming identical initialized model parameters and the same sequence of sampled training batches.
- Reserve a comparable tuning budget for the future Muon baseline.
- Separate exploratory single-seed runs from final multi-seed confirmation runs.
- Require at least three seeds for the final selected global SGDM configuration and for any configuration used to support a reported performance claim.
- Define run names and saved metadata containing optimizer, model, learning rate, momentum, weight decay, seed, and scaling rule.

### Task passing criteria

- The complete set of candidate configurations, or the exact rule used to generate them, is documented.
- The number of permitted tuning runs is explicit.
- Global SGDM, AdamW, and future Muon comparisons use the same processed-token and evaluation protocol.
- The protocol explicitly lists every matched control and explains any setting that cannot be identical across optimizer families.
- For the same seed and training configuration, a preflight test confirms that AdamW and global SGDM begin from identical model parameters and consume the same sequence of training batches.
- Exploratory tuning may use one seed, but final reported configurations require at least three seeds.
- Another researcher could reproduce the search and select the same winning configuration from the documented rules.
- No tuning sweep is executed as part of this task.

## Task 1.6 — Execute the approved baseline tuning protocol

### Task goal

Run the separately approved tuning protocol and establish the best tuned global SGDM baseline without changing the predefined search or selection rules after results are observed.

### Required changes

- Start this task only after Tasks 1.1 through 1.5 are approved and the user explicitly authorizes the tuning runs and their compute budget.
- Materialize and save the complete run manifest before launching, including every candidate configuration, seed, processed-token limit, and evaluation checkpoint.
- Run each approved configuration under the matched controls defined in Task 1.5.
- Record completed, failed, divergent, and interrupted runs; do not silently discard unsuccessful configurations.
- Select the best exploratory configuration using the predefined validation-loss criterion at the fixed processed-token checkpoint.
- Run the selected final configuration with at least three predefined seeds.
- Report validation loss and every run-level measurement defined in Task 1.3 for each seed, together with mean and standard deviation where applicable.
- Preserve the complete configurations, checkpoints, logs, parameter-group audits, and run metadata needed to reproduce the result.
- Do not make a causal SGDM-versus-Muon claim until full Muon has been implemented and tuned with a comparable budget.

### Task passing criteria

- Every entry in the approved run manifest has a recorded outcome.
- All successful tuning runs use the same matched controls and processed-token selection checkpoint.
- The winning exploratory configuration is selected solely by the predefined rule.
- The final selected configuration has results from at least three seeds.
- The final report includes individual-seed results, mean, standard deviation, failures or instability, and the exact winning configuration.
- No unapproved configurations or additional tuning runs are introduced after results are inspected.
- Results are reported as baseline measurements without unsupported causal claims about Muon.

## Variant completion criteria

- Tasks 1.1 through 1.6 are completed and approved sequentially; Task 1.6 additionally requires explicit authorization for its compute budget.
- Global SGDM uses one common rate for every eligible matrix and auxiliary AdamW for all other parameters.
- Update mechanics, grouping, checkpoint/resume behavior, and diagnostics are covered by tests.
- Existing AdamW behavior remains the default.
- AdamW can always be selected explicitly with `optimizer_name = 'adamw'` and remains available after every later variant is added.
- The approved tuning protocol has been executed, the final configuration has been evaluated with at least three seeds, and large or additional sweeps still require separate approval.
