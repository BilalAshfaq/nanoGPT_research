# Variant 1 Baseline Tuning and Comparison Protocol

## Status and authorization boundary

This document defines the complete Task 1.5 protocol. It does not authorize or
launch any tuning run. Task 1.6 may begin only after separate approval of this
protocol and its compute budget. Failed, divergent, and interrupted runs count
toward the stated limits and may not be replaced without approval.

## Primary selection rule

Select configurations using validation loss after exactly 999 completed
optimizer updates. With the common batch configuration, this is:

```text
tokens_per_update = 40 * 12 * 1024 = 491,520
selection_tokens = 999 * 491,520 = 491,028,480
```

Set `max_iters = 999`, `eval_interval = 333`, and `always_save_checkpoint =
True`. The evaluation checkpoints are 0, 333, 666, and 999 completed updates.
The repository's existing inclusive termination behavior performs one final
update after the step-999 evaluation, so the hard per-run limit is 1,000
updates, or 491,520,000 processed tokens. Only validation loss saved at step
999 is used for selection; the final uncheckpointed update is ignored.

Rank candidates by the unrounded step-999 validation loss. An exactly tied
loss is broken by the lower base learning rate and then the lower momentum
coefficient (`matrix_momentum` for SGDM or `beta1` for AdamW). Divergent,
failed, interrupted, or missing-step-999 runs cannot win.

## Matched training configuration

Every exploratory and confirmation run uses these controls:

| Setting | Fixed value |
|---|---|
| Model | nanoGPT GPT-2 124M configuration |
| Initialization | scratch |
| Layers / heads / embedding | 12 / 12 / 768 |
| Bias / dropout | `False` / `0.0` |
| Vocabulary / tokenizer | 50,304 / GPT-2 BPE |
| Dataset | prepared OpenWebText files from `data/openwebtext` |
| Sequence length | 1,024 |
| Micro-batch size | 12 |
| Global gradient accumulation | 40 |
| Effective batch | 480 sequences, 491,520 tokens |
| DDP world size | 8 |
| Precision | bfloat16 |
| Compilation | enabled |
| Gradient clipping | global norm 1.0 |
| Schedule | linear warmup then cosine decay |
| Warmup | 100 updates |
| Decay endpoint | step 999 |
| Minimum LR | 0.1 times each optimizer group's base LR |
| Matrix weight decay | 0.1, decoupled |
| Auxiliary weight decay | 0.1, AdamW decoupled |
| Evaluation | 200 batches at steps 0, 333, 666, and 999 |
| Checkpoint policy | always save at evaluation after step 0 |
| Diagnostics | disabled during broad tuning |
| Exploratory seed | 1,337 |
| Confirmation seeds | 1,337; 2,027; 4,099 |

All candidates must run on the same hardware class, DDP world size, PyTorch
version, precision, compilation setting, and prepared dataset files. Record
the hardware and Git metadata already emitted by `train.py`. A run with a
different control is invalid rather than an additional candidate.

For global SGDM, the common schedule multiplier is generated with
`learning_rate = 6e-4` and `min_lr = 6e-5`; it scales both the matrix and
auxiliary base rates without changing their ratio. For AdamW, set `min_lr` to
0.1 times that candidate's `learning_rate`.

## Global SGDM exploratory grid

Use the Cartesian product below, for exactly 12 exploratory runs:

```text
matrix_learning_rate in {0.03, 0.10, 0.30, 1.00}
matrix_momentum in {0.90, 0.95, 0.99}
```

All other global-SGDM settings are fixed:

```text
optimizer_name = global_sgdm
matrix_momentum_convention = ema
matrix_nesterov = False
matrix_weight_decay = 0.1
matrix_weight_decay_mode = decoupled
auxiliary_learning_rate = 6e-4
auxiliary_beta1 = 0.9
auxiliary_beta2 = 0.95
auxiliary_weight_decay = 0.1
auxiliary_weight_decay_mode = adamw_decoupled
seed = 1337
```

The grid tunes learning rate and momentum only. Weight decay, warmup, schedule
shape, minimum-LR ratio, clipping, and auxiliary AdamW are fixed before any
result is inspected.

## AdamW exploratory grid

Use the Cartesian product below, for exactly 12 exploratory runs:

```text
learning_rate in {3e-4, 6e-4, 1e-3, 2e-3}
beta1 in {0.85, 0.90, 0.95}
```

The remaining AdamW settings are fixed:

```text
optimizer_name = adamw
beta2 = 0.95
weight_decay = 0.1
seed = 1337
```

This remains the protected repository AdamW implementation. AdamW necessarily
owns all trainable parameters rather than using the experimental
matrix/auxiliary split. Model parameters, initialization, data sequence,
regularization coefficient, training budget, schedule shape, evaluation, and
all other controls remain matched.

## Run limits and confirmation

The maximum current baseline budget is:

| Family | Exploratory runs | Additional confirmation runs | Maximum |
|---|---:|---:|---:|
| Global SGDM | 12 at seed 1,337 | 2 at seeds 2,027 and 4,099 | 14 |
| AdamW | 12 at seed 1,337 | 2 at seeds 2,027 and 4,099 | 14 |
| Current approved-manifest ceiling | 24 | 4 | 28 |

The seed-1,337 winning exploratory run is the first of the three confirmation
seeds and is not rerun unless it was invalid for a predeclared reason. Report
all three seed results, their arithmetic mean, and sample standard deviation
(`n - 1` denominator). Report every unsuccessful run; do not silently discard
or replace it.

Reserve a separate future Full Muon budget of 12 exploratory plus 2 additional
confirmation runs under the same token and evaluation protocol. This 14-run
reservation is not authorized by Task 1.5 and cannot be borrowed by SGDM or
AdamW. Muon candidates will be specified before their results are observed.

## Deterministic preflight

Before materializing or launching the Task 1.6 manifest, run the matched
preflight test for AdamW and global SGDM with seed 1,337. For each path:

1. Reset the PyTorch seed to 1,337.
2. Construct the same model through the same model factory.
3. Construct the selected optimizer without taking an update.
4. Hash every named initialized model parameter.
5. Hash the first eight sampled training micro-batches in order.
6. Require identical model and ordered-batch fingerprints.

The reusable implementation is `shared_utils/matched_preflight.py`; focused
coverage is in `tests/test_baseline_preflight.py`. Any mismatch blocks all
tuning runs. The preflight does not compare post-update parameters because the
optimizers are expected to diverge after their first updates.

## Matched comparison controls

The final comparison must confirm all of the following from saved metadata:

- identical model architecture and initialization seed;
- identical prepared dataset and GPT-2 tokenizer;
- identical sampled data order for each shared seed;
- identical sequence length, batch size, accumulation, and DDP world size;
- identical selection and maximum processed-token budgets;
- identical warmup duration, cosine schedule shape, and minimum-LR ratio;
- identical weight-decay coefficient and decoupled semantics;
- identical gradient clipping, precision, compilation, and hardware class;
- identical evaluation batches, frequency, and checkpoints;
- identical W&B behavior and checkpoint policy;
- the common eligible/auxiliary audit recorded for both methods;
- the fixed auxiliary AdamW settings for every global-SGDM run.

Optimizer-specific learning rates and momentum parameters are intentionally
different because they are independently tuned.

## Run names and required artifacts

Use this exact structure:

```text
{optimizer}_gpt2-124m_lr{lr}_mom{momentum}_wd{wd}_seed{seed}_scale{rule}
```

Use `rule=global` for global SGDM and `rule=adamw` for AdamW. For AdamW,
`momentum` means `beta1`. Example:

```text
global_sgdm_gpt2-124m_lr0.1_mom0.95_wd0.1_seed1337_scaleglobal
adamw_gpt2-124m_lr0.0006_mom0.9_wd0.1_seed1337_scaleadamw
```

Preserve each run's complete config, checkpoint, `run_summary.json`, parameter
audit, optimizer state, Git commit, seed, hardware metadata, and train/validation
metrics. Task 1.6 must materialize the full 24-run exploratory manifest before
launching any candidate and may add only the four predefined confirmation runs
after applying the selection rule.

## Interpretation boundary

This protocol establishes tuned baseline measurements only. It does not permit
a causal SGDM-versus-Muon claim. Such a claim requires the separately
implemented and comparably tuned Full Muon baseline.
