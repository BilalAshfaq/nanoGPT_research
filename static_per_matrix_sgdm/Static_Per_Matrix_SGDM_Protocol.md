# Static per-matrix SGDM tuning protocol

This protocol defines the Task 2.4 comparison before any static-study result is
inspected. It does not authorize or launch a smoke, exploratory, or confirmation
run. The exact candidate design is locked in
`static_per_matrix_sgdm/task_2_4_candidate_design.json`.

## Comparison and unresolved Variant 1 dependency

The primary comparison is the best tuned global SGDM configuration selected by
the Task 1.6 fixed-token rule versus the best tuned static per-matrix SGDM
configuration selected below. Static momentum is not tuned: every candidate
must use the winning global-SGDM `matrix_momentum` from the reviewed Task 1.6
selection report. Until that report exists, no runnable Task 2.4 manifest may
be materialized or authorized.

After Variant 1 selection, materialize the exploratory manifest with:

```bash
python -m static_per_matrix_sgdm.utils.study_manifest \
  --variant-1-selection task_1_6_selection.json \
  --output experiment_manifests/task_2_5_static_exploratory.json
```

The materializer accepts only a completed global-SGDM winner whose momentum is
one of the predeclared Variant 1 candidates (`0.90`, `0.95`, or `0.99`). It
copies no result metric into the static design. The generated manifest remains
`launch_authorized: false` pending review, a successful separately approved
smoke run, and explicit launch approval.

## Allowed parameterization and exploratory budget

Only matrix-type multipliers are allowed in this first study. A multiplier is
shared across all 12 Transformer layers for each of `attention_qkv`,
`attention_output`, `mlp_input`, and `mlp_output`. Exact-parameter overrides are
empty. Every resolved 48-matrix mapping has geometric mean one, separating the
shared base LR from static heterogeneity.

The three locked profiles are:

| Order | Profile | Attention QKV/output | MLP input/output |
| ---: | --- | ---: | ---: |
| 0 | `attention_mild` | sqrt(2) | 1/sqrt(2) |
| 1 | `attention_strong` | 2 | 1/2 |
| 2 | `mlp_strong` | 1/2 | 2 |

The shared matrix base LR is tuned over `{0.03, 0.10, 0.30, 1.00}`. Its
Cartesian product with the three profiles gives exactly 12 exploratory runs at
seed `1337`, equal to Variant 1's 12 global-SGDM exploratory runs. No failed or
divergent candidate is replaced. Any additional multiplier, LR, or momentum
sweep requires a separately approved protocol.

The primary selection metric is the unrounded validation loss at step `999`
(`491,028,480` processed tokens). Lower is better. Exact metric ties are broken
by lower base LR and then the locked profile order shown above. The report must
show every candidate so sensitivity to both base LR and multiplier profile is
visible, not only the winner.

## Matched controls

The static study reuses the Task 1.6 GPT-2 124M/OpenWebText configuration:

- scratch initialization and the same seed/data order for each shared seed;
- 12 layers, 12 heads, width 768, context length 1024, dropout 0, no bias;
- batch size 12 with 40 gradient-accumulation micro-steps across two GPUs,
  giving 491,520 tokens per optimizer update;
- 1,000 attempted updates (`max_iters = 999`) and evaluations at steps
  `0`, `333`, `666`, and `999`, each using 200 evaluation batches;
- 100-step warmup and cosine decay through step 999 to 0.1 of each optimizer
  group's base rate;
- global gradient clipping at 1.0, bfloat16, TF32 enabled, and compilation;
- decoupled matrix weight decay 0.1 and unchanged auxiliary AdamW with LR
  `6e-4`, betas `(0.9, 0.95)`, and weight decay 0.1;
- the same environment, dataset fingerprints, hardware requirement, logging,
  checkpointing, and divergence accounting as Task 1.6.

Any mismatch blocks the static study.

## Confirmation and claims

After all 12 exploratory outcomes are final, select exactly one winner using
the frozen rule. Reuse its seed-`1337` result and run only seeds `2027` and
`4099` as confirmation. Exploratory and confirmation manifests remain separate.
Report all three individual unrounded losses, their mean and sample standard
deviation, failures or instability, base LR, fixed momentum, complete resolved
mapping, and SHA-256 mapping fingerprint.

No claim that static scaling improves global SGDM is permitted without three
successful seeds for both selected configurations and verification that their
matched controls and fingerprints agree.
