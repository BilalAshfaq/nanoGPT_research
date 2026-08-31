# Zero Eligible-Matrix Weight-Decay Follow-up

This follow-up preserves the original weight-decay-`0.1` studies as immutable
historical evidence. It creates new manifests, output directories, reports,
confirmation runs, and scheduler log names. It does not reinterpret or replace
the existing Task 1.6, 2.5, or 3.5 results.

## Controlled change

The only common training-control change is:

```text
matrix_weight_decay: 0.1 -> 0.0
```

Auxiliary AdamW remains unchanged at learning rate `6e-4`, betas
`(0.9, 0.95)`, and weight decay `0.1`. Model, data, batching, schedule shape,
clipping, precision, evaluation steps, token budget, hardware lock, and dataset
fingerprints remain inherited from Task 1.6.

The Static family fixes the previously selected `mlp_strong` mapping so that
all three families spend exactly 12 candidates retuning LR and momentum. The
raw Global and Static LR grids add `0.01` below the former boundary winner and
drop the extreme `1.0` point. Frobenius retains its bracketed LR grid and adds
momentum `0.8` below its former boundary winner.

## Artifact isolation

- Exploratory manifest:
  `experiment_manifests/zero_matrix_decay_followup_exploratory.json`
- Confirmation manifest generated after selection:
  `experiment_manifests/zero_matrix_decay_followup_confirmation.json`
- Run root: `nanogpt-study-runs/zero-matrix-decay-followup-v1`
- Selection report: `reports/zero_matrix_decay_followup_selection.json`
- Final report: `reports/zero_matrix_decay_followup_final.json`
- Scheduler log prefix: `logs/zero_matrix_decay_followup_v1-<job-id>`

The candidate design records canonical JSON SHA-256 hashes for all original
manifests and selection reports. Canonical hashing is independent of checkout
line endings, and materialization fails if protected JSON content changed.

## Authorization and claims

The exploratory manifest is materialized with `launch_authorized = false`.
Preparing or validating it launches no compute. After separate authorization,
all 36 candidates must receive final recorded outcomes before selection.

Seed `1337` is exploratory. The selected configuration from each family must
also complete uninterrupted at seeds `2027` and `4099` before a three-family
claim is permitted. Existing weight-decay-`0.1` results remain a separate
regularized ablation and must not be pooled with this follow-up.
