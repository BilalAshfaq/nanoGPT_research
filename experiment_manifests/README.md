# Optimizer experiment manifests

The generic runner accepts any optimizer name implemented by `train.py`.
Pilot and study artifacts are kept distinct by the manifest `purpose` and
`counts_toward_study_budget` fields.

## Pilot

From the repository root on Martin:

```bash
sbatch \
  --export=ALL,MANIFEST_PATH=experiment_manifests/pilot_global_sgdm.json \
  run_optimizer_manifest.slurm
```

## Task 1.6 preparation

Prepare OpenWebText, freeze the Python environment, and record the dataset
fingerprints from the repository root on Martin. OpenWebText requires roughly
54 GB for the Hugging Face cache plus roughly 17 GB for `train.bin`, so check
the available space first:

```bash
bash setup_nanogpt_env.sh
df -h /shared/home/bilal.ashfaq
mkdir -p /shared/home/bilal.ashfaq/nanogpt-job-logs

PREP_JOB_ID=$(sbatch --parsable \
  --export=ALL,NANOGPT_HF_HOME=/shared/home/bilal.ashfaq/huggingface-cache \
  prepare_openwebtext.slurm)
echo "OpenWebText preparation job: $PREP_JOB_ID"
```

The preparation job downloads and tokenizes OpenWebText, creates
`data/openwebtext/train.bin` and `val.bin`, fingerprints both files, and
validates the exploratory manifest. `NANOGPT_HF_HOME` deliberately overrides
any cluster-wide `HF_HOME` that points to a read-only cache. The job does not
request a GPU. Before downloading, it verifies the pinned
`Skylion007/openwebtext` source has the expected `text` field and 8,013,769
documents. Before fingerprinting, it verifies nanoGPT's expected train and
validation token counts for this pinned modern pipeline: 9,035,582,489 and
4,434,606 respectively. Their 9,040,017,095-token total matches nanoGPT's
historical corpus total. Monitor it with:

```bash
squeue -j "$PREP_JOB_ID"
tail -f "/shared/home/bilal.ashfaq/nanogpt-job-logs/owt_prepare-${PREP_JOB_ID}.out"
```

## Task 1.6 exploratory study

The exploratory manifest must have `"launch_authorized": true` before the
study can run. To queue the study immediately and start it only after dataset
preparation succeeds:

```bash
STUDY_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$PREP_JOB_ID" \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_1_6_exploratory.json \
  run_optimizer_manifest.slurm)

echo "Task 1.6 study job: $STUDY_JOB_ID"
squeue -j "$PREP_JOB_ID","$STUDY_JOB_ID"
```

If preparation has already completed successfully, submit the study without a
dependency:

```bash
STUDY_JOB_ID=$(sbatch --parsable \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_1_6_exploratory.json \
  run_optimizer_manifest.slurm)

echo "Task 1.6 study job: $STUDY_JOB_ID"
```

To execute one approved entry, add its manifest run id:

```bash
sbatch \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_1_6_exploratory.json,RUN_ID=global_sgdm_lr0.10_mom0.95 \
  run_optimizer_manifest.slurm
```

After all 24 exploratory entries have final outcomes, select the winners and
materialize the four still-unauthorized confirmation entries:

```bash
python -m shared_utils.experiment_results select \
  experiment_manifests/task_1_6_exploratory.json \
  --report task_1_6_selection.json \
  --confirmation-manifest experiment_manifests/task_1_6_confirmation.json
```

Generated JSON reports are placed automatically in the repository-root
`reports/` directory, so the command above writes
`reports/task_1_6_selection.json`.

Pilot results are rejected by the study selector. Failed, divergent, and
interrupted outcomes remain final and are not automatically replaced.

## Task 2.5 static per-matrix SGDM study

The checked-in exploratory manifest was materialized from
`reports/task_1_6_selection.json`. It fixes matrix momentum to the seed-1337
Variant 1 winner (`0.99`) and contains the 12 candidates frozen by Task 2.4.
Do not submit it until the smoke outcome below is `completed`.

From a clean `experiments-iter-1` checkout on Martin, validate both manifests:

```bash
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/task_2_5_static_smoke.json
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/task_2_5_static_exploratory.json
```

Run the required non-budget smoke first:

```bash
SMOKE_JOB_ID=$(sbatch --parsable \
  --time=01:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_2_5_static_smoke.json \
  run_optimizer_manifest.slurm)

echo "Task 2.5 smoke job: $SMOKE_JOB_ID"
squeue -j "$SMOKE_JOB_ID"
```

After it exits, require a completed outcome before proceeding:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('nanogpt-smoke-runs/task-2.5').resolve()
outcome_path = path / (
    'static_per_matrix_sgdm_smoke_retry-v2_lr0.03_mom0.99_'
    'seed1337_scalestatic-26f16b708d59/outcome.json'
)
if not outcome_path.is_file():
    raise SystemExit(f'missing retry smoke outcome: {outcome_path}')
outcome = json.loads(outcome_path.read_text())
print(outcome)
if outcome.get('status') != 'completed':
    raise SystemExit('Task 2.5 smoke did not complete successfully')
PY
```

Only after that check succeeds, submit the 12-run exploratory study:

```bash
STUDY_JOB_ID=$(sbatch --parsable \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_2_5_static_exploratory.json \
  run_optimizer_manifest.slurm)

echo "Task 2.5 exploratory job: $STUDY_JOB_ID"
squeue -j "$STUDY_JOB_ID"
```

The manifest runner records and does not replace completed, failed, divergent,
or interrupted outcomes. Keep the cluster checkout unchanged until the
sequential manifest job finishes.

After all 12 outcomes are final, select the seed-1337 static winner and
materialize the still-unauthorized, deferred confirmation manifest:

```bash
python -m shared_utils.experiment_results select \
  experiment_manifests/task_2_5_static_exploratory.json \
  --report task_2_5_static_selection.json \
  --confirmation-manifest experiment_manifests/task_2_5_static_confirmation.json
```

Create the explicitly exploratory one-seed comparison with Variant 1:

```bash
python -m static_per_matrix_sgdm.utils.study_results \
  --global-manifest experiment_manifests/task_1_6_exploratory.json \
  --global-selection reports/task_1_6_selection.json \
  --static-manifest experiment_manifests/task_2_5_static_exploratory.json \
  --static-selection reports/task_2_5_static_selection.json \
  --output task_2_5_seed1337_comparison.json
```

The selection and comparison commands place their generated report files in
`reports/`, including `reports/task_2_5_static_selection.json` and
`reports/task_2_5_seed1337_comparison.json`.

Seeds `2027` and `4099` remain deferred for both selected optimizers. The
one-seed comparison is exploratory and cannot support a final improvement
claim.

## Task 3.5 Frobenius-normalized SGDM study

Task 3.4 froze exactly 12 seed-1337 candidates: Frobenius base LRs
`{0.001, 0.003, 0.01, 0.03}` crossed with momenta
`{0.90, 0.95, 0.99}`. Every run fixes `frobenius_epsilon = 1e-12` and
`frobenius_shape_factor = 1.0`. The selection checkpoint is step 999 at
491,028,480 processed tokens, and exact ties are broken by lower Frobenius LR
and then lower momentum.

The smoke and exploratory manifests are checked in with
`"launch_authorized": false`. Validation is read-only and may be run now, but
the runner will refuse compute until the applicable manifest receives a
separately reviewed authorization change. Smoke authorization does not
authorize the 12-run study, and study authorization does not authorize the two
confirmation runs.

### 1. Synchronize and validate without launching

From a clean `experiments-iter-1` checkout on Martin, record the exact commit
and verify the branch and worktree before using any compute:

```bash
git branch --show-current
git status --short
git rev-parse HEAD
mkdir -p logs

python -m shared_utils.experiment_manifest validate \
  experiment_manifests/task_3_5_frobenius_smoke.json
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/task_3_4_frobenius_exploratory.json
```

The first command must print `experiments-iter-1`, `git status --short` must be
empty, and both manifests must validate. Do not rematerialize or edit the LR
grid after inspecting Variant 3 results.

Run the focused checks in the locked environment before requesting smoke
authorization:

```bash
python -m unittest \
  tests.test_baseline_preflight \
  tests.test_frobenius_normalization \
  tests.test_frobenius_normalized_sgdm \
  tests.test_frobenius_normalized_diagnostics \
  tests.test_frobenius_study_protocol \
  tests.test_frobenius_study_execution -v
```

This includes the deterministic save/load/next-update coverage. A failed test
blocks the smoke and study.

### 2. Run the separately authorized smoke

After explicit smoke-compute approval, review a change that sets only the smoke
manifest's `launch_authorized` field to `true`. Keep the exploratory manifest
unauthorized. Validate the authorized smoke again, then submit it:

```bash
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/task_3_5_frobenius_smoke.json

SMOKE_JOB_ID=$(sbatch --parsable \
  --time=01:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_3_5_frobenius_smoke.json \
  run_optimizer_manifest.slurm)

echo "Task 3.5 smoke job: $SMOKE_JOB_ID"
squeue -j "$SMOKE_JOB_ID"
tail -f "logs/optimizer_manifest-${SMOKE_JOB_ID}.out"
```

After the job exits, run this strict artifact check from the repository root:

```bash
python - <<'PY'
import json
from pathlib import Path

run_name = (
    'frobenius_normalized_sgdm_smoke_lr0.01_mom0.95_wd0.1_'
    'seed1337_scalefrobsqrtr-eps1em12-v1'
)
run_dir = Path('nanogpt-smoke-runs/task-3.5').resolve() / run_name
required = (
    'outcome.json',
    'resolved_run.json',
    'run_summary.json',
    'evaluation_metrics.jsonl',
    'optimizer_diagnostics.jsonl',
    'ckpt.pt',
)
missing = [name for name in required if not (run_dir / name).is_file()]
if missing:
    raise SystemExit(f'missing smoke artifacts: {missing}')

outcome = json.loads((run_dir / 'outcome.json').read_text())
summary = json.loads((run_dir / 'run_summary.json').read_text())
resolved = json.loads((run_dir / 'resolved_run.json').read_text())
evaluations = [
    json.loads(line)
    for line in (run_dir / 'evaluation_metrics.jsonl').read_text().splitlines()
    if line.strip()
]
diagnostics = [
    json.loads(line)
    for line in (run_dir / 'optimizer_diagnostics.jsonl').read_text().splitlines()
    if line.strip()
]

if outcome.get('status') != 'completed' or outcome.get('resumed', False):
    raise SystemExit(f'ineligible smoke outcome: {outcome}')
metrics = summary['metrics']
if metrics.get('numerical_status') != 'ok':
    raise SystemExit(f'smoke numerical event: {metrics}')
if metrics.get('divergence_status') != 'not_observed':
    raise SystemExit(f'smoke divergence: {metrics}')
if [record['step'] for record in evaluations] != [0, 1]:
    raise SystemExit(f'unexpected evaluation checkpoints: {evaluations}')
if [record['step'] for record in diagnostics] != [0, 1]:
    raise SystemExit(f'unexpected diagnostic checkpoints: {diagnostics}')
if any(len(record.get('matrices', [])) != 48 for record in diagnostics):
    raise SystemExit('smoke diagnostics do not contain all 48 eligible matrices')

settings = summary['optimizer']['settings']
expected = {
    'frobenius_learning_rate': 0.01,
    'frobenius_epsilon': 1e-12,
    'frobenius_shape_factor': 1.0,
}
for key, value in expected.items():
    if settings.get(key) != value:
        raise SystemExit(f'wrong optimizer setting {key}: {settings.get(key)}')
if summary['optimizer']['name'] != 'frobenius_normalized_sgdm':
    raise SystemExit('wrong smoke optimizer')
if resolved['run']['seed'] != 1337:
    raise SystemExit('wrong smoke seed')

epsilon_events = sum(
    matrix.get('epsilon_dominated') is True
    for record in diagnostics
    for matrix in record['matrices']
)
zero_events = sum(
    matrix.get('zero_momentum') is True
    for record in diagnostics
    for matrix in record['matrices']
)
print({
    'outcome': outcome,
    'evaluation_steps': [record['step'] for record in evaluations],
    'diagnostic_steps': [record['step'] for record in diagnostics],
    'epsilon_dominated_events': epsilon_events,
    'zero_momentum_events': zero_events,
    'checkpoint_bytes': (run_dir / 'ckpt.pt').stat().st_size,
})
PY
```

Any missing artifact, resumed execution, instability, divergence, wrong
optimizer setting, incomplete matrix diagnostics, or failed preflight blocks
the exploratory study. Do not silently replace the smoke; preserve its outcome
and diagnose it first.

### 3. Complete the final comparator-code audit

Before authorizing the study, compare both recorded comparator commits with
the exact clean commit that will launch Variant 3:

```bash
VARIANT3_COMMIT=$(git rev-parse HEAD)

git diff --name-status \
  2d5d860120c898157fa95311f0269ff0a143a65e.."$VARIANT3_COMMIT" -- \
  train.py model.py config/task_1_6_baseline.py shared_utils/

git diff --name-status \
  aeac82e0427f65633ddcff5d8a73c31167321e78.."$VARIANT3_COMMIT" -- \
  train.py model.py config/task_1_6_baseline.py shared_utils/
```

Review the semantic effect of every listed change. If one affects a matched
training control or measurement, rerun the affected comparator or obtain
explicit approval for a documented limitation. Artifact-location changes,
optimizer-name-specific Frobenius branches, outcome classification, and report
formatting must still be reviewed; they are not exempt merely because they are
expected.

The implementation-time assessment is preserved in
`frobenius_normalized_sgdm/task_3_5_implementation_audit.json`, which records
the committed Task 3.5 implementation as
`2cbca3a1f83de97de8a4e59e7df26cec5717c6e8`. The JSON file does not substitute
for checking the exact clean launch commit, including any later reviewed
reporting or authorization-only commits.

### 4. Authorize and launch exactly 12 exploratory runs

Only after the smoke and audit pass, obtain explicit approval for the full
12-run budget. Review a single-field change setting
`experiment_manifests/task_3_4_frobenius_exploratory.json`'s
`launch_authorized` field to `true`. Do not change any run, order, override,
seed, budget, comparator lock, or selection rule.

Validate once more, then submit the sequential manifest job:

```bash
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/task_3_4_frobenius_exploratory.json

STUDY_JOB_ID=$(sbatch --parsable \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_3_4_frobenius_exploratory.json \
  run_optimizer_manifest.slurm)

echo "Task 3.5 exploratory job: $STUDY_JOB_ID"
squeue -j "$STUDY_JOB_ID"
tail -f "logs/optimizer_manifest-${STUDY_JOB_ID}.out"
```

Keep the checkout and dataset unchanged until the job finishes. The runner
records completed, failed, divergent, interrupted, and numerically unstable
outcomes without replacing them. If a prior `running` outcome resumes from a
checkpoint, its final outcome records `"resumed": true`; such a run is
automatically excluded from winner selection because RNG state is not restored.

After the job exits, summarize all 12 outcomes before selection:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path(
    'experiment_manifests/task_3_4_frobenius_exploratory.json'
).read_text())
root = Path(manifest['output_root']).resolve()
rows = []
for run in manifest['runs']:
    path = root / run['run_name'] / 'outcome.json'
    outcome = json.loads(path.read_text()) if path.is_file() else {'status': 'missing'}
    rows.append({
        'run_id': run['run_id'],
        'status': outcome.get('status'),
        'resumed': outcome.get('resumed', False),
        'return_code': outcome.get('return_code'),
    })
print(json.dumps(rows, indent=2))
final = {'completed', 'failed', 'divergent', 'interrupted', 'numerically_unstable'}
if len(rows) != 12 or any(row['status'] not in final for row in rows):
    raise SystemExit('all 12 candidates must have final recorded outcomes')
PY
```

### 5. Select Variant 3 and generate the one-seed comparison

Select solely by the frozen step-999 rule and generate the still-unauthorized
Frobenius confirmation manifest:

```bash
python -m shared_utils.experiment_results select \
  experiment_manifests/task_3_4_frobenius_exploratory.json \
  --report task_3_5_frobenius_selection.json \
  --confirmation-manifest experiment_manifests/task_3_5_frobenius_confirmation.json
```

The report is written to `reports/task_3_5_frobenius_selection.json`. Selection
fails if any outcome is nonfinal, if no completed uninterrupted candidate
exists, or if the required optimizer family cannot be selected. The generated
confirmation manifest contains only seeds 2027 and 4099, preserves the winning
LR, momentum, epsilon, shape factor, normalization id, comparator locks, audit,
and claim gate, and remains unauthorized.

Create the direct, explicitly exploratory comparison with the immutable
Variant 1 and Variant 2 winners:

```bash
python -m frobenius_normalized_sgdm.utils.study_results exploratory \
  --global-manifest experiment_manifests/task_1_6_exploratory.json \
  --global-selection reports/task_1_6_selection.json \
  --static-manifest experiment_manifests/task_2_5_static_exploratory.json \
  --static-selection reports/task_2_5_static_selection.json \
  --frobenius-manifest experiment_manifests/task_3_4_frobenius_exploratory.json \
  --frobenius-selection reports/task_3_5_frobenius_selection.json \
  --frobenius-smoke-manifest experiment_manifests/task_3_5_frobenius_smoke.json \
  --output task_3_5_seed1337_three_family_comparison.json
```

This command verifies matched configs, token/evaluation controls, immutable
comparator identities, the Static mapping fingerprint, the Frobenius rule, and
runtime environment/dataset fingerprints. The original Variant 1 and 2 winner
directories referenced by their selection reports must still contain
`resolved_run.json`. If those runtime artifacts are unavailable, stop and
record the limitation; do not weaken the check or claim fully matched controls.

The broad study deliberately keeps diagnostics disabled to avoid comparison
overhead. The command strictly validates the required smoke outcome, resolved
configuration, runtime locks, checkpoint, evaluation records, and both 48-matrix
diagnostic checkpoints. It persists the smoke's epsilon-dominated and
zero-momentum event counts and fractions as separately labelled mechanical
evidence. Those values are not broad-study measurements or performance evidence.

### 6. Confirm only after separate approval

Do not launch `experiment_manifests/task_3_5_frobenius_confirmation.json`
until the seed-1337 selection and comparison have been reviewed and the two-run
confirmation budget is explicitly approved. After that approval, review a
single-field authorization change, validate, and submit:

```bash
CONFIRM_JOB_ID=$(sbatch --parsable \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_3_5_frobenius_confirmation.json \
  run_optimizer_manifest.slurm)

echo "Task 3.5 Frobenius confirmation job: $CONFIRM_JOB_ID"
squeue -j "$CONFIRM_JOB_ID"
```

Generate the Frobenius three-seed summary after both confirmation outcomes are
final:

```bash
python -m shared_utils.experiment_results report \
  experiment_manifests/task_3_4_frobenius_exploratory.json \
  experiment_manifests/task_3_5_frobenius_confirmation.json \
  --output task_3_5_frobenius_final.json
```

Only when the selected Global SGDM, Static Per-Matrix SGDM, and
Frobenius-Normalized SGDM configurations each have successful, uninterrupted
seeds 1337, 2027, and 4099 may the final matched report be generated:

```bash
python -m frobenius_normalized_sgdm.utils.study_results confirmed \
  --global-manifest experiment_manifests/task_1_6_exploratory.json \
  --global-selection reports/task_1_6_selection.json \
  --static-manifest experiment_manifests/task_2_5_static_exploratory.json \
  --static-selection reports/task_2_5_static_selection.json \
  --frobenius-manifest experiment_manifests/task_3_4_frobenius_exploratory.json \
  --frobenius-selection reports/task_3_5_frobenius_selection.json \
  --frobenius-smoke-manifest experiment_manifests/task_3_5_frobenius_smoke.json \
  --global-confirmation-manifest experiment_manifests/task_1_6_confirmation.json \
  --static-confirmation-manifest experiment_manifests/task_2_5_static_confirmation.json \
  --frobenius-confirmation-manifest experiment_manifests/task_3_5_frobenius_confirmation.json \
  --output task_3_5_confirmed_three_family_comparison.json
```

The final report includes individual seed records, means, sample standard
deviations, failures or ineligible resumed runs, clipping, numerical status,
timing, memory, and the exact selected configurations. Its H3 claim gate remains
blocked unless all three families have three matched successful seeds.

## Zero eligible-matrix weight-decay follow-up

This is a new, isolated follow-up. It does not modify or replace the Task 1.6,
2.5, or 3.5 manifests, reports, logs, or run directories. The historical
studies retain `matrix_weight_decay = 0.1`; the new inherited config documents
that prior value in a comment and then sets `matrix_weight_decay = 0.0`.

Materialize or verify the unauthorized 36-run manifest with:

```bash
python -m zero_matrix_decay_followup.utils.study_manifest
python -m zero_matrix_decay_followup.utils.study_manifest --check
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/zero_matrix_decay_followup_exploratory.json
```

The manifest contains exactly 12 LR/momentum candidates for each of Global,
Static, and Frobenius SGDM. It writes runs only under
`nanogpt-study-runs/zero-matrix-decay-followup-v1` and remains unauthorized
until the complete grid and compute budget receive separate approval.

When an authorized study is eventually submitted, override the scheduler log
paths so they cannot collide with earlier job logs:

```bash
ZERO_WD_JOB_ID=$(sbatch --parsable \
  --output=logs/zero_matrix_decay_followup_v1-%j.out \
  --error=logs/zero_matrix_decay_followup_v1-%j.err \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/zero_matrix_decay_followup_exploratory.json \
  run_optimizer_manifest.slurm)
```

After all 36 entries have final outcomes, use only the new report and
confirmation names:

```bash
python -m shared_utils.experiment_results select \
  experiment_manifests/zero_matrix_decay_followup_exploratory.json \
  --report zero_matrix_decay_followup_selection.json \
  --confirmation-manifest experiment_manifests/zero_matrix_decay_followup_confirmation.json

python -m shared_utils.experiment_results report \
  experiment_manifests/zero_matrix_decay_followup_exploratory.json \
  experiment_manifests/zero_matrix_decay_followup_confirmation.json \
  --output zero_matrix_decay_followup_final.json
```

The confirmation manifest generated by selection remains unauthorized and
contains seeds `2027` and `4099` for each selected family. The original
weight-decay-`0.1` results remain a separately reported regularized ablation.

## Global SGDM momentum-0.99 learning-rate exploration

This five-run follow-up reuses `config/zero_matrix_decay_followup.py`, including
the GPT-2 124M/OpenWebText controls, two-GPU setup, fixed token budget, zero
eligible-matrix weight decay, and unchanged auxiliary AdamW settings. It fixes
Global SGDM momentum to `0.99` and tests matrix learning rates `0.1`, `0.01`,
`0.001`, `0.0001`, and `0.00001` at seed `1337`.

Validate the manifest without launching compute:

```bash
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/global_sgdm_momentum_099_lr_exploratory.json
```

The manifest is checked in with `launch_authorized: false`. After the five-run
budget is explicitly approved, change only that field to `true`, validate it
again, and submit it through the existing manifest runner:

```bash
GLOBAL_LR_JOB_ID=$(sbatch --parsable \
  --output=logs/global_sgdm_momentum_099_lr_v1-%j.out \
  --error=logs/global_sgdm_momentum_099_lr_v1-%j.err \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/global_sgdm_momentum_099_lr_exploratory.json \
  run_optimizer_manifest.slurm)

echo "Global SGDM LR exploration job: $GLOBAL_LR_JOB_ID"
squeue -j "$GLOBAL_LR_JOB_ID"
tail -f "logs/global_sgdm_momentum_099_lr_v1-${GLOBAL_LR_JOB_ID}.out"
```

To run only one authorized candidate, pass its manifest run id:

```bash
sbatch \
  --output=logs/global_sgdm_momentum_099_lr_v1-%j.out \
  --error=logs/global_sgdm_momentum_099_lr_v1-%j.err \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/global_sgdm_momentum_099_lr_exploratory.json,RUN_ID=global_sgdm_lr0.001_mom0.99_mwd0 \
  run_optimizer_manifest.slurm
```

The five runs write only under
`nanogpt-study-runs/global-sgdm-momentum-099-lr-exploratory-v1`.

After all five entries have final outcomes, select the winner using unrounded
step-999 validation loss and write the selection report into `reports/`:

```bash
python -m shared_utils.experiment_results select \
  experiment_manifests/global_sgdm_momentum_099_lr_exploratory.json \
  --report global_sgdm_momentum_099_lr_selection.json \
  --confirmation-manifest experiment_manifests/global_sgdm_momentum_099_lr_confirmation.json
```

This produces `reports/global_sgdm_momentum_099_lr_selection.json`, containing
all five outcomes and the selected `global_sgdm` winner. It also materializes
an unauthorized confirmation manifest for seeds `2027` and `4099`; generating
that file does not authorize or launch those runs.
