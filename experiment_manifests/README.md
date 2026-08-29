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
