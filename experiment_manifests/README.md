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
request a GPU. Monitor it with:

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

Pilot results are rejected by the study selector. Failed, divergent, and
interrupted outcomes remain final and are not automatically replaced.
