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
fingerprints before requesting study authorization:

```bash
bash setup_nanogpt_env.sh
python -m shared_utils.experiment_manifest fingerprint-data \
  experiment_manifests/task_1_6_exploratory.json
python -m shared_utils.experiment_manifest validate \
  experiment_manifests/task_1_6_exploratory.json
```

The checked-in exploratory manifest intentionally has
`"launch_authorized": false`. It must remain false until the user explicitly
authorizes the Task 1.6 compute budget. Once authorized, the same generic
runner can execute every pending entry sequentially:

```bash
sbatch \
  --time=3-00:00:00 \
  --export=ALL,MANIFEST_PATH=experiment_manifests/task_1_6_exploratory.json \
  run_optimizer_manifest.slurm
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
