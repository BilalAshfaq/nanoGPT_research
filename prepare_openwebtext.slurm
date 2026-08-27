#!/bin/bash
#SBATCH --job-name=owt_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=nils
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/shared/home/bilal.ashfaq/nanogpt-job-logs/owt_prepare-%j.out
#SBATCH --error=/shared/home/bilal.ashfaq/nanogpt-job-logs/owt_prepare-%j.err

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
CONDA_ENV="nanogpt"
CONDA_SH="/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh"
MANIFEST_PATH="${MANIFEST_PATH:-experiment_manifests/task_1_6_exploratory.json}"
HF_HOME="${NANOGPT_HF_HOME:-/shared/home/bilal.ashfaq/huggingface-cache}"
TRAIN_DATA="$REPO_DIR/data/openwebtext/train.bin"
VAL_DATA="$REPO_DIR/data/openwebtext/val.bin"

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export HF_HOME PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME"

echo "===== OPENWEBTEXT PREPARATION ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO_DIR"
echo "Hugging Face cache: $HF_HOME"
echo "Manifest: $MANIFEST_PATH"
echo "Started: $(date --iso-8601=seconds)"
df -h /shared/home/bilal.ashfaq

if [[ "$(git branch --show-current)" != "experiments-iter-1" ]]; then
    echo "ERROR: Expected experiments-iter-1 branch."
    exit 1
fi

if [[ -f "$TRAIN_DATA" && -f "$VAL_DATA" ]]; then
    echo "OpenWebText binary files already exist; skipping preparation."
elif [[ -e "$TRAIN_DATA" || -e "$VAL_DATA" ]]; then
    echo "ERROR: Only one OpenWebText binary file exists."
    echo "Resolve the partial dataset before retrying; no files were overwritten."
    exit 1
else
    python -u data/openwebtext/prepare.py
fi

python -m shared_utils.experiment_manifest fingerprint-data "$MANIFEST_PATH"
python -m shared_utils.experiment_manifest validate "$MANIFEST_PATH"

echo "===== OPENWEBTEXT PREPARATION FINISHED ====="
ls -lh "$TRAIN_DATA" "$VAL_DATA" data/openwebtext/dataset_fingerprints.json
echo "Finished: $(date --iso-8601=seconds)"
