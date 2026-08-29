#!/bin/bash
#SBATCH --job-name=optimizer_manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --account=nils
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/optimizer_manifest-%j.out
#SBATCH --error=logs/optimizer_manifest-%j.err

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
CONDA_ENV="nanogpt"
CONDA_SH="/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh"
MANIFEST_PATH="${MANIFEST_PATH:-experiment_manifests/pilot_global_sgdm.json}"
RUN_ID="${RUN_ID:-}"

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

echo "===== OPTIMIZER MANIFEST RUNNER ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO_DIR"
echo "Manifest: $MANIFEST_PATH"
echo "Run ID: ${RUN_ID:-all}"
echo "Started: $(date --iso-8601=seconds)"

if [[ "$(git branch --show-current)" != "experiments-iter-1" ]]; then
    echo "ERROR: Expected experiments-iter-1 branch."
    exit 1
fi

python -m shared_utils.experiment_manifest validate "$MANIFEST_PATH"

COMMAND=(python -m shared_utils.experiment_manifest run "$MANIFEST_PATH")
if [[ -n "$RUN_ID" ]]; then
    COMMAND+=(--run-id "$RUN_ID")
fi
"${COMMAND[@]}"

echo "===== MANIFEST RUNNER FINISHED ====="
echo "Finished: $(date --iso-8601=seconds)"
