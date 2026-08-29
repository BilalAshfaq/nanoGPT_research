#!/bin/bash
#SBATCH --job-name=sgdm_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --account=nils
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=sgdm_pilot-%j.out
#SBATCH --error=sgdm_pilot-%j.err

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
CONDA_ENV="nanogpt"
CONDA_SH="/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh"
PILOT_ROOT="$REPO_DIR/nanogpt-pilot-runs"
ENV_FREEZE="$REPO_DIR/cluster_environment.freeze.txt"

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

echo "===== GLOBAL SGDM PILOT ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Repository: $REPO_DIR"
echo "Started: $(date --iso-8601=seconds)"

if [[ "$(git branch --show-current)" != "experiments-iter-1" ]]; then
    echo "ERROR: Expected experiments-iter-1 branch."
    exit 1
fi

if [[ ! -f "$ENV_FREEZE" ]]; then
    echo "ERROR: Missing $ENV_FREEZE; run bash setup_nanogpt_env.sh first."
    exit 1
fi

CURRENT_FREEZE="$(mktemp)"
trap 'rm -f "$CURRENT_FREEZE"' EXIT
python -m pip freeze > "$CURRENT_FREEZE"
diff -u "$ENV_FREEZE" "$CURRENT_FREEZE"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA devices:", torch.cuda.device_count())
print("bfloat16 supported:", torch.cuda.is_bf16_supported())

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("the pilot requires exactly two allocated CUDA GPUs")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("the pilot requires bfloat16 support")
PY

if [[ ! -f data/shakespeare_char/train.bin || \
      ! -f data/shakespeare_char/val.bin ]]; then
    python data/shakespeare_char/prepare.py
fi

PILOT_OUT_DIR="$PILOT_ROOT/global-sgdm-$SLURM_JOB_ID"
mkdir -p "$PILOT_OUT_DIR"
cp "$ENV_FREEZE" "$PILOT_OUT_DIR/cluster_environment.freeze.txt"

echo "Output directory: $PILOT_OUT_DIR"

torchrun \
    --standalone \
    --nproc_per_node=2 \
    train.py \
    config/pilot_global_sgdm.py \
    --out_dir="$PILOT_OUT_DIR"

echo "===== PILOT ARTIFACTS ====="
find "$PILOT_OUT_DIR" -maxdepth 2 -type f -print

echo "===== PILOT COMPLETED ====="
echo "These results verify progress only and are not Task 1.6 selection data."
echo "Finished: $(date --iso-8601=seconds)"
