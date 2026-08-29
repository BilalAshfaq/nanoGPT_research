#!/bin/bash
#SBATCH --job-name=sgdm_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --account=nils
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=sgdm_verify-%j.out
#SBATCH --error=sgdm_verify-%j.err

set -euo pipefail

REPO_DIR="$SLURM_SUBMIT_DIR"
CONDA_ENV="nanogpt"
CONDA_SH="/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh"
SMOKE_ROOT="$REPO_DIR/nanogpt-smoke-runs"
ENV_FREEZE="$REPO_DIR/cluster_environment.freeze.txt"

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

if [[ ! -f "$ENV_FREEZE" ]]; then
    echo "ERROR: Missing $ENV_FREEZE; run bash setup_nanogpt_env.sh first."
    exit 1
fi

CURRENT_FREEZE="$(mktemp)"
trap 'rm -f "$CURRENT_FREEZE"' EXIT
python -m pip freeze > "$CURRENT_FREEZE"
diff -u "$ENV_FREEZE" "$CURRENT_FREEZE"

echo "===== JOB INFORMATION ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"

echo "===== REPOSITORY CHECK ====="
CURRENT_BRANCH="$(git branch --show-current)"
echo "Branch: $CURRENT_BRANCH"
git status --short

if [[ "$CURRENT_BRANCH" != "experiments-iter-1" ]]; then
    echo "ERROR: Expected experiments-iter-1 branch."
    exit 1
fi

echo "===== GPU INFORMATION ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
nvidia-smi -L

echo "===== PYTORCH CHECK ====="
python - <<'PY'
import numpy
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("NumPy:", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("bfloat16 supported:", torch.cuda.is_bf16_supported())

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"Expected exactly 2 allocated GPUs, found {torch.cuda.device_count()}"
    )
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("Allocated GPUs do not support bfloat16")
PY

echo "===== UNIT TESTS ====="
python -m unittest discover -s tests -v

echo "===== PREPARING TINY SMOKE DATASET ====="
if [[ ! -f data/shakespeare_char/train.bin || \
      ! -f data/shakespeare_char/val.bin ]]; then
    python data/shakespeare_char/prepare.py
else
    echo "Prepared Shakespeare dataset already exists."
fi

SMOKE_OUT_DIR="$SMOKE_ROOT/global-sgdm-$SLURM_JOB_ID"
mkdir -p "$SMOKE_OUT_DIR"
cp "$ENV_FREEZE" "$SMOKE_OUT_DIR/cluster_environment.freeze.txt"

echo "===== TWO-GPU GLOBAL SGDM SMOKE TEST ====="
echo "Output directory: $SMOKE_OUT_DIR"

torchrun \
    --standalone \
    --nproc_per_node=2 \
    train.py \
    config/smoke_global_sgdm_diagnostics.py \
    --out_dir="$SMOKE_OUT_DIR" \
    --gradient_accumulation_steps=2 \
    --dtype=bfloat16

echo "===== GENERATED ARTIFACTS ====="
find "$SMOKE_OUT_DIR" -maxdepth 2 -type f -print

echo "===== VERIFICATION PASSED ====="
echo "Finished: $(date --iso-8601=seconds)"
