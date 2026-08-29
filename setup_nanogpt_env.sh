#!/bin/bash

set -euo pipefail

ENV_NAME="nanogpt"
CONDA_SH="/shared/home/bilal.ashfaq/miniconda3/etc/profile.d/conda.sh"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_FREEZE="$REPO_DIR/cluster_environment.freeze.txt"
JOB_LOG_DIR="/shared/home/bilal.ashfaq/nanogpt-job-logs"
SMOKE_RUN_DIR="$REPO_DIR/nanogpt-smoke-runs"
PILOT_RUN_DIR="$REPO_DIR/nanogpt-pilot-runs"
STUDY_RUN_DIR="$REPO_DIR/nanogpt-study-runs"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: Conda initialization script not found at $CONDA_SH"
    exit 1
fi

source "$CONDA_SH"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "===== CREATING CONDA ENVIRONMENT ====="
    conda create -y -n "$ENV_NAME" python=3.11 pip
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

conda activate "$ENV_NAME"

echo "===== UPGRADING INSTALL TOOLS ====="
python -m pip install --upgrade pip setuptools wheel

echo "===== INSTALLING CUDA PYTORCH ====="
python -m pip install \
    "torch==2.13.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130

if [[ -f "$ENV_FREEZE" ]]; then
    echo "===== RESTORING FROZEN ENVIRONMENT ====="
    python -m pip install -r "$ENV_FREEZE"
else
    echo "===== BOOTSTRAPPING NANOGPT DEPENDENCIES ====="
    python -m pip install \
        "numpy==2.4.6" \
        transformers \
        datasets \
        tiktoken \
        wandb \
        tqdm \
        requests
fi

echo "===== VERIFYING PACKAGES ====="
python -m pip check

python - <<'PY'
import datasets
import numpy
import requests
import tiktoken
import torch
import tqdm
import transformers
import wandb

print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("NumPy:", numpy.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)
print("tiktoken:", tiktoken.__version__)
print("W&B:", wandb.__version__)
print("CUDA visible during setup:", torch.cuda.is_available())
PY

mkdir -p "$JOB_LOG_DIR" "$SMOKE_RUN_DIR" "$PILOT_RUN_DIR" "$STUDY_RUN_DIR"

if [[ ! -f "$ENV_FREEZE" ]]; then
    python -m pip freeze > "$ENV_FREEZE"
    echo "Created environment lock: $ENV_FREEZE"
    echo "Review and commit this file before running scientific experiments."
else
    CURRENT_FREEZE="$(mktemp)"
    trap 'rm -f "$CURRENT_FREEZE"' EXIT
    python -m pip freeze > "$CURRENT_FREEZE"
    diff -u "$ENV_FREEZE" "$CURRENT_FREEZE"
    echo "Installed environment exactly matches $ENV_FREEZE"
fi

echo "===== ENVIRONMENT READY ====="
echo "Activate with: conda activate $ENV_NAME"
echo "Package snapshot: $ENV_FREEZE"
echo "Job logs: $JOB_LOG_DIR"
echo "Smoke outputs: $SMOKE_RUN_DIR"
echo "Pilot outputs: $PILOT_RUN_DIR"
echo "Study outputs: $STUDY_RUN_DIR"
