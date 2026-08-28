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
OPENWEBTEXT_DATASET_ID="Skylion007/openwebtext"
OPENWEBTEXT_REVISION="79d93d786212f7344586290adb811d4ae6a1762c"
EXPECTED_DOCUMENTS=8013769
EXPECTED_TRAIN_TOKENS=9035582489
EXPECTED_VAL_TOKENS=4434606

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
    echo "===== VERIFYING OPENWEBTEXT SOURCE ====="
    python - \
        "$OPENWEBTEXT_DATASET_ID" \
        "$OPENWEBTEXT_REVISION" \
        "$EXPECTED_DOCUMENTS" <<'PY'
import sys

from datasets import load_dataset_builder

dataset_id, revision, expected_documents = sys.argv[1:]
builder = load_dataset_builder(dataset_id, revision=revision)
features = set(builder.info.features or {})
train_split = (builder.info.splits or {}).get("train")

if features != {"text"}:
    raise RuntimeError(f"unexpected OpenWebText features: {sorted(features)}")
if train_split is None or train_split.num_examples != int(expected_documents):
    actual = None if train_split is None else train_split.num_examples
    raise RuntimeError(
        f"unexpected OpenWebText document count: expected {expected_documents}, got {actual}"
    )

print(f"Dataset source: {dataset_id}@{revision}")
print(f"Dataset features: {sorted(features)}")
print(f"Training documents: {train_split.num_examples}")
PY
    python -u data/openwebtext/prepare.py
fi

echo "===== VERIFYING OPENWEBTEXT TOKEN COUNTS ====="
python - \
    "$TRAIN_DATA" \
    "$VAL_DATA" \
    "$EXPECTED_TRAIN_TOKENS" \
    "$EXPECTED_VAL_TOKENS" <<'PY'
import sys
from pathlib import Path

train_path, val_path = map(Path, sys.argv[1:3])
expected_train_tokens, expected_val_tokens = map(int, sys.argv[3:5])

for path, expected_tokens in (
    (train_path, expected_train_tokens),
    (val_path, expected_val_tokens),
):
    byte_count = path.stat().st_size
    if byte_count % 2:
        raise RuntimeError(f"{path} has an invalid odd byte count: {byte_count}")
    token_count = byte_count // 2
    if token_count != expected_tokens:
        raise RuntimeError(
            f"unexpected token count for {path}: expected {expected_tokens}, got {token_count}"
        )
    print(f"{path}: {token_count} uint16 tokens")
PY

python -m shared_utils.experiment_manifest fingerprint-data "$MANIFEST_PATH"
python -m shared_utils.experiment_manifest validate "$MANIFEST_PATH"

echo "===== OPENWEBTEXT PREPARATION FINISHED ====="
ls -lh "$TRAIN_DATA" "$VAL_DATA" data/openwebtext/dataset_fingerprints.json
echo "Finished: $(date --iso-8601=seconds)"
