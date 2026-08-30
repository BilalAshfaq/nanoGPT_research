"""Matched-control checks shared by optimizer study comparisons."""

import json
import os

from shared_utils.experiment_manifest import resolve_run_config


COMMON_MATCHED_CONFIG_KEYS = (
    "dataset",
    "tokenizer",
    "gradient_accumulation_steps",
    "batch_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
    "weight_decay",
    "matrix_weight_decay",
    "matrix_momentum_convention",
    "matrix_weight_decay_mode",
    "matrix_nesterov",
    "auxiliary_learning_rate",
    "auxiliary_weight_decay",
    "auxiliary_beta1",
    "auxiliary_beta2",
    "auxiliary_weight_decay_mode",
    "max_iters",
    "decay_lr",
    "warmup_iters",
    "lr_decay_iters",
    "learning_rate",
    "min_lr",
    "grad_clip",
    "eval_interval",
    "eval_iters",
    "dtype",
    "compile",
)
MATCHED_RUN_KEYS = (
    "seed",
    "tokens_per_update",
    "selection_tokens",
    "max_processed_tokens",
    "evaluation_steps",
    "selection_step",
)


def read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def completed_winner(selection_report, optimizer_name, seed=1337):
    winner = selection_report.get("winners", {}).get(optimizer_name)
    if not isinstance(winner, dict) or winner.get("status") != "completed":
        raise ValueError(f"missing completed {optimizer_name} winner")
    if winner.get("resumed", False):
        raise ValueError(f"resumed {optimizer_name} winner is comparison-ineligible")
    if winner.get("seed") != seed:
        raise ValueError(f"comparison requires seed {seed}")
    record = winner.get("selection_record")
    if not isinstance(record, dict) or record.get("step") != 999:
        raise ValueError("winner lacks the fixed step-999 selection record")
    if record.get("processed_tokens") != 491_028_480:
        raise ValueError("winner selection token count does not match")
    return winner


def manifest_run(manifest, winner):
    matches = [
        run for run in manifest["runs"] if run["run_id"] == winner["run_id"]
    ]
    if len(matches) != 1:
        raise ValueError("winner does not identify exactly one manifest run")
    return matches[0]


def runtime_locks(result):
    path = os.path.join(result["output_directory"], "resolved_run.json")
    if not os.path.isfile(path):
        raise ValueError(f"run lacks resolved runtime metadata: {path}")
    resolved = read_json(path)
    try:
        return {
            "environment_sha256": resolved["environment"]["sha256"],
            "allocated_gpu_names": resolved["environment"][
                "allocated_gpu_names"
            ],
            "dataset_files": resolved["dataset"]["files"],
        }
    except KeyError as exc:
        raise ValueError("run runtime metadata lacks lock fingerprints") from exc


def verify_matched_controls(reference, candidates, config_keys):
    """Verify configs, manifest locks, run controls, and runtime fingerprints."""

    reference_name, reference_manifest, reference_run, reference_result = reference
    reference_config = resolve_run_config(reference_manifest, reference_run)
    reference_runtime = runtime_locks(reference_result)
    for name, manifest, run, result in candidates:
        candidate_config = resolve_run_config(manifest, run)
        mismatches = {
            key: {
                reference_name: reference_config[key],
                name: candidate_config[key],
            }
            for key in config_keys
            if reference_config[key] != candidate_config[key]
        }
        if mismatches:
            raise ValueError(
                f"matched study controls differ for {name}: "
                + ", ".join(sorted(mismatches))
            )
        for key in ("environment_lock", "dataset", "resources"):
            if reference_manifest[key] != manifest[key]:
                raise ValueError(f"manifest {key} does not match {reference_name}")
        for key in MATCHED_RUN_KEYS:
            if reference_run[key] != run[key]:
                raise ValueError(f"run control {key} does not match {reference_name}")
        if runtime_locks(result) != reference_runtime:
            raise ValueError(
                f"runtime environment or dataset fingerprints differ for {name}"
            )
    return reference_runtime
