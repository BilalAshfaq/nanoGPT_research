"""Build the explicitly exploratory seed-1337 Variant 1/Variant 2 comparison."""

import argparse
import json
import os

from shared_utils.experiment_manifest import load_manifest, resolve_run_config
from shared_utils.experiment_results import report_output_path


MATCHED_CONFIG_KEYS = (
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
    "matrix_momentum",
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
    "min_lr",
    "grad_clip",
    "eval_interval",
    "eval_iters",
    "dtype",
    "compile",
)


def _read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _winner(selection_report, optimizer_name):
    winner = selection_report.get("winners", {}).get(optimizer_name)
    if not isinstance(winner, dict) or winner.get("status") != "completed":
        raise ValueError(f"missing completed {optimizer_name} winner")
    if winner.get("seed") != 1337:
        raise ValueError("exploratory comparison requires seed 1337")
    record = winner.get("selection_record")
    if not isinstance(record, dict) or record.get("step") != 999:
        raise ValueError("winner lacks the fixed step-999 selection record")
    if record.get("processed_tokens") != 491_028_480:
        raise ValueError("winner selection token count does not match")
    return winner


def _manifest_run(manifest, winner):
    matches = [
        run for run in manifest["runs"] if run["run_id"] == winner["run_id"]
    ]
    if len(matches) != 1:
        raise ValueError("winner does not identify exactly one manifest run")
    return matches[0]


def _runtime_locks(winner):
    path = os.path.join(winner["output_directory"], "resolved_run.json")
    if not os.path.isfile(path):
        raise ValueError(f"winner lacks resolved runtime metadata: {path}")
    resolved = _read_json(path)
    try:
        return {
            "environment_sha256": resolved["environment"]["sha256"],
            "dataset_files": resolved["dataset"]["files"],
        }
    except KeyError as exc:
        raise ValueError("winner runtime metadata lacks lock fingerprints") from exc


def exploratory_comparison(
    global_manifest,
    global_selection,
    static_manifest,
    static_selection,
):
    global_winner = _winner(global_selection, "global_sgdm")
    static_winner = _winner(static_selection, "static_per_matrix_sgdm")
    global_run = _manifest_run(global_manifest, global_winner)
    static_run = _manifest_run(static_manifest, static_winner)

    global_config = resolve_run_config(global_manifest, global_run)
    static_config = resolve_run_config(static_manifest, static_run)
    mismatches = {
        key: {"global_sgdm": global_config[key], "static": static_config[key]}
        for key in MATCHED_CONFIG_KEYS
        if global_config[key] != static_config[key]
    }
    if mismatches:
        raise ValueError(
            "matched study controls differ: " + ", ".join(sorted(mismatches))
        )
    for key in ("environment_lock", "dataset", "resources"):
        if global_manifest[key] != static_manifest[key]:
            raise ValueError(f"manifest {key} does not match Variant 1")
    for key in (
        "seed",
        "tokens_per_update",
        "selection_tokens",
        "max_processed_tokens",
        "evaluation_steps",
        "selection_step",
    ):
        if global_run[key] != static_run[key]:
            raise ValueError(f"run control {key} does not match Variant 1")
    global_runtime_locks = _runtime_locks(global_winner)
    static_runtime_locks = _runtime_locks(static_winner)
    if global_runtime_locks != static_runtime_locks:
        raise ValueError("runtime environment or dataset fingerprints differ")

    global_loss = global_winner["selection_record"]["validation_loss"]
    static_loss = static_winner["selection_record"]["validation_loss"]
    return {
        "report_type": "exploratory_seed_1337_only",
        "claim_status": (
            "confirmation seeds 2027 and 4099 are deferred; no final "
            "improvement claim is permitted"
        ),
        "observed_measurements": {
            "global_sgdm": {
                "run_id": global_winner["run_id"],
                "validation_loss": global_loss,
                "learning_rate": global_winner["selection_values"][
                    "learning_rate"
                ],
                "momentum": global_winner["selection_values"]["momentum"],
            },
            "static_per_matrix_sgdm": {
                "run_id": static_winner["run_id"],
                "validation_loss": static_loss,
                "learning_rate": static_winner["selection_values"][
                    "learning_rate"
                ],
                "momentum": static_winner["selection_values"]["momentum"],
                "static_multiplier_configuration": static_run[
                    "static_multiplier_configuration"
                ],
            },
            "validation_loss_difference_static_minus_global": (
                static_loss - global_loss
            ),
        },
        "matched_controls": {
            "status": "verified",
            "seed": global_run["seed"],
            "selection_step": global_run["selection_step"],
            "selection_tokens": global_run["selection_tokens"],
            "environment_lock": global_manifest["environment_lock"],
            "dataset_lock": global_manifest["dataset"]["lock_file"],
            "runtime_fingerprints": global_runtime_locks,
            "config_keys": list(MATCHED_CONFIG_KEYS),
        },
        "interpretation": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-manifest", required=True)
    parser.add_argument("--global-selection", required=True)
    parser.add_argument("--static-manifest", required=True)
    parser.add_argument("--static-selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = exploratory_comparison(
        load_manifest(args.global_manifest),
        _read_json(args.global_selection),
        load_manifest(args.static_manifest),
        _read_json(args.static_selection),
    )
    output_path = report_output_path(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
