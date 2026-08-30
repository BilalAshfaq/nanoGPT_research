"""Build the explicitly exploratory seed-1337 Variant 1/Variant 2 comparison."""

import argparse
import json
import os

from shared_utils.experiment_manifest import load_manifest
from shared_utils.experiment_results import report_output_path
from shared_utils.study_comparison import (
    COMMON_MATCHED_CONFIG_KEYS,
    completed_winner,
    manifest_run,
    read_json,
    verify_matched_controls,
)


MATCHED_CONFIG_KEYS = COMMON_MATCHED_CONFIG_KEYS + ("matrix_momentum",)


def exploratory_comparison(
    global_manifest,
    global_selection,
    static_manifest,
    static_selection,
):
    global_winner = completed_winner(global_selection, "global_sgdm")
    static_winner = completed_winner(static_selection, "static_per_matrix_sgdm")
    global_run = manifest_run(global_manifest, global_winner)
    static_run = manifest_run(static_manifest, static_winner)
    global_runtime_locks = verify_matched_controls(
        ("global_sgdm", global_manifest, global_run, global_winner),
        [("static_per_matrix_sgdm", static_manifest, static_run, static_winner)],
        MATCHED_CONFIG_KEYS,
    )

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
        read_json(args.global_selection),
        load_manifest(args.static_manifest),
        read_json(args.static_selection),
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
