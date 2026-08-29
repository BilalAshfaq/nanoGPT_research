"""Materialize the frozen static-study grid after Variant 1 selection."""

import argparse
import copy
import json
import os
from types import SimpleNamespace

from shared_utils.experiment_manifest import validate_manifest
from static_per_matrix_sgdm.utils.static_multipliers import (
    resolve_static_multipliers,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DESIGN_PATH = os.path.join(
    REPOSITORY_ROOT,
    "static_per_matrix_sgdm",
    "task_2_4_candidate_design.json",
)
BASELINE_MANIFEST_PATH = os.path.join(
    REPOSITORY_ROOT,
    "experiment_manifests",
    "task_1_6_exploratory.json",
)
ELIGIBLE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _eligible_gpt2_124m_parameters():
    return tuple(
        SimpleNamespace(
            name=f"transformer.h.{layer_index}.{suffix}",
            parameter=object(),
        )
        for layer_index in range(12)
        for suffix in ELIGIBLE_SUFFIXES
    )


def _validated_profiles(design):
    profiles = []
    for profile in design["mapping_profiles"]:
        configuration = resolve_static_multipliers(
            _eligible_gpt2_124m_parameters(),
            default_multiplier=profile["default_multiplier"],
            matrix_type_multipliers=profile["matrix_type_multipliers"],
            exact_parameter_multipliers=profile["exact_parameter_multipliers"],
        )
        if configuration["fingerprint_sha256"] != profile["fingerprint_sha256"]:
            raise ValueError(
                f"mapping fingerprint changed for profile {profile['name']}"
            )
        if configuration["mapping_id"] != profile["mapping_id"]:
            raise ValueError(f"mapping id changed for profile {profile['name']}")
        profiles.append((profile, configuration))
    if [item[0]["order"] for item in profiles] != list(range(len(profiles))):
        raise ValueError("mapping profile order must be contiguous from zero")
    return profiles


def _variant_1_momentum(selection_report, design):
    if selection_report.get("manifest_id") != design["variant_1_source_manifest_id"]:
        raise ValueError("selection report is not from the locked Variant 1 manifest")
    winner = selection_report.get("winners", {}).get(
        design["variant_1_winner_group"]
    )
    if not isinstance(winner, dict):
        raise ValueError("selection report has no global-SGDM winner")
    if winner.get("status") != "completed":
        raise ValueError("Variant 1 global-SGDM winner must be completed")
    momentum = winner.get("selection_values", {}).get("momentum")
    if momentum not in design["allowed_variant_1_momenta"]:
        raise ValueError("Variant 1 winner has an undeclared momentum")
    return momentum


def materialize_manifest(
    selection_report,
    design=None,
    baseline_manifest=None,
    *,
    launch_authorized=False,
):
    """Return the exact 12-run manifest using the real Variant 1 winner."""

    design = copy.deepcopy(design if design is not None else _read_json(DESIGN_PATH))
    baseline = copy.deepcopy(
        baseline_manifest
        if baseline_manifest is not None
        else _read_json(BASELINE_MANIFEST_PATH)
    )
    if baseline.get("manifest_id") != design["variant_1_source_manifest_id"]:
        raise ValueError("locked Variant 1 source manifest is unavailable")
    momentum = _variant_1_momentum(selection_report, design)
    profiles = _validated_profiles(design)

    runs = []
    for learning_rate in design["base_learning_rates"]:
        for profile, configuration in profiles:
            mapping_id = configuration["mapping_id"]
            learning_rate_label = format(learning_rate, "g")
            momentum_label = format(momentum, "g")
            run_id = (
                f"static_lr{learning_rate_label}_mom{momentum_label}_"
                f"map{mapping_id}"
            )
            run_name = (
                "static_per_matrix_sgdm_gpt2-124m_"
                f"lr{learning_rate_label}_mom{momentum_label}_wd0.1_"
                f"seed{design['exploratory_seed']}_scalestatic-{mapping_id}"
            )
            runs.append(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "optimizer_name": "static_per_matrix_sgdm",
                    "seed": design["exploratory_seed"],
                    "max_iters": 999,
                    "tokens_per_update": 491520,
                    "selection_tokens": design["selection"]["selection_tokens"],
                    "max_processed_tokens": 491520000,
                    "evaluation_steps": [0, 333, 666, 999],
                    "selection_step": design["selection"]["selection_step"],
                    "selection_values": {
                        "learning_rate": learning_rate,
                        "mapping_order": profile["order"],
                        "mapping_name": profile["name"],
                        "mapping_id": mapping_id,
                        "mapping_fingerprint_sha256": configuration[
                            "fingerprint_sha256"
                        ],
                        "momentum": momentum,
                        "weight_decay": 0.1,
                        "scaling_rule": f"static-{mapping_id}",
                    },
                    "static_multiplier_configuration": configuration,
                    "overrides": {
                        "matrix_learning_rate": learning_rate,
                        "matrix_momentum": momentum,
                        "static_default_multiplier": profile[
                            "default_multiplier"
                        ],
                        "static_matrix_type_multipliers": copy.deepcopy(
                            profile["matrix_type_multipliers"]
                        ),
                        "static_exact_parameter_multipliers": copy.deepcopy(
                            profile["exact_parameter_multipliers"]
                        ),
                    },
                }
            )

    if len(runs) != design["expected_run_count"]:
        raise ValueError("frozen candidate design does not produce 12 runs")
    manifest = {
        "schema_version": baseline["schema_version"],
        "manifest_id": "task-2.5-static-exploratory-v1",
        "purpose": "study",
        "launch_authorized": launch_authorized,
        "counts_toward_study_budget": True,
        "config_file": "config/task_2_4_static.py",
        "environment_lock": baseline["environment_lock"],
        "preflight_tests": ["tests.test_baseline_preflight"],
        "dataset": copy.deepcopy(baseline["dataset"]),
        "resources": copy.deepcopy(baseline["resources"]),
        "output_root": "/shared/home/bilal.ashfaq/nanogpt-study-runs/task-2.5",
        "expected_run_count": len(runs),
        "selection": {
            "group_by": "optimizer_name",
            "expected_groups": ["static_per_matrix_sgdm"],
            "metric": design["selection"]["metric"],
            "mode": design["selection"]["mode"],
            "tie_break": design["selection"]["tie_break"],
        },
        "confirmation": {
            "manifest_id": "task-2.5-static-confirmation-v1",
            "additional_seeds": design["confirmation_seeds"],
            "run_name_template": (
                "{optimizer_name}_gpt2-124m_lr{learning_rate}_mom{momentum}_"
                "wd{weight_decay}_seed{seed}_scale{scaling_rule}"
            ),
        },
        "variant_1_winner": copy.deepcopy(
            selection_report["winners"][design["variant_1_winner_group"]]
        ),
        "candidate_design": {
            "design_id": design["design_id"],
            "path": os.path.relpath(DESIGN_PATH, REPOSITORY_ROOT).replace("\\", "/"),
        },
        "runs": runs,
    }
    validate_manifest(manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-1-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--launch-authorized", action="store_true")
    args = parser.parse_args(argv)
    manifest = materialize_manifest(
        _read_json(args.variant_1_selection),
        launch_authorized=args.launch_authorized,
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    authorization = "authorized" if manifest["launch_authorized"] else "unauthorized"
    print(f"materialized {authorization} manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
