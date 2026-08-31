"""Materialize the isolated zero eligible-matrix-decay follow-up manifest."""

import argparse
import copy
import hashlib
import json
import os

from shared_utils.experiment_manifest import repository_root, validate_manifest


DESIGN_PATH = os.path.join(
    repository_root(),
    "zero_matrix_decay_followup",
    "task_zero_matrix_decay_candidate_design.json",
)


def _read_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(relative_path):
    return os.path.join(repository_root(), *relative_path.split("/"))


def verify_protected_artifacts(design):
    """Reject materialization if an original manifest or report changed."""

    for relative_path, expected_hash in design["protected_artifacts"].items():
        path = _repository_path(relative_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing protected artifact: {relative_path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"protected artifact changed: {relative_path}")


def _static_configuration(design):
    specification = design["families"]["static_per_matrix_sgdm"]
    source = _read_json(_repository_path(specification["source_manifest"]))
    matches = [
        run
        for run in source["runs"]
        if run["run_id"] == specification["source_run_id"]
    ]
    if len(matches) != 1:
        raise ValueError("static source run must resolve exactly once")
    configuration = matches[0].get("static_multiplier_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("static source run lacks multiplier configuration")
    if configuration.get("mapping_id") != specification["mapping_id"]:
        raise ValueError("static source mapping id changed")
    return copy.deepcopy(configuration)


def _common_run(design, optimizer_name, run_id, run_name, selection, overrides):
    budget = design["budget"]
    return {
        "run_id": run_id,
        "run_name": run_name,
        "optimizer_name": optimizer_name,
        "seed": budget["exploratory_seed"],
        "max_iters": budget["max_iters"],
        "tokens_per_update": budget["tokens_per_update"],
        "selection_tokens": budget["selection_tokens"],
        "max_processed_tokens": budget["max_processed_tokens"],
        "evaluation_steps": budget["evaluation_steps"],
        "selection_step": budget["selection_step"],
        "selection_values": selection,
        "overrides": overrides,
    }


def _run_name(optimizer_name, learning_rate, momentum, scaling_rule):
    return (
        f"zero_matrix_decay_{optimizer_name}_gpt2-124m_"
        f"lr{learning_rate:g}_mom{momentum:g}_mwd0_seed1337_scale{scaling_rule}"
    )


def materialize_manifest(design=None):
    """Build and validate the frozen 36-run manifest without launching compute."""

    design = copy.deepcopy(design if design is not None else _read_json(DESIGN_PATH))
    verify_protected_artifacts(design)
    static_configuration = _static_configuration(design)
    runs = []

    for optimizer_name in (
        "global_sgdm",
        "static_per_matrix_sgdm",
        "frobenius_normalized_sgdm",
    ):
        family = design["families"][optimizer_name]
        for learning_rate in family["learning_rates"]:
            for momentum in family["momenta"]:
                scaling_rule = family["scaling_rule"]
                selection = {
                    "learning_rate": learning_rate,
                    "family_learning_rate": learning_rate,
                    "momentum": momentum,
                    "weight_decay": design["matrix_weight_decay"],
                    "scaling_rule": scaling_rule,
                }
                overrides = {"matrix_momentum": momentum}
                if optimizer_name == "frobenius_normalized_sgdm":
                    overrides.update({
                        "frobenius_learning_rate": learning_rate,
                        "frobenius_epsilon": family["frobenius_epsilon"],
                        "frobenius_shape_factor": family[
                            "frobenius_shape_factor"
                        ],
                    })
                    selection.update({
                        "frobenius_epsilon": family["frobenius_epsilon"],
                        "frobenius_shape_factor": family[
                            "frobenius_shape_factor"
                        ],
                        "normalization_rule_id": family[
                            "normalization_rule_id"
                        ],
                    })
                else:
                    overrides["matrix_learning_rate"] = learning_rate
                if optimizer_name == "static_per_matrix_sgdm":
                    overrides.update({
                        "static_default_multiplier": 1.0,
                        "static_matrix_type_multipliers": copy.deepcopy(
                            static_configuration["specification"][
                                "matrix_type_multipliers"
                            ]
                        ),
                        "static_exact_parameter_multipliers": copy.deepcopy(
                            static_configuration["specification"][
                                "exact_parameter_multipliers"
                            ]
                        ),
                    })
                    selection.update({
                        "mapping_name": family["mapping_name"],
                        "mapping_id": static_configuration["mapping_id"],
                        "mapping_fingerprint_sha256": static_configuration[
                            "fingerprint_sha256"
                        ],
                    })

                run_id = (
                    f"zero_wd_{optimizer_name}_lr{learning_rate:g}_"
                    f"mom{momentum:g}"
                )
                run = _common_run(
                    design,
                    optimizer_name,
                    run_id,
                    _run_name(
                        optimizer_name,
                        learning_rate,
                        momentum,
                        scaling_rule,
                    ),
                    selection,
                    overrides,
                )
                if optimizer_name == "static_per_matrix_sgdm":
                    run["static_multiplier_configuration"] = copy.deepcopy(
                        static_configuration
                    )
                runs.append(run)

    artifacts = design["artifact_names"]
    manifest = {
        "schema_version": 1,
        "manifest_id": "zero-matrix-decay-followup-exploratory-v1",
        "purpose": "study",
        "launch_authorized": False,
        "counts_toward_study_budget": True,
        "config_file": design["config_file"],
        "environment_lock": "cluster_environment.freeze.txt",
        "preflight_tests": [
            "tests.test_baseline_preflight",
            "tests.test_parameter_partition",
            "tests.test_zero_matrix_decay_followup",
        ],
        "dataset": {
            "name": "openwebtext",
            "files": [
                "data/openwebtext/train.bin",
                "data/openwebtext/val.bin",
            ],
            "lock_file": "data/openwebtext/dataset_fingerprints.json",
            "require_lock": True,
        },
        "resources": {
            "world_size": 2,
            "device_name_contains": "NVIDIA RTX PRO 6000 Blackwell",
        },
        "output_root": artifacts["output_root"],
        "expected_run_count": len(runs),
        "selection": {
            "group_by": "optimizer_name",
            "expected_groups": [
                "global_sgdm",
                "static_per_matrix_sgdm",
                "frobenius_normalized_sgdm",
            ],
            "metric": "validation_loss",
            "mode": "min",
            "tie_break": ["family_learning_rate", "momentum"],
        },
        "confirmation": {
            "manifest_id": "zero-matrix-decay-followup-confirmation-v1",
            "additional_seeds": design["budget"]["confirmation_seeds"],
            "run_name_template": (
                "zero_matrix_decay_{optimizer_name}_gpt2-124m_"
                "lr{learning_rate}_mom{momentum}_mwd0_seed{seed}_"
                "scale{scaling_rule}"
            ),
        },
        "candidate_design": {
            "design_id": design["design_id"],
            "path": "zero_matrix_decay_followup/"
            "task_zero_matrix_decay_candidate_design.json",
        },
        "claim_gate": {
            "required_successful_matched_seeds_per_family": 3,
            "families": [
                "global_sgdm",
                "static_per_matrix_sgdm",
                "frobenius_normalized_sgdm",
            ],
        },
        "outcome_policy": {
            "selection_eligibility": "completed_uninterrupted_only",
            "replacement": "forbidden_without_separately_approved_follow_up",
        },
        "protected_artifacts": copy.deepcopy(design["protected_artifacts"]),
        "report_names": {
            "selection": artifacts["selection_report"],
            "final": artifacts["final_report"],
        },
        "runs": runs,
    }
    if len(runs) != design["budget"]["total_exploratory_candidates"]:
        raise ValueError("materialized exploratory budget changed")
    validate_manifest(manifest)
    return manifest


def _serialized_json(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_new_or_identical(path, value):
    """Create a new artifact, or accept an existing byte-identical artifact."""

    serialized = _serialized_json(value)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as existing_file:
            if existing_file.read() != serialized:
                raise FileExistsError(f"refusing to overwrite changed artifact: {path}")
        return "unchanged"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialized)
    return "created"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "experiment_manifests/"
            "zero_matrix_decay_followup_exploratory.json"
        ),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = materialize_manifest()
    output_path = _repository_path(args.output)
    if args.check:
        if not os.path.isfile(output_path):
            raise FileNotFoundError(f"missing materialized manifest: {output_path}")
        with open(output_path, encoding="utf-8") as input_file:
            if input_file.read() != _serialized_json(manifest):
                raise ValueError("materialized zero-decay manifest changed")
        print(f"valid manifest: {manifest['manifest_id']}")
        return 0
    status = write_new_or_identical(output_path, manifest)
    print(f"{status}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
